# scripts/run_experiments.py
# Automatic experiments to produce GOLD labels for ROC/F1/Confusion/Calibration plots.
#
# What it does:
# - Samples "true" claims from your indexed KB (meta.json chunks)
# - Generates "wrong" (CONTRADICTED) claims by perturbing dates/numbers
# - Generates "unknown" (UNDER_SPECIFIED) claims unrelated to KB
# - Runs your Layer-1 retrieval + Layer-2 SLM gate (+ optional Layer-3 reasoner)
# - Saves run JSON files into outputs/runs/ with:
#     - per_fact[i]["gold_label"]
#     - per_fact[i]["gate_label"] (pred)
#     - p_yes, p_no, sim_max, uncertainty
#     - timings + variant
#
# After running, execute:
#   python scripts/make_plots.py
# and it will automatically generate ROC/F1/Confusion/Calibration plots.
#
# Usage examples:
#   python scripts/run_experiments.py --n_runs 30 --facts_per_run 6 --seed 7
#   python scripts/run_experiments.py --n_runs 50 --domain university
#
# Notes:
# - No GPT calls (cheap + fast). This evaluates your verifier gate behavior.
# - Requires that you already built the FAISS index + meta.json.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from src.retrieval.vector_store import search
from src.config import INDEX_DIR
from src.slm_gate.gate import gate_claim
from src.evaluation.logger import save_run

# Optional: If you want to include Layer-3 in experiments
# from src.reasoning.neurosymbolic_reasoner import reason


# -----------------------------
# Load KB chunks from meta.json
# -----------------------------

def load_meta() -> List[Dict[str, Any]]:
    meta_path = Path(INDEX_DIR) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found at: {meta_path}. Run scripts/build_index.py first.")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_by_domain(meta: List[Dict[str, Any]], domain: Optional[str]) -> List[Dict[str, Any]]:
    if not domain:
        return meta
    return [m for m in meta if m.get("domain") == domain]


# -----------------------------
# Sentence extraction (simple)
# -----------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

def extract_candidate_sentences(text: str) -> List[str]:
    # Keep lines that have at least some content and are not too short
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    # Filter very short strings; keep schedule-like lines too (contain date or code)
    out = []
    for p in parts:
        if len(p) >= 20:
            out.append(p)
        else:
            # allow short but structured lines
            if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", p) or re.search(r"\bCSE\d+\b", p):
                out.append(p)
    return out


def pick_true_claim(meta: List[Dict[str, Any]], rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    """Pick a sentence from a random chunk as a true (SUPPORTED) claim."""
    for _ in range(50):
        m = rng.choice(meta)
        sents = extract_candidate_sentences(m.get("text", ""))
        if not sents:
            continue
        claim = rng.choice(sents).strip()
        if claim:
            return claim, m
    # fallback: raw chunk
    m = rng.choice(meta)
    return (m.get("text", "")[:200].strip() or "The document contains information."), m


# -----------------------------
# Claim corruption utilities
# -----------------------------

def bump_integer(s: str, delta: int = 1) -> Optional[str]:
    m = re.search(r"\b(\d{1,4})\b", s)
    if not m:
        return None
    n = int(m.group(1))
    # avoid bumping years too aggressively unless it looks like a year
    new = n + delta
    return s[:m.start(1)] + str(new) + s[m.end(1):]


def corrupt_date_ddmmyy(s: str, rng: random.Random) -> Optional[str]:
    # matches 19/12/25 or 19/12/2025
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", s)
    if not m:
        return None
    dd = int(m.group(1))
    mm = int(m.group(2))
    yy = m.group(3)

    # Change the day by +1 or -1 (wrap minimally)
    dd2 = dd + (1 if rng.random() < 0.5 else -1)
    if dd2 <= 0:
        dd2 = 2
    if dd2 > 28:
        dd2 = 28

    new_date = f"{dd2:02d}/{mm:02d}/{yy}"
    return s[:m.start()] + new_date + s[m.end():]


def corrupt_time_ampm(s: str, rng: random.Random) -> Optional[str]:
    # Change "9 AM" -> "10 AM" or "5 PM" -> "6 PM"
    m = re.search(r"\b(\d{1,2})\s*(AM|PM)\b", s, flags=re.IGNORECASE)
    if not m:
        return None
    hour = int(m.group(1))
    ap = m.group(2)
    hour2 = hour + (1 if rng.random() < 0.5 else -1)
    hour2 = max(1, min(12, hour2))
    new = f"{hour2} {ap}"
    return s[:m.start()] + new + s[m.end():]


def corrupt_weekday(s: str, rng: random.Random) -> Optional[str]:
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    # Find any day (case-insensitive)
    for d in days:
        if re.search(rf"\b{d}\b", s, flags=re.IGNORECASE):
            # replace with a different day
            other = rng.choice([x for x in days if x.lower() != d.lower()])
            return re.sub(rf"\b{d}\b", other, s, flags=re.IGNORECASE)
    return None


def make_contradicted_claim(true_claim: str, rng: random.Random) -> str:
    # Try multiple corruption strategies; if none apply, bump integer if possible.
    funcs = [corrupt_date_ddmmyy, corrupt_time_ampm, corrupt_weekday]
    rng.shuffle(funcs)

    for fn in funcs:
        out = fn(true_claim, rng)
        if out and out != true_claim:
            return out

    out2 = bump_integer(true_claim, delta=1)
    if out2 and out2 != true_claim:
        return out2

    # last resort: add a negation flip (not always logically contradictory, but often)
    if " is " in true_claim:
        return true_claim.replace(" is ", " is not ", 1)
    return true_claim + " (NOT TRUE)"


def make_under_specified_claim(rng: random.Random) -> str:
    # Generic, likely absent from KB (safe "unknown" set).
    templates = [
        "The CEO of the company is {name}.",
        "The patient should take {dose} mg of {drug} daily.",
        "The next holiday is on {date}.",
        "The campus bus runs every {n} minutes.",
        "The store has {n} branches in Europe.",
    ]
    names = ["Alex Rahman", "Samira Khan", "John Smith", "Ayesha Islam"]
    drugs = ["Metformin", "Atorvastatin", "Aspirin", "Amoxicillin"]
    doses = ["10", "25", "50", "100"]
    dates = ["01/01/2027", "15/08/2026", "12/03/2028"]
    nvals = ["7", "12", "20", "45"]

    t = rng.choice(templates)
    return t.format(
        name=rng.choice(names),
        drug=rng.choice(drugs),
        dose=rng.choice(doses),
        date=rng.choice(dates),
        n=rng.choice(nvals),
    )


# -----------------------------
# Run one experiment
# -----------------------------

def run_single_experiment(
    meta: List[Dict[str, Any]],
    facts_per_run: int,
    k_retrieval: int,
    domain: Optional[str],
    variant: str,
    rng: random.Random,
    include_reasoner: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """
    Produces a single run payload with mixed gold labels:
      - SUPPORTED (true claims from KB)
      - CONTRADICTED (corrupted versions)
      - UNDER_SPECIFIED (unrelated claims)
    """
    timings: Dict[str, float] = {"l1_retrieval": 0.0, "l2_slm": 0.0, "l3_reasoner": 0.0}
    t0 = time.perf_counter()

    # Build a "query" string just for logging (not used for correctness)
    query = f"[EXPERIMENT] Mixed claims test (domain={domain or 'all'}, k={k_retrieval})"

    per_fact = []
    gold_counts = {"SUPPORTED": 0, "CONTRADICTED": 0, "UNDER_SPECIFIED": 0}

    # Decide proportions (you can change)
    # roughly: 40% supported, 40% contradicted, 20% underspecified
    n_supported = max(1, int(round(facts_per_run * 0.4)))
    n_contra = max(1, int(round(facts_per_run * 0.4)))
    n_under = max(0, facts_per_run - n_supported - n_contra)
    plan = (["SUPPORTED"] * n_supported) + (["CONTRADICTED"] * n_contra) + (["UNDER_SPECIFIED"] * n_under)
    rng.shuffle(plan)

    for gold in plan:
        if gold == "SUPPORTED":
            claim, _src = pick_true_claim(meta, rng)
        elif gold == "CONTRADICTED":
            true_claim, _src = pick_true_claim(meta, rng)
            claim = make_contradicted_claim(true_claim, rng)
        else:
            claim = make_under_specified_claim(rng)

        # Layer-1 retrieval for this claim
        t = time.perf_counter()
        retrieved = search(claim, k=k_retrieval, domain=domain)
        timings["l1_retrieval"] += time.perf_counter() - t

        # Layer-2 gate
        t = time.perf_counter()
        pred_label, scores, ctx = gate_claim(query, retrieved, claim)
        timings["l2_slm"] += time.perf_counter() - t

        # Optional Layer-3
        reasoning_json = None
        if include_reasoner:
            t = time.perf_counter()
            # reasoning_json = reason(claim, ctx)
            # If you want reasoner, uncomment import + line above
            reasoning_json = None
            timings["l3_reasoner"] += time.perf_counter() - t

        per_fact.append({
            "fact": claim,
            "retrieved": retrieved,
            "gate_label": pred_label,

            "p_yes": scores.get("p_yes", 0.0),
            "p_no": scores.get("p_no", 0.0),
            "sim_max": scores.get("sim_max", 0.0),
            "uncertainty": scores.get("uncertainty", 1.0),

            # ✅ GOLD LABEL (this enables ROC/F1/Confusion/Calibration)
            "gold_label": gold,

            # optional: keep context for debugging
            "context": ctx,

            # optional: reasoner output
            "reasoning": reasoning_json,
        })

        gold_counts[gold] += 1

    timings["total"] = time.perf_counter() - t0

    payload = {
        "query": query,
        "variant": variant,
        "domain": domain,
        "timings": timings,
        "gold_counts": gold_counts,
        "per_fact": per_fact,
    }

    path = save_run(payload)
    return payload, path


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=30, help="Number of experiment runs to generate")
    ap.add_argument("--facts_per_run", type=int, default=8, help="Facts per run (mix of gold labels)")
    ap.add_argument("--k", type=int, default=5, help="Top-K retrieval for each claim")
    ap.add_argument("--domain", type=str, default=None, help="Optional domain filter (e.g., university/company/medical)")
    ap.add_argument("--variant", type=str, default="full", help="Variant label for ablation plots (e.g., full/no_l2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--include_reasoner", action="store_true", help="Include Layer-3 reasoner (if enabled in code)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    meta_all = load_meta()
    meta = filter_by_domain(meta_all, args.domain)
    if not meta:
        raise ValueError(f"No meta chunks found for domain={args.domain}. Check meta.json domains.")

    print(f"[INFO] Loaded {len(meta)} chunks from meta.json (domain={args.domain or 'all'})")
    print(f"[INFO] Generating {args.n_runs} runs x {args.facts_per_run} facts (k={args.k})")

    saved = 0
    for i in range(args.n_runs):
        payload, path = run_single_experiment(
            meta=meta,
            facts_per_run=args.facts_per_run,
            k_retrieval=args.k,
            domain=args.domain,
            variant=args.variant,
            rng=rng,
            include_reasoner=args.include_reasoner,
        )
        saved += 1
        print(f"[OK] Run {i+1}/{args.n_runs} saved: {path} | gold_counts={payload['gold_counts']}")

    print(f"\nDone. Saved {saved} experiment runs into outputs/runs/.")
    print("Now run: python scripts/make_plots.py")


if __name__ == "__main__":
    main()