import math
import torch
from src.slm_gate.slm_loader import load_slm

PROMPT = """You are a strict fact-checker.
Decide whether the CLAIM is directly supported by the CONTEXT.
Ignore the user's question if it is ambiguous; focus ONLY on whether the CONTEXT supports the CLAIM.
Answer ONLY YES or NO.

CONTEXT:
{ctx}

CLAIM:
{claim}

Is the CLAIM supported by the CONTEXT?
"""

def _build_ctx(
    retrieved_chunks: list[dict],
    *,
    top_n: int = 2,
    max_chars: int = 1500,
    per_chunk_chars: int = 700,
) -> str:
    """
    Build a compact context to avoid huge prompts (prevents OOM / killed).
    - takes top_n retrieved chunks (already sorted by FAISS score)
    - deduplicates exact texts
    - truncates each chunk and total context
    """
    parts = []
    seen = set()
    total = 0

    for c in (retrieved_chunks or [])[:top_n]:
        t = (c.get("text") or "").strip()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)

        if len(t) > per_chunk_chars:
            t = t[:per_chunk_chars] + "…"

        piece = f"- {t}"
        if total + len(piece) > max_chars:
            break

        parts.append(piece)
        total += len(piece)

    return "\n".join(parts)


@torch.no_grad()
def _score_completion(tok, model, prompt: str, completion: str, *, max_prompt_tokens: int = 512) -> float:
    """
    Returns average log-prob of the completion tokens given the prompt.
    We TRUNCATE the prompt to max_prompt_tokens to prevent large forward passes.
    """
    prompt_ids = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    ).input_ids.to(model.device)

    comp_ids = tok(
        completion,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(model.device)

    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)

    out = model(input_ids=input_ids)
    logits = out.logits  # [1, seq, vocab]

    comp_len = comp_ids.shape[1]
    start = prompt_ids.shape[1] - 1  # logits index where completion begins

    logp = 0.0
    for i in range(comp_len):
        token_id = comp_ids[0, i]
        step_logits = logits[0, start + i]
        logp += torch.log_softmax(step_logits, dim=-1)[token_id].item()

    return logp / max(1, comp_len)


def yes_no_probs(question: str, context: str, claim: str) -> tuple[float, float]:
    tok, model = load_slm()
    prompt = PROMPT.format(ctx=context, claim=claim)

    yes_score = _score_completion(tok, model, prompt, " YES", max_prompt_tokens=512)
    no_score  = _score_completion(tok, model, prompt, " NO",  max_prompt_tokens=512)

    m = max(yes_score, no_score)
    ey = math.exp(yes_score - m)
    en = math.exp(no_score - m)
    p_yes = ey / (ey + en)
    p_no  = en / (ey + en)
    return p_yes, p_no


def gate_claim(
    question: str,
    retrieved_chunks: list[dict],
    claim: str,
    sim_support: float = 0.40,   # slightly lower for long PDF chunks
    sim_weak: float = 0.20,      # avoid early "p_yes=0.0" too often
    p_yes_support: float = 0.60,
    p_no_contra: float = 0.85,
    ctx_top_n: int = 2,
):
    sim_max = max([c.get("score", 0.0) for c in (retrieved_chunks or [])], default=0.0)

    # Build compact context (prevents huge prompts)
    context = _build_ctx(retrieved_chunks, top_n=ctx_top_n, max_chars=1500, per_chunk_chars=700)

    if sim_max < sim_weak:
        return "UNDER_SPECIFIED", {"p_yes": 0.0, "p_no": 0.0, "sim_max": sim_max, "uncertainty": 1.0}, context

    p_yes, p_no = yes_no_probs(question, context, claim)
    uncertainty = 1.0 - abs(p_yes - 0.5) * 2.0

    if sim_max >= sim_support and p_yes >= p_yes_support:
        label = "SUPPORTED"
    elif sim_max >= sim_support and p_no >= p_no_contra:
        label = "CONTRADICTED"
    else:
        label = "UNDER_SPECIFIED"

    return label, {"p_yes": p_yes, "p_no": p_no, "sim_max": sim_max, "uncertainty": uncertainty}, context