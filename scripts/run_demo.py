import time
import json
from src.llm.gpt_client import chat, GROUNDED_SYSTEM
from src.llm.atomic_facts import decompose_to_atomic_facts
from src.retrieval.vector_store import search
from src.slm_gate.gate import gate_claim
from src.reasoning.neurosymbolic_reasoner import reason
from src.evaluation.logger import save_run

# OPTIONAL: if you have query normalizer
# from src.utils.query_normalizer import expand_date_variants

def build_context_from_hits(hits, max_chars=2000):
    parts, total = [], 0
    for h in hits:
        t = h["text"].strip()
        if not t:
            continue
        chunk = f"- {t}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts)

def pipeline(user_query: str, variant: str = "full"):
    timings = {}

    t0 = time.perf_counter()

    # user_query = expand_date_variants(user_query)  # optional

    # Layer-1 retrieval for generation
    t = time.perf_counter()
    query_hits = search(user_query, k=5)
    query_context = build_context_from_hits(query_hits)
    timings["l1_retrieval"] = time.perf_counter() - t

    # LLM generation (GPT)
    t = time.perf_counter()
    llm_answer = chat(
        prompt=f"QUESTION: {user_query}\nAnswer clearly and concisely.",
        system=GROUNDED_SYSTEM,
        context=query_context
    )
    timings["l0_llm"] = time.perf_counter() - t

    # Atomic facts
    t = time.perf_counter()
    facts = decompose_to_atomic_facts(llm_answer)
    timings["atomic_decompose"] = time.perf_counter() - t

    results = []
    for fact in facts:
        # Verify each fact
        retrieved = search(fact, k=5)

        t = time.perf_counter()
        label, scores, ctx = gate_claim(user_query, retrieved, fact)
        timings["l2_slm"] = timings.get("l2_slm", 0.0) + (time.perf_counter() - t)

        t = time.perf_counter()
        reasoning_json = reason(fact, ctx)
        timings["l3_reasoner"] = timings.get("l3_reasoner", 0.0) + (time.perf_counter() - t)

        # ✅ Attach gold label here (see section 2)
        gold_label = None  # default (no gold yet)

        results.append({
            "fact": fact,
            "retrieved": retrieved,
            "gate_label": label,
            "p_yes": scores["p_yes"],
            "p_no": scores["p_no"],
            "sim_max": scores["sim_max"],
            "uncertainty": scores["uncertainty"],
            "reasoning": reasoning_json,

            # ✅ ADD THIS FIELD (even if None)
            "gold_label": gold_label,
        })

    timings["total"] = time.perf_counter() - t0

    # ✅ Attach variant + timings at the TOP LEVEL payload
    out = {
        "query": user_query,
        "variant": variant,          # ✅ HERE
        "timings": timings,          # ✅ HERE

        "query_context": query_context,
        "llm_answer": llm_answer,
        "atomic_facts": facts,
        "per_fact": results,
    }

    path = save_run(out)
    return out, path

if __name__ == "__main__":
    query = "What are the store working hours?"
    out, path = pipeline(query)
    print("✅ Saved run:", path)
    print(json.dumps(out, indent=2)[:2000])