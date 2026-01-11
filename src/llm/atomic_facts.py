import json
from src.llm.gpt_client import chat

FACT_PROMPT = """
Extract ATOMIC FACTS from the answer.

Rules:
- Include ONLY factual, checkable claims.
- Ignore advice, suggestions, and generic statements (e.g., "check website", "it depends").
- Keep each fact short and single-claim.
Return ONLY a JSON list of strings.

ANSWER:
{answer}
"""

def decompose_to_atomic_facts(answer: str) -> list[str]:
    out = chat(FACT_PROMPT.format(answer=answer), system="You output valid JSON only.")
    try:
        facts = json.loads(out)
        return [f.strip() for f in facts if isinstance(f, str) and f.strip()]
    except Exception:
        # fallback: simple split
        return [s.strip() for s in answer.split(".") if s.strip()]