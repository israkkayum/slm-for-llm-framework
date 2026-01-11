from src.llm.gpt_client import chat
from src.reasoning.rules import simple_rules

REASON_PROMPT = """
You must reason using ONLY the provided context.
Task:
- If claim is supported, keep it.
- If unsupported/uncertain, correct it using context OR say "INSUFFICIENT".
Return JSON:
{{"final_claim": "...", "verdict": "SUPPORTED|CORRECTED|REJECTED|INSUFFICIENT", "why": "..."}}

Context:
{ctx}

Claim:
{claim}

Rule signal:
{rule}
"""

def reason(claim: str, ctx: str):
    rule = simple_rules(claim)
    out = chat(REASON_PROMPT.format(ctx=ctx, claim=claim, rule=rule), system="Return JSON only.")
    return out  # keep as JSON string for logging