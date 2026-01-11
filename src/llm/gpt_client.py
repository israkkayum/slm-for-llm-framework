import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
)

GROUNDED_SYSTEM = """You are an enterprise assistant.
You MUST answer using ONLY the provided CONTEXT.
If the CONTEXT does not contain enough information, reply exactly:
INSUFFICIENT_CONTEXT
Do not guess or use outside knowledge."""

def chat(prompt: str, system: str = GROUNDED_SYSTEM, context: str = "") -> str:
    # Put context into the user message to force grounding
    user = f"CONTEXT:\n{context}\n\nPROMPT:\n{prompt}"
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        extra_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Hallucination Firewall",
        },
    )
    return resp.choices[0].message.content.strip()