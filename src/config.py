import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# OpenRouter expects namespaced model ids like "openai/gpt-4o-mini"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SLM_MODEL = os.getenv("SLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
TOP_K = int(os.getenv("TOP_K", "5"))