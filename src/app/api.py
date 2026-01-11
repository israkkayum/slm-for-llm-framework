from fastapi import FastAPI
from pydantic import BaseModel
from scripts.run_demo import pipeline
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="Hallucination Firewall API")

class QueryIn(BaseModel):
    query: str

@app.post("/ask")
def ask(q: QueryIn):
    out, path = pipeline(q.query)
    return {"run_file": path, "result": out}