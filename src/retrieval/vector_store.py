import os, json
import faiss
import numpy as np
from src.config import INDEX_DIR, TOP_K
from src.retrieval.embedder import embed_texts


def load_store():
    idx_path = os.path.join(INDEX_DIR, "faiss.index")
    meta_path = os.path.join(INDEX_DIR, "meta.json")
    index = faiss.read_index(idx_path)
    meta = json.load(open(meta_path))
    return index, meta


# 🔥 UPDATED search() with domain filter
def search(query: str, k: int = TOP_K, domain: str | None = None):
    index, meta = load_store()

    qv = embed_texts([query]).astype("float32")
    scores, ids = index.search(qv, k * 3)  # over-fetch

    results = []
    for score, i in zip(scores[0], ids[0]):
        if i == -1:
            continue

        item = meta[i]

        # Domain filtering
        if domain and item.get("domain") != domain:
            continue

        results.append({
            "score": float(score),
            **item
        })

        if len(results) >= k:
            break

    return results