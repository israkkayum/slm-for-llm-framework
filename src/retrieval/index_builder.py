import os, json
import faiss
import numpy as np
from src.retrieval.embedder import embed_texts
from src.config import INDEX_DIR

def build_faiss_index(chunks: list[dict]):
    texts = [c["text"] for c in chunks]
    vecs = embed_texts(texts).astype("float32")

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "meta.json"), "w") as f:
        json.dump(chunks, f, indent=2)

    return index