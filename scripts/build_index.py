import os
import json
from pathlib import Path

from src.retrieval.index_builder import build_faiss_index
from src.retrieval.loaders import load_document
from src.retrieval.chunker import chunk_text
from src.config import INDEX_DIR

RAW_DIR = "data/raw"

def iter_files(root: str):
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in [".pdf", ".txt", ".md"]:
            yield str(p)

def infer_domain(path: str) -> str:
    # data/raw/company/xxx.pdf -> company
    parts = Path(path).parts
    try:
        idx = parts.index("raw")
        return parts[idx+1] if idx+1 < len(parts) else "general"
    except ValueError:
        return "general"

def main():
    chunks = []
    for file_path in iter_files(RAW_DIR):
        domain = infer_domain(file_path)
        doc_id = os.path.basename(file_path)

        text = load_document(file_path)
        pieces = chunk_text(text, chunk_size=450, overlap=80)

        for i, ch in enumerate(pieces):
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": i,
                "text": ch,
                "source_path": file_path,
                "domain": domain,
                "tags": []
            })

    os.makedirs(INDEX_DIR, exist_ok=True)
    build_faiss_index(chunks)
    print(f"✅ Built index with {len(chunks)} chunks from: {RAW_DIR}")
    print(f"✅ Saved in: {INDEX_DIR}")

if __name__ == "__main__":
    main()