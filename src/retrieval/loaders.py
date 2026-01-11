import os
from typing import List, Dict

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_md(path: str) -> str:
    return load_txt(path)

def load_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        if t.strip():
            pages.append(t)
    return "\n".join(pages)

def load_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt"]:
        return load_txt(path)
    if ext in [".md"]:
        return load_md(path)
    if ext in [".pdf"]:
        return load_pdf(path)
    raise ValueError(f"Unsupported file: {path}")