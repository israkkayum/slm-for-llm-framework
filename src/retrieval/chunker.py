from typing import List

def chunk_text(text: str, chunk_size: int = 450, overlap: int = 80) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)

    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        cleaned = chunk.strip()
        if cleaned:
            chunks.append(cleaned)
        i += step

    return chunks