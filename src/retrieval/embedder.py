from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL

_model = None

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def embed_texts(texts: list[str]):
    return get_embedder().encode(texts, normalize_embeddings=True)