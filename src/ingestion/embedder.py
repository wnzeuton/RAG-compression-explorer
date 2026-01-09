import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    """
    Embed year-based chunks.

    Returns:
        embeddings: np.ndarray
        chunks_with_meta: List[dict]
    """
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)

    return embeddings, chunks