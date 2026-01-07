from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

def embed_chunks(chunks):
    """
    Compute embeddings for a list of chunks.

    Args:
        chunks (List[dict]): Each dict has 'doc_id', 'chunk_id', 'text'

    Returns:
        embeddings (np.ndarray): Array of shape (num_chunks, embedding_dim)
        chunk_meta (List[dict]): Same order as embeddings
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return embeddings, chunks