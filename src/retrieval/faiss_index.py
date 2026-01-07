import faiss
import numpy as np
import pickle
from pathlib import Path

def build_faiss_index(embeddings, save_path="data/embeddings/faiss.index"):
    """
    Build and save a FAISS index for retrieval.

    Args:
        embeddings (np.ndarray): Chunk embeddings
        save_path (str): Path to save the FAISS index

    Returns:
        index: FAISS index object
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # simple L2 distance
    index.add(np.array(embeddings).astype("float32"))
    
    # save index
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, save_path)
    
    return index

def load_faiss_index(path="data/embeddings/faiss.index"):
    """Load a saved FAISS index."""
    return faiss.read_index(path)

def save_chunk_metadata(chunks, path="data/embeddings/chunks.pkl"):
    """Save chunk metadata for retrieval."""
    with open(path, "wb") as f:
        pickle.dump(chunks, f)

def load_chunk_metadata(path="data/embeddings/chunks.pkl"):
    """Load chunk metadata."""
    with open(path, "rb") as f:
        return pickle.load(f)