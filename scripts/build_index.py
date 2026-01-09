from pathlib import Path
from src.ingestion.chunker import chunk_documents_from_files
from src.ingestion.embedder import embed_chunks
from src.retrieval.faiss_index import build_faiss_index, save_chunk_metadata
import numpy as np

RAW_DIR = Path("data/raw")
file_paths = list(RAW_DIR.glob("*.txt"))

chunks = chunk_documents_from_files(file_paths)
print(f"Created {len(chunks)} chunks from {len(file_paths)} files")

embeddings, chunks_with_meta = embed_chunks(chunks)
print(f"Created embeddings of shape {embeddings.shape}")

chunks_with_embeddings = []
for c, emb in zip(chunks_with_meta, embeddings):
    chunks_with_embeddings.append({
        "doc_name": c["doc_name"],
        "chunk_id": c["chunk_id"],
        "title": c["title"],
        "type": c["type"],
        "year": c["year"],
        "text": c["text"],
        "start_char": c["start_char"],
        "end_char": c["end_char"],
        "embedding": emb.astype("float32")
    })

index = build_faiss_index(
    np.array([c["embedding"] for c in chunks_with_embeddings])
)
print("FAISS index built and saved")

save_chunk_metadata(chunks_with_embeddings)
print("Chunk metadata saved")