from pathlib import Path
from src.ingestion.chunker import chunk_documents_from_files
from src.ingestion.embedder import embed_chunks
from src.retrieval.faiss_index import build_faiss_index, save_chunk_metadata

RAW_DIR = Path("data/raw")
file_paths = list(RAW_DIR.glob("*.txt"))  # get all txt files

chunks = chunk_documents_from_files(file_paths, separator="---")
print(f"Created {len(chunks)} chunks from {len(file_paths)} files")

embeddings, chunks_with_meta = embed_chunks(chunks)
print(f"Created embeddings of shape {embeddings.shape}")

chunks_with_embeddings = []
for c, emb in zip(chunks_with_meta, embeddings):
    chunks_with_embeddings.append({
        "doc_id": c["doc_id"],
        "chunk_id": c["chunk_id"],
        "text": c["text"],
        "embedding": emb
    })

index = build_faiss_index(embeddings)
print("FAISS index built and saved")

save_chunk_metadata(chunks_with_embeddings)
print("Chunk metadata saved")