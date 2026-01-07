from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_chunks
from src.retrieval.faiss_index import build_faiss_index, save_chunk_metadata
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

docs = load_documents()
print(f"Loaded {len(docs)} documents")

chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
print(f"Created {len(chunks)} chunks")

embeddings, chunks_with_meta = embed_chunks(chunks)
print(f"Created embeddings of shape {embeddings.shape}")

index = build_faiss_index(embeddings)
print("FAISS index built and saved")

save_chunk_metadata(chunks_with_meta)
print("Chunk metadata saved")