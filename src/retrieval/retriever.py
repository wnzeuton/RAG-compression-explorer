import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.retrieval.faiss_index import load_chunk_metadata
from src.config import EMBEDDING_MODEL, TOP_K


class Retriever:
    def __init__(self, top_k=TOP_K):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.top_k = top_k

        # Load ALL chunk metadata (with embeddings stored)
        self.chunks = load_chunk_metadata()

        # Cache embeddings for fast filtering
        self.embeddings = np.array(
            [c["embedding"] for c in self.chunks],
            dtype="float32"
        )

    def _build_temp_index(self, embeddings):
        """Build a temporary FAISS index for a subset of chunks"""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity
        index.add(embeddings)
        return index

    def query(
        self,
        query_text,
        max_year=None,
        allowed_entities=None,
        top_k=None
    ):
        """
        Args:
            query_text (str)
            max_year (int | None): time cutoff
            allowed_entities (set[str] | None): entity whitelist
            top_k (int | None)

        Returns:
            List[dict]: ranked chunks
        """
        top_k = top_k or self.top_k

        # ----------------------------------
        # Step 1: filter eligible chunks
        # ----------------------------------
        eligible_indices = []
        eligible_chunks = []

        for i, c in enumerate(self.chunks):
            if max_year is not None and c["year"] > max_year:
                continue
            if allowed_entities is not None and c["title"] not in allowed_entities:
                continue

            eligible_indices.append(i)
            eligible_chunks.append(c)

        if not eligible_chunks:
            return []

        eligible_embeddings = self.embeddings[eligible_indices]

        # ----------------------------------
        # Step 2: build temporary FAISS index
        # ----------------------------------
        index = self._build_temp_index(eligible_embeddings)

        # ----------------------------------
        # Step 3: embed query & search
        # ----------------------------------
        query_emb = self.model.encode(
            [query_text],
            normalize_embeddings=True
        ).astype("float32")

        distances, indices = index.search(query_emb, min(top_k, len(eligible_chunks)))

        # ----------------------------------
        # Step 4: return ranked results
        # ----------------------------------
        results = [eligible_chunks[i] for i in indices[0]]
        return results