"""
Thin retrieval wrapper over the FAISS index built by build_index.py.
Used by agents/evidence_retrieval_agent.py.
"""

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class KnowledgeRetriever:
    def __init__(self, index_dir: str = INDEX_DIR, embedding_model: str = EMBEDDING_MODEL):
        index_path = os.path.join(index_dir, "faiss.index")
        chunks_path = os.path.join(index_dir, "chunks.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"No FAISS index found at {index_path}. "
                f"Run `python knowledge_base/build_index.py` first."
            )
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.embedder = SentenceTransformer(embedding_model)

    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve top-K most relevant clinical evidence chunks for a query.
        Returns list of {text, source, score} sorted by relevance (Section
        5.4.3: cosine similarity retrieves Top-K relevant documents)."""
        query_vec = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "score": float(score),
            })
        return results
