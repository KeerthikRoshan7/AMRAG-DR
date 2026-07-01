"""
Knowledge Retrieval Pipeline (AM-RAG Section 5.8):
    Documents -> Cleaning -> Chunking -> Embedding -> Vector Database (FAISS)

Chunk size: 512 tokens, overlap 128 tokens (per Section 5.8 chunking
strategy). Embedding model defaults to BAAI/bge-small-en-v1.5 -- a strong,
CPU-friendly general embedder. Swap EMBEDDING_MODEL to a biomedical model
(e.g. "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb" or
"neuml/pubmedbert-base-embeddings") if you want closer alignment with the
paper's stated BioBERT/ClinicalBERT/BGE-M3 options -- bge-small is the
practical zero-cost-hardware default for local + HF Spaces deployment.

Run:
    python knowledge_base/build_index.py
Produces:
    knowledge_base/index/faiss.index
    knowledge_base/index/chunks.json   (chunk text + source metadata)
"""

import os
import json
import glob
import tiktoken
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DOC_DIR = os.path.join(os.path.dirname(__file__), "documents")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 128


def chunk_text(text: str, source: str, chunk_size=CHUNK_SIZE_TOKENS,
                overlap=CHUNK_OVERLAP_TOKENS):
    """Token-aware sliding-window chunking with overlap."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_str = enc.decode(chunk_tokens)
        chunks.append({"text": chunk_str, "source": source})
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def load_and_chunk_documents():
    all_chunks = []
    for path in sorted(glob.glob(os.path.join(DOC_DIR, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source_name = os.path.basename(path)
        all_chunks.extend(chunk_text(text, source_name))
    return all_chunks


def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("Loading + chunking documents...")
    chunks = load_and_chunk_documents()
    print(f"Produced {len(chunks)} chunks from {DOC_DIR}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    print("Embedding chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    # Inner product on normalized vectors == cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved FAISS index ({index.ntotal} vectors, dim={dim}) to {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
