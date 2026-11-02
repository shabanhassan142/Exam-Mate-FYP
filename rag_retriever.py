"""
RAG Retriever
=============
Loads the pre-built FAISS index and provides a simple retrieve() function.
Used by the question-generation pipeline to fetch relevant context chunks
before passing them to TinyLlama (or any other generator).
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_DB_DIR = Path("vector_db")
EMBED_MODEL   = "all-MiniLM-L6-v2"


class RAGRetriever:
    """Singleton-style retriever — load once, query many times."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    # ── Initialisation ────────────────────────────────────────────────────────
    def load(self):
        """Load FAISS index + metadata + embedding model into memory."""
        if self._loaded:
            return

        index_path    = VECTOR_DB_DIR / "faiss_index.bin"
        metadata_path = VECTOR_DB_DIR / "metadata.pkl"
        info_path     = VECTOR_DB_DIR / "db_info.json"

        if not index_path.exists():
            raise FileNotFoundError(
                "Vector DB not found. Run  python build_vector_db.py  first."
            )

        print("Loading RAG retriever...")
        self.index    = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        with open(info_path) as f:
            self.info = json.load(f)

        self.embedder = SentenceTransformer(EMBED_MODEL)
        self._loaded  = True
        print(f"  ✓ FAISS index loaded  ({self.index.ntotal} vectors, "
              f"courses: {', '.join(self.info['courses'])})")

    # ── Public API ────────────────────────────────────────────────────────────
    def retrieve(self, query: str, course_code: str = None,
                 top_k: int = 5) -> list[dict]:
        """
        Return the top-k most relevant chunks for *query*.

        Parameters
        ----------
        query       : natural-language question or topic
        course_code : optional filter (e.g. "AI", "DSA")
        top_k       : number of chunks to return

        Returns
        -------
        list of dicts with keys: text, course_code, source_file, chunk_index, score
        """
        if not self._loaded:
            self.load()

        # Embed & normalise query
        q_vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)

        # Search — fetch more if we need to filter by course
        fetch_k = top_k * 10 if course_code else top_k
        scores, indices = self.index.search(q_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            if course_code and meta["course_code"].upper() != course_code.upper():
                continue
            results.append({**meta, "score": float(score)})
            if len(results) == top_k:
                break

        return results

    def get_context(self, query: str, course_code: str = None,
                    top_k: int = 5, max_chars: int = 3000) -> str:
        """
        Convenience wrapper — returns a single string of concatenated chunks
        ready to be injected into a prompt.
        """
        chunks = self.retrieve(query, course_code=course_code, top_k=top_k)
        parts  = []
        total  = 0
        for c in chunks:
            text = c["text"]
            if total + len(text) > max_chars:
                text = text[: max_chars - total]
                parts.append(text)
                break
            parts.append(text)
            total += len(text)
        return "\n\n---\n\n".join(parts)

    def is_ready(self) -> bool:
        return self._loaded and VECTOR_DB_DIR.joinpath("faiss_index.bin").exists()


# Module-level singleton
retriever = RAGRetriever()
