from typing import List, Dict, Any, Optional, Tuple
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None

import numpy as np


class SemanticMatcher:
    """Lightweight semantic matcher using sentence-transformers.

    This prototype embeds per-question text on the fly and computes cosine
    similarities to find best matches. It's intentionally simple so it can be
    used without a heavy index (FAISS) for a quick integration.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        if SentenceTransformer is None:
            raise ImportError('sentence-transformers is not installed')

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return embeddings as a numpy array (num_texts, dim)."""
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings

    def find_similar_by_embeddings(self, new_questions: List[str], existing_questions: List[str],
                                   top_k: int = 1, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Find semantic matches between lists of questions.

        Returns a list of matches in the same format as the project's
        `find_similar_questions` expects for `similar_questions` entries.
        The function assumes questions are in the same order as their
        paper-scoped indices; mapping back to global paper IDs should be done
        by the caller if needed.
        """
        results = []

        # Clean inputs
        new_questions_clean = [q.strip() for q in new_questions]
        existing_questions_clean = [q.strip() for q in existing_questions]

        if len(existing_questions_clean) == 0 or len(new_questions_clean) == 0:
            return []

        # Embed
        emb_new = self.embed_texts(new_questions_clean)
        emb_existing = self.embed_texts(existing_questions_clean)

        # Compute cosine similarity matrix (num_new x num_existing)
        # Using util.pytorch_cos_sim when available gives same results, but
        # we can compute with numpy for simplicity.
        # Normalize embeddings
        def normalize(v):
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return v / norms

        emb_new_n = normalize(emb_new)
        emb_existing_n = normalize(emb_existing)

        sim_matrix = np.dot(emb_new_n, emb_existing_n.T)

        for i, row in enumerate(sim_matrix):
            best_indices = np.argsort(row)[::-1][:top_k]
            best_score = float(row[best_indices[0]]) if best_indices.size > 0 else 0.0

            if best_score < threshold:
                continue

            # For prototype we return only top_k=1 mapping per new question
            j = int(best_indices[0])

            match = {
                'new_question_index': i + 1,
                'existing_question_index': j + 1,
                'similarity_percentage': round(best_score * 100, 2),
                'new_question_text': new_questions_clean[i][:300] + ('...' if len(new_questions_clean[i]) > 300 else ''),
                'existing_question_text': existing_questions_clean[j][:300] + ('...' if len(existing_questions_clean[j]) > 300 else ''),
                'matched_concepts': []
            }

            results.append(match)

        # Prefer higher similarity first
        results.sort(key=lambda x: x['similarity_percentage'], reverse=True)
        return results


def get_matcher_safe() -> Optional[SemanticMatcher]:
    try:
        return SemanticMatcher()
    except Exception:
        return None
