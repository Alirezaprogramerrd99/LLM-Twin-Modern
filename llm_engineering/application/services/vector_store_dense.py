from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # Both expected normalized; still handle edge cases.
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class DenseVectorStore:
    """Stores dense vectors in memory. Not persistent; ideal for learning."""
    def __init__(self):
        # id -> (raw_text, vector)
        self._docs: Dict[str, Tuple[str, np.ndarray]] = {}

    def add(self, doc_id: str, text: str, vector: np.ndarray) -> None:
        self._docs[doc_id] = (text, vector)

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[str, float, str]]:
        results: List[Tuple[str, float, str]] = []
        for doc_id, (raw, vec) in self._docs.items():
            score = _cosine(query_vec, vec)
            results.append((doc_id, score, raw))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
