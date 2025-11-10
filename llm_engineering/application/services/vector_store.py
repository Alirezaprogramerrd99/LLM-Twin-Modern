from __future__ import annotations
from typing import Dict, List, Tuple
import math


def _cosine_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    # cosine = dot(a,b) / (||a|| * ||b||) for sparse dicts
    if not a or not b:
        return 0.0
    dot = 0.0
    for term, va in a.items():
        vb = b.get(term)
        if vb is not None:
            dot += va * vb
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """Stores (id, text, vector) in memory. Not persistent; great for learning."""

    def __init__(self, embedder):
        self.embedder = embedder
        #_docs is a dictionary that maps document IDs to a tuple containing the raw text and its corresponding term-frequency vector.
        self._docs: Dict[str, Tuple[str, Dict[str, float]]] = {}
        # id -> (raw_text, tf_vector)

    def add(self, doc_id: str, text: str) -> None:
        # Example: doc_id="doc1", text="This is a test.
        
        # store.add("d1", "Cats chase mice.")
        # _docs = {"d1": ("Cats chase mice.", {"cats":1.0,"chase":1.0,"mice":1.0})
        vec = self.embedder.embed(text)
        self._docs[doc_id] = (text, vec)

    def add_many(self, items: List[Tuple[str, str]]) -> None:
        
        # items = [("doc1", "This is a test."), ("doc2", "Another document.")]
        # store.add_many(items)
        # _docs = {"doc1": ("This is a test.", {...}), "doc2": ("Another document.", {...})}
        
        for doc_id, text in items:
            self.add(doc_id, text)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:  # 3-element tuple
        # Example: query="test", k=3
        qv = self.embedder.embed(query)
        
        # scored = [ ("doc_id", cosine_score, "raw_text"), ...]

        scored: List[Tuple[str, float, str]] = []
        for doc_id, (raw, vec) in self._docs.items():
            score = _cosine_sim(qv, vec)
            
            # scored = [("d1", 0.70, "Cats chase mice"), ("d2", 0.00, "Dogs like bones"), ("d3", 0.98, "A cat likes to chase") ]
            scored.append((doc_id, score, raw))
            
        
        # sort by score desc
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
