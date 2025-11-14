from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from loguru import logger
import numpy as np

from llm_engineering.application.settings import Settings
from llm_engineering.application.services.st_embedder import STEmbedder
from llm_engineering.application.services.vector_store_dense import DenseVectorStore


@dataclass
class RAGService:
    settings: Settings
    embedder: STEmbedder
    store: DenseVectorStore

    @classmethod
    def build(cls, settings: Settings) -> "RAGService":
        model_name = getattr(settings, "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        embedder = STEmbedder(model_name=model_name, device=None)
        store = DenseVectorStore()
        return cls(settings=settings, embedder=embedder, store=store)

    def index(self, items: List[Tuple[str, str]]) -> int:
        logger.info("Indexing {} document(s)", len(items))
        
        # items: List of (doc_id, text); embed and add to store
        for doc_id, text in items:
            vec = self.embedder.embed(text)  # np.ndarray (normalized)
            self.store.add(doc_id, text, vec)
        return len(items)

    def search(self, query: str, k: int = 5):
        qv = self.embedder.embed(query)
        return self.store.search(qv, k=k)
