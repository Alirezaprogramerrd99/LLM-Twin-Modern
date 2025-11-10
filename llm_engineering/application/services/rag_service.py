from dataclasses import dataclass
from typing import List, Tuple
from loguru import logger

from llm_engineering.application.settings import Settings
from llm_engineering.application.services.embedder import SimpleEmbedder
from llm_engineering.application.services.vector_store import InMemoryVectorStore


@dataclass
class RAGService:
    settings: Settings
    store: InMemoryVectorStore

    @classmethod
    def build(cls, settings: Settings) -> "RAGService":
        # You can pass stopwords or other options from settings later
        embedder = SimpleEmbedder(stopwords=["the", "a", "an", "and", "of", "to", "in"])
        store = InMemoryVectorStore(embedder=embedder)
        return cls(settings=settings, store=store)

    def index(self, items: List[Tuple[str, str]]) -> int:
        """Index a list of (id, text). Returns how many were added."""
        # this function adds documents to the vector store
        logger.info("Indexing {} document(s)", len(items))
        self.store.add_many(items)
        return len(items)

    def search(self, query: str, k: int = 5):
        logger.debug("Searching: query='{}', k={}", query, k)
        return self.store.search(query, k=k)
