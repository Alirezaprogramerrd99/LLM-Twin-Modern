from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from loguru import logger

from llm_engineering.application.settings import Settings
from llm_engineering.application.services.st_embedder import STEmbedder
from llm_engineering.application.services.vector_store_dense import DenseVectorStore
from llm_engineering.application.services.vector_store_qdrant import QdrantVectorStore
from llm_engineering.application.services.mongo_store import MongoDocumentStore


@dataclass
class RAGService:
    settings: Settings
    embedder: STEmbedder
    # store is either DenseVectorStore or QdrantVectorStore
    store: object
    backend: str  # "dense" or "qdrant"
    
    # Optional MongoDB document store for metadata (not used in this example)
    doc_store: MongoDocumentStore | None = None

    @classmethod
    def build(cls, settings: Settings) -> "RAGService":
        embedder = STEmbedder(model_name=settings.embedding_model_name, device=None)

        use_qdrant = bool(settings.use_qdrant or settings.qdrant_url)
        if use_qdrant:
            logger.info("Using Qdrant backend")
            
            store = QdrantVectorStore(
                url=str(settings.qdrant_url),            # type: ignore[arg-type]
                api_key=settings.qdrant_api_key,
                collection=settings.qdrant_collection,
                vector_size=settings.embedding_dim,
            )
            backend = "qdrant"
        else:
            logger.info("Using in-memory dense backend")
            store = DenseVectorStore()
            backend = "dense"
            
        # --- optional Mongo doc store ---
        
        # Initialize MongoDocumentStore if mongo_uri is provided and use_mongo is True (in settings).
        doc_store: MongoDocumentStore | None = None
        if settings.mongo_uri and settings.use_mongo:
            logger.info("Using MongoDB document store at {}", settings.mongo_uri)
            doc_store = MongoDocumentStore(
                uri=settings.mongo_uri,
                db_name=settings.mongo_db_name,
                collection_name=settings.mongo_collection_docs,
            )

        return cls(settings=settings, embedder=embedder, store=store, backend=backend, doc_store=doc_store if settings.use_mongo else None)

    def index(self, items: List[Tuple[str, str]]) -> int:
        logger.info("Indexing {} document(s) via {}", len(items), self.backend)
        
        # Optional MongoDB document store
        if self.doc_store is not None:
            self.doc_store.upsert_documents(items)
        
        
        if self.backend == "qdrant":
            return self.store.index_many(items, embed_func=self.embedder.embed)
        
        # dense in-memory
        for doc_id, text in items:
            self.store.add(doc_id, text, self.embedder.embed(text))
        return len(items)

    def search(self, query: str, k: int = 5):
        if self.backend == "qdrant":
            return self.store.search(query, k=k, embed_func=self.embedder.embed)
        
        
        # dense in-memory
        qv = self.embedder.embed(query)
        return self.store.search(qv, k=k)
