from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from loguru import logger

from llm_engineering.application.settings import Settings
from llm_engineering.application.services.st_embedder import STEmbedder
from llm_engineering.application.services.vector_store_dense import DenseVectorStore
from llm_engineering.application.services.vector_store_qdrant import QdrantVectorStore
from llm_engineering.application.services.mongo_store import MongoDocumentStore, MongoInteractionStore
from llm_engineering.application.services.llm_client import LLMClient




@dataclass
class RAGService:
    settings: Settings
    embedder: STEmbedder
    # store is either DenseVectorStore or QdrantVectorStore
    store: object
    backend: str  # "dense" or "qdrant"
    
    # Optional MongoDB document store for metadata (not used in this example)
    doc_store: MongoDocumentStore | None = None
    history_store: MongoInteractionStore | None = None
    
    # Optional LLM client for generation (not used in this example)
    llm: LLMClient | None = None
    

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
        history_store: MongoInteractionStore | None = None
        
        if settings.mongo_uri and settings.use_mongo:
            logger.info("Using MongoDB document store at {}", settings.mongo_uri)
            doc_store = MongoDocumentStore(
                uri=settings.mongo_uri,
                db_name=settings.mongo_db_name,
                collection_name=settings.mongo_collection_docs,
            )
            
            history_store = MongoInteractionStore(
                uri=settings.mongo_uri,
                db_name=settings.mongo_db_name,
                collection_name=settings.mongo_collection_history,
            )
            
        # --- optional LLM client (Ollama) ---
        llm_client = LLMClient.from_settings(settings)

        return cls(settings=settings, 
                   embedder=embedder, 
                   store=store, 
                   backend=backend, 
                   doc_store=doc_store if settings.use_mongo else None, 
                   history_store=history_store if settings.use_mongo else None,
                   llm=llm_client)

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
    
    
    
    def ask(self, question: str, k: int = 3) -> dict:
        """
        Full RAG step:
        1) retrieve top-k docs
        2) build a context string
        3) call the LLM to generate an answer
        """
        if self.llm is None:
            # Graceful failure if no API key configured
            return {
                "question": question,
                "answer": (
                    "LLM is not configured. Set OPENAI_API_KEY and OPENAI_MODEL "
                    "in your environment to enable /ask."
                ),
                "sources": [],
            }

        # example: hits = [
        #   ("doc3", 0.92, "Cats are natural predators and often chase small animals like mice."),
        #   ("doc1", 0.81, "Mice are common prey for domestic cats due to their size and movement."),
        #   ("doc7", 0.60, "Dogs usually do not chase mice as often as cats do.")]
    
        hits = self.search(question, k=k)  # [(doc_id, score, text), ...]

        if not hits:
            return {
                "question": question,
                "answer": "I couldn't find any relevant information in the knowledge base.",
                "sources": [],
            }

        # Build context for the LLM
        context_lines = []
        for idx, (doc_id, score, text) in enumerate(hits, start=1):
            context_lines.append(
                f"[{idx}] (score={score:.3f}, id={doc_id}) {text}"
            )  # for each creates line like: [1] (score=0.923, id=doc3) Cats are natural predators...
        context = "\n\n".join(context_lines)

        prompt = (
            "You are an assistant that answers questions using the provided context.\n"
            "Use ONLY the information in the context. If the answer is not clearly in the "
            "context, say you don't know.\n"
            "When possible, mention which snippet you used, e.g. [1] or [2].\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        # Generate answer using the LLM
        answer_text = self.llm.generate(prompt)
        
        
         # log to Mongo if configured
        if self.history_store is not None:
            self.history_store.log_interaction(question, answer_text, hits)

        return {
            "question": question,
            "answer": answer_text,
            "sources": [
                {"id": doc_id, "score": score, "text": text}
                for (doc_id, score, text) in hits
            ],
        }