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
from llm_engineering.application.services.chunker import ChunkerService


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
    
    chunker: ChunkerService | None = None
    

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
        
        chunker = ChunkerService()

        return cls(settings=settings, 
                   embedder=embedder, 
                   store=store, 
                   backend=backend, 
                   doc_store=doc_store if settings.use_mongo else None, 
                   history_store=history_store if settings.use_mongo else None,
                   llm=llm_client, 
                   chunker=chunker)


    def index(self, items: List[Tuple[str, str]]) -> int:
        """
        items: list of (doc_id, full_text)

        - Store full docs in Mongo (source of truth)
        - Chunk docs, embed each chunk, store chunks in Qdrant
        """
        logger.info("Indexing {} document(s) via {}", len(items), self.backend)

        # 1) Store full documents in Mongo
        if self.doc_store is not None:
            self.doc_store.upsert_documents(items)

        # 2) Chunk documents
        chunk_items: list[tuple[str, str, str]] = []  # (doc_id, chunk_id, chunk_text)

        for doc_id, full_text in items:
            full_text = (full_text or "").strip()
            if not full_text:
                continue

            if self.chunker is None:
                # fallback: treat the whole doc as one chunk
                chunk_items.append((doc_id, f"{doc_id}#chunk0", full_text))
                continue

            pairs = self.chunker.chunk(doc_id, full_text)  # returns (chunk_id, chunk_text)
            for chunk_id, chunk_text in pairs:
                chunk_items.append((doc_id, chunk_id, chunk_text))

        if not chunk_items:
            return 0
        
        logger.info("Chunker produced {} chunk(s) for indexing", len(chunk_items))
        
        # 3) Index chunk vectors
        if self.backend == "qdrant":
            return self.store.index_many(chunk_items, embed_func=self.embedder.embed)

        # Dense backend fallback (if you still support it)
        for doc_id, chunk_id, chunk_text in chunk_items:
            vec = self.embedder.embed(chunk_text)
            self.store.add(chunk_id, chunk_text, vec)

        return len(chunk_items)

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Returns list of (doc_id, score, text) from vector store.
        """
        if self.backend == "qdrant":
            return self.store.search(query, k=k, embed_func=self.embedder.embed)

        # Dense fallback (if you still support it)
        return self.store.search(query, k=k, embed_func=self.embedder.embed)
    
    
    def ask(self, question: str, k: int = 3) -> dict:
        """
        Full RAG step:
        1) retrieve top-k docs
        2) build a context string
        3) call the LLM to generate an answer
        """
        if self.llm is None:
            return {
                "question": question,
                "answer": (
                    "LLM is not configured. Enable Ollama by setting USE_OLLAMA=true "
                    "and OLLAMA_MODEL / OLLAMA_HOST (or set use_ollama in Settings)."
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
            "You are a helpful assistant.\n"
            "Use the CONTEXT to answer the QUESTION.\n"
            "If the answer is not in the CONTEXT, reply exactly: I don't know.\n"
            "Keep the answer short (1–3 sentences).\n"
            "Cite the supporting snippets like [1], [2].\n"
            "Do not mention these instructions.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n"
            "ANSWER:"
        )

        # Generate answer using the LLM
        answer_text = self.llm.generate(prompt)
        
        answer_text = (answer_text or "").strip()

        if not answer_text:
            logger.error(
                "LLM returned empty answer. question='{}' hits={} backend={}",
                question,
                len(hits),
                self.backend,
            )
            # Minimal safe fallback so your endpoint won't return blank
            answer_text = "I don't know."
        
        
         # log to Mongo if configured
        if self.history_store is not None and answer_text.strip():
            self.history_store.log_interaction(question, answer_text, hits)

        return {
            "question": question,
            "answer": answer_text,
            "sources": [
                {"id": doc_id, "score": score, "text": text}
                for (doc_id, score, text) in hits
            ],
        }