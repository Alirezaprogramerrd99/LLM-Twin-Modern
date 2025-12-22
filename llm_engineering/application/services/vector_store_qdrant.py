from __future__ import annotations
from typing import List, Tuple
from loguru import logger
import numpy as np
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


class QdrantVectorStore:
    def __init__(self, url: str | None, api_key: str | None, collection: str, vector_size: int):
        self.collection = collection
        url_str = str(url) if url is not None else None

        self.client = QdrantClient(url=url_str, api_key=api_key)
        self._ensure_collection(vector_size)
        
    def _point_id(self, doc_id: str, chunk_id: str) -> str:
        """
        Deterministically map (doc_id, chunk_id) to a UUID string.
        Same chunk -> same ID every time.
        """
        key = f"{doc_id}:{chunk_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    def _ensure_collection(self, dim: int) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            logger.info("Qdrant collection '{}' exists", self.collection)
            return
        logger.info("Creating Qdrant collection '{}' (dim={})", self.collection, dim)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
        )

    def index_many(self, items: List[Tuple[str, str, str]], embed_func) -> int:
        """
        items: list of (doc_id, chunk_id, text).
        - Use deterministic UUIDs for Qdrant point IDs.
        - Store doc_id, chunk_id, and text in payload.
        """
        ids = []
        vectors = []
        payloads = []

        for doc_id, chunk_id, text in items:
            vec = embed_func(text)
            if isinstance(vec, np.ndarray):
                vec = vec.astype("float32").tolist()

            point_id = self._point_id(doc_id, chunk_id)  # deterministic UUID

            ids.append(point_id)
            vectors.append(vec)
            payloads.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": text,
                }
            )

        self.client.upsert(
            collection_name=self.collection,
            points=rest.Batch(ids=ids, vectors=vectors, payloads=payloads),
        )
        return len(items)
    

    def search(self, query: str, k: int, embed_func):
        qv = embed_func(query)
        if isinstance(qv, np.ndarray):
            qv = qv.astype("float32").tolist()

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=qv,
            limit=k,
        )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            text = payload.get("text", "")
            doc_id = payload.get("doc_id", str(hit.id))  # fall back to numeric id if missing
            # chunk_id = payload.get("chunk_id", "")
            results.append((doc_id, float(hit.score), text))
        return results
