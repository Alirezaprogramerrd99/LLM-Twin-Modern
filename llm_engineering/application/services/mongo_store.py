from __future__ import annotations
from typing import Any, Iterable, Union, Dict, Tuple, List, Optional
from datetime import datetime, timezone

from loguru import logger
from pymongo import MongoClient, UpdateOne


class MongoDocumentStore:
    """Store raw documents & metadata in MongoDB."""
    

    DocTuple = Tuple[str, str]                 # (doc_id, text)
    DocDict = Dict[str, Any]                  # {"id":..., "text":..., "source":...}
    DocItem = Union[DocTuple, DocDict]

    def __init__(self, uri: str, db_name: str, collection_name: str):
        logger.info("Connecting to MongoDB at {}", uri)
        self.client = MongoClient(uri)
        self.collection = self.client[db_name][collection_name]

    def upsert_documents(self, items: List[DocItem]) -> int:
        """
        Accepts either:
        - (doc_id, text)
        - {"id": doc_id, "text": text, "source": "...", "title": "...", "url": "...", "tags": [...]}

        Uses doc_id as _id so we don't create duplicates on re-index.
        """
        if not items:
            return 0

        ops: list[UpdateOne] = []
        now = datetime.now(timezone.utc)

        for it in items:
            # --- normalize input ---
            if isinstance(it, tuple):
                doc_id, text = it
                meta = {"source": "manual", "title": None, "url": None, "tags": []}
            elif isinstance(it, dict):
                doc_id = it["id"]
                text = it["text"]
                meta = {
                    "source": it.get("source", "manual"),
                    "title": it.get("title"),
                    "url": it.get("url"),
                    "tags": it.get("tags") or [],
                }
            else:
                raise TypeError(f"Unsupported document item type: {type(it)}")

            # --- build update doc ---
            set_doc = {
                "text": text,
                "source": meta["source"],
                "tags": meta["tags"],
                "updated_at": now,
            }

            # only store optional fields if provided
            if meta["title"] is not None:
                set_doc["title"] = meta["title"]
            if meta["url"] is not None:
                set_doc["url"] = meta["url"]

            ops.append(
                UpdateOne(
                    {"_id": doc_id},
                    {
                        "$set": set_doc,
                        "$setOnInsert": {"created_at": now},
                    },
                    upsert=True,
                )
            )

        result = self.collection.bulk_write(ops, ordered=False)
        count = (result.upserted_count or 0) + (result.modified_count or 0)
        logger.info("Mongo upserted/updated {} document(s)", count)
        return count

    def get_texts(self, doc_ids: list[str]) -> dict[str, str]:
        """Fetch texts for given doc_ids. (We may use this later for /ask)."""
        if not doc_ids:
            return {}
        # Query MongoDB for documents with _id in doc_ids, projecting only the text field (1).
        cursor = self.collection.find({"_id": {"$in": doc_ids}}, {"text": 1})
        
        # return a dict mapping doc_id to text
        # doc.get("text", "") ensures we return an empty string if text is missing.
        return {str(doc["_id"]): doc.get("text", "") for doc in cursor}
    
class MongoInteractionStore:
    """Store /ask interactions (question, answer, sources) in MongoDB."""

    def __init__(self, uri: str, db_name: str, collection_name: str):
        logger.info("Connecting to MongoDB (interactions) at {}", uri)
        self.client = MongoClient(uri)
        self.collection = self.client[db_name][collection_name]

    def log_interaction(
        self,
        question: str,
        answer: str,
        hits: List[Tuple[str, float, str]],
    ) -> None:
        now = datetime.now(timezone.utc)
        doc = {
            "question": question,
            "answer": answer,
            "sources": [
                {"id": doc_id, "score": score, "text": text}
                for (doc_id, score, text) in hits
            ],
            "created_at": now,
        }
        self.collection.insert_one(doc)
        logger.info("Logged interaction for question: {!r}", question[:80])

    def recent(self, limit: int = 20) -> list[dict]:
        cursor = (
            self.collection
            .find({}, {"question": 1, "answer": 1, "sources": 1, "created_at": 1})
            .sort("created_at", -1)
            .limit(limit)
        )
        return list(cursor)
