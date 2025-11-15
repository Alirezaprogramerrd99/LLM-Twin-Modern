from __future__ import annotations
from typing import List, Tuple
from datetime import datetime, timezone

from loguru import logger
from pymongo import MongoClient, UpdateOne


class MongoDocumentStore:
    """Store raw documents & metadata in MongoDB."""

    def __init__(self, uri: str, db_name: str, collection_name: str):
        logger.info("Connecting to MongoDB at {}", uri)
        self.client = MongoClient(uri)
        self.collection = self.client[db_name][collection_name]

    def upsert_documents(self, items: List[Tuple[str, str]]) -> int:
        """
        items: list of (doc_id, text).
        Uses doc_id as _id so we don't create duplicates on re-index.
        """
        if not items:
            return 0

        # ops is lisf of UpdateOne(write) operations for bulk_write
        ops: list[UpdateOne] = []
        now = datetime.now(timezone.utc)
        
        # Prepare bulk upsert operations for each document.
        for doc_id, text in items:
            ops.append(
                UpdateOne(
                    {"_id": doc_id},
                    {
                        "$set": { # fields to update
                            "text": text,
                            "updated_at": now,
                        },
                        "$setOnInsert": { # fields to set only on insert;
                            "created_at": now,
                        },
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
