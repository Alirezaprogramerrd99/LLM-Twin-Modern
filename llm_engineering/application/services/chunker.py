from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


# doc1#chunk0: chars 0–800
# doc1#chunk1: chars 600–1400
# etc.

@dataclass
class ChunkerService:
    """
    Very simple text chunker:
    - Splits text into overlapping chunks of ~chunk_size characters.
    - chunk_overlap characters are shared between consecutive chunks.
    - Returns (chunk_id, chunk_text) pairs, where chunk_id encodes doc + index.
    """
    chunk_size: int = 800
    chunk_overlap: int = 200

    def chunk(self, doc_id: str, text: str) -> List[Tuple[str, str]]:
        """
        Given a full document, return [(chunk_id, chunk_text), ...].

        Example chunk_id: f"{doc_id}#chunk0", f"{doc_id}#chunk1", ...
        """
        text = text.strip()
        if not text:
            return []
        
        # a list of (chunk_id, chunk_text)
        chunks: list[tuple[str, str]] = []

        start = 0
        idx = 0
        n = len(text)

        while start < n:
            end = min(start + self.chunk_size, n)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = f"{doc_id}#chunk{idx}"
                chunks.append((chunk_id, chunk_text))
                idx += 1

            if end == n:
                break  # done

            # move start forward with overlap
            start = end - self.chunk_overlap

        return chunks