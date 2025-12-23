from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import re


@dataclass
class ChunkerService:
    """
    Web-first chunker (still simple + debuggable):
    - Normalizes whitespace
    - Splits into paragraph-like blocks (blank lines / line groups)
    - Drops obvious boilerplate-ish lines
    - Accumulates blocks into chunks by size targets
    - Uses block overlap (meaningful overlap)
    Returns (chunk_id, chunk_text) pairs.
    """

    target_chars = 350
    min_chars    = 120
    max_chars    = 700
    overlap_blocks = 1

    def chunk(self, doc_id: str, text: str) -> List[Tuple[str, str]]:
        text = self._normalize(text)
        if not text:
            return []

        blocks = self._to_blocks(text)
        if not blocks:
            return []

        # Merge tiny blocks so embeddings have enough signal
        blocks = self._merge_tiny_blocks(blocks)

        chunks: list[tuple[str, str]] = []
        current_blocks: list[str] = []
        current_len = 0
        idx = 0

        def flush():
            nonlocal idx, current_blocks, current_len
            if not current_blocks:
                return
            chunk_text = "\n\n".join(current_blocks).strip()
            if chunk_text:
                chunk_id = f"{doc_id}#chunk{idx}"
                chunks.append((chunk_id, chunk_text))
                idx += 1

        for b in blocks:
            blen = len(b)

            # If a single block is huge, split it (sentence-ish fallback)
            if blen > self.max_chars:
                for piece in self._split_large_block(b):
                    self._append_block(piece, chunks, doc_id, current_blocks_ref=lambda: current_blocks,
                                       set_current_blocks=lambda v: self._set_list("current_blocks", v, locals()),
                                       current_len_ref=lambda: current_len,
                                       set_current_len=lambda v: self._set_int("current_len", v, locals()),
                                       flush=flush)
                continue

            # If adding this block would exceed max, flush current chunk first
            if current_len > 0 and (current_len + blen + 2) > self.max_chars:
                flush()

                # overlap: carry last N blocks into next chunk
                if self.overlap_blocks > 0 and chunks:
                    prev_text = chunks[-1][1]
                    prev_blocks = prev_text.split("\n\n")
                    carry = prev_blocks[-self.overlap_blocks:]
                    current_blocks = carry[:]  # new list
                    current_len = sum(len(x) for x in current_blocks) + 2 * max(0, len(current_blocks) - 1)
                else:
                    current_blocks = []
                    current_len = 0

            # Append block
            current_blocks.append(b)
            current_len += blen + (2 if current_len > 0 else 0)

            # If we reached target and we're above min, flush
            if current_len >= self.target_chars and current_len >= self.min_chars:
                flush()
                # overlap carry
                if self.overlap_blocks > 0 and chunks:
                    prev_text = chunks[-1][1]
                    prev_blocks = prev_text.split("\n\n")
                    carry = prev_blocks[-self.overlap_blocks:]
                    current_blocks = carry[:]
                    current_len = sum(len(x) for x in current_blocks) + 2 * max(0, len(current_blocks) - 1)
                else:
                    current_blocks = []
                    current_len = 0

        # Flush remaining
        flush()

        # Final pass: self-contained start heuristic (cheap improvement)
        chunks = self._fix_pronoun_starts(chunks)

        return chunks

    # ---------------- helpers ----------------

    def _normalize(self, text: str) -> str:
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        # remove zero-width spaces
        text = text.replace("\u200b", " ").replace("\ufeff", " ")
        # collapse trailing spaces per line
        text = "\n".join(line.rstrip() for line in text.splitlines())
        # collapse many blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _to_blocks(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        # Case A: we already have paragraph breaks
        if "\n\n" in text:
            raw_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
            blocks: list[str] = []
            for b in raw_blocks:
                lines = [ln.strip() for ln in b.split("\n") if ln.strip()]
                lines = [ln for ln in lines if not self._is_boilerplate_line(ln)]
                if not lines:
                    continue

                # keep headings alone
                if len(lines) == 1 and self._looks_like_heading(lines[0]):
                    blocks.append(lines[0])
                else:
                    blocks.append(" ".join(lines))
            return blocks

        # Case B: no double-newlines → treat single newlines as structure
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        lines = [ln for ln in lines if not self._is_boilerplate_line(ln)]
        if not lines:
            return []

        blocks: list[str] = []
        current: list[str] = []

        def flush():
            nonlocal current
            if current:
                blocks.append(" ".join(current).strip())
                current = []

        for ln in lines:
            # headings become their own block boundary
            if self._looks_like_heading(ln) and len(ln) <= 120:
                flush()
                blocks.append(ln)
                continue

            current.append(ln)

            # flush once the block is "big enough"
            if sum(len(x) for x in current) >= 300:
                flush()

        flush()
        return blocks



    def _is_boilerplate_line(self, line: str) -> bool:
        l = line.lower()

        # very short junk / nav-like items
        if len(line) <= 2:
            return True

        # cookie/privacy common snippets
        bad_phrases = [
            "cookie", "privacy policy", "terms of service", "accept all", "manage cookies",
            "subscribe", "sign in", "log in", "newsletter", "all rights reserved"
        ]
        if any(p in l for p in bad_phrases) and len(line) < 120:
            return True

        # nav-like lines (lots of separators or menu-ish)
        if re.fullmatch(r"[-–—_=•\s]{5,}", line):
            return True

        # super link-y / breadcrumb-ish
        if line.count(" / ") >= 3 and len(line) < 120:
            return True

        return False

    def _looks_like_heading(self, line: str) -> bool:
        # e.g., "Overview", "Step 3", "Installation", "### Title"
        if line.startswith("#"):
            return True
        if len(line) <= 80 and re.match(r"^(step|chapter|overview|introduction|installation)\b", line.lower()):
            return True
        # Title Case-ish short lines
        if len(line) <= 60 and sum(ch.isupper() for ch in line) >= 2:
            return True
        return False

    def _merge_tiny_blocks(self, blocks: List[str]) -> List[str]:
        merged: list[str] = []
        buf = ""
        for b in blocks:
            if not buf:
                buf = b
                continue
            # merge if buffer too small
            if len(buf) < self.min_chars:
                buf = buf + "\n\n" + b
            else:
                merged.append(buf)
                buf = b
        if buf:
            merged.append(buf)
        return merged

    def _split_large_block(self, block: str) -> List[str]:
        # Sentence-ish split on . ! ? followed by space/newline (very simple)
        parts = re.split(r"(?<=[\.\!\?])\s+", block)
        pieces: list[str] = []
        cur = ""
        for p in parts:
            if not p:
                continue
            if len(cur) + len(p) + 1 <= self.max_chars:
                cur = (cur + " " + p).strip()
            else:
                if cur:
                    pieces.append(cur.strip())
                cur = p.strip()
        if cur:
            pieces.append(cur.strip())
        return pieces

    def _fix_pronoun_starts(self, chunks: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        pronouns = ("it ", "this ", "they ", "these ", "also ", "therefore ", "however ")
        fixed: list[tuple[str, str]] = []
        prev_text = ""
        for chunk_id, chunk_text in chunks:
            start = chunk_text.strip().lower()
            if prev_text and start.startswith(pronouns):
                # prepend a short tail of prev chunk (last paragraph) to give context
                tail = prev_text.split("\n\n")[-1].strip()
                chunk_text = (tail + "\n\n" + chunk_text).strip()
            fixed.append((chunk_id, chunk_text))
            prev_text = chunk_text
        return fixed
