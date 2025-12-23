from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import re

import httpx
import trafilatura
from bs4 import BeautifulSoup


@dataclass
class WebLoaderService:
    timeout_s: float = 25.0
    user_agent: str = "LLM-Twin-Modern/0.1 (+RAG web ingest)"

    def fetch(self, url: str) -> Tuple[Optional[str], str]:
        """
        Fetch a URL and extract main readable text.

        Returns:
            (title, text)

        Raises:
            httpx.HTTPError on network problems
            ValueError if extraction produces empty text
        """
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        # 1) Download HTML
        with httpx.Client(
            timeout=self.timeout_s,
            headers=headers,
            follow_redirects=True,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text

        # 2) Title (best-effort)
        title: Optional[str] = None
        try:
            meta = trafilatura.extract_metadata(html, url=url)
            if meta and meta.title:
                title = meta.title.strip() or None
        except Exception:
            title = None

        # 3) Try trafilatura main-text extraction
        text = ""
        try:
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_links=False,
                output_format="txt",
                url=url,
            )
            text = (extracted or "").strip()
        except Exception:
            text = ""

        # 4) Fallback: BeautifulSoup text extraction (more brute-force)
        if not text:
            soup = BeautifulSoup(html, "html.parser")

            # Remove non-content elements (common docs site noise)
            for tag in soup(["script", "style", "noscript", "svg"]):
                tag.decompose()

            # Optional: remove typical layout containers if present
            # (safe-ish, but keep conservative)
            for tag in soup.find_all(["nav", "header", "footer", "aside"]):
                tag.decompose()

            # If the page has a <main> element, prefer it
            main = soup.find("main")
            base = main if main else soup.body if soup.body else soup

            raw = base.get_text(separator="\n")

            # Normalize whitespace
            raw = raw.replace("\r\n", "\n").replace("\r", "\n")
            raw = re.sub(r"[ \t]+\n", "\n", raw)      # trailing spaces
            raw = re.sub(r"\n{3,}", "\n\n", raw)      # collapse blank lines

            # Drop very short lines that are usually menus/buttons
            lines = []
            for ln in raw.split("\n"):
                s = ln.strip()
                if not s:
                    lines.append("")  # keep paragraph breaks
                    continue
                if len(s) <= 2:
                    continue
                # drop cookie/nav-ish micro-lines
                low = s.lower()
                if any(p in low for p in ["cookie", "privacy", "terms", "sign in", "log in"]) and len(s) < 80:
                    continue
                lines.append(s)

            cleaned = "\n".join(lines)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            text = cleaned

            # Title fallback from HTML <title>
            if not title:
                try:
                    t = soup.title.string if soup.title else None
                    title = t.strip() if t else None
                except Exception:
                    title = None

        # 5) Final guard
        if not text or len(text) < 200:
            raise ValueError(
                "Could not extract readable text from the page (empty/too short extraction). "
                "This page may be heavily JS-rendered."
            )

        return title, text
