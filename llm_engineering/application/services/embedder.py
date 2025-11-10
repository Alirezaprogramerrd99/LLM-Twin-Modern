# Minimal "embedder": tokenize → term-frequency dict (no external libs)
from collections import Counter
from typing import Dict, List
import re


class SimpleEmbedder:
    def __init__(self, stopwords: List[str] | None = None):
        self.stopwords = set(stopwords or [])

    def _tokens(self, text: str) -> List[str]:
        # lowercase, keep words only
        toks = re.findall(r"[a-zA-Z]+", text.lower())  # toks will be like ['this', 'is', 'a', 'test']
        # for filltering stopwords
        return [t for t in toks if t not in self.stopwords]

    def embed(self, text: str) -> Dict[str, float]:
        # term frequency vector stored as a dict: {term: tf}
        toks = self._tokens(text)
        counts = Counter(toks)
        if not counts:
            return {}
        # normalize by max term frequency (common TF variant)
        max_tf = max(counts.values())
        return {term: c / max_tf for term, c in counts.items()} # return normalized tf vector a dict containing terms and their frequencies