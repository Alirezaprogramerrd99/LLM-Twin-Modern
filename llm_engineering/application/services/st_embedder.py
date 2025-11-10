from __future__ import annotations
from typing import List
from loguru import logger

# Sentence Transformers (uses PyTorch under the hood)
from sentence_transformers import SentenceTransformer
import numpy as np


class STEmbedder:
    """Thin wrapper around SentenceTransformer to embed text → np.ndarray."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        logger.info("Loading sentence-transformer model: {}", model_name)
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, text: str) -> np.ndarray:
        # Returns shape (d,)
        vec = self.model.encode(text, normalize_embeddings=True)  # cosine-ready
        # ensure 1D np.ndarray float32
        return np.asarray(vec, dtype=np.float32).reshape(-1)
