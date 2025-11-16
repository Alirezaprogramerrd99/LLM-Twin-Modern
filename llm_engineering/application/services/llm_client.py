from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from loguru import logger
import ollama 

from llm_engineering.application.settings import Settings


# We can make this class polymorphic later if we want to add more LLM backends.
@dataclass
class LLMClient:
    """Thin wrapper around a local Ollama model (e.g. qwen3:4b)."""
    model: str
    host: str

    @classmethod
    def from_settings(cls, settings: Settings) -> Optional["LLMClient"]:
        if not settings.use_ollama:
            logger.warning("use_ollama is False; LLMClient disabled.")
            return None

        logger.info(
            "Initialized Ollama LLMClient with model '{}' at {}",
            settings.ollama_model,
            settings.ollama_host,
        )
        return cls(model=settings.ollama_model, host=settings.ollama_host)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """
        Simple text-in → text-out using Ollama's chat endpoint.
        """
        # Build a single-turn chat message
        messages = [{"role": "user", "content": prompt}]

        # The ollama library uses the global host by default; we can override via Client if needed.
        # For now, assume default localhost:11434.
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            )
        except Exception as e:
            logger.error("Error calling Ollama: {}", e)
            return f"[LLM error: {e}]"

        # Response is a dict; the main text is in message.content
        msg = response.get("message") or {}
        return msg.get("content", "").strip()
