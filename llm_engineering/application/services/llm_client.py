from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any
from collections.abc import Mapping

from loguru import logger
import ollama

from llm_engineering.application.settings import Settings


@dataclass
class LLMClient:
    """Thin wrapper around a local Ollama model (e.g. phi3:latest, qwen3:4b)."""
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

    @staticmethod
    def _to_dict(resp: Any) -> dict:
        """
        Normalize Ollama responses to a plain dict when possible.

        The `ollama` python client may return:
        - dict-like mappings
        - Pydantic-like objects with .model_dump()
        - objects with .dict()
        - typed objects with attributes (.message.content, .response, etc.)
        """
        if resp is None:
            return {}

        # Already a mapping/dict
        if isinstance(resp, Mapping):
            return dict(resp)

        # Pydantic v2 style
        if hasattr(resp, "model_dump") and callable(getattr(resp, "model_dump")):
            try:
                return resp.model_dump()
            except Exception:
                pass

        # Pydantic v1 style
        if hasattr(resp, "dict") and callable(getattr(resp, "dict")):
            try:
                return resp.dict()
            except Exception:
                pass

        # Fallback: best-effort from __dict__
        if hasattr(resp, "__dict__"):
            try:
                return dict(resp.__dict__)
            except Exception:
                return {}

        return {}

    @classmethod
    def _extract_text(cls, resp: Any) -> str:
        """
        Ollama can return different shapes depending on chat/generate AND client version.

        Examples:
        - chat(): {"message": {"content": "..."}}
        - generate(): {"response": "..."}
        - typed: resp.message.content or resp.response
        """
        # 1) Try attribute access first (works for typed objects)
        try:
            msg = getattr(resp, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if content is not None:
                    return str(content).strip()
        except Exception:
            pass

        try:
            direct = getattr(resp, "response", None)
            if direct is not None:
                return str(direct).strip()
        except Exception:
            pass

        # 2) Normalize to dict and parse
        d = cls._to_dict(resp)
        if not d:
            return ""

        if "message" in d:
            msg = d.get("message") or {}
            # message may be dict OR object-like
            if isinstance(msg, Mapping):
                return str(msg.get("content") or "").strip()
            # object-like
            content = getattr(msg, "content", None)
            if content is not None:
                return str(content).strip()

        if "response" in d:
            return str(d.get("response") or "").strip()

        return ""

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.1) -> str:
        """
        Use chat() first, fall back to generate() if needed.
        Returns the assistant text or "" if nothing was produced.
        """
        messages = [{"role": "user", "content": prompt}]

        try:
            client = ollama.Client(host=self.host)

            # 1) Try chat()
            chat_resp = client.chat(
                model=self.model,
                messages=messages,
                options={"num_predict": max_tokens, "temperature": temperature},
            )
            text = self._extract_text(chat_resp)
            if text:
                return text

            logger.warning(
                "Ollama chat returned empty content; falling back to generate(). model='{}' host='{}'",
                self.model,
                self.host,
            )

            # 2) Fall back to generate()
            gen_resp = client.generate(
                model=self.model,
                prompt=prompt,
                options={"num_predict": max_tokens, "temperature": temperature},
            )
            text2 = self._extract_text(gen_resp)
            if text2:
                return text2

            # Helpful debug: log shapes/keys without dumping huge content
            chat_d = self._to_dict(chat_resp)
            gen_d = self._to_dict(gen_resp)
            logger.error(
                "Ollama returned empty from both chat and generate. "
                "chat_keys={} gen_keys={} model='{}' host='{}'",
                list(chat_d.keys()),
                list(gen_d.keys()),
                self.model,
                self.host,
            )
            return ""

        except Exception as e:
            logger.exception("Error calling Ollama (model='{}', host='{}')", self.model, self.host)
            return f"[LLM error: {e}]"
