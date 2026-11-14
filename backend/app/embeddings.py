"""Gemini text embeddings with explicit graceful fallback behavior."""
import os
from functools import lru_cache

from .llm import _client, LLMNotConfiguredError, LLMGenerationError

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float] | None:
    if not os.getenv("GEMINI_API_KEY", "").strip():
        return None
    try:
        response = _client().models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config={"task_type": task_type, "output_dimensionality": 768},
        )
        if not response.embeddings:
            return None
        return list(response.embeddings[0].values)
    except (LLMNotConfiguredError, LLMGenerationError):
        return None
    except Exception:
        # Retrieval remains available via TF-IDF if the embedding endpoint is
        # temporarily unavailable or the account has exhausted its free quota.
        return None


def embed_query(text: str) -> list[float] | None:
    return embed_text(text, task_type="RETRIEVAL_QUERY")
