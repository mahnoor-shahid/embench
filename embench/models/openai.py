"""OpenAI embeddings.

Reads the API key from the ``OPENAI_API_KEY`` environment variable.
"""
from __future__ import annotations

import numpy as np

from .base import BaseEmbeddingModel


class OpenAIModel(BaseEmbeddingModel):
    def __init__(self, model_id: str = "text-embedding-3-small", name: str | None = None):
        super().__init__(name or model_id)
        self.model_id = model_id
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai is required for OpenAIModel. It ships with embench, "
                "so reinstall it with: pip install embench"
            ) from exc
        self._client = OpenAI()

    def _encode(self, texts: list[str]) -> np.ndarray:
        # OpenAI rejects empty strings; substitute a single space.
        cleaned = [t if t.strip() else " " for t in texts]
        resp = self._client.embeddings.create(model=self.model_id, input=cleaned)
        # Preserve input order (API returns an index per item).
        ordered = sorted(resp.data, key=lambda d: d.index)
        return np.array([d.embedding for d in ordered], dtype=np.float32)
