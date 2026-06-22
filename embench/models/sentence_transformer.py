"""Local models via sentence-transformers.  ``pip install embench[local]``"""
from __future__ import annotations

import numpy as np

from .base import BaseEmbeddingModel


class SentenceTransformerModel(BaseEmbeddingModel):
    def __init__(self, model_id: str, device: str | None = None, name: str | None = None):
        super().__init__(name or model_id)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerModel. "
                "Install it with: pip install embench[local]"
            ) from exc
        self._model = SentenceTransformer(model_id, device=device)

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=False
        )
