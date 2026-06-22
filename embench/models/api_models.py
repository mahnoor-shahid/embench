"""Cohere and Voyage AI adapters.

Both providers distinguish *query* vs *document* embeddings via an
``input_type``. For benchmarking we embed everything as documents by
default, which is the conventional choice for corpus-side comparison;
override ``input_type`` if you want query-side embeddings.

    pip install embench[cohere]   # CohereModel
    pip install embench[voyage]   # VoyageModel
"""
from __future__ import annotations

import numpy as np

from .base import BaseEmbeddingModel


class CohereModel(BaseEmbeddingModel):
    def __init__(
        self,
        model_id: str = "embed-english-v3.0",
        input_type: str = "search_document",
        name: str | None = None,
    ):
        super().__init__(name or model_id)
        self.model_id = model_id
        self.input_type = input_type
        try:
            import cohere
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "cohere is required for CohereModel. "
                "Install it with: pip install embench[cohere]"
            ) from exc
        self._client = cohere.Client()

    def _encode(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embed(
            texts=texts, model=self.model_id, input_type=self.input_type
        )
        return np.array(resp.embeddings, dtype=np.float32)


class VoyageModel(BaseEmbeddingModel):
    def __init__(
        self,
        model_id: str = "voyage-3",
        input_type: str = "document",
        name: str | None = None,
    ):
        super().__init__(name or model_id)
        self.model_id = model_id
        self.input_type = input_type
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "voyageai is required for VoyageModel. "
                "Install it with: pip install embench[voyage]"
            ) from exc
        self._client = voyageai.Client()

    def _encode(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embed(texts, model=self.model_id, input_type=self.input_type)
        return np.array(resp.embeddings, dtype=np.float32)
