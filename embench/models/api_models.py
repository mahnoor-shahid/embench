"""Hosted-API embedding adapters: Cohere, Voyage, Google, Hugging Face.

Cohere and Voyage distinguish *query* vs *document* embeddings via an
``input_type``. For benchmarking we embed everything as documents by
default, which is the conventional choice for corpus-side comparison;
override ``input_type`` if you want query-side embeddings.

All SDKs these adapters need ship with ``embench`` itself. API keys are read
from the environment (see ``.env.example``); call ``embench.load_env()``
first if you keep them in a ``.env`` file.
"""
from __future__ import annotations

import os

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
                "cohere is required for CohereModel. It ships with embench, "
                "so reinstall it with: pip install embench"
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
                "voyageai is required for VoyageModel. It ships with embench, "
                "so reinstall it with: pip install embench"
            ) from exc
        self._client = voyageai.Client()

    def _encode(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embed(texts, model=self.model_id, input_type=self.input_type)
        return np.array(resp.embeddings, dtype=np.float32)


class GoogleModel(BaseEmbeddingModel):
    """Gemini embeddings via the Google GenAI SDK.

    Reads the API key from ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``). The
    default ``gemini-embedding-001`` returns one vector per input; avoid
    ``gemini-embedding-2``, which aggregates a batch into a single vector.
    """

    def __init__(self, model_id: str = "gemini-embedding-001", name: str | None = None):
        super().__init__(name or model_id)
        self.model_id = model_id
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "google-genai is required for GoogleModel. It ships with "
                "embench, so reinstall it with: pip install embench"
            ) from exc
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def _encode(self, texts: list[str]) -> np.ndarray:
        resp = self._client.models.embed_content(model=self.model_id, contents=texts)
        return np.array([e.values for e in resp.embeddings], dtype=np.float32)


class HuggingFaceModel(BaseEmbeddingModel):
    """Embeddings via the Hugging Face Inference API.

    Reads the token from ``HUGGINGFACE_API_KEY`` (or ``HF_TOKEN``). For
    *local* models with no API call, use ``SentenceTransformerModel``
    instead. The Inference API embeds one text per
    call, so this loops over the batch; if a model returns token-level
    vectors they are mean-pooled into a single sentence embedding.
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        provider: str = "hf-inference",
        name: str | None = None,
    ):
        super().__init__(name or model_id)
        self.model_id = model_id
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "huggingface_hub is required for HuggingFaceModel. It ships "
                "with embench, so reinstall it with: pip install embench"
            ) from exc
        token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
        self._client = InferenceClient(model=model_id, token=token, provider=provider)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            arr = np.asarray(self._client.feature_extraction(text), dtype=np.float32)
            if arr.ndim == 2:  # token-level output -> mean pool to one vector
                arr = arr.mean(axis=0)
            vecs.append(arr)
        return np.vstack(vecs)
