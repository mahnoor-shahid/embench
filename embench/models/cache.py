"""Disk-backed cache so re-running a comparison never re-pays API costs.

Wrap any model:

    model = CachedModel(OpenAIModel("text-embedding-3-small"))

Embeddings are keyed by (model name + text) so different models never
collide, and the cache is written to disk as a single compressed file
per model.
"""
from __future__ import annotations

import os
import time

import numpy as np

from ..utils import text_hash
from .base import BaseEmbeddingModel


class CachedModel(BaseEmbeddingModel):
    def __init__(self, model: BaseEmbeddingModel, cache_dir: str = ".embench_cache"):
        super().__init__(model.name)
        self.model = model
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._path = os.path.join(cache_dir, f"{text_hash(model.name)}.npz")
        self._store: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            data = np.load(self._path, allow_pickle=False)
            for key in data.files:
                self._store[key] = data[key]

    def _save(self) -> None:
        np.savez_compressed(self._path, **self._store)

    def _encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        # Not used directly: we override encode() to cache per text.
        return self.model._encode(texts)

    def encode(self, texts, batch_size: int = 32, show_progress: bool = False):
        texts = list(texts)
        # Count every requested text, but only the cache misses below count
        # as actually encoded -- so stats reflect the real (cached) cost.
        self.stats.n_texts += len(texts)
        keys = [text_hash(t, salt=self.model.name) for t in texts]
        missing = [t for t, k in zip(texts, keys) if k not in self._store]

        if missing:
            t0 = time.perf_counter()
            fresh = self.model.encode(
                missing, batch_size=batch_size, show_progress=show_progress
            )
            self.stats.seconds += time.perf_counter() - t0
            self.stats.n_encoded += len(missing)
            self.stats.n_chars += sum(len(t) for t in missing)
            for t, vec in zip(missing, fresh):
                self._store[text_hash(t, salt=self.model.name)] = vec
            self._save()

        return np.vstack([self._store[k] for k in keys]).astype(np.float32)
