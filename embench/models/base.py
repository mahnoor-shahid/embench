"""The single interface every embedding source is wrapped behind.

If you can implement ``_encode``, your model works with every task and
report in this package. That is the whole point of the abstraction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class BaseEmbeddingModel(ABC):
    """Adapter base class.

    Subclasses implement :meth:`_encode` for a single list of texts.
    Batching, progress, and array assembly are handled here so adapters
    stay tiny.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into a 2-D float array (n_texts, dim)."""
        raise NotImplementedError

    def encode(
        self,
        texts: Iterable[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        chunks: list[np.ndarray] = []
        n = len(texts)
        for start in range(0, n, batch_size):
            batch = texts[start : start + batch_size]
            chunks.append(np.asarray(self._encode(batch), dtype=np.float32))
            if show_progress:
                done = min(start + batch_size, n)
                print(f"  [{self.name}] encoded {done}/{n}", end="\r")
        if show_progress:
            print()
        return np.vstack(chunks)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
