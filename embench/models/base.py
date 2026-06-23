"""The single interface every embedding source is wrapped behind.

If you can implement ``_encode``, your model works with every task and
report in this package. That is the whole point of the abstraction.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class EncodeStats:
    """Per-model encoding counters, used to report speed and cost.

    ``n_texts`` is everything the caller asked to encode; ``n_encoded`` is
    what actually reached :meth:`_encode` (i.e. excludes cache hits), so it
    is a faithful proxy for API cost. ``seconds`` is wall time spent
    encoding only -- not metric computation.
    """

    n_texts: int = 0
    n_encoded: int = 0
    n_chars: int = 0
    seconds: float = 0.0

    def reset(self) -> None:
        self.n_texts = self.n_encoded = self.n_chars = 0
        self.seconds = 0.0

    @property
    def texts_per_sec(self) -> float:
        return self.n_encoded / self.seconds if self.seconds > 0 else 0.0


class BaseEmbeddingModel(ABC):
    """Adapter base class.

    Subclasses implement :meth:`_encode` for a single list of texts.
    Batching, progress, and array assembly are handled here so adapters
    stay tiny.
    """

    def __init__(self, name: str):
        self.name = name
        self.stats = EncodeStats()

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
        self.stats.n_texts += len(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        chunks: list[np.ndarray] = []
        n = len(texts)
        for start in range(0, n, batch_size):
            batch = texts[start : start + batch_size]
            t0 = time.perf_counter()
            encoded = np.asarray(self._encode(batch), dtype=np.float32)
            self.stats.seconds += time.perf_counter() - t0
            self.stats.n_encoded += len(batch)
            self.stats.n_chars += sum(len(t) for t in batch)
            chunks.append(encoded)
            if show_progress:
                done = min(start + batch_size, n)
                print(f"  [{self.name}] encoded {done}/{n}", end="\r")
        if show_progress:
            print()
        return np.vstack(chunks)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
