"""A dependency-free embedding model.

It uses the *hashing trick*: each word is hashed into one of ``dim`` buckets
and counted. This is not a good embedding model, but it is deterministic,
needs no downloads or API keys, and crucially it makes *similar texts get
similar vectors* -- so example output and tests produce sensible numbers
rather than noise. Use it as a baseline and a test fixture.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from .base import BaseEmbeddingModel

_TOKEN = re.compile(r"[a-z0-9]+")


def _stable_hash(token: str) -> int:
    # Python's built-in hash() is salted per process (PYTHONHASHSEED), which
    # would make vectors and cache keys non-deterministic. Use sha1 instead.
    return int.from_bytes(hashlib.sha1(token.encode("utf-8")).digest()[:8], "big")


class DummyModel(BaseEmbeddingModel):
    def __init__(self, dim: int = 256, name: str | None = None):
        super().__init__(name or f"dummy-hash-{dim}")
        self.dim = dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN.findall(text.lower()):
                h = _stable_hash(tok)
                bucket = h % self.dim
                sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
                out[i, bucket] += sign
        return out
