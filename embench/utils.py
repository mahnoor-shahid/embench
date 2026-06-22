"""Small, dependency-light helpers used across the package."""
from __future__ import annotations

import hashlib

import numpy as np


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows. Zero rows are left as zeros (no div-by-zero)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between rows of ``a`` and rows of ``b``.

    Returns an array of shape ``(len(a), len(b))``.
    """
    return normalize(a) @ normalize(b).T


def text_hash(text: str, salt: str = "") -> str:
    """Stable hash of a string, used for caching keys and the dummy model."""
    return hashlib.sha256((salt + "\x00" + text).encode("utf-8")).hexdigest()
