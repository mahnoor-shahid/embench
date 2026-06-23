"""Small, dependency-light helpers used across the package."""
from __future__ import annotations

import hashlib
import os

import numpy as np


def load_env(path: str = ".env", override: bool = False) -> dict[str, str]:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into ``os.environ``.

    embench's API model adapters read keys (``OPENAI_API_KEY`` etc.) from the
    environment. Call this once at startup so a local ``.env`` "just works":

        import embench as eb
        eb.load_env()

    By default existing environment variables win (``override=False``). If
    ``python-dotenv`` is installed it is used; otherwise a small built-in
    parser handles the common ``KEY=VALUE`` / ``# comment`` / quoted-value
    cases. Returns the keys that were applied. A missing file is a no-op.
    """
    if not os.path.exists(path):
        return {}

    try:  # prefer python-dotenv if available (handles edge cases)
        from dotenv import dotenv_values
    except ImportError:
        dotenv_values = None

    if dotenv_values is not None:
        parsed = {k: v for k, v in dotenv_values(path).items() if v is not None}
    else:
        parsed = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :]
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                parsed[key] = value

    applied = {}
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


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
