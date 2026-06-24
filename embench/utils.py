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


def paired_permutation_test(
    a, b, n_permutations: int = 10000, seed: int = 0
) -> float:
    """Two-sided paired randomization (Fisher) test on per-query metric values.

    ``a`` and ``b`` are aligned arrays -- one score per query, for two models on
    the *same* queries. Under the null (the two models are interchangeable) the
    sign of each per-query difference is equally likely to flip, so we randomly
    flip signs many times and ask how often the resampled mean difference is at
    least as extreme as the observed one. Returns a p-value in ``(0, 1]``.

    This mirrors the significance test ranx uses to decide whether one model
    *really* beats another on your data, but needs no extra dependency.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of the same length")
    diff = a - b
    n = diff.size
    if n == 0:
        return 1.0
    observed = abs(float(diff.mean()))
    if observed == 0.0:
        return 1.0

    rng = np.random.default_rng(seed)
    # Random +/-1 sign flips per query, vectorised over all permutations.
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, n))
    resampled = np.abs((signs * diff).mean(axis=1))
    # +1 in numerator and denominator counts the observed arrangement itself,
    # giving an unbiased, never-zero p-value.
    count = int(np.count_nonzero(resampled >= observed - 1e-12))
    return (count + 1) / (n_permutations + 1)
