"""Embedding model adapters.

Only the dummy model and the base class are imported eagerly. The real
adapters import heavy / optional dependencies lazily, so importing this
package never requires torch, openai, etc.
"""
from .base import BaseEmbeddingModel
from .cache import CachedModel
from .dummy import DummyModel


def __getattr__(name):  # PEP 562 lazy attribute access
    if name == "SentenceTransformerModel":
        from .sentence_transformer import SentenceTransformerModel

        return SentenceTransformerModel
    if name == "OpenAIModel":
        from .openai import OpenAIModel

        return OpenAIModel
    if name in ("CohereModel", "VoyageModel", "GoogleModel", "HuggingFaceModel"):
        from . import api_models

        return getattr(api_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseEmbeddingModel",
    "CachedModel",
    "DummyModel",
    "SentenceTransformerModel",
    "OpenAIModel",
    "CohereModel",
    "VoyageModel",
    "GoogleModel",
    "HuggingFaceModel",
]
