"""embench -- benchmark embedding models on *your own* data.

Quickstart:

    import embench as eb

    models = [
        eb.DummyModel(dim=256),                          # baseline, no deps
        eb.SentenceTransformerModel("all-MiniLM-L6-v2"), # pip install embench[local]
        eb.OpenAIModel("text-embedding-3-small"),        # pip install embench[openai]
    ]
    # Wrap in a cache so re-runs don't re-pay:
    models = [eb.CachedModel(m) for m in models]

    retrieval = eb.RetrievalDataset.from_json("my_data.json")

    bench = eb.Benchmark(models, tasks=[
        eb.RetrievalTask(retrieval, k_values=[1, 5, 10]),
    ])
    results = bench.run()
    print(results.to_table())
    print("best:", results.best_model("ndcg@10"))
"""
from .datasets import ClassificationDataset, ClusteringDataset, RetrievalDataset
from .models import (
    BaseEmbeddingModel,
    CachedModel,
    DummyModel,
)
from .reporting import BenchmarkResults
from .runners import Benchmark
from .tasks import ClassificationTask, ClusteringTask, RetrievalTask, Task
from .utils import load_env

__version__ = "0.1.0"

_LAZY_MODELS = (
    "SentenceTransformerModel",
    "OpenAIModel",
    "CohereModel",
    "VoyageModel",
    "GoogleModel",
    "HuggingFaceModel",
)


def __getattr__(name):
    # Lazy access to adapters with optional heavy dependencies.
    if name in _LAZY_MODELS:
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "load_env",
    # models
    "BaseEmbeddingModel",
    "DummyModel",
    "CachedModel",
    "SentenceTransformerModel",
    "OpenAIModel",
    "CohereModel",
    "VoyageModel",
    "GoogleModel",
    "HuggingFaceModel",
    # datasets
    "RetrievalDataset",
    "ClassificationDataset",
    "ClusteringDataset",
    # tasks
    "Task",
    "RetrievalTask",
    "ClassificationTask",
    "ClusteringTask",
    # runner + results
    "Benchmark",
    "BenchmarkResults",
]
