from .base import Task
from .classification import ClassificationTask
from .clustering import ClusteringTask
from .retrieval import RetrievalTask

__all__ = ["Task", "RetrievalTask", "ClassificationTask", "ClusteringTask"]
