"""Task base class.

A task pairs an evaluation procedure with a dataset. To add a new task
type, subclass :class:`Task`, set ``task_type``, declare which dataset
class you accept, and implement :meth:`evaluate`. The runner and the
reporting layer need no changes -- that is what "modular tasks" buys you.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models.base import BaseEmbeddingModel


class Task(ABC):
    #: short identifier, e.g. "retrieval", shown in reports
    task_type: str = "task"
    #: the dataset class this task consumes (for validation)
    dataset_class: type | None = None

    def __init__(self, dataset, name: str | None = None):
        if self.dataset_class is not None and not isinstance(dataset, self.dataset_class):
            raise TypeError(
                f"{type(self).__name__} expects a "
                f"{self.dataset_class.__name__}, got {type(dataset).__name__}"
            )
        self.dataset = dataset
        # name distinguishes multiple datasets of the same task type
        self.name = name or getattr(dataset, "name", self.task_type)

    @abstractmethod
    def evaluate(self, model: BaseEmbeddingModel) -> dict[str, float]:
        """Run the task for one model and return ``{metric_name: value}``."""
        raise NotImplementedError

    def evaluate_detailed(
        self, model: BaseEmbeddingModel
    ) -> tuple[dict[str, float], dict[str, list]]:
        """Like :meth:`evaluate`, but also return per-sample metric values.

        Returns ``(metrics, samples)`` where ``samples`` maps a metric name to
        its per-sample (e.g. per-query) values -- the raw input a significance
        test needs to tell real differences from noise. Tasks that have no
        meaningful per-sample breakdown return an empty ``samples`` dict; the
        runner calls this and keeps whatever samples it gets.
        """
        return self.evaluate(model), {}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
