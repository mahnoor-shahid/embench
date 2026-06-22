"""Orchestration: run every model against every task, collect results.

The runner is deliberately dumb -- it knows nothing about specific tasks
or models, only the two interfaces. All intelligence lives in the tasks.
"""
from __future__ import annotations

import time
import traceback

from ..models.base import BaseEmbeddingModel
from ..reporting.results import BenchmarkResults
from ..tasks.base import Task


class Benchmark:
    def __init__(self, models: list[BaseEmbeddingModel], tasks: list[Task] | None = None):
        self.models = list(models)
        self.tasks: list[Task] = list(tasks or [])

    def add_task(self, task: Task) -> "Benchmark":
        self.tasks.append(task)
        return self

    def run(self, verbose: bool = True, skip_errors: bool = True) -> BenchmarkResults:
        rows: list[dict] = []
        for model in self.models:
            for task in self.tasks:
                label = f"{model.name} | {task.task_type}:{task.name}"
                if verbose:
                    print(f"Running {label} ...")
                start = time.perf_counter()
                try:
                    metrics = task.evaluate(model)
                except Exception as exc:  # noqa: BLE001
                    if not skip_errors:
                        raise
                    print(f"  ! failed: {exc}")
                    if verbose:
                        traceback.print_exc()
                    continue
                elapsed = time.perf_counter() - start
                for metric, value in metrics.items():
                    rows.append(
                        {
                            "model": model.name,
                            "task": task.task_type,
                            "dataset": task.name,
                            "metric": metric,
                            "value": value,
                        }
                    )
                if verbose:
                    print(f"  done in {elapsed:.2f}s")
        return BenchmarkResults(rows)
