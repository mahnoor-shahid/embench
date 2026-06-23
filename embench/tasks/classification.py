"""Classification probe.

Trains a simple classifier on the embeddings with stratified
cross-validation. A good embedding space separates your categories with a
linear probe; this measures exactly that.
"""
from __future__ import annotations

import numpy as np

from ..datasets.base import ClassificationDataset
from ..models.base import BaseEmbeddingModel
from .base import Task


class ClassificationTask(Task):
    task_type = "classification"
    dataset_class = ClassificationDataset

    def __init__(
        self,
        dataset,
        method: str = "logreg",
        n_splits: int = 5,
        random_state: int = 42,
        name: str | None = None,
    ):
        super().__init__(dataset, name=name)
        if method not in ("logreg", "knn"):
            raise ValueError("method must be 'logreg' or 'knn'")
        self.method = method
        self.n_splits = n_splits
        self.random_state = random_state

    def _make_classifier(self):
        if self.method == "logreg":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=1000)
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(n_neighbors=5)

    def evaluate(self, model: BaseEmbeddingModel) -> dict[str, float]:
        from sklearn.model_selection import StratifiedKFold, cross_validate

        ds: ClassificationDataset = self.dataset
        X = model.encode(ds.texts)
        y = np.array(ds.labels)

        # cap folds so every fold can be stratified on the rarest class
        _, counts = np.unique(y, return_counts=True)
        n_splits = max(2, min(self.n_splits, int(counts.min())))

        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        res = cross_validate(
            self._make_classifier(),
            X,
            y,
            cv=cv,
            scoring=("accuracy", "f1_macro"),
        )
        # Report the std across folds too, so callers can tell whether a
        # difference between models is real or within noise.
        return {
            "accuracy": float(res["test_accuracy"].mean()),
            "accuracy_std": float(res["test_accuracy"].std()),
            "f1_macro": float(res["test_f1_macro"].mean()),
            "f1_macro_std": float(res["test_f1_macro"].std()),
        }
