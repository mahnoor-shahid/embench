"""Clustering evaluation.

Runs KMeans on the embeddings (k = number of ground-truth groups) and
scores the clustering against the true labels. Tests whether natural
groups in your data form clean, well-separated clusters in the embedding
space.
"""
from __future__ import annotations

import numpy as np

from ..datasets.base import ClusteringDataset
from ..models.base import BaseEmbeddingModel
from ..utils import normalize
from .base import Task


class ClusteringTask(Task):
    task_type = "clustering"
    dataset_class = ClusteringDataset

    def __init__(self, dataset, random_state: int = 42, name: str | None = None):
        super().__init__(dataset, name=name)
        self.random_state = random_state

    def evaluate(self, model: BaseEmbeddingModel) -> dict[str, float]:
        from sklearn.cluster import KMeans
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
            v_measure_score,
        )

        ds: ClusteringDataset = self.dataset
        # normalize so KMeans (Euclidean) approximates cosine geometry
        X = normalize(model.encode(ds.texts))
        y = np.array(ds.labels)
        k = len(np.unique(y))

        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        pred = km.fit_predict(X)

        return {
            "v_measure": float(v_measure_score(y, pred)),
            "ari": float(adjusted_rand_score(y, pred)),
            "nmi": float(normalized_mutual_info_score(y, pred)),
        }
