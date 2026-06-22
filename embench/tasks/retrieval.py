"""Retrieval evaluation: the core RAG use case.

Embeds corpus and queries, ranks the corpus per query by cosine
similarity, and reports standard IR metrics at each requested cutoff k.
Metrics are implemented directly on numpy (no extra deps) and support
graded relevance for nDCG.
"""
from __future__ import annotations

import numpy as np

from ..datasets.base import RetrievalDataset
from ..models.base import BaseEmbeddingModel
from ..utils import cosine_similarity
from .base import Task


def _dcg(relevances: np.ndarray) -> float:
    relevances = np.asarray(relevances, dtype=float)
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum((2.0**relevances - 1.0) / discounts))


class RetrievalTask(Task):
    task_type = "retrieval"
    dataset_class = RetrievalDataset

    def __init__(self, dataset, k_values=(1, 5, 10), name: str | None = None):
        super().__init__(dataset, name=name)
        self.k_values = tuple(sorted(k_values))

    def evaluate(self, model: BaseEmbeddingModel) -> dict[str, float]:
        ds: RetrievalDataset = self.dataset
        doc_ids = list(ds.corpus.keys())
        doc_pos = {did: i for i, did in enumerate(doc_ids)}
        query_ids = list(ds.queries.keys())

        doc_vecs = model.encode([ds.corpus[d] for d in doc_ids])
        query_vecs = model.encode([ds.queries[q] for q in query_ids])

        sims = cosine_similarity(query_vecs, doc_vecs)  # (n_queries, n_docs)
        max_k = min(max(self.k_values), len(doc_ids))
        # indices of the top max_k docs per query, best first
        top_idx = np.argsort(-sims, axis=1)[:, :max_k]

        scores = {f"ndcg@{k}": [] for k in self.k_values}
        for metric in ("recall", "precision", "mrr", "map"):
            for k in self.k_values:
                scores[f"{metric}@{k}"] = []

        for qi, qid in enumerate(query_ids):
            rels = ds.qrels[qid]
            n_relevant = sum(1 for r in rels.values() if r > 0)
            ranked = [doc_ids[j] for j in top_idx[qi]]
            ranked_rel = np.array([rels.get(d, 0) for d in ranked], dtype=float)
            ideal = np.array(sorted(rels.values(), reverse=True), dtype=float)

            for k in self.k_values:
                topk = ranked_rel[:k]
                hits = topk > 0
                num_hits = int(hits.sum())

                idcg = _dcg(ideal[:k])
                scores[f"ndcg@{k}"].append(_dcg(topk) / idcg if idcg > 0 else 0.0)
                scores[f"recall@{k}"].append(
                    num_hits / n_relevant if n_relevant else 0.0
                )
                scores[f"precision@{k}"].append(num_hits / k)

                # MRR: reciprocal rank of first hit within k
                first = np.argmax(hits) if hits.any() else -1
                scores[f"mrr@{k}"].append(1.0 / (first + 1) if first >= 0 else 0.0)

                # MAP: mean of precision at each hit position within k
                if num_hits:
                    ranks = np.where(hits)[0] + 1
                    precs = np.arange(1, num_hits + 1) / ranks
                    denom = min(k, n_relevant) if n_relevant else 1
                    scores[f"map@{k}"].append(float(precs.sum() / denom))
                else:
                    scores[f"map@{k}"].append(0.0)

        return {m: float(np.mean(v)) for m, v in scores.items()}
