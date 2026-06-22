"""Canonical dataset containers.

Each task consumes exactly one dataset type. Loaders convert common file
formats (JSON, CSV) into these, and each class validates itself so errors
surface early with a clear message rather than deep inside a metric.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field


@dataclass
class RetrievalDataset:
    """Queries, a corpus, and relevance judgements (qrels).

    queries: mapping query_id -> query text
    corpus:  mapping doc_id   -> document text
    qrels:   mapping query_id -> {doc_id: relevance}, relevance > 0 == relevant
             (graded relevances are supported and used by nDCG)
    """

    queries: dict[str, str]
    corpus: dict[str, str]
    qrels: dict[str, dict[str, int]]
    name: str = "retrieval"

    def __post_init__(self) -> None:
        if not self.queries:
            raise ValueError("RetrievalDataset has no queries")
        if not self.corpus:
            raise ValueError("RetrievalDataset has no corpus")
        for qid in self.qrels:
            if qid not in self.queries:
                raise ValueError(f"qrels reference unknown query id {qid!r}")
            for did in self.qrels[qid]:
                if did not in self.corpus:
                    raise ValueError(f"qrels reference unknown doc id {did!r}")
        unjudged = set(self.queries) - set(self.qrels)
        if unjudged:
            raise ValueError(
                f"{len(unjudged)} queries have no relevance judgements, e.g. "
                f"{next(iter(unjudged))!r}"
            )

    @classmethod
    def from_json(cls, path: str, name: str | None = None) -> "RetrievalDataset":
        """Load from a JSON file with keys ``queries``, ``corpus``, ``qrels``."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            queries={str(k): v for k, v in data["queries"].items()},
            corpus={str(k): v for k, v in data["corpus"].items()},
            qrels={
                str(q): {str(d): int(r) for d, r in rels.items()}
                for q, rels in data["qrels"].items()
            },
            name=name or data.get("name", "retrieval"),
        )


@dataclass
class ClassificationDataset:
    """Texts with categorical labels."""

    texts: list[str]
    labels: list
    name: str = "classification"

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.labels):
            raise ValueError("texts and labels must have the same length")
        if len(set(self.labels)) < 2:
            raise ValueError("classification needs at least 2 distinct labels")

    @classmethod
    def from_csv(
        cls,
        path: str,
        text_col: str = "text",
        label_col: str = "label",
        name: str | None = None,
    ) -> "ClassificationDataset":
        texts, labels = [], []
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                texts.append(row[text_col])
                labels.append(row[label_col])
        return cls(texts=texts, labels=labels, name=name or "classification")


@dataclass
class ClusteringDataset:
    """Texts with ground-truth group labels (used to score the clustering)."""

    texts: list[str]
    labels: list
    name: str = "clustering"

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.labels):
            raise ValueError("texts and labels must have the same length")

    @classmethod
    def from_csv(
        cls,
        path: str,
        text_col: str = "text",
        label_col: str = "label",
        name: str | None = None,
    ) -> "ClusteringDataset":
        texts, labels = [], []
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                texts.append(row[text_col])
                labels.append(row[label_col])
        return cls(texts=texts, labels=labels, name=name or "clustering")
