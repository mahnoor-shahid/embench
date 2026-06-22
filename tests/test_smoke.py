"""Tests: metric correctness on known inputs + full pipeline runs."""
import numpy as np

import embench as eb
from embench.tasks.retrieval import _dcg


def test_dcg_known_value():
    # rel = [3, 2, 3, 0, 1, 2] -> DCG = sum (2^rel - 1)/log2(i+1)
    rels = [3, 2, 3, 0, 1, 2]
    expected = sum((2.0**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))
    assert abs(_dcg(rels) - expected) < 1e-9


def test_perfect_retrieval_scores_one():
    # Build a dataset where each query's text is identical to its one
    # relevant doc, so a sane embedder must rank it first.
    queries = {"q1": "alpha beta", "q2": "gamma delta"}
    corpus = {"d1": "alpha beta", "d2": "gamma delta", "d3": "zeta eta theta"}
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    ds = eb.RetrievalDataset(queries, corpus, qrels)
    task = eb.RetrievalTask(ds, k_values=[1, 3])
    scores = task.evaluate(eb.DummyModel(dim=64))
    assert scores["recall@1"] == 1.0
    assert scores["ndcg@1"] == 1.0
    assert scores["mrr@1"] == 1.0


def test_full_pipeline_runs():
    queries = {"q1": "cat dog", "q2": "stocks market"}
    corpus = {
        "d1": "the cat and the dog played",
        "d2": "stocks and the market rose",
        "d3": "unrelated text here",
    }
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    retrieval = eb.RetrievalDataset(queries, corpus, qrels)

    texts = ["money bank finance", "goal team match sport"] * 4
    labels = ["fin", "sport"] * 4
    clf = eb.ClassificationDataset(texts, labels)
    clu = eb.ClusteringDataset(texts, labels)

    bench = eb.Benchmark(
        [eb.DummyModel(dim=128)],
        tasks=[
            eb.RetrievalTask(retrieval, k_values=[1, 3]),
            eb.ClassificationTask(clf),
            eb.ClusteringTask(clu),
        ],
    )
    results = bench.run(verbose=False)
    assert len(results) > 0
    df = results.to_dataframe()
    assert not df.empty


def test_dummy_is_deterministic():
    m = eb.DummyModel(dim=32)
    a = m.encode(["hello world"])
    b = m.encode(["hello world"])
    assert np.array_equal(a, b)


def test_dataset_validation_rejects_bad_qrels():
    try:
        eb.RetrievalDataset(
            queries={"q1": "x"},
            corpus={"d1": "y"},
            qrels={"q1": {"missing_doc": 1}},
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown doc id")
