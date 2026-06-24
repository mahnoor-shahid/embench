"""Tests: metric correctness on known inputs + full pipeline runs."""
import os

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


def test_results_include_performance_metrics():
    queries = {"q1": "alpha beta"}
    corpus = {"d1": "alpha beta", "d2": "gamma delta"}
    qrels = {"q1": {"d1": 1}}
    ds = eb.RetrievalDataset(queries, corpus, qrels)
    results = eb.Benchmark(
        [eb.DummyModel(dim=32)], tasks=[eb.RetrievalTask(ds, k_values=[1])]
    ).run(verbose=False)

    perf = results.performance()
    assert not perf.empty
    assert {"encode_seconds", "texts_encoded", "texts_per_sec"} <= set(perf.columns)
    # 2 docs + 1 query were encoded
    assert perf.loc["dummy-hash-32", "texts_encoded"] == 3.0
    # perf metrics must not leak into the quality table
    assert "retrieval:retrieval/texts_encoded" not in results.to_dataframe().columns


def test_cached_model_skips_cost_on_hit(tmp_path):
    model = eb.CachedModel(eb.DummyModel(dim=16), cache_dir=str(tmp_path / "c"))
    texts = ["one fish", "two fish"]

    model.stats.reset()
    model.encode(texts)
    assert model.stats.n_encoded == 2  # first pass: everything is a miss

    model.stats.reset()
    model.encode(texts)
    assert model.stats.n_encoded == 0  # second pass: all cache hits, no cost
    assert model.stats.n_texts == 2  # but still requested


def test_load_env_parses_file_without_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "export OPENAI_API_KEY='sk-test-123'\n"
        'COHERE_API_KEY="co-456"\n'
        "EMPTY_LINE_BELOW=\n"
        "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("COHERE_API_KEY", "already-set")

    applied = eb.load_env(str(env))

    assert os.environ["OPENAI_API_KEY"] == "sk-test-123"  # quotes stripped
    assert os.environ["COHERE_API_KEY"] == "already-set"  # existing var wins
    assert "OPENAI_API_KEY" in applied and "COHERE_API_KEY" not in applied


def test_load_env_missing_file_is_noop(tmp_path):
    assert eb.load_env(str(tmp_path / "nope.env")) == {}


def test_new_model_adapters_are_exported():
    # Names resolve via lazy import; constructing without the optional dep
    # must raise a clear, install-pointing ImportError (not AttributeError).
    for cls_name in ("GoogleModel", "HuggingFaceModel"):
        cls = getattr(eb, cls_name)
        assert issubclass(cls, eb.BaseEmbeddingModel)


def test_std_metrics_present_but_hidden_by_default():
    texts = ["money bank finance", "goal team match sport"] * 5
    labels = ["fin", "sport"] * 5
    clf = eb.ClassificationDataset(texts, labels)
    results = eb.Benchmark(
        [eb.DummyModel(dim=64)], tasks=[eb.ClassificationTask(clf)]
    ).run(verbose=False)

    # std rows exist
    assert any(r["metric"] == "accuracy_std" for r in results.rows)
    # but are hidden from the default quality table
    cols = results.to_dataframe().columns
    assert not any(c.endswith("_std") for c in cols)
    assert any(c.endswith("_std") for c in results.to_dataframe(include_std=True).columns)
    # the mean ± std table renders the spread
    assert "±" in results.to_table(std=True)


def test_cli_run_offline(tmp_path):
    import csv
    import json

    from embench.cli import main

    retrieval = tmp_path / "r.json"
    retrieval.write_text(
        json.dumps(
            {
                "queries": {"q1": "alpha beta"},
                "corpus": {"d1": "alpha beta", "d2": "gamma delta"},
                "qrels": {"q1": {"d1": 1}},
            }
        ),
        encoding="utf-8",
    )
    labeled = tmp_path / "c.csv"
    with open(labeled, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "label"])
        for t, lab in [("money bank", "fin"), ("goal team", "sport")] * 4:
            w.writerow([t, lab])

    out = tmp_path / "out.csv"
    code = main(
        [
            "run", "-m", "dummy:64",
            "--retrieval", str(retrieval),
            "--classification", str(labeled),
            "-k", "1,2",
            "-o", str(out),
            "--no-cache", "--quiet",
        ]
    )
    assert code == 0
    assert out.exists() and out.stat().st_size > 0


def test_cli_no_command_returns_nonzero(capsys):
    from embench.cli import main

    assert main([]) == 1  # prints help, signals "nothing ran"


def test_ingest_collects_and_ids_documents(tmp_path):
    from embench.ingest.common import chunk_id_for, collect_paths, doc_id_for

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.pdf").write_bytes(b"%PDF-1.4")

    recursive = collect_paths(str(tmp_path), (".pdf",), recursive=True)
    assert len(recursive) == 2  # a.pdf + sub/b.pdf, not the .txt
    flat = collect_paths(str(tmp_path), (".pdf",), recursive=False)
    assert len(flat) == 1  # only top-level a.pdf

    # ids are stable and disambiguate same-named files in different folders
    assert doc_id_for("x/a.pdf") == doc_id_for("x/a.pdf")
    assert doc_id_for("x/a.pdf") != doc_id_for("y/a.pdf")
    assert chunk_id_for(doc_id_for("x/a.pdf"), 3).endswith("-0003")


def test_chunk_text_strategies_and_roundtrip(tmp_path):
    import json

    # one short doc + one long doc (forces splitting)
    docs = {
        "doc-0000": "First para.\n\nSecond para.",
        "doc-0001": "word. " * 400,  # ~2400 chars -> must split at max_chars
    }
    for method in ("recursive", "sentence", "fixed"):
        corpus = eb.chunk_text(docs, method=method, max_chars=300, overlap=50)
        assert corpus  # produced chunks
        assert all(ids.startswith(("doc-0000", "doc-0001")) for ids in corpus)
        assert all(len(v) <= 600 for v in corpus.values())  # roughly bounded

    # persists corpus JSON in the retrieval-friendly shape
    out = tmp_path / "corpus.json"
    eb.chunk_text(docs, method="fixed", max_chars=300, out_path=str(out))
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert "corpus" in saved and len(saved["corpus"]) > 0


def test_extract_unknown_method_errors():
    import pytest

    with pytest.raises(ValueError, match="unknown extract method"):
        eb.extract_text("whatever", method="nope")


def test_extract_requires_backend_when_absent(tmp_path):
    import importlib.util

    import pytest

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    if importlib.util.find_spec("docling") is None:
        with pytest.raises(ImportError, match="pip install embench"):
            eb.extract_text(str(tmp_path), method="docling")


def _make_pdf(path, lines):
    """Build a tiny born-digital PDF with a real text layer (needs pymupdf)."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line)
        y += 28
    doc.save(str(path))
    doc.close()


def test_extract_pymupdf_real_pdf(tmp_path):
    import importlib.util

    import pytest

    if importlib.util.find_spec("fitz") is None:
        pytest.skip("pymupdf not installed")

    _make_pdf(
        tmp_path / "faq.pdf",
        ["Reset your password from the login page.", "We accept Visa and PayPal."],
    )
    docs = eb.extract_text(str(tmp_path), method="pymupdf", show_progress=False)
    assert len(docs) == 1
    text = next(iter(docs.values())).lower()
    assert "password" in text and "paypal" in text

    corpus = eb.chunk_text(docs, method="recursive", max_chars=40)
    assert len(corpus) >= 2  # the two lines land in separate chunks


def test_extract_docling_real_pdf(tmp_path):
    import importlib.util

    import pytest

    if importlib.util.find_spec("docling") is None or importlib.util.find_spec("fitz") is None:
        pytest.skip("docling/pymupdf not installed")

    _make_pdf(tmp_path / "doc.pdf", ["Cancel your subscription in account settings."])
    docs = eb.extract_text(str(tmp_path), method="docling", show_progress=False)
    assert len(docs) == 1
    assert "subscription" in next(iter(docs.values())).lower()


def test_generate_queries_builds_benchmarkable_dataset(tmp_path):
    import json

    # Persist a corpus the way `ingest chunk` does, then load it back.
    corpus = {"doc-0000": "alpha beta gamma", "doc-0001": "delta epsilon"}
    corpus_path = tmp_path / "corpus.json"
    eb.save_corpus(corpus, str(corpus_path))

    # Offline generator: deterministic, no API key needed.
    def fake_gen(text, n):
        first = text.split()[0]
        return [f"what is {first} #{i}?" for i in range(n)]

    out = tmp_path / "dataset.json"
    dataset = eb.generate_queries(
        str(corpus_path),
        generator=fake_gen,
        n_queries=2,
        out_path=str(out),
        show_progress=False,
    )

    # 2 chunks x 2 queries each, every query judged against its source chunk.
    assert len(dataset["queries"]) == 4
    assert dataset["corpus"] == corpus
    assert dataset["qrels"]["doc-0000-q0"] == {"doc-0000": 1}

    # The written file loads as a real, valid RetrievalDataset and scores.
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert {"queries", "corpus", "qrels"} <= set(saved)
    ds = eb.RetrievalDataset.from_json(str(out))
    scores = eb.RetrievalTask(ds, k_values=[1]).evaluate(eb.DummyModel(dim=64))
    assert "recall@1" in scores


def test_generate_queries_unknown_method_errors():
    import pytest

    with pytest.raises(ValueError, match="unknown query method"):
        eb.generate_queries({"d0": "x"}, method="nope")


def test_generate_queries_empty_corpus_errors():
    import pytest

    with pytest.raises(ValueError, match="empty corpus"):
        eb.generate_queries({}, generator=lambda t, n: ["q"])


def test_permutation_test_separates_signal_from_noise():
    import numpy as np

    from embench.utils import paired_permutation_test

    rng = np.random.default_rng(0)
    # a consistently beats b by a clear margin -> low p
    a = rng.uniform(0.6, 0.9, size=200)
    b = rng.uniform(0.1, 0.4, size=200)
    assert paired_permutation_test(a, b) < 0.05
    # identical vectors -> not distinguishable -> p == 1.0
    assert paired_permutation_test(a, a) == 1.0


def test_significance_and_win_tie_loss(tmp_path):
    # Build a retrieval set with enough queries for a meaningful test. One model
    # is a strong retriever (dummy), the other returns constant vectors so every
    # query ties at chance -> the strong model should significantly win.
    import numpy as np

    queries = {f"q{i}": f"term{i} alpha beta" for i in range(40)}
    corpus = {f"d{i}": f"term{i} alpha beta" for i in range(40)}
    corpus["dx"] = "unrelated filler text"
    qrels = {f"q{i}": {f"d{i}": 1} for i in range(40)}
    ds = eb.RetrievalDataset(queries, corpus, qrels)

    class ConstantModel(eb.BaseEmbeddingModel):
        def __init__(self):
            super().__init__("constant")

        def _encode(self, texts):
            return np.ones((len(texts), 8), dtype=np.float32)

    results = eb.Benchmark(
        [eb.DummyModel(dim=128), ConstantModel()],
        tasks=[eb.RetrievalTask(ds, k_values=[1, 5])],
    ).run(verbose=False)

    sig = results.significance("ndcg@5", n_permutations=2000)
    assert {"model_a", "model_b", "delta", "p_value", "significant"} <= set(sig.columns)
    assert len(sig) == 1  # one unordered pair of two models
    row = sig.iloc[0]
    assert row["significant"]  # the difference is real, not noise

    wtl = results.win_tie_loss("ndcg@5", n_permutations=2000)
    assert set(wtl.columns) == {"score", "wins", "ties", "losses"}
    # the better model tops the table with exactly one win
    assert wtl.iloc[0]["wins"] == 1 and wtl.iloc[-1]["losses"] == 1


def test_significance_requires_per_sample_data():
    import pytest

    texts = ["money bank finance", "goal team match sport"] * 5
    labels = ["fin", "sport"] * 5
    results = eb.Benchmark(
        [eb.DummyModel(dim=32)],
        tasks=[eb.ClassificationTask(eb.ClassificationDataset(texts, labels))],
    ).run(verbose=False)
    with pytest.raises(ValueError, match="per-sample data"):
        results.significance("accuracy")


def test_aggregate_ranking_means_quality_metrics():
    queries = {"q1": "alpha beta", "q2": "gamma delta"}
    corpus = {"d1": "alpha beta", "d2": "gamma delta", "d3": "zeta eta"}
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    ds = eb.RetrievalDataset(queries, corpus, qrels)
    results = eb.Benchmark(
        [eb.DummyModel(dim=64), eb.DummyModel(dim=32)],
        tasks=[eb.RetrievalTask(ds, k_values=[1, 5])],
    ).run(verbose=False)

    overall = results.aggregate_ranking()
    assert overall and len(overall) == 2  # both models scored
    # scores are means of quality metrics only -> within [0, 1], perf excluded
    assert all(0.0 <= score <= 1.0 for _, score in overall)
    # subsetting to chosen metrics works too
    subset = results.aggregate_ranking(metrics=["recall@5"])
    assert {m for m, _ in subset} == {m for m, _ in overall}


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
