"""End-to-end example using the dummy model (no API keys / downloads).

Swap the model list for real models to benchmark them on your own data:

    models = [
        eb.CachedModel(eb.SentenceTransformerModel("all-MiniLM-L6-v2")),
        eb.CachedModel(eb.OpenAIModel("text-embedding-3-small")),
    ]
"""
import os

import embench as eb

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "sample_data")

# Two dummy models of different widths, just to have something to compare.
models = [
    eb.DummyModel(dim=128),
    eb.DummyModel(dim=512),
]

retrieval = eb.RetrievalDataset.from_json(os.path.join(DATA, "retrieval.json"))
classification = eb.ClassificationDataset.from_csv(os.path.join(DATA, "labeled.csv"))
clustering = eb.ClusteringDataset.from_csv(os.path.join(DATA, "labeled.csv"))

bench = eb.Benchmark(
    models,
    tasks=[
        eb.RetrievalTask(retrieval, k_values=[1, 3, 5]),
        eb.ClassificationTask(classification, method="logreg"),
        eb.ClusteringTask(clustering),
    ],
)

results = bench.run()

print("\n=== Comparison table (quality) ===")
print(results.to_table())

print("\n=== Performance (speed / cost) ===")
print(results.performance())

print("\nBest on retrieval (ndcg@5):", results.best_model("ndcg@5"))
print("Best on classification (accuracy):", results.best_model("accuracy"))
print("Ranking by v_measure:", results.ranking("v_measure"))

results.to_csv(os.path.join(HERE, "results.csv"))
print("\nWrote results.csv")
