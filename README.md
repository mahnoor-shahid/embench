# embench

**Benchmark embedding models on _your own_ data.**

MTEB and friends rank models on public, generic datasets. The model at the
top of the leaderboard is often *not* the best one for your domain — your
legal contracts, your support tickets, your medical notes, your code. `embench`
lets you point a handful of candidate models (local or proprietary) at your own
labeled data and see which one actually wins for *your* task.

```python
import embench as eb

models = [
    eb.CachedModel(eb.SentenceTransformerModel("all-MiniLM-L6-v2")),
    eb.CachedModel(eb.OpenAIModel("text-embedding-3-small")),
]

retrieval = eb.RetrievalDataset.from_json("my_data.json")

results = eb.Benchmark(models, tasks=[
    eb.RetrievalTask(retrieval, k_values=[1, 5, 10]),
]).run()

print(results.to_table())
print("Winner:", results.best_model("ndcg@10"))
```

## Install

```bash
pip install embench              # core: dummy model + all tasks
pip install embench[local]       # + sentence-transformers (local/HF models)
pip install embench[openai]      # + OpenAI
pip install embench[cohere]      # + Cohere
pip install embench[voyage]      # + Voyage AI
pip install embench[all]         # everything
```

Core install pulls only numpy, scikit-learn, and pandas. Every model backend
is an optional extra, so you only install what you use.

## Tasks

| Task | What it measures | Metrics |
|------|------------------|---------|
| `RetrievalTask` | RAG / semantic search quality | nDCG@k, Recall@k, MRR@k, MAP@k, Precision@k |
| `ClassificationTask` | Whether the space separates your categories (linear probe, cross-validated) | accuracy, f1_macro |
| `ClusteringTask` | Whether natural groups form clean clusters (KMeans) | V-measure, ARI, NMI |

## Data formats

**Retrieval** — JSON with `queries`, `corpus`, and `qrels` (relevance
judgements; graded relevance supported):

```json
{
  "queries": { "q1": "how do I reset my password" },
  "corpus":  { "d1": "click forgot password on the login page" },
  "qrels":   { "q1": { "d1": 2 } }
}
```

**Classification / Clustering** — CSV with a text column and a label column:

```csv
text,label
the stock market rallied,finance
the team won the championship,sports
```

```python
eb.ClassificationDataset.from_csv("data.csv", text_col="text", label_col="label")
```

## Caching

Wrap any model in `CachedModel` so re-running a comparison never re-encodes
(or re-pays for) the same text. Embeddings are keyed by model name + text and
stored on disk.

```python
model = eb.CachedModel(eb.OpenAIModel("text-embedding-3-small"))
```

## Reading results

```python
results.to_table()                       # pretty comparison string
results.to_dataframe()                    # wide: models x metrics
results.best_model("ndcg@10")            # single best model for a metric
results.ranking("accuracy")              # all models ranked
results.to_csv("out.csv")                # export
```

## Extending it

**A new model backend:** subclass `BaseEmbeddingModel`, implement `_encode`.
Batching and progress are handled for you.

```python
class MyModel(eb.BaseEmbeddingModel):
    def _encode(self, texts: list[str]) -> np.ndarray:
        return my_embedding_call(texts)
```

**A new task type:** subclass `Task`, set `task_type` and `dataset_class`,
implement `evaluate(self, model) -> dict[str, float]`. The runner and
reporting need no changes — that is the whole design.

## Try it without any keys

```bash
python examples/quickstart.py
```

Uses a dependency-free `DummyModel` (a hashing-trick embedder) so the whole
pipeline runs offline. Use it as a baseline and a sanity check.

## License

MIT
