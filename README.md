# embench

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/mahnoor-shahid/embench)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-blue)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-blue)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-red)](https://scikit-learn.org/)


**Benchmark embedding models on _your own_ data.**

MTEB and friends rank models on public, generic datasets. The model at the
top of the leaderboard is often *not* the best one for your domain — your
legal contracts, your support tickets, your medical notes, your code. `embench`
lets you point a handful of candidate models (local or proprietary) at your own
labeled data and see which one actually wins for *your* task, how confidently, and at what speed/cost.

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
pip install embench              # core + all API backends + PDF ingestion (docling/pymupdf)
pip install embench[local]       # + local models via sentence-transformers (pulls in PyTorch)
pip install embench[all]         # everything, including the local backend
```

The base install runs every **API** backend and the **PDF ingestion** pipeline
out of the box. The only heavyweight left out is `sentence-transformers` (it
pulls in PyTorch), so local models are an opt-in extra — add `[local]` (or
`[all]`) when you want them.

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

## Ingesting documents (PDFs)

Don't have a clean JSON corpus yet? Turn a folder of PDFs into one. The pipeline
is **staged** so each step persists its output and you never redo expensive work:

```
data/
  documents/          # your PDFs (input)
  extracted_text/     # stage 1 → one .md per document
  corpus.json         # stage 2 → {chunk_id: text}
  dataset.json        # stage 3 → + queries + qrels (ready to benchmark)
```

```bash
# ingestion ships in the base install — nothing extra to add
embench ingest extract data/documents/ -o data/extracted_text/        # stage 1
embench ingest chunk   data/extracted_text/ -o data/corpus.json       # stage 2
# …or both at once:
embench ingest run     data/documents/ -o data/corpus.json

# stage 3 — synthesise queries + qrels with an LLM, yielding a full dataset:
embench ingest queries data/corpus.json -o data/dataset.json --method openai
# now it's benchmarkable directly:
embench run -m dummy:256 --retrieval data/dataset.json
```

**Each stage has swappable options:**

| Stage | Option | Notes |
|------|--------|-------|
| extract | `--method docling` *(default)* | layout/table aware, **OCR for scanned pages** ([IBM Docling](https://github.com/docling-project/docling)) |
| extract | `--method pymupdf` | fast, born-digital PDFs only (no OCR) |
| extract | `--method granite` | IBM **Granite-Docling VLM** — reads page images, best on complex/messy layouts |
| chunk | `--method recursive` *(default)* | packs paragraphs/sections up to `--max-chars` |
| chunk | `--method sentence` | never cuts mid-sentence |
| chunk | `--method fixed` | uniform sliding window with `--overlap` |
| queries | `--method openai` *(default)* | Chat Completions (`OPENAI_API_KEY`), default `gpt-4o-mini` |
| queries | `--method google` | Gemini (`GOOGLE_API_KEY`), default `gemini-2.5-flash` |
| queries | `--n-queries N` / `--max-chunks N` | questions per chunk / cap chunks for cost |

In Python the stages are plain functions — extract once, then re-chunk freely,
and synthesise the supervision with whichever model (or your own callable):

```python
from embench.ingest import extract_text, chunk_text, generate_queries
docs    = extract_text("data/documents/", method="docling", out_dir="data/extracted_text/")
corpus  = chunk_text(docs, method="recursive", max_chars=1000, out_path="data/corpus.json")
dataset = generate_queries(corpus, method="openai", n_queries=1, out_path="data/dataset.json")
```

Stages 1–2 produce the **corpus** half of a retrieval dataset; **stage 3** adds
the supervision a benchmark needs — `queries` and relevance judgements (`qrels`),
synthesised by asking an LLM for the questions each chunk answers. The result
loads straight into `RetrievalDataset.from_json` and scores. (Pass your own
`generator=callable(text, n) -> list[str]` to `generate_queries` to plug in any
model or run offline.)

## Caching

Wrap any model in `CachedModel` so re-running a comparison never re-encodes
(or re-pays for) the same text. Embeddings are keyed by model name + text and
stored on disk.

```python
model = eb.CachedModel(eb.OpenAIModel("text-embedding-3-small"))
```

## API keys

Hosted backends read their key from the environment
(`OPENAI_API_KEY`, `COHERE_API_KEY`, `VOYAGE_API_KEY`, `GOOGLE_API_KEY`,
`HUGGINGFACE_API_KEY`). Copy `.env.example` to `.env`, fill it in, and load it:

```python
import embench as eb
eb.load_env()            # reads .env into the environment (no-op if absent)

models = [eb.GoogleModel(), eb.HuggingFaceModel("sentence-transformers/all-MiniLM-L6-v2")]
```

`load_env()` uses `python-dotenv` if installed and otherwise a small built-in
parser, so it works with just the core install.

## Reading results

```python
results.to_table()                       # pretty comparison string (quality only)
results.to_table(std=True)               # cells as "mean ± std"
results.to_dataframe()                    # wide: models x quality metrics
results.performance()                     # wide: models x speed/cost metrics
results.best_model("ndcg@10")            # single best model for a metric
results.ranking("accuracy")              # all models ranked by one metric
results.aggregate_ranking()              # one overall score: mean across quality metrics
results.to_csv("out.csv")                # export
```

Each metric also carries a spread (`accuracy_std` over CV folds, retrieval
metric std over queries). It's hidden from the default table — use
`to_table(std=True)` (or the CLI `--std` flag) to see whether one model
*really* beats another or the gap is within noise.

### Is the difference real? (significance testing)

A higher number isn't always a real win — it can be noise on your particular
queries. embench answers this directly with a **paired randomization (Fisher)
test** on the per-query scores, the same idea ranx uses, but with no extra
dependency:

```python
results.significance("ndcg@10")   # tidy table: every model pair, delta, p-value, significant?
results.win_tie_loss("ndcg@10")   # per-model record: wins / ties / losses (p < 0.05)
```

A *win* is a higher mean that clears `p < max_p`; anything within noise is a
*tie*. This needs per-query data, so it applies to retrieval metrics. From the
CLI: `embench run ... --significance ndcg@10 --rank`.

### Speed and cost

Every run also records how long each model spent encoding and how many texts
it actually encoded (cache hits don't count, so this tracks real API cost).
These are kept out of the quality table and exposed separately:

```python
results.performance()          # encode_seconds, texts_encoded, texts_per_sec per model
results.ranking("texts_per_sec")  # fastest model first
results.to_dataframe(include_perf=True)  # quality + perf in one table
```

Choosing a model is a quality/speed/cost trade-off — `embench` shows all three.

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

## Command line

Run a benchmark from the shell — no Python needed. Models are
`backend:model_id` specs (repeat `-m`); tasks are file paths. A local `.env`
is loaded automatically, so API keys are picked up.

```bash
embench run \
  -m dummy:256 \
  -m openai:text-embedding-3-small \
  -m local:all-MiniLM-L6-v2 \
  --retrieval my_data.json \
  --classification labeled.csv \
  -k 1,5,10 \
  -o results.csv
```

Backends: `dummy`, `openai`, `cohere`, `voyage`, `google`, `hf:<id>` (Inference
API), `local:<id>` (sentence-transformers). Caching is on by default
(`--no-cache` to disable); add `--perf` to fold speed/cost into the table.
See `embench run -h` for all options.

## Try it without any keys

```bash
embench run -m dummy:128 -m dummy:512 \
  --retrieval sample_data/retrieval.json \
  --classification sample_data/labeled.csv \
  --clustering sample_data/labeled.csv
# or the scripted equivalent:
python examples/quickstart.py
```

Uses a dependency-free `DummyModel` (a hashing-trick embedder) so the whole
pipeline runs offline. Use it as a baseline and a sanity check.

## Development

After cloning, set up an isolated environment in one command:

```bash
# Windows (PowerShell)
./setup.ps1

# macOS / Linux
./setup.sh
```

This creates a local `.venv`, installs `embench` (editable, with the API
backends + dev tools), and seeds a `.env` from `.env.example` for your API
keys. Then:

```bash
# activate: .venv\Scripts\Activate.ps1   (Windows)  |  source .venv/bin/activate  (Unix)
pytest                      # run the test suite
```

To also work on the local backend (PyTorch), add it after setup:
`pip install -e ".[all]"`.

`.venv`, `.env`, and the embedding cache are git-ignored; never commit them.
Do commit `.env.example`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
```
MIT License - Feel free to use, modify, and distribute
Academic use encouraged - Please cite our work
Commercial use welcome - Attribution appreciated
```

## Support

For questions, suggestions, or collaboration opportunities, feel free to reach out. 


