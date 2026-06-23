"""Command-line interface: ``embench run -m <model> --retrieval data.json ...``

Build a benchmark from the shell without writing Python. Models are given as
``backend:model_id`` specs (repeat ``-m`` for several); tasks are given as
file paths. A local ``.env`` is loaded automatically so API keys are picked
up. Everything runs offline with the ``dummy`` backend, so you can try it
with no keys:

    embench run -m dummy:256 -m dummy:512 \
        --retrieval sample_data/retrieval.json \
        --classification sample_data/labeled.csv
"""
from __future__ import annotations

import argparse
import sys

from . import __version__, load_env
from .datasets import ClassificationDataset, ClusteringDataset, RetrievalDataset
from .runners import Benchmark

# backend -> (factory(model_id) , default_model_id). model_id "" means "use default".
_MODEL_HELP = """\
Model spec is BACKEND[:MODEL_ID], repeatable. Backends:
  dummy[:DIM]          offline hashing baseline (no deps/keys), default DIM=256
  openai[:ID]          OpenAI, default text-embedding-3-small
  cohere[:ID]          Cohere, default embed-english-v3.0
  voyage[:ID]          Voyage AI, default voyage-3
  google[:ID]          Gemini, default gemini-embedding-001
  hf:ID                Hugging Face Inference API (ID required)
  local:ID             local sentence-transformers model (ID required)
"""


def _build_model(spec: str):
    backend, _, model_id = spec.partition(":")
    backend = backend.strip().lower()
    model_id = model_id.strip()

    if backend == "dummy":
        from . import DummyModel

        return DummyModel(dim=int(model_id) if model_id else 256)
    if backend == "openai":
        from . import OpenAIModel

        return OpenAIModel(model_id or "text-embedding-3-small")
    if backend == "cohere":
        from . import CohereModel

        return CohereModel(model_id or "embed-english-v3.0")
    if backend == "voyage":
        from . import VoyageModel

        return VoyageModel(model_id or "voyage-3")
    if backend == "google":
        from . import GoogleModel

        return GoogleModel(model_id or "gemini-embedding-001")
    if backend in ("hf", "huggingface"):
        from . import HuggingFaceModel

        if not model_id:
            raise SystemExit(
                "error: hf model needs an id, e.g. "
                "hf:sentence-transformers/all-MiniLM-L6-v2"
            )
        return HuggingFaceModel(model_id)
    if backend in ("local", "st", "sentence-transformers"):
        from . import SentenceTransformerModel

        if not model_id:
            raise SystemExit("error: local model needs an id, e.g. local:all-MiniLM-L6-v2")
        return SentenceTransformerModel(model_id)
    raise SystemExit(f"error: unknown model backend {backend!r}. See `embench run -h`.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embench",
        description="Benchmark embedding models on your own data.",
    )
    parser.add_argument("--version", action="version", version=f"embench {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser(
        "run",
        help="run a benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_MODEL_HELP,
    )
    run.add_argument(
        "-m", "--model", action="append", required=True, metavar="SPEC",
        help="model spec BACKEND[:MODEL_ID]; repeat for several models",
    )
    run.add_argument(
        "--retrieval", action="append", default=[], metavar="JSON",
        help="retrieval dataset (JSON); repeatable",
    )
    run.add_argument(
        "--classification", action="append", default=[], metavar="CSV",
        help="classification dataset (CSV); repeatable",
    )
    run.add_argument(
        "--clustering", action="append", default=[], metavar="CSV",
        help="clustering dataset (CSV); repeatable",
    )
    run.add_argument(
        "-k", "--k-values", default="1,5,10",
        help="comma-separated cutoffs for retrieval metrics (default 1,5,10)",
    )
    run.add_argument("--text-col", default="text", help="CSV text column (default 'text')")
    run.add_argument("--label-col", default="label", help="CSV label column (default 'label')")
    run.add_argument("-o", "--output", metavar="CSV", help="write long-format results CSV")
    run.add_argument("--json", metavar="JSON", dest="json_out", help="write results JSON")
    run.add_argument("--no-cache", action="store_true", help="disable on-disk embedding cache")
    run.add_argument("--cache-dir", default=".embench_cache", help="cache directory")
    run.add_argument("--perf", action="store_true", help="include speed/cost columns in the table")
    run.add_argument("--std", action="store_true", help="annotate the table as 'mean ± std'")
    run.add_argument("--no-dotenv", action="store_true", help="do not auto-load a .env file")
    run.add_argument("-q", "--quiet", action="store_true", help="suppress per-run progress output")
    return parser


def _run(args) -> int:
    if not args.no_dotenv:
        load_env()

    from . import CachedModel

    models = []
    for spec in args.model:
        model = _build_model(spec)
        models.append(model if args.no_cache else CachedModel(model, cache_dir=args.cache_dir))

    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]

    tasks = []
    from .tasks import ClassificationTask, ClusteringTask, RetrievalTask

    for path in args.retrieval:
        tasks.append(RetrievalTask(RetrievalDataset.from_json(path), k_values=k_values))
    for path in args.classification:
        ds = ClassificationDataset.from_csv(
            path, text_col=args.text_col, label_col=args.label_col
        )
        tasks.append(ClassificationTask(ds))
    for path in args.clustering:
        ds = ClusteringDataset.from_csv(
            path, text_col=args.text_col, label_col=args.label_col
        )
        tasks.append(ClusteringTask(ds))

    if not tasks:
        raise SystemExit(
            "error: no tasks. Pass at least one of "
            "--retrieval / --classification / --clustering."
        )

    results = Benchmark(models, tasks=tasks).run(verbose=not args.quiet)

    print("\n=== Quality ===")
    if args.perf:
        print(results.to_dataframe(include_perf=True))
    else:
        print(results.to_table(std=args.std))
    perf = results.performance()
    if not args.perf and not perf.empty:
        print("\n=== Performance (speed / cost) ===")
        print(perf)

    if args.output:
        results.to_csv(args.output)
        print(f"\nWrote {args.output}")
    if args.json_out:
        results.to_json(args.json_out)
        print(f"Wrote {args.json_out}")
    return 0


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 1
    return _run(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
