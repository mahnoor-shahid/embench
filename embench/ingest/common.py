"""Shared helpers for the ingestion stages (extract -> chunk -> ...).

The pipeline mirrors a simple on-disk convention so each stage can persist its
output and the next stage can pick it up without redoing expensive work:

    data/
      documents/         # your source PDFs (input)
      extracted_text/    # stage 1 -> one .md per document
      corpus.json        # stage 2 -> {"corpus": {chunk_id: text}}
      dataset.json       # stage 3 -> + {"queries": ..., "qrels": ...}
"""
from __future__ import annotations

import glob
import json
import os

from ..utils import text_hash

# Docling supports more (docx, pptx, images...), but PDFs are the default focus.
DEFAULT_EXTENSIONS = (".pdf",)
_TEXT_EXTENSIONS = (".md", ".txt")


def collect_paths(source, extensions, recursive) -> list[str]:
    """Resolve ``source`` (a file, a directory, or a list) to document paths."""
    if isinstance(source, (list, tuple)):
        return [str(p) for p in source]

    source = str(source)
    if os.path.isdir(source):
        pattern = os.path.join(source, "**", "*") if recursive else os.path.join(source, "*")
        paths = [
            p
            for p in glob.glob(pattern, recursive=recursive)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in extensions
        ]
        return sorted(paths)
    if os.path.isfile(source):
        return [source]
    raise FileNotFoundError(f"no such file or directory: {source!r}")


def doc_id_for(path: str) -> str:
    """Stable, collision-safe document id: ``<stem>-<pathhash>``."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{stem}-{text_hash(path)[:8]}"


def chunk_id_for(doc_id: str, index: int) -> str:
    return f"{doc_id}-{index:04d}"


def query_id_for(chunk_id: str, index: int) -> str:
    """Id for a synthetic query generated from a chunk: ``<chunk_id>-q<n>``."""
    return f"{chunk_id}-q{index}"


def load_text_dir(directory: str) -> dict[str, str]:
    """Load a folder of extracted text files into ``{doc_id: text}``."""
    out: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(directory, "*"))):
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in _TEXT_EXTENSIONS:
            doc_id = os.path.splitext(os.path.basename(path))[0]
            with open(path, encoding="utf-8") as fh:
                out[doc_id] = fh.read()
    return out


def save_text_dir(docs: dict[str, str], directory: str) -> None:
    """Write ``{doc_id: text}`` as one ``<doc_id>.md`` file each."""
    os.makedirs(directory, exist_ok=True)
    for doc_id, text in docs.items():
        with open(os.path.join(directory, f"{doc_id}.md"), "w", encoding="utf-8") as fh:
            fh.write(text)


def save_corpus(corpus: dict[str, str], path: str) -> None:
    """Write a corpus as ``{"corpus": {...}}`` JSON (the corpus half of retrieval).

    Add ``queries`` and ``qrels`` (the supervision) before loading it with
    ``RetrievalDataset.from_json`` -- or let :func:`generate_queries` do it.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"corpus": corpus}, fh, ensure_ascii=False, indent=2)


def load_corpus(source) -> dict[str, str]:
    """Resolve ``source`` to a ``{chunk_id: text}`` corpus.

    Accepts a corpus mapping directly, or a path to JSON written by
    :func:`save_corpus` (``{"corpus": {...}}``) or any file with a top-level
    ``corpus`` key (e.g. a full dataset). A bare ``{id: text}`` JSON object is
    also accepted.
    """
    if isinstance(source, dict):
        return dict(source)
    with open(str(source), encoding="utf-8") as fh:
        data = json.load(fh)
    corpus = data.get("corpus", data) if isinstance(data, dict) else data
    if not isinstance(corpus, dict):
        raise ValueError(f"could not find a corpus in {source!r}")
    return {str(k): v for k, v in corpus.items()}


def save_dataset(dataset: dict, path: str) -> None:
    """Write a full RetrievalDataset (``queries``/``corpus``/``qrels``) as JSON.

    The result loads directly with ``RetrievalDataset.from_json``.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=2)
