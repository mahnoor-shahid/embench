"""Document-ingestion pipeline: documents (PDFs) -> a retrieval corpus.

Staged so each step persists its output and the next picks it up without
redoing expensive work (mirrors a simple ``data/`` folder convention):

    extract_text(...)     # stage 1: docling (OCR) or pymupdf  -> {doc_id: text}
    chunk_text(...)       # stage 2: recursive/sentence/fixed   -> {chunk_id: text}
    pdf_to_corpus(...)    # convenience: stages 1-2 in one call -> corpus
    generate_queries(...) # stage 3: LLM questions + qrels      -> RetrievalDataset

The extraction backends (docling, pymupdf) ship with embench itself.

Stages 1-2 produce only the *corpus* half of a RetrievalDataset. Stage 3 adds
the supervision a benchmark needs -- queries and relevance judgements (qrels) --
by asking an LLM for questions each chunk answers, yielding a dataset you can
load with ``RetrievalDataset.from_json`` and score directly.
"""
from .chunk import chunk_text
from .common import load_corpus, load_text_dir, save_corpus, save_dataset, save_text_dir
from .extract import extract_text
from .pdf import pdf_to_corpus
from .queries import generate_queries

__all__ = [
    "extract_text",
    "chunk_text",
    "pdf_to_corpus",
    "generate_queries",
    "save_corpus",
    "save_dataset",
    "save_text_dir",
    "load_text_dir",
    "load_corpus",
]
