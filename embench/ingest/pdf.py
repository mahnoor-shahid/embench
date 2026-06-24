"""One-shot convenience: documents -> chunked corpus, in a single call.

This just composes the two stages (:func:`extract_text` then
:func:`chunk_text`). Use the stages directly when you want to persist the
extracted text and re-chunk without paying for extraction/OCR again.
"""
from __future__ import annotations

from .chunk import chunk_text
from .common import DEFAULT_EXTENSIONS, save_corpus  # noqa: F401 (re-export)
from .extract import extract_text


def pdf_to_corpus(
    source,
    *,
    extract_method: str = "docling",
    chunk_method: str = "recursive",
    ocr: bool = True,
    chunk: bool = True,
    max_chars: int = 1000,
    overlap: int = 150,
    recursive: bool = True,
    extensions=DEFAULT_EXTENSIONS,
    show_progress: bool = True,
) -> dict[str, str]:
    """Extract and (by default) chunk documents into a ``{chunk_id: text}`` corpus.

    A thin wrapper over :func:`extract_text` + :func:`chunk_text`; see those for
    the per-stage options. Set ``chunk=False`` to keep one entry per document.
    """
    docs = extract_text(
        source,
        method=extract_method,
        ocr=ocr,
        recursive=recursive,
        extensions=extensions,
        show_progress=show_progress,
    )
    if not chunk:
        return docs
    return chunk_text(docs, method=chunk_method, max_chars=max_chars, overlap=overlap)
