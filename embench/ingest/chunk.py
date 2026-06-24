"""Stage 2 -- chunking. Multiple swappable strategies, all dependency-free.

    recursive  default. Pack paragraphs/sections (split on blank lines) up to a
               size budget; hard-split any oversized block. Best on the markdown
               docling produces.
    sentence   pack whole sentences up to the budget (no mid-sentence cuts).
    fixed      sliding character window with overlap. Simplest / most uniform.

Sizes are in characters (roughly 4 chars per token) to stay dependency-free.
Chunkers operate on the *text* the extract stage produced, so they can be
swapped without re-extracting.
"""
from __future__ import annotations

import re

from .common import chunk_id_for, load_text_dir, save_corpus


def _chunk_fixed(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    step = max(1, max_chars - overlap)
    return [text[i : i + max_chars] for i in range(0, len(text), step)]


def _pack(pieces: list[str], max_chars: int, joiner: str) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + len(joiner) + len(piece) > max_chars:
            chunks.append(cur)
            cur = piece
        else:
            cur = cur + joiner + piece if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


def _chunk_recursive(text: str, max_chars: int, overlap: int) -> list[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    out: list[str] = []
    small: list[str] = []
    for block in blocks:
        if len(block) > max_chars:
            out.extend(_pack(small, max_chars, "\n\n"))
            small = []
            out.extend(_chunk_fixed(block, max_chars, overlap))
        else:
            small.append(block)
    out.extend(_pack(small, max_chars, "\n\n"))
    return out


def _chunk_sentence(text: str, max_chars: int, overlap: int) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return _pack(sentences, max_chars, " ")


_CHUNKERS = {
    "recursive": _chunk_recursive,
    "sentence": _chunk_sentence,
    "fixed": _chunk_fixed,
}


def chunk_text(
    source,
    *,
    method: str = "recursive",
    max_chars: int = 1000,
    overlap: int = 150,
    out_path: str | None = None,
) -> dict[str, str]:
    """Split extracted text into a ``{chunk_id: text}`` corpus.

    source:    a ``{doc_id: text}`` mapping, or a directory of extracted text
               files (e.g. ``data/extracted_text/``).
    method:    'recursive' (default), 'sentence', or 'fixed'.
    max_chars: target chunk size (~4 chars per token).
    overlap:   characters shared between adjacent 'fixed' chunks.
    out_path:  if given, also write the corpus JSON there.
    """
    if method not in _CHUNKERS:
        raise ValueError(
            f"unknown chunk method {method!r}; choose from {sorted(_CHUNKERS)}"
        )
    chunker = _CHUNKERS[method]

    docs = load_text_dir(str(source)) if isinstance(source, str) else dict(source)

    corpus: dict[str, str] = {}
    for doc_id, text in docs.items():
        for i, piece in enumerate(chunker(text, max_chars, overlap)):
            piece = piece.strip()
            if piece:
                corpus[chunk_id_for(doc_id, i)] = piece

    if out_path:
        save_corpus(corpus, out_path)
    return corpus
