"""Stage 3 -- query + qrel generation. Turn a corpus into a benchmark.

Stages 1-2 produce only the *corpus* half of a retrieval benchmark. To score
retrieval you also need supervision: *queries* and *qrels* (which chunk answers
which query). This stage synthesises that with an LLM -- for each chunk, ask a
model to write questions the chunk answers, then mark that chunk relevant to
those questions (relevance 1). The result is a full ``RetrievalDataset``.

    openai    default. Chat Completions; default model ``gpt-4o-mini``.
    google    Gemini via the Google GenAI SDK; default ``gemini-2.5-flash``.

Both SDKs ship with embench itself (no extra needed); they read their keys
from the environment (``OPENAI_API_KEY`` / ``GOOGLE_API_KEY``) -- call
``embench.load_env()`` first if you keep them in a ``.env`` file. Pass your own
``generator`` callable to plug in any other model (and to run fully offline).
"""
from __future__ import annotations

import json
import re

from .common import load_corpus, query_id_for, save_dataset

_PROMPT = (
    "You are generating evaluation questions for a document-retrieval benchmark.\n"
    "Read the passage below and write {n} distinct, self-contained question(s) "
    "that the passage directly and specifically answers. Each question must make "
    "sense on its own (do not write 'according to the passage') and must be "
    "answerable *only* from this passage, not from general knowledge.\n"
    'Return ONLY a JSON array of question strings, e.g. ["...", "..."].\n\n'
    "Passage:\n{text}"
)


def _parse_questions(raw: str, n: int) -> list[str]:
    """Pull a list of question strings out of a model response, defensively."""
    raw = (raw or "").strip()
    if not raw:
        return []
    # Strip a ```json ... ``` fence if the model wrapped its answer.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):  # e.g. {"questions": [...]}
            data = next((v for v in data.values() if isinstance(v, list)), [])
        if isinstance(data, list):
            qs = [str(q).strip() for q in data if str(q).strip()]
            return qs[:n]
    except (ValueError, TypeError):
        pass
    # Fallback: one question per line, stripping list markers / numbering.
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip('"')
        if line:
            lines.append(line)
    return lines[:n]


def _openai_completer(model: str | None):
    from openai import OpenAI

    client = OpenAI()
    model = model or "gpt-4o-mini"

    def complete(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""

    return model, complete


def _google_completer(model: str | None):
    from google import genai

    client = genai.Client()
    model = model or "gemini-2.5-flash"

    def complete(prompt: str) -> str:
        resp = client.models.generate_content(model=model, contents=prompt)
        return resp.text or ""

    return model, complete


_COMPLETERS = {
    "openai": _openai_completer,
    "google": _google_completer,
}


def generate_queries(
    source,
    *,
    method: str = "openai",
    model: str | None = None,
    n_queries: int = 1,
    max_chunks: int | None = None,
    generator=None,
    name: str = "retrieval",
    out_path: str | None = None,
    show_progress: bool = True,
) -> dict:
    """Generate queries + qrels for a corpus, returning a full RetrievalDataset.

    source:     a ``{chunk_id: text}`` corpus, or a path to corpus JSON (as
                written by ``save_corpus`` / ``embench ingest chunk``).
    method:     'openai' (default) or 'google'. Ignored if ``generator`` given.
    model:      override the backend's default model id.
    n_queries:  questions to generate per chunk (default 1).
    max_chunks: only use the first N chunks (cost control); ``None`` = all.
    generator:  optional ``callable(text, n) -> list[str]`` to plug in any
                model or run offline; bypasses ``method``/``model`` entirely.
    out_path:   if given, also write the dataset JSON there (loadable with
                ``RetrievalDataset.from_json``).

    Returns ``{"queries": ..., "corpus": ..., "qrels": ..., "name": ...}``.
    Chunks the model returns no questions for are skipped.
    """
    corpus = load_corpus(source)
    if not corpus:
        raise ValueError(f"empty corpus from {source!r}; run extract+chunk first")

    if generator is None:
        if method not in _COMPLETERS:
            raise ValueError(
                f"unknown query method {method!r}; choose from {sorted(_COMPLETERS)}"
            )
        model, complete = _COMPLETERS[method](model)

        def generator(text, n):  # noqa: F811 - bind the chosen backend
            return _parse_questions(complete(_PROMPT.format(n=n, text=text)), n)

    items = list(corpus.items())
    if max_chunks is not None:
        items = items[:max_chunks]

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    for chunk_id, text in items:
        if show_progress:
            print(f"  [queries:{method}] {chunk_id}")
        for i, question in enumerate(generator(text, n_queries)):
            question = question.strip()
            if not question:
                continue
            qid = query_id_for(chunk_id, i)
            queries[qid] = question
            qrels[qid] = {chunk_id: 1}

    if not queries:
        raise ValueError(
            "no queries were generated (the model returned nothing parseable). "
            "Check your API key, or pass a `generator` callable."
        )

    dataset = {"queries": queries, "corpus": corpus, "qrels": qrels, "name": name}
    if out_path:
        save_dataset(dataset, out_path)
    return dataset
