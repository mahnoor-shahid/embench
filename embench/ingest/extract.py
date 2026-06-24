"""Stage 1 -- text extraction. Multiple swappable methods.

    docling   default. Layout/table aware, runs OCR on scanned pages via a
              traditional OCR engine. Robust but heavy.
    pymupdf   fast, born-digital PDFs only (no OCR). Good when you know the
              PDFs already contain a text layer.
    granite   IBM Granite-Docling, a vision-language model (VLM) that "reads"
              each page image. Best on complex/messy layouts; heaviest.
              Runs through Docling's VLM pipeline.

All extraction backends (docling, pymupdf) ship with embench itself.

Each method takes a document path and returns its text (markdown).
"""
from __future__ import annotations

from .common import (
    DEFAULT_EXTENSIONS,
    collect_paths,
    doc_id_for,
    save_text_dir,
)


def _extract_docling(path: str, ocr: bool) -> str:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "docling is required for method='docling'. It ships with embench, "
            "so reinstall it with: pip install embench"
        ) from exc

    if ocr:
        converter = DocumentConverter()  # OCR is enabled by default
    else:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption

        options = PdfPipelineOptions(do_ocr=False)
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
    return converter.convert(path).document.export_to_markdown()


def _extract_pymupdf(path: str, ocr: bool) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyMuPDF is required for method='pymupdf'. It ships with embench, "
            "so reinstall it with: pip install embench"
        ) from exc
    # PyMuPDF reads the existing text layer only -- no OCR for scanned pages.
    with fitz.open(path) as doc:
        return "\n\n".join(page.get_text() for page in doc)


def _extract_granite(path: str, ocr: bool) -> str:
    """Extract with IBM Granite-Docling (a VLM) via Docling's VLM pipeline.

    Unlike the OCR-based ``docling`` method, the VLM looks at the rendered page
    image and emits structured text directly -- stronger on complex layouts.
    ``ocr`` is ignored here (the model does not use a separate OCR engine).
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "docling (with VLM support) is required for method='granite'. It "
            "ships with embench, so reinstall it with: pip install embench"
        ) from exc

    # Prefer the explicit Granite-Docling model spec; fall back to the VLM
    # pipeline default (Granite-Docling in recent Docling releases). The exact
    # constant name has shifted across versions, so resolve it defensively.
    vlm_options = None
    try:
        from docling.datamodel import vlm_model_specs

        for spec_name in (
            "GRANITEDOCLING_TRANSFORMERS",
            "GRANITE_DOCLING_TRANSFORMERS",
            "GRANITEDOCLING_MLX",
        ):
            if hasattr(vlm_model_specs, spec_name):
                vlm_options = getattr(vlm_model_specs, spec_name)
                break
    except Exception:  # noqa: BLE001 - any import/layout change -> use default
        vlm_options = None

    pipeline_options = (
        VlmPipelineOptions(vlm_options=vlm_options)
        if vlm_options is not None
        else VlmPipelineOptions()
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
            )
        }
    )
    return converter.convert(path).document.export_to_markdown()


_EXTRACTORS = {
    "docling": _extract_docling,
    "pymupdf": _extract_pymupdf,
    "granite": _extract_granite,
}


def extract_text(
    source,
    *,
    method: str = "docling",
    ocr: bool = True,
    recursive: bool = True,
    extensions=DEFAULT_EXTENSIONS,
    out_dir: str | None = None,
    show_progress: bool = True,
) -> dict[str, str]:
    """Extract text from documents into ``{doc_id: text}``.

    source:     a document, a folder of documents, or a list of paths.
    method:     'docling' (OCR, robust) or 'pymupdf' (fast, born-digital).
    ocr:        run OCR on scanned pages (docling only).
    out_dir:    if given, also persist one ``<doc_id>.md`` per document there,
                so the chunk stage can run later without re-extracting.
    """
    if method not in _EXTRACTORS:
        raise ValueError(
            f"unknown extract method {method!r}; choose from {sorted(_EXTRACTORS)}"
        )
    extractor = _EXTRACTORS[method]

    extensions = tuple(e.lower() for e in extensions)
    paths = collect_paths(source, extensions, recursive)
    if not paths:
        raise ValueError(f"no documents found under {source!r} (extensions={extensions})")

    docs: dict[str, str] = {}
    for path in paths:
        if show_progress:
            import os

            print(f"  [extract:{method}] {os.path.basename(path)}")
        text = (extractor(path, ocr) or "").strip()
        if text:
            docs[doc_id_for(path)] = text

    if out_dir:
        save_text_dir(docs, out_dir)
    return docs
