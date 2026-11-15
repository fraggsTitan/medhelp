"""Data pipeline: PDFs -> pages -> chunks -> embeddings -> Qdrant.

Run it ON YOUR OWN MACHINE (it needs ``pip install -r requirements-ingest.txt`` and ~2 GB of RAM); the
deployed API never imports the heavy libraries used here. It finishes by writing ``data/keyword.sqlite.gz``,
the keyword index that ships with the backend (see ``keyword_db.py``).

    python -m app.ingestion               # fetch (if data/ is empty) + index only NEW/changed PDFs; safe to re-run
    python -m app.ingestion --reindex     # re-embed every PDF (upserts, no duplicates unless chunking changed)
    python -m app.ingestion --recreate    # drop the Qdrant collection first (use after changing chunking/embedding settings)
    python -m app.ingestion --no-fetch    # never download; index whatever is already in data/
    python -m app.ingestion --no-keyword-db   # skip writing data/keyword.sqlite.gz

Design notes
------------
* PDFs are read with LlamaIndex's ``SimpleDirectoryReader`` (one Document per page,
  carrying ``file_name`` and ``page_label`` metadata).
* Chunks are produced by ``SentenceSplitter(chunk_size=512, chunk_overlap=50)``. The
  splitter is given the *embedding model's own tokenizer*, so a "512-token" chunk
  really is <= 512 BGE tokens and is never silently truncated by the embedder.
* Every chunk gets a deterministic UUID derived from (file, physical page, chunk
  number). Re-running ingestion therefore *upserts* (overwrites) instead of
  creating duplicates.
* Heavy libraries (torch, llama-index, qdrant) are imported lazily inside functions
  so that ``--help`` and the unit tests start instantly.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from app import config, fetch_data, textutils
from app.embeddings import BGE_QUERY_INSTRUCTION  # noqa: F401  (passages are embedded WITHOUT it, queries WITH it)
from app.qdrant_store import INGEST_VERSION, collection_point_count, get_qdrant_client  # noqa: F401

logger = logging.getLogger("medhelp.ingest")

MIN_PAGE_CHARS = 30  # pages with less text than this (blank / scanned images) are skipped
_ID_NAMESPACE = uuid.UUID("6f1f0c0e-6b0f-4d6a-9d84-1d7f3d3f0a11")  # fixed; do not change

# Metadata keys hidden from the embedder / LLM text so they don't eat chunk budget.
# (INGEST_VERSION lives in qdrant_store.py: bump it when chunking / cleaning changes and chunks written by an
# older version are detected and rebuilt automatically.)

_EXCLUDED_FROM_EMBED = ["file_name", "publisher", "source_url", "page_label", "pdf_page", "ingest_version"]
_EXCLUDED_FROM_LLM = ["file_name", "publisher", "source_url", "pdf_page", "ingest_version"]

_embed_model: Any = None
_embed_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Shared factories (also used by app.engine so the API process loads the
# 1.3 GB embedding model exactly once)
# --------------------------------------------------------------------------- #
def get_embed_model() -> Any:
    """Return the process-wide HuggingFace embedding model (loaded on first use)."""
    global _embed_model
    with _embed_lock:
        if _embed_model is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            logger.info("Loading embedding model %s (first run downloads ~1.3 GB)...", config.EMBED_MODEL_NAME)
            _embed_model = HuggingFaceEmbedding(
                model_name=config.EMBED_MODEL_NAME,
                query_instruction=BGE_QUERY_INSTRUCTION,
                embed_batch_size=config.EMBED_BATCH_SIZE,
                normalize=True,
                max_length=512,
            )
        return _embed_model


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def pretty_title(filename: str) -> str:
    """'who_hypertension_2021.pdf' -> 'Who Hypertension 2021' (fallback for unknown PDFs)."""
    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    return " ".join(stem.split()).title() or filename


def make_node_id(file_name: str, pdf_page: int, chunk_index: int) -> str:
    """Deterministic UUID so re-ingesting the same PDF overwrites instead of duplicating."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{file_name}|{pdf_page}|{chunk_index}"))


def clean_text(text: str) -> str:
    """Remove characters that break JSON/Qdrant payloads or add no meaning."""
    return text.replace("\x00", "").replace("\ufffe", "").replace("\uffff", "")


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #
def read_pdf_pages(path: Path) -> list[tuple[str, str]]:
    """``[(page_label, text), ...]`` - one entry per physical page, read directly with pypdf.

    Deliberately NOT done through LlamaIndex's SimpleDirectoryReader: depending on the installed reader version
    it may return one Document for the whole PDF, which silently destroys page numbers, bibliography filtering
    and header stripping (every chunk then claims to be "page 1").
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    try:
        labels = list(reader.page_labels)
    except Exception:  # noqa: BLE001 - malformed /PageLabels
        labels = []
    pages: list[tuple[str, str]] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - one bad page must not lose the whole PDF
            logger.warning("%s page %d: text extraction failed (%s)", Path(path).name, i + 1, exc)
            text = ""
        label = labels[i] if i < len(labels) and str(labels[i]).strip() else str(i + 1)
        pages.append((str(label), text))
    return pages


def load_documents(data_dir: Path, only: Optional[Iterable[str]] = None) -> list[Any]:
    """Read PDFs in *data_dir* (or just the file names in *only*) into per-page LlamaIndex Documents.

    Cleaning applied per PDF: running headers/footers are stripped, and pages that are bibliography
    lists are dropped (they never contain an answer but rank highly for any topical query).
    """
    from llama_index.core import Document

    data_dir = Path(data_dir)
    manifest = fetch_data.load_manifest(data_dir)
    if only is not None:
        files = [data_dir / n for n in only]
    else:
        files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() == ".pdf")

    documents: list[Any] = []
    skipped_empty = skipped_refs = 0
    for path in files:
        info = manifest.get(path.name, {})
        raw = read_pdf_pages(path)
        texts = textutils.strip_repeated_lines([clean_text(t) for _, t in raw])
        for physical_page, ((label, _), text) in enumerate(zip(raw, texts), start=1):
            if len(text.strip()) < MIN_PAGE_CHARS:
                skipped_empty += 1
                continue
            if textutils.is_reference_text(text):
                skipped_refs += 1
                continue
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": path.name,
                        "document_name": info.get("title") or pretty_title(path.name),
                        "publisher": info.get("publisher", ""),
                        "source_url": info.get("url", ""),
                        "page_label": label,
                        "pdf_page": physical_page,
                        "ingest_version": INGEST_VERSION,
                    },
                    excluded_embed_metadata_keys=list(_EXCLUDED_FROM_EMBED),
                    excluded_llm_metadata_keys=list(_EXCLUDED_FROM_LLM),
                )
            )
    logger.info(
        "Loaded %d pages from %d PDF(s); skipped %d empty/image-only and %d bibliography page(s).",
        len(documents), len(files), skipped_empty, skipped_refs,
    )
    return documents


_tokenizer_cache: dict[str, Any] = {}


def _embedding_tokenizer() -> Any:
    """Tokenizer function matching the embedding model, or None to use LlamaIndex's default (cached)."""
    if "tok" in _tokenizer_cache:
        return _tokenizer_cache["tok"]
    _tokenizer_cache["tok"] = _load_embedding_tokenizer()
    return _tokenizer_cache["tok"]


def _load_embedding_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(config.EMBED_MODEL_NAME)
        tok.model_max_length = int(1e9)  # silence "sequence longer than 512" warnings; we only count tokens
        return lambda text: tok.encode(text, add_special_tokens=False)
    except Exception as exc:  # network problems, missing transformers, ...
        logger.warning("Could not load the embedding tokenizer (%s); using the default tokenizer.", exc)
        return None


def build_nodes(documents: list[Any]) -> list[Any]:
    """Split pages into overlapping chunks and assign deterministic IDs."""
    # Change 1: Import TokenTextSplitter instead of SentenceSplitter
    from llama_index.core.node_parser import TokenTextSplitter

    # Change 2: Initialize TokenTextSplitter
    splitter = TokenTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        tokenizer=_embedding_tokenizer(),
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

    counters: dict[tuple[str, int], int] = defaultdict(int)
    for node in nodes:
        key = (node.metadata.get("file_name", ""), int(node.metadata.get("pdf_page", 0)))
        node.id_ = make_node_id(key[0], key[1], counters[key])
        counters[key] += 1

    logger.info("Created %d chunks (size=%d, overlap=%d).", len(nodes), config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    return nodes

def indexed_state(client: Any, file_names: Iterable[str]) -> dict[str, Optional[dict[str, Any]]]:
    """For each file name: ``{"document_name", "ingest_version"}`` stored in Qdrant, or None if not indexed.

    Any Qdrant error is treated as "not indexed"; ingestion deletes a file's old points before re-adding it, so
    that is always safe.
    """
    from qdrant_client import models

    names = list(file_names)
    if not client.collection_exists(config.QDRANT_COLLECTION):
        return {n: None for n in names}
    out: dict[str, Optional[dict[str, Any]]] = {}
    for name in names:
        try:
            flt = models.Filter(must=[models.FieldCondition(key="file_name", match=models.MatchValue(value=name))])
            points, _ = client.scroll(
                collection_name=config.QDRANT_COLLECTION,
                scroll_filter=flt,
                limit=1,
                with_payload=["document_name", "ingest_version"],
            )
            out[name] = dict(points[0].payload or {}) if points else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not check index state of %s (%s); will index it.", name, exc)
            out[name] = None
    return out


def indexed_titles(client: Any, file_names: Iterable[str]) -> dict[str, Optional[str]]:
    """Backward-compatible view of :func:`indexed_state`: stored document title or None."""
    return {n: (v.get("document_name", "") if v else None) for n, v in indexed_state(client, file_names).items()}


def audit_collection(client: Any, expected_files: Optional[Iterable[str]] = None) -> dict[str, Any]:
    """Summarise what is really inside the collection (payload fields only, no vectors, no text)."""
    from collections import Counter

    fields = ["file_name", "document_name", "page_label", "pdf_page", "ingest_version"]
    per_file: Counter[str] = Counter()
    versions: Counter[str] = Counter()
    titles: dict[str, set[str]] = defaultdict(set)
    page_one = total = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION, limit=1000, offset=offset, with_payload=fields, with_vectors=False
        )
        for pt in points:
            pl = pt.payload or {}
            total += 1
            fname = pl.get("file_name") or "(no file_name)"
            per_file[fname] += 1
            versions[str(pl.get("ingest_version", "none"))] += 1
            titles[fname].add(str(pl.get("document_name", "")))
            page_one += str(pl.get("pdf_page", pl.get("page_label", ""))) == "1"
        if offset is None or not points:
            break
    expected = set(expected_files or [])
    return {
        "total_points": total,
        "files_in_index": len(per_file),
        "chunks_per_file": dict(per_file.most_common()),
        "ingest_versions": dict(versions),
        "files_with_several_titles": [f for f, t in titles.items() if len(t) > 1],
        "share_of_chunks_labelled_page_1": round(page_one / total, 3) if total else 0.0,
        "files_not_in_data_dir": sorted(set(per_file) - expected) if expected else [],
        "files_in_data_dir_but_not_indexed": sorted(expected - set(per_file)) if expected else [],
    }


def delete_file_points(client: Any, file_name: str) -> None:
    from qdrant_client import models

    flt = models.Filter(must=[models.FieldCondition(key="file_name", match=models.MatchValue(value=file_name))])
    client.delete(collection_name=config.QDRANT_COLLECTION, points_selector=models.FilterSelector(filter=flt))


def run_ingestion(
    recreate: bool = False, fetch_if_missing: bool = True, reindex: bool = False, keyword_db: bool = True
) -> dict[str, Any]:
    """End-to-end ingestion, one PDF at a time. Returns a small summary dict.

    By default only PDFs that are not yet in the index (or whose title changed in the manifest) are processed, so
    adding guidelines to an existing index is cheap and an interrupted run loses at most one document.
    """
    config.validate(need_llm=False)
    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs and fetch_if_missing:
        logger.info("No PDFs found in %s - running the automated fetch first.", data_dir)
        fetch_data.fetch_all(data_dir)
        pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(
            f"No PDFs available in {data_dir}. Run `python -m app.fetch_data` or add PDFs manually."
        )

    manifest = fetch_data.sync_manifest(data_dir)  # readable titles for PDFs added by hand
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    client = get_qdrant_client()
    if recreate and client.collection_exists(config.QDRANT_COLLECTION):
        logger.info("Dropping existing collection %s", config.QDRANT_COLLECTION)
        client.delete_collection(config.QDRANT_COLLECTION)

    # Decide what needs work: new files, changed titles, or chunks written by an older ingestion version.
    state = indexed_state(client, [p.name for p in pdfs])
    pending: list[Path] = []
    for pdf in pdfs:
        title = manifest.get(pdf.name, {}).get("title") or pretty_title(pdf.name)
        stored = state.get(pdf.name)
        if reindex or stored is None:
            pending.append(pdf)
        elif stored.get("ingest_version") != INGEST_VERSION:
            logger.info("%s: indexed by an older ingestion version (%s); rebuilding.", pdf.name, stored.get("ingest_version"))
            pending.append(pdf)
        elif stored.get("document_name") != title:
            logger.info("%s: title changed ('%s' -> '%s'); re-indexing.", pdf.name, stored.get("document_name"), title)
            pending.append(pdf)
    skipped = len(pdfs) - len(pending)
    logger.info("%d PDF(s) to index, %d already up to date.", len(pending), skipped)

    vector_store = QdrantVectorStore(client=client, collection_name=config.QDRANT_COLLECTION)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    total_pages = total_nodes = 0
    failed: list[str] = []
    for n, pdf in enumerate(pending, start=1):
        logger.info("[%d/%d] Indexing %s", n, len(pending), pdf.name)
        try:
            documents = load_documents(data_dir, only=[pdf.name])
            if not documents:
                logger.warning("%s has no extractable text (scanned images?) - skipped.", pdf.name)
                failed.append(pdf.name)
                continue
            nodes = build_nodes(documents)
            if state.get(pdf.name) is not None:  # drop the old chunks first: boundaries differ, so overwriting leaves orphans
                delete_file_points(client, pdf.name)
            VectorStoreIndex(nodes, storage_context=storage_context, embed_model=get_embed_model(), show_progress=True)
            total_pages += len(documents)
            total_nodes += len(nodes)
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not abort the whole run
            logger.error("Failed to index %s: %s", pdf.name, exc)
            failed.append(pdf.name)

    total = collection_point_count(client)
    logger.info("Done. Collection '%s' now holds %d vectors.", config.QDRANT_COLLECTION, total)

    keyword_summary: Any = "skipped"
    if keyword_db:
        # The deployed API searches keywords through this small SQLite file instead of rebuilding an in-memory
        # index on every cold start. Ship it with the backend (commit data/keyword.sqlite.gz).
        try:
            from app.keyword_db import build_from_qdrant

            keyword_summary = build_from_qdrant(client, config.QDRANT_COLLECTION, config.KEYWORD_DB_PATH)
        except Exception as exc:  # noqa: BLE001 - the vector index is already safe in Qdrant
            logger.error("Could not write the keyword database (%s). Run `python -m app.keyword_db build` later.", exc)
            keyword_summary = f"FAILED: {exc}"
    return {
        "pdfs": len(pdfs),
        "indexed_now": len(pending) - len(failed),
        "already_indexed": skipped,
        "failed": failed,
        "pages": total_pages,
        "nodes": total_nodes,
        "collection": config.QDRANT_COLLECTION,
        "points_in_collection": total,
        "keyword_db": keyword_summary,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Parse, chunk, embed and upload guideline PDFs to Qdrant.")
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild the Qdrant collection")
    parser.add_argument("--audit", action="store_true", help="print what is really inside the Qdrant collection and exit")
    parser.add_argument("--reindex", action="store_true", help="re-embed every PDF, not just new/changed ones")
    parser.add_argument("--no-fetch", action="store_true", help="do not download PDFs automatically")
    parser.add_argument("--no-keyword-db", action="store_true", help="do not (re)write data/keyword.sqlite.gz afterwards")
    args = parser.parse_args(argv)
    if args.audit:
        import json

        logging.basicConfig(level=config.LOG_LEVEL, format="%(levelname)s  %(message)s")
        names = [p.name for p in Path(config.DATA_DIR).glob("*.pdf")]
        print(json.dumps(audit_collection(get_qdrant_client(), names), indent=2))
        return 0

    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
    try:
        summary = run_ingestion(
            recreate=args.recreate, fetch_if_missing=not args.no_fetch, reindex=args.reindex, keyword_db=not args.no_keyword_db
        )
    except (config.ConfigError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("\nIngestion complete:")
    for key, value in summary.items():
        print(f"  {key:<22} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
