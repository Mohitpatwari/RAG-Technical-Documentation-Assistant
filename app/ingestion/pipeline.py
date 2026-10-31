"""
app/ingestion/pipeline.py
load -> chunk -> embed -> upsert.

Idempotent: before upserting a document, all existing chunks with the same doc_id are deleted,
so re-ingesting a shorter/changed document leaves no stale chunks behind. Chunk IDs are
deterministic, so re-ingesting an unchanged document yields the same index.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import load_file, load_url
from app.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 30   # drops heading-only / empty chunks that hurt retrieval
BATCH_SIZE = 100       # stay well under Chroma's max batch size


def ingest_documents(documents: List[Document], show_progress: bool = False) -> int:
    """Chunk, embed and upsert documents. Returns the number of chunks stored."""
    if not documents:
        logger.warning("ingest_documents called with empty list")
        return 0

    chunks = [c for c in chunk_documents(documents) if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]
    if not chunks:
        logger.warning("Chunking produced 0 usable chunks")
        return 0

    vs = get_vectorstore()

    # Remove stale chunks of the documents being (re-)ingested
    for doc_id in {d.metadata.get("doc_id") for d in documents if d.metadata.get("doc_id")}:
        try:
            vs._collection.delete(where={"doc_id": doc_id})  # pylint: disable=protected-access
        except Exception:  # noqa: BLE001
            logger.warning("Could not delete old chunks for doc_id=%s", doc_id, exc_info=True)

    if show_progress:
        print(f"  Upserting {len(chunks)} chunks into ChromaDB ...", end=" ", flush=True)

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vs.add_texts(
            texts=[c.page_content for c in batch],
            metadatas=[c.metadata for c in batch],
            ids=[c.metadata["chunk_id"] for c in batch],
        )

    if show_progress:
        print("done")
    logger.info("Upserted %d chunks from %d docs", len(chunks), len(documents))
    return len(chunks)


def ingest_file(path: Path, title: Optional[str] = None, doc_id: Optional[str] = None,
                show_progress: bool = False) -> int:
    return ingest_documents(load_file(path, title=title, doc_id=doc_id), show_progress=show_progress)


def ingest_url(url: str, title: Optional[str] = None, show_progress: bool = False) -> int:
    return ingest_documents(load_url(url, title=title), show_progress=show_progress)


def ingest_corpus(corpus_dir: Path, manifest_path: Optional[Path] = None) -> dict:
    """Ingest the corpus. With a manifest, `source` is set to the real documentation URL
    (so citations link to the web page, not a local path)."""
    docs: List[Document] = []
    doc_ids: List[str] = []

    if manifest_path and manifest_path.exists():
        for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
            fpath = corpus_dir / entry["filename"]
            if not fpath.exists():
                logger.warning("Corpus file missing: %s", fpath)
                continue
            doc_id = entry.get("doc_id", fpath.stem)
            loaded = load_file(fpath, title=entry.get("title"), doc_id=doc_id)
            for d in loaded:
                d.metadata["file_path"] = str(fpath)
                if entry.get("source_url"):
                    d.metadata["source"] = entry["source_url"]
            docs.extend(loaded)
            doc_ids.append(doc_id)
    else:
        for fpath in sorted(corpus_dir.glob("**/*")):
            if fpath.suffix.lower() in (".md", ".txt", ".html") and fpath.name != "manifest.json":
                docs.extend(load_file(fpath))
                doc_ids.append(fpath.stem)

    chunks = ingest_documents(docs, show_progress=True)
    return {"docs_ingested": len(docs), "chunks_ingested": chunks, "doc_ids": doc_ids}
