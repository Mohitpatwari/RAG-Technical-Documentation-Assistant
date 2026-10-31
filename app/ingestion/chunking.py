"""
app/ingestion/chunking.py
─────────────────────────
Two-stage chunking strategy:

Stage 1 — MarkdownHeaderTextSplitter
  Splits on H1/H2/H3 headers so that every chunk is topically coherent
  and carries a "section" breadcrumb (e.g. "Models > Field Types").
  This avoids embedding chunks that mix two unrelated topics.

Stage 2 — RecursiveCharacterTextSplitter
  Further splits long sections into ~800-char pieces with 150-char overlap.
  The recursive splitter tries separators in order:
    \\n\\n  →  \\n  →  `` ` `` `` ` `` `` ` `` (code fence boundary)  →  space
  By listing the code-fence separator explicitly we minimise mid-code-block
  splits: the splitter will prefer to break at a blank line outside a fence.

Chunk IDs are deterministic SHA-256 hashes of (doc_id + chunk_index) so
re-ingesting the same file does not create duplicate embeddings (idempotent
upsert).

Metadata attached to every chunk:
  source       — file path or URL
  title        — document title
  doc_id       — stable document identifier
  section      — header breadcrumb path (e.g. "Path Parameters > Examples")
  chunk_index  — position of this chunk within the document
  chunk_id     — deterministic unique chunk identifier (used as Chroma ID)
"""

from __future__ import annotations

import hashlib
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 900          # target chunk size in characters
CHUNK_OVERLAP = 150       # overlap to preserve cross-chunk context

# Headers to split on (level, metadata key)
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# Separator priority for the second-stage splitter.
# Listing the code-fence boundary first gives the splitter a chance to
# break at ``` rather than mid-code-block.
RECURSIVE_SEPARATORS = [
    "\n```\n",   # code block boundary
    "\n\n",      # paragraph break
    "\n",        # line break
    " ",         # word break (last resort)
    "",
]


def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Create a deterministic, collision-resistant chunk ID."""
    raw = f"{doc_id}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _build_section_path(metadata: dict) -> str:
    """Concatenate header values into a breadcrumb string."""
    parts = [metadata.get(k, "") for k in ("h1", "h2", "h3")]
    return " > ".join(p for p in parts if p)


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split raw documents into indexed, metadata-rich chunks.

    Args:
        documents: List of Documents (one per source file/URL).

    Returns:
        List of chunk Documents, each with full metadata.
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,   # keep headers in chunk text for context
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=RECURSIVE_SEPARATORS,
        keep_separator=True,
    )

    all_chunks: List[Document] = []

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        title = doc.metadata.get("title", "Untitled")
        doc_id = doc.metadata.get("doc_id", "unknown")

        logger.debug("Chunking document: %s (%d chars)", doc_id, len(doc.page_content))

        # Stage 1: split on Markdown headers
        try:
            header_splits = md_splitter.split_text(doc.page_content)
        except Exception:
            # Fallback for non-Markdown content
            header_splits = [Document(page_content=doc.page_content, metadata={})]

        # Stage 2: split long sections further
        stage2_chunks: List[Document] = []
        for split in header_splits:
            if len(split.page_content) <= CHUNK_SIZE:
                stage2_chunks.append(split)
            else:
                sub = char_splitter.split_documents([split])
                stage2_chunks.extend(sub)

        # Attach full metadata and deterministic IDs
        for idx, chunk in enumerate(stage2_chunks):
            section = _build_section_path(chunk.metadata)
            chunk_id = _make_chunk_id(doc_id, idx)

            chunk.metadata.update(
                {
                    "source": source,
                    "title": title,
                    "doc_id": doc_id,
                    "section": section,
                    "chunk_index": idx,
                    "chunk_id": chunk_id,
                }
            )
            all_chunks.append(chunk)

    logger.info(
        "Chunking complete: %d source docs → %d chunks", len(documents), len(all_chunks)
    )
    return all_chunks
