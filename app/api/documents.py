"""
app/api/documents.py
GET /documents: one entry per indexed document (chunks grouped by doc_id).
Returns 503 if the vector store cannot be read, so "empty corpus" and "broken store" differ.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from fastapi import APIRouter, HTTPException

from app.schemas import DocumentInfo, DocumentsResponse
from app.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/documents", response_model=DocumentsResponse, summary="List all indexed documents",
            responses={503: {"description": "Vector store unavailable"}})
def list_documents() -> DocumentsResponse:
    try:
        result = get_vectorstore()._collection.get(include=["metadatas"])  # pylint: disable=protected-access
        metadatas = result.get("metadatas") or []
    except Exception:  # noqa: BLE001
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=503, detail="Vector store unavailable.")

    doc_map: dict[str, dict] = defaultdict(lambda: {"title": "", "source": "", "chunk_count": 0})
    for meta in metadatas:
        if not meta:
            continue
        doc_id = meta.get("doc_id", "unknown")
        doc_map[doc_id]["title"] = meta.get("title", doc_id)
        doc_map[doc_id]["source"] = meta.get("source", "")
        doc_map[doc_id]["chunk_count"] += 1

    documents = [DocumentInfo(doc_id=k, **v) for k, v in sorted(doc_map.items())]
    return DocumentsResponse(total=len(documents), documents=documents)
