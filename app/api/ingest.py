"""
app/api/ingest.py
POST /ingest: multipart file upload (.md/.txt/.html) OR JSON {"urls": [...]}.

Blocking work (fetching, embedding) runs in a threadpool so the event loop stays free.
Behaviour note: if one URL/file in a request is invalid, the whole request fails with 400
and nothing is ingested.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.ingestion.loaders import load_file, load_url
from app.ingestion.pipeline import ingest_documents
from app.schemas import IngestResponse, URLIngestRequest

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".md", ".txt", ".html", ".htm"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB per file


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documentation files or URLs",
    responses={400: {"description": "Unsupported file type or unfetchable URL"},
               413: {"description": "File too large (> 10 MB)"}},
)
async def ingest_endpoint(request: Request) -> IngestResponse:
    """**Files:** `curl -X POST /ingest -F "files=@doc.md"`
    **URLs:** `{"urls": ["https://fastapi.tiangolo.com/tutorial/path-params/"]}`"""
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            body = URLIngestRequest.model_validate(json.loads(await request.body()))
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}")

        def _load_urls():
            docs = []
            for url in body.urls:
                docs.extend(load_url(str(url)))
            return docs

        try:
            docs = await run_in_threadpool(_load_urls)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    elif "multipart/form-data" in content_type:
        form = await request.form()
        uploads = form.getlist("files")
        if not uploads:
            raise HTTPException(status_code=400, detail="No files provided in form data.")

        docs = []
        for upload in uploads:
            filename = upload.filename or "unknown"
            ext = Path(filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400,
                                    detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")
            content = await upload.read(MAX_FILE_SIZE_BYTES + 1)
            if len(content) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(status_code=413, detail=f"File '{filename}' exceeds 10 MB limit.")

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)
            try:
                loaded = load_file(tmp_path, title=Path(filename).stem)
                for d in loaded:
                    d.metadata["source"] = filename
                    d.metadata["doc_id"] = Path(filename).stem
                docs.extend(loaded)
            finally:
                tmp_path.unlink(missing_ok=True)
    else:
        raise HTTPException(status_code=400,
                            detail="Unsupported Content-Type. Use 'multipart/form-data' or 'application/json'.")

    chunks = await run_in_threadpool(ingest_documents, docs)
    doc_ids: List[str] = [d.metadata.get("doc_id", "") for d in docs]
    return IngestResponse(documents_ingested=len(docs), chunks_ingested=chunks, doc_ids=doc_ids)
