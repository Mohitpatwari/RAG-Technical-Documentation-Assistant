"""
app/main.py
FastAPI entry point: lifespan (auto-ingest, graph warm-up, Mermaid export), CORS, request
logging, uniform JSON error envelope {error, detail, status_code}, health endpoint.
Endpoints are served at the root (/query, /ingest, /documents, /feedback, /health).
Start from the project root so .env is found.
"""

from __future__ import annotations
from fastapi.responses import JSONResponse, RedirectResponse

import logging
import time
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import documents, feedback, ingest, query
from app.config import settings
from app.schemas import ErrorDetail, HealthResponse
from app.vectorstore import get_collection_count

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting RAG Docs Assistant ...")
    if not settings.google_api_key:
        logger.warning("GOOGLE_API_KEY is empty - /query will return 503 until it is set in .env")

    try:
        count = get_collection_count()
        if count == 0:
            logger.info("Vector store is empty - running auto-ingestion (first start is slow) ...")
            from app.ingestion.pipeline import ingest_corpus

            corpus_dir = Path(settings.corpus_dir)
            if corpus_dir.exists() and any(corpus_dir.iterdir()):
                res = ingest_corpus(corpus_dir, manifest_path=corpus_dir / "manifest.json")
                logger.info("Auto-ingestion complete: %d docs, %d chunks", res["docs_ingested"], res["chunks_ingested"])
            else:
                logger.warning("Corpus dir '%s' is empty. Run: python scripts/fetch_corpus.py", corpus_dir)
        else:
            logger.info("Vector store ready: %d chunks indexed", count)
    except Exception as exc:  # noqa: BLE001
        logger.error("Auto-ingestion failed: %s", exc)

    try:
        from app.graph.builder import get_graph, save_mermaid_diagram

        get_graph()
        save_mermaid_diagram()
        logger.info("Graph compiled and ready (diagram saved to docs/graph.md)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Graph warm-up failed (will retry on first request): %s", exc)

    yield
    logger.info("Shutting down RAG Docs Assistant")


app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description="""
A self-corrective Retrieval-Augmented Generation assistant built with **LangGraph** and **FastAPI**.

1. `POST /query` classifies and rewrites your question, retrieves chunks from ChromaDB, grades each
   chunk for relevance, and generates a cited answer, retrying with a new query if retrieval fails.
2. `POST /ingest` adds documentation (file upload or URLs).
3. `GET /documents` lists indexed documents.
4. `POST /feedback` rates a response (use the `query_id` from `/query`).

Indexed corpus: **FastAPI** and **Pydantic** official docs.
""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path, response.status_code,
                (time.perf_counter() - start) * 1000)
    return response


def _envelope(status_code: int, error: str, detail=None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorDetail(error=error, detail=detail, status_code=status_code).model_dump(mode="json"),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    try:
        phrase = HTTPStatus(exc.status_code).phrase
    except ValueError:
        phrase = "Error"
    return _envelope(exc.status_code, phrase, exc.detail)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [{"loc": [str(p) for p in e["loc"]], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return _envelope(422, "Validation error", errors)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception")
    return _envelope(500, "Internal server error", "An unexpected error occurred.")


app.include_router(query.router, tags=["RAG"])
app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(documents.router, tags=["Documents"])
app.include_router(feedback.router, tags=["Feedback"])


@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Health check")
def health_check() -> HealthResponse:
    """`degraded` if the vector store is unreachable or empty."""
    try:
        count = get_collection_count()
        return HealthResponse(status="ok" if count > 0 else "degraded", indexed_chunks=count)
    except Exception:  # noqa: BLE001
        return HealthResponse(status="degraded", indexed_chunks=0)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")