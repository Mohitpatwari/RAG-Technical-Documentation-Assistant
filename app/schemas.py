"""
app/schemas.py
All FastAPI request/response Pydantic v2 models (the API contract in one place).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


# ── /query ─────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000,
                          description="The user's natural-language question.",
                          examples=["How do I declare path parameters in FastAPI?"])
    session_id: Optional[str] = Field(default=None, max_length=100,
                                      description="Optional session identifier for multi-turn conversations.")

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class SourceCitation(BaseModel):
    title: str = Field(description="Document title.")
    source: str = Field(description="Source URL or file name.")
    section: str = Field(description="Header path within the document.")
    snippet: str = Field(description="Short excerpt from the chunk (<= 200 chars).")


class QueryResponse(BaseModel):
    query_id: str = Field(description="Unique ID for this query (use in /feedback).")
    answer: str
    sources: List[SourceCitation] = Field(default_factory=list, description="Sources cited in the answer.")
    query_type: str
    rewritten_query: str = Field(description="The (possibly rewritten) query used for retrieval.")
    retries_used: int
    grounded: Optional[bool] = Field(default=None,
                                     description="Hallucination-check result (null if disabled or no answer).")
    used_web_search: bool = False
    trace: List[str] = Field(default_factory=list, description="Graph nodes visited, in order.")
    latency_ms: float


# ── /ingest ────────────────────────────────────────────────────────────────────
class URLIngestRequest(BaseModel):
    urls: List[HttpUrl] = Field(..., min_length=1, max_length=10,
                                description="Documentation URLs to fetch and index (max 10).")


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_ingested: int
    doc_ids: List[str]


# ── /documents ─────────────────────────────────────────────────────────────────
class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    source: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    total: int
    documents: List[DocumentInfo]


# ── /feedback ──────────────────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    query_id: str = Field(description="The query_id returned by /query.")
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    status: str = "accepted"
    query_id: str


# ── /health ────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str = "1.0.0"
    indexed_chunks: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Error envelope ─────────────────────────────────────────────────────────────
class ErrorDetail(BaseModel):
    error: str
    detail: Any = None
    status_code: int
