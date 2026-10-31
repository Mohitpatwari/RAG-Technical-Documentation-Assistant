"""
tests/test_api.py
─────────────────
FastAPI endpoint validation tests.

Uses httpx.AsyncClient with the FastAPI app directly (no running server needed).
Tests validation errors: 422, 400, 404, 413.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest_asyncio.fixture
async def client():
    """Async test client for the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ── /health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_ok(client):
    with patch("app.main.get_collection_count", return_value=42):
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")


# ── /query validation ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_empty_question_returns_422(client):
    resp = await client.post("/query", json={"question": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_query_blank_whitespace_returns_422(client):
    resp = await client.post("/query", json={"question": "   "})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_query_too_long_returns_422(client):
    long_question = "a" * 1001
    resp = await client.post("/query", json={"question": long_question})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_query_missing_question_returns_422(client):
    resp = await client.post("/query", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_query_exactly_1000_chars_is_valid(client):
    """A 1000-char question is at the limit and should be accepted by validation."""
    question = "a" * 1000
    # We don't actually run the graph — just mock it out
    mock_state = {
        "answer": "Test answer",
        "citations": [],
        "query_type": "conceptual",
        "rewritten_query": question,
        "retry_count": 0,
        "grounded": None,
        "used_web_search": False,
        "trace": ["analyze_query", "retrieve", "grade_documents", "generate"],
    }
    with patch("app.api.query.get_graph") as mock_get_graph:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_state
        mock_get_graph.return_value = mock_graph
        resp = await client.post("/query", json={"question": question})
    # Should not fail on validation (422); may fail on LLM but that's fine
    assert resp.status_code != 422


# ── /feedback validation ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_feedback_unknown_query_id_returns_404(client):
    resp = await client.post(
        "/feedback",
        json={
            "query_id": "nonexistent-id-12345",
            "rating": "up",
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_feedback_invalid_rating_returns_422(client):
    resp = await client.post(
        "/feedback",
        json={"query_id": "some-id", "rating": "meh"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_feedback_accepts_valid_known_id(client, tmp_path):
    """Register a query_id manually, then submit feedback."""
    from app.api.query import known_query_ids

    test_id = "test-feedback-query-id"
    known_query_ids.add(test_id)

    with patch("app.api.feedback.settings") as mock_settings:
        mock_settings.feedback_file = str(tmp_path / "feedback.jsonl")
        resp = await client.post(
            "/feedback",
            json={"query_id": test_id, "rating": "up", "comment": "Great answer!"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["query_id"] == test_id
    assert data["status"] == "accepted"

    known_query_ids.discard(test_id)


# ── /ingest validation ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_unsupported_file_type_returns_400(client):
    resp = await client.post(
        "/ingest",
        files={"files": ("document.pdf", b"fake pdf content", "application/pdf")},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_ingest_file_too_large_returns_413(client):
    large_content = b"x" * (11 * 1024 * 1024)  # 11 MB
    resp = await client.post(
        "/ingest",
        files={"files": ("big_doc.md", large_content, "text/markdown")},
    )
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_ingest_url_bad_content_type_returns_400(client):
    """Sending an invalid content type should return 400."""
    resp = await client.post(
        "/ingest",
        content=b"raw bytes",
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 400


# ── /documents ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_documents_returns_list(client):
    # conftest.py's autouse mock handles the vectorstore
    resp = await client.get("/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "documents" in data
