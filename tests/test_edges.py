"""
tests/test_edges.py
───────────────────
Unit tests for conditional routing functions in app/graph/edges.py.

Each test is pure (no LLM, no DB) — we only construct GraphState dicts
and assert that the routing function returns the expected node name.

Three branches covered for route_after_grading:
  1. Relevant documents exist        → "generate"
  2. No relevant docs, retries left  → "rewrite_query"
  3. Retries exhausted, no Tavily    → "fallback"
  4. Retries exhausted, Tavily set   → "web_search"

Two branches for route_after_web_search:
  5. Web results found               → "generate"
  6. No web results                  → "fallback"
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from app.graph.edges import route_after_grading, route_after_web_search


# ── Helpers ────────────────────────────────────────────────────────────────────

def _doc(content: str = "test") -> Document:
    return Document(page_content=content, metadata={"chunk_id": "abc"})


def _make_state(**kwargs) -> dict:
    """Build a minimal GraphState-like dict."""
    base = {
        "question": "What is FastAPI?",
        "retry_count": 0,
        "max_retries": 2,
        "relevant_documents": [],
        "documents": [],
        "failed_queries": [],
        "trace": [],
    }
    base.update(kwargs)
    return base


# ── route_after_grading ────────────────────────────────────────────────────────

class TestRouteAfterGrading:
    def test_branch_generate_when_relevant_docs(self):
        """Branch 1: relevant documents exist → generate."""
        state = _make_state(
            relevant_documents=[_doc("FastAPI path parameters"), _doc("validation")],
            retry_count=0,
        )
        result = route_after_grading(state)
        assert result == "generate"

    def test_branch_rewrite_when_no_relevant_and_retries_left(self):
        """Branch 2: no relevant docs, retries remaining → rewrite_query."""
        state = _make_state(
            relevant_documents=[],
            retry_count=0,
            max_retries=2,
        )
        with patch("app.graph.edges.settings") as mock_settings:
            mock_settings.max_retries = 2
            mock_settings.tavily_api_key = ""
            result = route_after_grading(state)
        assert result == "rewrite_query"

    def test_branch_fallback_when_retries_exhausted_no_tavily(self):
        """Branch 3: retries exhausted, no Tavily key → fallback."""
        state = _make_state(
            relevant_documents=[],
            retry_count=2,
            max_retries=2,
        )
        with patch("app.graph.edges.settings") as mock_settings:
            mock_settings.max_retries = 2
            mock_settings.tavily_api_key = ""
            result = route_after_grading(state)
        assert result == "fallback"

    def test_branch_web_search_when_retries_exhausted_tavily_set(self):
        """Branch 4: retries exhausted, Tavily key present → web_search."""
        state = _make_state(
            relevant_documents=[],
            retry_count=2,
            max_retries=2,
        )
        with patch("app.graph.edges.settings") as mock_settings:
            mock_settings.max_retries = 2
            mock_settings.tavily_api_key = "test_key_123"
            result = route_after_grading(state)
        assert result == "web_search"

    def test_boundary_retry_count_equals_max(self):
        """retry_count == max_retries means retries ARE exhausted."""
        state = _make_state(
            relevant_documents=[],
            retry_count=2,
            max_retries=2,
        )
        with patch("app.graph.edges.settings") as mock_settings:
            mock_settings.max_retries = 2
            mock_settings.tavily_api_key = ""
            result = route_after_grading(state)
        assert result == "fallback"

    def test_boundary_retry_count_one_less_than_max(self):
        """retry_count == max_retries - 1 means one retry remaining."""
        state = _make_state(
            relevant_documents=[],
            retry_count=1,
            max_retries=2,
        )
        with patch("app.graph.edges.settings") as mock_settings:
            mock_settings.max_retries = 2
            mock_settings.tavily_api_key = ""
            result = route_after_grading(state)
        assert result == "rewrite_query"

    def test_relevant_docs_take_priority_over_retries(self):
        """Even if retries are exhausted, relevant docs → generate."""
        state = _make_state(
            relevant_documents=[_doc("answer")],
            retry_count=99,
            max_retries=2,
        )
        result = route_after_grading(state)
        assert result == "generate"


# ── route_after_web_search ─────────────────────────────────────────────────────

class TestRouteAfterWebSearch:
    def test_generate_when_web_results_found(self):
        """Branch 5: web search found results → generate."""
        state = _make_state(relevant_documents=[_doc("web result")])
        result = route_after_web_search(state)
        assert result == "generate"

    def test_fallback_when_no_web_results(self):
        """Branch 6: no web results → fallback."""
        state = _make_state(relevant_documents=[])
        result = route_after_web_search(state)
        assert result == "fallback"
