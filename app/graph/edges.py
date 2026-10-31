"""
app/graph/edges.py
──────────────────
Conditional routing functions for the LangGraph workflow.

Design principles:
  • Each function is pure (no side effects, no LLM calls).
  • They receive only the state and return a string node name.
  • They are unit-testable independently of the graph.
  • They are the ONLY place where routing decisions are made.

Routing logic after grade_documents:
  ┌──────────────────────────────────────────────────────────┐
  │ relevant_documents not empty?                            │
  │   YES → "generate"                                       │
  │   NO  →                                                  │
  │         retry_count < max_retries?                       │
  │           YES → "rewrite_query"                          │
  │           NO  →                                          │
  │                 TAVILY_API_KEY set?                       │
  │                   YES → "web_search"                     │
  │                   NO  → "fallback"                       │
  └──────────────────────────────────────────────────────────┘

Routing logic after hallucination_check:
  Hallucination check always terminates the graph (END) because
  regeneration (if needed) happens inside the node itself.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.graph.state import GraphState

logger = logging.getLogger(__name__)


def route_after_grading(state: GraphState) -> str:
    """Decide what to do after grade_documents.

    Returns one of: "generate", "rewrite_query", "web_search", "fallback"
    """
    relevant = state.get("relevant_documents", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.max_retries)

    if relevant:
        logger.debug(
            "[route_after_grading] %d relevant docs → generate", len(relevant)
        )
        return "generate"

    if retry_count < max_retries:
        logger.debug(
            "[route_after_grading] no relevant docs, retry %d/%d → rewrite_query",
            retry_count + 1,
            max_retries,
        )
        return "rewrite_query"

    # Retries exhausted — try web search if enabled
    if settings.tavily_api_key:
        logger.debug("[route_after_grading] retries exhausted → web_search")
        return "web_search"

    logger.debug("[route_after_grading] retries exhausted → fallback")
    return "fallback"


def route_after_web_search(state: GraphState) -> str:
    """Decide what to do after web_search.

    If web search found results → generate; otherwise → fallback.
    """
    relevant = state.get("relevant_documents", [])
    if relevant:
        logger.debug("[route_after_web_search] web results found → generate")
        return "generate"
    logger.debug("[route_after_web_search] no web results → fallback")
    return "fallback"


def route_after_hallucination_check(state: GraphState) -> str:
    """Always terminate after hallucination_check (regeneration is inline)."""
    return "__end__"
