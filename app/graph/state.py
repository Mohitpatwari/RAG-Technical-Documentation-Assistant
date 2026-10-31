"""
app/graph/state.py
──────────────────
GraphState TypedDict — the single shared data structure that flows through
every node in the LangGraph workflow.

Design philosophy:
  • Every field is documented so a reviewer can understand the graph
    by reading this file alone.
  • Fields are never deleted — nodes only add or overwrite values.
  • `question` is the canonical user input and is never overwritten.
  • `rewritten_query` starts as a copy of `question` and may be updated
    by `rewrite_query` nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class GradeResult(TypedDict):
    """Per-chunk grading result produced by the grade_documents node."""

    chunk_id: str
    """Deterministic ID of the chunk being graded."""

    relevant: bool
    """True if the chunk is relevant to the (rewritten) query."""

    reason: str
    """Short (≤ 60 chars) explanation for the grade."""


class GraphState(TypedDict, total=False):
    """
    Shared state object threaded through every LangGraph node.

    `total=False` means all keys are optional at creation time;
    nodes add keys progressively.  Required fields are those that must
    be present before the graph is invoked (see builder.py).
    """

    # ── Input (set before the graph runs) ────────────────────────────────────
    question: str
    """Original user question.  NEVER overwritten by any node."""

    session_id: Optional[str]
    """Optional session identifier for multi-turn conversation memory."""

    # ── Query analysis ────────────────────────────────────────────────────────
    query_type: Literal["conceptual", "how-to", "troubleshooting", "api_reference"]
    """Classified intent of the question (set by analyze_query)."""

    rewritten_query: str
    """
    Current query used for retrieval.  Starts as a copy of `question`,
    then updated by `rewrite_query` nodes to escape retrieval dead-ends.
    """

    # ── Retrieval ─────────────────────────────────────────────────────────────
    documents: List[Document]
    """Raw chunks returned by the vector store (before grading)."""

    # ── Grading ───────────────────────────────────────────────────────────────
    grades: List[GradeResult]
    """Per-chunk grading results from grade_documents."""

    relevant_documents: List[Document]
    """Subset of `documents` that passed the relevance grader."""

    # ── Retry control ─────────────────────────────────────────────────────────
    retry_count: int
    """Number of query-rewrite / retrieval retries performed so far."""

    max_retries: int
    """Maximum allowed retries before falling back (default: from settings)."""

    failed_queries: List[str]
    """
    All queries that failed to retrieve relevant docs.
    Passed to the rewrite node so it can generate a genuinely different
    query rather than rephrasing the same one.
    """

    # ── Generation ────────────────────────────────────────────────────────────
    answer: str
    """Final generated answer (Markdown)."""

    citations: List[Dict[str, str]]
    """
    List of citation dicts, each with keys:
        source, title, section, snippet
    Inline citation markers [1], [2] in `answer` map to this list by index.
    """

    # ── Bonus: hallucination check ────────────────────────────────────────────
    grounded: Optional[bool]
    """
    True  → answer is supported by context (passed hallucination check).
    False → answer contains claims not found in context.
    None  → hallucination check was not run.
    """

    # ── Bonus: web search ─────────────────────────────────────────────────────
    used_web_search: bool
    """True if the Tavily web-search fallback was used."""

    # ── Bonus: conversation memory ────────────────────────────────────────────
    chat_history: List[BaseMessage]
    """
    Ordered list of prior messages in this session.
    Used by analyze_query to resolve follow-up questions
    (e.g. "what about its default value?" → resolves "field default value").
    """

    # ── Observability ─────────────────────────────────────────────────────────
    trace: List[str]
    """
    Ordered list of node names visited during this graph run.
    Returned in the API response for transparency and debugging.
    """

    status: str
    """Human-readable summary of the final outcome (e.g. 'answered', 'fallback')."""
