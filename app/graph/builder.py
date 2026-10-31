"""
app/graph/builder.py
────────────────────
Builds and compiles the LangGraph StateGraph.

Exports:
  build_graph()  → CompiledGraph   (call once and cache)
  get_graph()    → CompiledGraph   (cached singleton)

Graph topology:
  START
    │
    ▼
  analyze_query
    │
    ▼
  retrieve
    │
    ▼
  grade_documents ──────────────────── relevant ──────────────────► generate
    │                                                                    │
    ├─ no relevant, retries left ──► rewrite_query ──► retrieve          │
    │                                                                    │
    ├─ retries exhausted + Tavily ──► web_search ──► generate            │
    │                                                                    │
    └─ retries exhausted, no Tavily ──► fallback ────────────────────────┤
                                                                         │
                                         (hallucination_check enabled?)  │
                                           YES ──► hallucination_check ──┤
                                           NO  ──────────────────────────┘
                                                                         │
                                                                        END
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.graph.edges import (
    route_after_grading,
    route_after_hallucination_check,
    route_after_web_search,
)
from app.graph.nodes import (
    analyze_query,
    fallback,
    generate,
    grade_documents,
    hallucination_check,
    retrieve,
    rewrite_query,
    web_search,
)
from app.graph.state import GraphState

logger = logging.getLogger(__name__)


def build_graph(enable_memory: bool = True) -> object:
    """Build and compile the RAG StateGraph.

    Args:
        enable_memory: If True, attach an in-memory MemorySaver checkpointer
                       to support multi-turn conversations via session_id.

    Returns:
        A compiled LangGraph graph ready for .invoke() / .stream() calls.
    """
    workflow = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────────
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("fallback", fallback)

    # Bonus nodes (always registered; routing decides whether they're reached)
    workflow.add_node("web_search", web_search)
    workflow.add_node("hallucination_check", hallucination_check)

    # ── Fixed edges ────────────────────────────────────────────────────────────
    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("rewrite_query", "retrieve")  # corrective loop

    # ── Conditional edges ──────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
            "fallback": "fallback",
        },
    )

    workflow.add_conditional_edges(
        "web_search",
        route_after_web_search,
        {
            "generate": "generate",
            "fallback": "fallback",
        },
    )

    # After generate: optionally run hallucination check, then END
    if settings.enable_hallucination_check:
        workflow.add_edge("generate", "hallucination_check")
        workflow.add_conditional_edges(
            "hallucination_check",
            route_after_hallucination_check,
            {"__end__": END},
        )
    else:
        workflow.add_edge("generate", END)

    workflow.add_edge("fallback", END)

    # ── Checkpointer for conversation memory ──────────────────────────────────
    checkpointer: Optional[MemorySaver] = MemorySaver() if enable_memory else None

    compiled = workflow.compile(checkpointer=checkpointer)
    logger.info(
        "Graph compiled  memory=%s  hallucination_check=%s",
        enable_memory,
        settings.enable_hallucination_check,
    )
    return compiled


# Module-level cached graph instance
@lru_cache(maxsize=1)
def get_graph() -> object:
    """Return the cached compiled graph (built once per process)."""
    return build_graph(enable_memory=True)


def save_mermaid_diagram() -> str:
    """Generate and save a Mermaid diagram of the compiled graph.

    Returns the Mermaid string for embedding in the README.
    """
    import os
    from pathlib import Path

    graph = get_graph()
    try:
        mermaid = graph.get_graph().draw_mermaid()
    except Exception as exc:
        logger.warning("Could not generate Mermaid diagram: %s", exc)
        mermaid = "%%  Mermaid diagram generation failed\ngraph TD\n  A[Graph] --> B[End]"

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    output_path = docs_dir / "graph.md"
    output_path.write_text(f"```mermaid\n{mermaid}\n```\n", encoding="utf-8")
    logger.info("Mermaid diagram saved to %s", output_path)
    return mermaid
