"""
app/api/query.py
POST /query: runs the compiled LangGraph workflow.

Plain `def` route on purpose: graph.invoke() is blocking, and FastAPI runs sync routes in a
threadpool so one slow query does not block the whole server.
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, status

from app.graph.builder import get_graph
from app.llm import is_llm_error
from app.schemas import QueryRequest, QueryResponse, SourceCitation
from app.store import known_query_ids, register_query  # noqa: F401  (known_query_ids re-exported)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question about the documentation",
    responses={
        422: {"description": "Validation error (blank question or > 1000 chars)"},
        503: {"description": "LLM unavailable or misconfigured"},
    },
)
def query_endpoint(body: QueryRequest) -> QueryResponse:
    """Run the self-corrective RAG graph. `trace` lists every node visited;
    `query_id` can be sent to `POST /feedback`."""
    query_id = str(uuid.uuid4())
    start = time.perf_counter()

    # NOTE: chat_history is deliberately NOT passed. With the checkpointer it is restored
    # from the session's saved state; passing [] here would wipe the memory every turn.
    initial_state = {"question": body.question, "session_id": body.session_id}
    config = {"configurable": {"thread_id": body.session_id or query_id}}

    try:
        final_state = get_graph().invoke(initial_state, config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph invocation failed")
        if is_llm_error(exc):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The language model service is unavailable or misconfigured "
                       "(check GOOGLE_API_KEY / LLM_MODEL, or retry shortly).",
            )
        raise HTTPException(status_code=500, detail="Internal error while answering the question.")

    sources = [
        SourceCitation(
            title=c.get("title", ""), source=c.get("source", ""),
            section=c.get("section", ""), snippet=c.get("snippet", ""),
        )
        for c in final_state.get("citations", [])
    ]
    answer = final_state.get("answer", "")

    try:
        register_query(query_id, body.question, answer, [s.model_dump() for s in sources], body.session_id)
    except OSError:
        logger.warning("Could not persist query log", exc_info=True)
        known_query_ids.add(query_id)

    return QueryResponse(
        query_id=query_id,
        answer=answer,
        sources=sources,
        query_type=final_state.get("query_type", "conceptual"),
        rewritten_query=final_state.get("rewritten_query", body.question),
        retries_used=final_state.get("retry_count", 0),
        grounded=final_state.get("grounded"),
        used_web_search=final_state.get("used_web_search", False),
        trace=final_state.get("trace", []),
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
