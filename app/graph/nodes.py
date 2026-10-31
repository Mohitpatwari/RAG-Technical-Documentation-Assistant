"""
app/graph/nodes.py
All LangGraph node functions: (state) -> partial state update.

Happy path:      analyze_query -> retrieve -> grade_documents -> generate
Corrective path: ... grade_documents -> rewrite_query -> retrieve -> grade_documents ...
Fallback path:   ... grade_documents -> [web_search ->] fallback
Bonus path:      ... generate -> hallucination_check
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Dict, List, Literal, Tuple

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.graph.prompts import (
    ANALYZE_QUERY_PROMPT,
    FALLBACK_MESSAGE,
    GENERATE_PROMPT,
    HALLUCINATION_CHECK_PROMPT,
    REGENERATE_STRICT_PROMPT,
    REWRITE_QUERY_PROMPT,
)
from app.graph.prompts_batch import GRADE_BATCH_PROMPT
from app.graph.state import GradeResult, GraphState
from app.llm import get_llm, invoke_llm
from app.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

QueryType = Literal["conceptual", "how-to", "troubleshooting", "api_reference"]
REFUSAL_PREFIX = "I don't have enough information"
MAX_HISTORY_MESSAGES = 20
# [n] citation marker, but not code like items[1] or arr[0][1]
CITE_RE = re.compile(r"(?<![\w\])])\[(\d+)\]")


# ── Structured output schemas ──────────────────────────────────────────────────
class QueryAnalysis(BaseModel):
    query_type: QueryType = Field(description="conceptual, how-to, troubleshooting or api_reference")
    rewritten_query: str = Field(description="Expanded, self-contained search query (10-30 words).")


class ChunkGrade(BaseModel):
    id: int = Field(description="The chunk id from the prompt.")
    relevant: bool = Field(description="True if the chunk is relevant to the query.")
    reason: str = Field(description="Short reason (<= 60 chars).")


class BatchGrades(BaseModel):
    grades: List[ChunkGrade] = Field(description="Exactly one grade per chunk id.")


class HallucinationVerdict(BaseModel):
    grounded: bool = Field(description="True if the answer is fully supported by the context.")
    reason: str = Field(description="Brief explanation (<= 80 chars).")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _text(content) -> str:
    """Chat model .content can be a str or a list of parts (Gemini); normalise to str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(str(p.get("text", "")))
        return "".join(parts)
    return str(content)


def _format_chat_history(messages: List) -> str:
    if not messages:
        return "(no prior conversation)"
    lines = []
    for msg in messages[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {_text(msg.content)[:300]}")
    return "\n".join(lines)


def _format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        header = f"[{i}] {m.get('title', 'Unknown')} - {m.get('section', '')}\nSource: {m.get('source', '')}"
        parts.append(f"{header}\n\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _citation(doc: Document) -> Dict[str, str]:
    m = doc.metadata
    return {
        "source": m.get("source", ""),
        "title": m.get("title", ""),
        "section": m.get("section", ""),
        "snippet": doc.page_content[:200].replace("\n", " "),
    }


def _resolve_citations(answer: str, docs: List[Document]) -> Tuple[str, List[Dict[str, str]]]:
    """Keep only sources the answer actually cites, renumbered [1], [2]... in order of first use."""
    if answer.startswith(REFUSAL_PREFIX):
        return answer, []
    order: List[int] = []
    for m in CITE_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= len(docs) and n not in order:
            order.append(n)
    if not order:  # model forgot markers: be transparent and list everything it was given
        return answer, [_citation(d) for d in docs]
    mapping = {old: new for new, old in enumerate(order, 1)}

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        return f"[{mapping[n]}]" if n in mapping else ""

    return CITE_RE.sub(repl, answer), [_citation(docs[n - 1]) for n in order]


def _trace(state: GraphState, name: str) -> List[str]:
    return [*state.get("trace", []), name]


def _history_with(state: GraphState, question: str, answer: str) -> list:
    hist = [*state.get("chat_history", []), HumanMessage(content=question), AIMessage(content=answer)]
    return hist[-MAX_HISTORY_MESSAGES:]


# ── Node 1: analyze_query ──────────────────────────────────────────────────────
def analyze_query(state: GraphState) -> dict:
    """Classify the query and rewrite it for retrieval; also resets all per-turn state
    (important with the MemorySaver checkpointer, which carries state across turns)."""
    logger.info("[analyze_query] question=%r", state["question"])
    chain = ANALYZE_QUERY_PROMPT | get_llm().with_structured_output(QueryAnalysis)
    result: QueryAnalysis = invoke_llm(
        chain,
        {"question": state["question"], "chat_history": _format_chat_history(state.get("chat_history", []))},
    )
    logger.info("[analyze_query] type=%s rewritten=%r", result.query_type, result.rewritten_query)
    return {
        "query_type": result.query_type,
        "rewritten_query": result.rewritten_query,
        # per-turn reset
        "retry_count": 0,
        "max_retries": settings.max_retries,
        "failed_queries": [],
        "documents": [],
        "relevant_documents": [],
        "grades": [],
        "answer": "",
        "citations": [],
        "status": "",
        "used_web_search": False,
        "grounded": None,
        "trace": ["analyze_query"],
    }


# ── Node 2: retrieve ───────────────────────────────────────────────────────────
def retrieve(state: GraphState) -> dict:
    query = state.get("rewritten_query") or state["question"]
    logger.info("[retrieve] query=%r top_k=%d", query, settings.retrieval_top_k)
    docs = get_vectorstore().similarity_search(query, k=settings.retrieval_top_k)
    logger.info("[retrieve] retrieved %d chunks", len(docs))
    return {"documents": docs, "trace": _trace(state, "retrieve")}


# ── Node 3: grade_documents ────────────────────────────────────────────────────
def grade_documents(state: GraphState) -> dict:
    """Grade every retrieved chunk (relevant / irrelevant) in ONE structured LLM call.

    One call per question instead of one per chunk keeps latency and rate-limit usage low.
    The verdict is still per chunk: the model returns one entry per chunk id. A chunk the
    model forgets to grade is treated as irrelevant (the safe choice).
    """
    query = state.get("rewritten_query") or state["question"]
    docs = state.get("documents", [])

    grades: List[GradeResult] = []
    relevant: List[Document] = []

    if docs:
        chunks_text = "\n".join(
            f"<chunk id={i}>\n{d.page_content[:1200]}\n</chunk>" for i, d in enumerate(docs)
        )
        chain = GRADE_BATCH_PROMPT | get_llm().with_structured_output(BatchGrades)
        result: BatchGrades = invoke_llm(chain, {"query": query, "chunks": chunks_text})
        by_id = {g.id: g for g in result.grades}

        for i, doc in enumerate(docs):
            g = by_id.get(i)
            is_rel = bool(g and g.relevant)
            grades.append(GradeResult(chunk_id=doc.metadata.get("chunk_id", "unknown"),
                                      relevant=is_rel, reason=g.reason if g else "not graded"))
            if is_rel:
                relevant.append(doc)

    logger.info("[grade_documents] %d/%d relevant", len(relevant), len(docs))
    return {"grades": grades, "relevant_documents": relevant, "trace": _trace(state, "grade_documents")}


# ── Node 4: rewrite_query ──────────────────────────────────────────────────────
def rewrite_query(state: GraphState) -> dict:
    failed = [*state.get("failed_queries", []), state.get("rewritten_query") or state["question"]]
    retry_count = state.get("retry_count", 0) + 1
    failed_str = "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(failed))
    resp = invoke_llm(REWRITE_QUERY_PROMPT | get_llm(),
                      {"question": state["question"], "failed_queries": failed_str})
    new_query = _text(resp.content).strip()
    logger.info("[rewrite_query] retry=%d new=%r", retry_count, new_query)
    return {
        "rewritten_query": new_query,
        "retry_count": retry_count,
        "failed_queries": failed,
        "trace": _trace(state, "rewrite_query"),
    }


# ── Node 5: generate ───────────────────────────────────────────────────────────
def generate(state: GraphState) -> dict:
    docs = state.get("relevant_documents", [])
    resp = invoke_llm(
        GENERATE_PROMPT | get_llm(),
        {"question": state["question"], "query_type": state.get("query_type", "conceptual"),
         "context": _format_context(docs)},
    )
    answer, citations = _resolve_citations(_text(resp.content).strip(), docs)
    logger.info("[generate] answer=%d chars, %d citations", len(answer), len(citations))
    return {
        "answer": answer,
        "citations": citations,
        "status": "insufficient_context" if answer.startswith(REFUSAL_PREFIX) else "answered",
        "chat_history": _history_with(state, state["question"], answer),
        "trace": _trace(state, "generate"),
    }


# ── Node 6: fallback ───────────────────────────────────────────────────────────
def fallback(state: GraphState) -> dict:
    retries = state.get("retry_count", 0)
    answer = FALLBACK_MESSAGE.format(retries=retries, question=state["question"])
    return {
        "answer": answer,
        "citations": [],
        "grounded": None,  # no answer claims were made, so there is nothing to verify
        "status": "fallback",
        "chat_history": _history_with(state, state["question"],
                                      "(No relevant information found in the documentation.)"),
        "trace": _trace(state, "fallback"),
    }


# ── Bonus: web_search ──────────────────────────────────────────────────────────
def web_search(state: GraphState) -> dict:
    if not settings.tavily_api_key:
        return {"trace": _trace(state, "web_search_skipped")}
    query = state.get("rewritten_query") or state["question"]
    try:
        from tavily import TavilyClient

        result = TavilyClient(api_key=settings.tavily_api_key).search(query=query, max_results=3)
        web_docs = []
        for r in result.get("results", []):
            if not r.get("content"):
                continue
            url = r.get("url", "")
            web_docs.append(Document(
                page_content=r["content"],
                metadata={
                    "source": url, "title": r.get("title", "Web result"), "section": "",
                    "doc_id": "web_search", "source_type": "web",
                    "chunk_id": "web_" + hashlib.sha1(url.encode()).hexdigest()[:16],
                },
            ))
        return {"relevant_documents": web_docs, "used_web_search": bool(web_docs),
                "trace": _trace(state, "web_search")}
    except Exception as exc:  # noqa: BLE001
        logger.warning("[web_search] failed: %s", exc)
        return {"trace": _trace(state, "web_search_failed")}


# ── Bonus: hallucination_check ─────────────────────────────────────────────────
def hallucination_check(state: GraphState) -> dict:
    """Verify groundedness. If it fails, regenerate once with a strict prompt and re-verify.
    grounded=True  -> the final answer passed (first try or after regeneration)
    grounded=False -> the final answer still could not be verified (treat with caution)
    """
    docs = state.get("relevant_documents", [])
    context = _format_context(docs)
    llm = get_llm()
    verify = HALLUCINATION_CHECK_PROMPT | llm.with_structured_output(HallucinationVerdict)

    verdict: HallucinationVerdict = invoke_llm(verify, {"context": context, "answer": state.get("answer", "")})
    if verdict.grounded:
        return {"grounded": True, "trace": _trace(state, "hallucination_check_passed")}

    resp = invoke_llm(REGENERATE_STRICT_PROMPT | llm,
                      {"question": state["question"], "query_type": state.get("query_type", "conceptual"),
                       "context": context})
    answer, citations = _resolve_citations(_text(resp.content).strip(), docs)
    verdict2: HallucinationVerdict = invoke_llm(verify, {"context": context, "answer": answer})
    return {
        "answer": answer,
        "citations": citations,
        "grounded": bool(verdict2.grounded),
        "chat_history": [*state.get("chat_history", [])[:-1], AIMessage(content=answer)],
        "trace": _trace(state, "hallucination_check_regenerated"),
    }
