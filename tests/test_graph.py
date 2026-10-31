"""
tests/test_graph.py
End-to-end graph runs with a fake LLM and a mocked vector store (no API keys, no network).

The fake grader decides relevance from the *query text* inside the grading prompt, so the
retry test does not depend on call order or on how many chunks are retrieved:
  - first query ("path params")                       -> chunks graded irrelevant
  - rewritten query (contains "type hints")           -> chunks graded relevant
Prompts are really formatted (prompt | runnable), so prompt wiring is exercised too.
"""
from __future__ import annotations

import re
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from app.config import settings
from app.graph.builder import build_graph


class FakeLLM(BaseChatModel):
    mode: str = "retry"  # retry | irrelevant | relevant
    answer_text: str = "Path parameters use curly braces [1]."

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = str(messages[-1].content)
        out = "path parameters with type hints" if "Generate a new search query" in text else self.answer_text
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=out))])

    def with_structured_output(self, schema, **kwargs):
        mode = self.mode

        def run(prompt_value):
            text = prompt_value.to_string()
            name = schema.__name__
            if name == "QueryAnalysis":
                return schema(query_type="how-to", rewritten_query="path params")
            if name == "BatchGrades":
                rel = mode == "relevant" or (mode == "retry" and "type hints" in text)
                ids = [int(i) for i in re.findall(r"<chunk id=(\d+)>", text)]
                return schema(grades=[{"id": i, "relevant": rel, "reason": "test"} for i in ids])
            return schema(grounded=True, reason="ok")

        return RunnableLambda(run)


def _doc(i: int) -> Document:
    return Document(
        page_content=f"Path parameters in FastAPI are declared with curly braces (chunk {i}).",
        metadata={"source": f"https://example.com/{i}", "title": f"Doc {i}", "section": f"Sec {i}",
                  "doc_id": f"doc{i}", "chunk_id": f"c{i}", "chunk_index": i},
    )


def _setup(monkeypatch, llm: FakeLLM, n_docs: int = 1, memory: bool = False):
    vs = MagicMock()
    vs.similarity_search.return_value = [_doc(i) for i in range(1, n_docs + 1)]
    monkeypatch.setattr("app.graph.nodes.get_llm", lambda: llm)
    monkeypatch.setattr("app.graph.nodes.get_vectorstore", lambda: vs)
    monkeypatch.setattr(settings, "tavily_api_key", "")
    monkeypatch.setattr(settings, "max_retries", 2)
    monkeypatch.setattr(settings, "enable_hallucination_check", False)
    return build_graph(enable_memory=memory)


Q = {"question": "How do I use path parameters in FastAPI?"}


def test_retry_then_success(monkeypatch):
    graph = _setup(monkeypatch, FakeLLM(mode="retry"), n_docs=2)
    s = graph.invoke(Q)
    assert s["retry_count"] == 1
    assert s["status"] == "answered"
    assert s["trace"] == ["analyze_query", "retrieve", "grade_documents", "rewrite_query",
                          "retrieve", "grade_documents", "generate"]
    assert "type hints" in s["rewritten_query"]
    assert s["failed_queries"] == ["path params"]


def test_fallback_after_retries_exhausted(monkeypatch):
    graph = _setup(monkeypatch, FakeLLM(mode="irrelevant"))
    s = graph.invoke(Q)
    assert s["status"] == "fallback"
    assert s["retry_count"] == 2  # loop terminates at the limit
    assert s["citations"] == []
    assert s["grounded"] is None
    assert s["trace"][-1] == "fallback"


def test_happy_path_no_retry(monkeypatch):
    graph = _setup(monkeypatch, FakeLLM(mode="relevant"))
    s = graph.invoke(Q)
    assert s["retry_count"] == 0
    assert s["status"] == "answered"
    assert "rewrite_query" not in s["trace"]


def test_only_cited_sources_returned_and_renumbered(monkeypatch):
    llm = FakeLLM(mode="relevant", answer_text="Use curly braces [2]. Code like items[1] stays.")
    graph = _setup(monkeypatch, llm, n_docs=3)
    s = graph.invoke(Q)
    assert len(s["citations"]) == 1
    assert s["citations"][0]["title"] == "Doc 2"
    assert "[1]" in s["answer"] and "[2]" not in s["answer"]
    assert "items[1]" in s["answer"]


def test_chat_history_persists_and_trace_resets(monkeypatch):
    graph = _setup(monkeypatch, FakeLLM(mode="relevant"), memory=True)
    cfg = {"configurable": {"thread_id": "session-1"}}
    graph.invoke({"question": "first question"}, config=cfg)
    s2 = graph.invoke({"question": "follow up"}, config=cfg)
    assert len(s2["chat_history"]) == 4  # 2 turns x (human + AI)
    assert s2["trace"][0] == "analyze_query"
    assert s2["trace"].count("analyze_query") == 1  # trace reset each turn
