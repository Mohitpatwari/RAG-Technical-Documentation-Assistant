"""
ui/gradio_app.py
────────────────
Bonus: Minimal Gradio chat UI that wraps the FastAPI backend.

Features:
  - Chat interface with conversation history
  - Displays answer with inline citations as an expandable accordion
  - Thumbs up / thumbs down feedback buttons wired to POST /feedback
  - Shows the graph trace for transparency

Run:
    python ui/gradio_app.py

Or, with the API server running on localhost:8000, just open
http://localhost:7860 after starting this script.
"""

from __future__ import annotations

import uuid
from typing import List, Tuple

import gradio as gr
import httpx

API_BASE = "http://localhost:8000"
SESSION_ID = str(uuid.uuid4())

# Track the last query_id for feedback
_last_query_id: str = ""


def chat(
    message: str,
    history: List[Tuple[str, str]],
) -> Tuple[str, List[Tuple[str, str]], str, str]:
    """Send a message to the API and return the response."""
    global _last_query_id

    if not message.strip():
        return "", history, "", ""

    try:
        resp = httpx.post(
            f"{API_BASE}/query",
            json={"question": message, "session_id": SESSION_ID},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        error_msg = f"❌ API error: {exc}"
        history.append((message, error_msg))
        return "", history, "", ""

    _last_query_id = data.get("query_id", "")
    answer = data.get("answer", "No answer returned")
    sources = data.get("sources", [])
    trace = data.get("trace", [])
    latency = data.get("latency_ms", 0)
    retries = data.get("retries_used", 0)
    query_type = data.get("query_type", "")

    # Format sources
    sources_md = ""
    if sources:
        sources_md = "\n\n---\n**📚 Sources:**\n"
        for i, s in enumerate(sources, 1):
            sources_md += (
                f"\n[{i}] **{s['title']}**"
                + (f" — {s['section']}" if s.get("section") else "")
                + f"\n> {s['snippet'][:150]}…\n"
            )

    full_answer = answer + sources_md

    # Metadata line
    meta = (
        f"⏱ {latency:.0f}ms | 🏷 {query_type} | 🔄 {retries} retries"
        + (f" | 🌐 Web search used" if data.get("used_web_search") else "")
    )

    trace_str = " → ".join(trace)

    history.append((message, full_answer))
    return "", history, meta, trace_str


def submit_feedback(rating: str) -> str:
    """Send feedback to the API."""
    global _last_query_id
    if not _last_query_id:
        return "No query to rate yet."

    try:
        resp = httpx.post(
            f"{API_BASE}/feedback",
            json={"query_id": _last_query_id, "rating": rating},
            timeout=10.0,
        )
        resp.raise_for_status()
        return f"✅ Feedback '{rating}' submitted for query {_last_query_id[:8]}…"
    except Exception as exc:
        return f"❌ Feedback error: {exc}"


# ── Gradio interface ───────────────────────────────────────────────────────────

with gr.Blocks(
    title="RAG Docs Assistant",
    theme=gr.themes.Soft(primary_hue="indigo"),
    css="""
    .feedback-row { gap: 8px; }
    .meta-box textarea { font-size: 0.8em; color: #666; }
    """,
) as demo:
    gr.Markdown(
        """
        # 📚 RAG Technical Documentation Assistant
        Ask questions about **FastAPI** and **Pydantic** documentation.
        The assistant uses a self-corrective LangGraph workflow to retrieve and verify answers.
        """
    )

    chatbot = gr.Chatbot(label="Conversation", height=450, show_copy_button=True)
    msg = gr.Textbox(
        placeholder="Ask a question about FastAPI or Pydantic…",
        label="Your question",
        lines=2,
    )

    with gr.Row():
        submit_btn = gr.Button("Ask", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")

    with gr.Row(elem_classes="feedback-row"):
        gr.Markdown("**Rate this answer:**")
        thumbs_up = gr.Button("👍 Helpful", size="sm")
        thumbs_down = gr.Button("👎 Not helpful", size="sm")

    feedback_status = gr.Textbox(label="Feedback status", interactive=False, lines=1)

    with gr.Accordion("📊 Response metadata", open=False):
        meta_box = gr.Textbox(label="Stats", interactive=False, lines=1, elem_classes="meta-box")
        trace_box = gr.Textbox(label="Graph trace", interactive=False, lines=1)

    # Wire up events
    submit_btn.click(
        chat,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, meta_box, trace_box],
    )
    msg.submit(
        chat,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, meta_box, trace_box],
    )
    clear_btn.click(lambda: ([], "", ""), outputs=[chatbot, meta_box, trace_box])
    thumbs_up.click(lambda: submit_feedback("up"), outputs=feedback_status)
    thumbs_down.click(lambda: submit_feedback("down"), outputs=feedback_status)

    gr.Examples(
        examples=[
            "How do I declare path parameters in FastAPI?",
            "What is the difference between Pydantic v1 and v2 validators?",
            "How do background tasks work in FastAPI?",
            "What is dependency injection in FastAPI?",
        ],
        inputs=msg,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
