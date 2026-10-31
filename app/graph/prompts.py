"""
app/graph/prompts.py
────────────────────
All LLM prompts in one place.

Keeping prompts here (rather than inline in nodes.py) makes it easy to:
  • Review and tune them without reading node logic.
  • A/B test alternatives.
  • Localise or translate them if needed.

Each prompt is a ChatPromptTemplate or PromptTemplate ready for use with
LCEL (|) chains.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# ── 1. Query analysis ──────────────────────────────────────────────────────────
ANALYZE_QUERY_SYSTEM = """\
You are a query understanding assistant for a technical documentation RAG system.
The corpus covers FastAPI and Pydantic documentation.

Your tasks:
1. Classify the user's question into one of:
   - "conceptual"      : asking what something is or how it works
   - "how-to"          : asking how to accomplish a task
   - "troubleshooting" : asking why something doesn't work
   - "api_reference"   : asking about a specific class, function, or parameter

2. Rewrite the question into an optimal search query that:
   - Expands acronyms and adds relevant synonyms
   - Removes filler words
   - Resolves pronouns using the conversation history (e.g. "it" → the specific topic)
   - Is self-contained (no references to previous turns)
   - Is 10-30 words

If there is no conversation history, just clean and expand the question.
"""

ANALYZE_QUERY_HUMAN = """\
Conversation history (may be empty):
{chat_history}

User question: {question}
"""

ANALYZE_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", ANALYZE_QUERY_SYSTEM), ("human", ANALYZE_QUERY_HUMAN)]
)


# ── 2. Document grading ────────────────────────────────────────────────────────
GRADE_DOCUMENT_SYSTEM = """\
You are a relevance grader for a technical documentation retrieval system.

Given a search query and a retrieved document chunk, determine whether
the chunk contains information that would help answer the query.

Be strict but fair:
  - "relevant"   if the chunk directly answers or provides key context
  - "irrelevant" if the chunk is off-topic or only tangentially related

Provide a concise reason (≤ 60 characters).
"""

GRADE_DOCUMENT_HUMAN = """\
Search query: {query}

Document chunk:
---
{chunk}
---
"""

GRADE_DOCUMENT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", GRADE_DOCUMENT_SYSTEM), ("human", GRADE_DOCUMENT_HUMAN)]
)


# ── 3. Query rewriting ────────────────────────────────────────────────────────
REWRITE_QUERY_SYSTEM = """\
You are a query rewriting assistant for a technical documentation search system.

The following queries were tried but retrieved no relevant documents:
{failed_queries}

Your task: generate a NEW search query that:
  1. Approaches the topic from a different angle
  2. Uses different terminology or synonyms
  3. Is more specific OR more general (choose the opposite of what was tried)
  4. Is self-contained and 10-25 words

Do NOT repeat or slightly rephrase previous queries.
"""

REWRITE_QUERY_HUMAN = """\
Original question: {question}

Generate a new search query:
"""

REWRITE_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", REWRITE_QUERY_SYSTEM), ("human", REWRITE_QUERY_HUMAN)]
)


# ── 4. Answer generation ───────────────────────────────────────────────────────
GENERATE_SYSTEM = """\
You are a technical documentation assistant. Answer the user's question
using ONLY the provided context chunks. Do NOT use any external knowledge.

Answer style guidelines by query type:
  - conceptual      : explain clearly with definitions and examples
  - how-to          : provide numbered steps; include code examples from context
  - troubleshooting : list possible causes and solutions from the docs
  - api_reference   : describe parameters, types, defaults, and return values

Formatting rules:
  - Use Markdown (headers, code blocks, bullet lists)
  - Cite sources inline as [1], [2], etc., where the number corresponds to
    the order sources appear in the context below
  - If the context is insufficient to answer, say exactly:
    "I don't have enough information in the indexed documentation to answer this."
  - Never hallucinate or invent details not present in the context

Query type: {query_type}
"""

GENERATE_HUMAN = """\
Question: {question}

Context chunks (in priority order):
{context}

Answer:
"""

GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", GENERATE_SYSTEM), ("human", GENERATE_HUMAN)]
)


# ── 5. Fallback ────────────────────────────────────────────────────────────────
FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't find relevant information in the indexed documentation "
    "to answer your question after {retries} attempt(s).\n\n"
    "Your question was: **{question}**\n\n"
    "This could mean:\n"
    "- The topic is not covered in the current documentation corpus "
    "(FastAPI and Pydantic docs).\n"
    "- The question may need rephrasing.\n\n"
    "Please try rephrasing your question or consult the official docs at:\n"
    "- https://fastapi.tiangolo.com\n"
    "- https://docs.pydantic.dev"
)


# ── 6. Hallucination check (bonus) ─────────────────────────────────────────────
HALLUCINATION_CHECK_SYSTEM = """\
You are a fact-checking assistant. Your job is to verify that an AI-generated
answer is fully supported by the provided source documents.

Check every factual claim in the answer against the context.
Return:
  - grounded: true  → all claims are supported by the context
  - grounded: false → the answer contains claims not found in the context
  - reason: brief explanation (≤ 80 chars)
"""

HALLUCINATION_CHECK_HUMAN = """\
Source context:
{context}

Generated answer:
{answer}
"""

HALLUCINATION_CHECK_PROMPT = ChatPromptTemplate.from_messages(
    [("system", HALLUCINATION_CHECK_SYSTEM), ("human", HALLUCINATION_CHECK_HUMAN)]
)


# ── 7. Regeneration with strict grounding (bonus) ──────────────────────────────
REGENERATE_STRICT_SYSTEM = """\
You are a technical documentation assistant. A previous answer was flagged as
containing claims not supported by the source documents.

Generate a new, strictly grounded answer using ONLY the provided context.
If you cannot answer from the context alone, say so explicitly.
Do not add ANY information from your training data.

Query type: {query_type}
"""

REGENERATE_STRICT_HUMAN = """\
Question: {question}

Context:
{context}

Strictly grounded answer:
"""

REGENERATE_STRICT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", REGENERATE_STRICT_SYSTEM), ("human", REGENERATE_STRICT_HUMAN)]
)
