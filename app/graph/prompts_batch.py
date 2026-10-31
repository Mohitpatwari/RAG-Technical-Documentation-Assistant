"""
Prompt for batched chunk grading (all retrieved chunks graded in ONE LLM call).
Kept next to prompts.py. One call instead of one-per-chunk keeps a question at ~3 LLM calls,
which matters on rate-limited free tiers (e.g. Gemini free: 5 requests/minute).
Grading is still per chunk: the model must return one verdict per chunk id.
"""

from langchain_core.prompts import ChatPromptTemplate

GRADE_BATCH_SYSTEM = """\
You are a relevance grader for a technical documentation retrieval system.

You get a search query and several numbered document chunks. For EACH chunk decide whether it
contains information that would help answer the query.
  - relevant   : directly answers the query or provides key context
  - irrelevant : off-topic or only tangentially related

Be strict but fair. Return exactly one entry per chunk id, each with a short reason (<= 60 characters).
"""

GRADE_BATCH_HUMAN = """\
Search query: {query}

Chunks:
{chunks}
"""

GRADE_BATCH_PROMPT = ChatPromptTemplate.from_messages(
    [("system", GRADE_BATCH_SYSTEM), ("human", GRADE_BATCH_HUMAN)]
)
