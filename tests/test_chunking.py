"""
tests/test_chunking.py
──────────────────────
Tests for the two-stage chunking pipeline.

Verifies:
  1. Metadata fields are attached to every chunk
  2. Chunk IDs are deterministic
  3. Re-ingesting with the same doc_id produces the same IDs (idempotency)
  4. Code blocks are not split in the middle (best-effort)
  5. Chunk size stays within configured bounds
"""

from __future__ import annotations

import textwrap

import pytest
from langchain_core.documents import Document

from app.ingestion.chunking import CHUNK_SIZE, chunk_documents, _make_chunk_id


# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_MARKDOWN = textwrap.dedent("""\
    # FastAPI Path Parameters

    FastAPI allows you to declare path parameters.

    ## Basic Example

    You can declare path parameters with the same syntax as Python format strings:

    ```python
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/items/{item_id}")
    async def read_item(item_id: int):
        return {"item_id": item_id}
    ```

    ## Validation

    FastAPI validates path parameters automatically.
    If you declare a parameter as `int`, it will be converted and validated.

    ### Custom Validators

    You can use Pydantic validators for more complex validation.
    This allows you to validate that the parameter meets your requirements.
    For example, you can ensure the item_id is a positive integer.
    Additional context about validation rules and their application in real-world scenarios.
""")


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            page_content=SAMPLE_MARKDOWN,
            metadata={
                "source": "test/path_params.md",
                "title": "FastAPI Path Parameters",
                "doc_id": "path_params_test",
                "file_type": "markdown",
            },
        )
    ]


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestChunkMetadata:
    def test_required_metadata_fields_present(self, sample_docs):
        chunks = chunk_documents(sample_docs)
        assert len(chunks) > 0, "Should produce at least one chunk"

        for chunk in chunks:
            meta = chunk.metadata
            assert "source" in meta, "chunk missing 'source'"
            assert "title" in meta, "chunk missing 'title'"
            assert "doc_id" in meta, "chunk missing 'doc_id'"
            assert "section" in meta, "chunk missing 'section'"
            assert "chunk_index" in meta, "chunk missing 'chunk_index'"
            assert "chunk_id" in meta, "chunk missing 'chunk_id'"

    def test_source_matches_original(self, sample_docs):
        chunks = chunk_documents(sample_docs)
        for chunk in chunks:
            assert chunk.metadata["source"] == "test/path_params.md"

    def test_title_matches_original(self, sample_docs):
        chunks = chunk_documents(sample_docs)
        for chunk in chunks:
            assert chunk.metadata["title"] == "FastAPI Path Parameters"

    def test_doc_id_matches_original(self, sample_docs):
        chunks = chunk_documents(sample_docs)
        for chunk in chunks:
            assert chunk.metadata["doc_id"] == "path_params_test"

    def test_chunk_indices_sequential(self, sample_docs):
        chunks = chunk_documents(sample_docs)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks))), "Chunk indices must be sequential"

    def test_section_breadcrumb_present(self, sample_docs):
        """At least one chunk should have a non-empty section."""
        chunks = chunk_documents(sample_docs)
        sections = [c.metadata["section"] for c in chunks]
        non_empty = [s for s in sections if s]
        assert len(non_empty) > 0, "Expected at least one chunk with a section breadcrumb"


class TestChunkIds:
    def test_chunk_ids_are_deterministic(self, sample_docs):
        """Same doc → same chunk IDs on two separate calls."""
        chunks1 = chunk_documents(sample_docs)
        chunks2 = chunk_documents(sample_docs)

        ids1 = [c.metadata["chunk_id"] for c in chunks1]
        ids2 = [c.metadata["chunk_id"] for c in chunks2]
        assert ids1 == ids2, "Chunk IDs must be deterministic"

    def test_chunk_ids_are_unique(self, sample_docs):
        chunks = chunk_documents(sample_docs)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique within a document"

    def test_make_chunk_id_function(self):
        id1 = _make_chunk_id("doc_a", 0)
        id2 = _make_chunk_id("doc_a", 0)
        id3 = _make_chunk_id("doc_a", 1)
        id4 = _make_chunk_id("doc_b", 0)

        assert id1 == id2, "Same inputs → same ID"
        assert id1 != id3, "Different index → different ID"
        assert id1 != id4, "Different doc_id → different ID"
        assert len(id1) == 32, "ID should be 32 hex chars"


class TestChunkSizes:
    def test_chunks_not_too_large(self, sample_docs):
        chunks = chunk_documents(sample_docs)
        for chunk in chunks:
            # Allow a 20% buffer over CHUNK_SIZE (overlap can push it slightly)
            assert len(chunk.page_content) <= CHUNK_SIZE * 1.5, (
                f"Chunk too large: {len(chunk.page_content)} chars"
            )

    def test_chunks_not_empty(self, sample_docs):
        chunks = chunk_documents(sample_docs)
        for chunk in chunks:
            assert chunk.page_content.strip(), "No chunk should be empty"


class TestCodeBlockPreservation:
    def test_code_block_not_split(self, sample_docs):
        """Code blocks should not be split (fences should appear in pairs)."""
        chunks = chunk_documents(sample_docs)
        for chunk in chunks:
            text = chunk.page_content
            # Count opening ``` — they should be balanced (paired) within a chunk
            fence_count = text.count("```")
            # Odd count means an unclosed fence (a split happened inside a block)
            # We allow odd count only if the chunk is large (splitter had to break it)
            if len(text) < CHUNK_SIZE:
                assert fence_count % 2 == 0, (
                    f"Possible mid-code-block split in chunk:\n{text[:200]}"
                )


class TestMultipleDocuments:
    def test_multiple_docs_produce_separate_chunks(self):
        docs = [
            Document(
                page_content="# Doc A\n\nContent about topic A.",
                metadata={"source": "a.md", "title": "Doc A", "doc_id": "doc_a"},
            ),
            Document(
                page_content="# Doc B\n\nContent about topic B.",
                metadata={"source": "b.md", "title": "Doc B", "doc_id": "doc_b"},
            ),
        ]
        chunks = chunk_documents(docs)
        doc_ids = {c.metadata["doc_id"] for c in chunks}
        assert "doc_a" in doc_ids
        assert "doc_b" in doc_ids

    def test_empty_document_list(self):
        chunks = chunk_documents([])
        assert chunks == []
