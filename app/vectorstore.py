"""
app/vectorstore.py
Embedding factory + ChromaDB setup.

  - Local sentence-transformers embeddings (no API key).
  - Chroma persisted to disk, cosine distance. NOTE: Chroma returns *distance* (lower = better).
  - Singletons via lru_cache; tests using a temp dir must call get_vectorstore.cache_clear().
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Loading embedding model: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Connecting to ChromaDB: dir=%s collection=%s", persist_dir, settings.chroma_collection_name)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return Chroma(
        client=client,
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_collection_count() -> int:
    return get_vectorstore()._collection.count()  # pylint: disable=protected-access
