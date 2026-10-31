"""Shared fixtures: every test gets isolated storage so nothing touches real data/ files."""
import pytest

from app import store
from app.config import settings


@pytest.fixture(autouse=True)
def isolate_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "feedback_file", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setattr(settings, "queries_file", str(tmp_path / "queries.jsonl"))
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    store.reset()
    yield
    store.reset()
