"""
app/store.py
Tiny persistent query log (JSONL). /query appends a record after a successful answer;
/feedback checks query_ids against it, so feedback keeps working after a server restart.
`known_query_ids` is the in-memory index of the file.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from app.config import settings

logger = logging.getLogger(__name__)

known_query_ids: set[str] = set()
_lock = Lock()
_loaded = False


def reset() -> None:
    """Clear the in-memory index (used by tests)."""
    global _loaded
    with _lock:
        known_query_ids.clear()
        _loaded = False


def _path() -> Path:
    p = Path(settings.queries_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_once() -> None:
    global _loaded
    if _loaded:
        return
    p = Path(settings.queries_file)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                known_query_ids.add(json.loads(line)["query_id"])
            except (ValueError, KeyError):
                continue
    _loaded = True


def register_query(query_id: str, question: str, answer: str, sources: list, session_id: str | None) -> None:
    record = {
        "query_id": query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "sources": sources,
    }
    with _lock:
        with _path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        known_query_ids.add(query_id)


def is_known(query_id: str) -> bool:
    with _lock:
        if query_id in known_query_ids:
            return True
        _load_once()
        return query_id in known_query_ids
