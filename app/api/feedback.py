"""
app/api/feedback.py
POST /feedback: stores thumbs up/down ratings in a JSONL file.
The query_id must come from a previous /query (persisted in app.store, so it survives restarts).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.schemas import FeedbackRequest, FeedbackResponse
from app.store import is_known

logger = logging.getLogger(__name__)
router = APIRouter()
_write_lock = Lock()


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback for a query response",
    responses={404: {"description": "Unknown query_id"}},
)
def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    if not is_known(body.query_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown query_id '{body.query_id}'. Use the query_id returned by POST /query.",
        )

    record = {
        "query_id": body.query_id,
        "rating": body.rating,
        "comment": body.comment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(settings.feedback_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info("Feedback stored: query_id=%s rating=%s", body.query_id, body.rating)
    return FeedbackResponse(status="accepted", query_id=body.query_id)
