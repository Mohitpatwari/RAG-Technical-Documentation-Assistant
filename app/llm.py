"""
app/llm.py
LLM factory + retry helpers.

Retry policy: only *transient* failures (rate limit 429, timeouts, 5xx, connection errors)
are retried with exponential back-off. Auth errors, bad model names and validation errors
fail fast. The model's own built-in retries are set to 1 so retries do not nest.

Nodes should call `invoke_llm` / `batch_llm` instead of `.invoke()` / `.batch()`.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, List

from langchain_core.language_models import BaseChatModel
from tenacity import retry, retry_if_exception, stop_after_attempt

from app.config import settings

logger = logging.getLogger(__name__)


class LLMConfigError(RuntimeError):
    """Raised when the LLM is not configured (e.g. missing API key)."""


def _transient_types() -> tuple:
    types: list = [TimeoutError, ConnectionError]
    try:
        import httpx

        types += [httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError]
    except ImportError:  # pragma: no cover
        pass
    try:
        from google.api_core import exceptions as g

        types += [
            g.ResourceExhausted, g.TooManyRequests, g.ServiceUnavailable,
            g.DeadlineExceeded, g.InternalServerError, g.GatewayTimeout,
        ]
    except ImportError:  # pragma: no cover
        pass
    return tuple(types)


_TRANSIENT = _transient_types()
_TRANSIENT_HINTS = ("429", "rate limit", "quota", "resource exhausted", "503", "504", "timed out", "timeout")


def _chain(exc: BaseException):
    seen = 0
    while exc is not None and seen < 5:
        yield exc
        exc = exc.__cause__ or exc.__context__
        seen += 1


def is_transient(exc: BaseException) -> bool:
    for e in _chain(exc):
        if isinstance(e, _TRANSIENT):
            return True
    msg = str(exc).lower()
    return any(h in msg for h in _TRANSIENT_HINTS)


def is_llm_error(exc: BaseException) -> bool:
    """True if the failure came from the LLM provider/config (used to return 503 instead of 500)."""
    if isinstance(exc, LLMConfigError) or is_transient(exc):
        return True
    for e in _chain(exc):
        mod = type(e).__module__ or ""
        if mod.startswith(("google", "langchain_google_genai", "openai", "anthropic", "groq")):
            return True
    return False


def _wait_seconds(retry_state) -> float:
    """Honour the provider's 'retry in 18.9s' hint on 429s; otherwise exponential back-off."""
    exc = retry_state.outcome.exception()
    m = re.search(r"retry in ([\d.]+)\s*s", str(exc), re.IGNORECASE)
    if m:
        return min(float(m.group(1)) + 1.0, 30.0)
    return float(min(2 ** retry_state.attempt_number, 8))


llm_retry = retry(
    reraise=True,
    stop=stop_after_attempt(settings.llm_max_retries),
    wait=_wait_seconds,
    retry=retry_if_exception(is_transient),
    before_sleep=lambda rs: logger.warning(
        "Transient LLM error (attempt %d): %s - retrying", rs.attempt_number, rs.outcome.exception()
    ),
)


@llm_retry
def invoke_llm(runnable: Any, inputs: Any) -> Any:
    return runnable.invoke(inputs)


@llm_retry
def batch_llm(runnable: Any, inputs: List[Any], **kwargs: Any) -> List[Any]:
    return runnable.batch(inputs, **kwargs)


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return the configured chat model (cached). Swap providers by editing this function."""
    name = settings.llm_model.lower()
    temp = settings.llm_temperature
    logger.info("Initialising LLM: model=%s temperature=%s", settings.llm_model, temp)

    if "gemini" in name:
        if not settings.google_api_key:
            raise LLMConfigError("GOOGLE_API_KEY is not set. Add it to your .env file.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=temp,
            google_api_key=settings.google_api_key,
            timeout=settings.llm_timeout,
            max_retries=1,
        )
    if "gpt" in name:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.llm_model, temperature=temp, timeout=settings.llm_timeout, max_retries=1)
    if "claude" in name:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.llm_model, temperature=temp, timeout=settings.llm_timeout, max_retries=1)
    if "llama" in name or "mixtral" in name:
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.llm_model, temperature=temp, timeout=settings.llm_timeout, max_retries=1)

    raise LLMConfigError(f"Unsupported model '{settings.llm_model}'. Add a branch in app/llm.py.")
