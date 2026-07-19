"""
errors.py — Error Classification layer (bottom-left box in the diagram).

  Auth (401)     -> fatal, no retry
  Network / 5xx  -> retry x5, exponential backoff

Used by stt_deepgram.py whenever the Deepgram websocket drops or a REST
call fails, so the Audio Router / main app knows whether to give up on a
session or reconnect.
"""

from enum import Enum


class ErrorAction(Enum):
    FATAL_NO_RETRY = "fatal_no_retry"
    RETRY_WITH_BACKOFF = "retry_with_backoff"


class ClassifiedError(Exception):
    def __init__(self, action: ErrorAction, status_code: int | None, message: str):
        self.action = action
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{action.value}] ({status_code}) {message}")


def classify_error(status_code: int | None, message: str = "") -> ClassifiedError:
    """
    status_code: HTTP/WS close code if available, else None for pure network errors
    (DNS failure, connection reset, timeout, etc.)
    """
    if status_code == 401 or status_code == 403:
        return ClassifiedError(ErrorAction.FATAL_NO_RETRY, status_code, message or "Authentication failed")

    # Anything else (5xx, or no status code at all e.g. socket reset/timeout)
    # is treated as transient/network and is retryable.
    return ClassifiedError(ErrorAction.RETRY_WITH_BACKOFF, status_code, message or "Network/server error")


async def retry_with_backoff(coro_fn, max_attempts: int = 5, base_delay_s: float = 1.0):
    """
    Generic retry wrapper implementing "retry x5, backoff".
    coro_fn: a zero-arg async callable to retry.
    Raises the last ClassifiedError if all attempts are exhausted, or
    immediately re-raises a FATAL_NO_RETRY error without retrying.
    """
    import asyncio

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_fn()
        except ClassifiedError as e:
            if e.action == ErrorAction.FATAL_NO_RETRY:
                raise
            last_err = e
            if attempt < max_attempts:
                delay = base_delay_s * (2 ** (attempt - 1))  # exponential backoff
                await asyncio.sleep(delay)
    raise last_err