"""Shared engine plumbing: error classification and an explicit, logged retry policy.

Cloud SDK retries are disabled (see engines/gemini_client.py) so that every
attempt, every free-tier limit (HTTP 429) and every timeout is visible in the
event log. Our policy: retry only transient failures (timeout, network, 5xx,
408), never 429 or other 4xx, and never switch to another model.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")

RETRYABLE_KINDS = frozenset({"timeout", "network", "server"})


class EngineError(Exception):
    """A stage failed; ``kind`` drives recovery and reporting."""

    def __init__(self, stage: str, kind: str, message: str, status_code: int | None = None,
                 attempts: int = 1) -> None:
        super().__init__(message)
        self.stage = stage
        self.kind = kind  # timeout | network | server | rate_limit | client | empty | config | other
        self.status_code = status_code
        self.attempts = attempts

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_KINDS

    def to_dict(self) -> dict[str, Any]:
        # "error_kind", not "kind": these dicts are splatted into EventLogger.log(kind, **fields)
        return {"stage": self.stage, "error_kind": self.kind, "status_code": self.status_code,
                "message": str(self)[:500], "attempts": self.attempts}


def classify_exception(stage: str, exc: BaseException) -> EngineError:
    """Map SDK / HTTP exceptions to an :class:`EngineError` without leaking secrets."""
    if isinstance(exc, EngineError):
        return exc
    import httpx  # dependency of google-genai

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        text = str(getattr(exc, "message", "") or exc)
        body = str(getattr(exc, "body", "") or "")[:300]
        message = f"HTTP {status}: {text}" + (f" {body}" if body and body[:40] not in text else "")
        if status == 429:
            return EngineError(stage, "rate_limit", message, status)
        if status == 408 or status >= 500:
            return EngineError(stage, "server", message, status)
        return EngineError(stage, "client", message, status)
    # google-genai's Interactions client raises OpenAI-style APITimeoutError /
    # APIConnectionError (google.genai._gaos.lib.compat_errors).
    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & {"APITimeoutError", "TimeoutException", "TimeoutError"}:
        return EngineError(stage, "timeout", f"{type(exc).__name__}: {exc}")
    if names & {"APIConnectionError", "NetworkError", "RemoteProtocolError", "ConnectionError"}:
        return EngineError(stage, "network", f"{type(exc).__name__}: {exc}")
    if isinstance(exc, httpx.TimeoutException):
        return EngineError(stage, "timeout", f"{type(exc).__name__}: {exc}")
    if type(exc).__name__ == "NoResponseError":
        return EngineError(stage, "network", f"NoResponseError: {exc}")
    return EngineError(stage, "other", f"{type(exc).__name__}: {exc}")


@dataclass
class Attempt:
    number: int
    t_start_ns: int
    t_end_ns: int
    error: dict[str, Any] | None = None


def call_with_retries(stage: str, fn: Callable[[], T], max_retries: int,
                      on_attempt: Callable[[Attempt], None] | None = None,
                      backoff_s: float = 0.5, retry_timeouts: bool = True) -> tuple[T, list[Attempt]]:
    """Run ``fn``; retry transient failures up to ``max_retries`` times.

    ``retry_timeouts=False`` gives up after a timeout (it already cost the full
    timeout; in a live conversation a second wait is worse than failing fast).
    Returns ``(result, attempts)``; raises :class:`EngineError` with
    ``attempts`` set when giving up.
    """
    attempts: list[Attempt] = []
    for number in range(1, max_retries + 2):
        t0 = time.perf_counter_ns()
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - classified below
            err = classify_exception(stage, exc)
            attempt = Attempt(number, t0, time.perf_counter_ns(), err.to_dict())
            attempts.append(attempt)
            if on_attempt is not None:
                on_attempt(attempt)
            give_up = not err.retryable or number > max_retries or (err.kind == "timeout" and not retry_timeouts)
            if give_up:
                err.attempts = number
                raise err from exc
            time.sleep(backoff_s * number)
            continue
        attempt = Attempt(number, t0, time.perf_counter_ns())
        attempts.append(attempt)
        if on_attempt is not None:
            on_attempt(attempt)
        return result, attempts
    raise AssertionError("unreachable")
