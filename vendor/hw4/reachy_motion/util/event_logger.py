# Copied unchanged from HW3: ../reachy_hw3/reachy_s2s/evaluation/event_logger.py (HW3 HEAD 0d49b6b, file dated 2026-10-17).
# HW3 is read-only reference material (CLAUDE.md rule 1); edit this copy, never the original.
"""Thread-safe JSONL event log.

Every record is one JSON line ``{"t": <s since session start>, "kind": ..., ...}``.
``t`` comes from ``time.perf_counter_ns()``: on Windows / Python 3.12
``time.monotonic()`` and ``time.time()`` only tick every 15.6 ms, which is too
coarse for latency measurements.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

EventListener = Callable[[dict[str, Any]], None]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


class EventLogger:
    def __init__(self, path: str | Path | None, t0_ns: int | None = None, keep_recent: int = 1000) -> None:
        self.t0_ns = time.perf_counter_ns() if t0_ns is None else t0_ns
        self.path = Path(path) if path is not None else None
        self._fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8", newline="\n")  # LF on every OS (JSONL)
        self._lock = threading.Lock()
        self._recent: deque[dict[str, Any]] = deque(maxlen=keep_recent)
        self._listeners: list[EventListener] = []

    def rel(self, t_ns: int) -> float:
        """Seconds since session start for a ``perf_counter_ns`` timestamp."""
        return round((t_ns - self.t0_ns) / 1e9, 6)

    def log(self, kind: str, t_ns: int | None = None, **fields: Any) -> dict[str, Any]:
        t_ns = time.perf_counter_ns() if t_ns is None else t_ns
        record = {"t": self.rel(t_ns), "kind": kind, **fields}
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        with self._lock:
            if self._fh is not None:
                self._fh.write(line + "\n")
                self._fh.flush()
            self._recent.append(record)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(record)
            except Exception:  # a broken listener (e.g. closed GUI) must not stop logging
                log.exception("event listener failed")
        return record

    def add_listener(self, listener: EventListener) -> None:
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: EventListener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def recent(self, kind_prefix: str = "") -> list[dict[str, Any]]:
        with self._lock:
            return [r for r in self._recent if r["kind"].startswith(kind_prefix)]

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def read_events(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
