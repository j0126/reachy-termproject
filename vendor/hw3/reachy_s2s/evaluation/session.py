"""One experiment run = one directory under ``results/sessions/``.

Layout::

    results/sessions/<YYYYmmdd_HHMMSS>_<tag>/
        session.json     metadata (config snapshot, robot / backend info, ...)
        events.jsonl     timestamped events (EventLogger)
        app.log          Python logging output
        audio/           recordings (WAV + JSON sidecars)
"""

from __future__ import annotations

import json
import logging
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import __version__
from .event_logger import EventLogger, _json_default

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "run"


class Session:
    def __init__(self, results_dir: str | Path, tag: str, capture_logging: bool = True) -> None:
        self.t0_ns = time.perf_counter_ns()
        self.started_at = datetime.now().astimezone()
        self.id = f"{self.started_at:%Y%m%d_%H%M%S}_{_slug(tag)}"
        self.dir = Path(results_dir) / "sessions" / self.id
        suffix = 1
        while self.dir.exists():  # two runs within the same second
            suffix += 1
            self.dir = Path(results_dir) / "sessions" / f"{self.id}-{suffix}"
        self.id = self.dir.name
        self.audio_dir = self.dir / "audio"
        self.audio_dir.mkdir(parents=True)
        self.events = EventLogger(self.dir / "events.jsonl", self.t0_ns)
        self._log_handler: logging.Handler | None = None
        if capture_logging:
            self._log_handler = logging.FileHandler(self.dir / "app.log", encoding="utf-8")
            self._log_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
            logging.getLogger().addHandler(self._log_handler)
        self._meta: dict[str, Any] = {
            "session_id": self.id,
            "started_at": self.started_at.isoformat(timespec="milliseconds"),
            "clock": "time.perf_counter_ns; event 't' = seconds since session start",
            "reachy_s2s_version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        self.write_metadata()
        self.events.log("session.started", session_id=self.id)

    def t(self, t_ns: int | None = None) -> float:
        return self.events.rel(time.perf_counter_ns() if t_ns is None else t_ns)

    def path(self, name: str) -> Path:
        return self.dir / name

    def write_metadata(self, **updates: Any) -> None:
        self._meta.update(updates)
        with (self.dir / "session.json").open("w", encoding="utf-8") as fh:
            json.dump(self._meta, fh, ensure_ascii=False, indent=2, default=_json_default)

    def close(self) -> None:
        self.events.log("session.closed")
        self.write_metadata(closed_at=datetime.now().astimezone().isoformat(timespec="milliseconds"))
        self.events.close()
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None


def setup_console_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, "%H:%M:%S"))
        root.addHandler(handler)
