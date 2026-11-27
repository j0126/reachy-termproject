"""Shared CLI plumbing: arguments, backend construction, CSV summaries."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from ..audio.wavio import read_wav
from ..config import AppConfig
from ..evaluation.tables import append_csv_row  # noqa: F401  (re-exported for the apps)
from ..platform_timing import enable_high_resolution_timer
from ..robot.backends.base import AudioBackend
from ..robot.backends.mock import MockAudioBackend
from ..robot.connection import ReachyConnection

log = logging.getLogger(__name__)


def prepare_runtime(cfg: AppConfig) -> dict[str, Any]:
    """Process-wide setup shared by all apps; returns facts for session.json."""
    timer = enable_high_resolution_timer(1) if cfg.timing.high_resolution_timer else False
    # HTTP client libraries log every request at INFO; keep our logs readable
    # (and never risk dumping request details).
    for name in ("httpx", "httpcore", "google_genai", "google.genai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # Korean text on stdout: a Windows console is UTF-8 already, but a pipe /
    # redirect defaults to the ANSI code page (cp1252) and would crash on print.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8", errors="replace")
    return {"high_resolution_timer_1ms": timer}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", help="robot IP / hostname (default: REACHY_HOST or config)")
    parser.add_argument("--config", action="append",
                        help="TOML file overriding config/default.toml (repeatable, applied in order)")
    parser.add_argument(
        "--mock",
        metavar="WAV",
        help="no robot: replay this WAV as the microphone (speaker output is discarded)",
    )


def open_backend(cfg: AppConfig, mock_wav: str | None) -> tuple[AudioBackend, ReachyConnection | None, dict[str, Any]]:
    """Return ``(backend, connection or None, info for session.json)``."""
    hint_s = cfg.audio.output_latency_hint_ms / 1000.0
    if mock_wav:
        samples, rate = read_wav(mock_wav)
        backend = MockAudioBackend(samples, sample_rate=rate, channels=samples.shape[1], realtime=True)
        return backend, None, {"mock_wav": str(Path(mock_wav).resolve())}
    conn = ReachyConnection(cfg.robot, cfg.logging.level).open()
    try:
        return conn.audio_backend(hint_s), conn, conn.describe()
    except Exception:
        conn.close()
        raise


