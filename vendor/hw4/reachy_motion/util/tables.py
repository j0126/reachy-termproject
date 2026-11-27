# Copied unchanged from HW3: ../reachy_hw3/reachy_s2s/evaluation/tables.py (HW3 HEAD 0d49b6b, file dated 2026-10-17).
# HW3 is read-only reference material (CLAUDE.md rule 1); edit this copy, never the original.
"""CSV result tables (one row per run / turn)."""

from __future__ import annotations

import csv
import logging
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
_lock = threading.Lock()


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    """Append one row; the header is written when the file is new.

    An existing file keeps its header: unknown keys are dropped with a warning,
    missing ones are left blank (so old CSVs stay readable after code changes).
    """
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(row.keys())
        new = not path.exists() or path.stat().st_size == 0
        if not new:
            with path.open(newline="", encoding="utf-8") as fh:
                fieldnames = next(csv.reader(fh), fieldnames)
            dropped = [k for k in row if k not in fieldnames]
            if dropped:
                log.warning("%s: columns %s not in existing header; not written", path.name, dropped)
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="", extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow(row)
