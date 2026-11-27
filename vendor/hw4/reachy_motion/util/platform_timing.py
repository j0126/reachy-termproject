# Copied unchanged from HW3: ../reachy_hw3/reachy_s2s/platform_timing.py (HW3 HEAD 0d49b6b, file dated 2026-10-17).
# HW3 is read-only reference material (CLAUDE.md rule 1); edit this copy, never the original.
"""Process-wide timer resolution (Windows).

Windows' default timer tick is 15.625 ms. ``time.sleep`` (Python >= 3.11) uses a
high-resolution waitable timer, but timed lock waits (``Condition.wait``,
``Event.wait``, ``queue.get(timeout)``) wake on the system tick. Requesting a
1 ms period with ``timeBeginPeriod`` makes those waits accurate for this
process; it is released automatically at exit. No-op elsewhere.
"""

from __future__ import annotations

import atexit
import ctypes
import sys

_enabled_period_ms: int | None = None


def enable_high_resolution_timer(period_ms: int = 1) -> bool:
    """Request ``period_ms`` timer resolution; returns ``True`` if it is active."""
    global _enabled_period_ms
    if sys.platform != "win32":
        return False
    if _enabled_period_ms is not None:
        return True
    winmm = ctypes.WinDLL("winmm")
    if winmm.timeBeginPeriod(int(period_ms)) != 0:  # TIMERR_NOERROR == 0
        return False
    _enabled_period_ms = int(period_ms)
    atexit.register(winmm.timeEndPeriod, _enabled_period_ms)
    return True


def high_resolution_timer_period_ms() -> int | None:
    return _enabled_period_ms
