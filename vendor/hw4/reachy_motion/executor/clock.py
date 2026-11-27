"""Clocks for the executor: the real ``perf_counter_ns`` clock and a virtual clock for tests.

Everything is integer nanoseconds in the ``time.perf_counter_ns`` domain: on Windows
``time.monotonic`` / ``time.time`` tick only every 15.625 ms (Phase 1 SDK_ANALYSIS 5.4).
``time.sleep`` is high resolution on Windows with Python >= 3.11 (measured here: a 10 ms sleep
takes 10.3 ms median); timed ``Event.wait`` is not, unless ``timeBeginPeriod(1)`` is active
(``reachy_motion.util.platform_timing``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def now_ns(self) -> int: ...

    def sleep_until(self, t_ns: int) -> None: ...


class RealClock:
    """``perf_counter_ns`` + ``time.sleep``; optional busy-wait for the last ``spin_us`` microseconds."""

    def __init__(self, spin_us: int = 0) -> None:
        if spin_us < 0:
            raise ValueError("spin_us must be >= 0")
        self.spin_ns = int(spin_us) * 1000

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    def sleep_until(self, t_ns: int) -> None:
        remaining = t_ns - time.perf_counter_ns() - self.spin_ns
        if remaining > 0:
            time.sleep(remaining / 1e9)
        if self.spin_ns:
            while time.perf_counter_ns() < t_ns:
                pass


class VirtualClock:
    """Deterministic clock for tests: ``sleep_until`` jumps to the deadline plus an injected wake-up delay.

    ``wake_delay_ns(deadline_ns) -> int`` models late wake-ups (jitter); ``advance(ns)`` models time
    spent inside a call (e.g. a slow backend send).
    """

    def __init__(self, start_ns: int = 0, wake_delay_ns: Callable[[int], int] | None = None) -> None:
        self._now = int(start_ns)
        self.wake_delay_ns = wake_delay_ns
        self.sleeps = 0

    def now_ns(self) -> int:
        return self._now

    def sleep_until(self, t_ns: int) -> None:
        self.sleeps += 1
        target = max(self._now, int(t_ns))
        if self.wake_delay_ns is not None:
            target += max(0, int(self.wake_delay_ns(int(t_ns))))
        self._now = target

    def advance(self, ns: int) -> None:
        if ns < 0:
            raise ValueError("cannot go back in time")
        self._now += int(ns)
