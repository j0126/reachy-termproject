"""Fault injection around a Mock or MuJoCo backend: the stop / hold experiments of Phase 4 (never the robot).

``FaultInjector(backend)`` forwards every call and, on request, fails, delays or freezes it - the same knobs as
``MockBackend``: ``fail_send`` (``set_target`` raises, nothing forwarded), ``send_delay_s`` (time spent before
forwarding), ``fail_present`` (``present()`` raises), ``freeze_present`` (``present()`` repeats its last answer),
``daemon_error`` (``health()`` reports the daemon's error state). A backend with ``capabilities.real_hardware`` is
refused: faults are never injected on the robot.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from reachy_motion.backends.base import BackendError, PresentState

__all__ = ["FaultInjector"]


class FaultInjector:
    def __init__(self, inner: Any) -> None:
        caps = getattr(inner, "capabilities", None)
        if caps is None or getattr(caps, "real_hardware", True):
            raise ValueError("fault injection is for the Mock and MuJoCo backends only, never the robot")
        self.inner = inner
        self.name = f"faults({inner.name})"
        self.data_kind = inner.data_kind
        self.capabilities = caps
        self.fail_send = False
        self.send_delay_s = 0.0
        self.fail_present = False
        self.freeze_present = False
        self.daemon_error: str | None = None
        self._last: PresentState | None = None

    def connect(self) -> None:
        self.inner.connect()

    def close(self) -> None:
        self.inner.close()

    def set_target(self, head: np.ndarray, antennas: Sequence[float], body_yaw: float) -> None:
        if self.send_delay_s > 0:
            time.sleep(self.send_delay_s)
        if self.fail_send:
            raise BackendError("injected send failure (fail_send=True): the command was not forwarded")
        self.inner.set_target(head, antennas, body_yaw)

    def goto_target(self, head: np.ndarray, antennas: Sequence[float], body_yaw: float, duration: float,
                    method: str) -> None:
        self.inner.goto_target(head, antennas, body_yaw, duration, method)

    def present(self) -> PresentState:
        if self.fail_present:
            raise BackendError("injected present() failure (fail_present=True)")
        if self.freeze_present and self._last is not None:
            return self._last
        p = self.inner.present()
        self._last = p
        return p

    def health(self) -> dict[str, Any] | None:
        if self.daemon_error:
            return {"state": "error", "error": self.daemon_error, "motor_mode": None, "status_age_s": None}
        h = getattr(self.inner, "health", None)
        return h() if callable(h) else None
