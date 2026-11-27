"""Backend protocol: the only way the executor reaches a robot, a simulator or the mock.

Rules every backend follows (Phase 1 findings, docs/ARCHITECTURE.md sections 5-8):

* ``set_target`` / ``goto_target`` always carry head, antennas AND body yaw. The SDK's
  ``goto_target(body_yaw=0.0)`` default would turn the body back to 0 when omitted
  (reachy_mini/reachy_mini.py:680), so the executor never relies on defaults.
* A backend that can drive real motors must refuse to exist without an explicit approval
  (``reachy_motion.safety.approval``); in Phase 2 only the mock can run.
* ``data_kind`` labels everything a backend produces: "mock" (synthetic model), "simulated"
  (MuJoCo), "measured" (robot) - CLAUDE.md rule 6.
* Calls come from ONE thread (the executor's); the SDK client is not used concurrently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


class BackendError(RuntimeError):
    """A backend call failed (communication, refused command, not connected)."""


@dataclass(frozen=True)
class Capabilities:
    stream: bool = True  # set_target streaming
    goto: bool = True  # blocking goto_target
    present: bool = True  # measured (or modelled) present state
    torque_control: bool = False  # motor modes / torque on-off (robot only; MuJoCo is always "enabled")
    real_hardware: bool = False  # True only for the robot backend


@dataclass(frozen=True)
class PresentState:
    """Latest present state as the backend knows it (robot: daemon push cache, mock: model)."""

    t_ns: int  # time the state refers to (backend clock, perf_counter_ns domain)
    head: np.ndarray  # 4x4 head pose (m)
    antennas: tuple[float, float]  # rad [right, left]
    body_yaw: float  # rad
    joints: np.ndarray | None = None  # (7,) [body_yaw, stewart_1..6] rad, if known
    data_kind: str = "mock"


@dataclass(frozen=True)
class CommandRecord:
    """One command exactly as handed to the backend (what was sent, not what was planned)."""

    t_ns: int  # when the backend call started (perf_counter_ns domain or virtual clock)
    kind: str  # "set_target" | "goto_target"
    head: np.ndarray  # 4x4
    antennas: tuple[float, float]
    body_yaw: float
    duration: float | None = None  # goto only
    method: str | None = None  # goto only: Easing value ("linear", "minjerk", "ease_in_out", "cartoon")


@runtime_checkable
class Backend(Protocol):
    name: str
    data_kind: str  # "mock" | "simulated" | "measured"
    capabilities: Capabilities

    def connect(self) -> None: ...

    def close(self) -> None: ...

    def set_target(self, head: np.ndarray, antennas: Sequence[float], body_yaw: float) -> None: ...

    def goto_target(self, head: np.ndarray, antennas: Sequence[float], body_yaw: float, duration: float,
                    method: str) -> None: ...

    def present(self) -> PresentState: ...


def check_command(head: np.ndarray, antennas: Sequence[float], body_yaw: float) -> tuple[np.ndarray, tuple[float, float], float]:
    """Validate and normalise one command (shape, finiteness, explicit body yaw). Raises BackendError."""
    h = np.asarray(head, dtype=np.float64)
    if h.shape != (4, 4):
        raise BackendError(f"head must be 4x4, got {h.shape}")
    if len(antennas) != 2:
        raise BackendError("antennas must be (right, left)")
    a = (float(antennas[0]), float(antennas[1]))
    if body_yaw is None:
        raise BackendError("body_yaw must be explicit (SDK goto_target defaults it to 0.0)")
    b = float(body_yaw)
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(a)) and np.isfinite(b)):
        raise BackendError("non-finite command")
    return h, a, b
