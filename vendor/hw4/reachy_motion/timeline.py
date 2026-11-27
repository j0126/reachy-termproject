"""``Timeline``: the planner's output and the single source of truth for what an executor sends.

Sample ``k`` is meant for control tick ``k`` at ``t[k] = k / rate_hz`` after the motion starts. The
grid runs to ``K = ceil(duration * rate_hz)`` (plus one sample when ``duration`` falls exactly on the
grid it is that sample); samples at or after ``duration`` hold the end state, so the last sample is
exactly the end state - unlike a daemon goto, which stops before t = D (Phase 1 SDK_ANALYSIS 3.5).
Units: s, 4x4 head matrices in metres, rad.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from reachy_motion.safety.policy import SafetyReport

if TYPE_CHECKING:  # pragma: no cover
    from reachy_motion.library.spec import MotionSpec


@dataclass(frozen=True)
class Timeline:
    name: str  # motion name, or e.g. "transition"
    variant: str  # "expressive" | "neutral" | "" for ad-hoc timelines
    rate_hz: float
    duration: float  # s, motion length (the last sample is at t[-1] >= duration)
    t: np.ndarray  # (N,) s, t[k] = k / rate_hz
    head: np.ndarray  # (N, 4, 4)
    antennas: np.ndarray  # (N, 2) rad [right, left]
    body_yaw: np.ndarray  # (N,) rad
    head_key: np.ndarray  # (N,) int, index of the head key being travelled to or held (-1: track empty)
    antenna_key: np.ndarray  # (N,) int
    body_key: np.ndarray  # (N,) int
    spec_digest: str = ""  # MotionSpec.digest() of the planned spec ("" for ad-hoc timelines)

    def __len__(self) -> int:
        return int(self.t.shape[0])

    def validate(self) -> list[str]:
        n = len(self)
        problems = []
        if n < 1:
            problems.append("empty timeline")
        if self.rate_hz <= 0:
            problems.append("rate_hz <= 0")
        shapes = {"head": (n, 4, 4), "antennas": (n, 2), "body_yaw": (n,), "head_key": (n,), "antenna_key": (n,),
                  "body_key": (n,)}
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                problems.append(f"{name} shape {getattr(self, name).shape} != {shape}")
        if n > 1 and not np.allclose(np.diff(self.t), 1.0 / self.rate_hz, rtol=0, atol=1e-12):
            problems.append("t is not a uniform 1/rate_hz grid")
        if n and abs(self.t[0]) > 0:
            problems.append("t[0] != 0")
        if n and self.t[-1] + 1e-12 < self.duration:
            problems.append("last sample before the end of the motion")
        for name in ("head", "antennas", "body_yaw"):
            if not np.all(np.isfinite(getattr(self, name))):
                problems.append(f"{name} has non-finite values")
        return problems

    def sample(self, k: int) -> tuple[np.ndarray, tuple[float, float], float]:
        """(head 4x4, (right, left), body_yaw) for tick ``k``."""
        return self.head[k], (float(self.antennas[k, 0]), float(self.antennas[k, 1])), float(self.body_yaw[k])

    def digest(self) -> str:
        """SHA-256 of the sampled commands (rounded to 1e-9): binds approvals and logs to exact commands."""
        h = hashlib.sha256()
        h.update(f"{self.rate_hz:.9f}|{self.duration:.9f}|{len(self)}".encode())
        for arr in (self.head, self.antennas, self.body_yaw):
            # + 0.0 turns -0.0 into 0.0 so that values rounding to zero hash the same either way
            h.update((np.round(np.asarray(arr, dtype=np.float64), 9) + 0.0).tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class PreparedMotion:
    """A timeline plus the policy report it was checked against - the only input executors accept.

    ``runnable`` requires the report to pass the policy AND to belong to this exact timeline
    (``report.timeline_digest == timeline.digest()``): a report cannot be reused for other commands.
    """

    timeline: Timeline
    report: SafetyReport
    spec: "MotionSpec | None" = None

    @property
    def runnable(self) -> bool:
        return bool(self.report.passes_policy) and self.report.timeline_digest == self.timeline.digest()
