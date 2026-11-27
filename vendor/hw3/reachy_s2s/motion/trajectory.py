"""Smooth time profiles: minimum-jerk blends, bumps, and the startup / shutdown ramps.

The ramps follow the SDK's own goto path (``linear_pose_interpolation`` with a minimum-jerk
time law, as ``ReachyMini.goto_target`` / ``wake_up`` do on the daemon), but are evaluated here and
streamed with ``set_target`` so they can be stopped at any tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .limits import Limits
from .pose import Pose, pose_from_matrix


def minjerk(s: float) -> float:
    """0 -> 1 with zero velocity and acceleration at both ends."""
    s = min(1.0, max(0.0, s))
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def bump(s: float) -> float:
    """0 -> 1 -> 0 over s in [0, 1] (raised cosine; zero velocity at both ends)."""
    if s <= 0.0 or s >= 1.0:
        return 0.0
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * s))


def plateau(s: float, rise: float) -> float:
    """0 -> 1, hold, -> 0 over s in [0, 1]; ``rise`` is the fraction used for each ramp."""
    if s <= 0.0 or s >= 1.0:
        return 0.0
    if s < rise:
        return minjerk(s / rise)
    if s > 1.0 - rise:
        return minjerk((1.0 - s) / rise)
    return 1.0


@dataclass
class Blend:
    """Minimum-jerk move from ``start`` to ``end`` starting at ``t0_ns``."""

    start: Pose
    end: Pose
    t0_ns: int
    dur_s: float

    def fraction(self, t_ns: int) -> float:
        if self.dur_s <= 0:
            return 1.0
        return minjerk((t_ns - self.t0_ns) / 1e9 / self.dur_s)

    def at(self, t_ns: int) -> Pose:
        return self.start + (self.end - self.start).scaled(self.fraction(t_ns))

    def done(self, t_ns: int) -> bool:
        return (t_ns - self.t0_ns) / 1e9 >= self.dur_s


def ramp_duration(start: Pose, end: Pose, lim: Limits, min_s: float, margin: float = 1.15) -> float:
    """Shortest minimum-jerk duration that keeps peak speed and acceleration under the caps."""
    d = np.abs((end - start).vec())
    t_v = 1.875 * d / lim.vmax  # minimum-jerk peak speed = 1.875 * distance / T
    t_a = np.sqrt(5.7735 * d / lim.amax)  # peak acceleration = 5.7735 * distance / T^2
    return max(min_s, float(np.max(t_v)) * margin, float(np.max(t_a)) * margin)


def _slerp_rotation(r0: np.ndarray, r1: np.ndarray, s: float) -> np.ndarray:
    rel = r0.T @ r1
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rel) - 1.0) / 2.0)))
    if angle < 1e-9:
        return r0.copy()
    axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]) / (2 * math.sin(angle))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = angle * s
    return r0 @ (np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k))


@dataclass
class Ramp:
    """Startup / shutdown move between two head matrices and antenna pairs (SDK goto path)."""

    start_m: np.ndarray
    end_m: np.ndarray
    start_ant: tuple[float, float]
    end_ant: tuple[float, float]
    t0_ns: int
    dur_s: float

    def pose_at(self, t_ns: int) -> Pose:
        s = minjerk((t_ns - self.t0_ns) / 1e9 / self.dur_s) if self.dur_s > 0 else 1.0
        try:
            from reachy_mini.utils.interpolation import linear_pose_interpolation

            m = linear_pose_interpolation(self.start_m, self.end_m, s, yaw_as_scalar=True)
        except Exception:  # noqa: BLE001 - SDK not importable (tests): same idea, slerp + lerp
            m = np.eye(4)
            m[:3, :3] = _slerp_rotation(self.start_m[:3, :3], self.end_m[:3, :3], s)
            m[:3, 3] = self.start_m[:3, 3] + (self.end_m[:3, 3] - self.start_m[:3, 3]) * s
        ant = [a + (b - a) * s for a, b in zip(self.start_ant, self.end_ant)]
        return pose_from_matrix(m, ant)

    def done(self, t_ns: int) -> bool:
        return (t_ns - self.t0_ns) / 1e9 >= self.dur_s
