"""Safety limits applied to every motion command (docs/PHASE4_5_DESIGN.md, sections 7-8).

Order per command: pose box (conversation motions only) -> joint guard (PC inverse kinematics)
-> speed / acceleration limiter -> send. The planned motions stay well inside all of them; the
limits only catch bugs, sudden target changes and extreme settings.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ..config import MotionConfig
from .pose import NEUTRAL, Pose, torso_frame

log = logging.getLogger(__name__)

#: Stewart motor position limits (raw 0..4095) from the SDK's assets/config/hardware_config.yaml
#: (stewart_1 .. stewart_6). The servo firmware enforces them as well.
STEWART_RAW_LIMITS = ((1502, 2958), (1138, 2844), (1502, 2958), (1138, 2594), (1252, 2958), (1138, 2594))


def raw_to_deg(raw: float) -> float:
    return (raw - 2048) * 360.0 / 4096.0


#: The same limits as inverse-kinematics joint angles. The mapping joint = (raw - 2048) * 360 / 4096
#: is INFERRED, not read from the motor controller source (Rust): with it the neutral pose sits
#: mid-range (odd joints +35.9 in [-48, +80], even joints -35.9 in [-80, +48]). The guard therefore
#: also bounds the change from the neutral pose, which does not depend on the mapping.
STEWART_LIMITS_DEG = tuple((raw_to_deg(lo), raw_to_deg(hi)) for lo, hi in STEWART_RAW_LIMITS)


@dataclass(frozen=True)
class Limits:
    max_pitch_deg: float
    max_roll_deg: float
    max_yaw_deg: float
    max_offset_mm: float
    max_antenna_rad: float
    head_speed: float  # deg/s
    head_accel: float  # deg/s^2
    linear_speed: float  # mm/s
    linear_accel: float  # mm/s^2
    antenna_speed: float  # rad/s
    antenna_accel: float  # rad/s^2

    @classmethod
    def from_config(cls, m: MotionConfig) -> "Limits":
        return cls(m.max_pitch_deg, m.max_roll_deg, m.max_yaw_deg, m.max_offset_mm, m.max_antenna_rad,
                   m.max_head_speed_dps, m.max_head_accel_dps2, m.max_linear_speed_mms, m.max_linear_accel_mms2,
                   m.max_antenna_speed_rps, m.max_antenna_accel_rps2)

    @property
    def vmax(self) -> np.ndarray:
        return np.array([self.head_speed] * 3 + [self.linear_speed] * 3 + [self.antenna_speed] * 2)

    @property
    def amax(self) -> np.ndarray:
        return np.array([self.head_accel] * 3 + [self.linear_accel] * 3 + [self.antenna_accel] * 2)


def clamp_box(pose: Pose, lim: Limits, neutral: Pose = NEUTRAL) -> tuple[Pose, bool]:
    """Clamp a conversation pose to the allowed deviation from neutral (antennas: absolute angle)."""
    d = (pose - neutral).vec()
    hi = np.array([lim.max_roll_deg, lim.max_pitch_deg, lim.max_yaw_deg] + [lim.max_offset_mm] * 3)
    head = np.clip(d[:6], -hi, hi)
    out = (neutral + Pose.from_vec(np.concatenate([head, d[6:]]))).vec()
    out[6:] = np.clip(out[6:], -lim.max_antenna_rad, lim.max_antenna_rad)
    clamped = bool(np.any(np.abs(out - pose.vec()) > 1e-9))
    return Pose.from_vec(out), clamped


class RateLimiter:
    """Per-axis speed and acceleration limits with target feed-forward.

    A smooth target within the limits is followed exactly; a jump is turned into an
    accelerate - cruise - brake profile that stops at the new target (no overshoot for a
    target at rest).
    """

    def __init__(self, lim: Limits) -> None:
        self.vmax, self.amax = lim.vmax, lim.amax
        self.p: np.ndarray | None = None
        self.v = np.zeros(8)
        self.last_target: np.ndarray | None = None

    def reset(self, pose: Pose) -> None:
        self.p = pose.vec()
        self.v = np.zeros(8)
        self.last_target = self.p.copy()

    def step(self, target: Pose, dt: float) -> tuple[Pose, bool]:
        x = target.vec()
        if self.p is None or dt <= 0:
            self.reset(target)
            return target, False
        vt = (x - self.last_target) / dt  # the target's own velocity (feed-forward)
        e = self.last_target - self.p  # lag behind the previous target (0 while tracking exactly)
        self.last_target = x
        catch_up = np.sign(e) * np.minimum(np.abs(e) / dt, self._braking_speed(np.abs(e), dt))
        v_cmd = np.clip(vt + catch_up, -self.vmax, self.vmax)
        dv = np.clip(v_cmd - self.v, -self.amax * dt, self.amax * dt)
        self.v = self.v + dv
        self.p = self.p + self.v * dt
        limited = bool(np.any(np.abs(self.p - x) > 1e-6))
        return Pose.from_vec(self.p), limited

    def _braking_speed(self, dist: np.ndarray, dt: float) -> np.ndarray:
        """Highest speed that still stops within ``dist`` when braking at ``amax`` in steps of ``dt``.

        Discrete counterpart of sqrt(2 a d): with v = n a dt the distance covered while braking is
        a dt^2 n (n + 1) / 2, so n = sqrt(1/4 + 2 d / (a dt^2)) - 1/2 (no overshoot from the stepping).
        """
        adt = self.amax * dt
        return adt * (np.sqrt(0.25 + 2.0 * dist / (adt * dt)) - 0.5)


@dataclass
class GuardResult:
    ok: bool
    reason: str = ""
    max_change_deg: float | None = None
    min_margin_deg: float | None = None
    body_yaw_change_deg: float | None = None  # how far the daemon's IK would turn the torso (must stay ~0)
    scale: float = 1.0

    def summary(self) -> dict[str, object]:
        return {"ok": self.ok, "reason": self.reason, "scale": self.scale,
                "max_change_deg": None if self.max_change_deg is None else round(self.max_change_deg, 2),
                "min_margin_deg": None if self.min_margin_deg is None else round(self.min_margin_deg, 2),
                "body_yaw_change_deg": None if self.body_yaw_change_deg is None else round(self.body_yaw_change_deg, 2)}


class JointGuard:
    """Checks head poses with the SDK's analytical inverse kinematics on this PC (~4 us per call).

    ``conversation`` poses must keep every Stewart joint within ``max_change_deg`` of its neutral
    angle (the SDK speech wobbler reaches 11.4 deg; our motions about 7 deg) and ``margin_deg``
    away from the motor limits; ``ramp`` poses (to / from the sleep pose, the SDK wake / sleep
    path) only need the margin. The body yaw must not move (we never command it).

    Poses are in the torso frame. The torso may stand turned (R0, 2026-10-17: 8.2 deg with the torque
    off); ``set_body_yaw`` makes the IK use that angle, as the daemon does (its IK keeps the present
    body yaw target): with it the PC joints matched the daemon's within 0.1 deg, with 0 they were 7 deg off.
    """

    BODY_YAW_TOLERANCE_DEG = 3.0

    def __init__(self, max_change_deg: float = 12.0, margin_deg: float = 10.0, enabled: bool = True) -> None:
        self.max_change_deg = max_change_deg
        self.margin_deg = margin_deg
        self.available = False
        self.error: str | None = None
        self._kin = None
        self._q0: np.ndarray | None = None
        self.body_yaw_deg = 0.0
        if not enabled:
            self.error = "disabled in config"
            return
        try:
            from reachy_mini.kinematics.analytical_kinematics import AnalyticalKinematics

            self._kin = AnalyticalKinematics(automatic_body_yaw=True)  # as the daemon (SDK default)
            self._q0 = self.joints_deg(NEUTRAL)
            self.available = self._q0 is not None
        except Exception as exc:  # noqa: BLE001 - no kinematics -> box and speed limits only
            self.error = repr(exc)
            log.warning("joint guard unavailable: %s", exc)

    def set_body_yaw(self, yaw_deg: float) -> None:
        """Torso yaw the robot holds (read when the motors are turned on); poses stay in the torso frame."""
        self.body_yaw_deg = float(yaw_deg)
        if self._kin is not None:
            self._q0 = self.joints_deg(NEUTRAL)
            self.available = self._q0 is not None

    def joints_deg(self, pose: Pose) -> np.ndarray | None:
        """Body yaw + 6 Stewart joints (deg) the daemon would command for ``pose`` (torso frame)."""
        if self._kin is None:
            return None
        try:
            world = torso_frame(self.body_yaw_deg) @ pose.head_matrix()
            q = self._kin.ik(world, body_yaw=math.radians(self.body_yaw_deg))
        except Exception:  # noqa: BLE001 - treated as unreachable
            return None
        if q is None or np.any(np.isnan(q)):
            return None
        return np.degrees(np.asarray(q, dtype=np.float64))

    def check(self, pose: Pose, mode: str = "conversation") -> GuardResult:
        if not self.available:
            return GuardResult(True, "guard unavailable")
        q = self.joints_deg(pose)
        if q is None:
            return GuardResult(False, "unreachable (inverse kinematics)")
        margin = min(min(q[i + 1] - lo, hi - q[i + 1]) for i, (lo, hi) in enumerate(STEWART_LIMITS_DEG))
        change = float(np.max(np.abs(q[1:] - self._q0[1:])))
        body_change = float(q[0] - self.body_yaw_deg)
        res = GuardResult(True, "", change, float(margin), body_change)
        if margin < self.margin_deg:
            res.ok, res.reason = False, f"joint within {margin:.1f} deg of a motor limit"
        elif mode == "conversation" and change > self.max_change_deg:
            res.ok, res.reason = False, f"joint change {change:.1f} deg from neutral"
        elif abs(body_change) > self.BODY_YAW_TOLERANCE_DEG:
            res.ok, res.reason = False, f"the torso would turn ({body_change:.1f} deg)"
        return res

    def fit(self, pose: Pose, neutral: Pose = NEUTRAL, mode: str = "conversation") -> tuple[Pose | None, GuardResult]:
        """``pose`` if allowed, else its head deviation scaled down (0.75, 0.5, 0.25); None if nothing fits."""
        res = self.check(pose, mode)
        if res.ok:
            return pose, res
        head = (pose - neutral).head_only()
        for k in (0.75, 0.5, 0.25):
            cand = neutral.head_only() + head.scaled(k) + pose.antennas_only()
            sub = self.check(cand, mode)
            if sub.ok:
                sub.scale, sub.reason = k, f"scaled {k} ({res.reason})"
                return cand, sub
        return None, res


def minjerk_peak_speed(distance: float, duration: float) -> float:
    return 1.875 * abs(distance) / duration if duration > 0 else math.inf
