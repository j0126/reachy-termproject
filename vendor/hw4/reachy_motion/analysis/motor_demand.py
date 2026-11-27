"""Preliminary IK-based motor demand of single goto_target moves (Phase 1 hardware-constraint check).

For each (method, axis, amplitude, duration) the SDK's own GotoMove is evaluated densely from
neutral to the target, every sample goes through the daemon's IK engine (AnalyticalKinematics,
automatic body yaw), and the joint trajectories are differentiated numerically. Reported per row:
peak Stewart speed and its ratio to the XL330-M288-T no-load speed (5 V / 6 V), peak body-yaw
speed, the minimum margin to the URDF joint limits along the whole path (CARTOON's
anticipation/overshoot included) and the IK NaN count.

data_kind = computed. The no-load speed is an upper bound (speed drops with load torque) and
the real bus voltage is unknown, so a ratio well below 1 is necessary, not sufficient. MuJoCo does
not model XL330 speed/torque saturation, so Phase 3 (sim) only extends this computed table; the
real tracking limit can only be measured on the robot (Phase 4, after explicit approval).
"""

from __future__ import annotations

import numpy as np
from reachy_mini.utils.interpolation import InterpolationTechnique

from reachy_motion.analysis.goto_replay import (
    JOINT_NAMES,
    STEWART,
    GotoCase,
    HeadTarget,
    ik_joints,
    new_kinematics,
)
from reachy_motion.analysis.limits import BODY_MOTOR_PROXY, STEWART_MOTOR, urdf_joint_limits

AXIS_FIELD = {"pitch": "pitch_deg", "roll": "roll_deg", "yaw": "yaw_deg", "z": "z_mm", "x": "x_mm", "y": "y_mm"}

DEFAULT_CASES: tuple[tuple[str, float], ...] = (
    ("pitch", 20.0), ("pitch", 30.0), ("pitch", -30.0),
    ("roll", 20.0), ("roll", 30.0),
    ("yaw", 30.0), ("yaw", 60.0),
    ("z", 10.0), ("z", -20.0),
)
DEFAULT_DURATIONS: tuple[float, ...] = (0.3, 0.5, 1.0, 2.0)


def demand_row(method: InterpolationTechnique, axis: str, amplitude: float, duration: float,
               rate_hz: float = 1000.0) -> dict[str, object]:
    """One goto from neutral: peak joint speeds, spec ratios, limit margin, NaN count."""
    case = GotoCase(target=HeadTarget(**{AXIS_FIELD[axis]: amplitude}), duration=duration, method=method)
    move = case.build_move()
    kin = new_kinematics()
    t = np.linspace(0.0, duration, int(round(duration * rate_hz)) + 1)
    q = np.array([ik_joints(kin, h, float(b)) for h, _a, b in (move.evaluate(float(ti)) for ti in t)])
    nan_rows = int(np.isnan(q).any(axis=1).sum())
    qd = np.rad2deg(np.gradient(q, t, axis=0))
    limits = urdf_joint_limits()
    lo = np.array([limits[n][0] for n in JOINT_NAMES])
    hi = np.array([limits[n][1] for n in JOINT_NAMES])
    margin = np.rad2deg(np.minimum(q - lo, hi - q))  # (N, 7); negative = beyond the URDF limit
    finite = ~np.isnan(margin).any(axis=1)
    min_margin = float(np.min(margin[finite])) if finite.any() else float("nan")
    worst_joint = JOINT_NAMES[int(np.argmin(np.min(np.where(np.isnan(margin), np.inf, margin), axis=0)))]
    stewart_peak = float(np.nanmax(np.abs(qd[:, STEWART])))
    body_peak = float(np.nanmax(np.abs(qd[:, 0])))
    return {
        "method": method.value,
        "axis": axis,
        "amplitude": amplitude,
        "amplitude_unit": "mm" if axis in ("x", "y", "z") else "deg",
        "duration_s": duration,
        "peak_stewart_speed_deg_s": stewart_peak,
        "ratio_to_noload_5v": stewart_peak / STEWART_MOTOR.no_load_deg_s_5v,
        "ratio_to_noload_6v": stewart_peak / STEWART_MOTOR.no_load_deg_s_6v,
        "peak_body_speed_deg_s": body_peak,
        "body_ratio_to_proxy_noload_5v": body_peak / BODY_MOTOR_PROXY.no_load_deg_s_5v,
        "min_margin_to_urdf_limit_deg": min_margin,
        "worst_joint": worst_joint,
        "ik_nan_samples": nan_rows,
    }


def demand_grid(methods, cases=DEFAULT_CASES, durations=DEFAULT_DURATIONS) -> list[dict[str, object]]:
    return [demand_row(m, axis, amp, d) for m in methods for axis, amp in cases for d in durations]
