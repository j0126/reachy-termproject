"""Offline checks of a commanded pose sequence (dry runs, tests, motion.csv of a robot session).

Speeds and accelerations are finite differences of the commands that were sent (the deadband
skips unchanged targets, which only lowers the estimates); joint changes and motor-limit margins
come from the same inverse kinematics as the controller's joint guard.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .limits import JointGuard
from .pose import FIELDS, NEUTRAL, Pose


def command_stats(times_s: list[float] | np.ndarray, poses: list[Pose], guard: JointGuard | None = None,
                  neutral: Pose = NEUTRAL) -> dict[str, Any]:
    """Per-axis max deviation from neutral, max speed and max acceleration; joint change / margin."""
    t = np.asarray(times_s, dtype=np.float64)
    if len(poses) < 3:
        return {"commands": len(poses)}
    x = np.stack([p.vec() for p in poses])
    dev = np.abs(x - neutral.vec())
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    t, x = t[keep], x[keep]
    dt = np.diff(t)
    v = np.diff(x, axis=0) / dt[:, None]  # at the interval midpoints (sends are not evenly spaced:
    a = np.diff(v, axis=0) / ((dt[:-1] + dt[1:]) / 2.0)[:, None]  # the deadband skips unchanged targets)
    out: dict[str, Any] = {"commands": len(poses), "seconds": round(float(t[-1] - t[0]), 2),
                           "max_dev": {f: round(float(dev[:, i].max()), 3) for i, f in enumerate(FIELDS)},
                           "max_speed": {f: round(float(np.abs(v[:, i]).max()), 2) for i, f in enumerate(FIELDS)},
                           "max_accel": {f: round(float(np.abs(a[:, i]).max()), 1) if len(a) else None
                                         for i, f in enumerate(FIELDS)}}
    if guard is not None and guard.available:
        changes, margins, yaws = [], [], []
        q0 = guard.joints_deg(neutral)
        for p in poses:
            res = guard.check(p, "ramp")
            if res.max_change_deg is not None:
                changes.append(res.max_change_deg)
                margins.append(res.min_margin_deg)
                yaws.append(abs(res.body_yaw_change_deg))
        out.update(max_joint_change_deg=round(max(changes), 2) if changes else None,
                   min_limit_margin_deg=round(min(margins), 2) if margins else None,
                   max_body_yaw_change_deg=round(max(yaws), 3) if yaws else None,
                   neutral_joints_deg=None if q0 is None else [round(float(q), 2) for q in q0])
    return out


def read_motion_csv(path: str | Path) -> dict[str, list[Any]]:
    """``motion.csv`` of a session -> {"t", "poses", "state", "mode"} (one entry per command sent)."""
    out: dict[str, list[Any]] = {"t": [], "poses": [], "state": [], "mode": []}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out["t"].append(float(row["t"]))
            out["poses"].append(Pose(*(float(row[f]) for f in FIELDS)))
            out["state"].append(row.get("state", ""))
            out["mode"].append(row.get("mode", ""))
    return out


#: controller modes whose commands are conversation motion (box + joint-change limit) vs. ramps
CONVERSATION_MODES = ("HOLD", "ACTIVE")
RAMP_MODES = ("STARTING", "SHUTTING_DOWN")


def _merge(parts: list[dict[str, Any]]) -> dict[str, Any]:
    parts = [p for p in parts if p.get("commands", 0) >= 3 and "max_dev" in p]
    if not parts:
        return {"commands": 0}
    out: dict[str, Any] = {"commands": sum(p["commands"] for p in parts),
                           "seconds": round(sum(p["seconds"] for p in parts), 2)}
    for key in ("max_dev", "max_speed", "max_accel"):
        out[key] = {f: max((p[key][f] for p in parts if p[key][f] is not None), default=None) for f in FIELDS}
    for key, pick in (("max_joint_change_deg", max), ("min_limit_margin_deg", min),
                      ("max_body_yaw_change_deg", max)):
        values = [p[key] for p in parts if p.get(key) is not None]
        if values:
            out[key] = pick(values)
    return out


def phase_stats(log: dict[str, list[Any]], guard: JointGuard | None = None) -> dict[str, Any]:
    """:func:`command_stats` per phase (conversation / ramps), over contiguous runs of each phase."""
    result = {}
    for name, modes in (("conversation", CONVERSATION_MODES), ("ramps", RAMP_MODES)):
        parts, start = [], None
        for i, mode in enumerate(log["mode"] + [""]):
            if mode in modes and start is None:
                start = i
            elif mode not in modes and start is not None:
                parts.append(command_stats(log["t"][start:i], log["poses"][start:i], guard))
                start = None
        result[name] = _merge(parts)
    return result
