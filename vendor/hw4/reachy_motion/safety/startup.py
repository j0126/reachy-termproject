"""Start-up ramp: from wherever the head already is (e.g. the SDK sleep pose) to a normal pose such as REST.

The sleep pose lies outside the motion envelope (SDK constant: x -21 mm, z -44 mm; MuJoCo measured z -50 mm) and
close to a Stewart limit (about 1.3 deg in MuJoCo), so the normal initial policy refuses any ramp that STARTS there.
The ramp may start where the head already is, but it must never make anything worse than at its start:

* per Stewart joint j: margin_j(t) >= min(policy floor, margin_j(0) - tol_deg)  (each joint on its own - a joint that
  starts far from its limit keeps the normal 10 deg floor);
* per axis x / y / z and roll / pitch / relative yaw / antennas: the envelope widens only to the start value (+ tol),
  never beyond it, and only on the side where the start already is outside;
* nothing else is relaxed: IK NaN, Stewart / body / antenna speeds, psi wrap, body steps, automatic body yaw;
* the ramp's END pose must satisfy the normal policy (a start-up exception cannot carry the head somewhere unsafe).

Used by ``reachy-motion sim-play`` (MuJoCo starts in the sleep pose). Initial policy only - not a safety guarantee.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from reachy_motion.library.pose import REST, RobotState
from reachy_motion.library.spec import Easing
from reachy_motion.planner import plan_transition, transition_duration
from reachy_motion.safety.checker import analyze_timeline, check_timeline
from reachy_motion.safety.policy import DEFAULT_POLICY, SafetyPolicy, Violation
from reachy_motion.timeline import PreparedMotion


def _widen(rng: tuple[float, float], start: float, tol: float) -> tuple[float, float]:
    lo, hi = rng
    return (min(lo, start - tol), max(hi, start + tol))


def startup_policy(start: RobotState, policy: SafetyPolicy = DEFAULT_POLICY, tol_deg: float = 0.25,
                   tol_mm: float = 1.0) -> SafetyPolicy:
    """The policy envelope widened only as far as ``start`` already is (Stewart margins are handled per joint)."""
    p = start.head.to_deg_mm()
    rel_yaw = math.degrees(start.head.yaw - start.body_yaw)
    return replace(
        policy, name=policy.name + "+startup",
        x_range_mm=_widen(policy.x_range_mm, p["x_mm"], tol_mm),
        y_range_mm=_widen(policy.y_range_mm, p["y_mm"], tol_mm),
        z_range_mm=_widen(policy.z_range_mm, p["z_mm"], tol_mm),
        max_roll_deg=max(policy.max_roll_deg, abs(p["roll_deg"]) + tol_deg),
        max_pitch_deg=max(policy.max_pitch_deg, abs(p["pitch_deg"]) + tol_deg),
        max_rel_yaw_deg=max(policy.max_rel_yaw_deg, abs(rel_yaw) + tol_deg),
        max_body_yaw_deg=max(policy.max_body_yaw_deg, abs(math.degrees(start.body_yaw)) + tol_deg),
        max_antenna_abs_rad=max(policy.max_antenna_abs_rad, max(abs(a) for a in start.antennas) + math.radians(tol_deg)),
    )


def stewart_floor_violation(margin: np.ndarray, t: np.ndarray, policy_floor_deg: float,
                            tol_deg: float = 0.25) -> Violation | None:
    """Per-joint start-up rule on margins (N, 6) deg: joint j may never go below min(policy floor, its start - tol).

    Each joint has its own floor: a joint that starts far from its limit keeps the normal floor even when another
    joint starts beyond its limit (the SDK sleep joints are 0.8-9.5 deg past the URDF limits on stewart_1, 2, 5, 6).
    """
    margin = np.asarray(margin, dtype=float)
    floors = np.minimum(policy_floor_deg, margin[0] - tol_deg)
    bad = ~(margin >= floors[None, :] - 1e-9)  # NaN counts as bad
    if not bad.any():
        return None
    k, j = np.argwhere(bad)[0]
    worst = np.where(np.isnan(margin), -np.inf, margin)  # IK failure (NaN) = worst case
    jw = int(np.argmin(worst.min(axis=0) - floors))
    value = float(worst[:, jw].min())
    value = value if math.isfinite(value) else math.nan
    return Violation("startup_margin",
                     f"start-up ramp brings stewart_{jw + 1} closer to its limit than at the start "
                     f"({value:.2f} deg < floor {floors[jw]:.2f} deg; first stewart_{j + 1} at t = {float(t[k]):.3f} s)",
                     float(t[k]), value, float(floors[jw]))


def prepare_startup_ramp(start: RobotState, end: RobotState = REST, policy: SafetyPolicy = DEFAULT_POLICY,
                         rate_hz: float = 50.0, min_s: float = 1.5, duration: float | None = None,
                         tol_deg: float = 0.25, tol_mm: float = 1.0) -> PreparedMotion:
    """Plan a MIN_JERK ramp ``start`` -> ``end`` and check it with the start-up rules above (runnable only if all pass)."""
    d = duration if duration is not None else transition_duration(start, end, min_s=min_s)
    ramp = plan_transition(start, end, d, Easing.MIN_JERK, rate_hz, name="startup")
    a = analyze_timeline(ramp)
    margin = np.asarray(a.stewart_margin, dtype=float)  # (N, 6) deg to the URDF limits
    floors = np.minimum(policy.min_stewart_margin_deg, margin[0] - tol_deg)
    relaxed = replace(startup_policy(start, policy, tol_deg, tol_mm),
                      min_stewart_margin_deg=float(min(policy.min_stewart_margin_deg, np.nanmin(floors))))
    report = check_timeline(ramp, relaxed)
    extra: list[Violation] = []
    v = stewart_floor_violation(margin, a.t, policy.min_stewart_margin_deg, tol_deg)
    if v is not None:
        extra.append(v)
    end_report = check_timeline(plan_transition(end, end, 0.1, Easing.LINEAR, rate_hz, name="startup_end"), policy)
    if not end_report.passes_policy:
        extra.append(Violation("startup_end", "the ramp's end pose is outside the normal initial policy: "
                               + ", ".join(v.code for v in end_report.violations)))
    report = replace(report, violations=tuple(report.violations) + tuple(extra),
                     passes_policy=bool(report.passes_policy) and not extra)
    return PreparedMotion(ramp, report)


def prepare_return_ramp(forward: PreparedMotion, start: RobotState, policy: SafetyPolicy = DEFAULT_POLICY,
                        tol_deg: float = 0.25, tol_mm: float = 1.0) -> PreparedMotion:
    """The start-up ramp ``forward`` (from ``start``) played backwards: back to the resting pose the run began at, so
    the torque can go off where the head rested limp (Step C end of a run, docs/phase4/C1_R1_CHECKLIST.md).

    Same samples in reverse order (same per-step changes, so the same speeds and stream steps), checked again with
    the start-up rules mirrored: no joint closer to its limit than at the END (the resting pose), the envelope
    widened only as far as ``start``; nothing else relaxed. No "end inside the normal policy" rule: the end is,
    by construction, where this run's head already rested."""
    tl = forward.timeline
    rev = replace(tl, name=f"{tl.name}_return", head=tl.head[::-1].copy(), antennas=tl.antennas[::-1].copy(),
                  body_yaw=tl.body_yaw[::-1].copy(), head_key=tl.head_key[::-1].copy(),
                  antenna_key=tl.antenna_key[::-1].copy(), body_key=tl.body_key[::-1].copy(), spec_digest="")
    a = analyze_timeline(rev)
    margin = np.asarray(a.stewart_margin, dtype=float)
    floors = np.minimum(policy.min_stewart_margin_deg, margin[-1] - tol_deg)
    relaxed = replace(startup_policy(start, policy, tol_deg, tol_mm),
                      min_stewart_margin_deg=float(min(policy.min_stewart_margin_deg, np.nanmin(floors))))
    report = check_timeline(rev, relaxed)
    v = stewart_floor_violation(margin[::-1], a.t, policy.min_stewart_margin_deg, tol_deg)  # floors from the END
    extra = [] if v is None else [replace(v, code="return_margin")]
    report = replace(report, violations=tuple(report.violations) + tuple(extra),
                     passes_policy=bool(report.passes_policy) and not extra)
    return PreparedMotion(rev, report)
