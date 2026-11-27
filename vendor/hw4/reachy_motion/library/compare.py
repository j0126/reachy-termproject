"""Expressive vs Neutral comparison: per-key speeds, neutral-variant conformance and comparison conditions.

Pure functions on ``MotionSpec`` (SDK-free, no I/O). Every output is JSON-able - plain dict, list, str, float,
int, bool and None; numpy scalars in pose / antenna / yaw values and durations are converted (only
``MotionSpec.digest()``, used for ``spec_digest``, needs plain numbers in durations, holds and scale factors) - and
labelled ``data_kind = "computed"``: the numbers are derived from the DESIGN parameters, not simulated or measured
(CLAUDE.md rule 6).

What the numbers mean (task space, on the effective keys = ``MotionSpec.effective()``, i.e. with
``amplitude_scale`` and ``tempo_scale`` applied; joint / motor speeds are the safety checker's job, by IK):

* Head rotation of a key (``change_deg``) = the angle the head turns along the SDK's own path.
  ``GotoMove.evaluate`` (reachy_mini/motion/goto.py:57-64) calls ``linear_pose_interpolation(start, target, s,
  yaw_as_scalar=True)`` (reachy_mini/utils/interpolation.py:74-84): R(s) = Rz(psi0 + s dpsi) Res0 Exp(s v), with
  psi the Euler "xyz" yaw of each pose in (-180, 180] deg (reachy_mini/utils/rotation.py:211-225),
  Res = Rz(-psi) R the pitch/roll residual and v = rotvec(Res0^T Res1) (rotation.py:247-299). Its angular
  velocity per unit s is Rz(psi(s)) (dpsi z + Res0 v), so the head turns at a constant rate in s and the key's
  path is |dpsi z + Res0 v| long (closed form; the tests compare it with the SDK function integrated
  numerically). That equals the geodesic angle (``pose.rotation_distance``, column ``geodesic_deg``) for
  single-axis keys whose yaw changes by at most 180 deg, is slightly longer for multi-axis keys, and is the LONG
  way round when the Euler yaw changes by more than 180 deg (yaw -100 -> +100 deg: 200 deg through the front;
  yaw 170 -> 190 deg, which the SDK reads as -170 deg: 340 deg). ``comparison_issues`` reports such keys.
  Translation = straight-line distance (linear in s, interpolation.py:94-98).
* Antennas: the larger of the two angle changes; body: |yaw change|. Both lerp with the same time law
  (goto.py:65-72).
* Average speed = path / key duration. Nominal peak = k(easing) x average, k = max |ds/dtau| of the SDK time
  law ``time_trajectory`` (reachy_mini/utils/interpolation.py:117-153): linear 1, minjerk 1.875,
  ease_in_out 2, cartoon 3 + c2 = 5.595 with c2 = 1.70158 x 1.525 (interpolation.py:137-138). Every channel
  moves at a constant rate in s, so this is the exact peak of the commanded path. Same factors as Phase 1
  (docs/phase1/SDK_ANALYSIS.md section 3, ``analysis.interpolation.characterize``); cartoon also moves about
  10 % of the change backwards and beyond the target, which the path length does not count.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from reachy_motion.library.pose import Pose6, RobotState, nearest_rotation, rotation_distance
from reachy_motion.library.spec import MIN_KEY_DURATION_S, TRACKS, Easing, MotionSpec

__all__ = [
    "DATA_KIND", "PEAK_SPEED_FACTOR", "SEGMENT_COLUMNS", "MOVE_EPS_DEG", "MOVE_EPS_MM", "NEUTRAL_ANTENNA_EPS_RAD",
    "SPEED_UNIFORMITY_TOL", "SAME_POSE_TOL_DEG", "SAME_POSE_TOL_MM", "RETIME_DECIMALS", "ISSUE_SEVERITIES", "Issue",
    "segment_speeds", "state_difference", "neutral_conformance", "comparison_issues", "comparison_conditions",
    "median_translation_speed_mms", "retime_uniform_speed",
]

DATA_KIND = "computed"  # derived from the design parameters (CLAUDE.md rule 6)

_CARTOON_C2 = 1.70158 * 1.525  # reachy_mini/utils/interpolation.py:137-138
#: Peak / average speed ratio of each easing = max |ds/dtau| of the SDK time law. Keyed by the Easing
#: string values (JSON-friendly); ``Easing`` members work as keys too (a ``str`` enum hashes like its value).
PEAK_SPEED_FACTOR: dict[str, float] = {
    Easing.LINEAR.value: 1.0,  # s' = 1
    Easing.MIN_JERK.value: 1.875,  # s' = 30 tau^2 (1 - tau)^2, max at tau = 0.5
    Easing.EASE_IN_OUT.value: 2.0,  # s' = 4 tau for tau < 0.5
    Easing.CARTOON.value: 3.0 + _CARTOON_C2,  # s'(0.5) = 3 (c2 + 1) - 2 c2
}

MOVE_EPS_DEG = 1e-6  # a key "moves" if it changes its track by more than this (deg) ...
MOVE_EPS_MM = 1e-6  # ... or translates the head by more than this (mm)
#: Neutral antenna keys may not differ from the start antennas by more than this (exact check: an antenna key
#: at -10/+10 deg is 0.0019 deg away from the SDK rest angle and counts; motion files write ``rest`` for it).
NEUTRAL_ANTENNA_EPS_RAD = 1e-9
SPEED_UNIFORMITY_TOL = 0.15  # neutral head keys: |pace - median pace| <= 15 % of the median (see neutral_conformance)
SAME_POSE_TOL_DEG = 0.01  # "same start / end": well below one XL330 position tick (360/4096 = 0.088 deg)
SAME_POSE_TOL_MM = 0.01
#: Retimed durations are rounded to this many decimals of a second (1 us) so saved files read 0.5, not
#: 0.4999999999999999; far below the 5 ms safety-check grid and the 20 ms control tick (speed change
#: <= 2.5e-5 relative for a 20 ms key).
RETIME_DECIMALS = 6

#: Columns of every :func:`segment_speeds` row (``None`` where a column does not apply to the track).
SEGMENT_COLUMNS: tuple[str, ...] = (
    "track", "index", "label", "easing", "duration_s", "hold_s", "t_depart_s", "t_arrive_s", "t_release_s",
    "moves", "change_deg", "geodesic_deg", "avg_speed_dps", "peak_speed_dps",
    "translation_mm", "avg_trans_speed_mms", "peak_trans_speed_mms",
    "d_x_mm", "d_y_mm", "d_z_mm", "d_roll_deg", "d_pitch_deg", "d_yaw_deg",
    "d_right_deg", "d_left_deg", "d_body_yaw_deg",
)

ISSUE_SEVERITIES: tuple[str, ...] = ("warning", "info")


@dataclass(frozen=True)
class Issue:
    """A design-level finding (not a safety violation: the safety checker decides what may run)."""

    code: str  # e.g. "start_mismatch", "neutral_antennas"
    message: str
    severity: str = "warning"  # "warning" | "info"

    def __post_init__(self) -> None:
        if self.severity not in ISSUE_SEVERITIES:
            raise ValueError(f"severity {self.severity!r} not in {ISSUE_SEVERITIES}")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


# ====================================================================== helpers
def _easing_name(easing: Any) -> str:
    try:
        return Easing(easing).value
    except ValueError:
        return str(easing)


def _principle_name(p: Any) -> str:
    return str(getattr(p, "value", p))


def _avg(change: float, duration: float) -> float | None:
    """change / duration as a plain float; ``None`` when undefined (a non-positive duration that still moves)."""
    change, duration = float(change), float(duration)  # numpy float32 would survive the division (NEP 50)
    if duration > 0.0:
        return change / duration
    return 0.0 if change == 0.0 else None


def _peak(avg: float | None, easing: str) -> float | None:
    k = PEAK_SPEED_FACTOR.get(easing)
    return None if avg is None or k is None else avg * k


def _rotation_deg(a: Pose6, b: Pose6) -> float:
    """Geodesic angle between two head poses (deg): how far apart they are, not the path between them."""
    return math.degrees(rotation_distance(a.matrix(), b.matrix()))


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)  # same layout as the SDK's _axis_matrix("z") (rotation.py:38-45)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _euler_yaw(r: np.ndarray) -> float:
    """Euler "xyz" yaw (rad, in (-pi, pi]) exactly as ``Rotation.as_euler`` computes it (rotation.py:211-225)."""
    sin_pitch = float(np.clip(-r[2, 0], -1.0, 1.0))
    if math.sqrt(max(0.0, 1.0 - sin_pitch * sin_pitch)) > 1e-7:
        return math.atan2(float(r[1, 0]), float(r[0, 0]))
    return 0.0  # gimbal lock (pitch +-90 deg): the SDK pins yaw to 0


def _rotvec(m: np.ndarray) -> np.ndarray:
    """Rotation vector (norm <= pi) of a 3x3 rotation, Shepperd's method as ``Rotation.as_rotvec``
    (reachy_mini/utils/rotation.py:247-299), copied so this module stays SDK-free."""
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        scale = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * scale
        xyz = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / scale
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / scale
        xyz = np.array([0.25 * scale, (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale])
    elif m[1, 1] > m[2, 2]:
        scale = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / scale
        xyz = np.array([(m[0, 1] + m[1, 0]) / scale, 0.25 * scale, (m[1, 2] + m[2, 1]) / scale])
    else:
        scale = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / scale
        xyz = np.array([(m[0, 2] + m[2, 0]) / scale, (m[1, 2] + m[2, 1]) / scale, 0.25 * scale])
    if w < 0.0:  # canonical hemisphere: angle in [0, pi]
        xyz, w = -xyz, -w
    norm = float(np.linalg.norm(xyz))
    if norm < 1e-12:
        return 2.0 * xyz
    return xyz * (2.0 * math.atan2(norm, float(w)) / norm)


def _head_path(a: Pose6, b: Pose6) -> tuple[float, float]:
    """``(angle turned along the SDK goto path, Euler yaw change the SDK interpolates)``, both in deg.

    Closed form of the length of ``linear_pose_interpolation(a, b, s, yaw_as_scalar=True)`` for s in [0, 1]
    (reachy_mini/utils/interpolation.py:74-84): the angular velocity per unit s is Rz(psi(s)) (dpsi z + Res_a v),
    whose norm does not depend on s (module docstring).
    """
    ra, rb = nearest_rotation(a.matrix()), nearest_rotation(b.matrix())  # the SDK orthonormalises too
    psi_a, psi_b = _euler_yaw(ra), _euler_yaw(rb)
    res_a = _rot_z(-psi_a) @ ra
    res_b = _rot_z(-psi_b) @ rb
    w = res_a @ _rotvec(res_a.T @ res_b)
    w[2] += psi_b - psi_a
    return math.degrees(float(np.linalg.norm(w))), math.degrees(psi_b - psi_a)


def _translation_mm(a: Pose6, b: Pose6) -> float:
    return 1000.0 * float(np.linalg.norm(b.as_array()[:3] - a.as_array()[:3]))


def _antenna_change_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.degrees(max(abs(b[0] - a[0]), abs(b[1] - a[1])))


def _key_name(track: str, index: int, label: str) -> str:
    return f"{track}[{index}]" + (f" {label!r}" if label else "")


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _state_human(state: RobotState) -> dict[str, Any]:
    """Boundary adapter for reports: head in deg/mm, antennas [right, left] and body yaw in deg (plain floats)."""
    return {"head": {k: float(v) for k, v in state.head.to_deg_mm().items()},
            "antennas_deg": [float(math.degrees(a)) for a in state.antennas],
            "body_yaw_deg": float(math.degrees(state.body_yaw))}


# ====================================================================== per-key speeds
def segment_speeds(spec: MotionSpec) -> list[dict[str, Any]]:
    """One row per key and track (head, antennas, body order) of ``spec.effective()``; columns ``SEGMENT_COLUMNS``.

    ``change_deg``: head = angle turned along the SDK goto path from the previous key (or the start) - the
    rotation the speeds are based on (module docstring) -, ``geodesic_deg`` = the shortest angle between the two
    poses (``pose.rotation_distance``; head only); antennas = the larger |change| of the two, body = |yaw
    change|. ``avg_speed_dps`` = change_deg / duration, ``peak_speed_dps`` = nominal peak k(easing) x average.
    Head rows also carry the translation (mm, mm/s) and the signed Euler / position component changes ``d_*``
    of the key values. Times ``t_*`` are effective (s).
    """
    eff = spec.effective()
    rows: list[dict[str, Any]] = []
    for track in TRACKS:
        prev = eff.start_value(track)
        for i, (k, (t0, t1, t2)) in enumerate(zip(eff.track(track), eff.key_times(track))):
            easing = _easing_name(k.easing)
            duration = float(k.duration)
            row: dict[str, Any] = dict.fromkeys(SEGMENT_COLUMNS)
            row.update(track=track, index=i, label=str(k.label), easing=easing, duration_s=duration,
                       hold_s=float(k.hold), t_depart_s=float(t0), t_arrive_s=float(t1), t_release_s=float(t2))
            if track == "head":
                (path, _dpsi), trans = _head_path(prev, k.pose), _translation_mm(prev, k.pose)
                d = k.pose - prev
                avg_t = _avg(trans, duration)
                row.update(change_deg=path, geodesic_deg=_rotation_deg(prev, k.pose), translation_mm=trans,
                           avg_trans_speed_mms=avg_t, peak_trans_speed_mms=_peak(avg_t, easing),
                           d_x_mm=d.x * 1000.0, d_y_mm=d.y * 1000.0, d_z_mm=d.z * 1000.0,
                           d_roll_deg=math.degrees(d.roll), d_pitch_deg=math.degrees(d.pitch),
                           d_yaw_deg=math.degrees(d.yaw))
                moves = path > MOVE_EPS_DEG or trans > MOVE_EPS_MM
            elif track == "antennas":
                dr, dl = math.degrees(k.antennas[0] - prev[0]), math.degrees(k.antennas[1] - prev[1])
                row.update(change_deg=max(abs(dr), abs(dl)), d_right_deg=dr, d_left_deg=dl)
                moves = row["change_deg"] > MOVE_EPS_DEG
            else:
                dy = math.degrees(k.yaw - prev)
                row.update(change_deg=abs(dy), d_body_yaw_deg=dy)
                moves = abs(dy) > MOVE_EPS_DEG
            avg = _avg(row["change_deg"], duration)
            row.update(moves=bool(moves), avg_speed_dps=avg, peak_speed_dps=_peak(avg, easing))
            rows.append(row)
            prev = k.value
    return rows


def state_difference(a: RobotState, b: RobotState) -> dict[str, float]:
    """How far apart two robot states are: head rotation (deg, geodesic), head translation (mm),
    the larger antenna difference (deg) and the body yaw difference (deg)."""
    return {"head_rotation_deg": _rotation_deg(a.head, b.head), "head_translation_mm": _translation_mm(a.head, b.head),
            "antennas_max_deg": _antenna_change_deg(a.antennas, b.antennas),
            "body_yaw_deg": float(math.degrees(abs(b.body_yaw - a.body_yaw)))}


def _same(diff: dict[str, float], tol_deg: float, tol_mm: float) -> bool:
    return (diff["head_rotation_deg"] <= tol_deg and diff["antennas_max_deg"] <= tol_deg
            and diff["body_yaw_deg"] <= tol_deg and diff["head_translation_mm"] <= tol_mm)


# ====================================================================== neutral conformance
def _uniformity_issue(rows: list[dict[str, Any]]) -> Issue | None:
    """``neutral_speed_not_uniform`` from the moving head rows (see :func:`neutral_conformance`)."""
    moving = [r for r in rows if r["track"] == "head" and r["moves"] and r["avg_speed_dps"] is not None
              and r["avg_trans_speed_mms"] is not None]
    if len(moving) < 2:
        return None

    def rot(r: dict[str, Any]) -> float:  # angular speed of a key that rotates, else 0
        return float(r["avg_speed_dps"]) if r["change_deg"] > MOVE_EPS_DEG else 0.0

    def tra(r: dict[str, Any]) -> float:  # translation speed of a key that translates, else 0
        return float(r["avg_trans_speed_mms"]) if r["translation_mm"] > MOVE_EPS_MM else 0.0

    w_ref = max(rot(r) for r in moving)  # a uniform retiming moves the binding channel of every key at
    v_ref = max(tra(r) for r in moving)  # one speed, so that speed is the fastest key's

    def pace(r: dict[str, Any]) -> float:
        return max(rot(r) / w_ref if w_ref > 0.0 else 0.0, tra(r) / v_ref if v_ref > 0.0 else 0.0)

    paces = [pace(r) for r in moving]
    med = statistics.median(paces)
    off = [(r, p) for r, p in zip(moving, paces) if abs(p - med) > SPEED_UNIFORMITY_TOL * med]
    if not off:
        return None
    name = [_key_name("head", r["index"], r["label"]) for r, _ in off]
    if v_ref == 0.0:  # no key translates: the plain angular-speed rule
        listed = ", ".join(f"{n} {r['avg_speed_dps']:.3g} deg/s" for n, (r, _) in zip(name, off))
        what = f"head angular speeds not uniform: {listed} vs median {med * w_ref:.3g} deg/s"
    elif w_ref == 0.0:  # no key rotates
        listed = ", ".join(f"{n} {r['avg_trans_speed_mms']:.3g} mm/s" for n, (r, _) in zip(name, off))
        what = f"head translation speeds not uniform: {listed} vs median {med * v_ref:.3g} mm/s"
    else:
        listed = ", ".join(f"{n} {r['avg_speed_dps']:.3g} deg/s, {r['avg_trans_speed_mms']:.3g} mm/s (pace {p:.0%})"
                           for n, (r, p) in zip(name, off))
        what = (f"head speeds not uniform: {listed} vs median pace {med:.0%}; pace = the larger of angular speed / "
                f"{w_ref:.3g} deg/s and translation speed / {v_ref:.3g} mm/s (the fastest keys)")
    return Issue("neutral_speed_not_uniform",
                 f"{what} (> {SPEED_UNIFORMITY_TOL:.0%}); see retime_uniform_speed", "info")


def neutral_conformance(spec: MotionSpec) -> list[Issue]:
    """Design rules of a Neutral variant, checked on ``spec.effective()`` ([] for any other variant).

    * ``neutral_antennas`` (warning): an antenna key differs from the start antennas by > 1e-9 rad (exact:
      a key at -10/+10 deg is 0.0019 deg from the SDK rest angle and counts; leave the antenna track empty or
      write ``rest`` in the motion file).
    * ``neutral_sound`` (warning): a sound cue is set.
    * ``neutral_not_constant_speed`` (warning): a head or body key uses an easing other than linear.
    * ``neutral_speed_not_uniform`` (info): the moving head keys do not share one pace. A key's pace is the
      larger of its average angular speed / the fastest key's and its average translation speed / the fastest
      key's - the pace :func:`retime_uniform_speed` gives every key -, and a key is reported when its pace differs
      from the median pace by more than 15 %. Without translation this is exactly "average angular speed differs
      from the median of the rotating keys by more than 15 %"; with translation, keys whose duration is set by
      their translation are not reported for rotating slower.
    """
    if spec.variant != "neutral":
        return []
    eff = spec.effective()
    issues: list[Issue] = []

    s0 = eff.start.antennas
    moving = [(i, k) for i, k in enumerate(eff.antennas)
              if max(abs(k.antennas[0] - s0[0]), abs(k.antennas[1] - s0[1])) > NEUTRAL_ANTENNA_EPS_RAD]
    if moving:
        worst = max(_antenna_change_deg(s0, k.antennas) for _, k in moving)
        hint = (" (below one motor tick: write the start angles exactly - rest for the SDK rest angle, which is not "
                "-10/+10 deg - or leave the antenna track empty)" if worst < SAME_POSE_TOL_DEG else "")
        issues.append(Issue("neutral_antennas",
                            f"antenna track moves ({', '.join(_key_name('antennas', i, k.label) for i, k in moving)}; "
                            f"up to {worst:.4g} deg from the start); a neutral variant keeps the antennas still{hint}",
                            "warning"))
    if eff.sound is not None:
        issues.append(Issue("neutral_sound", f"sound cue {eff.sound.file!r} at {eff.sound.at_s:.3g} s; a neutral "
                                             f"variant has no sound effect", "warning"))
    curved = [f"{_key_name(t, i, k.label)} {_easing_name(k.easing)}" for t in ("head", "body")
              for i, k in enumerate(eff.track(t)) if _easing_name(k.easing) != Easing.LINEAR.value]
    if curved:
        issues.append(Issue("neutral_not_constant_speed", f"non-linear easing on {', '.join(curved)}; a neutral "
                                                          f"variant moves at constant speed (linear)", "warning"))
    uniform = _uniformity_issue(segment_speeds(eff))
    if uniform is not None:
        issues.append(uniform)
    return issues


# ====================================================================== comparison conditions
def _yaw_path_issues(spec: MotionSpec) -> list[Issue]:
    """Head keys whose SDK yaw path is not what the key values suggest (effective keys).

    ``yaw_wraps`` (warning): a key yaw outside (-180, 180] deg - the SDK reads the Euler yaw of the pose matrix,
    so it turns by a different angle than the designed yaw change (often the other way round).
    ``yaw_long_way`` (info): the yaw changes by more than 180 deg as designed - the SDK turns the head through
    the front, not the shorter way through the back; the speeds count that path.
    """
    eff = spec.effective()
    wraps: list[str] = []
    long_way: list[str] = []
    prev = eff.start.head
    for i, k in enumerate(eff.head):
        _path, dpsi = _head_path(prev, k.pose)
        designed = math.degrees(k.pose.yaw - prev.yaw)
        name = _key_name("head", i, k.label)
        if abs(dpsi - designed) > 1e-6:
            wraps.append(f"{name} designed {designed:+.1f} deg, SDK {dpsi:+.1f} deg")
        elif abs(dpsi) > 180.0:
            long_way.append(f"{name} {dpsi:+.1f} deg")
        prev = k.pose
    issues = []
    if wraps:
        issues.append(Issue("yaw_wraps", f"{spec.key}: the SDK interpolates the Euler yaw of each pose, in (-180, 180] "
                                         f"deg, so these keys turn differently from their yaw values: "
                                         f"{'; '.join(wraps)}; keep head yaw within +-180 deg", "warning"))
    if long_way:
        issues.append(Issue("yaw_long_way", f"{spec.key}: yaw changes by more than 180 deg ({', '.join(long_way)}); "
                                            f"the SDK turns the head through the front, the long way round, and "
                                            f"the speeds count that path", "info"))
    return issues


def _issues(expressive: MotionSpec, neutral: MotionSpec, start_diff: dict[str, float], end_diff: dict[str, float],
            tol_deg: float, tol_mm: float) -> list[Issue]:
    issues: list[Issue] = []
    if expressive.name != neutral.name:
        issues.append(Issue("name_mismatch", f"comparing different motions: {expressive.key} vs {neutral.key}"))
    if expressive.variant != "expressive" or neutral.variant != "neutral":
        issues.append(Issue("variant_mismatch", f"expected an expressive and a neutral variant, got "
                                                f"{expressive.key} and {neutral.key}"))
    if not _same(start_diff, tol_deg, tol_mm):
        issues.append(Issue("start_mismatch", f"start states differ: {_fmt_diff(start_diff)} "
                                              f"(tolerance {tol_deg} deg, {tol_mm} mm)"))
    if not _same(end_diff, tol_deg, tol_mm):
        issues.append(Issue("end_mismatch", f"end states differ: {_fmt_diff(end_diff)} "
                                            f"(tolerance {tol_deg} deg, {tol_mm} mm)"))
    issues.extend(neutral_conformance(neutral))
    issues.extend(_yaw_path_issues(expressive))
    issues.extend(_yaw_path_issues(neutral))
    return issues


def comparison_issues(expressive: MotionSpec, neutral: MotionSpec, *, tol_deg: float = SAME_POSE_TOL_DEG,
                      tol_mm: float = SAME_POSE_TOL_MM) -> list[Issue]:
    """The findings of :func:`comparison_conditions` as ``Issue`` objects (same order as its ``issues``).

    ``name_mismatch`` / ``variant_mismatch`` / ``start_mismatch`` / ``end_mismatch`` (warnings), then
    ``neutral_conformance(neutral)``, then the yaw-path notes of both variants (``yaw_wraps`` warning,
    ``yaw_long_way`` info).
    """
    start_diff = state_difference(expressive.start, neutral.start)
    end_diff = state_difference(expressive.end_state, neutral.end_state)
    return _issues(expressive, neutral, start_diff, end_diff, tol_deg, tol_mm)


def _speed_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    def agg(track: str, change: str, avg: str, peak: str, eps: float) -> dict[str, float]:
        moving = [r for r in rows if r["track"] == track and r[change] is not None and r[change] > eps]
        travel = float(sum(r["duration_s"] for r in moving))
        return {
            "moving_keys": len(moving),
            "travel_time_s": travel,
            "overall_avg": float(sum(r[change] for r in moving)) / travel if travel > 0.0 else 0.0,
            "max_key_avg": max((r[avg] for r in moving if r[avg] is not None), default=0.0),
            "max_nominal_peak": max((r[peak] for r in moving if r[peak] is not None), default=0.0),
        }

    return {
        "head_rotation_dps": agg("head", "change_deg", "avg_speed_dps", "peak_speed_dps", MOVE_EPS_DEG),
        "head_translation_mms": agg("head", "translation_mm", "avg_trans_speed_mms", "peak_trans_speed_mms",
                                    MOVE_EPS_MM),
        "antennas_dps": agg("antennas", "change_deg", "avg_speed_dps", "peak_speed_dps", MOVE_EPS_DEG),
        "body_dps": agg("body", "change_deg", "avg_speed_dps", "peak_speed_dps", MOVE_EPS_DEG),
    }


def _summary(spec: MotionSpec) -> dict[str, Any]:
    eff = spec.effective()
    s0 = eff.start
    return {
        "motion": spec.key,
        "spec_digest": spec.digest(),
        "draft": bool(spec.draft),
        "scale": {"amplitude": float(spec.amplitude_scale), "tempo": float(spec.tempo_scale)},
        "start_state": _state_human(s0),
        "end_state": _state_human(spec.end_state),
        "duration_s": float(eff.duration),
        "track_durations_s": {t: float(eff.track_duration(t)) for t in TRACKS},
        "key_counts": {t: len(eff.track(t)) for t in TRACKS},
        "antennas_used": any(max(abs(k.antennas[0] - s0.antennas[0]), abs(k.antennas[1] - s0.antennas[1]))
                             > NEUTRAL_ANTENNA_EPS_RAD for k in eff.antennas),
        "easings": {t: _unique([_easing_name(k.easing) for k in eff.track(t)]) for t in TRACKS},
        "max_head_rotation_deg": max((_rotation_deg(s0.head, k.pose) for k in eff.head), default=0.0),
        "max_head_translation_mm": max((_translation_mm(s0.head, k.pose) for k in eff.head), default=0.0),
        "max_antenna_change_deg": max((_antenna_change_deg(s0.antennas, k.antennas) for k in eff.antennas),
                                      default=0.0),
        "max_body_yaw_change_deg": max((float(math.degrees(abs(k.yaw - s0.body_yaw))) for k in eff.body),
                                       default=0.0),
        "speeds": _speed_summary(segment_speeds(eff)),
        "sound": None if eff.sound is None else {"file": str(eff.sound.file), "at_s": float(eff.sound.at_s)},
        "principles": [_principle_name(p.principle) for p in spec.principles],
    }


def _fmt_diff(d: dict[str, float]) -> str:
    return (f"head {d['head_rotation_deg']:.3g} deg / {d['head_translation_mm']:.3g} mm, antennas "
            f"{d['antennas_max_deg']:.3g} deg, body {d['body_yaw_deg']:.3g} deg")


def comparison_conditions(expressive: MotionSpec, neutral: MotionSpec, *, tol_deg: float = SAME_POSE_TOL_DEG,
                          tol_mm: float = SAME_POSE_TOL_MM) -> dict[str, Any]:
    """Conditions of the Expressive vs Neutral comparison as one JSON-able dict (pure function).

    Top level: ``data_kind``, ``basis``, ``tolerance``, ``same_start`` / ``start_difference``, ``same_end`` /
    ``end_difference`` (``state_difference`` of the effective start / end states; "same" = every part within
    ``tol_deg`` / ``tol_mm``), then per-variant fields ``{"expressive": ..., "neutral": ...}``: ``motion``,
    ``spec_digest``, ``draft``, ``scale``, ``start_state``, ``end_state``, ``duration_s``,
    ``track_durations_s``, ``key_counts``, ``antennas_used``, ``easings`` (per track, first-use order),
    ``max_head_rotation_deg`` / ``max_head_translation_mm`` / ``max_antenna_change_deg`` /
    ``max_body_yaw_change_deg`` (largest distance of any KEY from the start; cartoon overshoot between keys
    not included), ``speeds`` (per channel: moving keys, travel time, overall average = total path / travel
    time, fastest key average, largest nominal peak), ``sound`` and ``principles``.

    ``issues`` is a list of PLAIN DICTS ``{"code", "message", "severity"}`` (``Issue.to_dict()``, so that the
    result stays JSON-able): read them as ``i["code"]``, or call :func:`comparison_issues` for ``Issue`` objects.
    """
    start_diff = state_difference(expressive.start, neutral.start)
    end_diff = state_difference(expressive.end_state, neutral.end_state)
    issues = _issues(expressive, neutral, start_diff, end_diff, tol_deg, tol_mm)

    sx, sn = _summary(expressive), _summary(neutral)
    out: dict[str, Any] = {
        "data_kind": DATA_KIND,
        "basis": ("design parameters of the effective keys (amplitude/tempo applied); head rotation along the SDK "
                  "goto path (yaw as a scalar); nominal peak = k(easing) x average speed of the SDK time law; task "
                  "space, not joint speeds; not measured"),
        "tolerance": {"deg": float(tol_deg), "mm": float(tol_mm)},
        "same_start": _same(start_diff, tol_deg, tol_mm),
        "start_difference": start_diff,
        "same_end": _same(end_diff, tol_deg, tol_mm),
        "end_difference": end_diff,
    }
    for field in sx:
        out[field] = {"expressive": sx[field], "neutral": sn[field]}
    out["issues"] = [i.to_dict() for i in issues]
    return out


# ====================================================================== retiming
def _retimed(*durations: float) -> float:
    """Longest of the candidate durations and MIN_KEY_DURATION_S, rounded to RETIME_DECIMALS (1 us)."""
    return max(round(max(*durations, MIN_KEY_DURATION_S), RETIME_DECIMALS), MIN_KEY_DURATION_S)


def _positive(value: float, name: str) -> float:
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    return v


def median_translation_speed_mms(spec: MotionSpec) -> float | None:
    """Median average translation speed (mm/s) of the DESIGNED head keys that translate (> MOVE_EPS_MM).

    Keys with a non-positive or non-finite duration are skipped; ``None`` when no key qualifies. This is the
    translation speed :func:`retime_uniform_speed` uses when it is not given one.
    """
    speeds: list[float] = []
    prev = spec.start.head
    for k in spec.head:
        trans, duration = _translation_mm(prev, k.pose), float(k.duration)
        if trans > MOVE_EPS_MM and math.isfinite(duration) and duration > 0.0:
            speeds.append(trans / duration)
        prev = k.pose
    return float(statistics.median(speeds)) if speeds else None


def retime_uniform_speed(spec: MotionSpec, rot_speed_dps: float, trans_speed_mms: float | None = None,
                         set_linear: bool = True) -> MotionSpec:
    """Copy of ``spec`` whose moving head and body keys travel at one constant pace (Neutral design aid).

    Head key that moves (rotates > MOVE_EPS_DEG or translates > MOVE_EPS_MM): ``duration = max(rotation_deg /
    rot_speed_dps, translation_mm / translation speed, MIN_KEY_DURATION_S)``, the rotation measured along the SDK
    goto path (module docstring). The translation speed is ``trans_speed_mms`` or, when it is ``None``,
    :func:`median_translation_speed_mms` of the designed keys - so the head never translates faster than the
    design's median key whatever ``rot_speed_dps`` is, and a key's duration does not jump when it rotates a
    little (``ValueError`` if keys translate but none has a positive duration to take a speed from). Body key
    that moves: ``max(|yaw change| / rot_speed_dps, MIN)``. Durations are rounded to ``RETIME_DECIMALS`` (1 us)
    to drop float noise. Non-moving keys keep their duration; holds, labels, the start, the sound cue and the
    whole antenna track are unchanged. ``set_linear`` sets every head and body key to LINEAR easing.

    Works on the DESIGNED keys, not ``effective()``: ``amplitude_scale`` and ``tempo_scale`` still apply on
    top (the executed speed is about speed x amplitude_scale x tempo_scale).
    """
    rot_speed = _positive(rot_speed_dps, "rot_speed_dps")
    trans_speed = (median_translation_speed_mms(spec) if trans_speed_mms is None
                   else _positive(trans_speed_mms, "trans_speed_mms"))

    head = []
    prev = spec.start.head
    for i, k in enumerate(spec.head):
        (rot, _dpsi), trans = _head_path(prev, k.pose), _translation_mm(prev, k.pose)
        duration = k.duration
        translates = trans > MOVE_EPS_MM
        if rot > MOVE_EPS_DEG or translates:
            if translates and trans_speed is None:
                raise ValueError(f"{_key_name('head', i, k.label)} translates {trans:.3g} mm but no head key has a "
                                 f"positive duration to take a translation speed from; pass trans_speed_mms")
            duration = _retimed(rot / rot_speed, trans / trans_speed if translates and trans_speed else 0.0)
        head.append(replace(k, duration=duration, easing=Easing.LINEAR if set_linear else k.easing))
        prev = k.pose

    body = []
    prev_yaw = spec.start.body_yaw
    for k in spec.body:
        change = math.degrees(abs(k.yaw - prev_yaw))
        duration = _retimed(change / rot_speed) if change > MOVE_EPS_DEG else k.duration
        body.append(replace(k, duration=duration, easing=Easing.LINEAR if set_linear else k.easing))
        prev_yaw = k.yaw

    return replace(spec, head=tuple(head), body=tuple(body))
