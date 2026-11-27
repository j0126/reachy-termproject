"""Planner: ``MotionSpec`` -> ``Timeline`` with the SDK's own interpolation arithmetic (data_kind "computed").

What is reproduced (SDK 1.11.0, paths relative to site-packages):

* ``GotoMove.evaluate`` (reachy_mini/motion/goto.py:51-74): ``s = time_trajectory(t / D, method)`` (line 57),
  head = ``linear_pose_interpolation(start, target, s, yaw_as_scalar=True)`` (59-64), antennas and body yaw
  = ``start + (target - start) * s`` (65-72).
* ``time_trajectory`` (reachy_mini/utils/interpolation.py:117-153) raises outside [0, 1] (lines 121-122), so
  tau is clamped to [0, 1] before every call (floating-point rounding at a key's end could exceed 1).
* ``linear_pose_interpolation`` (interpolation.py:59-105): signed-scalar Euler yaw, rotation-vector SLERP of
  the pitch/roll residual (yaw_as_scalar branch, 74-84), linear translation (94-98). The path is not the
  SO(3) geodesic (Phase 1 SDK_ANALYSIS 3.4); see :func:`head_path_angle`.

Each track of a spec (head / antennas / body) is planned on its own as a chain of such gotos. The start of
key i is the previous key's value (key 0: the spec's start value) - the *commanded* value, whereas a daemon
goto starts from the *present* (measured) pose (reachy_mini/daemon/backend/abstract.py:944-977). Per track,
with depart_i / arrive_i taken from the effective spec (``MotionSpec.key_times`` arithmetic)::

    t < 0                          start value
    depart_i <= t < arrive_i       tau = clamp((t - depart_i) / duration_i, 0, 1), s = time_trajectory(tau, method_i),
                                   value = interp(value_{i-1}, value_i, s)        (value_{-1} = start value)
    arrive_i <= t < depart_{i+1}   value_i EXACTLY (hold = pause; no interpolation call)
    t >= arrive_last               value_last EXACTLY (a track that ends early waits for the longest one)
    empty track                    start value

A single-key spec whose three tracks share duration and easing therefore equals the SDK's
``GotoMove(...).evaluate(t)`` bit for bit for every t < D (tests/test_planner.py, V2.1).

Grid: ``t_k = k / rate_hz`` for k = 0..K, ``K = ceil(duration * rate_hz - 1e-9)`` (one more sample in the
pathological case where that leaves the last sample more than 1e-12 s before ``duration``). The last sample
is the exact end state; a daemon goto never evaluates t = D (abstract.py:904-925, SDK_ANALYSIS 3.5).
Timelines last at most ``MAX_DURATION_S`` (1 h), within which float64 keeps that grid as uniform as
``Timeline.validate`` demands. Every timeline returned here passes ``Timeline.validate()``; input that cannot
give one (non-finite start state, overflowing values, too long) raises ValueError instead.
The default rate is the executor tick rate, 50 Hz like the robot's control loop
(reachy_mini/daemon/backend/robot/backend.py:71).

Units: SI inside (s, m, rad, 4x4 head matrices in metres). Degrees / millimetres only at the boundary:
``segment_table`` columns and ``transition_duration`` arguments (``_deg``, ``_mm``; ``_dps`` = deg/s,
``_mms`` = mm/s). Nothing here talks to a daemon or a robot: every output is computed (``DATA_KIND``).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from reachy_mini.utils.interpolation import InterpolationTechnique, linear_pose_interpolation, time_trajectory
from reachy_mini.utils.rotation import Rotation

from reachy_motion.analysis.interpolation import characterize
from reachy_motion.library.pose import RobotState, rotation_distance, translation_distance
from reachy_motion.library.spec import MIN_KEY_DURATION_S, TRACKS, Easing, MotionSpec
from reachy_motion.timeline import Timeline

__all__ = [
    "DATA_KIND", "DEFAULT_RATE_HZ", "DUPLICATE_TOL", "GRID_EPS", "MAX_DURATION_S", "SEGMENT_COLUMNS",
    "concat", "easing_to_sdk", "evaluate", "grid_times", "head_path_angle", "peak_speed_ratio", "plan",
    "plan_transition", "segment_table", "transition_duration",
]

DATA_KIND = "computed"  # CLAUDE.md rule 6: SDK functions evaluated offline, nothing measured or simulated
DEFAULT_RATE_HZ = 50.0  # executor tick rate = robot control loop rate (daemon/backend/robot/backend.py:71)
GRID_EPS = 1e-9  # K = ceil(duration * rate - GRID_EPS): no extra sample for a duration a rounding error past a tick
DUPLICATE_TOL = 1e-12  # concat: second's sample 0 within this of first's last sample is dropped

#: Longest timeline (s) the planner samples or concatenates. ``Timeline.validate`` checks the grid spacing with an
#: ABSOLUTE tolerance of 1e-12 s, but t_k = k / rate_hz is exact only to half a float64 spacing: below 4096 s
#: (2**12) consecutive samples differ from 1 / rate_hz by at most 2**-41 s (about 4.5e-13 s), while past 8192 s
#: the check fails (1.4e-12 s at 9000 s, 50 Hz). One hour keeps every grid time below 4096 s at rates >= 1 Hz;
#: expressive motions last seconds. Numerical limit only, not a robot or safety limit.
MAX_DURATION_S = 3600.0

#: Column order of :func:`segment_table` rows. Every row has every column; the ones that do not apply to the
#: row's track are None (``yaw_deg`` / ``delta_yaw_deg`` are the head yaw on head rows, the body yaw on body rows).
SEGMENT_COLUMNS: tuple[str, ...] = (
    "track", "index", "label", "t_depart_s", "t_arrive_s", "t_release_s", "duration_s", "hold_s", "easing",
    "peak_ratio",
    "x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg", "right_deg", "left_deg",
    "delta_x_mm", "delta_y_mm", "delta_z_mm", "delta_roll_deg", "delta_pitch_deg", "delta_yaw_deg",
    "delta_right_deg", "delta_left_deg",
    "rot_deg", "trans_mm",
    "peak_rot_dps", "peak_rot_path_dps", "peak_trans_mms", "peak_antenna_dps", "peak_body_dps",
)


# ---------------------------------------------------------------------------------------------- easing mapping
def easing_to_sdk(e: Easing | str) -> InterpolationTechnique:
    """Library ``Easing`` -> SDK ``InterpolationTechnique``, 1:1 by value (interpolation.py:108-114)."""
    return InterpolationTechnique(Easing(e).value)


@lru_cache(maxsize=None)
def _v_peak(method: InterpolationTechnique) -> float:
    return float(characterize(method).v_peak)


def peak_speed_ratio(e: Easing | str) -> float:
    """k = peak / average speed of one key with this easing: 1, 1.875, 2, 5.595 (SDK_ANALYSIS 3.2).

    Computed (and cached) by ``analysis.interpolation.characterize`` from the SDK's own ``time_trajectory``.
    A key moving a channel by A in D seconds peaks at k * A / D (CARTOON included: its anticipation and
    overshoot are part of s(tau)).
    """
    return _v_peak(easing_to_sdk(e))


# ---------------------------------------------------------------------------------------------- track machinery
@dataclass(frozen=True)
class _Segment:
    depart: float  # s, travel to the key starts
    arrive: float  # s, key value reached (hold starts)
    release: float  # s, hold ends (= arrive + hold, MotionSpec.key_times arithmetic)
    duration: float  # s, travel time: tau = (t - depart) / duration
    method: InterpolationTechnique


def _segments(keys: Sequence[Any]) -> tuple[_Segment, ...]:
    """Key times with exactly the arithmetic of ``MotionSpec.key_times`` (next key departs at t + (d + hold))."""
    out: list[_Segment] = []
    t = 0.0
    for k in keys:
        out.append(_Segment(t, t + k.duration, t + k.duration + k.hold, k.duration, easing_to_sdk(k.easing)))
        t += k.duration + k.hold
    return tuple(out)


def _locate(segs: Sequence[_Segment], ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per sample: key index (-1 = before t = 0 or empty track) and s while travelling (NaN = exact value)."""
    n = ts.shape[0]
    idx = np.full(n, -1, dtype=np.int64)
    s = np.full(n, np.nan)
    if not segs:
        return idx, s
    departs = [g.depart for g in segs]
    for j in range(n):
        tj = float(ts[j])
        i = bisect_right(departs, tj) - 1
        idx[j] = i
        if i < 0:
            continue
        g = segs[i]
        if tj < g.arrive:
            tau = (tj - g.depart) / g.duration  # = t / D for key 0, as GotoMove.evaluate (goto.py:57)
            # Python floats on purpose: time_trajectory's powers must match the SDK call exactly
            s[j] = time_trajectory(min(max(tau, 0.0), 1.0), g.method)
    return idx, s


def _head_values(mats: Sequence[np.ndarray], idx: np.ndarray, s: np.ndarray) -> np.ndarray:
    """mats[0] = start pose, mats[i + 1] = key i. Travelling samples call the SDK; others copy a matrix exactly."""
    out = np.empty((idx.shape[0], 4, 4))
    for j in range(idx.shape[0]):
        i = int(idx[j])
        if math.isnan(s[j]):
            out[j] = mats[i + 1]
        else:
            out[j] = linear_pose_interpolation(mats[i], mats[i + 1], float(s[j]), yaw_as_scalar=True)
    return out


def _lerp_values(vals: np.ndarray, idx: np.ndarray, s: np.ndarray) -> np.ndarray:
    """vals[0] = start value, vals[i + 1] = key i; ``prev + (key - prev) * s`` exactly as goto.py:65-72."""
    out = vals[idx + 1]  # fancy indexing: a copy
    trav = ~np.isnan(s)
    if np.any(trav):
        i = idx[trav]
        prev, tgt = vals[i], vals[i + 1]
        out[trav] = prev + (tgt - prev) * s[trav].reshape((-1,) + (1,) * (vals.ndim - 1))
    return out


def _evaluate_effective(eff: MotionSpec, ts: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(head, antennas, body, (head_key, antenna_key, body_key)) of an effective spec at times ``ts``."""
    st = eff.start
    idx_h, s_h = _locate(_segments(eff.head), ts)
    head = _head_values([st.head.matrix()] + [k.pose.matrix() for k in eff.head], idx_h, s_h)
    idx_a, s_a = _locate(_segments(eff.antennas), ts)
    ant_vals = np.array([st.antennas] + [k.antennas for k in eff.antennas], dtype=np.float64)
    antennas = _lerp_values(ant_vals, idx_a, s_a)
    idx_b, s_b = _locate(_segments(eff.body), ts)
    body_vals = np.array([st.body_yaw] + [k.yaw for k in eff.body], dtype=np.float64)
    body = _lerp_values(body_vals, idx_b, s_b)
    return head, antennas, body, (idx_h, idx_a, idx_b)


def _end_time(eff: MotionSpec) -> float:
    """Latest arrive time over all tracks: from here on every track holds its last value exactly."""
    arrives = [_segments(eff.track(t))[-1].arrive for t in TRACKS if eff.track(t)]
    return max(arrives, default=0.0)


def _state_finite(state: RobotState) -> bool:
    """True when every channel of ``state`` (head pose, antennas, body yaw) is a finite number."""
    return all(math.isfinite(float(v)) for v in (*state.head.as_array(), *state.antennas, state.body_yaw))


def _check_state(state: RobotState, label: str) -> None:
    """ValueError for a state with a NaN / inf channel (e.g. a corrupted present-state read)."""
    if not _state_finite(state):
        raise ValueError(f"{label} state has a non-finite value")


def _checked(tl: Timeline, context: str) -> Timeline:
    """``tl`` if it passes ``Timeline.validate()``, else ValueError: nothing invalid leaves the planner.

    Inputs are checked before planning, so this only catches what those checks cannot see, e.g. finite but huge
    values whose difference overflows (antenna keys at +-1e308). The timeline is what executors send.
    """
    problems = tl.validate()
    if problems:
        raise ValueError(f"{context}: planned timeline is invalid: " + "; ".join(problems))
    return tl


def _numeric_problems(spec: MotionSpec) -> list[str]:
    """What would make the arithmetic itself fail (metadata such as name / intent is not checked here)."""
    problems: list[str] = []
    if not (math.isfinite(spec.tempo_scale) and spec.tempo_scale > 0.0):
        problems.append(f"tempo_scale {spec.tempo_scale} is not > 0")
    if not math.isfinite(spec.amplitude_scale):
        problems.append(f"amplitude_scale {spec.amplitude_scale} is not finite")
    if not _state_finite(spec.start):
        problems.append("start state has a non-finite value")
    for t in TRACKS:
        for i, k in enumerate(spec.track(t)):
            where = f"{t}[{i}]"
            try:
                easing_to_sdk(k.easing)
            except ValueError:
                problems.append(f"{where}: unknown easing {k.easing!r}")
            if not (math.isfinite(k.duration) and k.duration > 0.0):
                problems.append(f"{where}: duration {k.duration} is not > 0")
            if not (math.isfinite(k.hold) and k.hold >= 0.0):
                problems.append(f"{where}: hold {k.hold} < 0")
            vals = (k.value.as_array().tolist() if t == "head" else list(k.value) if t == "antennas" else [k.value])
            if not all(math.isfinite(float(v)) for v in vals):
                problems.append(f"{where}: non-finite value")
    return problems


def _effective_checked(spec: MotionSpec) -> MotionSpec:
    problems = _numeric_problems(spec)
    if problems:
        raise ValueError(f"cannot evaluate {spec.key}: " + "; ".join(problems))
    return spec.effective()


def _check_rate(rate_hz: float) -> float:
    rate = float(rate_hz)
    if not (math.isfinite(rate) and rate > 0.0):
        raise ValueError(f"rate_hz must be > 0, got {rate_hz}")
    return rate


def _as_times(t: float | np.ndarray) -> np.ndarray:
    ts = np.atleast_1d(np.asarray(t, dtype=np.float64)).ravel()
    if not np.all(np.isfinite(ts)):
        raise ValueError("times must be finite")
    return ts


def grid_times(duration: float, rate_hz: float) -> np.ndarray:
    """``k / rate_hz`` for k = 0..K, K = ceil(duration * rate_hz - GRID_EPS), so that t[-1] >= duration - 1e-12.

    ``GRID_EPS`` keeps a duration that lands on a tick up to rounding (e.g. 0.1 + 0.2) from getting an extra
    sample. If that leaves the last tick more than 1e-12 s before ``duration`` (a duration a hair past a tick,
    e.g. 1.0 + 1e-11 s at 50 Hz) one more sample is added, so ``Timeline.validate`` always holds.
    Raises ValueError unless 0 < duration <= ``MAX_DURATION_S`` (beyond it float64 cannot keep the grid uniform
    to the 1e-12 s that ``Timeline.validate`` checks).
    """
    rate = _check_rate(rate_hz)
    d = float(duration)
    if not (math.isfinite(d) and d > 0.0):
        raise ValueError(f"duration must be > 0, got {duration}")
    if d > MAX_DURATION_S:
        raise ValueError(f"duration {d} s exceeds MAX_DURATION_S = {MAX_DURATION_S} s (float64 grid limit)")
    k_last = max(0, math.ceil(d * rate - GRID_EPS))
    if k_last / rate + 1e-12 < d:
        k_last += 1
    return np.arange(k_last + 1, dtype=np.int64) / rate


# ---------------------------------------------------------------------------------------------- public API
def evaluate(spec: MotionSpec, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(head (N, 4, 4), antennas (N, 2) rad [right, left], body yaw (N,) rad) at arbitrary times ``t`` (s).

    Evaluates ``spec.effective()`` (amplitude / tempo knobs applied) with the per-track semantics of the module
    docstring. ``t`` may be a scalar (N = 1) or an array (flattened). Raises ValueError for non-finite times or
    a spec whose numbers cannot be evaluated (non-positive duration, negative hold, non-finite value, ...);
    metadata (name, intent, ...) is not checked here - :func:`plan` checks ``spec.validate()``.
    """
    eff = _effective_checked(spec)
    head, antennas, body, _keys = _evaluate_effective(eff, _as_times(t))
    return head, antennas, body


def plan(spec: MotionSpec, rate_hz: float = DEFAULT_RATE_HZ) -> Timeline:
    """Sample a valid spec on the uniform grid ``t_k = k / rate_hz`` (see :func:`grid_times`).

    Raises ValueError listing ``spec.validate()`` problems, plus a non-finite start state (``MotionSpec.validate``
    does not look at ``spec.start``) and a duration over ``MAX_DURATION_S``. The returned timeline always passes
    ``Timeline.validate()`` (else ValueError, e.g. when finite but huge key values overflow).
    ``Timeline.duration`` is ``spec.duration`` (the effective length, holds included); the last sample is the
    exact end state (evaluated at max(t_K, latest arrive) so that a duration a rounding error past t_K cannot
    leave it a hair short). Key index arrays: index of the key being travelled to or held, the last index after
    the last key, -1 for an empty track. ``spec_digest = spec.digest()``.
    """
    problems = spec.validate()
    if not _state_finite(spec.start):  # MotionSpec.validate (library/spec.py) never checks the start state
        problems.append("start state has a non-finite value")
    if not problems and spec.duration > MAX_DURATION_S:
        problems.append(f"duration {spec.duration} s exceeds MAX_DURATION_S = {MAX_DURATION_S} s")
    if problems:
        raise ValueError(f"cannot plan {spec.key}: " + "; ".join(problems))
    rate = _check_rate(rate_hz)
    eff = spec.effective()
    duration = float(spec.duration)
    t = grid_times(duration, rate)
    t_eval = t.copy()
    t_eval[-1] = max(float(t_eval[-1]), _end_time(eff))
    head, antennas, body, (kh, ka, kb) = _evaluate_effective(eff, t_eval)
    return _checked(Timeline(name=spec.name, variant=spec.variant, rate_hz=rate, duration=duration, t=t, head=head,
                             antennas=antennas, body_yaw=body, head_key=kh, antenna_key=ka, body_key=kb,
                             spec_digest=spec.digest()), f"cannot plan {spec.key}")


def plan_transition(start: RobotState, end: RobotState, duration: float, easing: Easing | str = Easing.MIN_JERK,
                    rate_hz: float = DEFAULT_RATE_HZ, name: str = "transition") -> Timeline:
    """Every channel from ``start`` to ``end`` together, like one SDK goto (goto.py:51-74), on the planner grid.

    Samples with t < duration equal ``GotoMove(start, end, duration, method).evaluate(t)``; the last sample is
    ``end`` exactly. ``duration`` must be >= MIN_KEY_DURATION_S (one 50 Hz tick) and <= ``MAX_DURATION_S``, both
    states finite (else ValueError). Key indices are 0 (one key per channel), variant "" and spec_digest ""
    (ad-hoc timeline).
    """
    d = float(duration)
    if not (math.isfinite(d) and d >= MIN_KEY_DURATION_S):
        raise ValueError(f"transition duration {duration} < {MIN_KEY_DURATION_S} s")
    _check_state(start, "start")
    _check_state(end, "end")
    rate = _check_rate(rate_hz)
    seg = (_Segment(0.0, d, d, d, easing_to_sdk(easing)),)
    t = grid_times(d, rate)
    t_eval = t.copy()
    t_eval[-1] = max(float(t_eval[-1]), d)
    idx, s = _locate(seg, t_eval)
    head = _head_values([start.head.matrix(), end.head.matrix()], idx, s)
    antennas = _lerp_values(np.array([start.antennas, end.antennas], dtype=np.float64), idx, s)
    body = _lerp_values(np.array([start.body_yaw, end.body_yaw], dtype=np.float64), idx, s)
    return _checked(Timeline(name=name, variant="", rate_hz=rate, duration=d, t=t, head=head, antennas=antennas,
                             body_yaw=body, head_key=idx.copy(), antenna_key=idx.copy(), body_key=idx.copy(),
                             spec_digest=""), f"cannot plan transition {name!r}")


def head_path_angle(start: np.ndarray, end: np.ndarray) -> float:
    """Angle (rad) the head turns along ``linear_pose_interpolation(start, end, s, yaw_as_scalar=True)`` per unit s.

    That path (interpolation.py:74-84) is R(s) = Rz(psi0 + s * dpsi) @ Q0 @ Exp(s * r), with psi = Euler yaw,
    Q0 the yaw-free residual of the start and r the residual rotation vector. Its angular velocity is
    s' * (dpsi * e_z + R(s) r), and since Rz keeps z components and Exp(s r) r = r, the norm
    |dpsi * e_z + Q0 r| is the same for every s (also for CARTOON's extrapolated s). So the head's angular speed
    is exactly this value * |s'(tau)| / D, and the angle swept for s in [0, 1] is this value. It is >= the
    geodesic ``rotation_distance(start, end)``: equal for single-axis moves, +0.14 % to +5.3 % for the combined
    yaw + pitch/roll moves in tests/test_planner.py (SDK_ANALYSIS 3.4: +0.81 %, +4.1 %). Computed with the SDK's
    own Rotation (reachy_mini/utils/rotation.py:68-101 from_matrix with SVD, 104-151 from_euler, 195-228
    as_euler, 230-301 as_rotvec, 304-312 algebra) in the order of interpolation.py:71-84; tests/test_planner.py
    checks it against finite differences of ``linear_pose_interpolation`` itself.
    """
    rot_start = Rotation.from_matrix(np.asarray(start, dtype=np.float64)[:3, :3])
    rot_end = Rotation.from_matrix(np.asarray(end, dtype=np.float64)[:3, :3])
    yaw_start = float(rot_start.as_euler("xyz")[2])
    yaw_end = float(rot_end.as_euler("xyz")[2])
    res_start = Rotation.from_euler("z", -yaw_start) * rot_start
    res_end = Rotation.from_euler("z", -yaw_end) * rot_end
    rotvec_rel = (res_start.inv() * res_end).as_rotvec()
    w = res_start.as_matrix() @ rotvec_rel
    w[2] += yaw_end - yaw_start
    return float(np.linalg.norm(w))


def transition_duration(start: RobotState, end: RobotState, max_rot_dps: float = 60.0, max_trans_mms: float = 50.0,
                        max_antenna_dps: float = 180.0, max_body_dps: float = 45.0, min_s: float = 0.5) -> float:
    """Shortest MIN_JERK duration (s) whose peak speeds k * distance / T (k = 1.875) stay under the caps.

    The caps are CONSERVATIVE DEFAULTS chosen for unhurried repositioning between motions - not measured robot
    limits (nothing was measured; the safety policy, not this function, decides what may run).
    Distances: head rotation in deg, translation in mm (straight line), antennas the larger absolute change of
    the two (deg), body yaw (deg). The head rotation distance is the larger of the geodesic
    ``rotation_distance`` and :func:`head_path_angle`: the SDK's yaw-as-scalar path is longer than the geodesic
    for combined yaw + pitch/roll moves, and with the geodesic alone the peak angular speed would exceed the
    cap by that ratio (5.3 % in a large combined test case). For single-axis moves both are equal, i.e. the
    result is exactly max(min_s, 1.875 * distance / cap) as specified. The result is at least
    ``max(min_s, MIN_KEY_DURATION_S)``. Raises ValueError, like :func:`plan_transition`, for a non-finite start
    or end state (a NaN distance would silently drop out of ``max``) and when a distance overflows to inf.
    """
    caps = {"max_rot_dps": max_rot_dps, "max_trans_mms": max_trans_mms, "max_antenna_dps": max_antenna_dps,
            "max_body_dps": max_body_dps}
    for label, v in caps.items():
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"{label} must be > 0, got {v}")
    if not (math.isfinite(min_s) and min_s >= 0.0):
        raise ValueError(f"min_s must be >= 0, got {min_s}")
    _check_state(start, "start")
    _check_state(end, "end")
    k = peak_speed_ratio(Easing.MIN_JERK)
    a, b = start.head.matrix(), end.head.matrix()
    rot_deg = math.degrees(max(rotation_distance(a, b), head_path_angle(a, b)))
    trans_mm = translation_distance(a, b) * 1000.0
    ant_deg = math.degrees(max(abs(e - s) for s, e in zip(start.antennas, end.antennas)))
    body_deg = math.degrees(abs(end.body_yaw - start.body_yaw))
    d = float(max(min_s, MIN_KEY_DURATION_S, k * rot_deg / max_rot_dps, k * trans_mm / max_trans_mms,
                  k * ant_deg / max_antenna_dps, k * body_deg / max_body_dps))
    if not math.isfinite(d):  # finite states can still be so far apart that a distance overflows
        raise ValueError(f"transition distance overflows (duration {d})")
    return d


def concat(first: Timeline, second: Timeline, name: str | None = None) -> Timeline:
    """``second`` played right after ``first`` on one uniform grid (same rate required, else ValueError).

    Second's sample 0 goes one tick after first's last sample (shift = first.t[-1] + 1 / rate). If it equals
    first's last sample (max abs difference <= DUPLICATE_TOL over head, antennas and body) it is dropped and
    second's sample 1 takes that tick (shift = first.t[-1]). ``t`` is rebuilt as k / rate; ``duration`` =
    shift + second.duration, at most ``MAX_DURATION_S`` (else ValueError). ``Timeline.validate`` lets second's
    duration lie up to 1e-12 s past its last sample; when rounding of that sum then leaves the result's last
    sample more than 1e-12 s before ``duration``, second's last sample (the end state) is repeated once more,
    as :func:`grid_times` adds a sample. Key index arrays are first's followed by second's, each still
    indexing its OWN keys (no offset; e.g. a transition's 0s, then the motion's indices). spec_digest is ""
    (no single spec); variant is kept when both agree, else "". Default name: "<first>+<second>".
    """
    for label, tl in (("first", first), ("second", second)):
        problems = tl.validate()
        if problems:
            raise ValueError(f"{label} timeline is invalid: " + "; ".join(problems))
    if not math.isclose(first.rate_hz, second.rate_hz, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(f"cannot concat timelines with different rates ({first.rate_hz} Hz, {second.rate_hz} Hz)")
    rate = float(first.rate_hz)
    duplicate = (float(np.max(np.abs(second.head[0] - first.head[-1]))) <= DUPLICATE_TOL
                 and float(np.max(np.abs(second.antennas[0] - first.antennas[-1]))) <= DUPLICATE_TOL
                 and abs(float(second.body_yaw[0]) - float(first.body_yaw[-1])) <= DUPLICATE_TOL)
    j0 = 1 if duplicate else 0
    n1 = len(first)
    shift = (n1 - j0) / rate  # grid time of second's sample 0
    duration = shift + float(second.duration)
    if duration > MAX_DURATION_S:
        raise ValueError(f"concatenated duration {duration} s exceeds MAX_DURATION_S = {MAX_DURATION_S} s")
    n = n1 + len(second) - j0
    t = np.arange(n, dtype=np.int64) / rate
    # the comparison of Timeline.validate itself; true only by rounding (second.t[-1] + 1e-12 >= second.duration)
    pad = bool(t[-1] + 1e-12 < duration)
    if pad:
        t = np.arange(n + 1, dtype=np.int64) / rate

    def join(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.concatenate([a, b[j0:], b[-1:]] if pad else [a, b[j0:]])

    return _checked(Timeline(
        name=name if name is not None else f"{first.name}+{second.name}",
        variant=first.variant if first.variant == second.variant else "",
        rate_hz=rate,
        duration=duration,
        t=t,
        head=join(first.head, second.head),
        antennas=join(first.antennas, second.antennas),
        body_yaw=join(first.body_yaw, second.body_yaw),
        head_key=join(first.head_key, second.head_key),
        antenna_key=join(first.antenna_key, second.antenna_key),
        body_key=join(first.body_key, second.body_key),
        spec_digest="",
    ), "cannot concat")


def segment_table(spec: MotionSpec) -> list[dict[str, Any]]:
    """One row per key per track with the EFFECTIVE parameters actually executed (amplitude / tempo applied).

    Columns: ``SEGMENT_COLUMNS`` (non-applicable ones None). Times in s (``MotionSpec.key_times`` arithmetic),
    targets and deltas from the previous value in deg / mm (head: Euler roll / pitch / yaw and x / y / z;
    antennas: right / left; body: yaw), ``peak_ratio`` = k of the easing (:func:`peak_speed_ratio`).
    Nominal peak channel speeds k * |delta| / duration: ``peak_rot_dps`` from the geodesic ``rot_deg``
    (``rotation_distance``), ``peak_trans_mms`` from ``trans_mm``, ``peak_antenna_dps`` the larger of the two
    antennas, ``peak_body_dps``. ``peak_rot_path_dps`` is the exact peak angular speed of the executed SDK head
    path (:func:`head_path_angle`, >= ``peak_rot_dps``). Task-space values only: motor speeds need IK (safety).
    """
    eff = _effective_checked(spec)
    rows: list[dict[str, Any]] = []
    for track in TRACKS:
        keys = eff.track(track)
        prev = eff.start_value(track)
        for i, (k, g) in enumerate(zip(keys, _segments(keys))):
            ratio = peak_speed_ratio(k.easing)
            row: dict[str, Any] = dict.fromkeys(SEGMENT_COLUMNS)
            row.update(track=track, index=i, label=k.label, t_depart_s=g.depart, t_arrive_s=g.arrive,
                       t_release_s=g.release, duration_s=k.duration, hold_s=k.hold, easing=Easing(k.easing).value,
                       peak_ratio=ratio)
            if track == "head":
                row.update(k.pose.to_deg_mm())
                row.update({f"delta_{c}": v for c, v in (k.pose - prev).to_deg_mm().items()})
                a, b = prev.matrix(), k.pose.matrix()
                geodesic = rotation_distance(a, b)
                rot_deg = math.degrees(geodesic)
                path_deg = math.degrees(max(geodesic, head_path_angle(a, b)))
                trans_mm = translation_distance(a, b) * 1000.0
                row.update(rot_deg=rot_deg, trans_mm=trans_mm, peak_rot_dps=ratio * rot_deg / k.duration,
                           peak_rot_path_dps=ratio * path_deg / k.duration,
                           peak_trans_mms=ratio * trans_mm / k.duration)
            elif track == "antennas":
                d_right, d_left = (math.degrees(v - p) for v, p in zip(k.antennas, prev))
                row.update(right_deg=math.degrees(k.antennas[0]), left_deg=math.degrees(k.antennas[1]),
                           delta_right_deg=d_right, delta_left_deg=d_left,
                           peak_antenna_dps=ratio * max(abs(d_right), abs(d_left)) / k.duration)
            else:
                d_yaw = math.degrees(k.yaw - prev)
                row.update(yaw_deg=math.degrees(k.yaw), delta_yaw_deg=d_yaw,
                           peak_body_dps=ratio * abs(d_yaw) / k.duration)
            rows.append(row)
            prev = k.value
    return rows
