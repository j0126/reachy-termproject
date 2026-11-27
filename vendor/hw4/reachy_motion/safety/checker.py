"""Static safety checker: a planned motion against the initial ``SafetyPolicy`` (data_kind "computed").

Everything here is computed offline with the SDK's own functions - nothing is measured or simulated - and a
plan that passes is "within the initial Phase 2 policy", NOT "mechanically safe": the 10 deg joint margin and
the 50 % no-load speed limits are an initial engineering policy (``safety.policy``), and every report carries
``POLICY_DISCLAIMER``.

Model (SDK 1.11.0, paths relative to site-packages):

* Commanded samples come from the planner (``planner.evaluate`` / ``planner.plan``), i.e. the SDK's
  ``time_trajectory`` + ``linear_pose_interpolation(..., yaw_as_scalar=True)`` exactly as ``GotoMove.evaluate``
  (reachy_mini/motion/goto.py:51-74; reachy_mini/utils/interpolation.py:59-105, 117-153). CARTOON's
  anticipation / overshoot are therefore part of the checked samples.
* Joints = IK of every sample with ONE ``AnalyticalKinematics()`` per analysis, the daemon's default engine
  (reachy_mini/daemon/backend/abstract.py:152, 210-213) with its default ``automatic_body_yaw=True``
  (reachy_mini/kinematics/analytical_kinematics.py:33): ``inverse_kinematics_safe`` moves the body joint to keep
  the head within 65 deg of it and the body within 160 deg (analytical_kinematics.py:80-95). The IK is analytical
  and stateless (Phase 1 SDK_ANALYSIS 4.2), so one instance serves every sample. Joint vector order
  ``[body_yaw, stewart_1..6]`` (``analysis.goto_replay.JOINT_NAMES``).
* An unreachable pose gives NaN joints; the daemon then raises (abstract.py:586-588) and the control loop only
  logs a throttled warning and keeps the previous joint target (reachy_mini/daemon/backend/robot/backend.py:253-261)
  - a silent stop. Hence ``ik_nan`` is a violation.
* Joint limits: the SDK URDF (``analysis.limits.urdf_joint_limits``). Motor speeds: ROBOTIS no-load datasheet
  speeds at 5 V (``analysis.limits``: XL330-M288-T 618 deg/s, XC330-M288-T proxy 486 deg/s, XL330-M077-T
  2298 deg/s) - upper bounds, not trackable speeds.
* psi = atan2(-R01, R00) (``library.pose.head_yaw_psi``) is the yaw the automatic body yaw clamps against. It
  wraps at +-180 deg while the clamp does not, so a psi wrap flips the body target by ~230-275 deg in one sample
  without any NaN (Phase 1 SDK_ANALYSIS 4.2): checked by ``psi_wrap``, ``body_step`` and ``stream_step``.

Grids:

* ``check_spec`` checks the DENSE grid (``policy.check_rate_hz``, ``analyze_spec``) and the STREAM timeline the
  executor sends (``planner.plan(spec, policy.stream_rate_hz)``). Per-sample checks (IK, margins, envelopes,
  antenna range, automatic body yaw, psi wrap) run on both; speeds (``np.gradient`` over t), ``body_step`` and the
  informational acceleration on the dense grid; ``stream_step`` (per-tick joint change) on the stream grid.
* ``check_timeline`` checks the given samples only (e.g. a transition + motion composite): every check runs on
  that grid, with ``n_dense = 0`` in the metrics; key-based warnings (LINEAR, CARTOON, draft) need a spec and are
  not produced.

Units: SI inside (s, m, rad); degrees / millimetres only in report messages, metrics and ``PlanAnalysis.rows``
(suffixes ``_deg``, ``_mm``, ``_dps`` = deg/s, ``_mms`` = mm/s, ``_dps2`` = deg/s^2). A ``Violation``'s ``value``
and ``limit`` use the unit of the policy field they are compared with (deg, mm, ratio; rad for
``antenna_range``; deg per tick for ``stream_step``; a sample count for ``ik_nan``).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reachy_motion import planner
from reachy_motion.analysis.goto_replay import JOINT_NAMES, STEWART, ik_joints, new_kinematics
from reachy_motion.analysis.interpolation import CARTOON_C2
from reachy_motion.analysis.limits import ANTENNA_MOTOR, BODY_MOTOR_PROXY, STEWART_MOTOR, urdf_joint_limits
from reachy_motion.library.pose import head_yaw_psi, pose_from_matrix
from reachy_motion.library.spec import TRACKS, Easing, MotionSpec
from reachy_motion.safety.policy import DEFAULT_POLICY, POLICY_DISCLAIMER, SafetyPolicy, SafetyReport, Violation
from reachy_motion.timeline import PreparedMotion, Timeline

__all__ = [
    "DATA_KIND", "METRIC_KEYS", "ROW_COLUMNS", "STEWART_NAMES", "VIOLATION_CODES", "WARNING_CODES", "PlanAnalysis",
    "analyze_spec", "analyze_timeline", "check_spec", "check_timeline", "prepare", "prepare_timeline",
]

DATA_KIND = "computed"  # CLAUDE.md rule 6: SDK functions evaluated offline
STEWART_NAMES: tuple[str, ...] = tuple(JOINT_NAMES[STEWART])
ANTENNA_NAMES: tuple[str, str] = ("antenna_right", "antenna_left")

#: Exact keys of ``SafetyReport.metrics`` (other modules read them; units in the names, ranges are [min, max]).
METRIC_KEYS: tuple[str, ...] = (
    "min_stewart_margin_deg", "min_stewart_margin_joint", "min_stewart_margin_t",
    "peak_stewart_speed_dps", "peak_stewart_speed_ratio", "peak_body_speed_dps", "peak_body_speed_ratio",
    "peak_antenna_speed_dps", "peak_antenna_speed_ratio",
    "max_abs_roll_deg", "max_abs_pitch_deg", "max_abs_rel_yaw_deg", "body_yaw_range_deg",
    "x_range_mm", "y_range_mm", "z_range_mm", "antenna_range_deg",
    "max_auto_body_shift_deg", "max_psi_step_deg", "max_body_step_deg", "max_stream_step_deg",
    "peak_joint_accel_dps2", "duration_s", "n_dense", "n_stream", "ik_nan_samples",
)

#: Violation codes (severity "violation": the plan is refused), in report order.
VIOLATION_CODES: tuple[str, ...] = (
    "ik_nan", "stewart_margin", "stewart_speed", "body_speed", "antenna_speed",
    "envelope_roll", "envelope_pitch", "envelope_rel_yaw", "envelope_body_yaw", "envelope_x", "envelope_y",
    "envelope_z", "antenna_range", "auto_body_shift", "body_step", "psi_wrap", "stream_step",
)
#: Warning codes (severity "warning": reported only), in report order.
WARNING_CODES: tuple[str, ...] = ("velocity_step", "cartoon_overshoot", "draft", "nan")

#: One-sided difference step (s) for the velocity on each side of a LINEAR key boundary.
VELOCITY_STEP_H_S = 1e-6
#: Velocity jumps below this (deg/s or mm/s) are floating-point noise, not a step.
VELOCITY_STEP_TOL = 1e-3
#: Excursions beyond the key values below this (deg or mm) count as none.
OVERSHOOT_TOL = 1e-6
#: tau samples per CARTOON key for the overshoot warning (plus the exact extremes below).
CARTOON_TAU_SAMPLES = 101
#: tau of the CARTOON extremes s_min / s_max: the derivative of the SDK's easeInOutBack phases
#: (reachy_mini/utils/interpolation.py:136-145) vanishes at tau = c2 / (3 (c2 + 1)) and 1 - that
#: (c2 = ``analysis.interpolation.CARTOON_C2``, the SDK constant 1.70158 * 1.525).
CARTOON_TAU_EXTREMES: tuple[float, float] = (CARTOON_C2 / (3.0 * (CARTOON_C2 + 1.0)),
                                             1.0 - CARTOON_C2 / (3.0 * (CARTOON_C2 + 1.0)))
_MAX_LISTED = 6  # items named in one warning message
_HEAD_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")


def _row_columns() -> tuple[str, ...]:
    cols = ["t_s", "x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg",
            "x_speed_mms", "y_speed_mms", "z_speed_mms", "roll_rate_dps", "pitch_rate_dps", "yaw_rate_dps",
            "antenna_right_deg", "antenna_left_deg", "antenna_right_speed_dps", "antenna_left_speed_dps",
            "body_cmd_yaw_deg", "psi_deg", "rel_yaw_deg", "auto_body_shift_deg", "ik_ok"]
    cols += [f"ik_{n}_deg" for n in JOINT_NAMES]
    cols += [f"ik_{n}_speed_dps" for n in JOINT_NAMES]
    cols += [f"ik_{n}_accel_dps2" for n in JOINT_NAMES]
    cols += [f"{n}_margin_deg" for n in STEWART_NAMES]
    cols += [f"{n}_speed_ratio" for n in STEWART_NAMES]
    cols += ["body_yaw_speed_ratio", "antenna_right_speed_ratio", "antenna_left_speed_ratio"]
    return tuple(cols)


#: Column order of :meth:`PlanAnalysis.rows` (human units; ``ik_ok`` is a bool).
ROW_COLUMNS: tuple[str, ...] = _row_columns()


# ---------------------------------------------------------------------------------------------- analysis
@dataclass(frozen=True)
class PlanAnalysis:
    """Per-sample arrays of one planned motion for GUI plots and the policy checks (SI units, computed).

    ``pose6`` comes from the head matrices (``library.pose.pose_from_matrix``) with roll and yaw unwrapped along
    t (continuous; equal to the Euler angles modulo 2 pi). ``psi`` is NOT unwrapped (the IK sees the raw value).
    Velocities are ``np.gradient`` over ``t`` (central differences; NaN next to IK-NaN samples).
    ``stewart_margin`` (deg) is the distance to the nearer URDF limit (negative = outside); the speed ratios are
    |speed| / no-load speed at 5 V of the joint's motor (body: IK body joint vs the XC330-M288-T proxy).
    """

    name: str
    variant: str
    rate_hz: float  # grid rate: t[k] = k / rate_hz
    source: str  # "spec" (analyze_spec) | "timeline" (analyze_timeline)
    t: np.ndarray  # (N,) s
    head: np.ndarray  # (N, 4, 4) commanded head pose (m)
    pose6: np.ndarray  # (N, 6) [x, y, z, roll, pitch, yaw] m / rad
    antennas: np.ndarray  # (N, 2) rad [right, left]
    body: np.ndarray  # (N,) rad, commanded body yaw
    joints: np.ndarray  # (N, 7) rad IK [body_yaw, stewart_1..6] (NaN = no IK solution)
    joint_velocity: np.ndarray  # (N, 7) rad/s
    joint_acceleration: np.ndarray  # (N, 7) rad/s^2 (informational; LINEAR ends show as spikes)
    pose_velocity: np.ndarray  # (N, 6) m/s, rad/s
    antenna_velocity: np.ndarray  # (N, 2) rad/s
    psi: np.ndarray  # (N,) rad, atan2(-R01, R00) of each head matrix
    rel_yaw: np.ndarray  # (N,) rad, Euler yaw - commanded body yaw, wrapped to [-pi, pi)
    auto_body_shift: np.ndarray  # (N,) rad, IK body joint - commanded body yaw
    stewart_margin: np.ndarray  # (N, 6) deg to the URDF limits
    stewart_speed_ratio: np.ndarray  # (N, 6) |speed| / 618 deg/s
    body_speed_ratio: np.ndarray  # (N,) |IK body joint speed| / 486 deg/s
    antenna_speed_ratio: np.ndarray  # (N, 2) |speed| / 2298 deg/s
    spec_digest: str = ""
    joint_names: tuple[str, ...] = JOINT_NAMES
    data_kind: str = DATA_KIND

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def ik_ok(self) -> np.ndarray:
        """(N,) bool: the IK of the sample has a solution (every joint finite)."""
        return np.all(np.isfinite(self.joints), axis=1)

    @property
    def stewart_speed_dps(self) -> np.ndarray:
        """(N, 6) |Stewart joint speed| in deg/s."""
        return np.degrees(np.abs(self.joint_velocity[:, STEWART]))

    def row_matrix(self) -> np.ndarray:
        """(N, len(ROW_COLUMNS)) float array in human units (``ik_ok`` as 1.0 / 0.0)."""
        p, v = self.pose6, self.pose_velocity
        cols = [self.t, p[:, 0] * 1e3, p[:, 1] * 1e3, p[:, 2] * 1e3, *np.degrees(p[:, 3:6]).T,
                v[:, 0] * 1e3, v[:, 1] * 1e3, v[:, 2] * 1e3, *np.degrees(v[:, 3:6]).T,
                *np.degrees(self.antennas).T, *np.degrees(self.antenna_velocity).T,
                np.degrees(self.body), np.degrees(self.psi), np.degrees(self.rel_yaw),
                np.degrees(self.auto_body_shift), self.ik_ok.astype(np.float64),
                *np.degrees(self.joints).T, *np.degrees(self.joint_velocity).T,
                *np.degrees(self.joint_acceleration).T, *self.stewart_margin.T, *self.stewart_speed_ratio.T,
                self.body_speed_ratio, *self.antenna_speed_ratio.T]
        return np.column_stack([np.asarray(c, dtype=np.float64) for c in cols])

    def rows(self) -> list[dict[str, Any]]:
        """One CSV-friendly dict per sample, keys = ``ROW_COLUMNS`` (s, mm, deg, mm/s, deg/s, deg/s^2, ratios)."""
        i_ok = ROW_COLUMNS.index("ik_ok")
        out = []
        for r in self.row_matrix():
            row: dict[str, Any] = {c: float(x) for c, x in zip(ROW_COLUMNS, r)}
            row["ik_ok"] = bool(r[i_ok] > 0.5)
            out.append(row)
        return out


def _wrap(a: Any) -> np.ndarray:
    """Angles (rad) wrapped to [-pi, pi)."""
    return (np.asarray(a, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def _gradient(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``np.gradient`` over t along axis 0 (zeros for a single sample)."""
    if y.shape[0] < 2:
        return np.zeros_like(y, dtype=np.float64)
    return np.gradient(y, t, axis=0)


def _analyze(name: str, variant: str, rate_hz: float, source: str, t: np.ndarray, head: np.ndarray,
             antennas: np.ndarray, body: np.ndarray, spec_digest: str) -> PlanAnalysis:
    t = np.asarray(t, dtype=np.float64)
    head = np.asarray(head, dtype=np.float64)
    antennas = np.asarray(antennas, dtype=np.float64)
    body = np.asarray(body, dtype=np.float64)
    n = t.shape[0]
    kin = new_kinematics()  # AnalyticalKinematics(), automatic_body_yaw=True (analytical_kinematics.py:33)
    joints = np.empty((n, len(JOINT_NAMES)))
    pose6 = np.empty((n, 6))
    psi = np.empty(n)
    for i in range(n):
        joints[i] = ik_joints(kin, head[i], float(body[i]))
        pose6[i] = pose_from_matrix(head[i]).as_array()
        psi[i] = head_yaw_psi(head[i])
    pose6[:, 3] = np.unwrap(pose6[:, 3])
    pose6[:, 5] = np.unwrap(pose6[:, 5])
    joint_velocity = _gradient(joints, t)
    antenna_velocity = _gradient(antennas, t)
    limits = urdf_joint_limits()
    lo = np.array([limits[j][0] for j in STEWART_NAMES])
    hi = np.array([limits[j][1] for j in STEWART_NAMES])
    q_st = joints[:, STEWART]
    qd_deg = np.degrees(np.abs(joint_velocity))
    return PlanAnalysis(
        name=name, variant=variant, rate_hz=float(rate_hz), source=source, t=t, head=head, pose6=pose6,
        antennas=antennas, body=body, joints=joints, joint_velocity=joint_velocity,
        joint_acceleration=_gradient(joint_velocity, t), pose_velocity=_gradient(pose6, t),
        antenna_velocity=antenna_velocity, psi=psi, rel_yaw=_wrap(pose6[:, 5] - body),
        auto_body_shift=joints[:, 0] - body, stewart_margin=np.degrees(np.minimum(q_st - lo, hi - q_st)),
        stewart_speed_ratio=qd_deg[:, STEWART] / STEWART_MOTOR.no_load_deg_s_5v,
        body_speed_ratio=qd_deg[:, 0] / BODY_MOTOR_PROXY.no_load_deg_s_5v,
        antenna_speed_ratio=np.degrees(np.abs(antenna_velocity)) / ANTENNA_MOTOR.no_load_deg_s_5v,
        spec_digest=spec_digest,
    )


def _end_time(spec: MotionSpec) -> float:
    """Latest key arrival (s): from here on every track holds its last value (``MotionSpec.key_times``)."""
    return max((spec.key_times(tr)[-1][1] for tr in TRACKS if spec.track(tr)), default=0.0)


def analyze_spec(spec: MotionSpec, rate_hz: float = DEFAULT_POLICY.check_rate_hz) -> PlanAnalysis:
    """Dense analysis of ``spec`` on ``t_k = k / rate_hz``, k = 0..K (``planner.grid_times``: t_K >= duration).

    Samples come from ``planner.evaluate`` (effective spec: amplitude / tempo applied); the last one is the exact
    end state (evaluated at max(t_K, latest key arrival), like ``planner.plan``). Raises ValueError for a spec
    whose numbers cannot be evaluated (the planner's own checks) or a non-positive / too long duration.
    """
    planner.evaluate(spec, np.zeros(1))  # the planner's numeric checks first (ValueError), before spec.duration
    t = planner.grid_times(spec.duration, rate_hz)
    t_eval = t.copy()
    t_eval[-1] = max(float(t_eval[-1]), _end_time(spec))
    head, antennas, body = planner.evaluate(spec, t_eval)
    return _analyze(spec.name, spec.variant, float(rate_hz), "spec", t, head, antennas, body, spec.digest())


def analyze_timeline(timeline: Timeline) -> PlanAnalysis:
    """Analysis of exactly the samples of ``timeline`` (what an executor sends). ValueError if it is invalid."""
    problems = timeline.validate()
    if problems:
        raise ValueError(f"timeline {timeline.name!r} is invalid: " + "; ".join(problems))
    return _analyze(timeline.name, timeline.variant, timeline.rate_hz, "timeline", timeline.t, timeline.head,
                    timeline.antennas, timeline.body_yaw, timeline.spec_digest)


# ---------------------------------------------------------------------------------------------- helpers
def _check_policy(policy: SafetyPolicy) -> None:
    """Refuse a policy the checker cannot apply (a NaN threshold would silently disable its check)."""
    if not isinstance(policy, SafetyPolicy):
        raise TypeError(f"policy must be a SafetyPolicy, got {type(policy).__name__}")
    for key, v in policy.to_dict().items():
        if key in ("name", "notes"):
            continue
        vals = list(v) if isinstance(v, (list, tuple)) else [v]
        if key.endswith("_range_mm") and len(vals) != 2:
            raise ValueError(f"policy {key} must be (low, high) in mm, got {v!r}")
        for x in vals:
            if isinstance(x, bool) or not isinstance(x, numbers.Real) or math.isnan(x):
                raise ValueError(f"policy {key} must be a number, got {v!r}")
    for key in ("check_rate_hz", "stream_rate_hz"):
        r = getattr(policy, key)
        if not (math.isfinite(r) and r > 0.0):
            raise ValueError(f"policy {key} must be > 0, got {r}")


def _nanmax(x: Any, empty: float = math.nan) -> float:
    """Max ignoring NaN; ``empty`` for an empty input, NaN when every value is NaN."""
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        return float(empty)
    a = a[~np.isnan(a)]
    return float(a.max()) if a.size else math.nan


def _nanmin(x: Any) -> float:
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[~np.isnan(a)]
    return float(a.min()) if a.size else math.nan


def _row_extreme(x: np.ndarray, highest: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per row of (N, k): (max or min ignoring NaN, column index); NaN where a row has no number."""
    a = np.asarray(x, dtype=np.float64)
    nan = np.isnan(a)
    filled = np.where(nan, -np.inf if highest else np.inf, a)
    idx = filled.argmax(axis=1) if highest else filled.argmin(axis=1)
    val = filled[np.arange(a.shape[0]), idx]
    return np.where(nan.all(axis=1), np.nan, val), idx


@dataclass
class _Acc:
    """One violation code accumulated over several sample grids: earliest offending time and worst value."""

    code: str
    text: str  # what is checked
    unit: str
    t: float | None = None
    score: float = -math.inf
    value: float | None = None
    limit: float | None = None
    where: str = ""
    grids: list[str] = field(default_factory=list)

    def add(self, grid: str, t: np.ndarray, bad: np.ndarray, score: np.ndarray, value: np.ndarray,
            limit: float | Callable[[int], float], where: Callable[[int], str]) -> None:
        """``bad``: offending samples (NaN never offends); the worst one maximises ``score``.

        ``limit`` / ``where`` are called right away with the worst sample's index.
        """
        idx = np.flatnonzero(np.asarray(bad, dtype=bool))
        if idx.size == 0:
            return
        sc = np.nan_to_num(np.asarray(score, dtype=np.float64)[idx], nan=-np.inf)
        j = int(idx[int(np.argmax(sc))])
        if grid not in self.grids:
            self.grids.append(grid)
        t_first = float(t[idx[0]])
        if self.t is None or t_first < self.t:
            self.t = t_first
        s = float(np.nan_to_num(float(np.asarray(score, dtype=np.float64)[j]), nan=-np.inf))
        if self.value is None or s > self.score:
            self.score = s
            self.value = float(np.asarray(value, dtype=np.float64)[j])
            self.limit = float(limit(j)) if callable(limit) else float(limit)
            self.where = where(j)

    def violation(self) -> Violation | None:
        if self.value is None:
            return None
        msg = (f"{self.text}: {self.value:.4g} {self.unit} {self.where} (policy limit {self.limit:g} {self.unit}; "
               f"first at t = {self.t:.3f} s; {' + '.join(self.grids)})")
        return Violation(self.code, msg, t=self.t, value=self.value, limit=self.limit)


def _new_accumulators() -> dict[str, _Acc]:
    texts = {
        "stewart_margin": ("Stewart joint too close to its URDF limit (margin)", "deg"),
        "stewart_speed": (f"Stewart joint speed / {STEWART_MOTOR.model} no-load speed at 5 V "
                          f"({STEWART_MOTOR.no_load_deg_s_5v:g} deg/s)", "x"),
        "body_speed": (f"IK body joint speed / {BODY_MOTOR_PROXY.model} no-load speed at 5 V "
                       f"({BODY_MOTOR_PROXY.no_load_deg_s_5v:g} deg/s)", "x"),
        "antenna_speed": (f"antenna speed / {ANTENNA_MOTOR.model} no-load speed at 5 V "
                          f"({ANTENNA_MOTOR.no_load_deg_s_5v:g} deg/s)", "x"),
        "envelope_roll": ("|head roll| outside the envelope", "deg"),
        "envelope_pitch": ("|head pitch| outside the envelope", "deg"),
        "envelope_rel_yaw": ("|head Euler yaw - commanded body yaw| outside the envelope", "deg"),
        "envelope_body_yaw": ("|commanded body yaw| outside the envelope", "deg"),
        "envelope_x": ("head x outside the envelope", "mm"),
        "envelope_y": ("head y outside the envelope", "mm"),
        "envelope_z": ("head z outside the envelope", "mm"),
        "antenna_range": ("|antenna angle| outside the policy range", "rad"),
        "auto_body_shift": ("automatic body yaw moves the IK body joint away from the commanded body yaw", "deg"),
        "body_step": ("IK body joint change between consecutive samples", "deg"),
        "psi_wrap": ("psi = atan2(-R01, R00) jumps between consecutive samples (a wrap flips the automatic body "
                     "yaw target)", "deg"),
        "stream_step": ("commanded joint change per executor tick", "deg"),
    }
    return {code: _Acc(code, *texts[code]) for code in VIOLATION_CODES if code != "ik_nan"}


# ---------------------------------------------------------------------------------------------- policy checks
def _policy_checks(policy: SafetyPolicy, grids: Sequence[tuple[str, PlanAnalysis]], speed: tuple[str, PlanAnalysis],
                   step: tuple[str, PlanAnalysis]) -> tuple[list[Violation], dict[str, Any]]:
    """Violations and metrics (without duration_s / n_dense / n_stream).

    ``grids``: every sample grid (per-sample checks); ``speed``: the grid of the speed, ``body_step`` and
    acceleration checks; ``step``: the grid of the per-tick ``stream_step`` check.
    """
    p = policy
    acc = _new_accumulators()
    nan_count, nan_t, nan_grids = 0, math.inf, []
    margin_best: tuple[float, str, float] = (math.nan, "", math.nan)  # (deg, joint, t)
    maxima: dict[str, list[float]] = {"roll": [], "pitch": [], "rel": [], "shift": [], "psi": []}
    values: dict[str, list[np.ndarray]] = {"body": [], "x": [], "y": [], "z": [], "ant": []}

    for label, a in grids:
        t = a.t

        def at(j: int, t: np.ndarray = t) -> str:
            return f"at t = {t[j]:.3f} s"

        # --- IK feasibility
        nan_rows = ~a.ik_ok
        if np.any(nan_rows):
            nan_count += int(nan_rows.sum())
            nan_t = min(nan_t, float(t[np.flatnonzero(nan_rows)[0]]))
            nan_grids.append(label)

        # --- Stewart margin to the URDF limits
        m_row, m_col = _row_extreme(a.stewart_margin, highest=False)
        acc["stewart_margin"].add(label, t, m_row < p.min_stewart_margin_deg, -m_row, m_row, p.min_stewart_margin_deg,
                                  lambda j: f"({STEWART_NAMES[m_col[j]]} at t = {t[j]:.3f} s)")
        if not np.all(np.isnan(m_row)):
            j = int(np.nanargmin(m_row))
            if math.isnan(margin_best[0]) or m_row[j] < margin_best[0]:
                margin_best = (float(m_row[j]), STEWART_NAMES[int(m_col[j])], float(t[j]))

        # --- task-space envelope (commanded values)
        roll, pitch = np.degrees(np.abs(a.pose6[:, 3])), np.degrees(np.abs(a.pose6[:, 4]))
        rel, body = np.degrees(np.abs(a.rel_yaw)), np.degrees(a.body)
        for code, v, lim in (("envelope_roll", roll, p.max_roll_deg), ("envelope_pitch", pitch, p.max_pitch_deg),
                             ("envelope_rel_yaw", rel, p.max_rel_yaw_deg),
                             ("envelope_body_yaw", np.abs(body), p.max_body_yaw_deg)):
            acc[code].add(label, t, v > lim, v, v, lim, at)
        maxima["roll"].append(_nanmax(roll))
        maxima["pitch"].append(_nanmax(pitch))
        maxima["rel"].append(_nanmax(rel))
        values["body"].append(body)
        for c, (axis, rng) in enumerate((("x", p.x_range_mm), ("y", p.y_range_mm), ("z", p.z_range_mm))):
            v = a.pose6[:, c] * 1e3
            lo, hi = float(rng[0]), float(rng[1])
            acc[f"envelope_{axis}"].add(label, t, (v < lo) | (v > hi), np.maximum(v - hi, lo - v), v,
                                        lambda j: hi if v[j] > hi else lo, at)
            values[axis].append(v)

        # --- antennas (rad, the policy field's unit)
        ant_abs = np.max(np.abs(a.antennas), axis=1)
        acc["antenna_range"].add(label, t, ant_abs > p.max_antenna_abs_rad, ant_abs, ant_abs, p.max_antenna_abs_rad, at)
        values["ant"].append(np.degrees(a.antennas).ravel())

        # --- automatic body yaw engaging (IK body joint vs commanded body yaw)
        shift = np.degrees(np.abs(a.auto_body_shift))
        acc["auto_body_shift"].add(label, t, shift > p.max_auto_body_shift_deg, shift, shift,
                                   p.max_auto_body_shift_deg, at)
        maxima["shift"].append(_nanmax(shift))

        # --- psi wrap between consecutive samples (raw psi, as the IK sees it)
        dpsi = np.degrees(np.abs(np.diff(a.psi)))
        psi = a.psi
        acc["psi_wrap"].add(label, t[1:], dpsi > p.max_psi_step_deg, dpsi, dpsi, p.max_psi_step_deg,
                            lambda j: f"(psi {math.degrees(psi[j]):.1f} -> {math.degrees(psi[j + 1]):.1f} deg at "
                                      f"t = {t[j + 1]:.3f} s)")
        maxima["psi"].append(_nanmax(dpsi, 0.0))

    # --- speeds, body steps and accelerations on the speed grid
    s_label, s = speed
    ts = s.t
    st_ratio, st_col = _row_extreme(s.stewart_speed_ratio, highest=True)
    acc["stewart_speed"].add(s_label, ts, st_ratio > p.max_stewart_speed_ratio, st_ratio, st_ratio,
                             p.max_stewart_speed_ratio,
                             lambda j: f"({STEWART_NAMES[st_col[j]]} {st_ratio[j] * STEWART_MOTOR.no_load_deg_s_5v:.0f}"
                                       f" deg/s at t = {ts[j]:.3f} s)")
    b_ratio = s.body_speed_ratio
    acc["body_speed"].add(s_label, ts, b_ratio > p.max_body_speed_ratio, b_ratio, b_ratio, p.max_body_speed_ratio,
                          lambda j: f"({b_ratio[j] * BODY_MOTOR_PROXY.no_load_deg_s_5v:.0f} deg/s at t = {ts[j]:.3f} s)")
    a_ratio, a_col = _row_extreme(s.antenna_speed_ratio, highest=True)
    acc["antenna_speed"].add(s_label, ts, a_ratio > p.max_antenna_speed_ratio, a_ratio, a_ratio,
                             p.max_antenna_speed_ratio,
                             lambda j: f"({ANTENNA_NAMES[a_col[j]]} {a_ratio[j] * ANTENNA_MOTOR.no_load_deg_s_5v:.0f}"
                                       f" deg/s at t = {ts[j]:.3f} s)")
    dbody = np.degrees(np.abs(np.diff(s.joints[:, 0])))
    acc["body_step"].add(s_label, ts[1:], dbody > p.max_body_step_deg, dbody, dbody, p.max_body_step_deg,
                         lambda j: f"(between t = {ts[j]:.3f} and {ts[j + 1]:.3f} s)")

    # --- per-tick commanded joint change on the step (stream) grid: |dq| <= ratio x no-load speed x tick
    k_label, k = step
    kt, dt = k.t, 1.0 / k.rate_hz
    lim_st = p.max_stewart_speed_ratio * STEWART_MOTOR.no_load_deg_s_5v * dt
    lim_b = p.max_body_speed_ratio * BODY_MOTOR_PROXY.no_load_deg_s_5v * dt
    dq = np.degrees(np.abs(np.diff(k.joints, axis=0)))
    st_step, st_step_col = _row_extreme(dq[:, STEWART], highest=True)
    b_step = dq[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):  # a zero limit makes any motion infinitely over it
        rel_st, rel_b = st_step / lim_st, b_step / lim_b
    use_b = np.nan_to_num(rel_b, nan=-np.inf) > np.nan_to_num(rel_st, nan=-np.inf)
    acc["stream_step"].add(
        k_label, kt[1:], (st_step > lim_st) | (b_step > lim_b), np.where(use_b, rel_b, rel_st),
        np.where(use_b, b_step, st_step), lambda j: lim_b if use_b[j] else lim_st,
        lambda j: (f"({'body_yaw' if use_b[j] else STEWART_NAMES[st_step_col[j]]} between t = {kt[j]:.3f} and "
                   f"{kt[j + 1]:.3f} s; limit = speed ratio x no-load speed x {dt * 1e3:.4g} ms tick)"))

    violations: list[Violation] = []
    if nan_count:
        violations.append(Violation(
            "ik_nan", f"IK has no solution (NaN joints) at {nan_count} sample(s), first at t = {nan_t:.3f} s "
                      f"({' + '.join(nan_grids)}): the daemon would keep the previous joint target and the head "
                      f"would silently stop", t=nan_t, value=float(nan_count), limit=0.0))
    for code in VIOLATION_CODES[1:]:
        v = acc[code].violation()
        if v is not None:
            violations.append(v)

    metrics: dict[str, Any] = {
        "min_stewart_margin_deg": margin_best[0],
        "min_stewart_margin_joint": margin_best[1],
        "min_stewart_margin_t": margin_best[2],
        "peak_stewart_speed_dps": _nanmax(s.stewart_speed_dps),
        "peak_stewart_speed_ratio": _nanmax(s.stewart_speed_ratio),
        "peak_body_speed_dps": _nanmax(np.degrees(np.abs(s.joint_velocity[:, 0]))),
        "peak_body_speed_ratio": _nanmax(s.body_speed_ratio),
        "peak_antenna_speed_dps": _nanmax(np.degrees(np.abs(s.antenna_velocity))),
        "peak_antenna_speed_ratio": _nanmax(s.antenna_speed_ratio),
        "max_abs_roll_deg": _nanmax(maxima["roll"]),
        "max_abs_pitch_deg": _nanmax(maxima["pitch"]),
        "max_abs_rel_yaw_deg": _nanmax(maxima["rel"]),
        "body_yaw_range_deg": [_nanmin(np.concatenate(values["body"])), _nanmax(np.concatenate(values["body"]))],
        **{f"{ax}_range_mm": [_nanmin(np.concatenate(values[ax])), _nanmax(np.concatenate(values[ax]))]
           for ax in ("x", "y", "z")},
        "antenna_range_deg": [_nanmin(np.concatenate(values["ant"])), _nanmax(np.concatenate(values["ant"]))],
        "max_auto_body_shift_deg": _nanmax(maxima["shift"]),
        "max_psi_step_deg": _nanmax(maxima["psi"], 0.0),
        "max_body_step_deg": _nanmax(dbody, 0.0),
        "max_stream_step_deg": _nanmax(np.concatenate([st_step, b_step]), 0.0),
        "peak_joint_accel_dps2": _nanmax(np.degrees(np.abs(s.joint_acceleration))),
        "ik_nan_samples": int(nan_count),
    }
    return violations, metrics


# ---------------------------------------------------------------------------------------------- spec warnings
def _velocity_step_warning(spec: MotionSpec, nan_notes: list[str]) -> Violation | None:
    """LINEAR keys start and stop at full speed (s'(0) = s'(1) = 1): velocity steps at their boundaries.

    Measured with one-sided differences of ``planner.evaluate`` (+ IK) at +-``VELOCITY_STEP_H_S`` around every
    departure / arrival of a LINEAR key (t < 0 evaluates to the start state: the robot is at rest before).
    ``value`` = the largest jump of a motor (Stewart / body IK joint or antenna) in deg/s.
    """
    eff = spec.effective()
    times: list[float] = []
    for track in TRACKS:
        for key, (dep, arr, _rel) in zip(eff.track(track), spec.key_times(track)):
            if Easing(key.easing) is Easing.LINEAR:
                times += [dep, arr]
    bounds: list[float] = []
    for x in sorted(times):
        if not bounds or x - bounds[-1] > 1e-12:
            bounds.append(x)
    if not bounds:
        return None
    h = VELOCITY_STEP_H_S
    tb = np.asarray(bounds, dtype=np.float64)
    nb = tb.shape[0]
    head, ant, body = planner.evaluate(spec, np.stack([tb - h, tb, tb + h], axis=1).ravel())
    kin = new_kinematics()
    q = np.array([ik_joints(kin, m, float(b)) for m, b in zip(head, body)]).reshape(nb, 3, len(JOINT_NAMES))
    pose = np.array([pose_from_matrix(m).as_array() for m in head]).reshape(nb, 3, 6)

    def jump(x: np.ndarray, angles: bool) -> np.ndarray:
        """|v_right - v_left| per boundary and channel from (nb, 3, C) samples at tb - h, tb, tb + h."""
        d1, d2 = x[:, 1] - x[:, 0], x[:, 2] - x[:, 1]
        if angles:
            d1, d2 = _wrap(d1), _wrap(d2)
        return np.abs(d2 - d1) / h

    rot = np.degrees(jump(pose[:, :, 3:], True))
    trans = jump(pose[:, :, :3], False) * 1e3
    antj = np.degrees(jump(ant.reshape(nb, 3, 2), False))
    bodyj = np.degrees(jump(body.reshape(nb, 3, 1), False))
    jointj = np.degrees(jump(q, False))
    joint_row, joint_col = _row_extreme(jointj, highest=True)
    ant_row = np.max(antj, axis=1)
    motor = np.fmax(joint_row, ant_row)  # Stewart / body / antenna motors (NaN only if every joint is NaN)
    for i in np.flatnonzero(np.isnan(jointj).any(axis=1)):
        nan_notes.append(f"velocity_step: no IK solution next to the LINEAR key boundary t = {tb[i]:.3f} s "
                         f"(joint-space jump unknown there)")
    task = np.max(np.concatenate([rot, trans, antj, bodyj], axis=1), axis=1)
    stepped = (task > VELOCITY_STEP_TOL) | (np.nan_to_num(motor, nan=0.0) > VELOCITY_STEP_TOL)
    if not np.any(stepped):
        return None
    value = _nanmax(motor)
    w = int(np.nanargmax(motor)) if not math.isnan(value) else -1
    if w < 0:
        where = "joint-space jump unknown: no IK solution"
    elif not math.isnan(joint_row[w]) and joint_row[w] >= ant_row[w]:
        where = f"{JOINT_NAMES[int(joint_col[w])]} at t = {tb[w]:.3f} s"
    else:
        where = f"{ANTENNA_NAMES[int(np.argmax(antj[w]))]} at t = {tb[w]:.3f} s"
    msg = (f"LINEAR keys start and stop at full speed: velocity steps at {int(np.sum(stepped))} key boundary(ies); "
           f"largest motor jump {value:.3g} deg/s ({where}); task space: head rotation {_nanmax(rot, 0.0):.3g} "
           f"deg/s, translation {_nanmax(trans, 0.0):.3g} mm/s, antennas {_nanmax(antj, 0.0):.3g} deg/s, body "
           f"{_nanmax(bodyj, 0.0):.3g} deg/s (MIN_JERK / EASE_IN_OUT / CARTOON start and end at rest)")
    return Violation("velocity_step", msg, t=float(tb[np.flatnonzero(stepped)[0]]),
                     value=None if math.isnan(value) else value, limit=None, severity="warning")


def _cartoon_overshoot_warning(spec: MotionSpec, nan_notes: list[str]) -> Violation | None:
    """How far CARTOON keys go beyond the values they move between (anticipation and overshoot).

    Each CARTOON key is evaluated on its own tau grid (``CARTOON_TAU_SAMPLES`` points plus the exact s extremes
    ``CARTOON_TAU_EXTREMES``); excursion = distance outside [min, max] of its values at tau = 0 and tau = 1. Head
    angles are Euler angles unwrapped along the key's path (deg), translations in mm. ``value`` = the largest
    angular excursion (deg), or the largest translation (mm) when no angle overshoots (the message names units).
    """
    eff = spec.effective()
    taus = np.unique(np.concatenate([np.linspace(0.0, 1.0, CARTOON_TAU_SAMPLES), CARTOON_TAU_EXTREMES]))
    items: list[tuple[float, str, float, float]] = []  # (t_first, text, worst angle deg, worst translation mm)
    for track in TRACKS:
        for i, (key, (dep, _arr, _rel)) in enumerate(zip(eff.track(track), spec.key_times(track))):
            if Easing(key.easing) is not Easing.CARTOON:
                continue
            ts = dep + taus * key.duration
            head, ant, body = planner.evaluate(spec, ts)
            if track == "head":
                pose = np.array([pose_from_matrix(m).as_array() for m in head])
                pose[:, 3] = np.unwrap(pose[:, 3])
                pose[:, 5] = np.unwrap(pose[:, 5])
                vals = np.column_stack([pose[:, :3] * 1e3, np.degrees(pose[:, 3:])])
                names, is_angle = _HEAD_FIELDS, (False, False, False, True, True, True)
            elif track == "antennas":
                vals, names, is_angle = np.degrees(ant), ("right", "left"), (True, True)
            else:
                vals, names, is_angle = np.degrees(body)[:, None], ("yaw",), (True,)
            if np.isnan(vals).any():
                nan_notes.append(f"cartoon_overshoot: non-finite value on {track}[{i}]")
                continue
            lo, hi = np.minimum(vals[0], vals[-1]), np.maximum(vals[0], vals[-1])
            exc = np.maximum(np.maximum(vals - hi, lo - vals), 0.0)  # (samples, channels)
            worst = exc.max(axis=0)
            if not np.any(worst > OVERSHOOT_TOL):
                continue
            ang = max(((float(worst[c]), names[c]) for c in range(len(names)) if is_angle[c]), default=(0.0, ""))
            lin = max(((float(worst[c]), names[c]) for c in range(len(names)) if not is_angle[c]), default=(0.0, ""))
            parts = ([f"{ang[1]} {ang[0]:.3g} deg"] if ang[0] > OVERSHOOT_TOL else []) + \
                    ([f"{lin[1]} {lin[0]:.3g} mm"] if lin[0] > OVERSHOOT_TOL else [])
            label = f" '{key.label}'" if key.label else ""
            first = int(np.flatnonzero(exc.max(axis=1) > OVERSHOOT_TOL)[0])
            items.append((float(ts[first]), f"{track}[{i}]{label} {', '.join(parts)}", ang[0], lin[0]))
    if not items:
        return None
    worst_angle = max(it[2] for it in items)
    value, unit = ((worst_angle, "deg") if worst_angle > OVERSHOOT_TOL else (max(it[3] for it in items), "mm"))
    listed = "; ".join(it[1] for it in items[:_MAX_LISTED])
    more = f"; ... ({len(items) - _MAX_LISTED} more)" if len(items) > _MAX_LISTED else ""
    msg = (f"CARTOON keys go beyond their key values (anticipation before the start value and overshoot past the "
           f"target, about 10 % of each move): {listed}{more}; worst {value:.3g} {unit}. These extrapolated poses "
           f"are part of the checked samples.")
    return Violation("cartoon_overshoot", msg, t=min(it[0] for it in items), value=value, limit=None,
                     severity="warning")


def _spec_warnings(spec: MotionSpec) -> list[Violation]:
    nan_notes: list[str] = []
    out = [w for w in (_velocity_step_warning(spec, nan_notes), _cartoon_overshoot_warning(spec, nan_notes))
           if w is not None]
    if spec.draft:
        out.append(Violation("draft", f"{spec.key} is a DRAFT (spec.draft = True): initial values for the student "
                                      f"to design and tune, not a signed-off motion", severity="warning"))
    if nan_notes:
        out.append(Violation("nan", "NaN in a warning computation: " + "; ".join(nan_notes), severity="warning"))
    return out


# ---------------------------------------------------------------------------------------------- public API
def _report(name: str, variant: str, policy: SafetyPolicy, violations: list[Violation], warnings: list[Violation],
            metrics: dict[str, Any], spec_digest: str, timeline_digest: str) -> SafetyReport:
    mismatch = set(METRIC_KEYS) ^ set(metrics)
    if mismatch:  # pragma: no cover - internal consistency
        raise RuntimeError(f"metrics keys mismatch: {sorted(mismatch)}")
    return SafetyReport(name=name, variant=variant, policy=policy, passes_policy=not violations,
                        violations=tuple(violations), warnings=tuple(warnings),
                        metrics={key: metrics[key] for key in METRIC_KEYS}, spec_digest=spec_digest,
                        timeline_digest=timeline_digest, data_kind=DATA_KIND, disclaimer=POLICY_DISCLAIMER)


def _check_spec(spec: MotionSpec, policy: SafetyPolicy, stream_rate_hz: float) -> tuple[SafetyReport, Timeline]:
    """The report of ``spec`` with the stream checks on ``plan(spec, stream_rate_hz)``; the policy is checked."""
    timeline = planner.plan(spec, stream_rate_hz)  # ValueError for an invalid spec
    dense = analyze_spec(spec, policy.check_rate_hz)
    stream = analyze_timeline(timeline)
    label_d, label_s = f"dense {dense.rate_hz:g} Hz grid", f"stream {stream.rate_hz:g} Hz grid"
    violations, metrics = _policy_checks(policy, [(label_d, dense), (label_s, stream)], (label_d, dense),
                                         (label_s, stream))
    metrics.update(duration_s=float(spec.duration), n_dense=len(dense), n_stream=len(stream))
    report = _report(spec.name, spec.variant, policy, violations, _spec_warnings(spec), metrics, spec.digest(),
                     timeline.digest())
    return report, timeline


def check_spec(spec: MotionSpec, policy: SafetyPolicy = DEFAULT_POLICY) -> SafetyReport:
    """Check ``spec`` against ``policy`` on the dense grid and on the stream timeline (see module docstring).

    ``spec_digest = spec.digest()``, ``timeline_digest = plan(spec, policy.stream_rate_hz).digest()``,
    ``passes_policy`` = no violation. Warnings (never refusing): ``velocity_step`` (LINEAR keys),
    ``cartoon_overshoot``, ``draft``, ``nan`` (a warning computation hit NaN). Raises ValueError for a spec the
    planner refuses (``planner.plan``) or a policy with NaN thresholds / non-positive rates (TypeError for a
    policy that is not a ``SafetyPolicy``).
    """
    _check_policy(policy)
    return _check_spec(spec, policy, policy.stream_rate_hz)[0]


def check_timeline(timeline: Timeline, policy: SafetyPolicy = DEFAULT_POLICY) -> SafetyReport:
    """Check exactly the samples of ``timeline`` (e.g. ``planner.concat`` of a transition and a motion).

    Every check runs on the timeline's own grid (speeds by ``np.gradient``, ``body_step`` per sample,
    ``stream_step`` per tick of 1 / ``timeline.rate_hz``); no dense resampling (``n_dense = 0``) and no key-based
    warnings. ``spec_digest = timeline.spec_digest``, ``timeline_digest = timeline.digest()``. Raises ValueError
    for an invalid timeline (``Timeline.validate``) or policy.
    """
    _check_policy(policy)
    a = analyze_timeline(timeline)
    label = f"timeline {a.rate_hz:g} Hz grid"
    violations, metrics = _policy_checks(policy, [(label, a)], (label, a), (label, a))
    metrics.update(duration_s=float(timeline.duration), n_dense=0, n_stream=len(a))
    return _report(timeline.name, timeline.variant, policy, violations, [], metrics, timeline.spec_digest,
                   timeline.digest())


def prepare(spec: MotionSpec, policy: SafetyPolicy = DEFAULT_POLICY, rate_hz: float | None = None) -> PreparedMotion:
    """``PreparedMotion(plan(spec, rate), report, spec)`` with rate = ``rate_hz`` or ``policy.stream_rate_hz``.

    The report is ``check_spec``'s with the stream checks (per-sample and ``stream_step``) run on the very
    timeline returned, so ``report.timeline_digest == timeline.digest()`` and ``runnable`` is True exactly when
    the policy passes. ValueError as for :func:`check_spec` (and for a non-positive rate).
    """
    _check_policy(policy)
    rate = policy.stream_rate_hz if rate_hz is None else float(rate_hz)
    report, timeline = _check_spec(spec, policy, rate)
    return PreparedMotion(timeline=timeline, report=report, spec=spec)


def prepare_timeline(timeline: Timeline, policy: SafetyPolicy = DEFAULT_POLICY) -> PreparedMotion:
    """``PreparedMotion(timeline, check_timeline(timeline, policy), spec=None)``."""
    return PreparedMotion(timeline=timeline, report=check_timeline(timeline, policy), spec=None)
