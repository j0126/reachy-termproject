"""Tick timing of the streaming executor: per-tick records, period / jitter statistics, CSV export.

The executor's REAL period and jitter must be measurable (Phase 2 requirement, V2.4: tick interval
p95 <= 21 ms on this PC, HW3 robot reference p95 <= 20.7 ms). Every loop iteration that reaches the
send step leaves one ``TickRecord`` with the deadline it slept until, the time it actually woke up
(both in the executor clock's ``perf_counter_ns`` domain, ``executor.clock``), how long the backend
call took and how many samples the loop skipped to catch up. ``TickLog.summary`` turns the records
into the numbers a report needs (effective rate, interval distribution, jitter, lateness, send time).

These are timings of the PC-side loop only. They say nothing about when the daemon wrote the command
to the motors (robot control loop 50 Hz, reachy_mini/daemon/backend/robot/backend.py:71, 200-228) or
about the WiFi path; with a ``VirtualClock`` they are synthetic.

``measure_tick_timing`` streams a constant REST command through the real ``Executor`` for a while and
returns its tick log (CLI ``timing`` command). Its plan is checked here (IK of the constant pose with
the SDK's AnalyticalKinematics, URDF margins) because a constant command has zero speed; real-robot
backends are refused, since streaming REST to a head that is elsewhere is a step command.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from reachy_motion.library.pose import REST
from reachy_motion.safety.policy import DEFAULT_POLICY, SafetyPolicy, SafetyReport, Violation
from reachy_motion.timeline import PreparedMotion, Timeline

if TYPE_CHECKING:  # pragma: no cover
    from reachy_motion.backends.base import Backend
    from reachy_motion.executor.clock import Clock

#: Columns of ``TickLog.to_csv`` (times relative to the first planned tick).
CSV_COLUMNS: tuple[str, ...] = ("tick", "k", "planned_s", "actual_s", "late_ms", "interval_ms", "send_ms",
                                "skipped_before")

INTERVAL_STATS: tuple[str, ...] = ("mean", "std", "min", "p50", "p95", "p99", "max")
LATE_STATS: tuple[str, ...] = ("mean", "p50", "p95", "p99", "max")
SEND_STATS: tuple[str, ...] = ("mean", "p95", "max")

#: A tick waking later than this counts as late (``TickLog.summary``; the executor's default setting).
DEFAULT_LATE_THRESHOLD_MS = 5.0

REST_HOLD_NAME = "timing_rest_hold"


@dataclass(frozen=True)
class TickRecord:
    """One executor tick (integer ns in the executor clock's domain).

    ``planned_ns`` is the deadline the loop slept until; ``actual_ns`` the wake-up time read right
    after the sleep, before the send. When the loop woke a period or more late it skipped
    ``skipped_before`` samples and sent sample ``k`` instead of ``k - skipped_before``, so
    ``planned_ns = t0 + offset(k - skipped_before)`` and ``late_ns`` is the full wake-up lateness
    (the time the sent sample was due is ``t0 + offset(k)``). ``ok`` is False only for the last tick
    of a run whose backend call raised (the command may or may not have reached the backend).
    """

    k: int  # timeline sample index sent at this tick
    tick: int  # loop iteration (0, 1, 2, ...; differs from k after skipped samples)
    planned_ns: int  # deadline slept until
    actual_ns: int  # wake-up time (before the send)
    send_ns: int = 0  # duration of the backend call
    skipped_before: int = 0  # samples skipped right before this one
    ok: bool = True  # False: the backend call raised (run stopped here)
    late_ns: int = field(init=False)  # actual_ns - planned_ns

    def __post_init__(self) -> None:
        object.__setattr__(self, "late_ns", int(self.actual_ns) - int(self.planned_ns))


def _stats(values: np.ndarray, keys: Sequence[str]) -> dict[str, float | None]:
    """Named statistics of ``values`` (population std, linear-interpolated percentiles); None when empty."""
    if values.size == 0:
        return {k: None for k in keys}
    out: dict[str, float | None] = {}
    for key in keys:
        if key == "mean":
            v = np.mean(values)
        elif key == "std":
            v = np.std(values)
        elif key == "min":
            v = np.min(values)
        elif key == "max":
            v = np.max(values)
        elif key.startswith("p"):
            v = np.percentile(values, float(key[1:]))
        else:  # pragma: no cover - programming error
            raise KeyError(key)
        out[key] = float(v)
    return out


@dataclass
class TickLog:
    """Tick records of one run plus free-form metadata (rate, clock, backend, outcome).

    Appended by the executor thread only; other threads read snapshots (``list(log.records)``).
    """

    records: list[TickRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def append(self, record: TickRecord) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def summary(self, nominal_period_ns: float | None = None,
                late_threshold_ms: float | None = None) -> dict[str, Any]:
        """Period / jitter statistics (ms unless the key says otherwise; JSON-serialisable).

        * ``effective_hz`` = (n - 1) / (t_last - t_first) over the wake-up times (None for n < 2)
        * ``interval_ms``  = distribution of the intervals between consecutive wake-ups (n - 1 values;
          an interval after skipped samples spans several periods - that gap is real)
        * ``jitter_ms``    = std of (interval - nominal period); ``jitter_rms_ms`` = RMS of the same
          deviation (also counts a mean period error)
        * ``late_ms``      = distribution of wake-up lateness, ``late_ticks`` = ticks later than
          ``late_threshold_ms``; ``send_ms`` = backend call durations
        * ``n_ticks`` = records, ``n_sent`` = successful backend calls, ``skipped_samples`` = total skipped

        ``nominal_period_ns`` defaults to ``meta["period_ns"]`` and ``late_threshold_ms`` to
        ``meta["late_threshold_ms"]`` (both set by the executor, so every summary of one run counts late
        ticks with the executor's own threshold), else ``DEFAULT_LATE_THRESHOLD_MS``.
        """
        if nominal_period_ns is None:
            nominal_period_ns = self.meta.get("period_ns")
            if nominal_period_ns is None:
                raise ValueError("nominal_period_ns is required (no period_ns in meta)")
        if late_threshold_ms is None:
            late_threshold_ms = float(self.meta.get("late_threshold_ms", DEFAULT_LATE_THRESHOLD_MS))
        period_ms = float(nominal_period_ns) / 1e6
        recs = list(self.records)
        n = len(recs)
        actual = np.array([r.actual_ns for r in recs], dtype=np.int64)
        intervals_ms = np.diff(actual).astype(np.float64) / 1e6
        late_ms = np.array([r.late_ns for r in recs], dtype=np.float64) / 1e6
        send_ms = np.array([r.send_ns for r in recs], dtype=np.float64) / 1e6
        span_ns = int(actual[-1] - actual[0]) if n else 0
        deviation = intervals_ms - period_ms
        return {
            "n_ticks": n,
            "n_sent": int(sum(1 for r in recs if r.ok)),
            "skipped_samples": int(sum(r.skipped_before for r in recs)),
            "duration_s": span_ns / 1e9,
            "effective_hz": (n - 1) * 1e9 / span_ns if n >= 2 and span_ns > 0 else None,
            "period_ms": period_ms,
            "interval_ms": _stats(intervals_ms, INTERVAL_STATS),
            "jitter_ms": float(np.std(deviation)) if deviation.size else None,
            "jitter_rms_ms": float(np.sqrt(np.mean(deviation ** 2))) if deviation.size else None,
            "late_ms": _stats(late_ms, LATE_STATS),
            "late_threshold_ms": float(late_threshold_ms),
            "late_ticks": int(np.count_nonzero(late_ms > late_threshold_ms)),
            "send_ms": _stats(send_ms, SEND_STATS),
            "meta": dict(self.meta),
        }

    def to_rows(self) -> list[dict[str, Any]]:
        """One dict per tick with ``CSV_COLUMNS`` keys; times in s relative to the first planned tick.

        ``interval_ms`` is the gap to the previous tick's wake-up (None on the first row).
        """
        recs = list(self.records)
        if not recs:
            return []
        base = recs[0].planned_ns
        rows: list[dict[str, Any]] = []
        prev: TickRecord | None = None
        for r in recs:
            rows.append({
                "tick": r.tick,
                "k": r.k,
                "planned_s": (r.planned_ns - base) / 1e9,
                "actual_s": (r.actual_ns - base) / 1e9,
                "late_ms": r.late_ns / 1e6,
                "interval_ms": None if prev is None else (r.actual_ns - prev.actual_ns) / 1e6,
                "send_ms": r.send_ns / 1e6,
                "skipped_before": r.skipped_before,
            })
            prev = r
        return rows

    def to_csv(self, path: str | Path) -> Path:
        """Write ``to_rows()`` with the ``CSV_COLUMNS`` header (ns resolution kept); returns the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        def fmt(col: str, v: Any) -> str:
            if v is None:
                return ""
            if col in ("planned_s", "actual_s"):
                return f"{v:.9f}"
            if col.endswith("_ms"):
                return f"{v:.6f}"
            return str(v)

        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_COLUMNS)
            for row in self.to_rows():
                writer.writerow([fmt(c, row[c]) for c in CSV_COLUMNS])
        return p


# ---------------------------------------------------------------------- stationary REST plan
def _constant_pose_report(tl: Timeline, policy: SafetyPolicy) -> SafetyReport:
    """Policy report for a timeline whose samples are all identical (zero commanded speed).

    Checks the single pose with the daemon's default IK engine (``AnalyticalKinematics`` with
    ``automatic_body_yaw=True``, reachy_mini/kinematics/analytical_kinematics.py:33, 80-95, through the
    Phase 1 helper ``analysis.goto_replay.ik_joints``) and the URDF joint limits
    (``analysis.limits.urdf_joint_limits``):
    IK finite, Stewart margin >= ``min_stewart_margin_deg``, IK body joint within
    ``max_auto_body_shift_deg`` of the commanded body yaw, |body yaw| <= ``max_body_yaw_deg``,
    |antenna| <= ``max_antenna_abs_rad``. Speed limits hold trivially (nothing moves).
    """
    from reachy_motion.analysis.goto_replay import JOINT_NAMES, STEWART, ik_joints, new_kinematics
    from reachy_motion.analysis.limits import urdf_joint_limits

    if not (np.all(tl.head == tl.head[0]) and np.all(tl.antennas == tl.antennas[0])
            and np.all(tl.body_yaw == tl.body_yaw[0])):
        raise ValueError("timeline is not constant")
    head, (a_r, a_l), body = tl.sample(0)
    q = ik_joints(new_kinematics(), np.array(head), body)
    limits = urdf_joint_limits()
    violations: list[Violation] = []
    metrics: dict[str, Any] = {"constant_command": True, "n_samples": len(tl), "max_command_step": 0.0}
    if not np.all(np.isfinite(q)):
        violations.append(Violation("ik_nan", "IK of the constant pose is not finite", t=0.0))
    else:
        margins = [min(v - limits[name][0], limits[name][1] - v)
                   for name, v in zip(JOINT_NAMES[STEWART], q[STEWART])]
        margin_deg = math.degrees(min(margins))
        shift_deg = math.degrees(abs(q[0] - body))
        metrics.update(min_stewart_margin_deg=margin_deg, auto_body_shift_deg=shift_deg,
                       ik_joints_deg=[math.degrees(v) for v in q])
        if margin_deg < policy.min_stewart_margin_deg:
            violations.append(Violation("stewart_margin", "Stewart joint too close to its URDF limit", t=0.0,
                                        value=margin_deg, limit=policy.min_stewart_margin_deg))
        if shift_deg > policy.max_auto_body_shift_deg:
            violations.append(Violation("auto_body_yaw", "automatic body yaw engages", t=0.0, value=shift_deg,
                                        limit=policy.max_auto_body_shift_deg))
    if abs(math.degrees(body)) > policy.max_body_yaw_deg:
        violations.append(Violation("envelope_body_yaw", "body yaw outside the policy", t=0.0,
                                    value=math.degrees(body), limit=policy.max_body_yaw_deg))
    if max(abs(a_r), abs(a_l)) > policy.max_antenna_abs_rad:
        violations.append(Violation("antenna_range", "antenna angle outside the policy", t=0.0,
                                    value=max(abs(a_r), abs(a_l)), limit=policy.max_antenna_abs_rad))
    note = Violation("constant_hold_check",
                     "constant command checked by executor.timing (one IK + URDF margins, zero speed), not by the "
                     "full plan checker; streaming it is a step command if the head is elsewhere",
                     severity="warning")
    return SafetyReport(name=tl.name, variant=tl.variant, policy=policy, passes_policy=not violations,
                        violations=tuple(violations), warnings=(note,), metrics=metrics,
                        spec_digest=tl.spec_digest, timeline_digest=tl.digest(), data_kind="computed")


def stationary_rest_motion(duration_s: float, rate_hz: float = 50.0,
                           policy: SafetyPolicy = DEFAULT_POLICY) -> PreparedMotion:
    """A checked plan that holds REST (head = SDK INIT_HEAD_POSE, antennas INIT_ANTENNAS, body 0).

    Samples ``k = 0..K`` at ``k / rate_hz`` with ``K = ceil(duration_s * rate_hz)`` (Timeline grid rule).
    """
    if not (math.isfinite(duration_s) and duration_s >= 0.0):
        raise ValueError("duration_s must be >= 0")
    if not (math.isfinite(rate_hz) and rate_hz > 0.0):
        raise ValueError("rate_hz must be > 0")
    k_end = max(0, math.ceil(duration_s * rate_hz - 1e-9))
    if k_end / rate_hz + 1e-12 < duration_s:
        k_end += 1
    n = k_end + 1
    keys = np.full(n, -1, dtype=np.int64)
    tl = Timeline(
        name=REST_HOLD_NAME, variant="", rate_hz=float(rate_hz), duration=float(duration_s),
        t=np.arange(n, dtype=np.float64) / rate_hz,
        head=np.repeat(REST.head.matrix()[None, :, :], n, axis=0),
        antennas=np.tile(np.asarray(REST.antennas, dtype=np.float64), (n, 1)),
        body_yaw=np.full(n, float(REST.body_yaw)),
        head_key=keys, antenna_key=keys.copy(), body_key=keys.copy(),
    )
    problems = tl.validate()
    if problems:  # pragma: no cover - construction above follows the grid rule
        raise ValueError("REST hold timeline invalid: " + "; ".join(problems))
    return PreparedMotion(timeline=tl, report=_constant_pose_report(tl, policy), spec=None)


def measure_tick_timing(backend: Backend, duration_s: float = 10.0, rate_hz: float = 50.0,
                        clock: Clock | None = None) -> TickLog:
    """Stream a stationary REST plan for ``duration_s`` through the ``Executor``; return its tick log.

    The backend must already be connected (the caller owns ``connect`` / ``close``). A backend with
    ``capabilities.real_hardware`` is refused (``ExecutionRefused``): REST streamed to a head that is
    elsewhere is a step command. A backend error stops the run and raises ``BackendError``. With the
    default ``RealClock`` this blocks for ``duration_s`` (call it from a worker / CLI thread).
    """
    from reachy_motion.backends.base import BackendError
    from reachy_motion.executor.executor import ExecutionRefused, Executor

    if getattr(getattr(backend, "capabilities", None), "real_hardware", False):
        raise ExecutionRefused("measure_tick_timing streams a fixed REST pose: refused for real-hardware backends")
    prepared = stationary_rest_motion(duration_s, rate_hz)
    result = Executor(backend, clock=clock, rate_hz=rate_hz).run(prepared)
    log = result.ticks
    log.meta["purpose"] = "executor tick timing (stationary REST hold)"
    if result.error is not None:
        raise BackendError(f"timing run stopped after {len(result.commands)} sends: {result.error}")
    return log


__all__ = ["CSV_COLUMNS", "DEFAULT_LATE_THRESHOLD_MS", "TickRecord", "TickLog", "stationary_rest_motion",
           "measure_tick_timing", "REST_HOLD_NAME"]
