"""Tracking metrics: how a measured trajectory follows the commanded one (Phase 4; also MuJoCo and Mock runs).

Time bases - both series must be on ONE clock, this PC's ``perf_counter_ns`` (converted to s):

* command   ``t_cmd``  = when the client SENT the target (``Executor`` ``CommandRecord.t_ns``). The daemon applies it
  at its next control tick; on the robot the head joints are written one tick later still (IK, then write on the
  next tick: daemon/backend/robot/backend.py:200-262). The commanded value never comes back from the daemon
  (docs/phase1/SDK_ANALYSIS.md), so the client's own record is the only source.
* measured  ``t_meas`` = when the client RECEIVED the state push the value came from (``SdkBackend.present().t_ns`` =
  arrival of the ``JointPositionsMsg``). State pushes carry no timestamp and the robot clock is not synced to this
  PC, so how old a measurement is on arrival is unknown.

``lag`` is therefore the END-TO-END delay seen from the PC: command path + control tick + motor response +
measurement path. It cannot be split into these parts without robot-side timestamps.

Alignment: the commands are samples of the intended trajectory, so the command is linearly interpolated between
sends (the half tick of the daemon's zero-order hold between targets thus counts as lag). Before the first send the
first target applies, after the last send the last target holds (the daemon keeps writing its last target:
robot/backend.py:204-210). Measurements are snapshots: repeated arrival stamps (the executor may read the same push
twice) are dropped, then the values are linearly interpolated. Both are evaluated on one uniform grid (default 200 Hz)
over the measured span.

Windows, per channel, from the command:

* hold     the command stays constant (change <= ``hold_eps``) for at least ``min_hold_s``;
* move     the command changes; the *response* window is every move plus the ``max_lag_s`` after it;
* settled  a hold minus its first ``settle_skip_s`` when a move precedes it (the whole hold otherwise).

Metrics per channel, in the channel's own SI unit (rad or m):

* ``static_bias``          mean(measured - command) over the settled parts of the holds (all holds together; each
                           step also reports its own ``hold_bias``, since the sag depends on the pose).
* ``lag_s``                the shift tau in [0, ``max_lag_s``] (``lag_resolution_s`` steps) that minimises
                           RMS(measured(t) - command(t - tau) - static_bias) over the response windows;
                           ``lag_at_bound`` flags a minimum at either end of the search range. ``lag_fit_r2`` =
                           1 - residual^2 / variance of the shifted command says how much of the motion a pure delay
                           explains: small, friction-dominated channels fit badly (in an offline MuJoCo 10-degree
                           pitch nod the 2-degree Stewart joints gave meaningless lags) - trust a lag only when
                           r2 >= 0.9, and read the channel with the largest command excursion first.
* ``dynamic_rmse``         RMS(measured - command - static_bias) over the response windows (delay included).
* ``dynamic_rmse_lag_compensated``  the same with command(t - lag_s): the shape error left after the delay.
* ``peak_error``           max |measured - command| over the whole span (bias and delay included), and when.
* ``amplitude_ratio``      peak-to-peak(measured) / peak-to-peak(command) over the span.
* ``overshoot``            per step (a move that ends in a hold): max(0, max over the hold of d * (measured -
                           final)), with d the direction of the step and ``final`` the mean measured value over the
                           settled part; ``overshoot_ratio`` divides by |final - initial|.
* ``settling_time_s``      per step: time from the end of the commanded move until the measurement stays within
                           max(``settle_rel_tol`` * |final - initial|, ``settle_abs_tol``) of ``final`` for the rest
                           of the hold; NaN when it never does within the hold.
* ``hold_ripple``          RMS of the measured value about its own straight-line fit over each settled part of the
                           holds (all holds together): shaking or chatter while the command stands still; a slow drift
                           is not counted. Encoder steps and sensor noise set its floor (added 2026-11-13: the
                           assignment's "shaking" as a number; shown, not a stop criterion).

Channel aggregates: ``overshoot`` and ``settling_time_s`` are the worst step; ``unsettled_steps`` counts the steps
that never settled. Pure numpy, no I/O except ``load_run`` (CSV files written by ``sim_review._save_run_files``).
``data_kind`` is whatever the input was (simulated for MuJoCo runs, measured for the robot, mock for Mock runs).
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_RATE_HZ", "DEFAULT_MAX_LAG_S", "DEFAULT_LAG_RESOLUTION_S", "DEFAULT_MIN_HOLD_S", "DEFAULT_SETTLE_SKIP_S",
    "DEFAULT_SETTLE_REL_TOL", "HOLD_EPS", "StepMetrics", "ChannelMetrics", "dedupe", "channel_metrics",
    "tracking_metrics", "RunData", "load_run", "LEGACY_ALIGNMENT_NOTE",
]

DEFAULT_RATE_HZ = 200.0
DEFAULT_MAX_LAG_S = 0.5
DEFAULT_LAG_RESOLUTION_S = 0.001
DEFAULT_MIN_HOLD_S = 0.2
DEFAULT_SETTLE_SKIP_S = 0.5
DEFAULT_SETTLE_REL_TOL = 0.05
HOLD_EPS = 1e-9
_MIN_SETTLED_S = 0.05  # a settled part shorter than this is not used


@dataclass(frozen=True)
class StepMetrics:
    """One commanded step: a move [t_move_start_s, t_move_end_s] followed by a hold until t_hold_end_s."""

    t_move_start_s: float
    t_move_end_s: float  # the command reaches the hold value
    t_hold_end_s: float
    command_step: float  # hold value - value before the move
    measured_initial: float
    measured_final: float  # mean over the settled part of the hold (NaN if the hold has none)
    hold_bias: float  # measured_final - commanded hold value (static bias of this hold alone)
    overshoot: float  # >= 0 (NaN without a settled part)
    overshoot_ratio: float  # overshoot / |final - initial|
    settling_time_s: float  # NaN: never settles within the hold


@dataclass(frozen=True)
class ChannelMetrics:
    name: str
    n_command: int
    n_measured: int  # after removing repeated arrival stamps
    duplicates_removed: int
    span_s: tuple[float, float]
    static_bias: float
    lag_s: float
    lag_at_bound: bool
    lag_fit_r2: float  # 1 - (lag-compensated residual)^2 / variance of the shifted command: < 0.9 = unreliable lag
    dynamic_rmse: float
    dynamic_rmse_lag_compensated: float
    peak_error: float
    peak_error_t_s: float
    amplitude_ratio: float
    command_p2p: float
    overshoot: float  # worst step (NaN without steps)
    settling_time_s: float  # worst settled step (NaN if none settled)
    unsettled_steps: int
    hold_ripple: float  # RMS about a straight line over the settled holds (NaN: no settled hold)
    steps: tuple[StepMetrics, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["span_s"] = list(self.span_s)
        d["steps"] = [asdict(s) for s in self.steps]
        return d


def _clean(t: Any, x: Any, what: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if t.ndim != 1 or x.ndim != 1 or t.shape != x.shape:
        raise ValueError(f"{what}: t and x must be 1-D arrays of the same length, got {t.shape} and {x.shape}")
    if t.size < 2:
        raise ValueError(f"{what}: need at least 2 samples, got {t.size}")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(x))):
        raise ValueError(f"{what}: non-finite time or value")
    if np.any(np.diff(t) < 0):
        raise ValueError(f"{what}: times must not decrease (one clock, in send / arrival order)")
    return t, x


def dedupe(t: Any, x: Any, keep: str = "first") -> tuple[np.ndarray, np.ndarray, int]:
    """Drop samples whose time equals the previous one: ``keep="first"`` for measurements (the same push read twice),
    ``"last"`` for commands (a later send at the same instant replaces the earlier one). Returns (t, x, n_removed)."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if keep not in ("first", "last"):
        raise ValueError("keep must be 'first' or 'last'")
    if t.size == 0:
        return t, x, 0
    if keep == "first":
        mask = np.concatenate(([True], np.diff(t) > 0))
    else:
        mask = np.concatenate((np.diff(t) > 0, [True]))
    return t[mask], x[mask], int(t.size - mask.sum())


def _intervals(t_cmd: np.ndarray, x_cmd: np.ndarray, eps: float, lo: float, hi: float
               ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """(holds, moves) of the edge-held, linearly interpolated command, clipped to [lo, hi]; maximal intervals."""
    moving = np.abs(np.diff(x_cmd)) > eps
    holds: list[tuple[float, float]] = []
    moves: list[tuple[float, float]] = []
    run_start = -math.inf
    i, n = 0, len(t_cmd)
    while i < n - 1:
        if not moving[i]:
            i += 1
            continue
        holds.append((run_start, float(t_cmd[i])))
        j = i
        while j < n - 1 and moving[j]:
            j += 1
        moves.append((float(t_cmd[i]), float(t_cmd[j])))
        run_start = float(t_cmd[j])
        i = j
    holds.append((run_start, math.inf))

    def clip(ivs: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out = []
        for a, b in ivs:
            a, b = max(a, lo), min(b, hi)
            if b > a:
                out.append((a, b))
        return out

    return clip(holds), clip(moves)


def _mask(grid: np.ndarray, ivs: Sequence[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(grid.shape, dtype=bool)
    for a, b in ivs:
        m |= (grid >= a - 1e-12) & (grid <= b + 1e-12)
    return m


def _rms(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e * e))) if e.size else math.nan


def channel_metrics(name: str, t_cmd: Any, x_cmd: Any, t_meas: Any, x_meas: Any, *,
                    rate_hz: float = DEFAULT_RATE_HZ, max_lag_s: float = DEFAULT_MAX_LAG_S,
                    lag_resolution_s: float = DEFAULT_LAG_RESOLUTION_S, min_hold_s: float = DEFAULT_MIN_HOLD_S,
                    settle_skip_s: float = DEFAULT_SETTLE_SKIP_S, settle_rel_tol: float = DEFAULT_SETTLE_REL_TOL,
                    settle_abs_tol: float = 0.0, hold_eps: float = HOLD_EPS) -> ChannelMetrics:
    """Tracking metrics of one channel (module docstring). Times in s on one clock, values in one SI unit."""
    for label, v, positive in (("rate_hz", rate_hz, True), ("max_lag_s", max_lag_s, False),
                               ("lag_resolution_s", lag_resolution_s, True), ("min_hold_s", min_hold_s, False),
                               ("settle_skip_s", settle_skip_s, False), ("settle_rel_tol", settle_rel_tol, False),
                               ("settle_abs_tol", settle_abs_tol, False), ("hold_eps", hold_eps, False)):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and (v > 0 if positive else v >= 0)):
            raise ValueError(f"{label} must be a finite number {'> 0' if positive else '>= 0'}, got {v!r}")
    tc, xc = _clean(t_cmd, x_cmd, f"{name} command")
    tm, xm = _clean(t_meas, x_meas, f"{name} measured")
    tc, xc, _ = dedupe(tc, xc, keep="last")
    tm, xm, removed = dedupe(tm, xm, keep="first")
    if tm.size < 2:
        raise ValueError(f"{name} measured: fewer than 2 distinct arrival times")
    lo, hi = float(tm[0]), float(tm[-1])
    dt = 1.0 / rate_hz
    grid = lo + dt * np.arange(int(math.floor((hi - lo) / dt + 1e-9)) + 1)
    if grid.size < 2:
        raise ValueError(f"{name}: measured span {hi - lo:.4g} s is shorter than two grid steps")
    c = np.interp(grid, tc, xc)
    m = np.interp(grid, tm, xm)
    err = m - c

    holds, moves = _intervals(tc, xc, hold_eps, lo, hi)
    long_holds = [h for h in holds if h[1] - h[0] >= min_hold_s - 1e-12]
    move_ends = {round(b, 9) for _, b in moves}
    settled = []
    for a, b in long_holds:
        start = a + settle_skip_s if round(a, 9) in move_ends else a
        if b - start >= _MIN_SETTLED_S:
            settled.append((start, b))
    settled_mask = _mask(grid, settled)
    static_bias = float(np.mean(err[settled_mask])) if settled_mask.any() else math.nan
    ripples = []
    for a, b in settled:  # shaking: what is left of each settled hold after its own straight line
        k = _mask(grid, [(a, b)])
        if int(k.sum()) >= 3:
            g, v = grid[k] - grid[k][0], m[k]
            ripples.append(v - np.polyval(np.polyfit(g, v, 1), g))
    hold_ripple = _rms(np.concatenate(ripples)) if ripples else math.nan

    response = _mask(grid, [(a, min(b + max_lag_s, hi)) for a, b in moves])
    bias0 = static_bias if math.isfinite(static_bias) else 0.0
    lag_s, at_bound, rmse_lc, r2 = math.nan, False, math.nan, math.nan
    if response.any():
        taus = np.arange(0.0, max_lag_s + lag_resolution_s / 2, lag_resolution_s)
        best = (math.inf, 0.0)
        for tau in taus:
            e = m[response] - np.interp(grid[response] - tau, tc, xc)
            e = e - (static_bias if math.isfinite(static_bias) else float(np.mean(e)))
            r = _rms(e)
            if r < best[0] - 1e-15:
                best = (r, float(tau))
        rmse_lc, lag_s = best
        at_bound = lag_s <= lag_resolution_s / 2 or lag_s >= taus[-1] - lag_resolution_s / 2
        var_c = float(np.var(np.interp(grid[response] - lag_s, tc, xc)))
        r2 = 1.0 - rmse_lc**2 / var_c if var_c > 0.0 else math.nan
    dyn_rmse = _rms(err[response] - bias0) if response.any() else math.nan

    k_peak = int(np.argmax(np.abs(err)))
    p2p_c = float(np.ptp(c))
    amp = float(np.ptp(m)) / p2p_c if p2p_c > hold_eps else math.nan

    steps: list[StepMetrics] = []
    finals: dict[float, float] = {}
    for a, b in settled:
        finals[round(b, 9)] = float(np.mean(m[_mask(grid, [(a, b)])]))
    hold_by_start = {round(a, 9): (a, b) for a, b in long_holds}
    prev_final = math.nan
    first_hold = long_holds[0] if long_holds else None
    if first_hold is not None and round(first_hold[1], 9) in finals and first_hold[0] <= lo + 1e-9:
        prev_final = finals[round(first_hold[1], 9)]
    for ms, me in moves:
        hold = hold_by_start.get(round(me, 9))
        if hold is None:
            prev_final = math.nan
            continue
        hs, he = hold
        step_cmd = float(np.interp(hs, tc, xc) - np.interp(ms, tc, xc))
        initial = prev_final if math.isfinite(prev_final) else float(np.interp(ms, tm, xm))
        final = finals.get(round(he, 9), math.nan)
        over = ratio = settle = math.nan
        if math.isfinite(final) and abs(step_cmd) > hold_eps:
            in_hold = _mask(grid, [(hs, he)])
            gh, mh = grid[in_hold], m[in_hold]
            d = 1.0 if step_cmd > 0 else -1.0
            over = max(0.0, float(np.max(d * (mh - final))))
            span = abs(final - initial)
            ratio = over / span if span > hold_eps else math.nan
            band = max(settle_rel_tol * span, settle_abs_tol)
            out = np.nonzero(np.abs(mh - final) > band)[0]
            if out.size == 0:
                settle = 0.0
            elif out[-1] < gh.size - 1:
                settle = float(gh[out[-1] + 1] - hs)
        steps.append(StepMetrics(t_move_start_s=ms, t_move_end_s=me, t_hold_end_s=he, command_step=step_cmd,
                                 measured_initial=initial, measured_final=final,
                                 hold_bias=final - float(np.interp(hs, tc, xc)), overshoot=over,
                                 overshoot_ratio=ratio, settling_time_s=settle))
        prev_final = final
    overs = [s.overshoot for s in steps if math.isfinite(s.overshoot)]
    judged = [s for s in steps if math.isfinite(s.measured_final) and abs(s.command_step) > hold_eps]
    settled_times = [s.settling_time_s for s in judged if math.isfinite(s.settling_time_s)]
    return ChannelMetrics(
        name=name, n_command=int(tc.size), n_measured=int(tm.size), duplicates_removed=removed, span_s=(lo, hi),
        static_bias=static_bias, lag_s=lag_s, lag_at_bound=bool(at_bound), lag_fit_r2=r2, dynamic_rmse=dyn_rmse,
        dynamic_rmse_lag_compensated=rmse_lc, peak_error=float(abs(err[k_peak])), peak_error_t_s=float(grid[k_peak]),
        amplitude_ratio=amp, command_p2p=p2p_c, overshoot=max(overs) if overs else math.nan,
        settling_time_s=max(settled_times) if settled_times else math.nan,
        unsettled_steps=len(judged) - len(settled_times), hold_ripple=hold_ripple, steps=tuple(steps))


def tracking_metrics(t_cmd: Any, x_cmd: Any, t_meas: Any, x_meas: Any, names: Sequence[str], *,
                     settle_abs_tols: Sequence[float] | None = None, **kw: Any) -> dict[str, ChannelMetrics]:
    """``channel_metrics`` for every column of ``x_cmd`` (N, C) / ``x_meas`` (M, C); ``settle_abs_tols`` per channel."""
    xc = np.asarray(x_cmd, dtype=np.float64)
    xm = np.asarray(x_meas, dtype=np.float64)
    if xc.ndim != 2 or xm.ndim != 2 or xc.shape[1] != len(names) or xm.shape[1] != len(names):
        raise ValueError(f"x_cmd / x_meas must be (N, {len(names)}) / (M, {len(names)})")
    tols = [0.0] * len(names) if settle_abs_tols is None else list(settle_abs_tols)
    if len(tols) != len(names):
        raise ValueError("settle_abs_tols must have one value per channel")
    return {n: channel_metrics(n, t_cmd, xc[:, i], t_meas, xm[:, i], settle_abs_tol=tols[i], **kw)
            for i, n in enumerate(names)}


# --------------------------------------------------------------------------------------- run files (sim_review)
LEGACY_ALIGNMENT_NOTE = (
    "commands.csv and present_*.csv each start at their own first sample (sim_review._save_run_files): the first "
    "present sample is the last push that arrived before the first send, so the measurement time base can start up "
    "to one push interval EARLIER than the command one. Measured events then appear late by that offset: lag may be "
    "overestimated by up to one push interval (about 20 ms). New recordings should keep absolute perf_counter_ns.")

POSE_CHANNELS: tuple[tuple[str, str, float], ...] = (  # (csv column, channel name, factor to SI)
    ("x_mm", "x", 1e-3), ("y_mm", "y", 1e-3), ("z_mm", "z", 1e-3),
    ("roll_deg", "roll", math.pi / 180), ("pitch_deg", "pitch", math.pi / 180), ("yaw_deg", "yaw", math.pi / 180))
ANTENNA_CHANNELS: tuple[tuple[str, str, float], ...] = (
    ("antenna_right_deg", "antenna_right", math.pi / 180), ("antenna_left_deg", "antenna_left", math.pi / 180))
STEWART_NAMES: tuple[str, ...] = tuple(f"stewart_{i}" for i in range(1, 7))


@dataclass(frozen=True)
class RunData:
    """One executor run as SI arrays. ``joints`` are [stewart_1..6]: command = IK of the commanded pose (daemon engine)."""

    run_dir: str
    data_kind: str
    t_cmd: np.ndarray
    t_meas: np.ndarray
    channels: tuple[str, ...]
    units: tuple[str, ...]  # "m" or "rad"
    x_cmd: np.ndarray  # (N, C)
    x_meas: np.ndarray  # (M, C)
    alignment_note: str


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no rows")
    out: dict[str, np.ndarray] = {}
    for key in rows[0]:
        try:
            out[key] = np.array([float(r[key]) for r in rows], dtype=np.float64)
        except (TypeError, ValueError):
            continue  # text columns (kind)
    return out


def load_run(run_dir: str | Path, with_joints: bool = True) -> RunData:
    """Read ``commands.csv`` + ``present_<data_kind>.csv`` of an executor run (``sim_review._save_run_files``).

    Channels: head pose x/y/z (m) and roll/pitch/yaw (rad), both antennas (rad) and, when the measured file has
    them and ``with_joints``, stewart_1..6 (rad): the commanded joints are the IK of each commanded pose with the
    daemon's default engine (``analysis.goto_replay.new_kinematics``), the analysis CLAUDE.md asks for (joint space).
    """
    run = Path(run_dir)
    presents = sorted(run.glob("present_*.csv"))
    if not (run / "commands.csv").is_file() or len(presents) != 1:
        raise ValueError(f"{run}: expected commands.csv and exactly one present_*.csv")
    data_kind = presents[0].stem.split("_", 1)[1]
    cmd, meas = _read_csv(run / "commands.csv"), _read_csv(presents[0])
    names, units, cols_c, cols_m = [], [], [], []
    for col, name, f in POSE_CHANNELS + ANTENNA_CHANNELS:
        if col in cmd and col in meas:
            names.append(name)
            units.append("m" if col.endswith("_mm") else "rad")
            cols_c.append(cmd[col] * f)
            cols_m.append(meas[col] * f)
    if with_joints and all(f"{n}_deg" in meas for n in STEWART_NAMES):
        from reachy_motion.analysis.goto_replay import ik_joints, new_kinematics
        from reachy_motion.library.pose import Pose6

        kin = new_kinematics()
        body = cmd.get("body_yaw_deg", np.zeros_like(cmd["t_s"])) * math.pi / 180
        q = np.array([ik_joints(kin, Pose6(*(cmd[c][k] * f for c, _, f in POSE_CHANNELS)).matrix(), float(body[k]))
                      for k in range(cmd["t_s"].size)])
        for i, n in enumerate(STEWART_NAMES):
            names.append(n)
            units.append("rad")
            cols_c.append(q[:, i + 1])
            cols_m.append(meas[f"{n}_deg"] * math.pi / 180)
    return RunData(run_dir=str(run), data_kind=data_kind, t_cmd=cmd["t_s"], t_meas=meas["t_s"],
                   channels=tuple(names), units=tuple(units), x_cmd=np.column_stack(cols_c),
                   x_meas=np.column_stack(cols_m), alignment_note=LEGACY_ALIGNMENT_NOTE)
