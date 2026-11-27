"""Safe start-up procedure (Phase 4 stage R1-0 proposal) - verified with the Mock backend and in MuJoCo only.

Nothing here can reach the robot: the robot backend is still locked (``backends.sdk``), and any robot use needs the
separate approval of CLAUDE.md rule 3. Why not the SDK's ``wake_up`` (comparison: docs/phase4/STARTUP_STOP.md): its
first goto starts from the daemon's FK pose, turns the body to 0 (``body_yaw`` default) and sweeps the antennas 165 deg
in 1.5-2 s, then a 20-degree roll in 0.2 s and back (reachy_mini.py:724-740; daemon/backend/abstract.py:1284-1323),
about 1945 deg/s^2 peak Stewart acceleration in the commanded trajectory against 7 deg/s^2 for the slow ramp below.

Procedure (``run_startup``):

1. ``check_start``: a finite present state WITH joints; every joint inside its URDF limit; the pose the daemon reports
   (its FK estimate) agrees with the joints - the IK of the pose is within ``FK_JOINT_TOL_DEG`` of them - because the
   ramp is a pose-space trajectory that starts from that pose (a mismatch would make the first target jump).
2. Torque - robot only, NOT done by this module (no robot connection exists yet): if R0 reports the motors limp, the
   robot runner calls the SDK's GLOBAL ``enable_motors()`` (no ``ids``), which pins every target to the last read
   joints, writes them into the goal registers and only then turns the torque on (daemon/backend/robot/
   backend.py:352-377). A per-motor enable (``ids``) skips the pinning (daemon/backend/abstract.py:1786-1792) and is
   never used. MuJoCo motors are always on and the Mock has no torque, so this step is skipped there.
3. ``watch_still``: for ``still_s`` the measured joints must stay within ``STILL_TOL_DEG`` of their first reading - a
   snap at torque-on, a head still settling or sagging, or a hand on it stops the procedure before any command.
4. ``plan_startup``: a MIN_JERK ramp from the re-read state to REST under the start-up rules of ``safety.startup``
   (no joint ever closer to its limit than at the start, the end pose inside the normal policy), at the slow caps
   ``SLOW_CAPS`` and at least ``MIN_STARTUP_S``.
5. Stream it with a ``StopPolicy`` (S1 on cancel, S2 on a failed or slow send or a lost measurement), recording the
   present state.
6. Hold ``hold_s`` at REST; report the end error against REST and the largest commanded joint step.

Antennas (2026-11-12, docs/phase4/STARTUP_STOP.md §11): ``antennas="rest"`` ramps them to the SDK rest angles like the
head; ``"hold"`` keeps them where they are (the ramp's end state is REST for the head and body, the present antennas),
and a pair of angles holds them at that pair, refused when it lies more than ``ANTENNA_HOLD_TOL_DEG`` from the present
reading. The robot's first start uses a held pair: the limp antennas read 157 / 130 deg in R0 (2026-11-11) against
the -10 / +10 deg rest, and their zero, sign and wrap live in the compiled motor controller (not verifiable from
source), so the start-up never sweeps them.

The caps and tolerances are conservative proposals for the first robot start, not measured limits.
"""

from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from reachy_motion.backends.base import PresentState
from reachy_motion.executor.executor import ExecutionResult, Executor, OnTick, StopPolicy
from reachy_motion.library.pose import REST, RobotState, pose_from_matrix, rotation_distance, translation_distance
from reachy_motion.planner import transition_duration
from reachy_motion.timeline import PreparedMotion

__all__ = ["SLOW_CAPS", "MIN_STARTUP_S", "FK_JOINT_TOL_DEG", "STILL_TOL_DEG", "ANTENNA_HOLD_TOL_DEG", "StartCheck",
           "StartupReport", "check_start", "startup_end", "plan_startup", "watch_still", "run_startup"]

#: Peak-speed caps of the start-up ramp (MIN_JERK, planner.transition_duration): head rotation deg/s, translation
#: mm/s, antennas deg/s, body yaw deg/s. Proposals, not measured robot limits.
SLOW_CAPS: dict[str, float] = {"max_rot_dps": 15.0, "max_trans_mms": 10.0, "max_antenna_dps": 60.0,
                               "max_body_dps": 10.0}
MIN_STARTUP_S = 2.0
FK_JOINT_TOL_DEG = 1.0  # as robot_r0 (Phase 1 K27: the per-tick FK lags 0.14-0.35 deg in motion)
STILL_TOL_DEG = 0.5
STILL_S = 1.0
HOLD_S = 1.0
ANTENNA_HOLD_TOL_DEG = 2.0  # a held antenna pair may differ this much from the present reading (R0 -> connect)
JOINT_NAMES = ("body_yaw",) + tuple(f"stewart_{i}" for i in range(1, 7))


def _deg(x: float) -> float:
    return math.degrees(x)


@dataclass
class StartCheck:
    ok: bool
    problems: list[str] = field(default_factory=list)
    state: RobotState | None = None
    joints_deg: list[float] | None = None
    min_stewart_margin_deg: float | None = None
    ik_gap_deg: float | None = None  # max |IK(reported pose) - joints| over the Stewart joints
    max_beyond_deg: float = 0.0  # how far the farthest joint is past its URDF limit (0 = all inside)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = None if self.state is None else {
            "head": self.state.head.to_deg_mm(), "antennas_deg": [_deg(a) for a in self.state.antennas],
            "body_yaw_deg": _deg(self.state.body_yaw)}
        return d


def check_start(p: PresentState, beyond_tol_deg: float = 0.0) -> StartCheck:
    """Step 1 (module docstring): may the start-up ramp begin from this present state? Joints past the URDF limits
    refuse, unless every one is past by at most ``beyond_tol_deg`` - only the robot connection passes that, for a
    LIMP head (``robot_r0.LIMP_BEYOND_TOL_DEG``, user decision 2026-11-17); the start-up ramps keep 0."""
    from reachy_motion.analysis.goto_replay import ik_joints, new_kinematics
    from reachy_motion.analysis.limits import urdf_joint_limits

    head = np.asarray(p.head, dtype=float)
    if (head.shape != (4, 4) or not np.all(np.isfinite(head)) or not all(math.isfinite(a) for a in p.antennas)
            or not math.isfinite(p.body_yaw)):
        return StartCheck(False, ["the present state is not finite"])
    if p.joints is None:
        return StartCheck(False, ["no joint reading: the start joints cannot be checked"])
    q = np.asarray(p.joints, dtype=float)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        return StartCheck(False, ["the joint reading is not 7 finite values"])
    lim = urdf_joint_limits()
    lo = np.array([lim[n][0] for n in JOINT_NAMES])
    hi = np.array([lim[n][1] for n in JOINT_NAMES])
    margin = np.minimum(q - lo, hi - q)
    problems = []
    beyond = [(n, _deg(-m)) for n, m in zip(JOINT_NAMES, margin) if m < 0]
    worst = max((x for _, x in beyond), default=0.0)
    if beyond and worst > beyond_tol_deg:
        problems.append("joints beyond the URDF limits: " + ", ".join(f"{n} {x:.1f} deg" for n, x in beyond)
                        + (f" (more than {beyond_tol_deg:g} deg past)" if beyond_tol_deg > 0 else "") + " (PHASE4_PLAN §0)")
    q_ik = np.asarray(ik_joints(new_kinematics(), head, float(p.body_yaw)), dtype=float)
    gap = math.nan
    if not np.all(np.isfinite(q_ik)):
        problems.append("the reported head pose has no IK solution")
    else:
        gap = _deg(float(np.max(np.abs(q_ik[1:] - q[1:]))))
        if gap > FK_JOINT_TOL_DEG:
            problems.append(f"the reported pose and the joints disagree by {gap:.2f} deg (> {FK_JOINT_TOL_DEG:g}): "
                            "a pose-space ramp would start with a jump")
    state = RobotState(head=pose_from_matrix(head), antennas=(float(p.antennas[0]), float(p.antennas[1])),
                       body_yaw=float(p.body_yaw))
    return StartCheck(not problems, problems, state, [_deg(v) for v in q], _deg(float(np.min(margin[1:]))), gap,
                      max_beyond_deg=worst)


def startup_end(start: RobotState, antennas: Any = "rest") -> RobotState:
    """The ramp's end state: REST for the head and body; antennas "rest", "hold" (= ``start``'s) or a pair (rad).
    A pair farther than ``ANTENNA_HOLD_TOL_DEG`` from ``start`` raises ValueError (holding never sweeps them)."""
    if isinstance(antennas, str):
        if antennas == "rest":
            return REST
        if antennas == "hold":
            return RobotState(head=REST.head, antennas=start.antennas, body_yaw=REST.body_yaw)
        raise ValueError(f"antennas must be 'rest', 'hold' or a pair of angles, got {antennas!r}")
    pair = (float(antennas[0]), float(antennas[1]))
    gap = max(abs(_deg(a - b)) for a, b in zip(pair, start.antennas))
    if not all(math.isfinite(a) for a in pair) or not gap <= ANTENNA_HOLD_TOL_DEG:
        raise ValueError(f"held antennas {[round(_deg(a), 2) for a in pair]} deg are {gap:.2f} deg from the present "
                         f"{[round(_deg(a), 2) for a in start.antennas]} deg (> {ANTENNA_HOLD_TOL_DEG:g}): the start-up "
                         "does not move the antennas - read R0 again")
    return RobotState(head=REST.head, antennas=pair, body_yaw=REST.body_yaw)


def plan_startup(start: RobotState, caps: dict[str, float] | None = None, min_s: float = MIN_STARTUP_S,
                 antennas: Any = "rest") -> PreparedMotion:
    """Step 4: the slow MIN_JERK ramp start -> REST (antennas per ``startup_end``), checked with the start-up rules
    (``prepared.runnable``)."""
    from reachy_motion.safety.startup import prepare_startup_ramp

    c = dict(SLOW_CAPS if caps is None else caps)
    end = startup_end(start, antennas)
    return prepare_startup_ramp(start, end, duration=transition_duration(start, end, min_s=min_s, **c))


def watch_still(backend: Any, clock: Any, seconds: float = STILL_S, tol_deg: float = STILL_TOL_DEG,
                period_s: float = 0.02) -> tuple[bool, float, int]:
    """Step 3: (still?, largest joint deviation from the first reading in deg, readings) over ``seconds``."""
    first: np.ndarray | None = None
    worst, n = 0.0, 0
    t_end = clock.now_ns() + int(round(seconds * 1e9))
    while True:
        p = backend.present()
        if p.joints is not None:
            q = np.asarray(p.joints, dtype=float)
            if first is None:
                first = q
            else:
                worst = max(worst, _deg(float(np.max(np.abs(q - first)))))
            n += 1
        now = clock.now_ns()
        if now >= t_end:
            break
        clock.sleep_until(min(t_end, now + int(round(period_s * 1e9))))
    return (first is not None and worst <= tol_deg), worst, n


@dataclass
class StartupReport:
    ok: bool
    stage: str  # where it ended: "check" | "still" | "plan" | "run" | "done"
    problems: list[str]
    data_kind: str
    check: dict[str, Any] | None = None
    still_max_drift_deg: float | None = None
    ramp_duration_s: float | None = None
    max_command_joint_step_deg: float | None = None  # largest Stewart / body step between consecutive samples
    peak_command_rotation_dps: float | None = None
    run: dict[str, Any] | None = None  # ExecutionResult.summary()
    end_error: dict[str, float] | None = None  # measured after the hold, against REST

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ramp_demand(prep: PreparedMotion) -> tuple[float, float]:
    from reachy_motion.safety.checker import analyze_timeline

    tl = prep.timeline
    q = np.asarray(analyze_timeline(tl).joints, dtype=float)
    step = _deg(float(np.max(np.abs(np.diff(q, axis=0))))) if len(tl) > 1 else 0.0
    rot = max((_deg(rotation_distance(tl.head[i], tl.head[i + 1])) * tl.rate_hz for i in range(len(tl) - 1)),
              default=0.0)
    return step, rot


def run_startup(backend: Any, executor: Executor, *, stop: StopPolicy | None = None, still_s: float = STILL_S,
                hold_s: float = HOLD_S, caps: dict[str, float] | None = None, cancel: threading.Event | None = None,
                on_tick: OnTick | None = None, antennas: Any = "rest") -> tuple[StartupReport, ExecutionResult | None]:
    """Steps 1 and 3-6 of the module docstring (step 2, torque, belongs to the future robot runner); ``antennas``
    as ``startup_end`` ("hold" or a pair: the antennas are not swept)."""
    clock = executor.clock
    kind = str(getattr(backend, "data_kind", ""))
    stop = StopPolicy() if stop is None else stop
    chk = check_start(backend.present())
    if not chk.ok:
        return StartupReport(False, "check", chk.problems, kind, chk.to_dict()), None
    still, drift, _ = watch_still(backend, clock, still_s)
    if not still:
        return StartupReport(False, "still", [f"the head moved {drift:.2f} deg (> {STILL_TOL_DEG:g}) while it should "
                                              f"hold still"], kind, chk.to_dict(), drift), None
    chk = check_start(backend.present())  # the ramp starts from this fresh reading
    if not chk.ok:
        return StartupReport(False, "check", chk.problems, kind, chk.to_dict(), drift), None
    try:
        end_state = startup_end(chk.state, antennas)  # type: ignore[arg-type]
    except ValueError as exc:
        return StartupReport(False, "plan", [str(exc)], kind, chk.to_dict(), drift), None
    prep = plan_startup(chk.state, caps, antennas=antennas)  # type: ignore[arg-type]  # end_state, checked above
    step, rot = _ramp_demand(prep)
    report = StartupReport(False, "plan", [], kind, chk.to_dict(), drift, float(prep.timeline.duration), step, rot)
    if not prep.runnable:
        report.problems = [f"{v.code}: {v.message}" for v in prep.report.violations]
        return report, None
    result = executor.run(prep, cancel=cancel, on_tick=on_tick, record_present=True, stop=stop)
    report.stage, report.run = "run", result.summary()
    if not result.completed:
        report.problems = [result.stop_reason or result.error or "the ramp did not complete"]
        return report, result
    clock.sleep_until(clock.now_ns() + int(round(hold_s * 1e9)))
    end = backend.present()
    rest = REST.head.matrix()
    report.end_error = {  # against the ramp's end state (REST head and body; the held antennas when held)
        "head_rotation_deg": _deg(rotation_distance(np.asarray(end.head, dtype=float), rest)),
        "head_translation_mm": 1000.0 * translation_distance(np.asarray(end.head, dtype=float), rest),
        "antennas_deg": _deg(max(abs(a - b) for a, b in zip(end.antennas, end_state.antennas))),
        "body_yaw_deg": _deg(abs(end.body_yaw - REST.body_yaw))}
    report.ok, report.stage = True, "done"
    return report, result
