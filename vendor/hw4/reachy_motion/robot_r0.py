"""Phase 4 stage R0: read-only robot check - REST GET only (CLAUDE.md rule 3), no motion, no client, no approval file.

Every request goes through ``backends.daemon_client.DaemonClient``: GET on its five whitelisted paths, no query string,
no redirect, no proxy. Nothing here builds a ``ReachyMini`` (its constructor sends commands to the daemon, rule 3),
opens a WebSocket, or calls anything that changes the daemon or the motors.

What R0 answers before any motor stage (docs/phase4/PHASE4_PLAN.md):

* the daemon: version 1.11.0 (rule 2), state, simulation flag, errors, control-loop statistics;
* the IK engine (the analyses reproduce the daemon's IK with ``AnalyticalKinematics``);
* the motor mode: enabled / disabled (limp) / gravity_compensation;
* where the head is: present head pose and body yaw, the distance to rest and to the SDK sleep pose
  (``reachy_mini.reachy_mini.SLEEP_HEAD_POSE``), and the present joints with their margins to the URDF limits. The
  joints come from ``/api/state/full?with_head_joints=true`` (added to the CLAUDE.md list on 2026-11-11: the motor
  controller's cached positions, ``get_last_position``, daemon/backend/robot/backend.py:480-506), in the same snapshot
  as the pose. The IK of the reported pose (the daemon's FK estimate) is compared with them: the SDK's goto and
  ``wake_up`` start from that pose (daemon/backend/abstract.py:944-977), and so does a pose-space ramp, so a pose
  that does not match the joints means the first target would jump. Without a joint reading the IK of the pose is
  used as an estimate (attention).

Why the start pose matters (2026-11-11, docs/phase4/PHASE4_PLAN.md §0): the SDK's ``SLEEP_HEAD_JOINT_POSITIONS``
(reachy_mini.py:56-64) lie 0.8-9.5 deg beyond the URDF limits on stewart_1/2/5/6, while the SDK's own sleep
motion (a pose-space goto to ``SLEEP_HEAD_POSE``) stays inside them. Where the limp robot's head really rests is not
in the source, so a start pose whose joints are beyond the limits BLOCKS the motor stages until it is explained.

What R0 cannot answer (2026-11-11, docs/phase4/STARTUP_STOP.md §8.2):

* head tracking: the daemon's enabled flag is in no allowed GET. ``face_target`` (/api/daemon/status) is only set
  while tracking is enabled and its detector has produced an observation (abstract.py:765-790; disabling resets it,
  712-719): set = tracking is on (block), but the default does not prove it off (enabled with weight 0 or before the
  first observation, 683-699, 729). Only ``no_media`` proves it off (no media server: the enable refuses, 683-689).
* a per-motor torque enable leaves the reported mode unchanged (robot/backend.py:646-667, 604-606);
* other clients sending commands: no allowed GET lists them.

``r1_entry`` (user decision 2026-11-12) therefore sorts the entry conditions into four categories instead of asking
for all of them to be "met" (which the allowed GETs can never give): ``auto`` (judged from this report: met / unmet),
``connection`` (not read but established and re-checked by the connection sequence itself, backends/robot_connect.py),
``operator`` (only the operator controls it: confirmed at the terminal right before the connection, otherwise no run)
and ``residual`` (nobody can verify or prevent it: recorded, never "verified"). Its verdict is ``blocked`` (an
``auto`` condition is unmet) or ``conditional`` - never "met".

Findings have a level: ``block`` (no motor stage until resolved), ``attention`` (a person looks first), ``info``.
A motor mode that was not read and a body yaw that was not read both block; neither is replaced by a default.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from reachy_motion.backends.base import BackendError
from reachy_motion.backends.daemon_client import DaemonClient, loop_stats

__all__ = ["EXPECTED_VERSION", "EXPECTED_ENGINE", "MIN_START_MARGIN_DEG", "Finding", "collect", "evaluate", "run_r0",
           "write_report", "report_markdown", "r1_entry", "ENTRY_CATEGORIES", "R1_CONNECTION", "R1_OPERATOR",
           "R1_RESIDUAL", "AWAKE_INIT_TOL", "INIT_ANTENNAS_DEG", "R1_CONNECTION_AWAKE", "init_pose_error",
           "LIMP_BEYOND_TOL_DEG"]

EXPECTED_VERSION = "1.11.0"  # CLAUDE.md rule 2
EXPECTED_ENGINE = "AnalyticalKinematics"  # daemon default (abstract.py:210-213)
MIN_START_MARGIN_DEG = 10.0  # initial policy joint margin (safety.policy DEFAULT_POLICY)
FK_JOINT_TOL_DEG = 1.0  # max |IK(reported pose) - measured joint| (Phase 1 K27: FK lags 0.14-0.35 deg in motion)
MIN_LOOP_HZ = 45.0  # HW3 saw 49.5-49.7 Hz on the robot (docs/phase1/SOURCE_REFERENCES.md CBT-08)
NEAR_SLEEP_DEG, NEAR_SLEEP_MM = 10.0, 15.0
BODY_YAW_ATTENTION_DEG = 5.0  # the motions assume body yaw 0
#: How far a LIMP head may rest past the URDF joint limits and still be connected (user decision 2026-11-17, "고쳐
#: 주세요" after "내가 고개를 들어야 연결이 되는데"). The daemon's goto_sleep ends limp ("Rest limp at the sleep pose, like
#: a fresh boot", daemon/backend/abstract.py:1371-1372) and the wireless robot boots limp (no wake-up on start,
#: daemon/app/services/wireless/launcher.sh:18); gravity then settles the head at its mechanical rest, which can lie a
#: little past the URDF limits. The official app wakes it from there with the same global enable_motors() pin the
#: connection uses (robot/backend.py:352-377). 10 deg covers the SDK's own sleep joints (SLEEP_HEAD_JOINT_POSITIONS,
#: up to 9.5 deg past, reachy_mini.py:56-64). Farther, or with the torque on, it still blocks. What the firmware does
#: with a goal past its EEPROM limit is not in the source (clamp or refusal, docs/phase1/SDK_ANALYSIS.md): the
#: connection's torque-on still check allows that much more movement and stops otherwise.
LIMP_BEYOND_TOL_DEG = 10.0
JOINT_NAMES = ("body_yaw",) + tuple(f"stewart_{i}" for i in range(1, 7))
_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
#: R1 entry categories, in display order (module docstring).
ENTRY_CATEGORIES = ("auto", "connection", "operator", "residual")
#: Not readable with the allowed GETs, but established by the connection sequence and re-checked there (a failed
#: re-check stops the sequence): name -> how.
R1_CONNECTION = {
    "head_tracking_off": "the sequence turns tracking and wobbling off with the torque OFF (stop_head_tracking, "
                         "disable_wobbling: abstract.py:702-719, after which step_head_tracking returns at once, "
                         "729-730) and reads face_target again before the enable",
    "no_partial_torque": "a per-motor enable leaves the reported mode unchanged (robot/backend.py:646-667, 604-606) "
                         "but the loop writes no goal without the global flag (203); the global enable_motors() "
                         "re-applies over it and pins every target to the present joints (614-626, 352-376), then the "
                         "sequence checks that nothing moved",
}
#: Only the operator controls it: confirmed at the terminal right before the connection; not confirmed -> no run.
R1_OPERATOR = {
    "controlled_environment": "this PC is the only client of the robot: no dashboard app or demo running, no SDK "
                              "script on another computer (if possible a network with only the robot and this PC); "
                              "nothing around the head and antennas, power switch within reach, a person ready to "
                              "support the head. If this cannot be controlled, the run does not happen",
}
#: Nobody can verify or prevent it with SDK 1.11.0: recorded with every run, never "verified".
R1_RESIDUAL = {
    "other_client": "no control lock in the SDK and no allowed GET lists the daemon's clients (direct SDK clients also "
                    "bypass the managed-app lock, daemon/app/routers/daemon.py:167-189): another client can still "
                    "command the robot during a run; the still checks and the tracking error show it only after the "
                    "head moved",
}


@dataclass(frozen=True)
class Finding:
    level: str  # "block" | "attention" | "info"
    code: str
    message: str


def collect(client: DaemonClient, timeout_s: float | None = None) -> dict[str, Any]:
    """The five read-only GETs, best effort: an answer or ``{"error": ...}`` per item (no retry, nothing else sent)."""
    calls = (("status", client.status), ("kinematics_info", client.kinematics_info),
             ("state_full", client.state_full_with_head_joints),  # pose + joints of one snapshot
             ("present_body_yaw", client.present_body_yaw), ("motors_status", client.motors_status))
    answers: dict[str, Any] = {}
    for key, call in calls:
        try:
            answers[key] = call(timeout_s)
        except BackendError as exc:
            answers[key] = {"error": str(exc)}
    return answers


def _tracking_state(face: Any, no_media: Any) -> str:
    """'on' | 'off' | 'unknown' from the only allowed-GET fields that touch head tracking (module docstring)."""
    if isinstance(face, dict) and (face.get("detected") is True
                                   or any(face.get(k) is not None for k in ("x", "y", "roll", "ts"))):
        return "on"
    return "off" if no_media is True else "unknown"


def _failed(answer: Any) -> bool:
    return isinstance(answer, dict) and set(answer) == {"error"}


def _deg(x: float) -> float:
    return math.degrees(x)


def evaluate(answers: dict[str, Any]) -> tuple[dict[str, Any], list[Finding]]:
    """(derived values, findings) from the raw answers of ``collect`` - pure computation, no request."""
    from reachy_mini.reachy_mini import SLEEP_HEAD_POSE  # a constant; importing the module constructs nothing

    from reachy_motion.analysis.goto_replay import ik_joints, new_kinematics
    from reachy_motion.analysis.limits import urdf_joint_limits
    from reachy_motion.library.pose import Pose6, rotation_distance, translation_distance

    f: list[Finding] = []
    derived: dict[str, Any] = {}
    status = answers.get("status")
    if status is None or _failed(status) or not isinstance(status, dict):
        f.append(Finding("block", "no_status", f"GET /api/daemon/status failed: {status!r}"))
        return derived, f
    sim, mockup = status.get("simulation_enabled"), status.get("mockup_sim_enabled")
    derived["data_kind"] = "simulated" if sim is True else "measured" if (sim is False and mockup is False) else None
    derived["robot_name"] = status.get("robot_name")
    derived["daemon_version"] = version = status.get("version")
    derived["daemon_state"] = state = status.get("state")
    if version != EXPECTED_VERSION:
        f.append(Finding("block", "version", f"daemon version {version!r}, expected {EXPECTED_VERSION} (rule 2: no "
                                             f"update, no downgrade; stop here)"))
    if state != "running":
        f.append(Finding("block", "state", f"daemon state {state!r}, expected 'running'"))
    backend = status.get("backend_status") if isinstance(status.get("backend_status"), dict) else {}
    errors = [e for e in (status.get("error"), backend.get("error")) if e]
    if errors:
        f.append(Finding("block", "daemon_error", f"daemon reports an error: {errors}"))
    if sim is True or mockup is True:
        f.append(Finding("info", "simulation", "the daemon is a simulation (not the robot): simulated data"))
    face = status.get("face_target")
    derived["head_tracking"] = tracking = _tracking_state(face, status.get("no_media"))
    if tracking == "on":
        f.append(Finding("block", "head_tracking_active", f"face_target {face!r} is set: daemon-side head tracking is "
                                                          "enabled and running (another client turned it on); its "
                                                          "aim overrides the enable pin at the next control tick"))
    stats = loop_stats(status)
    derived["control_loop_stats"] = stats
    if stats:
        hz, n_err = stats.get("mean_control_loop_frequency"), stats.get("nb_error")
        if isinstance(hz, (int, float)) and hz < MIN_LOOP_HZ:
            f.append(Finding("attention", "loop_rate", f"control loop {hz:.1f} Hz < {MIN_LOOP_HZ:g} Hz"))
        if isinstance(n_err, (int, float)) and n_err > 0:
            f.append(Finding("attention", "loop_errors", f"control loop reports nb_error = {n_err}"))

    kin_info = answers.get("kinematics_info")
    engine = kin_info.get("info", {}).get("engine") if isinstance(kin_info, dict) and not _failed(kin_info) else None
    derived["ik_engine"] = engine
    if engine != EXPECTED_ENGINE:
        f.append(Finding("attention", "ik_engine", f"IK engine {engine!r}, the analyses assume {EXPECTED_ENGINE}"))

    motors = answers.get("motors_status")
    mode = motors.get("mode") if isinstance(motors, dict) and not _failed(motors) else None
    derived["motor_mode"] = mode
    if mode == "disabled":
        f.append(Finding("info", "motors_disabled", "motors are limp (torque off). Turning torque on is a motor "
                                                    "action that needs the R1 approval; the SDK's enable_motors pins "
                                                    "the targets to the present joints first "
                                                    "(daemon/backend/robot/backend.py:352-362)"))
    elif mode == "gravity_compensation":
        f.append(Finding("attention", "motors_gravity", "motors in gravity compensation: position targets are ignored"))
    elif mode == "enabled":
        f.append(Finding("info", "motors_enabled", "torque on: the daemon holds its last target"))
    else:
        f.append(Finding("block", "motors_unknown", f"motor mode not read ({motors!r}): the torque state must be "
                                                    "known before any motor stage"))

    state_full = answers.get("state_full")
    pose = state_full.get("head_pose") if isinstance(state_full, dict) and not _failed(state_full) else None
    try:
        p = Pose6(*(float(pose[k]) for k in _POSE_KEYS))  # type: ignore[index]
    except (TypeError, KeyError, ValueError):
        f.append(Finding("block", "no_pose", f"no head pose in /api/state/full: {state_full!r}"))
        return derived, f
    body = answers.get("present_body_yaw")
    body_yaw = (float(body) if isinstance(body, (int, float)) and not isinstance(body, bool) and math.isfinite(body)
                else None)  # an unread body yaw stays unknown: never replaced by 0
    derived["head_pose"] = {k: round(v, 3) for k, v in p.to_deg_mm().items()}
    derived["body_yaw_deg"] = None if body_yaw is None else round(_deg(body_yaw), 3)
    ant = state_full.get("antennas_position")
    if isinstance(ant, list) and len(ant) == 2:
        derived["antennas_deg"] = [round(_deg(float(a)), 3) for a in ant]
    if body_yaw is None:
        f.append(Finding("block", "no_body_yaw", f"GET /api/state/present_body_yaw gave no number ({body!r}): the "
                                                 "body yaw is unknown (not replaced by 0) and the pose-joint "
                                                 "cross-check, which needs it, is skipped"))
    elif abs(_deg(body_yaw)) > BODY_YAW_ATTENTION_DEG:
        f.append(Finding("attention", "body_yaw", f"body yaw {_deg(body_yaw):.1f} deg: the motions assume 0 and a "
                                                  f"goto without body_yaw turns the body to 0 (reachy_mini.py:680)"))
    m = p.matrix()
    derived["distance_to_rest"] = {"rotation_deg": round(_deg(rotation_distance(m, np.eye(4))), 3),
                                   "translation_mm": round(1000 * translation_distance(m, np.eye(4)), 3)}
    sleep_rot = _deg(rotation_distance(m, SLEEP_HEAD_POSE))
    sleep_tr = 1000 * translation_distance(m, SLEEP_HEAD_POSE)
    derived["distance_to_sdk_sleep_pose"] = {"rotation_deg": round(sleep_rot, 3), "translation_mm": round(sleep_tr, 3)}
    if sleep_rot <= NEAR_SLEEP_DEG and sleep_tr <= NEAR_SLEEP_MM:
        f.append(Finding("info", "near_sleep", f"the head is near the SDK sleep pose ({sleep_rot:.1f} deg, "
                                               f"{sleep_tr:.1f} mm)"))

    q_ik = None if body_yaw is None else np.asarray(ik_joints(new_kinematics(), m, body_yaw), dtype=float)
    raw = state_full.get("head_joints")
    measured = (isinstance(raw, list) and len(raw) == 7
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in raw))
    lim = urdf_joint_limits()
    lo = np.array([lim[n][0] for n in JOINT_NAMES])
    hi = np.array([lim[n][1] for n in JOINT_NAMES])
    joints: dict[str, Any] = {}
    if measured:
        q = np.array(raw, dtype=float)
        joints["source"] = "measured: /api/state/full?with_head_joints=true (the daemon's cached motor positions)"
        if q_ik is None:
            joints["ik_of_pose_vs_joints_max_deg"] = None  # skipped: the body yaw is unknown (no_body_yaw blocks)
        elif np.all(np.isfinite(q_ik)):
            gap = _deg(float(np.max(np.abs(q_ik[1:] - q[1:]))))
            joints["ik_of_pose_vs_joints_max_deg"] = round(gap, 3)
            if gap > FK_JOINT_TOL_DEG:
                f.append(Finding("block", "fk_joint_mismatch", f"the reported head pose (the daemon's FK) and the joints "
                                                               f"disagree by {gap:.2f} deg (> {FK_JOINT_TOL_DEG:g}): a "
                                                               f"pose-space command (SDK goto / wake_up, our ramp) "
                                                               f"would start from the wrong pose and jump"))
        else:
            f.append(Finding("block", "ik_nan", "the reported head pose has no IK solution although the joints were "
                                                "read: the daemon's pose estimate is unusable - do not start"))
    else:
        f.append(Finding("attention", "no_joint_reading", f"no head_joints in the answer ({raw!r}): using the IK of "
                                                          f"the reported pose as an estimate"))
        if q_ik is None:  # no joints and no body yaw: nothing to estimate from (no_body_yaw blocks)
            derived["joints"] = joints
            return derived, f
        if not np.all(np.isfinite(q_ik)):
            f.append(Finding("block", "ik_nan", "the reported head pose has no IK solution: the joints cannot be "
                                                "estimated - do not start"))
            derived["joints"] = joints
            return derived, f
        q = q_ik
        joints["source"] = "estimate: IK of the reported head pose with the daemon's engine (computed)"
    margin = np.minimum(q - lo, hi - q)
    joints["joints_deg"] = {n: round(_deg(v), 2) for n, v in zip(JOINT_NAMES, q)}
    joints["margin_deg"] = {n: round(_deg(v), 2) for n, v in zip(JOINT_NAMES, margin)}
    joints["min_stewart_margin_deg"] = round(_deg(float(np.min(margin[1:]))), 2)
    derived["joints"] = joints
    beyond = [(n, _deg(-v)) for n, v in zip(JOINT_NAMES, margin) if v < 0]
    if beyond:
        worst = max(x for _, x in beyond)
        text = ", ".join(f"{n} {x:.1f} deg" for n, x in beyond)
        joints["max_beyond_deg"] = round(worst, 2)
        limp = derived.get("motor_mode") == "disabled"
        if limp and worst <= LIMP_BEYOND_TOL_DEG:  # user decision 2026-11-17 (LIMP_BEYOND_TOL_DEG)
            f.append(Finding("attention", "start_beyond_limits_limp", f"the limp head rests past the URDF limits ({text}; "
                             f"<= {LIMP_BEYOND_TOL_DEG:g} deg): the connection pins it there like the official app's "
                             f"enable - the firmware may pull a joint back to its limit when the torque comes on (up "
                             f"to {worst:.1f} deg) - and Wake-up then moves it inside"))
        else:
            why = (f"more than {LIMP_BEYOND_TOL_DEG:g} deg past" if limp else "past them with the torque on")
            f.append(Finding("block", "start_beyond_limits", f"start joints beyond the URDF limits ({text}; {why}): no "
                             f"motor stage (PHASE4_PLAN §0)"))
    elif _deg(float(np.min(margin[1:]))) < MIN_START_MARGIN_DEG:
        f.append(Finding("attention", "start_near_limits", f"start joint margin {joints['min_stewart_margin_deg']} deg "
                                                           f"< {MIN_START_MARGIN_DEG:g} deg"))
    return derived, f


#: AWAKE start (user decision 2026-11-12): the robot woken the usual way holds INIT - head identity, antennas
#: -/+0.1745 rad (SDK INIT_ANTENNAS_JOINT_POSITIONS, reachy_mini.py:67 = the motion files' "rest"), body 0
#: (daemon wake_up, daemon/backend/abstract.py:1284-1323). The MEASURED pose sags under gravity with P-only servos
#: (HW3: pitch about 4 deg below the command), so it is compared with INIT within these bounds (initial values).
AWAKE_INIT_TOL = {"rotation_deg": 6.0, "translation_mm": 8.0, "antenna_deg": 6.0, "body_deg": 3.0}
INIT_ANTENNAS_DEG = (math.degrees(-0.1745), math.degrees(0.1745))
#: what the connection establishes on an AWAKE robot (torque already on, nothing re-pinned)
R1_CONNECTION_AWAKE = {
    "head_tracking_off": "a set face_target refuses the run (tracking running = a conflict); otherwise "
                         "stop_head_tracking / disable_wobbling with the torque ON at INIT: their IK uses the INIT target "
                         "(a missing target is the identity = INIT, abstract.py:556-561), no step; still check after",
    "no_partial_torque": "a per-motor torque change is not readable (robot/backend.py:646-667); a limp axis shows as "
                         "INIT error or motion in the still checks, which refuse the run",
}


def init_pose_error(report: dict[str, Any]) -> dict[str, float] | None:
    """Measured distance from INIT (R0 answers): head rotation / translation, worst antenna, body yaw; None if unread."""
    d = report.get("derived") or {}
    dist, ant, body = d.get("distance_to_rest") or {}, d.get("antennas_deg"), d.get("body_yaw_deg")
    if "rotation_deg" not in dist or not isinstance(ant, list) or len(ant) != 2 or not isinstance(body, (int, float)):
        return None
    return {"rotation_deg": float(dist["rotation_deg"]), "translation_mm": float(dist["translation_mm"]),
            "antenna_deg": max(abs(float(a) - r) for a, r in zip(ant, INIT_ANTENNAS_DEG)), "body_deg": abs(float(body))}


def r1_entry(report: dict[str, Any], mode: str = "sleep") -> dict[str, Any]:
    """R1 entry conditions of an R0 report in four categories (module docstring). Statuses: ``auto`` met / unmet,
    ``connection`` by_connection, ``operator`` operator_confirms, ``residual`` residual. Verdict ``blocked`` when an
    ``auto`` condition is unmet, else ``conditional``; ``auto_met`` says whether the automatic checks passed. Nothing
    here is ever "met" as a whole: the other categories are not verified by R0.

    ``mode``: "sleep" (default) - the motors are limp, the connection pins them; "awake" - woken the usual way: the
    motors are on and the robot stands at INIT (``AWAKE_INIT_TOL``), nothing is re-pinned."""
    awake = mode == "awake"
    d = report.get("derived") or {}
    tracking = d.get("head_tracking")
    motors = ("motors_enabled", d.get("motor_mode") == "enabled", f"reported motor mode {d.get('motor_mode')!r} "
              "(AWAKE: woken the usual way)") if awake else \
        ("motors_disabled", d.get("motor_mode") == "disabled", f"reported motor mode {d.get('motor_mode')!r}")
    auto = [
        ("r0_not_blocking", not report.get("blocking", True), "no blocking finding"),
        ("measured_robot", report.get("data_kind") == "measured", f"data_kind {report.get('data_kind')!r}"),
        motors,
        ("tracking_not_seen_on", tracking != "on",
         {"off": "no_media: the daemon has no media server, tracking cannot be enabled (abstract.py:683-689)",
          "on": "face_target is set: tracking is enabled and running (another client or app is active)"}.get(
             tracking, "face_target is the default: no sign of tracking (which does not prove it off: weight 0 or no "
                       "observation yet, abstract.py:683-699, 729) - the connection turns it off")),
    ]
    if awake:
        err = init_pose_error(report)
        auto.append(("at_init", err is not None and all(err[k] <= AWAKE_INIT_TOL[k] for k in AWAKE_INIT_TOL),
                     f"measured distance from INIT {err} (bounds {AWAKE_INIT_TOL}, sag allowed)" if err is not None
                     else "pose, antennas or body yaw not read"))
    rows = [{"name": n, "category": "auto", "status": "met" if ok else "unmet", "detail": t} for n, ok, t in auto]
    rows += [{"name": n, "category": "connection", "status": "by_connection", "detail": t}
             for n, t in (R1_CONNECTION_AWAKE if awake else R1_CONNECTION).items()]
    rows += [{"name": n, "category": "operator", "status": "operator_confirms", "detail": t}
             for n, t in R1_OPERATOR.items()]
    rows += [{"name": n, "category": "residual", "status": "residual", "detail": t} for n, t in R1_RESIDUAL.items()]
    auto_met = all(ok for _, ok, _ in auto)
    return {"verdict": "conditional" if auto_met else "blocked", "auto_met": auto_met, "conditions": rows, "mode": mode,
            "note": "conditional = the automatic R0 checks passed; the connection sequence establishes and re-checks "
                    "the 'connection' rows, the operator confirms the 'operator' rows right before the connection "
                    "(else no run), the 'residual' risks stay. Not a verification."}


def run_r0(client: DaemonClient, timeout_s: float | None = None) -> dict[str, Any]:
    """Collect the answers, evaluate them, return the report (no file written)."""
    answers = collect(client, timeout_s)
    derived, findings = evaluate(answers)
    report = {"stage": "R0", "daemon_url": client.base_url, "requests": "GET only: " + ", ".join(sorted(
        ("/api/daemon/status", "/api/kinematics/info", "/api/state/full?with_head_joints=true",
         "/api/state/present_body_yaw", "/api/motors/status"))),
              "generated_local": datetime.now().isoformat(timespec="seconds"), "data_kind": derived.get("data_kind"),
              "answers": answers, "derived": derived, "findings": [asdict(x) for x in findings],
              "blocking": any(x.level == "block" for x in findings)}
    report["r1_entry"] = r1_entry(report)
    return report


def report_markdown(report: dict[str, Any]) -> str:
    d = report.get("derived", {})
    lines = [f"# R0 read-only check ({report['generated_local']})", "",
             f"- daemon: `{report['daemon_url']}`, data_kind **{report.get('data_kind')}**, {report['requests']}",
             f"- version {d.get('daemon_version')}, state {d.get('daemon_state')}, IK engine {d.get('ik_engine')}, "
             f"motors {d.get('motor_mode')}",
             f"- head pose (deg / mm): {d.get('head_pose')}, body yaw {d.get('body_yaw_deg')} deg, "
             f"antennas {d.get('antennas_deg')} deg",
             f"- distance to rest {d.get('distance_to_rest')}, to the SDK sleep pose "
             f"{d.get('distance_to_sdk_sleep_pose')}",
             f"- joints ({(d.get('joints') or {}).get('source')}): {d.get('joints')}", "",
             f"**Blocking: {'YES' if report['blocking'] else 'no'}**", "", "| level | code | message |", "|---|---|---|"]
    lines += [f"| {x['level']} | {x['code']} | {x['message']} |" for x in report["findings"]]
    r1 = report.get("r1_entry") or {}
    lines += ["", f"**R1 entry: {r1.get('verdict', '-')}** (automatic checks "
              f"{'passed' if r1.get('auto_met') else 'NOT passed'}; never 'verified': {r1.get('note', '')})", "",
              "| category | condition | status | detail |", "|---|---|---|---|"]
    lines += [f"| {c.get('category', '-')} | {c['name']} | {c['status']} | {c['detail']} |"
              for c in r1.get("conditions", [])]
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pj, pm = out / "r0_report.json", out / "r0_report.md"
    pj.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pm.write_text(report_markdown(report), encoding="utf-8")
    return [pj, pm]
