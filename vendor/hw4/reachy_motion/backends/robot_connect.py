"""Robot connection sequence (Phase 4 preparation). NOT wired: ``SdkBackend.connect()`` still refuses the robot.

What the SDK 1.11.0 sends while a client connects (docs/phase4/STARTUP_STOP.md §8; paths relative to
``site-packages/reachy_mini/``):

* ``ReachyMini(...)`` constructor, network mode, ``media_backend="no_media"``:
  C1 WebSocket connect + wait for the first joint / pose pushes - receive only (reachy_mini.py:582-586,
     io/ws_client.py:60-112);
  C2 ``SetAutomaticBodyYawCmd`` (reachy_mini.py:161, 1103) - the daemon only sets a kinematics flag, no IK request
     (daemon/backend/abstract.py:1159-1166);
  C3 status read - receive only (reachy_mini.py:342);
  C4 ``POST /api/media/release`` when the daemon has media (reachy_mini.py:350-357) - the daemon stops its media
     pipeline and relay (daemon/daemon.py:139-152), nothing motion-related. The backend keeps its media-server
     object (it is only cleared in ``close()``, abstract.py:481).
  The constructor therefore requests no IK and sets no target: it cannot move the head by itself.
* Commands a client sends afterwards that DO request IK: ``disable_wobbling`` -> ``set_speech_offsets(0)``
  (abstract.py:1744-1751, 668-680; on the robot because the media-server object stays) and ``stop_head_tracking``
  -> ``clear_tracking_aim`` (abstract.py:702-719). With no head target the IK of a missing target is the identity
  pose = REST (abstract.py:555-560), and with the torque on that joint goal is written at the next control tick
  (daemon/backend/robot/backend.py:203-210): a step to REST. With the torque off nothing is written.
* The global ``enable_motors()`` (no ``ids``) pins every target to the last read joints, writes them to the goal
  registers and then turns the torque on (robot/backend.py:352-377); when already enabled it does nothing (608-616).
  A per-motor enable skips the pin (abstract.py:1786-1792).
* What the pin does not cover (docs/phase4/STARTUP_STOP.md §8.1, real-SDK harness tests/test_sdk_offline_connect.py):
  an aim another client left running (tracking with a lost face aims at ``INIT_HEAD_POSE`` = identity) re-requests IK
  at the next tick and overrides the pin (abstract.py:729-756, robot/backend.py:249-256); and nothing locks the control
  thread's IK against the pin, so an IK result computed from the old target before the pin can land after it
  (abstract.py:586-594) whenever ``ik_required`` is set - IK never clears it.

Sequence (``robot_connect_sequence``) - the precondition is checked BEFORE the constructor runs, because the
constructor already sends C2 / C4 (a state change) and nothing after it can undo a jump that happened during it:

0. R0 (GET only, ``robot_r0``) just before, sorted by ``robot_r0.r1_entry`` (four categories, user decision
   2026-11-12): its ``auto`` conditions pass (no blocking finding, a measured robot, the motors DISABLED, tracking
   not seen on); the operator has confirmed the controlled environment at the terminal right before the connection
   (``operator_confirmed``: one client, no app, people and power switch ready - not confirmed, no run); the
   ``residual`` risks (another client) are written into the report, never "verified". The ``connection`` rows
   (tracking off, no partial torque) are not preconditions: steps 3 and 6-8 establish and re-check them. An
   enabled robot is refused: its daemon target cannot be read (dropped from /api/state/full).
1. construct: network mode, the approved host, port 8000, ``spawn_daemon=False``, ``media_backend="no_media"``.
2. on the motion socket: not a simulation, version 1.11.0, the approved host, motors still disabled (a race with
   another client) - otherwise disconnect, nothing else sent.
3. aim sources off WITH THE TORQUE OFF: ``stop_head_tracking``, ``disable_wobbling``. Their IK results are not
   written while the torque is off, and an aim left running by another client is gone before the pin.
4. re-measure: a still check with the torque off (``still_s / 2``: a head that moves now is moved by something
   else), then the present state - joints inside the URDF limits, pose and joints agree (``startup_procedure``).
   A LIMP head may rest up to ``robot_r0.LIMP_BEYOND_TOL_DEG`` past the limits (user decision 2026-11-17: the
   mechanical rest after the daemon's goto_sleep or a boot); step 8 then allows that much more movement.
5. target sync, torque still off: ``set_target(present pose, present body yaw)``. Step 3 left ``ik_required`` set,
   so the control thread keeps running IK; with the target synced, an IK result landing after the pin is the
   present pose (within the SDK's FK/IK round trip, 0.45 deg at the SDK sleep joints) instead of REST.
6. on the motion socket again: motors still disabled, ``face_target`` the default (tracking not turned back on).
7. global ``enable_motors()``: the daemon pins all targets to the present joints again, writes the goal registers,
   then turns the torque on - this also re-torques a per-motor enable left by another client (608-626).
8. still check with the torque on: joints within ``settle_tol_deg`` for ``still_s`` (a snap, a sag, or another client's
   commands show here) - on failure nothing more is sent (the daemon holds its target) and the report says so.
   Another client's commands are only DETECTED here, not prevented: a running move also makes step 5 a silent no-op
   (abstract.py:1661-1699) and keeps writing after the pin.

Every step is logged in ``ConnectReport.log``. Nothing here enables the torque partially or retries; the only pose
sent before the pin is the present one, with the torque off.

Torque state at the end (``torque_state``: never_on / on / off / off_unconfirmed / unknown). A stop after the enable
(step 8, a communication error, a stop request) is never followed by an AUTOMATIC torque-off: with the torque off the
head goes wherever gravity takes it. ``confirm_torque_off`` (the operator supports the head and types 'off',
``confirm_torque_off_interactively``) sends the global ``disable_motors()`` and reads the motor mode back; a torque-off
that cannot be read back is ``off_unconfirmed`` (a disconnected client drops commands without an error,
io/ws_client.py:135-160). Otherwise the report carries ``OPERATOR_PROCEDURE_KO``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from reachy_motion.settings import ROBOT_DAEMON_PORT, ROBOT_HOST

__all__ = ["EXPECTED_VERSION", "ConnectReport", "robot_connect_sequence", "CONSTRUCT_KWARGS", "C1Report", "run_c1",
           "C1_HOLD_S", "C1_OFF_WATCH_S", "OPERATOR_PROCEDURE_KO", "TORQUE_OFF_PROMPT_KO",
           "confirm_torque_off_interactively", "send_torque_off"]

EXPECTED_VERSION = "1.11.0"
STILL_S = 1.0
STILL_TOL_DEG = 0.5  # nothing sent by us yet: a head that moves more is moved by something else
#: After our own command (the torque enable; tracking / wobbling off at INIT) the head may settle a little: the robot
#: moved 0.53 deg right after a clean enable twice on 2026-11-12 (results/lab/*_robot_connection). More than this is a
#: snap or another client's command.
SETTLE_TOL_DEG = 2.0
#: Status messages read after the torque-off: the first may predate the command (get_status(wait=True) returns at once
#: when a message arrived since the last read, io/ws_client.py:194-196, 228-234; 2026-11-12 it read 'enabled').
TORQUE_OFF_READS = 3
OFF_DRIFT_WARN_DEG = 1.0  # a head that moves more after the torque-off was not resting where it was switched off
#: Constructor arguments besides host / port: never "auto" (it falls back to other hosts), never spawn a daemon.
CONSTRUCT_KWARGS: dict[str, Any] = {"connection_mode": "network", "spawn_daemon": False,
                                    "media_backend": "no_media", "automatic_body_yaw": True}
#: What the operator does when a run ends with the torque on or in an unknown state (printed and recorded).
OPERATOR_PROCEDURE_KO = (
    "1) 머리를 손으로 받친다: 토크가 꺼지면 머리는 중력으로 쉬던 자세까지 내려간다 (REST에서 최대 약 29°·51 mm). "
    "2) 받친 채 토크를 끈다: 프롬프트에 'off' 입력. 연결이 끊겼거나 'off'가 확인되지 않으면 전원 스위치로 끈다. "
    "3) 머리를 쉬던 자세에 천천히 내려놓는다. "
    "4) 원인 확인: 다른 클라이언트·앱이 의심되면 그 PC·앱부터 끈다. "
    "5) 다시 시작할 때는 R0부터 (GET만) 하고, 상태가 바뀌었으면 승인 양식도 다시 만든다.")
TORQUE_OFF_PROMPT_KO = ("토크가 켜져 있거나 상태를 알 수 없습니다. 머리를 손으로 받쳤으면 'off'를 입력하세요 (전체 토크 끄기). "
                        "다른 입력이면 토크를 그대로 둡니다 (그때는 머리를 받치고 전원 스위치로 끄기): ")


def confirm_torque_off_interactively(prompt: str = TORQUE_OFF_PROMPT_KO, input_fn: Callable[[str], str] | None = None,
                                     isatty_fn: Callable[[], bool] | None = None) -> bool:
    """The operator, holding the head, types 'off' at a terminal. No terminal (scripts, agents) -> False."""
    import builtins
    import sys

    try:
        tty = bool((isatty_fn or sys.stdin.isatty)())
    except (AttributeError, OSError, ValueError):
        tty = False
    if not tty:
        print("torque-off NOT confirmed: stdin is not an interactive terminal (the torque is left as it is)")
        return False
    try:
        answer = (input_fn or builtins.input)(prompt)
    except (EOFError, KeyboardInterrupt):
        return False
    return isinstance(answer, str) and answer.strip().lower() == "off"


def send_torque_off(mini: Any, log: list[str], reads: int = TORQUE_OFF_READS) -> str:
    """Global ``disable_motors()`` (SetTorqueCmd(on=False), reachy_mini.py:1074-1086; robot/backend.py:378-383), then
    the motor mode read back on the motion socket, over up to ``reads`` status messages (the first one may be older
    than the command: ``TORQUE_OFF_READS``). Returns "off" only when the daemon reports "disabled"."""
    try:
        mini.disable_motors()
    except Exception as exc:  # noqa: BLE001 - recorded, the operator procedure takes over
        log.append(f"disable_motors() failed: {exc!r}")
        return "unknown"
    mode = None
    for n in range(1, max(1, int(reads)) + 1):
        try:
            mode = _status_fields(mini.client.get_status(wait=True, timeout=2.0))["motor_mode"]
        except Exception as exc:  # noqa: BLE001
            log.append(f"disable_motors() sent; the motor mode could not be read back: {exc!r}")
            return "off_unconfirmed"
        if mode == "disabled":
            break
    log.append(f"disable_motors() sent; motor mode read back: {mode!r} (status message {n} of up to {reads})")
    return "off" if mode == "disabled" else "off_unconfirmed"


@dataclass
class ConnectReport:
    ok: bool = False
    stage: str = "precondition"  # where it ended
    problems: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    still_drift_deg: list[float] = field(default_factory=list)
    torque_enabled_by_us: bool = False
    entry: dict[str, Any] = field(default_factory=dict)  # robot_r0.r1_entry of the R0 the sequence started from
    operator_confirmed: bool = False  # the controlled environment, confirmed by the operator (not a verification)
    residual: list[str] = field(default_factory=list)  # residual risks carried by this run (never verified)
    torque_state: str = "never_on"  # never_on | on | off | off_unconfirmed | unknown (module docstring)
    operator_procedure: str = ""  # set when the torque is (or may be) on at the end

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _operator_torque_off(mini: Any, rep: Any, confirm: Callable[[str], bool] | None) -> None:
    """The torque is (or may be) on and the head may not rest where gravity would take it: no automatic torque-off.
    With ``confirm`` the operator supports the head and confirms; then the global torque-off is sent and read back."""
    rep.operator_procedure = OPERATOR_PROCEDURE_KO
    if confirm is None:
        rep.log.append(f"torque {rep.torque_state}: NOT switched off automatically - operator procedure")
        return
    try:
        ok = bool(confirm(TORQUE_OFF_PROMPT_KO))
    except (Exception, KeyboardInterrupt):  # noqa: BLE001
        ok = False
    if not ok:
        rep.log.append(f"the operator did not confirm a torque-off: torque {rep.torque_state} - operator procedure")
        return
    rep.log.append("the operator supports the head and confirmed the torque-off")
    rep.torque_state = send_torque_off(mini, rep.log)
    if rep.torque_state == "off":
        rep.operator_procedure = ""


def _status_fields(status: Any) -> dict[str, Any]:
    backend = getattr(status, "backend_status", None)
    face = getattr(status, "face_target", None)
    return {"version": getattr(status, "version", None), "simulation": getattr(status, "simulation_enabled", None),
            "mockup": getattr(status, "mockup_sim_enabled", None),
            "motor_mode": getattr(getattr(backend, "motor_control_mode", None), "value",
                                  getattr(backend, "motor_control_mode", None)),
            "state": getattr(getattr(status, "state", None), "value", getattr(status, "state", None)),
            # set only while tracking is enabled and observing (robot_r0 module docstring)
            "face_target_set": face is not None and (getattr(face, "detected", False) is True or any(
                getattr(face, k, None) is not None for k in ("x", "y", "roll", "ts")))}


def _present(mini: Any) -> Any:
    from reachy_motion.backends.base import PresentState

    joints, antennas = mini.get_current_joint_positions()
    head = np.asarray(mini.get_current_head_pose(), dtype=float).reshape(4, 4)
    q = np.asarray(joints, dtype=float).reshape(7)
    return PresentState(t_ns=0, head=head, antennas=(float(antennas[0]), float(antennas[1])), body_yaw=float(q[0]),
                        joints=q, data_kind="measured")


def _still(mini: Any, clock: Any, seconds: float, tol_deg: float) -> float:
    first = np.asarray(mini.get_current_joint_positions()[0], dtype=float)
    worst = 0.0
    t_end = clock.now_ns() + int(round(seconds * 1e9))
    while clock.now_ns() < t_end:
        clock.sleep_until(min(t_end, clock.now_ns() + 20_000_000))
        q = np.asarray(mini.get_current_joint_positions()[0], dtype=float)
        worst = max(worst, math.degrees(float(np.max(np.abs(q - first)))))
    return worst


def robot_connect_sequence(make_client: Callable[..., Any], r0: dict[str, Any], *, clock: Any,
                           host: str = ROBOT_HOST, port: int = ROBOT_DAEMON_PORT, still_s: float = STILL_S,
                           tol_deg: float = STILL_TOL_DEG, operator_confirmed: bool = False,
                           confirm_torque_off: Callable[[str], bool] | None = None, mode: str = "sleep",
                           settle_tol_deg: float = SETTLE_TOL_DEG) -> tuple[Any | None, ConnectReport]:
    """Steps 0-8 of the module docstring. Returns (client or None, report); a refused or failed sequence never
    leaves a connected client behind, and after step 7 it never sends anything else once a check fails - except the
    torque-off the operator confirms (``confirm_torque_off``, module docstring).

    ``mode="awake"`` (user decision 2026-11-12): the robot was woken the usual way (daemon ``wake_up``,
    daemon/backend/abstract.py:1284-1323) and holds INIT with the torque ON. Steps 3-8 are replaced by
    ``_awake_steps``: still check FIRST (another motion running refuses before anything is sent), joints and INIT
    distance, a set face_target refuses (tracking running), then tracking / wobbling off - with the torque on at INIT
    their IK input is the INIT target (a missing target is the identity = INIT, abstract.py:556-561), so nothing
    steps - and a still check. No ``set_target`` sync (it would command the sagged measured pose) and no
    ``enable_motors()`` (already enabled: nothing is re-pinned)."""
    from reachy_motion.robot_r0 import LIMP_BEYOND_TOL_DEG, r1_entry
    from reachy_motion.startup_procedure import check_start

    awake = mode == "awake"
    rep = ConnectReport()
    derived = r0.get("derived", {}) if isinstance(r0, dict) else {}
    pre = []
    if not isinstance(r0, dict) or r0.get("blocking", True):
        pre.append("the R0 report is missing or has a blocking finding")
    if isinstance(r0, dict) and r0.get("data_kind") != "measured":
        pre.append(f"R0 data_kind is {r0.get('data_kind')!r}, not a measured robot")
    if awake and derived.get("motor_mode") != "enabled":
        pre.append(f"the motors are {derived.get('motor_mode')!r}: the AWAKE start needs a robot woken the usual way")
    elif not awake and derived.get("motor_mode") != "disabled":
        pre.append(f"the motors are {derived.get('motor_mode')!r}: the sequence starts only from limp motors "
                   f"(an enabled robot with an unknown daemon target could move as soon as IK is requested; a robot "
                   f"woken the usual way uses mode='awake')")
    if isinstance(r0, dict) and str(r0.get("daemon_url", "")).split("//")[-1].split(":")[0] not in ("", host):
        pre.append(f"R0 was read from {r0.get('daemon_url')}, not {host}")
    # R1 entry, recomputed from the report itself (a stored verdict is never trusted): four categories
    entry = r1_entry(r0, mode) if isinstance(r0, dict) else {"verdict": "blocked", "auto_met": False, "conditions": []}
    rep.entry = entry
    if entry["verdict"] == "blocked":
        unmet = [c["name"] for c in entry["conditions"] if c["category"] == "auto" and c["status"] != "met"]
        pre.append("R1 entry blocked by the automatic R0 checks: " + (", ".join(unmet) or "no R0 report"))
    rep.operator_confirmed = operator_confirmed is True
    if not rep.operator_confirmed:
        pre.append("the operator has not confirmed the controlled environment (this PC the only client, no app, "
                   "people and power switch ready): if it cannot be controlled, the run does not happen")
    rep.residual = [c["name"] for c in entry["conditions"] if c["category"] == "residual"]
    if pre:
        rep.problems = pre
        rep.log.append("refused before constructing the client (nothing sent)")
        return None, rep
    rep.log.append("R1 entry " + entry["verdict"] + "; connection establishes: " + ", ".join(
        c["name"] for c in entry["conditions"] if c["category"] == "connection") + "; operator confirmed the "
        "environment; residual risks carried by this run (not verified, not prevented): " + ", ".join(rep.residual))

    rep.stage = "construct"
    rep.log.append(f"construct ReachyMini(host={host!r}, port={port}, {CONSTRUCT_KWARGS})")
    mini = make_client(host=host, port=port, **CONSTRUCT_KWARGS)
    try:
        rep.stage = "verify"
        st = _status_fields(mini.client.get_status(wait=True, timeout=2.0))
        rep.log.append(f"status on the motion socket: {st}")
        bad = []
        if st["simulation"] is not False or st["mockup"] is True:
            bad.append(f"not the robot (simulation={st['simulation']}, mockup={st['mockup']})")
        if st["version"] != EXPECTED_VERSION:
            bad.append(f"daemon version {st['version']!r}")
        if getattr(mini.client, "host", host) != host:
            bad.append(f"the client is connected to {getattr(mini.client, 'host', None)!r}")
        if awake:
            if st["motor_mode"] == "enabled":
                rep.torque_state = "on"  # woken the usual way; this sequence never changes the torque
            else:
                bad.append(f"the motors are {st['motor_mode']!r} now: not awake (changed since R0)")
            if st["face_target_set"]:
                bad.append("face_target is set: head tracking is running (a control conflict) - no run")
            if bad:
                raise _Stop(bad)
            _awake_steps(mini, rep, clock, still_s, tol_deg, check_start, settle_tol_deg)
            rep.ok, rep.stage = True, "ready"
            rep.log.append("ready (AWAKE): the robot stays at INIT with the torque on; trials run INIT -> INIT")
            return mini, rep
        if st["motor_mode"] != "disabled":
            bad.append(f"the motors are {st['motor_mode']!r} now (changed since R0)")
        if bad:
            raise _Stop(bad)

        rep.stage = "aim_sources_off"
        mini.stop_head_tracking()
        mini.disable_wobbling()
        rep.log.append("stop_head_tracking(), disable_wobbling() with the torque off (an IK result is not written)")

        rep.stage = "present"
        drift = _still(mini, clock, still_s / 2, tol_deg)
        rep.still_drift_deg.append(drift)
        if drift > tol_deg:
            raise _Stop([f"the head moved {drift:.2f} deg with the torque off (> {tol_deg:g}): something moves it"])
        present = _present(mini)
        chk = check_start(present, beyond_tol_deg=LIMP_BEYOND_TOL_DEG)  # limp: a little past the limits is the rest
        rep.log.append(f"present check: ok={chk.ok}, ik gap {chk.ik_gap_deg}, margin {chk.min_stewart_margin_deg}")
        if not chk.ok:
            raise _Stop(chk.problems)
        settle = settle_tol_deg
        if chk.max_beyond_deg > 0:  # user decision 2026-11-17 (robot_r0.LIMP_BEYOND_TOL_DEG)
            settle = settle_tol_deg + chk.max_beyond_deg
            rep.log.append(f"the limp head rests {chk.max_beyond_deg:.1f} deg past the URDF limits (<= "
                           f"{LIMP_BEYOND_TOL_DEG:g}): the enable pins it there like the official app's; the firmware "
                           f"may pull a joint back to its limit, so the torque-on still check allows {settle:.1f} deg")

        rep.stage = "sync"
        mini.set_target(head=np.asarray(present.head, dtype=float), body_yaw=float(present.body_yaw))
        rep.log.append("set_target(present pose, present body yaw) with the torque off")
        st = _status_fields(mini.client.get_status(wait=True, timeout=2.0))
        bad = []
        if st["motor_mode"] != "disabled":
            bad.append(f"the motors are {st['motor_mode']!r} now (changed during the connection)")
        if st["face_target_set"]:
            bad.append("face_target is set after stop_head_tracking: tracking is on again (another client)")
        if bad:
            raise _Stop(bad)

        rep.stage = "enable"
        rep.torque_state = "unknown"  # from here the enable may have been sent even if the call raises
        mini.enable_motors()  # global: the daemon pins all targets to the present joints, writes the goals, torque on
        rep.torque_enabled_by_us = True
        rep.torque_state = "on"
        rep.log.append("enable_motors() (global: the daemon pins all targets to the present joints again)")
        drift = _still(mini, clock, still_s, settle)
        rep.still_drift_deg.append(drift)
        if drift > settle:
            raise _Stop([f"the head moved {drift:.2f} deg after the torque came on (> {settle:g})"], hold=True)
    except _Stop as stop:
        rep.problems = stop.problems
        rep.log.append("stopped: nothing more is sent" + (
            "; the torque stays ON and the daemon holds its current target (the pin, unless another client changed it)"
            if stop.hold else "; the torque was never enabled"))
        if rep.torque_state != "never_on":
            _operator_torque_off(mini, rep, confirm_torque_off)
        _disconnect(mini, rep)
        return None, rep
    except (Exception, KeyboardInterrupt) as exc:  # a communication error or a stop request: recorded, never "safe"
        what = "stop requested" if isinstance(exc, KeyboardInterrupt) else "error"
        rep.problems = [f"{what} at stage {rep.stage}: {exc!r}"]
        if rep.torque_state == "on" and not isinstance(exc, KeyboardInterrupt):
            rep.torque_state = "unknown"  # the link may be gone: the torque cannot be read back
        rep.log.append(f"{what}: torque {rep.torque_state}")
        if rep.torque_state != "never_on":
            _operator_torque_off(mini, rep, confirm_torque_off)
        _disconnect(mini, rep)
        return None, rep
    except BaseException:
        _disconnect(mini, rep)
        raise
    rep.ok, rep.stage = True, "ready"
    rep.log.append("ready: the start-up procedure may run (startup_procedure.run_startup)")
    return mini, rep


def init_error(present: Any) -> dict[str, float]:
    """Measured distance of a present state from INIT (head identity, SDK INIT antennas, body 0)."""
    from reachy_motion.library.pose import rotation_distance, translation_distance
    from reachy_motion.robot_r0 import INIT_ANTENNAS_DEG

    head = np.asarray(present.head, dtype=float).reshape(4, 4)
    return {"rotation_deg": math.degrees(rotation_distance(head, np.eye(4))),
            "translation_mm": 1000.0 * translation_distance(head, np.eye(4)),
            "antenna_deg": max(abs(math.degrees(a) - r) for a, r in zip(present.antennas, INIT_ANTENNAS_DEG)),
            "body_deg": abs(math.degrees(present.body_yaw))}


def _awake_steps(mini: Any, rep: ConnectReport, clock: Any, still_s: float, tol_deg: float, check_start: Any,
                 settle_tol_deg: float = SETTLE_TOL_DEG) -> None:
    """``robot_connect_sequence(mode="awake")`` after the motion-socket check: nothing is sent until the head is
    seen still at INIT; then only the two aim-source switches, and a still check."""
    from reachy_motion.robot_r0 import AWAKE_INIT_TOL

    rep.stage = "still"
    drift = _still(mini, clock, still_s / 2, tol_deg)
    rep.still_drift_deg.append(drift)
    if drift > tol_deg:
        raise _Stop([f"the head moved {drift:.2f} deg with nothing sent by us (> {tol_deg:g}): another motion is "
                     "running"])
    rep.stage = "present"
    present = _present(mini)
    chk, err = check_start(present), init_error(present)
    rep.log.append(f"present check: ok={chk.ok}, margin {chk.min_stewart_margin_deg}; distance from INIT "
                   f"{ {k: round(v, 2) for k, v in err.items()} } (bounds {AWAKE_INIT_TOL})")
    bad = [] if chk.ok else list(chk.problems)
    bad += [f"not at INIT: {k} {err[k]:.2f} > {AWAKE_INIT_TOL[k]:g}" for k in AWAKE_INIT_TOL if err[k] > AWAKE_INIT_TOL[k]]
    if bad:
        raise _Stop(bad)
    rep.stage = "aim_sources_off"
    mini.stop_head_tracking()
    mini.disable_wobbling()
    rep.log.append("stop_head_tracking(), disable_wobbling() with the torque ON at INIT: their IK input is the INIT "
                   "target - no step; no set_target sync, no enable_motors (nothing re-pinned)")
    st = _status_fields(mini.client.get_status(wait=True, timeout=2.0))
    bad = []
    if st["motor_mode"] != "enabled":
        bad.append(f"the motors are {st['motor_mode']!r} now (changed during the connection)")
    if st["face_target_set"]:
        bad.append("face_target is set after stop_head_tracking: tracking is on again (another client)")
    if bad:
        raise _Stop(bad)
    rep.stage = "still_after"
    drift = _still(mini, clock, still_s, settle_tol_deg)
    rep.still_drift_deg.append(drift)
    if drift > settle_tol_deg:
        raise _Stop([f"the head moved {drift:.2f} deg after tracking / wobbling off (> {settle_tol_deg:g})"])


C1_HOLD_S = 3.0  # C1: watch the held pose this long with the torque on, sending nothing
C1_OFF_WATCH_S = 2.0  # then watch after the torque went off (where gravity takes the head back)


@dataclass
class C1Report:
    ok: bool = False
    stage: str = "connect"  # connect | hold | torque_off | done
    problems: list[str] = field(default_factory=list)
    connect: dict[str, Any] = field(default_factory=dict)  # the connection sequence's own report
    start_joints_deg: list[float] = field(default_factory=list)  # right after the global enable
    start_antennas_deg: list[float] = field(default_factory=list)
    hold_head_drift_deg: float | None = None  # largest joint change during the hold (torque on, nothing sent)
    hold_antenna_drift_deg: float | None = None
    torque_off_sent: bool = False
    off_head_drift_deg: float | None = None  # after the torque went off (offline models have no gravity)
    end_joints_deg: list[float] = field(default_factory=list)
    end_antennas_deg: list[float] = field(default_factory=list)
    torque_state: str = "never_on"  # never_on | on | off | off_unconfirmed | unknown (module docstring)
    operator_confirmed: bool = False
    residual: list[str] = field(default_factory=list)  # residual risks carried by this run (never verified)
    operator_procedure: str = ""  # set when the torque is (or may be) on at the end
    warnings: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _watch(mini: Any, clock: Any, seconds: float) -> tuple[float, float]:
    """(largest head-joint change, largest antenna change) in deg against the first reading, over ``seconds``."""
    q0, a0 = (np.asarray(v, dtype=float) for v in mini.get_current_joint_positions())
    head = ant = 0.0
    t_end = clock.now_ns() + int(round(seconds * 1e9))
    while clock.now_ns() < t_end:
        clock.sleep_until(min(t_end, clock.now_ns() + 20_000_000))
        q, a = (np.asarray(v, dtype=float) for v in mini.get_current_joint_positions())
        head = max(head, math.degrees(float(np.max(np.abs(q - q0)))))
        ant = max(ant, math.degrees(float(np.max(np.abs(a - a0)))))
    return head, ant


def run_c1(make_client: Callable[..., Any], r0: dict[str, Any], *, clock: Any, host: str = ROBOT_HOST,
           port: int = ROBOT_DAEMON_PORT, hold_s: float = C1_HOLD_S, off_watch_s: float = C1_OFF_WATCH_S,
           tol_deg: float = STILL_TOL_DEG, operator_confirmed: bool = False,
           confirm_torque_off: Callable[[str], bool] | None = None) -> C1Report:
    """C1 (docs/phase4/C1_R1_CHECKLIST.md): torque on where the head already is, and nothing else.

    1. ``robot_connect_sequence`` steps 0-8: R0 entry (automatic checks, the operator's confirmation, residual risks
       recorded), client, aim sources off with the torque OFF, target synced to the present pose, global
       ``enable_motors()`` (the daemon pins every target, head and antennas, to the present joints), still check.
    2. Hold ``hold_s``: no command at all; the head joints and the antennas are watched against their reading
       right after the enable (``tol_deg``).
    3. Global torque off (``send_torque_off``: ``disable_motors()`` and the motor mode read back) at that same pose,
       the pose the head rested at before the enable, so gravity has nowhere to take it; then a watch of
       ``off_watch_s`` (a larger movement is a warning: the head was not resting where it was switched off).

    A failure, a communication error or a stop request in 2 sends nothing automatically: the torque stays on (or
    its state is unknown), the operator supports the head and may confirm the torque-off (``confirm_torque_off``),
    else the report carries the operator procedure. The CLI calls this only when ``robot_r1.ROBOT_RUN_ENABLED`` is
    True (it is False)."""
    rep = C1Report()
    mini, crep = robot_connect_sequence(make_client, r0, clock=clock, host=host, port=port, tol_deg=tol_deg,
                                        operator_confirmed=operator_confirmed, confirm_torque_off=confirm_torque_off)
    rep.connect = crep.to_dict()
    rep.operator_confirmed, rep.residual = crep.operator_confirmed, list(crep.residual)
    if mini is None:
        rep.problems = list(crep.problems)
        rep.torque_state, rep.operator_procedure = crep.torque_state, crep.operator_procedure
        rep.log.append("C1 stopped in the connection sequence")
        return rep
    rep.torque_state = "on"
    try:
        q0, a0 = mini.get_current_joint_positions()
        rep.start_joints_deg = [round(math.degrees(v), 4) for v in q0]
        rep.start_antennas_deg = [round(math.degrees(v), 4) for v in a0]
        rep.stage = "hold"
        rep.log.append(f"hold {hold_s:g} s with the torque on, nothing sent")
        rep.hold_head_drift_deg, rep.hold_antenna_drift_deg = _watch(mini, clock, hold_s)
        if max(rep.hold_head_drift_deg, rep.hold_antenna_drift_deg) > tol_deg:
            rep.problems = [f"moved during the hold: head {rep.hold_head_drift_deg:.2f} deg, antennas "
                            f"{rep.hold_antenna_drift_deg:.2f} deg (> {tol_deg:g}): nothing more is sent "
                            "automatically; the torque stays ON and the daemon holds its target"]
            _operator_torque_off(mini, rep, confirm_torque_off)
            return rep
        rep.stage = "torque_off"
        rep.torque_off_sent = True
        rep.torque_state = send_torque_off(mini, rep.log)
        if rep.torque_state != "off":
            rep.problems = [f"the torque-off was not confirmed by the daemon ({rep.torque_state})"]
            rep.operator_procedure = OPERATOR_PROCEDURE_KO
            return rep
        rep.off_head_drift_deg, _ = _watch(mini, clock, off_watch_s)
        if rep.off_head_drift_deg > OFF_DRIFT_WARN_DEG:
            rep.warnings.append(f"the head moved {rep.off_head_drift_deg:.2f} deg after the torque-off "
                                f"(> {OFF_DRIFT_WARN_DEG:g}): it was not resting where it was switched off")
        q1, a1 = mini.get_current_joint_positions()
        rep.end_joints_deg = [round(math.degrees(v), 4) for v in q1]
        rep.end_antennas_deg = [round(math.degrees(v), 4) for v in a1]
        rep.ok, rep.stage = True, "done"
        return rep
    except (Exception, KeyboardInterrupt) as exc:  # a communication error or a stop request: recorded, never "safe"
        stop = isinstance(exc, KeyboardInterrupt)
        rep.problems = [f"{'stop requested' if stop else 'error'} at stage {rep.stage}: {exc!r}"]
        if rep.torque_state == "on" and not stop:
            rep.torque_state = "unknown"  # the link may be gone: the torque cannot be read back
        if rep.torque_state != "off":
            _operator_torque_off(mini, rep, confirm_torque_off)
        return rep
    finally:
        _disconnect(mini, rep)  # type: ignore[arg-type]


class _Stop(Exception):
    def __init__(self, problems: list[str], hold: bool = False) -> None:
        super().__init__("; ".join(problems))
        self.problems, self.hold = list(problems), hold


def _disconnect(mini: Any, rep: ConnectReport) -> None:
    try:
        mini.client.disconnect()  # client side only: closes the WebSocket, sends no command (io/ws_client.py:135-143)
        rep.log.append("client disconnected")
    except Exception as exc:  # noqa: BLE001
        rep.log.append(f"disconnect failed: {exc!r}")
