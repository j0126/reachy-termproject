"""``SdkBackend``: the reachy_mini SDK client behind the Backend protocol - local MuJoCo now, robot gated.

target="sim" (data_kind "simulated", CLAUDE.md rule 4). ``connect()`` spawns OUR MuJoCo daemon
(``SimDaemon``: --sim, 127.0.0.1, robot_name hw4_sim; headless unless ``viewer=True``) and runs every
interlock before the first command:

1. ``daemon.start()`` - ``SimDaemon.start`` runs ``assert_local_sim`` on GET /api/daemon/status;
2. the same REST check again (``assert_local_sim(get_status(port))``) right before the client is built;
3. refuse if anything listens on [::1]:port: for a non-default robot_name ``localhost_only`` dials the
   literal "localhost" and ignores ``host`` (reachy_mini/reachy_mini.py:471-490), "localhost" resolves to
   ::1 first on this PC, and the constructor already sends SetAutomaticBodyYawCmd (reachy_mini.py:161);
4. ``ReachyMini(robot_name="hw4_sim", host="127.0.0.1", connection_mode="localhost_only",
   spawn_daemon=False, media_backend="no_media", timeout=10.0)`` - never the default "auto", which
   silently falls back to reachy-mini.local (reachy_mini.py:530-555). With the daemon's --no-media,
   DaemonStatus.no_media is True and the client does not ask the daemon to release media
   (daemon/daemon.py:84, reachy_mini.py:350-358);
5. ``assert_motion_client_is_local_sim`` on the client's own WebSocket;
6. only then commands (docs/ARCHITECTURE.md section 8): ``set_automatic_body_yaw(True)`` - explicit, the
   constructor sent the same default (reachy_mini.py:113, 161, 1096-1103) and the safety checker's IK
   assumes it; ``disable_wobbling()`` (reachy_mini.py:269-273; a no-op on the client without audio,
   media/media_manager.py:408-416, then SetWobblingCmd) and ``stop_head_tracking()``
   (reachy_mini.py:288-291) - both features add offsets to the daemon's IK input
   (daemon/backend/abstract.py:1745-1772).

Any failure after the daemon was spawned disconnects the client and stops the daemon.

target="robot" (data_kind "measured"): the constructor raises ``RobotApprovalError`` unless armed=True,
a human-written approval FILE validates against the request (``reachy_motion.safety.approval.gate_problems``:
content + unchanged file) and ``confirm(approval)`` returns True - or, as a PROPOSAL that is refused while
``safety.gui_approval.GUI_APPROVAL_ENABLED`` is False, a connection approved in the Motion Lab dialog
(``GuiApproval``: kind, host / port, validity; docs/lab/GUI_APPROVAL_POLICY.md) - checked before anything else, and the
gate runs AGAIN after the confirmation (the prompt has no time limit) and in ``connect()``
(``recheck_approval``; callers re-check before every run). ``connect()`` then runs the robot connection sequence
(``_connect_robot``: R0 GET -> ``backends.robot_connect`` -> approval re-check -> state-push stamps); constructing
ReachyMini against the robot daemon is itself a state change, made only for a person's approved connection
(CLAUDE.md rule 3, 2026-11-12).

SDK behaviour relied on (reachy_mini 1.11.0, paths relative to site-packages/reachy_mini):
  * ``set_target`` -> one SetFullTargetCmd with head, antennas and body_yaw (reachy_mini.py:610-670), sent
    fire-and-forget by ``WSClient.send_command`` (io/ws_client.py:145-159). Every exception it raises comes
    before or instead of the send (argument checks, a dead client, a closed socket), so a failed
    set_target is not recorded. The daemon IGNORES set_target while a move (goto) runs
    (daemon/backend/abstract.py:1661-1699): do not mix goto and streaming.
  * ``goto_target`` sends a TaskRequest (``send_task_request``, io/ws_client.py:268-280), then blocks until
    the daemon reports the task finished, timeout duration + 1 s (reachy_mini.py:705-722,
    ws_client.py:282-299). A failure BEFORE the send (argument checks, a dead client ws_client.py:270-271,
    a closed socket) leaves nothing to record. A failure AFTER it (TimeoutError ws_client.py:291-292, the
    daemon's "Task failed with error" 293-296) means the daemon received the goto and may still be running
    it: the command IS recorded, its index goes to ``unconfirmed`` and ``CommandNotConfirmedError`` (a
    BackendError) is raised. An exception that does not tell which (or Ctrl+C during the wait) is treated
    as "possibly sent" and recorded the same way. Its ``body_yaw=0.0`` default (reachy_mini.py:680) is never
    relied on: the body yaw is always passed explicitly.
  * Present state = the latest pushed JointPositionsMsg / HeadPoseMsg (io/ws_client.py:183-226,
    reachy_mini.py:926-959): joints [body_yaw, stewart_1..6]; the MuJoCo backend publishes antennas in
    the command convention [right, left] (ctrl = -target, present = -qpos,
    daemon/backend/mujoco/backend.py:260, 328-335). Pushes carry no timestamp and the client keeps no
    arrival time, so ``connect()`` wraps the client's ``_dispatch`` (the receive thread calls
    ``self._dispatch(msg)`` for every parsed message, io/ws_client.py:175-179) and stamps each
    JointPositionsMsg / HeadPoseMsg with the backend clock on arrival. ``PresentState.t_ns`` is the arrival
    time of the JointPositionsMsg the joints / antennas / body yaw come from (DEVELOPMENT_PLAN R10: client
    reception time), so reading one push twice gives the same t_ns. The daemon sends the head pose right
    after the joints of each control tick (daemon/backend/mujoco/backend.py:262-277,
    robot/backend.py:263-274), so ``head`` may be one push older than ``joints``.
  * ``close()`` disconnects like the Phase 1 smoke run: ``mini.client.disconnect()`` (io/ws_client.py:135-143).
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from reachy_motion.backends.base import (
    Backend,
    BackendError,
    Capabilities,
    CommandRecord,
    PresentState,
    check_command,
)
from reachy_motion.backends.sim_daemon import (
    DEFAULT_PORT,
    SIM_HOST,
    SIM_ROBOT_NAME,
    SimDaemon,
    assert_local_sim,
    assert_motion_client_is_local_sim,
    get_kinematics_info,
    get_status,
    port_answers,
)
from reachy_motion.executor.clock import Clock, RealClock
from reachy_motion.safety.approval import (
    ApprovalError,
    ApprovalRequest,
    RobotApproval,
    confirm_interactively,
    format_problems,
    gate_problems,
)

log = logging.getLogger(__name__)

SIM = "sim"
ROBOT = "robot"
TARGETS: tuple[str, ...] = (SIM, ROBOT)
BACKEND_NAMES: tuple[str, ...] = ("mock", SIM, ROBOT)
ROBOT_NOT_IN_PHASE2 = "real-robot connection is implemented in Phase 4, after separate approval"
SIM_DAEMON_LOG = "sim_daemon.log"
PUSH_TIMEOUT_S = 2.0  # present(): longest wait for the first state push after connect (the daemon pushes at 50 Hz)
PRESENT_T_NS = ("arrival of the JointPositionsMsg (stamped in the SDK receive thread with the backend clock); "
                "head = latest HeadPoseMsg")


class RobotApprovalError(ApprovalError):
    """The robot backend was refused: not armed, approval missing / invalid / not a file, or not confirmed."""


class CommandNotConfirmedError(BackendError):
    """A command reached (or may have reached) the daemon, but its completion was not confirmed.

    The command IS in ``SdkBackend.commands`` (at ``index``; ``record`` is that entry) and in
    ``SdkBackend.unconfirmed``: the daemon may still be executing it (a goto keeps running after the
    client's wait times out, and set_target is ignored while it runs).
    """

    def __init__(self, message: str, record: CommandRecord, index: int, sent: bool | None) -> None:
        super().__init__(message)
        self.record = record
        self.index = index
        self.sent = sent  # True: the SDK had sent it; None: unknown, treated as sent


def _require_robot_approval(armed: Any, approval: Any, request: Any,
                            confirm: Callable[[RobotApproval], bool] | None) -> None:
    """Raise RobotApprovalError unless every robot condition holds (cheapest first, interactive last).

    The full gate (content + unchanged human-written file) runs before the confirmation and AGAIN after
    it: the prompt has no time limit, so the approval may have expired - or its file changed - meanwhile.
    """
    if armed is not True:
        raise RobotApprovalError("refused: the robot target needs armed=True (CLAUDE.md rule 3)")
    from reachy_motion.safety import gui_approval as ga  # lazy: no Qt, nothing sent

    if isinstance(approval, ga.GuiApproval):
        # A connection approved in the Motion Lab dialog (a proposal: refused while GUI_APPROVAL_ENABLED is False,
        # docs/lab/GUI_APPROVAL_POLICY.md). The dialog is the person's confirmation: no terminal prompt.
        r = approval.request
        problems = ga.problems(approval, kind="connect", host=r.host, port=r.port)
        if problems:
            raise RobotApprovalError("robot approval refused (GUI): " + "; ".join(problems))
        return
    if not isinstance(approval, RobotApproval):
        raise RobotApprovalError("refused: the robot target needs a human-written approval file "
                                 "(reachy_motion.safety.approval.load_approval)")
    if not isinstance(request, ApprovalRequest):
        raise RobotApprovalError("refused: the robot target needs an ApprovalRequest describing the planned "
                                 "timelines and their safety reports")
    problems = gate_problems(approval, request)
    if problems:
        raise RobotApprovalError(format_problems(problems))
    confirm_fn = confirm_interactively if confirm is None else confirm
    if confirm_fn(approval) is not True:
        raise RobotApprovalError("refused: the approval was not confirmed by a person at the terminal")
    problems = gate_problems(approval, request)
    if problems:
        raise RobotApprovalError(format_problems(problems, "robot approval refused (re-checked after the "
                                                           "confirmation):"))


def _is_gui_approval(approval: Any) -> bool:
    """A Motion Lab dialog approval (``safety.gui_approval.GuiApproval``), not a file approval."""
    from reachy_motion.safety.gui_approval import GuiApproval

    return isinstance(approval, GuiApproval)


def _interpolation_technique(method: Any) -> Any:
    """Library easing name / Easing member -> SDK InterpolationTechnique (utils/interpolation.py:108-114)."""
    from reachy_mini.utils.interpolation import InterpolationTechnique  # lazy: the SDK import takes ~1 s

    try:
        return InterpolationTechnique(getattr(method, "value", method))
    except (ValueError, TypeError):
        names = ", ".join(t.value for t in InterpolationTechnique)
        raise BackendError(f"unknown interpolation method {method!r} (expected one of {names})") from None


def _goto_was_sent(exc: BaseException) -> bool | None:
    """Had ``ReachyMini.goto_target`` put its TaskRequest on the wire when it raised ``exc``?

    True  - raised while waiting, after the send: TimeoutError (io/ws_client.py:291-292), the daemon's
            "Task failed with error: ..." (293-296), "Task not found." (286-287);
    False - raised before the send: argument checks and the GotoTaskRequest model (ValueError / TypeError,
            reachy_mini.py:695-719), a dead client (ConnectionError, ws_client.py:270-271), no socket
            (AssertionError, ws_client.py:277), a closed socket (websockets ConnectionClosed from ``ws.send``,
            ws_client.py:278: websockets/sync/connection.py:905-1012 raises it when the connection is not open
            or the frame could not be written);
    None  - anything else: unknown.
    """
    message = str(exc)
    if isinstance(exc, TimeoutError) or message.startswith("Task failed with error:") or message == "Task not found.":
        return True
    if isinstance(exc, (ValueError, TypeError, ConnectionError, AssertionError)):
        return False
    try:
        from websockets.exceptions import ConnectionClosed  # SDK dependency (io/ws_client.py:19, 180)
    except ImportError:  # pragma: no cover
        return None
    return False if isinstance(exc, ConnectionClosed) else None


class _PushStamps:
    """Arrival times (backend clock) of the daemon's state pushes, written by the SDK receive thread."""

    def __init__(self) -> None:
        self.lock = threading.Lock()  # held while the client cache and the stamps change, and while reading both
        self.joints_ns: int | None = None  # arrival of the latest JointPositionsMsg
        self.head_ns: int | None = None  # arrival of the latest HeadPoseMsg
        self.pushes = 0
        self.ready = threading.Event()  # set once both kinds arrived after the hook was installed


def _stamp_state_pushes(client: Any, clock: Clock) -> _PushStamps:
    """Wrap ``client._dispatch`` so that every JointPositionsMsg / HeadPoseMsg is stamped on arrival.

    reachy_mini 1.11.0: the receive thread parses each message and calls ``self._dispatch(msg)``
    (io/ws_client.py:175-179), which only replaces the cached message (183-190) - no arrival time is kept.
    The method is looked up on the instance, so an instance attribute wraps it; ``ReachyMini`` assigns its
    client once (reachy_mini.py:157), so the wrapper stays in place for the connection. Sends nothing. The
    cache update and the stamp happen under one lock that ``present()`` takes to read both consistently.
    """
    from reachy_mini.io.protocol import HeadPoseMsg, JointPositionsMsg  # io/protocol.py:995-1007

    original = getattr(client, "_dispatch", None)
    if not callable(original):
        raise RuntimeError("the SDK client has no _dispatch (reachy_mini 1.11.0 io/ws_client.py:183): state "
                           "arrival times cannot be stamped")
    stamps = _PushStamps()

    def dispatch(msg: Any) -> None:
        t_ns = clock.now_ns()  # message parsed just now (ws_client.py:175): arrival + JSON parsing
        with stamps.lock:
            original(msg)
            if isinstance(msg, JointPositionsMsg):
                stamps.joints_ns = t_ns
            elif isinstance(msg, HeadPoseMsg):
                stamps.head_ns = t_ns
            else:
                return
            stamps.pushes += 1
            if stamps.joints_ns is not None and stamps.head_ns is not None:
                stamps.ready.set()

    client._dispatch = dispatch
    return stamps


class SdkBackend:
    """Backend over ``reachy_mini.ReachyMini``: ``target="sim"`` (local MuJoCo) or ``"robot"`` (Phase 4)."""

    def __init__(self, target: str, *, port: int = DEFAULT_PORT, viewer: bool = False, armed: bool = False,
                 approval: RobotApproval | None = None, request: ApprovalRequest | None = None,
                 confirm: Callable[[RobotApproval], bool] | None = None,
                 client_factory: Callable[..., Any] | None = None,
                 daemon_factory: Callable[..., Any] | None = None,
                 log_dir: str | Path | None = None, clock: Clock | None = None) -> None:
        if target == ROBOT:
            # The approval gate runs before ANY other attribute is set or anything is constructed.
            _require_robot_approval(armed, approval, request, confirm)
        elif target != SIM:
            raise ValueError(f"unknown target {target!r} (expected 'sim' or 'robot')")
        elif armed or approval is not None or request is not None or confirm is not None:
            raise ValueError("armed / approval / request / confirm apply to the robot target only")

        self.target = target
        self.name = target
        self.port = int(port)
        self._clock: Clock = RealClock() if clock is None else clock
        self.commands: list[CommandRecord] = []
        self.unconfirmed: dict[int, str] = {}  # index in commands -> why its completion was not confirmed
        self.connection_info: dict[str, Any] = {}
        #: robot: the operator confirmed the controlled environment right before the connection (robot_r1
        #: operator_confirms_environment); the connection sequence refuses without it. Not a verification.
        self.operator_confirmed = False
        #: robot: "sleep" (limp start, the connection pins the motors) or "awake" (woken the usual way, at INIT)
        self.start_mode = "sleep"
        self.push_timeout_s = PUSH_TIMEOUT_S
        self._mini: Any = None
        self._daemon: Any = None
        self._stamps: _PushStamps | None = None
        if target == ROBOT:
            self.data_kind = "measured"
            self.capabilities = Capabilities(torque_control=True, real_hardware=True)
            self.viewer = False
            self.log_dir = None
            self.approval: Any = approval  # RobotApproval (file + terminal ID) or a GUI approval (gui_approval)
            self.request: Any = request
            if _is_gui_approval(approval):  # GUI approval: host / port are the approved ones
                self.host, self.port = approval.request.host, int(approval.request.port)
            else:
                self.host = request.host  # type: ignore[union-attr]  # validated above
            # No factory is kept: in Phase 2 nothing may build a client or a daemon for the robot.
            self._client_factory = None
            self._daemon_factory = None
        else:
            self.data_kind = "simulated"
            # MuJoCo motors are always enabled: mode/torque calls are no-ops (daemon/backend/mujoco/backend.py:337-347)
            self.capabilities = Capabilities(torque_control=False, real_hardware=False)
            self.viewer = bool(viewer)
            self.log_dir = None if log_dir is None else Path(log_dir)
            self.approval = None
            self.request = None
            self.host = SIM_HOST
            self._client_factory = client_factory
            self._daemon_factory = daemon_factory

    # ------------------------------------------------------------------ lifecycle
    @property
    def connected(self) -> bool:
        return self._mini is not None

    def recheck_approval(self) -> None:
        """Robot: run the whole approval gate again now, raise RobotApprovalError on any problem.

        Time passes between construction and use (the approval may expire, its file may change):
        ``connect()`` calls this, and Phase 4 must call it before every run. No-op for the sim target.
        """
        if self.target != ROBOT:
            return
        if _is_gui_approval(self.approval):  # GUI approval of this connection
            from reachy_motion.safety import gui_approval as ga

            gp = ga.problems(self.approval, kind="connect", host=self.host, port=self.port)
            if gp:
                raise RobotApprovalError("robot approval refused (GUI, re-checked before use): " + "; ".join(gp))
            return
        problems = gate_problems(self.approval, self.request)  # type: ignore[arg-type]
        if problems:
            raise RobotApprovalError(format_problems(problems, "robot approval refused (re-checked before use):"))

    def _connect_robot(self) -> None:
        """The robot connection that ``connect()`` runs (wired 2026-11-12, user decision: CLAUDE.md rule 3).

        Order (``backends.robot_connect``): the approval gate again -> R0, GET only (``robot_r0``) -> the connection
        sequence, which refuses an enabled or blocked robot, a blocked R1 entry (its automatic checks) or an
        environment the operator did not confirm (``operator_confirmed``) BEFORE the SDK constructor runs,
        turns tracking / wobbling off and syncs the target to the present pose with the torque off, then pins with
        the global ``enable_motors()`` -> the approval gate again (time passed during the still checks) ->
        state-push stamps. The client is the SDK's ``ReachyMini`` in network mode on the approved host (never "auto",
        never a spawned daemon)."""
        if self.target != ROBOT:
            raise BackendError("_connect_robot is for the robot target only")
        if self._mini is not None:
            return
        self.recheck_approval()
        from reachy_mini import ReachyMini  # reachy_mini/__init__.py exports it

        from reachy_motion.backends.daemon_client import DaemonClient
        from reachy_motion.backends.robot_connect import robot_connect_sequence
        from reachy_motion.robot_r0 import run_r0

        r0 = run_r0(DaemonClient(self.host, port=self.port, timeout_s=3.0), 3.0)
        mini, report = robot_connect_sequence(ReachyMini, r0, clock=self._clock, host=self.host, port=self.port,
                                              operator_confirmed=self.operator_confirmed is True, mode=self.start_mode)
        self.connection_info = {"r0_blocking": r0.get("blocking"), "r0_findings": r0.get("findings"),
                                "r1_entry": report.entry, "operator_confirmed": report.operator_confirmed,
                                "residual": report.residual, "connect": report.to_dict()}
        if mini is None:
            raise BackendError("robot connection refused: " + "; ".join(report.problems))
        try:
            self.recheck_approval()
        except BaseException:
            mini.client.disconnect()
            raise
        self._stamps = _stamp_state_pushes(mini.client, self._clock)
        self._mini = mini

    def connect(self) -> None:
        if self.target == ROBOT:  # user decision 2026-11-12 (CLAUDE.md rule 3): the human-approved robot connection
            self._connect_robot()
            return
        if self._mini is not None:
            return
        factory = SimDaemon if self._daemon_factory is None else self._daemon_factory
        log_path = None if self.log_dir is None else self.log_dir / SIM_DAEMON_LOG
        daemon = factory(port=self.port, headless=not self.viewer, log_path=log_path)
        mini = None
        try:
            start_status = daemon.start()  # SimDaemon.start: assert_local_sim before it returns
            rest_status = assert_local_sim(get_status(self.port))
            if port_answers("::1", self.port):
                raise RuntimeError(f"something listens on [::1]:{self.port}; refusing to build a client "
                                   "(the SDK dials 'localhost', which resolves to ::1 first)")
            mini = self._new_client()
            socket_status = assert_motion_client_is_local_sim(mini)  # before ANY command on this socket
            stamps = _stamp_state_pushes(mini.client, self._clock)  # receive side only: sends nothing
            mini.set_automatic_body_yaw(True)
            mini.disable_wobbling()
            mini.stop_head_tracking()
            kinematics = get_kinematics_info(self.port)
        except BaseException:
            self._cleanup(mini, daemon)
            raise
        self._daemon, self._mini, self._stamps = daemon, mini, stamps
        self.connection_info = {
            "data_kind": self.data_kind, "target": self.target, "host": SIM_HOST, "port": self.port,
            "viewer": self.viewer, "connected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "daemon_status_at_start": start_status, "daemon_status_rest": rest_status,
            "daemon_status_socket": socket_status, "kinematics_info": kinematics,
            "client_init": {"automatic_body_yaw": True, "wobbling": False, "head_tracking": False},
            "present_t_ns": PRESENT_T_NS,
            "daemon_log": None if log_path is None else str(log_path),
        }

    def _new_client(self) -> Any:
        factory = self._client_factory
        if factory is None:
            from reachy_mini import ReachyMini  # lazy: only the sim target ever builds a client

            factory = ReachyMini
        return factory(robot_name=SIM_ROBOT_NAME, host=SIM_HOST, port=self.port, connection_mode="localhost_only",
                       spawn_daemon=False, media_backend="no_media", timeout=10.0)

    @staticmethod
    def _cleanup(mini: Any, daemon: Any) -> None:
        try:
            if mini is not None:
                mini.client.disconnect()
        except Exception as e:  # keep going: the daemon must be stopped in any case
            log.warning("client disconnect failed during cleanup: %s", e)
        finally:
            if daemon is not None:
                daemon.stop()

    def close(self) -> None:
        """Disconnect the client, then stop the daemon. Idempotent."""
        mini, daemon = self._mini, self._daemon
        self._mini = self._daemon = None
        self._stamps = None
        if mini is not None or daemon is not None:
            self._cleanup(mini, daemon)

    def __enter__(self) -> "SdkBackend":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ commands
    def _client(self) -> Any:
        if self._mini is None:
            raise BackendError("not connected (call connect() first)")
        return self._mini

    def _record_unconfirmed(self, record: CommandRecord, reason: str) -> int:
        self.commands.append(record)
        index = len(self.commands) - 1
        self.unconfirmed[index] = reason
        log.warning("%s recorded without confirmation (%s): the daemon may still be running it", record.kind, reason)
        return index

    def set_target(self, head: np.ndarray, antennas: Sequence[float], body_yaw: float) -> None:
        mini = self._client()
        h, a, b = check_command(head, antennas, body_yaw)
        t_ns = self._clock.now_ns()
        try:
            mini.set_target(head=h, antennas=[a[0], a[1]], body_yaw=float(b))
        except Exception as e:  # raised before / instead of the send (module doc): nothing to record
            raise BackendError(f"set_target failed: {e}") from e
        self.commands.append(CommandRecord(t_ns=t_ns, kind="set_target", head=h.copy(), antennas=a, body_yaw=b))

    def goto_target(self, head: np.ndarray, antennas: Sequence[float], body_yaw: float, duration: float,
                    method: str) -> None:
        mini = self._client()
        h, a, b = check_command(head, antennas, body_yaw)
        d = float(duration)
        if not (math.isfinite(d) and d > 0.0):
            raise BackendError(f"goto duration must be > 0, got {duration!r}")
        technique = _interpolation_technique(method)
        record = CommandRecord(t_ns=self._clock.now_ns(), kind="goto_target", head=h.copy(), antennas=a, body_yaw=b,
                               duration=d, method=technique.value)
        try:
            mini.goto_target(head=h, antennas=[a[0], a[1]], duration=d, method=technique, body_yaw=float(b))
        except Exception as e:
            sent = _goto_was_sent(e)
            if sent is False:
                raise BackendError(f"goto_target not sent: {e}") from e
            index = self._record_unconfirmed(record, f"{type(e).__name__}: {e}")
            what = "sent but its completion was not confirmed" if sent else "outcome unknown (recorded as sent)"
            raise CommandNotConfirmedError(f"goto_target {what}: {e} - the daemon may still be running it",
                                           record, index, sent) from e
        except BaseException as e:  # Ctrl+C during the blocking wait: the goto may be running
            self._record_unconfirmed(record, f"interrupted ({type(e).__name__}) while waiting for completion")
            raise
        self.commands.append(record)

    def disable_motors(self) -> str:
        """Global torque off through the SDK (``ReachyMini.disable_motors()``: SetTorqueCmd(on=False),
        reachy_mini.py:1074-1086; robot/backend.py:378-383), then the motor mode read back on the motion socket
        (``robot_connect.send_torque_off``). Returns "off" | "off_unconfirmed" | "unknown". The robot target is
        locked like every command (``_client``); the MuJoCo backend ignores torque commands
        (daemon/backend/mujoco/backend.py:337-347), so the sim reads back "off_unconfirmed"."""
        from reachy_motion.backends.robot_connect import send_torque_off

        mini = self._client()
        lines: list[str] = []
        state = send_torque_off(mini, lines)
        self.connection_info.setdefault("torque_off_log", []).extend(lines)
        return state

    def health(self) -> dict[str, Any] | None:
        """The daemon's latest status message on the motion socket (DaemonStatus, pushed about once per second;
        io/ws_client.py:228-234 ``get_status(wait=False)`` returns the cached one): state, error, motor mode.
        None while no status arrived yet. Read by the executor's S3 check; sends nothing."""
        mini = self._client()
        try:
            st = mini.client.get_status(wait=False)
        except Exception:  # noqa: BLE001 - AssertionError before the first status: unknown, not an error
            return None
        from reachy_motion.backends.robot_connect import _status_fields

        backend = getattr(st, "backend_status", None)
        mode = getattr(backend, "motor_control_mode", None)
        state = getattr(st, "state", None)
        return {"state": str(getattr(state, "value", state)), "error": getattr(st, "error", None) or getattr(
                backend, "error", None), "motor_mode": None if mode is None else str(getattr(mode, "value", mode)),
                "status_age_s": None,
                # tracking running (another controller aiming the head): robot_connect._status_fields
                "face_target_set": bool(_status_fields(st)["face_target_set"])}

    def present(self) -> PresentState:
        """Latest pushed state; ``t_ns`` = arrival of its JointPositionsMsg (module doc), not the read time."""
        mini = self._client()
        stamps = self._stamps
        if stamps is None or not stamps.ready.wait(self.push_timeout_s):  # only the first call can wait
            raise BackendError(f"no state push from the daemon within {self.push_timeout_s:g} s after connect")
        try:
            with stamps.lock:  # the receive thread replaces the cache and its stamp under this lock
                t_ns = stamps.joints_ns
                head = np.asarray(mini.get_current_head_pose(), dtype=np.float64).reshape(4, 4)
                head_joints, antennas = mini.get_current_joint_positions()
            joints = np.asarray(head_joints, dtype=np.float64).reshape(7)
            ant = (float(antennas[0]), float(antennas[1]))
        except Exception as e:
            raise BackendError(f"present state unavailable: {e}") from e
        return PresentState(t_ns=int(t_ns), head=head, antennas=ant, body_yaw=float(joints[0]), joints=joints,
                            data_kind=self.data_kind)


def make_backend(name: str, **kw: Any) -> Backend:
    """'mock' -> MockBackend(**kw) (imported lazily), 'sim' / 'robot' -> SdkBackend(name, **kw)."""
    if name == "mock":
        from reachy_motion.backends.mock import MockBackend  # lazy: not needed for sim / robot

        return MockBackend(**kw)
    if name in TARGETS:
        return SdkBackend(name, **kw)
    raise ValueError(f"unknown backend {name!r} (expected one of {', '.join(BACKEND_NAMES)})")
