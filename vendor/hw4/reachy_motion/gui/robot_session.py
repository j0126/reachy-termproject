"""The Motion Lab's real-robot session (Qt-free): ONE SDK client, every action from a person's click, every motion
from its own GUI approval - and three locks in front of all of it (user decision 2026-11-12, docs/lab/GUI_ROBOT.md).

``RobotLabSession`` is a ``LabSession`` (one worker thread, a job queue, the live queue, ``save_run`` and the
Results tab's analysis) whose jobs are the robot's:

* ``status``     R0 again - the allowed GETs only (``robot_r0.run_r0``, report under results/phase4/r0/<time>/ like
                 ``robot-status``) - plus, while connected, the motion socket's own status and present state (nothing
                 sent);
* ``robot_connect``  the connection sequence (``backends.robot_connect``; ``SdkBackend._connect_robot``) in the start
                 mode R0 reads: "sleep" (limp: tracking / wobbling off with the torque off, target = the present pose,
                 global ``enable_motors()`` pins it - no motion) or "awake" (torque on at INIT: nothing re-pinned);
* ``wake_up``    the SDK client's wake-up gotos (``sdk_wake_up``: INIT 2 s, roll 20 deg and back) from where the
                 head is, recorded (commands = the gotos sent, present = measured);
* ``trial``      ONE R1 / R2 / R3 trial of the awake robot: ``robot_r1.build_plan(trial, None, awake=True)`` +
                 ``robot_r1.run_trial`` (INIT -> trial -> INIT, recorded in runs/ like every Motion Lab run, analysed
                 for the Results tab), with the Stop button (controlled stop S1) and the live view;
* ``torque_off`` the global torque-off after the operator confirmed that they support the head (read back, watched
                 2 s), then the client disconnects;
* ``disconnect`` the client only (the robot keeps its torque and target).

Single client: this worker is the only place that talks to the robot; jobs run one after another, so wake-up, trials
and status reads never overlap. Another program cannot be prevented (no SDK lock, no GET lists the clients:
``robot_r0.R1_RESIDUAL``); it is detected where it shows: the connection refuses a set ``face_target`` (tracking) or
a moving head, and every wake-up / trial re-reads the motion socket (motors enabled, no face_target, daemon not in
error) and checks that the head is still before sending (``_conflict_problems``, ``watch_still``).

Gates, checked in this order before anything is sent (``_gates``), then the approval of THIS action:

1. ``safety.gui_approval.GUI_APPROVAL_ENABLED`` - the GUI approval policy (True since 2026-11-12, CLAUDE.md rule 3);
2. ``robot_r1.ROBOT_RUN_ENABLED`` - the robot run switch (True since 2026-11-12);
3. ``SdkBackend`` itself: it accepts a GUI approval only while 1. is True, re-checks it, and ``connect()`` runs the
   robot connection sequence (``SdkBackend._connect_robot``).

Every action still needs its own approval from the person at the GUI. Tests drive the same jobs offline only: the
real SDK 1.11.0 daemon code (``link_factory`` / a stubbed ``reachy_mini.ReachyMini``: the offline harness, data_kind
``computed``), never the network.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from reachy_motion.gui.lab import LAB_ROOT, LabError, LabSession, new_session_dir, tick_summary
from reachy_motion.library.pose import RobotState, pose_from_matrix
from reachy_motion.safety.policy import DEFAULT_POLICY, SafetyPolicy
from reachy_motion.settings import ROBOT_DAEMON_PORT, ROBOT_HOST

DEG = 180.0 / math.pi
R0_ROOT = Path("results") / "phase4" / "r0"
ROBOT_RUN_LOCKED = "실행 스위치가 꺼져 있습니다: robot_r1.ROBOT_RUN_ENABLED = False (실제 로봇 연결·구동 없음)"
SDK_LOCKED = "SdkBackend가 실제 로봇 연결을 거부했습니다 (NotImplementedError): 아무것도 보내지 않음"
WAKE_START_TOL_DEG = 1.0  # the wake-up model was checked from these joints: the head must still be there
STATE_KO = {
    "disconnected": "연결 안 됨",
    "rest_torque_on": "연결됨 · 토크 켜짐 · 쉬던 자세에 고정 → 다음: Wake-up (승인)",
    "at_init": "INIT에서 대기 (깨어 있음, 토크 켜짐) → 시행 가능 (시행마다 승인)",
    "holding": ("정지·이상: 토크 켜짐(또는 모름), 마지막 목표 유지 → 머리를 받치고 '토크 OFF', 또는 'Wake-up'(승인)으로 "
                "INIT 복귀"),
}
LINK_FRESH_S = 1.0  # a state push newer than this: the SDK link answers (the daemon pushes about 50 times a second)
#: How a connection attempt ended, from what is actually known (``connect_outcome``).
CONNECT_OUTCOME_KO = {
    "refused_before_send": "명령 전송 전 거부: 아무것도 보내지 않음",
    "failed_after_partial_send": "일부 명령 전송 후 실패: 토크 켜기(enable_motors)는 보내지 않음",
    "failed_after_enable": "토크 켜기(enable_motors)를 보낸 뒤 실패: 토크가 켜져 있을 수 있음",
    "comm_lost_torque_unknown": "통신 끊김: 토크 상태를 확인하지 못함",
    "connected": "정상 연결",
}
NOT_TORQUE_OFF_KO = "연결 끊기·창 닫기는 SDK 연결만 닫습니다. 토크 OFF가 아닙니다 (토크를 끄려면 머리를 받치고 '토크 OFF')."
SINGLE_CLIENT_KO = ("이 GUI가 로봇의 유일한 제어 프로그램이어야 합니다: 같은 로봇을 CLI(robot-r1 등)·대시보드 앱·시작 앱·다른 PC에서 "
                    "동시에 제어하지 마세요. SDK에는 제어 잠금이 없어 막을 수 없고, 정지 확인·추종 오차로 뒤늦게 보일 뿐입니다.")


class RobotLabSession(LabSession):
    """See the module docstring. ``link_factory(host=, port=, approval=, r0=, mode=, clock=)`` returns a connected
    backend (tests: the offline harness); default ``SdkBackend("robot")`` with the GUI approval - locked.
    ``r0_reader(host, port)`` returns an R0 report (default: the allowed GETs)."""

    TARGETS = ("robot",)

    def __init__(self, root: Path | str = LAB_ROOT, on_event: Callable[[dict[str, Any]], None] | None = None,
                 host: str = ROBOT_HOST, port: int = ROBOT_DAEMON_PORT, policy: SafetyPolicy = DEFAULT_POLICY,
                 link_factory: Callable[..., Any] | None = None,
                 r0_reader: Callable[[str, int], dict[str, Any]] | None = None, clock: Any = None,
                 r0_root: Path | str = R0_ROOT, speaker_factory: Callable[[str, int], Any] | None = None) -> None:
        from reachy_motion.executor.clock import RealClock

        self.host = str(host)
        self.link_factory, self.r0_reader = link_factory, r0_reader
        #: the robot speaker (``sound.RobotSpeaker``: daemon REST); tests pass a stub - never the network
        self.speaker_factory = speaker_factory
        self._speaker_obj: Any = None  # one per connection (its upload cache goes with it)
        self.clock = RealClock() if clock is None else clock
        self.r0_root = Path(r0_root)
        self.robot_state = "disconnected"
        self.torque_state = "unknown"  # as this PC knows it: on | off | off_unconfirmed | unknown
        self.start_mode: str | None = None
        self.last_r0: dict[str, Any] | None = None
        self.last_r0_path: Path | None = None
        self.used_approvals: set[str] = set()
        self.r3_runs: list[str] = []  # R3 run folders of this connection (the next step's "after")
        self.log_dir: Path | None = None  # robot_log of this connection: connect / wake-up / torque-off records
        self.operator_procedure = ""
        # What the screen may say about the torque: only a read-back taken AFTER the last event that may have changed
        # it counts (``torque_view``); otherwise UNKNOWN, never a guessed OFF.
        self._torque_epoch = 0
        self._torque_event = "아직 읽지 않음 ('상태 읽기 (GET)')"
        self._readback: dict[str, Any] | None = None
        self._link_note = "없음 (SDK 클라이언트를 만들지 않음)"
        super().__init__("robot", root, on_event, port=port, viewer=False, policy=policy)

    # ------------------------------------------------------------------ GUI-thread API (non-blocking)
    def read_status(self) -> None:
        """R0 (GET only) and, while connected, the motion socket's status - nothing is sent."""
        self._submit("status")

    def connect_robot(self, approval: Any = None) -> None:
        """None (the GUI's click): the job reads the robot (GET) and approves the connection itself."""
        self._submit("robot_connect", approval=approval)

    def wake_up(self, approval: Any = None) -> None:
        """None (the GUI's click): the job checks the wake-up from where the head is and approves it itself."""
        self._submit("wake_up", approval=approval)

    def run_trial(self, trial: Any, session_dir: Path | str, notes: str = "", approval: Any = None) -> None:  # type: ignore[override]
        self._submit("trial", trial=trial, session_dir=Path(session_dir), notes=str(notes or ""), approval=approval)

    def torque_off(self, confirmed: bool) -> None:
        """``confirmed``: the operator supports the head and typed 'off' (the dialog)."""
        self._submit("torque_off", confirmed=confirmed is True)

    def disconnect(self) -> None:
        self._submit("disconnect")

    @property
    def connected(self) -> bool:
        return self.backend is not None

    # ------------------------------------------------------------------ helpers (worker thread)
    def _gates(self) -> list[str]:
        from reachy_motion import robot_r1
        from reachy_motion.safety import gui_approval as ga

        out = []
        if not ga.enabled():
            out.append(ga.POLICY_NOT_CONFIRMED)
        if robot_r1.ROBOT_RUN_ENABLED is not True:
            out.append(ROBOT_RUN_LOCKED)
        return out

    def _refuse(self, job: str, reasons: list[str], label: str | None = None) -> None:
        """Refused BEFORE anything was sent to the robot (allowed GETs at most)."""
        self._emit("refused", job=job, label=label, reasons=list(reasons), nothing_sent=True)

    def _torque_changed(self, event: str) -> None:
        """Something that may have changed the torque happened: earlier read-backs no longer count."""
        self._torque_epoch += 1
        self._torque_event = event

    def _note_readback(self, mode: Any, source: str) -> None:
        """A motor mode actually read from the daemon (R0 GET, a fresh SDK status, the torque-off read-back)."""
        if mode is None:
            return
        self._readback = {"mode": str(getattr(mode, "value", mode)), "source": source,
                          "at": datetime.now().strftime("%H:%M:%S"), "epoch": self._torque_epoch}

    def torque_view(self) -> dict[str, str]:
        """{"torque": ON | OFF | UNKNOWN, "torque_basis": why, "link": the SDK connection object}: ON / OFF only from a
        read-back newer than the last torque-relevant event (a command sent, a failure), else UNKNOWN."""
        rb = self._readback
        if rb is not None and rb["epoch"] == self._torque_epoch:
            torque = {"enabled": "ON", "disabled": "OFF"}.get(rb["mode"], "UNKNOWN")
            basis = f"{rb['source']}에서 {rb['at']}에 읽음: {rb['mode']}"
        else:
            torque, basis = "UNKNOWN", f"확인 안 됨 - {self._torque_event}"
        return {"torque": torque, "torque_basis": basis, "link": self._link_note}

    def _set_state(self, state: str, **extra: Any) -> None:
        self.robot_state = state
        self._emit("robot_state", state=state, state_ko=STATE_KO.get(state, state), torque_state=self.torque_state,
                   operator_procedure=self.operator_procedure, **self.torque_view(), **extra)

    def _log(self, kind: str, body: dict[str, Any]) -> Path:
        """One record of this connection (robot_log/NN_kind.json, data_kind stated in it)."""
        from reachy_motion.gui.model import _json_default

        if self.log_dir is None:
            self.log_dir = new_session_dir("robot", "connection", self.root)
        n = 1 + sum(1 for _ in self.log_dir.glob("*.json"))
        p = self.log_dir / f"{n:02d}_{kind}.json"
        body = dict(body, kind=kind, host=self.host, port=self.port,
                    generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        p.write_text(json.dumps(body, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        return p

    def _read_r0(self) -> tuple[dict[str, Any], Path]:
        """R0: the allowed GETs only (CLAUDE.md rule 3), written like ``robot-status`` writes it."""
        from reachy_motion.robot_r0 import write_report

        if self.r0_reader is not None:
            rep = self.r0_reader(self.host, self.port)
        else:
            from reachy_motion.backends.daemon_client import DaemonClient
            from reachy_motion.robot_r0 import run_r0

            rep = run_r0(DaemonClient(self.host, port=self.port, timeout_s=3.0), 3.0)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out, i = self.r0_root / stamp, 1
        while out.exists():
            i += 1
            out = self.r0_root / f"{stamp}_{i}"
        path = write_report(rep, out)[0]
        self.last_r0, self.last_r0_path = rep, path
        self._note_readback((rep.get("derived") or {}).get("motor_mode"), "R0 GET")  # None when nothing answered
        return rep, path

    def _socket_status(self) -> dict[str, Any]:
        """The connected client's view (nothing sent): the daemon status message and the latest pushed state."""
        out: dict[str, Any] = {}
        try:
            out["health"] = self.backend.health()
        except Exception as exc:  # noqa: BLE001
            out["health_error"] = f"{type(exc).__name__}: {exc}"
        try:
            p = self.backend.present()
            out["present"] = present_brief(p)
            out["present_age_s"] = age = max(0.0, (self.clock.now_ns() - int(p.t_ns)) / 1e9)
        except Exception as exc:  # noqa: BLE001
            out["present_error"] = f"{type(exc).__name__}: {exc}"
            self._link_note = f"SDK 객체 남음 · 상태를 읽지 못함 ({type(exc).__name__})"
            self._torque_changed("SDK 상태를 읽지 못함 - 지금 토크를 확인할 수 없음")
            return out
        if age <= LINK_FRESH_S:  # the link answers: the cached daemon status is current enough to read the mode
            self._link_note = f"열림 (SDK 객체 있음 · 마지막 수신 {age:.1f} s 전)"
            self._note_readback((out.get("health") or {}).get("motor_mode"), "SDK 소켓 상태")
        else:  # a silent link: an earlier read-back no longer says what the torque is now
            self._link_note = f"SDK 객체 남음 · 응답 없음 (마지막 수신 {age:.1f} s 전)"
            self._torque_changed(f"SDK 응답 없음 (마지막 수신 {age:.1f} s 전) - 지금 토크를 확인할 수 없음")
        return out

    def _conflict_problems(self) -> list[str]:
        """Another controller where it shows (nothing sent): torque changed, tracking on, daemon error."""
        try:
            h = self.backend.health()
        except Exception as exc:  # noqa: BLE001
            return [f"daemon 상태를 읽을 수 없음: {type(exc).__name__}: {exc}"]
        if not isinstance(h, dict):
            return ["daemon 상태 메시지를 아직 받지 못했습니다 (잠시 뒤 다시)"]
        out = []
        if str(h.get("motor_mode")) != "enabled":
            out.append(f"모터 모드 {h.get('motor_mode')!r}: 깨어 있지 않음 (다른 프로그램이 토크를 바꿨을 수 있음)")
        if h.get("face_target_set"):
            out.append("face_target이 설정됨: 헤드 트래킹이 실행 중 (다른 제어 프로그램) - 실행하지 않음")
        if str(h.get("state") or "").lower() in ("error", "stopping", "stopped") or h.get("error"):
            out.append(f"daemon 상태 {h.get('state')!r} {h.get('error') or ''}".strip())
        return out

    def _close_backend(self, note: str) -> None:
        backend, self.backend, self.executor = self.backend, None, None
        if backend is not None:
            backend.close()
        self._link_note = note

    def _present_joints_deg(self) -> np.ndarray | None:
        p = self.backend.present()
        return None if p.joints is None else np.degrees(np.asarray(p.joints, dtype=float))

    def _sampling(self) -> "_Sampler":
        return _Sampler(self.backend, real_time=type(self.clock).__name__ == "RealClock")

    # ------------------------------------------------------------------ jobs
    def _job_connect(self, cancel: threading.Event) -> None:
        self._refuse("connect", ["실제 로봇은 '연결' 버튼(연결 승인)으로만 연결합니다"])

    def _job_preview(self, cancel: threading.Event, spec: Any) -> None:
        self._refuse("preview", ["실제 로봇에서는 미리보기를 하지 않습니다: Mock·MuJoCo에서 보세요"])

    def _job_reset(self, cancel: threading.Event) -> None:
        self._refuse("reset", ["실제 로봇에는 자동 복귀가 없습니다: 'Wake-up'(승인)으로 INIT에 돌아가거나 머리를 받치고 '토크 OFF'"])

    def _job_status(self, cancel: threading.Event) -> None:
        out: dict[str, Any] = {"state": self.robot_state, "state_ko": STATE_KO.get(self.robot_state),
                               "torque_state": self.torque_state, "start_mode": self.start_mode,
                               "operator_procedure": self.operator_procedure}
        if self.backend is not None:  # the socket first: a silent link resets the torque to UNKNOWN ...
            out["socket"] = self._socket_status()
        try:  # ... and a fresh GET answer can then still read it back
            rep, path = self._read_r0()
            out.update(r0=r0_brief(rep), r0_path=str(path))
        except Exception as exc:  # noqa: BLE001 - shown; nothing was sent
            out["r0_error"] = f"{type(exc).__name__}: {exc}"
        self._emit("robot_status", **out, **self.torque_view())

    def _job_robot_connect(self, cancel: threading.Event, approval: Any) -> None:
        from reachy_motion import r1_readiness as rr
        from reachy_motion.executor.executor import Executor
        from reachy_motion.safety import gui_approval as ga

        if self.backend is not None:
            self._emit("status", text="이미 연결되어 있습니다")
            return
        self._speaker_obj = None  # a new connection uploads the effect sounds again
        rep: dict[str, Any] | None = None
        if approval is None:  # the click on '연결' is the approval (2026-11-12): read the robot first (GET only)
            gates = self._gates()
            if gates:
                self._refuse("robot_connect", gates)
                return
            rep, path = self._read_r0()
            self._emit("robot_status", state=self.robot_state, r0=r0_brief(rep), r0_path=str(path), **self.torque_view())
            req = ga.connect_request(rep, self.host, self.port)
            if not req.policy_ok:
                self._refuse("robot_connect", list(req.problems))
                return
            approval = ga.approve(req, ga.OPERATOR_ROWS_KO)
        probs = self._gates() + ga.problems(approval, kind="connect", host=self.host, port=self.port,
                                            used=self.used_approvals)
        if probs:
            self._refuse("robot_connect", probs)
            return
        if rep is None:
            rep, path = self._read_r0()  # fresh, GET only
        mode = rr.start_mode(rep)
        if mode != approval.request.start_mode:
            self._refuse("robot_connect", [f"R0가 승인 뒤 바뀌었습니다: 승인 때 {rr.MODE_KO.get(approval.request.start_mode)}"
                                           f", 지금 {rr.MODE_KO.get(mode)} - 다시 '상태 읽기' 후 연결을 승인하세요"])
            return
        self.used_approvals.add(approval.approval_id)
        self.state = "connecting"
        self._emit("connecting", host=self.host, port=self.port, start_mode=mode)
        try:
            backend = self._make_link(approval, rep, mode)
        except Exception as exc:  # noqa: BLE001 - classified by what is known (connect_outcome)
            self._connect_failed(exc, getattr(exc, "connect_report", None))
            return
        self.backend = backend
        self.executor = Executor(backend, clock=self.clock)
        self.data_kind = str(getattr(backend, "data_kind", "measured"))
        self.start_mode, self.torque_state, self.operator_procedure = mode, "on", ""
        self.r3_runs = []
        info = dict(getattr(backend, "connection_info", {}) or {})
        self.connection = {"mode": "robot", "data_kind": self.data_kind, "backend": str(getattr(backend, "name", "")),
                           "host": self.host, "port": self.port, "start_mode": mode,
                           "connected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "r0_report": str(path), "connect_approval": approval.record(),
                           "single_client": SINGLE_CLIENT_KO,
                           "residual": list(info.get("residual") or ["other_client"]),
                           "connect_report": info.get("connect")}
        self._link_note = "열림 (SDK 객체 있음)"
        if mode == "sleep":  # the sequence ended with the global enable_motors() and did not read the mode back
            self._torque_changed("연결이 enable_motors()를 보냄 - 모터 모드를 아직 다시 읽지 않음 ('상태 읽기')")
        elif (info.get("connect") or {}).get("torque_state") == "on":  # awake: the sequence read "enabled" on the socket
            self._note_readback("enabled", "연결 중 SDK 소켓 상태")
        log = self._log("connect", {"data_kind": self.data_kind, "connection": self.connection, "outcome": "connected",
                                    "torque_view": self.torque_view(),
                                    "note": "commands sent by the connection sequence: see connect_report.log"})
        self.connection["connection_log"] = str(self.log_dir)
        self.state = "ready"
        self._set_state("at_init" if mode == "awake" else "rest_torque_on", record=str(log))
        self._emit("robot_connected", start_mode=mode, record=str(log), connection=self.connection,
                   outcome="connected", outcome_ko=CONNECT_OUTCOME_KO["connected"], **self.torque_view())

    def _connect_failed(self, exc: BaseException, report: dict[str, Any] | None) -> None:
        """A connection attempt that did not end connected, told apart by what is known (``connect_outcome``)."""
        from reachy_motion.backends.robot_connect import OPERATOR_PROCEDURE_KO

        outcome = connect_outcome(exc, report)
        reasons = [f"{SDK_LOCKED} ({exc})" if isinstance(exc, NotImplementedError) else f"{type(exc).__name__}: {exc}"]
        if outcome == "refused_before_send":
            self._refuse("robot_connect", reasons)
            return
        if report is None:
            self._link_note = "없음 (ReachyMini 생성 중 실패: SDK 객체가 만들어지지 않음)"
        elif report.get("stage") == "ready":
            self._link_note = "닫힘 (연결 뒤 승인 재확인 실패: SdkBackend가 클라이언트를 정리)"
        else:
            self._link_note = "닫힘 (연결 순서가 SDK 클라이언트를 정리)"
        if outcome in ("failed_after_enable", "comm_lost_torque_unknown"):
            self._torque_changed(CONNECT_OUTCOME_KO[outcome])
            self.torque_state = "unknown"
            self.operator_procedure = (OPERATOR_PROCEDURE_KO + " 이 PC의 SDK 연결은 닫혔으므로 여기서 토크 OFF를 보낼 수 "
                                       "없습니다: 대시보드나 전원 스위치. '상태 읽기 (GET)'로 모터 모드를 확인할 수 있습니다.")
        else:  # this connection sent no torque command: an earlier read-back still holds
            self.operator_procedure = ""
        rec = self._log("connect_failed", {"data_kind": self.data_kind, "outcome": outcome,
                                           "outcome_ko": CONNECT_OUTCOME_KO[outcome], "reasons": reasons,
                                           "connect_report": report, "torque_view": self.torque_view()})
        self.state = "idle"
        self._set_state("disconnected", record=str(rec))
        self._emit("robot_connect_failed", outcome=outcome, outcome_ko=CONNECT_OUTCOME_KO[outcome], reasons=reasons,
                   nothing_sent=False, record=str(rec), operator_procedure=self.operator_procedure,
                   problems=list((report or {}).get("problems") or []), **self.torque_view())

    def _make_link(self, approval: Any, r0: dict[str, Any], mode: str) -> Any:
        if self.link_factory is not None:
            return self.link_factory(host=self.host, port=self.port, approval=approval, r0=r0, mode=mode,
                                     clock=self.clock)
        from reachy_motion.backends.sdk import SdkBackend

        backend = SdkBackend("robot", armed=True, approval=approval, clock=self.clock)  # refused while the flag is off
        backend.operator_confirmed = True  # the operator rows ticked in the connection dialog
        backend.start_mode = mode
        try:
            backend.connect()  # Phase 2: NotImplementedError - never a client
        except BaseException as exc:  # what the connection sequence knew (stage, torque) goes with the error
            exc.connect_report = (backend.connection_info or {}).get("connect")  # type: ignore[attr-defined]
            raise
        return backend

    def _job_wake_up(self, cancel: threading.Event, approval: Any) -> None:
        from reachy_motion.backends.robot_connect import OPERATOR_PROCEDURE_KO, init_error
        from reachy_motion.robot_r0 import AWAKE_INIT_TOL
        from reachy_motion.safety import gui_approval as ga
        from reachy_motion.sdk_wake_up import DURATION_S, send_wake_up
        from reachy_motion.startup_procedure import HOLD_S, watch_still

        probs = self._gates()
        if self.backend is None:
            probs.append("연결되지 않았습니다: 먼저 '연결'")
        if not probs and approval is None:  # the click is the approval: checked from where the head is now
            from reachy_motion.sdk_wake_up import check_wake_up

            state, joints = present_start_state(self.backend.present())
            req = ga.wake_up_request(check_wake_up(state, self.policy), self.host, self.port, joints)
            if req.policy_ok:
                approval = ga.approve(req)
            else:
                probs += list(req.problems)
        if not probs:
            probs += ga.problems(approval, kind="wake_up", host=self.host, port=self.port, used=self.used_approvals)
        if probs:
            self._refuse("wake_up", probs)
            return
        q = self._present_joints_deg()
        start = np.asarray(approval.request.start_joints_deg, dtype=float)
        if q is None or start.shape != q.shape or float(np.max(np.abs(q - start))) > WAKE_START_TOL_DEG:
            self._refuse("wake_up", ["머리가 승인할 때의 자세에 있지 않습니다 (> "
                                     f"{WAKE_START_TOL_DEG:g}°): 다시 승인하세요 (계산 검사가 그 자세에서 시작하는 동작이었음)"])
            return
        conflict = self._conflict_problems()
        if conflict:
            self._refuse("wake_up", conflict)
            return
        still, drift, _ = watch_still(self.backend, self.clock, seconds=0.5)
        if not still:
            self._refuse("wake_up", [f"아무것도 보내지 않았는데 머리가 {drift:.2f}° 움직였습니다: 다른 동작이 실행 중"])
            return
        self.used_approvals.add(approval.approval_id)
        n0 = len(getattr(self.backend, "commands", []) or [])
        self._emit("wake_up_started", duration_s=DURATION_S)
        error, sent = None, []
        t0 = int(self.clock.now_ns())
        with self._sampling() as sampler:
            try:
                sent = send_wake_up(self.backend, cancel, self.clock)
            except Exception as exc:  # noqa: BLE001 - recorded; never followed by an automatic action
                error = f"{type(exc).__name__}: {exc}"
            self.clock.sleep_until(self.clock.now_ns() + int(round(HOLD_S * 1e9)))  # settle before the INIT check
        cur = self.backend.present()
        err = init_error(cur)
        near = all(err[k] <= AWAKE_INIT_TOL[k] for k in AWAKE_INIT_TOL)
        cancelled = cancel.is_set()
        ok = error is None and not cancelled and len(sent) == 3 and near
        problems = ([f"오류: {error}"] if error else []) + (["정지 요청: 남은 goto를 보내지 않음"] if cancelled else []) + (
            [] if near else [f"INIT 근처가 아님: { {k: round(v, 2) for k, v in err.items()} } (허용 {AWAKE_INIT_TOL})"])
        self.operator_procedure = "" if ok else OPERATOR_PROCEDURE_KO
        if error is not None:
            self.torque_state = "unknown"
            self._torque_changed("Wake-up 중 오류 - 토크 확인 안 됨")
            self._link_note = "SDK 객체 남음 · 오류 뒤 응답 확인 안 됨 ('상태 읽기'로 확인)"
        cmds = list(getattr(self.backend, "commands", []) or [])[n0:]
        rec = self._log("sdk_wake_up", {
            "data_kind": self.data_kind, "ok": ok, "problems": problems,
            "approval": approval.record(), "sequence": "sdk_wake_up.gotos (ReachyMini.wake_up, reachy_mini.py:724-738, "
                                                       "no sound)",
            "sent": [{"label": lb, "t_s": (t - t0) / 1e9} for lb, t in sent],
            "commands_sent": [{"t_s": (c.t_ns - t0) / 1e9, "kind": c.kind, "duration": c.duration, "method": c.method,
                               "head": pose_from_matrix(c.head).to_deg_mm(),
                               "antennas_deg": [a * DEG for a in c.antennas], "body_yaw_deg": c.body_yaw * DEG}
                              for c in cmds],
            "present_measured": [dict(present_brief(p), t_s=(int(p.t_ns) - t0) / 1e9) for p in sampler.samples],
            "init_error_after": err, "torque_state": self.torque_state,
            "time_base": "t_s = seconds since the wake-up job started (this PC's clock); commands = what was sent "
                         "(the daemon interpolates each goto at ~100 Hz: the commanded trajectory is not observable), "
                         f"present = arrival of the measured state ({self.data_kind})"})
        self._set_state("at_init" if ok else "holding", record=str(rec))
        self._emit("wake_up_finished", ok=ok, problems=problems, init_error=err, record=str(rec), sent=len(sent),
                   present_samples=len(sampler.samples))

    def _r3_conditions(self) -> list[tuple[str, dict[str, Any]]]:
        """(run folder, trial condition) of this connection's R3 steps, latest first."""
        out = []
        for run in reversed(self.r3_runs):
            try:
                out.append((run, json.loads((Path(run) / "manifest.json").read_text(encoding="utf-8"))["trial"]["condition"]))
            except (OSError, ValueError, KeyError):
                continue
        return out

    def _r3_after(self, trial: Any) -> str | None:
        """The ladder's previous step (``robot_r1.r3_step_problems`` checks it), latest of this connection: the same
        amplitude and a longer D, or on the amplitude ladder the same D and a smaller A. The same pitch nod counts
        whichever ladder ran it (A 20 deg in D 1.0 s is a step of both)."""
        a, d = float(trial.condition["amplitude_deg"]), float(trial.condition["duration_s"])
        amplitude_ladder = trial.condition.get("vary") == "amplitude"
        for run, c in self._r3_conditions():
            ca, cd = float(c.get("amplitude_deg", math.nan)), float(c.get("duration_s", math.nan))
            if amplitude_ladder and abs(cd - d) <= 1e-9 and ca < a - 1e-9:
                return run
            if not amplitude_ladder and abs(ca - a) <= 1e-9 and cd > d + 1e-9:
                return run
        return None

    def _r3_base(self, trial: Any) -> str | None:
        """The latest first step of this ladder in this connection, the baseline of the lag growth: D 1.0 s at the
        same amplitude, or on the amplitude ladder A 10 deg at the same D."""
        from reachy_motion import robot_r1

        a, d = float(trial.condition["amplitude_deg"]), float(trial.condition["duration_s"])
        want = (robot_r1.R3_BASE_A_DEG, d) if trial.condition.get("vary") == "amplitude" else (a, robot_r1.R3_BASE_S)
        for run, c in self._r3_conditions():
            if abs(float(c.get("amplitude_deg", math.nan)) - want[0]) <= 1e-9 and \
                    abs(float(c.get("duration_s", math.nan)) - want[1]) <= 1e-9:
                return run
        return None

    # ------------------------------------------------------------------ effect sounds (reachy_motion.sound)
    def _speaker(self) -> Any:
        """The robot speaker of this connection (uploads are remembered until the next connection)."""
        if self._speaker_obj is None:
            from reachy_motion.sound import RobotSpeaker

            make = self.speaker_factory or (lambda host, port: RobotSpeaker(host, port))
            self._speaker_obj = make(self.host, self.port)
        return self._speaker_obj

    def _sound_cue(self, trial: Any, plan: Any) -> Any:
        """The CueTrigger of the trial's sound cue, ready before the stream starts, or None (no cue, sound off, or a
        problem - reported, and the motion runs without it). Robot speaker: ``GET /api/media/status`` (a daemon
        without media plays nothing), then the upload (once per connection and file); laptop: ``winsound``. The
        click that approved the trial approves its sound: these requests come only from that click (CLAUDE.md
        rule 3); the record goes into the run's manifest (``connection.sound``)."""
        from reachy_motion.sound import CueTrigger, cue_tick, file_sha256, sound_cue

        cue = sound_cue(getattr(trial, "spec", None))
        output = self.sound.output
        if cue is None or output not in ("robot", "laptop"):
            return None
        path = Path(self.sound.sounds_dir) / cue.file
        rec: dict[str, Any] = {"output": output, "file": cue.file, "at_s": float(cue.at_s),
                               "offset_ms": float(self.sound.offset_ms), "pre_s": float(trial.pre_s)}
        if not path.is_file():
            rec.update(ok=False, error=f"음원 파일 없음: {path} - 모션만 실행")
            self._emit("sound", **rec)
            return None
        rec["sha256"] = file_sha256(path)
        if output == "laptop":
            play: Callable[[], Any] = lambda: self.laptop.play(path)  # noqa: E731
        else:
            try:
                spk = self._speaker()
                status = spk.status()
                rec["media_status"] = status
                if status.get("no_media"):
                    raise RuntimeError("로봇 daemon이 미디어 없이(--no-media) 실행 중이라 스피커를 쓸 수 없음")
                daemon_path = spk.upload(path)
            except Exception as exc:  # noqa: BLE001 - shown; the motion is approved and runs without its sound
                rec.update(ok=False, error=f"효과음 준비 실패 ({type(exc).__name__}: {exc}) - 모션만 실행")
                self._emit("sound", **rec)
                return None
            rec["daemon_path"] = daemon_path
            play = lambda: spk.play(daemon_path)  # noqa: E731
        k = cue_tick(trial.pre_s, cue.at_s, self.sound.offset_ms, plan.run.timeline.rate_hz)
        return CueTrigger(k, play, rec)

    def trial_plan(self, trial: Any) -> tuple[Any, list[str]]:
        """(plan, problems) of one trial on the awake robot: the plan the approval dialog shows and the job runs
        (computed; Qt-free, also called from the GUI thread - reads only this session's R3 history)."""
        from reachy_motion import robot_r1

        try:
            plan = robot_r1.build_plan(trial, None, awake=True)
        except ValueError as exc:
            return None, [f"깨어 있는 로봇의 시행이 아닙니다: {exc}"]
        probs = []
        if trial.stage == "R3":
            probs += robot_r1.r3_step_problems(plan, self._r3_after(trial), robot=True, base=self._r3_base(trial),
                                               lang="ko")
        return plan, probs

    def _job_trial(self, cancel: threading.Event, trial: Any, session_dir: Path, notes: str = "",  # type: ignore[override]
                   approval: Any = None) -> None:
        from reachy_motion import robot_r1
        from reachy_motion.gui.results import load_result, run_brief
        from reachy_motion.safety import gui_approval as ga

        probs = self._gates()
        if self.backend is None:
            probs.append("연결되지 않았습니다")
        elif self.robot_state != "at_init":
            probs.append(f"로봇이 INIT 대기 상태가 아닙니다 ({STATE_KO.get(self.robot_state, self.robot_state)})")
        plan, plan_problems = self.trial_plan(trial)
        probs += plan_problems
        if plan is not None:
            probs += ga.problems(approval, kind="trial", host=self.host, port=self.port, label=trial.label,
                                 digest=plan.run.timeline.digest(), used=self.used_approvals)
        if not probs:
            probs += self._conflict_problems()
        if probs:
            self._refuse("trial", probs, label=trial.label)
            return
        self.used_approvals.add(approval.approval_id)

        def before_part(label: str) -> None:  # right before the stream: gates, this approval, another controller
            p = self._gates() + ga.problems(approval, kind="trial", host=self.host, port=self.port, label=trial.label,
                                            digest=plan.run.timeline.digest())
            p += self._conflict_problems()
            if p:
                raise LabError("; ".join(p))

        conn = dict(self.connection, trial_approval=approval.record())
        trig = self._sound_cue(trial, plan)  # the motion's effect sound (prepared before the stream: upload, GET)
        if trig is not None:
            conn["sound"] = trig.record  # filled in by the cue's thread; saved with the trial's run

        def live(label: str, n_part: int) -> Any:  # the live view, and the sound cue on the trial's own stream
            on_tick = self._live_callback(label, n_part)
            return trig.wrap(on_tick) if trig is not None and label == trial.label else on_tick

        self._drain_live()
        n = len(plan.run.timeline)
        self._emit("trial_started", label=trial.label, title=trial.title, stage=trial.stage, n_samples=n,
                   duration_s=float(plan.run.timeline.duration), main_channel=trial.main_channel, pre_s=trial.pre_s,
                   settle=None)
        self.executor.last_result = None
        out = robot_r1.run_trial(self.backend, plan, session_dir, notes=notes, connection=conn, clock=self.clock,
                                 torque_off=getattr(self.backend, "disable_motors", None), before_part=before_part,
                                 cancel=cancel, live=live, executor=self.executor)
        sound = self._sound_record(trig)
        if sound is not None:
            self._log("sound", dict(sound, trial=trial.label, data_kind="pc_measured",
                                    note="request times measured on this PC; the robot's own playback delay is not "
                                         "measured"))
        result = self.executor.last_result if out.get("digests") else None
        sent = result is not None
        if not sent:  # refused inside the runner (start gap, a moving head, before_part): nothing was sent
            self._set_state(self.robot_state)
            self._refuse("trial", list(out.get("problems") or ["시작 전에 멈춤"]), label=trial.label)
            return
        if not result.commands and result.cancelled:  # Stop before the first command: nothing sent, robot unchanged
            rec = self._log("trial_cancelled", {
                "data_kind": self.data_kind, "trial": trial.label, "commands_sent": 0,
                "stop_reason": result.stop_reason, "session_record": str(Path(session_dir) / f"session_{trial.label}.json"),
                "note": "Stop before the first command: nothing was sent, the robot stays where it was (INIT). The "
                        "runner's own record keeps its conservative end-of-run text."})
            self._set_state(self.robot_state, record=str(rec))
            self._emit("trial_cancelled", label=trial.label, stage=trial.stage, record=str(rec),
                       reason="첫 명령을 보내기 전에 정지: 명령 0개, 로봇은 그대로")
            return
        self.torque_state = out.get("torque_state", self.torque_state)
        if self.torque_state == "unknown":  # S2 / S3 / an exception: the end state is not known
            self._torque_changed("시행 중 통신·daemon 오류 - 토크 확인 안 됨")
            self._link_note = "SDK 객체 남음 · 오류 뒤 응답 확인 안 됨 ('상태 읽기'로 확인)"
        self.operator_procedure = out.get("operator_procedure", "") if sent and not out.get("ok") else ""
        run_dir = out.get("run_dir")
        loaded, brief = None, {}
        if run_dir:
            try:
                loaded = load_result(run_dir)
                brief = run_brief(loaded)
            except Exception as exc:  # noqa: BLE001 - the run is saved even when it cannot be analysed
                brief = {"analysis_error": f"{type(exc).__name__}: {exc}"}
        if trial.stage == "R3" and out.get("ok") and run_dir:
            self.r3_runs.append(str(run_dir))
        r3_next = None
        if trial.stage == "R3" and run_dir and sent:  # may the ladder go on? (the GUI shows it at once, in Korean)
            r3_next = {"amplitude_deg": float(trial.condition["amplitude_deg"]),
                       "duration_s": float(trial.condition["duration_s"]),
                       "vary": "amplitude" if trial.condition.get("vary") == "amplitude" else "duration",
                       "problems": robot_r1.r3_next_step_problems(run_dir, self._r3_base(trial), lang="ko",
                                                                  loaded=loaded)}
        evidence = sent_vs_measured(result, brief, trial.main_channel)
        state = "at_init" if (out.get("ok") or not sent) else "holding"
        self._set_state(state)
        self._emit("trial_finished", label=trial.label, stage=trial.stage, run_dir=run_dir,
                   completed=bool(result is not None and result.completed),
                   cancelled=bool(result is not None and result.cancelled) or (not sent and cancel.is_set()),
                   error=None if result is None else result.error,
                   stop_kind=None if result is None else result.stop_kind,
                   stop_reason=None if result is None else result.stop_reason,
                   ticks={} if result is None else tick_summary(result), metrics=brief, dropped_live=self.dropped_live,
                   loaded=loaded, reason=None if sent else "; ".join(out.get("problems") or []) or None,
                   r3_next=r3_next, sound=sound,
                   robot={"state": state, "torque_state": self.torque_state, "ok": bool(out.get("ok")),
                          "problems": list(out.get("problems") or []), "operator_procedure": self.operator_procedure,
                          "evidence": evidence, "session_record": str(Path(session_dir) / f"session_{trial.label}.json")})

    def _job_torque_off(self, cancel: threading.Event, confirmed: bool) -> None:
        from reachy_motion.robot_r1 import OFF_DRIFT_WARN_DEG, _off_drift

        if self.backend is None:
            self._refuse("torque_off", ["연결되지 않았습니다: 이 PC에서 보낼 수 없음 - 머리를 받치고 대시보드나 전원 스위치로 끄세요"])
            return
        if not confirmed:
            self._refuse("torque_off", ["머리를 받쳤다는 확인('off' 입력)이 없어 토크를 그대로 둡니다"])
            return
        self._torque_changed("disable_motors()를 보냄 - 다시 읽어 확인하지 못함")
        state = self.backend.disable_motors()
        if state == "off":  # send_torque_off read the motor mode back on the socket: "disabled"
            self._note_readback("disabled", "토크 OFF 직후 SDK 소켓 상태")
        drift = _off_drift(self.backend, self.clock) if state == "off" else math.nan
        self.torque_state = state
        warn = ([f"토크를 끈 뒤 머리가 {drift:.2f}° 움직였습니다: 받친 자세에서 내려간 것 (쉬던 자세가 아니었음)"]
                if state == "off" and drift > OFF_DRIFT_WARN_DEG else [])
        rec = self._log("torque_off", {"data_kind": self.data_kind, "torque_state": state, "by": "operator (head "
                                       "supported, typed 'off' in the GUI)", "off_drift_deg": drift, "warnings": warn,
                                       "torque_off_log": (getattr(self.backend, "connection_info", {}) or {}).get(
                                           "torque_off_log") or getattr(self.backend, "torque_log", None)})
        self.operator_procedure = "" if state == "off" else (
            "토크 OFF가 확인되지 않았습니다: 머리를 받친 채 전원 스위치로 끄세요")
        self._close_backend("닫힘 (토크 OFF 뒤 SDK 클라이언트 정리)")
        self.state = "idle"
        self._set_state("disconnected", record=str(rec))
        self._emit("robot_torque_off", torque_state=state, off_drift_deg=drift, warnings=warn, record=str(rec),
                   **self.torque_view())

    def _job_disconnect(self, cancel: threading.Event) -> None:
        if self.backend is None:
            self._emit("status", text="연결되어 있지 않습니다")
            return
        was = self.robot_state
        rec = self._log("disconnect", {"data_kind": self.data_kind, "state_before": was,
                                       "torque_state": self.torque_state,
                                       "note": "client disconnect only (io/ws_client.py:135-143): the robot keeps its "
                                               "torque and its last target"})
        self._close_backend("닫힘 (연결 끊기: SDK 클라이언트만 정리, 토크 OFF 아님)")
        self.state = "idle"
        self._set_state("disconnected", record=str(rec))
        self._emit("robot_disconnected", state_before=was, torque_state=self.torque_state, record=str(rec),
                   **self.torque_view())


# ---------------------------------------------------------------------------------------------- helpers
class _Sampler:
    """Present states during the blocking wake-up gotos: a thread at ~100 Hz on the real clock; with a virtual clock
    (the offline harness) one sample before and one after."""

    def __init__(self, backend: Any, real_time: bool) -> None:
        self.backend, self.real_time = backend, real_time
        self.samples: list[Any] = []
        self._done = threading.Event()
        self._th: threading.Thread | None = None

    def _read(self) -> None:
        try:
            self.samples.append(self.backend.present())
        except Exception:  # noqa: BLE001 - a missed reading only thins the record
            pass

    def _run(self) -> None:
        while not self._done.wait(0.01):
            self._read()

    def __enter__(self) -> "_Sampler":
        self._read()
        if self.real_time:
            self._th = threading.Thread(target=self._run, name="wake-up-sampler", daemon=True)
            self._th.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._done.set()
        if self._th is not None:
            self._th.join(timeout=1.0)
        self._read()


def present_brief(p: Any) -> dict[str, Any]:
    """A present state for people: head pose (deg / mm), antennas and body yaw (deg), joints (deg)."""
    pose = pose_from_matrix(np.asarray(p.head, dtype=float)).to_deg_mm()
    return {"head": {k: round(float(v), 2) for k, v in pose.items()},
            "antennas_deg": [round(float(a) * DEG, 2) for a in p.antennas], "body_yaw_deg": round(float(p.body_yaw) * DEG, 2),
            "joints_deg": None if p.joints is None else [round(float(q) * DEG, 2) for q in p.joints],
            "data_kind": getattr(p, "data_kind", None)}


def r0_brief(rep: dict[str, Any]) -> dict[str, Any]:
    """What the robot panel shows of an R0 report (GET only)."""
    from reachy_motion import r1_readiness as rr

    d = rep.get("derived") or {}
    return {"generated_local": rep.get("generated_local"), "daemon_url": rep.get("daemon_url"),
            "data_kind": rep.get("data_kind"), "daemon_version": d.get("daemon_version"),
            "daemon_state": d.get("daemon_state"), "motor_mode": d.get("motor_mode"),
            "head_tracking": d.get("head_tracking"), "head_pose": d.get("head_pose"),
            "antennas_deg": d.get("antennas_deg"), "body_yaw_deg": d.get("body_yaw_deg"),
            "joints_deg": (d.get("joints") or {}).get("joints_deg"), "distance_to_rest": d.get("distance_to_rest"),
            "blocking": rep.get("blocking"), "start_mode": rr.start_mode(rep),
            "block_reasons": rr.r0_block_reasons(rep) if rep.get("blocking", True) else [],
            # a limp head resting a little past the limits: connectable (user decision 2026-11-17)
            "limp_past_limits_deg": ((d.get("joints") or {}).get("max_beyond_deg") if any(
                x.get("code") == "start_beyond_limits_limp" for x in rep.get("findings") or []) else None),
            "findings": [f"[{x.get('level')}] {x.get('code')}" for x in rep.get("findings") or []]}


def r0_start_state(rep: dict[str, Any]) -> tuple[RobotState, list[float]] | None:
    """The state an R0 report measured (head pose, antennas, body) and its joints (deg) - the wake-up's start."""
    from reachy_motion.robot_r1 import r0_state

    d = rep.get("derived") or {}
    joints = (d.get("joints") or {}).get("joints_deg")
    try:
        state = r0_state(rep)
    except (KeyError, TypeError, ValueError):
        return None
    if not isinstance(joints, dict):
        return None
    from reachy_motion.startup_procedure import JOINT_NAMES

    return state, [float(joints[n]) for n in JOINT_NAMES]


def present_start_state(p: Any) -> tuple[RobotState, list[float]]:
    """The state a present reading measured, and its joints (deg)."""
    state = RobotState(head=pose_from_matrix(np.asarray(p.head, dtype=float)),
                       antennas=(float(p.antennas[0]), float(p.antennas[1])), body_yaw=float(p.body_yaw))
    return state, [float(q) * DEG for q in p.joints]


def state_from_brief(brief: dict[str, Any]) -> tuple[RobotState, list[float]] | None:
    """Back from ``present_brief`` (what the panel received) to a state and its joints (deg) - the wake-up dialog's
    start; the job re-checks the head against these joints before sending (``WAKE_START_TOL_DEG``)."""
    from reachy_motion.library.pose import Pose6

    try:
        head = Pose6.from_deg_mm(**{k: float(v) for k, v in brief["head"].items()})
        ant = brief["antennas_deg"]
        state = RobotState(head=head, antennas=(math.radians(float(ant[0])), math.radians(float(ant[1]))),
                           body_yaw=math.radians(float(brief["body_yaw_deg"])))
        joints = [float(q) for q in brief["joints_deg"]]
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return state, joints


def connect_outcome(exc: BaseException | None, report: dict[str, Any] | None) -> str:
    """How a connection attempt ended (``CONNECT_OUTCOME_KO``), from what is actually known: the exception and the
    connection sequence's report (``robot_connect.ConnectReport``: stage, torque_state, torque_enabled_by_us) - None
    when the sequence never returned one.

    * no exception -> connected;
    * the Phase 2 lock (NotImplementedError), or an approval refused before the sequence ran, or the sequence's
      "precondition" stage -> refused before anything was sent (the sequence checks R0 BEFORE constructing the client);
    * torque_state "unknown" (an error during / after the enable: the link may be gone) -> torque not known;
    * the global enable_motors() was sent -> failed after the torque enable;
    * anything else after the constructor (or the constructor itself, which sends SetAutomaticBodyYaw / the media
      release but never a torque command) -> failed after some commands."""
    from reachy_motion.safety.approval import ApprovalError

    if exc is None:
        return "connected"
    if report is None:
        return ("refused_before_send" if isinstance(exc, (NotImplementedError, ApprovalError))
                else "failed_after_partial_send")
    if report.get("stage") == "precondition":
        return "refused_before_send"
    if report.get("torque_state") == "unknown":
        return "comm_lost_torque_unknown"
    if report.get("torque_enabled_by_us"):
        return "failed_after_enable"
    return "failed_after_partial_send"


def sent_vs_measured(result: Any, brief: dict[str, Any], channel: str) -> dict[str, Any]:
    """What was SENT (commands, the executor's record) next to what was MEASURED (present states and the main
    channel's measured / commanded amplitude) - a command sent is not a motion seen."""
    if result is None:
        return {"commands_sent": 0, "present_samples": 0, "note": "nothing was sent"}
    m = brief.get(channel) if isinstance(brief, dict) else None
    ratio = m.get("amplitude_ratio") if isinstance(m, dict) else None
    p2p = m.get("command_p2p") if isinstance(m, dict) else None
    return {"commands_sent": len(result.commands), "present_samples": len(result.present),
            "channel": channel, "command_p2p": p2p, "amplitude_ratio_measured_over_commanded": ratio,
            "data_kind_present": str(result.data_kind),
            "note": "commands.csv = values sent; present_<data_kind>.csv = states measured on arrival"}


__all__ = ["RobotLabSession", "STATE_KO", "SINGLE_CLIENT_KO", "ROBOT_RUN_LOCKED", "SDK_LOCKED", "R0_ROOT",
           "CONNECT_OUTCOME_KO", "NOT_TORQUE_OFF_KO", "LINK_FRESH_S", "connect_outcome",
           "present_brief", "r0_brief", "r0_start_state", "present_start_state", "state_from_brief",
           "sent_vs_measured"]
