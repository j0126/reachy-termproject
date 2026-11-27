"""Experiment runner of the Motion Lab (Qt-free): one backend, ONE worker thread, a queue to the GUI.

Threads (the split of ``service.MotionService``):

* the GUI thread only submits jobs (connect, trial, preview, reset, close), calls ``stop`` and drains ``live`` (a
  bounded queue); events reach it through ``on_event``, called on the worker thread (the Qt layer re-emits them as
  queued signals);
* ONE worker thread owns the backend and its ``Executor`` and runs the jobs one after another. The 50 Hz send loop
  is the executor's own, in that thread; its ``on_tick`` callback only does ``live.put_nowait`` of the sample just
  sent and the latest present state (a full queue drops that display sample, never a tick). Drawing therefore never
  runs in the sending thread, and the tick timing of every run is recorded (pc_measured) and reported, so an
  effect of the GUI on the 50 Hz loop would show.

Targets: "mock" (``MockBackend(structure="robot")``, data_kind mock) and "sim" (``SdkBackend("sim")``: our MuJoCo
daemon on 127.0.0.1 behind every interlock of ``backends.sdk``, data_kind simulated; right after connecting, the
start-up ramp from the sleep pose to REST, ``safety.startup``). ``LabSession("robot")`` raises: the robot is served
by the subclass ``gui.robot_session.RobotLabSession`` (one SDK client, a GUI approval per action, and the run lock
and SdkBackend's approved connection in front of it).

A trial (``gui.experiments``): a checked transition to its start state if needed (not recorded), then the recorded
timeline (start hold + motion + end hold) through ``Executor.run(record_present=True, stop=StopPolicy())`` - Stop is
the verified controlled stop S1, a failed or slow send or a lost measurement the immediate stop S2
(docs/phase4/STARTUP_STOP.md §6, §9) - then the run folder: ``commands.csv`` and ``present_<data_kind>.csv`` on ONE
time base (s from the executor's start, plus the absolute ``t_ns``), ``ticks.csv`` (pc_measured), ``manifest.json``
and ``metrics.json`` (tracking metrics, ``gui.results``).
"""

from __future__ import annotations

import json
import math
import queue
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reachy_motion.library.pose import REST, RobotState, pose_from_matrix
from reachy_motion.safety.policy import DEFAULT_POLICY, SafetyPolicy

DEG = 180.0 / math.pi
MODES: dict[str, str] = {"mock": "mock", "sim": "simulated", "robot": "measured"}  # mode -> data_kind of its present state
SIM_PORT = 8017  # the Motion Lab's own local MuJoCo daemon (127.0.0.1); the CLI sim tools default to 8000
LIVE_MAX = 4000  # display samples buffered between two GUI polls (80 s at 50 Hz)
LAB_ROOT = Path("results") / "lab"
LAB_TOOL = "reachy-motion Motion Lab"
DATA_KIND_NOTES = {
    "mock": "Mock 백엔드: daemon 명령 경로의 합성 모델(backends/mock.py). 측정값 아님",
    "simulated": "MuJoCo: 마찰이 지배적이고 이 PC에서 실시간의 약 0.84배. 설계와 기록 절차 확인용이며 로봇 추종 "
                 "예측이 아님 (Phase 1 5.5)",
    "measured": "실제 로봇",
}


class LabError(RuntimeError):
    """A job of the Motion Lab could not run (shown in the GUI; nothing was sent unless stated)."""


class LabSession:
    """One backend (mock or sim) driven by one worker thread; see the module docstring."""

    #: The targets this class serves. The robot has its own subclass with every robot gate
    #: (``gui.robot_session.RobotLabSession``): ``LabSession("robot")`` itself is refused.
    TARGETS: tuple[str, ...] = ("mock", "sim")

    def __init__(self, mode: str, root: Path | str = LAB_ROOT, on_event: Callable[[dict[str, Any]], None] | None = None,
                 port: int = SIM_PORT, viewer: bool = True, policy: SafetyPolicy = DEFAULT_POLICY,
                 backend_factory: Callable[[], Any] | None = None) -> None:
        if mode not in MODES or mode not in self.TARGETS:
            raise LabError(f"LabSession에서는 {mode!r} 모드를 쓸 수 없습니다: Mock과 MuJoCo만. 실제 로봇은 RobotLabSession "
                           "(gui/robot_session.py: 연결·Wake-up·시행마다 GUI 승인)으로만 다룹니다")
        self.mode, self.data_kind = mode, MODES[mode]
        self.root = Path(root)
        self.on_event = on_event
        self.port, self.viewer, self.policy = int(port), bool(viewer), policy
        self._factory = backend_factory
        self.backend: Any = None
        self.executor: Any = None
        self.state = "idle"  # idle | connecting | ready | busy | closed
        self.connection: dict[str, Any] = {}
        self.live: queue.Queue = queue.Queue(maxsize=LIVE_MAX)
        self.dropped_live = 0
        self._jobs: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._cancels: list[threading.Event] = []
        self._last: RobotState | None = None
        self._closed = False
        from reachy_motion.sound import LaptopSpeaker, SoundSettings

        self.sound = SoundSettings()  # off unless the window shares its setting (LabSessions.sound)
        self.laptop = LaptopSpeaker()
        self._thread = threading.Thread(target=self._loop, name=f"motionlab-{mode}", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ GUI-thread API (non-blocking)
    def connect(self) -> None:
        self._submit("connect")

    def run_trial(self, trial: Any, session_dir: Path | str, notes: str = "") -> None:
        """Queue one trial; ``notes`` (the user's experiment memo) is saved with the run."""
        self._submit("trial", trial=trial, session_dir=Path(session_dir), notes=str(notes or ""))

    def preview(self, spec: Any) -> None:
        self._submit("preview", spec=spec)

    def stop(self) -> None:
        """Cancel the running job (controlled stop S1 at the next tick) and every queued one."""
        with self._lock:
            for ev in self._cancels:
                ev.set()
            self._cancels.clear()

    def reset(self) -> None:
        """Stop, then a checked transition back to REST."""
        self.stop()
        self._submit("reset")

    def close(self, timeout: float = 30.0) -> None:
        """Stop everything, close the backend (sim: the daemon and its viewer stop) and end the worker."""
        if self._closed:
            return
        self._closed = True
        self.stop()
        self._jobs.put(("close", {}, threading.Event()))
        self._thread.join(timeout)

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._cancels)

    def _submit(self, kind: str, **kw: Any) -> None:
        if self._closed:
            raise LabError("세션이 닫혔습니다")
        ev = threading.Event()
        with self._lock:
            self._cancels.append(ev)
        self._jobs.put((kind, kw, ev))

    def _forget(self, ev: threading.Event) -> None:
        with self._lock:
            if ev in self._cancels:
                self._cancels.remove(ev)

    def _emit(self, kind: str, **details: Any) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event({"kind": kind, "mode": self.mode, "data_kind": self.data_kind, **details})
        except Exception:  # noqa: BLE001 - a broken listener must not stop the worker
            pass

    # ------------------------------------------------------------------ worker
    def _loop(self) -> None:
        from reachy_motion.util.platform_timing import enable_high_resolution_timer

        enable_high_resolution_timer()
        while True:
            kind, kw, cancel = self._jobs.get()
            if kind == "close":
                self._shutdown()
                return
            if cancel.is_set():
                self._emit("skipped", job=kind, label=getattr(kw.get("trial"), "label", None))
                self._forget(cancel)
                continue
            try:
                getattr(self, f"_job_{kind}")(cancel=cancel, **kw)
            except Exception as exc:  # noqa: BLE001 - reported; the worker keeps serving
                self._emit("error", job=kind, error=f"{type(exc).__name__}: {exc}",
                           label=getattr(kw.get("trial"), "label", None))
            finally:
                self._forget(cancel)
                self.state = "ready" if self.backend is not None else "idle"

    def _shutdown(self) -> None:
        backend, self.backend = self.backend, None
        self.state = "closed"
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001
                self._emit("error", job="close", error=f"{type(exc).__name__}: {exc}")
        self._emit("closed")

    def _make_backend(self) -> Any:
        if self._factory is not None:
            return self._factory()
        if self.mode == "mock":
            from reachy_motion.backends.mock import MockBackend

            return MockBackend(structure="robot")
        from reachy_motion.backends.sdk import SdkBackend
        from reachy_motion.sdk_env import pin_mismatches

        bad = pin_mismatches()
        if bad:
            raise LabError(f"SDK 버전 고정이 맞지 않습니다 ({bad}). `python -m reachy_motion env`를 확인하세요")
        log_dir = self.root / "sim_daemon_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        return SdkBackend("sim", port=self.port, viewer=self.viewer, log_dir=log_dir)

    def _ensure_connected(self, cancel: threading.Event) -> None:
        if self.backend is not None:
            return
        from reachy_motion.executor.executor import Executor

        self.state = "connecting"
        self._emit("connecting", port=self.port if self.mode == "sim" else None, viewer=self.viewer)
        backend = self._make_backend()
        backend.connect()
        self.backend = backend
        self.executor = Executor(backend)
        self.connection = {"mode": self.mode, "data_kind": self.data_kind, "backend": str(backend.name),
                           "connected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           **({"sim": getattr(backend, "connection_info", {})} if self.mode == "sim" else {})}
        if self.mode == "sim":
            self._startup_ramp(cancel)
        else:
            self._last = REST  # the Mock starts at REST
        self.state = "ready"
        self._emit("ready", port=self.port if self.mode == "sim" else None)

    def _startup_ramp(self, cancel: threading.Event) -> None:
        """MuJoCo starts in the sleep pose: ramp to REST under the start-up rules (as ``sim-play`` does)."""
        from reachy_motion.safety.startup import prepare_startup_ramp

        prep = prepare_startup_ramp(self._present_state(), REST, self.policy)
        if not prep.runnable:
            raise LabError("측정 자세에서 시작하는 기동 램프가 거부됨: "
                           + "; ".join(v.message for v in prep.report.violations))
        self._emit("status", text=f"rest까지 기동 램프 ({prep.timeline.duration:.1f} s)")
        res = self.executor.run(prep, cancel=cancel)
        self._remember(res)
        if not res.completed:
            raise LabError("기동 램프가 끝나지 않음 (정지됨)")

    def _present_state(self) -> RobotState:
        p = self.backend.present()
        return RobotState(head=pose_from_matrix(p.head), antennas=(float(p.antennas[0]), float(p.antennas[1])),
                          body_yaw=float(p.body_yaw))

    def _remember(self, result: Any) -> None:
        if result is not None and result.last_sample is not None:
            head, ant, body = result.last_sample
            self._last = RobotState(head=pose_from_matrix(head), antennas=(float(ant[0]), float(ant[1])),
                                    body_yaw=float(body))

    def _stop_policy(self) -> Any:
        from reachy_motion.executor.executor import StopPolicy

        return StopPolicy()

    def _to_start(self, state: RobotState, cancel: threading.Event) -> bool:
        """Checked transition from the last commanded state to ``state`` (not recorded); True when there."""
        from reachy_motion.library.spec import Easing
        from reachy_motion.planner import plan_transition, transition_duration
        from reachy_motion.safety.checker import prepare_timeline
        from reachy_motion.service import same_state

        cur = self._last if self._last is not None else self._present_state()
        if same_state(cur, state):
            return True
        tl = plan_transition(cur, state, transition_duration(cur, state), Easing.MIN_JERK, self.executor.rate_hz,
                             name="to_start")
        job = prepare_timeline(tl, self.policy)
        if not job.runnable:
            raise LabError("시작 자세로 가는 전이가 초기 정책 밖: "
                           + ", ".join(v.code for v in job.report.violations))
        self._emit("status", text=f"시작 자세로 전이 ({tl.duration:.1f} s)")
        res = self.executor.run(job, cancel=cancel, stop=self._stop_policy())
        self._remember(res)
        return bool(res.completed)

    def _wait_still(self, cancel: threading.Event, window_s: float = 0.3, tol_deg: float = 0.03,
                    timeout_s: float = 6.0) -> dict[str, Any]:
        """Read the present joints (no command is sent) until they moved < ``tol_deg`` over ``window_s``
        (``sim_review.wait_still``; MuJoCo is friction-dominated and still creeps after a ramp)."""
        import time

        t0 = time.perf_counter()
        samples: list[tuple[float, Any]] = []
        spread = math.nan
        while not cancel.is_set():
            now = time.perf_counter()
            p = self.backend.present()
            if p.joints is not None:
                samples.append((now, [float(q) * DEG for q in p.joints]))
            recent = [q for t, q in samples if t >= now - window_s]
            if len(recent) >= 5 and samples[0][0] <= now - window_s:
                spread = max(max(col) - min(col) for col in zip(*recent))
                if spread < tol_deg:
                    return {"settled": True, "waited_s": round(now - t0, 3), "spread_deg": spread}
            if now - t0 > timeout_s:
                break
            time.sleep(0.02)
        return {"settled": False, "waited_s": round(time.perf_counter() - t0, 3), "spread_deg": spread}

    def _live_callback(self, label: str, n: int) -> Callable[[int, Any], None]:
        ex, live = self.executor, self.live

        def on_tick(k: int, _rec: Any) -> None:  # executor thread: O(1), never blocks
            r = ex.last_result
            if r is None or not r.commands:
                return
            c = r.commands[-1].record
            p = r.present[-1] if r.present else None
            item = (label, k, n, c.t_ns, c.head, c.antennas, c.body_yaw,
                    None if p is None else (p.t_ns, p.head, p.antennas, p.joints))
            try:
                live.put_nowait(item)
            except queue.Full:
                self.dropped_live += 1

        return on_tick

    def _drain_live(self) -> None:
        try:
            while True:
                self.live.get_nowait()
        except queue.Empty:
            pass

    # ------------------------------------------------------------------ jobs
    def _job_connect(self, cancel: threading.Event) -> None:
        self._ensure_connected(cancel)

    def _job_reset(self, cancel: threading.Event) -> None:
        if self.backend is None:
            self._emit("status", text="연결되지 않음: 복귀할 것이 없음")
            return
        ok = self._to_start(REST, cancel)
        self._emit("reset_done", completed=ok)

    # ------------------------------------------------------------------ effect sounds (reachy_motion.sound)
    def _laptop_cue(self, spec: Any, pre_s: float, rate_hz: float) -> Any:
        """The CueTrigger of ``spec``'s sound cue on this PC's speaker (output "laptop" only: a MuJoCo / Mock run has
        no robot speaker), or None. A missing file is reported and the motion runs without it."""
        from reachy_motion.sound import CueTrigger, cue_tick, sound_cue

        cue = sound_cue(spec)
        if cue is None or self.sound.output != "laptop":
            return None
        path = Path(self.sound.sounds_dir) / cue.file
        rec: dict[str, Any] = {"output": "laptop", "file": cue.file, "at_s": float(cue.at_s),
                               "offset_ms": float(self.sound.offset_ms), "pre_s": float(pre_s)}
        if not path.is_file():
            rec.update(ok=False, error=f"음원 파일 없음: {path}")
            self._emit("sound", **rec)
            return None
        k = cue_tick(pre_s, cue.at_s, self.sound.offset_ms, rate_hz)
        return CueTrigger(k, lambda: self.laptop.play(path), rec)

    @staticmethod
    def _sound_record(trig: Any) -> dict[str, Any] | None:
        """What the cue did (waits up to 1 s for a request still running)."""
        if trig is None:
            return None
        if trig.fired.is_set():
            trig.done.wait(1.0)
        return dict(trig.record)

    def _job_preview(self, cancel: threading.Event, spec: Any) -> None:
        from reachy_motion.safety.checker import prepare

        self._ensure_connected(cancel)
        prepared = prepare(spec, self.policy)
        if not prepared.runnable:
            self._emit("refused", label=spec.key, reasons=[v.code for v in prepared.report.violations])
            return
        if not self._to_start(spec.start, cancel):
            self._emit("preview_finished", key=spec.key, completed=False, cancelled=True, error=None)
            return
        n = len(prepared.timeline)
        self._drain_live()
        trig = self._laptop_cue(spec, 0.0, prepared.timeline.rate_hz)
        self._emit("preview_started", key=spec.key, duration_s=float(prepared.timeline.duration), n_samples=n)
        on_tick = self._live_callback(spec.key, n)
        res = self.executor.run(prepared, cancel=cancel, on_tick=on_tick if trig is None else trig.wrap(on_tick),
                                record_present=True, stop=self._stop_policy())
        self._remember(res)
        self._emit("preview_finished", key=spec.key, completed=bool(res.completed), cancelled=bool(res.cancelled),
                   error=res.error, stop_kind=res.stop_kind, ticks=tick_summary(res), sound=self._sound_record(trig))

    def _job_trial(self, cancel: threading.Event, trial: Any, session_dir: Path, notes: str = "") -> None:
        from reachy_motion.gui.experiments import check_trial
        from reachy_motion.gui.results import (load_result, run_brief, update_session_readme, write_metrics_json,
                                               write_run_readme)

        self._ensure_connected(cancel)
        check = check_trial(trial, self.policy)
        if not check.runnable:
            self._emit("refused", label=trial.label, reasons=check.notes)
            return
        if not self._to_start(trial.spec.start, cancel):
            self._emit("trial_finished", label=trial.label, stage=trial.stage, run_dir=None, completed=False,
                       cancelled=True, error=None, stop_kind=None, reason="시작 자세로 가는 전이 중에 정지됨")
            return
        prepared = check.prepared
        n = len(prepared.timeline)
        settle = self._wait_still(cancel)  # the recorded start hold should begin from a still head
        if cancel.is_set():
            self._emit("trial_finished", label=trial.label, stage=trial.stage, run_dir=None, completed=False,
                       cancelled=True, error=None, stop_kind=None, reason="머리가 멈추기를 기다리는 중에 정지됨")
            return
        self._drain_live()
        self._emit("trial_started", label=trial.label, title=trial.title, stage=trial.stage, n_samples=n,
                   duration_s=float(prepared.timeline.duration), main_channel=trial.main_channel, pre_s=trial.pre_s,
                   settle=settle)
        started_local = datetime.now().astimezone().isoformat(timespec="seconds")
        trig = self._laptop_cue(trial.spec, trial.pre_s, prepared.timeline.rate_hz)
        on_tick = self._live_callback(trial.label, n)
        result = self.executor.run(prepared, cancel=cancel, on_tick=on_tick if trig is None else trig.wrap(on_tick),
                                   record_present=True, stop=self._stop_policy())
        self._remember(result)
        sound = self._sound_record(trig)
        run_dir = next_run_dir(session_dir, trial.label)
        save_run(run_dir, trial, check, result, dict(self.connection, settle_before=settle,
                                                     **({"sound": sound} if sound else {})),
                 notes=notes, started_local=started_local)
        brief: dict[str, Any] = {}
        loaded = None
        try:  # analysed here, in the worker, after the run: the GUI thread only receives the result
            loaded = load_result(run_dir)
            write_metrics_json(run_dir / "metrics.json", loaded)
            brief = run_brief(loaded)
            write_run_readme(loaded)  # human-readable record for team members
            update_session_readme(session_dir)
        except Exception as exc:  # noqa: BLE001 - a run too short to analyse is still saved
            brief = {"analysis_error": f"{type(exc).__name__}: {exc}"}
        self._emit("trial_finished", label=trial.label, stage=trial.stage, run_dir=str(run_dir),
                   completed=bool(result.completed),
                   cancelled=bool(result.cancelled), error=result.error, stop_kind=result.stop_kind,
                   stop_reason=result.stop_reason, ticks=tick_summary(result), metrics=brief,
                   dropped_live=self.dropped_live, loaded=loaded, sound=sound)


# ---------------------------------------------------------------------------------------------- run files
def tick_summary(result: Any) -> dict[str, Any]:
    """The executor's own tick timing of a run (pc_measured on this PC)."""
    s = result.ticks.summary(int(round(1e9 / result.rate_hz)))
    iv = s.get("interval_ms") or {}
    return {"n_ticks": s.get("n_ticks"), "effective_hz": s.get("effective_hz"), "late_ticks": s.get("late_ticks"),
            "skipped_samples": s.get("skipped_samples"), "jitter_ms": s.get("jitter_ms"),
            "interval_p95_ms": iv.get("p95"), "interval_max_ms": iv.get("max"), "data_kind": "pc_measured"}


def next_run_dir(session_dir: Path | str, label: str) -> Path:
    runs = Path(session_dir) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    n = 1 + sum(1 for p in runs.iterdir() if p.is_dir())
    return runs / f"{n:02d}_{label}"


def _state_row(head: Any, antennas: Any, body: float) -> dict[str, float]:
    row = dict(pose_from_matrix(head).to_deg_mm())
    row.update(antenna_right_deg=float(antennas[0]) * DEG, antenna_left_deg=float(antennas[1]) * DEG,
               body_yaw_deg=float(body) * DEG)
    return row


def save_run(run_dir: Path, trial: Any, check: Any, result: Any, connection: dict[str, Any], notes: str = "",
             started_local: str = "") -> Path:
    """Write one trial: commands / present on ONE time base, ticks, manifest (data_kind of every file), the
    user's notes (notes.md, also in the manifest)."""
    from reachy_motion.analysis.goto_replay import JOINT_NAMES
    from reachy_motion.gui.model import _json_default, _write_rows
    from reachy_motion.sdk_env import environment_report

    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = int(result.started_ns)
    cmd_rows = []
    for k, c in result.commands:
        cmd_rows.append({"k": k, "t_ns": int(c.t_ns), "t_s": (c.t_ns - t0) / 1e9, "kind": c.kind,
                         **_state_row(c.head, c.antennas, c.body_yaw)})
    _write_rows(run_dir / "commands.csv", cmd_rows)
    kind = str(result.data_kind)
    pres_rows = []
    for p in result.present:
        row = {"t_ns": int(p.t_ns), "t_s": (p.t_ns - t0) / 1e9, **_state_row(p.head, p.antennas, p.body_yaw)}
        if p.joints is not None:
            row.update({f"{name}_deg": float(q) * DEG for name, q in zip(JOINT_NAMES, p.joints)})
        pres_rows.append(row)
    present_name = f"present_{kind}.csv"
    if pres_rows:
        _write_rows(run_dir / present_name, pres_rows)
    result.ticks.to_csv(run_dir / "ticks.csv")
    manifest = {
        "tool": LAB_TOOL,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trial": {"stage": trial.stage, "label": trial.label, "title": trial.title, "condition": trial.condition,
                  "main_channel": trial.main_channel, "pre_s": trial.pre_s, "post_s": trial.post_s},
        "spec": trial.spec.to_dict(), "spec_digest": trial.spec.digest(),
        "timeline_digest": result.timeline_digest, "check_computed": check.row(),
        "result": {"completed": result.completed, "cancelled": result.cancelled, "error": result.error,
                   "stop_kind": result.stop_kind, "stop_reason": result.stop_reason,
                   "stop_detail": result.stop_detail, "end_state_known": result.end_state_known,
                   "n_commands": len(result.commands), "n_present": len(result.present),
                   "warnings": list(result.warnings)},
        "time_base": "t_s of commands.csv and of the present file = seconds since the executor's start "
                     "(started_ns); t_ns = absolute perf_counter_ns on this PC. Command = send time, present = "
                     "arrival time (sim) / model time (mock).",
        "started_ns": t0,
        "started_local": started_local,
        "notes": notes,
        "data_kind": kind,
        "data_kind_by_file": {"commands.csv": "commanded (the values sent)", present_name: kind,
                              "ticks.csv": "pc_measured", "metrics.json": f"computed from {kind} data"},
        "data_kind_note": DATA_KIND_NOTES.get(kind, ""),
        "tick_timing_pc_measured": tick_summary(result),
        "connection": connection,
        "environment": environment_report(),
    }
    if notes.strip():
        (run_dir / "notes.md").write_text(notes.strip() + "\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
                                           encoding="utf-8")
    return run_dir


def new_session_dir(mode: str, stage: str, root: Path | str = LAB_ROOT) -> Path:
    d = Path(root) / f"{datetime.now():%Y%m%d_%H%M%S}_{mode}_{stage}"
    d.mkdir(parents=True, exist_ok=True)
    return d


__all__ = ["LabSession", "LabError", "MODES", "SIM_PORT", "LAB_ROOT", "LAB_TOOL", "DATA_KIND_NOTES", "save_run",
           "next_run_dir", "new_session_dir", "tick_summary"]
