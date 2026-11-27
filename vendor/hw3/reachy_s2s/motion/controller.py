"""MotionController: the single sender of robot motion commands (docs/PHASE4_5_DESIGN.md).

Modes (``mode``)

=============  ===============================================================================
TORQUE_OFF     motors off (the Phase 4 baseline); nothing is sent
STARTING       motors on, then a ramp from the present pose to neutral (the SDK wake path)
HOLD           motors on, neutral pose, no motion: Phase 5 "Motion OFF"
ACTIVE         motors on, state poses + event gestures + speaking motion: "Motion ON"
STOPPED        Motion Stop (safety stop 1): nothing sent, the robot holds its last target
SHUTTING_DOWN  normal shutdown: neutral -> sleep pose -> settle -> pose check -> torque off -> check
FAULT          a send, state read or check failed: nothing sent, the torque is left as it is
ESTOP          Emergency Torque OFF was requested (safety stop 2); nothing sent
=============  ===============================================================================

Threads: the motion thread ticks at ``rate_hz`` and is the only one that sends motion commands (SDK
WebSocket); ``[motion] thread_priority`` sets its Windows priority (normal by default; it needs about
2 % of one core). While the local TTS saturates every core (``[tts] threads = 0``) the thread drops to
about 33 Hz with gaps up to 0.2 s whatever its priority; a late tick then advances at most
``MAX_DT_PERIODS`` periods, so the robot lags behind the plan instead of jumping. A health thread reads the motor mode (REST, every 2 s),
the daemon's control-loop statistics (every 10 s) and watches the motion thread. Every other
thread (VAD, pipeline workers, the audio player, the GUI) only appends to deques.
``emergency_torque_off`` runs in the caller's thread over REST: it does not wait for the motion
thread or its command queue, and motion commands issued before it (e.g. a Motor ON) are dropped.
Motion errors never reach the conversation: they put the controller into FAULT and are logged.

Torso frame: every pose here (states, gestures, ramps, the neutral and sleep poses, motion.csv) is
relative to the torso, whose yaw is read at Motor ON and never commanded (``set_target`` without
``body_yaw``; the daemon's IK keeps the present body yaw target). Commands are turned into the world
frame on the way out, measured poses into the torso frame on the way in (R0 2026-10-17: the torso
stood at 8.2 deg with the torque off).
"""

from __future__ import annotations

import csv
import itertools
import logging
import math
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AppConfig
from ..evaluation.event_logger import EventLogger
from .engine import SPEAKING_MOTIONS, MotionEngine
from .limits import JointGuard, Limits, RateLimiter, clamp_box
from .pose import (
    FIELDS,
    NEUTRAL,
    SLEEP,
    SLEEP_HEAD_MATRIX,
    Pose,
    pose_from_matrix,
    rotation_distance_deg,
    torso_frame,
    translation_distance_mm,
)
from .presets import speech_started_gesture
from .sinks import MotionSink
from .speech import PlaybackTap, SpeechMotion
from .trajectory import Blend, Ramp, ramp_duration

log = logging.getLogger(__name__)

TORQUE_OFF, STARTING, HOLD, ACTIVE, STOPPED = "TORQUE_OFF", "STARTING", "HOLD", "ACTIVE", "STOPPED"
SHUTTING_DOWN, FAULT, ESTOP = "SHUTTING_DOWN", "FAULT", "ESTOP"
MODES = (TORQUE_OFF, STARTING, HOLD, ACTIVE, STOPPED, SHUTTING_DOWN, FAULT, ESTOP)
TORQUE_ON_MODES = (STARTING, HOLD, ACTIVE, STOPPED, SHUTTING_DOWN)
MOVING_MODES = (STARTING, HOLD, ACTIVE, SHUTTING_DOWN)

#: event kinds -> engine events (Cascaded: local VAD; Cloud: local VAD and/or the server's activity)
EVENT_MAP = {"vad.speech_start": "speech_start", "vad.speech_end": "speech_end", "turn.merged": "merged"}
CLOUD_ACTIVITY = {"ACTIVITY_START": "speech_start", "ACTIVITY_END": "speech_end"}
#: commands an Emergency Torque OFF cancels when they were issued before it
MOTION_COMMANDS = ("motor_on", "motor_off", "motion", "resume")
CHECK_DELAY_S = 0.3  # motor mode is read this long after a torque on / off command
SETTLE_S = 0.5  # after the sleep ramp, before the final pose check
GUARD_HOLD_LIMIT_S = 0.5  # joint guard refusing every target this long -> Motion Stop
HOLD_BLEND_S = 0.6  # ACTIVE -> HOLD (Motion OFF), STOPPED -> HOLD (Resume)
SHUTDOWN_NEUTRAL_MIN_S = 1.0  # first shutdown ramp (current pose -> neutral), at least
STATS_PERIOD_S = 5.0
HEALTH_PERIOD_S = 0.5
MOTOR_MODE_PERIOD_S = 2.0
DAEMON_STATS_PERIOD_S = 10.0
STALL_S = 1.0  # motion thread silent this long while the torque is on -> reported (watchdog)
LATE_TICK_MS = 5.0
MAX_DT_PERIODS = 1.5  # a late tick advances the limiter by at most this many periods (bounded step per command)
THREAD_PRIORITIES = {"above_normal": 1, "normal": 0, "below_normal": -1}  # Windows THREAD_PRIORITY_*
MAX_INTENSITY = 1.5  # [motion] intensity upper bound (1.0 = design amplitudes)


def _set_thread_priority(name: str) -> bool:
    """Priority of the calling thread (Windows; elsewhere unchanged)."""
    if sys.platform != "win32" or name == "normal":
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        return bool(kernel32.SetThreadPriority(kernel32.GetCurrentThread(), THREAD_PRIORITIES[name]))
    except Exception:  # noqa: BLE001 - keep the default priority
        return False


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values), q)), 2)


class MotionController:
    def __init__(self, cfg: AppConfig, sink: MotionSink, events: EventLogger | None = None,
                 session_dir: Path | None = None, clock: Callable[[], int] = time.perf_counter_ns) -> None:
        m = cfg.motion
        if m.sink not in ("robot", "dry_run"):
            raise ValueError(f"[motion] sink must be robot | dry_run, got {m.sink!r}")
        if not 5.0 <= m.rate_hz <= 100.0:
            raise ValueError("[motion] rate_hz must be between 5 and 100")
        if not 0.0 <= m.intensity <= MAX_INTENSITY:
            raise ValueError(f"[motion] intensity must be between 0 and {MAX_INTENSITY:g}")
        if m.thread_priority not in THREAD_PRIORITIES:
            raise ValueError(f"[motion] thread_priority must be one of {tuple(THREAD_PRIORITIES)}")
        self.cfg, self.mcfg = cfg, m
        self.sink = sink
        self.events = events
        self.clock = clock
        self.limits = Limits.from_config(m)
        self.guard = JointGuard(m.joint_max_change_deg, m.joint_limit_margin_deg, enabled=m.joint_guard)
        self.tap = PlaybackTap()
        self.speech = SpeechMotion(m.speech_gain, m.antenna_beat_rad, m.speech_sync_offset_ms)
        self.engine = MotionEngine(m, self.speech, clock())
        self.limiter = RateLimiter(self.limits)
        self.rate_hz = float(m.rate_hz)
        self.motion_enabled = bool(m.enabled)
        self.mode = TORQUE_OFF
        self.locked = False
        self.last_error: str | None = None
        self.needs_confirmation: str | None = None
        self.motor_mode_seen: str | None = None
        self.health_error: str | None = None
        self.estop_result: str | None = None
        self.daemon_stats: dict[str, Any] | None = None
        self.wobbling = False
        self.torso_yaw_deg = 0.0  # read at Motor ON; poses are in the torso frame (see the module doc)
        self._to_world_m = np.eye(4)
        self._to_local_m = np.eye(4)
        self._seq_counter = itertools.count(1)
        self._seq = 0  # Emergency Torque OFF generation; commands carry the one they were issued in
        self._cmd_seq = 0  # generation of the command being processed
        self._cmds: deque[tuple[str, Any, int]] = deque()
        self._inputs: deque[tuple[str, Any, int]] = deque(maxlen=1024)
        self._halt = threading.Event()
        self._estop = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._health: threading.Thread | None = None
        self._last_cmd: Pose | None = None  # last safety-checked target (sent or deadband-skipped)
        self._last_sent: Pose | None = None  # last target the robot received
        self._last_send_ns = 0
        self._last_tick_ns: int | None = None
        self._last_tick_wall = time.monotonic()
        self._stall_reported = False
        self._last_mode_read = self._last_daemon_read = -1e9
        self._ramp: Ramp | None = None
        self._phase = ""
        self._phase_t0 = 0
        self._startup_from: tuple[np.ndarray, list[float]] | None = None
        self._final_pose: tuple[float, float] | None = None
        self._hold_blend: Blend | None = None
        self._preview: tuple[Callable[[float], Pose], int] | None = None
        self._guard_fail_since: int | None = None
        self._attached: tuple[Any, Any] | None = None
        self.session_dir = session_dir
        self._csv_fh: Any = None
        self._csv: Any = None
        self._csv_flush_ns = 0
        self._csv_lock = threading.RLock()  # the motion thread writes; a GUI session switch closes / redirects
        self.totals = {"ticks": 0, "sends": 0, "faults": 0, "guard_rejected": 0, "estops": 0}
        self._stats = self._fresh_stats()
        self._stats_t0 = clock()
        self._cpu_t0 = time.thread_time()

    # ------------------------------------------------------------------ wiring
    def attach(self, state: Any, events: EventLogger | None) -> None:
        """Follow a StateTracker and an event log (call again after a session switch)."""
        self.detach()
        if events is not None:
            self.events = events
            events.add_listener(self.on_event)
        if state is not None:
            state.add_listener(self.on_state)
            self._inputs.append(("state", state.state.value, self.clock()))
        self._attached = (state, events)

    def detach(self) -> None:
        if self._attached is None:
            return
        state, events = self._attached
        if state is not None and hasattr(state, "remove_listener"):
            state.remove_listener(self.on_state)
        if events is not None:
            events.remove_listener(self.on_event)
        self._attached = None

    def set_session_dir(self, path: Path | None) -> None:
        with self._csv_lock:  # never close the file under the motion thread's writer (-> 'tick failed' FAULT)
            self._close_csv()
            self.session_dir = path

    def on_state(self, old: Any, new: Any, record: dict[str, Any]) -> None:  # any thread, never blocks
        self._inputs.append(("state", getattr(new, "value", str(new)), self.clock()))

    def on_event(self, record: dict[str, Any]) -> None:  # any thread, never blocks
        kind = record.get("kind", "")
        if kind == "cloud.voice_activity":
            mapped = CLOUD_ACTIVITY.get(record.get("type"))
        else:
            mapped = EVENT_MAP.get(kind)
        if mapped is not None:
            self._inputs.append(("event", mapped, self.clock()))

    # ----------------------------------------------------------- operator API
    def motor_on(self, force: bool = False) -> None:
        """Torque on, then ramp to neutral. An unknown initial pose needs ``force`` (operator confirmed)."""
        self._command("motor_on", force)

    def motor_off(self) -> None:
        """Normal shutdown: neutral -> sleep pose -> checks -> torque off (never on a failed check)."""
        self._command("motor_off", None)

    def set_motion(self, enabled: bool) -> None:
        """Motion ON (ACTIVE) / OFF (HOLD: neutral pose, torque stays on)."""
        self._check_unlocked()
        self._command("motion", bool(enabled))

    def motion_stop(self) -> None:
        """Safety stop 1: from the next tick nothing is sent; the robot holds its last target (torque on)."""
        self._halt.set()
        self._command("halt", None)

    def resume(self) -> None:
        """Leave Motion Stop: back to the neutral hold (and Motion ON if it is enabled)."""
        self._command("resume", None)

    def emergency_torque_off(self) -> str:
        """Safety stop 2: torque off right now from this thread (REST; independent of the motion queue).

        Motion commands issued before this call (Motor ON, Resume, ...) are dropped. Blocks for at
        most the REST timeout (2 s) plus the WebSocket fallback; call it from a worker thread in a GUI.
        """
        self._seq = next(self._seq_counter)
        self._estop.set()
        self._halt.set()
        self.totals["estops"] += 1
        t0 = time.perf_counter_ns()
        try:
            via = self.sink.emergency_torque_off()
            result = f"ok via {via}" if via else "ok"
        except Exception as exc:  # noqa: BLE001 - reported to the operator
            result = f"FAILED: {exc!r}"
            self.last_error = f"Emergency Torque OFF {result}"
        self.estop_result = result
        self._log("estop", result=result, ms=round((time.perf_counter_ns() - t0) / 1e6, 1), mode_before=self.mode)
        return result

    def set_intensity(self, value: float) -> None:
        self._check_unlocked()
        value = float(value)
        if not 0.0 <= value <= 1.5:
            raise ValueError("intensity must be between 0 and 1.5")
        self._command("intensity", value)

    def set_speaking_motion(self, kind: str) -> None:
        self._check_unlocked()
        if kind not in SPEAKING_MOTIONS:
            raise ValueError(f"speaking motion must be one of {SPEAKING_MOTIONS}")
        if kind == "sdk" and not self.mcfg.allow_sdk_wobble:
            raise PermissionError("SDK wobbling is disabled ([motion] allow_sdk_wobble = false): it bypasses "
                                  "the controller's limits and needs a separate approval")
        self._command("speaking", kind)

    def set_speech_gesture(self, kind: str) -> None:
        self._check_unlocked()
        speech_started_gesture(kind, self.mcfg.nod_deg)  # validates
        self._command("gesture", kind)

    def set_rate(self, hz: float) -> None:
        self._check_unlocked()
        if not 5.0 <= hz <= 100.0:
            raise ValueError("rate must be between 5 and 100 Hz")
        self._command("rate", float(hz))

    def set_preview(self, fn: Callable[[float], Pose] | None) -> None:
        """Preview / test hook: while ACTIVE, follow ``NEUTRAL + fn(s since set) * intensity`` instead of the
        state motion (``None`` returns to it). The targets still pass the box, joint guard and limiter."""
        self._command("preview", fn)

    def lock(self, locked: bool) -> None:
        """While a conversation runs the motion condition is fixed (Phase 5). Safety stops stay available."""
        self.locked = locked

    def _check_unlocked(self) -> None:
        if self.locked:
            raise RuntimeError("motion settings are fixed while a conversation runs")

    def _command(self, name: str, arg: Any) -> None:
        self._cmds.append((name, arg, self._seq))

    # ------------------------------------------------------------------ frames
    def to_world(self, head_local: np.ndarray) -> np.ndarray:
        return self._to_world_m @ head_local

    def to_local(self, head_world: np.ndarray) -> np.ndarray:
        return self._to_local_m @ head_world

    def _set_torso_yaw(self, yaw_deg: float) -> None:
        self.torso_yaw_deg = float(yaw_deg)
        self._to_world_m = torso_frame(self.torso_yaw_deg)
        self._to_local_m = self._to_world_m.T  # pure rotation: inverse = transpose
        self.guard.set_body_yaw(self.torso_yaw_deg)

    def _present_local(self) -> tuple[np.ndarray, list[float]]:
        head, antennas = self.sink.present()
        return self.to_local(np.asarray(head, dtype=np.float64)), list(antennas)

    # ------------------------------------------------------------------ status
    @property
    def display_mode(self) -> str:
        """``mode``, or ESTOP as soon as an emergency stop was requested (even if the motion thread is stuck)."""
        return ESTOP if self._estop.is_set() else self.mode

    def label(self) -> str:
        """Motion condition for result rows: torque_off | hold | on:<intensity>:<speaking>:<gesture> | <mode>."""
        mode = self.display_mode
        if mode == ACTIVE:
            return f"on:{self.engine.intensity:g}:{self.engine.speaking_motion}:{self.engine.speech_started}"
        return {TORQUE_OFF: "torque_off", HOLD: "hold"}.get(mode, mode.lower())

    def status(self) -> dict[str, Any]:
        return {"mode": self.display_mode, "state": self.engine.state, "label": self.label(), "sink": self.sink.name,
                "simulated": self.sink.simulated, "motion_enabled": self.motion_enabled,
                "intensity": self.engine.intensity, "speaking_motion": self.engine.speaking_motion,
                "speech_started": self.engine.speech_started, "rate_hz": self.rate_hz, "locked": self.locked,
                "wobbling": self.wobbling, "motor_mode": self.motor_mode_seen, "last_error": self.last_error,
                "health_error": self.health_error, "needs_confirmation": self.needs_confirmation,
                "estop_requested": self._estop.is_set(), "estop_result": self.estop_result,
                "halted": self._halt.is_set(), "guard": self.guard.available,
                "torso_yaw_deg": round(self.torso_yaw_deg, 2), **self.totals}

    def wait_for(self, modes: tuple[str, ...], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.mode in modes:
                return True
            time.sleep(0.02)
        return self.mode in modes

    # --------------------------------------------------------------- lifecycle
    def start(self, threads: bool = True) -> None:
        self._log("started", sink=self.sink.name, simulated=self.sink.simulated, guard=self.guard.available,
                  guard_error=self.guard.error, motion=asdict(self.mcfg))
        if not threads:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="motion", daemon=True)
        self._thread.start()
        self._health = threading.Thread(target=self._health_loop, name="motion-health", daemon=True)
        self._health.start()

    def close(self, timeout: float = 5.0) -> None:
        """Stop the threads. Does not move the robot: run ``motor_off`` first and wait for it."""
        self._stop.set()
        for t in (self._thread, self._health):
            if t is not None:
                t.join(timeout=timeout)
        self._thread = self._health = None
        if self._estop.is_set() and self.mode != ESTOP:  # the thread may have been stuck until now
            self._set_mode(ESTOP, "emergency torque off")
        self.detach()
        self._log("closed", final_mode=self.mode, totals=dict(self.totals))
        self._close_csv()

    def _run(self) -> None:
        if not _set_thread_priority(self.mcfg.thread_priority) and self.mcfg.thread_priority != "normal":
            log.debug("motion thread priority unchanged")
        self._thread_id = threading.get_ident()
        self._cpu_t0 = time.thread_time()
        next_t = self.clock()
        last = None
        while not self._stop.is_set():
            now = self.clock()
            st = self._stats
            late_ms = (now - next_t) / 1e6
            if late_ms > LATE_TICK_MS:
                st["late_ticks"] += 1
            st["max_late_ms"] = max(st["max_late_ms"], late_ms)
            if last is not None:
                st["intervals_ms"].append((now - last) / 1e6)
            last = now
            self.tick(now)
            self._last_tick_wall = time.monotonic()
            period = int(1e9 / self.rate_hz)
            next_t += period
            after = self.clock()
            if next_t < after:  # overran (e.g. a slow send): skip the missed ticks instead of bursting
                missed = (after - next_t) // period + 1
                self._stats["skipped_ticks"] += missed
                next_t += missed * period
            # time.sleep is high resolution on Windows (Python >= 3.11); Event.wait needs timeBeginPeriod
            time.sleep(max(0.0, (next_t - self.clock()) / 1e9))

    def _health_loop(self) -> None:
        while not self._stop.wait(HEALTH_PERIOD_S):
            self.health_check(time.monotonic())

    def health_check(self, now: float) -> None:
        """Watchdog + motor mode (REST, every 2 s) + daemon statistics (every 10 s); health thread."""
        if self.mode in TORQUE_ON_MODES and self._thread is not None:
            stalled = now - self._last_tick_wall
            if stalled > STALL_S and not self._stall_reported:
                self._stall_reported = True
                self.last_error = (f"motion thread stalled for {stalled:.1f} s (stuck send?) - "
                                   "use Emergency Torque OFF if the robot misbehaves")
                self._log("error", error=self.last_error)
            elif stalled <= STALL_S:
                self._stall_reported = False
        if self.mode != TORQUE_OFF and now - self._last_mode_read >= MOTOR_MODE_PERIOD_S:
            self._last_mode_read = now
            try:
                mode = self.sink.motor_mode()
                self.health_error = None
            except Exception as exc:  # noqa: BLE001 - REST hiccup; the motion thread decides on faults
                mode, self.health_error = None, f"motor mode read failed: {exc!r}"
            self.motor_mode_seen = mode
            if mode is not None and mode != "enabled" and self.mode in (HOLD, ACTIVE):
                self._command("external_change", mode)
        if now - self._last_daemon_read >= DAEMON_STATS_PERIOD_S:
            self._last_daemon_read = now
            try:
                self.daemon_stats = self.sink.daemon_stats()
                self._log("daemon", mode=self.mode, stats=self.daemon_stats)
            except Exception as exc:  # noqa: BLE001
                log.debug("daemon stats read failed: %r", exc)

    # -------------------------------------------------------------------- tick
    def tick(self, now: int | None = None) -> None:
        """One control step. Never raises: any failure (REST, WebSocket, a bug) puts the controller in FAULT."""
        try:
            self._tick(self.clock() if now is None else now)
        except Exception as exc:  # noqa: BLE001 - a motion failure must never reach the conversation
            log.exception("motion tick failed")
            self._fault(f"tick failed: {exc!r}")

    def _tick(self, now: int) -> None:
        period = 1.0 / self.rate_hz
        dt = period if self._last_tick_ns is None else max(1e-4, (now - self._last_tick_ns) / 1e9)
        if dt > MAX_DT_PERIODS * period:  # stalled (CPU contention): lag behind instead of jumping
            self._stats["stalled_ticks"] += 1
            dt = MAX_DT_PERIODS * period
        self._last_tick_ns = now
        self._stats["ticks"] += 1
        self.totals["ticks"] += 1
        if self._estop.is_set() and self.mode != ESTOP:
            self._set_mode(ESTOP, "emergency torque off")
            self.speech.discard(now)
            self._ramp = self._hold_blend = None
        self._process_commands(now)
        self._drain_inputs(now)
        if self._estop.is_set() or self.mode in (TORQUE_OFF, FAULT, ESTOP):
            self._maybe_stats(now)
            return
        if self._halt.is_set():
            if self.mode != STOPPED:
                self._set_wobbling(False)
                self._set_mode(STOPPED, "motion stop: holding the last target")
            self._maybe_stats(now)
            return
        if self.mode == STARTING:
            self._tick_starting(now, dt)
        elif self.mode == SHUTTING_DOWN:
            self._tick_shutdown(now, dt)
        elif self.mode == HOLD:
            target = NEUTRAL if self._hold_blend is None else NEUTRAL + self._hold_blend.at(now)
            self._emit(target, now, dt, "conversation")
        elif self.mode == ACTIVE:
            if self._preview is not None:
                fn, t0 = self._preview
                target = NEUTRAL + fn((now - t0) / 1e9).scaled(self.engine.intensity)
            else:
                target = self.engine.step(now)
            self._emit(target, now, dt, "conversation")
        self._maybe_stats(now)

    def _drain_inputs(self, now: int) -> None:
        while self._inputs:
            try:
                kind, value, t_in = self._inputs.popleft()
            except IndexError:
                break
            if kind == "state":
                if self.engine.set_state(value, now) and self.mode == ACTIVE:
                    self._log("state", state=value, cancel_ms=round((now - t_in) / 1e6, 1))
            elif kind == "event":
                gesture = self.engine.on_event(value, now)
                if gesture is not None and self.mode == ACTIVE:
                    self._log("gesture", gesture=gesture, event=value, latency_ms=round((now - t_in) / 1e6, 1))
        items = self.tap.drain()
        if self.mode == ACTIVE and self.engine.speaking_motion == "custom":
            self.speech.push(items, now)
        elif items:
            self.speech.discard(now)

    # ---------------------------------------------------------------- commands
    def _process_commands(self, now: int) -> None:
        while self._cmds:
            try:
                name, arg, seq = self._cmds.popleft()
            except IndexError:
                return
            if name in MOTION_COMMANDS and seq != self._seq:
                self._log("command_dropped", command=name, reason="issued before an Emergency Torque OFF")
                continue
            if self.mode == ESTOP and name in MOTION_COMMANDS and name != "motor_on":
                self._log("command_ignored", command=name, mode=self.mode)
                continue
            self._cmd_seq = seq
            try:
                getattr(self, f"_cmd_{name}")(arg, now)
            except Exception as exc:  # noqa: BLE001 - e.g. a REST / WebSocket failure during a step
                log.exception("motion command %s failed", name)
                self._fault(f"{name} failed: {exc!r}")

    def _cmd_motor_on(self, force: bool, now: int) -> None:
        if self.mode in MOVING_MODES:
            self._log("command_ignored", command="motor_on", mode=self.mode)
            return
        seq = self._cmd_seq
        self.needs_confirmation = None
        mode_before = self.sink.motor_mode()
        self._set_torso_yaw(math.degrees(self.sink.body_yaw()))  # held as it is: never commanded
        head, antennas = self._present_local()
        start = pose_from_matrix(head, antennas)
        near = self._near_known_pose(head)
        if near is None and not force:
            self.needs_confirmation = (f"unknown initial pose {start.rounded()} (torso frame; torso yaw "
                                       f"{self.torso_yaw_deg:.1f} deg)")
            self._log("needs_confirmation", start=start.rounded(), motor_mode=mode_before,
                      torso_yaw_deg=round(self.torso_yaw_deg, 2))
            return
        self._estop.clear()
        self._halt.clear()
        if seq != self._seq:  # an Emergency Torque OFF came in meanwhile
            self._estop.set()
            self._halt.set()
            return
        self.last_error = self.estop_result = None
        self._set_wobbling(False)  # nothing else may move the head
        self.sink.enable_motors()  # the daemon pins every target to the present pose first (no jump)
        if seq != self._seq:  # ... or while the torque was being turned on: make sure it is off again
            self.emergency_torque_off()
            return
        self._startup_from = (head, list(antennas))
        self._phase, self._phase_t0 = "enable_check", now
        self._set_mode(STARTING, "motors on requested", motor_mode_before=mode_before, start=start.rounded(),
                       start_pose=near or "unknown (confirmed by the operator)",
                       torso_yaw_deg=round(self.torso_yaw_deg, 2))

    def _cmd_motor_off(self, _arg: Any, now: int) -> None:
        if self.mode in (TORQUE_OFF, ESTOP, SHUTTING_DOWN):
            self._log("command_ignored", command="motor_off", mode=self.mode)
            return
        if self.mode == FAULT and self.sink.motor_mode() == "disabled":
            self._halt.clear()
            self._last_cmd = self._last_sent = None
            self._set_mode(TORQUE_OFF, "normal shutdown: the motors were already off")
            return
        moving = self.mode in MOVING_MODES
        self._halt.clear()
        self._set_wobbling(False)
        if moving and self._last_cmd is not None:
            start = self._last_cmd  # keep the limiter's velocity: no jerk at the switch
        else:  # STOPPED / FAULT: the robot rests at the last target sent (or wherever it is)
            if self._last_sent is not None:
                start = self._last_sent
            else:
                head, antennas = self._present_local()
                start = pose_from_matrix(head, antennas)
            self.limiter.reset(start)
            self._last_cmd = start
        dur = ramp_duration(start, NEUTRAL, self.limits, SHUTDOWN_NEUTRAL_MIN_S)
        self._ramp = Ramp(start.head_matrix(), NEUTRAL.head_matrix(), tuple(start.antennas),
                          tuple(NEUTRAL.antennas), now, dur)
        self._phase = "to_neutral"
        self._hold_blend = None
        self.speech.discard(now)
        self._set_mode(SHUTTING_DOWN, "normal shutdown requested", ramp_s=round(dur, 2))

    def _cmd_motion(self, enabled: bool, now: int) -> None:
        self.motion_enabled = enabled
        self._log("setting", motion_enabled=enabled)
        if enabled and self.mode == HOLD:
            self._enter_active(now)
        elif not enabled and self.mode == ACTIVE:
            self._set_wobbling(False)
            dev = (self._last_cmd - NEUTRAL) if self._last_cmd is not None else Pose()
            self._hold_blend = Blend(dev, Pose(), now, HOLD_BLEND_S)
            self.speech.discard(now)
            self._set_mode(HOLD, "motion off (neutral hold)")

    def _cmd_halt(self, _arg: Any, now: int) -> None:
        self._log("motion_stop_requested", mode=self.mode)

    def _cmd_resume(self, _arg: Any, now: int) -> None:
        if self.mode != STOPPED:
            self._log("command_ignored", command="resume", mode=self.mode)
            return
        mode = self.sink.motor_mode()
        self._halt.clear()
        if mode != "enabled":
            self._set_mode(TORQUE_OFF if mode == "disabled" else FAULT, f"resume refused: motors are {mode!r}")
            return
        held = self._last_sent if self._last_sent is not None else NEUTRAL
        self.limiter.reset(held)  # the robot is at rest at the last target sent
        self._last_cmd = held
        self._hold_blend = Blend(held - NEUTRAL, Pose(), now, HOLD_BLEND_S)
        self._set_mode(HOLD, "resumed from motion stop")
        if self.motion_enabled:
            self._enter_active(now)

    def _cmd_intensity(self, value: float, now: int) -> None:
        self.engine.intensity = value
        self._log("setting", intensity=value)

    def _cmd_speaking(self, kind: str, now: int) -> None:
        self.engine.speaking_motion = kind
        self.speech.discard(now)
        if self.mode == ACTIVE:
            self._apply_speaking_mode()
        self._log("setting", speaking_motion=self.engine.speaking_motion)

    def _cmd_gesture(self, kind: str, now: int) -> None:
        self.engine.speech_started = kind
        self._log("setting", speech_started_gesture=kind)

    def _cmd_rate(self, hz: float, now: int) -> None:
        self.rate_hz = hz
        self._log("setting", rate_hz=hz)

    def _cmd_preview(self, fn: Callable[[float], Pose] | None, now: int) -> None:
        if fn is None and self._preview is not None:
            self._rebase_engine(now)  # continue smoothly into the state pose
        self._preview = None if fn is None else (fn, now)
        self._log("preview", on=fn is not None)

    def _cmd_external_change(self, mode: str, now: int) -> None:
        if self.mode in (HOLD, ACTIVE):
            self._fault(f"motors are {mode!r} although the controller had them on (changed outside)")

    def _rebase_engine(self, now: int) -> None:
        """Start the engine's move into its state pose from the last command (engine works unscaled)."""
        k = self.engine.intensity
        dev = (self._last_cmd - NEUTRAL) if self._last_cmd is not None else Pose()
        self.engine.rebase(dev.scaled(1.0 / k) if k > 1e-6 else Pose(), now)

    def _enter_active(self, now: int) -> None:
        self._rebase_engine(now)
        self._hold_blend = None
        self._set_mode(ACTIVE, "motion on")
        self._apply_speaking_mode()

    def _apply_speaking_mode(self) -> None:
        """Custom and SDK speaking motion never run together; SDK wobbling needs explicit approval."""
        if self.engine.speaking_motion == "sdk":
            if not self.mcfg.allow_sdk_wobble:
                self._log("sdk_wobble_refused", reason="[motion] allow_sdk_wobble = false (bypasses our limits)")
                self.engine.speaking_motion = "off"
                self._set_wobbling(False)
                return
            self._set_wobbling(True)
        else:
            self._set_wobbling(False)

    def _set_wobbling(self, on: bool) -> None:
        if self.wobbling == on:
            return
        try:
            self.sink.set_wobbling(on)
            self.wobbling = on
            self._log("wobbling", on=on)
        except Exception as exc:  # noqa: BLE001
            self._log("error", error=f"set_wobbling({on}) failed: {exc!r}")
            if on:
                self.engine.speaking_motion = "off"

    # ------------------------------------------------------------ mode ticks
    def _tick_starting(self, now: int, dt: float) -> None:
        if self._phase == "enable_check":
            if (now - self._phase_t0) / 1e9 < CHECK_DELAY_S:
                return
            mode = self.sink.motor_mode()
            if mode != "enabled":
                self._fault(f"motors did not turn on (motor mode {mode!r})")
                return
            head, antennas = self._startup_from
            start = pose_from_matrix(head, antennas)
            dur = ramp_duration(start, NEUTRAL, self.limits, self.mcfg.startup_ramp_s)
            self._ramp = Ramp(head, NEUTRAL.head_matrix(), tuple(antennas), tuple(NEUTRAL.antennas), now, dur)
            self.limiter.reset(start)
            self._last_cmd = start
            self._phase = "ramp"
            self._log("ramp", phase="to_neutral", seconds=round(dur, 2))
            return
        if not self._emit(self._ramp.pose_at(now), now, dt, "ramp"):
            return
        if self._ramp.done(now):
            self._ramp = None
            self._hold_blend = None
            self._set_mode(HOLD, "at neutral")
            if self.motion_enabled:
                self._enter_active(now)

    def _tick_shutdown(self, now: int, dt: float) -> None:
        if self._phase in ("to_neutral", "to_sleep"):
            if not self._emit(self._ramp.pose_at(now), now, dt, "ramp"):
                return
            if not self._ramp.done(now):
                return
            if self._phase == "to_neutral":
                dur = ramp_duration(NEUTRAL, SLEEP, self.limits, self.mcfg.shutdown_ramp_s)
                self._ramp = Ramp(NEUTRAL.head_matrix(), SLEEP_HEAD_MATRIX, tuple(NEUTRAL.antennas),
                                  tuple(SLEEP.antennas), now, dur)
                self._phase = "to_sleep"
                self._log("ramp", phase="to_sleep", seconds=round(dur, 2))
            else:
                self._ramp = None
                self._phase, self._phase_t0 = "settle", now
            return
        if self._phase == "settle":
            if (now - self._phase_t0) / 1e9 < SETTLE_S:
                return
            try:
                head, _antennas = self._present_local()
            except Exception as exc:  # noqa: BLE001
                self._fault(f"cannot read the final pose ({exc!r}); torque left ON")
                return
            rot = rotation_distance_deg(head, SLEEP_HEAD_MATRIX)
            trans = translation_distance_mm(head, SLEEP_HEAD_MATRIX)
            if rot > self.mcfg.pose_tolerance_deg or trans > self.mcfg.pose_tolerance_mm:
                self._fault(f"final pose is {rot:.1f} deg / {trans:.1f} mm from the sleep pose; torque left ON")
                return
            self._final_pose = (rot, trans)
            self.sink.disable_motors()  # a failure raises -> FAULT (torque state unknown, reported)
            self._phase, self._phase_t0 = "torque_check", now
            return
        if self._phase == "torque_check" and (now - self._phase_t0) / 1e9 >= CHECK_DELAY_S:
            mode = self.sink.motor_mode()
            if mode != "disabled":
                self._fault(f"torque off not confirmed (motor mode {mode!r})")
                return
            rot, trans = self._final_pose or (None, None)
            self._last_cmd = self._last_sent = None
            self._phase = ""
            self._set_mode(TORQUE_OFF, "normal shutdown complete",
                           final_rotation_deg=None if rot is None else round(rot, 2),
                           final_translation_mm=None if trans is None else round(trans, 2))

    # ---------------------------------------------------------------- output
    def _emit(self, desired: Pose, now: int, dt: float, kind: str) -> bool:
        """Safety chain, then send. False if nothing could be sent (guard refusal or send failure)."""
        guard_mode = "conversation" if kind == "conversation" else "ramp"
        if kind == "conversation":
            desired, clamped = clamp_box(desired, self.limits)
            if clamped:
                self._stats["box_clamped"] += 1
            fitted, res = self.guard.fit(desired, NEUTRAL, guard_mode)
        else:
            res = self.guard.check(desired, guard_mode)
            fitted = desired if res.ok else None
        if fitted is not None:
            if res.scale < 1.0:
                self._stats["guard_scaled"] += 1
            out, limited = self.limiter.step(fitted, dt)
            if limited:
                self._stats["limited"] += 1
            final = self.guard.check(out, guard_mode)
            if final.ok:
                self._guard_fail_since = None
                return self._send(out, now)
            res = final
        # refused: send nothing, the robot holds the last target
        self._stats["guard_rejected"] += 1
        self.totals["guard_rejected"] += 1
        if self._guard_fail_since is None:
            self._guard_fail_since = now
            self._log("guard", action="hold", chain=kind, target=desired.rounded(), **res.summary())
        elif (now - self._guard_fail_since) / 1e9 > GUARD_HOLD_LIMIT_S and not self._halt.is_set():
            self._halt.set()
            self._log("guard", action="motion stop", chain=kind, target=desired.rounded(), **res.summary())
        if self._last_sent is not None:
            self.limiter.reset(self._last_sent)
        return False

    def _within_deadband(self, a: Pose, b: Pose) -> bool:
        d = np.abs(a.vec() - b.vec())
        m = self.mcfg
        return bool(np.all(d[0:3] < m.deadband_deg) and np.all(d[3:6] < m.deadband_mm)
                    and np.all(d[6:8] < m.deadband_rad))

    def _send(self, pose: Pose, now: int) -> bool:
        prev, self._last_cmd = self._last_cmd, pose
        last = self._last_sent
        if (last is not None and (now - self._last_send_ns) / 1e9 < self.mcfg.keepalive_s
                and self._within_deadband(pose, last)):
            # skip a small change, unless the target has come to rest here (then send the exact pose once)
            settled = prev is not None and bool(np.all(np.abs(pose.vec() - prev.vec()) < 1e-9))
            if not settled or bool(np.all(np.abs(pose.vec() - last.vec()) < 1e-9)):
                self._stats["deadband_skips"] += 1
                return True
        t0 = time.perf_counter_ns()
        try:
            self.sink.send(self.to_world(pose.head_matrix()), pose.antennas)
        except Exception as exc:  # noqa: BLE001 - communication failure: stop, leave the torque alone
            self._fault(f"send failed: {exc!r}")
            return False
        send_ms = (time.perf_counter_ns() - t0) / 1e6
        self._last_sent, self._last_send_ns = pose, now
        st = self._stats
        st["sends"] += 1
        self.totals["sends"] += 1
        st["send_ms"].append(send_ms)
        self._write_csv(now, pose, send_ms)
        return True

    # ------------------------------------------------------------- bookkeeping
    def _near_known_pose(self, head: np.ndarray) -> str | None:
        tol_d, tol_mm = self.mcfg.pose_tolerance_deg, self.mcfg.pose_tolerance_mm
        for name, ref in (("sleep", SLEEP_HEAD_MATRIX), ("neutral", np.eye(4))):
            if rotation_distance_deg(head, ref) <= tol_d and translation_distance_mm(head, ref) <= tol_mm:
                return name
        return None

    def _set_mode(self, mode: str, reason: str, **fields: Any) -> None:
        old, self.mode = self.mode, mode
        self._log("mode", from_mode=old, to_mode=mode, reason=reason, **fields)

    def _fault(self, reason: str) -> None:
        self.last_error = reason
        self.totals["faults"] += 1
        if self.mode == ESTOP or self._estop.is_set():
            self._log("error", error=reason)
            return
        if self.wobbling:
            try:
                self.sink.set_wobbling(False)
                self.wobbling = False
            except Exception:  # noqa: BLE001
                pass
        self._ramp = self._hold_blend = None
        self._phase = ""
        self._set_mode(FAULT, reason)
        self._log("error", error=reason)

    def _log(self, kind: str, **fields: Any) -> None:
        if self.events is not None:
            try:
                self.events.log(f"motion.{kind}", **fields)
            except Exception:  # noqa: BLE001 - logging must never break motion
                log.exception("motion event logging failed")

    @staticmethod
    def _fresh_stats() -> dict[str, Any]:
        return {"ticks": 0, "sends": 0, "send_ms": [], "intervals_ms": [], "deadband_skips": 0, "late_ticks": 0,
                "max_late_ms": 0.0, "skipped_ticks": 0, "stalled_ticks": 0, "limited": 0, "box_clamped": 0,
                "guard_scaled": 0, "guard_rejected": 0}

    def _stats_summary(self, now: int) -> dict[str, Any]:
        s = self._stats
        wall = max(1e-9, (now - self._stats_t0) / 1e9)
        cpu = None
        if threading.get_ident() == self._thread_id:  # thread_time() measures the calling thread
            cpu = round(100.0 * (time.thread_time() - self._cpu_t0) / wall, 2)
        return {"mode": self.mode, "state": self.engine.state, "rate_hz": self.rate_hz, "seconds": round(wall, 2),
                "ticks": s["ticks"], "tick_hz": round(s["ticks"] / wall, 1),
                "interval_ms_p50": _percentile(s["intervals_ms"], 50),
                "interval_ms_p95": _percentile(s["intervals_ms"], 95),
                "interval_ms_max": _percentile(s["intervals_ms"], 100),
                "late_ticks": s["late_ticks"], "max_late_ms": round(s["max_late_ms"], 2),
                "skipped_ticks": s["skipped_ticks"], "stalled_ticks": s["stalled_ticks"], "sends": s["sends"],
                "send_hz": round(s["sends"] / wall, 1),
                "send_ms_p50": _percentile(s["send_ms"], 50), "send_ms_max": _percentile(s["send_ms"], 100),
                "deadband_skips": s["deadband_skips"], "limited": s["limited"], "box_clamped": s["box_clamped"],
                "guard_scaled": s["guard_scaled"], "guard_rejected": s["guard_rejected"], "thread_cpu_pct": cpu,
                "speech_blocks": self.speech.blocks_fed, "speech_hops": self.speech.hops,
                "speech_flushes": self.speech.flushes, "speech_stale_drops": self.speech.dropped_stale}

    def _maybe_stats(self, now: int) -> None:
        if (now - self._stats_t0) / 1e9 < STATS_PERIOD_S:
            return
        self._log("stats", **self._stats_summary(now))
        self._stats = self._fresh_stats()
        self._stats_t0 = now
        self._cpu_t0 = time.thread_time()

    def _write_csv(self, now: int, pose: Pose, send_ms: float) -> None:
        with self._csv_lock:
            if not self.mcfg.log_commands or self.session_dir is None:
                return
            if self._csv is None:
                try:
                    self._csv_fh = (Path(self.session_dir) / "motion.csv").open("a", newline="", encoding="utf-8")
                    self._csv = csv.writer(self._csv_fh)
                    if self._csv_fh.tell() == 0:
                        self._csv.writerow(["t", "mode", "state", *FIELDS, "send_ms"])
                except OSError:
                    self.session_dir = None
                    return
            t = self.events.rel(now) if self.events is not None else round(now / 1e9, 6)
            self._csv.writerow([t, self.mode, self.engine.state, *(round(float(v), 4) for v in pose.vec()),
                                round(send_ms, 3)])
            if now - self._csv_flush_ns > 1_000_000_000:
                self._csv_fh.flush()
                self._csv_flush_ns = now

    def _close_csv(self) -> None:
        with self._csv_lock:
            if self._csv_fh is not None:
                try:
                    self._csv_fh.close()
                except OSError:
                    pass
            self._csv_fh = self._csv = None
