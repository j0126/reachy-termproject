"""MotionService: the non-blocking facade for the Cafe FSM, the CLI and the Motion Editor (ARCHITECTURE section 7).

Threads:
* the caller - ``play`` / ``cancel`` / ``stop`` only enqueue and return at once;
* a preparer - plans and checks each requested motion against the initial policy (``safety.checker.prepare``,
  70-140 ms for the drafts); a motion outside the policy ends REJECTED with an event (no exception, no command);
  only a motion that passed may preempt the running one;
* one worker - the ONLY thread that touches the backend. Before each motion it plans a transition from the last
  commanded state (or ``backend.present()`` before the first command) to the motion's start when they differ by more
  than ``library.compare.SAME_POSE_TOL_DEG`` / ``_MM`` (so -10 vs -9.998 deg antennas never add a blend), checks the
  composite timeline (``check_timeline``) and streams it through the ``Executor`` (50 Hz, tick timing recorded).

Preemption happens at the next tick (the executor's cancel check): a play with a higher priority than the running
motion, or ``interrupt=True``, preempts it; the new motion then blends from the last commanded sample.
"""

from __future__ import annotations

import itertools
import logging
import math
import threading
import time
from collections.abc import Callable
from enum import Enum, IntEnum
from typing import Any

from reachy_motion.executor.executor import ExecutionResult, Executor
from reachy_motion.library.compare import SAME_POSE_TOL_DEG, SAME_POSE_TOL_MM
from reachy_motion.library.pose import REST, RobotState, pose_from_matrix, rotation_distance, translation_distance
from reachy_motion.library.spec import Easing, MotionSpec
from reachy_motion.planner import concat, plan_transition, transition_duration
from reachy_motion.safety.checker import prepare, prepare_timeline
from reachy_motion.safety.policy import DEFAULT_POLICY, SafetyPolicy
from reachy_motion.timeline import PreparedMotion

log = logging.getLogger(__name__)
EventListener = Callable[[dict[str, Any]], None]


class Priority(IntEnum):
    IDLE = 0
    NORMAL = 1
    REACTION = 2
    CRITICAL = 3


class HandleState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    CANCELLED = "CANCELLED"
    PREEMPTED = "PREEMPTED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


TERMINAL = frozenset({HandleState.DONE, HandleState.CANCELLED, HandleState.PREEMPTED, HandleState.REJECTED,
                      HandleState.FAILED})


class MotionHandle:
    """One requested motion. ``wait`` blocks until a terminal state; ``cancel`` is non-blocking."""

    _ids = itertools.count(1)

    def __init__(self, service: "MotionService", key: str, priority: Priority, interrupt: bool,
                 spec: MotionSpec | None) -> None:
        self.id = next(self._ids)
        self.key = key
        self.priority = Priority(priority)
        self.interrupt = bool(interrupt)
        self.spec = spec
        self.state = HandleState.PENDING
        self.report: Any = None
        self.result: ExecutionResult | None = None
        self.reason: str | None = None
        self.prepared: PreparedMotion | None = None
        self._service = service
        self._done = threading.Event()
        self._cancel = threading.Event()
        self._preempted = False

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def cancel(self) -> None:
        self._service._cancel_handle(self)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def __repr__(self) -> str:
        return f"MotionHandle(#{self.id} {self.key} {self.state.value})"


def same_state(a: RobotState, b: RobotState, tol_deg: float = SAME_POSE_TOL_DEG, tol_mm: float = SAME_POSE_TOL_MM) -> bool:
    """Same pose within the library's 'same start / end' tolerance (well below one motor tick)."""
    ha, hb = a.head.matrix(), b.head.matrix()
    return (math.degrees(rotation_distance(ha, hb)) <= tol_deg and translation_distance(ha, hb) * 1000.0 <= tol_mm
            and max(abs(math.degrees(x - y)) for x, y in zip(a.antennas, b.antennas)) <= tol_deg
            and abs(math.degrees(a.body_yaw - b.body_yaw)) <= tol_deg)


class MotionService:
    def __init__(self, backend: Any, library: Any = None, policy: SafetyPolicy = DEFAULT_POLICY, rate_hz: float = 50.0,
                 clock: Any = None, events: Any = None, record_present: bool = False,
                 transition_caps: dict[str, float] | None = None) -> None:
        self.backend = backend
        self.library = library
        self.policy = policy
        self.record_present = record_present
        self.transition_caps = dict(transition_caps or {})
        self.executor = Executor(backend, clock=clock, rate_hz=rate_hz, events=events)
        self._cond = threading.Condition()
        self._prep_queue: list[MotionHandle] = []
        self._run_queue: list[MotionHandle] = []
        self._current: MotionHandle | None = None
        self._last_state: RobotState | None = None
        self._listeners: list[EventListener] = []
        self._idle: tuple[Any, str, float] | None = None
        self._last_activity = time.monotonic()
        self._closing = False
        self._threads: list[threading.Thread] = []
        self.last_result: ExecutionResult | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "MotionService":
        if self._threads:
            return self
        self._closing = False
        for name, target in (("motion-prepare", self._prepare_loop), ("motion-worker", self._work_loop)):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def close(self, timeout: float = 5.0) -> None:
        """Cancel everything and join the threads (the head holds its last commanded pose)."""
        with self._cond:
            self._closing = True
            self._idle = None
            for h in self._prep_queue + self._run_queue:
                self._finish(h, HandleState.CANCELLED, "service closed")
            self._prep_queue.clear()
            self._run_queue.clear()
            if self._current is not None:
                self._current._cancel.set()
            self._cond.notify_all()
        for t in self._threads:
            t.join(timeout)
        self._threads.clear()

    def __enter__(self) -> "MotionService":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ requests (non-blocking)
    def play(self, motion: str | MotionSpec, variant: str = "expressive", intensity: float = 1.0, tempo: float = 1.0,
             priority: Priority = Priority.NORMAL, interrupt: bool = False) -> MotionHandle:
        spec: MotionSpec | None = None
        reason = None
        try:
            spec = motion if isinstance(motion, MotionSpec) else self._library().get(motion, variant)
            if intensity != 1.0 or tempo != 1.0:
                spec = spec.scaled(amplitude=intensity, tempo=tempo)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
        key = spec.key if spec is not None else f"{motion}:{variant}"
        h = MotionHandle(self, key, priority, interrupt, spec)
        if reason is not None:
            self._reject(h, reason, [])
            return h
        with self._cond:
            if self._closing or not self._threads:
                self._finish(h, HandleState.CANCELLED, "service not running")
                return h
            self._prep_queue.append(h)
            self._last_activity = time.monotonic()
            self._cond.notify_all()
        return h

    def stop(self, blend_to_rest: bool = True) -> MotionHandle | None:
        """Cancel the running and queued motions; optionally blend back to REST (a checked transition)."""
        with self._cond:
            for h in self._prep_queue + self._run_queue:
                self._finish(h, HandleState.CANCELLED, "stop")
            self._prep_queue.clear()
            self._run_queue.clear()
            if self._current is not None:
                self._current._cancel.set()
            rest = None
            if blend_to_rest and self._threads and not self._closing:
                rest = MotionHandle(self, "transition:rest", Priority.CRITICAL, False, None)
                self._run_queue.append(rest)
            self._last_activity = time.monotonic()
            self._cond.notify_all()
        return rest

    def set_idle(self, motion: str | MotionSpec, variant: str = "neutral", after_s: float = 10.0) -> None:
        if after_s <= 0:
            raise ValueError("after_s must be > 0")
        with self._cond:
            self._idle = (motion, variant, float(after_s))
            self._last_activity = time.monotonic()
            self._cond.notify_all()

    def clear_idle(self) -> None:
        with self._cond:
            self._idle = None

    def on_event(self, callback: EventListener) -> None:
        with self._cond:
            self._listeners.append(callback)

    def remove_listener(self, callback: EventListener) -> None:
        with self._cond:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def status(self) -> dict[str, Any]:
        with self._cond:
            cur = self._current
            return {"running": None if cur is None else cur.key, "queued": [h.key for h in self._run_queue],
                    "preparing": [h.key for h in self._prep_queue], "idle": None if self._idle is None else
                    {"motion": getattr(self._idle[0], "key", self._idle[0]), "variant": self._idle[1],
                     "after_s": self._idle[2]}}

    # ------------------------------------------------------------------ internals
    def _library(self) -> Any:
        if self.library is None:
            from reachy_motion.library.registry import default_library

            self.library = default_library()
        return self.library

    def _emit(self, kind: str, h: MotionHandle, **details: Any) -> None:
        event = {"kind": kind, "key": h.key, "handle": h, "id": h.id, "t": time.perf_counter(), **details}
        with self._cond:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - a broken listener must not stop the robot thread
                log.exception("motion event listener failed")

    def _finish(self, h: MotionHandle, state: HandleState, reason: str | None = None) -> None:
        if h.done:
            return
        h.state = state
        if reason:
            h.reason = reason
        kind = {HandleState.DONE: "finished", HandleState.CANCELLED: "cancelled", HandleState.PREEMPTED: "preempted",
                HandleState.REJECTED: "rejected", HandleState.FAILED: "failed"}[state]
        details: dict[str, Any] = {"reason": h.reason}
        if h.result is not None:
            details["result"] = {k: h.result.summary().get(k) for k in ("completed", "cancelled", "error", "n_commands")}
        try:
            self._emit(kind, h, **details)  # delivered before wait() returns
        finally:
            h._done.set()

    def _reject(self, h: MotionHandle, reason: str, codes: list[str]) -> None:
        h.reason = reason
        self._finish(h, HandleState.REJECTED, reason)

    def _cancel_handle(self, h: MotionHandle) -> None:
        with self._cond:
            if h in self._prep_queue:
                self._prep_queue.remove(h)
                self._finish(h, HandleState.CANCELLED, "cancelled before start")
            elif h in self._run_queue:
                self._run_queue.remove(h)
                self._finish(h, HandleState.CANCELLED, "cancelled before start")
            elif h is self._current:
                h._cancel.set()
            self._cond.notify_all()

    def _prepare_loop(self) -> None:
        while True:
            with self._cond:
                while not self._prep_queue and not self._closing:
                    self._cond.wait()
                if self._closing:
                    return
                h = self._prep_queue.pop(0)
            try:
                prepared = prepare(h.spec, self.policy)
            except Exception as exc:  # noqa: BLE001 - invalid spec etc.: reject, never crash the service
                self._reject(h, f"{type(exc).__name__}: {exc}", [])
                continue
            h.report = prepared.report
            if not prepared.runnable:
                codes = [v.code for v in prepared.report.violations]
                self._reject(h, "outside the initial policy: " + ", ".join(codes), codes)
                continue
            h.prepared = prepared
            with self._cond:
                if h.done or self._closing:  # cancelled while it was being prepared
                    continue
                self._run_queue.append(h)
                self._run_queue.sort(key=lambda x: -int(x.priority))  # stable: FIFO within a priority
                cur = self._current
                # a higher priority always preempts; interrupt=True preempts an equal or lower priority only
                if cur is not None and (h.priority > cur.priority or (h.interrupt and h.priority >= cur.priority)):
                    cur._preempted = True
                    cur._cancel.set()
                self._cond.notify_all()

    def _work_loop(self) -> None:
        while True:
            with self._cond:
                while not self._run_queue and not self._closing:
                    timeout = None
                    if self._idle is not None and not self._prep_queue:
                        timeout = self._idle[2] - (time.monotonic() - self._last_activity)
                        if timeout <= 0:
                            motion, variant, _ = self._idle
                            self._last_activity = time.monotonic()
                            self._cond.release()
                            try:
                                self.play(motion, variant, priority=Priority.IDLE)
                            finally:
                                self._cond.acquire()
                            continue
                    self._cond.wait(timeout)
                if self._closing:
                    return
                h = self._run_queue.pop(0)
                if h.done:
                    continue
                self._current = h
            try:
                self._run(h)
            except Exception as exc:  # noqa: BLE001 - report and keep the service alive
                log.exception("motion run failed")
                self._finish(h, HandleState.FAILED, f"{type(exc).__name__}: {exc}")
            finally:
                with self._cond:
                    self._current = None
                    self._last_activity = time.monotonic()
                    self._cond.notify_all()

    def _start_state(self) -> RobotState:
        if self._last_state is not None:
            return self._last_state
        p = self.backend.present()
        return RobotState(head=pose_from_matrix(p.head), antennas=tuple(p.antennas), body_yaw=float(p.body_yaw))

    def _transition(self, start: RobotState, end: RobotState) -> PreparedMotion:
        tl = plan_transition(start, end, transition_duration(start, end, **self.transition_caps),
                             Easing.MIN_JERK, self.executor.rate_hz)
        return prepare_timeline(tl, self.policy)

    def _run(self, h: MotionHandle) -> None:
        start = self._start_state()
        if h.spec is None:  # blend to REST
            if same_state(start, REST):
                h.state = HandleState.RUNNING
                self._finish(h, HandleState.DONE, "already at rest")
                return
            job = self._transition(start, REST)
        elif same_state(start, h.spec.start):
            job = h.prepared
        else:
            motion = h.prepared
            tr = self._transition(start, h.spec.start)
            # composite timeline: no spec (the executor binds a spec to its own timeline only), so a sound cue of
            # this motion is not logged on blended runs; recorded in the handle reason.
            job = prepare_timeline(concat(tr.timeline, motion.timeline, name=h.key), self.policy)
            h.reason = "started with a transition from the last commanded pose"
        if not job.runnable:
            codes = [v.code for v in job.report.violations]
            self._reject(h, "transition outside the initial policy: " + ", ".join(codes), codes)
            return
        if h._cancel.is_set():  # cancelled / preempted while waiting
            self._finish(h, HandleState.PREEMPTED if h._preempted else HandleState.CANCELLED)
            return
        h.state = HandleState.RUNNING
        self._emit("started", h, duration_s=float(job.timeline.t[-1]))
        result = self.executor.run(job, cancel=h._cancel, record_present=self.record_present)
        h.result = result
        self.last_result = result
        if result.last_sample is not None:
            head, antennas, body = result.last_sample
            self._last_state = RobotState(head=pose_from_matrix(head), antennas=tuple(antennas), body_yaw=float(body))
        if result.completed:
            self._finish(h, HandleState.DONE)
        elif result.error:
            self._finish(h, HandleState.FAILED, result.error)
        elif h._preempted:
            self._finish(h, HandleState.PREEMPTED, "preempted by a higher-priority or interrupting motion")
        else:
            self._finish(h, HandleState.CANCELLED, "cancelled")
