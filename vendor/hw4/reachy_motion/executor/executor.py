"""Executor: plays a checked ``PreparedMotion`` through a ``Backend`` and records exactly what it sent.

Strategies (docs/ARCHITECTURE.md section 6):

* ``run`` (StreamStrategy, the default) - one ``set_target(head, antennas, body_yaw)`` per sample of the
  planner's ``Timeline`` at ``rate_hz`` (50 Hz = the robot's control loop,
  reachy_mini/daemon/backend/robot/backend.py:71). The SDK sends it as ONE ``SetFullTargetCmd``
  (reachy_mini/reachy_mini.py:650-656), so the head, antennas and body of a sample travel together.
* ``goto`` (GotoStrategy, Phase 3 interpolation experiments) - one blocking ``goto_target``. The daemon
  interpolates from the MEASURED present pose (reachy_mini/daemon/backend/abstract.py:966-977). The SDK
  client first puts the task on the WebSocket and only then blocks for up to duration + 1 s
  (reachy_mini.py:721-722; reachy_mini/io/ws_client.py:278 send, 282-299 wait, 292 ``TimeoutError``).
  After that timeout the daemon move keeps running, so a goto whose call raised is kept as
  ``ExecutionResult.failed_command`` with ``end_state_known=False``.

Schedule - absolute, adapted from HW3 ``MotionController._run``
(../reachy_hw3/reachy_s2s/motion/controller.py:392-419, HW3 HEAD 0d49b6b): deadlines
``t0 + offset(k)`` with ``offset(k) = round(k * 1e9 / rate_hz)`` ns (no accumulated drift), reached with
``clock.sleep_until``. A tick that wakes one period or more late skips ahead to the sample matching the
current time (``TickRecord.skipped_before``) instead of bursting the missed samples, but the final sample
is never skipped, so a completed run always ends exactly on the planned end state. Every tick is timed
(``executor.timing``): the real period and jitter of the loop are measurable.

Safety contract:

* Nothing is sent unless the plan is runnable (policy pass AND report digest == timeline digest), the
  timeline is valid and was planned at this executor's rate, and the backend has the capability; a
  refusal raises ``ExecutionRefused`` before any backend call.
* What was checked is what is sent: ``Timeline`` is frozen but its numpy arrays stay writeable (the
  planner's too), so a caller still holding them could change samples after the check. The executor
  therefore copies every array into a private read-only snapshot, checks the digest and validity of THAT
  snapshot and streams only from it.
* ``prepared.spec`` counts only if the timeline was planned from it (``timeline.spec_digest ==
  spec.digest()``, the planner's label). Streaming, the spec only supplies the sound cue, so an unbound
  spec is ignored with a warning (the commands come from the checked timeline either way). A goto takes
  its easing and duration from the spec, so it refuses an unbound spec, a report made for another spec
  and a plan whose duration is not the spec's. ``ExecutionResult.spec_digest`` is always the timeline's
  label. ``Timeline.digest()`` does not cover that label (timeline.py:69-76): a report that carries a
  ``spec_digest`` binds it; a deliberately relabelled timeline is outside this check.
* Body yaw is always explicit: ``goto_target`` defaults it to 0.0 (reachy_mini.py:680) and a
  ``set_target`` without it leaves the body target unchanged (abstract.py:1691-1699).
* Stop = stop sending. After a cancel or a backend exception the daemon keeps writing its last joint
  target on every control tick (robot/backend.py:200-228; there is no idle reset), so the robot HOLDS the
  last commanded pose - the last successful command, or the failed one if it got through before the
  error (``end_state_known=False``). The executor does not recover on its own (no blend to rest): the
  caller decides.
* Phase 4 stop / hold (``run(..., stop=StopPolicy())``, docs/phase4/STARTUP_STOP.md; without ``stop`` nothing
  below happens and the rules above stand unchanged - MotionService preemption relies on the immediate stop):

  - S1 controlled stop, on ``cancel``: the run keeps streaming for ``controlled_stop_s`` but slows down ALONG the
    checked path - planned time u advances at the rate r(tau) = 1 - 3 tau^2 + 2 tau^3 (tau: 0 -> 1 over the stop),
    so the head ends at u0 + controlled_stop_s / 2 with zero commanded speed and then HOLDS there. The extra samples
    lie between neighbouring checked samples (Pose6 / antennas / body linearly interpolated between them); no
    command leaves the checked trajectory and no speed is higher than planned.
  - S2 immediate stop: stop sending at once (the daemon holds the last target) when a ``set_target`` raises, when a
    send takes longer than ``max_send_s`` (degraded link: the SDK's own check refuses to send only after ~1-2 s
    without any message, io/ws_client.py:120-160), or when no FRESH present state (a newer ``t_ns``) arrived for
    ``max_present_age_s`` (measurement lost or failing). ``present()`` is then read after every send.
  - S3 daemon error: when the backend has ``health()`` (``SdkBackend``: the daemon's 1 Hz status message;
    ``MockBackend``: an injected error), it is read every ``health_check_s``; a state of error / stopping /
    stopped or an error text stops sending at once. ``ExecutionResult.stop_detail`` then separates what is KNOWN
    (the daemon's own report, the last command sent, how old the last fresh measurement is) from what is
    UNCERTAIN (whether the motors still hold torque after the daemon closed its motor controller - compiled code,
    robot/backend.py:337-342, abstract.py:440-446 -, whether the last command reached them, whether the head holds
    or sags) and names the operator procedure. Detection waits for the next status message (up to about 1 s).
  - ``ExecutionResult.stop_kind`` / ``stop_reason`` / ``stop_commands`` / ``stop_detail`` record which stop happened.
* One run at a time per executor (one backend, one sending thread - backends/base.py). Do not mix ``goto``
  and ``run`` on a daemon: ``set_target`` is ignored while a daemon move runs (abstract.py:1661-1668).
* ``run`` / ``goto`` block the calling thread (the executor never starts threads); a GUI calls them from
  a worker thread, ``on_tick`` runs in that thread and must not block.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from reachy_motion.backends.base import Backend, BackendError, CommandRecord, PresentState, check_command
from reachy_motion.executor.clock import Clock, RealClock
from reachy_motion.executor.timing import DEFAULT_LATE_THRESHOLD_MS, TickLog, TickRecord
from reachy_motion.library.pose import Pose6, pose_from_matrix
from reachy_motion.library.spec import TRACKS, Easing, MotionSpec, SoundCue
from reachy_motion.safety.policy import SafetyReport
from reachy_motion.timeline import PreparedMotion, Timeline
from reachy_motion.util.event_logger import EventLogger

log = logging.getLogger(__name__)

#: Event kinds are ``executor.run_start``, ``executor.run_end``, ``executor.sound_cue``.
EVENT_PREFIX = "executor."
#: A goto's target and duration must equal the checked plan's end state / key duration (m, rad, s).
GOTO_MATCH_TOL = 1e-9
#: Added to ``ExecutionResult.warnings`` when ``goto_target`` raised (see the module docstring).
GOTO_IN_FLIGHT_WARNING = (
    "goto_target raised after the call started: the SDK sends the task before it waits "
    "(reachy_mini.py:721-722), so the daemon may have received it and may still be moving toward the "
    "target - end state unknown")

OnTick = Callable[[int, TickRecord], None]
Sample = tuple[np.ndarray, tuple[float, float], float]

#: Timeline arrays copied into the executor's snapshot (command arrays as float64, as ``Timeline.digest`` reads them).
_FLOAT_ARRAYS: tuple[str, ...] = ("t", "head", "antennas", "body_yaw")
_KEY_ARRAYS: tuple[str, ...] = ("head_key", "antenna_key", "body_key")


class ExecutionRefused(RuntimeError):
    """The executor refused to start: nothing was sent to the backend."""


@dataclass(frozen=True)
class StopPolicy:
    """Phase 4 stop / hold rules of a streamed run (module docstring). Times in s; 0 disables that rule."""

    controlled_stop_s: float = 0.3  # S1 on cancel: slow down along the checked path to a stop in this time
    max_send_s: float = 0.1  # S2: a set_target that takes longer means a degraded link
    max_present_age_s: float = 0.2  # S2: no fresh present state for this long (10 ticks at 50 Hz)
    health_check_s: float = 0.2  # S3: read backend.health() this often (when the backend has it)

    def __post_init__(self) -> None:
        for name in ("controlled_stop_s", "max_send_s", "max_present_age_s", "health_check_s"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (math.isfinite(v) and v >= 0.0):
                raise ValueError(f"{name} must be a finite number >= 0, got {v!r}")


def _interp_sample(tl: Timeline, u: float) -> Sample:
    """The checked trajectory at planned time ``u`` (s): linear between the two neighbouring samples (Pose6 with
    the yaw step unwrapped, antennas, body yaw); clamped to the first / last sample."""
    n = len(tl)
    i = int(math.floor(u * tl.rate_hz + 1e-9))
    if i >= n - 1:
        return tl.sample(n - 1)
    if i < 0:
        return tl.sample(0)
    f = min(max((u - float(tl.t[i])) * tl.rate_hz, 0.0), 1.0)
    p0 = pose_from_matrix(tl.head[i]).as_array()
    p1 = pose_from_matrix(tl.head[i + 1]).as_array()
    p1[5] = p0[5] + math.remainder(p1[5] - p0[5], 2.0 * math.pi)
    head = Pose6.from_array(p0 + f * (p1 - p0)).matrix()
    a0, a1 = tl.antennas[i], tl.antennas[i + 1]
    antennas = (float(a0[0] + f * (a1[0] - a0[0])), float(a0[1] + f * (a1[1] - a0[1])))
    body = float(tl.body_yaw[i] + f * (tl.body_yaw[i + 1] - tl.body_yaw[i]))
    return head, antennas, body


class SentCommand(NamedTuple):
    """A command as handed to the backend, paired with the timeline sample index it came from."""

    k: int
    record: CommandRecord


@dataclass
class ExecutionResult:
    """Outcome of one ``run`` / ``goto``; filled in place while it runs (``Executor.last_result``).

    ``commands`` hold what the executor handed to the backend (head arrays are read-only copies) for
    calls that returned without error. A call that raised is kept as ``failed_command`` (the run stops
    there); it may have reached the backend, so ``end_state_known`` is then False. ``completed`` is True
    only when the final sample was sent. ``data_kind`` is the backend's label ("mock" / "simulated" /
    "measured"); tick timings are PC-side measurements (synthetic with a ``VirtualClock``, see
    ``ticks.meta["clock"]``).
    """

    key: str  # "name:variant" of the executed timeline
    strategy: str  # "stream" | "goto"
    rate_hz: float
    backend: str
    data_kind: str
    n_samples: int = 0
    completed: bool = False
    cancelled: bool = False
    error: str | None = None
    commands: list[SentCommand] = field(default_factory=list)
    ticks: TickLog = field(default_factory=TickLog)
    started_ns: int = 0  # t0 of the schedule (stream) / start of the goto call
    ended_ns: int = 0
    # Last command whose backend call returned without error (None if none did). After a failed call
    # (``failed_command`` set, ``end_state_known`` False) the robot may hold this pose, hold the failed
    # command, or (goto) still be moving toward it.
    last_sample: Sample | None = None
    failed_command: SentCommand | None = None  # the call that raised (may or may not have reached the backend)
    end_state_known: bool = True  # False once a backend call raised: the robot's final target is uncertain
    present: list[PresentState] = field(default_factory=list)  # if record_present
    sound_cues: list[dict[str, Any]] = field(default_factory=list)  # logged cues (no audio in Phase 2)
    warnings: list[str] = field(default_factory=list)  # non-fatal problems (on_tick / present() failures, ...)
    timeline_digest: str = ""
    spec_digest: str = ""  # the executed timeline's spec_digest label ("" for ad-hoc timelines)
    late_threshold_ms: float = DEFAULT_LATE_THRESHOLD_MS
    stop_kind: str | None = None  # "S1" (controlled) / "S2" (immediate) / "S3" (daemon error), else None
    stop_reason: str | None = None
    stop_commands: int = 0  # S1 slow-down commands sent after the cancel
    stop_detail: dict[str, Any] | None = None  # S3: what is known and what is uncertain

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable outcome + tick statistics (``TickLog.summary``)."""
        failed = self.failed_command
        return {
            "key": self.key, "strategy": self.strategy, "backend": self.backend, "data_kind": self.data_kind,
            "n_samples": self.n_samples, "completed": self.completed, "cancelled": self.cancelled,
            "error": self.error, "n_commands": len(self.commands),
            "last_k": self.commands[-1].k if self.commands else None,
            "failed_k": failed.k if failed is not None else None,
            "failed_kind": failed.record.kind if failed is not None else None,
            "end_state_known": self.end_state_known,
            "elapsed_s": (self.ended_ns - self.started_ns) / 1e9 if self.ended_ns else None,
            "timeline_digest": self.timeline_digest, "spec_digest": self.spec_digest,
            "sound_cues": [dict(c) for c in self.sound_cues], "warnings": list(self.warnings),
            "n_present": len(self.present),
            "stop_kind": self.stop_kind, "stop_reason": self.stop_reason, "stop_commands": self.stop_commands,
            "stop_detail": self.stop_detail,
            "ticks": self.ticks.summary(1e9 / self.rate_hz, self.late_threshold_ms),
        }


def _key(tl: Timeline) -> str:
    return f"{tl.name}:{tl.variant}" if tl.variant else tl.name


def _effective_cue(spec: MotionSpec | None) -> SoundCue | None:
    """The spec's sound cue with tempo applied (``MotionSpec.effective``), or None."""
    if spec is None or spec.sound is None:
        return None
    return spec.effective().sound


def _snapshot(tl: Timeline) -> Timeline:
    """``tl`` with private, read-only copies of all its arrays: the executor checks and sends only these."""
    arrays: dict[str, np.ndarray] = {}
    for name in _FLOAT_ARRAYS + _KEY_ARRAYS:
        src = getattr(tl, name)
        arr = np.array(src, dtype=np.float64) if name in _FLOAT_ARRAYS else np.array(src)  # always a copy
        arr.flags.writeable = False
        arrays[name] = arr
    return dataclasses.replace(tl, **arrays)


def _spec_unbound(tl: Timeline, spec: MotionSpec | None) -> str | None:
    """None if ``spec`` is None or ``tl`` was planned from it (planner label), else the reason it is not."""
    if spec is None:
        return None
    if not tl.spec_digest:
        return f"the timeline is ad hoc (empty spec_digest), not planned from {spec.key}"
    if spec.digest() != tl.spec_digest:
        return f"the timeline was planned from another spec than {spec.key} (spec_digest mismatch)"
    return None


def _note(notes: dict[str, list[Any]], what: str, exc: BaseException) -> None:
    """Count a non-fatal failure; log only the first one of each kind."""
    entry = notes.get(what)
    if entry is None:
        notes[what] = [1, repr(exc)]
        log.warning("executor: %s raised %r (motion continues)", what, exc)
    else:
        entry[0] += 1


class Executor:
    """Plays ``PreparedMotion`` objects through one backend (see the module docstring for the rules)."""

    def __init__(self, backend: Backend, clock: Clock | None = None, rate_hz: float = 50.0,
                 late_threshold_ms: float = DEFAULT_LATE_THRESHOLD_MS, events: EventLogger | None = None) -> None:
        if not isinstance(backend, Backend):
            raise TypeError(f"{type(backend).__name__} does not implement the Backend protocol")
        if clock is not None and not isinstance(clock, Clock):
            raise TypeError(f"{type(clock).__name__} does not implement the Clock protocol")
        rate = float(rate_hz)
        if not (math.isfinite(rate) and rate > 0.0):
            raise ValueError("rate_hz must be > 0")
        late = float(late_threshold_ms)
        if not (math.isfinite(late) and late >= 0.0):
            raise ValueError("late_threshold_ms must be >= 0")
        self.backend = backend
        self.clock: Clock = clock if clock is not None else RealClock()
        self.rate_hz = rate
        self.period_ns = 1e9 / rate  # float; deadlines are rounded per sample (offset_ns)
        self.late_threshold_ms = late
        self.events = events
        self.last_result: ExecutionResult | None = None
        self._busy = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    def offset_ns(self, k: int) -> int:
        """Deadline of sample ``k`` relative to t0 (ns)."""
        return int(round(k * self.period_ns))

    def due_sample(self, elapsed_ns: int, k: int, n: int) -> int:
        """The sample matching ``elapsed_ns``: largest ``j`` in [k, n - 1] with ``offset_ns(j) <= elapsed_ns``.

        Returns ``k`` when no later sample is due yet; never more than ``n - 1`` (the final sample is sent).
        """
        j = int(elapsed_ns // self.period_ns) if elapsed_ns > 0 else 0
        j = min(max(j, k), n - 1)
        while j + 1 <= n - 1 and self.offset_ns(j + 1) <= elapsed_ns:
            j += 1
        while j > k and self.offset_ns(j) > elapsed_ns:
            j -= 1
        return j

    # ------------------------------------------------------------------ stream
    def run(self, prepared: PreparedMotion, cancel: threading.Event | None = None, on_tick: OnTick | None = None,
            record_present: bool = False, stop: StopPolicy | None = None) -> ExecutionResult:
        """Stream every sample of ``prepared.timeline`` (blocks for about its duration).

        The samples come from a read-only snapshot taken and checked before the first send (later changes to
        the caller's arrays are not sent). ``cancel`` is checked before every send (no send once it is set;
        latency <= one period); ``on_tick(k, record)`` runs after each successful send in this thread
        (exceptions are counted in ``warnings``, the motion continues); ``record_present`` reads
        ``backend.present()`` after each send. A backend exception stops the run (``error``,
        ``failed_command``, ``completed=False``) and the robot holds the last target that got through.
        With ``stop`` (Phase 4) a cancel slows down along the path (S1) and a failed / slow send or a lost
        measurement stops at once (S2); ``present()`` is then read after every send (module docstring).
        Raises ``ExecutionRefused`` (nothing sent) when the plan or the backend does not qualify.
        """
        if not self._busy.acquire(blocking=False):
            raise ExecutionRefused("executor busy: one run at a time (one backend, one sending thread)")
        try:
            checked = self._check_prepared(prepared)
            tl = checked.timeline
            if abs(tl.rate_hz - self.rate_hz) > 1e-9 * self.rate_hz:
                raise ExecutionRefused(f"{_key(tl)}: timeline planned at {tl.rate_hz:g} Hz, executor runs at "
                                       f"{self.rate_hz:g} Hz (re-plan at the executor rate)")
            caps = self.backend.capabilities
            if not caps.stream:
                raise ExecutionRefused(f"backend {self.backend.name!r} cannot stream set_target")
            if record_present and not caps.present:
                raise ExecutionRefused(f"backend {self.backend.name!r} has no present state to record")
            if stop is not None and not isinstance(stop, StopPolicy):
                raise ExecutionRefused(f"stop must be a StopPolicy, got {type(stop).__name__}")
            if stop is not None and stop.max_present_age_s > 0 and not caps.present:
                raise ExecutionRefused(f"backend {self.backend.name!r} has no present state to watch (StopPolicy)")
            return self._stream(checked, cancel, on_tick, record_present, stop)
        finally:
            self._busy.release()

    def _send_sample(self, result: ExecutionResult, sample: Sample, k: int, tick: int, deadline: int, woke: int,
                     skipped: int) -> TickRecord | None:
        """Send one sample; record it. None (with ``error`` / ``failed_command`` set) if the backend raised."""
        clock = self.clock
        head, antennas, body = sample
        sent_head = np.array(head, dtype=np.float64)  # private copy: the record cannot change later
        sent_head.flags.writeable = False
        s0 = clock.now_ns()
        cmd = SentCommand(k, CommandRecord(t_ns=s0, kind="set_target", head=sent_head, antennas=antennas,
                                           body_yaw=body))
        try:
            self.backend.set_target(sent_head.copy(), antennas, body)
        except BaseException as exc:  # any backend failure stops the run; robot holds the last target
            result.ticks.append(TickRecord(k, tick, deadline, woke, clock.now_ns() - s0, skipped, ok=False))
            result.failed_command = cmd
            result.end_state_known = False
            if not isinstance(exc, Exception):  # KeyboardInterrupt & co: record, then propagate
                result.error = f"interrupted during set_target at sample {k}: {exc!r}"
                raise
            result.error = f"set_target failed at sample {k}: {exc!r}"
            log.warning("executor: %s - run stopped, the robot holds the last target", result.error)
            return None
        rec = TickRecord(k, tick, deadline, woke, clock.now_ns() - s0, skipped)
        result.ticks.append(rec)
        result.commands.append(cmd)
        result.last_sample = (sent_head, antennas, body)
        return rec

    def _after_send(self, result: ExecutionResult, rec: TickRecord, record_present: bool, stop: StopPolicy | None,
                    fresh: list[int], notes: dict[str, list[Any]]) -> tuple[str, str] | None:
        """Read / watch the present state (and the daemon's health) after a send.

        Returns (stop kind "S2" / "S3", reason) if the run must stop now, else None. ``fresh`` = [newest present t_ns
        seen, clock time it was first seen, clock time of the last health check] (mutated)."""
        reason = self._watch_present(result, rec, record_present, stop, fresh, notes)
        if reason is not None:
            if "measurement lost" in reason:
                self._mark_unconfirmed(result, fresh[1])
            return "S2", reason
        reason = self._check_health(result, stop, fresh, notes)
        return None if reason is None else ("S3", reason)

    @staticmethod
    def _mark_unconfirmed(result: ExecutionResult, last_fresh_ns: int) -> None:
        """S2 on a lost measurement: the link may be dead, and a dead SDK link drops sends WITHOUT an error
        (io/ws_client.py:135-160; seen in MuJoCo, docs/phase4/STARTUP_STOP.md §9). Commands sent after the last fresh
        measurement are therefore unconfirmed and the held target is one of them or the last confirmed one."""
        confirmed = [c for c in result.commands if c.record.t_ns <= last_fresh_ns]
        unconfirmed = len(result.commands) - len(confirmed)
        result.end_state_known = False
        result.stop_detail = {
            "known": {"commands_sent": len(result.commands), "unconfirmed_commands": unconfirmed,
                      "last_confirmed_k": confirmed[-1].k if confirmed else None,
                      "last_sent_k": result.commands[-1].k if result.commands else None,
                      "last_fresh_measurement_seen_ns": last_fresh_ns},
            "uncertain": ["whether the commands sent after the last fresh measurement reached the daemon: the robot "
                          "holds the last confirmed target or one of the unconfirmed ones"]}

    def _check_health(self, result: ExecutionResult, stop: StopPolicy | None, fresh: list[int],
                      notes: dict[str, list[Any]]) -> str | None:
        """S3: the backend's ``health()`` (if any) every ``stop.health_check_s``; fills ``stop_detail`` on a hit."""
        health = getattr(self.backend, "health", None)
        if stop is None or stop.health_check_s <= 0 or not callable(health):
            return None
        now = self.clock.now_ns()
        if now - fresh[2] < stop.health_check_s * 1e9:
            return None
        fresh[2] = now
        try:
            h = health()
        except Exception as exc:  # noqa: BLE001 - an unreadable status is not itself a daemon error (S2 covers loss)
            _note(notes, "health()", exc)
            return None
        if not isinstance(h, dict):
            return None
        state, err = str(h.get("state") or "").lower(), h.get("error")
        if state not in ("error", "stopping", "stopped") and not err:
            return None
        last = result.commands[-1] if result.commands else None
        result.stop_detail = {
            "known": {"daemon_state": h.get("state"), "daemon_error": err, "motor_mode_reported": h.get("motor_mode"),
                      "status_age_s": h.get("status_age_s"), "commands_sent": len(result.commands),
                      "last_command_k": None if last is None else last.k,
                      "last_command_t_ns": None if last is None else last.record.t_ns,
                      "last_fresh_measurement_age_s": (now - fresh[1]) / 1e9},
            "uncertain": [
                "whether the motors still hold torque: on a backend error the daemon closes its motor controller "
                "(daemon/backend/robot/backend.py:337-342, abstract.py:440-446) and what the compiled controller "
                "does to the torque is not in the source",
                "whether the last command reached the motors before the error",
                "whether the head holds, sags or is limp (a motor's own protection turns that motor off, "
                "docs/phase1/SOURCE_REFERENCES.md L19)",
                "the head's state after the error: the daemon no longer reads the motors"],
            "operator": "send nothing; keep a hand near the head; switch the robot off; after the restart run R0 "
                        "(robot-status) again before any command"}
        return f"daemon reports state={h.get('state')!r}, error={err!r}"

    def _watch_present(self, result: ExecutionResult, rec: TickRecord, record_present: bool, stop: StopPolicy | None,
                       fresh: list[int], notes: dict[str, list[Any]]) -> str | None:
        """The S2 checks after a send: a slow send, a lost measurement (reads ``present()``)."""
        reason = None
        if stop is not None and stop.max_send_s > 0 and rec.send_ns > stop.max_send_s * 1e9:
            reason = (f"set_target took {rec.send_ns / 1e6:.0f} ms (> {stop.max_send_s * 1e3:g} ms): the link is "
                      f"degraded")
        watch = stop is not None and stop.max_present_age_s > 0
        if record_present or watch:
            try:
                p = self.backend.present()
                if record_present:
                    result.present.append(p)
                if p.t_ns > fresh[0]:
                    fresh[0], fresh[1] = int(p.t_ns), self.clock.now_ns()
            except Exception as exc:  # noqa: BLE001 - recorded; with a StopPolicy the age check below decides
                _note(notes, "present()", exc)
        if reason is None and watch:
            age = self.clock.now_ns() - fresh[1]
            if age > stop.max_present_age_s * 1e9:  # type: ignore[union-attr]
                reason = (f"no fresh present state for {age / 1e6:.0f} ms (> {stop.max_present_age_s * 1e3:g} ms): "  # type: ignore[union-attr]
                          f"measurement lost")
        return reason

    def _stop_now(self, result: ExecutionResult, reason: str, kind: str = "S2") -> None:
        result.stop_kind, result.stop_reason = kind, reason
        if result.error is None:
            result.error = f"stopped ({kind}): {reason}"
        if kind == "S3":
            log.error("executor: S3 stop - %s; nothing more is sent; torque state UNKNOWN (see stop_detail)", reason)
        else:
            log.warning("executor: %s stop - %s; nothing more is sent, the robot holds the last target", kind, reason)

    def _controlled_stop(self, result: ExecutionResult, tl: Timeline, t0: int, stop: StopPolicy, record_present: bool,
                         on_tick: OnTick | None, fresh: list[int], notes: dict[str, list[Any]]) -> None:
        """S1: slow down along the checked path to a stop in ``stop.controlled_stop_s``, then hold (module doc)."""
        if not result.commands:
            result.stop_kind, result.stop_reason = "S1", "cancelled before the first send: nothing to slow down"
            return
        result.stop_kind = "S1"
        result.stop_reason = f"cancel: slowing down along the checked path over {stop.controlled_stop_s:g} s"
        k_last = result.commands[-1].k
        u0, u_max = float(tl.t[k_last]), float(tl.t[-1])
        span = stop.controlled_stop_s
        steps = max(1, int(math.ceil(span * self.rate_hz - 1e-9)))
        tick = len(result.ticks)
        for j in range(1, steps + 1):
            tau = min(1.0, j / (span * self.rate_hz))
            u = min(u_max, u0 + span * (tau - tau ** 3 + 0.5 * tau ** 4))  # integral of r = 1 - 3 tau^2 + 2 tau^3
            deadline = t0 + self.offset_ns(k_last + j)
            self.clock.sleep_until(deadline)
            woke = self.clock.now_ns()
            k_idx = min(int(math.floor(u * self.rate_hz + 1e-9)), len(tl) - 1)
            rec = self._send_sample(result, _interp_sample(tl, u), k_idx, tick, deadline, woke, 0)
            if rec is None:
                self._stop_now(result, f"set_target raised during the S1 slow-down: {result.error}")
                return
            result.stop_commands += 1
            tick += 1
            hit = self._after_send(result, rec, record_present, stop, fresh, notes)
            if on_tick is not None:
                try:
                    on_tick(k_idx, rec)
                except Exception as exc:  # noqa: BLE001
                    _note(notes, "on_tick", exc)
            if hit is not None:
                self._stop_now(result, hit[1], hit[0])
                return

    def _stream(self, checked: PreparedMotion, cancel: threading.Event | None, on_tick: OnTick | None,
                record_present: bool, stop: StopPolicy | None = None) -> ExecutionResult:
        tl = checked.timeline  # the private read-only snapshot (_check_prepared)
        n = len(tl)
        clock, backend = self.clock, self.backend
        result = self._new_result(checked, "stream")
        unbound = _spec_unbound(tl, checked.spec)
        if unbound is not None:
            msg = f"prepared.spec ignored: {unbound}; its sound cue is not logged"
            result.warnings.append(msg)
            log.warning("executor: %s", msg)
        cue = _effective_cue(None if unbound is not None else checked.spec)
        notes: dict[str, list[Any]] = {}
        self._log("run_start", **self._start_fields(result, tl, record_present=record_present,
                                                     stop=None if stop is None else dataclasses.asdict(stop)))
        k = tick = 0
        try:
            t0 = clock.now_ns()
            result.started_ns = t0
            fresh = [-(2 ** 63), t0, t0]  # newest present t_ns, when it was seen, last health check (start = t0)
            while k < n:
                if cancel is not None and cancel.is_set():
                    result.cancelled = True
                    break
                deadline = t0 + self.offset_ns(k)
                clock.sleep_until(deadline)
                woke = clock.now_ns()
                due = self.due_sample(woke - t0, k, n)  # woke >= one period late -> skip ahead (not past the end)
                skipped, k = due - k, due
                if cancel is not None and cancel.is_set():  # set while sleeping: do not send this sample
                    result.cancelled = True
                    break
                rec = self._send_sample(result, tl.sample(k), k, tick, deadline, woke, skipped)
                if rec is None:
                    if stop is not None:
                        result.stop_kind, result.stop_reason = "S2", f"set_target raised: {result.error}"
                    break
                if cue is not None and not result.sound_cues and float(tl.t[k]) + 1e-9 >= cue.at_s:
                    self._log_cue(result, cue, k, float(tl.t[k]), woke - t0)
                hit = self._after_send(result, rec, record_present, stop, fresh, notes)
                if on_tick is not None:
                    try:
                        on_tick(k, rec)
                    except Exception as exc:  # noqa: BLE001 - a broken GUI callback must not stop the motion
                        _note(notes, "on_tick", exc)
                if hit is not None:
                    self._stop_now(result, hit[1], hit[0])
                    break
                k += 1
                tick += 1
            else:
                result.completed = True
            if result.cancelled and stop is not None and stop.controlled_stop_s > 0:
                self._controlled_stop(result, tl, t0, stop, record_present, on_tick, fresh, notes)
            elif result.cancelled and stop is not None:
                result.stop_kind, result.stop_reason = "S2", "cancel with controlled_stop_s = 0: immediate stop"
        except BaseException as exc:  # KeyboardInterrupt, clock failure, bug: record it, then propagate
            if result.error is None:
                result.error = f"interrupted: {exc!r}"
            raise
        finally:
            self._finish(result, notes)
        return result

    # ------------------------------------------------------------------ goto
    def goto(self, head: np.ndarray, antennas: tuple[float, float] | list[float], body_yaw: float, duration: float,
             method: Easing | str, *, prepared: PreparedMotion, record_present: bool = False) -> ExecutionResult:
        """One blocking ``backend.goto_target`` (GotoStrategy), bound to a checked one-key plan.

        ``prepared`` must be runnable and describe exactly this goto: its timeline was planned from
        ``prepared.spec`` (``timeline.spec_digest == spec.digest()``, a non-empty report ``spec_digest`` equal
        too, ``timeline.duration == spec.duration``), the spec has at most one key per track, every key uses
        ``method`` and lasts ``duration``, and the plan ends at (``head``, ``antennas``, ``body_yaw``) within
        ``GOTO_MATCH_TOL``. A duration / easing sweep therefore needs one checked plan per spec. The plan
        starts at ``spec.start`` while the daemon starts from the measured present pose
        (abstract.py:966-977): with ``record_present`` the present state is recorded before and after the
        call so the difference is visible. ``method`` is an ``Easing`` / its value, the same strings as the
        SDK's ``InterpolationTechnique`` (reachy_mini/utils/interpolation.py:108-114). No cancel: the call
        blocks until the backend returns. If it raises, the command is kept as ``failed_command`` and
        ``end_state_known`` is False (the daemon may still run the move, see the module docstring).
        """
        if not self._busy.acquire(blocking=False):
            raise ExecutionRefused("executor busy: one run at a time (one backend, one sending thread)")
        try:
            checked, h, a, b, d, m = self._check_goto(head, antennas, body_yaw, duration, method, prepared,
                                                      record_present)
            return self._goto(checked, h, a, b, d, m, record_present)
        finally:
            self._busy.release()

    def _check_goto(self, head: np.ndarray, antennas: Any, body_yaw: float, duration: float, method: Any,
                    prepared: PreparedMotion, record_present: bool
                    ) -> tuple[PreparedMotion, np.ndarray, tuple[float, float], float, float, str]:
        try:
            m = Easing(method).value
        except ValueError:
            raise ExecutionRefused(f"unknown interpolation method {method!r} "
                                   f"(one of {[e.value for e in Easing]})") from None
        try:
            h, a, b = check_command(head, antennas, body_yaw)
        except BackendError as exc:
            raise ExecutionRefused(f"goto target rejected: {exc}") from None
        d = float(duration)
        if not (math.isfinite(d) and d > 0.0):  # the SDK refuses duration <= 0 (reachy_mini.py:700-703)
            raise ExecutionRefused("goto duration must be > 0")
        checked = self._check_prepared(prepared)
        tl, spec, report = checked.timeline, checked.spec, checked.report
        key = _key(tl)
        if spec is None:
            raise ExecutionRefused(f"{key}: goto needs prepared.spec to verify the easing and the duration")
        unbound = _spec_unbound(tl, spec)
        if unbound is not None:
            raise ExecutionRefused(f"{key}: goto refused: {unbound}; the checked plan must be plan(spec) of "
                                   f"exactly this spec (re-plan and re-check every swept spec)")
        if report.spec_digest and report.spec_digest != tl.spec_digest:
            raise ExecutionRefused(f"{key}: the safety report was made for another spec (report spec_digest "
                                   f"mismatch)")
        eff = spec.effective()
        n_keys = 0
        for track in TRACKS:
            keys = eff.track(track)
            if len(keys) > 1:
                raise ExecutionRefused(f"{key}: a goto is one key per track ({track} has {len(keys)})")
            for kf in keys:
                n_keys += 1
                try:
                    key_method = Easing(kf.easing).value
                except ValueError:
                    raise ExecutionRefused(f"{key}: {track} key has an unknown easing {kf.easing!r}") from None
                if key_method != m:
                    raise ExecutionRefused(f"{key}: plan was checked with {key_method!r} on {track}, goto asks {m!r}")
                if abs(kf.duration - d) > GOTO_MATCH_TOL:
                    raise ExecutionRefused(f"{key}: plan {track} key lasts {kf.duration:g} s, goto asks {d:g} s")
        if n_keys == 0:
            raise ExecutionRefused(f"{key}: plan has no key")
        if abs(float(tl.duration) - float(spec.duration)) > GOTO_MATCH_TOL:  # spec.duration is effective
            raise ExecutionRefused(f"{key}: checked plan lasts {float(tl.duration):g} s, the spec "
                                   f"{float(spec.duration):g} s (not planned from this spec)")
        end_head, end_ant, end_body = tl.sample(len(tl) - 1)
        if not (np.allclose(end_head, h, rtol=0.0, atol=GOTO_MATCH_TOL)
                and np.allclose(end_ant, a, rtol=0.0, atol=GOTO_MATCH_TOL)
                and abs(end_body - b) <= GOTO_MATCH_TOL):
            raise ExecutionRefused(f"{key}: goto target differs from the end state of the checked plan")
        caps = self.backend.capabilities
        if not caps.goto:
            raise ExecutionRefused(f"backend {self.backend.name!r} has no goto_target")
        if record_present and not caps.present:
            raise ExecutionRefused(f"backend {self.backend.name!r} has no present state to record")
        return checked, h, a, b, d, m

    def _goto(self, checked: PreparedMotion, head: np.ndarray, antennas: tuple[float, float], body: float,
              duration: float, method: str, record_present: bool) -> ExecutionResult:
        tl = checked.timeline
        k_end = len(tl) - 1
        clock, backend = self.clock, self.backend
        result = self._new_result(checked, "goto")
        notes: dict[str, list[Any]] = {}
        if checked.spec is not None and checked.spec.sound is not None:
            result.warnings.append("sound cue not logged: a goto is one blocking call")
        self._log("run_start", **self._start_fields(result, tl, record_present=record_present, method=method,
                                                    goto_duration_s=duration))
        try:
            if record_present:
                try:
                    result.present.append(backend.present())
                except Exception as exc:  # noqa: BLE001
                    _note(notes, "present()", exc)
            sent_head = np.array(head, dtype=np.float64)
            sent_head.flags.writeable = False
            s0 = clock.now_ns()
            result.started_ns = s0
            cmd = SentCommand(k_end, CommandRecord(t_ns=s0, kind="goto_target", head=sent_head, antennas=antennas,
                                                   body_yaw=body, duration=duration, method=method))
            try:
                backend.goto_target(sent_head.copy(), antennas, body, duration, method)
            except BaseException as exc:  # the task may already be on the wire: keep the command
                result.ticks.append(TickRecord(k=k_end, tick=0, planned_ns=s0, actual_ns=s0,
                                               send_ns=clock.now_ns() - s0, ok=False))
                result.failed_command = cmd
                result.end_state_known = False
                result.warnings.append(GOTO_IN_FLIGHT_WARNING)
                if not isinstance(exc, Exception):  # KeyboardInterrupt & co: record, then propagate
                    result.error = f"interrupted during goto_target: {exc!r}"
                    raise
                result.error = f"goto_target failed: {exc!r}"
                log.warning("executor: %s - the daemon move may still be running", result.error)
            else:
                result.ticks.append(TickRecord(k=k_end, tick=0, planned_ns=s0, actual_ns=s0,
                                               send_ns=clock.now_ns() - s0))
                result.commands.append(cmd)
                result.last_sample = (sent_head, antennas, body)
                result.completed = True
                if record_present:
                    try:
                        result.present.append(backend.present())
                    except Exception as exc:  # noqa: BLE001
                        _note(notes, "present()", exc)
        except BaseException as exc:
            if result.error is None:
                result.error = f"interrupted: {exc!r}"
            raise
        finally:
            self._finish(result, notes)
        return result

    # ------------------------------------------------------------------ helpers
    def _check_prepared(self, prepared: PreparedMotion) -> PreparedMotion:
        """Check ``prepared``; return it with a private read-only snapshot of its timeline (send only from it)."""
        if not isinstance(prepared, PreparedMotion):
            raise ExecutionRefused("executors accept only a PreparedMotion (timeline + its safety report)")
        tl, report, spec = prepared.timeline, prepared.report, prepared.spec
        if not isinstance(tl, Timeline):
            raise ExecutionRefused(f"prepared.timeline is a {type(tl).__name__}, not a Timeline")
        key = _key(tl)
        if not isinstance(report, SafetyReport):
            raise ExecutionRefused(f"{key}: prepared.report is a {type(report).__name__}, not a SafetyReport")
        if spec is not None and not isinstance(spec, MotionSpec):
            raise ExecutionRefused(f"{key}: prepared.spec is a {type(spec).__name__}, not a MotionSpec")
        if not report.passes_policy:
            codes = ", ".join(sorted({v.code for v in report.violations})) or "no violation listed"
            raise ExecutionRefused(f"{key}: plan is outside the safety policy ({codes})")
        try:
            snap = _snapshot(tl)
            digest = snap.digest()
            problems = snap.validate()
        except Exception as exc:  # noqa: BLE001 - malformed arrays (non-numeric, wrong dimension)
            raise ExecutionRefused(f"{key}: invalid timeline ({exc!r})") from None
        if report.timeline_digest != digest:
            raise ExecutionRefused(f"{key}: the safety report belongs to other commands (timeline digest mismatch)")
        checked = PreparedMotion(timeline=snap, report=report, spec=spec)
        if not checked.runnable:  # the contract's own rule, on the snapshot (kept in case it tightens)
            raise ExecutionRefused(f"{key}: plan is not runnable")
        if problems:
            raise ExecutionRefused(f"{key}: invalid timeline ({'; '.join(problems)})")
        return checked

    def _new_result(self, checked: PreparedMotion, strategy: str) -> ExecutionResult:
        tl = checked.timeline
        result = ExecutionResult(
            key=_key(tl), strategy=strategy, rate_hz=self.rate_hz, backend=str(self.backend.name),
            data_kind=str(self.backend.data_kind), n_samples=len(tl), timeline_digest=checked.report.timeline_digest,
            spec_digest=tl.spec_digest, late_threshold_ms=self.late_threshold_ms)
        result.ticks.meta.update(key=result.key, strategy=strategy, rate_hz=self.rate_hz, period_ns=self.period_ns,
                                 late_threshold_ms=self.late_threshold_ms, clock=type(self.clock).__name__,
                                 backend=result.backend, backend_data_kind=result.data_kind)
        self.last_result = result
        return result

    def _start_fields(self, result: ExecutionResult, tl: Timeline, **extra: Any) -> dict[str, Any]:
        return {"key": result.key, "strategy": result.strategy, "n_samples": len(tl), "rate_hz": self.rate_hz,
                "duration_s": float(tl.duration), "backend": result.backend, "data_kind": result.data_kind,
                "clock": type(self.clock).__name__, "timeline_digest": result.timeline_digest,
                "spec_digest": result.spec_digest, **extra}

    def _log_cue(self, result: ExecutionResult, cue: SoundCue, k: int, t_s: float, elapsed_ns: int) -> None:
        entry = {"file": cue.file, "at_s": float(cue.at_s), "k": k, "t_s": t_s, "elapsed_s": elapsed_ns / 1e9,
                 "played": False}  # Phase 2: metadata only, no audio
        result.sound_cues.append(entry)
        self._log("sound_cue", key=result.key, **entry)

    def _finish(self, result: ExecutionResult, notes: dict[str, list[Any]]) -> None:
        result.ended_ns = self.clock.now_ns()
        for what, (count, first) in notes.items():
            result.warnings.append(f"{what} raised {count} time(s); first: {first}")
        result.ticks.meta.update(completed=result.completed, cancelled=result.cancelled, error=result.error,
                                 end_state_known=result.end_state_known)
        self._log("run_end", **result.summary())

    def _log(self, kind: str, **fields: Any) -> None:
        if self.events is None:
            return
        try:
            self.events.log(EVENT_PREFIX + kind, **fields)
        except Exception:  # noqa: BLE001 - logging must never break motion (as in HW3 controller._log)
            log.exception("executor event logging failed")


# ---------------------------------------------------------------------- thread priority (Windows)
#: Windows THREAD_PRIORITY_* values, as in HW3 (../reachy_hw3/reachy_s2s/motion/controller.py:96).
THREAD_PRIORITIES: dict[str, int] = {"above_normal": 1, "normal": 0, "below_normal": -1}
_THREAD_PRIORITY_ERROR_RETURN = 0x7FFFFFFF


@functools.lru_cache(maxsize=1)
def _kernel32() -> Any:
    """A private kernel32 handle with typed prototypes (the shared ``ctypes.windll`` one is left untouched)."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentThread.argtypes = []
    k32.GetCurrentThread.restype = wintypes.HANDLE
    k32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
    k32.SetThreadPriority.restype = wintypes.BOOL
    k32.GetThreadPriority.argtypes = [wintypes.HANDLE]
    k32.GetThreadPriority.restype = ctypes.c_int
    return k32


def set_current_thread_priority(name: str) -> bool:
    """Priority of the calling thread (Windows; elsewhere unchanged). True only if it was changed.

    Copied from HW3 ``_set_thread_priority`` (../reachy_hw3/reachy_s2s/motion/controller.py:100-110,
    HW3 HEAD 0d49b6b) with two changes: unknown names raise ``ValueError`` (HW3 validated them in its
    config), and the Win32 calls use typed prototypes. The HW3 version passes ``GetCurrentThread()``'s
    pseudo-handle through ctypes' default 32-bit int, which on 64-bit Python is not the pseudo-handle any
    more: ``SetThreadPriority`` failed and the function returned False (checked on this PC, 2026-11-10,
    Python 3.12.14 x64). As in HW3, "normal" leaves the thread as it is and returns False. Call it at the
    start of the executor's worker thread; it lasts for that thread's lifetime.
    """
    if name not in THREAD_PRIORITIES:
        raise ValueError(f"thread priority must be one of {tuple(THREAD_PRIORITIES)}, got {name!r}")
    if sys.platform != "win32" or name == "normal":
        return False
    try:
        k32 = _kernel32()
        return bool(k32.SetThreadPriority(k32.GetCurrentThread(), THREAD_PRIORITIES[name]))
    except Exception:  # noqa: BLE001 - keep the default priority
        return False


def current_thread_priority() -> int | None:
    """Windows priority value of the calling thread (e.g. 0 = normal), None elsewhere or on failure."""
    if sys.platform != "win32":
        return None
    try:
        k32 = _kernel32()
        value = int(k32.GetThreadPriority(k32.GetCurrentThread()))
    except Exception:  # noqa: BLE001
        return None
    return None if value == _THREAD_PRIORITY_ERROR_RETURN else value


__all__ = ["EVENT_PREFIX", "GOTO_IN_FLIGHT_WARNING", "GOTO_MATCH_TOL", "ExecutionRefused", "ExecutionResult",
           "Executor", "SentCommand", "THREAD_PRIORITIES", "current_thread_priority", "set_current_thread_priority"]
