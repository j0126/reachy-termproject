"""Mock backend: a deterministic model of the daemon's command pipeline - no daemon, no robot, no network.

Everything it returns is labelled ``data_kind="mock"``: a synthetic model derived from the SDK source, never a
measurement and not a prediction of the robot (CLAUDE.md rule 6; Phase 1 SDK_ANALYSIS 5.2, 5.5). It lets the
executor, the service and the Motion Editor GUI run and be tested without any daemon. SDK references are
reachy-mini 1.11.0, paths relative to ``site-packages/reachy_mini/`` (``abstract.py`` = daemon/backend/abstract.py).

Present-state model
-------------------
A virtual control loop runs at ``control_rate_hz`` (50 Hz, like every SDK backend: robot/backend.py:71; MuJoCo
500 Hz physics with ``decimation = 10``, mujoco/backend.py:90-91). Its ticks are at ``t_k = connect_time + k * period``
on the injected clock (``period`` rounded to whole nanoseconds). A tick at time t sees every command received at or
before t.

1. Target. ``set_target`` replaces the target from its receipt time on (latest value wins, no queue - like
   ``SetFullTargetCmd`` in the daemon, daemon/backend/abstract.py:1691-1699). ``goto_target`` replays the SDK's own
   ``GotoMove`` (motion/goto.py:15-74) built the way ``Backend.goto_target`` builds it: it starts from the PRESENT
   head pose, body yaw and antennas, not from the previous target (abstract.py:944-977), and is evaluated on the
   ``play_move`` grid ``t_j = j / goto_eval_rate_hz`` while ``t_j < D`` (play_frequency = 100 Hz,
   abstract.py:826-829, 904-925), each sample held until the next one. Deliberate deviation: from ``t = D`` on the
   target is exactly the goto target (the daemon never evaluates t = D, so its last command falls short - Phase 1
   SDK_ANALYSIS 3.5). While a goto runs, other commands are ignored as in the daemon (``set_target``:
   abstract.py:1661-1699; a second goto: abstract.py:844-846); they are only logged and counted. A goto the SDK
   cannot evaluate - a head rotation block with det <= 0, which ``Rotation.from_matrix`` rejects
   (utils/rotation.py:82-88, called from utils/interpolation.py:71-72) - is refused at call time with
   ``BackendError``: ``play_move`` evaluates the move at once (abstract.py:904-913), the error comes back as a
   failed task (io/ws_server.py:179-196, io/ws_client.py:293-296), nothing is written and the daemon stays usable
   (abstract.py:941-942). The mock likewise writes nothing and does not block.
2. Written goal (the tick at which a target reaches the motors):

   * ``structure="robot"`` - real robot and mockup (robot/backend.py:200-262: write the current joint goals, read,
     then IK of the latest target for the NEXT tick; Phase 1 SDK_ANALYSIS 5.2): at tick k the head and body goal is
     the target known at tick k-1 and the antenna goal the target known at tick k, so the antennas lead the head and
     the body by one tick (20 ms at 50 Hz).
   * ``structure="mujoco"`` (mujoco/backend.py:227-260: read, IK, ``data.ctrl`` in the same step): every channel is
     written at tick k with the target known at tick k.
3. Present position. Every component of [x, y, z, roll, pitch, yaw, antenna_right, antenna_left, body_yaw] (the head
   as ``Pose6`` through ``pose_from_matrix`` / ``matrix``) follows its written goal with a first-order lag of time
   constant ``lag_tau_s``. The goal is held between ticks, so the state is the exact continuous response and, from
   tick to tick, the exact discretisation ``x(t_k+1) = x(t_k) + alpha * (g_k - x(t_k))``,
   ``alpha = 1 - exp(-dt / tau)``; ``lag_tau_s = 0`` follows the written goal instantly. Yaw angles are unwrapped
   against the previous target, so the lag never goes the long way round +-180 deg (``current_target()`` reports
   the head yaw wrapped back to [-pi, pi]).
4. ``present()`` advances the model lazily to ``clock.now_ns()`` (stretches without target changes are jumped in
   closed form) and returns that instant's state (``t_ns`` = that instant). ``joints`` = IK of the present head pose
   with the present body yaw (``analysis.goto_replay.ik_joints``: ``AnalyticalKinematics`` with automatic body yaw,
   the daemon default, kinematics/analytical_kinematics.py:33, 65-97). ``body_yaw`` is the modelled body channel;
   an automatic body-yaw shift (relative yaw > 65 deg) only shows in ``joints[0]``.

Not modelled - do not read mock output as robot behaviour: motor dynamics beyond the lag (P gains, saturation,
friction, position quantisation), joint limits, network / WebSocket latency, jitter of the daemon loops, the daemon's
IK-failure hold (an infeasible head target is followed here and ``joints`` comes back NaN, whereas the daemon keeps
the previous joint goal: abstract.py:586-588, robot/backend.py:253-261), head tracking and speech-wobble offsets.
A ``set_target`` head whose 3x3 block is not a rotation is followed here as its nearest rotation
(``pose_from_matrix``); the daemon hands the raw matrix to the rust IK (kinematics/analytical_kinematics.py:76-97;
outcome not checked).

The ``start`` state and ``reset`` states must be finite (``ValueError``), like every command (``check_command``).

Fault injection (public attributes, may be changed at any time)
---------------------------------------------------------------
``fail_send``: the send raises ``BackendError`` and the command never reaches the model. ``send_delay_s``: time spent
inside every send before the command reaches the model (``VirtualClock.advance``, otherwise ``clock.sleep_until``).
``block_send``: a ``threading.Event`` the send waits for; ``close()`` releases a waiting sender with
``BackendError``. Order inside a send: block, delay, fail.
Measurement faults (Phase 4 stop / hold tests): ``fail_present``: ``present()`` raises ``BackendError`` (a failed
read). ``freeze_present``: ``present()`` keeps returning the last state it returned, with its old ``t_ns`` (no new
state push arrives, like a stalled link); the model itself keeps running. ``daemon_error``: a text makes
``health()`` report the daemon in its error state (as RobotBackend after a motor read failure, Phase 1 L20); the
model keeps holding its last target - on the robot that is exactly what is NOT known (executor S3).

Logs and counters
-----------------
``commands``: every validated command, recorded when its call starts (``t_ns`` = call start) - including the ones
whose send failed or that were ignored during a goto. ``motor_command_count``: commands that reached the model and
changed its target (the number to check for "nothing was sent to the motors"). ``failed_send_count``: logged
commands whose call raised instead of reaching the model (``fail_send``, a goto the SDK cannot evaluate).
``ignored_command_count``: commands ignored while a goto ran. ``sound_cues``: ``(t_ns, file)`` from
``log_sound_cue`` (metadata only, no audio).
Both logs keep the newest ``max_records`` entries (default 200 000 = 66 min of 50 Hz streaming; a command record
is about 0.55 kB, so at most about 110 MB); ``dropped_record_count`` counts what was discarded.

Thread safety
-------------
One lock guards the model and the logs. It is never held while a send is blocked or delayed or while
``goto_target`` waits, so ``present()`` from a GUI thread keeps answering during a blocking goto. Commands are meant
to come from one executor thread (backends.base).
"""

from __future__ import annotations

import math
import numbers
import threading
from bisect import bisect_right
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from reachy_mini.motion.goto import GotoMove
from reachy_mini.utils.interpolation import InterpolationTechnique, time_trajectory

from reachy_motion.analysis.goto_replay import ik_joints, new_kinematics
from reachy_motion.backends.base import BackendError, Capabilities, CommandRecord, PresentState, check_command
from reachy_motion.executor.clock import Clock, RealClock
from reachy_motion.library.pose import REST, Pose6, RobotState, head_matrix, pose_from_matrix
from reachy_motion.library.spec import Easing

STRUCTURES: tuple[str, ...] = ("robot", "mujoco")
#: Accepted ``goto_target`` methods: the Easing values = SDK InterpolationTechnique values (utils/interpolation.py:108-114).
GOTO_METHODS: tuple[str, ...] = tuple(e.value for e in Easing)
DEFAULT_MAX_RECORDS = 200_000
PLAY_MOVE_RATE_HZ = 100.0  # play_move default play_frequency (abstract.py:829)

_YAW = 5  # index of the head yaw in the 9-vector [x, y, z, roll, pitch, yaw, ant_r, ant_l, body]
_ANT = slice(6, 8)
_BODY = 8
_TWO_PI = 2.0 * math.pi
_BLOCK_POLL_S = 0.02  # how often a sender blocked on ``block_send`` checks for close()


def _real(x: object, minimum: float, strict: bool) -> bool:
    """True for a finite real number (not a bool) above ``minimum`` (``strict``) or at least ``minimum``."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(float(x)):
        return False
    return float(x) > minimum if strict else float(x) >= minimum


def _unwrap(angle: float, ref: float) -> float:
    """``angle`` shifted by a multiple of 2 pi to lie within pi of ``ref``."""
    return float(angle + _TWO_PI * round((ref - angle) / _TWO_PI))


def _state_vec(state: RobotState) -> np.ndarray:
    return np.concatenate([state.head.as_array(), np.asarray(state.antennas, dtype=np.float64),
                           [float(state.body_yaw)]])


def _vec_state(v: np.ndarray) -> RobotState:
    return RobotState(head=Pose6.from_array(v[:6]), antennas=(float(v[6]), float(v[7])), body_yaw=float(v[_BODY]))


def _checked_state(state: object, label: str) -> np.ndarray:
    """9-vector of a RobotState: TypeError for another type, ValueError unless finite (commands: check_command)."""
    if not isinstance(state, RobotState):
        raise TypeError(f"{label} must be a RobotState, got {type(state).__name__}")
    v = _state_vec(state)
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{label} must be finite, like every command (backends.base.check_command): {state!r}")
    return v


@dataclass
class _Hold:
    """Constant target from ``t_ns`` on (a ``set_target``, the connect state or a ``reset``)."""

    t_ns: int
    value: np.ndarray  # (9,)

    def at(self, t_ns: int) -> np.ndarray:
        return self.value

    def next_change_after(self, t_ns: int) -> float:
        return math.inf


@dataclass
class _Goto:
    """SDK ``GotoMove`` sampled on the play_move grid from ``t_ns``; exactly ``end`` from ``end_ns`` on."""

    t_ns: int
    end_ns: int
    duration_s: float
    eval_period_ns: int
    move: GotoMove
    method: InterpolationTechnique
    end: np.ndarray  # (9,) the commanded target, yaw unwrapped like the samples
    yaw0: float  # present (unwrapped) yaw at the start
    yaw_span: float  # end Euler yaw - start Euler yaw: the SDK interpolates yaw as this signed scalar (goto.py:59-64)
    cache: dict[int, np.ndarray] = field(default_factory=dict)

    def _index(self, t_ns: int) -> int:
        return max(0, (t_ns - self.t_ns) // self.eval_period_ns)

    def at(self, t_ns: int) -> np.ndarray:
        if t_ns >= self.end_ns:
            return self.end
        j = self._index(t_ns)
        v = self.cache.get(j)
        if v is None:
            if len(self.cache) > 8:  # ticks only move forward: a few recent samples are enough
                self.cache.clear()
            v = self.cache[j] = self._evaluate(j)
        return v

    def _evaluate(self, j: int) -> np.ndarray:
        t = j * self.eval_period_ns / 1e9  # < duration_s by construction
        head, antennas, body_yaw = self.move.evaluate(t)
        p = pose_from_matrix(head).as_array()
        s = time_trajectory(min(t / self.duration_s, 1.0), self.method)  # utils/interpolation.py:117-123
        p[_YAW] = _unwrap(p[_YAW], self.yaw0 + self.yaw_span * s)
        return np.concatenate([p, np.asarray(antennas, dtype=np.float64), [float(body_yaw)]])

    def next_change_after(self, t_ns: int) -> float:
        if t_ns >= self.end_ns:
            return math.inf
        return min(self.t_ns + (self._index(t_ns) + 1) * self.eval_period_ns, self.end_ns)


class MockBackend:
    """Backend without a daemon: a deterministic model of the SDK command pipeline (see the module docstring)."""

    name: str = "mock"
    data_kind: str = "mock"
    capabilities: Capabilities = Capabilities(torque_control=False, real_hardware=False)

    def __init__(self, clock: Clock | None = None, structure: str = "robot", control_rate_hz: float = 50.0,
                 lag_tau_s: float = 0.0, start: RobotState = REST, *,
                 goto_eval_rate_hz: float = PLAY_MOVE_RATE_HZ, max_records: int = DEFAULT_MAX_RECORDS) -> None:
        if structure not in STRUCTURES:
            raise ValueError(f"structure must be one of {STRUCTURES}, got {structure!r}")
        for label, rate in (("control_rate_hz", control_rate_hz), ("goto_eval_rate_hz", goto_eval_rate_hz)):
            if not _real(rate, 0.0, strict=True):
                raise ValueError(f"{label} must be a positive finite number, got {rate!r}")
        if not _real(lag_tau_s, 0.0, strict=False):
            raise ValueError(f"lag_tau_s must be a finite number >= 0, got {lag_tau_s!r}")
        state0 = _checked_state(start, "start")
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1:
            raise ValueError(f"max_records must be an integer >= 1, got {max_records!r}")
        self._clock: Clock = clock if clock is not None else RealClock()
        self._structure = structure
        self._rate_hz = float(control_rate_hz)
        self._period_ns = max(1, round(1e9 / self._rate_hz))
        self._eval_period_ns = max(1, round(1e9 / float(goto_eval_rate_hz)))
        self._tau_s = float(lag_tau_s)
        self._decay = math.exp(-self._period_ns / 1e9 / self._tau_s) if self._tau_s > 0 else 0.0  # = 1 - alpha
        self._kin = new_kinematics()
        self._lock = threading.RLock()

        # fault injection (see the module docstring)
        self.fail_send: bool = False
        self.send_delay_s: float = 0.0
        self.block_send: threading.Event | None = None
        self.fail_present: bool = False
        self.freeze_present: bool = False
        self.daemon_error: str | None = None
        self._last_present: PresentState | None = None

        # logs and counters
        self._commands: deque[CommandRecord] = deque(maxlen=max_records)
        self._sound_cues: deque[tuple[int, str]] = deque(maxlen=max_records)
        self._delivered = 0
        self._failed = 0
        self._ignored = 0
        self._dropped = 0

        # model: committed tick k (-1 = none yet), x = position at t_k, g = goal written at tick k
        self._state0 = state0
        self._connected = False
        self._generation = 0  # bumped by connect() and close(): a call that started before is stale
        self._t0: int | None = None
        self._last_ns = -(2 ** 63)
        self._k = -1
        self._x = self._state0.copy()
        self._g = self._state0.copy()
        self._segments: list[_Hold | _Goto] = []
        self._starts: list[int] = []
        self._ik_key: bytes | None = None
        self._ik_value: np.ndarray | None = None

    # ------------------------------------------------------------------ connection
    def connect(self) -> None:
        """Start the virtual control loop now, from ``start`` (or the pose left at the last ``close``). Idempotent."""
        with self._lock:
            if self._connected:
                return
            self._last_ns = max(self._last_ns, int(self._clock.now_ns()))
            t0 = self._last_ns
            v = self._state0.copy()
            self._t0, self._k, self._x, self._g = t0, -1, v, v.copy()
            self._segments, self._starts = [_Hold(t0, v.copy())], [t0]
            self._connected = True
            self._generation += 1

    def close(self) -> None:
        """Stop the loop; the next ``connect()`` starts from the pose reached now. Idempotent.

        Always closes: should the model fail to evaluate that pose, the error propagates after the connection is
        closed and the next ``connect()`` starts from the previous start pose.
        """
        with self._lock:
            if not self._connected:
                return
            try:
                self._state0 = self._present_vec(self._now())
            finally:
                self._connected = False
                self._generation += 1
                self._segments, self._starts = [], []

    def __enter__(self) -> "MockBackend":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ commands
    def set_target(self, head: np.ndarray, antennas: Sequence[float], body_yaw: float) -> None:
        """Stream one target (``check_command`` rules: 4x4 head, (right, left) antennas, explicit body yaw)."""
        generation, rec = self._begin("set_target", head, antennas, body_yaw)
        self._transmit(generation)
        with self._lock:
            self._check_generation(generation)
            t = self._now()
            if self._goto_running(t):
                self._ignored += 1  # daemon: "Ignoring set_target command: a move is currently running"
                return
            self._advance(t)
            p = pose_from_matrix(rec.head).as_array()
            p[_YAW] = _unwrap(p[_YAW], self._target_at(t)[_YAW])
            self._push(_Hold(t, np.concatenate([p, np.asarray(rec.antennas, dtype=np.float64), [rec.body_yaw]])))
            self._delivered += 1

    def goto_target(self, head: np.ndarray, antennas: Sequence[float], body_yaw: float, duration: float,
                    method: str) -> None:
        """SDK-like blocking goto (module docstring, point 1): returns ``duration`` later on the injected clock.

        A goto the SDK's ``GotoMove`` cannot evaluate (head rotation block with det <= 0) raises ``BackendError`` at
        once, without blocking or writing anything, as the daemon fails such a goto task at its first sample.
        """
        m = getattr(method, "value", method)  # accepts Easing / InterpolationTechnique members too
        if not isinstance(m, str) or m not in GOTO_METHODS:
            raise BackendError(f"unknown interpolation method {method!r}; expected one of {GOTO_METHODS}")
        if not _real(duration, 0.0, strict=True):
            # the SDK client refuses duration <= 0 as well (reachy_mini.py:700-703)
            raise BackendError(f"duration must be a positive finite number of seconds, got {duration!r}")
        generation, rec = self._begin("goto_target", head, antennas, body_yaw, duration=float(duration), method=m)
        self._transmit(generation)
        with self._lock:
            self._check_generation(generation)
            t = self._now()
            if self._goto_running(t):
                self._ignored += 1  # daemon: "Ignoring play_move request: another move is running" (returns at once)
                return
            start = self._present_vec(t)
            try:  # sample 0 now: play_move evaluates the move at once (abstract.py:904-913), before writing anything
                goto = self._new_goto(t, start, rec)
                goto.at(t)
            except ValueError as exc:  # det <= 0: Rotation.from_matrix, utils/rotation.py:82-88
                self._failed += 1
                raise BackendError(f"mock: goto_target refused, the SDK GotoMove cannot evaluate it ({exc}); the "
                                   "daemon fails such a goto task at its first sample") from exc
            self._push(goto)
            self._delivered += 1
        self._clock.sleep_until(goto.end_ns)  # the SDK client blocks until the daemon reports the move finished
        with self._lock:
            if self._generation != generation:
                raise BackendError("mock: connection closed while goto_target was running")

    def log_sound_cue(self, file: str) -> None:
        """Record a sound cue ``(t_ns, file)``: metadata only, the mock plays no audio."""
        if not isinstance(file, str) or not file.strip():
            raise BackendError(f"a sound cue needs a file name, got {file!r}")
        with self._lock:
            self._require_connected()
            self._log(self._sound_cues, (self._now(), file))

    # ------------------------------------------------------------------ state
    def health(self) -> dict[str, Any]:
        """The daemon health a robot backend would report (``daemon_error`` fault injection, module docstring)."""
        err = self.daemon_error
        return {"state": "error" if err else "running", "error": err, "motor_mode": "enabled", "status_age_s": 0.0}

    def present(self) -> PresentState:
        """The modelled state at ``clock.now_ns()`` (data_kind "mock", ``joints`` = SDK IK of that state).

        Measurement faults (module docstring): ``fail_present`` raises, ``freeze_present`` repeats the last answer."""
        if self.fail_present:
            raise BackendError("mock: injected present() failure (fail_present=True)")
        with self._lock:
            self._require_connected()
            if self.freeze_present and self._last_present is not None:
                return self._last_present
            t = self._now()
            v = self._present_vec(t)
            head = head_matrix(Pose6.from_array(v[:6]))
            joints = self._joints(v, head)
            state = PresentState(t_ns=t, head=head, antennas=(float(v[6]), float(v[7])), body_yaw=float(v[_BODY]),
                                 joints=joints, data_kind=self.data_kind)
            self._last_present = state
        return state

    def current_target(self) -> RobotState:
        """The target the model holds at ``clock.now_ns()`` (before the write delay and the lag).

        The head yaw is wrapped to [-pi, pi]; the model itself keeps it unwrapped (module docstring, point 3).
        """
        with self._lock:
            self._require_connected()
            t = self._now()
            self._advance(t)
            v = self._target_at(t).copy()
        v[_YAW] = math.remainder(v[_YAW], _TWO_PI)
        return _vec_state(v)

    def reset(self, state: RobotState = REST, clear_log: bool = False) -> None:
        """Put the model at ``state`` now (target = goal = present), which also ends a running goto.

        While disconnected, ``state`` becomes the start of the next ``connect()``. Logs and counters are kept
        unless ``clear_log``. ``state`` must be finite (ValueError). The model's history is replaced without being
        evaluated, so a reset also recovers a model that cannot be evaluated any more.
        """
        v = _checked_state(state, "state")
        with self._lock:
            if self._connected:
                assert self._t0 is not None
                t = self._now()
                # every tick before t is committed at once, as _advance(t) would leave it - with the new state
                self._k = max(self._k, (t - self._t0 - 1) // self._period_ns)
                self._x, self._g = v.copy(), v.copy()
                self._segments, self._starts = [_Hold(t, v.copy())], [t]
            else:
                self._state0 = v.copy()
            if clear_log:
                self._commands.clear()
                self._sound_cues.clear()
                self._delivered = self._failed = self._ignored = self._dropped = 0

    # ------------------------------------------------------------------ introspection
    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def structure(self) -> str:
        return self._structure

    @property
    def control_rate_hz(self) -> float:
        return self._rate_hz

    @property
    def lag_tau_s(self) -> float:
        return self._tau_s

    @property
    def tick_period_ns(self) -> int:
        return self._period_ns

    @property
    def connect_time_ns(self) -> int | None:
        """Time of tick 0 of the current (or last) connection; None before the first ``connect()``."""
        with self._lock:
            return self._t0

    def tick_time_ns(self, k: int) -> int:
        """Time of control tick ``k`` of the current (or last) connection."""
        with self._lock:
            if self._t0 is None:
                raise BackendError("mock backend was never connected")
            return self._t0 + int(k) * self._period_ns

    @property
    def commands(self) -> list[CommandRecord]:
        """Snapshot of the command log (newest ``max_records``), in call order."""
        with self._lock:
            return list(self._commands)

    @property
    def sound_cues(self) -> list[tuple[int, str]]:
        with self._lock:
            return list(self._sound_cues)

    @property
    def motor_command_count(self) -> int:
        """Commands that reached the model (not failed, not ignored) since the last ``reset(clear_log=True)``."""
        with self._lock:
            return self._delivered

    @property
    def failed_send_count(self) -> int:
        with self._lock:
            return self._failed

    @property
    def ignored_command_count(self) -> int:
        with self._lock:
            return self._ignored

    @property
    def dropped_record_count(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def max_records(self) -> int:
        return int(self._commands.maxlen or 0)

    def __repr__(self) -> str:
        return (f"MockBackend(structure={self._structure!r}, control_rate_hz={self._rate_hz}, "
                f"lag_tau_s={self._tau_s}, connected={self._connected})")

    # ------------------------------------------------------------------ internals: calls
    def _require_connected(self) -> None:
        if not self._connected:
            raise BackendError("mock backend is not connected (call connect() first)")

    def _check_generation(self, generation: int) -> None:
        if not self._connected or self._generation != generation:
            raise BackendError("mock: connection closed before the command reached the model")

    def _now(self) -> int:
        """Clock reading, never earlier than a previous one (keeps the lazily advanced model causal)."""
        self._last_ns = max(self._last_ns, int(self._clock.now_ns()))
        return self._last_ns

    def _log(self, log: deque[Any], item: Any) -> None:
        if len(log) == log.maxlen:
            self._dropped += 1
        log.append(item)

    def _begin(self, kind: str, head: np.ndarray, antennas: Sequence[float], body_yaw: float,
               duration: float | None = None, method: str | None = None) -> tuple[int, CommandRecord]:
        """Validate and log one command at the start of its call; returns (connection generation, record)."""
        with self._lock:
            self._require_connected()
            h, a, b = check_command(head, antennas, body_yaw)
            rec = CommandRecord(t_ns=self._now(), kind=kind, head=h.copy(), antennas=a, body_yaw=b,
                                duration=duration, method=method)
            self._log(self._commands, rec)
            return self._generation, rec

    def _transmit(self, generation: int) -> None:
        """Fault injection between the call and the model; runs without the lock (present() stays responsive)."""
        gate = self.block_send
        if gate is not None:
            while not gate.wait(_BLOCK_POLL_S):
                with self._lock:
                    if not self._connected or self._generation != generation:
                        raise BackendError("mock: connection closed while the send was blocked")
        delay_s = self.send_delay_s
        if not _real(delay_s, 0.0, strict=False):
            raise ValueError(f"send_delay_s must be a finite number >= 0, got {delay_s!r}")
        if delay_s > 0:
            delay_ns = int(round(float(delay_s) * 1e9))
            advance = getattr(self._clock, "advance", None)  # VirtualClock: time spent inside the call
            if callable(advance):
                advance(delay_ns)
            else:
                self._clock.sleep_until(self._clock.now_ns() + delay_ns)
        if self.fail_send:
            with self._lock:
                self._failed += 1
            raise BackendError("mock: injected send failure (fail_send=True); the command did not reach the model")

    def _goto_running(self, t: int) -> bool:
        seg = self._segments[-1]
        return isinstance(seg, _Goto) and t < seg.end_ns

    def _new_goto(self, t_ns: int, start: np.ndarray, rec: CommandRecord) -> _Goto:
        """``GotoMove`` from the present state ``start`` to the command, as Backend.goto_target builds it."""
        assert rec.duration is not None and rec.method is not None
        method = InterpolationTechnique(rec.method)
        start_head = head_matrix(Pose6.from_array(start[:6]))
        target = pose_from_matrix(rec.head).as_array()
        yaw_span = float(target[_YAW] - pose_from_matrix(start_head).yaw)
        target[_YAW] = start[_YAW] + yaw_span
        move = GotoMove(start_head_pose=start_head, target_head_pose=rec.head.copy(),
                        start_antennas=start[_ANT].copy(),
                        target_antennas=np.asarray(rec.antennas, dtype=np.float64),
                        start_body_yaw=float(start[_BODY]), target_body_yaw=float(rec.body_yaw),
                        duration=float(rec.duration), method=method)
        end = np.concatenate([target, np.asarray(rec.antennas, dtype=np.float64), [rec.body_yaw]])
        return _Goto(t_ns=t_ns, end_ns=t_ns + max(1, round(rec.duration * 1e9)), duration_s=float(rec.duration),
                     eval_period_ns=self._eval_period_ns, move=move, method=method, end=end,
                     yaw0=float(start[_YAW]), yaw_span=yaw_span)

    def _push(self, seg: _Hold | _Goto) -> None:
        if (len(self._segments) > 1 and self._starts[-1] == seg.t_ns and isinstance(seg, _Hold)
                and isinstance(self._segments[-1], _Hold)):
            # same instant: the latest value wins anyway, so replace instead of piling up (memory stays bounded
            # even if time never moves); segment 0 is kept - it is what the robot head holds before its next tick
            self._segments[-1] = seg
            return
        self._segments.append(seg)
        self._starts.append(seg.t_ns)

    # ------------------------------------------------------------------ internals: model
    def _target_at(self, t: int) -> np.ndarray:
        """Target known at time ``t`` (latest command received at or before t; the connect state before that)."""
        i = max(bisect_right(self._starts, t) - 1, 0)
        return self._segments[i].at(t)

    def _next_change_after(self, t: int) -> float:
        """Earliest time after ``t`` at which ``_target_at`` may change (inf: never, given the commands so far)."""
        i = max(bisect_right(self._starts, t) - 1, 0)
        nxt = self._starts[i + 1] if i + 1 < len(self._starts) else math.inf
        return min(nxt, self._segments[i].next_change_after(t))

    def _goals(self, k: int) -> np.ndarray:
        """Goal written at tick ``k`` (module docstring, point 2)."""
        assert self._t0 is not None
        t = self._t0 + k * self._period_ns
        now = self._target_at(t)
        if self._structure == "mujoco":
            return now.copy()
        g = self._target_at(t - self._period_ns).copy()  # head + body: IK of tick k-1, written at tick k
        g[_ANT] = now[_ANT]  # antennas: no IK, written at the first tick that knows them
        return g

    def _advance(self, t: int) -> None:
        """Commit every tick before ``t``: no later command can change them (receipt times never decrease)."""
        assert self._t0 is not None
        period = self._period_ns
        while True:
            t_next = self._t0 + (self._k + 1) * period
            if t_next >= t:
                return
            t_k = t_next - period
            held = self._target_at(t_k)  # constant until the next change, for both structures
            limit = min(t, self._next_change_after(t_k))
            if t_next < limit and np.array_equal(self._g, held):
                k_end = (limit - 1 - self._t0) // period  # last tick before ``limit``: all its goals are ``held``
                self._x = held + (self._x - held) * self._decay ** (k_end - self._k)
                self._g = held
                self._k = k_end
            else:
                self._x = self._g + (self._x - self._g) * self._decay
                self._g = self._goals(self._k + 1)
                self._k += 1
            self._prune()

    def _prune(self) -> None:
        """Drop the segments that end before the committed tick (keeps memory bounded)."""
        assert self._t0 is not None
        i = bisect_right(self._starts, self._t0 + self._k * self._period_ns) - 1
        if i > 0:
            del self._segments[:i]
            del self._starts[:i]

    def _present_vec(self, t: int) -> np.ndarray:
        """Present 9-vector at ``t`` (continuous first-order response to the goal held since the last tick)."""
        assert self._t0 is not None
        self._advance(t)
        t_next = self._t0 + (self._k + 1) * self._period_ns
        if self._tau_s == 0.0:
            return self._goals(self._k + 1) if t_next == t else self._g.copy()
        dt_s = (t - (t_next - self._period_ns)) / 1e9
        return self._g + (self._x - self._g) * math.exp(-dt_s / self._tau_s)

    def _joints(self, v: np.ndarray, head: np.ndarray) -> np.ndarray:
        """IK [body_yaw, stewart_1..6] of the present head and body yaw (cached while the state is unchanged)."""
        key = np.concatenate([v[:6], v[_BODY:]]).tobytes()
        if key != self._ik_key or self._ik_value is None:
            self._ik_value = ik_joints(self._kin, head, float(v[_BODY]))
            self._ik_key = key
        return self._ik_value.copy()
