"""System B (Phase 3): Gemini Live speech-to-speech on the robot.

Robot mic -> AudioManager subscription ``cloud_uplink`` -> 16-bit PCM blocks -> Live
WebSocket -> 24 kHz response audio -> ``AudioManager.open_stream`` -> robot speaker.
Full duplex by default (``[cloud] uplink_while_speaking``). Same public interface as
:class:`~reachy_s2s.conversation.manager.CascadedPipeline` (start / stop /
reset_history / save_conversation / add_listener / turns / history), so the dashboard
and the CLI drive both systems the same way. Only one of them runs at a time.

Threads, none of which waits for another:
* capture (AudioManager) -> **uplink** thread: PCM conversion, 40 ms blocks, half-duplex
  policy -> hands blocks to the asyncio loop (bounded; the oldest block is dropped and
  counted if the network stalls, nothing blocks the capture);
* **cloud-loop** (asyncio): connect / send / receive / reconnect / reset;
* player (AudioManager) paces the received audio to the robot;
* **local VAD** (Silero, the cascade's exact settings): *measurement only*. It gives the
  same ``speech_end_est`` as the cascade, so ``speech_end_to_push_ms`` means the same
  thing for both systems. Turn taking is the server's (automatic VAD).
* one short watcher thread per response (first push -> SPEAKING, end of play-out -> row).

Events: server-side ones are ``cloud.*`` (only what the API actually sends: e.g.
``cloud.voice_activity`` appears only if the server emits it), local ones are ``vad.*``
and ``playback.*``. Cloud-internal ASR / LLM / TTS times are not observable and are not
reported. ``output_est_*`` / ``response_latency_est_ms`` use the ASSUMED output latency.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..audio.signals import silence, tone
from ..audio.wavio import write_wav
from ..config import AppConfig
from ..engines.base import EngineError
from ..engines.gemini_live import (
    INPUT_MIME,
    INPUT_RATE,
    OUTPUT_RATE,
    CloudEvent,
    GeminiLiveConnector,
    build_live_config,
    classify_live_error,
    config_summary,
    float_to_pcm16,
    network_probe,
    parse_server_message,
    pcm16_to_float,
    validate_cloud_config,
)
from ..engines.vad import FrameModel, Utterance, VadWorker
from ..evaluation.session import Session
from ..evaluation.tables import append_csv_row
from .history import ConversationHistory, Exchange
from .manager import heard_prefix, motion_condition, motion_span
from .state import InteractionState, StateTracker

log = logging.getLogger(__name__)
S = InteractionState
_STREAM_END = object()


@dataclass
class CloudTurnRecord:
    """One server response (times: s since session start). Shared columns keep the cascade's names."""

    turn_id: int
    utterance_id: int | None = None  # local VAD utterance answered by this response (None: none found)
    status: str = "responding"  # responding | done | interrupted | aborted | no_response | stopped
    system: str = "cloud"
    variant: str = ""  # [cloud] variant (B)
    motion: str = "none"  # Phase 4.5 motion condition (MotionController.label(); "none" = no controller)
    connection_id: int | None = None
    error_stage: str | None = None
    error_kind: str | None = None
    error_status: int | None = None
    error_message: str | None = None
    # local VAD (measurement only, same estimator as the cascade)
    speech_start_est: float | None = None
    speech_end_est: float | None = None
    endpoint: float | None = None
    vad_endpoint_delay_ms: float | None = None
    speech_s: float | None = None
    pauses_ms: list[int] = field(default_factory=list)
    clipping_rate: float | None = None
    input_peak_dbfs: float | None = None
    input_rms_dbfs: float | None = None
    merged_utterances: int = 0  # earlier local utterances answered by the same response
    # uplink / server events (local receive times)
    uplink_speech_end_sent: float | None = None  # block containing the local speech end was sent
    cloud_activity_start: float | None = None  # server voice_activity (only if the server sends it)
    cloud_activity_end: float | None = None
    cloud_first_output: float | None = None
    cloud_first_audio: float | None = None
    cloud_generation_complete: float | None = None
    cloud_turn_complete: float | None = None
    turn_complete_reason: str | None = None
    interrupted: bool = False
    cloud_audio_chunks: int = 0
    cloud_audio_s: float | None = None
    # transcripts (same column names as the cascade: transcript = user, response = robot)
    transcript: str | None = None
    response: str | None = None
    # usage (from usage_metadata, if sent)
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    thought_tokens: int | None = None
    total_tokens: int | None = None
    # robot audio queue (same names as the cascade)
    queue_enqueue: float | None = None
    queue_first_push: float | None = None
    queue_last_push: float | None = None
    output_est_start: float | None = None  # ESTIMATE (assumed output latency)
    output_est_end: float | None = None
    output_hint_ms: float | None = None
    playback_gain_db: float | None = None
    playback_clipped: int | None = None
    playback_status: str | None = None
    # derived (PC-measured unless *_est)
    speech_end_to_cloud_audio_ms: float | None = None
    uplink_to_cloud_audio_ms: float | None = None
    cloud_audio_to_push_ms: float | None = None
    speech_end_to_push_ms: float | None = None  # the common metric (same definition as the cascade)
    response_latency_est_ms: float | None = None
    history_exchanges: int | None = None
    # Phase 4 barge-in (same column names as the cascade where the meaning is the same)
    user_interrupt_start: float | None = None  # first local-VAD speech start while this response ran (est.)
    user_interrupt_phase: str | None = None  # processing | speaking
    bargein_activity_start: float | None = None  # server ACTIVITY_START received while this response ran
    cloud_interrupted: float | None = None  # server "interrupted" received (the server's barge-in decision)
    bargein_detect_delay_ms: float | None = None  # cloud_interrupted - user_interrupt_start
    playback_stop_delay_ms: float | None = None  # local stop request -> local queue + robot buffer cleared
    heard_est_s: float | None = None  # ESTIMATE (assumed output latency)
    heard_ratio_est: float | None = None  # heard / received audio
    sent_unheard_s: float | None = None  # received from the server but never played (still in its context)
    premature_suspect: bool = False  # the user spoke again soon after the local endpoint of this utterance
    resume_after_endpoint_ms: float | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["pauses_ms"] = " ".join(str(p) for p in self.pauses_ms)
        return row


def _handle_id(handle: str | None) -> str | None:
    return None if not handle else hashlib.sha256(handle.encode()).hexdigest()[:8]


class CloudS2SPipeline:
    kind = "cloud"

    def __init__(self, audio: Any, cfg: AppConfig, session: Session, state: StateTracker,
                 connector: Any = None, vad_model: FrameModel | None = None) -> None:
        validate_cloud_config(cfg.cloud)
        self.audio = audio
        self.cfg = cfg
        self.ccfg = cfg.cloud
        self.session = session
        self.events = session.events
        self.state = state
        self.connector = connector if connector is not None else GeminiLiveConnector()
        #: Phase 4.5: returns the motion condition for each turn row (MotionController.label)
        self.motion_label: Callable[[], str] | None = None
        self.history = ConversationHistory(max_exchanges=100_000)  # display / log only: the server keeps the context
        self.vad = VadWorker(audio, cfg.vad, self.events, self._on_local_utterance, model=vad_model,
                             on_speech_start=self._on_local_speech_start)
        self.turns: list[CloudTurnRecord] = []
        self.interrupted_responses: list[dict[str, Any]] = []
        self._stops: dict[int, Any] = {}  # turn id -> StopRequest of the server interruption
        self._last_local_end: tuple[int, int] | None = None  # (utterance id, speech end index)
        self._premature: dict[int, dict[str, Any]] = {}  # utterance id -> suspicion found before it was linked
        self._turn_ids = itertools.count(1)
        self._listeners: list[Callable[[str, dict[str, Any]], None]] = []
        self.running = False
        self.status = "idle"  # idle | connecting | connected | reconnecting | error | stopped
        self.connection_id = 0
        self.uplink_stats = {"blocks": 0, "bytes": 0, "dropped_queue": 0, "dropped_disconnected": 0,
                             "paused_chunks": 0, "stream_end": 0}
        self._lock = threading.RLock()
        self._local_start_cond = threading.Condition(self._lock)  # a local speech start was linked to a response
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._loop_thread: threading.Thread | None = None
        self._uplink_thread: threading.Thread | None = None
        self._mic_q: Any = None
        self._send_q: asyncio.Queue | None = None
        self._control: asyncio.Event | None = None
        self._reset_requested = False
        self._reconnect_requested = False
        self._resume_handle: str | None = None
        self._sent: deque[tuple[int, int]] = deque(maxlen=3000)  # (end sample index, t_send_ns)
        self._conn_first_index: int | None = None
        # The server's voice_activity audio_offset counts the audio of the whole (resumed) session,
        # not of the current connection (observed 2026-10-17 after a resume): keep both counts.
        self._conn_samples = 0  # sent on the current connection
        self._session_samples_before = 0  # sent on earlier connections of the same server session
        self._current: CloudTurnRecord | None = None
        self._handle: Any = None
        self._robot_audio: dict[int, list[np.ndarray]] = {}  # turn_id -> received chunks
        self._ignore_audio = False
        self._done_events: dict[int, threading.Event] = {}
        self._pending_utts: deque[Utterance] = deque(maxlen=20)
        self._user_parts: list[str] = []
        self._last_activity_start_ns: int | None = None
        self._last_activity_end_ns: int | None = None
        self._last_usage: dict[str, Any] = {}
        self._last_session_s = 0.0
        self._watchers: list[threading.Thread] = []
        self.rows_written = 0

    # ------------------------------------------------------------ control
    def add_listener(self, fn: Callable[[str, dict[str, Any]], None]) -> None:
        """``fn(kind, payload)``: kinds ``turn``, ``history`` and ``status`` (connection state)."""
        self._listeners.append(fn)

    def _notify(self, kind: str, payload: dict[str, Any]) -> None:
        for fn in list(self._listeners):
            try:
                fn(kind, payload)
            except Exception:
                log.exception("pipeline listener failed")

    def _set_status(self, status: str, **info: Any) -> None:
        self.status = status
        self.events.log("cloud.status", status=status, connection_id=self.connection_id, **info)
        self._notify("status", {"status": status, "connection_id": self.connection_id, "model": self.ccfg.model,
                                "voice": self.ccfg.voice or "(server default)", **info})

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._stop.clear()
        self._reset_requested = self._reconnect_requested = False
        self._set_status("connecting")
        self.state.set(S.IDLE, "cloud connecting")
        self.vad.start()
        self._mic_q = self.audio.subscribe("cloud_uplink")
        self._loop_ready.clear()
        self._loop_thread = threading.Thread(target=self._loop_main, name="cloud-loop", daemon=True)
        self._loop_thread.start()
        self._loop_ready.wait(5)
        self._uplink_thread = threading.Thread(target=self._uplink_main, name="cloud-uplink", daemon=True)
        self._uplink_thread.start()
        self.events.log("cloud.started", model=self.ccfg.model, voice=self.ccfg.voice or None,
                        cloud=asdict(self.ccfg), vad=asdict(self.cfg.vad), output_gain_db=self.cfg.audio.output_gain_db,
                        system_prompt_chars=len(self.cfg.llm.system_prompt.strip()))

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self._stop.set()
        self._wake_loop()
        self.vad.stop()
        self.audio.unsubscribe("cloud_uplink")
        for thread in (self._uplink_thread, self._loop_thread):
            if thread is not None:
                thread.join(timeout=10)
        self.audio.stop_playback("cloud stop")
        self._abort_current("stopped", "pipeline stopped")
        for watcher in self._watchers:  # every response gets its row before stop() returns
            watcher.join(timeout=5)
        self._watchers.clear()
        self.state.set(S.IDLE, "cloud stop")
        self._set_status("stopped")
        self.save_conversation()
        self.events.log("cloud.stopped", turns=len(self.turns), uplink=dict(self.uplink_stats))

    def reset_history(self) -> int:
        """Forget the conversation: a new Live session without a resumption handle."""
        dropped = self.history.reset()
        with self._lock:
            self._resume_handle = None
            self._reset_requested = True
            self._pending_utts.clear()
            self._user_parts.clear()
        self.audio.stop_playback("cloud reset")
        self.events.log("conversation.reset", dropped_exchanges=dropped,
                        cloud="reconnect without resumption handle (server context discarded)")
        self._wake_loop()
        self.save_conversation()
        self._notify("history", {"transcript": []})
        return dropped

    def reconnect(self) -> None:
        """Close the WebSocket and reconnect with the last resumption handle (context kept).

        Exercises the same path as a server GoAway; used to verify on the real service
        that a resumed connection still knows the conversation.
        """
        with self._lock:
            self._reconnect_requested = True
            has_handle = bool(self._resume_handle)
        self.events.log("cloud.reconnect_requested", has_handle=has_handle)
        self._wake_loop()

    def save_conversation(self) -> None:
        data = {
            "session_id": self.session.id,
            "system": "cloud",
            "variant": self.ccfg.variant,
            "model": self.ccfg.model,
            "saved_at_t": self.session.t(),
            "resets": self.history.resets,
            "connections": self.connection_id,
            "history": self.history.transcript(),
            "interrupted_responses": list(self.interrupted_responses),
            "turns": [t.to_row() for t in self.turns],
        }
        self.session.path("conversation.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")

    def _wake_loop(self) -> None:
        loop, control = self._loop, self._control
        if loop is not None and control is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(control.set)

    def _rel(self, t_ns: int | None) -> float | None:
        return None if t_ns is None else self.events.rel(t_ns)

    # ------------------------------------------------------------- uplink
    def _uplink_main(self) -> None:
        block = int(INPUT_RATE * self.ccfg.uplink_chunk_ms / 1000)
        parts: list[np.ndarray] = []
        n = 0
        paused = False
        while not self._stop.is_set():
            chunk = self._mic_q.get(timeout=0.1) if self._mic_q is not None else None
            if chunk is None:
                continue
            if not self._connected.is_set():
                self.uplink_stats["dropped_disconnected"] += 1
                parts, n = [], 0
                continue
            if self.ccfg.uplink_while_speaking == "pause" and self.audio.is_speaking():
                if not paused:
                    paused = True
                    parts, n = [], 0
                    self._post(_STREAM_END)
                    self.events.log("cloud.uplink_paused", reason="robot speaking (half duplex)")
                self.uplink_stats["paused_chunks"] += 1
                continue
            if paused:
                paused = False
                self.events.log("cloud.uplink_resumed")
            mono = self.audio.mono(chunk.samples)
            parts.append(mono)
            n += mono.shape[0]
            if n >= block:
                end_index = chunk.start_index + chunk.samples.shape[0]
                self._post((float_to_pcm16(np.concatenate(parts)), end_index))
                parts, n = [], 0

    def _post(self, item: Any) -> None:
        loop = self._loop
        if loop is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._enqueue_send, item)

    def _enqueue_send(self, item: Any) -> None:  # loop thread
        q = self._send_q
        if q is None:
            return
        limit = max(1, self.ccfg.uplink_queue_ms // self.ccfg.uplink_chunk_ms)
        while q.qsize() >= limit:
            q.get_nowait()
            self.uplink_stats["dropped_queue"] += 1
            if self.uplink_stats["dropped_queue"] % 25 == 1:
                self.events.log("cloud.uplink_dropped", dropped_total=self.uplink_stats["dropped_queue"],
                                queue_ms=self.ccfg.uplink_queue_ms)
        q.put_nowait(item)

    async def _send_loop(self, session: Any) -> None:
        from google.genai import types

        first = True
        while True:
            item = await self._send_q.get()
            if item is _STREAM_END:
                await session.send_realtime_input(audio_stream_end=True)
                self.uplink_stats["stream_end"] += 1
                continue
            pcm, end_index = item
            await session.send_realtime_input(audio=types.Blob(data=pcm, mime_type=INPUT_MIME))
            t = time.perf_counter_ns()
            self._sent.append((end_index, t))
            self._conn_samples += len(pcm) // 2
            self.uplink_stats["blocks"] += 1
            self.uplink_stats["bytes"] += len(pcm)
            if first:
                first = False
                self._conn_first_index = end_index - len(pcm) // 2
                self.events.log("cloud.uplink_started", t_ns=t, connection_id=self.connection_id,
                                first_index=self._conn_first_index, block_ms=self.ccfg.uplink_chunk_ms)

    def _sent_time(self, index: int) -> int | None:
        for end, t in self._sent.copy():  # copy() is atomic; the loop thread appends concurrently
            if end >= index:
                return t
        return None

    # --------------------------------------------------------- async core
    def _loop_main(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception:
            log.exception("cloud loop crashed")
            self.events.log("cloud.crash")

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._send_q = asyncio.Queue()
        self._control = asyncio.Event()
        self._loop_ready.set()
        failures = 0
        while not self._stop.is_set():
            with self._lock:
                fresh = self._reset_requested
                self._reset_requested = self._reconnect_requested = False
                handle = None if fresh else self._resume_handle
            self._control.clear()
            outcome = await self._session_once(handle, failures)
            if self._stop.is_set():
                break
            if outcome in ("reset", "reconnect"):
                failures = 0
                continue
            if self._last_session_s >= 10.0:  # it worked for a while: start counting again
                failures = 0
            failures += 1  # "failed", or "closed" by the server (counted so a closing loop backs off)
            if failures > self.ccfg.reconnect_attempts:
                self._set_status("error", failures=failures)
                self.state.set(S.ERROR, "cloud connection failed", failures=failures)
                self._error_tone()
                await self._control.wait()  # Reset / Stop retries from scratch
                failures = 0
                continue
            self._set_status("reconnecting", attempt=failures)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._control.wait(), timeout=self.ccfg.reconnect_backoff_s * failures)

    async def _session_once(self, handle: str | None, attempt: int) -> str:
        self.connection_id += 1
        cid = self.connection_id
        if handle is None:  # a fresh server session: its audio clock starts at 0
            self._session_samples_before = 0
        self._conn_samples = 0
        self._conn_first_index = None
        config = build_live_config(self.ccfg, self.cfg.llm.system_prompt, handle)
        self.events.log("cloud.connecting", connection_id=cid, model=self.ccfg.model, resumed=bool(handle),
                        handle_id=_handle_id(handle), attempt=attempt, config=config_summary(config))
        if self.ccfg.network_probe:
            probe = await asyncio.get_running_loop().run_in_executor(None, network_probe)
            self.events.log("cloud.network_probe", connection_id=cid, **probe)
        t0 = time.perf_counter_ns()
        outcome = "failed"
        try:
            async with contextlib.AsyncExitStack() as stack:
                connect = asyncio.ensure_future(stack.enter_async_context(self.connector.connect(self.ccfg.model, config)))
                waiter = asyncio.ensure_future(self._control.wait())
                done, _ = await asyncio.wait({connect, waiter}, timeout=self.ccfg.connect_timeout_s,
                                             return_when=asyncio.FIRST_COMPLETED)
                if connect not in done:
                    connect.cancel()
                    waiter.cancel()
                    with contextlib.suppress(BaseException):
                        await connect
                    if waiter in done:
                        return self._control_outcome()
                    raise TimeoutError(f"connect timed out after {self.ccfg.connect_timeout_s} s")
                waiter.cancel()
                session = connect.result()
                await self._drain_send_queue()
                self._connected.set()
                self.events.log("cloud.connected", connection_id=cid, resumed=bool(handle),
                                connect_ms=round((time.perf_counter_ns() - t0) / 1e6, 1))
                self._set_status("connected", resumed=bool(handle))
                if self.state.state in (S.IDLE, S.ERROR):
                    self.state.set(S.LISTENING, "cloud connected")
                tasks = {asyncio.ensure_future(self._send_loop(session)),
                         asyncio.ensure_future(self._receive_loop(session)),
                         asyncio.ensure_future(self._control.wait())}
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    exc = task.exception()
                    if exc is not None:
                        raise exc
                outcome = self._control_outcome()
        except Exception as exc:  # noqa: BLE001 - every failure is classified and logged
            err = classify_live_error(exc)
            outcome = "closed" if err.kind == "closed" else "failed"
            self.events.log("cloud.error", connection_id=cid, **err.to_dict())
            if err.kind == "rate_limit":
                self.events.log("cloud.rate_limit", connection_id=cid, message=str(err)[:300])
        finally:
            self._connected.clear()
            self._session_samples_before += self._conn_samples  # a resumed session continues this clock
            self._last_session_s = (time.perf_counter_ns() - t0) / 1e9
            self.events.log("cloud.disconnected", connection_id=cid, outcome=outcome,
                            duration_s=round(self._last_session_s, 2), samples_sent=self._conn_samples,
                            session_samples=self._session_samples_before)
            if outcome != "stop":
                self._abort_current("reset" if outcome == "reset" else "aborted", f"connection ended ({outcome})")
        return outcome

    def _control_outcome(self) -> str:
        if self._stop.is_set():
            return "stop"
        if self._reset_requested:
            return "reset"
        return "reconnect"

    async def _drain_send_queue(self) -> None:
        while self._send_q is not None and not self._send_q.empty():
            self._send_q.get_nowait()

    async def _receive_loop(self, session: Any) -> None:
        while True:
            async for msg in session.receive():
                for ev in parse_server_message(msg):
                    try:
                        self._on_cloud_event(ev)
                    except Exception:  # a handler bug must not kill the session
                        log.exception("cloud event handler failed (%s)", ev.kind)
                        self.events.log("cloud.handler_error", event=ev.kind)

    # ------------------------------------------------------- server events
    def _on_cloud_event(self, ev: CloudEvent) -> None:  # loop thread: keep it short, never block
        k, d, t = ev.kind, ev.data, ev.t_ns
        if k == "audio":
            self._on_audio(d["pcm"], d["rate"], t)
        elif k in ("input_transcript", "interim_input_transcript"):
            if d.get("text"):
                if k == "input_transcript":
                    self._user_parts.append(d["text"])
                    if self._current is not None:
                        self._current.transcript = "".join(self._user_parts).strip() or None
                self.events.log(f"cloud.{k}", t_ns=t, text=d["text"], finished=d.get("finished"))
        elif k == "output_transcript":
            turn = self._current or self._begin_response(t)
            turn.response = ((turn.response or "") + d.get("text", "")) or None
            self.events.log("cloud.output_transcript", t_ns=t, turn_id=turn.turn_id, text=d.get("text"))
            self._notify("turn", turn.to_row())
        elif k == "interrupted":
            self._on_interrupted(t)
        elif k == "generation_complete":
            if self._current is not None:
                self._current.cloud_generation_complete = self._rel(t)
                self._end_stream()
            self.events.log("cloud.generation_complete", t_ns=t,
                            turn_id=None if self._current is None else self._current.turn_id)
        elif k == "turn_complete":
            self._on_turn_complete(t, d.get("reason"))
        elif k == "voice_activity":
            self._on_voice_activity(d.get("type"), d.get("audio_offset"), t)
        elif k == "usage":
            self._last_usage = d
            target = self._current or (self.turns[-1] if self.turns else None)
            if target is not None:
                target.prompt_tokens = d.get("prompt_token_count")
                target.response_tokens = d.get("response_token_count")
                target.thought_tokens = d.get("thoughts_token_count")
                target.total_tokens = d.get("total_token_count")
            self.events.log("cloud.usage", t_ns=t, **d)
        elif k == "resumption":
            if d.get("resumable") and d.get("handle"):
                self._resume_handle = d["handle"]
            self.events.log("cloud.resumption_update", t_ns=t, resumable=d.get("resumable"),
                            handle_id=_handle_id(d.get("handle")))
        elif k == "go_away":
            self.events.log("cloud.go_away", t_ns=t, time_left=d.get("time_left"))
            self._reconnect_requested = True
            if self._current is None:  # idle: reconnect now; otherwise after this response
                self._control.set()
        elif k == "setup_complete":
            self.events.log("cloud.setup_complete", t_ns=t, connection_id=self.connection_id)
        elif k in ("waiting_for_input", "vad_signal", "tool_call"):
            self.events.log(f"cloud.{k}", t_ns=t, **d)
        # "text" / "thought" parts are not spoken; they are not logged one by one

    def offset_to_index(self, offset: str | None) -> int | None:
        """Server ``audio_offset`` ("12.34s", session audio time) -> local mic sample index (approximate)."""
        if not offset or self._conn_first_index is None:
            return None
        try:
            samples = int(round(float(str(offset).rstrip("s")) * INPUT_RATE))
        except ValueError:
            return None
        return self._conn_first_index + samples - self._session_samples_before

    def _on_voice_activity(self, kind: str | None, offset: str | None, t: int) -> None:
        index = self.offset_to_index(offset)
        self.events.log("cloud.voice_activity", t_ns=t, type=kind, audio_offset=offset, index=index)
        if kind == "ACTIVITY_START":
            self._last_activity_start_ns = t
            current = self._current
            if current is not None and current.bargein_activity_start is None:  # user speech during a response
                current.bargein_activity_start = self._rel(t)
        elif kind == "ACTIVITY_END":
            self._last_activity_end_ns = t
            if self.state.state == S.LISTENING and self._current is None:
                self.state.set(S.PROCESSING, "server end of speech")

    def _begin_response(self, t: int) -> CloudTurnRecord:
        turn = CloudTurnRecord(turn_id=next(self._turn_ids), connection_id=self.connection_id,
                               variant=self.ccfg.variant, motion=motion_condition(self.motion_label))
        turn.cloud_first_output = self._rel(t)
        turn.output_hint_ms = round(self.audio.backend.output_latency_hint_s * 1000, 1)
        turn.playback_gain_db = self.cfg.audio.output_gain_db
        turn.transcript = "".join(self._user_parts).strip() or None
        if self._last_activity_start_ns is not None:
            turn.cloud_activity_start = self._rel(self._last_activity_start_ns)
        if self._last_activity_end_ns is not None:
            turn.cloud_activity_end = self._rel(self._last_activity_end_ns)
        self._last_activity_start_ns = self._last_activity_end_ns = None
        with self._lock:
            self._link_utterance(turn, t)
            self._current = turn
            self.turns.append(turn)
            self._done_events[turn.turn_id] = threading.Event()
        self._robot_audio[turn.turn_id] = []
        self._ignore_audio = False
        self._handle = self.audio.open_stream(OUTPUT_RATE, label=f"cloud_turn_{turn.turn_id}",
                                              meta={"turn_id": turn.turn_id, "connection_id": self.connection_id})
        turn.queue_enqueue = self._rel(self._handle.item.t_enqueued_ns)
        if self.state.state == S.LISTENING:
            self.state.set(S.PROCESSING, "cloud response started", turn_id=turn.turn_id)
        self.events.log("cloud.response_started", t_ns=t, turn_id=turn.turn_id, utterance_id=turn.utterance_id,
                        connection_id=self.connection_id)
        watcher = threading.Thread(target=self._watch_turn, args=(turn, self._handle),
                                   name=f"cloud-turn-{turn.turn_id}", daemon=True)
        self._watchers = [w for w in self._watchers if w.is_alive()] + [watcher]
        watcher.start()
        self._notify("turn", turn.to_row())
        return turn

    def _on_audio(self, pcm: bytes, rate: int, t: int) -> None:
        if self._ignore_audio:
            return
        turn = self._current or self._begin_response(t)
        samples = pcm16_to_float(pcm)
        if turn.cloud_first_audio is None:
            turn.cloud_first_audio = self._rel(t)
            self.events.log("cloud.first_audio", t_ns=t, turn_id=turn.turn_id, rate=rate, samples=samples.shape[0])
        turn.cloud_audio_chunks += 1
        turn.cloud_audio_s = round((turn.cloud_audio_s or 0.0) + samples.shape[0] / rate, 4)
        if self.ccfg.save_turn_audio:
            self._robot_audio.setdefault(turn.turn_id, []).append(samples)
        if self._handle is not None and not self._handle.item.ended:
            self._handle.write(samples)

    def _end_stream(self) -> None:
        if self._handle is not None and not self._handle.item.ended:
            self._handle.end()

    def _on_interrupted(self, t: int) -> None:
        turn = self._current
        self.events.log("cloud.interrupted", t_ns=t, turn_id=None if turn is None else turn.turn_id)
        if turn is None:
            return
        turn.interrupted = True
        turn.cloud_interrupted = self._rel(t)
        if turn.user_interrupt_start is not None:
            turn.bargein_detect_delay_ms = round((turn.cloud_interrupted - turn.user_interrupt_start) * 1000, 1)
        self._ignore_audio = True
        self._stops[turn.turn_id] = self.audio.stop_playback("cloud interrupted")
        self.state.set(S.INTERRUPTED, "server interrupted the response", turn_id=turn.turn_id)

    def _on_turn_complete(self, t: int, reason: str | None) -> None:
        turn = self._current
        self.events.log("cloud.turn_complete", t_ns=t, turn_id=None if turn is None else turn.turn_id, reason=reason)
        self._ignore_audio = False
        if turn is None:
            # the server ended a turn without output (e.g. it heard noise); nothing to play
            if self.state.state in (S.PROCESSING, S.INTERRUPTED):
                self.state.set(S.LISTENING, "cloud turn ended without output")
            self._user_parts.clear()
            return
        turn.cloud_turn_complete = self._rel(t)
        turn.turn_complete_reason = reason
        self._end_stream()
        with self._lock:
            self._current = None
            self._user_parts.clear()
            done = self._done_events.get(turn.turn_id)
        if done is not None:
            done.set()
        if self._reconnect_requested and self._control is not None:  # GoAway arrived during this response
            self._control.set()

    def _abort_current(self, status: str, message: str) -> None:
        with self._lock:
            turn, self._current = self._current, None
            done = None if turn is None else self._done_events.get(turn.turn_id)
        if turn is None:
            return
        turn.status = status
        if status == "aborted":  # connection lost mid-response: what already arrived still plays out
            turn.error_stage, turn.error_kind, turn.error_message = "cloud", "disconnected", message
        self._end_stream()
        if status == "reset":
            self.audio.stop_playback("cloud reset")
        if done is not None:
            done.set()

    # ------------------------------------------------------ local measure
    def _on_local_speech_start(self, index: int, t_ns: int) -> None:  # VAD thread, measurement only
        if not self.running:
            return
        est = self.audio.estimate_arrival_ns(index)
        premature = None
        with self._lock:
            last, self._last_local_end = self._last_local_end, None  # only the first speech after an endpoint
            if last is not None:
                after_ms = (index - last[1]) * 1000 / INPUT_RATE - self.cfg.vad.silence_timeout_ms
                if after_ms <= self.cfg.cascaded.premature_window_ms:
                    premature = {"utterance_id": last[0], "resume_after_endpoint_ms": round(after_ms)}
                    linked = next((t for t in reversed(self.turns) if t.utterance_id == last[0]), None)
                    if linked is not None:
                        self._apply_premature(linked, premature)
                    else:  # applied if the server answers that utterance on its own (_fill_utterance)
                        self._premature[last[0]] = premature
            turn = self._current
            if turn is None and self.turns and est is not None:
                last_turn = self.turns[-1]
                # the server may interrupt (and close) the response before our VAD confirms the speech
                if last_turn.cloud_interrupted is not None and self._rel(est) <= last_turn.cloud_interrupted:
                    turn = last_turn
            phase = None
            if turn is not None and turn.user_interrupt_start is None and est is not None:
                start = self._rel(est)
                phase = "speaking" if turn.queue_first_push is not None and start >= turn.queue_first_push \
                    else "processing"
                turn.user_interrupt_start, turn.user_interrupt_phase = start, phase
                if turn.cloud_interrupted is not None:
                    turn.bargein_detect_delay_ms = round((turn.cloud_interrupted - start) * 1000, 1)
                self._local_start_cond.notify_all()
        if premature is not None:
            gaps, gap_ms = self.audio.gaps_between(last[1], index)
            self.events.log("endpoint.premature_suspect", t_ns=t_ns, **premature, source="local VAD",
                            pause_ms=round((index - last[1]) * 1000 / INPUT_RATE), speech_start_index=index,
                            t_speech_start_est=self._rel(est), input_gaps=gaps, input_gap_ms=gap_ms)
        if phase is not None:
            self.events.log("user.speech_while_busy", t_ns=t_ns, turn_id=turn.turn_id, phase=phase, index=index,
                            t_speech_start_est=self._rel(est), action="server decides (automatic VAD)")

    @staticmethod
    def _apply_premature(turn: CloudTurnRecord, info: dict[str, Any] | None) -> None:
        if info is not None:
            turn.premature_suspect = True
            turn.resume_after_endpoint_ms = info["resume_after_endpoint_ms"]

    def _on_local_utterance(self, utt: Utterance) -> None:  # VAD thread
        if not self.running:
            return
        with self._lock:
            self._last_local_end = (utt.id, utt.speech_end_index)
            current = self._current
            if current is not None and current.utterance_id is None and current.cloud_first_output is not None \
                    and utt.t_speech_end_est_ns is not None \
                    and self.events.rel(utt.t_speech_end_est_ns) < current.cloud_first_output:
                self._fill_utterance(current, utt)  # the server answered before our endpoint
                return
            self._pending_utts.append(utt)
        if self.state.state == S.LISTENING:
            self.state.set(S.PROCESSING, "local speech end (waiting for the cloud)", utterance_id=utt.id)
        loop = self._loop
        if loop is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(loop.call_later, self.ccfg.response_timeout_s,
                                          self._check_no_response, utt)

    def _link_utterance(self, turn: CloudTurnRecord, t_first_ns: int) -> None:
        candidates = [u for u in self._pending_utts
                      if (u.t_speech_end_est_ns or u.t_endpoint_ns) <= t_first_ns]
        if not candidates:
            return
        utt = candidates[-1]
        for u in candidates:
            self._pending_utts.remove(u)
        turn.merged_utterances = len(candidates) - 1
        self._fill_utterance(turn, utt)

    def _fill_utterance(self, turn: CloudTurnRecord, utt: Utterance) -> None:
        turn.utterance_id = utt.id
        self._apply_premature(turn, self._premature.pop(utt.id, None))
        turn.speech_start_est = self._rel(utt.t_speech_start_est_ns)
        turn.speech_end_est = self._rel(utt.t_speech_end_est_ns)
        turn.endpoint = self._rel(utt.t_endpoint_ns)
        if utt.t_speech_end_est_ns is not None:
            turn.vad_endpoint_delay_ms = round((utt.t_endpoint_ns - utt.t_speech_end_est_ns) / 1e6, 1)
        turn.speech_s = round(utt.speech_duration_s, 3)
        turn.pauses_ms = [round((b - a) * 1000 / utt.sample_rate) for a, b in utt.pauses]
        turn.clipping_rate = round(utt.clipping_rate, 6)
        turn.input_peak_dbfs, turn.input_rms_dbfs = round(utt.peak_dbfs, 2), round(utt.rms_dbfs, 2)
        turn.uplink_speech_end_sent = self._rel(self._sent_time(utt.speech_end_index))
        if self.ccfg.save_turn_audio:
            write_wav(self.session.audio_dir / f"turn_{turn.turn_id:04d}_user.wav", utt.audio, utt.sample_rate)

    def _check_no_response(self, utt: Utterance) -> None:  # loop thread, response_timeout_s after the endpoint
        with self._lock:
            if utt not in self._pending_utts:
                return  # answered (or merged into an answer)
            self._pending_utts.remove(utt)
            turn = CloudTurnRecord(turn_id=next(self._turn_ids), status="no_response", connection_id=self.connection_id,
                                   motion=motion_condition(self.motion_label))
            self._fill_utterance(turn, utt)
            self.turns.append(turn)
        self.events.log("cloud.no_response", utterance_id=utt.id, turn_id=turn.turn_id,
                        waited_s=self.ccfg.response_timeout_s)
        if self.state.state == S.PROCESSING and self._current is None:
            self.state.set(S.LISTENING, "no cloud response")
        self._write_row(turn)

    # ------------------------------------------------------- per response
    def _watch_turn(self, turn: CloudTurnRecord, handle: Any) -> None:
        handle.wait_first_push()
        item = handle.item
        if item.status == "playing" and item.t_first_push_ns is not None:
            turn.queue_first_push = self._rel(item.t_first_push_ns)
            if self.state.state in (S.LISTENING, S.PROCESSING):
                self.state.set(S.SPEAKING, "first cloud audio pushed", turn_id=turn.turn_id)
            self._notify("turn", turn.to_row())
        done = self._done_events.get(turn.turn_id)
        if done is not None:
            done.wait()
        handle.wait(timeout=600, until_heard=True)
        self._finish_turn(turn, handle)

    def _finish_turn(self, turn: CloudTurnRecord, handle: Any) -> None:
        item = handle.item
        turn.queue_first_push = self._rel(item.t_first_push_ns)
        turn.queue_last_push = self._rel(item.t_last_push_ns)
        turn.output_est_start = self._rel(item.t_output_start_est_ns)
        turn.output_est_end = self._rel(item.t_output_end_est_ns)
        turn.playback_clipped = item.clipped_samples
        turn.playback_status = item.status
        final = ("interrupted" if turn.interrupted else "done") if turn.status == "responding" else turn.status
        if turn.speech_end_est is not None:
            if turn.cloud_first_audio is not None:
                turn.speech_end_to_cloud_audio_ms = round((turn.cloud_first_audio - turn.speech_end_est) * 1000, 1)
            if turn.queue_first_push is not None:
                turn.speech_end_to_push_ms = round((turn.queue_first_push - turn.speech_end_est) * 1000, 1)
                turn.response_latency_est_ms = round(turn.speech_end_to_push_ms + (turn.output_hint_ms or 0.0), 1)
        if turn.uplink_speech_end_sent is not None and turn.cloud_first_audio is not None:
            turn.uplink_to_cloud_audio_ms = round((turn.cloud_first_audio - turn.uplink_speech_end_sent) * 1000, 1)
        if turn.cloud_first_audio is not None and turn.queue_first_push is not None:
            turn.cloud_audio_to_push_ms = round((turn.queue_first_push - turn.cloud_first_audio) * 1000, 1)
        received = self._robot_audio.pop(turn.turn_id, None)
        if self.ccfg.save_turn_audio and received:
            write_wav(self.session.audio_dir / f"turn_{turn.turn_id:04d}_robot.wav", np.concatenate(received),
                      OUTPUT_RATE)
        stop = self._stops.pop(turn.turn_id, None)
        if stop is not None:
            self._record_interruption(turn, item, stop)
        if final in ("done", "interrupted") and (turn.transcript or turn.response):
            text = turn.response or ""
            self.history.add(Exchange(turn.turn_id, turn.transcript or "(자막 없음)",
                                      text + (" [끊김]" if turn.interrupted else ""), {}, []))
        turn.history_exchanges = len(self.history)
        turn.status = final  # only now: "done" means the row is about to be written
        self._write_row(turn)
        self._notify("history", {"transcript": self.history.transcript()})
        with self._lock:
            idle = self._current is None
        if self.running and idle and self.state.state in (S.SPEAKING, S.INTERRUPTED, S.PROCESSING):
            self.state.set(S.LISTENING, "cloud response finished", turn_id=turn.turn_id)

    def _record_interruption(self, turn: CloudTurnRecord, item: Any, stop: Any) -> None:
        """Server barge-in: how fast our output stopped and how much of the answer was heard (estimate).

        The server keeps its own context of an interrupted answer (we cannot edit it); what it
        received / sent is logged next to what the user presumably heard.
        """
        stop.done.wait(2.0)  # the player clears the robot buffer in its own thread
        turn.playback_stop_delay_ms = stop.stop_ms
        with self._local_start_cond:  # our VAD confirms speech after min_speech_ms; the server may be faster
            self._local_start_cond.wait_for(lambda: turn.user_interrupt_start is not None, timeout=0.6)
        heard = 0.0
        if item.t_output_start_est_ns is not None:
            pushed_s = item.frames_pushed / item.out_rate
            heard = max(0.0, min(pushed_s, (stop.t_request_ns - item.t_output_start_est_ns) / 1e9))
        received = turn.cloud_audio_s or 0.0
        turn.heard_est_s = round(heard, 3)
        if received > 0:
            turn.heard_ratio_est = round(min(1.0, heard / received), 3)
            turn.sent_unheard_s = round(max(0.0, received - heard), 3)
        record = {"turn_id": turn.turn_id, "policy": "server", "heard_est_s": turn.heard_est_s,
                  "heard_ratio_est": turn.heard_ratio_est, "received_audio_s": turn.cloud_audio_s,
                  "sent_unheard_s": turn.sent_unheard_s, "user_text": turn.transcript, "full_text": turn.response,
                  "heard_text": heard_prefix(turn.response or "", turn.heard_ratio_est or 0.0),
                  "playback_stop_delay_ms": turn.playback_stop_delay_ms,
                  "bargein_detect_delay_ms": turn.bargein_detect_delay_ms}
        self.events.log("conversation.interrupted_response", **record)
        self.interrupted_responses.append(record)

    def _write_row(self, turn: CloudTurnRecord) -> None:
        turn.motion = motion_span(turn.motion, self.motion_label)
        self.events.log("turn.end", **turn.to_row())
        append_csv_row(self.session.path("turns.csv"), turn.to_row())
        self.save_conversation()
        self.rows_written += 1
        self._notify("turn", turn.to_row())

    def _error_tone(self) -> None:
        if not self.cfg.cascaded.error_tone:
            return
        rate = OUTPUT_RATE
        beep = np.concatenate([tone(440.0, 0.12, rate, amplitude=0.25), silence(0.08, rate),
                               tone(330.0, 0.18, rate, amplitude=0.25)])
        with contextlib.suppress(Exception):
            self.audio.play(beep, rate, label="cloud_error_tone")
