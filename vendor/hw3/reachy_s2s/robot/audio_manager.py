"""Audio manager: microphone fan-out, recording and paced speaker playback.

Threads (none of them ever waits for the GUI or an engine):

* capture thread  -- ``backend.read()`` -> ring buffer, active recording and one
  :class:`DropOldestQueue` per subscriber (mel monitor, VAD, cloud uplink, ...);
* player thread   -- pulls queued :class:`PlaybackItem` audio, pushes it in
  ``playback_chunk_ms`` blocks while staying ``playback_lead_ms`` ahead of real
  time, and handles interruption (drop queue + ``backend.clear_output()``).

Timing vocabulary used in the event log (all ``perf_counter_ns`` based):

* ``t_push``       -- audio handed to the backend (= "queued to the robot");
* ``t_output_est`` -- ``t_push`` (or the end of audio already queued) plus the
  backend's ``output_latency_hint_s``: an *estimate* of when it is heard. The
  real value is measured acoustically (``apps/loopback_probe.py``).
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.buffers import AudioRingBuffer, DropOldestQueue
from ..audio.convert import as_frames, match_channels, rms_dbfs, to_mono
from ..audio.resample import StreamResampler
from ..audio.types import AudioChunk
from ..audio.wavio import write_wav
from ..config import AudioConfig
from ..evaluation.event_logger import EventLogger
from .backends.base import AudioBackend

log = logging.getLogger(__name__)

_NS = 1_000_000_000


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------
class Recording:
    """Microphone audio captured between ``start_recording`` and ``stop_recording``."""

    def __init__(self, name: str, sample_rate: int, channels: int, t_request_ns: int) -> None:
        self.name = name
        self.sample_rate = sample_rate
        self.channels = channels
        self.t_request_ns = t_request_ns
        self.t_stop_ns: int | None = None
        self.start_index: int | None = None
        self._chunks: list[np.ndarray] = []
        #: ``(end_index, t_arrival_ns)`` per chunk: maps the sample clock to PC time.
        self.anchors: list[tuple[int, int]] = []

    def append(self, chunk: AudioChunk) -> None:
        if self.start_index is None:
            self.start_index = chunk.start_index
        self._chunks.append(chunk.samples)
        self.anchors.append((chunk.end_index, chunk.t_arrival_ns))

    @property
    def frames(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros((0, self.channels), np.float32)
        return np.vstack(self._chunks)

    @property
    def num_frames(self) -> int:
        return sum(c.shape[0] for c in self._chunks)

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.sample_rate

    def mono(self, channel: str | int = "left") -> np.ndarray:
        return to_mono(self.frames, channel)

    def save(self, directory: str | Path, t0_ns: int) -> dict[str, str]:
        """Write ``<name>.wav`` (all channels, 16-bit) and ``<name>.json`` (timing)."""
        directory = Path(directory)
        wav_path = write_wav(directory / f"{self.name}.wav", self.frames, self.sample_rate)
        meta = {
            "name": self.name,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "num_frames": self.num_frames,
            "duration_s": round(self.duration_s, 6),
            "start_index": self.start_index,
            "t_request": round((self.t_request_ns - t0_ns) / _NS, 6),
            "t_stop": None if self.t_stop_ns is None else round((self.t_stop_ns - t0_ns) / _NS, 6),
            "anchors_note": "[end_index, t_arrival (s since session start)] per received chunk",
            "anchors": [[end, round((t - t0_ns) / _NS, 6)] for end, t in self.anchors],
        }
        json_path = directory / f"{self.name}.json"
        json_path.write_text(json.dumps(meta), encoding="utf-8")
        return {"wav": str(wav_path), "json": str(json_path)}


# --------------------------------------------------------------------------
# Playback
# --------------------------------------------------------------------------
class PlaybackItem:
    """One utterance / sound queued for the speaker (mono at the backend rate)."""

    _ids = itertools.count(1)

    def __init__(self, label: str, source_rate: int, out_rate: int) -> None:
        self.id = next(self._ids)
        self.label = label
        self.source_rate = source_rate
        self.out_rate = out_rate
        self.resampler = StreamResampler(source_rate, out_rate, channels=1)
        self.pending: deque[np.ndarray] = deque()
        self.pending_frames = 0
        self.ended = False
        self.status = "queued"  # queued | playing | done | interrupted | failed
        self.source_frames = 0
        self.frames_pushed = 0
        self.t_enqueued_ns = time.perf_counter_ns()
        self.t_first_push_ns: int | None = None
        self.t_output_start_est_ns: int | None = None
        self.t_last_push_ns: int | None = None
        self.t_output_end_est_ns: int | None = None
        self.clipped_samples = 0  # after the player's fixed output gain
        self.first_pushed = threading.Event()  # also set if the item ends without any push
        self.pushed_all = threading.Event()
        # set when stop_playback() hit this item (barge-in metrics)
        self.t_interrupt_request_ns: int | None = None
        self.t_interrupt_cleared_ns: int | None = None
        self.interrupt_reason: str | None = None

    def take(self, max_frames: int) -> np.ndarray:
        parts, n = [], 0
        while self.pending and n < max_frames:
            head = self.pending[0]
            need = max_frames - n
            if head.shape[0] <= need:
                parts.append(self.pending.popleft())
                n += head.shape[0]
            else:
                parts.append(head[:need])
                self.pending[0] = head[need:]
                n += need
        self.pending_frames -= n
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)


class StopRequest:
    """One ``stop_playback`` call, filled in by the player thread once it was handled.

    ``t_cleared_ns`` is when the local queue and the robot buffer were emptied (``None``: nothing
    was queued or audible, so nothing had to be cleared). ``done`` is set either way.
    """

    def __init__(self, reason: str, t_request_ns: int) -> None:
        self.reason = reason
        self.t_request_ns = t_request_ns
        self.t_cleared_ns: int | None = None
        self.item_ids: list[int] = []
        self.done = threading.Event()

    @property
    def stop_ms(self) -> float | None:
        """Request -> local queue and robot buffer cleared (measured on the PC)."""
        return None if self.t_cleared_ns is None else round((self.t_cleared_ns - self.t_request_ns) / 1e6, 1)


class PlaybackHandle:
    """What callers get back from ``play`` / ``open_stream``."""

    def __init__(self, item: PlaybackItem, player: "Player") -> None:
        self._item = item
        self._player = player

    @property
    def id(self) -> int:
        return self._item.id

    @property
    def status(self) -> str:
        return self._item.status

    @property
    def item(self) -> PlaybackItem:
        return self._item

    def write(self, samples: np.ndarray) -> None:
        """Append source-rate audio (streaming, e.g. TTS chunks)."""
        self._player.write(self._item, samples)

    def end(self) -> None:
        """No more audio for this item (flushes the resampler)."""
        self._player.end(self._item)

    def wait_first_push(self, timeout: float | None = None) -> bool:
        """Block until the first block reached the backend (or the item ended without one)."""
        return self._item.first_pushed.wait(timeout)

    def wait(self, timeout: float | None = None, until_heard: bool = True) -> bool:
        """Wait until everything was pushed and (optionally) its estimated end is heard."""
        deadline = None if timeout is None else time.perf_counter() + timeout
        if not self._item.pushed_all.wait(timeout):
            return False
        if until_heard and self._item.t_output_end_est_ns is not None:
            remaining = (self._item.t_output_end_est_ns - time.perf_counter_ns()) / _NS
            if deadline is not None:
                remaining = min(remaining, deadline - time.perf_counter())
            if remaining > 0:
                time.sleep(remaining)
        return True


class Player:
    def __init__(
        self,
        backend: AudioBackend,
        chunk_ms: float,
        lead_ms: float,
        events: EventLogger,
        mono_channel: str | int = "left",
        gain_db: float = 0.0,
    ) -> None:
        self._backend = backend
        self._events = events
        self._rate = backend.output_rate
        self.gain_db = float(gain_db)
        self._gain = np.float32(10.0 ** (self.gain_db / 20.0))
        self._chunk_frames = max(1, int(round(self._rate * chunk_ms / 1000.0)))
        self._lead_ns = int(lead_ms * 1e6)
        self._hint_ns = int(backend.output_latency_hint_s * _NS)
        self._mono_channel = mono_channel
        self._cond = threading.Condition()
        self._queue: deque[PlaybackItem] = deque()
        self._current: PlaybackItem | None = None
        self._interrupt: StopRequest | None = None
        self._stopping = False
        self._thread: threading.Thread | None = None
        #: Virtual playhead: when the backend will have consumed everything pushed.
        self._pushed_until_ns = 0
        #: Estimated end of audible output (playhead + latency hint; "now" after a clear).
        self._audible_until_ns = 0
        self._underrun_logged_for: int | None = None
        #: Phase 4.5 playback taps (motion): told about every block pushed and every clear, in this
        #: thread. Copy-on-write tuple so the player never takes a lock for them.
        self._taps: tuple[Any, ...] = ()
        self.tap_errors = 0

    # -- public (any thread) ------------------------------------------------
    def add_tap(self, tap: Any) -> None:
        """``tap.on_block(mono_samples, t_audible_ns, rate)`` / ``tap.on_flush(t_ns)``; must not block."""
        if tap not in self._taps:
            self._taps = (*self._taps, tap)

    def remove_tap(self, tap: Any) -> None:
        self._taps = tuple(t for t in self._taps if t is not tap)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="audio-player", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.interrupt("shutdown")
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def enqueue(self, label: str, source_rate: int, meta: dict[str, Any] | None = None) -> PlaybackHandle:
        item = PlaybackItem(label, source_rate, self._rate)
        with self._cond:
            self._queue.append(item)
            self._cond.notify_all()
        self._events.log("playback.enqueued", t_ns=item.t_enqueued_ns, id=item.id, label=label,
                         source_rate=source_rate, **(meta or {}))
        return PlaybackHandle(item, self)

    def write(self, item: PlaybackItem, samples: np.ndarray) -> None:
        if item.ended:
            raise RuntimeError(f"playback item {item.id} already ended")
        mono = to_mono(samples, self._mono_channel) if np.ndim(samples) == 2 else np.asarray(samples, np.float32)
        out = item.resampler.process(mono)
        with self._cond:
            item.source_frames += mono.shape[0]
            if item.status in ("interrupted", "failed"):
                return
            if out.shape[0]:
                item.pending.append(out)
                item.pending_frames += out.shape[0]
            self._cond.notify_all()

    def end(self, item: PlaybackItem) -> None:
        tail = item.resampler.flush() if not item.ended else np.zeros(0, np.float32)
        with self._cond:
            if tail.shape[0] and item.status not in ("interrupted", "failed"):
                item.pending.append(tail)
                item.pending_frames += tail.shape[0]
            item.ended = True
            self._cond.notify_all()

    def interrupt(self, reason: str) -> StopRequest:
        """Drop everything queued and clear the robot buffer (asynchronously, in the player thread).

        Requests made before the player handled a pending one share it (same object returned).
        """
        t_ns = time.perf_counter_ns()
        with self._cond:
            if self._interrupt is None:
                self._interrupt = StopRequest(reason, t_ns)
            request = self._interrupt
            self._cond.notify_all()
        return request

    def is_active(self) -> bool:
        """Something queued/pushing, or pushed audio presumably still audible."""
        with self._cond:
            busy = self._current is not None or bool(self._queue)
        return busy or time.perf_counter_ns() < self._audible_until_ns

    # -- player thread -------------------------------------------------------
    def _run(self) -> None:
        waiting_logged = False
        while True:
            action = None
            with self._cond:
                while action is None:
                    if self._interrupt is not None:
                        request, self._interrupt = self._interrupt, None
                        victims = ([self._current] if self._current else []) + list(self._queue)
                        self._current = None
                        self._queue.clear()
                        action = ("interrupt", request, victims)
                        break
                    if self._stopping:
                        return
                    if self._current is None and self._queue:
                        self._current = self._queue.popleft()
                    item = self._current
                    if item is None:
                        self._cond.wait()
                        continue
                    if not self._backend.output_ready():
                        if not waiting_logged:
                            self._events.log("playback.waiting_output_ready", id=item.id)
                            waiting_logged = True
                        self._cond.wait(0.05)
                        continue
                    if item.pending_frames == 0:
                        if item.ended:
                            self._current = None
                            action = ("finish", item)
                            break
                        if item.frames_pushed and self._underrun_logged_for != item.id \
                                and time.perf_counter_ns() > self._pushed_until_ns:
                            self._underrun_logged_for = item.id
                            action = ("underrun", item)
                            break
                        self._cond.wait(0.01)  # streaming source is slower than real time
                        continue
                    ahead = self._pushed_until_ns - time.perf_counter_ns()
                    if ahead > self._lead_ns:
                        self._cond.wait(min((ahead - self._lead_ns) / _NS, 0.01))
                        continue
                    chunk = item.take(self._chunk_frames)
                    if item.ended and item.pending_frames == 0 and chunk.shape[0] < self._chunk_frames:
                        # Pad the last block: the SDK transposes (n, ch) arrays with
                        # n < ch, and a few ms of trailing silence is harmless.
                        chunk = np.concatenate([chunk, np.zeros(self._chunk_frames - chunk.shape[0], np.float32)])
                    action = ("push", item, chunk)
            kind = action[0]
            if kind == "interrupt":
                self._do_interrupt(*action[1:])
            elif kind == "finish":
                self._finish(action[1])
            elif kind == "underrun":
                self._events.log("playback.underrun", id=action[1].id)
            else:
                self._push(action[1], action[2])

    def _push(self, item: PlaybackItem, mono_chunk: np.ndarray) -> None:
        if self.gain_db != 0.0:
            mono_chunk = mono_chunk * self._gain
        over = np.abs(mono_chunk) > 1.0
        if over.any():
            item.clipped_samples += int(np.count_nonzero(over))
            mono_chunk = np.clip(mono_chunk, -1.0, 1.0)
        now = time.perf_counter_ns()
        start_ns = max(now, self._pushed_until_ns)  # queued behind audio still playing
        try:
            self._backend.write(match_channels(mono_chunk, self._backend.output_channels))
        except Exception as exc:
            log.exception("backend write failed")
            item.status = "failed"
            item.first_pushed.set()
            item.pushed_all.set()
            with self._cond:
                if self._current is item:
                    self._current = None
            self._events.log("playback.error", id=item.id, error=repr(exc))
            return
        self._pushed_until_ns = start_ns + int(mono_chunk.shape[0] * _NS / self._rate)
        self._audible_until_ns = self._pushed_until_ns + self._hint_ns
        if item.t_first_push_ns is None:
            item.status = "playing"
            item.t_first_push_ns = now
            item.t_output_start_est_ns = start_ns + self._hint_ns
            item.first_pushed.set()
            self._events.log(
                "playback.first_push", t_ns=now, id=item.id, label=item.label,
                t_output_est=self._events.rel(item.t_output_start_est_ns),
                queue_wait_ms=round((now - item.t_enqueued_ns) / 1e6, 2),
            )
        item.frames_pushed += mono_chunk.shape[0]
        item.t_last_push_ns = now
        for tap in self._taps:
            try:
                tap.on_block(mono_chunk, start_ns + self._hint_ns, self._rate)
            except Exception:  # noqa: BLE001 - motion must never disturb playback
                self.tap_errors += 1

    def _finish(self, item: PlaybackItem) -> None:
        item.status = "done"
        item.t_output_end_est_ns = self._pushed_until_ns + self._hint_ns
        item.first_pushed.set()
        item.pushed_all.set()
        self._events.log(
            "playback.finished", id=item.id, label=item.label,
            frames_pushed=item.frames_pushed,
            duration_s=round(item.frames_pushed / self._rate, 4),
            gain_db=self.gain_db, clipped_samples=item.clipped_samples,
            t_last_push=None if item.t_last_push_ns is None else self._events.rel(item.t_last_push_ns),
            t_output_end_est=self._events.rel(item.t_output_end_est_ns),
        )

    def _do_interrupt(self, request: StopRequest, victims: list[PlaybackItem]) -> None:
        try:
            self._clear(request, victims)
        finally:
            request.done.set()

    def _clear(self, request: StopRequest, victims: list[PlaybackItem]) -> None:
        reason, t_req = request.reason, request.t_request_ns
        audible = time.perf_counter_ns() < self._audible_until_ns
        if not victims and not audible:
            return
        t_clear_start = time.perf_counter_ns()
        clear_error = None
        try:
            self._backend.clear_output()
        except Exception as exc:  # e.g. daemon REST call failed
            log.exception("clear_output failed")
            clear_error = repr(exc)
        t_cleared = time.perf_counter_ns()
        self._pushed_until_ns = t_cleared
        self._audible_until_ns = t_cleared
        for tap in self._taps:
            try:
                tap.on_flush(t_cleared)
            except Exception:  # noqa: BLE001
                self.tap_errors += 1
        request.t_cleared_ns, request.item_ids = t_cleared, [v.id for v in victims]
        for v in victims:
            v.status = "interrupted"
            v.t_interrupt_request_ns, v.t_interrupt_cleared_ns, v.interrupt_reason = t_req, t_cleared, reason
            v.first_pushed.set()
            v.pushed_all.set()
        current = victims[0] if victims else None
        heard_est = None
        if current is not None and current.t_output_start_est_ns is not None:
            heard_est = max(0.0, (t_req - current.t_output_start_est_ns) / _NS)
        self._events.log(
            "playback.interrupted", reason=reason,
            t_request=self._events.rel(t_req),
            clear_ms=round((t_cleared - t_clear_start) / 1e6, 2),
            stop_latency_ms=round((t_cleared - t_req) / 1e6, 2),
            id=None if current is None else current.id,
            label=None if current is None else current.label,
            pushed_s=None if current is None else round(current.frames_pushed / self._rate, 4),
            heard_est_s=None if heard_est is None else round(heard_est, 4),
            dropped_items=max(0, len(victims) - 1),
            error=clear_error,
        )


# --------------------------------------------------------------------------
# Audio manager
# --------------------------------------------------------------------------
class AudioManager:
    def __init__(self, backend: AudioBackend, cfg: AudioConfig, events: EventLogger | None = None) -> None:
        self.backend = backend
        self.cfg = cfg
        self.events = events if events is not None else EventLogger(None)
        self.mic_channel = cfg.mic_channel
        self.input_rate = backend.input_rate
        self.input_channels = backend.input_channels
        self.ring = AudioRingBuffer(int(cfg.ring_seconds * self.input_rate), self.input_channels)
        self.player = Player(backend, cfg.playback_chunk_ms, cfg.playback_lead_ms, self.events, cfg.mic_channel,
                             gain_db=cfg.output_gain_db)
        #: recent (end_index, t_arrival_ns) anchors for estimate_arrival_ns()
        self._anchors: deque[tuple[int, int]] = deque(maxlen=max(1, int(3.0 * self.input_rate / 320)))
        self._anchors_lock = threading.Lock()
        self.gaps = 0
        #: recent input gaps (at_index, gap_ms, t_arrival_ns) so turns can tell Wi-Fi gaps from VAD errors
        self.gap_log: deque[tuple[int, float, int]] = deque(maxlen=2000)
        self._subs: dict[str, DropOldestQueue[AudioChunk]] = {}
        self._subs_lock = threading.Lock()
        self._recording: Recording | None = None
        self._rec_lock = threading.Lock()
        self._capture_thread: threading.Thread | None = None
        self._stop_capture = threading.Event()
        self._first_input = threading.Event()
        self.started = False
        # capture statistics (written by the capture thread only)
        self._next_index = 0
        self.chunks_captured = 0
        self.read_errors = 0
        self.min_chunk_frames: int | None = None
        self.max_chunk_frames = 0
        self.t_first_input_ns: int | None = None
        self.t_last_input_ns: int | None = None
        self._rate_ref: tuple[int, int] | None = None  # (frames, t_ns) after 1 s warm-up

    def set_event_logger(self, events: EventLogger) -> None:
        """Send further audio/playback events to another session's log."""
        self.events = events
        self.player._events = events

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self.backend.open()
        self.events.log("audio.backend_opened", **self.backend.describe())
        self._stop_capture.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, name="audio-capture", daemon=True)
        self._capture_thread.start()
        if not self._first_input.wait(self.cfg.first_input_timeout_s):
            self._shutdown_capture()
            self.backend.close()
            raise TimeoutError(
                f"no microphone audio from {self.backend.name} within {self.cfg.first_input_timeout_s:.0f} s"
            )
        deadline = time.perf_counter() + self.cfg.first_input_timeout_s
        while not self.backend.output_ready() and time.perf_counter() < deadline:
            time.sleep(0.05)
        ready = self.backend.output_ready()
        self.events.log("audio.output_ready" if ready else "audio.output_not_ready")
        if not ready:
            log.warning("speaker path not ready yet; playback will wait for it")
        self.player.start()
        self.started = True

    def stop(self) -> None:
        if not self.started and self._capture_thread is None:
            return
        self.player.stop()
        self._shutdown_capture()
        self.backend.close()
        self.started = False
        self.events.log("audio.stopped", **self.input_stats())

    def _shutdown_capture(self) -> None:
        self._stop_capture.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5)
            self._capture_thread = None
        with self._subs_lock:
            for queue in self._subs.values():
                queue.close()

    # -- capture thread ------------------------------------------------------
    def _capture_loop(self) -> None:
        while not self._stop_capture.is_set():
            try:
                data = self.backend.read()
            except Exception as exc:
                self.read_errors += 1
                if self.read_errors <= 5 or self.read_errors % 100 == 0:
                    log.exception("audio read failed")
                    self.events.log("audio.read_error", count=self.read_errors, error=repr(exc))
                time.sleep(0.05)
                continue
            if data is None or len(data) == 0:
                continue
            t_ns = time.perf_counter_ns()
            frames = as_frames(data)
            chunk = AudioChunk(frames, self.input_rate, self._next_index, t_ns)
            self._next_index = chunk.end_index
            self.ring.write(frames, chunk.start_index)
            self._update_stats(chunk)
            with self._rec_lock:
                if self._recording is not None:
                    self._recording.append(chunk)
            with self._subs_lock:
                queues = list(self._subs.values())
            for queue in queues:
                queue.put(chunk)

    def _update_stats(self, chunk: AudioChunk) -> None:
        n = chunk.num_frames
        self.chunks_captured += 1
        self.min_chunk_frames = n if self.min_chunk_frames is None else min(self.min_chunk_frames, n)
        self.max_chunk_frames = max(self.max_chunk_frames, n)
        if self.t_last_input_ns is not None:
            gap_ms = (chunk.t_arrival_ns - self.t_last_input_ns) / 1e6
            if gap_ms > self.cfg.gap_warn_ms:
                self.gaps += 1
                self.gap_log.append((chunk.start_index, gap_ms, chunk.t_arrival_ns))
                self.events.log("audio.gap", t_ns=chunk.t_arrival_ns, gap_ms=round(gap_ms, 1),
                                at_index=chunk.start_index)
        with self._anchors_lock:
            self._anchors.append((chunk.end_index, chunk.t_arrival_ns))
        self.t_last_input_ns = chunk.t_arrival_ns
        if self.t_first_input_ns is None:
            self.t_first_input_ns = chunk.t_arrival_ns
            self.events.log("audio.first_input", t_ns=chunk.t_arrival_ns, frames=n, channels=chunk.channels,
                            sample_rate=chunk.sample_rate)
            self._first_input.set()
        elif self._rate_ref is None and chunk.t_arrival_ns - self.t_first_input_ns >= _NS:
            self._rate_ref = (chunk.end_index, chunk.t_arrival_ns)

    # -- input API -----------------------------------------------------------
    def subscribe(self, name: str, maxsize: int | None = None) -> DropOldestQueue[AudioChunk]:
        queue: DropOldestQueue[AudioChunk] = DropOldestQueue(maxsize or self.cfg.subscriber_queue_chunks)
        with self._subs_lock:
            if name in self._subs:
                raise ValueError(f"subscriber {name!r} already exists")
            self._subs[name] = queue
        return queue

    def unsubscribe(self, name: str) -> None:
        with self._subs_lock:
            queue = self._subs.pop(name, None)
        if queue is not None:
            queue.close()

    def mono(self, frames: np.ndarray) -> np.ndarray:
        return to_mono(frames, self.mic_channel)

    def level_dbfs(self, window_s: float = 0.05) -> float:
        return rms_dbfs(self.mono(self.ring.read_last(int(window_s * self.input_rate))))

    @property
    def input_index(self) -> int:
        return self.ring.end_index

    def gaps_between(self, start_index: int, end_index: int) -> tuple[int, float]:
        """Input gaps (arrival pauses > ``gap_warn_ms``) whose first late chunk starts in [start, end): (count, ms)."""
        hits = [ms for index, ms, _t in self.gap_log.copy() if start_index <= index < end_index]
        return len(hits), round(float(sum(hits)), 1)

    def estimate_arrival_ns(self, index: int) -> int | None:
        """PC time at which input sample ``index`` reached this process, without queueing.

        Lower envelope of the last ~3 s of chunk arrivals at the nominal rate:
        chunks are only ever late (tick quantization, GIL, Wi-Fi jitter), so the
        earliest consistent arrival is the best local estimate. Local so that
        clock drift and dropouts elsewhere in the session do not bias it.
        """
        with self._anchors_lock:
            anchors = list(self._anchors)
        if not anchors:
            return None
        ns_per_sample = 1e9 / self.input_rate
        offset = min(t - end * ns_per_sample for end, t in anchors)
        return int(offset + index * ns_per_sample)

    def input_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "frames": self._next_index,
            "chunks": self.chunks_captured,
            "audio_s": round(self._next_index / self.input_rate, 3),
            "chunk_frames_min": self.min_chunk_frames,
            "chunk_frames_max": self.max_chunk_frames,
            "read_errors": self.read_errors,
            "gaps": self.gaps,
        }
        if self.t_first_input_ns is not None and self.t_last_input_ns is not None:
            stats["wall_s"] = round((self.t_last_input_ns - self.t_first_input_ns) / _NS, 3)
        if self._rate_ref is not None and self.t_last_input_ns is not None:
            ref_frames, ref_t = self._rate_ref
            span = (self.t_last_input_ns - ref_t) / _NS
            if span > 0.5:
                stats["measured_rate_hz"] = round((self._next_index - ref_frames) / span, 1)
        with self._subs_lock:
            stats["subscriber_drops"] = {name: q.dropped for name, q in self._subs.items()}
        return stats

    # -- recording -------------------------------------------------------------
    @property
    def is_recording(self) -> bool:
        with self._rec_lock:
            return self._recording is not None

    def start_recording(self, name: str) -> None:
        rec = Recording(name, self.input_rate, self.input_channels, time.perf_counter_ns())
        with self._rec_lock:
            if self._recording is not None:
                raise RuntimeError("already recording")
            self._recording = rec
        self.events.log("record.started", t_ns=rec.t_request_ns, name=name)

    def stop_recording(self) -> Recording:
        t_ns = time.perf_counter_ns()
        with self._rec_lock:
            rec, self._recording = self._recording, None
        if rec is None:
            raise RuntimeError("not recording")
        rec.t_stop_ns = t_ns
        self.events.log("record.stopped", t_ns=t_ns, name=rec.name, frames=rec.num_frames,
                        duration_s=round(rec.duration_s, 3), start_index=rec.start_index)
        return rec

    # -- playback --------------------------------------------------------------
    def play(self, samples: np.ndarray, sample_rate: int, label: str = "",
             meta: dict[str, Any] | None = None) -> PlaybackHandle:
        """Queue a complete signal (any rate, mono or ``(n, ch)``).

        ``meta`` is copied into the ``playback.enqueued`` event (e.g. TTS text).
        """
        handle = self.player.enqueue(label, sample_rate, meta)
        handle.write(samples)
        handle.end()
        return handle

    def open_stream(self, sample_rate: int, label: str = "",
                    meta: dict[str, Any] | None = None) -> PlaybackHandle:
        """Queue a streamed signal: call ``write`` per chunk, then ``end``."""
        return self.player.enqueue(label, sample_rate, meta)

    def stop_playback(self, reason: str = "user") -> StopRequest:
        """Stop all robot output now; the returned request reports when the buffers were cleared."""
        return self.player.interrupt(reason)

    def add_playback_tap(self, tap: Any) -> None:
        """Phase 4.5: see what is played (speaking motion). Called in the player thread; must not block."""
        self.player.add_tap(tap)

    def remove_playback_tap(self, tap: Any) -> None:
        self.player.remove_tap(tap)

    def is_speaking(self) -> bool:
        return self.player.is_active()
