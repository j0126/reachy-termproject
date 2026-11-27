"""Voice activity detection: Silero VAD v6 (ONNX) + an utterance segmenter.

* :class:`SileroVadModel` - streaming wrapper around the ``silero_vad_v6.onnx``
  that ships with faster-whisper (inputs ``input[seq, 576]``, LSTM ``h``/``c``).
  Each call takes one 512-sample (32 ms @ 16 kHz) frame plus the previous
  frame's last 64 samples as context, exactly like the official wrapper.
* :class:`VadSegmenter` - pure state machine over frame probabilities
  (start confirmation, hysteresis, pauses, silence-timeout endpoint).
* :class:`VadWorker` - thread: mic subscription -> frames -> model ->
  segmenter -> :class:`Utterance` (audio cut from the AudioManager ring buffer).

Timing vocabulary (``perf_counter_ns``):
  ``t_speech_end_est`` - estimated PC arrival time of the last speech sample
  (what the VAD *recognised* as the end of speech; 32 ms frame resolution).
  ``t_endpoint`` - when the segmenter decided the turn is over
  (= speech end + silence timeout + processing). The user's *true* acoustic
  end of speech is earlier by the unknown mic-path latency.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..audio.convert import clipping_rate, peak_dbfs, rms_dbfs, to_mono
from ..config import VadConfig
from ..evaluation.event_logger import EventLogger

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME = 512
CONTEXT = 64


def default_model_path() -> Path:
    from faster_whisper.utils import get_assets_path

    return Path(get_assets_path()) / "silero_vad_v6.onnx"


class FrameModel(Protocol):
    def reset(self) -> None: ...
    def __call__(self, frame: np.ndarray) -> float: ...


class SileroVadModel:
    def __init__(self, path: str | Path | None = None) -> None:
        import onnxruntime

        self.path = Path(path) if path is not None else default_model_path()
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(str(self.path), providers=["CPUExecutionProvider"],
                                                     sess_options=opts)
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), np.float32)
        self._c = np.zeros((1, 1, 128), np.float32)
        self._context = np.zeros(CONTEXT, np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (FRAME,):
            raise ValueError(f"Silero expects {FRAME} samples, got {frame.shape}")
        x = np.concatenate([self._context, frame])[None, :]
        prob, self._h, self._c = self._session.run(None, {"input": x, "h": self._h, "c": self._c})
        self._context = frame[-CONTEXT:].copy()
        return float(np.asarray(prob).reshape(-1)[0])

    def probs(self, audio: np.ndarray) -> np.ndarray:
        """Batch over whole frames of ``audio`` (fresh state; for analysis/tests)."""
        audio = np.asarray(audio, dtype=np.float32)
        n = audio.shape[0] // FRAME
        frames = audio[: n * FRAME].reshape(n, FRAME)
        context = np.vstack([np.zeros((1, CONTEXT), np.float32), frames[:-1, -CONTEXT:]]) if n else frames
        x = np.concatenate([context, frames], axis=1)
        h = np.zeros((1, 1, 128), np.float32)
        c = np.zeros((1, 1, 128), np.float32)
        prob, _, _ = self._session.run(None, {"input": x, "h": h, "c": c})
        return np.asarray(prob, dtype=np.float32).reshape(-1)


# ----------------------------------------------------------------- segmenter
@dataclass
class SegEvent:
    kind: str  # speech_start | pause | speech_end
    index: int  # absolute sample index (start / end of speech, start of pause)
    end_index: int | None = None  # pause end
    forced: bool = False


class VadSegmenter:
    """Frame-probability state machine (no model, no threads)."""

    def __init__(self, cfg: VadConfig, sample_rate: int = SAMPLE_RATE) -> None:
        self.cfg = cfg
        ms = sample_rate / 1000.0
        self._min_speech = cfg.min_speech_ms * ms
        self._silence_timeout = cfg.silence_timeout_ms * ms
        self._min_pause = cfg.min_pause_ms * ms
        self._max_utt = cfg.max_utterance_s * sample_rate
        self.reset()

    def reset(self) -> None:
        self.in_speech = False
        self.confirmed = False
        self.speech_start: int | None = None
        self.speech_samples = 0
        self.silence_start: int | None = None
        self.pauses: list[tuple[int, int]] = []

    def push(self, prob: float, start: int, end: int) -> list[SegEvent]:
        """Feed one frame ``[start, end)`` with speech probability ``prob``."""
        cfg = self.cfg
        events: list[SegEvent] = []
        if not self.in_speech:
            if prob >= cfg.threshold:
                self.in_speech, self.speech_start, self.speech_samples = True, start, end - start
                self.silence_start, self.pauses, self.confirmed = None, [], False
                if self.speech_samples >= self._min_speech:
                    self.confirmed = True
                    events.append(SegEvent("speech_start", start))
            return events

        speechy = prob >= cfg.threshold or (self.silence_start is None and prob >= cfg.neg_threshold)
        if speechy:
            if self.silence_start is not None:
                if start - self.silence_start >= self._min_pause and self.confirmed:
                    self.pauses.append((self.silence_start, start))
                    events.append(SegEvent("pause", self.silence_start, start))
                self.silence_start = None
            self.speech_samples += end - start
            if not self.confirmed and self.speech_samples >= self._min_speech:
                self.confirmed = True
                events.append(SegEvent("speech_start", self.speech_start))
        elif self.silence_start is None:
            self.silence_start = start

        if self.silence_start is not None and end - self.silence_start >= self._silence_timeout:
            events.extend(self._finish(self.silence_start, forced=False))
        elif self.speech_start is not None and end - self.speech_start >= self._max_utt:
            events.extend(self._finish(end if self.silence_start is None else self.silence_start, forced=True))
        return events

    def _finish(self, speech_end: int, forced: bool) -> list[SegEvent]:
        confirmed = self.confirmed
        self.in_speech, self.confirmed, self.silence_start = False, False, None
        if not confirmed:  # too short: a click or a cough
            self.speech_start = None
            return []
        return [SegEvent("speech_end", speech_end, forced=forced)]


@dataclass
class OfflineSegment:
    start: int  # first speech sample
    end: int  # last speech sample (silence start)
    endpoint: int  # sample at which the silence timeout fired (end + timeout), or the track end
    pauses: list[tuple[int, int]] = field(default_factory=list)
    closed: bool = True  # False: the track ended before the silence timeout


def segment_probs(probs: np.ndarray, cfg: VadConfig, first_index: int = 0) -> list[OfflineSegment]:
    """Run the live segmenter over a recorded probability track (same rules as :class:`VadWorker`)."""
    seg = VadSegmenter(cfg)
    out: list[OfflineSegment] = []
    start = None
    for i, p in enumerate(probs):
        f_start = first_index + i * FRAME
        for ev in seg.push(float(p), f_start, f_start + FRAME):
            if ev.kind == "speech_start":
                start = ev.index
            elif ev.kind == "speech_end" and start is not None:
                out.append(OfflineSegment(start, ev.index, f_start + FRAME, list(seg.pauses)))
                start = None
    if start is not None and seg.confirmed:
        end = seg.silence_start if seg.silence_start is not None else first_index + len(probs) * FRAME
        out.append(OfflineSegment(start, end, first_index + len(probs) * FRAME, list(seg.pauses), closed=False))
    return out


# ------------------------------------------------------------- barge-in
class BargeInDetector:
    """Confirms a user interruption from per-frame speech probabilities (Phase 4, A1).

    Used live (fed by :class:`VadWorker` while the robot speaks or prepares an answer)
    and offline (double-talk / pause analysis) with the same rule:

    * frames with ``prob >= threshold`` add to the active speech time;
    * a run of frames below ``neg_threshold`` lasting ``reset_silence_ms`` resets it;
    * it fires once when active speech reaches ``min_speech_ms`` (HF speech-to-speech
      uses 384 ms of active speech for the same decision), then re-arms after the reset
      silence. ``start_index`` is the first speech frame of the run that fired.
    """

    def __init__(self, threshold: float = 0.6, neg_threshold: float = 0.35, min_speech_ms: float = 400.0,
                 reset_silence_ms: float = 300.0, frame_ms: float = FRAME * 1000 / SAMPLE_RATE) -> None:
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.min_speech_ms = min_speech_ms
        self.reset_silence_ms = reset_silence_ms
        self.frame_ms = frame_ms
        self.reset()

    def reset(self) -> None:
        self.active_ms = 0.0
        self.silence_ms = 0.0
        self.fired = False
        self.start_index: int | None = None

    def push(self, prob: float, start_index: int) -> bool:
        """Feed one frame; True exactly when the interruption is confirmed."""
        if prob >= self.threshold:
            if self.start_index is None:
                self.start_index = start_index
            self.active_ms += self.frame_ms
            self.silence_ms = 0.0
        elif prob < self.neg_threshold:
            self.silence_ms += self.frame_ms
            if self.silence_ms >= self.reset_silence_ms:
                self.active_ms, self.fired, self.start_index = 0.0, False, None
        if not self.fired and self.active_ms >= self.min_speech_ms:
            self.fired = True
            return True
        return False

    def scan(self, probs: np.ndarray, first_index: int = 0) -> list[tuple[int, int]]:
        """Offline: [(start_index, fire_index)] for a whole probability track (32 ms frames)."""
        self.reset()
        hits = []
        for i, p in enumerate(probs):
            index = first_index + i * FRAME
            if self.push(float(p), index):
                hits.append((self.start_index, index + FRAME))
        return hits


# ----------------------------------------------------------------- worker
@dataclass
class Utterance:
    id: int
    sample_rate: int
    audio: np.ndarray  # mono float32, pre-roll + speech + post-roll
    audio_start_index: int
    speech_start_index: int
    speech_end_index: int
    t_speech_start_detect_ns: int
    t_speech_start_est_ns: int | None
    t_speech_end_est_ns: int | None
    t_endpoint_ns: int
    pauses: list[tuple[int, int]] = field(default_factory=list)
    forced: bool = False
    max_prob: float = 0.0
    clipping_rate: float = 0.0
    peak_dbfs: float = -240.0
    rms_dbfs: float = -240.0

    @property
    def speech_duration_s(self) -> float:
        return (self.speech_end_index - self.speech_start_index) / self.sample_rate

    def summary(self, rel: Callable[[int], float]) -> dict[str, Any]:
        return {
            "utterance_id": self.id,
            "speech_start_index": self.speech_start_index,
            "speech_end_index": self.speech_end_index,
            "speech_duration_s": round(self.speech_duration_s, 3),
            "t_speech_start_est": None if self.t_speech_start_est_ns is None else rel(self.t_speech_start_est_ns),
            "t_speech_end_est": None if self.t_speech_end_est_ns is None else rel(self.t_speech_end_est_ns),
            "t_endpoint": rel(self.t_endpoint_ns),
            "pauses_ms": [round((b - a) * 1000 / self.sample_rate) for a, b in self.pauses],
            "forced": self.forced,
            "max_prob": round(self.max_prob, 3),
            "clipping_rate": round(self.clipping_rate, 6),
            "peak_dbfs": round(self.peak_dbfs, 2),
            "rms_dbfs": round(self.rms_dbfs, 2),
        }


class VadWorker:
    """Runs the VAD on the live mic stream in its own thread."""

    def __init__(self, audio: Any, cfg: VadConfig, events: EventLogger,
                 on_utterance: Callable[[Utterance], None], model: FrameModel | None = None,
                 channel: str | int | None = None,
                 on_frame: Callable[[float, int, int], None] | None = None,
                 on_speech_start: Callable[[int, int], None] | None = None) -> None:
        if audio.input_rate != SAMPLE_RATE:
            raise ValueError(f"VAD needs {SAMPLE_RATE} Hz input, got {audio.input_rate}")
        self.audio = audio
        self.cfg = cfg
        self.events = events
        self.on_utterance = on_utterance
        self.on_frame = on_frame  # (prob, frame_start_index, t_ns) per 32 ms frame (barge-in detection)
        self.on_speech_start = on_speech_start  # (speech start index, t_ns) when a speech run is confirmed
        self.model = model if model is not None else SileroVadModel()
        self.channel = audio.mic_channel if channel is None else channel
        self.segmenter = VadSegmenter(cfg)
        self.last_prob = 0.0
        self._queue = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._buf = np.zeros(0, np.float32)
        self._buf_start: int | None = None  # absolute index of _buf[0]
        self._utt_ids = 0
        self._t_start_detect = 0
        self._max_prob = 0.0
        self.frames = 0
        self.frame_time_ms = 0.0

    def start(self) -> None:
        self.model.reset()
        self.segmenter.reset()
        self._buf, self._buf_start = np.zeros(0, np.float32), None
        self._queue = self.audio.subscribe("vad")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vad", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._queue is not None:
            self.audio.unsubscribe("vad")
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None

    def speech_in_progress(self) -> int | None:
        """Start index of the confirmed speech run being segmented right now, else ``None``.

        Read from other threads without a lock: the answer may be one frame (32 ms) old.
        """
        seg = self.segmenter
        start = seg.speech_start
        return start if seg.in_speech and seg.confirmed and start is not None else None

    def _run(self) -> None:
        while not self._stop.is_set():
            chunk = self._queue.get(timeout=0.1)
            if chunk is None:
                continue
            self._feed(to_mono(chunk.samples, self.channel), chunk.start_index)

    def _feed(self, mono: np.ndarray, start_index: int) -> None:
        if self._buf_start is None:
            self._buf_start = start_index
        expected = self._buf_start + self._buf.shape[0]
        if start_index != expected:  # the subscription dropped chunks (consumer too slow)
            self.events.log("vad.input_gap", expected_index=expected, got_index=start_index)
            self.model.reset()
            self.segmenter.reset()
            self._buf, self._buf_start = np.zeros(0, np.float32), start_index
        self._buf = np.concatenate([self._buf, mono]) if self._buf.size else mono.copy()
        while self._buf.shape[0] >= FRAME:
            frame, self._buf = self._buf[:FRAME], self._buf[FRAME:]
            f_start = self._buf_start
            self._buf_start += FRAME
            t0 = time.perf_counter_ns()
            prob = self.model(frame)
            self.frame_time_ms = (time.perf_counter_ns() - t0) / 1e6
            self.frames += 1
            self.last_prob = prob
            was_in_speech = self.segmenter.in_speech
            events = self.segmenter.push(prob, f_start, f_start + FRAME)
            if was_in_speech:
                self._max_prob = max(self._max_prob, prob)
            elif self.segmenter.in_speech:  # a new (possibly unconfirmed) speech run
                self._max_prob = prob
            for event in events:
                self._handle(event)
            if self.on_frame is not None:
                try:
                    self.on_frame(prob, f_start, t0)
                except Exception:
                    log.exception("frame callback failed")

    def _handle(self, event: SegEvent) -> None:
        now = time.perf_counter_ns()
        if event.kind == "speech_start":
            self._t_start_detect = now
            est = self.audio.estimate_arrival_ns(event.index)
            self.events.log("vad.speech_start", t_ns=now, index=event.index,
                            t_speech_start_est=None if est is None else self.events.rel(est))
            if self.on_speech_start is not None:
                try:
                    self.on_speech_start(event.index, now)
                except Exception:
                    log.exception("speech-start callback failed")
        elif event.kind == "pause":
            self.events.log("vad.pause", t_ns=now, start_index=event.index, end_index=event.end_index,
                            pause_ms=round((event.end_index - event.index) / SAMPLE_RATE * 1000))
        elif event.kind == "speech_end":
            self._emit_utterance(event, now)

    def _emit_utterance(self, event: SegEvent, t_endpoint: int) -> None:
        seg = self.segmenter
        start = seg.speech_start if seg.speech_start is not None else event.index
        pre = int(self.cfg.pre_roll_ms * SAMPLE_RATE / 1000)
        post = int(self.cfg.post_roll_ms * SAMPLE_RATE / 1000)
        audio_start = max(0, start - pre)
        frames = self.audio.ring.read(audio_start, event.index + post)
        audio_start = max(audio_start, self.audio.ring.start_index)
        mono = to_mono(frames, self.channel) if frames.size else np.zeros(0, np.float32)
        speech_part = mono[start - audio_start: event.index - audio_start] if mono.size else mono
        self._utt_ids += 1
        t_start_est = self.audio.estimate_arrival_ns(start)
        t_end_est = self.audio.estimate_arrival_ns(event.index)
        utt = Utterance(
            id=self._utt_ids, sample_rate=SAMPLE_RATE, audio=mono, audio_start_index=audio_start,
            speech_start_index=start, speech_end_index=event.index,
            t_speech_start_detect_ns=self._t_start_detect, t_speech_start_est_ns=t_start_est,
            t_speech_end_est_ns=t_end_est, t_endpoint_ns=t_endpoint, pauses=list(seg.pauses),
            forced=event.forced, max_prob=self._max_prob,
            clipping_rate=clipping_rate(speech_part), peak_dbfs=peak_dbfs(speech_part),
            rms_dbfs=rms_dbfs(speech_part),
        )
        self._max_prob = 0.0
        seg.speech_start = None
        self.events.log("vad.speech_end", t_ns=t_endpoint, **utt.summary(self.events.rel),
                        endpoint_delay_ms=None if t_end_est is None else round((t_endpoint - t_end_est) / 1e6, 1))
        try:
            self.on_utterance(utt)
        except Exception:
            log.exception("utterance callback failed")
