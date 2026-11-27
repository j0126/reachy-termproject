"""Cascaded speech-to-speech: VAD -> ASR -> LLM -> TTS -> robot speaker.

A0 (Phase 2 baseline, ``[cascaded] barge_in = false``) is half duplex:
IDLE -> LISTENING -> PROCESSING -> SPEAKING -> LISTENING. An utterance is
processed only if its speech started while LISTENING; anything that starts
during PROCESSING/SPEAKING is logged as ``utterance.ignored``. Each utterance id
is processed at most once.

A1 (Phase 4, ``barge_in = true``) keeps listening while it is busy:

* speech that starts while an answer is being prepared (PROCESSING, the robot
  is still silent) cancels that answer; the cancelled transcript is joined to
  the next utterance (``merge_cancelled_input``);
* speech confirmed by :class:`BargeInDetector` while the robot speaks stops the
  playback (local queue + robot buffer): SPEAKING -> INTERRUPTED -> LISTENING.
  The answer stays in the history as far as it was presumably heard, marked as
  interrupted (``interrupted_history``);
* results that arrive after an interruption are discarded at the next
  checkpoint (after ASR / LLM / TTS, before and during playback). A turn is
  interrupted at most once; the interrupting utterance becomes the next turn.

Both variants log suspected premature endpoints (speech again within
``premature_window_ms`` of an endpoint), user speech while busy, and the input
gaps (Wi-Fi) inside each utterance and its end-of-speech silence.

Failures (timeouts, HTTP 4xx/5xx, free-tier 429, empty results) end the turn
with ``turn.error`` (stage, kind, status code, attempts), an optional local
error beep, and a return to LISTENING. The conversation history only contains
exchanges the user actually heard; the model is never switched automatically.

Per turn the session gets: events in events.jsonl, one row in turns.csv,
``audio/turn_NNNN_user.wav`` (VAD segment) and ``_robot.wav`` (TTS), and an
updated conversation.json.
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..audio.buffers import DropOldestQueue
from ..audio.signals import silence, tone
from ..audio.wavio import write_wav
from ..config import AppConfig
from ..engines.base import Attempt, EngineError
from ..engines.vad import SAMPLE_RATE, BargeInDetector, FrameModel, Utterance, VadWorker
from ..evaluation.session import Session
from ..evaluation.tables import append_csv_row
from .history import ConversationHistory, Exchange
from .state import InteractionState, StateTracker

log = logging.getLogger(__name__)
S = InteractionState

HISTORY_POLICIES = ("heard_marked", "drop", "full_marked")
HEARD_MARK = "… (사용자가 끼어들어 여기서 멈춤)"
FULL_MARK = " (사용자가 끼어들어 끝까지 듣지 못함)"
#: an interrupted answer with at most this much audio left unheard is kept as complete
FULLY_HEARD_TAIL_S = 0.4


@dataclass
class TurnRecord:
    """Everything measured for one user utterance (times: s since session start)."""

    turn_id: int
    utterance_id: int
    variant: str = ""  # A0 | A1 | ... ([cascaded] variant)
    motion: str = "none"  # Phase 4.5 motion condition (MotionController.label(); "none" = no controller)
    tts_threads: int | None = None  # local TTS CPU threads ([tts] threads; 0 = all cores, the Phase 2-4 setting)
    status: str = "processing"  # processing | done | empty_transcript | error | stopped | cancelled | interrupted
    error_stage: str | None = None
    error_kind: str | None = None
    error_status: int | None = None
    error_message: str | None = None
    # VAD
    speech_start_est: float | None = None
    speech_end_est: float | None = None  # VAD-recognised end of speech (estimated PC arrival)
    endpoint: float | None = None  # endpoint decision
    vad_endpoint_delay_ms: float | None = None
    speech_s: float | None = None
    pauses_ms: list[int] = field(default_factory=list)
    clipping_rate: float | None = None
    input_peak_dbfs: float | None = None
    input_rms_dbfs: float | None = None
    # ASR
    asr_start: float | None = None
    asr_end: float | None = None
    asr_ms: float | None = None
    transcript: str | None = None
    asr_language_prob: float | None = None
    asr_no_speech_prob: float | None = None
    # LLM
    llm_start: float | None = None
    llm_end: float | None = None
    llm_ms: float | None = None
    llm_model: str | None = None
    llm_status: str | None = None
    llm_attempts: int | None = None
    llm_input_tokens: int | None = None
    llm_output_tokens: int | None = None
    llm_thought_tokens: int | None = None
    response: str | None = None
    # TTS
    tts_start: float | None = None
    tts_end: float | None = None
    tts_ms: float | None = None
    tts_model: str | None = None
    tts_attempts: int | None = None
    tts_audio_s: float | None = None
    tts_sample_rate: int | None = None
    # robot audio queue
    queue_enqueue: float | None = None
    queue_first_push: float | None = None
    queue_last_push: float | None = None
    output_est_start: float | None = None  # ESTIMATE (assumed output latency)
    output_est_end: float | None = None
    output_hint_ms: float | None = None
    playback_gain_db: float | None = None
    playback_clipped: int | None = None
    # derived (PC-measured unless *_est)
    endpoint_to_push_ms: float | None = None
    speech_end_to_push_ms: float | None = None
    response_latency_est_ms: float | None = None
    history_exchanges: int | None = None
    # Phase 4 turn-taking (A0 only logs, A1 acts)
    merged_text: str | None = None  # A1: cancelled turns' text + this transcript (what the LLM got)
    merged_from: list[int] = field(default_factory=list)  # A1: the cancelled turns joined into this one
    bargein_of_turn: int | None = None  # A1: this utterance interrupted that turn
    user_interrupt_start: float | None = None  # user speech start while busy (local VAD, est.; may be echo)
    user_interrupt_phase: str | None = None  # processing | speaking
    interrupted: bool = False
    interrupt_kind: str | None = None  # processing (answer not played yet) | speaking (barge-in detector)
    interrupt_stage: str | None = None  # where the answer was dropped: asr | llm | tts | playback
    bargein_detect: float | None = None  # interruption confirmed (PC time)
    bargein_detect_delay_ms: float | None = None  # confirmation - user speech start (est.)
    playback_stop_delay_ms: float | None = None  # stop request -> local queue + robot buffer cleared
    heard_est_s: float | None = None  # ESTIMATE: answer audio heard before the stop (output latency assumed)
    heard_ratio_est: float | None = None
    history_policy: str | None = None  # heard_marked | drop | full_marked | complete | cancelled
    premature_suspect: bool = False  # the user spoke again soon after this utterance's endpoint
    resume_after_endpoint_ms: float | None = None
    input_gaps_speech: int | None = None  # input arrival gaps (> gap_warn_ms) inside the speech
    input_gap_ms_speech: float | None = None
    input_gaps_silence: int | None = None  # ... inside the end-of-speech silence that made the endpoint
    input_gap_ms_silence: float | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["pauses_ms"] = " ".join(str(p) for p in self.pauses_ms)
        row["merged_from"] = " ".join(str(t) for t in self.merged_from)
        return row


@dataclass
class _Interrupt:
    """A confirmed user interruption of the turn in progress (A1)."""

    turn_id: int
    kind: str  # processing | speaking
    start_index: int  # first input sample of the interrupting speech
    t_detect_ns: int
    reason: str
    stop: Any = None  # StopRequest when the answer's playback had to be stopped


class _Interrupted(Exception):
    def __init__(self, info: _Interrupt, stage: str) -> None:
        super().__init__(f"turn {info.turn_id} interrupted ({info.kind}) at {stage}")
        self.info, self.stage = info, stage


def motion_condition(label: Callable[[], str] | None) -> str:
    """Motion condition right now (Phase 4.5); never raises into the conversation."""
    if label is None:
        return "none"
    try:
        return str(label())
    except Exception:  # noqa: BLE001
        return "unknown"


def motion_span(start: str, label: Callable[[], str] | None) -> str:
    """``start`` if the condition did not change during the turn, else ``start->end``."""
    end = motion_condition(label)
    return start if end == start else f"{start}->{end}"


def heard_prefix(text: str, ratio: float) -> str:
    """Words of ``text`` presumably heard when playback stopped after ``ratio`` of its audio.

    An estimate: every character is assumed to take the same time, and a word cut in the
    middle is dropped. ``""`` when not even the first word was heard.
    """
    text = text.strip()
    cut = int(len(text) * min(max(ratio, 0.0), 1.0))
    if cut >= len(text):
        return text
    head = text[:cut]
    if not text[cut].isspace():  # stopped inside a word
        head = head[: max(head.rfind(" "), head.rfind("\n")) + 1]
    return head.strip()


class CascadedPipeline:
    def __init__(
        self,
        audio: Any,
        cfg: AppConfig,
        session: Session,
        state: StateTracker,
        asr: Any,
        llm: Any,
        tts: Any,
        vad_model: FrameModel | None = None,
        history: ConversationHistory | None = None,
    ) -> None:
        cc = cfg.cascaded
        if cc.interrupted_history not in HISTORY_POLICIES:
            raise ValueError(f"[cascaded] interrupted_history must be one of {HISTORY_POLICIES}, "
                             f"got {cc.interrupted_history!r}")
        if cc.barge_in and cc.variant == "A0":
            raise ValueError('[cascaded] barge_in = true is not the A0 baseline: set variant (e.g. "A1") '
                             "so the results stay separable")
        self.audio = audio
        self.cfg = cfg
        self.session = session
        self.events = session.events
        self.state = state
        self.asr, self.llm, self.tts = asr, llm, tts
        self.history = history if history is not None else ConversationHistory(cfg.llm.max_history_exchanges)
        self.variant = cc.variant
        self.barge_in = cc.barge_in
        #: Phase 4.5: returns the motion condition for each turn row (MotionController.label)
        self.motion_label: Callable[[], str] | None = None
        self.tts_threads = cfg.tts.threads if cfg.tts.provider == "supertonic" else None
        self.vad = VadWorker(audio, cfg.vad, self.events, self._on_utterance, model=vad_model,
                             on_frame=self._on_frame if self.barge_in else None,
                             on_speech_start=self._on_speech_start)
        self.turns: list[TurnRecord] = []
        self.interrupted_responses: list[dict[str, Any]] = []
        self._turn_ids = itertools.count(1)
        # A1 can accept the interrupting utterance while the interrupted turn is still winding down
        self._queue_size = 4 if self.barge_in else 1
        self._queue: DropOldestQueue[Utterance] = DropOldestQueue(self._queue_size)
        self._seen: set[int] = set()
        self._busy = threading.Event()
        self._stop = threading.Event()
        self._listen_since_ns = 0
        self._worker: threading.Thread | None = None
        self._listeners: list[Callable[[str, dict[str, Any]], None]] = []
        self.running = False
        # Turn hand-off between the VAD thread and the worker; everything below is guarded by it.
        self._lock = threading.Lock()
        self._turn: TurnRecord | None = None  # turn being processed
        self._phase = ""  # "processing" | "speaking" while that turn can be interrupted
        self._item: Any = None  # its playback item
        self._interrupt: _Interrupt | None = None
        self._accept: tuple[int, int | None] | None = None  # A1: (input index, interrupted turn id)
        self._carry: list[tuple[int, str]] = []  # A1: (turn id, user text) of cancelled turns
        self._bargein_utts: dict[int, int | None] = {}  # utterance id -> turn it interrupted
        self._last_endpoint: tuple[int, int] | None = None  # (utterance id, speech end index), accepted only
        self._premature: dict[int, dict[str, Any]] = {}  # utterance id -> suspicion found before its turn began
        # VAD thread only
        self._detector = BargeInDetector(cc.bargein_threshold, cfg.vad.neg_threshold, cc.bargein_min_speech_ms,
                                         cc.bargein_reset_silence_ms)
        self._detector_turn: int | None = None

    # ------------------------------------------------------------ control
    def add_listener(self, fn: Callable[[str, dict[str, Any]], None]) -> None:
        """``fn(kind, payload)`` for GUIs: kinds are ``turn`` and ``history``."""
        self._listeners.append(fn)

    def _notify(self, kind: str, payload: dict[str, Any]) -> None:
        for fn in list(self._listeners):
            try:
                fn(kind, payload)
            except Exception:
                log.exception("pipeline listener failed")

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._queue = DropOldestQueue(self._queue_size)  # a stopped pipeline closed the previous one
        self._busy.clear()
        with self._lock:
            self._turn, self._phase, self._item, self._interrupt, self._accept = None, "", None, None, None
            self._carry, self._last_endpoint = [], None
            self._bargein_utts.clear()
            self._premature.clear()
        self.running = True
        self._enter_listening("cascaded start")
        self.vad.start()
        self._worker = threading.Thread(target=self._run, name="cascaded", daemon=True)
        self._worker.start()
        self.events.log("cascaded.started", vad=asdict(self.cfg.vad), cascaded=asdict(self.cfg.cascaded),
                        asr=getattr(self.asr, "load_info", None) or {"engine": getattr(self.asr, "name", "?")},
                        llm={"model": self.cfg.llm.model, **getattr(self.llm, "params", {})},
                        tts={"model": self.cfg.tts.model, **getattr(self.tts, "params", {})},
                        output_gain_db=self.cfg.audio.output_gain_db)

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self._stop.set()
        self.vad.stop()
        self._queue.close()
        self.audio.stop_playback("cascaded stop")
        if self._worker is not None:
            self._worker.join(timeout=60)
        self.state.set(S.IDLE, "cascaded stop")
        self.save_conversation()
        self.events.log("cascaded.stopped", turns=len(self.turns))

    def reset_history(self) -> int:
        dropped = self.history.reset()
        with self._lock:
            carried, self._carry = len(self._carry), []
        self.events.log("conversation.reset", dropped_exchanges=dropped, dropped_carry=carried)
        self.save_conversation()
        self._notify("history", {"transcript": []})
        return dropped

    def save_conversation(self) -> None:
        data = {
            "session_id": self.session.id,
            "variant": self.variant,
            "saved_at_t": self.session.t(),
            "resets": self.history.resets,
            "history": self.history.transcript(),
            "interrupted_responses": list(self.interrupted_responses),
            "turns": [t.to_row() for t in self.turns],
        }
        path = self.session.path("conversation.json")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    # --------------------------------------------------------- utterances
    def _enter_listening(self, reason: str) -> None:
        self._listen_since_ns = time.perf_counter_ns()
        self.state.set(S.LISTENING, reason)

    def _on_utterance(self, utt: Utterance) -> None:  # VAD thread
        if not self.running:
            return
        if utt.id in self._seen:
            self.events.log("utterance.duplicate", utterance_id=utt.id)
            return
        self._seen.add(utt.id)
        with self._lock:
            accept, how = self._accept, None
            if accept is not None and utt.speech_end_index > accept[0]:  # A1: speech that interrupted a turn
                self._accept, how = None, "interrupting speech"
                self._bargein_utts[utt.id] = accept[1]
            elif not self._busy.is_set() and utt.t_speech_start_detect_ns >= self._listen_since_ns:
                how = "listening"
            if how is not None:
                self._last_endpoint = (utt.id, utt.speech_end_index)
                self._busy.set()  # claim now: a second utterance cannot sneak in before the worker runs
                self._queue.put(utt)
            state = self.state.state.value
        if how is None:
            self.events.log("utterance.ignored", utterance_id=utt.id, reason="started while busy",
                            state=state, speech_s=round(utt.speech_duration_s, 2))
        elif accept is not None and how == "interrupting speech":
            self.events.log("utterance.accepted", utterance_id=utt.id, reason=how, bargein_of_turn=accept[1],
                            state=state, speech_s=round(utt.speech_duration_s, 2))

    def _on_speech_start(self, index: int, t_ns: int) -> None:  # VAD thread (all variants)
        if not self.running:
            return
        est = self.audio.estimate_arrival_ns(index)
        premature = None
        with self._lock:
            last, self._last_endpoint = self._last_endpoint, None  # only the first speech after an endpoint
            turn, phase = self._turn, self._phase
            if last is not None:
                after_ms = (index - last[1]) * 1000 / SAMPLE_RATE - self.cfg.vad.silence_timeout_ms
                if after_ms <= self.cfg.cascaded.premature_window_ms:
                    premature = {"utterance_id": last[0], "resume_after_endpoint_ms": round(after_ms)}
                    if turn is not None and turn.utterance_id == last[0]:
                        self._apply_premature(turn, premature)
                    else:  # its turn has not started yet
                        self._premature[last[0]] = premature
            busy = turn is not None and phase in ("processing", "speaking")
            if busy and turn.user_interrupt_start is None:
                turn.user_interrupt_start, turn.user_interrupt_phase = self._rel(est), phase
        if premature is not None:
            gaps, gap_ms = self.audio.gaps_between(last[1], index)
            self.events.log("endpoint.premature_suspect", t_ns=t_ns, **premature,
                            pause_ms=round((index - last[1]) * 1000 / SAMPLE_RATE), speech_start_index=index,
                            t_speech_start_est=self._rel(est), input_gaps=gaps, input_gap_ms=gap_ms,
                            state=self.state.state.value)
        if not busy:
            return
        if not self.barge_in:
            action = "ignored (A0)"
        else:
            action = "cancel answer" if phase == "processing" else "barge-in detector decides"
        self.events.log("user.speech_while_busy", t_ns=t_ns, turn_id=turn.turn_id, phase=phase, index=index,
                        t_speech_start_est=self._rel(est), action=action)
        if self.barge_in and phase == "processing":
            self._interrupt_turn("processing", index, t_ns, "speech before the answer was played")

    def _on_frame(self, prob: float, index: int, t_ns: int) -> None:  # VAD thread, every 32 ms frame (A1)
        with self._lock:
            turn = self._turn
            armed = turn is not None and self._phase == "speaking" and self._interrupt is None
        if not armed:
            return
        if self._detector_turn != turn.turn_id:  # a new answer started playing: count from here
            self._detector.reset()
            self._detector_turn = turn.turn_id
        if self._detector.push(prob, index):
            self._interrupt_turn("speaking", self._detector.start_index, time.perf_counter_ns(),
                                 "barge-in while the robot speaks", active_ms=round(self._detector.active_ms))

    def _interrupt_turn(self, kind: str, start_index: int, t_detect_ns: int, reason: str, **detail: Any) -> None:
        """Confirm a user interruption of the turn in progress (A1; at most once per turn)."""
        with self._lock:
            turn = self._turn
            if (not self.running or turn is None or self._interrupt is not None
                    or self._phase not in ("processing", "speaking")):
                return
            info = _Interrupt(turn.turn_id, kind, start_index, t_detect_ns, reason)
            self._interrupt = info
            self._accept = (start_index, turn.turn_id)  # the interrupting utterance is processed when it ends
            item = self._item
            if item is not None:  # under the lock: the worker cannot start playing after this point
                info.stop = self.audio.stop_playback("barge-in")
            self.state.set(S.INTERRUPTED, reason, turn_id=turn.turn_id)
        start_est = self.audio.estimate_arrival_ns(start_index)
        self.events.log("bargein.detected", t_ns=t_detect_ns, turn_id=turn.turn_id, interrupt_kind=kind, reason=reason,
                        start_index=start_index, t_speech_start_est=self._rel(start_est),
                        detect_delay_ms=None if start_est is None else round((t_detect_ns - start_est) / 1e6, 1),
                        playback_id=None if item is None else item.id, **detail)

    def _run(self) -> None:
        while not self._stop.is_set():
            utt = self._queue.get(timeout=0.2)
            if utt is None:
                continue
            try:
                self._process(utt)
            except Exception as exc:  # never let the worker die
                log.exception("turn crashed")
                self.events.log("turn.crash", utterance_id=utt.id, error=repr(exc))
            finally:
                self._turn_finished()

    def _turn_finished(self) -> None:
        last_turn = self.turns[-1].turn_id if self.turns else None
        overlap = None
        with self._lock:
            self._turn, self._phase, self._item, self._interrupt = None, "", None, None
            # A1: the user started answering during the last moments of the robot's speech (too short
            # for the barge-in detector): accept that utterance instead of ignoring it like A0.
            if self.barge_in and self._accept is None and self._detector_turn == last_turn \
                    and self._detector.start_index is not None:
                overlap = self.vad.speech_in_progress()
                if overlap is not None:
                    self._accept = (overlap, None)
            if self.running:
                self._enter_listening("turn finished")
            if len(self._queue) == 0:
                self._busy.clear()
        if overlap is not None:
            self.events.log("bargein.overlap_at_end", turn_id=last_turn, speech_start_index=overlap)

    # --------------------------------------------------------------- turn
    def _rel(self, t_ns: int | None) -> float | None:
        return None if t_ns is None else self.events.rel(t_ns)

    def _attempt_logger(self, turn: TurnRecord, stage: str) -> Callable[[Attempt], None]:
        def log_attempt(a: Attempt) -> None:
            self.events.log(f"{stage}.attempt", turn_id=turn.turn_id, attempt=a.number,
                            ms=round((a.t_end_ns - a.t_start_ns) / 1e6, 1), error=a.error)
        return log_attempt

    @staticmethod
    def _apply_premature(turn: TurnRecord, info: dict[str, Any] | None) -> None:
        if info is not None:
            turn.premature_suspect = True
            turn.resume_after_endpoint_ms = info["resume_after_endpoint_ms"]

    @staticmethod
    def _user_text(turn: TurnRecord) -> str:
        return turn.merged_text if turn.merged_text is not None else (turn.transcript or "")

    def _checkpoint(self, stage: str) -> None:
        """Drop the turn's results if the user interrupted it meanwhile (A1; no-op for A0)."""
        with self._lock:
            info = self._interrupt
        if info is not None:
            raise _Interrupted(info, stage)

    def _process(self, utt: Utterance) -> None:
        turn = TurnRecord(turn_id=next(self._turn_ids), utterance_id=utt.id, variant=self.variant,
                          motion=motion_condition(self.motion_label), tts_threads=self.tts_threads)
        self.turns.append(turn)
        turn.speech_start_est = self._rel(utt.t_speech_start_est_ns)
        turn.speech_end_est = self._rel(utt.t_speech_end_est_ns)
        turn.endpoint = self._rel(utt.t_endpoint_ns)
        if utt.t_speech_end_est_ns is not None:
            turn.vad_endpoint_delay_ms = round((utt.t_endpoint_ns - utt.t_speech_end_est_ns) / 1e6, 1)
        turn.speech_s = round(utt.speech_duration_s, 3)
        turn.pauses_ms = [round((b - a) * 1000 / utt.sample_rate) for a, b in utt.pauses]
        turn.clipping_rate = round(utt.clipping_rate, 6)
        turn.input_peak_dbfs, turn.input_rms_dbfs = round(utt.peak_dbfs, 2), round(utt.rms_dbfs, 2)
        silence_end = utt.speech_end_index + int(self.cfg.vad.silence_timeout_ms * utt.sample_rate / 1000)
        turn.input_gaps_speech, turn.input_gap_ms_speech = self.audio.gaps_between(
            utt.speech_start_index, utt.speech_end_index)
        turn.input_gaps_silence, turn.input_gap_ms_silence = self.audio.gaps_between(
            utt.speech_end_index, silence_end)
        with self._lock:
            self._turn, self._phase, self._item, self._interrupt = turn, "processing", None, None
            turn.bargein_of_turn = self._bargein_utts.pop(utt.id, None)
            self._apply_premature(turn, self._premature.pop(utt.id, None))
            self.state.set(S.PROCESSING, "endpoint", turn_id=turn.turn_id)
        self.events.log("turn.started", turn_id=turn.turn_id, utterance_id=utt.id, variant=self.variant,
                        bargein_of_turn=turn.bargein_of_turn)
        if self.barge_in:
            ongoing = self.vad.speech_in_progress()
            if ongoing is not None:  # the user is still talking: let them finish, answer everything at once
                self._interrupt_turn("processing", ongoing, time.perf_counter_ns(), "user still speaking")
        if self.cfg.cascaded.save_turn_audio:
            write_wav(self.session.audio_dir / f"turn_{turn.turn_id:04d}_user.wav", utt.audio, utt.sample_rate)
        self._notify("turn", turn.to_row())

        try:
            self._asr(turn, utt)
            self._merge_carry(turn)
            self._checkpoint("asr")
            if not self._user_text(turn):
                turn.status = "empty_transcript"
                self.events.log("turn.skipped", turn_id=turn.turn_id, reason="empty transcript")
                return
            self._llm(turn)
            try:
                self._checkpoint("llm")
                tts_result = self._tts(turn)
                self._checkpoint("tts")
                if self._stop.is_set():
                    raise EngineError("playback", "stopped", "pipeline stopped before playback")
                self._speak(turn, tts_result, utt)
            except _Interrupted:
                raise  # the interrupted-history policy decides what stays
            except Exception:
                if self.history.pop_last(turn.turn_id):  # the user never heard this answer
                    self.events.log("conversation.rollback", turn_id=turn.turn_id)
                raise
            turn.status = "done" if turn.status == "processing" else turn.status
        except _Interrupted as intr:
            self._on_interrupted(turn, intr)
        except EngineError as err:
            self._fail(turn, err)
        except Exception as exc:  # engine bug: report it like any other failure
            log.exception("turn %d failed", turn.turn_id)
            self._fail(turn, EngineError("turn", "other", f"{type(exc).__name__}: {exc}"))
        finally:
            with self._lock:  # from here on nothing else writes into this record
                self._turn, self._phase = None, ""
            turn.history_exchanges = len(self.history)
            turn.motion = motion_span(turn.motion, self.motion_label)
            self.events.log("turn.end", **turn.to_row())
            append_csv_row(self.session.path("turns.csv"), turn.to_row())
            self.save_conversation()
            self._notify("turn", turn.to_row())
            self._notify("history", {"transcript": self.history.transcript()})

    def _merge_carry(self, turn: TurnRecord) -> None:
        """A1: prepend what the user said in turns that were cancelled before they were answered."""
        with self._lock:
            carry, self._carry = self._carry, []
        if not carry:
            return
        turn.merged_from = [turn_id for turn_id, _ in carry]
        turn.merged_text = " ".join(t for t in [*(text for _, text in carry), turn.transcript or ""] if t)
        self.events.log("turn.merged", turn_id=turn.turn_id, merged_from=turn.merged_from,
                        transcript=turn.transcript, text=turn.merged_text)
        self._notify("turn", turn.to_row())

    def _asr(self, turn: TurnRecord, utt: Utterance) -> None:
        t0 = time.perf_counter_ns()
        self.events.log("asr.start", t_ns=t0, turn_id=turn.turn_id, audio_s=round(utt.audio.shape[0] / 16000, 3))
        result = self.asr.transcribe(utt.audio, utt.sample_rate)
        turn.asr_start, turn.asr_end = self._rel(result.t_start_ns), self._rel(result.t_end_ns)
        turn.asr_ms = round(result.duration_ms, 1)
        turn.transcript = result.text.strip()
        turn.asr_language_prob = result.language_probability
        turn.asr_no_speech_prob = result.no_speech_prob
        self.events.log("asr.end", t_ns=result.t_end_ns, turn_id=turn.turn_id, ms=turn.asr_ms,
                        text=turn.transcript, language=result.language,
                        language_prob=result.language_probability, no_speech_prob=result.no_speech_prob,
                        avg_logprob=result.avg_logprob, audio_s=round(result.audio_s, 3))
        self._notify("turn", turn.to_row())

    def _llm(self, turn: TurnRecord) -> None:
        text = self._user_text(turn)
        self.events.log("llm.start", turn_id=turn.turn_id, model=self.cfg.llm.model, params=self.llm.params,
                        history_exchanges=len(self.history), user_text=text)
        result = self.llm.respond(self.history, text, on_attempt=self._attempt_logger(turn, "llm"))
        trimmed = self.history.add(Exchange(turn.turn_id, text, result.text,
                                            result.user_step, result.model_steps))
        turn.llm_start, turn.llm_end = self._rel(result.t_start_ns), self._rel(result.t_end_ns)
        turn.llm_ms = round(result.duration_ms, 1)
        turn.llm_model, turn.llm_status, turn.llm_attempts = result.model, result.status, result.attempts
        turn.llm_input_tokens = result.usage.get("total_input_tokens")
        turn.llm_output_tokens = result.usage.get("total_output_tokens")
        turn.llm_thought_tokens = result.usage.get("total_thought_tokens")
        turn.response = result.text
        self.events.log("llm.end", t_ns=result.t_end_ns, turn_id=turn.turn_id, ms=turn.llm_ms,
                        model=result.model, status=result.status, attempts=result.attempts, usage=result.usage,
                        input_steps=result.input_steps, model_steps=len(result.model_steps),
                        history_trimmed=trimmed, response=result.text)
        self._notify("turn", turn.to_row())

    def _tts(self, turn: TurnRecord) -> Any:
        self.events.log("tts.start", turn_id=turn.turn_id, model=self.cfg.tts.model, params=self.tts.params,
                        chars=len(turn.response or ""))
        result = self.tts.synthesize(turn.response, on_attempt=self._attempt_logger(turn, "tts"))
        turn.tts_start, turn.tts_end = self._rel(result.t_start_ns), self._rel(result.t_end_ns)
        turn.tts_ms = round(result.generation_ms, 1)
        turn.tts_model, turn.tts_attempts = result.model, result.attempts
        turn.tts_audio_s, turn.tts_sample_rate = round(result.duration_s, 3), result.sample_rate
        extra: dict[str, Any] = {}
        spoken = getattr(result, "spoken_text", None)
        if spoken and spoken != result.text:  # e.g. numbers spelled out in Hangul for a local TTS
            extra["spoken_text"] = spoken
        if getattr(result, "info", None):
            extra["tts_info"] = result.info
        self.events.log("tts.end", t_ns=result.t_end_ns, turn_id=turn.turn_id, ms=turn.tts_ms,
                        audio_s=turn.tts_audio_s, sample_rate=result.sample_rate, mime_type=result.mime_type,
                        bytes=result.bytes_received, attempts=result.attempts, **extra)
        if self.cfg.cascaded.save_turn_audio:
            write_wav(self.session.audio_dir / f"turn_{turn.turn_id:04d}_robot.wav", result.audio,
                      result.sample_rate)
        self._notify("turn", turn.to_row())
        return result

    def _speak(self, turn: TurnRecord, tts_result: Any, utt: Utterance) -> None:
        with self._lock:  # an interruption either sees this item (and stops it) or prevents it
            if self._interrupt is not None:
                raise _Interrupted(self._interrupt, "tts")
            handle = self.audio.play(tts_result.audio, tts_result.sample_rate, label=f"turn_{turn.turn_id}",
                                     meta={"turn_id": turn.turn_id, "text": (turn.response or "")[:200]})
            self._item = handle.item
        item = handle.item
        if not handle.wait_first_push(timeout=10):
            self.audio.stop_playback("first push timeout")
            raise EngineError("playback", "timeout", "audio was not accepted by the robot within 10 s")
        with self._lock:
            info = self._interrupt
            if info is None and item.t_first_push_ns is not None:
                self._phase = "speaking"
                self.state.set(S.SPEAKING, "first audio pushed", t_ns=item.t_first_push_ns, turn_id=turn.turn_id)
        if info is not None:
            raise _Interrupted(info, "playback")
        if item.t_first_push_ns is None:  # failed / stopped before any push
            raise EngineError("playback", "other", f"playback {item.status}")
        turn.queue_enqueue = self._rel(item.t_enqueued_ns)
        turn.queue_first_push = self._rel(item.t_first_push_ns)
        turn.output_est_start = self._rel(item.t_output_start_est_ns)
        turn.output_hint_ms = self.cfg.audio.output_latency_hint_ms
        turn.endpoint_to_push_ms = round((item.t_first_push_ns - utt.t_endpoint_ns) / 1e6, 1)
        if utt.t_speech_end_est_ns is not None:
            turn.speech_end_to_push_ms = round((item.t_first_push_ns - utt.t_speech_end_est_ns) / 1e6, 1)
            turn.response_latency_est_ms = round(turn.speech_end_to_push_ms + turn.output_hint_ms, 1)
        self._notify("turn", turn.to_row())
        # Stay SPEAKING until everything was pushed AND its estimated audible end
        # (last push + assumed output latency) has passed; check for stop / barge-in meanwhile.
        while not item.pushed_all.wait(0.1):
            if self._stop.is_set() or self._interrupt is not None:
                break
        while not self._stop.is_set() and self._interrupt is None and item.t_output_end_est_ns is not None:
            remaining = (item.t_output_end_est_ns - time.perf_counter_ns()) / 1e9
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining))
        if self._stop.is_set() and item.status != "done":
            self.audio.stop_playback("cascaded stop")
            turn.status = "stopped"
        turn.queue_last_push = self._rel(item.t_last_push_ns)
        turn.output_est_end = self._rel(item.t_output_end_est_ns)
        turn.playback_gain_db = self.cfg.audio.output_gain_db
        turn.playback_clipped = item.clipped_samples
        self._checkpoint("playback")

    def _heard_s(self, turn: TurnRecord, item: Any, info: _Interrupt) -> float:
        """ESTIMATED seconds of the answer heard before playback stopped (assumed output latency)."""
        stop = info.stop
        if stop is not None:
            stop.done.wait(2.0)  # the player clears the robot buffer in its own thread
            turn.playback_stop_delay_ms = stop.stop_ms
        if item is None or item.t_output_start_est_ns is None:
            return 0.0
        turn.queue_last_push = self._rel(item.t_last_push_ns)
        t_stop = stop.t_request_ns if stop is not None else info.t_detect_ns
        pushed_s = item.frames_pushed / item.out_rate
        return max(0.0, min(pushed_s, (t_stop - item.t_output_start_est_ns) / 1e9))

    def _on_interrupted(self, turn: TurnRecord, intr: _Interrupted) -> None:
        """A1: record the interruption and keep in the history only what the user heard."""
        info, cc = intr.info, self.cfg.cascaded
        with self._lock:
            item = self._item
        turn.interrupted = True
        turn.interrupt_kind, turn.interrupt_stage = info.kind, intr.stage
        start_est = self.audio.estimate_arrival_ns(info.start_index)
        turn.user_interrupt_start, turn.user_interrupt_phase = self._rel(start_est), info.kind
        turn.bargein_detect = self._rel(info.t_detect_ns)
        if start_est is not None:
            turn.bargein_detect_delay_ms = round((info.t_detect_ns - start_est) / 1e6, 1)
        heard_s = self._heard_s(turn, item, info)
        response, total = turn.response or "", turn.tts_audio_s or 0.0
        heard_text = ""
        if item is not None and item.t_first_push_ns is not None and total > 0:
            turn.heard_est_s, turn.heard_ratio_est = round(heard_s, 3), round(min(1.0, heard_s / total), 3)
            heard_text = heard_prefix(response, heard_s / total)
        if heard_text and total - heard_s <= FULLY_HEARD_TAIL_S:
            policy = "complete"  # only the last moment was cut: keep the answer as it is
        elif heard_text:
            policy = cc.interrupted_history
        else:
            policy = "cancelled"  # nothing of the answer was heard
        turn.history_policy = policy
        turn.status = "cancelled" if policy == "cancelled" else "interrupted"
        user_text = self._user_text(turn)
        stored, carried = None, False
        if policy in ("cancelled", "drop"):
            if self.history.pop_last(turn.turn_id):
                self.events.log("conversation.rollback", turn_id=turn.turn_id, reason=policy)
            if cc.merge_cancelled_input and user_text:
                with self._lock:
                    self._carry.append((turn.turn_id, user_text))
                carried = True
        elif policy == "heard_marked":
            stored = heard_text.rstrip(" ,.!?…~") + HEARD_MARK
            self.history.replace_last(turn.turn_id, stored)
        elif policy == "full_marked":
            stored = response + FULL_MARK
            self.history.replace_last(turn.turn_id, stored)
        record = {"turn_id": turn.turn_id, "interrupt_kind": info.kind, "stage": intr.stage, "policy": policy,
                  "heard_est_s": turn.heard_est_s, "heard_ratio_est": turn.heard_ratio_est, "user_text": user_text,
                  "full_text": response, "heard_text": heard_text, "stored_text": stored, "carried": carried}
        self.events.log("turn.cancelled" if policy == "cancelled" else "conversation.interrupted_response",
                        **record)
        self.interrupted_responses.append(record)

    def _fail(self, turn: TurnRecord, err: EngineError) -> None:
        turn.error_stage, turn.error_kind, turn.error_status = err.stage, err.kind, err.status_code
        turn.error_message = str(err)[:300]
        with self._lock:
            self._phase = "error"  # no barge-in on the error beep
            info = self._interrupt
        if info is not None and err.kind != "stopped":  # the user had already interrupted: no beep
            self.events.log("turn.error", turn_id=turn.turn_id, after_interrupt=True, **err.to_dict())
            self._on_interrupted(turn, _Interrupted(info, err.stage))
            return
        if err.kind == "stopped":
            turn.status = "stopped"
            self.events.log("turn.stopped", turn_id=turn.turn_id, stage=err.stage)
            return
        turn.status = "error"
        self.events.log("turn.error", turn_id=turn.turn_id, **err.to_dict())
        self.state.set(S.ERROR, f"{err.stage} {err.kind}", turn_id=turn.turn_id)
        if self.cfg.cascaded.error_tone and not self._stop.is_set():
            beep = np.concatenate([tone(660.0, 0.12, 16000, 0.2), silence(0.06, 16000), tone(440.0, 0.12, 16000, 0.2)])
            self.audio.play(beep, 16000, label="error_tone").wait(timeout=3)
