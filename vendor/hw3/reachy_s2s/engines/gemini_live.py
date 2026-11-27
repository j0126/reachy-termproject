"""Gemini Live (``bidiGenerateContent``) glue: session config, PCM conversion, server events.

No audio devices or robot code here (see ``conversation/cloud_pipeline.py``).
Facts checked on 2026-10-17 (docs/PHASE3_PLAN.md):

* input: raw 16-bit PCM, little-endian, 16 kHz, mono, MIME ``audio/pcm;rate=16000``;
  output: 24 kHz 16-bit PCM;
* automatic server VAD by default (silence ~800 ms, configurable), barge-in via
  ``server_content.interrupted``; ``turn_complete`` / ``generation_complete``;
  input / output transcription; ``GoAway``; session resumption handles (2 h);
* ``voice_activity`` (ACTIVITY_START / ACTIVITY_END + ``audio_offset``) is parsed
  by google-genai 2.31 but is not in the Live guide: it is reported only when
  the server really sends it. ``voice_activity_detection_signal`` is
  "allowlisted only" and only logged if it appears.

``receive()`` of the SDK ends after each ``turn_complete``; callers loop over it.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import CloudConfig
from .base import EngineError

log = logging.getLogger(__name__)

INPUT_RATE = 16000
OUTPUT_RATE = 24000
INPUT_MIME = f"audio/pcm;rate={INPUT_RATE}"
API_HOST = "generativelanguage.googleapis.com"

_START = {"high": "START_SENSITIVITY_HIGH", "low": "START_SENSITIVITY_LOW"}
_END = {"high": "END_SENSITIVITY_HIGH", "low": "END_SENSITIVITY_LOW"}
_HANDLING = {"interrupts": "START_OF_ACTIVITY_INTERRUPTS", "no_interruption": "NO_INTERRUPTION"}


# ------------------------------------------------------------------ PCM
def float_to_pcm16(x: np.ndarray) -> bytes:
    """float32 [-1, 1] mono -> 16-bit little-endian PCM bytes (clipped)."""
    x = np.clip(np.asarray(x, np.float32).reshape(-1), -1.0, 1.0)
    return np.round(x * 32767.0).astype("<i2").tobytes()


def pcm16_to_float(data: bytes) -> np.ndarray:
    n = len(data) // 2
    return np.frombuffer(data[: 2 * n], dtype="<i2").astype(np.float32) / 32768.0


def rate_from_mime(mime: str | None, default: int = OUTPUT_RATE) -> int:
    m = re.search(r"rate=(\d+)", mime or "")
    return int(m.group(1)) if m else default


# --------------------------------------------------------------- config
def validate_cloud_config(cfg: CloudConfig) -> None:
    problems = []
    if cfg.start_sensitivity not in ("", *_START):
        problems.append(f"start_sensitivity {cfg.start_sensitivity!r}")
    if cfg.end_sensitivity not in ("", *_END):
        problems.append(f"end_sensitivity {cfg.end_sensitivity!r}")
    if cfg.activity_handling not in _HANDLING:
        problems.append(f"activity_handling {cfg.activity_handling!r}")
    if cfg.uplink_while_speaking not in ("stream", "pause"):
        problems.append(f"uplink_while_speaking {cfg.uplink_while_speaking!r}")
    if cfg.uplink_chunk_ms <= 0:
        problems.append("uplink_chunk_ms must be > 0")
    if problems:
        raise EngineError("cloud", "config", "invalid [cloud] settings: " + ", ".join(problems))


def build_live_config(cfg: CloudConfig, system_prompt: str, resume_handle: str | None = None) -> Any:
    """``types.LiveConnectConfig`` for an audio-out Live session (``resume_handle`` keeps the context)."""
    from google.genai import types

    validate_cloud_config(cfg)
    vad: dict[str, Any] = {"silence_duration_ms": cfg.silence_duration_ms}
    if cfg.prefix_padding_ms:
        vad["prefix_padding_ms"] = cfg.prefix_padding_ms
    if cfg.start_sensitivity:
        vad["start_of_speech_sensitivity"] = _START[cfg.start_sensitivity]
    if cfg.end_sensitivity:
        vad["end_of_speech_sensitivity"] = _END[cfg.end_sensitivity]
    kwargs: dict[str, Any] = {
        "response_modalities": ["AUDIO"],
        "system_instruction": system_prompt.strip(),
        "realtime_input_config": types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(**vad),
            activity_handling=_HANDLING[cfg.activity_handling]),
    }
    if cfg.voice:
        kwargs["speech_config"] = types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=cfg.voice)))
    if cfg.input_transcription:
        kwargs["input_audio_transcription"] = types.AudioTranscriptionConfig()
    if cfg.output_transcription:
        kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()
    if cfg.session_resumption:
        kwargs["session_resumption"] = types.SessionResumptionConfig(handle=resume_handle)
    if cfg.context_compression:
        kwargs["context_window_compression"] = types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow())
    return types.LiveConnectConfig(**kwargs)


def config_summary(config: Any) -> dict[str, Any]:
    """Loggable view of a LiveConnectConfig (no system prompt text, no resumption handle)."""
    dump = config.model_dump(exclude_none=True, mode="json") if hasattr(config, "model_dump") else dict(config)
    dump.pop("system_instruction", None)
    if isinstance(dump.get("session_resumption"), dict):
        dump["session_resumption"] = {"resumed": bool(dump["session_resumption"].get("handle"))}
    return dump


# ---------------------------------------------------------------- events
@dataclass
class CloudEvent:
    kind: str
    t_ns: int
    data: dict[str, Any] = field(default_factory=dict)


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", None) or str(value)


def parse_server_message(msg: Any, t_ns: int | None = None) -> list[CloudEvent]:
    """One LiveServerMessage -> ordered events (audio, transcripts, turn state, session state)."""
    t = time.perf_counter_ns() if t_ns is None else t_ns
    out: list[CloudEvent] = []
    if getattr(msg, "setup_complete", None) is not None:
        out.append(CloudEvent("setup_complete", t))
    sc = getattr(msg, "server_content", None)
    if sc is not None:
        turn = getattr(sc, "model_turn", None)
        for part in (getattr(turn, "parts", None) or []) if turn is not None else []:
            blob = getattr(part, "inline_data", None)
            if blob is not None and getattr(blob, "data", None):
                out.append(CloudEvent("audio", t, {"pcm": bytes(blob.data),
                                                   "rate": rate_from_mime(getattr(blob, "mime_type", None))}))
            elif getattr(part, "text", None):
                out.append(CloudEvent("thought" if getattr(part, "thought", False) else "text", t,
                                      {"chars": len(part.text)}))
        for name, kind in (("input_transcription", "input_transcript"),
                           ("interim_input_transcription", "interim_input_transcript"),
                           ("output_transcription", "output_transcript")):
            tr = getattr(sc, name, None)
            if tr is not None and (getattr(tr, "text", None) or getattr(tr, "finished", None)):
                out.append(CloudEvent(kind, t, {"text": getattr(tr, "text", None) or "",
                                                "finished": bool(getattr(tr, "finished", False))}))
        if getattr(sc, "interrupted", None):
            out.append(CloudEvent("interrupted", t))
        if getattr(sc, "generation_complete", None):
            out.append(CloudEvent("generation_complete", t))
        if getattr(sc, "waiting_for_input", None):
            out.append(CloudEvent("waiting_for_input", t))
        if getattr(sc, "turn_complete", None):
            out.append(CloudEvent("turn_complete", t, {"reason": _enum_name(getattr(sc, "turn_complete_reason", None))}))
    va = getattr(msg, "voice_activity", None)
    if va is not None:
        out.append(CloudEvent("voice_activity", t, {"type": _enum_name(getattr(va, "voice_activity_type", None)),
                                                    "audio_offset": getattr(va, "audio_offset", None)}))
    sig = getattr(msg, "voice_activity_detection_signal", None)
    if sig is not None:
        out.append(CloudEvent("vad_signal", t, {"type": _enum_name(getattr(sig, "vad_signal_type", None))}))
    usage = getattr(msg, "usage_metadata", None)
    if usage is not None:
        keys = ("prompt_token_count", "response_token_count", "thoughts_token_count", "total_token_count",
                "cached_content_token_count")
        out.append(CloudEvent("usage", t, {k: getattr(usage, k) for k in keys if getattr(usage, k, None) is not None}))
    go_away = getattr(msg, "go_away", None)
    if go_away is not None:
        out.append(CloudEvent("go_away", t, {"time_left": getattr(go_away, "time_left", None)}))
    upd = getattr(msg, "session_resumption_update", None)
    if upd is not None:
        out.append(CloudEvent("resumption", t, {"handle": getattr(upd, "new_handle", None),
                                                "resumable": getattr(upd, "resumable", None)}))
    if getattr(msg, "tool_call", None) is not None:
        out.append(CloudEvent("tool_call", t))
    return out


# ---------------------------------------------------------------- errors
def classify_live_error(exc: BaseException) -> EngineError:
    """WebSocket close codes / handshake HTTP errors -> EngineError(stage "cloud")."""
    if isinstance(exc, EngineError):
        return exc
    code = getattr(exc, "code", None)
    details = str(getattr(exc, "details", None) or getattr(exc, "message", None) or exc)
    text = f"{type(exc).__name__}: {details}"[:500]
    low = details.lower()
    response = getattr(exc, "response", None)
    http_status = getattr(response, "status_code", None)
    if isinstance(http_status, int):  # handshake rejected (websockets InvalidStatus)
        code = http_status
    if "quota" in low or "resource_exhausted" in low or "rate limit" in low or code == 429:
        return EngineError("cloud", "rate_limit", text, code)
    if "api key" in low or "permission" in low or code in (401, 403):
        return EngineError("cloud", "config", text, code)
    if isinstance(exc, TimeoutError) or type(exc).__name__ in ("TimeoutError", "TimeoutException"):
        return EngineError("cloud", "timeout", text, code)
    if code in (1000, 1001):
        return EngineError("cloud", "closed", text, code)
    if code == 1006 or isinstance(exc, (ConnectionError, OSError)):
        return EngineError("cloud", "network", text, code)
    if isinstance(code, int) and (code == 1011 or 500 <= code < 600):
        return EngineError("cloud", "server", text, code)
    if isinstance(code, int) and 400 <= code < 500:
        return EngineError("cloud", "client", text, code)
    return EngineError("cloud", "other", text, code)


# -------------------------------------------------------------- connector
class GeminiLiveConnector:
    """``connect(model, config)`` -> async context manager yielding an SDK ``AsyncSession``."""

    name = "gemini-live"

    def __init__(self, client: Any = None) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from .gemini_client import make_client

            self._client = make_client("cloud")
        return self._client

    @contextlib.asynccontextmanager
    async def connect(self, model: str, config: Any) -> AsyncIterator[Any]:
        async with self.client.aio.live.connect(model=model, config=config) as session:
            yield session


def network_probe(timeout_s: float = 5.0) -> dict[str, Any]:
    """Network round trip to the API host (not a model request, no key sent). Never raises.

    ``connect_ms`` = DNS + TCP + TLS + first GET; ``rtt_ms`` = a second GET on the
    same connection (about one round trip + the host's tiny 404 handling).
    """
    import httpx

    t0 = time.perf_counter_ns()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            client.get(f"https://{API_HOST}/", follow_redirects=False)
            t1 = time.perf_counter_ns()
            r = client.get(f"https://{API_HOST}/", follow_redirects=False)
            t2 = time.perf_counter_ns()
        return {"ok": True, "connect_ms": round((t1 - t0) / 1e6, 1), "rtt_ms": round((t2 - t1) / 1e6, 1),
                "status": r.status_code}
    except Exception as exc:  # noqa: BLE001 - informational
        return {"ok": False, "elapsed_ms": round((time.perf_counter_ns() - t0) / 1e6, 1),
                "error": f"{type(exc).__name__}: {exc}"[:200]}
