"""TTS: Gemini (Interactions API) -> float32 mono PCM for the robot player.

Request: ``response_format={"type": "audio", "mime_type": "audio/wav",
"sample_rate": N}`` and ``generation_config={"speech_config": [{"voice": V}]}``.
WAV is requested (not ``audio/l16``) so the header states the encoding and
rate unambiguously. Generation time (request -> audio bytes) is measured
here; queueing / pushing to the robot is measured by the AudioManager.
"""

from __future__ import annotations

import base64
import io
import logging
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import TtsConfig
from .base import Attempt, EngineError, call_with_retries

log = logging.getLogger(__name__)


@dataclass
class TtsResult:
    audio: np.ndarray  # float32 mono
    sample_rate: int
    text: str
    model: str
    voice: str
    mime_type: str | None
    t_start_ns: int
    t_end_ns: int
    attempts: int = 1
    bytes_received: int = 0
    spoken_text: str | None = None  # text actually synthesized, if the engine rewrote it (numbers -> Hangul)
    info: dict[str, Any] = field(default_factory=dict)  # engine-specific facts for the log

    @property
    def duration_s(self) -> float:
        return self.audio.shape[0] / self.sample_rate if self.sample_rate else 0.0

    @property
    def generation_ms(self) -> float:
        return (self.t_end_ns - self.t_start_ns) / 1e6


def decode_audio(data: bytes, rate_hint: int, mime_type: str | None = None) -> tuple[np.ndarray, int, str]:
    """WAV (preferred) or headerless 16-bit PCM -> (float32 mono, rate, how)."""
    if data[:4] == b"RIFF":
        with wave.open(io.BytesIO(data), "rb") as wf:
            channels, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if width != 2:
            raise EngineError("tts", "other", f"unsupported WAV sample width {width}")
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)
        return pcm, rate, "wav"
    log.warning("TTS returned headerless audio (%s); assuming 16-bit little-endian PCM @ %d Hz", mime_type, rate_hint)
    pcm = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    return pcm, rate_hint, "raw-l16le-assumed"


class GeminiTTS:
    name = "gemini-tts"

    def __init__(self, cfg: TtsConfig, client: Any = None) -> None:
        self.cfg = cfg
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from .gemini_client import make_client

            self._client = make_client("tts")
        return self._client

    @property
    def params(self) -> dict[str, Any]:
        return {"provider": "gemini", "voice": self.cfg.voice, "sample_rate": self.cfg.sample_rate,
                "style": self.cfg.style or None,
                "mime_type": "audio/wav", "timeout_s": self.cfg.timeout_s, "max_retries": self.cfg.max_retries,
                "retry_on_timeout": self.cfg.retry_on_timeout}

    def synthesize(self, text: str, on_attempt: Callable[[Attempt], None] | None = None) -> TtsResult:
        cfg = self.cfg
        content: dict[str, Any] = {"type": "text", "text": text}
        if cfg.style:
            content["annotations"] = [{"type": "speech_metadata", "style": cfg.style}]

        def call() -> Any:
            return self.client.interactions.create(
                model=cfg.model,
                store=False,
                input=[{"type": "user_input", "content": [content]}],
                response_format={"type": "audio", "mime_type": "audio/wav", "sample_rate": cfg.sample_rate},
                generation_config={"speech_config": [{"voice": cfg.voice}]},
                timeout=cfg.timeout_s,
            )

        interaction, attempts = call_with_retries("tts", call, cfg.max_retries, on_attempt,
                                                  retry_timeouts=cfg.retry_on_timeout)
        audio_part = getattr(interaction, "output_audio", None)
        encoded = getattr(audio_part, "data", None) if audio_part is not None else None
        if not encoded:
            raise EngineError("tts", "empty", f"no audio in response (status={getattr(interaction, 'status', None)})",
                              attempts=len(attempts))
        data = base64.b64decode(encoded) if isinstance(encoded, str) else bytes(encoded)
        mime = getattr(audio_part, "mime_type", None)
        pcm, rate, how = decode_audio(data, getattr(audio_part, "sample_rate", None) or cfg.sample_rate, mime)
        if pcm.size == 0:
            raise EngineError("tts", "empty", "decoded audio is empty", attempts=len(attempts))
        return TtsResult(
            audio=pcm, sample_rate=rate, text=text, model=cfg.model, voice=cfg.voice,
            mime_type=f"{mime} ({how})", t_start_ns=attempts[0].t_start_ns, t_end_ns=attempts[-1].t_end_ns,
            attempts=len(attempts), bytes_received=len(data),
        )
