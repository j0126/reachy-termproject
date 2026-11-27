"""Local TTS: Supertonic 3 (Supertone, ONNX Runtime on the CPU) -> float32 mono PCM.

Same contract as :class:`~reachy_s2s.engines.tts.GeminiTTS` (``synthesize``
-> ``TtsResult``), so the pipeline does not change. Chosen after the Gemini
free tier allowed 10 TTS requests/day (HTTP 429, 2026-10-16). Usage follows
Hugging Face speech-to-speech's ``supertonic_tts_handler.py`` (Apache-2.0):
``TTS(auto_download=True)``, ``get_voice_style``, ``synthesize(lang=...)``,
one warm-up call.

* Model files are pinned by the ``supertonic`` package (1.3.1: HF
  ``Supertone/supertonic-3`` @ 724fb5ab) and cached in ``~/.cache/supertonic3``.
* CPU only: the GPU already holds Whisper and the local LLM.
* The text is prepared for speech (``ko_tts_text.normalize_for_tts``:
  numbers in Hangul, fixed pronunciations); characters the model cannot read
  are dropped and reported. ``TtsResult.spoken_text`` is what was synthesized;
  the history and logs keep the original reply.
* A fixed ``[tts] gain_db`` is applied (constant within an experiment). If a
  reply would exceed -0.2 dBFS it is scaled down and ``limited_db`` is reported.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..config import TtsConfig
from .base import Attempt, EngineError
from .ko_tts_text import normalize_for_tts
from .tts import TtsResult

log = logging.getLogger(__name__)

PEAK_LIMIT = 10 ** (-0.2 / 20)


class SupertonicTTS:
    name = "supertonic"

    def __init__(self, cfg: TtsConfig, engine: Any = None) -> None:
        self.cfg = cfg
        self._engine = engine
        self._style: Any = None
        self.load_info: dict[str, Any] = {}

    @property
    def params(self) -> dict[str, Any]:
        return {"provider": "supertonic", "voice": self.cfg.voice, "lang": self.cfg.lang, "speed": self.cfg.speed,
                "total_steps": self.cfg.total_steps, "gain_db": self.cfg.gain_db, "threads": self.cfg.threads or None,
                "normalize_numbers": self.cfg.normalize_numbers,
                "sample_rate": getattr(self._engine, "sample_rate", None)}

    @property
    def engine(self) -> Any:
        if self._engine is None:
            self.load()
        return self._engine

    def load(self) -> dict[str, Any]:
        """Load the ONNX models and the voice, then warm up once. Returns timings for the event log."""
        t0 = time.perf_counter_ns()
        if self._engine is None:
            try:
                from supertonic import TTS
                from supertonic.config import get_model_revision
            except ImportError as exc:
                raise EngineError("tts", "config", "supertonic is not installed (uv pip install supertonic==1.3.1)") \
                    from exc
            threads = self.cfg.threads or None
            try:
                self._engine = TTS(model=self.cfg.model, auto_download=True, intra_op_num_threads=threads)
            except Exception as exc:  # noqa: BLE001 - bad model name, download failure, ORT error
                raise EngineError("tts", "config", f"cannot load {self.cfg.model}: {exc}") from exc
            revision = get_model_revision(self.cfg.model)
        else:
            revision = getattr(self._engine, "revision", None)
        try:
            self._style = self._engine.get_voice_style(voice_name=self.cfg.voice)
        except Exception as exc:  # noqa: BLE001
            raise EngineError("tts", "config", f"unknown Supertonic voice {self.cfg.voice!r}: {exc}") from exc
        load_ms = (time.perf_counter_ns() - t0) / 1e6
        t1 = time.perf_counter_ns()
        self._engine.synthesize("안녕하세요.", voice_style=self._style, lang=self.cfg.lang)
        self.load_info = {"engine": self.name, "model": self.cfg.model, "model_revision": revision,
                          "model_dir": str(getattr(self._engine, "model_dir", "")) or None,
                          "sample_rate": self._engine.sample_rate, "load_ms": round(load_ms, 1),
                          "warmup_ms": round((time.perf_counter_ns() - t1) / 1e6, 1)}
        return self.load_info

    def prepare_text(self, text: str) -> tuple[str, list[str]]:
        """Text actually synthesized + characters dropped because the model cannot read them."""
        spoken = normalize_for_tts(text, [tuple(p) for p in self.cfg.pronunciations]) \
            if self.cfg.normalize_numbers else text
        processor = getattr(getattr(self.engine, "model", None), "text_processor", None)
        dropped: list[str] = []
        if processor is not None:
            ok, dropped = processor.validate_text(spoken)
            if not ok:
                spoken = "".join(" " if c in dropped else c for c in spoken)
        return " ".join(spoken.split()), dropped

    def synthesize(self, text: str, on_attempt: Callable[[Attempt], None] | None = None) -> TtsResult:
        engine = self.engine
        if self._style is None:
            self._style = engine.get_voice_style(voice_name=self.cfg.voice)
        t0 = time.perf_counter_ns()
        spoken, dropped = self.prepare_text(text)
        try:
            if not spoken.strip():
                raise ValueError("nothing left to speak after text preparation")
            wav, _duration = engine.synthesize(spoken, voice_style=self._style, lang=self.cfg.lang,
                                               speed=self.cfg.speed, total_steps=self.cfg.total_steps)
            audio = np.asarray(wav, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                raise ValueError("synthesized audio is empty")
        except Exception as exc:  # noqa: BLE001 - reported as a TTS failure of this turn
            err = EngineError("tts", "empty" if isinstance(exc, ValueError) else "other",
                              f"supertonic: {type(exc).__name__}: {exc}")
            if on_attempt is not None:
                on_attempt(Attempt(1, t0, time.perf_counter_ns(), err.to_dict()))
            raise err from exc
        info: dict[str, Any] = {"spoken_chars": len(spoken)}
        if dropped:
            info["dropped_chars"] = dropped
            log.warning("TTS dropped characters the model cannot read: %s", dropped)
        if self.cfg.gain_db:
            audio = audio * np.float32(10 ** (self.cfg.gain_db / 20))
        peak = float(np.max(np.abs(audio)))
        if peak > PEAK_LIMIT:
            info["limited_db"] = round(20 * np.log10(PEAK_LIMIT / peak), 2)
            audio = audio * np.float32(PEAK_LIMIT / peak)
        t1 = time.perf_counter_ns()
        if on_attempt is not None:
            on_attempt(Attempt(1, t0, t1))
        return TtsResult(audio=audio, sample_rate=int(engine.sample_rate), text=text, model=self.cfg.model,
                         voice=self.cfg.voice, mime_type="pcm/f32 (local ONNX)", t_start_ns=t0, t_end_ns=t1,
                         attempts=1, bytes_received=0, spoken_text=spoken, info=info)
