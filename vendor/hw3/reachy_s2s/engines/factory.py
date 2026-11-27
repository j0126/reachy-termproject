"""Build the Phase 2 engines once per process (models stay loaded across sessions)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..config import AppConfig, LlmConfig, TtsConfig
from ..evaluation.event_logger import EventLogger
from .base import EngineError


def make_llm(cfg: LlmConfig) -> Any:
    """The text LLM selected by ``[llm] provider`` (not yet contacted)."""
    if cfg.provider == "openai_compat":
        from .llm_openai import OpenAICompatLLM

        return OpenAICompatLLM(cfg)
    if cfg.provider == "gemini":
        from .llm import GeminiLLM

        return GeminiLLM(cfg)
    raise EngineError("llm", "config", f"unknown [llm] provider {cfg.provider!r} (gemini | openai_compat)")


def make_tts(cfg: TtsConfig) -> Any:
    """The TTS selected by ``[tts] provider`` (not yet loaded / contacted)."""
    if cfg.provider == "supertonic":
        from .tts_supertonic import SupertonicTTS

        return SupertonicTTS(cfg)
    if cfg.provider == "gemini":
        from .tts import GeminiTTS

        return GeminiTTS(cfg)
    raise EngineError("tts", "config", f"unknown [tts] provider {cfg.provider!r} (gemini | supertonic)")


def gemini_key_needed(cfg: AppConfig, echo_llm: bool = False) -> list[str]:
    """Stages that call the Gemini API with this config (empty: no API key needed)."""
    stages = []
    if cfg.llm.provider == "gemini" and not echo_llm:
        stages.append("llm")
    if cfg.tts.provider == "gemini":
        stages.append("tts")
    return stages


@dataclass
class Engines:
    asr: Any
    llm: Any
    tts: Any
    info: dict[str, Any]


def build_engines(cfg: AppConfig, events: EventLogger | None = None, fake: bool = False,
                  echo_llm: bool = False) -> Engines:
    """Load ASR (timed: load / warm-up), create the LLM and TTS.

    ``[llm] provider`` selects the LLM: ``gemini`` (Interactions API) or
    ``openai_compat`` (e.g. the local llama.cpp server; it must already be
    running, see ``apps.llm_server``). ``[tts] provider`` selects ``gemini``
    or ``supertonic`` (local, loaded and warmed up here).
    ``fake=True`` returns the stand-in engines (no GPU / API) for dry runs.
    ``echo_llm=True`` keeps the real ASR/TTS but replaces the LLM with a
    deterministic echo (hardware tests while the cloud LLM is unavailable).
    Raises :class:`EngineError` (kind ``config``) when an API key is missing or
    the LLM server is unreachable, before any audio is processed.
    """
    if fake:
        from .fakes import FakeASR, FakeLLM, FakeTTS

        engines = Engines(FakeASR(texts=["(가짜 인식 결과)"]), FakeLLM(), FakeTTS(), {"fake": True})
    else:
        from .asr import WhisperASR
        from .gemini_client import api_key_present

        stages = gemini_key_needed(cfg, echo_llm)
        if stages and not api_key_present():
            raise EngineError(stages[0], "config", f"GEMINI_API_KEY is not set (needed by: {', '.join(stages)}; "
                                                   "copy .env.example to .env and fill it in)")
        t0 = time.perf_counter_ns()
        if echo_llm:
            from .fakes import EchoLLM

            llm: Any = EchoLLM()
            llm_info = {"model": "echo (stand-in, not a language model)", **llm.params}
        else:
            llm = make_llm(cfg.llm)
            llm_info = {"model": cfg.llm.model, **llm.params}
            if cfg.llm.provider == "openai_compat":
                llm_info["server"] = llm.check_server()  # fail fast, before the ASR model loads
            else:
                _ = llm.client
        tts = make_tts(cfg.tts)
        if cfg.tts.provider == "gemini":
            _ = tts.client  # create the client now (key check, SDK retries disabled)
            tts_load: dict[str, Any] = {}
        else:
            tts_load = tts.load()  # ONNX models + voice + one warm-up synthesis (timed)
        asr = WhisperASR(cfg.asr)
        asr_info = asr.load()
        if not echo_llm and cfg.llm.warmup:  # after the ASR load: a local LLM then shares the GPU as in a turn
            llm_info["warmup"] = llm.warmup()
        engines = Engines(asr, llm, tts, {
            "asr": asr_info,
            "llm": llm_info,
            "tts": {"model": cfg.tts.model, **tts.params, **({"load": tts_load} if tts_load else {})},
            "total_ms": round((time.perf_counter_ns() - t0) / 1e6, 1),
        })
    if events is not None:
        events.log("engines.ready", **engines.info)
    return engines
