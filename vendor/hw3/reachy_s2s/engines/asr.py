"""Speech recognition with faster-whisper (CTranslate2) on the local GPU.

Timing is reported in three separate parts (perf_counter_ns):
``load_ms`` (weights -> GPU), ``warmup_ms`` (first inference: CUDA/cuDNN
initialisation) and, per utterance, ``t_start``/``t_end`` around the decode
(including iteration of faster-whisper's lazy segment generator).

Windows: CTranslate2 4.x needs cuBLAS 12 and cuDNN 9. They come from the pip
wheels ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` (``.[cuda]`` extra) and
must be on the DLL search path before CTranslate2 loads them.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import AsrConfig
from .base import EngineError

log = logging.getLogger(__name__)
_DLL_HANDLES: list[Any] = []


def ensure_cuda_dlls() -> list[str]:
    """Put the pip-installed CUDA libraries on the DLL search path (Windows; idempotent)."""
    if sys.platform != "win32":
        return []
    added = []
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.submodule_search_locations:
            continue
        bin_dir = os.path.join(list(spec.submodule_search_locations)[0], "bin")
        if not os.path.isdir(bin_dir) or bin_dir in os.environ.get("PATH", ""):
            continue
        _DLL_HANDLES.append(os.add_dll_directory(bin_dir))
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        added.append(bin_dir)
    return added


def gpu_memory_used_mb() -> int | None:
    """Best-effort GPU memory in use (nvidia-smi), for VRAM diagnostics."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


@dataclass
class AsrResult:
    text: str
    language: str | None
    language_probability: float | None
    audio_s: float
    t_start_ns: int
    t_end_ns: int
    segments: list[dict[str, Any]] = field(default_factory=list)
    no_speech_prob: float | None = None
    avg_logprob: float | None = None

    @property
    def duration_ms(self) -> float:
        return (self.t_end_ns - self.t_start_ns) / 1e6

    @property
    def real_time_factor(self) -> float:
        return (self.duration_ms / 1000.0) / self.audio_s if self.audio_s else 0.0


class WhisperASR:
    name = "faster-whisper"

    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.model: Any = None
        self.load_info: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        """Load the model (and warm it up). Returns timings for the event log."""
        dlls = ensure_cuda_dlls() if self.cfg.device == "cuda" else []
        from faster_whisper import WhisperModel

        mem_before = gpu_memory_used_mb() if self.cfg.device == "cuda" else None
        t0 = time.perf_counter_ns()
        try:
            try:
                self.model = WhisperModel(self.cfg.model, device=self.cfg.device,
                                          compute_type=self.cfg.compute_type, local_files_only=True)
                source = "local cache"
            except Exception as exc:  # not cached yet -> download once (same model)
                log.info("model not in local cache (%s); downloading %s", exc, self.cfg.model)
                self.model = WhisperModel(self.cfg.model, device=self.cfg.device,
                                          compute_type=self.cfg.compute_type)
                source = "downloaded"
        except Exception as exc:
            raise EngineError("asr", "config", f"cannot load {self.cfg.model} on {self.cfg.device} "
                                               f"({self.cfg.compute_type}): {exc}") from exc
        load_ms = (time.perf_counter_ns() - t0) / 1e6
        warmup_ms = None
        if self.cfg.warmup:
            noise = (0.001 * np.random.default_rng(0).standard_normal(16000)).astype(np.float32)
            t1 = time.perf_counter_ns()
            self.transcribe(noise)
            warmup_ms = (time.perf_counter_ns() - t1) / 1e6
        mem_after = gpu_memory_used_mb() if self.cfg.device == "cuda" else None
        self.load_info = {
            "engine": self.name, "model": self.cfg.model, "device": self.cfg.device,
            "compute_type": self.cfg.compute_type, "source": source,
            "load_ms": round(load_ms, 1), "warmup_ms": None if warmup_ms is None else round(warmup_ms, 1),
            "gpu_mem_before_mb": mem_before, "gpu_mem_after_mb": mem_after, "cuda_dll_dirs_added": len(dlls),
        }
        return self.load_info

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> AsrResult:
        if self.model is None:
            raise EngineError("asr", "config", "ASR model not loaded")
        if sample_rate != 16000:
            raise EngineError("asr", "config", f"expected 16 kHz audio, got {sample_rate}")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        cfg = self.cfg
        t0 = time.perf_counter_ns()
        try:
            segments, info = self.model.transcribe(
                audio,
                language=cfg.language or None,
                beam_size=cfg.beam_size,
                vad_filter=False,  # our own VAD already cut the utterance
                condition_on_previous_text=False,
                without_timestamps=True,
                initial_prompt=cfg.initial_prompt or None,
                hotwords=cfg.hotwords or None,
            )
            segs = list(segments)  # decoding happens while iterating
        except Exception as exc:
            raise EngineError("asr", "other", f"{type(exc).__name__}: {exc}") from exc
        t1 = time.perf_counter_ns()
        text = "".join(s.text for s in segs).strip()
        return AsrResult(
            text=text,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            audio_s=audio.shape[0] / sample_rate,
            t_start_ns=t0,
            t_end_ns=t1,
            segments=[{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segs],
            no_speech_prob=max((s.no_speech_prob for s in segs), default=None),
            avg_logprob=float(np.mean([s.avg_logprob for s in segs])) if segs else None,
        )
