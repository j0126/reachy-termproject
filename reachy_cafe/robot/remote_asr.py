"""음성 인식을 다른 PC(데스크톱 GPU)에 맡긴다 — 현장 두뇌 PC에 NVIDIA GPU가 없을 때의 예비 경로.

11/27 실측: GPU 없는 노트북(Core Ultra 7 155H) CPU로는 large-v3-turbo가 말 한 번에 6–14 s, small이 약 2 s였다
(데스크톱 GPU 0.2–0.5 s). [speech] asr = "remote"이면 손님 발화(16 kHz mono)를 데스크톱에서 돌리는
`python -m reachy_cafe.asr_server`로 보내 글자를 받는다. 데스크톱이 응답하지 않으면 이 PC CPU의 작은 모델
([speech] asr_fallback, 기본 small)로 대신하고, asr_retry_s 동안은 데스크톱을 건너뛰었다가 다시 시도한다
(매 발화마다 시간 초과를 기다리지 않게).

요청: POST {asr_url}/transcribe  본문 = int16 PCM, 쿼리 = sample_rate·language·beam_size·initial_prompt·hotwords
REACHY_ASR_TOKEN 환경 변수를 정하면 X-Token 헤더로 보낸다 (서버도 같은 값일 때만 받음).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from dataclasses import replace
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)
TOKEN_ENV = "REACHY_ASR_TOKEN"


def encode_pcm(audio: np.ndarray) -> bytes:
    return (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def decode_pcm(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


class RemoteASR:
    """HW3 WhisperASR와 같은 모양(cfg·load·transcribe)이라 KioskSpeech가 그대로 쓴다."""

    name = "remote"

    def __init__(self, cfg: Any, url: str, *, timeout_s: float = 3.0, retry_s: float = 20.0, fallback: Any = None,
                 token: str | None = None, clock: Callable[[], float] = time.monotonic,
                 opener: Callable[..., Any] | None = None):
        self.cfg = cfg  # 언어·초기 프롬프트·beam은 KioskSpeech가 바꾸고, 요청마다 서버로 보낸다
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.retry_s = retry_s
        self.fallback = fallback  # 이 PC의 WhisperASR(CPU) 또는 None
        self.token = os.environ.get(TOKEN_ENV, "") if token is None else token
        self.clock = clock
        self._open = opener or urllib.request.urlopen
        self.model: Any = None  # WhisperASR와 같은 속성 (KioskSpeech가 확인할 수 있음)
        self.server: dict[str, Any] = {}
        self.down_until = 0.0
        self.last_source: str | None = None  # "remote" | "fallback"
        self.last_error: str | None = None   # 이번 transcribe에서 데스크톱 요청이 실패했을 때만

    def _request(self, path: str, data: bytes | None = None, params: dict[str, Any] | None = None,
                 timeout: float | None = None) -> dict[str, Any]:
        url = self.url + path + ("?" + urllib.parse.urlencode(params) if params else "")
        req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
        if data is not None:
            req.add_header("Content-Type", "application/octet-stream")
        if self.token:
            req.add_header("X-Token", self.token)
        with self._open(req, timeout=timeout or self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def load(self) -> dict[str, Any]:
        """데스크톱이 지금 안 떠 있어도 실패로 치지 않는다 (나중에 켜지면 그때부터 씀). 대비책 모델은 미리 올린다."""
        info: dict[str, Any] = {"model": f"remote {self.url}", "device": "remote", "compute_type": ""}
        try:
            self.server = self._request("/health", timeout=min(self.timeout_s, 3.0))
            info["model"] = f"remote {self.server.get('model')} ({self.server.get('device')}) @ {self.url}"
            info["compute_type"] = str(self.server.get("compute_type") or "")
        except Exception as exc:  # noqa: BLE001 - 연결 거부·시간 초과·HTTP 오류 모두 "지금은 안 됨"
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.down_until = self.clock() + self.retry_s
            log.warning("remote ASR %s not reachable yet: %s", self.url, exc)
        if self.fallback is not None:
            try:
                self.fallback.load()
                info["fallback"] = f"{self.fallback.cfg.model} ({self.fallback.cfg.device})"
            except Exception as exc:  # noqa: BLE001
                log.warning("local fallback ASR could not load: %s", exc)
                self.fallback = None
        return info

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Any:
        from reachy_s2s.engines.asr import AsrResult

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        self.last_error = None
        if self.clock() >= self.down_until:
            cfg = self.cfg
            params = {"sample_rate": sample_rate, "language": cfg.language or "", "beam_size": cfg.beam_size,
                      "initial_prompt": cfg.initial_prompt or "", "hotwords": cfg.hotwords or ""}
            t0 = time.perf_counter_ns()
            try:
                data = self._request("/transcribe", data=encode_pcm(audio), params=params)
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.down_until = self.clock() + self.retry_s
            else:
                self.last_source = "remote"
                return AsrResult(text=str(data.get("text") or ""), language=data.get("language"),
                                 language_probability=data.get("language_probability"),
                                 audio_s=audio.shape[0] / sample_rate, t_start_ns=t0, t_end_ns=time.perf_counter_ns(),
                                 segments=list(data.get("segments") or []), no_speech_prob=data.get("no_speech_prob"),
                                 avg_logprob=data.get("avg_logprob"))
        if self.fallback is None or self.fallback.model is None:
            raise RuntimeError(f"remote ASR unavailable ({self.last_error or 'retrying later'}), no local fallback")
        cfg = self.cfg
        self.fallback.cfg = replace(self.fallback.cfg, language=cfg.language, initial_prompt=cfg.initial_prompt,
                                    hotwords=cfg.hotwords)
        self.last_source = "fallback"
        return self.fallback.transcribe(audio, sample_rate)
