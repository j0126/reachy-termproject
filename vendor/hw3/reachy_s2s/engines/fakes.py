"""Stand-in engines for unit tests and offline dry runs (no GPU, no API, no cost).

They mimic the real engines' result objects and timing fields, can be slowed
down, and can be told to fail with a given :class:`EngineError` kind.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..audio.signals import tone
from ..conversation.history import ConversationHistory
from .asr import AsrResult
from .base import Attempt, EngineError
from .llm import LlmResult, user_step
from .tts import TtsResult


class FakeASR:
    name = "fake-asr"

    def __init__(self, texts: list[str] | None = None, delay_s: float = 0.05) -> None:
        self.texts = list(texts or ["안녕하세요"])
        self.delay_s = delay_s
        self.calls = 0
        self.load_info = {"engine": self.name}
        self.fail_kind: str | None = None

    def load(self) -> dict[str, Any]:
        return self.load_info

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> AsrResult:
        t0 = time.perf_counter_ns()
        time.sleep(self.delay_s)
        if self.fail_kind:
            raise EngineError("asr", self.fail_kind, "fake ASR failure")
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return AsrResult(text=text, language="ko", language_probability=1.0, audio_s=len(audio) / sample_rate,
                         t_start_ns=t0, t_end_ns=time.perf_counter_ns(), no_speech_prob=0.0)


class FakeLLM:
    name = "fake-llm"

    def __init__(self, delay_s: float = 0.05, reply: Callable[[str, int], str] | None = None) -> None:
        self.delay_s = delay_s
        self.reply = reply or (lambda text, n: f"{text}라고 하셨네요. 지금까지 {n}번 이야기했어요.")
        self.fail_kinds: list[str] = []  # consumed one per call
        self.calls = 0
        self.seen_inputs: list[list[dict[str, Any]]] = []

    @property
    def params(self) -> dict[str, Any]:
        return {"fake": True}

    def respond(self, history: ConversationHistory, text: str,
                on_attempt: Callable[[Attempt], None] | None = None) -> LlmResult:
        t0 = time.perf_counter_ns()
        self.calls += 1
        steps = history.api_input() + [user_step(text)]
        self.seen_inputs.append(steps)
        time.sleep(self.delay_s)
        if self.fail_kinds:
            kind = self.fail_kinds.pop(0)
            status = 429 if kind == "rate_limit" else (500 if kind == "server" else None)
            raise EngineError("llm", kind, f"fake {kind}", status)
        reply = self.reply(text, len(history) + 1)
        model_steps = [{"type": "model_output", "content": [{"type": "text", "text": reply}]}]
        return LlmResult(text=reply, status="completed", model="fake-llm", params=self.params, t_start_ns=t0,
                         t_end_ns=time.perf_counter_ns(), user_step=user_step(text), model_steps=model_steps,
                         usage={"total_input_tokens": 10 * len(steps), "total_output_tokens": 12},
                         input_steps=len(steps))


class EchoLLM(FakeLLM):
    """Deterministic stand-in for the LLM (repeats what ASR heard). Not a language model:
    used to verify VAD/ASR/TTS/robot audio on hardware while the cloud LLM is unavailable."""

    name = "echo-llm"

    def __init__(self) -> None:
        super().__init__(delay_s=0.0, reply=lambda text, n: f"{text}, 라고 들었어요.")

    @property
    def params(self) -> dict[str, Any]:
        return {"stand_in": "echo (no language model)"}


class FakeTTS:
    name = "fake-tts"

    def __init__(self, delay_s: float = 0.05, sample_rate: int = 24000, seconds_per_char: float = 0.02) -> None:
        self.delay_s = delay_s
        self.sample_rate = sample_rate
        self.seconds_per_char = seconds_per_char
        self.fail_kind: str | None = None

    @property
    def params(self) -> dict[str, Any]:
        return {"fake": True, "sample_rate": self.sample_rate}

    def synthesize(self, text: str, on_attempt: Callable[[Attempt], None] | None = None) -> TtsResult:
        t0 = time.perf_counter_ns()
        time.sleep(self.delay_s)
        if self.fail_kind:
            raise EngineError("tts", self.fail_kind, "fake TTS failure")
        duration = max(0.2, min(3.0, len(text) * self.seconds_per_char))
        audio = tone(330.0, duration, self.sample_rate, amplitude=0.2)
        return TtsResult(audio=audio, sample_rate=self.sample_rate, text=text, model="fake-tts", voice="none",
                         mime_type="audio/wav (fake)", t_start_ns=t0, t_end_ns=time.perf_counter_ns())
