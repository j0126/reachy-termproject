"""KioskSpeech (HW3 파이프라인 축소판)를 HW3 MockAudioBackend + FakeASR/FakeTTS + 에너지 VAD로 실제 시간에 돌린다.

확인하는 것: 듣는 중 발화 → speech.start → speech.utterance, TTS 캐시, 로봇이 말하는 중 바지인 확인,
로봇 말과 겹친 짧은 소리는 버림, 예/아니요 문맥에서 끝점 단축.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from reachy_cafe.bus import EventLog
from reachy_cafe.config import load_config
from reachy_cafe.robot.hwpaths import ensure_hw_paths

np = pytest.importorskip("numpy")
CFG = load_config(env={})
ensure_hw_paths(CFG)  # HW3 코드(vendor/hw3)를 못 찾으면 아래에서 건너뜀
pytest.importorskip("reachy_s2s")
pytest.importorskip("soxr")

from reachy_s2s.engines.fakes import FakeASR, FakeTTS  # noqa: E402
from reachy_s2s.robot.backends.mock import MockAudioBackend  # noqa: E402

from reachy_cafe.robot.speech import HALLUCINATIONS, KioskSpeech, _squash, load_hw3_config  # noqa: E402

RATE = 16000


class FakeBus:
    def __init__(self) -> None:
        self.log = EventLog(None)
        self.events: list[tuple[str, dict]] = []
        self._lock = threading.Lock()

    def publish_threadsafe(self, kind: str, **data) -> None:
        with self._lock:
            self.events.append((kind, data))

    log_threadsafe = publish_threadsafe

    def of(self, kind: str) -> list[dict]:
        with self._lock:
            return [d for k, d in self.events if k == kind]


class EnergyVad:
    """Silero 대신: 프레임 RMS가 크면 말소리 (테스트용 FrameModel)."""

    def reset(self) -> None:
        pass

    def __call__(self, frame: np.ndarray) -> float:
        return 0.95 if float(np.sqrt(np.mean(np.square(frame)))) > 0.02 else 0.02


def source(*parts: tuple[float, bool]) -> np.ndarray:
    chunks = []
    for seconds, voiced in parts:
        n = int(seconds * RATE)
        t = np.arange(n) / RATE
        chunks.append((0.3 * np.sin(2 * np.pi * 220 * t) if voiced else np.zeros(n)).astype(np.float32))
    mono = np.concatenate(chunks)
    return np.stack([mono, mono], axis=1)


def make_speech(tmp_path, audio: np.ndarray, texts=("아메리카노 주세요",)):
    bus = FakeBus()
    hw3 = load_hw3_config(CFG)
    backend = MockAudioBackend(audio, sample_rate=RATE, channels=2)
    speech = KioskSpeech(CFG, bus, backend, hw3=hw3, asr=FakeASR(list(texts)), tts={"ko": FakeTTS(), "en": FakeTTS()},
                         vad_model=EnergyVad(), cache_dir=tmp_path / "tts")
    return bus, backend, speech


def wait_for(pred, timeout: float = 6.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_utterance_while_listening_is_published(tmp_path):
    bus, _backend, speech = make_speech(tmp_path, source((0.5, False), (1.0, True), (1.5, False)))
    speech.start()
    try:
        assert wait_for(lambda: bus.of("speech.utterance"), 6.0)
        utt = bus.of("speech.utterance")[0]
        assert utt["text"] == "아메리카노 주세요"
        assert utt["why"] == "listening"
        assert 0.8 < utt["speech_s"] < 1.3
        assert bus.of("speech.start") and not bus.of("speech.start")[0].get("bargein")
    finally:
        speech.close()


def test_tts_is_cached_on_disk_and_played(tmp_path):
    bus, backend, speech = make_speech(tmp_path, source((3.0, False)))

    async def run() -> None:
        speech.start(listen=False)
        await speech.speak("안녕하세요! 주문 도와드릴까요?", "ko")
        await speech.speak("안녕하세요! 주문 도와드릴까요?", "ko")

    try:
        asyncio.run(run())
        tts = bus.of("speech.tts")
        assert [e["cached"] for e in tts] == [False, True]
        assert len(list((tmp_path / "tts").glob("*.wav"))) == 1
        assert backend.written.shape[0] > 0  # 로봇 스피커로 보낸 샘플 (MockAudioBackend 기록)
        assert not speech.is_speaking()
    finally:
        speech.close()


def test_bargein_is_confirmed_while_robot_speaks(tmp_path):
    # 로봇이 말하는 동안(약 3 s) 1.0 s부터 1.5 s 동안 손님이 말함 → 0.4 s 지나 바지인 확인
    bus, _backend, speech = make_speech(tmp_path, source((1.0, False), (1.5, True), (2.5, False)),
                                        texts=("잠깐만요",))

    async def run() -> None:
        speech.start()
        task = asyncio.create_task(speech.speak("가" * 150, "ko"))  # FakeTTS: 최대 3 s
        assert await asyncio.to_thread(wait_for, lambda: any(e.get("bargein") for e in bus.of("speech.start")), 4.0)
        speech.stop_speech()  # BehaviorManager가 하는 일
        await asyncio.wait_for(task, 10.0)
        assert await asyncio.to_thread(wait_for, lambda: bus.of("speech.utterance"), 5.0)

    try:
        asyncio.run(run())
        assert bus.of("speech.utterance")[0]["why"] == "bargein"
    finally:
        speech.close()


def test_short_noise_over_robot_speech_is_ignored(tmp_path):
    bus, _backend, speech = make_speech(tmp_path, source((0.8, False), (0.32, True), (3.5, False)))

    async def run() -> None:
        speech.start()
        await speech.speak("가" * 150, "ko")
        await asyncio.to_thread(wait_for, lambda: bus.of("speech.ignored"), 3.0)

    try:
        asyncio.run(run())
        assert not any(e.get("bargein") for e in bus.of("speech.start"))
        assert not bus.of("speech.utterance")
        assert bus.of("speech.ignored")[0]["reason"] == "robot_speaking"
    finally:
        speech.close()


def test_yesno_context_shortens_the_endpoint(tmp_path):
    _bus, _backend, speech = make_speech(tmp_path, source((0.5, False)))
    speech.listen_context("yesno")
    assert speech.vad.segmenter._silence_timeout == pytest.approx(500 * 16)
    speech.listen_context("open")
    assert speech.vad.segmenter._silence_timeout == pytest.approx(800 * 16)


def test_hallucination_filter_matches_common_whisper_phrases():
    assert _squash("시청해 주셔서 감사합니다.") in HALLUCINATIONS
    assert _squash("아메리카노 주세요") not in HALLUCINATIONS


def test_no_gpu_falls_back_to_cpu_unless_device_is_pinned(monkeypatch):
    import reachy_cafe.robot.speech as speech_mod

    monkeypatch.setattr(speech_mod, "cuda_device_count", lambda: 0)
    hw3 = load_hw3_config(CFG)
    assert (hw3.asr.device, hw3.asr.compute_type) == ("cpu", "int8")  # 맥·GPU 없는 PC: 자동으로 CPU
    pinned = {**CFG, "speech": {**CFG["speech"], "asr_device": "cuda"}}
    assert load_hw3_config(pinned).asr.device == "cuda"  # 설정에서 정했으면 그대로


def test_cuda_load_failure_retries_on_cpu(tmp_path):
    from dataclasses import replace

    class FlakyASR:
        def __init__(self, cfg):
            self.cfg, self.model = cfg, None

        def load(self):
            if self.cfg.device == "cuda":
                raise RuntimeError("cuDNN library not found")
            self.model = "ok"
            return {"device": self.cfg.device, "compute_type": self.cfg.compute_type}

    bus, _backend, speech = make_speech(tmp_path, source((0.5, False)))
    speech.asr = FlakyASR(replace(speech.hw3.asr, device="cuda", compute_type="float16"))
    info = speech._load_asr()
    assert info == {"device": "cpu", "compute_type": "int8"}
    assert any(e.get("fallback") == "cpu int8" for e in bus.of("error"))
