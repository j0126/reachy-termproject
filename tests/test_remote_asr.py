"""원격 음성 인식: 현장 PC에 NVIDIA GPU가 없을 때 인식만 데스크톱 GPU에 맡기는 예비 경로 (11/27).

실제 네트워크 대신 urllib 요청을 FastAPI TestClient로 보내 RemoteASR ↔ asr_server 왕복을 확인한다.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.parse
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("reachy_s2s")

from fastapi.testclient import TestClient  # noqa: E402
from reachy_s2s.config import AsrConfig  # noqa: E402
from reachy_s2s.engines.asr import AsrResult, WhisperASR  # noqa: E402

from reachy_cafe.asr_server import create_app  # noqa: E402
from reachy_cafe.robot.remote_asr import RemoteASR, decode_pcm, encode_pcm  # noqa: E402
from reachy_cafe.robot.speech import KioskSpeech  # noqa: E402

URL = "http://desktop:8001"
AUDIO = (0.1 * np.sin(np.linspace(0, 400, 16000))).astype(np.float32)  # 1초
KO_PROMPT = "카페 주문: 아메리카노, 아인슈페너, 히비스커스 티."


def asr_cfg(**overrides: object) -> AsrConfig:
    values: dict[str, object] = dict(model="large-v3-turbo", device="cuda", compute_type="float16", language="ko",
                                     beam_size=5, initial_prompt="", hotwords="", warmup=False)
    values.update(overrides)
    return AsrConfig(**values)  # type: ignore[arg-type]


class FakeEngine:
    """WhisperASR 대신: 받은 설정·오디오를 기록하고 정해진 글자를 돌려준다."""

    def __init__(self, cfg: AsrConfig, text: str = "아메리카노 주세요"):
        self.cfg = cfg
        self.model = object()
        self.text = text
        self.calls: list[tuple[AsrConfig, int, int]] = []
        self.loaded = False

    def load(self) -> dict[str, object]:
        self.loaded = True
        return {"model": self.cfg.model}

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> AsrResult:
        self.calls.append((self.cfg, len(audio), sample_rate))
        return AsrResult(text=self.text, language=self.cfg.language, language_probability=1.0,
                         audio_s=len(audio) / sample_rate, t_start_ns=0, t_end_ns=150_000_000,
                         no_speech_prob=0.01, avg_logprob=-0.1)


class Opener:
    """urllib.request.urlopen 대신 TestClient로 보낸다. client=None이면 연결 거부처럼 실패한다."""

    def __init__(self, client: TestClient | None):
        self.client = client
        self.calls = 0

    def __call__(self, req: urllib.request.Request, timeout: float | None = None) -> io.BytesIO:
        self.calls += 1
        if self.client is None:
            raise urllib.error.URLError("connection refused")
        parts = urllib.parse.urlsplit(req.full_url)
        resp = self.client.request(req.get_method(), parts.path + (f"?{parts.query}" if parts.query else ""),
                                   content=req.data, headers=dict(req.header_items()))
        if resp.status_code >= 400:
            raise urllib.error.HTTPError(req.full_url, resp.status_code, resp.text, None, None)  # type: ignore[arg-type]
        return io.BytesIO(resp.content)


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def desktop(text: str = "아메리카노 주세요", token: str = "") -> tuple[FakeEngine, Opener]:
    engine = FakeEngine(asr_cfg(), text)
    return engine, Opener(TestClient(create_app(engine, token)))


def test_pcm_round_trip_keeps_the_audio() -> None:
    assert np.max(np.abs(decode_pcm(encode_pcm(AUDIO)) - AUDIO)) < 1 / 16000


def test_desktop_gets_the_language_and_menu_prompt_and_returns_whisper_fields() -> None:
    engine, opener = desktop()
    client = RemoteASR(asr_cfg(device="remote", initial_prompt=KO_PROMPT), URL, opener=opener)
    result = client.transcribe(AUDIO)
    assert result.text == "아메리카노 주세요" and client.last_source == "remote" and client.last_error is None
    assert result.no_speech_prob == 0.01 and result.avg_logprob == -0.1 and result.audio_s == pytest.approx(1.0)
    sent_cfg, n_samples, rate = engine.calls[0]
    assert (sent_cfg.language, sent_cfg.initial_prompt, sent_cfg.beam_size) == ("ko", KO_PROMPT, 5)
    assert (n_samples, rate) == (16000, 16000)
    # 언어를 바꾸면(영어 손님) 다음 요청부터 그 언어로
    client.cfg = asr_cfg(device="remote", language="en", initial_prompt="Cafe order: Americano.")
    client.transcribe(AUDIO)
    assert (engine.calls[1][0].language, engine.calls[1][0].initial_prompt) == ("en", "Cafe order: Americano.")


def test_load_reports_the_desktop_model() -> None:
    _, opener = desktop()
    info = RemoteASR(asr_cfg(device="remote"), URL, opener=opener).load()
    assert "large-v3-turbo" in info["model"] and "cuda" in info["model"] and info["device"] == "remote"


def test_desktop_down_uses_local_cpu_then_tries_the_desktop_again_after_retry_s() -> None:
    clock = Clock()
    fallback = FakeEngine(asr_cfg(model="small", device="cpu", compute_type="int8"), text="레귤러요")
    fallback.model = None  # load() 전에는 없음
    fallback.load = lambda: setattr(fallback, "model", object()) or {}  # type: ignore[method-assign]
    opener = Opener(None)
    client = RemoteASR(asr_cfg(device="remote", initial_prompt=KO_PROMPT), URL, opener=opener, fallback=fallback,
                       clock=clock, retry_s=20.0)
    info = client.load()  # 데스크톱이 꺼져 있어도 시작은 된다
    assert info["fallback"] == "small (cpu)" and opener.calls == 1

    assert client.transcribe(AUDIO).text == "레귤러요" and client.last_source == "fallback"
    assert opener.calls == 1 and client.last_error is None  # 재시도 시간 전: 데스크톱을 기다리지 않음
    assert fallback.calls[0][0].initial_prompt == KO_PROMPT  # 메뉴 이름 프롬프트도 대비책에 넘긴다

    clock.t += 21
    client.transcribe(AUDIO)
    assert opener.calls == 2 and client.last_source == "fallback" and "URLError" in (client.last_error or "")
    client.transcribe(AUDIO)
    assert opener.calls == 2  # 방금 실패했으니 다시 20초 동안은 바로 대비책

    _, up = desktop("따뜻한 라떼 하나 주세요")
    client._open = up
    clock.t += 21
    assert client.transcribe(AUDIO).text == "따뜻한 라떼 하나 주세요" and client.last_source == "remote"


def test_desktop_down_without_fallback_raises_so_speech_logs_an_error() -> None:
    client = RemoteASR(asr_cfg(device="remote"), URL, opener=Opener(None), retry_s=0.0)
    client.load()  # 시작에서는 실패로 치지 않는다 (데스크톱이 나중에 켜질 수 있음)
    with pytest.raises(RuntimeError, match="remote ASR unavailable"):
        client.transcribe(AUDIO)


def test_token_protects_the_desktop_server() -> None:
    engine, opener = desktop(token="s3cret")
    raw = opener.client
    assert raw is not None
    assert raw.get("/health").status_code == 401
    assert raw.get("/health", headers={"X-Token": "s3cret"}).json()["model"] == "large-v3-turbo"
    assert RemoteASR(asr_cfg(device="remote"), URL, opener=opener, token="s3cret").transcribe(AUDIO).text
    wrong = RemoteASR(asr_cfg(device="remote"), URL, opener=opener, token="nope")
    with pytest.raises(RuntimeError):
        wrong.transcribe(AUDIO)
    assert "401" in (wrong.last_error or "")


def test_server_rejects_a_body_that_is_not_16bit_pcm() -> None:
    _, opener = desktop()
    assert opener.client is not None
    assert opener.client.post("/transcribe", content=b"\x00\x01\x02").status_code == 400


def test_speech_builds_remote_asr_with_a_cpu_fallback_from_config() -> None:
    hw3 = SimpleNamespace(asr=asr_cfg())
    speech = {"asr": "remote", "asr_url": "http://10.8.139.133:8001", "asr_timeout_s": 2.5, "asr_retry_s": 15.0,
              "asr_fallback": "small"}
    asr = KioskSpeech._make_asr(SimpleNamespace(cfg={"speech": speech}, hw3=hw3))  # type: ignore[arg-type]
    assert isinstance(asr, RemoteASR) and asr.url == "http://10.8.139.133:8001"
    assert (asr.timeout_s, asr.retry_s, asr.cfg.device, asr.cfg.model) == (2.5, 15.0, "remote", "large-v3-turbo")
    assert isinstance(asr.fallback, WhisperASR)
    assert (asr.fallback.cfg.model, asr.fallback.cfg.device, asr.fallback.cfg.compute_type) == ("small", "cpu", "int8")

    no_fallback = KioskSpeech._make_asr(SimpleNamespace(cfg={"speech": {**speech, "asr_fallback": ""}}, hw3=hw3))  # type: ignore[arg-type]
    assert no_fallback.fallback is None
    local = KioskSpeech._make_asr(SimpleNamespace(cfg={"speech": {"asr": "whisper"}}, hw3=hw3))  # type: ignore[arg-type]
    assert isinstance(local, WhisperASR) and local.cfg.device == "cuda"
