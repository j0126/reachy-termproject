"""음성 입출력 (IMPLEMENTATION_PLAN.md 5.4·5.5절): HW3 A1 파이프라인에서 LLM을 빼고 키오스크에 맞게 줄였다.

입력: 로봇 마이크(XVF3800, 16 kHz) → HW3 AudioManager → Silero VAD(VadWorker) → faster-whisper(WhisperASR)
      → 버스 이벤트 speech.start / speech.utterance (규칙 기반 NLU는 FSM이 처리)
출력: 문장 → TTS 캐시(디스크) 또는 Supertonic 합성 → AudioManager.play → 로봇 스피커

바지인: 로봇이 말하는 동안 시작된 발화는 HW3 BargeInDetector(확률 0.6 이상이 400 ms)가 확인해야 speech.start를
보낸다. XVF3800이 로봇 자신의 소리를 마이크에서 지우지만(HW3 확인) 짧은 잡음 때문에 말을 끊지 않기 위해서다.
확인되지 않은 발화는 버리되, 로봇 말이 끝나기 직전(end_overlap_s 이내)에 시작한 대답은 받아들인다 (HW3 A1과 같음).

실패: TTS나 스피커가 안 되면 말 대신 예상 시간만큼 기다린다. 화면 문구는 그대로이므로 화면만으로 주문을 이어 간다.
ASR이 안 되면 발화를 버리고 로그를 남긴다 (오퍼레이터 패널의 텍스트 입력으로 계속 진행 가능).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import queue
import re
import threading
import time
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..behavior.mock_robot import MockRobot
from ..config import ROOT
from .hwpaths import resolve_hw3_file

log = logging.getLogger(__name__)

# Whisper가 잡음·무음에서 자주 지어내는 문장 (공백·문장부호를 지운 소문자로 비교)
HALLUCINATIONS = frozenset({
    "시청해주셔서감사합니다", "시청해주셔서고맙습니다", "구독과좋아요부탁드립니다", "구독과좋아요", "좋아요와구독부탁드립니다",
    "다음영상에서만나요", "mbc뉴스", "mbc뉴스이덕영입니다", "자막제공", "한글자막", "thankyouforwatching",
    "thanksforwatching", "pleasesubscribe", "you", "bye",
})


def _squash(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def load_hw3_config(cfg: dict[str, Any]) -> Any:
    """HW3에서 검증한 설정(config/default.toml + [speech] hw3_configs)에 키오스크 값만 덮어쓴다."""
    from reachy_s2s.config import load_config

    speech = cfg["speech"]
    paths = [resolve_hw3_file(cfg, p) for p in speech.get("hw3_configs", [])]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        log.warning("HW3 config files not found (using HW3 defaults instead): %s", missing)
    overrides = {
        "robot": {"host": cfg["robot"]["host"]},
        "asr": {"model": speech.get("asr_model") or None, "device": speech.get("asr_device") or None,
                "compute_type": speech.get("asr_compute_type") or None},
        "vad": {"silence_timeout_ms": float(speech.get("silence_timeout_ms", 800))},
    }
    hw3 = load_config([p for p in paths if p.is_file()], overrides)
    if hw3.tts.provider != "supertonic":  # 설정 파일이 없으면 HW3 기본값(Gemini)이 들어오므로 로컬 Supertonic으로 고정
        hw3 = replace(hw3, tts=replace(hw3.tts, provider="supertonic", model="supertonic-3", voice="M1",
                                       gain_db=6.0, threads=2))
    if hw3.asr.device == "cuda" and not speech.get("asr_device") and cuda_device_count() == 0:
        # GPU가 없는 PC(맥, NVIDIA 없는 리눅스·윈도우): CPU int8로. 설정에서 asr_device를 직접 정했으면 그대로 둔다
        log.warning("no CUDA GPU: Whisper %s runs on the CPU (int8)", hw3.asr.model)
        hw3 = replace(hw3, asr=cpu_asr(hw3.asr))
    return hw3


def cuda_device_count() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 - ctranslate2가 없거나 CUDA 드라이버 문제
        return 0


def cpu_asr(asr_cfg: Any) -> Any:
    return replace(asr_cfg, device="cpu", compute_type="int8")


def prepare_cuda_libraries() -> list[str]:
    """pip로 설치한 CUDA 라이브러리(nvidia-cublas-cu12, nvidia-cudnn-cu12)를 CTranslate2가 찾게 한다.

    Windows는 HW3 WhisperASR.load가 DLL 경로를 넣는다 (ensure_cuda_dlls). Linux는 실행 중에 LD_LIBRARY_PATH를 바꿔도
    동적 로더가 다시 읽지 않으므로, 같은 이름(SONAME)의 라이브러리를 미리 메모리에 올려 둔다 (나중의 dlopen이 그것을 씀).
    시작 스크립트(start_linux.sh)는 LD_LIBRARY_PATH도 설정한다. macOS에는 CUDA가 없다.
    """
    import sys

    if not sys.platform.startswith("linux"):
        return []
    import ctypes
    import importlib.util

    loaded: list[str] = []
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.submodule_search_locations:
            continue
        lib_dir = Path(list(spec.submodule_search_locations)[0]) / "lib"
        # 의존 순서: cublasLt → cublas, cudnn 본체 → 나머지 (이미 올라간 SONAME으로 서로 찾는다)
        names = sorted(lib_dir.glob("lib*.so*"), key=lambda p: (0 if "Lt" in p.name else 1,
                                                                0 if p.name.startswith("libcudnn.so") else 1, p.name))
        for lib in names:
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                loaded.append(lib.name)
            except OSError as exc:
                log.debug("preload %s failed: %s", lib.name, exc)
    return loaded


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0, rate


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    tmp = path.with_suffix(".tmp")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(rate))
        wf.writeframes(pcm.tobytes())
    tmp.replace(path)


def menu_prompt(menu: Any, lang: str) -> str:
    """Whisper initial_prompt: 메뉴 이름을 미리 알려 주면 '아인슈페너', '히비스커스' 같은 단어를 더 잘 받아 적는다."""
    names = [item.name.get(lang) or item.name.get("ko", "") for item in getattr(menu, "items", [])]
    names = [n for n in names if n]
    if not names:
        return ""
    head = "카페 주문: " if lang == "ko" else "Cafe order: "
    return head + ", ".join(names) + "."


class KioskSpeech:
    def __init__(self, cfg: dict[str, Any], bus: Any, backend: Any, *, hw3: Any = None, asr: Any = None,
                 tts: dict[str, Any] | None = None, vad_model: Any = None, prompts: dict[str, str] | None = None,
                 cache_dir: str | Path | None = None, events_path: str | Path | None = None):
        from reachy_s2s.engines.vad import BargeInDetector, VadWorker
        from reachy_s2s.evaluation.event_logger import EventLogger
        from reachy_s2s.robot.audio_manager import AudioManager

        sp = cfg["speech"]
        self.cfg = cfg
        self.bus = bus
        self.hw3 = hw3 if hw3 is not None else load_hw3_config(cfg)
        self.events = EventLogger(events_path)  # HW3 형식의 상세 오디오 로그 (재생·VAD 타이밍)
        self.audio = AudioManager(backend, self.hw3.audio, self.events)
        self.vad = VadWorker(self.audio, self.hw3.vad, self.events, self._on_utterance, model=vad_model,
                             on_frame=self._on_frame, on_speech_start=self._on_speech_start)
        cc = self.hw3.cascaded
        self.detector = BargeInDetector(cc.bargein_threshold, self.hw3.vad.neg_threshold, cc.bargein_min_speech_ms,
                                        cc.bargein_reset_silence_ms)
        self.asr = asr
        self._tts: dict[str, Any] = dict(tts or {})
        self.prompts = dict(prompts or {})
        self.lang = cfg["session"]["default_lang"]
        self.silence_ms = {"open": float(sp.get("silence_timeout_ms", 800)),
                           "yesno": float(sp.get("yesno_silence_timeout_ms", 500))}
        self.context = "open"
        self.min_utt_s = float(sp.get("min_utterance_s", 0.25))
        self.max_no_speech = float(sp.get("max_no_speech_prob", 0.7))
        self.overlap_s = float(sp.get("end_overlap_s", 1.0))
        self.cache_dir = Path(cache_dir) if cache_dir is not None else ROOT / sp.get("tts_cache_dir", "assets/tts_cache")
        self.ready = {"audio": False, "asr": False, "tts": False}
        self._cache: dict[str, tuple[np.ndarray, int]] = {}
        self._cache_lock = threading.Lock()
        self._tts_lock = threading.Lock()  # Supertonic 엔진 하나로 한 번에 한 문장만 합성
        self._speaking = False
        self._speak_id = 0
        self._speech_end_t = -math.inf
        self._reset_detector = False
        self._run: dict[str, Any] | None = None  # VAD 스레드 전용: 지금 진행 중인 발화 구간
        self._last_bargein: int | None = None
        self._asr_queue: queue.Queue[Any] = queue.Queue()
        self._pending_synth: list[tuple[str, str]] = []
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # ---- 시작·종료 -------------------------------------------------------------
    def start(self, listen: bool = True) -> dict[str, Any]:
        """마이크가 들어올 때까지 막힌다 (HW3 first_input_timeout_s). 모델 로딩은 백그라운드 스레드에서.

        listen=False: 말하기만 한다 ([speech] asr = "text", 발화는 오퍼레이터 패널에서 텍스트로 넣는다).
        """
        self._stop.clear()
        self.audio.start()
        self.ready["audio"] = True
        workers: list[tuple[str, Any]] = [("speech-tts", self._tts_loader)]
        if listen:
            self.vad.start()
            self.listen_context(self.context)
            workers.append(("speech-asr", self._asr_loop))
        for name, target in workers:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        return {"audio": self.audio.backend.describe(), "listen": listen}

    def close(self) -> None:
        self._stop.set()
        try:
            self.vad.stop()
        except Exception as exc:  # noqa: BLE001
            log.warning("VAD stop failed: %s", exc)
        try:
            self.audio.stop()
        except Exception as exc:  # noqa: BLE001
            log.warning("audio stop failed: %s", exc)
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        close = getattr(self.events, "close", None)
        if callable(close):
            close()

    # ---- 설정 ----------------------------------------------------------------
    def set_lang(self, lang: str) -> None:
        self.lang = lang

    def listen_context(self, context: str) -> None:
        """예/아니요를 기다릴 때는 끝점을 짧게 (500 ms), 그 밖에는 800 ms (HW3 T3 실험 값)."""
        from reachy_s2s.engines.vad import SAMPLE_RATE

        self.context = context if context in self.silence_ms else "open"
        # VadSegmenter는 끝점 시간을 샘플 수로 들고 있다 (engines/vad.py VadSegmenter.__init__)
        self.vad.segmenter._silence_timeout = self.silence_ms[self.context] * SAMPLE_RATE / 1000.0

    # ---- 말하기 ---------------------------------------------------------------
    def is_speaking(self) -> bool:
        return self._speaking

    async def speak(self, text: str, lang: str) -> None:
        self._speak_id += 1
        my_id = self._speak_id
        self._speaking = True
        self._reset_detector = True  # 이번 문장부터 바지인 시간을 센다 (HW3: 새 대답이 시작될 때 reset)
        t0 = time.perf_counter()
        try:
            audio, rate, cached = await asyncio.to_thread(self._audio_for, text, lang)
            synth_ms = (time.perf_counter() - t0) * 1000.0
            if not self.ready["audio"]:
                raise RuntimeError("robot audio is not running")
            handle = self.audio.play(audio, rate, label="tts", meta={"text": text, "lang": lang})
            duration = audio.shape[0] / float(rate)
            self.bus.log_threadsafe("speech.tts", cached=cached, synth_ms=round(synth_ms), duration_s=round(duration, 2),
                                    lang=lang, playback_id=handle.id)
            await asyncio.to_thread(handle.wait, duration + 5.0, True)
        except asyncio.CancelledError:
            self._stop_playback("cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - 음성 채널이 죽어도 화면으로 계속
            self.bus.log_threadsafe("error", where="tts", error=f"{type(exc).__name__}: {exc}", text=text)
            await asyncio.sleep(MockRobot.estimate_duration(text, lang))
        finally:
            if my_id == self._speak_id:  # 그사이 새 문장이 시작됐으면 그 문장의 상태를 건드리지 않는다
                self._speaking = False
                self._speech_end_t = time.monotonic()
                run = self._run
                if (run is not None and run["during"] and not run["announced"]
                        and self._speech_end_t - run["t"] <= self.overlap_s):
                    # 로봇 말 끝 무렵 시작한 대답이 아직 이어지는 중 (_on_utterance가 받아들일 발화와 같은 조건)
                    run["announced"] = True
                    self.bus.publish_threadsafe("speech.start", source="vad", overlap=True)

    def stop_speech(self) -> None:
        self._stop_playback("bargein")

    def _stop_playback(self, reason: str) -> None:
        try:
            self.audio.stop_playback(reason)
        except Exception as exc:  # noqa: BLE001
            log.warning("stop_playback failed: %s", exc)

    def presynthesize(self, items: list[tuple[str, str]]) -> None:
        """고정 문장을 미리 합성해 디스크 캐시에 둔다 (TTS가 준비되면 백그라운드에서)."""
        self._pending_synth.extend(items)

    # ---- TTS -----------------------------------------------------------------
    def _tts_for(self, lang: str) -> Any:
        tts = self._tts.get(lang)
        if tts is not None:
            return tts
        from reachy_s2s.engines.tts_supertonic import SupertonicTTS

        base = self._tts.get("ko") if lang != "ko" else None
        # 숫자 읽기 변환(16명 → 열여섯 명)은 한국어 전용이라 영어에서는 끈다
        tcfg = replace(self.hw3.tts, lang=lang, normalize_numbers=bool(self.hw3.tts.normalize_numbers and lang == "ko"))
        tts = SupertonicTTS(tcfg, engine=None if base is None else base.engine)  # ONNX 엔진은 언어끼리 공유
        self._tts[lang] = tts
        return tts

    def _cache_key(self, text: str, lang: str) -> str:
        t = self.hw3.tts
        raw = f"{t.model}|{t.voice}|{lang}|{t.speed}|{t.total_steps}|{t.gain_db}|{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    def _audio_for(self, text: str, lang: str) -> tuple[np.ndarray, int, bool]:
        key = self._cache_key(text, lang)
        with self._cache_lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit[0], hit[1], True
        path = self.cache_dir / f"{key}.wav"
        if path.is_file():
            try:
                audio, rate = read_wav(path)
                with self._cache_lock:
                    self._cache[key] = (audio, rate)
                return audio, rate, True
            except Exception as exc:  # noqa: BLE001 - 깨진 캐시 파일은 다시 합성
                log.warning("bad TTS cache file %s: %s", path, exc)
        with self._tts_lock:
            result = self._tts_for(lang).synthesize(text)
        audio, rate = np.asarray(result.audio, dtype=np.float32).reshape(-1), int(result.sample_rate)
        with self._cache_lock:
            self._cache[key] = (audio, rate)
        try:
            write_wav(path, audio, rate)
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot write TTS cache %s: %s", path, exc)
        return audio, rate, False

    def _tts_loader(self) -> None:
        t0 = time.perf_counter()
        try:
            tts = self._tts_for("ko")
            info = tts.load() if hasattr(tts, "load") else {}
            self.ready["tts"] = True
            self.bus.log_threadsafe("speech.tts_ready", load_s=round(time.perf_counter() - t0, 1),
                                    **{k: v for k, v in (info or {}).items() if k in ("model", "sample_rate")})
        except Exception as exc:  # noqa: BLE001
            self.bus.log_threadsafe("error", where="tts_load", error=f"{type(exc).__name__}: {exc}")
            return
        done = 0
        t1 = time.perf_counter()
        while self._pending_synth and not self._stop.is_set():
            text, lang = self._pending_synth.pop(0)
            try:
                _audio, _rate, cached = self._audio_for(text, lang)
                done += 0 if cached else 1
            except Exception as exc:  # noqa: BLE001
                log.warning("presynthesis failed for %r: %s", text, exc)
        self.bus.log_threadsafe("speech.presynthesized", new=done, seconds=round(time.perf_counter() - t1, 1))

    # ---- 듣기 (VAD 스레드) -------------------------------------------------------
    def _on_speech_start(self, index: int, _t_ns: int) -> None:
        during = self._speaking
        self._run = {"index": index, "t": time.monotonic(), "during": during, "bargein": False,
                     "announced": not during}
        if not during:
            self.bus.publish_threadsafe("speech.start", source="vad")

    def _on_frame(self, prob: float, index: int, _t_ns: int) -> None:
        if not self._speaking:
            return
        if self._reset_detector:
            self._reset_detector = False
            self.detector.reset()
        if self.detector.push(prob, index):
            self._last_bargein = self.detector.start_index if self.detector.start_index is not None else index
            run = self._run
            if run is not None:
                run["bargein"] = True
                run["announced"] = True
            self.bus.publish_threadsafe("speech.start", source="vad", bargein=True,
                                        active_ms=round(self.detector.active_ms))

    def _on_utterance(self, utt: Any) -> None:
        run, self._run = self._run, None
        if utt.speech_duration_s < self.min_utt_s:
            self.bus.log_threadsafe("speech.ignored", reason="too_short", speech_s=round(utt.speech_duration_s, 2))
            return
        why = "listening"
        if run is not None and run["during"]:
            barge = run["bargein"] or (self._last_bargein is not None
                                       and utt.speech_start_index <= self._last_bargein <= utt.speech_end_index)
            if barge:
                why = "bargein"
            elif self._speech_end_t >= run["t"] and self._speech_end_t - run["t"] <= self.overlap_s:
                why = "end_overlap"
            else:
                self.bus.log_threadsafe("speech.ignored", reason="robot_speaking",
                                        speech_s=round(utt.speech_duration_s, 2))
                return
        self._asr_queue.put((utt, self.lang, why))

    # ---- ASR (전용 스레드) -------------------------------------------------------
    def _make_asr(self) -> Any:
        from reachy_s2s.engines.asr import WhisperASR

        return WhisperASR(self.hw3.asr)

    def _load_asr(self) -> dict[str, Any]:
        """ASR 로딩. GPU(CUDA)에서 실패하면(드라이버·cuDNN 문제) CPU int8로 한 번 더 시도한다 — 음성 인식이
        통째로 죽는 대신 느려지는 쪽으로 (데모 중 실패 처리)."""
        if not hasattr(self.asr, "load"):
            return {}
        acfg = getattr(self.asr, "cfg", None)
        if acfg is not None and acfg.device == "cuda":
            prepare_cuda_libraries()
        try:
            return self.asr.load() or {}
        except Exception as exc:  # noqa: BLE001
            if acfg is None or acfg.device != "cuda":
                raise
            self.bus.log_threadsafe("error", where="asr_load_cuda", error=f"{type(exc).__name__}: {exc}",
                                    fallback="cpu int8")
            self.asr.cfg = cpu_asr(acfg)
            self.asr.model = None
            return self.asr.load() or {}

    def _asr_loop(self) -> None:
        t0 = time.perf_counter()
        try:
            if self.asr is None:
                self.asr = self._make_asr()
            info = self._load_asr()
            self.ready["asr"] = True
            self.bus.log_threadsafe("speech.asr_ready", load_s=round(time.perf_counter() - t0, 1),
                                    **{k: v for k, v in (info or {}).items() if k in ("model", "device", "compute_type")})
        except Exception as exc:  # noqa: BLE001
            self.bus.log_threadsafe("error", where="asr_load", error=f"{type(exc).__name__}: {exc}")
        while not self._stop.is_set():
            try:
                utt, lang, why = self._asr_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not self.ready["asr"]:
                self.bus.log_threadsafe("speech.ignored", reason="asr_unavailable")
                continue
            self._transcribe(utt, lang, why)

    def _configure_asr(self, lang: str) -> None:
        acfg = getattr(self.asr, "cfg", None)
        if acfg is None:
            return
        prompt = self.prompts.get(lang, acfg.initial_prompt)
        if acfg.language != lang or acfg.initial_prompt != prompt:
            self.asr.cfg = replace(acfg, language=lang, initial_prompt=prompt)

    def _transcribe(self, utt: Any, lang: str, why: str) -> None:
        self._configure_asr(lang)
        t0 = time.perf_counter()
        try:
            result = self.asr.transcribe(utt.audio, utt.sample_rate)
        except Exception as exc:  # noqa: BLE001
            self.bus.log_threadsafe("error", where="asr", error=f"{type(exc).__name__}: {exc}")
            return
        asr_ms = (time.perf_counter() - t0) * 1000.0
        text = (result.text or "").strip()
        nsp = result.no_speech_prob
        detail = {"speech_s": round(utt.speech_duration_s, 2), "asr_ms": round(asr_ms),
                  "no_speech_prob": None if nsp is None else round(float(nsp), 3), "lang": lang}
        reason = None
        if not text:
            reason = "empty"
        elif _squash(text) in HALLUCINATIONS:
            reason = "hallucination"
        elif nsp is not None and nsp > self.max_no_speech:
            reason = "no_speech"
        if reason:
            self.bus.log_threadsafe("speech.ignored", reason=reason, text=text, **detail)
            return
        conf = None if result.avg_logprob is None else round(float(math.exp(min(0.0, result.avg_logprob))), 3)
        latency = (time.perf_counter_ns() - int(utt.t_endpoint_ns)) / 1e9 if getattr(utt, "t_endpoint_ns", None) else None
        self.bus.publish_threadsafe("speech.utterance", text=text, conf=conf, source="asr", why=why,
                                    latency_s=None if latency is None else round(latency, 3), **detail)
