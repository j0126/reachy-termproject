"""데스크톱 사전 점검: 로봇을 움직이지 않고, 이 PC와 로봇이 연결 점검(spike)·데모를 돌릴 준비가 됐는지 본다.

  .venv\\Scripts\\python -X utf8 tools\\preflight.py                 # PC + 파일 + 네트워크 + 로봇(읽기만)
  .venv\\Scripts\\python -X utf8 tools\\preflight.py --load-models   # Whisper(GPU)·Supertonic·MediaPipe를 실제로 올려 1회 실행
  .venv\\Scripts\\python -X utf8 tools\\preflight.py --no-robot      # 로봇 없이 PC만
  .venv\\Scripts\\python -X utf8 tools\\preflight.py --fix-media     # 로봇 카메라·오디오가 '해제'돼 있으면 다시 켬

모터 명령은 보내지 않는다: 로봇에는 GET만 보내고, --fix-media일 때만 미디어를 다시 잡는 POST를 보낸다
(HW3 apps/check_setup.py와 같은 방식). 결과는 logs/preflight_<시각>.json, 마지막에 다음에 할 일을 출력한다.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reachy_cafe.config import load_config  # noqa: E402
from reachy_cafe.robot.hwpaths import _resolve, ensure_hw_paths, find_face_model, resolve_hw3_file  # noqa: E402

SDK_VERSION = "1.11.0"
OS = "windows" if sys.platform == "win32" else "mac" if sys.platform == "darwin" else "linux"
# (배포 이름, 꼭 필요한지, 고정 버전, 해당 OS) — pyproject.toml의 [robot]·[speech]·[cuda]와 같다
PACKAGES = [
    ("fastapi", True, None, None), ("uvicorn", True, None, None), ("websockets", True, None, None),
    ("rapidfuzz", True, None, None), ("numpy", True, None, None), ("reachy-mini", True, SDK_VERSION, None),
    ("reachy-mini-rust-kinematics", True, "1.0.3", None), ("mediapipe", True, "1.0.1", None), ("PyYAML", True, None, None),
    ("faster-whisper", True, None, None), ("ctranslate2", True, None, None), ("supertonic", True, "1.3.1", None),
    ("soxr", True, None, None), ("python-dotenv", True, None, None), ("onnxruntime", True, None, None),
    ("gstreamer-bundle", True, None, ("windows", "mac")), ("PyGObject", True, None, ("linux",)),
]
CUDA_PACKAGES = ("nvidia-cublas-cu12", "nvidia-cudnn-cu12")
INSTALL_HINT = {"windows": "start_windows.bat setup", "mac": "start_mac.command 의 1) 설치",
                "linux": "./start_linux.sh setup"}
FIREWALL_HINT = {
    "windows": "처음 실행 때 Windows 방화벽 창에서 '개인 네트워크 허용'",
    "mac": "시스템 설정 > 네트워크 > 방화벽에서 Python 수신 허용, 처음 로봇 연결 때 '로컬 네트워크' 접근 허용 (macOS 15+)",
    "linux": "방화벽을 쓰면 sudo ufw allow 8000/tcp",
}
GST_HINT = {
    "windows": "reachy-mini 다시 설치 (gstreamer-bundle이 GStreamer를 함께 설치함)",
    "mac": "reachy-mini 다시 설치 (gstreamer-bundle이 GStreamer를 함께 설치함)",
    "linux": "GStreamer 1.22+ apt 패키지 + Rust webrtc 플러그인 빌드, GST_PLUGIN_PATH 설정 (docs/desktop_check.md 'Linux 추가 준비')",
}
WEBRTC_ELEMENTS = ("webrtcsrc", "webrtcbin", "nicesrc", "opusenc", "rtpopuspay", "appsink", "appsrc", "audioconvert",
                   "audioresample", "videoconvert")


class Report:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def add(self, group: str, status: str, item: str, detail: str = "") -> None:
        self.items.append({"group": group, "status": status, "item": item, "detail": detail})
        print(f"  [{status:4s}] {item}" + (f" — {detail}" if detail else ""), flush=True)

    def count(self, status: str, group: str | None = None) -> int:
        return sum(1 for i in self.items if i["status"] == status and (group is None or i["group"] == group))


def version_of(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def http_json(url: str, timeout: float = 3.0, data: bytes | None = None) -> Any:
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - 로봇·localhost만
        return json.loads(resp.read().decode("utf-8") or "null")


# ---- 1 PC ----------------------------------------------------------------------------
def check_os(r: Report) -> None:
    machine = platform.machine().lower()
    if OS == "mac":
        ver = platform.mac_ver()[0] or "0"
        major = int(ver.split(".")[0] or 0)
        arm = machine in ("arm64", "aarch64")
        r.add("pc", "OK" if arm and major >= 14 else "FAIL", f"macOS {ver} ({machine})",
              "" if arm and major >= 14 else "Apple Silicon(M1 이상) + macOS 14 이상 필요: mediapipe 1.0.1·onnxruntime "
              "1.27.0에 Intel 맥·macOS 13 이하용 패키지가 없음")
    elif OS == "linux":
        libc, libc_ver = platform.libc_ver()
        try:
            distro = platform.freedesktop_os_release().get("PRETTY_NAME", "Linux")
        except OSError:
            distro = "Linux"
        ok = libc != "glibc" or tuple(int(x) for x in (libc_ver or "0.0").split(".")[:2]) >= (2, 28)
        r.add("pc", "OK" if ok else "FAIL", f"{distro} ({machine}, {libc} {libc_ver})",
              "" if ok else "glibc 2.28 이상 필요 (mediapipe·onnxruntime manylinux_2_28). Ubuntu 24.04 권장")
    else:
        r.add("pc", "OK", f"{platform.system()} {platform.release()} ({machine})")


def check_pc(r: Report, asr_device: str) -> None:
    print("\n[1] PC·설치")
    check_os(r)
    v = sys.version_info
    r.add("pc", "OK" if (v.major, v.minor) == (3, 12) else "FAIL", f"Python {v.major}.{v.minor}.{v.micro}",
          "" if (v.major, v.minor) == (3, 12) else "Python 3.12 필요")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    r.add("pc", "OK" if in_venv else "WARN", f"가상환경 {sys.prefix}", "" if in_venv else ".venv 밖에서 실행 중")
    for dist, required, pin, systems in PACKAGES:
        if systems is not None and OS not in systems:
            continue
        ver = version_of(dist)
        if ver is None:
            r.add("pc", "FAIL" if required else "WARN", f"패키지 {dist} 없음", INSTALL_HINT[OS])
        elif pin is not None and ver != pin:
            r.add("pc", "WARN", f"패키지 {dist} {ver}", f"HW3·HW4에서 검증한 버전은 {pin}")
        else:
            r.add("pc", "OK", f"패키지 {dist} {ver}")
    n_gpu = 0
    try:
        import ctranslate2

        n_gpu = ctranslate2.get_cuda_device_count()
    except Exception:  # noqa: BLE001
        pass
    name = ""
    if shutil.which("nvidia-smi"):
        try:
            name = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            name = ""
    if n_gpu > 0:
        r.add("pc", "OK", f"CUDA GPU {n_gpu}개" + (f": {name}" if name else ""), "Whisper를 GPU에서 실행")
        for dist in CUDA_PACKAGES:
            ver = version_of(dist)
            r.add("pc", "OK" if ver else "WARN", f"패키지 {dist} " + (ver or "없음"),
                  "" if ver else "GPU 음성 인식용 CUDA 라이브러리: 설치에서 cuda 포함")
    elif asr_device == "cuda":
        r.add("pc", "FAIL", "CUDA GPU 없음", "config [speech] asr_device = \"cuda\"로 고정돼 있음: 비우면 CPU로 자동 전환")
    else:
        r.add("pc", "INFO", "CUDA GPU 없음 → Whisper를 CPU(int8)에서 실행",
              "인식 시간은 --load-models로 확인 (느리면 [speech] asr_model = \"small\")")


def check_media(r: Report) -> None:
    """SDK의 WebRTC 카메라·오디오가 쓰는 GStreamer 요소 (webrtcsrc는 GStreamer Rust 플러그인)."""
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        missing = [e for e in WEBRTC_ELEMENTS if Gst.ElementFactory.find(e) is None]
        version = Gst.version()
        old = (version[0], version[1]) < (1, 22)
        status = "FAIL" if missing or old else "OK"
        detail = GST_HINT[OS] if status == "FAIL" else "로봇 카메라·마이크·스피커(WebRTC)를 쓸 수 있음"
        r.add("media", status, f"GStreamer {version[0]}.{version[1]}.{version[2]}"
              + (f", 없는 요소: {', '.join(missing)}" if missing else ", WebRTC 요소 모두 있음"), detail)
    except Exception as exc:  # noqa: BLE001
        r.add("media", "FAIL", "GStreamer(PyGObject)를 불러올 수 없음", f"{type(exc).__name__}: {exc} → {GST_HINT[OS]}")


# ---- 2 파일 -------------------------------------------------------------------------
def check_files(r: Report, cfg: dict[str, Any]) -> None:
    print("\n[2] HW 코드·모델 파일")
    ensure_hw_paths(cfg)
    for module, key in (("reachy_motion", "reachy_hw4"), ("reachy_s2s", "hw3")):
        try:
            mod = importlib.import_module(module)
            where = Path(mod.__file__).resolve().parent
            folder = _resolve(cfg["hw"][key])
            r.add("files", "OK", f"{module}: {where}", "config [hw] 폴더" if folder in where.parents else "설치된 패키지")
        except Exception as exc:  # noqa: BLE001
            r.add("files", "FAIL", f"{module}를 불러올 수 없음", f"{exc} → config [hw] {key} 경로 확인")
    try:
        from reachy_motion.library.registry import default_library, motions_dir
        from reachy_motion.safety.checker import prepare

        library = default_library()
        missing = library.missing()
        bad = [s.key for s in library if not prepare(s).runnable]
        sounds = sorted(p.name for p in (motions_dir() / "sounds").glob("*.wav"))
        ok = not missing and not bad and len(sounds) >= 4
        r.add("files", "OK" if ok else "FAIL", f"HW4 모션 {len(library)}개 (HW4 안전 검사 통과), 효과음 {len(sounds)}개",
              f"없음 {missing}, 검사 실패 {bad}" if not ok else str(motions_dir()))
    except Exception as exc:  # noqa: BLE001
        r.add("files", "FAIL", "HW4 모션 라이브러리", repr(exc))
    try:
        from reachy_cafe.robot.speech import load_hw3_config

        found = [p for p in cfg["speech"].get("hw3_configs", []) if resolve_hw3_file(cfg, p).is_file()]
        hw3 = load_hw3_config(cfg)
        r.add("files", "OK" if len(found) == len(cfg["speech"].get("hw3_configs", [])) else "WARN",
              f"HW3 설정 {found}", f"ASR {hw3.asr.model} ({hw3.asr.device}, {hw3.asr.compute_type}), "
              f"TTS {hw3.tts.provider} {hw3.tts.voice} gain {hw3.tts.gain_db} dB")
        try:
            import faster_whisper.utils as fwu
            from huggingface_hub import try_to_load_from_cache

            repo = fwu._MODELS.get(hw3.asr.model, hw3.asr.model)
            cached = isinstance(try_to_load_from_cache(repo, "model.bin"), str) or Path(hw3.asr.model).is_dir()
            r.add("files", "OK" if cached else "WARN", f"Whisper {hw3.asr.model} 모델 캐시",
                  "있음" if cached else "없음 → 첫 실행 때 약 1.6 GB를 내려받음 (인터넷 필요)")
            vad = Path(fwu.get_assets_path()) / "silero_vad_v6.onnx"
            r.add("files", "OK" if vad.is_file() else "FAIL", "Silero VAD v6 (faster-whisper 포함)", str(vad))
        except Exception as exc:  # noqa: BLE001
            r.add("files", "WARN", "Whisper·VAD 모델 확인 실패", repr(exc))
    except Exception as exc:  # noqa: BLE001
        r.add("files", "FAIL", "HW3 설정을 읽을 수 없음", repr(exc))
    face = find_face_model(cfg)
    r.add("files", "OK" if face else "FAIL", "얼굴 모델 face_landmarker.task",
          str(face) if face else "HW2 모델을 models\\에 복사 (README 2단계)")
    cache = ROOT / cfg["speech"].get("tts_cache_dir", "assets/tts_cache")
    try:
        cache.mkdir(parents=True, exist_ok=True)
        probe = cache / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        r.add("files", "OK", f"TTS 캐시 폴더 쓰기 가능: {cache}")
    except OSError as exc:
        r.add("files", "FAIL", f"TTS 캐시 폴더에 쓸 수 없음: {cache}", repr(exc))


# ---- 3 모델 실제 로딩 ------------------------------------------------------------------
def check_models(r: Report, cfg: dict[str, Any]) -> None:
    print("\n[3] 모델 실제 로딩 (--load-models, 1–2분)")
    import numpy as np

    from reachy_cafe.robot.speech import load_hw3_config

    hw3 = load_hw3_config(cfg)
    try:
        from reachy_s2s.engines.asr import WhisperASR

        asr = WhisperASR(hw3.asr)
        info = asr.load()
        t0 = time.perf_counter()
        asr.transcribe((0.001 * np.random.default_rng(1).standard_normal(32000)).astype(np.float32))
        ms = (time.perf_counter() - t0) * 1000
        r.add("models", "OK" if info.get("device") == "cuda" else "WARN",
              f"Whisper {info.get('model')} {info.get('device')} 로딩 {info.get('load_ms')} ms",
              f"2초 소리 인식 {ms:.0f} ms (HW3 기준 GPU면 수백 ms)")
    except Exception as exc:  # noqa: BLE001
        r.add("models", "FAIL", "Whisper 로딩 실패", f"{exc} (CUDA DLL 문제면 .[cuda] 설치 확인)")
    try:
        from dataclasses import replace

        from reachy_s2s.engines.tts_supertonic import SupertonicTTS

        ko = SupertonicTTS(hw3.tts)
        info = ko.load()
        res = ko.synthesize("안녕하세요! 리치 카페입니다. 주문 도와드릴까요?")
        en = SupertonicTTS(replace(hw3.tts, lang="en", normalize_numbers=False), engine=ko.engine)
        res_en = en.synthesize("Hello! Welcome to Reachy Cafe. May I take your order?")
        r.add("models", "OK", f"Supertonic {hw3.tts.voice} 로딩 {info.get('load_ms')} ms",
              f"한국어 {res.generation_ms:.0f} ms → {res.duration_s:.1f} s 음성, 영어 {res_en.generation_ms:.0f} ms")
    except Exception as exc:  # noqa: BLE001
        r.add("models", "FAIL", "Supertonic TTS 로딩 실패", repr(exc))
    try:
        from reachy_cafe.robot.vision import FacePerception

        face = find_face_model(cfg)
        lm = FacePerception(cfg, None, None, str(face) if face else None)._create_landmarker()
        import mediapipe as mp

        lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((480, 640, 3), np.uint8)), 1)
        lm.close()
        r.add("models", "OK", "MediaPipe FaceLandmarker (3명, VIDEO 모드) 실행")
    except Exception as exc:  # noqa: BLE001
        r.add("models", "FAIL", "MediaPipe 얼굴 인식 실행 실패", repr(exc))
    try:
        from reachy_s2s.engines.vad import SileroVadModel

        vad = SileroVadModel()
        vad(np.zeros(512, np.float32))
        r.add("models", "OK", "Silero VAD 실행")
    except Exception as exc:  # noqa: BLE001
        r.add("models", "FAIL", "Silero VAD 실행 실패", repr(exc))


# ---- 4 네트워크 -------------------------------------------------------------------------
def check_network(r: Report, cfg: dict[str, Any]) -> str:
    print("\n[4] 네트워크·키오스크 서버")
    from reachy_cafe.app import lan_ip

    ip = lan_ip()
    robot = str(cfg["robot"]["host"])
    local = robot.startswith("127.") or robot == "localhost"  # 같은 PC의 시뮬 daemon
    same = (ip.split(".")[:3] == robot.split(".")[:3]
            if robot.count(".") == 3 and ip.count(".") == 3 and not local else None)
    if ip == "localhost":
        r.add("net", "WARN", "이 PC의 네트워크 IP를 찾지 못함", "공유기(또는 핫스팟)에 연결돼 있는지 확인")
    elif same is False:
        r.add("net", "WARN", f"이 PC {ip} ↔ 로봇 {robot}: 다른 네트워크",
              "같은 공유기·핫스팟에 연결하거나, 로봇 IP가 바뀌었으면 --robot-host / config [robot] host")
    else:
        r.add("net", "OK", f"이 PC의 IP {ip}" + (f" (로봇 {robot}과 같은 네트워크)" if same else ""),
              f"Pi 4 pi_agent/kiosk.env의 BRAIN_HOST={ip}")
    port = int(cfg["server"]["port"])
    free = True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            free = False
    r.add("net", "OK" if free else "WARN", f"키오스크 서버 포트 {port} " + ("비어 있음" if free else "사용 중"),
          "" if free else "다른 프로그램(키오스크·HW3 대시보드 등)을 끄거나 --port로 바꾸기")
    r.add("net", "INFO", "방화벽 (Pi 4가 이 PC의 키오스크 포트로 접속)", FIREWALL_HINT[OS])
    return ip


# ---- 5 로봇 (읽기만) ----------------------------------------------------------------------
def check_robot(r: Report, cfg: dict[str, Any], fix_media: bool) -> bool:
    host, port = cfg["robot"]["host"], int(cfg["robot"].get("port", 8000))
    print(f"\n[5] 로봇 {host}:{port} (읽기만, 모터 명령 없음)")
    base = f"http://{host}:{port}"
    times, status = [], None
    for _ in range(5):
        t0 = time.perf_counter()
        try:
            status = http_json(f"{base}/api/daemon/status", timeout=3.0)
        except Exception as exc:  # noqa: BLE001
            r.add("robot", "FAIL", "로봇 daemon에 연결할 수 없음",
                  f"{type(exc).__name__}: IP(config [robot] host 또는 --robot-host), 같은 공유기, 로봇 전원 확인")
            return False
        times.append((time.perf_counter() - t0) * 1000)
    s = status or {}
    r.add("robot", "OK" if max(times) < 200 else "WARN", f"응답 시간 중앙값 {statistics.median(times):.0f} ms, "
          f"최대 {max(times):.0f} ms", "" if max(times) < 200 else "Wi-Fi가 느림: 공유기 가까이, 다른 기기 줄이기")
    ver, state = s.get("version"), s.get("state")
    r.add("robot", "OK" if ver == SDK_VERSION else "FAIL", f"daemon {ver}, 상태 {state}, 이름 {s.get('robot_name')}",
          "" if ver == SDK_VERSION else f"PC의 reachy-mini {SDK_VERSION}과 같아야 함")
    if s.get("simulation_enabled"):
        r.add("robot", "WARN", "시뮬레이터 daemon에 연결됨", "실제 로봇 IP인지 확인")
    backend = s.get("backend_status") or {}
    r.add("robot", "OK" if not backend.get("error") and not s.get("error") else "FAIL",
          f"모터 모드 {backend.get('motor_control_mode')}", str(backend.get("error") or s.get("error") or ""))
    if s.get("no_media"):
        r.add("robot", "FAIL", "daemon이 미디어 없이 실행 중 (no_media)", "카메라·마이크·스피커를 쓸 수 없음: 로봇 재시작")
    try:
        media = http_json(f"{base}/api/media/status", timeout=3.0)
        released = bool((media or {}).get("released"))
        if released and fix_media:
            http_json(f"{base}/api/media/acquire", timeout=15.0, data=b"")
            released = bool((http_json(f"{base}/api/media/status", timeout=3.0) or {}).get("released"))
            r.add("robot", "OK" if not released else "FAIL", "로봇 카메라·오디오 다시 잡음 (--fix-media)")
        else:
            r.add("robot", "FAIL" if released else "OK", "로봇 카메라·오디오 " + ("해제됨" if released else "사용 가능"),
                  "다른 프로그램이 해제한 채 끝남 → --fix-media" if released else "")
    except Exception as exc:  # noqa: BLE001
        r.add("robot", "WARN", "미디어 상태를 읽지 못함", repr(exc))
    try:
        kin = (http_json(f"{base}/api/kinematics/info", timeout=3.0) or {}).get("info", {})
        engine = str(kin.get("engine"))
        same = "analytical" in engine.lower()
        r.add("robot", "OK" if same else "WARN", f"daemon IK {engine}",
              "시선 레이어 관절 검사(JointGuard)와 같은 IK" if same else "JointGuard는 AnalyticalKinematics 기준: 여유를 더 둘 것")
    except Exception as exc:  # noqa: BLE001
        r.add("robot", "WARN", "IK 정보를 읽지 못함", repr(exc))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reachy Café 데스크톱 사전 점검 (로봇은 읽기만)")
    parser.add_argument("--config", action="append", default=[], help="덮어쓸 TOML")
    parser.add_argument("--robot-host", help="로봇 IP (기본: config robot.host / REACHY_HOST)")
    parser.add_argument("--no-robot", action="store_true", help="로봇 확인 생략")
    parser.add_argument("--load-models", action="store_true", help="Whisper·Supertonic·MediaPipe·VAD를 실제로 실행")
    parser.add_argument("--fix-media", action="store_true", help="로봇 카메라·오디오가 해제돼 있으면 다시 잡음")
    args = parser.parse_args(argv)
    cfg = load_config(*args.config)
    if args.robot_host:
        cfg["robot"]["host"] = args.robot_host
    r = Report()
    ensure_hw_paths(cfg)
    try:
        from reachy_cafe.robot.speech import load_hw3_config

        asr_device = load_hw3_config(cfg).asr.device  # GPU가 없으면 이미 cpu로 바뀌어 있음
    except Exception:  # noqa: BLE001 - HW3를 못 읽는 문제는 [2]에서 보고
        asr_device = cfg["speech"].get("asr_device") or "cuda"
    check_pc(r, asr_device)
    print("\n[1b] 로봇 카메라·오디오 경로 (GStreamer WebRTC)")
    check_media(r)
    check_files(r, cfg)
    if args.load_models:
        check_models(r, cfg)
    ip = check_network(r, cfg)
    robot_ok = None if args.no_robot else check_robot(r, cfg, args.fix_media)
    out = ROOT / "logs" / f"preflight_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"host": cfg["robot"]["host"], "lan_ip": ip, "items": r.items}, ensure_ascii=False,
                              indent=2), encoding="utf-8")
    fails, warns = r.count("FAIL"), r.count("WARN")
    print(f"\n결과: FAIL {fails}, WARN {warns} → {out}")
    nxt = {"windows": "start_windows.bat spike", "mac": "start_mac.command 의 3) 연결 점검",
           "linux": "./start_linux.sh spike"}[OS]
    if fails:
        print("FAIL 항목을 먼저 고친 뒤 다시 실행하세요.")
    elif robot_ok:
        print(f"준비 완료 → 다음: {nxt}")
    elif robot_ok is None:
        print("PC 준비 완료 (로봇은 확인하지 않음).")
    if not args.load_models and not fails:
        print("GPU 음성 인식까지 확인하려면 --load-models를 붙여 한 번 더 실행하세요.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
