"""Check this PC before running anything: Python and packages, API key, GPU and local models, the local LLM
server, and the robot. Prints what works, what does not and how to fix it. Sends no motor command.

    python -m reachy_s2s.apps.check_setup                 # everything (robot from .env REACHY_HOST / config)
    python -m reachy_s2s.apps.check_setup --host 192.168.0.20
    python -m reachy_s2s.apps.check_setup --no-robot      # PC only
    python -m reachy_s2s.apps.check_setup --fix-media     # also give the daemon its camera / audio back if released

Ends with the modes this PC can run: Full Evaluation (A0 / A1 / B) or Cloud B demo only.
Fixes: TROUBLESHOOTING.md. The API key value is never printed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, load_config

EXPECTED_PYTHON = (3, 12)
SDK_VERSION = "1.11.0"
results: list[tuple[str, str, str, str]] = []  # (group, status, item, detail)


def report(group: str, status: str, item: str, detail: str = "") -> None:
    results.append((group, status, item, detail))
    print(f"  [{status:4s}] {item}" + (f" — {detail}" if detail else ""))


def http_json(url: str, timeout: float = 3.0, data: bytes | None = None) -> Any:
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local robot / localhost only
        return json.loads(resp.read().decode("utf-8") or "null")


def version_of(module: str) -> str | None:
    try:
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001 - reported as missing
        return None
    try:
        from importlib.metadata import version

        return version({"reachy_mini": "reachy-mini", "PySide6": "PySide6-Essentials",
                        "google.genai": "google-genai"}.get(module, module))
    except Exception:  # noqa: BLE001
        return getattr(mod, "__version__", "installed")


def check_pc() -> dict[str, bool]:
    print("\n[1] PC / 설치")
    ok = {}
    v = sys.version_info
    ok["python"] = (v.major, v.minor) == EXPECTED_PYTHON
    report("pc", "OK" if ok["python"] else "FAIL", f"Python {v.major}.{v.minor}.{v.micro} ({platform.system()} "
           f"{platform.release()})", "" if ok["python"] else "Python 3.12가 필요합니다 (QUICK_START.md 1단계)")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    report("pc", "OK" if in_venv else "WARN", f"가상환경: {sys.prefix}", "" if in_venv else ".venv 밖에서 실행 중입니다")
    needed = {"reachy_mini": True, "PySide6": True, "numpy": True, "dotenv": True, "google.genai": True,
              "faster_whisper": False, "ctranslate2": False, "supertonic": False}
    missing_core = []
    for module, core in needed.items():
        ver = version_of(module)
        if ver is None:
            report("pc", "FAIL" if core else "WARN", f"패키지 {module}", "설치되지 않음 → setup.bat 다시 실행")
            if core:
                missing_core.append(module)
        elif module == "reachy_mini" and ver != SDK_VERSION:
            report("pc", "WARN", f"패키지 reachy-mini {ver}", f"로봇 데몬과 같은 {SDK_VERSION}이어야 합니다")
        else:
            report("pc", "OK", f"패키지 {module} {ver}")
    ok["packages"] = not missing_core
    env = PROJECT_ROOT / ".env"
    report("pc", "OK" if env.exists() else "WARN", ".env 파일",
           "" if env.exists() else ".env.example을 .env로 복사하고 값을 채우세요")
    key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    ok["api_key"] = key
    report("pc", "OK" if key else "FAIL", "GEMINI_API_KEY " + ("설정됨 (값은 표시하지 않음)" if key else "없음"),
           "" if key else "B(Cloud)를 쓰려면 .env에 GEMINI_API_KEY=... 를 넣으세요 (키는 공유·커밋 금지)")
    return ok


def check_local(cfg: Any) -> dict[str, bool]:
    print("\n[2] 로컬 A0 / A1 (GPU 필요: Full Evaluation)")
    ok = {}
    try:
        import ctranslate2

        n_gpu = ctranslate2.get_cuda_device_count()
    except Exception:  # noqa: BLE001
        n_gpu = 0
    gpu_name = ""
    if shutil.which("nvidia-smi"):
        try:
            gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                      capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            gpu_name = ""
    ok["gpu"] = n_gpu > 0
    report("local", "OK" if ok["gpu"] else "FAIL", f"NVIDIA GPU (CUDA) {n_gpu}개" + (f": {gpu_name}" if gpu_name else ""),
           "" if ok["gpu"] else "GPU가 없으면 A0/A1(로컬 ASR·LLM)을 실행할 수 없습니다. B만 쓰세요")
    exe = Path(os.path.expandvars(cfg.llm_server.exe))
    ok["llama_exe"] = exe.is_file()
    report("local", "OK" if ok["llama_exe"] else "FAIL", f"llama-server: {exe}",
           "" if ok["llama_exe"] else "llama.cpp Windows 빌드를 이 경로에 설치하세요 (QUICK_START.md Full Evaluation)")
    try:
        from huggingface_hub import try_to_load_from_cache

        gguf = try_to_load_from_cache(cfg.llm_server.model_repo, cfg.llm_server.model_file,
                                      revision=cfg.llm_server.model_revision)
        ok["gguf"] = isinstance(gguf, str)
        report("local", "OK" if ok["gguf"] else "WARN", f"LLM 모델 {cfg.llm_server.model_file}",
               "다운로드됨" if ok["gguf"] else "아직 없음 → start_llm_server.bat가 처음 한 번 내려받습니다(약 5 GB)")
        import faster_whisper.utils as fwu

        repo = fwu._MODELS.get(cfg.asr.model, cfg.asr.model)
        asr = try_to_load_from_cache(repo, "model.bin")
        ok["asr_model"] = isinstance(asr, str) or Path(cfg.asr.model).is_dir()
        report("local", "OK" if ok["asr_model"] else "WARN", f"ASR 모델 {cfg.asr.model}",
               "다운로드됨" if ok["asr_model"] else "아직 없음 → 처음 Load engines 때 내려받습니다(약 1.6 GB)")
    except Exception as exc:  # noqa: BLE001
        report("local", "WARN", "모델 캐시 확인 실패", repr(exc))
    base = cfg.llm.base_url.rstrip("/")
    try:
        models = http_json(f"{base}/models", timeout=2.0)
        names = [m.get("id") for m in (models or {}).get("data", [])]
        ok["llm_server"] = True
        report("local", "OK", f"로컬 LLM 서버 응답: {base}", ", ".join(n for n in names if n))
    except Exception:  # noqa: BLE001
        ok["llm_server"] = False
        report("local", "WARN", f"로컬 LLM 서버 꺼짐: {base}", "A0/A1 전에 start_llm_server.bat를 실행해 두세요")
    return ok


def check_robot(cfg: Any, fix_media: bool) -> dict[str, bool]:
    host, port = cfg.robot.host, cfg.robot.port
    print(f"\n[3] 로봇 {host}:{port} (읽기만 함, 모터 명령 없음)")
    ok = {"robot": False}
    base = f"http://{host}:{port}"
    try:
        status = http_json(f"{base}/api/daemon/status", timeout=3.0)
    except Exception as exc:  # noqa: BLE001
        report("robot", "FAIL", "로봇 데몬에 연결할 수 없음",
               f"{exc.__class__.__name__}: IP(.env REACHY_HOST), 같은 Wi-Fi, 로봇 전원을 확인하세요 (TROUBLESHOOTING.md 1)")
        return ok
    ok["robot"] = True
    ver = (status or {}).get("version")
    report("robot", "OK" if ver == SDK_VERSION else "WARN", f"데몬 {ver}, 상태 {(status or {}).get('state')}",
           "" if ver == SDK_VERSION else f"PC의 reachy-mini {SDK_VERSION}과 버전이 다릅니다")
    try:
        motors = http_json(f"{base}/api/motors/status", timeout=3.0)
        report("robot", "OK", f"모터 상태 {(motors or {}).get('mode', motors)}")
    except Exception as exc:  # noqa: BLE001
        report("robot", "WARN", "모터 상태를 읽지 못함", repr(exc))
    try:
        media = http_json(f"{base}/api/media/status", timeout=3.0)
        released = bool((media or {}).get("released"))
        if released and fix_media:
            http_json(f"{base}/api/media/acquire", timeout=15.0, data=b"")
            media = http_json(f"{base}/api/media/status", timeout=3.0)
            released = bool((media or {}).get("released"))
            report("robot", "OK" if not released else "FAIL", "로봇 카메라·오디오 다시 켬 (--fix-media)")
        else:
            report("robot", "FAIL" if released else "OK", "로봇 카메라·오디오(미디어)" + (" 해제됨" if released else " 사용 가능"),
                   "다른 프로그램이 해제한 채 끝났습니다 → check_setup.bat --fix-media" if released else "")
        ok["media"] = not released
    except Exception as exc:  # noqa: BLE001
        report("robot", "WARN", "미디어 상태를 읽지 못함", repr(exc))
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", help="robot IP (default: .env REACHY_HOST / config)")
    parser.add_argument("--config", action="append", help="extra config file(s)")
    parser.add_argument("--no-robot", action="store_true", help="skip the robot checks")
    parser.add_argument("--fix-media", action="store_true", help="re-acquire the daemon's camera / audio if released")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Korean text in a cp949 / cp1252 console
    print(f"Reachy HW3 환경 점검 — {PROJECT_ROOT}")
    try:  # the apps read .env through load_config; do it first so the key check sees it
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass
    pc = check_pc()
    if not pc["packages"]:
        print("\n[결과] 필수 패키지가 없습니다. setup.bat를 실행한 뒤 다시 점검하세요.")
        return 1
    cfg = load_config(args.config or ["config/local_e4b_supertonic.toml"], overrides={"robot": {"host": args.host}})
    local = check_local(cfg)
    robot = {"robot": False} if args.no_robot else check_robot(cfg, args.fix_media)

    full = pc["python"] and local.get("gpu") and local.get("llama_exe")
    print("\n[결과]")
    print(f"  Full Evaluation (A0 / A1 / B): {'가능' if full else '불가'}"
          + ("" if full else " — GPU 또는 llama-server가 없습니다. A0/A1 대신 B를 쓰세요"))
    if full and not local.get("llm_server"):
        print("    · A0/A1 대화 전에 start_llm_server.bat를 실행해 두세요")
    print(f"  Cloud B 데모: {'가능' if pc['api_key'] else '불가 (GEMINI_API_KEY 없음)'}")
    if not args.no_robot:
        print(f"  로봇 연결: {'가능' if robot.get('robot') else '불가'}"
              + ("" if robot.get("media", True) else " — 로봇 오디오가 꺼져 있음 (--fix-media)"))
    fails = [r for r in results if r[1] == "FAIL"]
    if fails:
        print(f"  해결할 항목 {len(fails)}개: TROUBLESHOOTING.md 참고")
    return 0 if pc["api_key"] or full else 1


if __name__ == "__main__":
    sys.exit(main())
