"""git 없이 옮길 묶음을 OS별로 만든다: 키오스크 코드 + HW3·HW4 코드 + HW2 얼굴 모델 + 그 OS의 시작 스크립트.

11/27부터 HW 코드는 저장소 안의 vendor/에 들어 있어(tools/vendor_hw.py) git clone 하나로도 실행된다. 이 묶음은
git을 쓸 수 없을 때를 위한 것이다. config [hw]가 저장소 밖의 HW 폴더를 가리킬 때만 그 폴더를 따로 넣는다.

  ReachyCafe/
    termproject/            키오스크 (vendor/hw3·vendor/hw4, models/face_landmarker.task, start_<os> 스크립트 포함)
    README_<os>.txt

  windows → reachy_cafe_windows_<날짜>.zip      (start_windows.bat, 줄바꿈 CRLF)
  mac     → reachy_cafe_mac_<날짜>.zip          (start_mac.command, 실행 권한 포함)
  linux   → reachy_cafe_linux_<날짜>.tar.gz     (start_linux.sh + scripts/linux_gstreamer_webrtc.sh, 실행 권한 포함)

빼는 것: 가상환경, 로그·결과·상태, 캐시(__pycache__, TTS 캐시), 비밀(.env, kiosk.env), 발표 자료(pptx), 보정 파일.

  .venv\\Scripts\\python -X utf8 tools\\make_bundle.py              # 세 OS 모두 → dist\\
  .venv\\Scripts\\python -X utf8 tools\\make_bundle.py --os mac     # 하나만
  .venv\\Scripts\\python -X utf8 tools\\make_bundle.py --no-hw      # 데스크톱에 HW 폴더가 이미 같은 배치로 있을 때
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reachy_cafe.config import load_config  # noqa: E402
from reachy_cafe.robot.hwpaths import _resolve, find_face_model  # noqa: E402

TOP = "ReachyCafe"
OSES = ("windows", "mac", "linux")
SKIP_DIRS = {".venv", "venv", "__pycache__", ".pytest_cache", ".git", ".idea", ".vscode", ".claude", "logs", "state",
             "dist", "results", "node_modules", "tts_cache"}
SKIP_FILES = ["*.pyc", "*.pyo", ".env", "kiosk.env", "*.pptx", "*.task", "Thumbs.db", ".DS_Store", "*.egg-info",
              "calibration.toml"]  # 보정 값은 그 자리(데스크톱·배치)에서 잰 것만 쓴다
# 시작 스크립트는 그 OS 묶음에만 넣는다
OS_ONLY = {"start_windows.bat": "windows", "start_mac.command": "mac", "start_linux.sh": "linux",
           "scripts/linux_gstreamer_webrtc.sh": "linux"}
EXECUTABLE = (".sh", ".command")
HW_PARTS = {
    "reachy_hw4": ["reachy_motion", "motions", "pyproject.toml", "README.md"],
    "hw3/reachy-hw3-main": ["reachy_s2s", "config", "pyproject.toml", "README.md", "TROUBLESHOOTING.md"],
}
README = {
    "windows": """Reachy Café 데스크톱 묶음 - Windows ({date})

필요: Windows 10/11 64비트, 인터넷(처음 설치 때), uv. NVIDIA GPU가 있으면 음성 인식이 GPU에서 돈다 (없으면 CPU).
1) 이 폴더(ReachyCafe)를 새 폴더에 둡니다. 예: C:\\ReachyCafe  (기존 HW3 폴더 위에 덮어쓰지 마세요)
2) termproject\\start_windows.bat 를 더블클릭 → 메뉴 1) 설치. PowerShell에서는:
     cd C:\\ReachyCafe\\termproject
     .\\start_windows.bat setup
     .\\start_windows.bat preflight
3) 점검 순서와 할 일: termproject\\docs\\desktop_check.md
""",
    "mac": """Reachy Café 데스크톱 묶음 - macOS ({date})

필요: Apple Silicon(M1 이상) + macOS 14 이상, 인터넷(처음 설치 때), uv. 음성 인식은 CPU에서 돈다 (맥에는 CUDA 없음).
1) 이 폴더(ReachyCafe)를 원하는 곳에 둡니다. 예: ~/ReachyCafe
2) termproject/start_mac.command 를 더블클릭 → 메뉴 1) 설치.
   "확인되지 않은 개발자" 경고로 열리지 않으면: 우클릭 > 열기, 또는 터미널에서
     xattr -dr com.apple.quarantine ~/ReachyCafe
     chmod +x ~/ReachyCafe/termproject/start_mac.command
3) 처음 로봇에 연결할 때 macOS가 '로컬 네트워크' 접근을 물으면 허용하세요.
4) 점검 순서와 할 일: termproject/docs/desktop_check.md
""",
    "linux": """Reachy Café 데스크톱 묶음 - Linux ({date})

필요: Ubuntu 24.04(x86_64) 권장 (GStreamer 1.22+), 인터넷(처음 설치 때), uv. NVIDIA GPU가 있으면 GPU 음성 인식.
리눅스는 GStreamer를 직접 설치해야 한다 (Windows·맥은 pip 패키지에 들어 있음).
1) 풀기:  tar xzf reachy_cafe_linux_*.tar.gz -C ~   (→ ~/ReachyCafe)
2) 시스템 패키지 + 로봇 카메라·마이크용 GStreamer webrtc 플러그인 (한 번):
     cd ~/ReachyCafe/termproject
     ./scripts/linux_gstreamer_webrtc.sh
3) 설치와 점검:
     ./start_linux.sh setup
     ./start_linux.sh preflight
4) 점검 순서와 할 일: termproject/docs/desktop_check.md
""",
}


def skipped(rel: Path) -> bool:
    if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts[:-1]):
        return True
    return any(fnmatch.fnmatch(rel.name, pattern) for pattern in SKIP_FILES)


def collect(target_os: str, with_hw: bool = True) -> tuple[list[tuple[str, bytes, int]], Path | None]:
    """(묶음 안 경로, 내용, 권한) 목록. 줄바꿈은 .bat → CRLF, .sh·.command → LF로 맞춘다."""
    cfg = load_config()
    items: list[tuple[str, bytes, int]] = []

    def add_file(path: Path, arc: str) -> None:
        data = path.read_bytes()
        if arc.endswith(".bat"):
            data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        elif arc.endswith(EXECUTABLE):
            data = data.replace(b"\r\n", b"\n")
        items.append((arc, data, 0o755 if arc.endswith(EXECUTABLE) else 0o644))

    def add_tree(src: Path, arc: str, project: bool = False) -> None:
        if src.is_file():
            add_file(src, arc)
            return
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(src)
            if skipped(rel):
                continue
            only = OS_ONLY.get(rel.as_posix()) if project else None
            if only is not None and only != target_os:
                continue
            add_file(path, f"{arc}/{rel.as_posix()}")

    add_tree(ROOT, f"{TOP}/termproject", project=True)
    face = find_face_model(cfg)
    if face is not None:
        add_file(face, f"{TOP}/termproject/models/face_landmarker.task")
    if with_hw:
        for key, rel_name in (("reachy_hw4", "reachy_hw4"), ("hw3", "hw3/reachy-hw3-main")):
            folder = _resolve(cfg["hw"][key])
            if folder.is_relative_to(ROOT):  # vendor/ 복사본: 위에서 termproject와 함께 이미 들어감
                continue
            for part in HW_PARTS[rel_name]:
                src = folder / part
                if src.exists():
                    add_tree(src, f"{TOP}/{rel_name}/{part}")
    text = README[target_os].format(date=time.strftime("%Y-%m-%d %H:%M"))
    if target_os == "windows":
        text = text.replace("\n", "\r\n")
    items.append((f"{TOP}/README_{target_os}.txt", text.encode("utf-8"), 0o644))
    return items, face


def write_archive(items: list[tuple[str, bytes, int]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    now = time.localtime()
    if out.name.endswith(".tar.gz"):
        with tarfile.open(out, "w:gz") as tf:
            for arc, data, mode in items:
                info = tarfile.TarInfo(arc)
                info.size, info.mode, info.mtime = len(data), mode, time.mktime(now)
                tf.addfile(info, io.BytesIO(data))
        return
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for arc, data, mode in items:
            zi = zipfile.ZipInfo(arc, date_time=now[:6])
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.create_system = 3  # Unix: 맥 압축 해제(Archive Utility)·unzip이 실행 권한을 살린다
            zi.external_attr = (0o100000 | mode) << 16
            zf.writestr(zi, data)


def build(target_os: str, out_dir: Path, with_hw: bool = True, stamp: str | None = None) -> dict[str, object]:
    items, face = collect(target_os, with_hw)
    stamp = stamp or time.strftime("%Y%m%d_%H%M")
    out = out_dir / f"reachy_cafe_{target_os}_{stamp}.{'tar.gz' if target_os == 'linux' else 'zip'}"
    write_archive(items, out)
    return {"os": target_os, "archive": str(out), "files": len(items),
            "raw_mb": round(sum(len(d) for _a, d, _m in items) / 1e6, 1), "mb": round(out.stat().st_size / 1e6, 1),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "face_model": str(face) if face else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="데스크톱으로 옮길 Reachy Café 묶음 (OS별)")
    parser.add_argument("--os", choices=(*OSES, "all"), default="all", help="만들 OS (기본: 셋 다)")
    parser.add_argument("--out-dir", default=str(ROOT / "dist"), help="저장 폴더 (기본 dist/)")
    parser.add_argument("--no-hw", action="store_true", help="HW3·HW4 코드를 넣지 않음")
    args = parser.parse_args(argv)
    stamp = time.strftime("%Y%m%d_%H%M")
    for target in (OSES if args.os == "all" else (args.os,)):
        info = build(target, Path(args.out_dir), with_hw=not args.no_hw, stamp=stamp)
        print(f"{info['os']:8s} {info['archive']}  ({info['files']} files, {info['mb']} MB)  sha256 {info['sha256']}")
        if info["face_model"] is None:
            print("  주의: 얼굴 모델을 찾지 못해 넣지 못했습니다 (config [hw] face_model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
