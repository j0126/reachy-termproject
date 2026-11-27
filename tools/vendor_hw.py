"""HW2·HW3·HW4에서 키오스크가 쓰는 부분을 이 저장소의 vendor/로 복사한다 (git clone 하나로 실행되게).

  vendor/hw4/   ← reachy_hw4/          reachy_motion/ (모션 실행·안전 검사), motions/ (YAML + 효과음)
  vendor/hw3/   ← hw3/reachy-hw3-main/ reachy_s2s/ (VAD·ASR·TTS·오디오), config/ (HW3에서 검증한 설정)
  models/face_landmarker.task ← HW2(reachy-chamchamcham) 얼굴 모델 (MediaPipe)

원본 HW 폴더는 읽기만 한다. 무엇을 가져오고 무엇을 빼는지(캐시·.env·egg-info)는 tools/make_bundle.py와 같다.
HW 코드가 바뀌면 다시 실행한다 (vendor/를 지우고 새로 복사):

  .venv\\Scripts\\python -X utf8 tools\\vendor_hw.py
  .venv\\Scripts\\python -X utf8 tools\\vendor_hw.py --hw4 D:\\reachy_hw4 --hw3 D:\\hw3\\reachy-hw3-main --face D:\\face_landmarker.task
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.make_bundle import HW_PARTS, skipped  # noqa: E402

VENDOR = ROOT / "vendor"
FACE_TARGET = ROOT / "models" / "face_landmarker.task"
DEFAULTS = {"hw4": ROOT.parent / "reachy_hw4", "hw3": ROOT.parent / "hw3" / "reachy-hw3-main"}
FACE_CANDIDATES = [ROOT.parent.parent / "reachy-chamchamcham" / "models" / "face_landmarker.task",
                   ROOT.parent / "hw2" / "reachy-chamchamcham" / "models" / "face_landmarker.task"]
PARTS = {"hw4": HW_PARTS["reachy_hw4"], "hw3": HW_PARTS["hw3/reachy-hw3-main"]}


def copy_tree(src: Path, dst: Path) -> int:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return 1
    count = 0
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if not path.is_file() or skipped(rel):
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HW 코드를 vendor/로 복사")
    parser.add_argument("--hw4", type=Path, default=DEFAULTS["hw4"])
    parser.add_argument("--hw3", type=Path, default=DEFAULTS["hw3"])
    parser.add_argument("--face", type=Path, default=None, help="face_landmarker.task (기본: HW2 폴더에서 찾음)")
    args = parser.parse_args(argv)

    sources = {"hw4": args.hw4.resolve(), "hw3": args.hw3.resolve()}
    for key, folder in sources.items():
        missing = [p for p in PARTS[key] if not (folder / p).exists() and p.endswith(("_motion", "_s2s"))]
        if missing or not folder.is_dir():
            print(f"[{key}] {folder}에 {', '.join(PARTS[key][:2])}가 없습니다 (--{key}로 위치를 알려 주세요)")
            return 1
    face = args.face or next((p for p in FACE_CANDIDATES if p.is_file()), None)
    if face is None or not face.is_file():
        print("HW2 얼굴 모델(face_landmarker.task)을 찾지 못했습니다 (--face로 위치를 알려 주세요)")
        return 1

    if VENDOR.exists():
        shutil.rmtree(VENDOR)
    lines = []
    for key, folder in sources.items():
        total = 0
        for part in PARTS[key]:
            if (folder / part).exists():
                total += copy_tree(folder / part, VENDOR / key / part)
        lines.append(f"- `vendor/{key}/` ← `{folder.name if key == 'hw4' else 'hw3/' + folder.name}`: "
                     f"{', '.join(PARTS[key])} ({total}개 파일)")
        print(f"[{key}] {folder} → vendor/{key} ({total}개 파일)")
    FACE_TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(face, FACE_TARGET)
    print(f"[hw2] {face} → models/face_landmarker.task ({face.stat().st_size / 1e6:.1f} MB)")

    (VENDOR / "README.md").write_text(
        "# vendor: 키오스크가 쓰는 HW 코드의 복사본\n\n"
        "`git clone` 하나로 실행되도록 HW3·HW4 코드에서 키오스크가 쓰는 부분을 여기 복사했다 "
        f"({time.strftime('%Y-%m-%d')}, `tools/vendor_hw.py`). `config/default.toml`의 `[hw]`가 이 폴더를 가리키고, "
        "`reachy_cafe/robot/hwpaths.py`가 앱 시작 때 sys.path 앞에 넣는다.\n\n"
        + "\n".join(lines)
        + "\n- `models/face_landmarker.task` ← HW2(reachy-chamchamcham) 얼굴 모델 (MediaPipe)\n\n"
        "여기서 고치지 않는다. 원본 HW 폴더에서 고친 뒤 `tools/vendor_hw.py`로 다시 복사한다. "
        "의존 패키지는 이 저장소의 `pyproject.toml`(robot·speech·cuda·sim)에 HW3·HW4와 같은 버전으로 들어 있다.\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
