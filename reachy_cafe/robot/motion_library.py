"""HW4 모션 라이브러리 + 카페에서 더한 모션 (termproject/motions/).

HW4 모션 네 개(aha_recommend, boredom, success, failure)는 ../reachy_hw4/motions/의 YAML을 그대로 쓰고,
카페가 더한 모션(greeting = 손님 인사)은 같은 형식(reachy-motion/v1)으로 이 프로젝트의 motions/ 폴더에 둔다.
HW4 폴더는 고치지 않는다.

HW4 `MotionSpec.validate()`는 과제의 모션 이름 네 개(`reachy_motion.library.spec.MOTION_NAMES`)만 받는다.
카페 모션도 HW4 planner·safety checker(prepare)를 그대로 거치게, 이 프로세스 안에서만 그 이름 목록에
카페 모션 이름을 더한다 (`allow_cafe_motions`). 다른 검사(키 길이, 정책: 관절 여유·속도·범위)는 그대로다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CAFE_MOTIONS_DIR = ROOT / "motions"
CAFE_MOTION_NAMES: tuple[str, ...] = ("greeting",)


def allow_cafe_motions() -> None:
    """HW4 이름 검사(MotionSpec.validate)가 카페 모션 이름도 받게 한다. 여러 번 불러도 같다."""
    from reachy_motion.library import spec as hw4_spec

    extra = tuple(name for name in CAFE_MOTION_NAMES if name not in hw4_spec.MOTION_NAMES)
    if extra:
        hw4_spec.MOTION_NAMES = (*hw4_spec.MOTION_NAMES, *extra)


def cafe_motion_files(directory: Path | None = None) -> list[Path]:
    d = Path(directory) if directory is not None else CAFE_MOTIONS_DIR
    return sorted(d.glob("*.yaml")) if d.is_dir() else []


def cafe_library(hw4_dir: Path | None = None, cafe_dir: Path | None = None) -> Any:
    """HW4 모션(기본: HW4 motions_dir) + 카페 모션을 한 MotionLibrary로. 이름이 겹치면 ValueError (HW4 register)."""
    allow_cafe_motions()
    from reachy_motion.library.io import load_spec_file
    from reachy_motion.library.registry import MotionLibrary, motions_dir

    d = Path(hw4_dir) if hw4_dir is not None else motions_dir()
    if not d.is_dir():
        raise FileNotFoundError(f"HW4 motion folder not found: {d}")
    files = sorted(d.glob("*.yaml")) + cafe_motion_files(cafe_dir)
    return MotionLibrary(load_spec_file(f) for f in files)
