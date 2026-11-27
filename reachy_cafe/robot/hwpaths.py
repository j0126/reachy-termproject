"""HW 코드(HW3 reachy_s2s, HW4 reachy_motion)를 import할 수 있게 한다.

데스크톱에서는 `uv pip install -e ..\\reachy_hw4 -e ..\\hw3\\reachy-hw3-main`로 설치하는 것을 권장한다.
설치하지 않았으면 config [hw]의 경로를 sys.path에 넣는다 (HW 폴더는 수정하지 않는다).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from ..config import ROOT


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def ensure_hw_paths(cfg: dict[str, Any]) -> list[str]:
    """설치되지 않은 HW 패키지의 폴더를 sys.path 앞에 넣고, 넣은 경로 목록을 돌려준다."""
    hw = cfg.get("hw", {})
    added: list[str] = []
    for module, key in (("reachy_motion", "reachy_hw4"), ("reachy_s2s", "hw3")):
        if importlib.util.find_spec(module) is not None:
            continue
        folder = _resolve(hw.get(key, ""))
        if (folder / module).is_dir():
            sys.path.insert(0, str(folder))
            added.append(str(folder))
    return added


def find_face_model(cfg: dict[str, Any]) -> Path | None:
    """MediaPipe face_landmarker.task (HW2 모델 파일). 후보 경로를 차례로 찾는다."""
    candidates = cfg.get("hw", {}).get("face_model", [])
    if isinstance(candidates, str):
        candidates = [candidates]
    for candidate in candidates:
        path = _resolve(candidate)
        if path.is_file():
            return path
    return None


def resolve_hw3_file(cfg: dict[str, Any], relative: str) -> Path:
    return _resolve(Path(cfg.get("hw", {}).get("hw3", "")) / relative)
