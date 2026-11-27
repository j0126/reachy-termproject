"""HW 코드(HW3 reachy_s2s, HW4 reachy_motion)를 import할 수 있게 한다.

기본은 이 저장소 안의 복사본 vendor/hw3·vendor/hw4다 (tools/vendor_hw.py, git clone 하나로 실행). config [hw]의
폴더를 sys.path 맨 앞에 넣어, PC에 따로 설치된 HW 패키지(예: 예전에 pip install -e로 넣은 원본)보다 먼저 쓴다.
그래서 노트북·데스크톱·NVIDIA 노트북이 모두 같은 코드로 돈다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..config import ROOT


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def ensure_hw_paths(cfg: dict[str, Any]) -> list[str]:
    """config [hw] 폴더를 sys.path 맨 앞에 넣고 넣은 경로 목록을 돌려준다. 폴더에 패키지가 없으면(경로를 비움)
    설치된 패키지를 그대로 쓴다. HW 모듈을 import하기 전에 불러야 한다 (이미 불러온 모듈은 바뀌지 않음)."""
    hw = cfg.get("hw", {})
    added: list[str] = []
    for module, key in (("reachy_motion", "reachy_hw4"), ("reachy_s2s", "hw3")):
        folder = _resolve(hw.get(key) or "")
        if not (folder / module).is_dir() or str(folder) in sys.path:
            continue
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
