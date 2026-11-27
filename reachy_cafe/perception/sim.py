"""시뮬레이션 얼굴 (로봇 카메라 없이 개발·리허설할 때 오퍼레이터 패널에서 고른다).

approach·passing은 시간에 따라 바뀐다 (프리셋을 고른 순간부터).
"""

from __future__ import annotations

from typing import Callable

from .human_state import Face


def _approach(dt: float) -> list[Face]:
    h = min(0.30, 0.07 + 0.09 * dt)          # 약 2.5초 동안 멀리서 가까이
    return [Face(1, h, 0.50, 0.45, yaw=0.0, pitch=0.0)]


def _passing(dt: float) -> list[Face]:
    if dt > 3.0:
        return []
    return [Face(1, 0.09, 0.05 + 0.30 * dt, 0.45, yaw=65.0, pitch=0.0)]  # 옆모습으로 지나감


STATIC: dict[str, list[Face] | None] = {
    "none": [],
    "one_near": [Face(1, 0.30, 0.50, 0.45, yaw=0.0)],
    "one_near_screen": [Face(1, 0.29, 0.52, 0.47, yaw=35.0, pitch=18.0)],
    "one_far": [Face(1, 0.10, 0.50, 0.45, yaw=0.0)],
    "one_away": [Face(1, 0.28, 0.50, 0.45, yaw=-55.0)],
    "two_near": [Face(1, 0.30, 0.40, 0.45, yaw=0.0), Face(2, 0.26, 0.66, 0.45, yaw=10.0)],
    "stale": None,  # 프레임이 끊긴 상황 (카메라 고장)
}
DYNAMIC: dict[str, Callable[[float], list[Face]]] = {"approach": _approach, "passing": _passing}
PRESETS = (*STATIC, *DYNAMIC)

LABELS = {
    "none": "없음", "one_near": "1명 가까이 (로봇 봄)", "one_near_screen": "1명 가까이 (화면 봄)",
    "one_far": "1명 멀리", "one_away": "1명 딴 곳 봄", "two_near": "2명", "stale": "프레임 끊김",
    "approach": "다가옴", "passing": "지나감",
}


class SimPerception:
    def __init__(self, preset: str = "none"):
        self.preset = preset if preset in PRESETS else "none"
        self.started = 0.0
        self._pending_start = True

    def set(self, preset: str) -> bool:
        if preset not in PRESETS:
            return False
        self.preset = preset
        self._pending_start = True
        return True

    def faces(self, now: float = 0.0) -> list[Face] | None:
        if self._pending_start:
            self.started = now
            self._pending_start = False
        if self.preset in DYNAMIC:
            return DYNAMIC[self.preset](now - self.started)
        faces = STATIC[self.preset]
        return None if faces is None else [Face(f.id, f.h, f.cx, f.cy, f.yaw, f.pitch) for f in faces]
