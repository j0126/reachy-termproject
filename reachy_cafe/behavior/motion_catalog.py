"""모션의 길이와 head 트랙 키프레임 라벨의 시작 시각 (변형별).

YAML의 head 트랙에서 duration_s + hold_s를 누적해 계산한 값이다. HW4 모션 네 개는 vendor/hw4/motions/,
카페에서 더한 greeting(손님 인사)은 termproject/motions/에 있다. mock 로봇과 BehaviorManager가 쓰고,
실제 로봇에서는 MotionPlayer(HW4 MotionService 포크)가 YAML에서 직접 라벨 이벤트를 낸다.
tests/test_greeting.py가 이 표를 YAML과 비교한다 (YAML을 고치면 여기도 고칠 것).
neutral 변형에는 expressive와 다른 라벨이 있다. 없는 라벨은 기다리지 않는다 (RealMotionHandle과 같음).
"""

from __future__ import annotations

MOTIONS: dict[str, dict[str, dict]] = {
    "aha_recommend": {
        "expressive": {"duration": 3.5, "labels": {"think": 0.0, "pop": 0.8, "present": 1.7, "back": 2.8}},
        "neutral": {"duration": 2.6, "labels": {"present": 0.0, "back": 1.7}},
    },
    "boredom": {
        "expressive": {"duration": 7.9, "labels": {"settle": 0.0, "look-left": 0.9, "look-right": 2.6,
                                                   "slump": 4.9, "back": 6.8}},
        "neutral": {"duration": 7.7, "labels": {"look-left": 0.0, "look-right": 2.45, "back": 6.25}},
    },
    "success": {
        "expressive": {"duration": 2.1, "labels": {"crouch": 0.0, "pop": 0.3, "nod-down": 0.95,
                                                   "nod-up": 1.25, "back": 1.55}},
        "neutral": {"duration": 2.0, "labels": {"nod-down": 0.0, "nod-up": 0.5, "nod-down-2": 1.0, "back": 1.5}},
    },
    "failure": {
        "expressive": {"duration": 3.7, "labels": {"flinch": 0.0, "slump": 0.2, "shake-left": 1.5,
                                                   "shake-right": 1.85, "center": 2.2, "recover": 2.8}},
        "neutral": {"duration": 2.0, "labels": {"left": 0.0, "right": 0.5, "center": 1.5}},
    },
    # 카페 모션: 손님이 오면 고개 숙여 인사 (termproject/motions/greeting.*.yaml, 초안)
    "greeting": {
        "expressive": {"duration": 2.75, "labels": {"notice": 0.0, "bow": 0.4, "rise": 1.3, "back": 2.25}},
        "neutral": {"duration": 1.9, "labels": {"bow": 0.0, "back": 1.1}},
    },
}


def timeline(name: str, variant: str = "expressive") -> tuple[float, dict[str, float]]:
    variants = MOTIONS.get(name)
    if variants is None:
        return 1.0, {}
    entry = variants.get(variant) or variants["expressive"]
    return float(entry["duration"]), dict(entry["labels"])
