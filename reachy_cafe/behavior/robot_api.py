"""BehaviorManager가 로봇에 기대하는 인터페이스. MockRobot과 실제 로봇(2단계)이 같은 모양을 구현한다.

2단계 실제 로봇 구현의 연결 지점 (IMPLEMENTATION_PLAN.md 2·5장):
- 연결 하나: ReachyMini(host, port=8000, connection_mode="network", media_backend="webrtc")를 공유
- play_motion / stop_motion / set_idle_motion / gaze / glance / posture
    → HW4 MotionService를 포크한 MotionPlayer (SharedMiniBackend, PreparedMotion 캐시, 라벨 이벤트, 시선 레이어)
- speak / stop_speech / is_speaking / listen_context
    → HW3 A1 기반 KioskPipeline (TTS 캐시 → AudioManager.play, clear_player로 바지인, 상태별 endpoint)
- 카메라 → HW2 FaceLandmarker 확장이 vision.faces 이벤트를 EventBus.publish_threadsafe로 보냄
- 음성 인식 → KioskPipeline이 speech.start / speech.utterance 이벤트를 보냄
"""

from __future__ import annotations

from typing import Protocol


class MotionHandle(Protocol):
    name: str

    async def wait_label(self, label: str, timeout: float) -> bool: ...

    async def wait_done(self, timeout: float | None = None) -> bool: ...


class RobotOutputs(Protocol):
    def play_motion(self, name: str, variant: str = "expressive") -> MotionHandle: ...

    def stop_motion(self, blend_to_rest: bool = True) -> None: ...

    def set_idle_motion(self, name: str | None, after_s: float = 20.0) -> None: ...

    def gaze(self, target: str) -> None: ...

    async def glance(self, target: str, hold_s: float) -> None: ...

    def posture(self, name: str) -> None: ...

    def gesture(self, name: str) -> None: ...

    async def speak(self, text: str, lang: str) -> None: ...

    def stop_speech(self) -> None: ...

    def is_speaking(self) -> bool: ...

    def listen_context(self, context: str) -> None: ...
