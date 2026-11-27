"""로봇 없이 개발할 때 쓰는 가짜 몸. 실제로 움직이는 대신 로그를 남기고, 말하기·모션 시간만 흉내 낸다."""

from __future__ import annotations

import asyncio
from typing import Any

from ..bus import EventLog
from .motion_catalog import timeline


class MockMotionHandle:
    def __init__(self, name: str, variant: str, duration: float, labels: dict[str, float], log: EventLog):
        self.name = name
        self.variant = variant
        self.duration = duration
        self.labels = labels
        self.is_idle = False
        self._log = log
        self._label_events = {label: asyncio.Event() for label in labels}
        self._done = asyncio.Event()
        self._task: asyncio.Task[Any] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        try:
            for label, at in sorted(self.labels.items(), key=lambda kv: kv[1]):
                await asyncio.sleep(max(0.0, t0 + at - loop.time()))
                self._label_events[label].set()
                self._log.log("robot.motion_label", motion=self.name, label=label)
            await asyncio.sleep(max(0.0, t0 + self.duration - loop.time()))
            self._log.log("robot.motion_done", motion=self.name)
        finally:
            self._release()

    def _release(self) -> None:
        for event in self._label_events.values():
            event.set()
        self._done.set()

    def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            self._log.log("robot.motion_cancel", motion=self.name)
        self._release()

    def done(self) -> bool:
        return self._done.is_set()

    async def wait_label(self, label: str, timeout: float) -> bool:
        event = self._label_events.get(label)
        if event is None:
            return False
        try:
            await asyncio.wait_for(event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_done(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._done.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False


class MockRobot:
    """robot_api.RobotOutputs의 mock 구현."""

    def __init__(self, log: EventLog, speech_rate: float = 1.0, time_scale: float = 1.0):
        self.log = log
        self.speech_rate = speech_rate
        self.time_scale = time_scale  # 테스트에서 시간을 줄일 때 (0.05 = 20배 빠르게)
        self.gaze_target = "wander"
        self.posture_name = "neutral"
        self.context = "open"
        self._motion: MockMotionHandle | None = None
        self._speaking = False
        self._idle_task: asyncio.Task[Any] | None = None

    # ---- 모션 ---------------------------------------------------------------
    def play_motion(self, name: str, variant: str = "expressive") -> MockMotionHandle:
        if self._motion is not None and not self._motion.done():
            self._motion.cancel()  # MotionPlayer처럼 새 모션이 이전 모션을 선점
        duration, labels = timeline(name, variant)
        duration *= self.time_scale
        labels = {label: at * self.time_scale for label, at in labels.items()}
        handle = MockMotionHandle(name, variant, duration, labels, self.log)
        self._motion = handle
        self.log.log("robot.motion", motion=name, variant=variant, duration=duration)
        handle.start()
        return handle

    def stop_motion(self, blend_to_rest: bool = True) -> None:
        if self._motion is not None and not self._motion.done():
            self._motion.cancel()
            self.log.log("robot.motion_stop", blend_to_rest=blend_to_rest)

    def set_idle_motion(self, name: str | None, after_s: float = 20.0) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None
        if self._motion is not None and self._motion.is_idle and not self._motion.done():
            self._motion.cancel()
        if name:
            self._idle_task = asyncio.create_task(self._idle_loop(name, after_s))
        self.log.log("robot.idle_motion", motion=name, after_s=after_s)

    async def _idle_loop(self, name: str, after_s: float) -> None:
        while True:
            await asyncio.sleep(after_s)
            if self._speaking or (self._motion is not None and not self._motion.done()):
                continue
            handle = self.play_motion(name)
            handle.is_idle = True
            await handle.wait_done()

    # ---- 시선·자세 ----------------------------------------------------------
    def gaze(self, target: str) -> None:
        if target != self.gaze_target:
            self.gaze_target = target
            self.log.log("robot.gaze", target=target)

    async def glance(self, target: str, hold_s: float) -> None:
        previous = self.gaze_target
        self.log.log("robot.glance", target=target, hold_s=hold_s)
        await asyncio.sleep((hold_s + 0.4) * self.time_scale)  # 왕복 시간 포함
        self.gaze_target = previous if previous != target else "user"

    def posture(self, name: str) -> None:
        if name != self.posture_name:
            self.posture_name = name
            self.log.log("robot.posture", posture=name)

    def gesture(self, name: str) -> None:
        # 실제 로봇에서는 MotionPlayer가 시선 레이어 위에 짧은 오프셋(끄덕임·안테나 흔들기)을 더한다
        self.log.log("robot.gesture", gesture=name)

    # ---- 음성 ---------------------------------------------------------------
    @staticmethod
    def estimate_duration(text: str, lang: str) -> float:
        if lang == "ko":
            seconds = 0.4 + len(text.replace(" ", "")) / 7.0
        else:
            seconds = 0.4 + len(text.split()) / 2.7
        return min(8.0, max(0.8, seconds))

    async def speak(self, text: str, lang: str) -> None:
        self._speaking = True
        try:
            await asyncio.sleep(self.estimate_duration(text, lang) / self.speech_rate * self.time_scale)
        finally:
            self._speaking = False

    def stop_speech(self) -> None:
        if self._speaking:
            self.log.log("robot.say_stop")
        self._speaking = False

    def is_speaking(self) -> bool:
        return self._speaking

    def listen_context(self, context: str) -> None:
        self.context = context
