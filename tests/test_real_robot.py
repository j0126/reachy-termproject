"""RealRobot의 모션 경로: MotionPlayer(HW4 MockBackend) 이벤트가 asyncio 핸들의 라벨·완료로 넘어오는지.

연결·음성·카메라 없이 player만 붙여서 BehaviorManager가 쓰는 wait_label / wait_done 동작을 확인한다.
"""

from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("reachy_motion")
pytest.importorskip("reachy_mini")

from reachy_motion.backends.mock import MockBackend  # noqa: E402

from reachy_cafe.bus import EventBus, EventLog  # noqa: E402
from reachy_cafe.config import load_config  # noqa: E402
from reachy_cafe.robot.motion import MotionPlayer  # noqa: E402
from reachy_cafe.robot.real import RealRobot  # noqa: E402


def make_robot():
    cfg = load_config(env={})
    log = EventLog(None)
    records: list[dict] = []
    log.add_listener(records.append)
    bus = EventBus(log)
    robot = RealRobot(cfg, bus)
    backend = MockBackend(structure="robot")
    backend.connect()
    robot.player = MotionPlayer(backend, motion_cfg={"label_delay_s": 0.0})
    robot.player.on_event(robot._on_motion_event)
    return robot, bus, records


def test_labels_and_completion_reach_asyncio_handles():
    robot, bus, records = make_robot()

    async def run() -> None:
        loop = asyncio.get_running_loop()
        bus.bind_loop(loop)
        robot._loop = loop
        robot.player.start()
        try:
            t0 = time.perf_counter()
            handle = robot.play_motion("success", "expressive")
            assert await handle.wait_label("nod-down", 3.0)
            assert time.perf_counter() - t0 > 0.9  # YAML 0.95 s (+ 전이)
            assert not await handle.wait_label("slump", 2.5)  # success에는 slump가 없음 → 바로 False
            assert await handle.wait_done(5.0)
            assert handle.state == "finished"
            bad = robot.play_motion("no_such_motion", "expressive")
            assert await bad.wait_done(1.0) and bad.state == "rejected"
        finally:
            robot.player.close()

    asyncio.run(run())
    kinds = [r["kind"] for r in records]
    assert "robot.motion_started" in kinds and "robot.motion_label" in kinds and "robot.motion_finished" in kinds


def test_new_motion_preempts_the_running_one():
    robot, bus, _records = make_robot()

    async def run() -> None:
        loop = asyncio.get_running_loop()
        bus.bind_loop(loop)
        robot._loop = loop
        robot.player.start()
        try:
            first = robot.play_motion("boredom", "expressive")
            await asyncio.sleep(0.6)
            second = robot.play_motion("success", "expressive")
            assert await first.wait_done(2.0) and first.state == "preempted"
            assert await second.wait_done(5.0) and second.state == "finished"
        finally:
            robot.player.close()

    asyncio.run(run())


def test_without_motion_the_fallback_keeps_timing():
    cfg = load_config(env={})
    robot = RealRobot(cfg, EventBus(EventLog(None)))

    async def run() -> None:
        handle = robot.play_motion("aha_recommend", "expressive")  # player 없음 → mock 시간표
        assert await handle.wait_label("present", 3.0)

    asyncio.run(run())
