"""MotionPlayer (HW4 MotionService 확장)를 HW4 MockBackend 위에서 실제 시간으로 돌려 본다.

로봇 없이 확인하는 것: 시선 스트리밍 방향·범위, 키프레임 라벨 시각, 효과음 큐, 선점, 끄덕임 오프셋.
(실제 로봇에서의 지연·소리 경로는 tools/spike_connection.py로 따로 잰다.)
"""

from __future__ import annotations

import math
import time

import pytest

pytest.importorskip("reachy_motion")
pytest.importorskip("reachy_mini")

from reachy_motion.backends.mock import MockBackend  # noqa: E402
from reachy_motion.library.pose import Pose6, RobotState, pose_from_matrix  # noqa: E402
from reachy_motion.service import HandleState, Priority  # noqa: E402

from reachy_cafe.robot.motion import ENVELOPE, MotionPlayer, clamp_state  # noqa: E402


def make_player(**cfg):
    backend = MockBackend(structure="robot")
    backend.connect()
    events: list[dict] = []
    sounds: list[str] = []
    player = MotionPlayer(backend, motion_cfg={"label_delay_s": 0.0, **cfg}, sound_player=sounds.append)
    player.on_event(events.append)
    return backend, player, events, sounds


def wait_for(pred, timeout: float = 6.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def last_head(backend) -> Pose6:
    return pose_from_matrix(backend.commands[-1].head)


def test_gaze_streams_toward_screen_and_stays_in_range():
    backend, player, _events, _sounds = make_player()
    player.set_gaze("screen")
    player.start()
    try:
        time.sleep(1.5)
        head = last_head(backend)
        # 화면은 로봇의 왼쪽 아래 (+yaw = 로봇 왼쪽, +pitch = 숙임). yaw는 설정 한계 45°에서 잘린다
        assert 30.0 < math.degrees(head.yaw) <= 45.5
        assert 10.0 < math.degrees(head.pitch) < 30.0
        assert player.gaze_steps >= 65  # 1.5 s × 50 Hz = 75 (Windows 타이머 해상도에 막히지 않는지)
        sends = player.gaze_sends
        time.sleep(1.0)  # 목표에 도착한 뒤에는 keepalive(0.5 s)로만 보낸다
        assert player.gaze_sends - sends <= 3
    finally:
        player.close()


def test_user_ray_is_followed_but_clamped():
    backend, player, _events, _sounds = make_player()
    player.set_gaze("user")
    player.set_user_ray([0.2, 1.0, 0.0])  # 거의 옆(약 79°): 안전 범위로 잘려야 한다
    player.start()
    try:
        wait_for(lambda: player.gaze_sends > 60, 3.0)
        yaw = math.degrees(last_head(backend).yaw)
        assert 40.0 < yaw <= 45.5
    finally:
        player.close()


def test_motion_labels_follow_yaml_times_and_sound_fires_once():
    _backend, player, events, sounds = make_player()
    player.start()
    try:
        time.sleep(0.3)  # 시선이 REST에서 벗어나 있어 전이(transition)가 앞에 붙는 경우
        h = player.play("success", "expressive")
        assert h.wait(8.0)
        assert h.state == HandleState.DONE
        started = next(e for e in events if e["kind"] == "started" and e["id"] == h.id)
        labels = [e for e in events if e["kind"] == "label" and e["id"] == h.id]
        assert [e["label"] for e in labels] == ["crouch", "pop", "nod-down", "nod-up", "back"]
        nod = next(e for e in labels if e["label"] == "nod-down")
        expected = started["offset_s"] + 0.95  # success.expressive.yaml head 트랙: nod-down이 0.95 s에 출발
        assert abs((nod["t"] - started["t"]) - expected) < 0.15
        assert sounds == ["success.wav"]
    finally:
        player.close()


def test_idle_motion_is_cancelled_and_preempted():
    _backend, player, _events, _sounds = make_player()
    player.start()
    try:
        h1 = player.play("boredom", "expressive", priority=Priority.IDLE)
        assert wait_for(lambda: h1.state == HandleState.RUNNING, 3.0)
        player.cancel_idle_motion()  # 사람이 나타남 → 지루함 모션을 바로 멈춘다
        assert h1.wait(2.0) and h1.state == HandleState.CANCELLED
        h2 = player.play("boredom", "expressive", priority=Priority.IDLE)
        assert wait_for(lambda: h2.state == HandleState.RUNNING, 3.0)
        h3 = player.play("success", "expressive", priority=Priority.NORMAL, interrupt=True)
        assert h2.wait(2.0) and h2.state == HandleState.PREEMPTED
        assert h3.wait(6.0) and h3.state == HandleState.DONE
    finally:
        player.close()


def test_nod_gesture_adds_a_visible_pitch_offset():
    backend, player, _events, _sounds = make_player()
    player.set_gaze("still")
    player.start()
    try:
        time.sleep(0.4)
        base = math.degrees(last_head(backend).pitch)
        player.gesture("nod")
        peak = base
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.7:
            peak = max(peak, math.degrees(last_head(backend).pitch))
            time.sleep(0.01)
        assert peak - base > 3.0
    finally:
        player.close()


def test_posture_offset_does_not_accumulate_while_holding_gaze():
    # 회귀: '그대로 보기'에서 현재 자세를 목표로 쓰면 생각하는 자세(yaw +8°)가 매 틱 더해져 머리가 흘러갔다
    backend, player, _events, _sounds = make_player()
    player.set_gaze("still")
    player.set_posture("thinking")
    player.start()
    try:
        time.sleep(1.2)
        yaw1 = math.degrees(last_head(backend).yaw)
        time.sleep(1.0)
        yaw2 = math.degrees(last_head(backend).yaw)
        assert 6.0 < yaw2 < 10.0  # 시작 방향(정면) + 자세 오프셋 8°
        assert abs(yaw2 - yaw1) < 0.5
    finally:
        player.close()


def test_park_returns_to_rest_and_stays_there():
    # 회귀 (MuJoCo 점검에서 발견): stop(blend_to_rest)만 하면 전이 뒤 시선 레이어가 다시 화면 쪽으로 끌어갔다
    backend, player, _events, _sounds = make_player()
    player.set_gaze("screen")
    player.set_posture("thinking")
    player.start()
    try:
        time.sleep(1.0)
        h = player.park()
        assert h is not None and h.wait(6.0)
        time.sleep(1.0)
        head = last_head(backend)
        assert abs(math.degrees(head.yaw)) < 0.5 and abs(math.degrees(head.pitch)) < 0.5
        assert abs(math.degrees(head.roll)) < 0.5
    finally:
        player.close()


def test_clamp_state_keeps_targets_inside_the_envelope():
    wild = RobotState(head=Pose6.from_deg_mm(roll_deg=50, pitch_deg=60, yaw_deg=-80, z_mm=40), antennas=(4.0, -4.0))
    s = clamp_state(wild)
    assert s.head.pitch == pytest.approx(ENVELOPE["pitch"])
    assert s.head.yaw == pytest.approx(-ENVELOPE["yaw"])
    assert s.head.roll == pytest.approx(ENVELOPE["roll"])
    assert s.head.z == pytest.approx(ENVELOPE["z"][1])
    assert s.antennas == (ENVELOPE["antenna"], -ENVELOPE["antenna"])
