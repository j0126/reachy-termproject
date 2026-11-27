"""로봇 첫 연결의 모터 켜는 순서 (HW4 backends/robot_connect.py와 같게): 가짜 ReachyMini로 호출 순서를 본다."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from reachy_cafe.robot.connection import RobotLink


class FakeMini:
    def __init__(self, motor_mode: str | None):
        self.calls: list[str] = []
        self.targets: list[dict[str, Any]] = []
        backend = SimpleNamespace(motor_control_mode=SimpleNamespace(value=motor_mode)) if motor_mode else None
        status = SimpleNamespace(backend_status=backend)
        self.client = SimpleNamespace(get_status=lambda wait=True, timeout=2.0: status)
        self.head = np.eye(4)
        self.head[2, 3] = -0.01  # 처진(limp) 머리

    def stop_head_tracking(self) -> None:
        self.calls.append("stop_head_tracking")

    def disable_wobbling(self) -> None:
        self.calls.append("disable_wobbling")

    def enable_motors(self) -> None:
        self.calls.append("enable_motors")

    def get_current_joint_positions(self) -> tuple[list[float], list[float]]:
        return [0.05, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1], [-0.17, 0.17]

    def get_current_head_pose(self) -> np.ndarray:
        return self.head

    def set_target(self, head: Any = None, body_yaw: float | None = None, **_kw: Any) -> None:
        self.calls.append("set_target")
        self.targets.append({"head": np.asarray(head), "body_yaw": body_yaw})


def link_with(mini: FakeMini) -> RobotLink:
    link = RobotLink({"host": "127.0.0.1"})
    link.mini = mini
    return link


def test_limp_robot_syncs_the_target_to_the_present_pose_before_torque_on() -> None:
    """두 끄기 명령이 daemon의 ik_required를 켜 두므로, 목표를 지금 자세로 맞춘 뒤 켜야 고개가 튀지 않는다."""
    mini = FakeMini("disabled")
    link_with(mini).prepare_motion()
    assert mini.calls == ["set_target", "stop_head_tracking", "disable_wobbling", "set_target", "enable_motors"]
    assert np.allclose(mini.targets[0]["head"], mini.head) and mini.targets[0]["body_yaw"] == 0.05


def test_awake_robot_gets_a_target_before_the_switches_and_no_enable() -> None:
    """토크가 켜진 daemon에 머리 목표가 없으면 두 끄기 명령의 IK가 정면(REST)이라 고개가 튄다 (MuJoCo 시뮬 22.7°).
    그래서 먼저 지금 자세를 목표로 주고, enable은 하지 않는다."""
    mini = FakeMini("enabled")
    steps = link_with(mini).prepare_motion()
    assert mini.calls == ["set_target", "stop_head_tracking", "disable_wobbling"]
    assert any("skipped" in s for s in steps)


def test_unknown_motor_state_takes_the_safe_path() -> None:
    mini = FakeMini(None)
    link_with(mini).prepare_motion()
    assert mini.calls[-2:] == ["set_target", "enable_motors"]
