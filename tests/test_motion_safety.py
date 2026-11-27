"""시선 레이어가 보낼 수 있는 자세가 HW4 안전 정책 안에 있는지 (HW4 검사기·IK·URDF 한계로 확인).

감정 모션(YAML)은 HW4 checker가 미리 검사하지만, 시선 레이어는 실행 중에 목표를 만들어 50 Hz로 보낸다.
1) 정적: 시선 방향(좌우·상하 극단) × 자세 5종 × 끄덕임·안테나 흔들기 → JointGuard로 보정한 목표를 REST에서
   HW4 plan_transition + prepare_timeline(DEFAULT_POLICY)으로 검사 → 전부 runnable
2) 동적: MockBackend 위에서 시선을 극단 사이로 마구 바꾸며 보낸 명령 전체의 IK 관절 여유·관절 속도·안테나 속도·
   자동 몸통 회전을 HW4 정책 값과 비교
"""

from __future__ import annotations

import itertools
import math
import time

import pytest

pytest.importorskip("reachy_motion")
pytest.importorskip("reachy_mini")

import numpy as np  # noqa: E402
from reachy_motion.analysis.goto_replay import ik_joints, new_kinematics  # noqa: E402
from reachy_motion.analysis.limits import ANTENNA_MOTOR, STEWART_MOTOR, urdf_joint_limits  # noqa: E402
from reachy_motion.backends.mock import MockBackend  # noqa: E402
from reachy_motion.library.pose import REST, Pose6, RobotState  # noqa: E402
from reachy_motion.library.spec import Easing  # noqa: E402
from reachy_motion.planner import plan_transition, transition_duration  # noqa: E402
from reachy_motion.safety.checker import STEWART_NAMES, prepare_timeline  # noqa: E402
from reachy_motion.safety.policy import DEFAULT_POLICY  # noqa: E402

from reachy_cafe.robot.motion import POSTURES, JointGuard, MotionPlayer, clamp_state  # noqa: E402


def composed(yaw_deg: float, pitch_deg: float, posture: str, nod: bool = False, wiggle: float = 0.0) -> RobotState:
    """MotionPlayer._target_state와 같은 합성: 기본 시선 + 자세 오프셋 + 제스처 오프셋."""
    off, ant = POSTURES[posture]
    head = Pose6(x=off.x, y=off.y, z=off.z, roll=off.roll,
                 pitch=math.radians(pitch_deg) + off.pitch + (math.radians(8.0) if nod else 0.0),
                 yaw=math.radians(yaw_deg) + off.yaw)
    return clamp_state(RobotState(head=head, antennas=(REST.antennas[0] + ant[0] + wiggle,
                                                        REST.antennas[1] + ant[1] + wiggle)))


def test_unguarded_extremes_would_break_the_policy():
    """보정이 왜 필요한지: 카드 슬롯 쪽(좌 45°, 아래 25°)에서 끄덕이면 Stewart 여유가 정책 한계 밑으로."""
    s = composed(45, 25, "neutral", nod=True)
    tl = plan_transition(REST, s, transition_duration(REST, s), Easing.MIN_JERK, 50.0)
    assert not prepare_timeline(tl).runnable


def test_guarded_gaze_targets_pass_the_hw4_policy():
    guard = JointGuard()
    failures = []
    for yaw, pitch, posture, nod, wiggle in itertools.product((-45, -25, 0, 25, 45), (-25, 0, 25), POSTURES,
                                                              (False, True), (0.0, 0.35)):
        s = guard.project(composed(yaw, pitch, posture, nod, wiggle))
        tl = plan_transition(REST, s, transition_duration(REST, s), Easing.MIN_JERK, 50.0)
        prepared = prepare_timeline(tl)
        if not prepared.runnable:
            failures.append(((yaw, pitch, posture, nod, wiggle), [v.code for v in prepared.report.violations]))
    assert not failures, failures[:5]
    assert guard.projections > 0  # 실제로 줄인 경우가 있어야 이 테스트가 의미 있다


def test_projection_keeps_as_much_of_the_gaze_as_possible():
    guard = JointGuard()
    raw = composed(45, 25, "perk_up", nod=True)
    safe = guard.project(raw)
    assert guard.ok(safe)
    assert math.degrees(safe.head.yaw) > 30.0  # 정면으로 확 돌아가지 않고 같은 방향을 최대한 유지
    floor = math.degrees(guard.min_margin)  # 기본 14° (config [motion] gaze_min_joint_margin_deg)
    assert floor <= guard.margin_deg(safe) < floor + 1.5


def test_fast_transitions_from_any_gaze_into_every_motion_pass_the_policy():
    """MotionPlayer._run과 같은 경로: 시선 자세 → (빨라진) 전이 → 감정 모션, 합친 타임라인을 HW4가 검사."""
    from reachy_cafe.robot.motion_library import cafe_library  # HW4 모션 + 카페 모션(greeting)
    from reachy_motion.planner import concat
    from reachy_motion.safety.checker import prepare

    backend = MockBackend(structure="robot")
    player = MotionPlayer(backend, motion_cfg={})  # 기본 전이 상한 (config/default.toml과 같은 값)
    guard = player.guard
    library = cafe_library()
    starts = [guard.project(composed(y, p, posture)) for y, p, posture in
              ((45, 25, "perk_up"), (-45, 25, "tilt"), (45, -25, "thinking"), (0, 0, "listening"), (-45, -25, "neutral"))]
    slow = []
    for spec in library:
        motion = prepare(spec, player.policy)
        for start in starts:
            tr = player._transition(start, spec.start)
            job = prepare_timeline(concat(tr.timeline, motion.timeline, name=spec.key), player.policy)
            assert job.runnable, (spec.key, start, [v.code for v in job.report.violations])
            slow.append(float(tr.timeline.t[-1]))
    assert max(slow) < 0.9  # 가장 먼 시선에서도 모션이 1초 안에 시작 (HW4 기본 상한이면 약 1.7 s)


def test_gaze_stream_stays_within_joint_margin_and_speed_limits():
    backend = MockBackend(structure="robot")
    backend.connect()
    player = MotionPlayer(backend, motion_cfg={"label_delay_s": 0.0})
    player.start()
    try:
        # 화면 ↔ 슬롯 ↔ 손님 극단 ↔ 둘러보기, 자세·끄덕임·안테나를 바꾸며 가장 빠른 움직임을 만든다
        script = [("slot", "perk_up", "nod"), ("user", "tilt", None), ("screen", "thinking", "antenna_wiggle"),
                  ("user", "listening", "nod"), ("slot", "neutral", "nod"), ("wander", "perk_up", "antenna_wiggle")]
        rays = {"user": [0.3, -1.0, -0.6]}  # 오른쪽 아래 멀리 (한계 밖)
        for gaze, posture, gesture in script:
            if gaze == "user":
                player.set_user_ray(rays["user"])
            player.set_gaze(gaze)
            player.set_posture(posture)
            if gesture:
                player.gesture(gesture)
            for _ in range(7):
                if gaze == "user":
                    player.set_user_ray(rays["user"])  # 신선도 유지 (실제로는 카메라가 15 Hz로 갱신)
                time.sleep(0.1)
    finally:
        player.close()
    cmds = [c for c in backend.commands if c.kind == "set_target"]
    assert len(cmds) > 150
    kin = new_kinematics()
    limits = urdf_joint_limits()
    lo = np.array([limits[n][0] for n in STEWART_NAMES])
    hi = np.array([limits[n][1] for n in STEWART_NAMES])
    t = np.array([c.t_ns for c in cmds]) / 1e9
    joints = np.array([ik_joints(kin, c.head, c.body_yaw) for c in cmds])
    antennas = np.array([c.antennas for c in cmds])
    assert np.all(np.isfinite(joints))
    q = joints[:, 1:7]
    margin = np.degrees(np.minimum(q - lo, hi - q)).min()
    assert margin >= DEFAULT_POLICY.min_stewart_margin_deg, margin
    shift = np.degrees(np.abs(joints[:, 0] - np.array([c.body_yaw for c in cmds]))).max()
    assert shift <= DEFAULT_POLICY.max_auto_body_shift_deg, shift
    dt = np.diff(t)
    keep = dt > 0.005  # 같은 순간의 중복 기록 제외
    q_speed = np.degrees(np.abs(np.diff(q, axis=0)))[keep] / dt[keep, None]
    a_speed = np.degrees(np.abs(np.diff(antennas, axis=0)))[keep] / dt[keep, None]
    assert q_speed.max() <= DEFAULT_POLICY.max_stewart_speed_ratio * STEWART_MOTOR.no_load_deg_s_5v, q_speed.max()
    assert a_speed.max() <= DEFAULT_POLICY.max_antenna_speed_ratio * ANTENNA_MOTOR.no_load_deg_s_5v, a_speed.max()
