"""모션 소유자 하나 (IMPLEMENTATION_PLAN.md 5.7절).

`MotionPlayer`는 HW4 `MotionService`를 상속해 그대로 쓰고 세 가지만 더한다.
1. 쉬는 동안의 시선 레이어: 손님 방향(월드 광선)·화면·슬롯·둘러보기를 50 Hz `set_target`으로 스트리밍.
   자세 오프셋(듣기·생각·갸웃 등)과 짧은 제스처(끄덕임·안테나 흔들기)를 그 위에 더한다.
2. 키프레임 라벨 이벤트: YAML head 트랙 라벨(예: Aha!의 `present`)이 시작되는 tick에 "label" 이벤트.
3. 효과음: YAML `sound.at_s`에 맞춰 로봇 스피커(HW4 RobotSpeaker, daemon REST)로 재생.

모든 로봇 명령은 MotionService의 워커 스레드 하나에서만 나간다. 감정 모션이 끝나면 `_last_state`에서 이어서
시선 스트리밍을 하므로, 다음 모션은 HW4의 전이 계획(plan_transition)으로 자연스럽게 이어진다.
`SharedMiniBackend`는 HW4 Backend 프로토콜을 공유 연결 위에서 구현한다 (SdkBackend와 같은 송신·상태 읽기).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from reachy_mini.vision.look_at import look_at_world_pose
from reachy_motion.backends.base import BackendError, Capabilities, PresentState, check_command
from reachy_motion.backends.sdk import _interpolation_technique, _stamp_state_pushes
from reachy_motion.executor.clock import RealClock
from reachy_motion.library.pose import REST, Pose6, RobotState, pose_from_matrix
from reachy_motion.planner import concat
from reachy_motion.safety.checker import prepare, prepare_timeline
from reachy_motion.service import HandleState, MotionService, Priority, same_state
from reachy_motion.sound import cue_tick, sound_cue

log = logging.getLogger(__name__)

# 자세: (머리 오프셋, 안테나 오프셋[right, left] rad). 안테나 +right / -left = 세우기 (HW4 antennas_droop의 반대)
POSTURES: dict[str, tuple[Pose6, tuple[float, float]]] = {
    "neutral": (Pose6(), (0.0, 0.0)),
    "listening": (Pose6.from_deg_mm(roll_deg=6.0, pitch_deg=-3.0), (0.20, -0.20)),
    "thinking": (Pose6.from_deg_mm(roll_deg=-4.0, pitch_deg=-8.0, yaw_deg=8.0), (0.30, -0.05)),
    "perk_up": (Pose6.from_deg_mm(pitch_deg=-4.0, z_mm=5.0), (0.35, -0.35)),
    "tilt": (Pose6.from_deg_mm(roll_deg=12.0), (0.10, -0.10)),
}
GESTURES = {"nod": 0.5, "antenna_wiggle": 0.9}  # 길이 (s)

# 시선 레이어가 보내는 최종 자세의 범위. HW4 safety/policy.py(roll·pitch 40°, 상대 yaw 60°, x -15..25 mm,
# y ±40 mm, z -40..24 mm, 안테나 3.05 rad)보다 안쪽으로 잡았다. 감정 모션은 HW4 checker가 따로 검사한다.
ENVELOPE = {"roll": math.radians(25.0), "pitch": math.radians(35.0), "yaw": math.radians(55.0),
            "x": (-0.010, 0.020), "y": (-0.030, 0.030), "z": (-0.030, 0.015), "antenna": 2.8}


def clamp_state(state: RobotState) -> RobotState:
    h = state.head
    r, p, y = ENVELOPE["roll"], ENVELOPE["pitch"], ENVELOPE["yaw"]
    head = Pose6(x=min(max(h.x, ENVELOPE["x"][0]), ENVELOPE["x"][1]),
                 y=min(max(h.y, ENVELOPE["y"][0]), ENVELOPE["y"][1]),
                 z=min(max(h.z, ENVELOPE["z"][0]), ENVELOPE["z"][1]),
                 roll=min(max(h.roll, -r), r), pitch=min(max(h.pitch, -p), p), yaw=min(max(h.yaw, -y), y))
    a = ENVELOPE["antenna"]
    antennas = (min(max(state.antennas[0], -a), a), min(max(state.antennas[1], -a), a))
    return RobotState(head=head, antennas=antennas, body_yaw=state.body_yaw)


class JointGuard:
    """시선 레이어 목표를 HW4 안전 검사기와 같은 기준으로 관절 공간에서 확인한다.

    각도 범위(ENVELOPE)만으로는 부족하다: 좌우 45° + 아래 25°(카드 슬롯 쪽)에서 Stewart 관절 여유가 10.2°로
    HW4 정책 한계(10°)에 닿고, 여기에 끄덕임·perk_up이 더해지면 넘는다 (tests/test_motion_safety.py).
    그래서 daemon과 같은 IK(AnalyticalKinematics, automatic_body_yaw)와 SDK URDF 한계로 여유를 계산하고,
    부족하면 정면(중립) 쪽으로 줄여 안전한 가장 가까운 자세를 찾는다. IK 1회 약 6 µs.
    """

    def __init__(self, min_margin_deg: float = 14.0, max_auto_body_shift_deg: float = 2.0):
        from reachy_motion.analysis.goto_replay import ik_joints, new_kinematics
        from reachy_motion.analysis.limits import urdf_joint_limits
        from reachy_motion.safety.checker import STEWART_NAMES

        self._ik = ik_joints
        self._kin = new_kinematics()
        limits = urdf_joint_limits()
        self._lo = np.array([limits[n][0] for n in STEWART_NAMES])
        self._hi = np.array([limits[n][1] for n in STEWART_NAMES])
        self.min_margin = math.radians(min_margin_deg)
        self.max_shift = math.radians(max_auto_body_shift_deg)
        self.projections = 0

    def margin_deg(self, state: RobotState) -> float:
        """가장 가까운 Stewart 관절 한계까지 여유 (도). IK 불가면 -inf."""
        joints = self._ik(self._kin, state.head.matrix(), float(state.body_yaw))
        if not np.all(np.isfinite(joints)):
            return -math.inf
        q = joints[1:7]
        return math.degrees(float(np.min(np.minimum(q - self._lo, self._hi - q))))

    def ok(self, state: RobotState) -> bool:
        joints = self._ik(self._kin, state.head.matrix(), float(state.body_yaw))
        if not np.all(np.isfinite(joints)):
            return False
        q = joints[1:7]
        if float(np.min(np.minimum(q - self._lo, self._hi - q))) < self.min_margin:
            return False
        return abs(float(joints[0]) - float(state.body_yaw)) <= self.max_shift  # 자동 몸통 회전이 끼어들지 않게

    def project(self, state: RobotState, anchor: RobotState | None = None) -> RobotState:
        """state가 안전하면 그대로, 아니면 anchor(기본: 정면) 쪽으로 머리 자세를 줄인 가장 먼 안전한 자세."""
        if self.ok(state):
            return state
        self.projections += 1
        base = (anchor.head if anchor is not None else Pose6()).as_array()
        goal = state.head.as_array()
        lo, hi = 0.0, 1.0
        for _ in range(10):  # 이분 탐색: 1/1024 정밀도
            mid = (lo + hi) / 2.0
            cand = RobotState(head=Pose6.from_array(base + mid * (goal - base)), antennas=state.antennas,
                              body_yaw=state.body_yaw)
            if self.ok(cand):
                lo = mid
            else:
                hi = mid
        return RobotState(head=Pose6.from_array(base + lo * (goal - base)), antennas=state.antennas,
                          body_yaw=state.body_yaw)


class SharedMiniBackend:
    """HW4 Backend 프로토콜 (backends/base.py) over 공유 ReachyMini. 연결의 주인은 RobotLink."""

    name = "robot"
    data_kind = "measured"

    def __init__(self, mini: Any, push_timeout_s: float = 2.0):
        self.capabilities = Capabilities(torque_control=False, real_hardware=True)
        self._mini = mini
        self._clock = RealClock()
        self._stamps: Any = None
        self.push_timeout_s = push_timeout_s

    def connect(self) -> None:
        # SdkBackend와 같은 방식: daemon 상태 push가 도착한 시각을 기록 (present()의 t_ns)
        self._stamps = _stamp_state_pushes(self._mini.client, self._clock)

    def close(self) -> None:
        self._stamps = None

    def set_target(self, head: np.ndarray, antennas: Sequence[float], body_yaw: float) -> None:
        h, a, b = check_command(head, antennas, body_yaw)
        try:
            self._mini.set_target(head=h, antennas=[a[0], a[1]], body_yaw=float(b))
        except Exception as exc:
            raise BackendError(f"set_target failed: {exc}") from exc

    def goto_target(self, head: np.ndarray, antennas: Sequence[float], body_yaw: float, duration: float,
                    method: str) -> None:
        h, a, b = check_command(head, antennas, body_yaw)
        self._mini.goto_target(head=h, antennas=[a[0], a[1]], duration=float(duration),
                               method=_interpolation_technique(method), body_yaw=float(b))

    def present(self) -> PresentState:
        stamps = self._stamps
        if stamps is None or not stamps.ready.wait(self.push_timeout_s):
            raise BackendError(f"no state push from the daemon within {self.push_timeout_s:g} s")
        try:
            with stamps.lock:
                t_ns = stamps.joints_ns
                head = np.asarray(self._mini.get_current_head_pose(), dtype=np.float64).reshape(4, 4)
                head_joints, antennas = self._mini.get_current_joint_positions()
            joints = np.asarray(head_joints, dtype=np.float64).reshape(7)
        except Exception as exc:
            raise BackendError(f"present state unavailable: {exc}") from exc
        return PresentState(t_ns=int(t_ns), head=head, antennas=(float(antennas[0]), float(antennas[1])),
                            body_yaw=float(joints[0]), joints=joints, data_kind=self.data_kind)


class MotionPlayer(MotionService):
    def __init__(self, backend: Any, library: Any = None, motion_cfg: dict[str, Any] | None = None,
                 screen_point: Sequence[float] = (0.20, 0.22, -0.12), slot_point: Sequence[float] = (0.22, 0.20, -0.28),
                 sound_player: Callable[[str], Any] | None = None):
        cfg = motion_cfg or {}
        # 감정 모션 앞에 붙는 전이(지금 자세 → 모션 시작 자세) 속도 상한. HW4 기본값(회전 60°/s, 최소 0.5 s)으로는
        # 카드 슬롯을 보던 자세에서 Success까지 1.7 s가 걸려 결제 완료 화면이 늦게 떴다 (MuJoCo 점검).
        # 빨라진 전이도 매번 HW4 prepare_timeline이 정책(관절 속도 등)으로 검사한다 (tests/test_motion_safety.py).
        caps = {"max_rot_dps": float(cfg.get("transition_max_rot_dps", 150.0)),
                "max_antenna_dps": float(cfg.get("transition_max_antenna_dps", 360.0)),
                "min_s": float(cfg.get("transition_min_s", 0.3))}
        super().__init__(backend, library=library, rate_hz=50.0, transition_caps=caps)
        self.gaze_rate_hz = float(cfg.get("gaze_rate_hz", 50.0))
        self.max_speed = math.radians(float(cfg.get("gaze_max_speed_dps", 120.0)))
        self.smoothing = float(cfg.get("gaze_smoothing", 0.25))
        self.yaw_limit = math.radians(float(cfg.get("gaze_yaw_limit_deg", 45.0)))
        self.pitch_limit = math.radians(float(cfg.get("gaze_pitch_limit_deg", 25.0)))
        self.ray_timeout = float(cfg.get("user_ray_timeout_s", 1.0))
        self.keepalive = float(cfg.get("keepalive_s", 0.5))
        # 명령 → 실제 머리 움직임 지연 (HW4 실측 약 0.19 s). 라벨을 이만큼 늦게 내 화면·음성이 '보이는' 동작과 맞게 한다
        self.label_delay = float(cfg.get("label_delay_s", 0.1))
        self.sound_offset_ms = float(cfg.get("sound_offset_ms", 0.0))
        self.sound_player = sound_player
        self.guard = JointGuard(float(cfg.get("gaze_min_joint_margin_deg", 14.0)))
        self._points = {"screen": pose_from_matrix(look_at_world_pose(*map(float, screen_point))),
                        "slot": pose_from_matrix(look_at_world_pose(*map(float, slot_point)))}
        self._gaze_lock = threading.Lock()
        self._gaze = "wander"
        self._glance: tuple[str, float] | None = None
        self._user_ray: np.ndarray | None = None
        self._user_ray_t = -math.inf
        self._posture = "neutral"
        self._gesture: tuple[str, float] | None = None
        self._last_send = -math.inf
        self._cache: dict[int, tuple[Any, Any]] = {}
        self._base: tuple[float, float] | None = None  # 기본 시선 (yaw, pitch), 워커 스레드 전용
        self.gaze_steps = 0  # 시선 루프가 돈 횟수 (약 50 Hz)
        self.gaze_sends = 0  # 실제로 보낸 명령 수 (멈춰 있으면 keepalive 간격으로만 보냄)

    # ---- 시선·자세 (어느 스레드에서 불러도 됨) ----------------------------------------
    def set_gaze(self, target: str) -> None:
        with self._gaze_lock:
            self._gaze = target

    def set_point(self, name: str, xyz: Sequence[float]) -> Pose6:
        """화면·슬롯 위치(로봇 기준 m: x 앞, y 왼쪽, z 위)를 바꾼다 (보정: tools/spike_connection.py --aim).
        돌려주는 값은 그 점을 보는 머리 자세 (범위·관절 검사 전)."""
        pose = pose_from_matrix(look_at_world_pose(*map(float, xyz)))
        with self._gaze_lock:
            self._points = {**self._points, name: pose}
        return pose

    def set_user_ray(self, ray: Sequence[float] | None) -> None:
        with self._gaze_lock:
            self._user_ray = None if ray is None else np.asarray(ray, dtype=np.float64)
            self._user_ray_t = time.perf_counter()

    def glance(self, target: str, hold_s: float) -> None:
        with self._gaze_lock:
            self._glance = (target, time.perf_counter() + float(hold_s))

    def set_posture(self, name: str) -> None:
        with self._gaze_lock:
            self._posture = name if name in POSTURES else "neutral"

    def gesture(self, name: str) -> None:
        if name in GESTURES:
            with self._gaze_lock:
                self._gesture = (name, time.perf_counter())

    def cancel_idle_motion(self) -> None:
        """사람이 나타나면 돌고 있던 Boredom(IDLE 우선순위)을 바로 멈춘다."""
        with self._cond:
            self._idle = None
            if self._current is not None and self._current.priority == Priority.IDLE:
                self._current._cancel.set()

    def park(self) -> Any:
        """끝낼 때: 모션을 멈추고 REST로 (HW4 검사를 거친 전이), 그 뒤에도 시선 레이어가 REST를 유지한다.

        stop(blend_to_rest=True)만 하면 전이가 끝난 뒤 시선 레이어가 다시 화면·손님 쪽으로 머리를 끌어간다
        (MuJoCo 시뮬 점검에서 발견, tools/sim_check.py rest 단계).
        """
        self.clear_idle()
        self.set_posture("neutral")
        self.set_gaze("rest")
        return self.stop(blend_to_rest=True)

    def preload(self, names: Sequence[str], variants: Sequence[str] = ("expressive",)) -> None:
        """부팅 때 모든 모션을 미리 prepare (매번 64–206 ms 절약)."""
        library = self._library()
        for name in names:
            for variant in variants:
                try:
                    spec = library.get(name, variant)
                    self._cache[id(spec)] = (spec, prepare(spec, self.policy))
                except Exception as exc:  # noqa: BLE001
                    log.warning("preload %s:%s failed: %s", name, variant, exc)

    # ---- MotionService 확장 --------------------------------------------------------
    def _prepared_for(self, spec: Any) -> Any:
        hit = self._cache.get(id(spec))
        if hit is not None and hit[0] is spec:
            return hit[1]
        return prepare(spec, self.policy)

    def _prepare_loop(self) -> None:  # 기반 클래스와 같고, 캐시만 쓴다
        while True:
            with self._cond:
                while not self._prep_queue and not self._closing:
                    self._cond.wait()
                if self._closing:
                    return
                h = self._prep_queue.pop(0)
            try:
                prepared = self._prepared_for(h.spec)
            except Exception as exc:  # noqa: BLE001
                self._reject(h, f"{type(exc).__name__}: {exc}", [])
                continue
            h.report = prepared.report
            if not prepared.runnable:
                codes = [v.code for v in prepared.report.violations]
                self._reject(h, "outside the initial policy: " + ", ".join(codes), codes)
                continue
            h.prepared = prepared
            with self._cond:
                if h.done or self._closing:
                    continue
                self._run_queue.append(h)
                self._run_queue.sort(key=lambda x: -int(x.priority))
                cur = self._current
                if cur is not None and (h.priority > cur.priority or (h.interrupt and h.priority >= cur.priority)):
                    cur._preempted = True
                    cur._cancel.set()
                self._cond.notify_all()

    def _work_loop(self) -> None:
        """기반 클래스의 워커 + 쉬는 동안 시선 스트리밍.

        주기는 절대 시각으로 맞춘다. Windows에서 시간 제한이 있는 Condition.wait는 15.6 ms 단위로 깨어나
        (HW4 executor/clock.py) 50 Hz가 24 Hz로 떨어지므로, 고해상도인 time.sleep(Python 3.11+)을 쓴다.
        새 모션은 다음 틱(최대 20 ms)에 시작된다.
        """
        period = 1.0 / self.gaze_rate_hz
        next_t = time.perf_counter()
        while True:
            h = None
            idle_play: tuple[Any, str] | None = None
            with self._cond:
                if self._closing:
                    return
                if self._run_queue:
                    h = self._run_queue.pop(0)
                    if h.done:
                        continue
                    self._current = h
                elif (self._idle is not None and not self._prep_queue
                      and time.monotonic() - self._last_activity >= self._idle[2]):
                    idle_play = (self._idle[0], self._idle[1])
                    self._last_activity = time.monotonic()
            if h is None:
                if idle_play is not None:
                    self.play(idle_play[0], idle_play[1], priority=Priority.IDLE)
                try:
                    self._gaze_step()
                except Exception as exc:  # noqa: BLE001 - 시선 오류가 모션 스레드를 죽이지 않게
                    log.warning("gaze step failed: %s", exc)
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()  # 늦었으면 몰아서 보내지 않고 지금부터 다시
                continue
            next_t = time.perf_counter()
            try:
                self._run(h)
            except Exception as exc:  # noqa: BLE001
                log.exception("motion run failed")
                self._finish(h, HandleState.FAILED, f"{type(exc).__name__}: {exc}")
            finally:
                with self._cond:
                    self._current = None
                    self._last_activity = time.monotonic()
                    self._cond.notify_all()

    def _run(self, h: Any) -> None:
        """기반 클래스 `_run`과 같고, executor에 on_tick(라벨·효과음)을 넘긴다."""
        start = self._start_state()
        offset = 0.0
        if h.spec is None:
            if same_state(start, REST):
                h.state = HandleState.RUNNING
                self._finish(h, HandleState.DONE, "already at rest")
                return
            job = self._transition(start, REST)
        elif same_state(start, h.spec.start):
            job = h.prepared
        else:
            transition = self._transition(start, h.spec.start)
            offset = float(transition.timeline.t[-1])
            job = prepare_timeline(concat(transition.timeline, h.prepared.timeline, name=h.key), self.policy)
            h.reason = "started with a transition from the last commanded pose"
        if not job.runnable:
            codes = [v.code for v in job.report.violations]
            self._reject(h, "transition outside the initial policy: " + ", ".join(codes), codes)
            return
        if h._cancel.is_set():
            self._finish(h, HandleState.PREEMPTED if h._preempted else HandleState.CANCELLED)
            return
        h.state = HandleState.RUNNING
        self._emit("started", h, duration_s=float(job.timeline.t[-1]), offset_s=round(offset, 3))
        result = self.executor.run(job, cancel=h._cancel, on_tick=self._ticker(h, offset),
                                   record_present=self.record_present)
        h.result = result
        self.last_result = result
        if result.last_sample is not None:
            head, antennas, body = result.last_sample
            self._last_state = RobotState(head=pose_from_matrix(head), antennas=tuple(antennas), body_yaw=float(body))
        if result.completed:
            self._finish(h, HandleState.DONE)
        elif result.error:
            self._finish(h, HandleState.FAILED, result.error)
        elif h._preempted:
            self._finish(h, HandleState.PREEMPTED, "preempted by a higher-priority or interrupting motion")
        else:
            self._finish(h, HandleState.CANCELLED, "cancelled")

    def _ticker(self, h: Any, offset: float) -> Callable[[int, Any], None]:
        rate = self.executor.rate_hz
        labels: list[tuple[float, str]] = []
        if h.spec is not None:
            eff = h.spec.effective()
            for (t_depart, _arrive, _release), key in zip(eff.key_times("head"), eff.track("head")):
                if key.label:
                    labels.append((offset + t_depart + self.label_delay, key.label))
            labels.sort()
        cue = sound_cue(h.spec) if (h.spec is not None and self.sound_player is not None) else None
        k_cue = cue_tick(offset, cue.at_s, self.sound_offset_ms, rate) if cue is not None else None
        state = {"next": 0, "sound": False}

        def on_tick(k: int, _rec: Any = None) -> None:
            t = k / rate
            while state["next"] < len(labels) and labels[state["next"]][0] <= t + 1e-9:
                at, label = labels[state["next"]]
                state["next"] += 1
                self._emit("label", h, label=label, at_s=round(at, 3))
            if k_cue is not None and not state["sound"] and k >= k_cue:
                state["sound"] = True
                threading.Thread(target=self._play_sound, args=(cue.file,), name="motion-sound", daemon=True).start()

        return on_tick

    def _play_sound(self, filename: str) -> None:
        try:
            if self.sound_player is not None:
                self.sound_player(filename)
        except Exception as exc:  # noqa: BLE001 - 소리가 안 나도 모션은 계속
            log.warning("motion sound %s failed: %s", filename, exc)

    # ---- 시선 레이어 -----------------------------------------------------------
    def _gaze_step(self) -> None:
        now = time.perf_counter()  # 시선 레이어 시각: Windows monotonic은 15.6 ms 단위
        self.gaze_steps += 1
        current = self._last_state
        if current is None:
            try:
                p = self.backend.present()
                current = RobotState(head=pose_from_matrix(p.head), antennas=tuple(p.antennas),
                                     body_yaw=float(p.body_yaw))
            except Exception:  # noqa: BLE001
                current = REST
        dt = 1.0 / self.gaze_rate_hz
        # 목표는 각도 범위로 자르고, 관절 여유가 부족하면 정면 쪽으로 줄인다 (JointGuard = HW4 검사기 기준)
        target = self.guard.project(clamp_state(self._target_state(now, current)))
        # 지금 자세는 자르지 않는다. 감정 모션이 범위 밖에서 끝났어도 거기서부터 천천히 들어오게 (순간 점프 없음)
        new = self._approach(current, target, dt)
        if not self.guard.ok(new):
            if self.guard.ok(current):  # 한 틱 직선 이동이 위험 영역을 스침: 지금 자세 쪽으로 줄인다
                new = self.guard.project(new, anchor=current)
            elif self.guard.margin_deg(new) < self.guard.margin_deg(current):
                # 이미 여유가 부족한 자리(모션 끝·시작 자세)에서는 여유가 줄어드는 쪽으로 가지 않고 정면 쪽으로
                new = self._approach(current, RobotState(head=Pose6(), antennas=target.antennas), dt)
        moved = (float(np.max(np.abs(new.head.as_array() - current.head.as_array()))) > 1e-4
                 or max(abs(a - b) for a, b in zip(new.antennas, current.antennas)) > 1e-4)
        if moved or now - self._last_send >= self.keepalive:
            self.backend.set_target(new.head.matrix(), new.antennas, new.body_yaw)
            self._last_send = now
            self.gaze_sends += 1
        self._last_state = new

    def _target_state(self, now: float, current: RobotState) -> RobotState:
        with self._gaze_lock:
            glance = self._glance if self._glance and now < self._glance[1] else None
            target = glance[0] if glance else self._gaze
            ray, ray_t = self._user_ray, self._user_ray_t
            posture = POSTURES.get(self._posture, POSTURES["neutral"])
            gesture = self._gesture
        if target == "rest":  # park(): 기본 자세 그대로 (자세·제스처 오프셋 없음)
            self._base = (0.0, 0.0)
            return REST
        head: Pose6 | None = None
        if target == "user" and ray is not None and now - ray_t <= self.ray_timeout:
            head = pose_from_matrix(look_at_world_pose(float(ray[0]), float(ray[1]), float(ray[2])))
        elif target in self._points:
            head = self._points[target]
        elif target == "wander":
            head = Pose6.from_deg_mm(yaw_deg=18.0 * math.sin(2 * math.pi * now / 14.0),
                                     pitch_deg=-2.0 + 2.0 * math.sin(2 * math.pi * now / 9.0))
        if head is not None:
            base = (head.yaw, head.pitch)
        elif self._base is not None:  # still, 또는 손님 방향 정보가 끊김: 마지막 '기본 시선'을 유지
            base = self._base
        else:  # 처음: 지금 머리 방향에서 시작
            base = (current.head.yaw, current.head.pitch)
        # 기본 시선(자세·제스처 오프셋을 더하기 전)을 따로 기억한다. 현재 머리 자세를 그대로 유지 목표로 쓰면
        # 매 틱 자세 오프셋이 다시 더해져 머리가 한쪽으로 흘러간다.
        yaw = max(-self.yaw_limit, min(self.yaw_limit, base[0]))
        pitch = max(-self.pitch_limit, min(self.pitch_limit, base[1]))
        self._base = (yaw, pitch)
        offset, ant_offset = posture
        g_head = Pose6()
        g_ant = (0.0, 0.0)
        if gesture is not None:
            name, t0 = gesture
            dt = now - t0
            length = GESTURES.get(name, 0.0)
            if dt <= length:
                if name == "nod":
                    g_head = Pose6.from_deg_mm(pitch_deg=8.0 * math.sin(math.pi * dt / length))
                elif name == "antenna_wiggle":
                    w = 0.35 * math.sin(2 * math.pi * 2.0 * dt / length)
                    g_ant = (w, w)
        head = Pose6(x=offset.x + g_head.x, y=offset.y + g_head.y, z=offset.z + g_head.z,
                     roll=offset.roll + g_head.roll, pitch=pitch + offset.pitch + g_head.pitch,
                     yaw=yaw + offset.yaw + g_head.yaw)
        antennas = (REST.antennas[0] + ant_offset[0] + g_ant[0], REST.antennas[1] + ant_offset[1] + g_ant[1])
        return RobotState(head=head, antennas=antennas, body_yaw=0.0)

    def _approach(self, cur: RobotState, tgt: RobotState, dt: float) -> RobotState:
        c, g = cur.head.as_array(), tgt.head.as_array()
        step = self.smoothing * (g - c)
        max_ang, max_lin = self.max_speed * dt, 0.05 * dt
        step[3:] = np.clip(step[3:], -max_ang, max_ang)
        step[:3] = np.clip(step[:3], -max_lin, max_lin)
        head = Pose6.from_array(c + step)
        max_ant = 3.0 * dt
        antennas = tuple(a + max(-max_ant, min(max_ant, self.smoothing * 2.0 * (b - a)))
                         for a, b in zip(cur.antennas, tgt.antennas))
        return RobotState(head=head, antennas=antennas, body_yaw=cur.body_yaw + 0.1 * (tgt.body_yaw - cur.body_yaw))
