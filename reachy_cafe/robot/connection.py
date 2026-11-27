"""로봇 연결 하나 (IMPLEMENTATION_PLAN.md 2·5.1절).

카메라(HW2)·오디오(HW3)·모션(HW4)이 이 `ReachyMini` 객체 하나를 공유한다. 두 번째 SDK 클라이언트를
`no_media`로 붙이면 daemon이 미디어를 놓아 오디오가 죽으므로 (HW3 확인) 절대 연결을 두 개 만들지 않는다.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class RobotLink:
    def __init__(self, robot_cfg: dict[str, Any]):
        self.cfg = robot_cfg
        self.mini: Any = None
        self.info: dict[str, Any] = {}

    # ---- 연결 ---------------------------------------------------------------
    def open(self) -> Any:
        from reachy_mini import ReachyMini

        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {}
        if self.cfg.get("robot_name"):  # MuJoCo 시뮬 daemon은 다른 이름(hw4_sim)을 쓴다 (tools/sim_check.py)
            kwargs["robot_name"] = self.cfg["robot_name"]
        self.mini = ReachyMini(host=self.cfg["host"], port=int(self.cfg.get("port", 8000)),
                               connection_mode=self.cfg.get("connection_mode", "network"),
                               media_backend=self.cfg.get("media_backend", "webrtc"), spawn_daemon=False,
                               automatic_body_yaw=True, timeout=float(self.cfg.get("connect_timeout_s", 10.0)),
                               **kwargs)
        self.info = {"host": self.cfg["host"], "connect_s": round(time.perf_counter() - t0, 2),
                     "media_backend": self.cfg.get("media_backend", "webrtc")}
        try:
            import importlib.metadata as md

            self.info["sdk_version"] = md.version("reachy-mini")
        except Exception:  # noqa: BLE001
            pass
        return self.mini

    def prepare_motion(self) -> list[str]:
        """모션을 스트리밍하기 전에 daemon 쪽 자동 동작을 끄고 모터를 켠다 (HW4 backends/robot_connect.py 순서).

        1. 먼저 목표를 지금 자세로 맞춘다(set_target). head tracking(daemon 자체 얼굴 추적)·wobbling(TTS에 맞춘 머리
           흔들기)을 끄는 명령은 daemon의 ik_required를 켜서 IK를 다시 계산하게 하는데, 머리 목표가 없으면 그 결과가
           정면(REST)이다. 토크가 켜져 있으면 그대로 쓰여 고개가 튄다 (MuJoCo 시뮬: 잠든 자세에서 1.5 s에 22.7°).
        2. 두 기능을 끈다. 시선은 우리가 정한다.
        3. 토크가 꺼져 있으면 목표를 한 번 더 지금 자세로 맞춘 뒤 enable_motors (HW4 robot_connect.py 순서:
           enable 직전에 계산된 IK가 pin 뒤에 쓰여도 지금 자세라서 튀지 않는다, SDK abstract.py:586-594).
           이미 켜져 있으면 enable은 하지 않는다. 처진 측정 자세를 명령해 생기는 몇 도의 차이는 바로 뒤
           wake_up(INIT으로 2 s goto)이 없앤다.
        """
        steps: list[str] = []
        enabled = self._motors_enabled()
        steps.append(f"motors {'enabled' if enabled else 'disabled' if enabled is False else 'unknown'}")
        self._sync_target(steps)
        for name in ("stop_head_tracking", "disable_wobbling"):
            self._call(name, steps)
        if enabled is True:
            steps.append("enable_motors skipped (already enabled)")
        else:
            self._sync_target(steps)
            self._call("enable_motors", steps)
        self.info["prepare"] = steps
        return steps

    def _sync_target(self, steps: list[str]) -> None:
        mini = self.mini
        try:
            joints, _antennas = mini.get_current_joint_positions()
            head = np.asarray(mini.get_current_head_pose(), dtype=np.float64).reshape(4, 4)
            body_yaw = float(np.asarray(joints, dtype=np.float64).reshape(7)[0])
            mini.set_target(head=head, body_yaw=body_yaw)
            steps.append("set_target(present)")
        except Exception as exc:  # noqa: BLE001 - 그래도 다음 단계는 한다 (enable은 daemon이 목표를 현재 관절로 pin)
            log.warning("set_target(present) failed: %s", exc)
            steps.append(f"set_target(present) FAILED: {exc}")

    def _call(self, name: str, steps: list[str]) -> None:
        fn = getattr(self.mini, name, None)
        if fn is None:
            return
        try:
            fn()
            steps.append(name)
        except Exception as exc:  # noqa: BLE001 - 하나가 실패해도 나머지는 시도하고 기록
            log.warning("%s failed: %s", name, exc)
            steps.append(f"{name} FAILED: {exc}")

    def _motors_enabled(self) -> bool | None:
        """daemon 상태의 motor_control_mode (HW4 robot_connect._status_fields). 모르면 None."""
        try:
            status = self.mini.client.get_status(wait=True, timeout=2.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("motor status unavailable: %s", exc)
            return None
        mode = getattr(getattr(status, "backend_status", None), "motor_control_mode", None)
        mode = getattr(mode, "value", mode)
        return None if mode is None else str(mode) == "enabled"

    def wake_up(self) -> None:
        """SDK wake_up: INIT 자세로 2초 goto + 깨어나는 소리 (모션 서비스를 시작하기 전에만 부른다)."""
        if self.mini is not None:
            self.mini.wake_up()

    def goto_sleep(self) -> None:
        if self.mini is not None:
            self.mini.goto_sleep()

    def close(self) -> None:
        mini, self.mini = self.mini, None
        if mini is not None:
            try:
                mini.__exit__(None, None, None)  # HW3 ReachyConnection.close()와 같은 방식
            except Exception as exc:  # noqa: BLE001
                log.warning("closing the robot connection failed: %s", exc)

    # ---- 카메라 기하 ----------------------------------------------------------
    def camera_model(self) -> dict[str, Any] | None:
        """K, D, 해상도, 머리→카메라 변환. 카메라가 아직 준비되지 않았으면 None."""
        mini = self.mini
        camera = getattr(getattr(mini, "media", None), "camera", None) if mini is not None else None
        try:
            if camera is None or camera.K is None or camera.D is None:
                return None
            # SDK 1.11: K는 현재 해상도로 다시 맞춘 값, resolution은 (width, height), T_head_cam은 CAD 기본값
            return {"K": np.asarray(camera.K, dtype=np.float64), "D": np.asarray(camera.D, dtype=np.float64),
                    "resolution": tuple(int(x) for x in camera.resolution),
                    "T_head_cam": np.asarray(mini.T_head_cam, dtype=np.float64)}
        except (RuntimeError, AttributeError):  # 해상도가 아직 정해지지 않음
            return None


def pixel_ray_world(u: float, v: float, model: dict[str, Any], head_pose: np.ndarray) -> np.ndarray:
    """카메라 픽셀 (u, v) → 로봇 기준(월드) 방향 벡터.

    SDK `look_at_image_pose`와 같은 계산이지만 프레임을 받은 순간의 머리 자세를 쓴다. 이렇게 월드 방향으로
    바꿔 두면 머리가 움직이는 중에도 시선 목표가 흔들리지 않는다 (`look_at_world_pose(*ray)`로 바로 쓸 수 있음).
    """
    from reachy_mini.media.camera_utils import undistort_points

    x_n, y_n = undistort_points(u, v, model["K"], model["D"])
    ray_cam = np.array([x_n, y_n, 1.0], dtype=np.float64)
    ray = (np.asarray(head_pose, dtype=np.float64) @ model["T_head_cam"])[:3, :3] @ ray_cam
    norm = float(np.linalg.norm(ray))
    return ray / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
