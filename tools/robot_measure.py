"""점검 도구(sim_check.py, spike_connection.py)가 같이 쓰는 측정 도우미.

- Recorder: 보낸 명령과 daemon이 보낸 상태(측정값)를 시각과 함께 모은다
- gaze_lag: 시선을 크게 바꿀 때 명령 yaw와 측정 yaw가 각각 변화의 90%에 닿는 시각 차이 (명령 → 실제 움직임 지연)
- direction_error_deg: 머리 정면 축끼리의 각도 (시선 방향 오차; 전체 회전 오차와 달리 roll은 빠진다)
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np


class Recorder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cmds: list[tuple[float, np.ndarray, tuple[float, float]]] = []
        self.meas: list[tuple[float, np.ndarray, tuple[float, float], np.ndarray]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def wrap(self, backend: Any) -> None:
        """backend.set_target을 감싸 보낸 명령을 기록한다 (모션 워커 스레드에서 불림)."""
        original = backend.set_target

        def set_target(head, antennas, body_yaw):
            original(head, antennas, body_yaw)
            with self.lock:
                self.cmds.append((time.perf_counter(), np.array(head, dtype=float),
                                  (float(antennas[0]), float(antennas[1]))))

        backend.set_target = set_target

    def sample(self, backend: Any, hz: float = 100.0) -> None:
        """backend.present()(SDK 캐시, 네트워크 요청 없음)를 hz로 읽어 기록한다."""
        def loop() -> None:
            while not self._stop.is_set():
                try:
                    p = backend.present()
                    with self.lock:
                        self.meas.append((time.perf_counter(), np.array(p.head, dtype=float), tuple(p.antennas),
                                          np.array(p.joints, dtype=float)))
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(1.0 / hz)

        self._thread = threading.Thread(target=loop, name="measure-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)

    def window(self, t0: float, t1: float) -> tuple[list, list]:
        with self.lock:
            return ([c for c in self.cmds if t0 <= c[0] <= t1], [m for m in self.meas if t0 <= m[0] <= t1])


def gaze_lag(rec: Recorder, t0: float, t1: float, min_step_deg: float = 5.0) -> float | None:
    from reachy_motion.library.pose import pose_from_matrix

    cmds, meas = rec.window(t0, t1)
    if len(cmds) < 5 or len(meas) < 5:
        return None
    cy = np.array([pose_from_matrix(c[1]).yaw for c in cmds])
    my = np.array([pose_from_matrix(m[1]).yaw for m in meas])
    a, b = cy[0], cy[-1]
    if abs(b - a) < math.radians(min_step_deg):
        return None
    level = a + 0.9 * (b - a)
    sign = 1 if b > a else -1
    tc = next((c[0] for c, y in zip(cmds, cy) if sign * (y - level) >= 0), None)
    tm = next((m[0] for m, y in zip(meas, my) if sign * (y - level) >= 0), None)
    return None if tc is None or tm is None else round(tm - tc, 3)


def direction_error_deg(cmd_head: np.ndarray, meas_head: np.ndarray) -> float:
    a, b = np.asarray(cmd_head)[:3, 0], np.asarray(meas_head)[:3, 0]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))
