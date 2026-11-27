"""Where motion commands go: the robot, or a simulation (dry runs, tests).

Only the MotionController's thread calls ``send`` / ``enable_motors`` / ``disable_motors`` /
``set_wobbling``. ``emergency_torque_off`` uses the daemon's REST API, independent of the
WebSocket the commands use, so it works from any thread even if a command is stuck.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from .pose import SLEEP, Pose, pose_from_matrix


class MotionSink(Protocol):
    name: str
    simulated: bool

    def send(self, head: np.ndarray, antennas: list[float]) -> None: ...
    def enable_motors(self) -> None: ...
    def disable_motors(self) -> None: ...
    def motor_mode(self) -> str | None: ...
    def present(self) -> tuple[np.ndarray, list[float]]: ...
    def body_yaw(self) -> float: ...
    def set_wobbling(self, on: bool) -> None: ...
    def emergency_torque_off(self) -> str: ...
    def daemon_stats(self) -> dict[str, Any] | None: ...


class RobotSink:
    """Reachy Mini through the SDK client the audio already uses (``ReachyConnection.mini``)."""

    name = "robot"
    simulated = False

    def __init__(self, mini: Any, daemon: Any) -> None:
        self.mini = mini
        self.daemon = daemon  # robot.connection.DaemonClient (REST)

    def send(self, head: np.ndarray, antennas: list[float]) -> None:
        # body_yaw omitted -> None -> unchanged (never commanded during conversation)
        self.mini.set_target(head=head, antennas=antennas)

    def enable_motors(self) -> None:
        self.mini.enable_motors()  # daemon pins every target to the present pose first (no snap)

    def disable_motors(self) -> None:
        self.mini.disable_motors()

    def motor_mode(self) -> str | None:
        return self.daemon.motor_mode()

    def present(self) -> tuple[np.ndarray, list[float]]:
        """Measured head pose (world frame, 4x4) and antennas (SDK cached state, no request)."""
        head = np.asarray(self.mini.get_current_head_pose(), dtype=np.float64)
        antennas = list(self.mini.get_current_joint_positions()[1])
        return head, antennas

    def body_yaw(self) -> float:
        """Measured torso yaw (rad): joint 0 of the head joints (SDK cached state)."""
        return float(self.mini.get_current_joint_positions()[0][0])

    def set_wobbling(self, on: bool) -> None:
        if on:
            self.mini.enable_wobbling()
        else:
            self.mini.disable_wobbling()

    def emergency_torque_off(self) -> str:
        """REST first (its own HTTP connection); the SDK WebSocket only if REST fails."""
        try:
            self.daemon.set_motor_mode("disabled", timeout_s=2.0)
            return "rest"
        except Exception as rest_exc:  # noqa: BLE001 - try the other channel before giving up
            try:
                self.mini.disable_motors()
            except Exception as ws_exc:  # noqa: BLE001
                raise ConnectionError(f"REST: {rest_exc!r}; WebSocket: {ws_exc!r}") from ws_exc
            return f"websocket (REST failed: {rest_exc!r})"

    def daemon_stats(self) -> dict[str, Any] | None:
        return self.daemon.control_loop_stats()


class PresentSampler:
    """Measured head pose in a thread (``sink.present()``: the SDK's cached state, no request), for
    command-vs-present tracking errors. ``samples`` = [(perf_counter_ns, Pose)]."""

    def __init__(self, sink: Any, hz: float = 25.0) -> None:
        self.sink = sink
        self.period = 1.0 / hz
        self.samples: list[tuple[int, Pose]] = []
        self.errors = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="present-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            try:
                head, antennas = self.sink.present()
                self.samples.append((time.perf_counter_ns(), pose_from_matrix(head, antennas)))
            except Exception:  # noqa: BLE001 - measurement only
                self.errors += 1


def simulated_like_robot(mini: Any, clock: Callable[[], int] = time.perf_counter_ns) -> "SimulatedSink":
    """Dry run that starts from the robot's measured head pose, antennas and torso yaw (SDK cached state:
    reads only). Falls back to the default sleep pose if the state cannot be read."""
    try:
        head = np.asarray(mini.get_current_head_pose(), dtype=np.float64)
        joints, antennas = mini.get_current_joint_positions()
        start = pose_from_matrix(head, list(antennas))
        return SimulatedSink(start=start, clock=clock, body_yaw_deg=math.degrees(float(joints[0])),
                             origin="robot state (read only)")
    except Exception:  # noqa: BLE001 - e.g. no state received yet: a plain simulation
        return SimulatedSink(clock=clock)


class SimulatedSink:
    """No robot. Follows commands instantly; failures can be injected (tests)."""

    name = "simulated"
    simulated = True

    def __init__(self, start: Pose = SLEEP, mode: str = "disabled",
                 clock: Callable[[], int] = time.perf_counter_ns, body_yaw_deg: float = 0.0,
                 origin: str = "default sleep pose") -> None:
        self.origin = origin  # where the simulated head starts (logged with the controller start)
        self.clock = clock  # timestamps of ``sent`` / ``calls`` (a virtual clock in fast dry runs)
        self.pose = start  # world frame, like the robot's measured pose
        self.body_yaw_deg = body_yaw_deg
        self.mode = mode
        self.wobbling = False
        self.calls: list[tuple[str, int]] = []  # (call, perf_counter_ns)
        self.sent: list[tuple[int, Pose]] = []
        self.fail_send = False
        self.fail_present = False
        self.fail_disable = False
        self.fail_mode_read = False
        self.fail_estop = False
        self.block_send: threading.Event | None = None  # tests: a send that hangs (stuck WebSocket)
        self.stay_put = False  # tests: the head does not follow the commands (pose check must fail)
        self._lock = threading.Lock()

    MAX_LOG = 50_000  # long dry runs: keep the newest half when full (bounded memory)

    def _log(self, call: str) -> None:
        with self._lock:
            self.calls.append((call, self.clock()))
            if len(self.calls) > self.MAX_LOG:
                del self.calls[: self.MAX_LOG // 2]

    def send(self, head: np.ndarray, antennas: list[float]) -> None:
        if self.block_send is not None:
            self.block_send.wait()
        if self.fail_send:
            raise ConnectionError("simulated send failure")
        pose = pose_from_matrix(head, antennas)
        with self._lock:
            self.sent.append((self.clock(), pose))
            if len(self.sent) > self.MAX_LOG:
                del self.sent[: self.MAX_LOG // 2]
            if self.mode == "enabled" and not self.stay_put:
                self.pose = pose

    def enable_motors(self) -> None:
        self._log("enable_motors")
        self.mode = "enabled"

    def disable_motors(self) -> None:
        self._log("disable_motors")
        if self.fail_disable:
            raise ConnectionError("simulated disable failure")
        self.mode = "disabled"

    def motor_mode(self) -> str | None:
        if self.fail_mode_read:
            raise ConnectionError("simulated REST failure")
        return self.mode

    def present(self) -> tuple[np.ndarray, list[float]]:
        if self.fail_present:
            raise ConnectionError("simulated state read failure")
        return self.pose.head_matrix(), self.pose.antennas

    def body_yaw(self) -> float:
        if self.fail_present:
            raise ConnectionError("simulated state read failure")
        return math.radians(self.body_yaw_deg)

    def set_wobbling(self, on: bool) -> None:
        self._log(f"wobbling_{'on' if on else 'off'}")
        self.wobbling = on

    def emergency_torque_off(self) -> str:
        self._log("emergency_torque_off")
        if self.fail_estop:
            raise ConnectionError("simulated REST and WebSocket failure")
        self.mode = "disabled"
        return "simulated"

    def daemon_stats(self) -> dict[str, Any] | None:
        return {"mean_control_loop_frequency": 50.0, "max_control_loop_interval": 0.021, "simulated": True}
