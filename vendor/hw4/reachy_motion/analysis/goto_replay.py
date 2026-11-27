"""Offline replay of the daemon's goto_target command pipeline (no daemon, no robot).

Reproduces, with the SDK's own classes (SDK 1.11.0, paths relative to site-packages):
  1. Backend.goto_target builds GotoMove(start = present pose, target, duration, method)
     (reachy_mini/daemon/backend/abstract.py:944-977, reachy_mini/motion/goto.py:15-74).
  2. Backend.play_move evaluates it at play_frequency = 100 Hz while time.time()-t0 < D and
     stores the result as the backend's targets (abstract.py:826-925, 596-668).
  3. The real-robot control loop runs at 50 Hz (reachy_mini/daemon/backend/robot/backend.py:71,
     154-198). Each tick it first writes the *current* joint target to the motors
     (backend.py:203-228), then reads the present joints and recomputes IK of the latest target
     pose (backend.py:251-257 -> abstract.py:544-594). The IK result is therefore written on
     the *next* tick.
  IK engine: AnalyticalKinematics() with automatic_body_yaw=True, the daemon default
  (abstract.py:152, 210-213; reachy_mini/kinematics/analytical_kinematics.py:33, 80-95).

Everything returned is a *model of the commanded signal* derived from source code. It is not a
measurement of the robot or of MuJoCo (the MuJoCo backend has its own loop; see the Phase 1
report before reusing control_rate for simulation).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from reachy_mini.kinematics import AnalyticalKinematics
from reachy_mini.motion.goto import GotoMove
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import InterpolationTechnique, distance_between_poses

# Head joint vector order used by the backends: [0] body rotation, [1:] Stewart motors
# (robot/backend.py:206-209 writes target[1:] to the platform and target[0] to the body).
JOINT_NAMES: tuple[str, ...] = (
    "body_yaw",
    "stewart_1",
    "stewart_2",
    "stewart_3",
    "stewart_4",
    "stewart_5",
    "stewart_6",
)
STEWART = slice(1, 7)

EVAL_RATE_HZ = 100.0  # play_move default play_frequency (abstract.py:829)
ROBOT_CONTROL_RATE_HZ = 50.0  # RobotBackend.control_loop_frequency (robot/backend.py:71)


@dataclass(frozen=True)
class HeadTarget:
    """Head pose in the SDK convention of create_head_pose (reachy_mini/utils/__init__.py:14-47):
    translation in mm, rotation as roll/pitch/yaw in degrees, Rotation.from_euler("xyz")."""

    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    def matrix(self) -> np.ndarray:
        return create_head_pose(
            x=self.x_mm,
            y=self.y_mm,
            z=self.z_mm,
            roll=self.roll_deg,
            pitch=self.pitch_deg,
            yaw=self.yaw_deg,
            mm=True,
            degrees=True,
        )


@dataclass(frozen=True)
class GotoCase:
    """One goto_target call, starting from rest at `start`."""

    target: HeadTarget
    duration: float
    method: InterpolationTechnique
    start: HeadTarget = HeadTarget()
    body_yaw_start_deg: float = 0.0
    body_yaw_target_deg: float = 0.0
    antennas_start_deg: tuple[float, float] = (0.0, 0.0)
    antennas_target_deg: tuple[float, float] = (0.0, 0.0)

    def build_move(self) -> GotoMove:
        """The SDK's own GotoMove, constructed as Backend.goto_target does."""
        return GotoMove(
            start_head_pose=self.start.matrix(),
            target_head_pose=self.target.matrix(),
            start_antennas=np.deg2rad(np.array(self.antennas_start_deg, dtype=float)),
            target_antennas=np.deg2rad(np.array(self.antennas_target_deg, dtype=float)),
            start_body_yaw=float(np.deg2rad(self.body_yaw_start_deg)),
            target_body_yaw=float(np.deg2rad(self.body_yaw_target_deg)),
            duration=self.duration,
            method=self.method,
        )


def new_kinematics() -> AnalyticalKinematics:
    """The daemon's default IK engine (abstract.py:210-213): AnalyticalKinematics()."""
    return AnalyticalKinematics()


def ik_joints(kin: AnalyticalKinematics, pose: np.ndarray, body_yaw_rad: float) -> np.ndarray:
    """7 joint angles [body_yaw, stewart_1..6] in rad (NaN entries = infeasible)."""
    return np.asarray(kin.ik(pose, body_yaw=body_yaw_rad), dtype=float)


def eval_times(duration: float, rate: float = EVAL_RATE_HZ) -> np.ndarray:
    """Ideal play_move instants k/rate, kept while k/rate < duration (abstract.py:904)."""
    k = np.arange(int(np.ceil(duration * rate)) + 1)
    t = k / rate
    return t[t < duration]


@dataclass
class GotoReplay:
    case: GotoCase
    t_dense: np.ndarray  # continuous-time reference, s
    joints_dense: np.ndarray  # (N, 7) rad, IK of the reference
    t_eval: np.ndarray  # play_move evaluation instants, s
    joints_eval: np.ndarray  # (M, 7) rad, IK of each evaluated target
    t_ctrl: np.ndarray  # control-loop ticks, s
    joints_written: np.ndarray  # (K, 7) rad, joint goal written at each tick
    ik_failures: int  # NaN joint vectors among dense + evaluated samples

    def joint_velocity_dense(self) -> np.ndarray:
        """(N, 7) rad/s - the speed the motors are asked for if they tracked the reference."""
        return np.gradient(self.joints_dense, self.t_dense, axis=0)

    def written_zoh(self, t: np.ndarray) -> np.ndarray:
        """Zero-order hold of the written goals at times t.

        Before tick 0 the goal is the pre-move one (the start pose, joints_eval[0]); a tick at
        exactly t is already in effect. Times are compared on an integer microsecond grid.
        """
        t_us = _to_us(np.atleast_1d(np.asarray(t, dtype=float)))
        idx = np.searchsorted(_to_us(self.t_ctrl), t_us, side="right") - 1
        out = self.joints_written[np.clip(idx, 0, len(self.t_ctrl) - 1)].copy()
        out[idx < 0] = self.joints_eval[0]
        return out


def _to_us(t: np.ndarray) -> np.ndarray:
    """Seconds -> integer microseconds, so that coinciding instants compare exactly."""
    return np.round(np.asarray(t, dtype=float) * 1_000_000).astype(np.int64)


def _check_control_args(control_rate: float, control_phase: float, write_delay_ticks: int) -> None:
    if not 0.0 <= control_phase < 1.0 / control_rate:
        raise ValueError("control_phase must be in [0, 1/control_rate)")
    if write_delay_ticks not in (0, 1):
        raise ValueError("write_delay_ticks must be 0 (MuJoCo) or 1 (robot)")


def control_sampling(
    t_eval: np.ndarray,
    joints_eval: np.ndarray,
    duration: float,
    control_rate: float = ROBOT_CONTROL_RATE_HZ,
    control_phase: float = 0.0,
    write_delay_ticks: int = 1,
    tail_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """(t_ctrl, joints_written): what the control loop writes at each tick for given evaluated targets.

    Head joints only. The target held at a tick is the latest evaluated sample (the start pose
    before t0). write_delay_ticks: 1 = real robot / mockup (the head goal written in tick j is the
    IK computed in tick j-1, robot/backend.py:200-261); 0 = MuJoCo (IK then ctrl in the same tick,
    mujoco/backend.py:245-260). Antenna targets skip IK and are written at the next tick on the
    robot, so on hardware they lead the head by one tick - not modelled here.
    """
    _check_control_args(control_rate, control_phase, write_delay_ticks)
    period = 1.0 / control_rate
    n_ticks = int(np.ceil((duration + tail_s) / period)) + 1
    # integer microseconds: phase + n*period in floating point lands just before an evaluation
    # instant it should coincide with (0.01 + 3*0.02 < 0.07) and would pick the previous sample
    t_ctrl_us = int(round(control_phase * 1_000_000)) + np.arange(n_ticks, dtype=np.int64) * int(
        round(period * 1_000_000))
    t_ctrl = t_ctrl_us / 1_000_000
    idx = np.searchsorted(_to_us(t_eval), t_ctrl_us, side="right") - 1
    ik_at_tick = np.where(idx[:, None] >= 0, joints_eval[np.clip(idx, 0, None)], joints_eval[0])
    if write_delay_ticks == 1:
        joints_written = np.vstack([joints_eval[:1], ik_at_tick[:-1]])
    else:
        joints_written = ik_at_tick.copy()
    return t_ctrl, joints_written


def pipeline_lag(replay: GotoReplay, joint: int, control_phase: float, write_delay_ticks: int,
                 grid_s: float = 0.0005, max_lag: float = 0.1) -> float:
    """Best-fit lag (s) of the written joint goal behind the continuous reference, for one control phase.

    The ZOH is sampled at the centres of a `grid_s` grid: ticks fall on grid points (phase and
    period are multiples of it), so sampling at the points themselves would bias the mean hold by
    half a cell and tie the lag search between two neighbouring values.
    """
    from reachy_motion.analysis.metrics import estimate_lag

    duration = replay.case.duration
    t_ctrl, written = control_sampling(replay.t_eval, replay.joints_eval, duration,
                                       control_phase=control_phase, write_delay_ticks=write_delay_ticks)
    r = replace(replay, t_ctrl=t_ctrl, joints_written=written)
    t_fine = np.arange(0.0, duration + 0.1, grid_s) + grid_s / 2
    return estimate_lag(r.t_dense, r.joints_dense[:, joint], t_fine, r.written_zoh(t_fine)[:, joint],
                        max_lag=max_lag, step=grid_s)


def replay_goto(
    case: GotoCase,
    eval_rate: float = EVAL_RATE_HZ,
    control_rate: float = ROBOT_CONTROL_RATE_HZ,
    control_phase: float = 0.0,
    dense_rate: float = 1000.0,
    tail_s: float = 0.1,
    write_delay_ticks: int = 1,
) -> GotoReplay:
    """Model the commanded joint signal of one goto_target call (see module docstring).

    control_phase in [0, 1/control_rate): offset of the first control tick after play_move's t0
    (the two loops run in different threads, so the phase is arbitrary).
    write_delay_ticks: 1 = real-robot backend (goal written before IK in each tick,
    robot/backend.py:200-261); 0 = MuJoCo backend (IK then ctrl write in the same tick,
    mujoco/backend.py:245-260).
    """
    _check_control_args(control_rate, control_phase, write_delay_ticks)
    kin = new_kinematics()
    move = case.build_move()
    duration = case.duration

    def joints_at(t: float) -> np.ndarray:
        head, _antennas, body_yaw = move.evaluate(min(max(t, 0.0), duration))
        return ik_joints(kin, head, float(body_yaw))

    t_dense = np.linspace(0.0, duration, int(round(duration * dense_rate)) + 1)
    joints_dense = np.array([joints_at(t) for t in t_dense])

    t_eval = eval_times(duration, eval_rate)
    joints_eval = np.array([joints_at(t) for t in t_eval])
    t_ctrl, joints_written = control_sampling(
        t_eval, joints_eval, duration, control_rate, control_phase, write_delay_ticks, tail_s
    )

    failures = int(np.isnan(joints_dense).any(axis=1).sum() + np.isnan(joints_eval).any(axis=1).sum())
    return GotoReplay(
        case=case,
        t_dense=t_dense,
        joints_dense=joints_dense,
        t_eval=t_eval,
        joints_eval=joints_eval,
        t_ctrl=t_ctrl,
        joints_written=joints_written,
        ik_failures=failures,
    )


def orthonormalize(rot: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix (SVD). The SDK's FK output and SLEEP_HEAD_POSE have det ~1.0005."""
    u, _, vt = np.linalg.svd(np.asarray(rot, dtype=float))
    return u @ vt


def rotation_error_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Angle between two rotations after SVD orthonormalisation.

    The SDK's trace-based delta_angle_between_mat_rot (utils/interpolation.py:156-179) clips
    (tr - 1)/2 to [-1, 1]; with a matrix scaled by 1.0005 it reads errors below ~1.5 deg as 0.
    """
    rel = orthonormalize(r_a) @ orthonormalize(r_b).T
    return float(np.rad2deg(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))))


def _pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = np.asarray(a), np.asarray(b)
    d_pos_m, _, _ = distance_between_poses(a, b)
    return d_pos_m * 1000.0, rotation_error_deg(a[:3, :3], b[:3, :3])


def fk_roundtrip(target: HeadTarget, body_yaw_rad: float = 0.0, steps: int = 20) -> dict[str, float]:
    """FK(IK(pose)) error, cold and warm-started.

    The FK of AnalyticalKinematics is a stateful Newton iteration (analytical_kinematics.py:99-142;
    the rust object keeps the last solution). 'cold': a fresh engine (FK state = sleep pose,
    analytical_kinematics.py:57-60) with 20 iterations, as at robot start-up
    (robot/backend.py:176-179). 'warm': start converged at neutral and walk to the pose in
    `steps` small increments with 3 iterations each (the default, as the 50 Hz loop calls fk
    every tick), then 20 iterations. Large cold jumps can converge to another FK branch.
    """
    goal = target.matrix()
    cold = new_kinematics()
    q_goal = ik_joints(cold, goal, body_yaw_rad)
    cold_err = _pose_error(cold.fk(q_goal, no_iterations=20), goal)

    warm = new_kinematics()
    warm.fk(ik_joints(warm, np.eye(4), 0.0), no_iterations=50)
    base = asdict(target)
    for a in np.linspace(0.0, 1.0, steps + 1)[1:]:
        step = HeadTarget(**{k: v * a for k, v in base.items()})
        warm.fk(ik_joints(warm, step.matrix(), body_yaw_rad * a), no_iterations=3)
    warm_err = _pose_error(warm.fk(q_goal, no_iterations=20), goal)
    return {
        "fk_cold_pos_err_mm": cold_err[0],
        "fk_cold_ang_err_deg": cold_err[1],
        "fk_warm_pos_err_mm": warm_err[0],
        "fk_warm_ang_err_deg": warm_err[1],
    }


def ik_table(poses: dict[str, HeadTarget], body_yaw_deg: float = 0.0) -> list[dict[str, object]]:
    """IK (deg) of each pose, plus FK(IK(pose)) round-trip errors. Computed, not measured."""
    kin = new_kinematics()  # IK is analytical and stateless: one engine is enough
    body_yaw_rad = float(np.deg2rad(body_yaw_deg))
    rows: list[dict[str, object]] = []
    for name, target in poses.items():
        joints = ik_joints(kin, target.matrix(), body_yaw_rad)
        ok = bool(np.all(np.isfinite(joints)))
        row: dict[str, object] = {"pose": name, **asdict(target), "body_yaw_cmd_deg": body_yaw_deg}
        row["ik_ok"] = ok
        for i, jn in enumerate(JOINT_NAMES):
            row[f"{jn}_deg"] = float(np.rad2deg(joints[i])) if ok else float("nan")
        if ok:
            row.update(fk_roundtrip(target, body_yaw_rad))
        rows.append(row)
    return rows


TASK_COORDS: tuple[str, ...] = ("x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg")


def ik_sensitivity(at: HeadTarget = HeadTarget(), h: float = 0.1) -> np.ndarray:
    """Central-difference Jacobian d(joint deg) / d(task coordinate) at `at`, shape (7, 6).

    Units: deg per mm for x/y/z columns, deg per deg for roll/pitch/yaw columns.
    Multiply by a task-space speed to get the joint speed it requires (locally).
    """
    kin = new_kinematics()
    base = asdict(at)
    jac = np.zeros((len(JOINT_NAMES), len(TASK_COORDS)))
    for k, coord in enumerate(TASK_COORDS):
        plus = HeadTarget(**{**base, coord: base[coord] + h})
        minus = HeadTarget(**{**base, coord: base[coord] - h})
        j_plus = ik_joints(kin, plus.matrix(), 0.0)
        j_minus = ik_joints(kin, minus.matrix(), 0.0)
        jac[:, k] = np.rad2deg(j_plus - j_minus) / (2 * h)
    return jac
