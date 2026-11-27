"""Head pose in SI units (``Pose6``) and the robot state a motion starts and ends in.

Adapted from HW3 ``../reachy_hw3/reachy_s2s/motion/pose.py`` (HW3 HEAD 0d49b6b): the same extrinsic
"xyz" rotation formula, the inverse conversion and the SVD orthonormalisation. Changed for HW4:
SI units inside (m, rad; CLAUDE.md code rules), deg/mm only through the explicit ``*_deg_mm``
adapters, and the antennas are kept out of the head pose (they are a separate keyframe track).

Axes and signs follow the SDK (``reachy_mini.utils.create_head_pose``,
reachy_mini/utils/__init__.py:14-47: ``Rotation.from_euler("xyz", [roll, pitch, yaw])``, i.e.
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)): x forward, y to the robot's left, z up. Signs checked on the
robot in HW2/HW3: +pitch lowers the head, +yaw turns to the robot's own left, antenna order is
[right, left]. The roll sign has NOT been checked on the robot (HW3 PHASE4_5_DESIGN.md:73).

The SDK is not imported here (the library layer stays SDK-free, docs/ARCHITECTURE.md section 3);
tests compare ``Pose6.matrix()`` with ``create_head_pose(..., degrees=False)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# SDK 1.11.0 constants (reachy_mini/reachy_mini.py:54, 67-76), copied so the library stays SDK-free.
INIT_ANTENNAS: tuple[float, float] = (-0.1745, 0.1745)  # rad [right, left]; "~10 deg offset to reduce shaking at vertical"
SLEEP_ANTENNAS: tuple[float, float] = (-3.05, 3.05)  # rad [right, left]; folded
SLEEP_HEAD_MATRIX = np.array([
    [0.911, 0.004, 0.413, -0.021],
    [-0.004, 1.0, -0.001, 0.001],
    [-0.413, -0.001, 0.911, -0.044],
    [0.0, 0.0, 0.0, 1.0],
])

_POSE_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class Pose6:
    """Head pose relative to the neutral pose (SDK INIT_HEAD_POSE = identity), SI units."""

    x: float = 0.0  # m, forward
    y: float = 0.0  # m, robot's left
    z: float = 0.0  # m, up
    roll: float = 0.0  # rad (sign not verified on the robot)
    pitch: float = 0.0  # rad, + = head down
    yaw: float = 0.0  # rad, + = robot's own left

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.roll, self.pitch, self.yaw], dtype=np.float64)

    @classmethod
    def from_array(cls, v: np.ndarray | list[float] | tuple[float, ...]) -> "Pose6":
        a = [float(x) for x in v]
        if len(a) != 6:
            raise ValueError(f"Pose6 needs 6 values, got {len(a)}")
        return cls(*a)

    @classmethod
    def from_deg_mm(cls, roll_deg: float = 0.0, pitch_deg: float = 0.0, yaw_deg: float = 0.0,
                    x_mm: float = 0.0, y_mm: float = 0.0, z_mm: float = 0.0) -> "Pose6":
        """Boundary adapter: degrees / millimetres in, SI out."""
        return cls(x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0,
                   math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg))

    def to_deg_mm(self) -> dict[str, float]:
        """Boundary adapter for reports and CLI output (keys carry the unit suffix)."""
        return {"x_mm": self.x * 1000.0, "y_mm": self.y * 1000.0, "z_mm": self.z * 1000.0,
                "roll_deg": math.degrees(self.roll), "pitch_deg": math.degrees(self.pitch),
                "yaw_deg": math.degrees(self.yaw)}

    def __add__(self, other: "Pose6") -> "Pose6":
        return Pose6.from_array(self.as_array() + other.as_array())

    def __sub__(self, other: "Pose6") -> "Pose6":
        return Pose6.from_array(self.as_array() - other.as_array())

    def scaled(self, k: float) -> "Pose6":
        return Pose6.from_array(self.as_array() * float(k))

    def matrix(self) -> np.ndarray:
        """4x4 homogeneous head pose in metres, like ``create_head_pose(x, y, z, roll, pitch, yaw, degrees=False)``."""
        return head_matrix(self)


def rotation_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Extrinsic x-y-z rotation in radians (= ``Rotation.from_euler("xyz", ...)`` used by the SDK)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def head_matrix(pose: Pose6) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rotation_xyz(pose.roll, pose.pitch, pose.yaw)
    m[:3, 3] = [pose.x, pose.y, pose.z]
    return m


def pose_from_matrix(m: np.ndarray) -> Pose6:
    """Inverse of :func:`head_matrix` (extrinsic xyz, pitch kept in (-90, 90) deg).

    Matrices measured by the daemon are not exactly orthonormal (det about 1.0005); the rotation is
    orthonormalised first (SVD), as in HW3 and Phase 1 (``analysis.goto_replay.orthonormalize``).
    """
    a = np.asarray(m, dtype=np.float64)
    r = nearest_rotation(a)
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    roll = math.atan2(r[2, 1], r[2, 2])
    yaw = math.atan2(r[1, 0], r[0, 0])
    t = a[:3, 3]
    return Pose6(float(t[0]), float(t[1]), float(t[2]), roll, pitch, yaw)


def nearest_rotation(m: np.ndarray) -> np.ndarray:
    """Closest proper rotation to the 3x3 block of ``m`` (SVD)."""
    u, _s, vt = np.linalg.svd(np.asarray(m, dtype=np.float64)[:3, :3])
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] = -u[:, -1]
    return u @ vt


def rotation_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (rad) of the relative rotation between two poses (orthonormalised first)."""
    rel = nearest_rotation(a) @ nearest_rotation(b).T
    s = np.linalg.norm([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]) / 2.0
    return math.atan2(s, (np.trace(rel) - 1.0) / 2.0)


def translation_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance (m) between the positions of two 4x4 poses."""
    return float(np.linalg.norm(np.asarray(a)[:3, 3] - np.asarray(b)[:3, 3]))


def head_yaw_psi(m: np.ndarray) -> float:
    """Head yaw (rad) as the SDK's automatic body yaw sees it: atan2(-R01, R00).

    This is the yaw that ``inverse_kinematics_safe`` clamps against the body
    (reachy_mini/kinematics/analytical_kinematics.py:80-95, Phase 1 SDK_ANALYSIS section 4.2). It
    equals the Euler yaw only for a level head and wraps at +-180 deg.
    """
    r = np.asarray(m, dtype=np.float64)
    return math.atan2(-r[0, 1], r[0, 0])


def antennas_droop(d: float, base: tuple[float, float] = INIT_ANTENNAS) -> tuple[float, float]:
    """Both antennas turned by ``d`` rad toward the folded (sleep) side: right ``-d``, left ``+d``.

    ``d < 0`` turns them the other way (toward and past vertical, "perk"). The direction is
    INFERRED from the SDK sleep pose (right -3.05, left +3.05 rad = folded down) and HW3's "perk"
    helper; it has not been checked on the robot (Phase 1 SDK_ANALYSIS section 8).
    """
    return (base[0] - d, base[1] + d)


@dataclass(frozen=True)
class RobotState:
    """Everything a motion commands: head pose, antennas [right, left] (rad) and body yaw (rad)."""

    head: Pose6 = field(default_factory=Pose6)
    antennas: tuple[float, float] = INIT_ANTENNAS
    body_yaw: float = 0.0

    def __post_init__(self) -> None:
        if len(self.antennas) != 2:
            raise ValueError("antennas must be (right, left)")
        object.__setattr__(self, "antennas", (float(self.antennas[0]), float(self.antennas[1])))


#: Rest pose shared by every motion as start and end (head straight = SDK INIT_HEAD_POSE, antennas at the
#: SDK initial angles, body 0). Expressive and neutral variants both start and end here.
REST = RobotState()
