"""Head + antenna pose as plain numbers, and conversion to the SDK's 4x4 head matrix.

A :class:`Pose` is (roll, pitch, yaw in degrees; x, y, z in mm; right / left antenna in rad).
Axes follow the SDK (``reachy_mini.utils.create_head_pose``): x forward, y left, z up, extrinsic
"xyz" Euler angles. Signs verified on the robot in HW2: +yaw turns to the robot's own left (the
user's right), +pitch lowers the head (HW2 "sad" +10 deg, "happy" -5 deg), antenna order is
[right, left]. The roll sign was not checked in HW2 (to verify on the robot).

The SDK is not imported here (any ``reachy_mini`` import pulls GStreamer in); the matrix maths is
the same formula and a test compares it with the SDK.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

FIELDS = ("roll", "pitch", "yaw", "x", "y", "z", "ant_r", "ant_l")
ANGLES = slice(0, 3)
OFFSETS = slice(3, 6)
ANTENNAS = slice(6, 8)


@dataclass(frozen=True)
class Pose:
    roll: float = 0.0  # deg
    pitch: float = 0.0  # deg, + = head down
    yaw: float = 0.0  # deg, + = robot's own left
    x: float = 0.0  # mm, forward
    y: float = 0.0  # mm, left
    z: float = 0.0  # mm, up
    ant_r: float = 0.0  # rad, right antenna
    ant_l: float = 0.0  # rad, left antenna

    def vec(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=np.float64)

    @staticmethod
    def from_vec(v: np.ndarray | list[float]) -> "Pose":
        return Pose(*(float(a) for a in v))

    def __add__(self, other: "Pose") -> "Pose":
        return Pose.from_vec(self.vec() + other.vec())

    def __sub__(self, other: "Pose") -> "Pose":
        return Pose.from_vec(self.vec() - other.vec())

    def scaled(self, k: float) -> "Pose":
        return Pose.from_vec(self.vec() * k)

    def head_only(self) -> "Pose":
        return Pose(self.roll, self.pitch, self.yaw, self.x, self.y, self.z, 0.0, 0.0)

    def antennas_only(self) -> "Pose":
        return Pose(ant_r=self.ant_r, ant_l=self.ant_l)

    def head_matrix(self) -> np.ndarray:
        return head_matrix(self.x, self.y, self.z, self.roll, self.pitch, self.yaw)

    @property
    def antennas(self) -> list[float]:
        return [self.ant_r, self.ant_l]

    def rounded(self) -> dict[str, float]:
        return {name: round(float(v), 3) for name, v in zip(FIELDS, self.vec())}


def perk(s: float) -> Pose:
    """Antenna offset: s > 0 raises both antennas (HW2 'happy' direction), s < 0 lowers them."""
    return Pose(ant_r=s, ant_l=-s)


def rotation_xyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Extrinsic x-y-z rotation (= ``Rotation.from_euler("xyz", ...)`` used by the SDK)."""
    r, p, y = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def head_matrix(x_mm: float, y_mm: float, z_mm: float, roll_deg: float, pitch_deg: float,
                yaw_deg: float) -> np.ndarray:
    """4x4 head pose in metres, like ``create_head_pose(x, y, z, roll, pitch, yaw, mm=True, degrees=True)``."""
    m = np.eye(4)
    m[:3, :3] = rotation_xyz(roll_deg, pitch_deg, yaw_deg)
    m[:3, 3] = [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0]
    return m


def pose_from_matrix(m: np.ndarray, antennas: list[float] | tuple[float, float]) -> Pose:
    """Inverse of :func:`head_matrix` (extrinsic xyz; pitch kept in (-90, 90) deg)."""
    r = np.asarray(m)[:3, :3]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    roll = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    t = np.asarray(m)[:3, 3] * 1000.0
    return Pose(roll, pitch, yaw, float(t[0]), float(t[1]), float(t[2]), float(antennas[0]), float(antennas[1]))


def torso_frame(yaw_deg: float) -> np.ndarray:
    """4x4 rotation about z by the torso (body) yaw: world pose = torso_frame(yaw) @ pose in the torso frame."""
    m = np.eye(4)
    m[:3, :3] = rotation_xyz(0.0, 0.0, yaw_deg)
    return m


def nearest_rotation(m: np.ndarray) -> np.ndarray:
    """Closest proper rotation to a 3x3 matrix (SVD). The daemon's head pose matrices and the SDK sleep pose
    constant are not orthonormal (determinant about 1.0005)."""
    u, _s, vt = np.linalg.svd(np.asarray(m, dtype=np.float64)[:3, :3])
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] = -u[:, -1]
    return u @ vt


def rotation_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    # orthonormalize first: with the raw trace, the 1.0005 scale made angles below about 2 deg read as 0
    rel = nearest_rotation(a) @ nearest_rotation(b).T
    sin = np.linalg.norm([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]) / 2.0
    return math.degrees(math.atan2(sin, (np.trace(rel) - 1.0) / 2.0))


def translation_distance_mm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a)[:3, 3] - np.asarray(b)[:3, 3]) * 1000.0)


# SDK 1.11.0 constants (reachy_mini/reachy_mini.py:54-76), copied so this module stays light.
INIT_ANTENNAS = (-0.1745, 0.1745)  # "~10 deg offset to reduce shaking at vertical"
SLEEP_ANTENNAS = (-3.05, 3.05)
SLEEP_HEAD_MATRIX = np.array([
    [0.911, 0.004, 0.413, -0.021],
    [-0.004, 1.0, -0.001, 0.001],
    [-0.413, -0.001, 0.911, -0.044],
    [0.0, 0.0, 0.0, 1.0],
])

#: Neutral pose: head straight (SDK INIT_HEAD_POSE = identity), antennas at the SDK initial angles.
NEUTRAL = Pose(ant_r=INIT_ANTENNAS[0], ant_l=INIT_ANTENNAS[1])
#: The SDK sleep pose as a Pose (about 24 deg pitch down, 44 mm lower, antennas folded).
SLEEP = pose_from_matrix(SLEEP_HEAD_MATRIX, SLEEP_ANTENNAS)
