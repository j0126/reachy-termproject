"""State poses and event gestures (docs/PHASE4_5_DESIGN.md section 4).

All values are deviations from the neutral pose at intensity 1.0 (the engine scales them). They
were sized so that every planned move stays under the default speed / acceleration caps (40 deg/s,
400 deg/s^2, 15 mm/s, 150 mm/s^2, 1.5 rad/s, 10 rad/s^2) and changes no Stewart joint by more than
about 7 deg (offline inverse kinematics, see the design); tests check both.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from .pose import Pose, perk
from .trajectory import bump, plateau

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class StatePreset:
    name: str
    offset: Pose  # static deviation from neutral
    transition_s: float  # minimum-jerk move from the current pose
    dwell_s: float = 0.0  # keep the previous pose this long first (PROCESSING: skipped by short Cloud turns)
    life: Callable[[float], Pose] | None = None  # small continuous motion, t = s since the move started


def _breathing(t: float, scale: float) -> Pose:
    return Pose(pitch=1.0 * scale * math.sin(TWO_PI * 0.2 * t),
                z=1.0 * scale * math.sin(TWO_PI * 0.2 * t + math.pi / 2))


def _glance(t: float) -> Pose:
    """Look 5 deg to one side for 2 s every 12.5 s (first after 8 s), alternating sides."""
    if t < 8.0:
        return Pose()
    k = int((t - 8.0) // 12.5)
    s = (t - 8.0 - 12.5 * k) / 2.0
    side = 1.0 if k % 2 == 0 else -1.0
    return Pose(yaw=5.0 * side * plateau(s, 0.3))


def idle_life(t: float) -> Pose:
    return _breathing(t, 1.0) + perk(0.04 * math.sin(TWO_PI * 0.1 * t)) + _glance(t)


def listening_life(t: float) -> Pose:
    return _breathing(t, 0.5)


def thinking_life(t: float) -> Pose:
    """Right antenna fidget; after 3 s the head sweeps slowly from its +6 deg yaw to -6 deg and back."""
    sweep = 0.0 if t < 3.0 else -12.0 * 0.5 * (1.0 - math.cos(TWO_PI * (t - 3.0) / 4.0))
    return Pose(yaw=sweep, ant_r=0.06 * math.sin(TWO_PI * 0.7 * t))


LISTENING_POSE = Pose(pitch=-4.0, z=1.5) + perk(0.20)

PRESETS: dict[str, StatePreset] = {
    "IDLE": StatePreset("IDLE", Pose(pitch=2.0), 0.8, life=idle_life),
    "LISTENING": StatePreset("LISTENING", LISTENING_POSE, 0.4, life=listening_life),
    "PROCESSING": StatePreset("PROCESSING", Pose(roll=6.0, yaw=6.0, pitch=-4.0, ant_r=0.30), 0.6, dwell_s=0.25,
                              life=thinking_life),
    "SPEAKING": StatePreset("SPEAKING", Pose(pitch=-2.0) + perk(0.10), 0.4),
    # barge-in: drop whatever was moving and settle into the listening pose
    "INTERRUPTED": StatePreset("INTERRUPTED", LISTENING_POSE, 0.35),
    "ERROR": StatePreset("ERROR", Pose(pitch=4.0) + perk(-0.15), 0.5),
}


@dataclass(frozen=True)
class Gesture:
    name: str
    duration_s: float
    shape: Callable[[float], Pose]  # s in [0, 1] -> deviation; zero (with zero velocity) at both ends
    head: bool = True  # moves the head (not used on the head while SDK wobbling drives it)

    def at(self, s: float) -> Pose:
        return self.shape(s) if 0.0 < s < 1.0 else Pose()


def nod(depth_deg: float = 3.0) -> Gesture:
    """Small nod: head down by ``depth_deg`` and back (0.6 s), antennas lift a little."""
    return Gesture("nod", 0.6, lambda s: Pose(pitch=depth_deg * bump(s)) + perk(0.10 * bump(s)))


ANTENNA_PERK = Gesture("antenna", 0.6, lambda s: perk(0.15 * bump(s)), head=False)
INTERRUPT_PERK = Gesture("interrupt", 0.6, lambda s: perk(0.08 * bump(s)), head=False)
#: small "no" head shake (yaw; the roll sign was never checked on the robot) with antennas lowered
ERROR_SHAKE = Gesture("error", 1.4, lambda s: Pose(yaw=2.5 * math.sin(2.0 * TWO_PI * s) * bump(s))
                      + perk(-0.10 * bump(s)))
MERGED_NOD = Gesture("merged_nod", 0.5, lambda s: Pose(pitch=2.0 * bump(s)))

#: antennas lift while the user speaks (LISTENING), on top of the listening pose (minimum-jerk moves)
ATTENTION_PERK = 0.10
ATTENTION_ATTACK_S = 0.4
ATTENTION_RELEASE_S = 0.6


def speech_started_gesture(kind: str, nod_deg: float) -> Gesture | None:
    if kind == "nod":
        return nod(nod_deg)
    if kind == "antenna":
        return ANTENNA_PERK
    if kind == "off":
        return None
    raise ValueError(f"speech_started_gesture must be nod | antenna | off, got {kind!r}")
