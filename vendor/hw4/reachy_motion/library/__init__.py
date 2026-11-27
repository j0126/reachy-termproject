"""Motion library: SI pose types, keyframe specs and the A/B/C catalog. SDK-free (plain data)."""

from reachy_motion.library.pose import (
    INIT_ANTENNAS,
    REST,
    SLEEP_ANTENNAS,
    Pose6,
    RobotState,
    antennas_droop,
)
from reachy_motion.library.spec import (
    MOTION_NAMES,
    VARIANTS,
    AntennaKey,
    BodyKey,
    Easing,
    HeadKey,
    MotionSpec,
    Principle,
    PrincipleUse,
)

__all__ = [
    "INIT_ANTENNAS", "REST", "SLEEP_ANTENNAS", "Pose6", "RobotState", "antennas_droop",
    "MOTION_NAMES", "VARIANTS", "AntennaKey", "BodyKey", "Easing", "HeadKey", "MotionSpec", "Principle",
    "PrincipleUse",
]
