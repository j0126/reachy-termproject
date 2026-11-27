"""State-based expressive robot motion (Phase 4.5, docs/PHASE4_5_DESIGN.md).

Importing this package does not import the Reachy Mini SDK (GStreamer); the SDK's kinematics,
interpolation and speech-sway modules are loaded lazily when a controller is created / used.
"""

from .controller import (
    ACTIVE,
    ESTOP,
    FAULT,
    HOLD,
    MODES,
    SHUTTING_DOWN,
    STARTING,
    STOPPED,
    TORQUE_OFF,
    TORQUE_ON_MODES,
    MotionController,
)
from .pose import NEUTRAL, SLEEP, Pose
from .sinks import MotionSink, RobotSink, SimulatedSink
from .speech import PlaybackTap

__all__ = [
    "ACTIVE", "ESTOP", "FAULT", "HOLD", "MODES", "SHUTTING_DOWN", "STARTING", "STOPPED", "TORQUE_OFF",
    "TORQUE_ON_MODES", "MotionController", "MotionSink", "NEUTRAL", "Pose", "PlaybackTap", "RobotSink", "SLEEP",
    "SimulatedSink",
]
