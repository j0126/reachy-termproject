"""The SDK client's wake-up, as the Motion Lab sends it to a connected robot (and ``sim-startup`` replays it in MuJoCo).

``ReachyMini.wake_up()`` (reachy_mini 1.11.0, paths relative to site-packages/reachy_mini/, reachy_mini.py:724-738)::

    goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS_JOINT_POSITIONS, duration=2)  # MIN_JERK, body_yaw=0.0 (680)
    time.sleep(0.1)
    media.play_sound("wake_up.wav")  # no_media client: audio is None -> a warning only (media/media_manager.py:264-278)
    goto_target(roll 20 deg, duration=0.2)
    goto_target(INIT_HEAD_POSE, duration=0.2)

``send_wake_up`` sends the same three gotos through the backend's ``goto_target`` (each one recorded as a command; a
completion the daemon did not confirm raises, backends/sdk.py), with the antennas passed as their INIT values where
wake_up passes none (the first goto put them there) and the body yaw 0.0 passed explicitly. The sound is left out
(it is a no-op for this client). This is NOT the daemon's ``Backend.wake_up`` (REST ``POST /api/move/play/wake_up``,
daemon/backend/abstract.py:1284-1323): that one ends with the wake-up callback, which on the wireless robot launches
the configured startup app once (daemon/app/main.py:176-206, daemon/app/startup_app.py:305-314) - a second controller
next to ours. A client's gotos go through ``Backend.goto_target`` -> ``play_move`` (abstract.py:944-977) and never
call it. The goto itself starts from the daemon's MEASURED pose (``get_present_head_pose``, abstract.py:966-977).

``check_wake_up`` models the gotos with the planner (one goto = one MIN_JERK ``plan_transition`` = ``GotoMove`` on
the grid) from a start state and checks them: the 2 s goto to INIT under the start-up rules of that start (the
envelope widened only as far as the start already is: ``safety.startup.prepare_startup_ramp``), the pause, the roll
and the way back under the default initial policy. ``computed``: the daemon interpolates the real gotos at about
100 Hz from its own measured pose; ``send_wake_up`` records what was sent, the caller records what was measured.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from reachy_motion.library.pose import REST, Pose6, RobotState
from reachy_motion.safety.policy import DEFAULT_POLICY, SafetyPolicy

GOTO_INIT_S = 2.0  # reachy_mini.py:726
PAUSE_S = 0.1  # reachy_mini.py:727
ROLL_DEG = 20.0  # reachy_mini.py:732-734
ROLL_S = 0.2  # reachy_mini.py:735, 738
DURATION_S = GOTO_INIT_S + PAUSE_S + 2 * ROLL_S
#: INIT = the SDK's INIT_HEAD_POSE (identity) + INIT_ANTENNAS_JOINT_POSITIONS + body 0 (reachy_mini.py:54, 67) = REST
INIT = REST
ROLLED = replace(REST, head=Pose6.from_deg_mm(roll_deg=ROLL_DEG))


def gotos() -> list[tuple[str, np.ndarray, list[float], float]]:
    """(label, head 4x4, antennas, duration) of the three gotos, built with the SDK's own constants and rotation."""
    from reachy_mini.reachy_mini import INIT_ANTENNAS_JOINT_POSITIONS, INIT_HEAD_POSE
    from reachy_mini.utils.rotation import Rotation as R

    ant = [float(a) for a in INIT_ANTENNAS_JOINT_POSITIONS]
    roll = np.array(INIT_HEAD_POSE, dtype=float).copy()
    roll[:3, :3] = R.from_euler("xyz", [ROLL_DEG, 0, 0], degrees=True).as_matrix()
    init = np.array(INIT_HEAD_POSE, dtype=float)
    return [(f"goto INIT {GOTO_INIT_S:.1f} s + antennas", init, ant, GOTO_INIT_S),
            (f"goto roll {ROLL_DEG:g} deg {ROLL_S:.1f} s", roll, ant, ROLL_S),
            (f"goto INIT {ROLL_S:.1f} s", init.copy(), ant, ROLL_S)]


def send_wake_up(backend: Any, cancel: threading.Event | None = None, clock: Any = None) -> list[tuple[str, int]]:
    """The three gotos (module docstring) on ``backend.goto_target(head, antennas, body_yaw, duration, method)``;
    returns (label, send time ns) of each goto that was started. A set ``cancel`` sends no further goto (one that
    runs is not interrupted: the daemon interpolates it, abstract.py:904-925). ``clock`` (now_ns / sleep_until) times
    the pause; default the real clock. Raises what ``goto_target`` raises."""
    if clock is None:
        from reachy_motion.executor.clock import RealClock

        clock = RealClock()
    sent: list[tuple[str, int]] = []
    for i, (label, head, ant, d) in enumerate(gotos()):
        if cancel is not None and cancel.is_set():
            break
        sent.append((label, int(clock.now_ns())))
        backend.goto_target(head, ant, 0.0, d, "minjerk")
        if i == 0:
            clock.sleep_until(clock.now_ns() + int(round(PAUSE_S * 1e9)))
    return sent


@dataclass
class WakeUpCheck:
    """The computed check of the wake-up gotos from one start state (module docstring)."""

    runnable: bool
    problems: list[str]
    stewart_speed_ratio: float
    antenna_speed_ratio: float
    min_margin_deg: float
    max_joint_travel_deg: float
    max_travel_joint: str
    digest: str
    start_text: str
    timeline: Any = field(repr=False, default=None)
    data_kind: str = "computed"


def check_wake_up(start: RobotState, policy: SafetyPolicy = DEFAULT_POLICY, rate_hz: float = 50.0) -> WakeUpCheck:
    """Model and check the wake-up from ``start`` (module docstring). Never raises for a bad start: refused."""
    from reachy_motion.library.spec import Easing
    from reachy_motion.planner import concat, plan_transition
    from reachy_motion.robot_r1 import ramp_travel
    from reachy_motion.safety.checker import prepare_timeline
    from reachy_motion.safety.startup import prepare_startup_ramp

    p = start.head.to_deg_mm()
    text = (f"roll {p['roll_deg']:.1f}°, pitch {p['pitch_deg']:.1f}°, yaw {p['yaw_deg']:.1f}°, z {p['z_mm']:.1f} mm, "
            f"안테나 {math.degrees(start.antennas[0]):.0f}° / {math.degrees(start.antennas[1]):.0f}°, 몸통 "
            f"{math.degrees(start.body_yaw):.1f}°")
    try:
        first = prepare_startup_ramp(start, INIT, policy, rate_hz, duration=GOTO_INIT_S)
        tail = concat(concat(plan_transition(INIT, INIT, PAUSE_S, Easing.MIN_JERK, rate_hz, name="pause"),
                             plan_transition(INIT, ROLLED, ROLL_S, Easing.MIN_JERK, rate_hz, name="roll")),
                      plan_transition(ROLLED, INIT, ROLL_S, Easing.MIN_JERK, rate_hz, name="back"))
        rest = prepare_timeline(tail, policy)
        model = replace(concat(first.timeline, tail, name="sdk_wake_up"), variant="model")
    except ValueError as exc:
        return WakeUpCheck(False, [f"계획할 수 없음: {exc}"], math.nan, math.nan, math.nan, math.nan, "-", "", text)
    problems = [f"INIT까지 goto (기동 규칙): {v.code} ({v.message})" for v in first.report.violations]
    problems += [f"roll·복귀 (초기 정책): {v.code} ({v.message})" for v in rest.report.violations]
    m1, m2 = first.report.metrics, rest.report.metrics
    travel = ramp_travel(first)

    def worst(key: str, f: Callable[..., float]) -> float:
        vals = [float(m[key]) for m in (m1, m2) if isinstance(m.get(key), (int, float)) and math.isfinite(m[key])]
        return f(vals) if vals else math.nan

    return WakeUpCheck(bool(first.runnable and rest.runnable), problems,
                       worst("peak_stewart_speed_ratio", max), worst("peak_antenna_speed_ratio", max),
                       worst("min_stewart_margin_deg", min), float(travel["max_deg"]), str(travel["joint"]),
                       model.digest(), text, model)


__all__ = ["GOTO_INIT_S", "PAUSE_S", "ROLL_DEG", "ROLL_S", "DURATION_S", "INIT", "ROLLED", "gotos", "send_wake_up",
           "WakeUpCheck", "check_wake_up"]
