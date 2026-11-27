"""Position / velocity / acceleration characteristics of the SDK's four interpolation methods.

Source of truth (SDK 1.11.0, paths relative to site-packages):
    reachy_mini/utils/interpolation.py:108-114  InterpolationTechnique enum
    reachy_mini/utils/interpolation.py:117-153  time_trajectory(t, method) -> s(tau)
    reachy_mini/motion/goto.py:57-72            GotoMove.evaluate: s drives the head-pose
                                                interpolation and the antenna / body-yaw lerps
Every s(tau) value here comes from *calling* ``time_trajectory``. The closed-form derivatives
in this module are our own derivation; they exist to cross-check the numerical derivatives
(tests/test_interpolation_analysis.py) and to give exact one-sided values at the boundaries
and at the tau = 0.5 seam of the piecewise methods.

Physical scaling for a 1-D coordinate moving by amplitude A in duration D:
    q(t)   = q0 + A * s(t / D)
    q'(t)  = (A / D)    * s'(tau)
    q''(t) = (A / D**2) * s''(tau)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from reachy_mini.utils.interpolation import InterpolationTechnique, time_trajectory

METHODS: tuple[InterpolationTechnique, ...] = (
    InterpolationTechnique.LINEAR,
    InterpolationTechnique.MIN_JERK,
    InterpolationTechnique.EASE_IN_OUT,
    InterpolationTechnique.CARTOON,
)

# easeInOutBack constants exactly as written in the SDK (interpolation.py:137-138).
CARTOON_C1 = 1.70158
CARTOON_C2 = CARTOON_C1 * 1.525

# Ideal play_move evaluation period: play_frequency = 100 Hz (daemon/backend/abstract.py:829,866).
PLAY_MOVE_PERIOD_S = 0.01


def sdk_s(method: InterpolationTechnique, tau: np.ndarray | float) -> np.ndarray:
    """s(tau) evaluated by the SDK's own ``time_trajectory`` (tau must lie in [0, 1])."""
    tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
    return np.fromiter(
        (time_trajectory(float(t), method) for t in tau_arr), dtype=float, count=tau_arr.size
    )


def analytic_ds(method: InterpolationTechnique, tau: np.ndarray | float) -> np.ndarray:
    """ds/dtau - our closed-form derivative of the SDK formula (right-continuous at 0.5)."""
    t = np.atleast_1d(np.asarray(tau, dtype=float))
    if method is InterpolationTechnique.LINEAR:
        return np.ones_like(t)
    if method is InterpolationTechnique.MIN_JERK:
        return 30 * t**2 - 60 * t**3 + 30 * t**4
    if method is InterpolationTechnique.EASE_IN_OUT:
        return np.where(t < 0.5, 4 * t, 4 * (1 - t))
    if method is InterpolationTechnique.CARTOON:
        c2 = CARTOON_C2
        u, w = 2 * t, 2 * t - 2
        return np.where(
            t < 0.5, 3 * (c2 + 1) * u**2 - 2 * c2 * u, 3 * (c2 + 1) * w**2 + 2 * c2 * w
        )
    raise ValueError(f"unknown method {method!r}")


def analytic_dds(method: InterpolationTechnique, tau: np.ndarray | float) -> np.ndarray:
    """d2s/dtau2 - our closed-form second derivative (right-continuous at 0.5)."""
    t = np.atleast_1d(np.asarray(tau, dtype=float))
    if method is InterpolationTechnique.LINEAR:
        return np.zeros_like(t)
    if method is InterpolationTechnique.MIN_JERK:
        return 60 * t - 180 * t**2 + 120 * t**3
    if method is InterpolationTechnique.EASE_IN_OUT:
        return np.where(t < 0.5, 4.0, -4.0)
    if method is InterpolationTechnique.CARTOON:
        c2 = CARTOON_C2
        u, w = 2 * t, 2 * t - 2
        return np.where(t < 0.5, 12 * (c2 + 1) * u - 4 * c2, 12 * (c2 + 1) * w + 4 * c2)
    raise ValueError(f"unknown method {method!r}")


@dataclass(frozen=True)
class Characteristics:
    """Normalised (A = 1, D = 1) characteristics of one interpolation method."""

    method: str
    s_min: float
    tau_at_s_min: float
    s_max: float
    tau_at_s_max: float
    anticipation: float  # max(0, -s_min): fraction of A moved backwards before departing
    overshoot: float  # max(0, s_max - 1): fraction of A beyond the goal
    v_peak: float  # max |s'|; also the peak/average velocity ratio (average = 1)
    tau_at_v_peak: float
    v_start: float  # s'(0+)
    v_end: float  # s'(1-)
    a_peak: float  # sup |s''| including one-sided limits
    a_start: float  # s''(0+)
    a_end: float  # s''(1-)
    v_seam_jump: float  # |s'(0.5-) - s'(0.5+)|
    a_seam_jump: float  # |s''(0.5-) - s''(0.5+)|
    velocity_continuous: bool  # starts and ends at rest with no velocity step
    acceleration_continuous: bool  # no acceleration step at 0, 0.5 or 1
    max_abs_err_numeric_vs_analytic_v: float  # cross-check of our derivation vs the SDK

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def characterize(method: InterpolationTechnique, n: int = 200_001) -> Characteristics:
    """Characterise one method on a dense grid of n points (n odd so that tau = 0.5 is a node)."""
    if n % 2 == 0:
        n += 1
    tau = np.linspace(0.0, 1.0, n)
    s = sdk_s(method, tau)
    v_num = np.gradient(s, tau, edge_order=2)
    v_ana = analytic_ds(method, tau)
    # Compare away from the tau = 0.5 seam (one-sided formulas differ there by design).
    h = tau[1] - tau[0]
    away = np.abs(tau - 0.5) > 2 * h
    cross_err = float(np.max(np.abs(v_num[away] - v_ana[away])))

    i_min, i_max = int(np.argmin(s)), int(np.argmax(s))
    i_vpk = int(np.argmax(np.abs(v_ana)))
    eps = 1e-12
    a_left_seam = float(analytic_dds(method, 0.5 - eps)[0])
    a_right_seam = float(analytic_dds(method, 0.5)[0])
    a_grid = np.abs(analytic_dds(method, tau))
    a_peak = float(max(a_grid.max(), abs(a_left_seam), abs(a_right_seam)))

    v_start, v_end = float(analytic_ds(method, 0.0)[0]), float(analytic_ds(method, 1.0)[0])
    a_start, a_end = float(analytic_dds(method, 0.0)[0]), float(analytic_dds(method, 1.0)[0])
    v_seam_jump = abs(float(analytic_ds(method, 0.5 - eps)[0]) - float(analytic_ds(method, 0.5)[0]))
    seam_jump = abs(a_left_seam - a_right_seam)
    tol = 1e-9
    # The robot is at rest before and after a goto, so a nonzero boundary value is a step.
    velocity_continuous = abs(v_start) < tol and abs(v_end) < tol and v_seam_jump < 1e-6
    acceleration_continuous = (
        velocity_continuous and abs(a_start) < tol and abs(a_end) < tol and seam_jump < 1e-6
    )
    return Characteristics(
        method=method.value,
        s_min=float(s[i_min]),
        tau_at_s_min=float(tau[i_min]),
        s_max=float(s[i_max]),
        tau_at_s_max=float(tau[i_max]),
        anticipation=max(0.0, -float(s[i_min])),
        overshoot=max(0.0, float(s[i_max]) - 1.0),
        v_peak=float(np.abs(v_ana[i_vpk])),
        tau_at_v_peak=float(tau[i_vpk]),
        v_start=v_start,
        v_end=v_end,
        a_peak=a_peak,
        a_start=a_start,
        a_end=a_end,
        v_seam_jump=v_seam_jump,
        a_seam_jump=seam_jump,
        velocity_continuous=velocity_continuous,
        acceleration_continuous=acceleration_continuous,
        max_abs_err_numeric_vs_analytic_v=cross_err,
    )


def physical_peaks(
    chars: Characteristics, amplitude: float, duration: float
) -> dict[str, float | str]:
    """Peak magnitudes for a 1-D move of `amplitude` (any unit, e.g. deg) in `duration` s.

    A method whose velocity steps at the ends (LINEAR: s' = 1 at tau = 0 and 1 while the robot is at
    rest before and after) has an impulsive acceleration there, so peak_accel_per_s2 is inf and the
    size of the step is given in velocity_step_at_ends_per_s (in discrete time the step happens within
    one evaluation interval: (A/D)/dt per sample).
    """
    step = max(abs(chars.v_start), abs(chars.v_end)) * amplitude / duration
    return {
        "method": chars.method,
        "amplitude": amplitude,
        "duration_s": duration,
        "peak_velocity_per_s": chars.v_peak * amplitude / duration,
        "peak_accel_per_s2": chars.a_peak * amplitude / duration**2 if chars.velocity_continuous else float("inf"),
        "velocity_step_at_ends_per_s": step,
        "overshoot_beyond_goal": chars.overshoot * amplitude,
        "anticipation_before_start": chars.anticipation * amplitude,
    }


def endpoint_shortfall(
    method: InterpolationTechnique, duration: float, last_gap: float = PLAY_MOVE_PERIOD_S
) -> float:
    """1 - s(1 - g/D): fraction of the amplitude never commanded when the last evaluation of
    Backend.play_move falls a full gap g = `last_gap` before D.

    play_move evaluates only while ``time.time() - t0 < move.duration`` and nothing after the
    loop commands t = D (daemon/backend/abstract.py:904-925), so t_last lies in [D - g, D) where g
    is the gap that would have followed. On the robot's Linux daemon each gap is at least the
    nominal period (the loop sleeps period - elapsed after every evaluation, abstract.py:921-925)
    plus wake-up latency, so with g = the nominal 10 ms this is the value for an ideal schedule and
    the true worst case is 1 - s(1 - g_max/D) >= this value (g_max unmeasured on the robot). On this
    Windows PC asyncio timers follow a 15.625 ms clock, so single gaps can be shorter or longer.
    Negative values are a residual overshoot (CARTOON approaches the goal from above).
    """
    tau_last = max(0.0, 1.0 - last_gap / duration)
    return 1.0 - float(time_trajectory(tau_last, method))
