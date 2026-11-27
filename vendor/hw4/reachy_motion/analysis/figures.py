"""Report figures for Phase 1 (matplotlib, static PNG, light theme).

Colour follows identity in a fixed order (dataviz reference palette, categorical slots 1-6);
line style is a secondary encoding so the four methods stay distinguishable in grayscale/CVD.
Every figure title says the data are *computed* from SDK functions - none is a measurement.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reachy_mini.utils.interpolation import InterpolationTechnique  # noqa: E402

from reachy_motion.analysis.goto_replay import (  # noqa: E402
    JOINT_NAMES,
    GotoCase,
    GotoReplay,
    HeadTarget,
    replay_goto,
)
from reachy_motion.analysis.interpolation import (  # noqa: E402
    METHODS,
    analytic_dds,
    analytic_ds,
    characterize,
    endpoint_shortfall,
    sdk_s,
)
from reachy_motion.sdk_env import installed_version  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

METHOD_COLOR = {
    InterpolationTechnique.LINEAR: "#2a78d6",
    InterpolationTechnique.MIN_JERK: "#eb6834",
    InterpolationTechnique.EASE_IN_OUT: "#1baf7a",
    InterpolationTechnique.CARTOON: "#eda100",
}
METHOD_DASH = {
    InterpolationTechnique.LINEAR: "-",
    InterpolationTechnique.MIN_JERK: (0, (6, 2)),
    InterpolationTechnique.EASE_IN_OUT: (0, (3, 1.5, 1, 1.5)),
    InterpolationTechnique.CARTOON: (0, (1.2, 1.2)),
}
MOTOR_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

LW = 1.6
COMPUTED_NOTE = (f"computed from reachy_mini {installed_version('reachy-mini')} source functions"
                 " - not a measurement")


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        }
    )


def _plain_log_ticks(ax: plt.Axes, axis: str, ticks: list[float]) -> None:
    """Readable ticks (0.2, 0.5, 1, 2 ...) instead of 2x10^-1 on a log axis."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    target.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _pos: f"{v:g}"))
    target.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _finish(fig: plt.Figure, title: str, path: Path, bottom: float = 0.03) -> Path:
    fig.suptitle(title, color=INK, fontsize=11, fontweight="bold", x=0.01, ha="left")
    # Reserve a strip at the bottom so the provenance note never overlaps axes or legends
    # (figures with an 'outside lower' legend need a taller strip).
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, bottom, 1.0, 1.0 - bottom))  # (left, bottom, width, height)
    fig.text(0.01, 0.006, COMPUTED_NOTE, color=MUTED, fontsize=7.5, ha="left", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_normalized_profiles(path: Path) -> Path:
    """s, s', s'' of the four methods on normalised time (A = 1, D = 1)."""
    _style()
    tau = np.linspace(0.0, 1.0, 4001)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True, constrained_layout=True)
    for m in METHODS:
        c = characterize(m, n=20001)
        kw = dict(color=METHOD_COLOR[m], linestyle=METHOD_DASH[m], linewidth=LW)
        axes[0].plot(tau, sdk_s(m, tau), label=m.value, **kw)
        axes[1].plot(tau, analytic_ds(m, tau), label=f"{m.value} (peak {c.v_peak:.3g})", **kw)
        # Break the line at the tau = 0.5 seam so acceleration steps are not drawn as ramps.
        dds = analytic_dds(m, tau)
        seam = np.argmax(tau >= 0.5)
        axes[2].plot(tau[:seam], dds[:seam], **kw)
        axes[2].plot(tau[seam:], dds[seam:], label=m.value, **kw)
    axes[0].set_ylabel("s(τ)  position fraction")
    axes[1].set_ylabel("ds/dτ  (velocity × D/A)")
    axes[2].set_ylabel("d²s/dτ²  (accel × D²/A)")
    axes[2].set_xlabel("τ = t / D")
    for ax in axes:
        ax.axhline(0, color=AXIS, linewidth=0.8, zorder=0)
    axes[0].axhline(1, color=AXIS, linewidth=0.8, zorder=0)
    cart = characterize(InterpolationTechnique.CARTOON, n=20001)
    axes[0].annotate(
        f"cartoon dips {cart.anticipation:.1%} below start (τ={cart.tau_at_s_min:.3f})",
        xy=(cart.tau_at_s_min, cart.s_min), xytext=(0.30, -0.25), color=INK_2, fontsize=8,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8),
    )
    axes[0].annotate(
        f"and overshoots {cart.overshoot:.1%} past goal (τ={cart.tau_at_s_max:.3f})",
        xy=(cart.tau_at_s_max, cart.s_max), xytext=(0.36, 1.18), color=INK_2, fontsize=8,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8),
    )
    axes[0].set_ylim(-0.32, 1.30)
    axes[0].legend(loc="lower right", ncols=2)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Position fraction s(τ) - SDK time_trajectory()")
    axes[1].set_title("Normalised velocity: peak = k, so q̇_peak = k·A/D")
    axes[2].set_title("Normalised acceleration: steps at τ = 0, 0.5, 1 are infinite jerk")
    # linear's acceleration is 0 inside (0, 1), but its velocity jumps 0 -> 1 at tau = 0 and 1 -> 0 at tau = 1:
    # acceleration impulses the plot cannot draw (the Neutral motions are linear; added 2026-11-13).
    axes[2].text(0.02, -27.0, "linear: velocity steps 0→1 at τ = 0 and 1→0 at τ = 1\n"
                 "= acceleration impulses (infinite, not drawn)", color=INK_2, fontsize=8, va="center")
    return _finish(fig, "Four goto_target interpolation methods (normalised)", path)


def fig_physical_example(
    path: Path, amplitude_deg: float = 20.0, duration_s: float = 1.0
) -> Path:
    """Head pitch goto of `amplitude_deg` in `duration_s`: position, velocity, peak-vs-duration."""
    _style()
    t = np.linspace(0.0, duration_s, 2001)
    tau = t / duration_s
    durations = np.geomspace(0.2, 3.0, 60)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), constrained_layout=True)
    for m in METHODS:
        c = characterize(m, n=20001)
        kw = dict(color=METHOD_COLOR[m], linestyle=METHOD_DASH[m], linewidth=LW)
        axes[0].plot(t, amplitude_deg * sdk_s(m, tau), label=m.value, **kw)
        axes[1].plot(t, amplitude_deg / duration_s * analytic_ds(m, tau), label=m.value, **kw)
        axes[2].plot(durations, c.v_peak * amplitude_deg / durations, label=m.value, **kw)
    axes[0].set_ylabel("pitch command (deg)")
    axes[1].set_ylabel("pitch rate (deg/s)")
    axes[0].set_xlabel("time (s)")
    axes[1].set_xlabel("time (s)")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    _plain_log_ticks(axes[2], "x", [0.2, 0.3, 0.5, 1, 2, 3])
    _plain_log_ticks(axes[2], "y", [5, 10, 20, 50, 100, 200, 500])
    axes[2].set_xlabel("duration D (s)")
    axes[2].set_ylabel("peak pitch rate (deg/s)")
    axes[0].set_title(f"Commanded pitch 0 → {amplitude_deg:g}° in D = {duration_s:g} s")
    axes[1].set_title("Commanded pitch rate (task space, before IK)")
    axes[2].set_title(f"Peak task-space rate vs duration for A = {amplitude_deg:g}°  (k·A/D)")
    for ax in axes:
        ax.legend(loc="best", ncols=2, fontsize=8)
    return _finish(fig, "Interpolation in physical units", path)


def fig_joint_space(replays: dict[InterpolationTechnique, GotoReplay], path: Path) -> Path:
    """Commanded Stewart motor angles and speeds (IK of the reference) per method."""
    _style()
    fig, axes = plt.subplots(
        2, len(replays), figsize=(11.5, 6.0), sharex=True, sharey="row", constrained_layout=True
    )
    for col, (m, r) in enumerate(replays.items()):
        q = np.rad2deg(r.joints_dense)
        qd = np.rad2deg(r.joint_velocity_dense())
        for j in range(1, 7):
            axes[0, col].plot(r.t_dense, q[:, j], color=MOTOR_COLOR[j - 1], linewidth=1.3,
                              label=JOINT_NAMES[j])
            axes[1, col].plot(r.t_dense, qd[:, j], color=MOTOR_COLOR[j - 1], linewidth=1.3)
        axes[0, col].set_title(m.value)
        axes[1, col].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("motor goal (deg)")
    axes[1, 0].set_ylabel("motor speed required (deg/s)")
    # The top-row panels are empty around 0 deg (odd motors > 0, even motors < 0): put the legend there.
    axes[0, 0].legend(loc="center left", ncols=2, fontsize=7.5)
    case = next(iter(replays.values())).case
    return _finish(
        fig,
        f"IK image of a pitch 0 → {case.target.pitch_deg:g}° goto (D = {case.duration:g} s): "
        "Stewart motors",
        path,
    )


def fig_endpoint_shortfall(path: Path, sample_interval: float = 0.01) -> Path:
    """|1 - s(1 - g/D)| vs duration for a last gap g = sample_interval (real gaps are >= the period)."""
    _style()
    durations = np.geomspace(0.2, 3.0, 80)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for m in METHODS:
        vals = np.array([abs(endpoint_shortfall(m, d, sample_interval)) for d in durations])
        ax.plot(durations, 100 * vals, color=METHOD_COLOR[m], linestyle=METHOD_DASH[m],
                linewidth=LW, label=m.value)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _plain_log_ticks(ax, "x", [0.2, 0.3, 0.5, 1, 2, 3])
    _plain_log_ticks(ax, "y", [1e-4, 1e-3, 1e-2, 1e-1, 1, 5])
    ax.set_xlabel("duration D (s)")
    ax.set_ylabel("|1 − s(τ_last)|  (% of amplitude)")
    ax.set_title(
        f"play_move stops at t < D: never-commanded fraction if the last gap = {sample_interval*1000:g} ms"
    )
    ax.legend(loc="lower left", ncols=2, fontsize=8)
    ax.text(0.99, 0.97, "linear: g/D short of the goal (real gaps ≥ 10 ms)\ncartoon: residual overshoot",
            transform=ax.transAxes, ha="right", va="top", color=INK_2, fontsize=8)
    return _finish(fig, "Final commanded target of goto_target", path)


def fig_pipeline_timing(path: Path, case: GotoCase | None = None, window=(0.40, 0.56)) -> Path:
    """Reference vs 100 Hz evaluated target vs 50 Hz written goal, for one Stewart motor."""
    _style()
    case = case or GotoCase(
        target=HeadTarget(pitch_deg=20.0), duration=1.0, method=InterpolationTechnique.LINEAR
    )
    r = replay_goto(case)
    j = int(np.argmax(np.ptp(r.joints_dense[:, 1:], axis=0))) + 1
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    m_ref = (r.t_dense >= window[0]) & (r.t_dense <= window[1])
    ax.plot(r.t_dense[m_ref], np.rad2deg(r.joints_dense[m_ref, j]), color=INK_2, linewidth=LW,
            label="continuous reference (IK of GotoMove)")
    t_eval_ext = np.append(r.t_eval, r.t_eval[-1] + 0.01)
    ax.stairs(np.rad2deg(r.joints_eval[:, j]), t_eval_ext, color=METHOD_COLOR[InterpolationTechnique.LINEAR],
              linewidth=LW, label="target after play_move (100 Hz)")
    t_ctrl_ext = np.append(r.t_ctrl, r.t_ctrl[-1] + (r.t_ctrl[1] - r.t_ctrl[0]))
    ax.stairs(np.rad2deg(r.joints_written[:, j]), t_ctrl_ext, color=METHOD_COLOR[InterpolationTechnique.MIN_JERK],
              linewidth=LW, linestyle=(0, (6, 2)), label="goal written to motor (50 Hz, IK one tick earlier)")
    for tc in r.t_ctrl[(r.t_ctrl >= window[0]) & (r.t_ctrl <= window[1])]:
        ax.axvline(tc, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlim(*window)
    t_probe = np.linspace(window[0], window[1], 400)
    shown = np.concatenate([
        np.rad2deg(r.joints_dense[m_ref, j]),
        np.rad2deg(r.written_zoh(t_probe)[:, j]),
    ])
    pad = 0.08 * np.ptp(shown)
    ax.set_ylim(shown.min() - pad, shown.max() + pad)
    ax.set_xlabel("time since goto start (s)   (vertical lines = control ticks)")
    ax.set_ylabel(f"{JOINT_NAMES[j]} (deg)")
    ax.set_title(f"Command pipeline, {case.method.value} pitch 0 → {case.target.pitch_deg:g}° in {case.duration:g} s")
    rising = r.joints_dense[-1, j] > r.joints_dense[0, j]
    ax.legend(loc="upper left" if rising else "lower left", fontsize=8)
    return _finish(fig, "goto_target command path: evaluate (100 Hz) → IK → write (50 Hz)", path)
