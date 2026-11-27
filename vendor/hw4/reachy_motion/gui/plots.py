"""Qt-free matplotlib figures for the Motion Editor and its PNG export (data_kind = computed).

Every figure shows planned values computed with the SDK's own interpolation and IK functions (via the planner and the
safety checker) - not measurements. Figures are built on ``matplotlib.figure.Figure`` directly (no pyplot), so the same
code draws into a Qt canvas and into a PNG file.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from reachy_motion.analysis.limits import ANTENNA_MOTOR, BODY_MOTOR_PROXY, STEWART_MOTOR
from reachy_motion.gui.ko import use_korean_matplotlib
from reachy_motion.safety.policy import DEFAULT_POLICY, SafetyPolicy

use_korean_matplotlib()  # Korean figure text (Malgun Gothic on Windows); DejaVu Sans as the glyph fallback

DEG = 180.0 / math.pi
# Same reference palette as the Phase 1 figures (reachy_motion/analysis/figures.py); line style is a second encoding.
C3 = ("#2a78d6", "#eb6834", "#1baf7a")
C6 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
EXPRESSIVE = dict(color="#eb6834", linestyle="-", label="expressive")
NEUTRAL = dict(color="#2a78d6", linestyle=(0, (5, 2)), label="neutral")
LW = 1.4
NOTE = "SDK 보간·IK로 계산한 계획값 (computed, 측정 아님)"
#: roll + = the head's left side up (Rx of the SDK's create_head_pose turns +y, the robot's left, toward +z): the head
#: tilts to the robot's right. Computed from the SDK frame; the roll sign is not checked on the robot (library.pose).
ROLL_NOTE = "* roll 부호는 SDK 좌표계로 계산 (실제 로봇 미확인)"


def new_figure(width: float = 10.0, height: float = 7.0) -> Figure:
    fig = Figure(figsize=(width, height), dpi=100, layout="constrained")
    fig.set_facecolor("#fcfcfb")
    return fig


def _style(ax: Any, ylabel: str, xlabel: bool = False) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=8.5)
    ax.tick_params(labelsize=8, colors=INK_2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if xlabel:
        ax.set_xlabel("시간 (s)", color=INK_2, fontsize=8.5)


def _keys(ax: Any, spec: Any, track: str) -> None:
    for _d, arrive, release in spec.key_times(track):
        ax.axvline(arrive, color=MUTED, linewidth=0.5, linestyle=":")
        if release > arrive:
            ax.axvspan(arrive, release, color=GRID, alpha=0.35, linewidth=0)


def _cursor(ax: Any, t: float | None) -> None:
    if t is not None:
        ax.axvline(t, color=INK, linewidth=1.0)


def draw_motion(fig: Figure, bundle: Any, cursor_s: float | None = None) -> None:
    """Position (left) and velocity (right) of head rotation, head translation, antennas and body."""
    fig.clear()
    a, spec = bundle.analysis, bundle.spec
    if a is None:
        _message(fig, bundle)
        return
    t = np.asarray(a.t)
    pose = np.asarray(a.pose6)
    vel = np.asarray(a.pose_velocity)
    ant = np.asarray(a.antennas)
    ant_v = np.asarray(a.antenna_velocity)
    body = np.asarray(a.body)
    body_v = np.gradient(body, t) if t.size > 1 else np.zeros_like(body)
    axes = fig.subplots(3, 2, sharex=True)
    rows = [
        ("머리 회전 (deg)", "회전 속도 (deg/s)", [(pose[:, 3] * DEG, vel[:, 3] * DEG, "roll (+ 오른쪽 기울임*)"),
                                              (pose[:, 4] * DEG, vel[:, 4] * DEG, "pitch (+ 아래)"),
                                              (pose[:, 5] * DEG, vel[:, 5] * DEG, "yaw (+ 왼쪽)")], "head"),
        ("머리 이동 (mm)", "이동 속도 (mm/s)", [(pose[:, 0] * 1e3, vel[:, 0] * 1e3, "x 앞"),
                                             (pose[:, 1] * 1e3, vel[:, 1] * 1e3, "y 왼쪽"),
                                             (pose[:, 2] * 1e3, vel[:, 2] * 1e3, "z 위")], "head"),
        ("안테나 / 몸통 (deg)", "속도 (deg/s)", [(ant[:, 0] * DEG, ant_v[:, 0] * DEG, "오른쪽 안테나"),
                                             (ant[:, 1] * DEG, ant_v[:, 1] * DEG, "왼쪽 안테나"),
                                             (body * DEG, body_v * DEG, "몸통 yaw")], "antennas"),
    ]
    for r, (ylab, vlab, series, track) in enumerate(rows):
        for (pos, v, label), color in zip(series, C3):
            axes[r, 0].plot(t, pos, color=color, linewidth=LW, label=label)
            axes[r, 1].plot(t, v, color=color, linewidth=LW, label=label)
        for c in (0, 1):
            _keys(axes[r, c], spec, track)
            _cursor(axes[r, c], cursor_s)
        _style(axes[r, 0], ylab, r == 2)
        _style(axes[r, 1], vlab, r == 2)
        axes[r, 0].legend(fontsize=7, loc="best", frameon=False)
    verdict = _verdict(bundle)
    fig.suptitle(f"{spec.key}  {'초안  ' if spec.draft else ''}길이 {spec.duration:.2f} s  |  {verdict}\n"
                 f"{NOTE}  ·  점선 = 키 도착, 음영 = 정지(pause)  ·  {ROLL_NOTE}", fontsize=9.5, color=INK, x=0.01,
                 ha="left")


def draw_joints(fig: Figure, bundle: Any, policy: SafetyPolicy = DEFAULT_POLICY, cursor_s: float | None = None) -> None:
    """IK joint angles, margin to the URDF limits and required speed / no-load speed, with the policy lines."""
    fig.clear()
    a, spec = bundle.analysis, bundle.spec
    if a is None:
        _message(fig, bundle)
        return
    t = np.asarray(a.t)
    q = np.asarray(a.joints) * DEG
    qd = np.abs(np.asarray(a.joint_velocity)) * DEG
    margin = np.asarray(a.stewart_margin)
    ant_v = np.abs(np.asarray(a.antenna_velocity)) * DEG
    axes = fig.subplots(3, 1, sharex=True)
    for j in range(6):
        axes[0].plot(t, q[:, 1 + j], color=C6[j], linewidth=LW, label=f"stewart_{j + 1}")
        axes[1].plot(t, margin[:, j], color=C6[j], linewidth=LW, label=f"stewart_{j + 1}")
    axes[0].plot(t, q[:, 0], color=INK_2, linewidth=LW, linestyle=(0, (4, 2)), label="몸통 (IK)")
    axes[0].legend(fontsize=7, ncols=4, loc="best", frameon=False)
    axes[1].axhline(policy.min_stewart_margin_deg, color="#c0392b", linewidth=1.0, linestyle="--",
                    label=f"초기 정책 {policy.min_stewart_margin_deg:g} deg")
    axes[1].legend(fontsize=7, ncols=4, loc="best", frameon=False)
    ratios = [
        (np.max(qd[:, 1:7], axis=1) / STEWART_MOTOR.no_load_deg_s_5v, "Stewart (최대) / 618 deg/s", C6[0]),
        (qd[:, 0] / BODY_MOTOR_PROXY.no_load_deg_s_5v, "몸통 / 486 deg/s (대용)", C6[2]),
        (np.max(ant_v, axis=1) / ANTENNA_MOTOR.no_load_deg_s_5v, "안테나 / 2298 deg/s", C6[3]),
    ]
    for r, label, color in ratios:
        axes[2].plot(t, r, color=color, linewidth=LW, label=label)
    axes[2].axhline(policy.max_stewart_speed_ratio, color="#c0392b", linewidth=1.0, linestyle="--",
                    label=f"초기 정책 {policy.max_stewart_speed_ratio:g}")
    axes[2].legend(fontsize=7, ncols=2, loc="best", frameon=False)
    _style(axes[0], "IK 관절각 (deg)")
    _style(axes[1], "URDF 한계까지 여유 (deg)")
    _style(axes[2], "속도 / 무부하 속도 @5 V", True)
    for ax in axes:
        _keys(ax, spec, "head")
        _cursor(ax, cursor_s)
    fig.suptitle(f"{spec.key}  관절 (IK, 자동 body yaw)  |  {_verdict(bundle)}\n{NOTE}  ·  한계: 추정한 원시 각도 "
                 "대응의 URDF; 무부하 속도는 상한; 초기 정책 통과는 기구적 안전 보장이 아님", fontsize=9.5, color=INK,
                 x=0.01, ha="left")


def draw_compare(fig: Figure, expressive: Any, neutral: Any) -> None:
    """Expressive vs neutral of the same motion on one time axis (computed plans)."""
    fig.clear()
    if expressive.analysis is None or neutral.analysis is None:
        _message(fig, expressive if expressive.analysis is None else neutral)
        return
    axes = fig.subplots(2, 2, sharex=True)
    panels = [(0, 0, 4, "pitch (deg, + 아래)"), (0, 1, 5, "yaw (deg, + 왼쪽)"), (1, 0, 3, "roll (deg, + 오른쪽 기울임*)")]
    for b, style in ((expressive, EXPRESSIVE), (neutral, NEUTRAL)):
        a = b.analysis
        t = np.asarray(a.t)
        pose = np.asarray(a.pose6)
        for r, c, idx, _ in panels:
            axes[r, c].plot(t, pose[:, idx] * DEG, linewidth=LW, **style)
        ant = np.asarray(a.antennas) * DEG
        axes[1, 1].plot(t, ant[:, 0], linewidth=LW, **style)
        axes[1, 1].plot(t, ant[:, 1], linewidth=LW * 0.8, color=style["color"], linestyle=(0, (1, 1.5)))
    for r, c, _, label in panels:
        _style(axes[r, c], label, r == 1)
    _style(axes[1, 1], "안테나 (deg)", True)
    axes[0, 0].legend(fontsize=7, frameon=False)
    from matplotlib.lines import Line2D

    axes[1, 1].legend(handles=[Line2D([], [], color=INK_2, linewidth=LW, label="오른쪽 (E 실선 · N 파선)"),
                               Line2D([], [], color=INK_2, linewidth=LW * 0.8, linestyle=(0, (1, 1.5)), label="왼쪽 (점선)")],
                      fontsize=7, frameon=False)
    same_start = _same_state(expressive.spec.start, neutral.spec.start)
    same_end = _same_state(expressive.spec.end_state, neutral.spec.end_state)
    fig.suptitle(f"{expressive.spec.name}: expressive {expressive.spec.duration:.2f} s vs neutral "
                 f"{neutral.spec.duration:.2f} s  |  같은 시작: {'예' if same_start else '아니오'}, 같은 끝: "
                 f"{'예' if same_end else '아니오'}\n{NOTE}  ·  {ROLL_NOTE}", fontsize=9.5, color=INK, x=0.01, ha="left")


def _same_state(a: Any, b: Any, tol: float = 1e-9) -> bool:
    return (bool(np.allclose(a.head.as_array(), b.head.as_array(), atol=tol))
            and bool(np.allclose(a.antennas, b.antennas, atol=tol)) and abs(a.body_yaw - b.body_yaw) <= tol)


def _verdict(bundle: Any) -> str:
    if bundle.problems:
        return "spec 오류"
    if bundle.error:
        return "분석 오류"
    if bundle.report is None:
        return "검사 안 함"
    n = len(bundle.report.violations)
    return "초기 정책 안" if bundle.report.passes_policy else f"초기 정책 위반 ({n})"


def _message(fig: Figure, bundle: Any) -> None:
    text = "; ".join(bundle.problems) if bundle.problems else (bundle.error or "no analysis")
    fig.text(0.02, 0.5, f"{bundle.spec.key}: 분석할 수 없음\n{text}", fontsize=10, color="#c0392b", wrap=True)
