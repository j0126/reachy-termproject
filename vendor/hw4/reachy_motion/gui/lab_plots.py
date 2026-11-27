"""Qt-free matplotlib figures of the Motion Lab: commanded vs measured, the R1 amplitude ratios, the R3 ladder, the
limit map, expressive vs neutral, and the live view. Every title names the data_kind of what it shows (computed /
mock / simulated / measured); colour = run (fixed order, at most six runs at once), line style = commanded (dashed)
vs measured (solid).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from reachy_motion.gui import ko
from reachy_motion.gui.plots import C6, GRID, INK, INK_2, LW, MUTED, _style, new_figure
from reachy_motion.gui.results import R1_GROUPS, ROLE_AXIS, ROLE_HEAD, ROLE_JOINT

MAX_RUNS = 6
CMD_STYLE = dict(linestyle=(0, (4, 2)), linewidth=1.1)
MEAS_STYLE = dict(linestyle="-", linewidth=LW)
KIND_MARKER = {"measured": "o", "simulated": "s", "mock": "^"}
PRESENT_NAME = {"measured": "실측 (measured)", "simulated": "시뮬레이션 (simulated, MuJoCo)", "mock": "Mock 모델 (mock)"}
VARIANT_COLOR = {"expressive": "#eb6834", "neutral": "#2a78d6"}


def _empty(fig: Figure, text: str) -> None:
    fig.clear()
    fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=10, color=MUTED, wrap=True)


def _kinds(runs: list[Any]) -> str:
    return ", ".join(sorted({r.data_kind for r in runs}))


def draw_tracking(fig: Figure, runs: list[Any], channels: tuple[str, ...], title: str) -> None:
    """Commanded (dashed) vs measured (solid) per channel; one colour per run."""
    if not runs:
        _empty(fig, "선택한 시행이 없습니다.\n왼쪽에서 세션을 불러와 시행을 체크하거나, 실험 탭에서 시행하세요.")
        return
    fig.clear()
    shown = runs[:MAX_RUNS]
    chans = [c for c in channels if any(c in r.data.channels for r in shown)]
    if not chans:
        _empty(fig, "선택한 시행에 이 채널이 없습니다.")
        return
    ncols = 2
    nrows = math.ceil(len(chans) / ncols)
    axes = fig.subplots(nrows, ncols, sharex=True, squeeze=False)
    for idx, ch in enumerate(chans):
        ax = axes[idx // ncols][idx % ncols]
        unit = ""
        for i, r in enumerate(shown):
            if ch not in r.data.channels:
                continue
            tc, c, tm, m = r.series(ch)
            unit = r.unit(ch)[1]
            ax.plot(tc, c, color=C6[i], **CMD_STYLE)
            ax.plot(tm, m, color=C6[i], **MEAS_STYLE)
        _style(ax, f"{ch} ({unit})", xlabel=idx // ncols == nrows - 1)
    for j in range(len(chans), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    handles = [Line2D([], [], color=C6[i], linewidth=LW, label=f"{r.label} [{r.data_kind}]") for i, r in
               enumerate(shown)]
    handles += [Line2D([], [], color=INK_2, **CMD_STYLE, label="명령값 (보낸 값)"),
                Line2D([], [], color=INK_2, **MEAS_STYLE, label="측정값")]
    fig.legend(handles=handles, fontsize=7, frameon=False, loc="outside lower center",
               ncols=min(4, len(handles)))
    more = f"  (시행 {len(runs)}개 중 앞 {MAX_RUNS}개)" if len(runs) > MAX_RUNS else ""
    present = " / ".join(PRESENT_NAME.get(r, r) for r in sorted({r.data_kind for r in shown}))
    fig.suptitle(f"{title}  |  data_kind: {_kinds(shown)}  |  실선 = {present}{more}\n"
                 "시간 = 실행기 시작부터 s (Motion Lab 시행). 지연은 PC에서 본 전체 지연",
                 fontsize=9.5, color=INK, x=0.01, ha="left")


MAX_PANELS = 4


def draw_main_channel(fig: Figure, runs: list[Any]) -> None:
    """Each run's main channel (its trial axis) only: commanded (dashed) vs measured (solid). Runs that share a
    main channel share a panel (same unit); colour = run, in the order of the list (the same as the key table)."""
    if not runs:
        _empty(fig, "선택한 시행이 없습니다.\n왼쪽에서 세션을 불러와 시행을 체크하거나, 실험 탭에서 시행하세요.")
        return
    fig.clear()
    shown = [r for r in runs[:MAX_RUNS] if r.main_channel in r.data.channels]
    if not shown:
        _empty(fig, "선택한 시행에 주 채널 기록이 없습니다.")
        return
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(shown):
        groups.setdefault(r.main_channel, []).append(i)
    chans = list(groups)[:MAX_PANELS]
    axes = fig.subplots(len(chans), 1, sharex=True, squeeze=False)
    for k, ch in enumerate(chans):
        ax = axes[k][0]
        unit = ""
        for i in groups[ch]:
            tc, c, tm, m = shown[i].series(ch)
            unit = shown[i].unit(ch)[1]
            ax.plot(tc, c, color=C6[i], **CMD_STYLE)
            ax.plot(tm, m, color=C6[i], **MEAS_STYLE)
        _style(ax, f"{ch} ({unit})", xlabel=k == len(chans) - 1)
    handles = [Line2D([], [], color=C6[i], linewidth=LW, label=f"{r.label} [{r.data_kind}]") for i, r in
               enumerate(shown) if r.main_channel in chans]
    handles += [Line2D([], [], color=INK_2, **CMD_STYLE, label="명령값 (보낸 값)"),
                Line2D([], [], color=INK_2, **MEAS_STYLE, label="측정값")]
    fig.legend(handles=handles, fontsize=7, frameon=False, loc="outside lower center", ncols=min(4, len(handles)))
    dropped = len(runs) - len([r for r in shown if r.main_channel in chans])
    more = f"  (시행 {len(runs)}개 중 {dropped}개는 그리지 않음: 한 번에 {MAX_RUNS}개, 채널 {MAX_PANELS}개까지)" if dropped else ""
    present = " / ".join(PRESENT_NAME.get(k, k) for k in sorted({r.data_kind for r in shown}))
    legacy = any(getattr(r, "legacy_time_base", False) for r in shown)
    fig.suptitle(f"주 채널 (시행 축): 명령값 vs 측정값  |  data_kind: {_kinds(shown)}  |  실선 = {present}{more}\n"
                 + ("시간 = 실행기 시작부터 s." if not legacy else
                    "시간 = 실행기 시작부터 s (sim-verify 시행은 두 CSV의 시간 기준 시작점이 달라 어긋날 수 있음).")
                 + " 지연은 PC에서 본 전체 지연", fontsize=9.5, color=INK, x=0.01, ha="left")


#: R3 panels (one name per quantity, as in results.METRIC_LABELS). Shaking is shown, not a stop criterion.
R3_METRICS = (("lag_ms", "지연 Lag (ms)"), ("shape_error_pct", "지연 보정 RMSE (진폭 %)"),
              ("amplitude_ratio", "진폭비"), ("overshoot_pct", "Overshoot (진폭 %)"),
              ("hold_ripple", "정지 중 떨림 RMS"), ("peak_error", "Peak Error (지연 포함)"),
              ("settling_time_s", "정착 시간 (s)"))
LIMIT_COLOR = "#c0392b"


def _ladder(vary: str) -> tuple[str, str, float, str]:
    """(x key, x label, the ladder's first step, its unit text) of the duration or the amplitude ladder."""
    from reachy_motion.robot_r1 import R3_BASE_A_DEG, R3_BASE_S

    if vary == "amplitude":
        return "amplitude_deg", "A 진폭 (deg)", R3_BASE_A_DEG, "°"
    return "duration_s", "D 편도 이동 시간 (s)", R3_BASE_S, " s"


def r3_breakdown(rows: list[dict[str, Any]], vary: str = "duration") -> dict[str, float]:
    """data_kind -> the first step along the ladder (D long -> short; A small -> large on the amplitude ladder) whose
    trial-axis numbers end the robot ladder (``robot_r1.R3_LIMITS``; the same test the robot makes before its next
    step, the lag measured against the ladder's first step). Kinds without one are absent."""
    from reachy_motion.robot_r1 import R3_LIMITS

    xkey, _, first, _ = _ladder(vary)
    out: dict[str, float] = {}
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("role") == ROLE_AXIS and r.get("vary", "duration") == vary and r.get(xkey) is not None:
            by_kind.setdefault(r["data_kind"], []).append(r)
    for kind, pts in by_kind.items():
        pts = sorted(pts, key=lambda p: float(p[xkey]) * (-1.0 if vary == "duration" else 1.0))
        base = next((p for p in pts if abs(float(p[xkey]) - first) < 1e-9), None)
        base_lag = base["lag_ms"] if base is not None and base.get("lag_fit_r2", 0.0) >= 0.9 else None
        for p in pts:
            lag_bad = p.get("lag_fit_r2", 0.0) >= 0.9 and (
                p["lag_ms"] > base_lag + R3_LIMITS["lag_increase_ms_max"] if base_lag is not None
                else p["lag_ms"] > R3_LIMITS["lag_ms_max_without_base"])
            if (p["amplitude_ratio"] < R3_LIMITS["amplitude_ratio_min"]
                    or p["shape_error_pct"] > 100.0 * R3_LIMITS["shape_error_ratio_max"]
                    or p["overshoot_pct"] > 100.0 * R3_LIMITS["overshoot_ratio_max"] or lag_bad):
                out[kind] = float(p[xkey])
                break
    return out


def draw_r3(fig: Figure, rows: list[dict[str, Any]]) -> None:
    """Metrics along ONE R3 ladder (the first in ``rows``: its varied parameter and fixed value; other ladders are
    named, not drawn): x = the varied parameter; one series per data_kind and channel role."""
    from reachy_motion.robot_r1 import R3_LIMITS, r3_base_name

    if not rows:
        _empty(fig, "선택한 시행 중 R3 시행이 없습니다.\n실험 탭에서 R3 사다리(MuJoCo 또는 실제 로봇)를 실행하세요.")
        return
    fig.clear()
    vary = rows[0]["vary"]
    fkey = "amplitude_deg" if vary == "duration" else "duration_s"
    fixed = float(rows[0][fkey])
    shown = [r for r in rows if r["vary"] == vary and r.get(fkey) is not None and abs(float(r[fkey]) - fixed) < 1e-9]
    hidden = len({r["run_dir"] if "run_dir" in r else r["run"] for r in rows}) - len(
        {r["run_dir"] if "run_dir" in r else r["run"] for r in shown})
    xkey, xlabel, first, xunit = _ladder(vary)
    axes = fig.subplots(2, 4, squeeze=False)
    limits = {"amplitude_ratio": R3_LIMITS["amplitude_ratio_min"],
              "shape_error_pct": 100.0 * R3_LIMITS["shape_error_ratio_max"],
              "overshoot_pct": 100.0 * R3_LIMITS["overshoot_ratio_max"]}
    base_lags = {r["data_kind"]: r["lag_ms"] for r in shown if r.get("role") == ROLE_AXIS
                 and abs(float(r[xkey]) - first) < 1e-9 and r.get("lag_fit_r2", 0.0) >= 0.9}
    breakdown = r3_breakdown(shown, vary)
    series: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in shown:
        series.setdefault((r["data_kind"], r["role"]), []).append(r)
    for i, (key, label) in enumerate(R3_METRICS):
        ax = axes[i // 4][i % 4]
        for j, ((kind, role), pts) in enumerate(sorted(series.items())):
            x = np.array([p[xkey] for p in pts], dtype=float)
            y = np.array([p.get(key, math.nan) for p in pts], dtype=float)  # runs analysed before a metric existed
            color = C6[j % len(C6)]
            ax.scatter(x, y, s=26, color=color, marker=KIND_MARKER.get(kind, "o"), zorder=3)  # every run
            xs = sorted(set(x.tolist()))  # the line: the mean of the repetitions at each step
            means = [float(np.nanmean(y[x == v])) if np.isfinite(y[x == v]).any() else math.nan for v in xs]
            ax.plot(xs, means, color=color, marker=KIND_MARKER.get(kind, "o"), markersize=5, linewidth=LW,
                    label=f"{role} [{kind}]", zorder=2)
        unit = shown[0]["unit"] if key in ("dynamic_rmse", "peak_error", "overshoot", "hold_ripple") else ""
        _style(ax, "", xlabel=False)
        ax.set_title(f"{label}{' (' + unit + ')' if unit else ''}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel(xlabel, color=INK_2, fontsize=8.5)
        limit = limits.get(key)
        if limit is not None:  # the robot ladder's stop line (robot_r1.R3_LIMITS)
            ax.axhline(limit, color=LIMIT_COLOR, linestyle=(0, (4, 3)), linewidth=1.0)
        if key == "lag_ms":
            for kind, base_lag in base_lags.items():
                ax.axhline(base_lag + R3_LIMITS["lag_increase_ms_max"], color=LIMIT_COLOR, linestyle=(0, (4, 3)),
                           linewidth=1.0)
        for kind, x_break in breakdown.items():
            ax.axvline(x_break, color=LIMIT_COLOR, linewidth=1.4, alpha=0.6)
        if vary == "duration":
            ax.invert_xaxis()  # shorter D (faster) to the right
    handles, labels = axes[0][0].get_legend_handles_labels()
    spare = axes[1][3]  # 7 panels: the eighth holds the legend (never over a line)
    spare.axis("off")
    spare.legend(handles, labels, fontsize=8, frameon=False, loc="center")
    p_vary, p_fixed, fixed_unit = ("D", "A", "°") if vary == "duration" else ("A", "D", " s")
    broke = ", ".join(f"{kind} {p_vary} {x:g}{xunit}" for kind, x in sorted(breakdown.items()))
    head = (f"R3 사다리: {p_vary} 변경 ({p_fixed} = {fixed:g}{fixed_unit} 고정)  |  "
            f"data_kind: {', '.join(sorted({r['data_kind'] for r in shown}))}  |  "
            + (f"추종이 무너진 단계 (세로 빨간 선): {broke}" if broke else "기준을 넘은 단계 없음")
            + (f"  |  다른 사다리의 시행 {hidden}개는 그리지 않음" if hidden > 0 else ""))
    fig.suptitle(head + "\n빨간 점선 = 실제 로봇 다음 단계 멈춤 기준: 진폭비 0.8, 지연 보정 RMSE·오버슈트 15 % (명령 진폭 대비), "
                 f"지연 {r3_base_name(vary)} + 100 ms (시행 축)\n정지 중 떨림은 표시만 한다 (멈춤 기준 아님). "
                 "simulated / mock 값은 로봇 한계가 아니다.", fontsize=9.5, color=INK, x=0.01, ha="left")


PAIR_METRICS = (("dynamic_rmse", "RMSE (지연 포함)"), ("peak_error", "Peak Error (지연 포함)"),
                ("lag_ms", "지연 Lag (ms)"), ("amplitude_ratio", "진폭비"))


def draw_pairs(fig: Figure, rows: list[dict[str, Any]], role: str = ROLE_JOINT) -> None:
    """Expressive vs neutral bars per motion (mean over repetitions), one data_kind per bar group."""
    rows = [r for r in rows if r["role"] == role]
    if not rows:
        _empty(fig, "선택한 시행 중 같은 모션·같은 data_kind의 expressive / neutral 쌍이 없습니다.\n"
                    "R2를 변형 '둘 다'로 실행하거나 sim-verify 시행을 불러오세요.")
        return
    fig.clear()
    groups = sorted({(r["motion"], r["data_kind"]) for r in rows})
    axes = fig.subplots(2, 2, squeeze=False)
    x = np.arange(len(groups))
    width = 0.38
    for i, (key, label) in enumerate(PAIR_METRICS):
        ax = axes[i // 2][i % 2]
        for j, variant in enumerate(("expressive", "neutral")):
            vals = []
            for g in groups:
                pts = [r[key] for r in rows if (r["motion"], r["data_kind"]) == g and r["variant"] == variant]
                pts = [p for p in pts if isinstance(p, (int, float)) and math.isfinite(p)]
                vals.append(float(np.mean(pts)) if pts else np.nan)
            ax.bar(x + (j - 0.5) * width, vals, width * 0.92, color=VARIANT_COLOR[variant], label=variant)
        ax.set_xticks(x, [f"{m}\n[{k}]" for m, k in groups], fontsize=7)
        _style(ax, label)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.grid(False, axis="x")
    fig.legend(handles=[Patch(color=VARIANT_COLOR[v], label=v) for v in ("expressive", "neutral")], fontsize=8,
               frameon=False, loc="outside lower center", ncols=2)  # below the panels: never over a bar
    fig.suptitle(f"Expressive vs Neutral ({role}, 두 변형에 같은 채널)  |  data_kind: "
                 f"{', '.join(sorted({r['data_kind'] for r in rows}))}\n막대 = 불러온 반복의 평균. 각도·관절은 deg, "
                 "이동은 mm. 추종 오차 = 명령을 따라간 정확도이며 감정 표현력·의도 인식률이 아님", fontsize=9.5, color=INK,
                 x=0.01, ha="left")


R1_AXIS_COLOR = {"yaw": C6[0], "pitch": C6[1], "roll": C6[2], "z": C6[4], "antenna_right": C6[3],
                 "antenna_left": C6[5]}  # one colour per axis across the panels (one shared legend)


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def draw_r1(fig: Figure, rows: list[dict[str, Any]]) -> None:
    """R1 small motions (``results.r1_rows``): commanded amplitude vs amplitude ratio, one panel per unit group
    (head rotation deg / z mm / antennas deg). A marker per run (shape = data_kind), a line through the mean of each
    amplitude (both signs and the repetitions together; measured solid, others dashed)."""
    rows = [r for r in rows if _finite(r.get("amplitude_ratio")) and _finite(r.get("amplitude"))]
    if not rows:
        _empty(fig, "선택한 시행 중 R1 시행이 없습니다.\n실험 탭 R1에서 축과 진폭을 바꿔 가며 실행하세요 (진폭은 직접 입력 가능).")
        return
    fig.clear()
    groups = [(title, names) for title, names in R1_GROUPS if any(r["axis"] in names for r in rows)]
    axs = fig.subplots(1, len(groups), squeeze=False, sharey=True)[0]
    handles: dict[tuple[str, str], Line2D] = {}
    top = 1.15
    for k, (ax, (title, names)) in enumerate(zip(axs, groups)):
        unit = "deg"
        for axis, kind in sorted({(r["axis"], r["data_kind"]) for r in rows if r["axis"] in names},
                                 key=lambda t: (names.index(t[0]), t[1])):
            ps = [r for r in rows if (r["axis"], r["data_kind"]) == (axis, kind)]
            unit = ps[0]["unit"]
            color = R1_AXIS_COLOR.get(axis, INK_2)
            style = "-" if kind == "measured" else (0, (4, 2))
            ax.scatter([p["amplitude"] for p in ps], [p["amplitude_ratio"] for p in ps], s=30, color=color,
                       marker=KIND_MARKER.get(kind, "o"), edgecolors="white", linewidths=0.8, zorder=3)
            amps = sorted({float(p["amplitude"]) for p in ps})
            means = [float(np.mean([p["amplitude_ratio"] for p in ps if float(p["amplitude"]) == a])) for a in amps]
            ax.plot(amps, means, color=color, linewidth=LW, linestyle=style, zorder=2)
            top = max(top, 1.08 * max(p["amplitude_ratio"] for p in ps))
            handles[(axis, kind)] = Line2D([], [], color=color, marker=KIND_MARKER.get(kind, "o"), linewidth=LW,
                                           linestyle=style, label=f"{ko.label(ko.AXES, axis)} [{kind}]")
        ax.axhline(1.0, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        _style(ax, "진폭비 = 움직인 크기 ÷ 명령 크기" if k == 0 else "")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel(f"명령 진폭 ({unit})", color=INK_2, fontsize=8.5)
        ax.set_xlim(left=0.0)
    axs[0].set_ylim(0.0, top)
    axs[0].annotate("1 = 명령만큼 움직임", xy=(0.0, 1.0), xycoords=("axes fraction", "data"), xytext=(4, 4),
                    textcoords="offset points", ha="left", fontsize=7.5, color=MUTED)
    fig.legend(handles=list(handles.values()), fontsize=8, frameon=False, loc="outside lower center",
               ncols=min(4, len(handles)))
    fig.suptitle(f"R1 작은 동작: 명령 진폭 vs 진폭비  |  data_kind: {', '.join(sorted({r['data_kind'] for r in rows}))}\n"
                 "점 = 시행 하나 (+/− 방향·반복 포함), 선 = 같은 진폭의 평균. 1보다 작을수록 명령보다 덜 움직였다.\n"
                 "작은 명령에서 거의 안 움직이면 데드밴드다. simulated / mock은 로봇 측정이 아니다.",
                 fontsize=9.5, color=INK, x=0.01, ha="left")


MOTION_COLOR = {"aha_recommend": C6[0], "boredom": C6[2], "success": C6[1], "failure": C6[3]}
MOTION_SHORT = {"aha_recommend": "aha", "boredom": "boredom", "success": "success", "failure": "failure"}


def limit_overlay(runs: list[Any], r3: list[dict[str, Any]]) -> dict[str, Any]:
    """What the selected runs add to the limit map: motions and R3 steps (pitch 20 deg, duration ladder) run on the
    robot (measured), and the step where each data_kind's duration ladder broke down (``r3_breakdown``)."""
    from reachy_motion.gui.experiments import R3_DEFAULT_AMPLITUDE_DEG

    ladder = [x for x in r3 if x.get("vary", "duration") == "duration"
              and abs(float(x.get("amplitude_deg") or 0.0) - R3_DEFAULT_AMPLITUDE_DEG) < 1e-9]
    return {"robot_motions": {(r.condition.get("motion"), r.condition.get("variant")) for r in runs
                              if r.stage == "R2" and r.data_kind == "measured"},
            "r3_measured": {float(r.condition["duration_s"]) for r in runs
                            if r.stage == "R3" and r.data_kind == "measured" and r.condition.get("vary", "duration")
                            == "duration" and abs(float(r.condition.get("amplitude_deg") or 0.0)
                                                  - R3_DEFAULT_AMPLITUDE_DEG) < 1e-9},
            "breakdown": r3_breakdown(ladder, "duration"),
            "kinds": sorted({r.data_kind for r in runs})}


def _place_labels(fig: Figure, ax: Any, items: list[tuple[float, float, str, str]],
                  obstacles: list[tuple[float, float, float]], fontsize: float = 7.5) -> None:
    """Text next to points without covering markers or each other: per label the first of eight offsets whose box
    (estimated from the text length; no renderer needed) misses every marker (x, y, radius pt) and label placed so
    far. ``items``: (x, y, text, colour) in data units; the axes limits must be final."""
    px = fig.dpi / 72.0
    to_disp = ax.transData.transform
    marks = [(*to_disp((x, y)), r * px) for x, y, r in obstacles]
    boxes: list[tuple[float, float, float, float]] = []
    cands = ((7, 4, "left", "bottom"), (7, -4, "left", "top"), (-7, 4, "right", "bottom"), (-7, -4, "right", "top"),
             (0, 9, "center", "bottom"), (0, -9, "center", "top"), (14, 0, "left", "center"), (-14, 0, "right", "center"))

    def box(cx: float, cy: float, dx: float, dy: float, ha: str, va: str, text: str) -> tuple[float, ...]:
        w, h = 0.58 * fontsize * px * len(text), 1.15 * fontsize * px
        x0 = cx + dx * px - (w if ha == "right" else w / 2 if ha == "center" else 0.0)
        y0 = cy + dy * px - (h if va == "top" else h / 2 if va == "center" else 0.0)
        return x0, y0, x0 + w, y0 + h

    def hits(b: tuple[float, ...]) -> int:
        n = sum(1 for (mx, my, r) in marks if b[0] - r < mx < b[2] + r and b[1] - r < my < b[3] + r)
        return n + sum(1 for o in boxes if b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3])

    for x, y, text, color in items:
        cx, cy = to_disp((x, y))
        best = min(cands, key=lambda c: (hits(box(cx, cy, *c, text)), cands.index(c)))
        boxes.append(box(cx, cy, *best, text))
        ax.annotate(text, (x, y), xytext=(best[0], best[1]), textcoords="offset points", ha=best[2], va=best[3],
                    fontsize=fontsize, color=color, zorder=6)


def draw_limit_map(fig: Figure, points: list[dict[str, Any]], overlay: dict[str, Any] | None = None) -> None:
    """Where the confirmed motions sit against the R3 duration ladder (``experiments.limit_points``, computed):
    x = peak Stewart speed / no-load speed, y = peak joint acceleration, log-log. ``overlay`` (``limit_overlay``):
    rings = run on the robot (measured), an X = the R3 step where tracking broke down."""
    pts = [p for p in points if _finite(p.get("speed_ratio")) and _finite(p.get("accel_dps2"))
           and p["speed_ratio"] > 0 and p["accel_dps2"] > 0]
    if not pts:
        _empty(fig, "계산할 모션이 없습니다 (motions 폴더를 확인하세요).")
        return
    ov = overlay or {}
    fig.clear()
    ax = fig.subplots(1, 1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    r3 = sorted([p for p in pts if p["kind"] == "r3"], key=lambda p: -p["duration_s"])
    motions = [p for p in pts if p["kind"] == "motion"]
    from reachy_motion.safety.policy import DEFAULT_POLICY

    cap = DEFAULT_POLICY.max_stewart_speed_ratio
    xs = [p["speed_ratio"] for p in pts] + [cap]
    ys = [p["accel_dps2"] for p in pts]
    ax.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
    ax.set_ylim(min(ys) / 1.8, max(ys) * 1.8)
    ax.plot([p["speed_ratio"] for p in r3], [p["accel_dps2"] for p in r3], color=INK_2, linewidth=1.0, zorder=1)
    obstacles: list[tuple[float, float, float]] = []
    for p in r3:
        ax.scatter(p["speed_ratio"], p["accel_dps2"], s=34, marker="o", zorder=3, linewidths=1.2, edgecolors=INK_2,
                   facecolors=INK_2 if p["robot_scope"] else "white")
        obstacles.append((p["speed_ratio"], p["accel_dps2"], 5.0))
    for p in motions:
        color = MOTION_COLOR.get(p["motion"], INK_2)
        expressive = p["variant"] == "expressive"
        ax.scatter(p["speed_ratio"], p["accel_dps2"], s=64, marker="s" if expressive else "D", zorder=4,
                   linewidths=1.6, edgecolors=color if not expressive else "white",
                   facecolors=color if expressive else "white")
        obstacles.append((p["speed_ratio"], p["accel_dps2"], 7.0))
    robot_motions = ov.get("robot_motions") or set()
    r3_measured = ov.get("r3_measured") or set()
    for p in motions:
        if (p["motion"], p["variant"]) in robot_motions:
            ax.scatter(p["speed_ratio"], p["accel_dps2"], s=200, facecolors="none", edgecolors=INK, linewidths=1.0,
                       zorder=5)
    for p in r3:
        if any(abs(p["duration_s"] - d) < 1e-9 for d in r3_measured):
            ax.scatter(p["speed_ratio"], p["accel_dps2"], s=150, facecolors="none", edgecolors=INK, linewidths=1.0,
                       zorder=5)
    broke_text = []
    for kind, d in sorted((ov.get("breakdown") or {}).items()):
        p = next((q for q in r3 if abs(q["duration_s"] - d) < 1e-9), None)
        if p is None:
            continue
        color = LIMIT_COLOR if kind == "measured" else MUTED
        ax.scatter(p["speed_ratio"], p["accel_dps2"], s=120, marker="x", color=color, linewidths=2.2, zorder=6)
        ax.axvline(p["speed_ratio"], color=color, linewidth=0.9, linestyle=(0, (1, 2)), zorder=1)
        ax.axhline(p["accel_dps2"], color=color, linewidth=0.9, linestyle=(0, (1, 2)), zorder=1)
        broke_text.append(f"{kind} D {d:g} s" + ("" if kind == "measured" else " (로봇 한계 아님)"))
    ax.axvline(cap, color=LIMIT_COLOR, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    ax.annotate(f"초기 정책 속도 상한 {cap:g}", xy=(cap, 0.0), xycoords=("data", "axes fraction"), xytext=(-4, 4),
                textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=LIMIT_COLOR)
    ticks_x = [v for v in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0) if ax.get_xlim()[0] <= v <= ax.get_xlim()[1]]
    ticks_y = [v for v in (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
               if ax.get_ylim()[0] <= v <= ax.get_ylim()[1]]
    for axis, ticks in ((ax.xaxis, ticks_x), (ax.yaxis, ticks_y)):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        axis.set_minor_locator(NullLocator())
    _style(ax, "최고 관절 가속도 (deg/s², 몸통 + Stewart)")
    ax.set_xlabel("최고 Stewart 관절 속도 ÷ 무부하 속도 (XL330 618 deg/s @5 V)", color=INK_2, fontsize=8.5)
    labels = [(p["speed_ratio"], p["accel_dps2"], f"D {p['duration_s']:g}", INK_2) for p in r3]
    labels += [(p["speed_ratio"], p["accel_dps2"], f"{MOTION_SHORT.get(p['motion'], p['motion'])} "
                f"{'E' if p['variant'] == 'expressive' else 'N'}", MOTION_COLOR.get(p["motion"], INK_2))
               for p in motions]
    _place_labels(fig, ax, labels, obstacles)
    handles = [Line2D([], [], color=MOTION_COLOR[m], marker="s", linestyle="none", markersize=7,
                      label=ko.label(ko.MOTIONS, m).split(" (")[0]) for m in MOTION_COLOR
               if any(p["motion"] == m for p in motions)]
    handles += [Line2D([], [], color=INK_2, marker="s", linestyle="none", markersize=7, label="Expressive"),
                Line2D([], [], color=INK_2, marker="D", markerfacecolor="white", linestyle="none", markersize=6,
                       label="Neutral (linear)"),
                Line2D([], [], color=INK_2, marker="o", linewidth=1.0, label="R3 pitch 20° (채움 = 실제 로봇 범위)"),
                Line2D([], [], color=INK, marker="o", markerfacecolor="none", markersize=11, linestyle="none",
                       label="실제 로봇에서 실행함 (measured)"),
                Line2D([], [], color=LIMIT_COLOR, marker="x", linestyle="none", markersize=8, markeredgewidth=2,
                       label="R3에서 추종이 무너진 단계")]
    fig.legend(handles=handles, fontsize=7.5, frameon=False, loc="outside lower center", ncols=5)
    shown = ", ".join(ov.get("kinds") or []) or "없음"
    fig.suptitle("한계 지도: 모션과 R3 사다리가 요구하는 최고 관절 속도·가속 (computed: SDK 보간·IK 계산값)\n"
                 f"겹쳐 표시한 시행: {shown}" + (f"  |  무너진 단계: {', '.join(broke_text)}" if broke_text else "")
                 + "\n오른쪽·위일수록 빠르고 급하다. R3는 D가 짧을수록 오른쪽 위로 간다. 가속 한계는 데이터시트에 없어 "
                 "R3에서 무너진 단계(점선 십자)로 본다.\nNeutral(linear)의 가속은 시작·끝의 속도 계단을 200 Hz 표본으로 "
                 "잰 값이다 (이론상 무한대). 초기 정책 통과는 기구적 안전 보장이 아니다.",
                 fontsize=9.5, color=INK, x=0.01, ha="left")


def live_figure() -> tuple[Figure, dict[str, Any]]:
    """The live view: the trial's main channel, commanded (dashed) vs present state (solid)."""
    fig = new_figure(7.0, 3.6)
    ax = fig.subplots(1, 1)
    lines = {"cmd": ax.plot([], [], color=C6[0], **CMD_STYLE, label="명령값")[0],
             "meas": ax.plot([], [], color=C6[0], **MEAS_STYLE, label="측정값")[0]}
    _style(ax, "주 채널", xlabel=True)
    ax.set_xlim(0.0, 5.0)  # until the first trial sets the run's own range
    ax.set_ylim(-5.0, 5.0)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    title = fig.suptitle("실시간: 시행을 실행하면 명령값(점선)과 측정값(실선)이 그려집니다", fontsize=9.5,
                         color=INK, x=0.01, ha="left")
    return fig, {"ax": ax, "lines": lines, "title": title}


__all__ = ["draw_tracking", "draw_main_channel", "draw_r3", "draw_r1", "draw_limit_map", "limit_overlay",
           "r3_breakdown", "draw_pairs", "live_figure", "MAX_RUNS", "PRESENT_NAME", "ROLE_HEAD"]
