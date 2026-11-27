"""Results of the Motion Lab (Qt-free): load recorded runs, tracking metrics, R3 and expressive-vs-neutral
comparisons, CSV / JSON exports and a report summary.

A run folder holds ``commands.csv`` + ``present_<data_kind>.csv`` (Motion Lab trials; sim-verify runs; later robot
runs). Its data_kind is the present file's suffix and nothing else - ``mock`` (Mock model), ``simulated``
(MuJoCo), ``measured`` (robot) - and every row, figure title and summary line carries it. Nothing here creates,
edits or extrapolates a measurement: a kind that was not loaded is reported as missing.

Metrics: ``analysis.tracking`` (lag, dynamic RMSE, static bias, peak error, amplitude ratio, overshoot, settling
time) with the ``tracking`` CLI's settling floors (0.2 deg / 0.2 mm, cli_phase4.SETTLE_ABS_TOL). Runs written
before the Motion Lab (sim-verify) start their two CSV time bases at different samples
(``tracking.LEGACY_ALIGNMENT_NOTE``): they are flagged ``legacy_time_base``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

KINDS = ("measured", "simulated", "mock")
METRIC_FIELDS = ("command_p2p", "static_bias", "dynamic_rmse", "dynamic_rmse_lag_compensated", "peak_error",
                 "lag_ms", "lag_fit_r2", "amplitude_ratio", "overshoot", "overshoot_ratio", "settling_time_s",
                 "hold_ripple")
#: One name per quantity everywhere (tables, figures, robot messages; 2026-11-13). RMSE and Peak Error keep the
#: delay (on the robot ~190 ms of it): their labels say so; the lag-compensated RMSE is the shape error.
METRIC_LABELS = {"command_p2p": "명령 진폭", "static_bias": "Static Bias (정적 편향)", "dynamic_rmse": "RMSE (지연 포함)",
                 "dynamic_rmse_lag_compensated": "지연 보정 RMSE", "peak_error": "Peak Error (지연 포함)",
                 "lag_ms": "지연 Lag (ms)", "lag_fit_r2": "Lag r²", "amplitude_ratio": "진폭비", "overshoot": "Overshoot",
                 "overshoot_ratio": "Overshoot 비", "settling_time_s": "정착 시간 (s)",
                 "shape_error_pct": "지연 보정 RMSE (진폭 %)", "overshoot_pct": "Overshoot (진폭 %)",
                 "hold_ripple": "정지 중 떨림 RMS", "axis": "축", "amplitude": "명령 진폭", "sign": "방향"}
ROLE_AXIS, ROLE_JOINT, ROLE_HEAD = "시행 축", "대표 관절", "주 머리 축"
POSE_CHANNELS = ("roll", "pitch", "yaw", "x", "y", "z")
JOINT_CHANNELS = tuple(f"stewart_{i}" for i in range(1, 7))
SETTLE_ABS_TOL = {"rad": math.radians(0.2), "m": 0.2e-3}  # the `tracking` CLI's floors (cli_phase4.SETTLE_ABS_TOL)


@dataclass
class LoadedRun:
    run_dir: Path
    label: str
    data_kind: str
    stage: str  # "R1" / "R2" / "R3" (Motion Lab) or "sim-verify"
    condition: dict[str, Any]
    main_channel: str
    data: Any  # analysis.tracking.RunData
    metrics: dict[str, Any]  # channel -> tracking.ChannelMetrics
    manifest: dict[str, Any] = field(default_factory=dict)
    legacy_time_base: bool = False
    representative: str = ""  # the Stewart joint with the largest commanded excursion

    @property
    def completed(self) -> bool | None:
        r = self.manifest.get("result")
        return None if not r else bool(r.get("completed"))

    @property
    def name(self) -> str:
        return f"{self.label} [{self.data_kind}]"

    def unit(self, channel: str) -> tuple[float, str]:
        u = self.data.units[self.data.channels.index(channel)]
        return (180.0 / math.pi, "deg") if u == "rad" else (1000.0, "mm")

    def series(self, channel: str, display: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(t_cmd, cmd, t_meas, meas) of one channel, in deg / mm when ``display``."""
        i = self.data.channels.index(channel)
        k = self.unit(channel)[0] if display else 1.0
        return self.data.t_cmd, self.data.x_cmd[:, i] * k, self.data.t_meas, self.data.x_meas[:, i] * k


def find_runs(root: Path | str) -> list[Path]:
    """Every folder under ``root`` (itself included) with commands.csv and exactly one present_*.csv."""
    root = Path(root)
    out = []
    for p in [root, *sorted(x for x in root.rglob("*") if x.is_dir())]:
        if (p / "commands.csv").is_file() and len(list(p.glob("present_*.csv"))) == 1:
            out.append(p)
    return out


def _legacy_condition(run_dir: Path) -> dict[str, Any]:
    parts = run_dir.name.split(".")
    if len(parts) == 2 and parts[1] in ("expressive", "neutral"):
        return {"motion": parts[0], "variant": parts[1]}
    return {}


def load_result(run_dir: Path | str) -> LoadedRun:
    """Read a run folder and compute its tracking metrics (computed from the run's own data_kind)."""
    from reachy_motion.analysis.tracking import load_run, tracking_metrics
    from reachy_motion.gui.lab import LAB_TOOL

    run_dir = Path(run_dir)
    data = load_run(run_dir)
    mpath = run_dir / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else {}
    lab = manifest.get("tool") == LAB_TOOL
    trial = manifest.get("trial", {}) if lab else {}
    metrics = tracking_metrics(data.t_cmd, data.x_cmd, data.t_meas, data.x_meas, data.channels,
                               settle_abs_tols=[SETTLE_ABS_TOL[u] for u in data.units])
    joints = [c for c in data.channels if c.startswith("stewart_")]
    rep = max(joints, key=lambda c: metrics[c].command_p2p) if joints else data.channels[0]
    condition = dict(trial.get("condition", {})) if lab else _legacy_condition(run_dir)
    main = trial.get("main_channel") or "pitch"
    if main not in data.channels:
        main = data.channels[0]
    return LoadedRun(run_dir=run_dir, label=trial.get("label") or run_dir.name, data_kind=data.data_kind,
                     stage=trial.get("stage") or "sim-verify", condition=condition, main_channel=main, data=data,
                     metrics=metrics, manifest=manifest, legacy_time_base=not lab, representative=rep)


def _display(run: LoadedRun, channel: str) -> dict[str, Any]:
    m = run.metrics[channel]
    k, unit = run.unit(channel)
    ratios = [s.overshoot_ratio for s in m.steps if math.isfinite(s.overshoot_ratio)]
    return {"command_p2p": m.command_p2p * k, "static_bias": m.static_bias * k, "dynamic_rmse": m.dynamic_rmse * k,
            "dynamic_rmse_lag_compensated": m.dynamic_rmse_lag_compensated * k, "peak_error": m.peak_error * k,
            "peak_error_t_s": m.peak_error_t_s, "lag_ms": m.lag_s * 1000.0, "lag_fit_r2": m.lag_fit_r2,
            "lag_at_bound": m.lag_at_bound, "amplitude_ratio": m.amplitude_ratio, "overshoot": m.overshoot * k,
            "overshoot_ratio": max(ratios) if ratios else math.nan, "settling_time_s": m.settling_time_s,
            "unsettled_steps": m.unsettled_steps, "hold_ripple": m.hold_ripple * k, "unit": unit}


def channels_for(run: LoadedRun, which: str) -> list[str]:
    if which == "representative":
        return list(dict.fromkeys([run.main_channel, run.representative]))
    if which == "pose":
        return [c for c in POSE_CHANNELS if c in run.data.channels]
    if which == "joints":
        return [c for c in JOINT_CHANNELS if c in run.data.channels]
    return list(run.data.channels)


def metric_rows(runs: list[LoadedRun], which: str = "representative") -> list[dict[str, Any]]:
    """One row per run and channel, deg / mm / ms, with the data_kind of the run."""
    rows = []
    for r in runs:
        for c in channels_for(r, which):
            role = ROLE_AXIS if c == r.main_channel else (ROLE_JOINT if c == r.representative else "")
            rows.append({"run": r.label, "data_kind": r.data_kind, "stage": r.stage,
                         "condition": condition_text(r.condition), "channel": c, "role": role, **_display(r, c),
                         "legacy_time_base": r.legacy_time_base, "run_dir": str(r.run_dir)})
    return rows


def condition_text(c: dict[str, Any]) -> str:
    if "duration_s" in c and "amplitude_deg" in c:
        return f"A {c['amplitude_deg']:g}°, D {c['duration_s']:g} s"
    if "axis" in c:
        return f"{c['axis']} {c['amplitude']:+g} {c.get('unit', '')}"
    if "motion" in c:
        return f"{c['motion']} {c.get('variant', '')}"
    return ""


def r3_rows(runs: list[LoadedRun]) -> list[dict[str, Any]]:
    """R3 steps (one parameter varied), sorted along the ladder; trial axis and representative joint."""
    rows = []
    for r in runs:
        if r.stage != "R3":
            continue
        c = r.condition
        for ch, role in ((r.main_channel, ROLE_AXIS), (r.representative, ROLE_JOINT)):
            d = _display(r, ch)
            p2p = abs(d["command_p2p"]) if d["command_p2p"] == d["command_p2p"] else 0.0
            # the robot ladder's own criteria (robot_r1.R3_LIMITS) as % of the commanded peak-to-peak
            d["shape_error_pct"] = 100.0 * d["dynamic_rmse_lag_compensated"] / p2p if p2p > 0 else math.nan
            d["overshoot_pct"] = 100.0 * d["overshoot"] / p2p if p2p > 0 else math.nan
            rows.append({"run": r.label, "data_kind": r.data_kind, "vary": c.get("vary", "duration"),
                         "amplitude_deg": c.get("amplitude_deg"), "duration_s": c.get("duration_s"),
                         "rep": c.get("rep"), "channel": ch, "role": role, **d})
    rows.sort(key=lambda x: (x["data_kind"], x["vary"], x["role"], -(x["duration_s"] or 0) if x["vary"] == "duration"
                             else (x["amplitude_deg"] or 0)))
    return rows


R1_GROUPS = (("머리 회전 (deg)", ("yaw", "pitch", "roll")), ("머리 위아래 z (mm)", ("z",)),
             ("안테나 (deg)", ("antenna_right", "antenna_left")))
R1_UNIT = {"z": "mm"}  # every other R1 axis is in deg


def r1_rows(runs: list[LoadedRun]) -> list[dict[str, Any]]:
    """R1 small motions on their trial axis: the commanded amplitude and how much of it the head or antenna did
    (amplitude ratio; the robot's dead band and steady-state error show here), with lag and the hold's shaking.
    Sorted by data_kind, axis, amplitude."""
    rows = []
    for r in runs:
        c = r.condition
        if r.stage != "R1" or "axis" not in c or r.main_channel not in r.metrics:
            continue
        d = _display(r, r.main_channel)
        a = float(c.get("amplitude") or 0.0)
        rows.append({"run": r.label, "data_kind": r.data_kind, "axis": c["axis"], "amplitude": abs(a),
                     "sign": "+" if a > 0 else "-", "unit": d["unit"], "rep": c.get("rep"), "channel": r.main_channel,
                     "amplitude_ratio": d["amplitude_ratio"], "lag_ms": d["lag_ms"], "lag_fit_r2": d["lag_fit_r2"],
                     "static_bias": d["static_bias"], "hold_ripple": d["hold_ripple"], "run_dir": str(r.run_dir)})
    order = {ax: i for i, (_, axes) in enumerate(R1_GROUPS) for ax in axes}
    rows.sort(key=lambda x: (x["data_kind"], order.get(x["axis"], 9), x["axis"], x["amplitude"], x["sign"],
                             x["rep"] or 0))
    return rows


def pair_rows(runs: list[LoadedRun]) -> list[dict[str, Any]]:
    """Expressive vs neutral of the same motion and data_kind (R2 trials or sim-verify runs), on the SAME channels:
    the expressive run's dominant head axis and its representative joint, read in both variants."""
    groups: dict[tuple[str, str], dict[str, list[LoadedRun]]] = {}
    for r in runs:
        m, v = r.condition.get("motion"), r.condition.get("variant")
        if m and v in ("expressive", "neutral"):
            groups.setdefault((m, r.data_kind), {}).setdefault(v, []).append(r)
    rows = []
    for (motion, kind), by in sorted(groups.items()):
        if not (by.get("expressive") and by.get("neutral")):
            continue
        ref = by["expressive"][0]
        channels = ((ref.main_channel, ROLE_HEAD), (ref.representative, ROLE_JOINT))
        for variant in ("expressive", "neutral"):
            for r in by[variant]:
                for ch, role in channels:
                    if ch in r.data.channels:
                        rows.append({"motion": motion, "data_kind": kind, "variant": variant, "run": r.label,
                                     "channel": ch, "role": role, **_display(r, ch)})
    return rows


def run_brief(run: LoadedRun) -> dict[str, Any]:
    """Short summary for the experiment log: trial axis + representative joint (deg / mm / ms)."""
    out = {"data_kind": run.data_kind, "legacy_time_base": run.legacy_time_base}
    for ch in channels_for(run, "representative"):
        d = _display(run, ch)
        out[ch] = {k: d[k] for k in ("unit", "command_p2p", "static_bias", "dynamic_rmse", "peak_error", "lag_ms",
                                     "lag_fit_r2", "amplitude_ratio", "overshoot", "settling_time_s")}
    return out


# ---------------------------------------------------------------------------------------------- first-user views
NOTES_FILE = "notes.md"
KEY_METRICS = ("dynamic_rmse", "peak_error", "lag_ms", "amplitude_ratio")  # shown first (RMSE, Peak, Lag, ratio)
CAUSAL_DISCLAIMER = "이 문장들은 계산된 값의 크기와 변화만 적은 것이며, 그 원인을 뜻하지 않는다."
#: Expressive vs Neutral: what the tracking metrics are NOT (shown wherever the pair is compared and exported).
EN_METRIC_NOTE = ("Expressive vs Neutral의 추종 오차 지표(RMSE·Peak Error·Lag·진폭비)는 로봇이 명령한 움직임을 얼마나 "
                  "정확히 따라갔는지만 나타낸다. 감정 표현력이나 사람이 의도를 알아보는 정도(의도 인식률)를 뜻하지 않으며, "
                  "그것은 사람 대상 평가로 따로 재야 한다.")
EN_METRIC_NOTE_SHORT = ("이 숫자는 '명령을 얼마나 정확히 따라갔는지'입니다. 감정이 잘 전달되는지(표현력·의도 인식률)는 "
                        "사람 평가로 따로 재야 합니다.")
SIMPLE_DISCLAIMER = "숫자만 정리했으며, 왜 그렇게 됐는지(원인)는 판단하지 않았습니다."
KIND_KO = {"measured": "실제 로봇 측정", "simulated": "MuJoCo 시뮬레이션", "mock": "Mock 모델"}
CHANNEL_KO = {"yaw": "머리 좌우 회전(yaw)", "pitch": "머리 끄덕임(pitch)", "roll": "머리 갸웃(roll)",
              "x": "머리 앞뒤 이동(x)", "y": "머리 좌우 이동(y)", "z": "머리 위아래 이동(z)"}


def channel_ko(ch: str) -> str:
    """Everyday name of a channel for the Basic-mode summary (Stewart joints keep their name)."""
    if ch in CHANNEL_KO:
        return CHANNEL_KO[ch]
    return f"가장 많이 움직인 머리 관절({ch})" if ch.startswith("stewart_") else ch


def session_of(run: LoadedRun) -> Path:
    """The folder a run belongs to: <session>/runs/<run> (Motion Lab, sim-verify) or the run's parent."""
    p = Path(run.run_dir).parent
    return p.parent if p.name == "runs" else p


def read_notes(run_dir: Path | str) -> str:
    """The user's notes of a run: notes.md (editable) or, failing that, the notes saved in the manifest."""
    p = Path(run_dir) / NOTES_FILE
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    m = Path(run_dir) / "manifest.json"
    if m.is_file():
        try:
            return str(json.loads(m.read_text(encoding="utf-8")).get("notes") or "").strip()
        except ValueError:
            return ""
    return ""


def write_notes(run_dir: Path | str, text: str) -> Path:
    p = Path(run_dir) / NOTES_FILE
    p.write_text(text.strip() + "\n", encoding="utf-8")
    return p


def run_order(run: LoadedRun) -> tuple[Any, ...]:
    """Reading order of runs: session, stage, then along the R3 ladder (D long -> short, A small -> large) or
    expressive before neutral, then repetition."""
    c = run.condition
    rep = int(c.get("rep") or 0)
    if run.stage == "R3":
        vary = str(c.get("vary") or "duration")
        v = float(c.get("duration_s") or 0.0) if vary == "duration" else float(c.get("amplitude_deg") or 0.0)
        return (session_of(run).name, run.stage, vary, -v if vary == "duration" else v, rep, run.label)
    return (session_of(run).name, run.stage, str(c.get("motion") or c.get("axis") or ""),
            0 if c.get("variant") == "expressive" else 1, rep, run.label)


def main_metrics_rows(runs: list[LoadedRun]) -> list[dict[str, Any]]:
    """One row per run on its main channel (trial axis / dominant head axis): RMSE, Peak Error, Lag, amplitude
    ratio first - the numbers a first-time reader looks for."""
    rows = []
    for r in runs:
        d = _display(r, r.main_channel)
        rows.append({"run": r.label, "session": session_of(r).name, "stage": r.stage,
                     "condition": condition_text(r.condition), "data_kind": r.data_kind, "channel": r.main_channel,
                     "unit": d["unit"], **{k: d[k] for k in KEY_METRICS}, "lag_fit_r2": d["lag_fit_r2"],
                     "notes": read_notes(r.run_dir), "run_dir": str(r.run_dir)})
    return rows


def mixing_warnings(runs: list[LoadedRun]) -> list[str]:
    """What is mixed in the current selection (so conditions and data sources are not compared by mistake)."""
    out = []
    kinds = sorted({r.data_kind for r in runs})
    if len(kinds) > 1:
        out.append(f"서로 다른 데이터 출처(data_kind: {', '.join(kinds)})가 함께 선택됐습니다. 비교는 같은 출처끼리 하세요.")
    stages = sorted({r.stage for r in runs})
    if len(stages) > 1:
        out.append(f"서로 다른 실험 단계({', '.join(stages)})가 함께 선택됐습니다. 조건이 다른 시행입니다.")
    sessions = sorted({session_of(r).name for r in runs})
    if len(sessions) > 1:
        out.append(f"{len(sessions)}개 세션의 시행이 섞여 있습니다: {', '.join(sessions)}")
    if any(r.legacy_time_base for r in runs):
        out.append("이전 형식(sim-verify) 시행이 포함돼 있습니다: 시간 기준이 달라 Lag이 최대 약 20 ms 크게 나올 수 있습니다.")
    return out


def _v(v: Any, unit: str) -> str:
    if not (isinstance(v, (int, float)) and math.isfinite(v)):
        return "-"
    return {"deg": f"{v:.2f}°", "mm": f"{v:.2f} mm", "ms": f"{v:.0f} ms"}.get(unit, f"{v:.2f}")


def _ro(text: str) -> str:
    """Korean particle (으)로 after a value: a bare number is read digit by digit (0, 3, 6 end in a consonant)."""
    return text + ("으로" if text[-1:] in "036" and text[-1:] != "" else "로")


def _change(a: float, b: float, unit: str, tol: float = 1e-9) -> str:
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (a, b)):
        return "비교할 수 없었다(값 없음)"
    if abs(b - a) <= tol:
        return f"{_ro(_v(a, unit))} 같았다"
    return f"{_v(a, unit)}에서 {_ro(_v(b, unit))} {'커졌다' if b > a else '작아졌다'}"


def _trend(vals: list[float]) -> str:
    if len(vals) < 3 or not all(math.isfinite(v) for v in vals):
        return ""
    d = np.diff(vals)
    return " (단계마다 커짐)" if np.all(d > 0) else (" (단계마다 작아짐)" if np.all(d < 0) else " (단계마다 일정하지 않음)")


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return float(np.mean(xs)) if xs else math.nan


def _r1_groups(runs: list[LoadedRun]) -> list[tuple[tuple[str, str], dict[float, list[float]]]]:
    """(data_kind, axis) -> {|amplitude|: amplitude ratios} of the R1 runs, for axes with at least two amplitudes."""
    groups: dict[tuple[str, str], dict[float, list[float]]] = {}
    for x in r1_rows(runs):
        groups.setdefault((x["data_kind"], x["axis"]), {}).setdefault(float(x["amplitude"]), []).append(
            x["amplitude_ratio"])
    return [(k, v) for k, v in sorted(groups.items()) if len(v) >= 2]


def describe(runs: list[LoadedRun]) -> list[str]:
    """Plain sentences on the selected runs built ONLY from computed numbers: sizes and directions of change,
    never a cause (CAUSAL_DISCLAIMER closes the list)."""
    if not runs:
        return ["선택한 시행이 없다."]
    n = kind_counts(runs)
    out = [f"선택한 시행 {len(runs)}개: " + ", ".join(f"{k} {n[k]}개" for k in KINDS if n[k]) + "."
           + ("" if n["measured"] else " 실제 로봇 측정(measured) 결과는 없다.")]
    used: set[str] = set()
    groups: dict[tuple[str, str, float], list[LoadedRun]] = {}
    for r in runs:
        if r.stage == "R3":
            c = r.condition
            vary = c.get("vary", "duration")
            fixed = c.get("amplitude_deg") if vary == "duration" else c.get("duration_s")
            groups.setdefault((r.data_kind, vary, float(fixed)), []).append(r)
    for (kind, vary, fixed), rs in sorted(groups.items()):
        key = "duration_s" if vary == "duration" else "amplitude_deg"
        by_val: dict[float, list[LoadedRun]] = {}
        for r in rs:
            by_val.setdefault(float(r.condition[key]), []).append(r)
        if len(by_val) < 2:
            continue
        order = sorted(by_val, reverse=vary == "duration")  # D long -> short, A small -> large
        ch = rs[0].main_channel
        unit = rs[0].unit(ch)[1]
        ser = {m: [_mean([_display(r, ch)[m] for r in by_val[x]]) for x in order] for m in KEY_METRICS}
        reps = any(len(by_val[x]) > 1 for x in order)
        p, pu, fu = ("D", " s", "°") if vary == "duration" else ("A", "°", " s")
        out.append(f"R3 ({kind}, {'A' if vary == 'duration' else 'D'} {fixed:g}{fu} 고정, 단계 {len(order)}개"
                   f"{', 반복 평균' if reps else ''}): {p}를 {order[0]:g}{pu}에서 {order[-1]:g}{pu}로 바꾸는 동안 "
                   f"{ch} 기준 RMSE(지연 포함)는 {_change(ser['dynamic_rmse'][0], ser['dynamic_rmse'][-1], unit)}"
                   f"{_trend(ser['dynamic_rmse'])}, Peak Error(지연 포함)는 "
                   f"{_change(ser['peak_error'][0], ser['peak_error'][-1], unit)}"
                   f"{_trend(ser['peak_error'])}, Lag은 {_change(ser['lag_ms'][0], ser['lag_ms'][-1], 'ms')}"
                   f"{_trend(ser['lag_ms'])}, 진폭비는 {_change(ser['amplitude_ratio'][0], ser['amplitude_ratio'][-1], '')}"
                   f"{_trend(ser['amplitude_ratio'])}.")
        used.update(r.label + str(r.run_dir) for r in rs)
    for (kind, axis), pts in _r1_groups(runs):
        amps = sorted(pts)
        ratios = [_mean(pts[a]) for a in amps]
        unit = R1_UNIT.get(axis, "deg")
        out.append(f"R1 {axis} ({kind}, 방향·반복 평균): 명령 진폭을 {amps[0]:g}에서 {amps[-1]:g} {unit}로 키우는 동안 "
                   f"진폭비는 {_change(ratios[0], ratios[-1], '')}{_trend(ratios)}.")
        used.update(r.label + str(r.run_dir) for r in runs if r.stage == "R1" and r.data_kind == kind
                    and r.condition.get("axis") == axis)
    all_pairs = pair_rows(runs)
    for motion, kind in sorted({(x["motion"], x["data_kind"]) for x in all_pairs}):
        parts = []
        for role in (ROLE_HEAD, ROLE_JOINT):  # the key table's channel first, then the representative joint
            pts = [x for x in all_pairs if (x["motion"], x["data_kind"], x["role"]) == (motion, kind, role)]
            if not pts:
                continue
            e = {m: _mean([x[m] for x in pts if x["variant"] == "expressive"]) for m in KEY_METRICS}
            ne = {m: _mean([x[m] for x in pts if x["variant"] == "neutral"]) for m in KEY_METRICS}
            unit, ch = pts[0]["unit"], pts[0]["channel"]
            diff = e["dynamic_rmse"] - ne["dynamic_rmse"]
            cmp = "크다" if diff > 0 else ("작다" if diff < 0 else "같다")
            parts.append(f"{role} {ch} 기준 RMSE(지연 포함)는 Expressive {_v(e['dynamic_rmse'], unit)}, Neutral "
                         f"{_ro(_v(ne['dynamic_rmse'], unit))} Expressive가 {_v(abs(diff), unit)} {cmp} (Peak Error "
                         f"{_v(e['peak_error'], unit)} 대 {_v(ne['peak_error'], unit)}, Lag {_v(e['lag_ms'], 'ms')} 대 "
                         f"{_v(ne['lag_ms'], 'ms')}, 진폭비 {_v(e['amplitude_ratio'], '')} 대 "
                         f"{_v(ne['amplitude_ratio'], '')})")
        if parts:
            out.append(f"{motion} ({kind}): " + "; ".join(parts) + ".")
        used.update(r.label + str(r.run_dir) for r in runs if r.condition.get("motion") == motion and r.data_kind == kind)
    if all_pairs:
        out.append(EN_METRIC_NOTE)
    rest = [r for r in runs if r.label + str(r.run_dir) not in used]
    for r in rest[:8]:
        d = _display(r, r.main_channel)
        out.append(f"{r.label} ({r.data_kind}, {r.stage}, {condition_text(r.condition) or '-'}): {r.main_channel} 명령 "
                   f"진폭 {_v(d['command_p2p'], d['unit'])}, RMSE(지연 포함) {_v(d['dynamic_rmse'], d['unit'])}, "
                   f"Peak Error(지연 포함) {_v(d['peak_error'], d['unit'])}, Lag {_v(d['lag_ms'], 'ms')}, "
                   f"진폭비 {_v(d['amplitude_ratio'], '')}.")
    if len(rest) > 8:
        out.append(f"(나머지 시행 {len(rest) - 8}개는 핵심 지표 표를 보세요.)")
    low = [r.label for r in runs if math.isfinite(r.metrics[r.main_channel].lag_fit_r2)
           and r.metrics[r.main_channel].lag_fit_r2 < 0.9]
    if low:
        out.append(f"Lag의 r2가 0.9보다 작은 시행({', '.join(low[:5])}{' 외' if len(low) > 5 else ''})은 순수 지연으로 "
                   "잘 설명되지 않으므로 Lag 값을 참고만 한다.")
    out.append(CAUSAL_DISCLAIMER)
    return out


def _arrow(a: float, b: float, unit: str, tol: float = 1e-9) -> str:
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (a, b)):
        return "값 없음"
    if abs(b - a) <= tol:
        return f"{_v(a, unit)} (같음)"
    return f"{_v(a, unit)} → {_v(b, unit)} ({'커짐' if b > a else '작아짐'})"


def describe_simple(runs: list[LoadedRun]) -> list[str]:
    """The same computed facts as :func:`describe`, as short lines for a first-time reader (Basic mode): one number
    per line, everyday words first, the English metric name in brackets. Never a cause."""
    if not runs:
        return ["선택한 시행이 없습니다."]
    n = kind_counts(runs)
    out = [f"시행 {len(runs)}개 (" + ", ".join(f"{KIND_KO[k]} {n[k]}개" for k in KINDS if n[k]) + ")."
           + ("" if n["measured"] else " 이 중 실제 로봇 측정값은 없습니다.")]
    used: set[str] = set()
    groups: dict[tuple[str, str, float], list[LoadedRun]] = {}
    for r in runs:
        if r.stage == "R3":
            c = r.condition
            vary = c.get("vary", "duration")
            fixed = c.get("amplitude_deg") if vary == "duration" else c.get("duration_s")
            groups.setdefault((r.data_kind, vary, float(fixed)), []).append(r)
    for (kind, vary, fixed), rs in sorted(groups.items()):
        key = "duration_s" if vary == "duration" else "amplitude_deg"
        by_val: dict[float, list[LoadedRun]] = {}
        for r in rs:
            by_val.setdefault(float(r.condition[key]), []).append(r)
        if len(by_val) < 2:
            continue
        order = sorted(by_val, reverse=vary == "duration")  # D long -> short, A small -> large
        ch = rs[0].main_channel
        unit = rs[0].unit(ch)[1]
        ser = {m: [_mean([_display(r, ch)[m] for r in by_val[x]]) for x in order] for m in KEY_METRICS}
        what = (f"진폭 {fixed:g}°는 그대로 두고, 이동 시간을 {order[0]:g} s에서 {order[-1]:g} s로 줄였습니다"
                if vary == "duration" else
                f"이동 시간 {fixed:g} s는 그대로 두고, 진폭을 {order[0]:g}°에서 {order[-1]:g}°로 키웠습니다")
        reps = " (반복 평균)" if any(len(by_val[x]) > 1 for x in order) else ""
        out.append(f"R3 끄덕임 ({KIND_KO.get(kind, kind)}, {channel_ko(ch)} 기준){reps}: {what}.")
        out.append(f"평균 오차 (RMSE, 지연 포함): {_arrow(ser['dynamic_rmse'][0], ser['dynamic_rmse'][-1], unit)}")
        out.append(f"가장 큰 오차 (Peak Error, 지연 포함): {_arrow(ser['peak_error'][0], ser['peak_error'][-1], unit)}")
        out.append(f"늦게 따라간 시간 (Lag): {_arrow(ser['lag_ms'][0], ser['lag_ms'][-1], 'ms')}")
        out.append(f"움직인 크기 ÷ 명령 크기 (진폭비, 1이면 명령만큼): "
                   f"{_arrow(ser['amplitude_ratio'][0], ser['amplitude_ratio'][-1], '')}")
        used.update(r.label + str(r.run_dir) for r in rs)
    for (kind, axis), pts in _r1_groups(runs):
        amps = sorted(pts)
        unit = R1_UNIT.get(axis, "deg")
        out.append(f"R1 {channel_ko(axis)} ({KIND_KO.get(kind, kind)}): 명령을 {amps[0]:g}에서 {amps[-1]:g} {unit}로 "
                   f"키웠을 때 움직인 크기 ÷ 명령 크기 (진폭비, 1이면 명령만큼): "
                   f"{_arrow(_mean(pts[amps[0]]), _mean(pts[amps[-1]]), '')}")
        used.update(r.label + str(r.run_dir) for r in runs if r.stage == "R1" and r.data_kind == kind
                    and r.condition.get("axis") == axis)
    all_pairs = pair_rows(runs)
    for motion, kind in sorted({(x["motion"], x["data_kind"]) for x in all_pairs}):
        for role in (ROLE_HEAD, ROLE_JOINT):
            pts = [x for x in all_pairs if (x["motion"], x["data_kind"], x["role"]) == (motion, kind, role)]
            if not pts:
                continue
            e = _mean([x["dynamic_rmse"] for x in pts if x["variant"] == "expressive"])
            ne = _mean([x["dynamic_rmse"] for x in pts if x["variant"] == "neutral"])
            unit, ch = pts[0]["unit"], pts[0]["channel"]
            if all(math.isfinite(x) for x in (e, ne)):
                cmp = (f"Expressive가 {_v(abs(e - ne), unit)} {'큽니다' if e > ne else '작습니다'}" if e != ne else
                       "두 변형이 같습니다")
            else:
                cmp = "비교할 값이 없습니다"
            out.append(f"{motion} ({KIND_KO.get(kind, kind)}), {channel_ko(ch)}의 평균 오차 (RMSE, 지연 포함): Expressive "
                       f"{_v(e, unit)}, Neutral {_v(ne, unit)}. {cmp}.")
        used.update(r.label + str(r.run_dir) for r in runs if r.condition.get("motion") == motion and r.data_kind == kind)
    if all_pairs:
        out.append(EN_METRIC_NOTE_SHORT)
    rest = [r for r in runs if r.label + str(r.run_dir) not in used]
    for r in rest[:8]:
        d = _display(r, r.main_channel)
        out.append(f"{r.label} ({KIND_KO.get(r.data_kind, r.data_kind)}, {channel_ko(r.main_channel)}): 평균 오차(지연 "
                   f"포함) {_v(d['dynamic_rmse'], d['unit'])}, 가장 큰 오차 {_v(d['peak_error'], d['unit'])}, 늦게 따라간 시간 "
                   f"{_v(d['lag_ms'], 'ms')}, 진폭비 {_v(d['amplitude_ratio'], '')}.")
    if len(rest) > 8:
        out.append(f"(나머지 시행 {len(rest) - 8}개는 핵심 지표 표를 보세요.)")
    low = [r.label for r in runs if math.isfinite(r.metrics[r.main_channel].lag_fit_r2)
           and r.metrics[r.main_channel].lag_fit_r2 < 0.9]
    if low:
        out.append(f"늦게 따라간 시간(Lag)은 참고만 하세요 (단순한 지연으로 잘 설명되지 않음, r2 < 0.9): "
                   f"{', '.join(low[:5])}{' 외' if len(low) > 5 else ''}.")
    out.append(SIMPLE_DISCLAIMER)
    return out


def run_readme(run: LoadedRun) -> str:
    """Human-readable record of one run (README.md in the run folder) for team members."""
    from reachy_motion.gui.lab import DATA_KIND_NOTES

    m = run.manifest
    trial = m.get("trial", {})
    res_ = m.get("result", {})
    L = [f"# {trial.get('title', run.label)}", "",
         f"- 시행 이름: `{run.label}` (세션 `{session_of(run).name}`)",
         f"- 단계: {run.stage}, 조건: {condition_text(run.condition) or '-'}",
         f"- 데이터 출처(data_kind): **{run.data_kind}** - {DATA_KIND_NOTES.get(run.data_kind, '')}",
         f"- 실행 시각(이 PC): {m.get('started_local', '-')}",
         f"- 완료: {'예' if res_.get('completed') else '아니오'}"
         + (f", 정지 {res_.get('stop_kind')}: {res_.get('stop_reason')}" if res_.get("stop_kind") else ""),
         "", f"## 결과 (computed from {run.data_kind} data)", "",
         "| 채널 | 역할 | 명령 진폭 | RMSE (지연 포함) | 지연 보정 RMSE | Peak Error (지연 포함) | 지연 Lag (r²) | 진폭비 | "
         "Overshoot | 정착 시간 | 정지 중 떨림 RMS |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in metric_rows([run], "representative"):
        u = row["unit"]
        L.append(f"| {row['channel']} | {row['role']} | {_v(row['command_p2p'], u)} | {_v(row['dynamic_rmse'], u)} | "
                 f"{_v(row['dynamic_rmse_lag_compensated'], u)} | {_v(row['peak_error'], u)} | "
                 f"{_v(row['lag_ms'], 'ms')} ({_n(row['lag_fit_r2'])}) | {_n(row['amplitude_ratio'])} | "
                 f"{_v(row['overshoot'], u)} | {_n(row['settling_time_s'])} s | {_v(row['hold_ripple'], u)} |")
    notes = read_notes(run.run_dir)
    L += ["", "## 메모", "", notes or "(없음)", "",
          "## 파일", "",
          "- `commands.csv` 보낸 명령 / `present_<data_kind>.csv` 측정값 (같은 시간 기준) / `ticks.csv` 실행기 주기 "
          "(pc_measured) / `manifest.json` 조건·검사·환경 / `metrics.json` 추종 지표 (SI)",
          "- 다른 PC에서 보기: Motion Lab → ③ 결과 → '다른 폴더 불러오기'로 이 폴더(또는 세션 폴더)를 고르면 같은 "
          "그래프와 지표가 나온다."]
    return "\n".join(L) + "\n"


def write_run_readme(run: LoadedRun) -> Path:
    p = Path(run.run_dir) / "README.md"
    p.write_text(run_readme(run), encoding="utf-8")
    return p


def update_session_readme(session_dir: Path | str) -> Path:
    """README.md of a session folder: every run with its condition, data_kind, time and main-channel result
    (read from each run's manifest.json and metrics.json; nothing recomputed)."""
    session = Path(session_dir)
    L = [f"# Motion Lab 세션 `{session.name}`", "",
         "| 시행 | 단계 | 조건 | data_kind | 실행 시각 | 완료 | 채널 | RMSE | Peak Error | Lag ms | 진폭비 | 메모 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run_dir in sorted(p for p in (session / "runs").glob("*") if p.is_dir()):
        try:
            man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        trial = man.get("trial", {})
        ch = trial.get("main_channel", "pitch")
        k = 1000.0 if ch in ("x", "y", "z") else 180.0 / math.pi
        unit = "mm" if ch in ("x", "y", "z") else "deg"
        vals = {"dynamic_rmse": math.nan, "peak_error": math.nan, "lag_s": math.nan, "amplitude_ratio": math.nan}
        try:
            met = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"][ch]
            vals = {key: (met.get(key) if met.get(key) is not None else math.nan) for key in vals}
        except (OSError, ValueError, KeyError):
            pass
        notes = read_notes(run_dir).replace("\n", " ")
        L.append(f"| {trial.get('label', run_dir.name)} | {trial.get('stage', '-')} | "
                 f"{condition_text(trial.get('condition', {})) or '-'} | {man.get('data_kind')} | "
                 f"{man.get('started_local', '-')} | {'예' if man.get('result', {}).get('completed') else '아니오'} | {ch} | "
                 f"{_v(vals['dynamic_rmse'] * k, unit)} | {_v(vals['peak_error'] * k, unit)} | "
                 f"{_v(vals['lag_s'] * 1000.0, 'ms')} | {_n(vals['amplitude_ratio'])} | {notes[:60]} |")
    L += ["", "값은 각 시행의 data_kind로 계산했다(simulated/mock은 로봇 측정이 아님). 자세한 내용은 각 시행 폴더의 README.md."]
    p = session / "README.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def kind_counts(runs: list[LoadedRun]) -> dict[str, int]:
    return {k: sum(1 for r in runs if r.data_kind == k) for k in KINDS}


def source_banner(runs: list[LoadedRun]) -> str:
    """One line on where the shown numbers come from (shown above every result view)."""
    if not runs:
        return "불러온 결과가 없습니다. 왼쪽에서 세션을 불러오거나 실험 탭에서 시행하세요."
    n = kind_counts(runs)
    parts = [f"{k}: {n[k]}" for k in KINDS if n[k]]
    text = "데이터 출처 - " + ", ".join(parts)
    if not n["measured"]:
        text += "   |   실제 로봇 측정(measured) 결과 없음: 여기 보이는 값은 로봇 측정값이 아닙니다"
    return text


# ---------------------------------------------------------------------------------------------- export
def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        return "" if not math.isfinite(v) else f"{v:.6g}"
    return v


def write_rows_csv(path: Path | str, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # BOM: Excel reads the Korean text
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})
    return path


def _json_safe(o: Any) -> Any:
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.generic):
        return _json_safe(o.item())
    if isinstance(o, Path):
        return str(o)
    return o


def metrics_payload(run: LoadedRun) -> dict[str, Any]:
    from reachy_motion.analysis.tracking import LEGACY_ALIGNMENT_NOTE

    return {"run": run.label, "run_dir": str(run.run_dir), "data_kind": run.data_kind, "stage": run.stage,
            "condition": run.condition, "main_channel": run.main_channel, "representative": run.representative,
            "legacy_time_base": run.legacy_time_base,
            "time_alignment": LEGACY_ALIGNMENT_NOTE if run.legacy_time_base else "one time base (Motion Lab run)",
            "units": "SI (rad, m, s)", "metrics": {c: m.to_dict() for c, m in run.metrics.items()},
            "parameters": {"settle_abs_tol": {"rad_deg": 0.2, "m_mm": 0.2}, "settle_rel_tol": 0.05,
                           "analysis": "reachy_motion.analysis.tracking"},
            "data_kind_of_metrics": f"computed from {run.data_kind} data"}


def write_metrics_json(path: Path | str, runs: LoadedRun | list[LoadedRun]) -> Path:
    path = Path(path)
    items = [runs] if isinstance(runs, LoadedRun) else list(runs)
    body = metrics_payload(items[0]) if isinstance(runs, LoadedRun) else {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_kinds": kind_counts(items), "runs": [metrics_payload(r) for r in items]}
    path.write_text(json.dumps(_json_safe(body), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _n(v: Any, fmt: str = ".2f") -> str:
    return format(v, fmt) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else "-"


def summary_markdown(runs: list[LoadedRun]) -> str:
    """Report summary (Korean) of the loaded runs; every table names its data_kind."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n = kind_counts(runs)
    L = [f"# Motion Lab 결과 요약 (자동 생성 {now})", "",
         f"데이터 출처: measured {n['measured']}개, simulated {n['simulated']}개, mock {n['mock']}개.", ""]
    if not runs:
        return "\n".join(L + ["불러온 결과가 없다. 실험 탭에서 시행하거나 결과 폴더를 불러온다.", ""])
    if not n["measured"]:
        L += ["**실제 로봇 측정(measured) 결과 없음.** 아래 값은 모두 시뮬레이션(simulated) 또는 Mock 모델(mock)에서 "
              "계산했다. 로봇의 추종 성능이 아니다.", ""]
    L += ["## 1. 시행 목록", "", "| 시행 | 단계 | 조건 | data_kind | 완료 | 대표 관절 |", "|---|---|---|---|---|---|"]
    for r in runs:
        done = {True: "예", False: "아니오", None: "-"}[r.completed]
        L.append(f"| {r.label} | {r.stage} | {condition_text(r.condition)} | {r.data_kind} | {done} | {r.representative} |")
    L += ["", "## 2. 추종 지표 (시행 축 + 대표 관절, deg / mm / ms)", "",
          "| 시행 | data_kind | 채널 | 명령 진폭 | 정적 편향 | RMSE (지연 포함) | 지연 보정 RMSE | Peak (지연 포함) | 지연 ms (r²) | "
          "진폭비 | Overshoot | 정착 s | 떨림 RMS |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in metric_rows(runs, "representative"):
        L.append(f"| {row['run']} | {row['data_kind']} | {row['channel']} ({row['unit']}) | {_n(row['command_p2p'])} | "
                 f"{_n(row['static_bias'])} | {_n(row['dynamic_rmse'])} | {_n(row['dynamic_rmse_lag_compensated'])} | "
                 f"{_n(row['peak_error'])} | {_n(row['lag_ms'], '.0f')} ({_n(row['lag_fit_r2'])}) | "
                 f"{_n(row['amplitude_ratio'])} | {_n(row['overshoot'])} | {_n(row['settling_time_s'])} | "
                 f"{_n(row['hold_ripple'], '.3f')} |")
    r1 = r1_rows(runs)
    if r1:
        L += ["", "## 3. R1 작은 동작 (시행 축: 명령 진폭 대비 움직인 비율)", "",
              "| data_kind | 축 | 명령 진폭 | 방향 | 반복 | 진폭비 | 지연 ms (r²) | 정적 편향 | 떨림 RMS |",
              "|---|---|---|---|---|---|---|---|---|"]
        for x in r1:
            L.append(f"| {x['data_kind']} | {x['axis']} | {_n(x['amplitude'], 'g')} {x['unit']} | {x['sign']} | "
                     f"{x['rep'] or '-'} | {_n(x['amplitude_ratio'])} | {_n(x['lag_ms'], '.0f')} ({_n(x['lag_fit_r2'])}) | "
                     f"{_n(x['static_bias'])} | {_n(x['hold_ripple'], '.3f')} |")
    r3 = [x for x in r3_rows(runs) if x["role"] == ROLE_AXIS]
    if r3:
        L += ["", "## 4. R3 단계 비교 (pitch, 한 번에 한 파라미터)", "",
              "| data_kind | 변경 | A deg | D s | 지연 ms (r²) | RMSE (지연 포함) | 지연 보정 RMSE % | Peak (지연 포함) | "
              "진폭비 | Overshoot % | 정착 s | 떨림 RMS |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in r3:
            L.append(f"| {x['data_kind']} | {x['vary']} | {_n(x['amplitude_deg'], 'g')} | {_n(x['duration_s'], 'g')} | "
                     f"{_n(x['lag_ms'], '.0f')} ({_n(x['lag_fit_r2'])}) | {_n(x['dynamic_rmse'])} | "
                     f"{_n(x['shape_error_pct'], '.1f')} | {_n(x['peak_error'])} | {_n(x['amplitude_ratio'])} | "
                     f"{_n(x['overshoot_pct'], '.1f')} | {_n(x['settling_time_s'])} | {_n(x['hold_ripple'], '.3f')} |")
    pairs = [x for x in pair_rows(runs) if x["role"] == ROLE_JOINT]
    if pairs:
        L += ["", "## 5. Expressive vs Neutral (대표 관절)", "", f"> {EN_METRIC_NOTE}", "",
              "| 모션 | data_kind | 변형 | 시행 | 채널 | 명령 진폭 | RMSE (지연 포함) | Peak (지연 포함) | 지연 ms | 진폭비 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for x in pairs:
            L.append(f"| {x['motion']} | {x['data_kind']} | {x['variant']} | {x['run']} | {x['channel']} | "
                     f"{_n(x['command_p2p'])} | {_n(x['dynamic_rmse'])} | {_n(x['peak_error'])} | "
                     f"{_n(x['lag_ms'], '.0f')} | {_n(x['amplitude_ratio'])} |")
    L += ["", "## 6. 해석 주의", "",
          "- simulated = MuJoCo: 마찰이 지배적이고 이 PC에서 실시간보다 느리다. 설계와 기록 절차 확인용이며 로봇 추종 예측이 아니다.",
          "- mock = Mock 백엔드의 합성 모델 출력이다(측정 아님).",
          "- 지연 = PC에서 본 전체 지연(명령 전송 → 측정 도착). r² < 0.9이면 지연 값을 믿지 않는다.",
          "- RMSE·Peak Error에는 지연이 들어 있다(실제 로봇 약 190 ms). 동작 모양의 오차는 지연 보정 RMSE로 본다.",
          "- R3 멈춤 기준(`robot_r1.R3_LIMITS`, 2026-11-13): 진폭비 < 0.8, 지연 보정 RMSE·오버슈트 > 명령 진폭의 15 %, 지연이 "
          "사다리 첫 단계(D 1 s, 진폭 사다리는 A 10°)보다 100 ms 넘게 늘어남. 정지 중 떨림은 표시만 한다(기준 아님).",
          "- 추종 오차 지표는 명령 추종의 정확도다. Expressive와 Neutral의 감정 표현력이나 의도 인식률을 뜻하지 않는다."]
    if any(r.legacy_time_base for r in runs):
        L.append("- legacy 시행(sim-verify)은 두 CSV의 시간 원점이 달라 지연이 최대 약 20 ms 과대일 수 있다.")
    return "\n".join(L) + "\n"


__all__ = ["LoadedRun", "find_runs", "load_result", "metric_rows", "r3_rows", "r1_rows", "R1_GROUPS", "pair_rows",
           "run_brief",
           "session_of", "read_notes", "write_notes", "main_metrics_rows", "mixing_warnings", "describe", "run_readme",
           "run_order", "describe_simple", "EN_METRIC_NOTE", "EN_METRIC_NOTE_SHORT", "SIMPLE_DISCLAIMER", "KIND_KO",
           "channel_ko",
           "write_run_readme", "update_session_readme", "KEY_METRICS", "CAUSAL_DISCLAIMER",
           "kind_counts", "source_banner", "write_rows_csv", "write_metrics_json", "summary_markdown",
           "condition_text", "channels_for", "METRIC_FIELDS", "METRIC_LABELS", "KINDS", "ROLE_AXIS", "ROLE_JOINT",
           "ROLE_HEAD"]
