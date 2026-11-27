"""Phase 3, first task: play every motion file in the local MuJoCo sim and prepare the material for a human review.

One daemon session (127.0.0.1, interlocked by ``SdkBackend('sim')``): start-up ramp from the sleep pose
(``safety.startup``), then for each motion: wait until the head is still, read the measured start, play it
(policy-checked ``PreparedMotion``), hold, read the measured end. Writes per-motion runs (commands / present / ticks),
``summary.json`` and a generated ``review.md`` (planned vs measured start/end, keyframes, pair comparison,
policy metrics, playback commands).

data_kind: plans and policy checks are ``computed``; MuJoCo measurements are ``simulated`` (friction-dominated,
~0.84x real time on this PC: use them to judge the design by eye, not as robot predictions - Phase 1 5.5);
tick timings are ``pc_measured``. Antenna direction and the roll sign are NOT verified on the robot.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from reachy_motion.analysis.goto_replay import JOINT_NAMES
from reachy_motion.analysis.limits import urdf_joint_limits
from reachy_motion.library.compare import comparison_conditions
from reachy_motion.library.pose import pose_from_matrix
from reachy_motion.library.spec import MOTION_NAMES, MotionSpec

DEG = 180.0 / math.pi
ORDER = [(n, v) for n in MOTION_NAMES for v in ("expressive", "neutral")]


def wait_still(backend: Any, window_s: float = 0.3, tol_deg: float = 0.03, timeout_s: float = 6.0) -> dict[str, Any]:
    """Poll the measured joints until body + Stewart moved < tol_deg over window_s (MuJoCo is friction-dominated)."""
    t0 = time.perf_counter()
    samples: list[tuple[float, np.ndarray]] = []
    spread = math.nan
    while True:
        now = time.perf_counter()
        p = backend.present()
        if p.joints is not None:
            samples.append((now, np.asarray(p.joints, dtype=float) * DEG))
        recent = [q for t, q in samples if t >= now - window_s]
        if len(recent) >= 5 and samples[0][0] <= now - window_s:
            spread = float(np.ptp(np.array(recent), axis=0).max())
            if spread < tol_deg:
                return {"settled": True, "waited_s": now - t0, "spread_deg": spread}
        if now - t0 > timeout_s:
            return {"settled": False, "waited_s": now - t0, "spread_deg": spread}
        time.sleep(0.02)


def measured_state(backend: Any) -> dict[str, Any]:
    p = backend.present()
    head = pose_from_matrix(p.head).to_deg_mm()
    return {"head": {k: float(v) for k, v in head.items()},
            "antennas_deg": [float(a * DEG) for a in p.antennas], "body_yaw_deg": float(p.body_yaw * DEG),
            "joints_deg": None if p.joints is None else [float(q * DEG) for q in p.joints]}


def planned_state(state: Any) -> dict[str, Any]:
    return {"head": {k: float(v) for k, v in state.head.to_deg_mm().items()},
            "antennas_deg": [float(a * DEG) for a in state.antennas], "body_yaw_deg": float(state.body_yaw * DEG)}


def state_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """Largest absolute differences (deg for angles, mm for translation)."""
    ang = max(abs(a["head"][k] - b["head"][k]) for k in ("roll_deg", "pitch_deg", "yaw_deg"))
    trans = max(abs(a["head"][k] - b["head"][k]) for k in ("x_mm", "y_mm", "z_mm"))
    ant = max(abs(x - y) for x, y in zip(a["antennas_deg"], b["antennas_deg"]))
    return {"head_angle_deg": float(ang), "head_trans_mm": float(trans), "antenna_deg": float(ant),
            "body_deg": float(abs(a["body_yaw_deg"] - b["body_yaw_deg"]))}


def measured_margins(present: list[Any]) -> dict[str, float]:
    """Min margin of the MEASURED Stewart joints to the URDF limits over a run (simulated)."""
    lim = urdf_joint_limits()
    lo = np.array([lim[n][0] for n in JOINT_NAMES[1:]]) * DEG
    hi = np.array([lim[n][1] for n in JOINT_NAMES[1:]]) * DEG
    q = np.array([np.asarray(p.joints, float)[1:] * DEG for p in present if p.joints is not None])
    if q.size == 0:
        return {"min_margin_deg": math.nan, "joint": ""}
    m = np.minimum(q - lo, hi - q)
    j = int(np.argmin(m.min(axis=0)))
    return {"min_margin_deg": float(m.min()), "joint": JOINT_NAMES[1 + j]}


def _key_moves(spec: MotionSpec, min_deg: float) -> list[tuple[str, str, float]]:
    """(label, dominant axis, change) of every head key whose rotation CHANGE from the previous value >= min_deg."""
    axes = ("roll_deg", "pitch_deg", "yaw_deg")
    eff = spec.effective()
    prev = eff.start.head.to_deg_mm()
    out = []
    for k in eff.head:
        v = k.pose.to_deg_mm()
        delta = {a: v[a] - prev[a] for a in axes}
        ax = max(axes, key=lambda a: abs(delta[a]))
        if abs(delta[ax]) >= min_deg:
            out.append((k.label, ax, delta[ax]))
        prev = v
    return out


def functional_match(expressive: MotionSpec, neutral: MotionSpec, min_deg: float = 2.0) -> dict[str, Any]:
    """Does the expressive variant contain the neutral's basic act? For every neutral head key, take its rotation
    CHANGE from the previous key (dominant axis and size) and list the expressive keys whose change has the same
    dominant axis, the same sign and 0.5-2x the size (designed keys; a review aid, not a proof)."""
    expr = _key_moves(expressive, min_deg)
    rows, ok = [], True
    for label, ax, d in _key_moves(neutral, min_deg):
        hits = [f"{el} (Δ{eax.replace('_deg', '')} {ed:+.1f})" for el, eax, ed in expr
                if eax == ax and ed * d > 0 and 0.5 <= abs(ed) / abs(d) <= 2.0]
        ok &= bool(hits)
        rows.append({"neutral_key": f"{label} (Δ{ax.replace('_deg', '')} {d:+.1f})", "expressive_match": ", ".join(hits)
                     or None})
    return {"matches": rows, "all_matched": ok}


def run_review(out_dir: Path, specs: dict[tuple[str, str], MotionSpec], port: int = 8000, viewer: bool = False,
               hold_s: float = 1.0, startup: Any = None) -> dict[str, Any]:
    """Play every spec in ORDER in one MuJoCo session; returns the summary written to ``summary.json``."""
    from reachy_motion.backends.sdk import SdkBackend
    from reachy_motion.executor.executor import Executor
    from reachy_motion.planner import segment_table
    from reachy_motion.safety.checker import check_spec, prepare
    from reachy_motion.sdk_env import environment_report

    out_dir.mkdir(parents=True, exist_ok=True)
    backend = SdkBackend("sim", port=port, viewer=viewer, log_dir=out_dir / "daemon")
    backend.connect()
    motions: dict[str, Any] = {}
    try:
        executor = Executor(backend)
        sleep_state = measured_state(backend)
        ok, ramp = startup(backend, executor)
        if not ok:
            raise RuntimeError("start-up ramp refused")
        for key in ORDER:
            spec = specs.get(key)
            if spec is None:
                continue
            report = check_spec(spec)
            entry: dict[str, Any] = {"policy_pass": bool(report.passes_policy), "policy_metrics": report.metrics,
                                     "warnings": [w.code for w in report.warnings],
                                     "violations": [v.code for v in report.violations],
                                     "planned_start": planned_state(spec.start), "planned_end": planned_state(spec.end_state),
                                     "duration_s": spec.duration, "segments": segment_table(spec)}
            if not report.passes_policy:
                entry["skipped"] = "outside the initial policy"
                motions[spec.key] = entry
                continue
            entry["settle_before"] = wait_still(backend)
            entry["measured_start"] = measured_state(backend)
            result = executor.run(prepare(spec), record_present=True)
            time.sleep(hold_s)
            entry["settle_after"] = wait_still(backend)
            entry["measured_end"] = measured_state(backend)
            entry["completed"] = bool(result.completed)
            entry["error"] = result.error
            entry["measured_margin"] = measured_margins(result.present)
            ticks = result.ticks.summary(int(round(1e9 / executor.rate_hz)))
            entry["ticks_pc_measured"] = {k: ticks.get(k) for k in ("n_ticks", "skipped_samples", "effective_hz")}
            entry["start_error"] = state_diff(entry["measured_start"], entry["planned_start"])
            entry["end_error"] = state_diff(entry["measured_end"], entry["planned_end"])
            run_dir = out_dir / "runs" / f"{spec.name}.{spec.variant}"
            _save_run_files(run_dir, result)
            motions[spec.key] = entry
    finally:
        backend.close()
    pairs = {}
    for name in MOTION_NAMES:
        e, n = specs.get((name, "expressive")), specs.get((name, "neutral"))
        if e is None or n is None:
            continue
        cc = comparison_conditions(e, n)
        me, mn = motions.get(e.key, {}), motions.get(n.key, {})
        meas = {}
        if "measured_start" in me and "measured_start" in mn:
            meas = {"start": state_diff(me["measured_start"], mn["measured_start"]),
                    "end": state_diff(me["measured_end"], mn["measured_end"])}
        pairs[name] = {"planned_same_start": cc["same_start"], "planned_same_end": cc["same_end"],
                       "issues": cc["issues"], "durations_s": {"expressive": e.duration, "neutral": n.duration},
                       "measured_difference_expr_vs_neutral": meas, "functional_act": functional_match(e, n)}
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_kind": {"plans_and_policy": "computed", "measured_poses_and_margins": "simulated",
                      "ticks": "pc_measured"},
        "note": "MuJoCo is friction-dominated and slower than real time on this PC: judge the motions by eye, do not "
                "read the measured errors as robot tracking performance. Antenna direction and roll sign are "
                "unverified on the robot.",
        "sleep_state_measured": sleep_state, "startup_ramp_policy": ramp.report.policy.name if ramp else None,
        "motions": motions, "pairs": pairs, "environment": environment_report(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_jd),
                                          encoding="utf-8")
    (out_dir / "review.md").write_text(review_markdown(summary, specs), encoding="utf-8")
    return summary


def _save_run_files(run_dir: Path, result: Any) -> None:
    from reachy_motion.gui.model import _write_rows, command_rows

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(run_dir / "commands.csv", command_rows(result))
    result.ticks.to_csv(run_dir / "ticks.csv")
    rows = []
    t0 = result.present[0].t_ns if result.present else 0
    for p in result.present:
        row = {"t_s": (p.t_ns - t0) / 1e9}
        row.update({f"{k}": v for k, v in pose_from_matrix(p.head).to_deg_mm().items()})
        row.update({"antenna_right_deg": p.antennas[0] * DEG, "antenna_left_deg": p.antennas[1] * DEG})
        if p.joints is not None:
            row.update({f"{n}_deg": float(q * DEG) for n, q in zip(JOINT_NAMES, p.joints)})
        rows.append(row)
    _write_rows(run_dir / "present_simulated.csv", rows)


def _jd(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _f(v: Any, fmt: str = ".1f") -> str:
    return format(v, fmt) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else "-"


def review_markdown(s: dict[str, Any], specs: dict[tuple[str, str], MotionSpec]) -> str:
    L = ["# MuJoCo 검토 자료 (자동 생성)", "",
         f"생성 {s['generated_utc']} · `python -m reachy_motion sim-verify`. 계획·정책 = computed, MuJoCo 측정 = simulated "
         "(마찰 지배적·실시간보다 느림 → 눈으로 판단용, 로봇 추종 예측 아님), tick = pc_measured.", "",
         "**미검증**: 안테나 방향(droop = right −/left +, perk = 0° 쪽)은 SDK sleep 자세에서 추론했고, roll 부호는 실제 로봇에서 "
         "확인하지 않았다. 초기 정책 통과는 기구적 안전 보장이 아니다.", "",
         "## 1. 실행 결과와 시작·종료 자세 (계획 vs MuJoCo 측정)", "",
         "| 모션 | 실행 | 정지 후 시작 | 측정 시작 (pitch/yaw/roll °, z mm, 안테나 R/L °) | 측정 종료 | 시작 오차 각/이동/안테나 | 종료 오차 각/이동/안테나 | 측정 최소 관절 여유 | tick (건너뜀) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for key, m in s["motions"].items():
        if "measured_start" not in m:
            L.append(f"| {key} | {m.get('skipped', '-')} | | | | | | | |")
            continue

        def st(x: dict[str, Any]) -> str:
            h = x["head"]
            return (f"{_f(h['pitch_deg'])}/{_f(h['yaw_deg'])}/{_f(h['roll_deg'])}, {_f(h['z_mm'])}, "
                    f"{_f(x['antennas_deg'][0])}/{_f(x['antennas_deg'][1])}")

        se, ee = m["start_error"], m["end_error"]
        L.append(f"| {key} | {'완료' if m['completed'] else m['error']} | {'예' if m['settle_before']['settled'] else '아니오'} | "
                 f"{st(m['measured_start'])} | {st(m['measured_end'])} | {_f(se['head_angle_deg'], '.2f')}°/"
                 f"{_f(se['head_trans_mm'], '.2f')} mm/{_f(se['antenna_deg'], '.2f')}° | {_f(ee['head_angle_deg'], '.2f')}°/"
                 f"{_f(ee['head_trans_mm'], '.2f')} mm/{_f(ee['antenna_deg'], '.2f')}° | "
                 f"{_f(m['measured_margin']['min_margin_deg'])}° ({m['measured_margin']['joint']}) | "
                 f"{m['ticks_pc_measured']['n_ticks']} ({m['ticks_pc_measured']['skipped_samples']}) |")
    L += ["", "계획 시작·종료는 8개 모두 rest(head 0, 안테나 −9.998/+9.998°, 몸통 0). 오차 = |측정 − 계획|의 최댓값.", "",
          "## 2. 키프레임 (설계값에 배율 적용 = 실제 실행 파라미터)", ""]
    for key, m in s["motions"].items():
        L += [f"**{key}** — 길이 {m['duration_s']:.2f} s", "",
              "| 트랙 | 라벨 | 도착 s | duration s | pause s | 보간 | 주요 목표값 |", "|---|---|---|---|---|---|---|"]
        for r in m["segments"]:
            if r["track"] == "head":
                vals = ", ".join(f"{k.replace('_deg', '').replace('_mm', '')} {r[k]:+.1f}" for k in
                                 ("pitch_deg", "yaw_deg", "roll_deg", "x_mm", "y_mm", "z_mm") if abs(r[k] or 0) > 1e-9)
                vals = vals or "rest(0)"
            elif r["track"] == "antennas":
                vals = f"R {r['right_deg']:+.1f}, L {r['left_deg']:+.1f}"
            else:
                vals = f"body {r['yaw_deg']:+.1f}"
            L.append(f"| {r['track']} | {r['label']} | {r['t_arrive_s']:.2f} | {r['duration_s']:.2f} | {r['hold_s']:.2f} | "
                     f"{r['easing']} | {vals} |")
        L.append("")
    L += ["## 3. Expressive vs Neutral", "",
          "| 모션 | 계획 같은 시작 / 종료 | 비교 문제 | 길이 E / N s | 측정 시작 차 (각°/이동 mm) | 측정 종료 차 | 기본 동작 일치 (neutral 키 → expressive 키) |",
          "|---|---|---|---|---|---|---|"]
    for name, p in s["pairs"].items():
        meas = p["measured_difference_expr_vs_neutral"]
        ms = f"{_f(meas['start']['head_angle_deg'], '.2f')}/{_f(meas['start']['head_trans_mm'], '.2f')}" if meas else "-"
        me = f"{_f(meas['end']['head_angle_deg'], '.2f')}/{_f(meas['end']['head_trans_mm'], '.2f')}" if meas else "-"
        fm = "; ".join(f"{r['neutral_key']} → {r['expressive_match'] or '없음'}" for r in p["functional_act"]["matches"])
        L.append(f"| {name} | {'예' if p['planned_same_start'] else '아니오'} / {'예' if p['planned_same_end'] else '아니오'} | "
                 f"{len(p['issues'])} | {p['durations_s']['expressive']:.2f} / {p['durations_s']['neutral']:.2f} | "
                 f"{ms} | {me} | {'✓ ' if p['functional_act']['all_matched'] else '✗ '}{fm} |")
    L += ["", "## 4. 전체 궤적 검사 (계획, computed, 초기 정책)", "",
          "| 모션 | 판정 | IK NaN | 최소 관절 여유 ° (관절) | Stewart / 몸통 / 안테나 속도 비 | 최대 roll / pitch / 상대 yaw ° | z 범위 mm | 경고 |",
          "|---|---|---|---|---|---|---|---|"]
    for key, m in s["motions"].items():
        x = m["policy_metrics"]
        L.append(f"| {key} | {'통과' if m['policy_pass'] else '위반 ' + ','.join(m['violations'])} | {x.get('ik_nan_samples')} | "
                 f"{_f(x.get('min_stewart_margin_deg'))} ({x.get('min_stewart_margin_joint')}) | "
                 f"{_f(x.get('peak_stewart_speed_ratio'), '.2f')} / {_f(x.get('peak_body_speed_ratio'), '.2f')} / "
                 f"{_f(x.get('peak_antenna_speed_ratio'), '.2f')} | {_f(x.get('max_abs_roll_deg'))} / "
                 f"{_f(x.get('max_abs_pitch_deg'))} / {_f(x.get('max_abs_rel_yaw_deg'))} | "
                 f"{_f((x.get('z_range_mm') or [math.nan])[0])}…{_f((x.get('z_range_mm') or [0, math.nan])[1])} | "
                 f"{', '.join(m['warnings'])} |")
    L += ["", "## 5. 눈으로 확인하는 재생 명령 (MuJoCo viewer, 127.0.0.1)", "", "```"]
    for key in s["motions"]:
        n, v = key.split(":")
        L.append(f".venv\\Scripts\\python -m reachy_motion sim-play {n} --variant {v}")
    L += ["```", "", "재생 후 Enter = 다시, r = 파일 다시 읽기(수정 반영), v = 다른 변형, q = 종료.", ""]
    return "\n".join(L)
