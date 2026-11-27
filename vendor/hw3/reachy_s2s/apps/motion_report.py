"""Summarise the motion side of dashboard / preview sessions (Phase 4.5): what the controller did and
what the conversation looked like meanwhile.

    python -m reachy_s2s.apps.motion_report results/sessions/<id>_dashboard [more session dirs]

Per session (also written to <session>/motion_report.json):
* motion: sink (simulated or robot), mode sequence, state reactions (state change -> first tick that
  applied it), speech-start reactions, safety guard / limiter / box counts, errors, emergency stops,
  control loop statistics (tick rate, intervals, late ticks, sends, thread CPU);
* commands (motion.csv): per phase (conversation / ramps) the deviation from neutral, speed,
  acceleration, Stewart joint change and motor-limit margin (PC inverse kinematics);
* conversation: turns per variant / motion / TTS threads, speech end -> push, input gaps (all, and
  during TTS synthesis);
* resources: process and per-thread CPU (``system.resources``, if the monitor was on).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config
from ..evaluation.event_logger import read_events
from ..evaluation.resource_monitor import resource_summary
from ..motion.analysis import phase_stats, read_motion_csv
from ..motion.limits import JointGuard


def _stats(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {"n": len(values), "median": round(float(np.median(values)), 1), "max": round(float(max(values)), 1)}


def _windows(events: list[dict[str, Any]], start: str, end: str) -> list[tuple[float, float]]:
    starts = [e["t"] for e in events if e["kind"] == start]
    ends = [e["t"] for e in events if e["kind"] == end]
    return [(s, e) for s, e in zip(starts, ends)]


def report(session_dir: Path, guard: JointGuard | None) -> dict[str, Any]:
    events = read_events(session_dir / "events.jsonl")
    kinds = lambda k: [e for e in events if e["kind"] == k]  # noqa: E731
    started = kinds("motion.started")
    out: dict[str, Any] = {"session": session_dir.name}
    if started:
        st = [e for e in kinds("motion.stats")]
        states = kinds("motion.state")
        gestures = kinds("motion.gesture")
        out["motion"] = {
            "sink": started[0].get("sink"), "simulated": started[0].get("simulated"),
            "joint_guard": started[0].get("guard"),
            "modes": [f"{e['to_mode']} ({e['reason']})" for e in kinds("motion.mode")],
            "state_reactions": len(states),
            "state_reaction_ms": _stats([e["cancel_ms"] for e in states if e.get("cancel_ms") is not None]),
            "states_seen": sorted({e["state"] for e in states}),
            "gestures": {g: sum(1 for e in gestures if e["gesture"] == g) for g in sorted({e["gesture"] for e in gestures})},
            "gesture_latency_ms": _stats([e["latency_ms"] for e in gestures if e.get("latency_ms") is not None]),
            "guard_events": [{k: e.get(k) for k in ("action", "chain", "reason")} for e in kinds("motion.guard")],
            "errors": [e.get("error") for e in kinds("motion.error")],
            "estops": [e.get("result") for e in kinds("motion.estop")],
            "settings": [{k: v for k, v in e.items() if k not in ("t", "kind")} for e in kinds("motion.setting")],
            "control": {
                "windows": len(st),
                "tick_hz_min": min((e["tick_hz"] for e in st), default=None),
                "interval_ms_p95_max": max((e["interval_ms_p95"] for e in st if e.get("interval_ms_p95")), default=None),
                "interval_ms_max": max((e["interval_ms_max"] for e in st if e.get("interval_ms_max")), default=None),
                "late_ticks": sum(e.get("late_ticks", 0) for e in st),
                "stalled_ticks": sum(e.get("stalled_ticks", 0) for e in st),
                "sends": sum(e.get("sends", 0) for e in st),
                "deadband_skips": sum(e.get("deadband_skips", 0) for e in st),
                "limited": sum(e.get("limited", 0) for e in st),
                "box_clamped": sum(e.get("box_clamped", 0) for e in st),
                "guard_scaled": sum(e.get("guard_scaled", 0) for e in st),
                "guard_rejected": sum(e.get("guard_rejected", 0) for e in st),
                "thread_cpu_pct_max": max((e["thread_cpu_pct"] for e in st if e.get("thread_cpu_pct") is not None),
                                          default=None),
            },
        }
        starts = [e for e in kinds("motion.mode") if e.get("to_mode") == "STARTING"]
        torso = [e.get("torso_yaw_deg") for e in starts if e.get("torso_yaw_deg") is not None]
        out["motion"]["torso_yaw_deg"] = torso
        if len(set(torso)) > 1:
            out["motion"]["note"] = "the torso yaw differed between motor starts; joints use the first one"
        csv_path = session_dir / "motion.csv"
        if csv_path.exists():
            if guard is not None:
                guard.set_body_yaw(torso[0] if torso else 0.0)
            out["commands"] = phase_stats(read_motion_csv(csv_path), guard)
    turns_path = session_dir / "turns.csv"
    if turns_path.exists():
        with turns_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        groups: dict[str, list[dict[str, str]]] = {}
        for r in rows:
            key = f"{r.get('variant') or '-'} | motion {r.get('motion') or 'none'} | tts_threads {r.get('tts_threads') or '-'}"
            groups.setdefault(key, []).append(r)
        out["turns"] = {key: {"turns": len(rs), "statuses": {s: sum(1 for r in rs if r["status"] == s)
                                                            for s in sorted({r["status"] for r in rs})},
                              "speech_end_to_push_ms": _stats([float(r["speech_end_to_push_ms"]) for r in rs
                                                               if r.get("speech_end_to_push_ms")]),
                              "tts_ms": _stats([float(r["tts_ms"]) for r in rs if r.get("tts_ms")]),
                              "interrupted": sum(1 for r in rs if r.get("interrupted") == "True")}
                        for key, rs in groups.items()}
    gaps = kinds("audio.gap")
    tts = _windows(events, "tts.start", "tts.end")
    in_tts = [g for g in gaps if any(s - 0.05 <= g["t"] <= e + 0.3 for s, e in tts)]
    out["input_gaps"] = {"all": len(gaps), "during_tts": len(in_tts), "tts_windows": len(tts),
                         "gap_ms": _stats([g["gap_ms"] for g in gaps])}
    res = resource_summary(events)
    if res:
        out["resources"] = res
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+", type=Path)
    args = parser.parse_args(argv)
    m = load_config().motion
    guard = JointGuard(m.joint_max_change_deg, m.joint_limit_margin_deg)
    for d in args.sessions:
        r = report(d, guard if guard.available else None)
        (d / "motion_report.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(r, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
