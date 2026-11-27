"""Summarise the turns of one or more sessions (turns.csv) - stage latencies, failures, turn-taking.

    python -m reachy_s2s.apps.turn_report results/sessions/<id>_dashboard [more session dirs]

Latency fields (ms): vad_endpoint_delay (VAD speech end -> endpoint decision),
asr, llm, tts (stage durations), endpoint_to_push and speech_end_to_push
(measured on this PC), response_latency_est (+ ASSUMED output latency).

Turn-taking (Phase 4), per variant (A0 / A1 / B ...); ``--by`` groups by other turn-row columns
instead, joined with "+": ``motion`` (Phase 4.5 motion condition; "none" in older files) and
``tts_threads`` (local TTS threads; 0 = the Phase 2-4 setting), e.g. ``--by variant+tts_threads``:
* user speech while busy: a local-VAD speech start while the system prepared or spoke an
  answer (may include robot echo); "while speaking, not stopped" are missed or ignored barge-ins;
* interruptions: A1 = barge-in detector (speaking) or new speech while preparing (cancelled);
  B = the server's ``interrupted``; detection delay, playback stop delay, heard share (estimate);
* recovery: the turn answering the interrupting speech finished normally;
* false-interruption suspects: no follow-up turn, or the follow-up heard nothing (empty transcript);
* premature endpoint suspects: the user spoke again within ``premature_window_ms`` of an endpoint.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

FIELDS = ("vad_endpoint_delay_ms", "asr_ms", "llm_ms", "tts_ms", "endpoint_to_push_ms",
          "speech_end_to_push_ms", "response_latency_est_ms", "tts_audio_s", "speech_s")
TURN_TAKING_FIELDS = ("bargein_detect_delay_ms", "playback_stop_delay_ms", "heard_ratio_est", "sent_unheard_s")


def load_turns(session_dirs: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for d in session_dirs:
        path = d / "turns.csv"
        if path.is_file():
            with path.open(encoding="utf-8") as fh:
                rows.extend(dict(r, session=d.name) for r in csv.DictReader(fh))
    return rows


def _num(value: str | None) -> float | None:
    return None if value in (None, "", "None") else float(value)


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"n": len(values), "median": round(float(np.median(values)), 1 if max(values) > 2 else 3),
            "min": round(min(values), 3), "max": round(max(values), 3)}


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    out: dict[str, object] = {"turns": len(rows)}
    statuses: dict[str, int] = {}
    for r in rows:
        key = r["status"] + (f" ({r['error_stage']} {r['error_kind']} {r['error_status']})".rstrip()
                             if r.get("error_stage") else "")
        statuses[key] = statuses.get(key, 0) + 1
    out["status"] = statuses
    done = [r for r in rows if r["status"] == "done"]
    for f in FIELDS:
        vals = [float(r[f]) for r in done if r.get(f) not in (None, "", "None")]
        if vals:
            out[f] = {"n": len(vals), "median": round(float(np.median(vals)), 1),
                      "min": round(min(vals), 1), "max": round(max(vals), 1)}
    return out


def _follow_up(rows: list[dict[str, str]], r: dict[str, str]) -> dict[str, str] | None:
    """The turn that answered the speech which interrupted ``r`` (same session)."""
    same = [x for x in rows if x["session"] == r["session"]]
    tid = r["turn_id"]
    for x in same:  # A1 links it explicitly
        if x.get("bargein_of_turn") == tid or tid in (x.get("merged_from") or "").split():
            return x
    if r.get("system") == "cloud":  # B: the server answers the interrupting speech in its next response
        later = [x for x in same if int(x["turn_id"]) > int(tid)]
        return min(later, key=lambda x: int(x["turn_id"])) if later else None
    return None


def _answered(r: dict[str, str]) -> bool:
    """The turn's answer started playing (it may have been interrupted again afterwards)."""
    return r["status"] in ("done", "interrupted") and _num(r.get("queue_first_push")) is not None


def _recovers(rows: list[dict[str, str]], r: dict[str, str], depth: int = 0) -> bool:
    """Following the interrupting turns, the conversation reaches a normally completed turn."""
    nxt = _follow_up(rows, r)
    if nxt is None or depth > 20:
        return False
    if nxt["status"] == "done":
        return True
    return nxt.get("interrupted") == "True" and _recovers(rows, nxt, depth + 1)


GROUP_COLUMNS = ("variant", "motion", "tts_threads")
_MISSING = {"motion": "none", "tts_threads": "-"}  # value in files written before the column existed


def grouping(text: str) -> str:
    cols = text.split("+")
    if not cols or any(c not in GROUP_COLUMNS for c in cols):
        raise argparse.ArgumentTypeError(f"use columns from {GROUP_COLUMNS} joined with +")
    return text


def _group(r: dict[str, str], by: str) -> str:
    parts = []
    for col in by.split("+"):
        if col == "variant":
            parts.append(r.get("variant") or "(none)")
        else:
            parts.append(f"{col} {r.get(col) or _MISSING[col]}")
    return " | ".join(parts)


def turn_taking(rows: list[dict[str, str]], by: str = "variant") -> dict[str, dict[str, object]]:
    """Phase 4 counters and timings per variant (or per motion condition, Phase 4.5)."""
    by_variant: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_variant.setdefault(_group(r, by), []).append(r)
    out: dict[str, dict[str, object]] = {}
    for variant, vr in by_variant.items():
        busy = [r for r in vr if r.get("user_interrupt_start") not in (None, "", "None")]
        interrupted = [r for r in vr if r.get("interrupted") == "True"]
        speaking_busy = [r for r in busy if r.get("user_interrupt_phase") == "speaking"]
        recovered = false_suspects = bargein_success = 0
        for r in interrupted:
            nxt = _follow_up(rows, r)
            if nxt is None or nxt["status"] == "empty_transcript":
                false_suspects += 1
                continue
            recovered += _recovers(rows, r)
            # stopped while speaking AND the interrupting speech got an answer (PHASE4_PLAN §6)
            bargein_success += r.get("user_interrupt_phase") == "speaking" and _answered(nxt)
        latency = [v for v in (_num(r.get("speech_end_to_push_ms")) for r in vr if r["status"] == "done")
                   if v is not None]
        stats = {"turns": len(vr), "user_speech_while_busy": len(busy),
                 "while_speaking": len(speaking_busy),
                 "while_speaking_not_stopped": sum(1 for r in speaking_busy if r.get("interrupted") != "True"),
                 "bargein_success": f"{bargein_success}/{len(speaking_busy)}",
                 "speech_end_to_push_ms_median": round(float(np.median(latency))) if latency else None,
                 "interruptions": len(interrupted),
                 "cancelled_before_playback": sum(1 for r in vr if r["status"] == "cancelled"),
                 "recovered": recovered, "false_interruption_suspects": false_suspects,
                 "premature_suspects": sum(1 for r in vr if r.get("premature_suspect") == "True"),
                 "history_policy": {p: sum(1 for r in interrupted if r.get("history_policy") == p)
                                    for p in sorted({r.get("history_policy") for r in interrupted} - {None, ""})}}
        for f in TURN_TAKING_FIELDS:
            s = _stats([v for v in (_num(r.get(f)) for r in interrupted) if v is not None])
            if s:
                stats[f] = s
        out[variant] = stats
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--by", type=grouping, default="variant",
                        help=f"turn-taking groups: columns from {GROUP_COLUMNS} joined with + "
                             "(default: variant, as in Phase 4)")
    args = parser.parse_args(argv)
    rows = load_turns(args.sessions)
    for r in rows:
        extra = ""
        if r.get("interrupted") == "True":
            extra = (f"  [interrupted {r.get('interrupt_kind') or 'server'}: detect {r.get('bargein_detect_delay_ms')} ms,"
                     f" stop {r.get('playback_stop_delay_ms')} ms, heard {r.get('heard_ratio_est')},"
                     f" history {r.get('history_policy') or '-'}]")
        user = r.get("merged_text") or r.get("transcript")
        print(f"[{r['session'][:15]} #{r['turn_id']} {r.get('variant') or ''}] {r['status']:<16} user={user!r}{extra}\n"
              f"    robot={r.get('response')!r}")
    summary = summarize(rows)
    print(f"\nturns: {summary['turns']}  status: {summary['status']}")
    for f in FIELDS:
        if f in summary:
            s = summary[f]
            print(f"  {f:<26} median {s['median']:>8}  min {s['min']:>8}  max {s['max']:>8}  (n={s['n']})")
    for variant, s in turn_taking(rows, args.by).items():
        print(f"\nturn-taking [{variant}]: " + ", ".join(f"{k} {v}" for k, v in s.items()
                                                       if k not in TURN_TAKING_FIELDS))
        for f in TURN_TAKING_FIELDS:
            if f in s:
                v = s[f]
                print(f"  {f:<26} median {v['median']:>8}  min {v['min']:>8}  max {v['max']:>8}  (n={v['n']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
