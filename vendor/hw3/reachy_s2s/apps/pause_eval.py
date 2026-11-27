"""Offline pause-handling comparison on recordings from ``apps.pause_record`` (no robot, no API).

    python -m reachy_s2s.apps.pause_eval results/sessions/<id>_pause_record [--timeouts 800,1500,2000]

Every line was meant as ONE utterance. For each line and each silence timeout the recording
goes through Silero + the live segmenter ([vad] settings, only ``silence_timeout_ms`` changes):

* segments: utterances the VAD would hand to the cascade; more than one = premature endpoint;
* pauses_ms: silences inside the line (from a run with a very long timeout), i.e. what the
  user actually did, independent of the setting;
* endpoint_delay_ms: endpoint decision - end of the last speech (what the user waits before
  the cascade even starts ASR);
* input_gaps: Wi-Fi arrival gaps inside the line (from the recording's chunk timing), so
  endpoint errors with a gap nearby can be told apart from VAD errors.

Writes pause_eval.csv and pause_eval.json next to the recordings and prints per-timeout rates.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.convert import to_mono
from ..audio.wavio import read_wav
from ..config import load_config
from ..engines.vad import SAMPLE_RATE, SileroVadModel, segment_probs


def arrival_gaps(meta_path: Path, warn_ms: float) -> list[tuple[int, float]]:
    """(end_index, gap_ms) where consecutive chunks arrived more than ``warn_ms`` apart."""
    if not meta_path.is_file():
        return []
    anchors = json.loads(meta_path.read_text(encoding="utf-8")).get("anchors") or []
    out = []
    for (_e0, t0), (e1, t1) in zip(anchors, anchors[1:]):
        gap = (t1 - t0) * 1000
        if gap > warn_ms:
            out.append((e1, round(gap, 1)))
    return out


def evaluate(session_dir: Path, timeouts: list[int], vad: Any = None) -> dict[str, Any]:
    cfg = load_config()
    items = json.loads((session_dir / "items.json").read_text(encoding="utf-8"))
    vad = vad if vad is not None else SileroVadModel()
    rows: list[dict[str, Any]] = []
    for item in items:
        wav = session_dir / "audio" / f"{item['item']}.wav"
        x, rate = read_wav(wav)
        if rate != SAMPLE_RATE:
            raise SystemExit(f"{wav}: {rate} Hz")
        probs = vad.probs(to_mono(x, cfg.audio.mic_channel))
        whole = segment_probs(probs, replace(cfg.vad, silence_timeout_ms=60_000, max_utterance_s=600))
        pauses = [round((b - a) * 1000 / SAMPLE_RATE) for s in whole for a, b in s.pauses]
        gaps = arrival_gaps(wav.with_suffix(".json"), cfg.audio.gap_warn_ms)
        for timeout in timeouts:
            segs = segment_probs(probs, replace(cfg.vad, silence_timeout_ms=timeout))
            last = segs[-1] if segs else None
            rows.append({
                "item": item["item"], "text": item["text"], "has_pause": item["has_pause"], "timeout_ms": timeout,
                "segments": len(segs), "premature": len(segs) > 1,
                "pauses_ms": " ".join(str(p) for p in pauses), "max_pause_ms": max(pauses) if pauses else 0,
                "speech_s": round(sum(s.end - s.start for s in segs) / SAMPLE_RATE, 2),
                "endpoint_delay_ms": round((last.endpoint - last.end) * 1000 / SAMPLE_RATE)
                if last is not None and last.closed else None,
                "closed": None if last is None else last.closed,
                "input_gaps": len(gaps), "input_gap_ms": round(sum(g for _e, g in gaps), 1),
            })
    summary: dict[str, Any] = {}
    for timeout in timeouts:
        sel = [r for r in rows if r["timeout_ms"] == timeout]
        paused = [r for r in sel if r["has_pause"]]
        plain = [r for r in sel if not r["has_pause"]]
        delays = [r["endpoint_delay_ms"] for r in sel if r["endpoint_delay_ms"] is not None]
        summary[str(timeout)] = {
            "premature_rate_paused": round(sum(r["premature"] for r in paused) / len(paused), 3) if paused else None,
            "premature_rate_plain": round(sum(r["premature"] for r in plain) / len(plain), 3) if plain else None,
            "premature_rate_all": round(sum(r["premature"] for r in sel) / len(sel), 3) if sel else None,
            "endpoint_delay_ms_median": round(float(np.median(delays))) if delays else None,
            "items_with_input_gaps": sum(1 for r in sel if r["input_gaps"]),
        }
    with (session_dir / "pause_eval.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result = {"session": session_dir.name, "timeouts": timeouts, "vad": {"threshold": cfg.vad.threshold,
              "neg_threshold": cfg.vad.neg_threshold, "min_pause_ms": cfg.vad.min_pause_ms},
              "summary": summary, "rows": rows}
    (session_dir / "pause_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", help="results/sessions/<id>_pause_record")
    parser.add_argument("--timeouts", default="800,1500,2000", help="silence timeouts to compare (ms)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    result = evaluate(Path(args.session_dir), [int(t) for t in args.timeouts.split(",")])
    for r in result["rows"]:
        print(f"{r['item']:<3} {r['timeout_ms']:>5} ms  segments {r['segments']}  pauses {r['pauses_ms'] or '-':<12} "
              f"endpoint +{r['endpoint_delay_ms']} ms  gaps {r['input_gaps']}  {r['text']}")
    print("\nsummary:", json.dumps(result["summary"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
