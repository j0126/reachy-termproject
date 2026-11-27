"""Measure availability and latency of candidate LLMs on the Gemini API (same key).

    python -m reachy_s2s.apps.llm_probe --list
    python -m reachy_s2s.apps.llm_probe --models gemini-3.8-flash,gemini-3.6-flash --repeats 4

Models are called round-robin (one request each per round, ``--interval``
seconds apart) so all candidates see the same time-of-day load. No retries:
every 503 / 429 / timeout counts. Uses the project's LLM wrapper and system
prompt; thinking is "low" for gemini-3* models and omitted otherwise (Gemma 4
and 2.5 models reject thinking_level). Writes results/llm_probe.csv (one row per
request) and prints a per-model summary. Measurement only - the configured
model is not changed.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import numpy as np

from ..config import load_config
from ..conversation.history import ConversationHistory
from ..engines.base import EngineError
from ..engines.gemini_client import make_client
from ..engines.llm import GeminiLLM
from ..evaluation.session import setup_console_logging
from ..evaluation.tables import append_csv_row
from .common import prepare_runtime

PROMPT = "Reachy Mini가 뭐야? 한 문장으로 답해 줘."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append",
                        help="TOML file overriding config/default.toml (repeatable, applied in order)")
    parser.add_argument("--list", action="store_true", help="list text models available to this key")
    parser.add_argument("--models", default="gemini-3.8-flash", help="comma-separated model ids")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--interval", type=float, default=8.0, help="seconds between requests")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    setup_console_logging("WARNING")
    prepare_runtime(cfg)

    if args.list:
        client = make_client()  # keep a reference: the pager fetches lazily and a GC'd client is closed
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            name = getattr(m, "name", "")
            if "generateContent" in actions and "tts" not in name and "image" not in name:
                print(name.removeprefix("models/"))
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    llms = {m: GeminiLLM(replace(cfg.llm, model=m, max_retries=0, timeout_s=args.timeout,
                                 thinking_level="low" if m.startswith("gemini-3") else "")) for m in models}
    stats: dict[str, list] = {m: [] for m in models}
    out = cfg.logging.results_path / "llm_probe.csv"
    for r in range(args.repeats):
        for m in models:
            t_wall = time.strftime("%Y-%m-%d %H:%M:%S")
            t0 = time.perf_counter()
            row = {"time": t_wall, "model": m, "round": r + 1, "ok": False, "ms": None, "error_kind": "",
                   "status": "", "thought_tokens": "", "output_tokens": "", "reply": ""}
            try:
                res = llms[m].respond(ConversationHistory(), PROMPT)
                row.update(ok=True, ms=round(res.duration_ms), thought_tokens=res.usage.get("total_thought_tokens"),
                           output_tokens=res.usage.get("total_output_tokens"), reply=res.text[:120])
            except EngineError as e:
                row.update(ms=round((time.perf_counter() - t0) * 1000), error_kind=e.kind, status=e.status_code)
            stats[m].append(row)
            append_csv_row(out, row)
            print(f"{t_wall} {m:<28} {'OK ' if row['ok'] else 'ERR'} {row['ms']:>6} ms "
                  f"{row['error_kind']} {row['status']} {row['reply']}")
            time.sleep(args.interval)
    print("\nsummary (no retries):")
    for m, rows in stats.items():
        ok = [x["ms"] for x in rows if x["ok"]]
        errs = {}
        for x in rows:
            if not x["ok"]:
                key = f"{x['error_kind']} {x['status']}".strip()
                errs[key] = errs.get(key, 0) + 1
        med = f"{np.median(ok):.0f}" if ok else "-"
        print(f"  {m:<28} success {len(ok)}/{len(rows)}  median {med} ms  "
              f"min {min(ok) if ok else '-'}  max {max(ok) if ok else '-'}  errors {errs}")
    print(f"[..] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
