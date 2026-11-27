"""Check how the configured LLM reacts to an interrupted answer in its history (Phase 4, A1).

    python -m reachy_s2s.apps.history_probe --config config/local_e4b_supertonic.toml --runs 3

A1 stores an interrupted answer as "the part the user heard + a marker"
(``[cascaded] interrupted_history = "heard_marked"``, the marker is
``conversation.manager.HEARD_MARK``). This replays short scripted dialogs with
such a history through the same code path as the pipeline (system prompt,
client-side history) and checks every reply:

* the reply must not repeat the marker text (the model imitating our bookkeeping);
* the reply must use the user's correction (e.g. 14명, not 16명);
* "what were you saying?" should refer to the cut-off topic.

No audio, no robot. One CSV row per reply goes to ``results/history_probe.csv``.
Use a local LLM (llama-server) - with provider gemini every reply is one request
against the free-tier daily limit.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass

from ..config import load_config
from ..conversation.history import ConversationHistory, Exchange
from ..conversation.manager import FULL_MARK, HEARD_MARK
from ..engines.base import EngineError
from ..engines.factory import make_llm
from ..engines.llm import user_step
from ..evaluation.session import setup_console_logging
from ..evaluation.tables import append_csv_row
from .common import prepare_runtime
from .llm_bench import system_prompt_id

MARKER_WORDS = re.compile(r"끼어들|여기서\s*멈춤|끝까지\s*듣지")


@dataclass(frozen=True)
class Probe:
    name: str
    history: tuple[tuple[str, str], ...]  # (user, assistant as stored by A1)
    question: str
    expect: tuple[str, ...] = ()  # any of these (spaces ignored) must appear
    forbid: tuple[str, ...] = ()


def probes() -> tuple[Probe, ...]:
    return (
        Probe("correction during speech",
              (("회의는 3시고 16명이 와.", "네, 회의는 3시에 16명이" + HEARD_MARK),
               ("잠깐, 16명이 아니라 14명이야.", "알겠어요, 참석자는 14명으로 기억할게요.")),
              "회의에 몇 명 온다고 했지?", expect=("14명", "열네명"), forbid=("16명",)),
        Probe("correction right after the cut",
              (("회의는 3시고 16명이 와.", "네, 회의는 3시에 16명이" + HEARD_MARK),),
              "잠깐, 16명이 아니라 14명이야. 그럼 몇 명이지?", expect=("14명", "열네명"), forbid=("16명",)),
        Probe("what were you saying",
              (("오늘 날씨 어때?", "오늘은 맑고 기온은" + HEARD_MARK),),
              "방금 무슨 말 하다가 멈췄어?", expect=("날씨", "기온", "맑")),
        Probe("topic change after the cut",
              (("주말에 뭐 하면 좋을까?", "날씨가 좋으면 공원 산책도 좋고," + HEARD_MARK),),
              "아니 그거 말고, 내 이름 기억해? 나는 준수야.", expect=("준수",)),
        Probe("full_marked",
              (("오늘 날씨 어때?", "오늘은 맑고 기온은 20도 정도예요. 산책하기 좋아요." + FULL_MARK),),
              "방금 뭐라고 했어? 다시 짧게 말해 줘.", expect=("맑", "20도", "이십도")),
    )


def check(probe: Probe, reply: str) -> list[str]:
    failed = []
    compact = re.sub(r"\s+", "", reply)
    if MARKER_WORDS.search(reply):
        failed.append("imitates marker")
    if probe.expect and not any(e in compact for e in probe.expect):
        failed.append(f"missing:{probe.expect[0]}")
    for bad in probe.forbid:
        if bad in compact:
            failed.append(f"forbidden:{bad}")
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append",
                        help="TOML file overriding config/default.toml (repeatable, applied in order)")
    parser.add_argument("--runs", type=int, default=3, help="replies per probe (sampling temperature applies)")
    parser.add_argument("--label", default="", help="free text stored in the CSV")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    setup_console_logging("WARNING")
    prepare_runtime(cfg)
    if cfg.llm.provider == "gemini":
        print(f"[!!] provider gemini: {args.runs * len(probes())} requests against the free-tier daily limit.")
    llm = make_llm(cfg.llm)
    if hasattr(llm, "check_server"):
        llm.check_server()
    out = cfg.logging.results_path / "history_probe.csv"
    prompt_id = system_prompt_id(cfg.llm.system_prompt)
    total = bad = 0
    for probe in probes():
        for run in range(1, args.runs + 1):
            history = ConversationHistory(cfg.llm.max_history_exchanges)
            for i, (user, assistant) in enumerate(probe.history, start=1):
                history.add(Exchange(i, user, assistant, user_step(user),
                                     [{"type": "model_output", "content": [{"type": "text", "text": assistant}]}]))
            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "label": args.label, "provider": cfg.llm.provider,
                   "model": cfg.llm.model, "prompt_id": prompt_id, "marker": HEARD_MARK, "probe": probe.name,
                   "run": run, "ok": False, "ms": None, "failed": "", "question": probe.question, "reply": ""}
            t0 = time.perf_counter()
            try:
                res = llm.respond(history, probe.question)
            except EngineError as err:
                row.update(ms=round((time.perf_counter() - t0) * 1000), failed=f"error:{err.kind}")
            else:
                failed = check(probe, res.text)
                row.update(ok=not failed, ms=round(res.duration_ms), failed=";".join(failed), reply=res.text)
            append_csv_row(out, row)
            total += 1
            bad += 0 if row["ok"] else 1
            print(f"{probe.name:<32} run {run}  {'OK ' if row['ok'] else 'BAD'} {row['ms']:>5} ms  "
                  f"{row['failed'] or '-'}\n    Q: {probe.question}\n    R: {row['reply']}")
    print(f"\n{total - bad}/{total} replies passed ({cfg.llm.model}, prompt {prompt_id}). [..] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
