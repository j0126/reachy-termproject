"""Korean multi-turn dialog benchmark for the configured LLM (any provider).

    python -m reachy_s2s.apps.llm_bench --config config/llm_local_gemma4.toml --runs 3 --label E4B
    python -m reachy_s2s.apps.llm_bench --config config/llm_local_gemma4.toml --with-asr \
        --asr-wav results/sessions/<id>/audio/turn_0001_user.wav --label E4B+whisper

Runs a fixed 8-turn Korean script through the same code path as the
cascaded pipeline (system prompt, client-side history, retry policy) and
checks every reply automatically:

* facts: the name, the meeting date / time and the *corrected* head count
  must come back when asked (and the old count must not);
* style: sentence count, length, no markdown / emoji, Hangul share.

``--with-asr`` loads faster-whisper first (GPU memory as in the pipeline);
``--asr-wav`` also transcribes that file before every turn, so the GPU
alternates between ASR and LLM like in a real turn. One CSV row per turn goes
to ``results/llm_bench.csv``. Measurement only: no audio, no robot, and the
cloud TTS is not called.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import load_config
from ..conversation.history import ConversationHistory, Exchange
from ..engines.asr import gpu_memory_used_mb
from ..engines.base import EngineError
from ..engines.factory import make_llm
from ..evaluation.session import setup_console_logging
from ..evaluation.tables import append_csv_row
from .common import prepare_runtime


@dataclass(frozen=True)
class Turn:
    text: str
    expect: tuple[tuple[str, ...], ...] = ()  # every group must match (any alternative), spaces ignored
    forbid: tuple[str, ...] = ()
    expect_re: tuple[str, ...] = ()  # regexes that must match the reply with spaces removed
    note: str = ""

    @property
    def checks_facts(self) -> bool:
        return bool(self.expect or self.forbid or self.expect_re)


SCRIPT: tuple[Turn, ...] = (
    Turn("안녕, 내 이름은 준수야.", expect=(("준수",),), note="name"),
    Turn("다음 주 회의는 11월 13일 오후 3시 20분이고, 14명이 참석해. 기억해 줘.", note="facts in"),
    Turn("아, 참석 인원이 16명으로 바뀌었어.", expect=(("16명", "열여섯명"),), note="correction"),
    Turn("회의 날짜랑 시간, 인원을 다시 말해 줘.",
         expect=(("11월13일",), ("3시20분", "15시20분", "세시이십분"), ("16명", "열여섯명")),
         forbid=("14명",), note="recall after correction"),
    Turn("내 이름이 뭐라고 했지?", expect=(("준수",),), note="name recall"),
    Turn("오늘 좀 피곤하네.", note="empathy"),
    # Reachy Mini has no wheels / arms: claiming to move with them is a hallucination ("바퀴나 팔은 없어요" is fine)
    Turn("너는 어떻게 움직여? 짧게 설명해 줘.", forbid=("바퀴로", "바퀴를", "바퀴가있", "팔로", "팔을움직"),
         note="self description"),
    Turn("고개 한번 끄덕여 줄래?",
         expect_re=(r"(움직|끄덕)\S{0,12}(없|못|어렵|어려|않|꺼져)|못\S{0,3}(움직|끄덕)|아직",),
         note="motion request (Phase 2: speech only)"),
    Turn("고마워, 다음에 또 이야기하자.", note="closing"),
)

MARKDOWN = re.compile(r"(\*\*|__|`|^\s*#|^\s*[-*•]\s|^\s*\d+\.\s)", re.M)
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")
SENTENCE_END = re.compile(r"[.!?。！？]+(?=\s|$)")
HANGUL = re.compile(r"[가-힣]")
LETTER = re.compile(r"[^\W\d_]")


@dataclass
class Checks:
    failed: list[str] = field(default_factory=list)
    sentences: int = 0
    chars: int = 0
    hangul_ratio: float = 0.0


def system_prompt_id(prompt: str) -> str:
    """Short, stable id of the system prompt, so result rows say which prompt produced them."""
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()[:8]


def check_reply(turn: Turn, reply: str) -> Checks:
    c = Checks(chars=len(reply))
    compact = re.sub(r"\s+", "", reply)
    for group in turn.expect:
        if not any(alt in compact for alt in group):
            c.failed.append(f"missing:{group[0]}")
    for bad in turn.forbid:
        if bad in compact:
            c.failed.append(f"forbidden:{bad}")
    for pattern in turn.expect_re:
        if not re.search(pattern, compact):
            c.failed.append(f"missing:/{pattern}/")
    c.sentences = max(1, len(SENTENCE_END.findall(reply.strip()))) if reply.strip() else 0
    if c.sentences > 3:
        c.failed.append("style:>3 sentences")
    if MARKDOWN.search(reply):
        c.failed.append("style:markdown")
    if EMOJI.search(reply):
        c.failed.append("style:emoji")
    letters = LETTER.findall(reply)
    c.hangul_ratio = round(len(HANGUL.findall(reply)) / len(letters), 2) if letters else 0.0
    if c.hangul_ratio < 0.5:
        c.failed.append("style:not Korean")
    return c


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append",
                        help="TOML file overriding config/default.toml (repeatable, applied in order)")
    parser.add_argument("--runs", type=int, default=3, help="how many times the whole dialog is run")
    parser.add_argument("--label", default="", help="free text stored in the CSV (e.g. E4B+whisper)")
    parser.add_argument("--with-asr", action="store_true", help="load faster-whisper first (shared GPU)")
    parser.add_argument("--asr-wav", help="with --with-asr: transcribe this WAV before every turn")
    parser.add_argument("--no-warmup", action="store_true", help="skip the warm-up request")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    setup_console_logging("WARNING")
    prepare_runtime(cfg)
    if cfg.llm.provider == "gemini":
        print("[!!] provider gemini: every turn is one request against the free-tier daily limit "
              f"({args.runs * len(SCRIPT)} requests).")

    llm = make_llm(cfg.llm)
    server = llm.check_server() if hasattr(llm, "check_server") else {}
    asr, wav = None, None
    if args.with_asr:
        from ..audio.convert import to_mono
        from ..audio.wavio import read_wav
        from ..engines.asr import WhisperASR

        asr = WhisperASR(cfg.asr)
        info = asr.load()
        print(f"[..] ASR {cfg.asr.model} {cfg.asr.compute_type}: load {info.get('load_ms')} ms, "
              f"GPU memory {info.get('gpu_mem_before_mb')} -> {info.get('gpu_mem_after_mb')} MiB")
        if args.asr_wav:
            samples, rate = read_wav(args.asr_wav)
            if rate != 16000:
                raise SystemExit("--asr-wav must be 16 kHz (the robot's rate)")
            wav = to_mono(samples, "left")
    if not args.no_warmup and hasattr(llm, "warmup"):
        print(f"[..] warm-up: {llm.warmup()}")
    gpu_mem = gpu_memory_used_mb()
    print(f"[..] {cfg.llm.provider} {cfg.llm.model} at {getattr(llm, 'base_url', '')} "
          f"server={server.get('llama_cpp', server.get('models'))} GPU memory now {gpu_mem} MiB")

    out = cfg.logging.results_path / "llm_bench.csv"
    prompt_id = system_prompt_id(cfg.llm.system_prompt)
    print(f"[..] system prompt id {prompt_id} (sha256 of the stripped [llm] system_prompt)")
    rows: list[dict] = []
    for run in range(1, args.runs + 1):
        history = ConversationHistory(cfg.llm.max_history_exchanges)
        for i, turn in enumerate(SCRIPT, start=1):
            asr_ms = None
            if asr is not None and wav is not None:
                asr_ms = round(asr.transcribe(wav, 16000).duration_ms, 1)
            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "label": args.label, "provider": cfg.llm.provider,
                   "model": cfg.llm.model, "prompt_id": prompt_id, "run": run, "turn": i, "note": turn.note,
                   "ok": False, "ms": None,
                   "error_kind": "", "status_code": "", "input_tokens": None, "output_tokens": None,
                   "cache_n": None, "prompt_ms": None, "predicted_ms": None, "tok_s": None, "checks_failed": "",
                   "sentences": None, "chars": None, "hangul_ratio": None, "asr_ms": asr_ms,
                   "gpu_mem_mb": gpu_mem, "user": turn.text, "reply": ""}
            t0 = time.perf_counter()
            try:
                res = llm.respond(history, turn.text)
            except EngineError as err:
                row.update(ms=round((time.perf_counter() - t0) * 1000), error_kind=err.kind,
                           status_code=err.status_code or "")
            else:
                history.add(Exchange(i, turn.text, res.text, res.user_step, res.model_steps))
                timings = res.usage.get("server_timings") or {}
                checks = check_reply(turn, res.text)
                row.update(ok=True, ms=round(res.duration_ms), input_tokens=res.usage.get("total_input_tokens"),
                           output_tokens=res.usage.get("total_output_tokens"), cache_n=timings.get("cache_n"),
                           prompt_ms=timings.get("prompt_ms"), predicted_ms=timings.get("predicted_ms"),
                           tok_s=timings.get("predicted_per_second"), checks_failed=";".join(checks.failed),
                           sentences=checks.sentences, chars=checks.chars, hangul_ratio=checks.hangul_ratio,
                           reply=res.text)
            rows.append(row)
            append_csv_row(out, row)
            flag = "OK " if row["ok"] else "ERR"
            print(f"run {run} turn {i} {flag} {row['ms']:>6} ms  {row['checks_failed'] or row['error_kind'] or '-'}\n"
                  f"    U: {turn.text}\n    R: {row['reply']}")

    ok = [r for r in rows if r["ok"]]
    print(f"\nsummary  {cfg.llm.model}  label={args.label!r}  runs={args.runs}")
    if ok:
        ms = np.array([r["ms"] for r in ok], dtype=float)
        print(f"  latency  median {np.median(ms):.0f} ms  p90 {np.percentile(ms, 90):.0f}  max {ms.max():.0f}  "
              f"(n={len(ms)}, errors {len(rows) - len(ok)})")
        tok = [r["tok_s"] for r in ok if r["tok_s"]]
        if tok:
            print(f"  decode   median {np.median(tok):.1f} tok/s; output tokens median "
                  f"{np.median([r['output_tokens'] or 0 for r in ok]):.0f}")
        fact_turns = [r for r in ok if SCRIPT[r["turn"] - 1].checks_facts]
        fact_pass = sum(1 for r in fact_turns if not any(f.startswith(("missing", "forbidden"))
                                                          for f in r["checks_failed"].split(";") if f))
        style_bad = sum(1 for r in ok if any(f.startswith("style") for f in r["checks_failed"].split(";") if f))
        print(f"  facts    {fact_pass}/{len(fact_turns)} fact turns correct; style violations in "
              f"{style_bad}/{len(ok)} replies")
    print(f"[..] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
