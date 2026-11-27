"""Replay the pause-script recordings through the real cascade, without the robot (Phase 4).

    python -m reachy_s2s.apps.pause_replay results/sessions/<id>_pause_record
    python -m reachy_s2s.apps.pause_replay <dir> --variant A0=config/local_e4b_supertonic.toml \
        --variant A1=config/local_e4b_supertonic.toml,config/a1.toml

Every line of ``apps.pause_record`` was meant as ONE utterance. For each variant and line the
recording is fed in real time to a fresh CascadedPipeline on the mock backend: local Silero /
faster-whisper / LLM server / TTS exactly as configured, robot output goes nowhere, no API.
(The LLM server must be running; ``--fake-engines`` rehearses the tool without models.)

Per (variant, line) one row in ``results/pause_replay.csv`` (``--fake-engines``:
``results/_mock_and_tests/pause_replay_mock.csv``):

* turns: how many turns the cascade made of the one intended utterance, and their statuses
  (A0 answers the first fragment and ignores the rest; A1 cancels it and merges);
* answered_text: what the LLM got for the answer that was finally played (merged in A1);
* speech_end_to_push_ms of that answer: the user's last word -> first robot audio push;
* premature-endpoint suspects and cancelled turns.

The replay sessions go to ``results/_mock_and_tests/`` (mock input, kept apart from robot runs).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..audio.wavio import read_wav
from ..config import AppConfig, load_config
from ..conversation.manager import CascadedPipeline
from ..conversation.state import InteractionState, StateTracker
from ..engines.factory import build_engines
from ..evaluation.session import Session, setup_console_logging
from ..evaluation.tables import append_csv_row
from ..robot.audio_manager import AudioManager
from ..robot.backends.mock import MockAudioBackend
from .common import prepare_runtime

DEFAULT_VARIANTS = (
    ("A0", ["config/local_e4b_supertonic.toml"]),
    ("A1", ["config/local_e4b_supertonic.toml", "config/a1.toml"]),
    ("A1-1500", ["config/local_e4b_supertonic.toml", "config/a1_pause1500.toml"]),
    ("A1-2000", ["config/local_e4b_supertonic.toml", "config/a1_pause2000.toml"]),
)
ENGINE_SECTIONS = ("asr", "llm", "tts")


def parse_variant(text: str) -> tuple[str, list[str]]:
    name, _, files = text.partition("=")
    if not name or not files:
        raise argparse.ArgumentTypeError("use NAME=config1.toml[,config2.toml]")
    return name, files.split(",")


def replay_line(cfg: AppConfig, engines: Any, wav: Path, results: Path, label: str,
                settle_s: float, max_s: float) -> tuple[CascadedPipeline, Session]:
    """Feed one recording in real time; return when the cascade is idle again (or after ``max_s``)."""
    x, rate = read_wav(wav)
    if rate != 16000:
        raise SystemExit(f"{wav}: {rate} Hz (robot recordings are 16 kHz)")
    session = Session(results, label, capture_logging=False)
    session.write_metadata(app="pause_replay", source=str(wav), config=cfg.to_dict())
    backend = MockAudioBackend(x, realtime=True, output_latency_hint_s=cfg.audio.output_latency_hint_ms / 1000)
    audio = AudioManager(backend, cfg.audio, session.events)
    state = StateTracker(session.events)
    pipeline = CascadedPipeline(audio, cfg, session, state, engines.asr, engines.llm, engines.tts)
    audio.start()
    pipeline.start()
    source_s = x.shape[0] / rate
    t0 = time.perf_counter()
    idle_since = None
    try:
        while time.perf_counter() - t0 < max_s:
            time.sleep(0.1)
            idle = (time.perf_counter() - t0 > source_s and state.state == InteractionState.LISTENING
                    and not pipeline.vad.speech_in_progress() and not audio.is_speaking())
            idle_since = (idle_since or time.perf_counter()) if idle else None
            if idle_since is not None and time.perf_counter() - idle_since >= settle_s:
                break
    finally:
        pipeline.stop()
        audio.stop()
        session.close()
    return pipeline, session


def summarize_line(pipeline: CascadedPipeline) -> dict[str, Any]:
    turns = pipeline.turns
    answered = [t for t in turns if t.queue_first_push is not None and t.status in ("done", "interrupted")]
    final = answered[-1] if answered else None
    return {
        "turns": len(turns),
        "statuses": " ".join(t.status for t in turns),
        "transcripts": " | ".join(t.transcript or "" for t in turns),
        "answers": len(answered),
        "answered_text": None if final is None else (final.merged_text or final.transcript),
        "response": None if final is None else final.response,
        "speech_end_to_push_ms": None if final is None else final.speech_end_to_push_ms,
        "vad_endpoint_delay_ms": None if final is None else final.vad_endpoint_delay_ms,
        "cancelled": sum(1 for t in turns if t.status == "cancelled"),
        "premature_suspects": sum(1 for t in turns if t.premature_suspect),
        "ignored_utterances": None,  # filled from the event log
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path, help="results/sessions/<id>_pause_record")
    parser.add_argument("--variant", action="append", type=parse_variant,
                        help="NAME=config.toml[,more.toml] (repeatable; default A0, A1, A1-1500, A1-2000)")
    parser.add_argument("--items", default="", help="comma-separated item ids (default: all)")
    parser.add_argument("--settle", type=float, default=1.5, help="idle seconds after the line before stopping")
    parser.add_argument("--fake-engines", action="store_true", help="stand-in engines (rehearse the tool)")
    parser.add_argument("--out", type=Path, help="CSV path instead of results/pause_replay.csv")
    args = parser.parse_args(argv)
    variants = args.variant or list(DEFAULT_VARIANTS)
    cfgs = [(name, load_config(files)) for name, files in variants]
    base = cfgs[0][1]
    for name, cfg in cfgs[1:]:  # one engine set for all variants: their engine settings must match
        diff = [s for s in ENGINE_SECTIONS if getattr(cfg, s) != getattr(base, s)]
        if diff:
            raise SystemExit(f"variant {name} changes {diff}; replay one engine set at a time")
    setup_console_logging("WARNING")
    prepare_runtime(base)
    items = json.loads((args.session_dir / "items.json").read_text(encoding="utf-8"))
    if args.items:
        wanted = set(args.items.split(","))
        items = [i for i in items if i["item"] in wanted]
    results = base.logging.results_path / "_mock_and_tests"
    engine_session = Session(results, "pause_replay_engines", capture_logging=False)
    print("[..] loading engines once for all variants ...")
    engines = build_engines(base, engine_session.events, fake=args.fake_engines)
    engine_session.write_metadata(engines=engines.info, source=str(args.session_dir))
    engine_session.close()
    out = args.out or (results / "pause_replay_mock.csv" if args.fake_engines
                       else base.logging.results_path / "pause_replay.csv")
    for name, cfg in cfgs:
        cfg = replace(cfg, cascaded=replace(cfg.cascaded, save_turn_audio=False))
        for item in items:
            wav = args.session_dir / "audio" / f"{item['item']}.wav"
            pipeline, session = replay_line(cfg, engines, wav, results, f"pause_replay_{name}_{item['item']}",
                                            settle_s=args.settle, max_s=60.0)
            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": args.session_dir.name,
                   "variant": cfg.cascaded.variant, "label": name, "fake_engines": args.fake_engines,
                   "silence_timeout_ms": cfg.vad.silence_timeout_ms, "barge_in": cfg.cascaded.barge_in,
                   "item": item["item"], "has_pause": item["has_pause"], "script": item["text"],
                   **summarize_line(pipeline), "replay_session": session.dir.name}
            events = [json.loads(line) for line in (session.dir / "events.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
            row["ignored_utterances"] = sum(1 for e in events if e["kind"] == "utterance.ignored")
            append_csv_row(out, row)
            print(f"{name:<8} {item['item']}  turns {row['turns']} [{row['statuses']}]  answers {row['answers']}  "
                  f"ignored {row['ignored_utterances']}  push {row['speech_end_to_push_ms']} ms\n"
                  f"         heard: {row['transcripts']}\n         LLM got: {row['answered_text']}")
    print(f"[..] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
