"""Basic cascaded speech-to-speech on the robot, without GUI.

    python -m reachy_s2s.apps.cascaded                       # robot + Silero + faster-whisper + Gemini
    python -m reachy_s2s.apps.cascaded --fake-engines --mock speech.wav   # dry run: no GPU / API / robot

Type a command + Enter while it runs:
    r  reset the conversation history      n  start a new session (new results folder)
    s  save the conversation log now       q  quit

Per session: results/sessions/<id>_cascaded/{events.jsonl, turns.csv,
conversation.json, audio/turn_NNNN_user.wav, audio/turn_NNNN_robot.wav, ...}.
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
from typing import Any

from ..config import load_config
from ..conversation.manager import CascadedPipeline
from ..conversation.state import StateTracker
from ..engines.base import EngineError
from ..engines.factory import build_engines
from ..evaluation.session import Session, setup_console_logging
from ..robot.audio_manager import AudioManager
from .common import add_common_args, open_backend, prepare_runtime

log = logging.getLogger("cascaded")


def _print_turn(kind: str, payload: dict[str, Any]) -> None:
    if kind != "turn" or payload.get("status") == "processing":
        return
    t = payload
    print(f"\n[turn {t['turn_id']}] {t['status']} ({t.get('variant')})"
          + (f"  ({t['error_stage']}: {t['error_kind']} {t['error_status'] or ''})" if t.get("error_stage") else ""))
    print(f"  사용자: {t.get('transcript')}")
    if t.get("merged_text") is not None:
        print(f"  합친 입력(turn {t.get('merged_from')} + 이번): {t['merged_text']}")
    print(f"  로봇  : {t.get('response')}")
    if t.get("interrupted"):
        print(f"  끼어들기: {t.get('interrupt_kind')} / {t.get('interrupt_stage')}, 확정 지연 "
              f"{t.get('bargein_detect_delay_ms')} ms, 재생 중단 {t.get('playback_stop_delay_ms')} ms, "
              f"들린 비율(추정) {t.get('heard_ratio_est')} -> 이력 {t.get('history_policy')}")
    if t.get("premature_suspect"):
        print(f"  조기 종료 의심: 종료 판정 {t.get('resume_after_endpoint_ms')} ms 뒤 다시 말함")
    print(f"  VAD endpoint delay {t.get('vad_endpoint_delay_ms')} ms | ASR {t.get('asr_ms')} ms | "
          f"LLM {t.get('llm_ms')} ms | TTS {t.get('tts_ms')} ms | "
          f"speech end -> robot push {t.get('speech_end_to_push_ms')} ms "
          f"(+{t.get('output_hint_ms')} ms assumed output = {t.get('response_latency_est_ms')} ms est.)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--fake-engines", action="store_true", help="stand-in ASR/LLM/TTS (no GPU, no API)")
    parser.add_argument("--echo-llm", action="store_true",
                        help="real VAD/ASR/TTS but the LLM is replaced by a deterministic echo (hardware tests)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides={"robot": {"host": args.host}})
    setup_console_logging(cfg.logging.level)
    runtime = prepare_runtime(cfg)

    session = Session(cfg.logging.results_path, "cascaded")
    conn = audio = pipeline = None
    try:
        print("[..] loading engines (ASR model load + warm-up) ...")
        engines = build_engines(cfg, session.events, fake=args.fake_engines, echo_llm=args.echo_llm)
        print(f"[OK] engines: {engines.info}")
        backend, conn, robot_info = open_backend(cfg, args.mock)
        audio = AudioManager(backend, cfg.audio, session.events)
        audio.start()

        def new_pipeline(sess: Session) -> CascadedPipeline:
            sess.write_metadata(app="cascaded", args=vars(args), config=cfg.to_dict(), robot=robot_info,
                                backend=backend.describe(), runtime=runtime, engines=engines.info)
            state = StateTracker(sess.events)
            p = CascadedPipeline(audio, cfg, sess, state, engines.asr, engines.llm, engines.tts)
            p.add_listener(_print_turn)
            return p

        pipeline = new_pipeline(session)
        pipeline.start()
        print("[OK] listening - speak to the robot.  Commands: r=reset  n=new session  s=save  q=quit")

        commands: queue.Queue[str] = queue.Queue()
        threading.Thread(target=lambda: [commands.put(line.strip()) for line in sys.stdin], daemon=True).start()
        while True:
            try:
                cmd = commands.get(timeout=0.5)
            except queue.Empty:
                continue
            if cmd == "q":
                break
            if cmd == "r":
                print(f"[..] history reset ({pipeline.reset_history()} exchanges dropped)")
            elif cmd == "s":
                pipeline.save_conversation()
                print(f"[OK] saved {session.path('conversation.json')}")
            elif cmd == "n":
                pipeline.stop()
                session.close()
                session = Session(cfg.logging.results_path, "cascaded")
                audio.set_event_logger(session.events)
                pipeline = new_pipeline(session)
                pipeline.start()
                print(f"[OK] new session {session.dir}")
    except EngineError as exc:
        print(f"\n[FAIL] {exc.stage} {exc.kind}: {exc}", file=sys.stderr)
        session.events.log("app.error", **exc.to_dict())
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        if pipeline is not None:
            pipeline.stop()
        if audio is not None:
            audio.stop()
        if conn is not None:
            conn.close()
        session.close()
        print(f"[..] session: {session.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
