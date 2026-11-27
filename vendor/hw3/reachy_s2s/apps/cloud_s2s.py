"""Cloud speech-to-speech (System B, Gemini Live) on the robot, without GUI.

    python -m reachy_s2s.apps.cloud_s2s                           # robot + Gemini Live (GEMINI_API_KEY)
    python -m reachy_s2s.apps.cloud_s2s --fake-cloud --mock speech.wav   # dry run: fake Live server, no robot

Type a command + Enter while it runs:
    r  reset the conversation (new Live session, server context discarded)
    c  reconnect, resuming the Live session (context kept)
    n  start a new session (new results folder)      s  save the log now      q  quit

Per session: results/sessions/<id>_cloud/{events.jsonl, turns.csv, conversation.json,
audio/turn_NNNN_user.wav (local VAD segment), audio/turn_NNNN_robot.wav (received audio)}.
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
from typing import Any

from ..config import load_config
from ..conversation.cloud_pipeline import CloudS2SPipeline
from ..conversation.state import StateTracker
from ..engines.gemini_client import api_key_present
from ..evaluation.session import Session, setup_console_logging
from ..robot.audio_manager import AudioManager
from .common import add_common_args, open_backend, prepare_runtime

log = logging.getLogger("cloud_s2s")


def _print(kind: str, payload: dict[str, Any]) -> None:
    if kind == "status":
        extra = {k: v for k, v in payload.items() if k in ("attempt", "failures", "resumed")}
        print(f"[cloud] {payload['status']} (connection {payload.get('connection_id')}) {extra or ''}")
        return
    if kind != "turn" or payload.get("status") == "responding":
        return
    t = payload
    print(f"\n[turn {t['turn_id']}] {t['status']}" + (" (interrupted)" if t.get("interrupted") else ""))
    print(f"  사용자: {t.get('transcript')}")
    print(f"  로봇  : {t.get('response')}")
    print(f"  speech end -> cloud audio {t.get('speech_end_to_cloud_audio_ms')} ms | uplink -> cloud audio "
          f"{t.get('uplink_to_cloud_audio_ms')} ms | speech end -> robot push {t.get('speech_end_to_push_ms')} ms "
          f"(+{t.get('output_hint_ms')} ms assumed output = {t.get('response_latency_est_ms')} ms est.)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--fake-cloud", action="store_true", help="scripted fake Live server (no network, no key)")
    parser.add_argument("--voice", help="override [cloud] voice")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides={"robot": {"host": args.host}, "cloud": {"voice": args.voice}})
    setup_console_logging(cfg.logging.level)
    runtime = prepare_runtime(cfg)
    if not args.fake_cloud and not api_key_present():
        print("[FAIL] GEMINI_API_KEY is not set (.env)", file=sys.stderr)
        return 1

    def connector() -> Any:
        if args.fake_cloud:
            from ..engines.fake_live import FakeLiveConnector

            return FakeLiveConnector()
        from ..engines.gemini_live import GeminiLiveConnector

        return GeminiLiveConnector()

    session = Session(cfg.logging.results_path, "cloud")
    conn = audio = pipeline = None
    try:
        backend, conn, robot_info = open_backend(cfg, args.mock)
        audio = AudioManager(backend, cfg.audio, session.events)
        audio.start()

        def new_pipeline(sess: Session) -> CloudS2SPipeline:
            sess.write_metadata(app="cloud_s2s", args=vars(args), config=cfg.to_dict(), robot=robot_info,
                                backend=backend.describe(), runtime=runtime)
            p = CloudS2SPipeline(audio, cfg, sess, StateTracker(sess.events), connector=connector())
            p.add_listener(_print)
            return p

        pipeline = new_pipeline(session)
        pipeline.start()
        print("[OK] streaming to the cloud - speak to the robot.  Commands: r=reset  n=new session  s=save  q=quit")
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
                print(f"[..] conversation reset ({pipeline.reset_history()} exchanges dropped, new Live session)")
            elif cmd == "c":
                pipeline.reconnect()
                print("[..] reconnecting with the session handle (context kept)")
            elif cmd == "s":
                pipeline.save_conversation()
                print(f"[OK] saved {session.path('conversation.json')}")
            elif cmd == "n":
                pipeline.stop()
                session.close()
                session = Session(cfg.logging.results_path, "cloud")
                audio.set_event_logger(session.events)
                pipeline = new_pipeline(session)
                pipeline.start()
                print(f"[OK] new session {session.dir}")
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
