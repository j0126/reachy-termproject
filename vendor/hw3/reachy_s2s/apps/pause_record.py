"""Record scripted utterances from the robot mic for the pause-handling comparison (Phase 4).

    python -m reachy_s2s.apps.pause_record [--seconds 7]
    python -m reachy_s2s.apps.pause_eval results/sessions/<id>_pause_record

Ten lines, each meant as ONE utterance: five without pauses and five with a pause where
"…" is shown (the hint says how long). The robot stays silent; no API is used. For every
line: countdown, then ``--seconds`` of recording. Saved in the session folder:
audio/<item>.wav (+ .json chunk timing, used to find Wi-Fi gaps) and items.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from ..config import load_config
from ..evaluation.session import Session, setup_console_logging
from ..robot.audio_manager import AudioManager
from .common import add_common_args, open_backend, prepare_runtime

# (id, line, pause hint or "")
ITEMS = [
    ("c1", "오늘 날씨가 어때?", ""),
    ("c2", "내 이름은 준수야.", ""),
    ("c3", "회의는 11월 13일 오후 3시야.", ""),
    ("c4", "너는 어떤 로봇이야?", ""),
    ("c5", "다음 주에 같이 산책 가고 싶어.", ""),
    ("p1", "내 이름은 … 준수야.", "… 에서 1초 쉬기"),
    ("p2", "회의는 … 11월 13일이야.", "… 에서 1초 쉬기"),
    ("p3", "참석자는 14명 … 아니 16명이야.", "… 에서 1.5초 쉬기"),
    ("p4", "오늘은 … 기분이 어때?", "… 에서 1.5초 쉬기"),
    ("p5", "음 … 날씨가 좋으면 … 산책 가고 싶어.", "… 마다 1초씩 쉬기"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--seconds", type=float, default=7.0, help="recording length per line")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides={"robot": {"host": args.host}})
    setup_console_logging(cfg.logging.level)
    runtime = prepare_runtime(cfg)
    session = Session(cfg.logging.results_path, "pause_record")
    conn = audio = None
    items = []
    try:
        backend, conn, robot_info = open_backend(cfg, args.mock)
        session.write_metadata(app="pause_record", args=vars(args), config=cfg.to_dict(), robot=robot_info,
                               runtime=runtime)
        audio = AudioManager(backend, cfg.audio, session.events)
        audio.start()
        print("\n각 문장을 한 번에 말해 주세요. '…' 표시에서는 안내한 만큼 쉬었다가 이어서 말합니다.")
        for item_id, line, hint in ITEMS:
            print(f"\n>>> [{item_id}] {line}" + (f"   ({hint})" if hint else ""))
            for k in (3, 2, 1):
                print(f"    {k} ...", flush=True)
                time.sleep(1.0)
            print("    GO", flush=True)
            session.events.log("pause_record.item", item=item_id, text=line, pause_hint=hint)
            audio.start_recording(item_id)
            time.sleep(args.seconds)
            rec = audio.stop_recording()
            rec.save(session.audio_dir, session.t0_ns)
            gaps, gap_ms = audio.gaps_between(rec.start_index, rec.start_index + rec.num_frames)
            items.append({"item": item_id, "text": line, "has_pause": bool(hint), "pause_hint": hint,
                          "seconds": round(rec.duration_s, 2), "input_gaps": gaps, "input_gap_ms": gap_ms})
            print(f"    saved ({rec.duration_s:.1f} s, input gaps {gaps})")
    except KeyboardInterrupt:
        print("\n[..] stopped")
    finally:
        if audio is not None:
            audio.stop()
        if conn is not None:
            conn.close()
        (session.dir / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        session.close()
        print(f"[..] {len(items)} lines in {session.dir}\n     next: python -m reachy_s2s.apps.pause_eval {session.dir}")
    return 0 if items else 1


if __name__ == "__main__":
    sys.exit(main())
