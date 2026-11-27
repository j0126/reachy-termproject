"""Phase 1 check without GUI: Reachy Mini mic -> record -> save -> play back on its speaker.

    python -m reachy_s2s.apps.audio_check --info              # daemon + mic DSP settings only (no WebRTC)
    python -m reachy_s2s.apps.audio_check --seconds 5         # record 5 s, save, play it back
    python -m reachy_s2s.apps.audio_check --seconds 5 --tone  # play a test tone first
    python -m reachy_s2s.apps.audio_check --speaker-test --seconds 0   # listening test only
    python -m reachy_s2s.apps.audio_check --no-play
    python -m reachy_s2s.apps.audio_check --mock some.wav     # same pipeline, no robot

The playback of the recording is peak-normalized (-3 dBFS, max +30 dB) because
robot recordings of a quiet room peak near -30 dBFS and are inaudible as-is;
``--no-normalize`` replays the raw samples.

``--speaker-test`` plays, while you listen next to the robot: (1) the daemon's
built-in test sound (REST, no WebRTC), (2) a quiet and (3) a loud 440 Hz tone
through our WebRTC playback path. It also reads XVF3800 AEC_AECCONVERGED
before/after (it turns 1 once the echo canceller has seen speaker output).

Outputs: results/sessions/<id>_audio_check/{session.json, events.jsonl, app.log,
audio/mic.wav, audio/mic.json, summary.json} and one row in results/audio_check.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any

from ..audio.convert import normalize_peak
from ..audio.signals import tone
from ..config import load_config
from ..evaluation.audio_stats import recording_stats
from ..evaluation.session import Session, setup_console_logging
from ..robot.audio_manager import AudioManager
from ..robot.connection import DaemonClient, ReachyConnection
from .common import add_common_args, append_csv_row, open_backend, prepare_runtime

log = logging.getLogger("audio_check")


def _meter(db: float, width: int = 30, lo: float = -80.0, hi: float = 0.0) -> str:
    filled = int(round(width * min(1.0, max(0.0, (db - lo) / (hi - lo)))))
    return "#" * filled + "-" * (width - filled)


def _aec_converged(conn: ReachyConnection | None) -> Any:
    if conn is None:
        return None
    try:
        return conn.daemon.get("/api/audio/config/parameter/AEC_AECCONVERGED").get("values")
    except Exception as exc:  # informative only
        return f"unavailable: {exc}"


def _speaker_test(audio: AudioManager, conn: ReachyConnection | None, session: Session, rate: int) -> dict:
    result: dict[str, Any] = {"aec_converged_before": _aec_converged(conn)}
    print("[..] SPEAKER TEST - listen next to the robot:")
    if conn is not None:
        print("     (1) daemon built-in test sound (REST, not WebRTC)")
        session.events.log("speaker_test.rest_sound")
        try:
            result["rest_sound"] = conn.daemon.post("/api/volume/test-sound")
        except Exception as exc:
            result["rest_sound"] = f"failed: {exc}"
        time.sleep(4.0)
    for label, amp in (("speaker_test_quiet_-16dBFS", 0.15), ("speaker_test_loud_-4dBFS", 0.6)):
        print(f"     ({'2' if 'quiet' in label else '3'}) WebRTC 440 Hz tone, {label.split('_')[-1]}")
        audio.play(tone(440.0, 1.5, rate, amplitude=amp), rate, label=label).wait(timeout=10)
        time.sleep(2.0)
    result["aec_converged_after"] = _aec_converged(conn)
    print(f"     AEC_AECCONVERGED {result['aec_converged_before']} -> {result['aec_converged_after']}"
          "  (1 = echo canceller saw speaker output)")
    print("     Write down which of (1)/(2)/(3) you heard; the session events mark each one.")
    return result


def _record_and_play(args: argparse.Namespace, audio: AudioManager, session: Session, summary: dict,
                     rate: int) -> None:
    print(f"[..] recording {args.seconds:.1f} s - speak to the robot now")
    audio.start_recording("mic")
    t_end = time.perf_counter() + args.seconds
    while time.perf_counter() < t_end:
        db = audio.level_dbfs(0.1)
        print(f"\r     level {_meter(db)} {db:6.1f} dBFS", end="", flush=True)
        time.sleep(0.1)
    print()
    rec = audio.stop_recording()
    paths = rec.save(session.audio_dir, session.t0_ns)
    stats = recording_stats(rec.frames, rec.sample_rate)
    summary.update(recording=paths, recording_stats=stats)
    print(f"[OK] saved {paths['wav']}")
    print("     " + json.dumps(stats))
    if args.no_play or not rec.num_frames:
        return
    samples, gain_db = rec.mono(audio.mic_channel), 0.0
    if not args.no_normalize:
        samples, gain_db = normalize_peak(samples)
    print(f"[..] playing the recording back on the robot speaker (gain {gain_db:+.1f} dB)")
    handle = audio.play(samples, rec.sample_rate, label="playback_mic", meta={"gain_db": round(gain_db, 2)})
    finished = handle.wait(timeout=rec.duration_s + 10)
    item = handle.item
    summary["playback"] = {
        "status": item.status,
        "finished": finished,
        "gain_db": round(gain_db, 2),
        "queue_wait_ms": None if item.t_first_push_ns is None
        else round((item.t_first_push_ns - item.t_enqueued_ns) / 1e6, 2),
        "pushed_s": round(item.frames_pushed / rate, 3),
    }
    print(f"[OK] playback {item.status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--seconds", type=float, default=5.0, help="recording length, 0 = no recording (default 5)")
    parser.add_argument("--no-play", action="store_true", help="do not play the recording back")
    parser.add_argument("--no-normalize", action="store_true", help="replay the recording without peak normalization")
    parser.add_argument("--tone", action="store_true", help="play a 440 Hz test tone before recording")
    parser.add_argument("--speaker-test", action="store_true", help="audible speaker test (see above)")
    parser.add_argument("--info", action="store_true", help="print daemon / mic settings and exit")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides={"robot": {"host": args.host}})
    setup_console_logging(cfg.logging.level)
    runtime = prepare_runtime(cfg)

    if args.info:
        info = DaemonClient(cfg.robot.host, cfg.robot.port).summary()
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0 if "status_error" not in info else 1

    session = Session(cfg.logging.results_path, "audio_check")
    conn = None
    audio = None
    summary: dict = {"session_id": session.id, "ok": False}
    try:
        t_connect = time.perf_counter()
        backend, conn, robot_info = open_backend(cfg, args.mock)
        summary["connect_s"] = round(time.perf_counter() - t_connect, 3)
        session.write_metadata(app="audio_check", args=vars(args), config=cfg.to_dict(),
                               robot=robot_info, backend=backend.describe(), runtime=runtime)
        audio = AudioManager(backend, cfg.audio, session.events)
        t_start = time.perf_counter()
        audio.start()
        summary["first_input_s"] = round(time.perf_counter() - t_start, 3)
        print(f"\n[OK] {backend.name}: mic audio flowing "
              f"({backend.input_rate} Hz x {backend.input_channels} ch), "
              f"speaker path ready={backend.output_ready()}")

        rate = backend.output_rate
        if args.speaker_test:
            summary["speaker_test"] = _speaker_test(audio, conn, session, rate)

        if args.tone:
            print("[..] playing 440 Hz test tone (0.6 s, -10 dBFS)")
            handle = audio.play(tone(440.0, 0.6, rate, amplitude=0.3), rate, label="test_tone_440Hz")
            handle.wait(timeout=10)

        if args.seconds > 0:
            _record_and_play(args, audio, session, summary, rate)

        summary["input_stats"] = audio.input_stats()
        summary["ok"] = True
    except Exception as exc:
        log.exception("audio check failed")
        session.events.log("app.error", error=repr(exc))
        summary["error"] = repr(exc)
        print(f"\n[FAIL] {exc}", file=sys.stderr)
    finally:
        if audio is not None:
            audio.stop()
        if conn is not None:
            conn.close()
        (session.dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
        stats = summary.get("recording_stats", {})
        istats = summary.get("input_stats", {})
        append_csv_row(cfg.logging.results_path / "audio_check.csv", {
            "session_id": session.id,
            "ok": summary["ok"],
            "backend": "mock" if args.mock else f"reachy@{cfg.robot.host}",
            "connect_s": summary.get("connect_s"),
            "first_input_s": summary.get("first_input_s"),
            "duration_s": stats.get("duration_s"),
            "measured_rate_hz": istats.get("measured_rate_hz"),
            "chunk_frames_min": istats.get("chunk_frames_min"),
            "chunk_frames_max": istats.get("chunk_frames_max"),
            "ch0_rms_dbfs": stats.get("ch0_rms_dbfs"),
            "ch0_peak_dbfs": stats.get("ch0_peak_dbfs"),
            "ch0_floor_dbfs_p10": stats.get("ch0_floor_dbfs_p10"),
            "clipped_samples": stats.get("clipped_samples"),
            "lr_identical": stats.get("lr_identical"),
            "lr_corr": stats.get("lr_corr"),
            "playback_status": (summary.get("playback") or {}).get("status"),
            "playback_gain_db": (summary.get("playback") or {}).get("gain_db"),
            "playback_queue_wait_ms": (summary.get("playback") or {}).get("queue_wait_ms"),
            "error": summary.get("error", ""),
        })
        session.close()
        print(f"[..] session: {session.dir}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
