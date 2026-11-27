"""Measure the acoustic round trip push -> robot speaker -> robot mic -> PC.

    python -m reachy_s2s.apps.loopback_probe                 # 6 chirps, default level
    python -m reachy_s2s.apps.loopback_probe --repeats 10 --amp 0.5
    python -m reachy_s2s.apps.loopback_probe --mock-delay 0.35   # self-test without robot

Plays short chirps on the robot speaker while recording its mic, then finds
each chirp with a normalized cross-correlation. Keep the room quiet. A low
score / "not detected" usually means the XVF3800 echo canceller removed the
chirp from the processed mic channel (that itself is a useful finding).

Outputs: results/sessions/<id>_loopback/{loopback.json, audio/loopback.wav, ...}
and one row in results/loopback_probe.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np

from ..audio.signals import chirp
from ..config import load_config
from ..evaluation.loopback import fit_sample_clock, locate_probes
from ..evaluation.session import Session, setup_console_logging
from ..robot.audio_manager import AudioManager
from ..robot.backends.mock import MockAudioBackend
from .common import add_common_args, append_csv_row, open_backend, prepare_runtime

log = logging.getLogger("loopback_probe")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--interval", type=float, default=1.5, help="seconds between chirps")
    parser.add_argument("--amp", type=float, default=0.3, help="chirp amplitude (0-1)")
    parser.add_argument("--f0", type=float, default=300.0)
    parser.add_argument("--f1", type=float, default=3400.0)
    parser.add_argument("--dur", type=float, default=0.25, help="chirp length in seconds")
    parser.add_argument("--threshold", type=float, default=0.3, help="minimum NCC for a detection")
    parser.add_argument("--mock-delay", type=float, help="no robot: simulated loopback delay in seconds")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides={"robot": {"host": args.host}})
    setup_console_logging(cfg.logging.level)
    runtime = prepare_runtime(cfg)
    session = Session(cfg.logging.results_path, "loopback")
    session.write_metadata(runtime=runtime)
    conn = None
    audio = None
    result: dict = {"session_id": session.id, "ok": False}
    try:
        if args.mock_delay is not None:
            backend = MockAudioBackend(sample_rate=16000, channels=2, realtime=True,
                                       loopback_delay_s=args.mock_delay, loopback_gain=0.5)
            robot_info = {"mock_loopback_delay_s": args.mock_delay}
        else:
            backend, conn, robot_info = open_backend(cfg, args.mock)
        session.write_metadata(app="loopback_probe", args=vars(args), config=cfg.to_dict(),
                               robot=robot_info, backend=backend.describe())
        audio = AudioManager(backend, cfg.audio, session.events)
        audio.start()
        rate = backend.output_rate
        probe = chirp(args.f0, args.f1, args.dur, rate, amplitude=args.amp)

        audio.start_recording("loopback")
        time.sleep(1.0)  # baseline
        pushes: list[tuple[str, int]] = []
        for i in range(args.repeats):
            t_next = time.perf_counter() + args.interval
            handle = audio.play(probe, rate, label=f"probe_{i}")
            handle.wait(timeout=5, until_heard=False)
            if handle.item.t_first_push_ns is not None:
                pushes.append((f"probe_{i}", handle.item.t_first_push_ns))
            print(f"[..] chirp {i + 1}/{args.repeats} pushed")
            time.sleep(max(0.0, t_next - time.perf_counter()))
        time.sleep(1.5)
        rec = audio.stop_recording()
        paths = rec.save(session.audio_dir, session.t0_ns)

        clock = fit_sample_clock(rec.anchors, rec.sample_rate)
        mono = rec.mono(cfg.audio.mic_channel)
        found = locate_probes(mono, rec.start_index, clock, probe, pushes, rec.sample_rate,
                              search_s=min(2.0, args.interval), threshold=args.threshold)
        rtts = [p.round_trip_ms for p in found if p.detected]
        result.update(
            ok=True,
            recording=paths,
            sample_clock={"ns_per_sample": clock.ns_per_sample, "implied_rate_hz": 1e9 / clock.ns_per_sample},
            probes=[{"label": p.label, "detected": p.detected, "score": round(p.score, 3),
                     "round_trip_ms": None if p.round_trip_ms is None else round(p.round_trip_ms, 1)}
                    for p in found],
            detected=len(rtts),
            round_trip_ms_median=None if not rtts else round(float(np.median(rtts)), 1),
            round_trip_ms_mean=None if not rtts else round(float(np.mean(rtts)), 1),
            round_trip_ms_std=None if len(rtts) < 2 else round(float(np.std(rtts, ddof=1)), 1),
            round_trip_ms_min=None if not rtts else round(float(np.min(rtts)), 1),
            round_trip_ms_max=None if not rtts else round(float(np.max(rtts)), 1),
        )
        for p in result["probes"]:
            print(f"     {p['label']}: detected={p['detected']} score={p['score']} rtt={p['round_trip_ms']} ms")
        print(f"[OK] detected {len(rtts)}/{len(found)}; median round trip = {result['round_trip_ms_median']} ms")
    except Exception as exc:
        log.exception("loopback probe failed")
        session.events.log("app.error", error=repr(exc))
        result["error"] = repr(exc)
        print(f"\n[FAIL] {exc}", file=sys.stderr)
    finally:
        if audio is not None:
            audio.stop()
        if conn is not None:
            conn.close()
        (session.dir / "loopback.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        append_csv_row(cfg.logging.results_path / "loopback_probe.csv", {
            "session_id": session.id,
            "ok": result["ok"],
            "backend": "mock" if (args.mock or args.mock_delay is not None) else f"reachy@{cfg.robot.host}",
            "repeats": args.repeats,
            "amp": args.amp,
            "detected": result.get("detected"),
            "rtt_median_ms": result.get("round_trip_ms_median"),
            "rtt_mean_ms": result.get("round_trip_ms_mean"),
            "rtt_std_ms": result.get("round_trip_ms_std"),
            "rtt_min_ms": result.get("round_trip_ms_min"),
            "rtt_max_ms": result.get("round_trip_ms_max"),
            "scores": " ".join(str(p["score"]) for p in result.get("probes", [])),
            "error": result.get("error", ""),
        })
        session.close()
        print(f"[..] session: {session.dir}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
