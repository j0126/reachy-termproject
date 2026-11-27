"""Supertonic CPU threads: synthesis time and what it does to the rest of the process (Phase 4.5).

    python -m reachy_s2s.apps.tts_threads_bench --config config/local_e4b_supertonic.toml           # PC, silent mock mic
    python -m reachy_s2s.apps.tts_threads_bench --config config/local_e4b_supertonic.toml --robot   # robot mic, read only

For every ``[tts] threads`` value (default 0 = the Phase 2-4 setting, 4, 2) the model is loaded and
warmed up, then the same sentences (real E4B answers from the Phase 4 robot sessions) are synthesised
in blocks, without and with the motion controller (simulated motors only, Motion ON, SPEAKING, its
speech tap fed with the synthesised audio as if it were playing). Pass 2 runs the settings in reverse
order and swaps the block order. Meanwhile the microphone is captured as in a conversation: a silent
mock stream, or the robot's over WebRTC (``--robot``: nothing is played, no motor command is sent;
the motor mode is only read, before and after).

``--load dashboard`` adds the dashboard's other work in this process, as in a conversation: the live
VAD, the mel monitor and a 30 fps spectrogram (offscreen); the synthesis then runs in a worker thread,
like the pipeline's. The Phase 4 input gaps (2026-10-17) appeared only under that load.

Per synthesis, one row in results/tts_threads_bench.csv: synthesis time, audio length, process CPU,
microphone arrival gaps (> [audio] gap_warn_ms) and motion tick intervals during the synthesis.
The summary per (threads, motion) goes to the console and the session folder (summary.json).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AppConfig, load_config
from ..conversation.state import InteractionState as S
from ..conversation.state import StateTracker
from ..evaluation.session import Session, setup_console_logging
from ..evaluation.tables import append_csv_row
from ..motion.controller import ACTIVE, MotionController
from ..motion.pose import NEUTRAL
from ..motion.sinks import SimulatedSink
from ..robot.audio_manager import AudioManager
from ..robot.backends.mock import MockAudioBackend
from ..robot.connection import DaemonClient
from .common import add_common_args, open_backend, prepare_runtime

#: answers E4B actually gave in the Phase 4 robot sessions (15-96 characters)
SENTENCES = (
    "회의는 11월 13일이에요.",
    "네, 말씀하세요. 제가 듣고 있어요.",
    "회의 참석자는 14명이군요. 제가 잘 기억해 두겠습니다.",
    "안녕하세요. 저는 Reachy Mini예요. 무엇을 도와드릴까요?",
    "저는 Reachy Mini라는 작은 탁상용 로봇이에요. 제 몸은 바퀴와 팔은 없답니다. 제가 무엇을 도와드릴까요?",
    "한국의 봄은 꽃이 피고 따뜻해지며 생기가 넘쳐요. 여름은 무덥고 장마가 있답니다. 가을은 풍요롭고 단풍이 "
    "아름다운 계절이에요. 그리고 겨울은 춥고 눈이 많이 내리기도 해요.",
)
LATE_INTERVAL_MS = 30.0  # a 50 Hz tick interval longer than this counts as late (1.5 periods)
PAUSE_S = 0.4  # idle time between syntheses (lets the ONNX threads settle)
GAP_MARGIN_S = 0.25  # a stall shows as a late chunk arriving after it: count gaps until this long after


def _pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


class SpeechFeeder:
    """Feeds the last synthesised audio to the controller's speech tap in 20 ms blocks, like the player."""

    def __init__(self, hint_s: float) -> None:
        self.hint_s = hint_s
        self.controller: MotionController | None = None
        self.audio: tuple[np.ndarray, int] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bench-feeder", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        pos = 0
        while not self._stop.wait(0.02):
            ctl, audio = self.controller, self.audio
            if ctl is None or audio is None:
                continue
            x, rate = audio
            n = max(1, int(rate * 0.02))
            if pos + n > x.shape[0]:
                pos = 0
            ctl.tap.on_block(x[pos:pos + n], time.perf_counter_ns() + int(self.hint_s * 1e9), rate)
            pos += n


def make_controller(cfg: AppConfig, session: Session, tick_log: deque) -> MotionController:
    """Motion ON on simulated motors only (this tool never creates a robot sink), already at neutral."""
    mcfg = replace(cfg.motion, enabled=True, sink="dry_run", startup_ramp_s=0.3, log_commands=False)
    ctl = MotionController(replace(cfg, motion=mcfg), SimulatedSink(start=NEUTRAL), session.events)
    inner = ctl.tick

    def logged_tick(now: int | None = None) -> None:
        tick_log.append(time.perf_counter_ns())
        inner(now)

    ctl.tick = logged_tick  # type: ignore[method-assign]
    return ctl


def window_stats(t0: int, t1: int, audio: AudioManager, ticks: list[int], motion: bool) -> dict[str, Any]:
    gaps = [ms for _i, ms, t in audio.gap_log.copy() if t0 <= t <= t1 + int(GAP_MARGIN_S * 1e9)]
    out: dict[str, Any] = {"gaps": len(gaps), "gap_ms_total": round(sum(gaps), 1),
                           "gap_ms_max": round(max(gaps), 1) if gaps else 0.0,  # same columns in every row (CSV)
                           "tick_hz": None, "tick_interval_p95_ms": None, "tick_interval_max_ms": None,
                           "late_ticks": None}
    if motion:
        inside = [t for t in ticks if t0 <= t <= t1]
        iv = np.diff(inside) / 1e6 if len(inside) > 1 else np.zeros(0)
        out.update(tick_hz=round((len(inside) - 1) / ((inside[-1] - inside[0]) / 1e9), 1) if len(inside) > 2 else None,
                   tick_interval_p95_ms=_pct(list(iv), 95), tick_interval_max_ms=round(float(iv.max()), 1) if iv.size else None,
                   late_ticks=int(np.sum(iv > LATE_INTERVAL_MS)) if iv.size else None)
    return out


def run_with_dashboard_load(audio: AudioManager, cfg: AppConfig, session: Session, body: Any) -> None:
    """Run ``body`` in a worker thread while this (main) thread does the dashboard's other work:
    live VAD, mel monitor, 30 fps spectrogram (offscreen)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import pyqtgraph as pg
    from PySide6 import QtCore, QtWidgets

    from ..audio.mel import mel_band_edges
    from ..audio.mel_monitor import MelMonitor
    from ..engines.vad import VadWorker
    from ..gui.mel_view import MelSpectrogramView

    pg.setConfigOptions(antialias=False, imageAxisOrder="col-major")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    mel = cfg.mel
    centers = mel_band_edges(mel.n_mels, mel.fmin_hz, mel.fmax_hz)[1:-1]
    view = MelSpectrogramView(mel.n_mels, mel.display_seconds, centers, (mel.display_db_min, mel.display_db_max))
    view.resize(900, 400)
    view.show()
    monitor = MelMonitor(audio.subscribe("mel"), audio.input_rate, mel, cfg.audio.mic_channel)
    monitor.start()
    vad = VadWorker(audio, cfg.vad, session.events, on_utterance=lambda _utt: None)
    vad.start()
    errors: list[BaseException] = []

    def work() -> None:
        try:
            body()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
            errors.append(exc)

    worker = threading.Thread(target=work, name="bench-synthesis", daemon=True)
    worker.start()
    render = QtCore.QTimer()
    render.timeout.connect(lambda: view.update_image(monitor.snapshot()))
    render.start(33)
    done = QtCore.QTimer()
    done.timeout.connect(lambda: None if worker.is_alive() else app.quit())
    done.start(200)
    app.exec()
    render.stop()
    done.stop()
    vad.stop()
    monitor.stop()
    audio.unsubscribe("mel")
    view.close()
    if errors:
        raise errors[0]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for threads in dict.fromkeys(r["threads"] for r in rows):
        for motion in (False, True):
            rs = [r for r in rows if r["threads"] == threads and r["motion"] == motion]
            if not rs:
                continue
            synth = [r["synth_ms"] for r in rs]
            s: dict[str, Any] = {"threads": threads, "motion": motion, "n": len(rs),
                                 "synth_ms_median": _pct(synth, 50), "synth_ms_p90": _pct(synth, 90),
                                 "synth_ms_max": round(max(synth), 1),
                                 "rtf_median": round(float(np.median([r["rtf"] for r in rs])), 3),
                                 "cpu_pct_mean": round(float(np.mean([r["cpu_pct"] for r in rs])), 0),
                                 "gaps": sum(r["gaps"] for r in rs),
                                 "gap_ms_total": round(sum(r["gap_ms_total"] for r in rs), 1),
                                 "gap_ms_max": max(r["gap_ms_max"] for r in rs)}
            if motion:
                hz = [r["tick_hz"] for r in rs if r.get("tick_hz") is not None]
                p95 = [r["tick_interval_p95_ms"] for r in rs if r.get("tick_interval_p95_ms") is not None]
                mx = [r["tick_interval_max_ms"] for r in rs if r.get("tick_interval_max_ms") is not None]
                s.update(tick_hz_min=min(hz) if hz else None, tick_hz_median=_pct(hz, 50),
                         tick_interval_p95_ms_median=_pct(p95, 50), tick_interval_max_ms=max(mx) if mx else None,
                         late_ticks=sum(r.get("late_ticks") or 0 for r in rs))
            out.append(s)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--threads", default="0,4,2", help="comma list of [tts] threads values (0 = all cores)")
    parser.add_argument("--passes", type=int, default=2, help="passes over the settings (even passes reverse the order)")
    parser.add_argument("--robot", action="store_true",
                        help="capture the robot microphone over WebRTC (read only: nothing played, no motor command)")
    parser.add_argument("--fake-tts", action="store_true", help="stand-in TTS (rehearse the tool, no model)")
    parser.add_argument("--load", choices=("none", "dashboard"), default="none",
                        help="dashboard: also run the live VAD, mel monitor and a 30 fps spectrogram (offscreen)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides={"robot": {"host": args.host}})
    setup_console_logging("WARNING")
    runtime = prepare_runtime(cfg)
    settings = [int(t) for t in args.threads.split(",") if t.strip()]
    if cfg.tts.provider != "supertonic" and not args.fake_tts:
        raise SystemExit("[tts] provider is not supertonic: use --config config/local_e4b_supertonic.toml")
    mic = "robot" if args.robot else "mock"
    real = cfg.logging.results_path
    results = real if args.robot else real / "_mock_and_tests"
    out_csv = (real / "tts_threads_bench.csv") if not args.fake_tts else (real / "_mock_and_tests" /
                                                                           "tts_threads_bench_mock.csv")
    session = Session(results, f"tts_threads_bench_{mic}" + ("_dashboard" if args.load == "dashboard" else ""))
    daemon = DaemonClient(cfg.robot.host, cfg.robot.port) if args.robot else None
    meta: dict[str, Any] = {"app": "tts_threads_bench", "args": vars(args), "config": cfg.to_dict(),
                            "runtime": runtime, "logical_cpus": os.cpu_count(), "sentences": SENTENCES}
    if daemon is not None:
        meta["motor_mode_before"] = daemon.motor_mode()
    conn = None
    if args.robot:
        backend, conn, meta["robot"] = open_backend(cfg, None)
    else:
        backend = MockAudioBackend(np.zeros((16000, 2), np.float32), realtime=True)
    session.write_metadata(**meta)
    audio = AudioManager(backend, cfg.audio, session.events)
    audio.start()
    feeder = SpeechFeeder(cfg.audio.output_latency_hint_ms / 1000.0)
    feeder.start()
    tick_log: deque[int] = deque(maxlen=200_000)
    rows: list[dict[str, Any]] = []
    proc = None
    try:
        import psutil

        proc = psutil.Process()
    except ImportError:
        pass
    def measure() -> None:
        for p in range(args.passes):
            order = settings if p % 2 == 0 else list(reversed(settings))
            for threads in order:
                if args.fake_tts:
                    from ..engines.fakes import FakeTTS

                    tts: Any = FakeTTS(delay_s=0.05, seconds_per_char=0.01)
                    load: dict[str, Any] = {"fake": True}
                else:
                    from ..engines.tts_supertonic import SupertonicTTS

                    tts = SupertonicTTS(replace(cfg.tts, threads=threads))
                    load = tts.load()  # load + one warm-up synthesis
                    tts.synthesize(SENTENCES[3])  # a second warm-up at full length
                session.events.log("bench.loaded", threads=threads, **load)
                blocks = (False, True) if p % 2 == 0 else (True, False)
                for motion in blocks:
                    ctl = None
                    if motion:
                        ctl = make_controller(cfg, session, tick_log)
                        state = StateTracker(session.events)
                        ctl.attach(state, session.events)
                        ctl.start()
                        ctl.motor_on()
                        if not ctl.wait_for((ACTIVE,), 10):
                            raise RuntimeError(f"simulated controller did not reach ACTIVE ({ctl.mode})")
                        state.set(S.SPEAKING, "bench")
                        feeder.controller = ctl
                    for k, text in enumerate(SENTENCES):
                        time.sleep(PAUSE_S)
                        c0 = proc.cpu_times() if proc else None
                        t0 = time.perf_counter_ns()
                        result = tts.synthesize(text)
                        t1 = time.perf_counter_ns()
                        c1 = proc.cpu_times() if proc else None
                        wall = (t1 - t0) / 1e9
                        audio_s = result.audio.shape[0] / result.sample_rate
                        if motion:
                            feeder.audio = (result.audio.astype(np.float32), result.sample_rate)
                        cpu = None if c0 is None else round(100.0 * ((c1.user - c0.user) + (c1.system - c0.system))
                                                            / wall, 0)
                        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "session": session.id, "mic": mic,
                               "pass": p + 1, "threads": threads, "motion": motion, "sentence": k,
                               "chars": len(text), "synth_ms": round(wall * 1000, 1), "audio_s": round(audio_s, 3),
                               "rtf": round(wall / audio_s, 3) if audio_s else None, "cpu_pct": cpu,
                               **window_stats(t0, t1, audio, list(tick_log), motion)}
                        row["load"] = args.load
                        rows.append(row)
                        append_csv_row(out_csv, row)
                        print(f"  pass {p + 1} threads {threads} motion {'on ' if motion else 'off'} #{k} "
                              f"{row['synth_ms']:7.1f} ms  cpu {cpu}%  gaps {row['gaps']}"
                              + (f"  ticks {row.get('tick_hz')} Hz max {row.get('tick_interval_max_ms')} ms"
                                 if motion else ""), flush=True)
                    if ctl is not None:
                        feeder.controller = None
                        feeder.audio = None
                        ctl.close()
                del tts

    try:
        print(f"[..] mic: {mic}; load {args.load}; settings {settings}; {args.passes} passes; {len(SENTENCES)} "
              f"sentences; CPUs {os.cpu_count()}; session {session.dir}")
        if args.load == "dashboard":
            run_with_dashboard_load(audio, cfg, session, measure)
        else:
            measure()
        summary = summarize(rows)
        print("\n threads motion   n  synth median / p90 / max (ms)  RTF   CPU%  gaps (total ms, max)  ticks")
        for s in summary:
            ticks = (f"{s['tick_hz_min']}-{s['tick_hz_median']} Hz, p95 {s['tick_interval_p95_ms_median']} ms, "
                     f"max {s['tick_interval_max_ms']} ms, late {s['late_ticks']}") if s["motion"] else "-"
            print(f" {s['threads']:>7} {'on ' if s['motion'] else 'off':>6} {s['n']:>3}  {s['synth_ms_median']:>7} / "
                  f"{s['synth_ms_p90']:>7} / {s['synth_ms_max']:>7}  {s['rtf_median']:5.3f} {s['cpu_pct_mean']:>5}  "
                  f"{s['gaps']:>3} ({s['gap_ms_total']}, {s['gap_ms_max']})  {ticks}")
        (session.dir / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1,
                                                             ensure_ascii=False), encoding="utf-8")
    finally:
        feeder.stop()
        audio.stop()
        if conn is not None:
            conn.close()
        if daemon is not None:
            try:
                session.write_metadata(motor_mode_after=daemon.motor_mode())
            except Exception as exc:  # noqa: BLE001
                session.write_metadata(motor_mode_after=f"unavailable: {exc!r}")
        session.close()
        print(f"[..] {out_csv}\n[..] session: {session.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
