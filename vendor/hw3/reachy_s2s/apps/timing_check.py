"""Timing diagnostics: clock resolution, timed-wait accuracy, thread wake-up latency,
and how much scheduling / network delay is mixed into recorded chunk arrival times.

    python -m reachy_s2s.apps.timing_check                 # all checks, existing recordings
    python -m reachy_s2s.apps.timing_check --skip-waits    # only clocks + recordings

Writes results/sessions/<id>_timing_check/timing.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..config import load_config
from ..evaluation.loopback import fit_sample_clock
from ..evaluation.session import Session, setup_console_logging
from ..platform_timing import enable_high_resolution_timer

_CLOCKS: dict[str, Callable[[], int]] = {
    "monotonic_ns": time.monotonic_ns,
    "perf_counter_ns": time.perf_counter_ns,
    "time_ns": time.time_ns,
}


def _pct(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "n": int(arr.size),
        "min": round(float(arr.min()), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "max": round(float(arr.max()), 3),
    }


def clock_resolution(samples: int = 200_000) -> dict[str, Any]:
    """Declared resolution and the smallest observed non-zero step of each clock."""
    out: dict[str, Any] = {}
    for name, fn in _CLOCKS.items():
        info = time.get_clock_info(name.removesuffix("_ns"))
        prev, steps = fn(), []
        for _ in range(samples):
            now = fn()
            if now != prev:
                steps.append(now - prev)
                prev = now
        out[name] = {
            "implementation": info.implementation,
            "declared_resolution_ms": info.resolution * 1000,
            "observed_min_step_ms": round(min(steps) / 1e6, 6) if steps else None,
            "observed_median_step_ms": round(float(np.median(steps)) / 1e6, 6) if steps else None,
        }
    return out


def timed_wait_accuracy(targets_ms=(1.0, 5.0, 10.0), repeats: int = 40) -> dict[str, Any]:
    """Actual duration of timed waits (what the player / queues rely on)."""
    cond = threading.Condition()
    event = threading.Event()

    def cond_wait(s: float) -> None:
        with cond:
            cond.wait(s)

    waits: dict[str, Callable[[float], Any]] = {
        "time.sleep": time.sleep,
        "Event.wait": event.wait,
        "Condition.wait": cond_wait,
    }
    out: dict[str, Any] = {}
    for name, fn in waits.items():
        for target in targets_ms:
            actual = []
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                fn(target / 1000.0)
                actual.append((time.perf_counter_ns() - t0) / 1e6)
            out[f"{name}({target:g} ms)"] = _pct(actual)
    return out


def wake_latency(repeats: int = 200, busy_thread: bool = False) -> dict[str, Any]:
    """notify() in one thread -> wait() returns in another (player wake-up path)."""
    cond = threading.Condition()
    state = {"t_notify": 0, "flag": False}
    latencies: list[float] = []
    stop = threading.Event()

    def burner() -> None:  # CPU-bound pure-Python work, competes for the GIL
        x = 0
        while not stop.is_set():
            for i in range(2000):
                x += i * i

    def waiter() -> None:
        for _ in range(repeats):
            with cond:
                while not state["flag"]:
                    cond.wait()
                latencies.append((time.perf_counter_ns() - state["t_notify"]) / 1e6)
                state["flag"] = False

    if busy_thread:
        threading.Thread(target=burner, daemon=True).start()
    w = threading.Thread(target=waiter, daemon=True)
    w.start()
    for _ in range(repeats):
        time.sleep(0.002)
        with cond:
            state["t_notify"] = time.perf_counter_ns()
            state["flag"] = True
            cond.notify()
        while state["flag"]:
            time.sleep(0.0002)
    w.join(timeout=5)
    stop.set()
    return {"switch_interval_ms": sys.getswitchinterval() * 1000, **_pct(latencies)}


def arrival_jitter(results_dir: Path) -> list[dict[str, Any]]:
    """Inter-arrival deltas and residuals vs. the lower-envelope sample clock."""
    rows = []
    for meta_path in sorted(results_dir.glob("sessions/*/audio/*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        anchors = meta.get("anchors") or []
        if len(anchors) < 50:
            continue
        rate = meta["sample_rate"]
        pairs = [(int(a[0]), int(round(a[1] * 1e9))) for a in anchors]
        clock = fit_sample_clock(pairs, rate)
        t = np.array([p[1] for p in pairs], dtype=np.float64)
        idx = np.array([p[0] for p in pairs], dtype=np.float64)
        residual_ms = (t - (clock.offset_ns + clock.ns_per_sample * idx)) / 1e6
        rows.append({
            "recording": f"{meta_path.parent.parent.name}/{meta_path.stem}",
            "seconds": round(meta["num_frames"] / rate, 1),
            "inter_arrival_ms": _pct(np.diff(t) / 1e6),
            "residual_ms": _pct(residual_ms),
            "implied_rate_hz": round(1e9 / clock.ns_per_sample, 1),
        })
    return rows


def _arrival_stats(anchors: list[tuple[int, int]], rate: int) -> dict[str, Any]:
    idx = np.array([a[0] for a in anchors], dtype=np.float64)
    t = np.array([a[1] for a in anchors], dtype=np.float64)
    nominal = t - idx * 1e9 / rate
    residual = (nominal - nominal.min()) / 1e6
    return {"inter_arrival_ms": _pct(np.diff(t) / 1e6), "residual_ms": _pct(residual)}


def robot_capture(cfg: Any, seconds: float) -> dict[str, Any]:
    """Mic-only capture on the robot: default timer vs. timeBeginPeriod(1)."""
    from ..robot.audio_manager import AudioManager
    from ..robot.connection import ReachyConnection

    conn = ReachyConnection(cfg.robot, "WARNING").open()
    audio = AudioManager(conn.audio_backend(cfg.audio.output_latency_hint_ms / 1000), cfg.audio)
    out: dict[str, Any] = {}
    try:
        audio.start()
        time.sleep(1.0)  # let the start-up backlog drain
        for label in ("default_timer", "timeBeginPeriod1"):
            if label == "timeBeginPeriod1":
                enable_high_resolution_timer(1)
                time.sleep(0.5)
            audio.start_recording(label)
            time.sleep(seconds)
            rec = audio.stop_recording()
            out[label] = _arrival_stats(rec.anchors, rec.sample_rate)
            ia, rs = out[label]["inter_arrival_ms"], out[label]["residual_ms"]
            print(f"robot {label:<18} inter-arrival p1 {ia['min']:5.1f} p50 {ia['p50']:5.1f} p99 {ia['p99']:5.1f} ms | "
                  f"residual p50 {rs['p50']:5.1f} p95 {rs['p95']:5.1f} max {rs['max']:5.1f} ms")
    finally:
        audio.stop()
        conn.close()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append",
                        help="TOML file overriding config/default.toml (repeatable, applied in order)")
    parser.add_argument("--host", help="robot IP (for --robot-capture)")
    parser.add_argument("--skip-waits", action="store_true", help="skip the wait / wake-up measurements")
    parser.add_argument("--robot-capture", type=float, metavar="S",
                        help="also capture S seconds from the robot mic with the default timer and with "
                             "timeBeginPeriod(1) (no sound is played)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides={"robot": {"host": args.host}})
    setup_console_logging("WARNING")
    session = Session(cfg.logging.results_path, "timing_check", capture_logging=False)
    report: dict[str, Any] = {"python": sys.version.split()[0], "platform": sys.platform}
    try:
        report["clocks"] = clock_resolution()
        print(json.dumps(report["clocks"], indent=1))
        if args.robot_capture:  # before anything raises the timer resolution in this process
            report["robot_capture"] = robot_capture(cfg, args.robot_capture)
        if not args.skip_waits:
            already = args.robot_capture is not None  # timer was raised by the robot capture
            label = "timeBeginPeriod1" if already else "default"
            report[f"timed_waits_{label}"] = timed_wait_accuracy()
            report[f"wake_latency_idle_{label}"] = wake_latency()
            report[f"wake_latency_gil_busy_{label}"] = wake_latency(busy_thread=True)
            if not already and enable_high_resolution_timer(1):
                report["timed_waits_timeBeginPeriod1"] = timed_wait_accuracy()
                report["wake_latency_idle_timeBeginPeriod1"] = wake_latency()
                report["wake_latency_gil_busy_timeBeginPeriod1"] = wake_latency(busy_thread=True)
            for key in ("timed_waits_default", "timed_waits_timeBeginPeriod1"):
                for name, stats in report.get(key, {}).items():
                    print(f"{key:<30} {name:<26} p50 {stats['p50']:7.3f} ms  p95 {stats['p95']:7.3f}  max {stats['max']:7.3f}")
            for key in sorted(k for k in report if k.startswith("wake_latency")):
                s = report[key]
                print(f"{key:<38} p50 {s['p50']:.3f} ms  p95 {s['p95']:.3f}  p99 {s['p99']:.3f}  max {s['max']:.3f}")
        report["arrival_jitter"] = arrival_jitter(cfg.logging.results_path)
        for row in report["arrival_jitter"]:
            r = row["residual_ms"]
            print(f"{row['recording']:<45} {row['seconds']:6.1f}s  residual p50 {r['p50']:6.2f}  "
                  f"p95 {r['p95']:6.2f}  p99 {r['p99']:6.2f}  max {r['max']:7.2f} ms  rate {row['implied_rate_hz']}")
    finally:
        (session.dir / "timing.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        session.close()
        print(f"[..] {session.dir / 'timing.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
