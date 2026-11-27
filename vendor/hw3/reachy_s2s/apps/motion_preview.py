"""Motion preview and staged robot checks (Phase 4.5). A DRY RUN unless --live is given.

    python -m reachy_s2s.apps.motion_preview                    # dry run in real time, all steps
    python -m reachy_s2s.apps.motion_preview --fast             # dry run on a virtual clock (a few seconds)
    python -m reachy_s2s.apps.motion_preview --estop            # REST torque off now, then exit (any terminal)
    python -m reachy_s2s.apps.motion_preview --pose             # read only: motor mode + present pose (no command)
    python -m reachy_s2s.apps.motion_preview --watch 600        # read only, REST GET: motor mode + head pose log
    python -m reachy_s2s.apps.motion_preview --fast --from-robot --steps enable   # dry run from the robot's real pose

Robot (only with the operator at the robot; asks you to type "move"; every step waits for Enter):

    python -m reachy_s2s.apps.motion_preview --live --steps enable                 # torque on -> neutral -> off
    python -m reachy_s2s.apps.motion_preview --live --steps signs --intensity 0.3  # axis signs, small pulses
    python -m reachy_s2s.apps.motion_preview --live --steps states,speech --intensity 0.5

Steps: enable (Motor ON -> neutral -> hold), signs (one axis at a time), states (IDLE / LISTENING
+ nod / PROCESSING / SPEAKING / INTERRUPTED / ERROR), speech (speaking motion driven by a WAV or a
synthetic speech-like signal; the audio is NOT played), estop (Motor ON -> hold -> Emergency
Torque OFF). Every run ends with the normal shutdown (neutral -> sleep pose -> checks -> torque off)
unless an emergency stop ended it. Ctrl+C = Motion Stop, then the normal shutdown.

Results: robot runs in results/sessions/<id>_motion_preview_live/, dry runs (simulated motors) in
results/_mock_and_tests/sessions/<id>_motion_preview_dry/: events.jsonl, motion.csv, preview.json, and in
live runs present.csv (the measured head pose, to compare with the commands).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.wavio import read_wav
from ..config import load_config
from ..conversation.state import InteractionState as S
from ..conversation.state import StateTracker
from ..evaluation.session import Session, setup_console_logging
from ..motion.analysis import phase_stats, read_motion_csv
from ..motion.controller import ACTIVE, ESTOP, FAULT, HOLD, MAX_INTENSITY, STOPPED, TORQUE_OFF, MotionController
from ..motion.pose import Pose, pose_from_matrix
from ..motion.sinks import RobotSink, SimulatedSink, simulated_like_robot
from ..motion.trajectory import bump
from ..robot.connection import DaemonClient, ReachyConnection
from .common import add_common_args, prepare_runtime

STEPS = ("enable", "signs", "states", "speech", "estop")
#: one axis at a time (intensity-scaled), with what the operator should see (signs from HW2; roll unverified)
SIGN_PULSES = (
    ("pitch +6 deg", Pose(pitch=6.0), "head tilts DOWN (nods forward)"),
    ("yaw +10 deg", Pose(yaw=10.0), "head turns to the ROBOT's own left (your right when facing it)"),
    ("roll +5 deg", Pose(roll=5.0), "head tilts sideways - NOTE which side (not verified in HW2)"),
    ("z +3 mm", Pose(z=3.0), "head rises slightly"),
    ("antennas perk +0.3 rad", Pose(ant_r=0.3, ant_l=-0.3), "both antennas RAISE (HW2 'happy' direction)"),
)
PULSE_S = 2.0
CHUNK_S = 0.02


class _Clock:
    """Real time, or a virtual clock that the runner advances (``--fast``)."""

    def __init__(self, virtual: bool, start_ns: int) -> None:
        self.virtual = virtual
        self.t = start_ns

    def __call__(self) -> int:
        return self.t if self.virtual else time.perf_counter_ns()


class Runner:
    def __init__(self, ctl: MotionController, clock: _Clock, live: bool) -> None:
        self.ctl, self.clock, self.live = ctl, clock, live
        self.audio_feed: list[tuple[int, np.ndarray, int]] = []  # fast mode: (t_audible_ns, block, rate)

    def wait(self, seconds: float) -> None:
        if not self.clock.virtual:
            time.sleep(seconds)
            return
        period = int(1e9 / self.ctl.rate_hz)
        end = self.clock.t + int(seconds * 1e9)
        while self.clock.t < end:
            self.clock.t += period
            while self.audio_feed and self.audio_feed[0][0] - int(0.2e9) <= self.clock.t:  # player lead ~200 ms
                t_aud, block, rate = self.audio_feed.pop(0)
                self.ctl.tap.on_block(block, t_aud, rate)
            self.ctl.tick(self.clock.t)

    def until(self, modes: tuple[str, ...], timeout: float) -> bool:
        if not self.clock.virtual:
            return self.ctl.wait_for(modes, timeout)
        waited = 0.0
        while self.ctl.mode not in modes and waited < timeout:
            self.wait(0.1)
            waited += 0.1
        return self.ctl.mode in modes


def _speech_signal(rate: int, seconds: float = 4.0) -> np.ndarray:
    """Speech-like test signal: harmonic voice with syllable (4 Hz) and phrase envelopes and a pause."""
    t = np.arange(int(rate * seconds)) / rate
    f0 = 140.0 + 20.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    voice = sum(np.sin(k * phase) / k for k in range(1, 6))
    syll = np.abs(np.sin(2 * np.pi * 2.0 * t)) ** 0.6
    phrase = ((t % 2.0) < 1.6).astype(np.float64)
    return (0.25 * voice * syll * phrase).astype(np.float32)


def _feed_audio(runner: Runner, ctl: MotionController, samples: np.ndarray, rate: int, hint_s: float) -> float:
    """Hand the blocks to the controller's playback tap like the audio player would (not played)."""
    n = max(1, int(rate * CHUNK_S))
    blocks = [samples[i:i + n] for i in range(0, len(samples), n)]
    start = runner.clock() + int(hint_s * 1e9)
    if runner.clock.virtual:
        runner.audio_feed = [(start + int(i * CHUNK_S * 1e9), b, rate) for i, b in enumerate(blocks)]
        return len(blocks) * CHUNK_S

    def feeder() -> None:
        t0 = time.perf_counter_ns()
        for i, b in enumerate(blocks):
            due = t0 + int(i * CHUNK_S * 1e9) - int(0.2e9)  # the player keeps ~200 ms ahead
            delay = (due - time.perf_counter_ns()) / 1e9
            if delay > 0:
                time.sleep(delay)
            ctl.tap.on_block(b, start + int(i * CHUNK_S * 1e9), rate)

    threading.Thread(target=feeder, name="preview-audio", daemon=True).start()
    return len(blocks) * CHUNK_S


class PresentRecorder:
    """Live runs: the measured head pose (SDK cached state, no extra requests) at 25 Hz -> present.csv,
    in the controller's torso frame (as motion.csv)."""

    def __init__(self, mini: Any, path: Path, t0_ns: int, ctl: MotionController) -> None:
        self.mini, self.path, self.t0_ns, self.ctl = mini, path, t0_ns, ctl
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="preview-present", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "roll", "pitch", "yaw", "x", "y", "z", "ant_r", "ant_l", "torso_yaw_deg"])
            while not self._stop.wait(0.04):
                try:
                    head = np.asarray(self.mini.get_current_head_pose(), dtype=np.float64)
                    ant = list(self.mini.get_current_joint_positions()[1])
                except Exception:  # noqa: BLE001 - measurement only
                    continue
                p = pose_from_matrix(self.ctl.to_local(head), ant)
                w.writerow([round((time.perf_counter_ns() - self.t0_ns) / 1e9, 4),
                            *(round(float(v), 4) for v in p.vec()), round(self.ctl.torso_yaw_deg, 3)])


def _prompt(live: bool, text: str) -> bool:
    if not live:
        print(f"[..] {text}")
        return True
    answer = input(f"[??] {text} - Enter = run, q = stop and shut down: ").strip().lower()
    return answer != "q"


def run_steps(runner: Runner, ctl: MotionController, state: StateTracker, steps: list[str],
              args: argparse.Namespace, cfg: Any) -> str:
    live = runner.live
    if not _prompt(live, "Motor ON: torque on, ramp from the present pose to neutral"):
        return "skipped"
    ctl.motor_on()
    if not runner.until((HOLD, ACTIVE, FAULT, ESTOP), 25.0) or ctl.mode in (FAULT, ESTOP):
        if ctl.needs_confirmation:
            print(f"[!!] {ctl.needs_confirmation}: not started (move the head near the sleep pose, or use the GUI)")
        return f"startup ended in {ctl.mode}: {ctl.last_error}"
    print(f"[OK] {ctl.mode} at neutral")
    runner.wait(args.hold if "enable" in steps else 1.0)
    for step in steps:
        if step == "enable":
            continue
        if ctl.mode in (FAULT, ESTOP, STOPPED):
            return f"stopped before '{step}': {ctl.mode} {ctl.last_error or ''}"
        if step == "estop":
            if not _prompt(live, "EMERGENCY TORQUE OFF test: the head will DROP to its rest position"):
                return "stopped by the operator"
            result = ctl.emergency_torque_off()
            runner.wait(0.5)
            print(f"[..] emergency torque off: {result}; motor mode now {ctl.sink.motor_mode()}")
            return "estop"
        if not _prompt(live, f"step '{step}' (intensity {ctl.engine.intensity:g})"):
            return "stopped by the operator"
        ctl.set_motion(True)
        runner.until((ACTIVE,), 2.0)
        if step == "signs":
            for name, pose, expect in SIGN_PULSES:
                print(f"   {name} x{ctl.engine.intensity:g}: expect {expect}")
                ctl.set_preview(lambda t, p=pose: p.scaled(bump(t / PULSE_S)))
                runner.wait(PULSE_S + 0.8)
            ctl.set_preview(None)
        elif step == "states":
            for target, dwell, note in ((S.LISTENING, 1.5, "listening pose, antennas up"),
                                        ("speech_start", 1.5, "user starts speaking: small nod / antenna perk"),
                                        ("speech_end", 0.3, ""),
                                        (S.PROCESSING, 4.0, "thinking: head tilt, antenna fidget, slow look"),
                                        (S.SPEAKING, 2.0, "speaking pose"),
                                        (S.INTERRUPTED, 0.8, "barge-in: drop everything, back to listening"),
                                        (S.LISTENING, 1.5, ""),
                                        (S.ERROR, 2.0, "error: small shake, antennas down"),
                                        (S.IDLE, 3.0, "idle breathing")):
                if isinstance(target, S):
                    print(f"   {target.value}: {note}")
                    state.set(target, "preview")
                else:
                    if note:
                        print(f"   {target}: {note}")
                    ctl.events.log(f"vad.{target}", source="preview")
                runner.wait(dwell)
        elif step == "speech":
            if args.wav:
                samples, rate = read_wav(args.wav)
                samples = samples.mean(axis=1) if samples.ndim == 2 else samples
            else:
                rate = 16000
                samples = _speech_signal(rate)
            state.set(S.SPEAKING, "preview speech")
            dur = _feed_audio(runner, ctl, samples.astype(np.float32), rate,
                              cfg.audio.output_latency_hint_ms / 1000.0)
            print(f"   speaking motion for {dur:.1f} s ({'WAV ' + args.wav if args.wav else 'synthetic voice'}; "
                  "not played)")
            runner.wait(dur + cfg.audio.output_latency_hint_ms / 1000.0 + 0.5)
            state.set(S.IDLE, "preview speech done")
            runner.wait(1.5)
        ctl.set_motion(False)
        runner.wait(1.0)
    return "done"


def watch(cfg: Any, seconds: float, period_s: float = 1.0) -> int:
    """Independent observer for dry runs (R0): REST GET only (motor mode, present head pose, antennas)
    every ``period_s`` -> watch.csv / watch.json. Sends nothing to the robot."""
    from ..motion.pose import rotation_distance_deg, translation_distance_mm

    daemon = DaemonClient(cfg.robot.host, cfg.robot.port, timeout_s=2.0)
    session = Session(cfg.logging.results_path, "motion_watch")
    modes: dict[str, int] = {}
    first: np.ndarray | None = None
    max_rot = max_trans = 0.0
    samples = errors = 0
    print(f"[..] watching {cfg.robot.host} for {seconds:.0f} s (REST GET only, every {period_s:g} s); "
          f"Ctrl+C ends early. Session {session.dir}")
    t_end = time.monotonic() + seconds
    try:
        with session.path("watch.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "motor_mode", "roll", "pitch", "yaw", "x", "y", "z", "ant0", "ant1", "torso_yaw_deg",
                        "rotation_from_start_deg", "translation_from_start_mm"])
            while time.monotonic() < t_end:
                t = time.perf_counter_ns()
                try:
                    mode = daemon.motor_mode() or "?"
                    m = np.asarray(daemon.get("/api/state/present_head_pose?use_pose_matrix=true",
                                              timeout_s=2.0)["m"], dtype=np.float64).reshape(4, 4)
                    ant = list(daemon.get("/api/state/present_antenna_joint_positions", timeout_s=2.0))
                    torso = math.degrees(float(daemon.get("/api/state/present_body_yaw", timeout_s=2.0)))
                except Exception as exc:  # noqa: BLE001 - keep watching
                    errors += 1
                    session.events.log("watch.error", error=repr(exc))
                    time.sleep(period_s)
                    continue
                first = m if first is None else first
                rot, trans = rotation_distance_deg(m, first), translation_distance_mm(m, first)
                max_rot, max_trans = max(max_rot, rot), max(max_trans, trans)
                modes[mode] = modes.get(mode, 0) + 1
                samples += 1
                p = pose_from_matrix(m, ant)
                w.writerow([session.events.rel(t), mode, *(round(float(v), 4) for v in p.vec()), round(torso, 3),
                            round(rot, 3), round(trans, 3)])
                fh.flush()
                if mode != "disabled" and modes[mode] == 1:
                    print(f"[!!] motor mode is now {mode!r}")
                time.sleep(max(0.0, period_s - (time.perf_counter_ns() - t) / 1e9))
    except KeyboardInterrupt:
        pass
    summary = {"samples": samples, "motor_modes": modes, "errors": errors,
               "max_rotation_from_start_deg": round(max_rot, 3), "max_translation_from_start_mm": round(max_trans, 3)}
    session.path("watch.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    session.close()
    print(f"[..] {json.dumps(summary)}\n[..] session: {session.dir}")
    return 0


def read_pose(cfg: Any) -> int:
    """Connect without media, read the motor mode and the present pose; sends no command."""
    from ..motion.pose import NEUTRAL, SLEEP_HEAD_MATRIX, rotation_distance_deg, translation_distance_mm

    from ..motion.limits import JointGuard
    from ..motion.pose import torso_frame

    daemon = DaemonClient(cfg.robot.host, cfg.robot.port)
    print(f"[..] motor mode (REST): {daemon.motor_mode()}")
    conn = ReachyConnection(replace(cfg.robot, media_backend="no_media"), cfg.logging.level).open()
    try:
        head = np.asarray(conn.mini.get_current_head_pose(), dtype=np.float64)
        joints, ant = conn.mini.get_current_joint_positions()
        ant = list(ant)
    finally:
        conn.close()
    reported = np.degrees(np.asarray(joints, dtype=np.float64))
    torso = float(reported[0])
    local = torso_frame(torso).T @ head  # the controller works in the torso frame
    print(f"[..] torso (body) yaw: {torso:.1f} deg (held as it is, never commanded)")
    print(f"[..] present head pose, world frame: {pose_from_matrix(head, ant).rounded()}")
    print(f"     torso frame:                      {pose_from_matrix(local, ant).rounded()}")
    guard = JointGuard(cfg.motion.joint_max_change_deg, cfg.motion.joint_limit_margin_deg)
    guard.set_body_yaw(torso)
    ours = guard.joints_deg(pose_from_matrix(local, ant))
    print(f"[..] joints reported by the daemon (deg; body yaw, stewart 1-6): {np.round(reported, 1).tolist()}")
    if ours is not None:  # validates the joint model the joint guard uses (PC inverse kinematics)
        print(f"     PC inverse kinematics (with the torso yaw):              {np.round(ours, 1).tolist()}"
              f"  (max difference {np.max(np.abs(ours[1:] - reported[1:])):.2f} deg)")
    tol_d, tol_mm = cfg.motion.pose_tolerance_deg, cfg.motion.pose_tolerance_mm
    for name, ref in (("sleep", SLEEP_HEAD_MATRIX), ("neutral", NEUTRAL.head_matrix())):
        rot, trans = rotation_distance_deg(local, ref), translation_distance_mm(local, ref)
        ok = rot <= tol_d and trans <= tol_mm
        print(f"     {name:8s}: {rot:5.1f} deg, {trans:5.1f} mm (torso frame) -> {'within' if ok else 'OUTSIDE'} "
              f"the start tolerance ({tol_d:g} deg / {tol_mm:g} mm)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--live", action="store_true", help="send the commands to the robot (asks to type 'move')")
    parser.add_argument("--fast", action="store_true", help="dry run on a virtual clock (no waiting)")
    parser.add_argument("--steps", default="enable,signs,states,speech",
                        help=f"comma list from {', '.join(STEPS)} (default: %(default)s)")
    parser.add_argument("--intensity", type=float, help="motion intensity (default: 0.3 live, config value dry)")
    parser.add_argument("--rate", type=float, choices=(25.0, 50.0), help="control rate (default: config)")
    parser.add_argument("--hold", type=float, default=5.0, help="seconds at neutral in the enable step")
    parser.add_argument("--wav", help="speech step: drive the speaking motion with this WAV (it is not played)")
    parser.add_argument("--estop", action="store_true", help="only send Emergency Torque OFF over REST, then exit")
    parser.add_argument("--from-robot", action="store_true",
                        help="dry run starting from the robot's measured pose and torso yaw (reads only)")
    parser.add_argument("--watch", type=float, metavar="S",
                        help="read only (REST GET): log the motor mode and head pose for S seconds, then exit")
    parser.add_argument("--pose", action="store_true",
                        help="read only: motor mode, present head pose and its distance to the sleep / neutral poses")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides={"robot": {"host": args.host}})
    setup_console_logging(cfg.logging.level)

    if args.pose:
        return read_pose(cfg)
    if args.watch:
        return watch(cfg, args.watch)

    if args.estop:
        daemon = DaemonClient(cfg.robot.host, cfg.robot.port)
        try:
            daemon.set_motor_mode("disabled", timeout_s=2.0)
            print(f"[OK] torque off requested over REST; motor mode now: {daemon.motor_mode()}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] emergency torque off: {exc!r}", file=sys.stderr)
            return 1

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown steps {unknown}; choose from {STEPS}")
    if args.live and args.fast:
        parser.error("--fast is for dry runs only")
    if args.live and args.from_robot:
        parser.error("--from-robot is for dry runs (a live run starts from the robot's pose anyway)")
    intensity = args.intensity if args.intensity is not None else (0.3 if args.live else cfg.motion.intensity)
    if not 0.0 <= intensity <= MAX_INTENSITY:  # checked before connecting (the controller would refuse it later)
        parser.error(f"--intensity must be between 0 and {MAX_INTENSITY:g}")
    overrides = {"intensity": intensity, "enabled": False}
    if args.rate:
        overrides["rate_hz"] = args.rate
    cfg = replace(cfg, motion=replace(cfg.motion, **overrides))
    runtime = prepare_runtime(cfg)

    if args.live:
        print(f"[!!] LIVE: the robot at {cfg.robot.host} will turn its motor torque ON and MOVE "
              f"(steps {steps}, intensity {intensity:g}).")
        print("     Keep hands away. Ctrl+C = Motion Stop + normal shutdown. Hard stop from another terminal:")
        print("     python -m reachy_s2s.apps.motion_preview --estop")
        if input('[??] type "move" to continue: ').strip() != "move":
            print("[..] cancelled; nothing was sent")
            return 0

    results = cfg.logging.results_path if args.live else cfg.logging.results_path / "_mock_and_tests"
    session = Session(results, f"motion_preview_{'live' if args.live else 'dry'}")
    clock = _Clock(args.fast, session.t0_ns)
    conn = recorder = None
    if args.live:
        conn = ReachyConnection(replace(cfg.robot, media_backend="no_media"), cfg.logging.level).open()
        sink: Any = RobotSink(conn.mini, conn.daemon)
        recorder = None  # needs the controller (torso frame), created below
    else:
        sink = SimulatedSink(clock=clock)
        if args.from_robot:  # read the real pose once, then simulate (nothing is sent)
            reader = ReachyConnection(replace(cfg.robot, media_backend="no_media"), cfg.logging.level).open()
            try:
                sink = simulated_like_robot(reader.mini, clock)
            finally:
                reader.close()
            print(f"[..] simulated motors start from the robot's state: {sink.pose.rounded()}, torso yaw "
                  f"{sink.body_yaw_deg:.1f} deg ({sink.origin})")
    try:
        session.write_metadata(app="motion_preview", args=vars(args), config=cfg.to_dict(), runtime=runtime,
                               robot=conn.describe() if conn is not None else None)
        ctl = MotionController(cfg, sink, session.events, session.dir, clock=clock)
        if conn is not None:
            recorder = PresentRecorder(conn.mini, session.path("present.csv"), session.t0_ns, ctl)
        state = StateTracker(session.events)
        ctl.attach(state, session.events)
        ctl.start(threads=not args.fast)
    except BaseException:
        # nothing was sent yet; closing the connection gives the daemon its media back (a no-media client
        # releases it) and lets the process exit (the SDK client threads kept it alive)
        if conn is not None:
            conn.close()
        session.close()
        raise
    runner = Runner(ctl, clock, args.live)
    if recorder is not None:
        recorder.start()
    print(f"[..] {'LIVE robot' if args.live else 'DRY RUN (simulated sink)'}; joint guard "
          f"{'on' if ctl.guard.available else 'UNAVAILABLE: ' + str(ctl.guard.error)}; session {session.dir}")
    outcome = "error"
    try:
        outcome = run_steps(runner, ctl, state, steps, args, cfg)
    except KeyboardInterrupt:
        ctl.motion_stop()
        outcome = "interrupted (Ctrl+C): motion stop"
        print("\n[!!] Motion Stop")
    finally:
        if ctl.mode not in (TORQUE_OFF, ESTOP):
            print("[..] normal shutdown: neutral -> sleep pose -> checks -> torque off")
            ctl.motor_off()
            runner.until((TORQUE_OFF, FAULT, ESTOP), 30.0)
        final = ctl.mode
        if final == FAULT:
            print(f"[FAIL] shutdown FAULT: {ctl.last_error}\n       the torque was left ON. Check the robot, then "
                  "retry, or: python -m reachy_s2s.apps.motion_preview --estop", file=sys.stderr)
        if recorder is not None:
            recorder.stop()
        ctl.close()
        if conn is not None:
            conn.close()
    path = session.path("motion.csv")
    log = read_motion_csv(path) if path.exists() else {"t": [], "poses": [], "state": [], "mode": []}
    stats = phase_stats(log, ctl.guard)
    summary = {"outcome": outcome, "final_mode": final, "intensity": intensity, "steps": steps, "live": args.live,
               "fast": args.fast, "torso_yaw_deg": round(ctl.torso_yaw_deg, 3), "totals": ctl.totals,
               "commands": stats}
    session.path("preview.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    session.close()
    print(f"[..] outcome: {outcome}; final mode {final}; totals {ctl.totals}")
    for phase, st in stats.items():
        if st.get("commands", 0) < 3:
            continue
        print(f"[..] {phase}: {st['commands']} commands in {st['seconds']} s; max deviation from neutral "
              f"{_fmt(st['max_dev'])}")
        print(f"     max speed {_fmt(st['max_speed'])}; max accel {_fmt(st['max_accel'])}")
        if st.get("max_joint_change_deg") is not None:
            print(f"     joints: max change from neutral {st['max_joint_change_deg']} deg, min motor-limit margin "
                  f"{st['min_limit_margin_deg']} deg, torso turn {st['max_body_yaw_change_deg']} deg")
    print(f"[..] session: {session.dir}")
    return 0 if final in (TORQUE_OFF, ESTOP) else 2


def _fmt(d: dict[str, Any]) -> str:
    return ", ".join(f"{k} {v:g}" for k, v in d.items() if v is not None and not math.isnan(v))


if __name__ == "__main__":
    sys.exit(main())
