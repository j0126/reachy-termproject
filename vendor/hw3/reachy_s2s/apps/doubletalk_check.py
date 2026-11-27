"""Double-talk diagnostic: how much of the user's voice survives in the robot mic while
the robot speaks, and whether the robot's own voice fools the VAD (before barge-in, Phase 4).

    python -m reachy_s2s.apps.doubletalk_check --config config/local_e4b_supertonic.toml
    python -m reachy_s2s.apps.doubletalk_check --robot-audio noise --no-asr
    python -m reachy_s2s.apps.doubletalk_check --reanalyse results/sessions/<id>_doubletalk   # no robot

Conditions (``--seconds`` each; the user reads the same sentence in C and D):
    A_quiet               nobody speaks, robot silent        -> noise floor
    C_user_only           user reads the sentence, robot silent
    B_robot_only[<v>]     robot speaks with voice <v>, user silent  -> residual robot echo / false triggers
    D_double_talk[<v>]    robot speaks with voice <v> while the user reads the sentence
for every robot voice in ``--robot-audio`` (default: the two real systems' voices,
``supertonic`` = System A's Supertonic M1 at its +6 dB gain and ``cloud`` = a Gemini Live
Puck answer recorded in Phase 3; also ``noise`` or ``wav:<path>``; ``gemini_tts`` uses the
free-tier TTS quota).

Per condition: mic level, Silero speech-frame ratio, level of the speech frames, clipping
rate, input gaps (Wi-Fi) and estimated lost audio, the utterances the A0 VAD would cut
([vad] settings), the interruptions the A1 barge-in detector would confirm ([cascaded]
bargein_* settings) and, with ASR, transcript + character error rate (C and D). Robot-only
conditions with VAD segments or barge-in triggers are false-interruption risks.
Outputs: session folder (WAV per condition, doubletalk.json) and results/doubletalk_check.csv.

Phase 4.5 motion conditions (``--motion dry_run | robot``; default ``off`` = the Phase 4 tool above).
The motors are turned on first (normal start: present pose -> neutral) and A-D are measured with the
head held at neutral (Motion OFF = torque ON + hold, the Phase 5 control condition), then with Motion ON:
    E_motion_only         nobody speaks; LISTENING motion with a speech-start reaction every
                          ``--nod-every`` s (injected)            -> motor noise, false speech detection
    E_user_listening      user reads the sentence; LISTENING motion, reaction from the live VAD
    F_robot_motion[<v>]   robot speaks + speaking motion, user silent  -> false barge-in triggers
    F_double_talk_motion[<v>]  robot speaks + speaking motion while the user reads -> barge-in detection
Extra per condition: motion control (tick intervals, commands, reactions, live VAD speech starts) and,
on the robot, the commanded-vs-measured head pose error. ``robot`` drives the real motors: it asks you
to type "move" and ends with the normal shutdown (never cuts the torque on a failed check). ``dry_run``
uses simulated motors (no motor command). Motion runs go to results/doubletalk_motion.csv
(``--mock``: results/_mock_and_tests/), so the Phase 4 table keeps its columns.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.convert import clipping_rate, rms_dbfs, to_mono
from ..audio.resample import resample
from ..audio.wavio import read_wav, write_wav
from ..config import PROJECT_ROOT, load_config
from ..conversation.state import InteractionState, StateTracker
from ..engines.vad import FRAME, SAMPLE_RATE, BargeInDetector, SileroVadModel, VadWorker, segment_probs
from ..evaluation.session import Session, setup_console_logging
from ..evaluation.tables import append_csv_row
from ..motion.controller import ESTOP, FAULT, HOLD, TORQUE_OFF, MotionController
from ..motion.pose import pose_from_matrix
from ..motion.sinks import PresentSampler, RobotSink, SimulatedSink
from ..robot.audio_manager import AudioManager
from .common import add_common_args, open_backend, prepare_runtime

DEFAULT_PROMPT = "오늘 날씨가 정말 좋아서 공원으로 산책을 가고 싶어요."
ROBOT_SENTENCE = ("안녕하세요. 저는 리치 미니입니다. 지금은 이중 발화 시험을 하고 있어요. "
                  "제 말이 끝날 때까지 계속 읽어 주세요.")
CLOUD_SAMPLE = PROJECT_ROOT / "results" / "sessions" / "20261017_010608_live_probe" / "audio" / "turn_01_robot.wav"
MOTION_MODES = ("off", "dry_run", "robot")
CONDITION_RE = re.compile(r"(A_quiet|C_user_only|B_robot_only|D_double_talk|E_motion_only|E_user_listening|"
                          r"F_robot_motion|F_double_talk_motion)(?:_(.+)_)?")
USER_SPEAKS = ("C_", "D_", "E_user", "F_double")  # conditions where the user reads the sentence (ASR / CER)
NOD_RELEASE_S = 0.6  # injected speech start -> speech end (E_motion_only)


def speech_like_noise(seconds: float, rate: int, seed: int = 0) -> np.ndarray:
    """Pink noise with a 4 Hz syllable-like envelope at about -20 dBFS RMS."""
    n = int(seconds * rate)
    spec = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / rate)
    spec[1:] /= np.sqrt(f[1:])
    spec[(f < 100) | (f > 5000)] = 0
    x = np.fft.irfft(spec, n)
    t = np.arange(n) / rate
    x *= (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)) ** 2
    return (x / (np.sqrt(np.mean(x * x)) + 1e-12) * 0.1).astype(np.float32)


def cer(reference: str, hypothesis: str) -> float:
    norm = lambda s: re.sub(r"[\s\.,!?~\-]", "", s)  # noqa: E731
    ref, hyp = norm(reference), norm(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def analyse(mono: np.ndarray, vad: Any, cfg: Any | None = None) -> dict[str, Any]:
    """Level / VAD statistics of one condition; with ``cfg`` also A0 segments and A1 barge-in triggers."""
    probs = vad.probs(mono)
    n = probs.shape[0]
    frames = mono[: n * FRAME].reshape(n, FRAME) if n else np.zeros((0, FRAME), np.float32)
    speech = probs >= 0.5
    quiet = probs < 0.1
    stats: dict[str, Any] = {
        "rms_dbfs": round(rms_dbfs(mono), 2),
        "speech_frame_ratio": round(float(speech.mean()) if n else 0.0, 3),
        "mean_speech_prob": round(float(probs.mean()) if n else 0.0, 3),
        "max_speech_prob": round(float(probs.max()) if n else 0.0, 3),
        "speech_frames_rms_dbfs": round(rms_dbfs(frames[speech]), 2) if speech.any() else None,
        # median level of clearly non-speech frames: a floor that survives someone talking in the condition
        "non_speech_dbfs": round(float(np.median([rms_dbfs(f) for f in frames[quiet]])), 2) if quiet.any() else None,
        "clipping_rate": round(clipping_rate(mono), 6),
    }
    if cfg is not None:
        segs = segment_probs(probs, cfg.vad)
        stats.update(vad_segments=len(segs),
                     vad_speech_s=round(sum(s.end - s.start for s in segs) / SAMPLE_RATE, 2),
                     vad_first_start_s=round(segs[0].start / SAMPLE_RATE, 2) if segs else None)
        c = cfg.cascaded
        hits = BargeInDetector(c.bargein_threshold, cfg.vad.neg_threshold, c.bargein_min_speech_ms,
                               c.bargein_reset_silence_ms).scan(probs)
        stats.update(bargein_triggers=len(hits),
                     bargein_first_s=round(hits[0][1] / SAMPLE_RATE, 2) if hits else None,
                     bargein_first_delay_ms=round((hits[0][1] - hits[0][0]) * 1000 / SAMPLE_RATE) if hits else None)
    return stats


def robot_sources(names: list[str], cfg: Any, rate: int, seconds: float) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for name in names:
        if name == "supertonic":
            from ..engines.tts_supertonic import SupertonicTTS

            tcfg = cfg.tts if cfg.tts.provider == "supertonic" else replace(
                cfg.tts, provider="supertonic", model="supertonic-3", voice="M1", gain_db=6.0)
            result = SupertonicTTS(tcfg).synthesize(ROBOT_SENTENCE)
            out.append((f"supertonic_{tcfg.voice}", resample(result.audio, result.sample_rate, rate)))
        elif name == "cloud":
            x, sr = read_wav(CLOUD_SAMPLE)
            out.append(("cloud_puck", resample(to_mono(x), sr, rate)))
        elif name == "noise":
            out.append(("noise", speech_like_noise(seconds, rate)))
        elif name == "gemini_tts":
            from ..engines.tts import GeminiTTS

            result = GeminiTTS(cfg.tts).synthesize(ROBOT_SENTENCE)
            out.append(("gemini_tts", resample(result.audio, result.sample_rate, rate)))
        elif name.startswith("wav:"):
            x, sr = read_wav(name[4:])
            out.append((Path(name[4:]).stem, resample(to_mono(x), sr, rate)))
        else:
            raise SystemExit(f"unknown robot audio {name!r} (supertonic, cloud, noise, gemini_tts, wav:<path>)")
    return out


def summarize(conditions: dict[str, Any], labels: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Per robot voice: residual echo, false-trigger risk, user speech under double talk."""
    a, c = conditions["A_quiet"], conditions["C_user_only"]
    warnings = []
    if a["vad_segments"]:
        warnings.append("A_quiet contains speech: its overall level is not a noise floor (use non_speech_dbfs)")
    floor = a["non_speech_dbfs"] if a.get("non_speech_dbfs") is not None else a["rms_dbfs"]
    summary: dict[str, Any] = {}
    for label in labels:
        b, d = conditions[f"B_robot_only[{label}]"], conditions[f"D_double_talk[{label}]"]
        s: dict[str, Any] = {
            # B overall level over A's non-speech floor; can be negative: the XVF3800 attenuates the mic
            # while the robot plays (measured 2026-10-17)
            "robot_residual_over_floor_db": round(b["rms_dbfs"] - floor, 2),
            "robot_only_max_speech_prob": b.get("max_speech_prob"),
            "robot_only_vad_segments": b["vad_segments"], "robot_only_bargein_triggers": b["bargein_triggers"],
            "speech_ratio_user_only": c["speech_frame_ratio"], "speech_ratio_double_talk": d["speech_frame_ratio"],
            "double_talk_bargein_first_s": d["bargein_first_s"], "cer_user_only": c.get("cer"),
            "cer_double_talk": d.get("cer"),
        }
        if c.get("speech_frames_rms_dbfs") is not None and d.get("speech_frames_rms_dbfs") is not None:
            s["user_speech_attenuation_db"] = round(c["speech_frames_rms_dbfs"] - d["speech_frames_rms_dbfs"], 2)
        summary[label] = s
    return summary, warnings


def summarize_motion(conditions: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    """Motion ON (E / F) against the same session's Motion OFF hold conditions (A / C / B / D)."""
    a, c = conditions["A_quiet"], conditions["C_user_only"]
    e1, e2 = conditions["E_motion_only"], conditions["E_user_listening"]
    floor = a["non_speech_dbfs"] if a.get("non_speech_dbfs") is not None else a["rms_dbfs"]
    c_ratio = c["speech_frame_ratio"]
    out: dict[str, Any] = {"listening": {
        "motion_noise_over_floor_db": round(e1["rms_dbfs"] - floor, 2),
        "motion_only_max_speech_prob": e1.get("max_speech_prob"), "motion_only_vad_segments": e1["vad_segments"],
        "motion_only_bargein_triggers": e1["bargein_triggers"],
        "motion_only_live_vad_starts": (e1.get("motion") or {}).get("live_vad_starts"),
        "user_speech_ratio_hold": c_ratio, "user_speech_ratio_listening_motion": e2["speech_frame_ratio"],
        "user_detection_ratio": round(e2["speech_frame_ratio"] / c_ratio, 3) if c_ratio else None,
        "user_utterance_detected": e2["vad_segments"] >= 1,
        "cer_hold": c.get("cer"), "cer_listening_motion": e2.get("cer"),
        "reactions_motion_only": (e1.get("motion") or {}).get("gestures"),
        "reactions_user": (e2.get("motion") or {}).get("gestures"),
    }}
    for label in labels:
        b, d = conditions[f"B_robot_only[{label}]"], conditions[f"D_double_talk[{label}]"]
        f1, f2 = conditions[f"F_robot_motion[{label}]"], conditions[f"F_double_talk_motion[{label}]"]
        out[label] = {
            "speaking_motion_over_robot_only_db": round(f1["rms_dbfs"] - b["rms_dbfs"], 2),
            "robot_only_bargein_triggers_hold": b["bargein_triggers"],
            "robot_motion_bargein_triggers": f1["bargein_triggers"], "robot_motion_vad_segments": f1["vad_segments"],
            "robot_motion_max_speech_prob": f1.get("max_speech_prob"),
            "speech_ratio_double_talk_hold": d["speech_frame_ratio"],
            "speech_ratio_double_talk_motion": f2["speech_frame_ratio"],
            "bargein_first_s_hold": d["bargein_first_s"], "bargein_first_s_motion": f2["bargein_first_s"],
            "bargein_delay_ms_hold": d["bargein_first_delay_ms"], "bargein_delay_ms_motion": f2["bargein_first_delay_ms"],
            "cer_double_talk_hold": d.get("cer"), "cer_double_talk_motion": f2.get("cer"),
        }
    ctrl = [cond["motion"] for name, cond in conditions.items() if name.startswith(("E_", "F_")) and cond.get("motion")]
    if ctrl:
        out["control"] = {
            "tick_hz_min": min((m["tick_hz"] for m in ctrl if m.get("tick_hz") is not None), default=None),
            "tick_interval_max_ms": max((m["tick_interval_max_ms"] for m in ctrl
                                         if m.get("tick_interval_max_ms") is not None), default=None),
            "late_ticks": sum(m.get("late_ticks") or 0 for m in ctrl),
            "track_err_deg_max": max((m["track_err_deg_max"] for m in ctrl
                                      if m.get("track_err_deg_max") is not None), default=None),
            "simulated": ctrl[0].get("simulated"),
        }
    return out


class MotionRig:
    """Motors + controller + live VAD for the motion conditions (dry run: simulated motors only)."""

    def __init__(self, mode: str, cfg: Any, conn: Any, audio: AudioManager, session: Session,
                 intensity: float | None) -> None:
        self.mode = mode
        mcfg = replace(cfg.motion, enabled=False, intensity=cfg.motion.intensity if intensity is None else intensity)
        self.cfg = replace(cfg, motion=mcfg)
        self.sink: Any = SimulatedSink() if mode == "dry_run" else RobotSink(conn.mini, conn.daemon)
        self.audio = audio
        self.session = session
        self.ticks: deque[int] = deque(maxlen=500_000)
        self.cmds: deque[tuple[int, Any]] = deque(maxlen=500_000)
        self.seen: list[tuple[int, dict[str, Any]]] = []  # (perf_counter_ns, record) of reactions / VAD starts
        send = self.sink.send

        def logged_send(head: Any, antennas: list[float]) -> None:
            send(head, antennas)  # world frame
            self.cmds.append((time.perf_counter_ns(), pose_from_matrix(self.ctl.to_local(head), antennas)))

        self.sink.send = logged_send
        self.ctl = MotionController(self.cfg, self.sink, session.events, session.dir)
        tick = self.ctl.tick

        def logged_tick(now: int | None = None) -> None:
            self.ticks.append(time.perf_counter_ns())
            tick(now)

        self.ctl.tick = logged_tick  # type: ignore[method-assign]
        self.state = StateTracker(session.events)
        self.ctl.attach(self.state, session.events)
        session.events.add_listener(self._watch)
        self.present = PresentSampler(self.sink)
        self.vad = VadWorker(audio, cfg.vad, session.events, on_utterance=lambda _utt: None)
        self.started = False

    def _watch(self, record: dict[str, Any]) -> None:
        if record["kind"] in ("motion.gesture", "vad.speech_start"):
            self.seen.append((time.perf_counter_ns(), record))

    def start(self) -> None:
        self.audio.add_playback_tap(self.ctl.tap)
        self.ctl.start()
        self.started = True
        self.ctl.motor_on()
        if not self.ctl.wait_for((HOLD, FAULT, ESTOP), 30.0) or self.ctl.mode != HOLD:
            raise RuntimeError(f"motors did not reach the neutral hold ({self.ctl.mode}): "
                               f"{self.ctl.needs_confirmation or self.ctl.last_error}")
        self.present.start()
        self.vad.start()

    def phase(self, phase: str) -> None:
        """hold | listening | speaking (before a condition's countdown, so the move is not recorded)."""
        if phase == "hold":
            self.state.set(InteractionState.LISTENING, "doubletalk")
            self.ctl.set_motion(False)
        else:
            self.ctl.set_motion(True)
            self.state.set(InteractionState.LISTENING if phase == "listening" else InteractionState.SPEAKING,
                           "doubletalk")

    def inject_reactions(self, seconds: float, every: float) -> threading.Thread:
        """E_motion_only: a speech start / end pair every ``every`` s (the reaction the user's speech would cause)."""
        def run() -> None:
            t_end = time.perf_counter() + seconds
            while time.perf_counter() + NOD_RELEASE_S < t_end:
                self.session.events.log("vad.speech_start", source="injected")
                time.sleep(NOD_RELEASE_S)
                self.session.events.log("vad.speech_end", source="injected")
                time.sleep(max(0.0, every - NOD_RELEASE_S))

        th = threading.Thread(target=run, name="doubletalk-reactions", daemon=True)
        th.start()
        return th

    def window(self, t0: int, t1: int) -> dict[str, Any]:
        """Motion control during one condition."""
        inside = [t for t in self.ticks if t0 <= t <= t1]
        iv = np.diff(inside) / 1e6 if len(inside) > 1 else np.zeros(0)
        period_ms = 1000.0 / self.ctl.rate_hz
        sent = [(t, pose) for t, pose in self.cmds if t0 <= t <= t1]
        out: dict[str, Any] = {
            "mode": self.ctl.mode, "state": self.ctl.engine.state, "label": self.ctl.label(),
            "simulated": self.sink.simulated,
            "tick_hz": round((len(inside) - 1) / ((inside[-1] - inside[0]) / 1e9), 1) if len(inside) > 2 else None,
            "tick_interval_p95_ms": round(float(np.percentile(iv, 95)), 1) if iv.size else None,
            "tick_interval_max_ms": round(float(iv.max()), 1) if iv.size else None,
            "late_ticks": int(np.sum(iv > 1.5 * period_ms)) if iv.size else None,
            "commands": len(sent),
            "gestures": sum(1 for t, r in self.seen if t0 <= t <= t1 and r["kind"] == "motion.gesture"),
            "live_vad_starts": sum(1 for t, r in self.seen if t0 <= t <= t1 and r["kind"] == "vad.speech_start"
                                   and r.get("source") != "injected"),
        }
        if sent:
            dev = np.max(np.abs(np.stack([p.vec() for _t, p in sent])[:, :3]), axis=0)
            out["max_roll_pitch_yaw_deg"] = [round(float(v), 2) for v in dev]
        cmd_t = [t for t, _p in self.cmds]
        errs = []
        for t, present in self.present.samples:
            if t0 <= t <= t1:
                i = bisect.bisect_right(cmd_t, t) - 1
                if i >= 0:
                    local = pose_from_matrix(self.ctl.to_local(present.head_matrix()), present.antennas)
                    errs.append(float(np.max(np.abs((local - self.cmds[i][1]).vec()[:3]))))
        if errs:
            out.update(track_err_deg_median=round(float(np.median(errs)), 2),
                       track_err_deg_p95=round(float(np.percentile(errs, 95)), 2),
                       track_err_deg_max=round(max(errs), 2))
        return out

    def shutdown(self) -> str:
        """Normal shutdown (neutral -> sleep -> checks -> torque off). Never cuts the torque on a failure."""
        self.vad.stop()
        self.present.stop()
        if self.started and self.ctl.mode not in (TORQUE_OFF, ESTOP):
            self.state.set(InteractionState.LISTENING, "doubletalk end")
            self.ctl.motor_off()
            self.ctl.wait_for((TORQUE_OFF, FAULT, ESTOP), 40.0)
        final = self.ctl.mode
        self.audio.remove_playback_tap(self.ctl.tap)
        self.session.events.remove_listener(self._watch)
        self.ctl.close()
        return final


def reanalyse(session_dir: Path, cfg: Any, use_asr: bool) -> int:
    """Recompute doubletalk.json's statistics from the saved WAVs (no robot), e.g. with other settings."""
    old = json.loads((session_dir / "doubletalk.json").read_text(encoding="utf-8"))
    prompt = old.get("prompt", DEFAULT_PROMPT)
    vad = SileroVadModel()
    asr = None
    if use_asr:
        from ..engines.asr import WhisperASR

        asr = WhisperASR(cfg.asr)
        asr.load()
    conditions: dict[str, Any] = {}
    labels: list[str] = []
    for wav in sorted((session_dir / "audio").glob("*.wav")):
        m = CONDITION_RE.fullmatch(wav.stem)
        if not m:
            continue
        name = m.group(1) if m.group(2) is None else f"{m.group(1)}[{m.group(2)}]"
        if m.group(1) == "B_robot_only":
            labels.append(m.group(2))
        x, rate = read_wav(wav)
        mono = to_mono(x, cfg.audio.mic_channel)
        stats = analyse(mono, vad, cfg)
        if asr is not None and name.startswith(USER_SPEAKS):
            text = asr.transcribe(mono, rate).text
            stats.update(transcript=text, cer=round(cer(prompt, text), 3))
        old_motion = (old.get("conditions", {}).get(name) or {}).get("motion")
        if old_motion is not None:  # recorded live, cannot be recomputed from the WAV
            stats["motion"] = old_motion
        conditions[name] = stats
    summary, warnings = summarize(conditions, labels)
    motion_summary = summarize_motion(conditions, labels) if "E_motion_only" in conditions else None
    out = {"source_session": session_dir.name, "reanalysed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "prompt": prompt, "warnings": warnings,
           "vad": {"threshold": cfg.vad.threshold, "silence_timeout_ms": cfg.vad.silence_timeout_ms},
           "bargein": {k: getattr(cfg.cascaded, k) for k in
                       ("bargein_threshold", "bargein_min_speech_ms", "bargein_reset_silence_ms")},
           "conditions": conditions, "summary": summary, "motion_summary": motion_summary}
    path = session_dir / "doubletalk_reanalysis.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for w in warnings:
        print(f"[!!] {w}")
    for name, stats in conditions.items():
        print(f"    {name}: {json.dumps(stats, ensure_ascii=False)}")
    for label, s in summary.items():
        print(f"    {label}: {json.dumps(s, ensure_ascii=False)}")
    if motion_summary:
        for label, s in motion_summary.items():
            print(f"    motion {label}: {json.dumps(s, ensure_ascii=False)}")
    print(f"[..] {path}")
    return 0


def countdown(text: str, seconds: float = 3.0) -> None:
    print(f"\n>>> {text}")
    steps = max(0, int(round(seconds)))
    for k in range(steps, 0, -1):
        print(f"    {k} ...", flush=True)
        time.sleep(seconds / steps)
    print("    GO", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--seconds", type=float, default=6.0, help="length of each condition (default 6)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="sentence the user reads in C and D")
    parser.add_argument("--robot-audio", default="supertonic,cloud",
                        help="comma list: supertonic, cloud, noise, gemini_tts, wav:<path>")
    parser.add_argument("--no-asr", action="store_true", help="skip transcribing C and D")
    parser.add_argument("--reanalyse", type=Path, metavar="SESSION_DIR",
                        help="recompute the statistics of a saved run from its WAVs (no robot; current settings)")
    parser.add_argument("--motion", choices=MOTION_MODES, default="off",
                        help="add the Motion ON conditions E / F: dry_run (simulated motors) or robot (real motors, "
                             "asks to type 'move'); off = the Phase 4 tool")
    parser.add_argument("--intensity", type=float, help="motion intensity for E / F (default: [motion] intensity)")
    parser.add_argument("--nod-every", type=float, default=1.5,
                        help="E_motion_only: one speech-start reaction every S seconds (default 1.5)")
    parser.add_argument("--countdown", type=float, default=3.0, help="countdown before each condition (s)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides={"robot": {"host": args.host}})
    setup_console_logging(cfg.logging.level)
    runtime = prepare_runtime(cfg)
    if args.reanalyse:
        return reanalyse(args.reanalyse, cfg, use_asr=not args.no_asr)
    motion = args.motion != "off"
    if args.motion == "robot":
        if args.mock:
            raise SystemExit("--motion robot drives the real motors: it cannot run with --mock")
        intensity = cfg.motion.intensity if args.intensity is None else args.intensity
        print(f"[!!] --motion robot: the robot at {cfg.robot.host} turns its motor torque ON and MOVES "
              f"(intensity {intensity:g}); it ends with the normal shutdown. Hard stop from another terminal:\n"
              "     python -m reachy_s2s.apps.motion_preview --estop")
        if input('[??] type "move" to continue: ').strip() != "move":
            print("[..] cancelled; nothing was started")
            return 0
    results = cfg.logging.results_path / "_mock_and_tests" if (motion and args.mock) else cfg.logging.results_path
    session = Session(results, "doubletalk_motion" if motion else "doubletalk")
    conn = audio = rig = None
    report: dict[str, Any] = {"session_id": session.id, "prompt": args.prompt, "ok": False,
                              "vad": {"threshold": cfg.vad.threshold, "silence_timeout_ms": cfg.vad.silence_timeout_ms},
                              "bargein": {k: getattr(cfg.cascaded, k) for k in
                                          ("bargein_threshold", "bargein_min_speech_ms", "bargein_reset_silence_ms")},
                              "motion": args.motion}
    try:
        vad = SileroVadModel()
        asr = None
        if not args.no_asr:
            from ..engines.asr import WhisperASR

            asr = WhisperASR(cfg.asr)
            report["asr"] = asr.load()
        backend, conn, robot_info = open_backend(cfg, args.mock)
        session.write_metadata(app="doubletalk_check", args=vars(args), config=cfg.to_dict(), robot=robot_info,
                               runtime=runtime)
        audio = AudioManager(backend, cfg.audio, session.events)
        audio.start()
        rate = backend.output_rate
        if motion:
            rig = MotionRig(args.motion, cfg, conn, audio, session, args.intensity)
            report["motion_config"] = {"intensity": rig.cfg.motion.intensity, "rate_hz": rig.cfg.motion.rate_hz,
                                       "speaking_motion": rig.cfg.motion.speaking_motion,
                                       "speech_started_gesture": rig.cfg.motion.speech_started_gesture,
                                       "sink": rig.sink.name, "nod_every_s": args.nod_every}
            print("[..] motors on: present pose -> neutral (Motion OFF = hold) ...")
            rig.start()
        sources = robot_sources([s.strip() for s in args.robot_audio.split(",") if s.strip()], cfg, rate,
                                args.seconds)
        report["robot_audio"] = {label: {"seconds": round(x.shape[0] / rate, 2), "rms_dbfs": round(rms_dbfs(x), 2),
                                          "peak": round(float(np.max(np.abs(x))), 3)} for label, x in sources}
        report["robot_audio_output_gain_db"] = cfg.audio.output_gain_db
        for label, x in sources:
            write_wav(session.audio_dir / f"robot_signal_{label}.wav", x, rate)
        # The sentence is shown only where it is read: shown earlier, people read it during A_quiet (2026-10-17).
        hold = "hold" if motion else None  # A-D: with motion, the head is held at neutral (Motion OFF)
        plan: list[tuple[str, str, np.ndarray | None, str | None]] = [
            ("A_quiet", "아무 말도 하지 마세요 (잡음 측정). 읽을 문장은 다음 단계에서 보여 드립니다", None, hold),
            ("C_user_only", f"문장을 읽어 주세요: {args.prompt}", None, hold),
        ]
        for label, x in sources:
            plan.append((f"B_robot_only[{label}]", "로봇만 말합니다. 조용히 계세요", x, hold))
            plan.append((f"D_double_talk[{label}]", f"로봇이 말하는 동안 같은 문장을 읽어 주세요: {args.prompt}", x, hold))
        if motion:
            plan.append(("E_motion_only", "아무 말도 하지 마세요. 로봇이 듣는 동작(반응 포함)을 합니다 (모터 소음 측정)",
                         None, "listening"))
            plan.append(("E_user_listening", f"로봇이 듣는 동작을 하는 동안 문장을 읽어 주세요: {args.prompt}", None,
                         "listening"))
            for label, x in sources:
                plan.append((f"F_robot_motion[{label}]", "로봇이 말하면서 움직입니다. 조용히 계세요", x, "speaking"))
                plan.append((f"F_double_talk_motion[{label}]",
                             f"로봇이 말하며 움직이는 동안 같은 문장을 읽어 주세요: {args.prompt}", x, "speaking"))
        conditions: dict[str, Any] = {}
        for name, instruction, robot, phase in plan:
            if rig is not None:
                rig.phase(phase)
            countdown(instruction, args.countdown)
            session.events.log("doubletalk.condition", name=name, robot_plays=robot is not None, motion_phase=phase)
            rec_name = re.sub(r"[^A-Za-z0-9_]", "_", name)
            t_cond0 = time.perf_counter_ns()
            audio.start_recording(rec_name)
            handle = audio.play(robot, rate, label=f"doubletalk_{rec_name}") if robot is not None else None
            injector = rig.inject_reactions(args.seconds, args.nod_every) \
                if rig is not None and name == "E_motion_only" else None
            time.sleep(args.seconds)
            if handle is not None:
                audio.stop_playback("condition over")
            rec = audio.stop_recording()
            t_cond1 = time.perf_counter_ns()
            if injector is not None:
                injector.join(timeout=2)
            rec.save(session.audio_dir, session.t0_ns)
            mono = rec.mono(cfg.audio.mic_channel)
            stats = analyse(mono, vad, cfg)
            gaps, gap_ms = audio.gaps_between(rec.start_index, rec.start_index + rec.num_frames)
            wall_s = (rec.t_stop_ns - rec.t_request_ns) / 1e9 if rec.t_stop_ns else None
            stats.update(input_gaps=gaps, input_gap_ms=gap_ms,
                         lost_s_est=None if wall_s is None else round(max(0.0, wall_s - rec.duration_s), 2))
            if asr is not None and name.startswith(USER_SPEAKS):
                result = asr.transcribe(mono)
                stats.update(transcript=result.text, cer=round(cer(args.prompt, result.text), 3))
            if rig is not None:
                stats["motion"] = rig.window(t_cond0, t_cond1)
            conditions[name] = stats
            print(f"    {name}: {json.dumps(stats, ensure_ascii=False)}")
        report["conditions"] = conditions
        summary, warnings = summarize(conditions, [label for label, _x in sources])
        for warning in warnings:
            print(f"[!!] {warning}")
        report["warnings"] = warnings
        report["summary"] = summary
        if motion:
            report["motion_summary"] = summarize_motion(conditions, [label for label, _x in sources])
        report["ok"] = True
        print("\n[OK] summary per robot voice:")
        for label, s in summary.items():
            print(f"    {label}: {json.dumps(s, ensure_ascii=False)}")
        for label, s in (report.get("motion_summary") or {}).items():
            print(f"    motion {label}: {json.dumps(s, ensure_ascii=False)}")
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = repr(exc)
        print(f"\n[FAIL] {exc!r}", file=sys.stderr)
        if rig is not None:
            rig.ctl.motion_stop()
    finally:
        if rig is not None:
            if audio is not None:
                audio.stop_playback("doubletalk end")
            final = rig.shutdown()
            report["motion_final_mode"] = final
            print(f"[..] motors: {final}")
            if final == FAULT:
                print(f"[!!] normal shutdown FAULT: {rig.ctl.last_error}\n     the torque was left ON. Check the robot, "
                      "then: python -m reachy_s2s.apps.motion_preview --estop", file=sys.stderr)
        if audio is not None:
            audio.stop()
        if conn is not None:
            conn.close()
        (session.dir / "doubletalk.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        if motion:
            write_motion_rows(report, session, results, args)
            session.close()
            print(f"[..] session: {session.dir}")
            return 0 if report["ok"] else 1
        for label, s in (report.get("summary") or {"(none)": {}}).items():
            append_csv_row(cfg.logging.results_path / "doubletalk_check.csv", {
                "session_id": session.id, "ok": report["ok"], "robot_voice": label,
                **{k: s.get(k) for k in ("robot_residual_over_floor_db", "robot_only_max_speech_prob",
                                         "robot_only_vad_segments",
                                         "robot_only_bargein_triggers", "speech_ratio_user_only",
                                         "speech_ratio_double_talk", "user_speech_attenuation_db",
                                         "double_talk_bargein_first_s", "cer_user_only", "cer_double_talk")},
                "error": report.get("error", ""),
            })
        session.close()
        print(f"[..] session: {session.dir}")
    return 0 if report["ok"] else 1


def write_motion_rows(report: dict[str, Any], session: Session, results: Path, args: argparse.Namespace) -> None:
    """One row per robot voice in doubletalk_motion.csv (the Phase 4 doubletalk_check.csv keeps its columns)."""
    ms = report.get("motion_summary") or {}
    listening = ms.get("listening", {})
    control = ms.get("control", {})
    cfg_m = report.get("motion_config", {})
    voices = [k for k in ms if k not in ("listening", "control")] or ["(none)"]
    for label in voices:
        append_csv_row(results / ("doubletalk_motion_mock.csv" if args.mock else "doubletalk_motion.csv"), {
            "session_id": session.id, "ok": report["ok"], "motion": args.motion,
            "intensity": cfg_m.get("intensity"), "rate_hz": cfg_m.get("rate_hz"),
            "speech_started_gesture": cfg_m.get("speech_started_gesture"), "robot_voice": label,
            **{f"listening_{k}": v for k, v in listening.items()},
            **{k: v for k, v in (ms.get(label) or {}).items()},
            **{f"control_{k}": v for k, v in control.items()},
            "final_mode": report.get("motion_final_mode"), "error": report.get("error", ""),
        })


if __name__ == "__main__":
    sys.exit(main())
