"""Minimal Gemini Live check without the robot: stream recorded utterances, log what the server sends.

    python -m reachy_s2s.apps.live_probe --wav a.wav --wav b.wav [--voice Puck] [--gap 3.0]

One Live session with the cloud system's settings ([cloud] model, server VAD, transcription;
[llm] system prompt). Each 16 kHz WAV is streamed in real time in 40 ms blocks, followed by
``--gap`` seconds of silence so the server VAD can end the turn. Every server message is logged
with its local receive time (events.jsonl). Printed / saved (probe.json):

* the event kinds that actually arrived (e.g. whether ``voice_activity`` is sent at all),
* user and robot transcripts, response audio length and sample rate,
* per utterance: last voiced block sent -> first response audio received (ms),
* usage metadata if the server reports it.

Response audio goes to audio/turn_NN_robot.wav. No robot is needed. API usage: one session of
about (sum of WAV lengths + gaps) seconds of streamed audio.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.convert import to_mono
from ..audio.wavio import read_wav, write_wav
from ..config import load_config
from ..engines.gemini_client import api_key_present
from ..engines.gemini_live import (
    INPUT_MIME,
    INPUT_RATE,
    GeminiLiveConnector,
    build_live_config,
    classify_live_error,
    config_summary,
    float_to_pcm16,
    network_probe,
    parse_server_message,
    pcm16_to_float,
)
from ..evaluation.session import Session, setup_console_logging
from .common import prepare_runtime

BLOCK_MS = 40


def last_voiced_sample(x: np.ndarray, rate: int, frame_ms: int = 20, below_peak_db: float = 30.0) -> int:
    """End of the last 20 ms frame within ``below_peak_db`` of the loudest frame (approximate speech end)."""
    n = max(1, int(rate * frame_ms / 1000))
    frames = x[: len(x) // n * n].reshape(-1, n)
    if frames.size == 0:
        return len(x)
    db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12)
    voiced = np.nonzero(db > db.max() - below_peak_db)[0]
    return int((voiced[-1] + 1) * n) if voiced.size else len(x)


async def run_probe(cfg: Any, wavs: list[np.ndarray], gap_s: float, session: Session, tail_s: float) -> dict:
    from google.genai import types

    events = session.events
    config = build_live_config(cfg.cloud, cfg.llm.system_prompt)
    events.log("probe.config", model=cfg.cloud.model, config=config_summary(config))
    probe = await asyncio.get_running_loop().run_in_executor(None, network_probe)
    events.log("cloud.network_probe", **probe)
    kinds: Counter = Counter()
    result: dict[str, Any] = {"model": cfg.cloud.model, "voice": cfg.cloud.voice or "(server default)",
                              "network": probe, "turns": [], "errors": []}
    marks: list[dict[str, Any]] = []  # per utterance: when its last voiced block was sent
    responses: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_input: list[str] = []
    all_sent = asyncio.Event()

    async def sender(sess: Any) -> None:
        block = int(INPUT_RATE * BLOCK_MS / 1000)
        t_next = time.perf_counter()
        for i, x in enumerate(wavs):
            end = last_voiced_sample(x, INPUT_RATE)
            padded = np.concatenate([x, np.zeros(int(gap_s * INPUT_RATE), np.float32)])
            mark: dict[str, Any] = {"utterance": i + 1, "audio_s": round(len(x) / INPUT_RATE, 3),
                                    "speech_end_s": round(end / INPUT_RATE, 3)}
            events.log("probe.utterance_start", utterance=i + 1, audio_s=mark["audio_s"])
            for start in range(0, len(padded), block):
                t_next += BLOCK_MS / 1000
                await asyncio.sleep(max(0.0, t_next - time.perf_counter()))
                await sess.send_realtime_input(audio=types.Blob(data=float_to_pcm16(padded[start:start + block]),
                                                                mime_type=INPUT_MIME))
                if "t_speech_end_sent_ns" not in mark and start + block >= end:
                    mark["t_speech_end_sent_ns"] = time.perf_counter_ns()
                    events.log("probe.speech_end_sent", t_ns=mark["t_speech_end_sent_ns"], utterance=i + 1)
            marks.append(mark)
        all_sent.set()

    async def receiver(sess: Any) -> None:
        nonlocal current
        while True:
            async for msg in sess.receive():
                for ev in parse_server_message(msg):
                    kinds[ev.kind] += 1
                    if ev.kind == "audio":
                        if current is None:
                            current = {"t_first_audio_ns": ev.t_ns, "audio": [], "rate": ev.data["rate"],
                                       "output": "", "input": "".join(pending_input)}
                            pending_input.clear()
                            responses.append(current)
                            events.log("cloud.first_audio", t_ns=ev.t_ns, response=len(responses),
                                       rate=ev.data["rate"])
                        current["audio"].append(pcm16_to_float(ev.data["pcm"]))
                        continue
                    data = {k: v for k, v in ev.data.items() if k != "handle"}
                    if ev.kind == "resumption":
                        data["has_handle"] = bool(ev.data.get("handle"))
                    events.log(f"cloud.{ev.kind}", t_ns=ev.t_ns, **data)
                    if ev.kind == "input_transcript":  # usually streams while the user speaks
                        if current is not None:
                            current["input"] += ev.data.get("text", "")
                        else:
                            pending_input.append(ev.data.get("text", ""))
                    elif ev.kind == "output_transcript" and current is not None:
                        current["output"] += ev.data.get("text", "")
                    elif ev.kind == "usage":
                        result.setdefault("usage", []).append(ev.data)
                    elif ev.kind == "turn_complete":
                        if current is not None:
                            current["t_turn_complete_ns"] = ev.t_ns
                            current["reason"] = ev.data.get("reason")
                        current = None

    connector = GeminiLiveConnector()
    t0 = time.perf_counter_ns()
    try:
        async with connector.connect(cfg.cloud.model, config) as sess:
            events.log("cloud.connected", connect_ms=round((time.perf_counter_ns() - t0) / 1e6, 1))
            recv = asyncio.ensure_future(receiver(sess))
            send = asyncio.ensure_future(sender(sess))
            await send
            await asyncio.wait({recv}, timeout=tail_s)  # let the last answer finish
            recv.cancel()
            await asyncio.gather(recv, return_exceptions=True)
            if recv.done() and not recv.cancelled() and recv.exception() is not None:
                raise recv.exception()
    except Exception as exc:  # noqa: BLE001 - reported
        err = classify_live_error(exc)
        events.log("cloud.error", **err.to_dict())
        result["errors"].append(err.to_dict())
    result["event_kinds"] = dict(kinds)
    for i, r in enumerate(responses, 1):
        audio = np.concatenate(r["audio"]) if r["audio"] else np.zeros(0, np.float32)
        write_wav(session.audio_dir / f"turn_{i:02d}_robot.wav", audio, r["rate"])
        sent = [m for m in marks if m.get("t_speech_end_sent_ns", 1 << 62) < r["t_first_audio_ns"]]
        mark = sent[-1] if sent else None
        result["turns"].append({
            "response": i, "after_utterance": None if mark is None else mark["utterance"],
            "speech_end_sent_to_first_audio_ms": None if mark is None
            else round((r["t_first_audio_ns"] - mark["t_speech_end_sent_ns"]) / 1e6, 1),
            "audio_s": round(len(audio) / r["rate"], 3), "rate": r["rate"],
            "user_transcript": r["input"] or None, "robot_transcript": r["output"] or None,
            "turn_complete_reason": r.get("reason"),
        })
    result["utterances"] = [{k: v for k, v in m.items() if not k.startswith("t_")} for m in marks]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append",
                        help="TOML file overriding config/default.toml (repeatable, applied in order)")
    parser.add_argument("--wav", action="append", required=True, help="16 kHz utterance to send (repeatable)")
    parser.add_argument("--voice", help="override [cloud] voice (prebuilt voice name)")
    parser.add_argument("--gap", type=float, default=3.0, help="silence after each utterance (s)")
    parser.add_argument("--tail", type=float, default=12.0, help="wait this long for the last answer (s)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides={"cloud": {"voice": args.voice}})
    setup_console_logging("WARNING")
    prepare_runtime(cfg)
    if not api_key_present():
        print("[FAIL] GEMINI_API_KEY is not set (.env)", file=sys.stderr)
        return 1
    wavs = []
    for path in args.wav:
        x, rate = read_wav(path)
        if rate != INPUT_RATE:
            raise SystemExit(f"{path}: {rate} Hz, need {INPUT_RATE} Hz")
        wavs.append(to_mono(x, "left"))
    session = Session(cfg.logging.results_path, "live_probe")
    session.write_metadata(app="live_probe", args=vars(args), config=cfg.to_dict())
    try:
        result = asyncio.run(run_probe(cfg, wavs, args.gap, session, args.tail))
    finally:
        session.close()
    (session.dir / "probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[..] {session.dir}")
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
