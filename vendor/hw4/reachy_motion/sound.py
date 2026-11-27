"""Effect sounds of the expressive motions (user decision 2026-11-14: "로봇 스피커로"). Qt-free.

The signed-off motion files name a sound cue (``SoundCue``: file + time in the motion): aha_recommend expressive
``aha.wav`` at 0.85 s (the pop), success expressive ``success.wav`` at 0.3 s (the pop), and since revision 2
(2026-11-14, user decision "둘 다 넣기") failure expressive ``failure.wav`` at 0.2 s (the head drops) and boredom
expressive ``boredom.wav`` at 4.9 s (the slump). Neutral variants have none (the assignment: "no antenna movement or
sound"). The sounds follow the motions' V/A design targets: valence -> pitch direction and mode (rising / major for
the positive cues, falling / minor or a downward slide for the negative ones), arousal -> tempo, register and
loudness (aha / success fast, high, -6 dBFS; failure slower, an octave lower, -8 dBFS; boredom slow, low, -10 dBFS).
The audio is synthesised here (original - no third-party audio) into ``motions/sounds/``; the team may replace a
file with its own under the same name.

Where the sound comes out (``SoundSettings.output``):

* ``"robot"``  the robot's built-in speaker through the SDK 1.11.0 daemon REST API: the file is uploaded once per
  connection (``POST /api/media/sounds/upload``, multipart field ``file``; it answers the daemon path) and played at
  the cue (``POST /api/media/play_sound`` ``{"file": <daemon path>}``) - daemon/app/routers/media.py:77-103 and
  207-262 (site-packages reachy_mini). The daemon plays it with its own GStreamer playbin (media/media_server.py:
  1344-1390), also while our connection has released the media pipeline (that stop only touches the sender pipeline,
  media_server.py:1339-1342) - not tried on the robot yet. A daemon sound would drive the head wobbler when wobbling
  is on (media_server.py:1387-1389); our connection turns wobbling off (backends/robot_connect.py:311, 409).
  ``GET /api/media/status`` says whether the daemon has media at all (``no_media``: then nothing plays).
* ``"laptop"`` this PC's speaker (``winsound``, Windows): the fallback, and the only output of MuJoCo previews.
* ``"off"``    nothing (the default of every session: tests and offline runs never send a sound).

Timing: the cue fires from the executor's ``on_tick`` (after each send, in the executor thread) at the first tick
whose time reaches ``pre_s + at_s + offset`` of the stream; the request then runs in its own thread, so the 50 Hz
loop never waits for it. ``offset_ms`` lets the team move the sound against the motion it sees (on the robot the head
follows its command about 0.19 s late, measured 2026-11-12; the speaker path's own delay is not measured).
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import urllib.request
import uuid
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

RATE = 44100
SOUND_OUTPUTS = ("robot", "laptop", "off")
OUTPUT_KO = {"robot": "로봇 스피커", "laptop": "노트북 스피커", "off": "끄기"}
UPLOAD_PREFIX = "hw4_"  # the daemon keeps uploads in /tmp/reachy_mini_sounds/<name>: ours stay apart
UPLOAD_TIMEOUT_S = 10.0  # the daemon probes an upload with GStreamer (up to 5 s, routers/media.py:244-247)
PLAY_TIMEOUT_S = 3.0
OFFSET_RANGE_MS = (-500, 800)


@dataclass
class SoundSettings:
    """Where the effect sounds go; shared by every session of one Motion Lab window (``LabSessions.sound``)."""

    output: str = "off"
    offset_ms: float = 0.0
    sounds_dir: Path = field(default_factory=lambda: Path("motions") / "sounds")


# ---------------------------------------------------------------------------------------------- synthesis
def _bell(t: np.ndarray, start: float, freq: float, tau: float, amp: float) -> np.ndarray:
    """A plucked bell tone from ``start``: fundamental + two soft partials, 4 ms attack, exponential decay."""
    u = t - start
    on = u >= 0.0
    env = np.where(on, (1.0 - np.exp(-np.maximum(u, 0.0) / 0.004)) * np.exp(-np.maximum(u, 0.0) / tau), 0.0)
    tone = (np.sin(2 * np.pi * freq * u) + 0.35 * np.sin(2 * np.pi * 2 * freq * u)
            + 0.12 * np.sin(2 * np.pi * 3.01 * freq * u))
    return amp * env * tone


def _finish(x: np.ndarray, peak: float = 0.5) -> np.ndarray:
    """Normalise to ``peak`` (-6 dBFS) and fade the last 30 ms out (no click at the end)."""
    x = x * (peak / float(np.max(np.abs(x))))
    n = int(0.03 * RATE)
    x[-n:] *= np.linspace(1.0, 0.0, n)
    return x


def synth_aha(rate: int = RATE) -> np.ndarray:
    """'Aha!': two quick rising bell tones (E6 then B6) - the light-bulb moment of the pop (0.5 s)."""
    t = np.arange(int(0.5 * rate)) / rate
    return _finish(_bell(t, 0.0, 1318.5, 0.10, 0.8) + _bell(t, 0.085, 1975.5, 0.16, 1.0))


def synth_success(rate: int = RATE) -> np.ndarray:
    """'Success': a bright rising major arpeggio C6-E6-G6-C7, the last note ringing (0.6 s)."""
    t = np.arange(int(0.6 * rate)) / rate
    x = sum(_bell(t, i * 0.07, f, 0.09, 0.8) for i, f in enumerate((1046.5, 1318.5, 1568.0)))
    return _finish(x + _bell(t, 0.21, 2093.0, 0.22, 1.0))


def _glide(t: np.ndarray, start: float, f0: float, f1: float, glide_s: float, attack: float, tau: float,
           partials: tuple[tuple[int, float], ...]) -> np.ndarray:
    """A tone from ``start`` whose pitch slides exponentially from ``f0`` to ``f1`` in ``glide_s`` (then stays),
    with the given (harmonic, weight) partials, an ``attack`` rise and an exponential decay ``tau``."""
    u = np.maximum(t - start, 0.0)
    k = math.log(f1 / f0) / glide_s
    phase = np.where(u <= glide_s, f0 * np.expm1(k * u) / k if k != 0.0 else f0 * u,
                     (f0 * np.expm1(k * glide_s) / k if k != 0.0 else f0 * glide_s) + f1 * (u - glide_s))
    env = np.where(t >= start, (1.0 - np.exp(-u / attack)) * np.exp(-u / tau), 0.0)
    return env * sum(w * np.sin(2 * np.pi * n * phase) for n, w in partials)


DARK = ((1, 1.0), (2, 0.25), (3, 0.06))  # fewer bright partials than the bells: a darker colour


def synth_failure(rate: int = RATE) -> np.ndarray:
    """'Oh-oh': a falling minor third, E5 then C#5 drooping to C5 - lower, slower, darker and softer (-8 dBFS) than
    the positive cues: negative valence, middling arousal (0.75 s; motion 2026-11-14 revision 2)."""
    t = np.arange(int(0.75 * rate)) / rate
    x = _glide(t, 0.0, 659.3, 659.3, 0.2, 0.008, 0.12, DARK) + _glide(t, 0.2, 554.4, 523.3, 0.45, 0.010, 0.25, DARK)
    return _finish(x, peak=0.4)


def synth_boredom(rate: int = RATE) -> np.ndarray:
    """A sigh: a soft tone sliding slowly down (D5 to F#4) with a little breath noise, quiet (-10 dBFS): negative
    valence, low arousal (1.1 s; motion 2026-11-14 revision 2). The noise is seeded: the file is reproducible."""
    t = np.arange(int(1.1 * rate)) / rate
    tone = _glide(t, 0.0, 587.3, 370.0, 0.9, 0.12, 0.45, ((1, 1.0), (2, 0.18)))
    box = np.ones(25) / 25.0  # two passes = a triangular low-pass: breath, not hiss (little above ~2 kHz)
    noise = np.convolve(np.convolve(np.random.default_rng(9).standard_normal(t.size), box, mode="same"), box,
                        mode="same")
    breath = 0.25 * noise / float(np.max(np.abs(noise))) * (1.0 - np.exp(-t / 0.08)) * np.exp(-t / 0.35)
    return _finish(tone + breath, peak=0.3)


SYNTHS: dict[str, Callable[[], np.ndarray]] = {"aha.wav": synth_aha, "success.wav": synth_success,
                                               "failure.wav": synth_failure, "boredom.wav": synth_boredom}


def write_wav(path: Path | str, x: np.ndarray, rate: int = RATE) -> Path:
    """16-bit mono PCM WAV (what the daemon's GStreamer probe and winsound both read)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def write_default_sounds(folder: Path | str) -> list[Path]:
    """(Re)write the synthesised sounds of the signed-off cues into ``folder``."""
    return [write_wav(Path(folder) / name, synth()) for name, synth in SYNTHS.items()]


# ---------------------------------------------------------------------------------------------- the cue
def sound_cue(spec: Any) -> Any:
    """The spec's effective sound cue (tempo applied) or None."""
    return spec.effective().sound if spec is not None else None


def cue_tick(pre_s: float, at_s: float, offset_ms: float, rate_hz: float) -> int:
    """The first stream tick whose time reaches ``pre_s + at_s + offset`` (>= 0)."""
    return max(0, int(math.ceil((pre_s + at_s + offset_ms / 1000.0) * rate_hz - 1e-9)))


class CueTrigger:
    """Fires ``play()`` once, in its own thread, at the first tick >= ``k_cue`` of one stream (``on_tick(k, rec)``
    of the executor: called after each send, must not block). ``record`` collects what happened (pc_measured
    times: when the request left this PC and how long it took)."""

    def __init__(self, k_cue: int, play: Callable[[], Any], record: dict[str, Any]) -> None:
        self.k_cue, self._play, self.record = int(k_cue), play, record
        self.fired = threading.Event()
        self.done = threading.Event()
        self.record.update({"tick": self.k_cue, "fired": False})

    def on_tick(self, k: int, _rec: Any = None) -> None:
        if k < self.k_cue or self.fired.is_set():
            return
        self.fired.set()
        self.record.update(fired=True, fired_tick=int(k), t_fire_ns=time.perf_counter_ns())
        threading.Thread(target=self._run, name="sound-cue", daemon=True).start()

    def _run(self) -> None:
        t0 = time.perf_counter_ns()
        try:
            answer = self._play()
            self.record.update(ok=True, answer=answer)
        except Exception as exc:  # noqa: BLE001 - reported, the motion goes on
            self.record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            self.record["request_ms"] = (time.perf_counter_ns() - t0) / 1e6
            self.done.set()

    def wrap(self, inner: Callable[[int, Any], None] | None) -> Callable[[int, Any], None]:
        def on_tick(k: int, rec: Any) -> None:
            self.on_tick(k, rec)
            if inner is not None:
                inner(k, rec)
        return on_tick


# ---------------------------------------------------------------------------------------------- outputs
class LaptopSpeaker:
    """This PC's speaker, asynchronous (returns at once): ``winsound`` on Windows, ``afplay`` on macOS, ``paplay`` /
    ``aplay`` on Linux (the first one found)."""

    def __init__(self, player: Callable[[str], None] | None = None) -> None:
        self._player = player

    def play(self, path: Path | str) -> str:
        if self._player is not None:
            self._player(str(path))
            return "played (test player)"
        import sys

        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return "winsound"
        import shutil
        import subprocess

        for tool in (("afplay",) if sys.platform == "darwin" else ("paplay", "aplay")):
            exe = shutil.which(tool)
            if exe:
                subprocess.Popen([exe, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return tool
        raise RuntimeError("이 PC에서 소리를 낼 프로그램이 없습니다 (macOS afplay, Linux paplay/aplay)")


class RobotSpeaker:
    """The robot's built-in speaker through the daemon REST API (module docstring). ``opener`` is
    ``urllib.request.urlopen``-like (tests: a stub; never the network)."""

    def __init__(self, host: str, port: int, opener: Callable[..., Any] | None = None) -> None:
        self.base = f"http://{host}:{int(port)}/api/media"
        self._open = opener or urllib.request.urlopen
        self.uploaded: dict[str, str] = {}  # sha256 of the file -> its path on the daemon

    def _json(self, req: urllib.request.Request, timeout: float) -> dict[str, Any]:
        with self._open(req, timeout=timeout) as resp:
            body = resp.read()
        return json.loads(body.decode("utf-8")) if body else {}

    def status(self) -> dict[str, Any]:
        """``GET /api/media/status``: available / released / no_media (nothing sent to the robot)."""
        return self._json(urllib.request.Request(f"{self.base}/status", method="GET"), PLAY_TIMEOUT_S)

    def upload(self, path: Path | str) -> str:
        """``POST /api/media/sounds/upload`` once per file content; returns the daemon path to play."""
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest in self.uploaded:
            return self.uploaded[digest]
        name = UPLOAD_PREFIX + Path(path).name
        boundary = "----hw4sound" + uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                f"Content-Type: audio/wav\r\n\r\n").encode("ascii") + data + f"\r\n--{boundary}--\r\n".encode("ascii")
        req = urllib.request.Request(f"{self.base}/sounds/upload", data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        answer = self._json(req, UPLOAD_TIMEOUT_S)
        daemon_path = str(answer.get("path") or "")
        if answer.get("status") != "ok" or not daemon_path:
            raise RuntimeError(f"upload answered {answer!r}")
        self.uploaded[digest] = daemon_path
        return daemon_path

    def play(self, daemon_path: str) -> dict[str, Any]:
        """``POST /api/media/play_sound``: the daemon starts the file on the speaker and answers at once."""
        req = urllib.request.Request(f"{self.base}/play_sound", data=json.dumps({"file": daemon_path}).encode("utf-8"),
                                     method="POST", headers={"Content-Type": "application/json"})
        return self._json(req, PLAY_TIMEOUT_S)


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = ["RATE", "SOUND_OUTPUTS", "OUTPUT_KO", "SoundSettings", "synth_aha", "synth_success", "synth_failure",
           "synth_boredom", "SYNTHS", "write_wav",
           "write_default_sounds", "sound_cue", "cue_tick", "CueTrigger", "LaptopSpeaker", "RobotSpeaker",
           "file_sha256", "OFFSET_RANGE_MS"]
