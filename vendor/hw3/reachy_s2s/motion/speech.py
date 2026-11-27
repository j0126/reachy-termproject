"""Custom speaking motion from the PCM actually sent to the robot (design section 3, option B).

* :class:`PlaybackTap` is registered with the AudioManager player. The player thread hands it
  every block it sends (after the output gain, i.e. what the robot plays) together with the
  block's estimated audible time (virtual playhead + ``[audio] output_latency_hint_ms``), and a
  flush marker when playback is stopped (barge-in). It only appends to a deque.
* :class:`SpeechMotion` runs in the motion thread. Blocks are analysed when their audible time
  (+ ``speech_sync_offset_ms``) has come.

The motion follows the SDK's speech sway (``reachy_mini/motion/speech_tapper.py``, ``SwayRollRT``,
SDK 1.11.0): the same voice-activity hysteresis, envelope, loudness mapping and the same six
oscillators (frequencies, amplitudes, phases from seed 7), scaled by ``speech_gain`` (0.25 = a
quarter of the SDK amplitude). Two differences, both for smoothness under our speed limits:

1. the SDK multiplies the oscillators by the loudness of the latest 20 ms frame, so the target
   jumps every 50 ms hop (with real TTS audio our rate limiter was saturated in 95 % of the ticks
   and rang); here the hop amplitude is smoothed (attack 0.1 s, release 0.3 s) and interpolated;
2. the oscillators are evaluated at every control tick (continuous time), not once per hop.

The hop amplitude then passes two one-pole filters at the control rate (no kinks at the hop
boundaries). The smoothing delays the amplitude by about 0.2 s, so blocks are analysed
``ANALYSIS_LEAD_S`` before their estimated audible time (``speech_sync_offset_ms`` adds to that;
0 = this best estimate). A flush (barge-in) drops the pending blocks and fades the motion out;
stale blocks are dropped.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from .pose import Pose, perk

# --- SDK SwayRollRT constants (reachy_mini 1.11.0 motion/speech_tapper.py) ---------------------
FRAME_S = 0.02
HOP_S = 0.05
SWAY_MASTER = 1.5
SENS_DB_OFFSET = 4.0
VAD_DB_ON, VAD_DB_OFF = -35.0, -45.0
VAD_ATTACK_HOPS, VAD_RELEASE_HOPS = 1, 5
SWAY_ATTACK_HOPS, SWAY_RELEASE_HOPS = 1, 5
ENV_FOLLOW_GAIN = 0.65
DB_LOW, DB_HIGH, GAMMA = -46.0, -18.0, 0.9
#: (Pose field, frequency Hz, amplitude in deg or mm), in the SDK's phase order
OSCILLATORS = (("pitch", 2.2, 4.5), ("yaw", 0.6, 7.5), ("roll", 1.3, 2.25),
               ("x", 0.35, 4.5), ("y", 0.45, 3.75), ("z", 0.25, 2.25))
SDK_SEED = 7
# --- ours ----------------------------------------------------------------------------------------
AMP_ATTACK_S = 0.10  # smoothing of the hop amplitude (loudness x envelope)
AMP_RELEASE_S = 0.30
AMP_TAU_S = 0.06  # two cascaded one-pole filters on the hop amplitude, at the control rate
ANALYSIS_LEAD_S = 0.15  # analyse this much before the audible time: compensates the smoothing delay
AMP_ZERO = 1e-3  # below this (target 0) the amplitude snaps to 0: a fade-out ends after about 0.35 s
STALE_S = 0.5  # blocks this late (stalled thread) are dropped instead of replayed
BEAT_TAU_S = 0.08  # antenna beat: two cascaded one-pole filters on the block loudness


def rms_dbfs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return -120.0
    return 20.0 * math.log10(math.sqrt(float(np.mean(x * x)) + 1e-12) + 1e-12)


def loudness_gain(db: float) -> float:
    """SDK ``_loudness_gain``: dBFS -> 0..1 (offset, range, gamma)."""
    t = min(1.0, max(0.0, (db + SENS_DB_OFFSET - DB_LOW) / (DB_HIGH - DB_LOW)))
    return t ** GAMMA


def loudness(block: np.ndarray) -> float:
    return loudness_gain(rms_dbfs(block)) if np.size(block) else 0.0


class SwayAmplitude:
    """Per-hop sway amplitude (0..1.5) from PCM: SDK VAD hysteresis x envelope x loudness, smoothed."""

    def __init__(self, rate: int) -> None:
        self.rate = int(rate)
        self.hop = max(1, int(self.rate * HOP_S))
        self.frame = max(1, int(self.rate * FRAME_S))
        self.reset()

    def reset(self) -> None:
        self.carry = np.zeros(0, np.float32)
        self.vad_on = False
        self.above = self.below = 0
        self.up = self.down = 0
        self.env = 0.0
        self.amp = 0.0

    def feed(self, pcm: np.ndarray) -> list[float]:
        self.carry = np.concatenate([self.carry, np.asarray(pcm, np.float32).reshape(-1)])
        out = []
        while self.carry.size >= self.hop:
            hop, self.carry = self.carry[: self.hop], self.carry[self.hop:]
            db = rms_dbfs(hop[-self.frame:])
            if db >= VAD_DB_ON:
                self.above += 1
                self.below = 0
                if not self.vad_on and self.above >= VAD_ATTACK_HOPS:
                    self.vad_on = True
            elif db <= VAD_DB_OFF:
                self.below += 1
                self.above = 0
                if self.vad_on and self.below >= VAD_RELEASE_HOPS:
                    self.vad_on = False
            if self.vad_on:
                self.up, self.down = min(SWAY_ATTACK_HOPS, self.up + 1), 0
                target = self.up / SWAY_ATTACK_HOPS
            else:
                self.down, self.up = min(SWAY_RELEASE_HOPS, self.down + 1), 0
                target = 1.0 - self.down / SWAY_RELEASE_HOPS
            self.env = min(1.0, max(0.0, self.env + ENV_FOLLOW_GAIN * (target - self.env)))
            raw = loudness_gain(db) * SWAY_MASTER * self.env
            tau = AMP_ATTACK_S if raw > self.amp else AMP_RELEASE_S
            self.amp += (1.0 - math.exp(-HOP_S / tau)) * (raw - self.amp)
            out.append(self.amp)
        return out


def _sdk_phases() -> tuple[float, ...]:
    rng = np.random.default_rng(SDK_SEED)
    return tuple(float(rng.random() * 2 * math.pi) for _ in OSCILLATORS)


class PlaybackTap:
    """Thread-safe hand-off from the audio player to the motion thread (append / popleft only)."""

    def __init__(self, maxlen: int = 3000) -> None:
        self._items: deque[tuple[str, int, Any, int]] = deque(maxlen=maxlen)

    def on_block(self, samples: np.ndarray, t_audible_ns: int, rate: int) -> None:  # player thread
        self._items.append(("block", int(t_audible_ns), samples, int(rate)))

    def on_flush(self, t_ns: int) -> None:  # player thread (stop_playback / barge-in)
        self._items.append(("flush", int(t_ns), None, 0))

    def drain(self) -> list[tuple[str, int, Any, int]]:
        out = []
        while True:
            try:
                out.append(self._items.popleft())
            except IndexError:
                return out


class SpeechMotion:
    def __init__(self, gain: float = 0.25, beat_rad: float = 0.08, sync_offset_ms: float = 0.0) -> None:
        self.gain = gain
        self.beat_rad = beat_rad
        self.sync_offset_ns = int(sync_offset_ms * 1e6)
        self.phases = _sdk_phases()
        self._sway: SwayAmplitude | None = None
        self._pending: deque[tuple[int, np.ndarray, int]] = deque()
        self._amp_target = self._amp1 = self._amp2 = 0.0
        self._osc_t0 = 0
        self._beat_target = self._beat1 = self._beat2 = 0.0
        self._last_ns: int | None = None
        self._speaking = False
        self.blocks_fed = 0
        self.hops = 0
        self.flushes = 0
        self.dropped_stale = 0
        self.last_amplitude = 0.0

    # -- input ---------------------------------------------------------------
    def push(self, items: list[tuple[str, int, Any, int]], now_ns: int) -> None:
        for kind, t_ns, samples, rate in items:
            if kind == "flush":
                self.flush(now_ns)
            else:
                self._pending.append((t_ns + self.sync_offset_ns - int(ANALYSIS_LEAD_S * 1e9), samples, rate))

    def flush(self, now_ns: int) -> None:
        """Barge-in / stop: drop the pending blocks, restart the analysis, fade what is moving."""
        self._pending.clear()
        if self._sway is not None:
            self._sway.reset()
        self._beat_target = self._amp_target = 0.0
        self.flushes += 1

    def discard(self, now_ns: int) -> None:
        """Not driving the robot (motion off / hold): forget everything, output zero."""
        self._pending.clear()
        if self._sway is not None:
            self._sway.reset()
        self._amp_target = self._amp1 = self._amp2 = 0.0
        self._beat_target = self._beat1 = self._beat2 = 0.0
        self._last_ns = now_ns

    def reset_output(self, now_ns: int) -> None:
        """Start a new speaking phase from zero (the engine did not include us before)."""
        self._amp_target = self._amp1 = self._amp2 = 0.0
        self._osc_t0 = now_ns
        self._beat1 = self._beat2 = 0.0

    # -- output --------------------------------------------------------------
    def step(self, now_ns: int, speaking: bool) -> Pose:
        """Offsets for this tick (head sway + antenna beat), before the motion intensity."""
        dt = 0.02 if self._last_ns is None else max(0.0, (now_ns - self._last_ns) / 1e9)
        self._last_ns = now_ns
        hop_amp: float | None = None
        while self._pending and self._pending[0][0] <= now_ns:
            t_ns, samples, rate = self._pending.popleft()
            if (now_ns - t_ns) / 1e9 > STALE_S:
                self.dropped_stale += 1
                continue
            if self._sway is None or self._sway.rate != rate:
                self._sway = SwayAmplitude(rate)
            block = np.asarray(samples, dtype=np.float32).reshape(-1)
            amps = self._sway.feed(block)
            if amps:
                hop_amp = amps[-1]
                self.hops += len(amps)
            self._beat_target = loudness(block)
            self.blocks_fed += 1
        if speaking and not self._speaking:
            self.reset_output(now_ns)
        self._speaking = speaking
        if not speaking:
            self._amp_target = self._beat_target = 0.0
        elif hop_amp is not None:
            self._amp_target = hop_amp
        a = 1.0 - math.exp(-dt / AMP_TAU_S) if dt > 0 else 0.0
        self._amp2 += a * (self._amp1 - self._amp2)  # second pole first: zero initial slope after a step
        self._amp1 += a * (self._amp_target - self._amp1)
        if self._amp_target == 0.0 and max(self._amp1, self._amp2) < AMP_ZERO:
            self._amp1 = self._amp2 = 0.0
        a = 1.0 - math.exp(-dt / BEAT_TAU_S) if dt > 0 else 0.0
        self._beat2 += a * (self._beat1 - self._beat2)
        self._beat1 += a * (self._beat_target - self._beat1)
        amp = self._amp2
        self.last_amplitude = amp
        if amp <= 1e-6:
            return perk(self.beat_rad * self._beat2)
        t = (now_ns - self._osc_t0) / 1e9
        k = amp * self.gain
        values = {name: k * a_ * math.sin(2 * math.pi * f * t + ph)
                  for (name, f, a_), ph in zip(OSCILLATORS, self.phases)}
        return Pose(**values) + perk(self.beat_rad * self._beat2)
