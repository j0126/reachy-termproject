"""Hardware-free backend for unit tests and offline runs (no robot needed).

* Input: replays ``source`` (e.g. a WAV recorded from the robot) in fixed-size
  chunks, paced in real time or as fast as it is read; afterwards silence
  (``pad_with_silence``), a loop, or end-of-stream.
* Output: every ``write`` is kept with its ``perf_counter_ns`` timestamp.
* Loopback: written audio re-enters the input ``loopback_delay_s`` later
  (scaled by ``loopback_gain``), like a speaker heard by the mic with a known
  latency. Used to test the loopback latency probe end to end.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from ...audio.convert import as_frames
from .base import AudioBackend


class MockAudioBackend(AudioBackend):
    def __init__(
        self,
        source: np.ndarray | None = None,
        sample_rate: int = 16000,
        channels: int = 2,
        chunk_ms: float = 20.0,
        realtime: bool = True,
        loop: bool = False,
        pad_with_silence: bool = True,
        loopback_delay_s: float | None = None,
        loopback_gain: float = 1.0,
        output_latency_hint_s: float = 0.0,
        output_ready_after_s: float = 0.0,
    ) -> None:
        self.name = "mock"
        self._rate = sample_rate
        self._channels = channels
        self._chunk = max(1, int(round(sample_rate * chunk_ms / 1000.0)))
        src = np.zeros((0, channels), np.float32) if source is None else as_frames(source)
        if src.shape[1] != channels:
            src = np.repeat(src[:, :1], channels, axis=1)
        self._source = src
        self._realtime = realtime
        self._loop = loop
        self._pad = pad_with_silence
        self._loopback_delay = None if loopback_delay_s is None else int(round(loopback_delay_s * sample_rate))
        self._loopback_gain = loopback_gain
        self.output_latency_hint_s = output_latency_hint_s
        self._ready_after_s = output_ready_after_s

        self._lock = threading.Lock()
        self._pos = 0  # frames delivered by read()
        self._t_open_ns: int | None = None
        # Loopback: output samples scheduled at absolute input positions.
        self._echo = np.zeros((0, channels), np.float32)
        self._echo_start = 0
        self.writes: list[tuple[int, np.ndarray]] = []
        self.clear_calls: list[int] = []
        self.opened = False
        self.closed = False

    # -- properties --------------------------------------------------------
    @property
    def input_rate(self) -> int:
        return self._rate

    @property
    def output_rate(self) -> int:
        return self._rate

    @property
    def input_channels(self) -> int:
        return self._channels

    @property
    def output_channels(self) -> int:
        return self._channels

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        self._t_open_ns = time.perf_counter_ns()
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def output_ready(self) -> bool:
        if self._t_open_ns is None:
            return False
        return (time.perf_counter_ns() - self._t_open_ns) / 1e9 >= self._ready_after_s

    # -- input -------------------------------------------------------------
    def read(self) -> np.ndarray | None:
        if self._t_open_ns is None:
            raise RuntimeError("backend not opened")
        with self._lock:
            pos = self._pos
        if self._realtime:
            due_ns = self._t_open_ns + int((pos + self._chunk) * 1e9 / self._rate)
            wait = (due_ns - time.perf_counter_ns()) / 1e9
            if wait > 0:
                time.sleep(min(wait, 0.05))
                if time.perf_counter_ns() < due_ns:
                    return None
        block = self._source_block(pos, self._chunk)
        if block is None:
            time.sleep(0.01)
            return None
        with self._lock:
            block = self._mix_echo(block, pos)
            self._pos = pos + block.shape[0]
        return block

    def _source_block(self, pos: int, n: int) -> np.ndarray | None:
        total = self._source.shape[0]
        if self._loop and total > 0:
            idx = (np.arange(pos, pos + n) % total)
            return self._source[idx].copy()
        if pos < total:
            block = self._source[pos : pos + n]
            if block.shape[0] < n and self._pad:
                block = np.vstack([block, np.zeros((n - block.shape[0], self._channels), np.float32)])
            return block.copy()
        if self._pad:
            return np.zeros((n, self._channels), np.float32)
        return None

    def _mix_echo(self, block: np.ndarray, pos: int) -> np.ndarray:
        if self._echo.shape[0] == 0:
            return block
        lo = max(pos, self._echo_start)
        hi = min(pos + block.shape[0], self._echo_start + self._echo.shape[0])
        if hi > lo:
            block = block.copy()
            block[lo - pos : hi - pos] += self._echo[lo - self._echo_start : hi - self._echo_start]
        # forget echo that is now in the past
        consumed = pos + block.shape[0] - self._echo_start
        if consumed > 0:
            self._echo = self._echo[consumed:]
            self._echo_start += consumed
        return block

    # -- output ------------------------------------------------------------
    def write(self, frames: np.ndarray) -> None:
        frames = as_frames(frames).copy()
        self.writes.append((time.perf_counter_ns(), frames))
        if self._loopback_delay is None:
            return
        with self._lock:
            echo_end = self._echo_start + self._echo.shape[0]
            # The device plays queued output contiguously; an empty queue starts
            # playing "now" (current input position) + the loopback delay.
            start = max(self._pos + self._loopback_delay, echo_end)
            if start > echo_end:
                gap = np.zeros((start - echo_end, self._channels), np.float32)
                if self._echo.shape[0] == 0:
                    self._echo_start = start
                else:
                    self._echo = np.vstack([self._echo, gap])
            self._echo = np.vstack([self._echo, frames * self._loopback_gain])

    def clear_output(self) -> None:
        self.clear_calls.append(time.perf_counter_ns())
        with self._lock:
            # Drop echo that has not reached the "mic" yet.
            keep = max(0, self._pos - self._echo_start)
            self._echo = self._echo[:keep]

    @property
    def written(self) -> np.ndarray:
        if not self.writes:
            return np.zeros((0, self._channels), np.float32)
        return np.vstack([w for _, w in self.writes])

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(realtime=self._realtime, loopback_delay_frames=self._loopback_delay)
        return info
