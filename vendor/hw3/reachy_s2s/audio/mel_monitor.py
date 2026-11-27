"""Background log-mel computation for live display (independent of speech processing).

The monitor owns one microphone subscription and its own thread; renderers
(GUI timer, offline dumps) only call :meth:`MelMonitor.snapshot`.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..config import MelConfig
from .buffers import DropOldestQueue
from .convert import to_mono
from .mel import StreamingLogMel
from .types import AudioChunk


class MelMonitor:
    def __init__(
        self,
        queue: DropOldestQueue[AudioChunk],
        sample_rate: int,
        cfg: MelConfig,
        channel: str | int = "left",
    ) -> None:
        self._queue = queue
        self.cfg = cfg
        self.mel = StreamingLogMel.from_ms(
            sample_rate,
            cfg.win_ms,
            cfg.hop_ms,
            n_fft=cfg.n_fft,
            n_mels=cfg.n_mels,
            fmin=cfg.fmin_hz,
            fmax=cfg.fmax_hz,
            floor_db=cfg.floor_db,
        )
        self.capacity = max(1, int(round(cfg.display_seconds * self.mel.frame_rate)))
        self._img = np.full((self.capacity, cfg.n_mels), cfg.floor_db, dtype=np.float32)
        self._written = 0
        self._channel = channel
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_chunk_arrival_ns: int | None = None
        self.compute_ms_last = 0.0

    @property
    def frames_total(self) -> int:
        with self._lock:
            return self._written

    def set_channel(self, channel: str | int) -> None:
        with self._lock:
            self._channel = channel

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="mel-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            chunk = self._queue.get(timeout=0.1)
            if chunk is None:
                continue
            self.process_chunk(chunk)

    def process_chunk(self, chunk: AudioChunk) -> None:
        t0 = time.perf_counter()
        with self._lock:
            channel = self._channel
        frames = self.mel.process(to_mono(chunk.samples, channel))
        with self._lock:
            n = frames.shape[0]
            if n >= self.capacity:
                frames = frames[-self.capacity :]
                self._written += n - self.capacity
                n = self.capacity
            pos = self._written % self.capacity
            first = min(n, self.capacity - pos)
            self._img[pos : pos + first] = frames[:first]
            if first < n:
                self._img[: n - first] = frames[first:]
            self._written += n
            self.last_chunk_arrival_ns = chunk.t_arrival_ns
        self.compute_ms_last = (time.perf_counter() - t0) * 1000.0

    def snapshot(self) -> np.ndarray:
        """``(capacity, n_mels)`` dB image, oldest frame first."""
        with self._lock:
            return np.roll(self._img, -(self._written % self.capacity), axis=0)
