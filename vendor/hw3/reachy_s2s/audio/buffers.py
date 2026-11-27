"""Thread-safe buffers used between the capture thread and its consumers."""

from __future__ import annotations

import threading
from collections import deque
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


class DropOldestQueue(Generic[T]):
    """Bounded FIFO whose producer never blocks: when full, the oldest item is dropped.

    The capture thread must never wait for a slow consumer (GUI, ASR, ...),
    so each consumer gets its own queue and pays for its own slowness.
    """

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._items: deque[T] = deque()
        self._maxsize = maxsize
        self._cond = threading.Condition()
        self.dropped = 0
        self.closed = False

    def put(self, item: T) -> None:
        with self._cond:
            if len(self._items) >= self._maxsize:
                self._items.popleft()
                self.dropped += 1
            self._items.append(item)
            self._cond.notify()

    def get(self, timeout: float | None = None) -> T | None:
        """Next item, or ``None`` on timeout / when closed and empty."""
        with self._cond:
            if not self._items and not self.closed:
                self._cond.wait(timeout)
            if self._items:
                return self._items.popleft()
            return None

    def drain(self) -> list[T]:
        with self._cond:
            items = list(self._items)
            self._items.clear()
            return items

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)


class AudioRingBuffer:
    """Fixed-capacity history of the input stream addressed by absolute sample index.

    ``write`` appends frames; ``read(start, end)`` returns the frames whose
    absolute indices are in ``[start, end)`` if they are still held. Used for
    pre-roll (audio just before a detected speech onset) and level meters.
    """

    def __init__(self, capacity_frames: int, channels: int) -> None:
        if capacity_frames < 1:
            raise ValueError("capacity_frames must be >= 1")
        self._data = np.zeros((capacity_frames, channels), dtype=np.float32)
        self._capacity = capacity_frames
        self._channels = channels
        self._end = 0  # absolute index one past the newest frame
        self._lock = threading.Lock()

    @property
    def end_index(self) -> int:
        with self._lock:
            return self._end

    @property
    def start_index(self) -> int:
        with self._lock:
            return max(0, self._end - self._capacity)

    def write(self, frames: np.ndarray, start_index: int | None = None) -> None:
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != self._channels:
            raise ValueError(f"expected (n, {self._channels}) frames, got {frames.shape}")
        with self._lock:
            if start_index is not None and start_index != self._end:
                # Keep the absolute clock consistent with the producer's index
                # (gaps are filled with silence so indices stay aligned).
                if start_index < self._end:
                    raise ValueError("start_index went backwards")
                gap = start_index - self._end
                self._write_locked(np.zeros((min(gap, self._capacity), self._channels), np.float32))
                self._end = start_index
            self._write_locked(frames)

    def _write_locked(self, frames: np.ndarray) -> None:
        n = frames.shape[0]
        if n == 0:
            return
        if n >= self._capacity:
            frames = frames[-self._capacity :]
            self._end += n - self._capacity
            n = self._capacity
        pos = self._end % self._capacity
        first = min(n, self._capacity - pos)
        self._data[pos : pos + first] = frames[:first]
        if first < n:
            self._data[: n - first] = frames[first:]
        self._end += n

    def read(self, start: int, end: int) -> np.ndarray:
        """Frames with absolute indices in ``[start, end)``, clipped to what is held."""
        with self._lock:
            lo = max(start, self._end - self._capacity, 0)
            hi = min(end, self._end)
            if hi <= lo:
                return np.zeros((0, self._channels), dtype=np.float32)
            idx = np.arange(lo, hi) % self._capacity
            return self._data[idx].copy()

    def read_last(self, n: int) -> np.ndarray:
        with self._lock:
            end = self._end
        return self.read(end - n, end)
