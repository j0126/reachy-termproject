"""Sample-rate conversion (libsoxr).

The robot path runs at 16 kHz while TTS / realtime APIs typically emit 24 kHz
(or 22.05/44.1/48 kHz). ``StreamResampler`` keeps the filter state between
chunks so streamed audio is converted without clicks at chunk boundaries.
"""

from __future__ import annotations

import numpy as np
import soxr


def resample(x: np.ndarray, in_rate: int, out_rate: int, quality: str = "HQ") -> np.ndarray:
    """One-shot conversion of a complete signal (1-D or ``(n, channels)``)."""
    arr = np.asarray(x, dtype=np.float32)
    if in_rate == out_rate or arr.shape[0] == 0:
        return arr.copy()
    return soxr.resample(arr, in_rate, out_rate, quality=quality).astype(np.float32, copy=False)


class StreamResampler:
    """Chunk-by-chunk conversion for one continuous stream.

    Call :meth:`process` for every chunk and :meth:`flush` once at the end of
    the stream (it returns the samples still held in the filter).
    """

    def __init__(self, in_rate: int, out_rate: int, channels: int = 1, quality: str = "HQ") -> None:
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.channels = channels
        self._passthrough = in_rate == out_rate
        self._stream = None if self._passthrough else soxr.ResampleStream(
            in_rate, out_rate, channels, dtype="float32", quality=quality
        )
        self._finished = False

    def _shape(self, x: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(x, dtype=np.float32)
        if self.channels == 1 and arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr

    def process(self, x: np.ndarray) -> np.ndarray:
        if self._finished:
            raise RuntimeError("stream already flushed")
        arr = self._shape(x)
        if self._passthrough:
            return arr.copy()
        return self._stream.resample_chunk(arr, last=False)

    def flush(self) -> np.ndarray:
        if self._finished:
            raise RuntimeError("stream already flushed")
        self._finished = True
        empty_shape = (0,) if self.channels == 1 else (0, self.channels)
        empty = np.zeros(empty_shape, dtype=np.float32)
        if self._passthrough:
            return empty
        return self._stream.resample_chunk(empty, last=True)
