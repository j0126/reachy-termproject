"""Audio data containers shared by backends, the audio manager and consumers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    """A block of captured microphone audio.

    Attributes:
        samples: float32 array, shape ``(n, channels)``, exactly as the backend
            delivered it (robot: 16 kHz stereo).
        sample_rate: samples per second.
        start_index: position of ``samples[0]`` in the input stream since the
            capture started (sample clock; never resets while capturing).
        t_arrival_ns: ``time.perf_counter_ns()`` when the capture thread
            received the chunk on this PC. The acoustic time of the last
            sample is earlier by the (unknown) input latency.
    """

    samples: np.ndarray
    sample_rate: int
    start_index: int
    t_arrival_ns: int

    @property
    def num_frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def end_index(self) -> int:
        return self.start_index + self.num_frames

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.sample_rate
