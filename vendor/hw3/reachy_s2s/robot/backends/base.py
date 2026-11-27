"""Backend interface used by :class:`reachy_s2s.robot.audio_manager.AudioManager`.

Threading contract (the AudioManager guarantees it):

* ``read`` is called only from the capture thread;
* ``write`` and ``clear_output`` are called only from the player thread;
* ``open`` / ``close`` are called from the owner thread while the other two
  threads are not running.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AudioBackend(ABC):
    name: str = "abstract"
    #: Nominal delay between ``write`` and the sound leaving the speaker.
    output_latency_hint_s: float = 0.0

    @property
    @abstractmethod
    def input_rate(self) -> int: ...

    @property
    @abstractmethod
    def output_rate(self) -> int: ...

    @property
    @abstractmethod
    def input_channels(self) -> int: ...

    @property
    @abstractmethod
    def output_channels(self) -> int: ...

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Next captured block ``(n, input_channels)`` float32, or ``None``.

        May block briefly (tens of ms) but must return so the capture thread
        can notice a stop request.
        """

    @abstractmethod
    def write(self, frames: np.ndarray) -> None:
        """Queue ``(n, output_channels)`` float32 frames at ``output_rate``."""

    @abstractmethod
    def clear_output(self) -> None:
        """Drop audio already queued for the speaker (barge-in)."""

    def output_ready(self) -> bool:
        """``True`` once ``write`` will actually reach the speaker."""
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "input_rate": self.input_rate,
            "output_rate": self.output_rate,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "output_latency_hint_s": self.output_latency_hint_s,
        }
