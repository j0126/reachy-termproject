"""Reachy Mini audio through the official SDK ``MediaManager`` (reachy-mini 1.11.0).

Verified against the installed SDK source:

* WEBRTC (Wireless): ``GstWebRTCClient``. Mic = Opus over WebRTC -> appsink
  F32LE 16 kHz stereo; ``get_audio_sample`` waits <= 20 ms. Speaker =
  ``push_audio_sample`` -> appsrc -> Opus -> daemon (300 ms jitter buffer) ->
  ALSA. ``start_recording``/``start_playing`` are no-ops (streaming starts
  when the media manager is created). ``clear_player`` flushes the local send
  queue and POSTs ``/api/media/clear_incoming_audio``.
* LOCAL (Lite, daemon on this PC): ``GStreamerAudio`` with WASAPI. Record and
  playback share ONE pipeline: ``stop_recording``/``stop_playing`` stop both and
  ``clear_player`` briefly pauses capture too.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...audio.convert import match_channels
from .base import AudioBackend


class ReachyMediaBackend(AudioBackend):
    def __init__(self, media: Any, output_latency_hint_s: float) -> None:
        """``media`` is ``ReachyMini.media`` (a ``MediaManager``)."""
        if media is None or getattr(media, "audio", None) is None:
            raise RuntimeError(
                "Reachy media/audio is not available (media_backend='no_media' or the daemon has no media)."
            )
        self._media = media
        self.kind = media.backend.value  # "webrtc" | "local"
        self.name = f"reachy-{self.kind}"
        self.output_latency_hint_s = output_latency_hint_s

    @property
    def input_rate(self) -> int:
        return int(self._media.get_input_audio_samplerate())

    @property
    def output_rate(self) -> int:
        return int(self._media.get_output_audio_samplerate())

    @property
    def input_channels(self) -> int:
        return int(self._media.get_input_channels())

    @property
    def output_channels(self) -> int:
        return int(self._media.get_output_channels())

    def open(self) -> None:
        self._media.start_recording()
        self._media.start_playing()

    def close(self) -> None:
        # The media manager belongs to the ReachyMini instance; it is released by
        # ReachyConnection.close() (calling stop_* here would also stop the LOCAL
        # pipeline for anyone else sharing it).
        pass

    def read(self) -> np.ndarray | None:
        data = self._media.get_audio_sample()
        if data is None:
            return None
        return np.asarray(data, dtype=np.float32)

    def write(self, frames: np.ndarray) -> None:
        self._media.push_audio_sample(match_channels(frames, self.output_channels))

    def clear_output(self) -> None:
        # MediaManager has no clear wrapper; clear_player() is public on both
        # GStreamerAudio and GstWebRTCClient.
        self._media.audio.clear_player()

    def output_ready(self) -> bool:
        # SDK 1.11.0 internals: the WebRTC send chain (appsrc) is created only
        # after the audio track is negotiated; pushing earlier drops audio with
        # an "AppSrc is not initialized" warning.
        return getattr(self._media.audio, "_appsrc", None) is not None

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["sdk_media_backend"] = self.kind
        return info
