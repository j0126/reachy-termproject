"""Streaming log-mel spectrogram (numpy only).

Conventions (documented so values are comparable across runs):

* periodic Hann window of ``win_length`` samples, zero-padded to ``n_fft``;
* power normalized by ``sum(window)**2``, so a full-scale sine centred on an
  FFT bin has power 0.25 (-6 dB) in that bin;
* mel scale and filter normalization follow librosa's defaults
  (Slaney scale, ``norm="slaney"`` area normalization), i.e. the same family
  of filters as Whisper's front end; ``htk=True`` switches to the HTK scale;
* output is ``10*log10(mel_power)`` clamped at ``floor_db``.

``StreamingLogMel.process`` accepts arbitrary chunk sizes and returns exactly
the frames a one-shot computation over the concatenated signal would return
(frame ``i`` covers samples ``[i*hop, i*hop + win_length)``).
"""

from __future__ import annotations

import numpy as np

# Slaney mel scale constants (librosa.hz_to_mel with htk=False)
_F_SP = 200.0 / 3.0
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def hz_to_mel(freq_hz: np.ndarray | float, htk: bool = False) -> np.ndarray:
    f = np.asarray(freq_hz, dtype=np.float64)
    if htk:
        return 2595.0 * np.log10(1.0 + f / 700.0)
    mels = f / _F_SP
    log_region = f >= _MIN_LOG_HZ
    return np.where(
        log_region,
        _MIN_LOG_MEL + np.log(np.maximum(f, _MIN_LOG_HZ) / _MIN_LOG_HZ) / _LOGSTEP,
        mels,
    )


def mel_to_hz(mels: np.ndarray | float, htk: bool = False) -> np.ndarray:
    m = np.asarray(mels, dtype=np.float64)
    if htk:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    freqs = _F_SP * m
    log_region = m >= _MIN_LOG_MEL
    return np.where(log_region, _MIN_LOG_HZ * np.exp(_LOGSTEP * (m - _MIN_LOG_MEL)), freqs)


def mel_band_edges(n_mels: int, fmin: float, fmax: float, htk: bool = False) -> np.ndarray:
    """``n_mels + 2`` frequencies: lower edge, centres, upper edge of the triangles."""
    mels = np.linspace(hz_to_mel(fmin, htk), hz_to_mel(fmax, htk), n_mels + 2)
    return mel_to_hz(mels, htk)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    htk: bool = False,
    norm: str | None = "slaney",
) -> np.ndarray:
    """Triangular filters, shape ``(n_mels, n_fft // 2 + 1)`` (librosa-compatible)."""
    fmax = sample_rate / 2.0 if fmax is None else fmax
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    edges = mel_band_edges(n_mels, fmin, fmax, htk)
    widths = np.diff(edges)
    ramps = np.subtract.outer(edges, fft_freqs)
    weights = np.zeros((n_mels, fft_freqs.size))
    for i in range(n_mels):
        lower = -ramps[i] / widths[i]
        upper = ramps[i + 2] / widths[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    if norm == "slaney":
        weights *= (2.0 / (edges[2 : n_mels + 2] - edges[:n_mels]))[:, None]
    elif norm is not None:
        raise ValueError(f"unsupported norm {norm!r}")
    return weights.astype(np.float32)


class StreamingLogMel:
    """Chunk-wise log-mel computation for one continuous mono stream."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float | None = None,
        floor_db: float = -120.0,
        htk: bool = False,
    ) -> None:
        if win_length > n_fft:
            raise ValueError("win_length must be <= n_fft")
        if hop_length < 1:
            raise ValueError("hop_length must be >= 1")
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.floor_db = floor_db
        self.window = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win_length) / win_length)).astype(
            np.float32
        )
        self._power_norm = float(np.sum(self.window, dtype=np.float64) ** 2)
        self.filters = mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax, htk)
        self.band_edges_hz = mel_band_edges(n_mels, fmin, sample_rate / 2.0 if fmax is None else fmax, htk)
        self._floor_power = 10.0 ** (floor_db / 10.0)
        self._buf = np.zeros(0, dtype=np.float32)
        self.frames_emitted = 0

    @classmethod
    def from_ms(cls, sample_rate: int, win_ms: float, hop_ms: float, **kwargs) -> "StreamingLogMel":
        return cls(
            sample_rate,
            win_length=int(round(sample_rate * win_ms / 1000.0)),
            hop_length=int(round(sample_rate * hop_ms / 1000.0)),
            **kwargs,
        )

    @property
    def center_freqs_hz(self) -> np.ndarray:
        return self.band_edges_hz[1:-1]

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self.frames_emitted = 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Append mono samples; return the newly completed frames ``(n, n_mels)`` in dB."""
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("StreamingLogMel expects mono (1-D) audio")
        buf = np.concatenate((self._buf, x)) if self._buf.size else x
        if buf.size < self.win_length:
            self._buf = buf.copy()
            return np.zeros((0, self.n_mels), dtype=np.float32)
        n_frames = 1 + (buf.size - self.win_length) // self.hop_length
        frames = np.lib.stride_tricks.sliding_window_view(buf, self.win_length)[
            :: self.hop_length
        ][:n_frames]
        spec = np.fft.rfft(frames * self.window, n=self.n_fft, axis=1)
        power = (spec.real**2 + spec.imag**2) / self._power_norm
        mel_power = power.astype(np.float32) @ self.filters.T
        out = 10.0 * np.log10(np.maximum(mel_power, self._floor_power))
        self._buf = buf[n_frames * self.hop_length :].copy()
        self.frames_emitted += n_frames
        return out.astype(np.float32)


def log_mel_spectrogram(samples: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
    """One-shot version of :class:`StreamingLogMel` (same frames, same values)."""
    return StreamingLogMel(sample_rate, **kwargs).process(samples)
