"""Acoustic loopback analysis: when does pushed audio come back through the mic?

Why it matters: the cascaded system knows when it *pushed* the first TTS
sample (``t_push``) and when the end of user speech *arrived* on the PC
(``t_arrival(j_end)``). Both the speaker path and the mic path add delay, so

    response latency at the robot = t_push - t_arrival(j_end) + L_rt

where ``L_rt`` (push -> speaker -> air -> mic -> PC) is what this module measures
by playing a chirp and locating it in the recorded mic stream with a
normalized cross-correlation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_NS = 1e9


@dataclass(frozen=True)
class SampleClock:
    """Linear map between the input sample index and PC ``perf_counter_ns``.

    ``offset_ns`` is the *lower envelope* of the chunk arrival times: chunks
    can arrive late (network jitter, bursts) but never early, so the earliest
    consistent arrival line is the best estimate of when each sample reached
    the PC without queueing.
    """

    offset_ns: float
    ns_per_sample: float

    def time_of(self, index: float) -> float:
        return self.offset_ns + self.ns_per_sample * index

    def index_at(self, t_ns: float) -> float:
        return (t_ns - self.offset_ns) / self.ns_per_sample


def fit_sample_clock(anchors: list[tuple[int, int]], sample_rate: int, fit_slope: bool = True) -> SampleClock:
    """Fit ``t_arrival ~ offset + slope * end_index`` (slope from least squares)."""
    if not anchors:
        raise ValueError("no anchors")
    idx = np.array([a[0] for a in anchors], dtype=np.float64)
    t = np.array([a[1] for a in anchors], dtype=np.float64)
    slope = _NS / sample_rate
    if fit_slope and len(anchors) >= 50 and idx[-1] - idx[0] >= 5 * sample_rate:
        fitted = np.polyfit(idx - idx[0], t, 1)[0]
        # accept only plausible clock skews (< 0.5 %), otherwise keep nominal
        if abs(fitted / slope - 1.0) < 0.005:
            slope = fitted
    offset = float(np.min(t - slope * idx))
    return SampleClock(offset, slope)


def normalized_xcorr(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    """NCC in [-1, 1] for every lag ``k`` with ``x[k:k+len(template)]`` inside ``x``."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    n_t = t.shape[0]
    if x.shape[0] < n_t:
        return np.zeros(0)
    n = 1 << int(np.ceil(np.log2(x.shape[0] + n_t)))
    corr = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(t, n)), n)[: x.shape[0] - n_t + 1]
    energy = np.concatenate(([0.0], np.cumsum(x * x)))
    window_energy = energy[n_t:] - energy[:-n_t]
    denom = np.sqrt(np.maximum(window_energy, 1e-20) * np.sum(t * t))
    return corr / denom


@dataclass(frozen=True)
class ProbeResult:
    label: str
    t_push_ns: int
    detected: bool
    score: float
    index: int | None
    t_arrival_ns: float | None

    @property
    def round_trip_ms(self) -> float | None:
        if self.t_arrival_ns is None:
            return None
        return (self.t_arrival_ns - self.t_push_ns) / 1e6


def locate_probes(
    mic: np.ndarray,
    mic_start_index: int,
    clock: SampleClock,
    template: np.ndarray,
    pushes: list[tuple[str, int]],
    sample_rate: int,
    search_s: float = 2.0,
    threshold: float = 0.3,
) -> list[ProbeResult]:
    """Find each pushed probe in the mic recording.

    The search window of probe ``i`` starts at its push and ends ``search_s``
    later or at the next push, whichever comes first (so the next chirp's echo
    is never taken for this one; use an interval longer than the round trip).
    Inside the window the earliest peak reaching 80 % of the window maximum is
    taken and refined to the local maximum within +-5 ms.

    Args:
        mic: mono recording; ``mic[0]`` has absolute input index ``mic_start_index``.
        pushes: ``(label, t_first_push_ns)`` per probe, in push order.
        search_s: look for the echo up to this long after the push.
        threshold: minimum NCC to accept a detection.
    """
    ncc = normalized_xcorr(mic, template)
    refine = max(1, int(0.005 * sample_rate))
    results = []
    for i, (label, t_push) in enumerate(pushes):
        lo = int(np.floor(clock.index_at(t_push))) - mic_start_index
        hi = lo + int(search_s * sample_rate)
        if i + 1 < len(pushes):
            hi = min(hi, int(np.floor(clock.index_at(pushes[i + 1][1]))) - mic_start_index)
        lo, hi = max(lo, 0), min(hi, ncc.shape[0])
        if hi <= lo:
            results.append(ProbeResult(label, t_push, False, 0.0, None, None))
            continue
        window = ncc[lo:hi]
        first = int(np.argmax(window >= 0.8 * float(np.max(window))))
        a, b = max(0, first - refine), min(window.shape[0], first + refine + 1)
        k = lo + a + int(np.argmax(window[a:b]))
        score = float(ncc[k])
        if score >= threshold:
            index = mic_start_index + k
            results.append(ProbeResult(label, t_push, True, score, index, clock.time_of(index)))
        else:
            results.append(ProbeResult(label, t_push, False, score, None, None))
    return results
