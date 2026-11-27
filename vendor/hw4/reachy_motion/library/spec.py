"""Motion specifications as plain data (no SDK import; docs/ARCHITECTURE.md section 4).

A ``MotionSpec`` is three independent keyframe tracks that all start from the same ``start`` state:

* ``head``     - ``HeadKey``: 6-D head pose (``Pose6``, SI units, relative to the neutral pose)
* ``antennas`` - ``AntennaKey``: (right, left) antenna angles in rad
* ``body``     - ``BodyKey``: body yaw in rad (always explicit: ``goto_target`` defaults it to 0.0)

Each key moves its track from the previous value (or the start value) to the key's value in
``duration`` seconds with the key's ``easing``, then holds it (``hold`` = pause) for ``hold`` seconds.
Tracks run in parallel from t = 0; a track that ends early holds its last value. Separate tracks are
what make overlapping action / follow-through possible: one SDK ``goto_target`` drives every channel
with the same time law (Phase 1 SDK_ANALYSIS section 3.4), so channel offsets are planned here.
(Phase 0 sketched one ``Keyframe`` holding every channel; Phase 2 splits it into tracks.)

``amplitude_scale`` and ``tempo_scale`` are global edit knobs kept separate from the keys: the
planner applies them (amplitude about the start state, all durations / holds divided by tempo).
``effective()`` returns the spec with both baked into the keys - the parameters actually executed.

Valence / arousal are the designer's EXPECTED values (design targets on [-1, 1]); nobody has rated
these motions, so they are not results (CLAUDE.md rule 6). The shipped motions are DRAFTS: the
student designs and tunes the final ones (``motions/*.yaml``, Motion Editor GUI).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from reachy_motion.library.pose import REST, Pose6, RobotState

MOTION_NAMES: tuple[str, ...] = ("aha_recommend", "boredom", "success", "failure")
VARIANTS: tuple[str, ...] = ("expressive", "neutral")
MIN_KEY_DURATION_S = 0.02  # one 50 Hz control tick; shorter keys could never be commanded


class Easing(str, Enum):
    """Library easing names; the planner maps them 1:1 to SDK ``InterpolationTechnique`` values."""

    LINEAR = "linear"  # constant speed inside the key (velocity steps at both ends)
    MIN_JERK = "minjerk"
    EASE_IN_OUT = "ease_in_out"
    CARTOON = "cartoon"  # +-10.0 % anticipation and overshoot built into the time law (Phase 1 section 3.2)


class Principle(str, Enum):
    """Disney's 12 principles of animation (Thomas & Johnston 1981) that apply to a 9-DOF head robot."""

    SQUASH_AND_STRETCH = "squash_and_stretch"
    ANTICIPATION = "anticipation"
    STAGING = "staging"
    POSE_TO_POSE = "pose_to_pose"
    FOLLOW_THROUGH = "follow_through_and_overlapping_action"
    SLOW_IN_SLOW_OUT = "slow_in_slow_out"
    ARCS = "arcs"
    SECONDARY_ACTION = "secondary_action"
    TIMING = "timing"
    EXAGGERATION = "exaggeration"
    APPEAL = "appeal"


@dataclass(frozen=True)
class PrincipleUse:
    """Where and how one principle is applied in a spec (free text, cites tracks / key labels)."""

    principle: Principle
    how: str


@dataclass(frozen=True)
class SoundCue:
    """Optional sound effect cue (metadata only in Phase 2: logged, not played). Neutral variants have none."""

    file: str  # file name, e.g. "aha.wav"
    at_s: float  # start time within the motion (s, before tempo scaling)


@dataclass(frozen=True)
class HeadKey:
    duration: float  # s, travel time from the previous value to ``pose``
    pose: Pose6
    easing: Easing = Easing.MIN_JERK
    hold: float = 0.0  # s, pause after arriving
    label: str = ""

    @property
    def value(self) -> Pose6:
        return self.pose


@dataclass(frozen=True)
class AntennaKey:
    duration: float
    antennas: tuple[float, float]  # rad [right, left]
    easing: Easing = Easing.MIN_JERK
    hold: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.antennas) != 2:
            raise ValueError("antennas must be (right, left)")
        object.__setattr__(self, "antennas", (float(self.antennas[0]), float(self.antennas[1])))

    @property
    def value(self) -> tuple[float, float]:
        return self.antennas


@dataclass(frozen=True)
class BodyKey:
    duration: float
    yaw: float  # rad
    easing: Easing = Easing.MIN_JERK
    hold: float = 0.0
    label: str = ""

    @property
    def value(self) -> float:
        return self.yaw


TRACKS: tuple[str, ...] = ("head", "antennas", "body")


@dataclass(frozen=True)
class MotionSpec:
    name: str  # one of MOTION_NAMES
    variant: str  # one of VARIANTS
    intent: str  # what the motion should communicate
    valence: float  # EXPECTED (design target) on [-1, 1]
    arousal: float  # EXPECTED (design target) on [-1, 1]
    principles: tuple[PrincipleUse, ...]
    head: tuple[HeadKey, ...] = ()
    antennas: tuple[AntennaKey, ...] = ()
    body: tuple[BodyKey, ...] = ()
    start: RobotState = field(default_factory=lambda: REST)
    description: str = ""
    amplitude_scale: float = 1.0  # applied by the planner about the start state (1.0 = as designed)
    tempo_scale: float = 1.0  # applied by the planner: durations and holds / tempo_scale
    sound: SoundCue | None = None
    draft: bool = True  # True until the student signs the design off

    # ------------------------------------------------------------------ structure
    @property
    def key(self) -> str:
        return f"{self.name}:{self.variant}"

    def track(self, name: str) -> tuple[Any, ...]:
        if name not in TRACKS:
            raise ValueError(f"unknown track {name!r}")
        return getattr(self, name)

    def start_value(self, track: str) -> Any:
        return {"head": self.start.head, "antennas": self.start.antennas, "body": self.start.body_yaw}[track]

    # ------------------------------------------------------------------ effective (executed) parameters
    def effective(self) -> "MotionSpec":
        """The spec with ``amplitude_scale`` / ``tempo_scale`` baked into the keys (factors reset to 1.0)."""
        a, tempo = self.amplitude_scale, self.tempo_scale
        if a == 1.0 and tempo == 1.0:
            return self
        s0 = self.start

        def head_key(k: HeadKey) -> HeadKey:
            return replace(k, pose=s0.head + (k.pose - s0.head).scaled(a), duration=k.duration / tempo,
                           hold=k.hold / tempo)

        def ant_key(k: AntennaKey) -> AntennaKey:
            vals = tuple(s + (v - s) * a for s, v in zip(s0.antennas, k.antennas))
            return replace(k, antennas=vals, duration=k.duration / tempo, hold=k.hold / tempo)

        def body_key(k: BodyKey) -> BodyKey:
            return replace(k, yaw=s0.body_yaw + (k.yaw - s0.body_yaw) * a, duration=k.duration / tempo,
                           hold=k.hold / tempo)

        sound = None if self.sound is None else replace(self.sound, at_s=self.sound.at_s / tempo)
        return replace(self, head=tuple(head_key(k) for k in self.head),
                       antennas=tuple(ant_key(k) for k in self.antennas),
                       body=tuple(body_key(k) for k in self.body), sound=sound,
                       amplitude_scale=1.0, tempo_scale=1.0)

    def track_duration(self, name: str) -> float:
        """Effective length of one track (s), holds included."""
        return float(sum(k.duration + k.hold for k in self.track(name))) / self.tempo_scale

    @property
    def duration(self) -> float:
        """Effective total length (s): the longest track including its holds."""
        return max(self.track_duration(t) for t in TRACKS)

    def end_value(self, track: str) -> Any:
        keys = self.effective().track(track)
        return keys[-1].value if keys else self.start_value(track)

    @property
    def end_state(self) -> RobotState:
        return RobotState(head=self.end_value("head"), antennas=self.end_value("antennas"),
                          body_yaw=float(self.end_value("body")))

    def key_times(self, track: str) -> list[tuple[float, float, float]]:
        """Effective [(t_depart, t_arrive, t_release)] per key: travel starts, target reached, hold ends."""
        out, t = [], 0.0
        for k in self.effective().track(track):
            out.append((t, t + k.duration, t + k.duration + k.hold))
            t += k.duration + k.hold
        return out

    # ------------------------------------------------------------------ checks
    def validate(self) -> list[str]:
        """Problems with the data itself (empty list = well formed). Physical limits are the safety checker's job."""
        problems: list[str] = []
        if self.name not in MOTION_NAMES:
            problems.append(f"name {self.name!r} not in {MOTION_NAMES}")
        if self.variant not in VARIANTS:
            problems.append(f"variant {self.variant!r} not in {VARIANTS}")
        for label, v in (("valence", self.valence), ("arousal", self.arousal)):
            if not (isinstance(v, (int, float)) and math.isfinite(v) and -1.0 <= v <= 1.0):
                problems.append(f"{label} {v} outside [-1, 1]")
        if not any(self.track(t) for t in TRACKS):
            problems.append("no keys on any track")
        if not self.intent.strip():
            problems.append("intent is empty")
        if not self.principles:
            problems.append("no animation principle recorded")
        for t in TRACKS:
            for i, k in enumerate(self.track(t)):
                where = f"{t}[{i}]{' ' + k.label if k.label else ''}"
                if not isinstance(k.easing, Easing):
                    problems.append(f"{where}: easing {k.easing!r} is not an Easing")
                if not (math.isfinite(k.duration) and k.duration >= MIN_KEY_DURATION_S):
                    problems.append(f"{where}: duration {k.duration} < {MIN_KEY_DURATION_S} s")
                if not (math.isfinite(k.hold) and k.hold >= 0.0):
                    problems.append(f"{where}: hold {k.hold} < 0")
                vals = (k.value.as_array().tolist() if t == "head"
                        else list(k.value) if t == "antennas" else [k.value])
                if not all(math.isfinite(float(x)) for x in vals):
                    problems.append(f"{where}: non-finite value")
        if not (math.isfinite(self.amplitude_scale) and self.amplitude_scale >= 0.0):
            problems.append(f"amplitude_scale {self.amplitude_scale} < 0")
        if not (math.isfinite(self.tempo_scale) and self.tempo_scale > 0.0):
            problems.append(f"tempo_scale {self.tempo_scale} <= 0")
        elif any(k.duration / self.tempo_scale < MIN_KEY_DURATION_S for t in TRACKS for k in self.track(t)):
            problems.append(f"tempo_scale {self.tempo_scale} makes a key shorter than {MIN_KEY_DURATION_S} s")
        if self.sound is not None:
            raw_duration = max(float(sum(k.duration + k.hold for k in self.track(t))) for t in TRACKS)
            if not (math.isfinite(self.sound.at_s) and 0.0 <= self.sound.at_s <= raw_duration):
                problems.append(f"sound cue at {self.sound.at_s} s outside the motion")
            if not self.sound.file.strip():
                problems.append("sound cue without a file name")
        return problems

    # ------------------------------------------------------------------ transforms
    def scaled(self, amplitude: float = 1.0, tempo: float = 1.0) -> "MotionSpec":
        """Multiply the global amplitude / tempo knobs (intensity, Phase 3 sweeps); keys stay as designed."""
        if amplitude < 0.0 or not math.isfinite(amplitude):
            raise ValueError("amplitude must be >= 0")
        if tempo <= 0.0 or not math.isfinite(tempo):
            raise ValueError("tempo must be > 0")
        return replace(self, amplitude_scale=self.amplitude_scale * amplitude, tempo_scale=self.tempo_scale * tempo)

    # ------------------------------------------------------------------ records
    def to_dict(self) -> dict[str, Any]:
        """Canonical JSON-able form (SI units) - used for manifests and the plan digest."""

        def key_dict(t: str, k: Any) -> dict[str, Any]:
            v = (dict(zip(("x", "y", "z", "roll", "pitch", "yaw"), k.value.as_array().tolist())) if t == "head"
                 else list(k.value) if t == "antennas" else float(k.value))
            return {"duration": k.duration, "value": v, "easing": k.easing.value, "hold": k.hold, "label": k.label}

        return {
            "name": self.name, "variant": self.variant, "intent": self.intent,
            "valence_expected": self.valence, "arousal_expected": self.arousal,
            "principles": [{"principle": p.principle.value, "how": p.how} for p in self.principles],
            "start": {"head": dict(zip(("x", "y", "z", "roll", "pitch", "yaw"), self.start.head.as_array().tolist())),
                      "antennas": list(self.start.antennas), "body_yaw": self.start.body_yaw},
            "tracks": {t: [key_dict(t, k) for k in self.track(t)] for t in TRACKS},
            "amplitude_scale": self.amplitude_scale, "tempo_scale": self.tempo_scale,
            "sound": None if self.sound is None else {"file": self.sound.file, "at_s": self.sound.at_s},
            "draft": self.draft, "description": self.description,
        }

    def digest(self) -> str:
        """SHA-256 of the canonical form (12 significant digits): changes whenever the motion changes."""
        def rounded(o: Any) -> Any:
            if isinstance(o, float):
                return float(f"{o:.12g}")
            if isinstance(o, dict):
                return {k: rounded(v) for k, v in o.items()}
            if isinstance(o, list):
                return [rounded(v) for v in o]
            return o

        blob = json.dumps(rounded(self.to_dict()), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
