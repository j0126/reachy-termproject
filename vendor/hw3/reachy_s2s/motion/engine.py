"""Motion engine: interaction state + events + time -> target pose (no threads, no I/O).

Layers per tick (design section 5), as deviations from the neutral pose:
state base pose (minimum-jerk move + small continuous "life" motion) + one event gesture +
listening attention + speaking motion; the sum is scaled by the intensity.

On a state change everything that was moving (gesture, speaking sway, attention, the old
state's life motion) is frozen into the start of the move to the new state: the old motion is
cancelled at that tick, and nothing jumps. The move lasts the preset's ``transition_s`` or longer
when the distance (at the current intensity) needs it to stay under the speed / acceleration
caps, so the limiter downstream stays a safety net. The PROCESSING pose waits ``dwell_s`` first,
so short Cloud turns go from LISTENING to SPEAKING without ever starting the thinking pose.
"""

from __future__ import annotations


from ..config import MotionConfig
from .limits import Limits
from .pose import NEUTRAL, Pose, perk
from .presets import (
    ATTENTION_ATTACK_S,
    ATTENTION_PERK,
    ATTENTION_RELEASE_S,
    ERROR_SHAKE,
    INTERRUPT_PERK,
    MERGED_NOD,
    PRESETS,
    Gesture,
    StatePreset,
    speech_started_gesture,
)
from .speech import SpeechMotion
from .trajectory import Blend, minjerk, ramp_duration

SPEAKING_MOTIONS = ("custom", "sdk", "off")
GESTURE_FADE_S = 0.15  # a gesture replaced by another fades out this fast
SPEECH_START_DEDUP_S = 0.5  # local VAD and the server's ACTIVITY_START report the same speech start
TRANSITION_MARGIN = 1.25  # state moves: planned peak speed / acceleration at most 1 / 1.25 of the caps
GESTURE_TRANSITION_MARGIN = 2.0  # ... at most half of them when a gesture starts with the move
GESTURE_STATES = ("INTERRUPTED", "ERROR")


class MotionEngine:
    def __init__(self, mcfg: MotionConfig, speech: SpeechMotion | None = None, start_ns: int = 0) -> None:
        if mcfg.speaking_motion not in SPEAKING_MOTIONS:
            raise ValueError(f"[motion] speaking_motion must be one of {SPEAKING_MOTIONS}")
        speech_started_gesture(mcfg.speech_started_gesture, mcfg.nod_deg)  # validates the name
        self.intensity = float(mcfg.intensity)
        self.speaking_motion = mcfg.speaking_motion
        self.speech_started = mcfg.speech_started_gesture
        self.nod_deg = mcfg.nod_deg
        self.merged_nod = False
        self.speech = speech
        self.limits = Limits.from_config(mcfg)
        self.last_transition_s = 0.0
        self.state = "IDLE"
        self.state_t0 = start_ns
        self._preset: StatePreset = PRESETS["IDLE"]
        self._blend = Blend(Pose(), self._preset.offset, start_ns, self._preset.transition_s)
        self._life_t0 = start_ns
        self._gesture: tuple[Gesture, int] | None = None
        self._fade: tuple[Pose, int] | None = None
        self._attention = Blend(Pose(), Pose(), start_ns, 0.0)  # antenna lift while the user speaks
        self._attention_target = 0.0
        self._last_speech_start_ns: int | None = None
        self._last_t: int | None = None
        self.last_dev = Pose()  # last composed deviation (unscaled)

    # -- inputs ------------------------------------------------------------------
    def set_state(self, state: str, t_ns: int) -> bool:
        """Switch to ``state`` at ``t_ns`` (cancels what was moving). False if unknown / unchanged."""
        if state == self.state or state not in PRESETS:
            return False
        preset = PRESETS[state]
        margin = GESTURE_TRANSITION_MARGIN if state in GESTURE_STATES else TRANSITION_MARGIN
        self._move_to(preset, self.last_dev, t_ns, preset.dwell_s, margin)
        self.state, self.state_t0 = state, t_ns
        if state == "INTERRUPTED":
            self._start_gesture(INTERRUPT_PERK, t_ns)
        elif state == "ERROR":
            self._start_gesture(ERROR_SHAKE, t_ns)
        return True

    def rebase(self, dev: Pose, t_ns: int) -> None:
        """Continue from ``dev`` (e.g. the neutral hold pose when Motion turns on) into the current state."""
        self.last_dev = dev
        self._move_to(self._preset, dev, t_ns, 0.0, TRANSITION_MARGIN)

    def _move_to(self, preset: StatePreset, start: Pose, t_ns: int, dwell_s: float, margin: float) -> None:
        self._preset = preset
        t0 = t_ns + int(dwell_s * 1e9)
        k = max(self.intensity, 1e-6)
        dur = ramp_duration(start.scaled(k), preset.offset.scaled(k), self.limits, preset.transition_s,
                            margin=margin)
        self.last_transition_s = dur
        self._blend = Blend(start, preset.offset, t0, dur)
        self._life_t0 = t0
        self._gesture = None
        self._fade = None
        self._attention = Blend(Pose(), Pose(), t_ns, 0.0)
        self._attention_target = 0.0
        if self.speech is not None:
            self.speech.reset_output(t_ns)

    def on_event(self, kind: str, t_ns: int) -> str | None:
        """``speech_start`` / ``speech_end`` / ``merged``; returns the gesture started, if any."""
        if kind == "speech_start" and self.state == "LISTENING":
            last = self._last_speech_start_ns
            if last is not None and (t_ns - last) / 1e9 < SPEECH_START_DEDUP_S:
                return None
            self._last_speech_start_ns = t_ns
            self._set_attention(1.0, t_ns)
            gesture = speech_started_gesture(self.speech_started, self.nod_deg)
            if gesture is not None:
                self._start_gesture(gesture, t_ns)
                return gesture.name
        elif kind == "speech_end":
            self._set_attention(0.0, t_ns)
        elif kind == "merged" and self.merged_nod and self.state == "PROCESSING":
            self._start_gesture(MERGED_NOD, t_ns)
            return MERGED_NOD.name
        return None

    def _set_attention(self, target: float, t_ns: int) -> None:
        if target == self._attention_target:
            return
        dur = ATTENTION_ATTACK_S if target > self._attention_target else ATTENTION_RELEASE_S
        self._attention = Blend(self._attention.at(t_ns), perk(ATTENTION_PERK * target), t_ns, dur)
        self._attention_target = target

    def _gesture_offset(self, t_ns: int) -> Pose:
        if self._gesture is None:
            return Pose()
        g, t0 = self._gesture
        s = (t_ns - t0) / 1e9 / g.duration_s
        if s >= 1.0:
            self._gesture = None
            return Pose()
        off = g.at(s)
        if self.speaking_motion == "sdk" and self.state == "SPEAKING":
            off = off.antennas_only()  # SDK wobbling owns the head while speaking
        return off

    def _start_gesture(self, g: Gesture, t_ns: int) -> None:
        current = self._gesture_offset(t_ns) + self._fade_offset(t_ns)
        if current != Pose():
            self._fade = (current, t_ns)
        self._gesture = (g, t_ns)

    def _fade_offset(self, t_ns: int) -> Pose:
        if self._fade is None:
            return Pose()
        off, t0 = self._fade
        f = (t_ns - t0) / 1e9 / GESTURE_FADE_S
        if f >= 1.0:
            self._fade = None
            return Pose()
        return off.scaled(1.0 - minjerk(f))

    # -- output --------------------------------------------------------------------
    def step(self, t_ns: int) -> Pose:
        """Absolute target pose for ``t_ns`` (box, joint guard and speed limits come after)."""
        self._last_t = t_ns
        dev = self._blend.at(t_ns)
        life = self._preset.life
        if life is not None and t_ns >= self._life_t0:
            dev = dev + life((t_ns - self._life_t0) / 1e9).scaled(self._blend.fraction(t_ns))
        dev = dev + self._gesture_offset(t_ns) + self._fade_offset(t_ns)
        if self.state == "LISTENING":
            dev = dev + self._attention.at(t_ns)
        if self.speech is not None:
            offsets = self.speech.step(t_ns, speaking=self.state == "SPEAKING")
            if self.speaking_motion == "custom" and self.state == "SPEAKING":
                dev = dev + offsets
        self.last_dev = dev
        return NEUTRAL + dev.scaled(self.intensity)
