"""HW4 experiment trials R1 / R2 / R3 for the Motion Lab (Qt-free; docs/phase4/PHASE4_PLAN.md §3, §5.3, §7).

A trial is one recorded run: a hold at the start state (>= 1 s), the motion, a hold at its end state (>= 2 s), so
the record contains the still parts the tracking metrics need (static bias, overshoot, settling time).

* R1  one axis, a very small move: rest -> target in 1.0 s (minjerk), 1.0 s at the target, back in 1.0 s.
      Amplitude ladder 2 -> 5 -> 10 deg (z 1 -> 3 -> 5 mm, one antenna alone 5 -> 15 -> 30 deg); the GUI also takes
      a typed amplitude inside ``R1_MIN_AMPLITUDE`` .. ``R1_MAX_AMPLITUDE`` (``r1_amplitude_problem``).
* R2  one confirmed motion file, unchanged (``motions/*.yaml``).
* R3  a pitch nod: rest -> +A in D -> 1.0 s -> rest in D. One ladder at a time: D with A fixed (default A 20 deg,
      D 1.0 ... 0.2 s on the robot) or A with D fixed (small A first; on the robot at D >= 1 s).

``limit_points`` gives the "limit map" of the Results tab: the computed peak Stewart speed ratio and peak joint
acceleration of the confirmed motions and of the R3 duration ladder.

R1 / R3 motions are test specs built from the library types; ``name`` / ``variant`` are only the carriers the spec
format requires (the trial label names the run). Every trial goes through the motion engine: ``prepare(spec)``
(dense + stream check of the motion against the initial policy) and ``prepare_timeline`` of the recorded timeline.
A trial outside the policy cannot be prepared, so it cannot run anywhere. Nothing here sends a command; every
number is ``computed``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from reachy_motion.library.pose import REST, Pose6
from reachy_motion.library.spec import (AntennaKey, Easing, HeadKey, MotionSpec, Principle, PrincipleUse)
from reachy_motion.safety.policy import DEFAULT_POLICY, SafetyPolicy, SafetyReport

STAGES = ("R0", "R1", "R2", "R3")
PRE_S = 1.0  # PHASE4_PLAN §7: >= 1 s still before the motion
POST_S = 2.0  # >= 2 s still after it
R1_MOVE_S, R1_HOLD_S = 1.0, 1.0
#: axis -> (Pose6 / antenna field, unit, amplitude ladder, tracking channel)
R1_AXES: dict[str, tuple[str, str, tuple[float, ...], str]] = {
    "yaw": ("yaw_deg", "deg", (2.0, 5.0, 10.0), "yaw"),
    "pitch": ("pitch_deg", "deg", (2.0, 5.0, 10.0), "pitch"),
    "roll": ("roll_deg", "deg", (2.0, 5.0, 10.0), "roll"),
    "z": ("z_mm", "mm", (1.0, 3.0, 5.0), "z"),
    "antenna_right": ("right_deg", "deg", (5.0, 15.0, 30.0), "antenna_right"),
    "antenna_left": ("left_deg", "deg", (5.0, 15.0, 30.0), "antenna_left"),
}
#: R1 amplitudes typed in the GUI (2026-11-13, user decision: values between the ladder steps for the dead-band
#: curve - on the robot 2 deg gave an amplitude ratio of 0.12-0.20 and 10 deg 0.53-0.69, measured 2026-11-12). R1
#: stays a small-motion check: bigger moves are R2 / R3. Every value still goes through the initial policy.
R1_MIN_AMPLITUDE = 0.1
R1_MAX_AMPLITUDE = {"yaw": 15.0, "pitch": 15.0, "roll": 15.0, "z": 8.0, "antenna_right": 45.0, "antenna_left": 45.0}
R3_HOLD_S = 1.0
R3_DEFAULT_AMPLITUDE_DEG = 20.0
#: The R3 duration ladder, also on the robot (one step at a time). 0.25 / 0.2 s added 2026-11-13 (user decision): their
#: speed ratio (0.18 / 0.22) and peak joint acceleration (1391 / 2166 deg/s^2) stay inside what the robot already ran
#: with aha_recommend expressive (0.386, 2941 deg/s^2, measured 2026-11-12); all values computed (gui check_trial).
R3_DEFAULT_DURATIONS = (1.0, 0.6, 0.4, 0.3, 0.25, 0.2)
R3_SIM_ONLY_DURATIONS = (0.15, 0.1)  # computed and MuJoCo only (0.15 s: 3825 deg/s^2, beyond the robot's tested envelope)
#: The amplitude ladder (D fixed): small A first, one step at a time; on the robot at D >= 1 s only
#: (robot_r1.r3_amplitude_step_problems). A separate plan after the D ladder.
R3_DEFAULT_AMPLITUDES = (10.0, 20.0, 30.0)
R3_DEFAULT_FIXED_DURATION_S = 1.0
REAL_SEPARATE_APPROVAL_BELOW_S = 0.3  # shorter: only after the previous step's measurement, separately approved
REAL_MIN_DURATION_S = 0.2  # shorter R3 steps are outside the real-robot scope: computed / MuJoCo only


@dataclass(frozen=True)
class Trial:
    stage: str  # "R1" | "R2" | "R3"
    label: str  # file-safe run name, e.g. "r3_pitch20_D0.60_rep1"
    title: str  # for people
    condition: dict[str, Any]  # what was set: axis / amplitude / D / motion / variant / rep
    spec: MotionSpec  # the planned motion (R2: the confirmed file; R1 / R3: a test spec)
    pre_s: float = PRE_S
    post_s: float = POST_S
    main_channel: str = "pitch"  # tracking channel shown live and summarised first


@dataclass
class TrialCheck:
    """The motion engine's verdict on a trial (computed): may it run, what does it demand, where may it run."""

    trial: Trial
    runnable: bool
    report: SafetyReport | None  # dense check of the motion (initial policy)
    prepared: Any = None  # PreparedMotion of the recorded timeline (holds + motion), or None
    record_s: float = math.nan  # length of the recorded timeline
    real_scope: str = ""  # where a real-robot run of this trial would stand (PHASE4_PLAN), text
    notes: list[str] = field(default_factory=list)

    @property
    def metrics(self) -> dict[str, Any]:
        return {} if self.report is None else dict(self.report.metrics)

    def row(self) -> dict[str, Any]:
        """Display row: required speed ratios, joint margin, verdict (computed)."""
        m = self.metrics
        return {"label": self.trial.label, "title": self.trial.title, **self.trial.condition,
                "record_s": self.record_s, "stewart_speed_ratio": m.get("peak_stewart_speed_ratio"),
                "antenna_speed_ratio": m.get("peak_antenna_speed_ratio"),
                "body_speed_ratio": m.get("peak_body_speed_ratio"),
                "min_margin_deg": m.get("min_stewart_margin_deg"), "min_margin_joint": m.get("min_stewart_margin_joint"),
                "policy": "통과" if self.runnable else "거부",
                "violations": "" if self.report is None else ", ".join(v.code for v in self.report.violations),
                "warnings": "" if self.report is None else ", ".join(w.code for w in self.report.warnings),
                "real_scope": self.real_scope, "data_kind": "computed"}


def _test_spec(intent: str, head: Sequence[HeadKey] = (), antennas: Sequence[AntennaKey] = (),
               start: Any = REST) -> MotionSpec:
    return MotionSpec(name="success", variant="expressive", intent=intent, valence=0.0, arousal=0.0,
                      principles=(PrincipleUse(Principle.TIMING, "experiment trial: a test motion, not a design"),),
                      head=tuple(head), antennas=tuple(antennas), start=start, draft=False,
                      description="HW4 experiment test motion (Motion Lab); name / variant are format carriers only")


def r1_trial(axis: str, amplitude: float, sign: int = 1, rep: int = 1, move_s: float = R1_MOVE_S,
             hold_s: float = R1_HOLD_S, antennas: tuple[float, float] | None = None) -> Trial:
    """One R1 trial: ``axis`` +- ``amplitude`` (deg, mm for z) from rest and back (PHASE4_PLAN §3 R1).

    ``antennas`` (rad, [right, left]): start the trial with the antennas there instead of at the SDK rest - the
    robot's limp antennas as R0 read them, which the start-up does not move (docs/phase4/STARTUP_STOP.md §11). A head
    trial then holds them for the whole record; an antenna trial moves one antenna +-amplitude from there."""
    if axis not in R1_AXES:
        raise ValueError(f"unknown R1 axis {axis!r} (one of {', '.join(R1_AXES)})")
    fname, unit, _, channel = R1_AXES[axis]
    a = math.copysign(abs(float(amplitude)), sign)
    if not math.isfinite(a) or a == 0.0:
        raise ValueError("the R1 amplitude must be a non-zero number")
    title = f"R1 {axis} {a:+g} {unit}"
    start = REST if antennas is None else replace(REST, antennas=(float(antennas[0]), float(antennas[1])))
    if axis.startswith("antenna"):
        r, left = start.antennas
        target = (r + math.radians(a), left) if axis == "antenna_right" else (r, left + math.radians(a))
        spec = _test_spec(title, antennas=(AntennaKey(move_s, target, Easing.MIN_JERK, hold_s, "target"),
                                           AntennaKey(move_s, start.antennas, Easing.MIN_JERK, 0.0, "back")),
                          start=start)
    else:
        target_pose = Pose6.from_deg_mm(**{fname: a})
        spec = _test_spec(title, head=(HeadKey(move_s, target_pose, Easing.MIN_JERK, hold_s, "target"),
                                       HeadKey(move_s, Pose6(), Easing.MIN_JERK, 0.0, "back")), start=start)
    label = f"r1_{axis}_{'p' if a > 0 else 'm'}{abs(a):g}{unit}_rep{rep}"
    cond = {"axis": axis, "amplitude": a, "unit": unit, "move_s": move_s, "hold_s": hold_s, "rep": rep}
    if antennas is not None:
        cond["antennas_start_deg"] = [round(math.degrees(v), 3) for v in start.antennas]
    return Trial("R1", label, f"{title} #{rep}" + (" (안테나 유지)" if antennas is not None else ""), cond, spec,
                 main_channel=channel)


def dominant_channel(spec: MotionSpec) -> str:
    """The head rotation axis with the largest designed excursion (what a person looks at first)."""
    best, name = -1.0, "pitch"
    for axis in ("pitch", "yaw", "roll"):
        vals = [abs(math.degrees(getattr(k.pose, axis))) for k in spec.effective().head]
        if vals and max(vals) > best:
            best, name = max(vals), axis
    return name


def r2_trial(spec: MotionSpec, rep: int = 1, pre_s: float = PRE_S, post_s: float = POST_S) -> Trial:
    """One R2 trial: a confirmed motion exactly as in its file, with the still parts before and after."""
    return Trial("R2", f"r2_{spec.name}_{spec.variant}_rep{rep}", f"R2 {spec.key} #{rep}",
                 {"motion": spec.name, "variant": spec.variant, "rep": rep, "duration_s": spec.duration,
                  "draft": spec.draft, "spec_digest": spec.digest()[:12]}, spec, pre_s, post_s,
                 main_channel=dominant_channel(spec))


def r3_trial(amplitude_deg: float, duration_s: float, rep: int = 1, hold_s: float = R3_HOLD_S,
             easing: Easing = Easing.MIN_JERK, vary: str = "duration") -> Trial:
    """One R3 step: pitch +A in D, hold, back in D (PHASE4_PLAN §5.3; D = one-way travel time)."""
    a, d = float(amplitude_deg), float(duration_s)
    if not (math.isfinite(a) and a > 0 and math.isfinite(d) and d > 0):
        raise ValueError("R3 needs A > 0 deg and D > 0 s")
    title = f"R3 pitch {a:g}°, D {d:g} s"
    spec = _test_spec(title, head=(HeadKey(d, Pose6.from_deg_mm(pitch_deg=a), easing, hold_s, "up"),
                                   HeadKey(d, Pose6(), easing, 0.0, "back")))
    return Trial("R3", f"r3_pitch{a:g}_D{d:.2f}_rep{rep}", f"{title} #{rep}",
                 {"amplitude_deg": a, "duration_s": d, "easing": Easing(easing).value, "hold_s": hold_s, "rep": rep,
                  "vary": vary}, spec, main_channel="pitch")


def r3_ladder(vary: str, values: Sequence[float], fixed: float, rep: int = 1) -> list[Trial]:
    """The steps of one ladder; only ONE parameter changes (``vary`` = "duration" or "amplitude")."""
    if vary == "duration":
        return [r3_trial(fixed, d, rep, vary=vary) for d in values]
    if vary == "amplitude":
        return [r3_trial(a, fixed, rep, vary=vary) for a in values]
    raise ValueError("vary must be 'duration' or 'amplitude'")


def parse_values(text: str) -> list[float]:
    """'1.0, 0.6 0.4;0.3' -> [1.0, 0.6, 0.4, 0.3] (finite positive numbers, order kept)."""
    out = []
    for tok in text.replace(";", ",").replace(" ", ",").split(","):
        if tok.strip():
            v = float(tok)
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{tok!r} is not a positive number")
            out.append(v)
    if not out:
        raise ValueError("no values")
    return out


def record_timeline(trial: Trial, motion_timeline: Any) -> Any:
    """Start hold + motion + end hold on the motion's grid, named after the trial (planner.concat)."""
    from reachy_motion.planner import concat, plan_transition

    tl = motion_timeline
    rate = tl.rate_hz
    if trial.pre_s > 0:
        tl = concat(plan_transition(trial.spec.start, trial.spec.start, trial.pre_s, Easing.MIN_JERK, rate,
                                    name="hold_before"), tl)
    if trial.post_s > 0:
        end = trial.spec.end_state
        tl = concat(tl, plan_transition(end, end, trial.post_s, Easing.MIN_JERK, rate, name="hold_after"))
    return replace(tl, name=trial.label, variant="")


def real_scope(trial: Trial, runnable: bool) -> str:
    """Where a real-robot run of this trial would stand (PHASE4_PLAN §3, §5.3). Text only: Step A never runs it."""
    if not runnable:
        return "초기 정책 밖: 어디서도 실행하지 않음"
    if trial.stage == "R3":
        d = float(trial.condition["duration_s"])
        if d < REAL_MIN_DURATION_S - 1e-12:
            return f"실제 로봇 제외 (D < {REAL_MIN_DURATION_S:g} s): 계산·MuJoCo만"
        if trial.condition.get("vary") == "amplitude":
            if d < R3_DEFAULT_FIXED_DURATION_S - 1e-9:
                return (f"진폭 사다리는 실제 로봇에서 D ≥ {R3_DEFAULT_FIXED_DURATION_S:g} s만 (짧은 D는 D 사다리로): "
                        "계산·MuJoCo만")
            return "실제 로봇 후보: 진폭 사다리 (작은 A부터 한 단계씩, 앞 단계 실측이 기준 안일 때만)"
        if d < REAL_SEPARATE_APPROVAL_BELOW_S - 1e-12:
            return "앞 단계를 실측한 뒤 따로 승인해야 실제 로봇 가능"
        return "실제 로봇 후보 (PHASE4_PLAN §5.3), 단계마다 따로 승인"
    if trial.stage == "R1":
        first = R1_AXES[trial.condition["axis"]][2][0]
        return ("실제 로봇 후보: R1 첫 단계" if abs(trial.condition["amplitude"]) <= first + 1e-9 else
                "더 작은 R1 단계에서 이상이 없을 때만 실제 로봇 가능")
    return "R1을 마친 뒤에만 실제 로봇 가능 (PHASE4_PLAN §3 R2)"


def check_trial(trial: Trial, policy: SafetyPolicy = DEFAULT_POLICY) -> TrialCheck:
    """``prepare(spec)`` (dense) and ``prepare_timeline`` of the recorded timeline: both must pass to run."""
    from reachy_motion.safety.checker import prepare, prepare_timeline

    problems = trial.spec.validate()
    if problems:
        return TrialCheck(trial, False, None, notes=["모션 데이터 오류: " + "; ".join(problems)],
                          real_scope=real_scope(trial, False))
    try:
        motion = prepare(trial.spec, policy)
    except ValueError as exc:
        return TrialCheck(trial, False, None, notes=[f"계획할 수 없음: {exc}"], real_scope=real_scope(trial, False))
    notes: list[str] = []
    prepared = None
    record_s = math.nan
    if motion.runnable:
        prepared = prepare_timeline(record_timeline(trial, motion.timeline), policy)
        record_s = float(prepared.timeline.duration)
        if not prepared.runnable:
            notes.append("정지 구간을 포함한 기록 타임라인이 정책 밖: "
                         + ", ".join(v.code for v in prepared.report.violations))
    else:
        notes.append("초기 정책 밖: " + ", ".join(v.code for v in motion.report.violations))
    runnable = bool(motion.runnable and prepared is not None and prepared.runnable)
    return TrialCheck(trial, runnable, motion.report, prepared if runnable else None, record_s,
                      real_scope(trial, runnable), notes)


def r1_amplitude_problem(axis: str, amplitude: float) -> str | None:
    """Why a typed R1 amplitude (|value|, deg or mm) is not an R1 step, or None."""
    unit = R1_AXES[axis][1]
    hi = R1_MAX_AMPLITUDE[axis]
    a = abs(float(amplitude))
    if not (math.isfinite(a) and R1_MIN_AMPLITUDE - 1e-12 <= a <= hi + 1e-12):
        return (f"진폭 {a:g} {unit}는 R1 범위 밖입니다 ({R1_MIN_AMPLITUDE:g}–{hi:g} {unit}). R1은 작은 동작 확인이고, "
                "더 큰 동작은 R2·R3에서 합니다")
    return None


_LIMIT_POINTS: dict[tuple[Any, ...], list[dict[str, Any]]] = {}


def limit_points(library: Any, policy: SafetyPolicy = DEFAULT_POLICY) -> list[dict[str, Any]]:
    """The "limit map" (data_kind computed): peak Stewart speed ratio (peak joint speed / XL330 no-load speed at 5 V)
    and peak joint acceleration (body + Stewart, the dense 200 Hz check) of every confirmed motion (E / N) and of the
    R3 pitch nod along the duration ladder (A 20 deg; ``robot_scope`` = D >= ``REAL_MIN_DURATION_S``). Neutral
    motions are LINEAR: their velocity steps at the start and end show as large sampled accelerations (infinite in
    theory). Cached per motion digests (about 2 s the first time)."""
    from reachy_motion.library.spec import MOTION_NAMES

    specs = []
    for name in MOTION_NAMES:
        for variant in ("expressive", "neutral"):
            try:
                specs.append(library.get(name, variant))
            except (KeyError, FileNotFoundError):
                continue
    key = (tuple(s.digest() for s in specs), repr(policy))
    if key in _LIMIT_POINTS:
        return [dict(p) for p in _LIMIT_POINTS[key]]
    pts: list[dict[str, Any]] = []
    for s in specs:
        c = check_trial(r2_trial(s), policy)
        m = c.metrics
        pts.append({"kind": "motion", "motion": s.name, "variant": s.variant, "duration_s": s.duration,
                    "speed_ratio": m.get("peak_stewart_speed_ratio"), "accel_dps2": m.get("peak_joint_accel_dps2"),
                    "runnable": c.runnable, "robot_scope": c.runnable, "data_kind": "computed"})
    for d in R3_DEFAULT_DURATIONS + R3_SIM_ONLY_DURATIONS:
        c = check_trial(r3_trial(R3_DEFAULT_AMPLITUDE_DEG, d), policy)
        m = c.metrics
        pts.append({"kind": "r3", "amplitude_deg": R3_DEFAULT_AMPLITUDE_DEG, "duration_s": d,
                    "speed_ratio": m.get("peak_stewart_speed_ratio"), "accel_dps2": m.get("peak_joint_accel_dps2"),
                    "runnable": c.runnable, "robot_scope": c.runnable and d >= REAL_MIN_DURATION_S - 1e-12,
                    "data_kind": "computed"})
    _LIMIT_POINTS[key] = pts
    return [dict(p) for p in pts]


__all__ = ["STAGES", "PRE_S", "POST_S", "R1_AXES", "R1_MIN_AMPLITUDE", "R1_MAX_AMPLITUDE", "R3_DEFAULT_DURATIONS",
           "R3_SIM_ONLY_DURATIONS", "R3_DEFAULT_AMPLITUDE_DEG", "R3_DEFAULT_AMPLITUDES", "R3_DEFAULT_FIXED_DURATION_S",
           "Trial", "TrialCheck", "r1_trial", "r2_trial", "r3_trial", "r3_ladder", "parse_values", "record_timeline",
           "check_trial", "real_scope", "dominant_channel", "r1_amplitude_problem", "limit_points"]
