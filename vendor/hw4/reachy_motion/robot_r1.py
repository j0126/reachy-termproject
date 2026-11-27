"""C1 and R1 runners (Steps B / C): the terminal side of the first robot start and of one R1 trial, prepared and
tested on the Mock and in the SDK offline harness only (docs/phase4/C1_R1_CHECKLIST.md, FIRST_ROBOT_TEST.md).

* C1 (``robot-c1``): torque on where the head rests, hold, torque off - no motion (``backends.robot_connect.run_c1``,
  run and recorded by ``c1_stage``). Its approval binds ``C1:hold_present``: a hold of the APPROVED R0 state (a fresh
  R0 must match it within ``STATE_TOL``), checked with the start-up envelope of that state.
* R1 (``robot-r1``): one R1 trial (``gui.experiments.r1_trial``, the Motion Lab's definition and checks) between two
  ramps planned from the approved R0 state. The ramps and the trial are approved SEPARATELY (user decision
  2026-11-12), each approval binding its own policy:
  - R1-0 ``ramp_request``: ``R1:startup_hold`` (the slow ramp from the resting pose to REST with the antennas held -
    never swept, STARTUP_STOP.md §11) and ``R1:return_hold`` (that ramp backwards), start-up policy of this start.
    NOT a small motion: the Stewart joints travel up to ~90 deg (``ramp_travel``, computed: 89.9 deg on stewart_5
    from the 2026-11-11 R0) - ``ramp_summary`` is printed by approval-template and by the gate before the confirmation.
  - R1 ``trial_request``: ``R1:<trial>`` only, default initial policy.
  Each part is recorded as its own run (``runs/01_r1_startup_ramp``, ``02_<trial>``, ``03_r1_return_ramp``: commands,
  present, ticks, manifest with the data_kind), which the Results tab and the analysis modules read as they are.
* R2 / R3 (``robot-r2`` / ``robot-r3``, 2026-11-12): the same runner around any Motion Lab trial (``build_plan``) -
  a confirmed motion (``r2_plan``) or one step of the R3 duration ladder (``r3_plan``, one step at a time:
  ``r3_step_problems``). A trial whose start antennas are not where R0 read them is refused - the antennas are never
  moved to their rest angle automatically; ``hold_antennas`` is the labelled head-only diagnosis. On the robot both
  need a clean measured C1 and R1 first (``prerequisite_problems``); the same R1-0 ramp approval covers them.

End of a run (``run_trial``, user decision 2026-11-12). The torque is switched off automatically only when nothing
went wrong (nothing was sent yet, or all three parts completed) and a fresh present state is within ``END_TOL`` of
where the head rested when the run started (with the torque off it then stays where it is); the torque-off is read
back and watched for ``OFF_WATCH_S``. A stop anywhere else - S1 stop request, S2 link / measurement loss, S3 daemon
error, an exception - is never followed by an automatic torque-off or an automatic return (with the torque off the
head goes wherever gravity takes it, up to ~29 deg / 51 mm from REST): the torque stays on at the last target or its
state is unknown, the failure is recorded with where it stopped, the operator supports the head and may confirm the
torque-off (``confirm_torque_off``), else the record carries ``OPERATOR_PROCEDURE_KO``.

``--target mock``: a Mock robot that starts where R0 saw the real one; the same timelines and records (data_kind
``mock``). The Mock has no torque: no torque-off is sent and none is claimed.
``--target robot``: a fresh R0 (allowed GETs only), the readiness (``r1_readiness``), the fresh state compared with the
approved R0, the approval gate of every part (armed + a human-written approval file + its id typed at a terminal),
then ``ROBOT_RUN_ENABLED`` (True since 2026-11-12; False stops here: no client, nothing sent). The operator then
confirms the controlled environment (``operator_confirms_environment``; not confirmed - no run); C1 runs ``c1_stage``,
R1 / R2 / R3 connect through ``SdkBackend.connect`` and run. A daemon on a loopback address is never the robot: its R0
is labelled ``fake_daemon``. The Motion Lab GUI (``gui.robot_session``) is the main path; this CLI is for debugging.

The approval requests come from ``ramp_request`` / ``trial_request`` / ``c1_request`` - the same functions
``approval-template --r1 ... --part ramp|trial`` / ``--c1`` print, so they always agree. The labels exist because the
approval gate requires each report's ``name:variant`` among the requested motions (safety/approval.py:686-707); a
label does not change a digest (timeline.py:69-76 hashes the samples only).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from reachy_motion.library.pose import Pose6, RobotState

ROBOT_RUN_ENABLED = True  # user decision 2026-11-12 (CLAUDE.md rule 3): runs a person approved (GUI or CLI)
PREFLIGHT_OUT = Path("results") / "phase4" / "r1_preflight"
FAKE_KIND = "fake_daemon"
C1_KEY, RAMP_VARIANT, RETURN_VARIANT = "C1:hold_present", "startup_hold", "return_hold"
RAMP_LABEL, RETURN_LABEL = "r1_startup_ramp", "r1_return_ramp"
#: how far a fresh R0 may be from the approved one (else: R0 again, approval again)
STATE_TOL = {"joint_deg": 1.0, "antenna_deg": 2.0}
#: the automatic torque-off at the end needs the head this close to where it rested at the start of the run
#: (initial policy values, not a mechanical guarantee)
END_TOL = {"rotation_deg": 2.0, "translation_mm": 2.0, "antenna_deg": 2.0, "body_deg": 2.0}
OFF_WATCH_S = 2.0  # watch after the torque-off (as C1)
OFF_DRIFT_WARN_DEG = 1.0
PRESENT_FRESH_S = 0.5  # a present state older than this is not evidence of where the head is
ENV_TOKEN = "controlled"  # what the operator types to confirm the controlled environment
R2_DEFAULT_MOTIONS = ("aha_recommend", "boredom", "success")  # (A) intention, (B) state, (C) reaction: 3 E/N pairs
R3_BASE_S = 1.0  # the base step of the R3 duration ladder (gui.experiments.R3_DEFAULT_DURATIONS[0])
R3_BASE_A_DEG = 10.0  # the base step of the R3 amplitude ladder (gui.experiments.R3_DEFAULT_AMPLITUDES[0])
#: When the previous R3 step ends the ladder ("tracking broke down"), relative to the robot baseline measured
#: 2026-11-12 at D 1.0 s (pitch 20 deg): amplitude ratio 0.92-1.03, lag 184-188 ms (r2 0.96-0.97), lag-compensated
#: RMSE 1.39-1.65 deg (7-8 % of the amplitude), overshoot ~0 deg. The former absolute peak error 10 deg / lag 200 ms
#: tripped on that constant delay alone (D 0.6 s: about 0.19 s x 62.5 deg/s = 11.8 deg, computed); revised 2026-11-13
#: (user decision): the delay counts only when it GROWS, the shape error only after the delay is removed.
R3_LIMITS = {
    "amplitude_ratio_min": 0.8,  # the head no longer reaches the commanded amplitude
    "shape_error_ratio_max": 0.15,  # lag-compensated RMSE / commanded peak-to-peak (baseline 0.07-0.08)
    "overshoot_ratio_max": 0.15,  # worst overshoot / commanded peak-to-peak (baseline ~0)
    "lag_increase_ms_max": 100.0,  # lag (fit r2 >= 0.9) above this ladder's D 1.0 s lag
    "lag_ms_max_without_base": 300.0,  # the same, when this ladder's D 1.0 s run is not known (CLI --after only)
}


def parse_r1(text: str) -> tuple[str, float]:
    """``"pitch:+2"`` -> ("pitch", 2.0); the sign is part of the amplitude (default +)."""
    from reachy_motion.gui.experiments import R1_AXES

    axis, sep, amp = str(text).partition(":")
    axis = axis.strip()
    if not sep or axis not in R1_AXES:
        raise ValueError(f"--r1 must be AXIS:AMPLITUDE with AXIS one of {', '.join(R1_AXES)} (e.g. pitch:+2), "
                         f"got {text!r}")
    a = float(amp)
    if not math.isfinite(a) or a == 0.0:
        raise ValueError("the R1 amplitude must be a non-zero number")
    return axis, a


def load_r0(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def r0_antennas(report: dict[str, Any] | None) -> tuple[float, float] | None:
    ant = ((report or {}).get("derived") or {}).get("antennas_deg")
    if isinstance(ant, list) and len(ant) == 2 and all(isinstance(a, (int, float)) for a in ant):
        return (math.radians(float(ant[0])), math.radians(float(ant[1])))
    return None


def r0_state(report: dict[str, Any]) -> RobotState:
    """Where R0 saw the robot: head pose, antennas and body yaw."""
    d = report.get("derived") or {}
    hp = d.get("head_pose")
    ant = r0_antennas(report)
    if not isinstance(hp, dict) or ant is None or not isinstance(d.get("body_yaw_deg"), (int, float)):
        raise ValueError("the R0 report has no head pose, antennas or body yaw")
    head = Pose6.from_deg_mm(**{k: float(hp[k]) for k in ("x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg")})
    return RobotState(head=head, antennas=ant, body_yaw=math.radians(float(d["body_yaw_deg"])))


def _labelled(prep: Any, name: str, variant: str) -> Any:
    """The same timeline and report under the approval's name (samples, checks and digest unchanged)."""
    from reachy_motion.timeline import PreparedMotion

    return PreparedMotion(replace(prep.timeline, name=name, variant=variant),
                          replace(prep.report, name=name, variant=variant), getattr(prep, "spec", None))


@dataclass(frozen=True)
class R1Plan:
    trial: Any  # gui.experiments.Trial
    check: Any  # gui.experiments.TrialCheck (display: speed ratios, margins)
    start: RobotState  # where the run starts: the approved R0 state (or the trial's start without R0)
    ramp: Any  # PreparedMotion R1:startup_hold (start-up policy of this start)
    run: Any  # PreparedMotion R1:<trial label> (default initial policy, recorded)
    back: Any  # PreparedMotion R1:return_hold (start-up policy of this start)

    @property
    def parts(self) -> tuple[Any, ...]:
        """The timelines that are streamed: ramp, trial, return - or the trial alone (AWAKE: no ramps)."""
        return tuple(p for p in (self.ramp, self.run, self.back) if p is not None)

    @property
    def runnable(self) -> bool:
        """The ramps pass the start-up policy of this start, the trial the default initial policy."""
        return bool(self.check.runnable) and all(p.runnable for p in self.parts)


def build_plan(trial: Any, report: dict[str, Any] | None = None, awake: bool = False) -> R1Plan:
    """Any Motion Lab trial (R1 / R2 / R3) between the two ramps planned from the approved R0 state. The trial must
    start where the start-up ramp ends: REST, body 0, the antennas where R0 read them. Start antennas more than
    ``startup_procedure.ANTENNA_HOLD_TOL_DEG`` away from R0 are refused (ValueError): the antennas are never moved to
    their rest angle automatically (STARTUP_STOP.md §11; user decision 2026-11-12).

    ``awake`` (user decision 2026-11-12): the robot was woken the usual way and stands at INIT (= REST with the SDK
    rest antennas, body 0), so there are no ramps: the trial alone, which must start AND end at INIT."""
    from reachy_motion.gui.experiments import check_trial
    from reachy_motion.library.pose import REST
    from reachy_motion.safety.startup import prepare_return_ramp
    from reachy_motion.startup_procedure import plan_startup, startup_end

    check = check_trial(trial)
    if check.prepared is None:
        raise ValueError(f"{trial.title}: outside the initial policy - " + "; ".join(check.notes))
    if awake:
        for what, state in (("start", trial.spec.start), ("end", trial.spec.end_state)):
            g = _gap(state, REST)
            if max(g.values()) > 1e-3:
                raise ValueError(f"{trial.title}: an AWAKE trial must {what} at INIT (REST, SDK rest antennas, body 0): "
                                 f"{ {k: round(v, 3) for k, v in g.items()} }")
        return R1Plan(trial, check, trial.spec.start, None, _labelled(check.prepared, trial.stage, trial.label), None)
    start = r0_state(report) if report else trial.spec.start
    end = startup_end(start, trial.spec.start.antennas)  # ValueError when the antennas would have to move
    gap = _gap(trial.spec.start, end)
    if max(gap["rotation_deg"], gap["translation_mm"], gap["body_deg"]) > 1e-6:
        raise ValueError(f"{trial.title}: does not start where the start-up ramp ends (REST, body 0): {gap}")
    ramp = plan_startup(start, antennas=trial.spec.start.antennas)
    back = prepare_return_ramp(ramp, start)
    # Two approvals (ramp_request / trial_request), one policy each (safety/approval.py approval_template): the ramps
    # under the start-up policy of THIS start (the default one widened only as far as the resting pose; the template
    # lists the widened fields), the trial under the default initial policy (check_trial).
    return R1Plan(trial, check, start, _labelled(ramp, "R1", RAMP_VARIANT),
                  _labelled(check.prepared, trial.stage, trial.label), _labelled(back, "R1", RETURN_VARIANT))


def build(r1: str, rep: int = 1, report: dict[str, Any] | None = None, awake: bool = False) -> R1Plan:
    """One R1 trial and its two ramps; with an R0 report everything starts where R0 read the robot and the
    antennas stay there. ``awake``: no ramps, the trial starts at INIT with the SDK rest antennas."""
    from reachy_motion.gui.experiments import r1_trial

    axis, a = parse_r1(r1)
    held = None if awake or not report else r0_antennas(report)
    return build_plan(r1_trial(axis, abs(a), 1 if a > 0 else -1, int(rep), antennas=held), report, awake)


def hold_antennas(trial: Any, antennas: tuple[float, float]) -> Any:
    """The trial with the antennas held where R0 read them (user decision 2026-11-12: a head-only diagnosis until
    the antenna start and direction are checked). Its antenna keys are removed and it starts with these antennas;
    labelled ``_antheld`` - never reported as the confirmed motion as designed."""
    held = (float(antennas[0]), float(antennas[1]))
    removed = len(trial.spec.antennas)
    spec = replace(trial.spec, antennas=(), start=replace(trial.spec.start, antennas=held))
    cond = dict(trial.condition, antennas_held_deg=[round(math.degrees(v), 3) for v in held],
                antenna_keys_removed=removed)
    note = " (안테나 고정" + (f", 안테나 동작 {removed}개 제거: 머리만 진단)" if removed else ")")
    return replace(trial, label=trial.label + "_antheld", title=trial.title + note, condition=cond, spec=spec)


def _held(trial: Any, report: dict[str, Any] | None) -> Any:
    ant = r0_antennas(report) if report else None
    if ant is None:
        raise ValueError("--antennas hold needs an R0 report with the antenna angles (--r0)")
    return hold_antennas(trial, ant)


def r2_plan(motion: str, variant: str, rep: int = 1, report: dict[str, Any] | None = None,
            hold: bool = False, awake: bool = False) -> R1Plan:
    """One confirmed motion (``motions/<motion>.<variant>.yaml``, draft: false) as an R2 trial between the ramps.
    ``hold``: the head-only diagnosis with the antennas held where R0 read them (``hold_antennas``)."""
    from reachy_motion.gui.experiments import r2_trial
    from reachy_motion.library.io import load_spec_file
    from reachy_motion.library.registry import motions_dir

    spec = load_spec_file(motions_dir() / f"{motion}.{variant}.yaml")
    if spec.draft:
        raise ValueError(f"{spec.key} is a draft: R2 runs the confirmed motions only")
    trial = r2_trial(spec, int(rep))
    if hold and awake:
        raise ValueError("--antennas hold is for the limp start: an AWAKE robot already has its antennas at INIT, "
                         "so the confirmed motion runs as designed")
    return build_plan(_held(trial, report) if hold else trial, report, awake)


def r3_plan(amplitude_deg: float, duration_s: float, rep: int = 1, report: dict[str, Any] | None = None,
            hold: bool = False, awake: bool = False) -> R1Plan:
    """One R3 step (``gui.experiments.r3_trial``: pitch +A in D, hold, back in D) between the ramps (or alone,
    ``awake``)."""
    from reachy_motion.gui.experiments import r3_trial

    trial = r3_trial(float(amplitude_deg), float(duration_s), int(rep))
    if hold and awake:
        raise ValueError("--antennas hold is for the limp start: an AWAKE robot already has its antennas at INIT")
    return build_plan(_held(trial, report) if hold else trial, report, awake)


#: R3 step messages: (English for the CLI, Korean for the Motion Lab GUI). Percentages with one decimal: 15.3 % is
#: not shown as "15 % > 15 %".
_R3_MSG: dict[str, tuple[str, str]] = {
    "amplitude": ("previous step: amplitude ratio {x:.2f} < {lim}",
                  "진폭비 {x:.2f} < {lim}: 명령한 크기만큼 움직이지 못함"),
    "shape": ("previous step: lag-compensated RMSE {e:.2f} deg = {r:.1%} of the amplitude > {lim:.0%}",
              "지연 보정 RMSE {e:.2f}° = 진폭의 {r:.1%} > {lim:.0%}: 동작 모양이 무너짐"),
    "overshoot": ("previous step: overshoot {e:.2f} deg = {r:.1%} of the amplitude > {lim:.0%}",
                  "오버슈트 {e:.2f}° = 진폭의 {r:.1%} > {lim:.0%}"),
    "lag_base": ("previous step: lag {lag:.0f} ms > {what} lag {base:.0f} ms + {inc:g}",
                 "지연 {lag:.0f} ms > {what}의 지연 {base:.0f} ms + {inc:g} ms: 지연이 늘어남"),
    "lag_abs": ("previous step: lag {lag:.0f} ms > {lim:g} ms", "지연 {lag:.0f} ms > {lim:g} ms"),
    "scope": ("D {d:g} s < {m:g} s: outside the real-robot scope (computed / MuJoCo only)",
              "D {d:g} s는 실제 로봇 범위 밖 (D < {m:g} s는 계산·MuJoCo에서만)"),
    "needs_after": ("D {d:g} s is shorter than the base step {b:g} s: --after <the previous step's run folder> is "
                    "required (one step at a time, after looking at its result)",
                    "D {d:g} s는 앞 단계가 먼저: 같은 연결에서 D {b:g} s부터 한 단계씩 (앞 단계 결과를 보고)"),
    "unreadable": ("--after {run}: not a readable run folder ({err})", "앞 단계 기록을 읽을 수 없음 ({err})"),
    "not_r3": ("--after is not an R3 step", "앞 단계가 R3 시행이 아님"),
    "amplitude_differs": ("--after has amplitude {pa}, this step {a:g}: one setting at a time",
                          "앞 단계 진폭 {pa}°, 이 단계 {a:g}°: 한 번에 한 설정만"),
    "wrong_previous": ("--after has D {pd:g} s: the step before D {d:g} s is D {need:g} s",
                       "D {d:g} s의 앞 단계는 D {need:g} s (지금 앞 단계 D {pd:g} s)"),
    "not_longer": ("--after has D {pd:g} s: the previous step must be longer than {d:g} s",
                   "앞 단계 D {pd:g} s는 이 단계 {d:g} s보다 길어야 함"),
    "antennas": ("--after used another antenna mode", "앞 단계의 안테나 방식이 다름"),
    "not_measured": ("--after is {kind!r} data: the robot's next step needs the robot's measurement",
                     "앞 단계가 {kind} 데이터: 실제 로봇 다음 단계는 실제 로봇 측정이 필요함"),
    "not_completed": ("the previous step did not complete", "앞 단계가 끝까지 완료되지 않음"),
    "stopped": ("the previous step stopped: {why}", "앞 단계가 정지함: {why}"),
    "base_unreadable": ("the {what} run of this ladder is not readable ({err})",
                        "이 사다리의 {what} 기록을 읽을 수 없음 ({err})"),
    "metrics_unreadable": ("previous step metrics not readable: {err}", "앞 단계 지표를 읽을 수 없음: {err}"),
    # the amplitude ladder (D fixed, A grows; 2026-11-13 user decision: small A first, one step at a time)
    "a_scope": ("amplitude ladder at D {d:g} s: on the robot only at D >= {b:g} s (a large A in a short D exceeds the "
                "acceleration the robot has run; shorter D: the duration ladder)",
                "진폭 사다리 D {d:g} s: 실제 로봇은 D ≥ {b:g} s에서만 (큰 A·짧은 D는 실제 로봇에서 돌려 본 가속을 넘음, 짧은 D는 D 사다리로)"),
    "needs_after_a": ("A {a:g} deg is larger than the base step {b:g} deg: --after <the previous step's run folder> is "
                      "required (one step at a time, after looking at its result)",
                      "A {a:g}°는 앞 단계가 먼저: 같은 연결에서 A {b:g}°부터 한 단계씩 (앞 단계 결과를 보고)"),
    "duration_differs": ("--after has D {pd:g} s, this step {d:g} s: one setting at a time",
                         "앞 단계 D {pd:g} s, 이 단계 {d:g} s: 한 번에 한 설정만"),
    "wrong_previous_a": ("--after has A {pa:g} deg: the step before A {a:g} deg is A {need:g} deg",
                         "A {a:g}°의 앞 단계는 A {need:g}° (지금 앞 단계 A {pa:g}°)"),
    "not_smaller": ("--after has A {pa:g} deg: the previous step must be smaller than {a:g} deg",
                    "앞 단계 A {pa:g}°는 이 단계 {a:g}°보다 작아야 함"),
}


def _r3_msg(key: str, lang: str, **kw: Any) -> str:
    return _R3_MSG[key][1 if lang == "ko" else 0].format(**kw)


def r3_base_name(vary: str | None) -> str:
    """The first step of a ladder, as the messages name it: "D 1 s" (duration ladder) or "A 10°" (amplitude)."""
    return f"A {R3_BASE_A_DEG:g}°" if vary == "amplitude" else f"D {R3_BASE_S:g} s"


def r3_metric_problems(m: Any, k: float, base_lag_s: float | None = None, lang: str = "en",
                       base_what: str | None = None) -> list[str]:
    """``R3_LIMITS`` on a step's main-channel metrics ``m`` (SI; ``k`` = factor to deg). ``base_lag_s``: the lag of
    this ladder's first step (D 1.0 s, or A 10 deg on the amplitude ladder; fit r2 >= 0.9), or None (then the absolute
    lag bound applies); ``base_what`` names it in the message. ``lang``: "en" | "ko"."""
    lim = R3_LIMITS
    out = []
    p2p = abs(float(m.command_p2p))
    if m.amplitude_ratio < lim["amplitude_ratio_min"]:
        out.append(_r3_msg("amplitude", lang, x=m.amplitude_ratio, lim=lim["amplitude_ratio_min"]))
    if p2p > 0 and m.dynamic_rmse_lag_compensated / p2p > lim["shape_error_ratio_max"]:
        out.append(_r3_msg("shape", lang, e=m.dynamic_rmse_lag_compensated * k, r=m.dynamic_rmse_lag_compensated / p2p,
                           lim=lim["shape_error_ratio_max"]))
    if p2p > 0 and m.overshoot == m.overshoot and m.overshoot / p2p > lim["overshoot_ratio_max"]:
        out.append(_r3_msg("overshoot", lang, e=m.overshoot * k, r=m.overshoot / p2p, lim=lim["overshoot_ratio_max"]))
    if m.lag_fit_r2 >= 0.9:
        lag_ms = m.lag_s * 1000.0
        if base_lag_s is not None and lag_ms > base_lag_s * 1000.0 + lim["lag_increase_ms_max"]:
            out.append(_r3_msg("lag_base", lang, lag=lag_ms, what=base_what or r3_base_name("duration"),
                               base=base_lag_s * 1000.0, inc=lim["lag_increase_ms_max"]))
        elif base_lag_s is None and lag_ms > lim["lag_ms_max_without_base"]:
            out.append(_r3_msg("lag_abs", lang, lag=lag_ms, lim=lim["lag_ms_max_without_base"]))
    return out


def r3_next_step_problems(run: str | Path, base: str | Path | None = None, lang: str = "en",
                          loaded: Any = None) -> list[str]:
    """Why the R3 ladder stops after the step recorded in ``run`` (empty = the next step may run: a shorter D, or a
    larger A on the amplitude ladder): not completed, an S2 / S3 stop or an error, or ``R3_LIMITS`` (``base`` = this
    ladder's first-step run folder - D 1.0 s, or A 10 deg - which may be ``run`` itself). The same test
    ``r3_step_problems`` makes before the next step; the GUI shows it at once."""
    from reachy_motion.gui.results import load_result

    run = Path(run)
    problems = []
    try:
        man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        loaded = loaded if loaded is not None else load_result(run)
    except (OSError, ValueError, KeyError) as exc:
        return [_r3_msg("unreadable", lang, run=run, err=f"{type(exc).__name__}: {exc}")]
    what = r3_base_name(((man.get("trial") or {}).get("condition") or {}).get("vary"))
    res = man.get("result") or {}
    if not res.get("completed"):
        problems.append(_r3_msg("not_completed", lang))
    if res.get("stop_kind") in ("S2", "S3") or res.get("error"):
        problems.append(_r3_msg("stopped", lang, why=f"{res.get('stop_kind')} {res.get('stop_reason') or res.get('error')}"))
    base_lag = None
    if base is not None:
        try:
            bm = (loaded if Path(base).resolve() == run.resolve() else load_result(base)).metrics["pitch"]
            if bm.lag_fit_r2 >= 0.9:
                base_lag = float(bm.lag_s)
        except (OSError, ValueError, KeyError, AttributeError) as exc:
            problems.append(_r3_msg("base_unreadable", lang, what=what, err=f"{type(exc).__name__}: {exc}"))
    try:
        problems += r3_metric_problems(loaded.metrics["pitch"], loaded.unit("pitch")[0], base_lag, lang, what)
    except (KeyError, AttributeError, TypeError) as exc:
        problems.append(_r3_msg("metrics_unreadable", lang, err=repr(exc)))
    return problems


def r3_step_problems(plan: R1Plan, after: str | Path | None, robot: bool,
                     base: str | Path | None = None, lang: str = "en") -> list[str]:
    """R3 goes one step at a time (PHASE4_PLAN §5.3): the base step is ``R3_BASE_S``; a shorter step needs the
    previous step's run folder (``after``): same amplitude and antenna mode, the next longer duration of the default
    ladder (else any longer one), and ``r3_next_step_problems`` of that step (completed, no S2 / S3 stop, within
    ``R3_LIMITS``; ``base`` = this ladder's D 1.0 s run folder, for the lag growth). On the robot the previous step must
    be measured and D < ``REAL_MIN_DURATION_S`` is out of scope. The amplitude ladder (condition ``vary`` =
    "amplitude") mirrors it: ``r3_amplitude_step_problems``. ``lang``: "en" (CLI) | "ko" (GUI)."""
    from reachy_motion.gui.experiments import R3_DEFAULT_DURATIONS, REAL_MIN_DURATION_S

    c = plan.trial.condition
    a, d = float(c["amplitude_deg"]), float(c["duration_s"])
    problems = []
    if robot and d < REAL_MIN_DURATION_S - 1e-12:
        problems.append(_r3_msg("scope", lang, d=d, m=REAL_MIN_DURATION_S))
    if c.get("vary") == "amplitude":
        return problems + r3_amplitude_step_problems(plan, after, robot, base, lang)
    if after is None:
        if d < R3_BASE_S - 1e-9:
            problems.append(_r3_msg("needs_after", lang, d=d, b=R3_BASE_S))
        return problems
    run = Path(after)
    try:
        man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return problems + [_r3_msg("unreadable", lang, run=run, err=f"{type(exc).__name__}: {exc}")]
    t = man.get("trial") or {}
    pc = t.get("condition") or {}
    pd = float(pc.get("duration_s", math.nan))
    ladder = [x for x in R3_DEFAULT_DURATIONS if x > d + 1e-9]
    if t.get("stage") != "R3":
        problems.append(_r3_msg("not_r3", lang))
    elif not abs(float(pc.get("amplitude_deg", math.nan)) - a) <= 1e-9:
        problems.append(_r3_msg("amplitude_differs", lang, pa=pc.get("amplitude_deg"), a=a))
    elif d in R3_DEFAULT_DURATIONS and ladder and abs(pd - ladder[-1]) > 1e-9:
        problems.append(_r3_msg("wrong_previous", lang, pd=pd, d=d, need=ladder[-1]))
    elif not pd > d:
        problems.append(_r3_msg("not_longer", lang, pd=pd, d=d))
    if pc.get("antennas_held_deg") != c.get("antennas_held_deg"):
        problems.append(_r3_msg("antennas", lang))
    if robot and man.get("data_kind") != "measured":
        problems.append(_r3_msg("not_measured", lang, kind=man.get("data_kind")))
    return problems + r3_next_step_problems(run, base, lang)


def r3_amplitude_step_problems(plan: R1Plan, after: str | Path | None, robot: bool,
                               base: str | Path | None = None, lang: str = "en") -> list[str]:
    """The amplitude ladder (D fixed, A grows; user decision 2026-11-13), the mirror of the duration ladder: the base
    step is A <= ``R3_BASE_A_DEG``; a larger A needs the previous step's run folder (``after``): same D and antenna
    mode, the next smaller amplitude of the default ladder (else any smaller one), and ``r3_next_step_problems`` of
    that step (``base`` = the A 10 deg run of this ladder, for the lag growth). On the robot the previous step must be
    measured and the ladder runs at D >= ``R3_BASE_S`` only: A 30 deg in D 0.2 s needs 3521 deg/s^2 (computed), more
    than the 2941 deg/s^2 of aha_recommend expressive the robot ran (2026-11-12) - a shorter D is the duration
    ladder's question."""
    from reachy_motion.gui.experiments import R3_DEFAULT_AMPLITUDES

    c = plan.trial.condition
    a, d = float(c["amplitude_deg"]), float(c["duration_s"])
    problems = []
    if robot and d < R3_BASE_S - 1e-9:
        problems.append(_r3_msg("a_scope", lang, d=d, b=R3_BASE_S))
    if after is None:
        if a > R3_BASE_A_DEG + 1e-9:
            problems.append(_r3_msg("needs_after_a", lang, a=a, b=R3_BASE_A_DEG))
        return problems
    run = Path(after)
    try:
        man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return problems + [_r3_msg("unreadable", lang, run=run, err=f"{type(exc).__name__}: {exc}")]
    t = man.get("trial") or {}
    pc = t.get("condition") or {}
    pa, pd = float(pc.get("amplitude_deg", math.nan)), float(pc.get("duration_s", math.nan))
    smaller = [x for x in R3_DEFAULT_AMPLITUDES if x < a - 1e-9]
    if t.get("stage") != "R3":
        problems.append(_r3_msg("not_r3", lang))
    elif not abs(pd - d) <= 1e-9:
        problems.append(_r3_msg("duration_differs", lang, pd=pd, d=d))
    elif a in R3_DEFAULT_AMPLITUDES and smaller and abs(pa - smaller[-1]) > 1e-9:
        problems.append(_r3_msg("wrong_previous_a", lang, pa=pa, a=a, need=smaller[-1]))
    elif not pa < a:
        problems.append(_r3_msg("not_smaller", lang, pa=pa, a=a))
    if pc.get("antennas_held_deg") != c.get("antennas_held_deg"):
        problems.append(_r3_msg("antennas", lang))
    if robot and man.get("data_kind") != "measured":
        problems.append(_r3_msg("not_measured", lang, kind=man.get("data_kind")))
    return problems + r3_next_step_problems(run, base, lang)


def prerequisite_problems(c1: dict[str, Any] | None, r1_sessions: list[dict[str, Any]], moves_antennas: bool,
                          awake: bool = False) -> list[str]:
    """Robot R2 / R3 only after a clean, measured C1 and R1 (user decision 2026-11-12: what C1 / R1 did not verify
    is not assumed safe): C1 ok with the torque read back off; an R1 head trial ok (a limp-start run ends with the
    torque off, an AWAKE run stays on at INIT); and when the trial moves the antennas, an R1 trial of EACH antenna ok
    (their direction is otherwise unverified). ``awake``: no C1 - the robot was woken the usual way."""
    problems = []
    if awake:
        pass
    elif not c1:
        problems.append("no C1 report (--c1-report c1_report.json)")
    elif not (c1.get("data_kind") == "measured" and c1.get("ok") is True and c1.get("torque_state") == "off"):
        problems.append(f"C1 not clean: data_kind {c1.get('data_kind')!r}, ok {c1.get('ok')!r}, "
                        f"torque {c1.get('torque_state')!r}")
    clean = [str(s.get("trial", "")) for s in r1_sessions if s.get("data_kind") == "measured" and s.get("ok") is True
             and s.get("torque_state") in ("off", "on")]
    if not any(lbl.startswith("r1_") and "antenna" not in lbl for lbl in clean):
        problems.append("no clean measured R1 head trial (--r1-session session_<R1 trial>.json)")
    if moves_antennas and not all(any(lbl.startswith(f"r1_{side}") for lbl in clean)
                                  for side in ("antenna_right", "antenna_left")):
        problems.append("this trial moves the antennas: a clean measured R1 trial of EACH antenna is required first "
                        "(their direction is unverified)")
    return problems


def check_prerequisites(c1_path: str | None, r1_paths: list[str] | None, moves_antennas: bool,
                        awake: bool = False) -> list[str]:
    """``prerequisite_problems`` for the record files given on the command line."""
    try:
        c1 = json.loads(Path(c1_path).read_text(encoding="utf-8")) if c1_path else None
        r1s = [json.loads(Path(p).read_text(encoding="utf-8")) for p in (r1_paths or [])]
    except (OSError, ValueError) as exc:
        return [f"record not readable: {exc}"]
    return prerequisite_problems(c1, r1s, moves_antennas, awake)


def ramp_travel(prep: Any) -> dict[str, Any]:
    """Stewart joint travel over a ramp: per joint the range of its IK joint angle over the samples (computed with
    the SDK kinematics, ``safety.checker.analyze_timeline``) and the largest."""
    from reachy_motion.safety.checker import analyze_timeline
    from reachy_motion.startup_procedure import JOINT_NAMES

    q = np.asarray(analyze_timeline(prep.timeline).joints, dtype=float)
    travel = np.degrees(np.ptp(q[:, 1:7], axis=0))
    k = int(np.argmax(travel))
    return {"max_deg": round(float(travel[k]), 1), "joint": JOINT_NAMES[1 + k], "data_kind": "computed",
            "per_joint_deg": {JOINT_NAMES[1 + i]: round(float(v), 1) for i, v in enumerate(travel)}}


def ramp_summary(plan: R1Plan) -> str:
    """The R1-0 ramp for the approval: it is not a small motion (computed)."""
    from reachy_motion.library.pose import REST, rotation_distance, translation_distance

    if plan.ramp is None:
        return ("AWAKE start: no start-up / return ramp - the robot stands at INIT (woken the usual way) and the trial "
                "runs INIT -> trial -> INIT")
    t = ramp_travel(plan.ramp)
    m = plan.ramp.report.metrics
    h0, h1 = plan.start.head.matrix(), REST.head.matrix()
    return (f"R1-0 start-up ramp (approved separately; NOT a small motion): resting pose -> REST "
            f"{plan.ramp.timeline.duration:.1f} s and back {plan.back.timeline.duration:.1f} s, head "
            f"{math.degrees(rotation_distance(h0, h1)):.1f} deg / {1000.0 * translation_distance(h0, h1):.1f} mm, "
            f"Stewart joint travel up to {t['max_deg']:.1f} deg ({t['joint']}), peak Stewart speed ratio "
            f"{m.get('peak_stewart_speed_ratio', math.nan):.4f}, min joint margin "
            f"{m.get('min_stewart_margin_deg', math.nan):.2f} deg, antennas held (computed)")


def _request(parts: tuple[Any, ...], host: str, stage: str = "R1") -> Any:
    from reachy_motion.safety.approval import ApprovalRequest

    return ApprovalRequest(host=host, stage=stage, motions=tuple(f"{p.timeline.name}:{p.timeline.variant}"
                                                                 for p in parts),
                           timeline_digests=tuple(p.timeline.digest() for p in parts),
                           reports=tuple(p.report for p in parts))


def ramp_request(plan: R1Plan, host: str) -> Any:
    """R1-0 approval (stage R1): the start-up ramp and the return ramp (``ramp_summary``). Same R0 and antenna
    mode -> same ramps, so one R1-0 approval covers the R1, R2 and R3 runs of that state until it expires."""
    return _request((plan.ramp, plan.back), host)


def trial_request(plan: R1Plan, host: str) -> Any:
    """The trial's own approval (stage = the trial's: R1 / R2 / R3): this one trial."""
    return _request((plan.run,), host, plan.trial.stage)


def trials_request(plans: list[R1Plan], host: str) -> Any:
    """One approval for several trials of one stage (R2: the E/N pairs). The gate accepts it for every single-trial
    request it contains (safety/approval.py: request motions and digests must be IN the approval)."""
    stages = {p.trial.stage for p in plans}
    if len(stages) != 1:
        raise ValueError(f"one approval covers one stage, got {sorted(stages)}")
    return _request(tuple(p.run for p in plans), host, stages.pop())


def c1_plan(report: dict[str, Any], hold_s: float | None = None) -> Any:
    """C1's approval object: holding the approved R0 state for the hold time (C1 streams nothing; this timeline
    stands for 'the targets stay at this state'), checked with the start-up envelope of that state."""
    from reachy_motion.backends.robot_connect import C1_HOLD_S
    from reachy_motion.planner import plan_transition
    from reachy_motion.safety.checker import prepare_timeline
    from reachy_motion.safety.startup import startup_policy

    start = r0_state(report)
    tl = plan_transition(start, start, float(hold_s or C1_HOLD_S), "linear", name="C1")
    prep = prepare_timeline(tl, startup_policy(start))
    return _labelled(prep, *C1_KEY.split(":"))


def c1_request(report: dict[str, Any], host: str) -> Any:
    from reachy_motion.safety.approval import ApprovalRequest

    prep = c1_plan(report)
    return ApprovalRequest(host=host, stage="R1", motions=(C1_KEY,), timeline_digests=(prep.timeline.digest(),),
                           reports=(prep.report,))


def _gap(cur: Any, planned: Any) -> dict[str, float]:
    """Distance between two states (``RobotState`` or ``PresentState``)."""
    from reachy_motion.library.pose import rotation_distance, translation_distance

    def mat(s: Any) -> np.ndarray:
        h = s.head
        return np.asarray(h.matrix() if hasattr(h, "matrix") else h, dtype=float)

    a, b = mat(cur), mat(planned)
    return {"rotation_deg": math.degrees(rotation_distance(a, b)),
            "translation_mm": 1000.0 * translation_distance(a, b),
            "antenna_deg": max(abs(math.degrees(x - y)) for x, y in zip(cur.antennas, planned.antennas)),
            "body_deg": abs(math.degrees(cur.body_yaw - planned.body_yaw))}


# ---------------------------------------------------------------------------------------------- ramp run records
@dataclass(frozen=True)
class _PartSpec:
    """Stands in for a MotionSpec in a ramp's run record (``save_run`` writes ``to_dict()`` and ``digest()``)."""

    data: dict[str, Any]
    timeline_digest: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)

    def digest(self) -> str:
        return self.timeline_digest


@dataclass(frozen=True)
class _PartTrial:
    stage: str
    label: str
    title: str
    condition: dict[str, Any]
    spec: _PartSpec
    pre_s: float = 0.0
    post_s: float = 0.0
    main_channel: str = "pitch"


@dataclass(frozen=True)
class _PartCheck:
    trial: _PartTrial
    prep: Any

    @property
    def runnable(self) -> bool:
        return bool(self.prep.runnable)

    def row(self) -> dict[str, Any]:
        m, r = dict(self.prep.report.metrics), self.prep.report
        return {"label": self.trial.label, "title": self.trial.title, **self.trial.condition,
                "record_s": float(self.prep.timeline.duration), "stewart_speed_ratio": m.get("peak_stewart_speed_ratio"),
                "antenna_speed_ratio": m.get("peak_antenna_speed_ratio"), "body_speed_ratio": m.get("peak_body_speed_ratio"),
                "min_margin_deg": m.get("min_stewart_margin_deg"), "min_margin_joint": m.get("min_stewart_margin_joint"),
                "policy": "통과" if self.runnable else "거부", "violations": ", ".join(v.code for v in r.violations),
                "warnings": ", ".join(w.code for w in r.warnings),
                "real_scope": "R1-0 start-up / return ramp, approved separately from the R1 trial", "data_kind": "computed"}


def _part_record(plan: R1Plan, prep: Any, label: str) -> tuple[_PartTrial, _PartCheck]:
    which = "start-up ramp (resting pose -> REST)" if label == RAMP_LABEL else "return ramp (REST -> resting pose)"
    t = ramp_travel(prep)
    cond = {"part": label, "approval": "R1-0", "antennas_held_deg": [round(math.degrees(a), 3)
                                                                       for a in plan.start.antennas],
            "stewart_travel_max_deg": t["max_deg"], "stewart_travel_joint": t["joint"], "for_trial": plan.trial.label}
    trial = _PartTrial("R1", label, f"R1-0 {which}, before / after {plan.trial.title}", cond,
                       _PartSpec({"kind": "planned timeline (start-up procedure, not a motion file)",
                                  "name": prep.timeline.name, "variant": prep.timeline.variant,
                                  "duration_s": float(prep.timeline.duration), "stewart_travel_deg": t},
                                 prep.timeline.digest()))
    return trial, _PartCheck(trial, prep)


def _analyse(run_dir: Path) -> dict[str, Any]:
    from reachy_motion.gui.results import load_result, write_metrics_json, write_run_readme

    loaded = load_result(run_dir)
    write_metrics_json(run_dir / "metrics.json", loaded)
    write_run_readme(loaded)
    m = loaded.metrics[loaded.main_channel]
    k = loaded.unit(loaded.main_channel)[0]
    return {"channel": loaded.main_channel, "rmse": m.dynamic_rmse * k, "peak_error": m.peak_error * k,
            "lag_ms": m.lag_s * 1000.0, "amplitude_ratio": m.amplitude_ratio}


def run_trial(backend: Any, plan: R1Plan, session_dir: Path, *, notes: str = "",
              connection: dict[str, Any] | None = None, clock: Any = None,
              torque_off: Callable[[], str] | None = None,
              confirm_torque_off: Callable[[str], bool] | None = None,
              before_part: Callable[[str], None] | None = None, cancel: Any = None,
              live: Callable[[str, int], Any] | None = None, executor: Any = None) -> dict[str, Any]:
    """The approved timelines in order, each recorded as its own run: still check, start-up ramp (from the approved
    start, which the present state must match), hold, the trial, the return ramp - then the end of the run (module
    docstring). ``torque_off`` sends the global torque-off and returns the read-back state (``SdkBackend.disable_motors``:
    "off" | "off_unconfirmed" | "unknown"); None = a backend without torque (Mock). ``before_part(label)`` may raise
    to refuse a part (the robot re-checks its approval there). ``backend`` is connected (Mock; on the robot after the
    connection sequence, so the head rests where it was limp, the torque on).

    Motion Lab (GUI, 2026-11-12): ``cancel`` (threading.Event) is the Stop button - the executor's controlled stop S1,
    after which the run is not clean (no automatic torque-off, no automatic return); ``live(label, n_samples)``
    returns the executor's on_tick callback of that part (the live view); ``executor`` reuses the session's one."""
    from reachy_motion.backends.robot_connect import OPERATOR_PROCEDURE_KO, TORQUE_OFF_PROMPT_KO
    from reachy_motion.executor.executor import Executor, StopPolicy
    from reachy_motion.gui.lab import next_run_dir, save_run
    from reachy_motion.gui.results import update_session_readme
    from reachy_motion.startup_procedure import HOLD_S, STILL_TOL_DEG, watch_still

    ex = executor if executor is not None else Executor(backend, clock=clock)
    has_torque = torque_off is not None
    out: dict[str, Any] = {"ok": False, "stage": "start", "problems": [], "digests": [], "runs": {},
                           "data_kind": getattr(backend, "data_kind", None),
                           "torque_state": "on" if has_torque else "no torque control (mock: nothing to switch off)",
                           "torque_off_sent": False, "warnings": [],
                           "residual": list((connection or {}).get("residual", []))}
    conn = dict(connection or {}, approved_timelines={p.timeline.variant: p.timeline.digest() for p in plan.parts},
                approvals={"R1-0": [f"R1:{RAMP_VARIANT}", f"R1:{RETURN_VARIANT}"], "R1": [f"R1:{plan.trial.label}"]})
    rest = backend.present()  # where the head rests (on the robot: limp until the connection's enable)
    out["start_gap"] = gap = _gap(rest, plan.start)
    sent = False
    completed = 0

    def part(prep: Any, label: str, trial: Any, check: Any) -> Any:
        nonlocal sent
        out["stage"] = label
        if before_part is not None:
            before_part(label)
        started_local = datetime.now().astimezone().isoformat(timespec="seconds")
        sent = True
        interrupted: BaseException | None = None
        try:
            result = ex.run(prep, cancel=cancel, on_tick=None if live is None else live(label, len(prep.timeline)),
                            record_present=True, stop=StopPolicy())
        except (Exception, KeyboardInterrupt) as exc:  # the executor recorded it and re-raised: save what ran first
            interrupted, result = exc, ex.last_result
            if result is None:
                raise
        out["digests"].append(result.timeline_digest)
        run_dir = next_run_dir(session_dir, label)
        save_run(run_dir, trial, check, result, dict(conn, part=label), notes=notes, started_local=started_local)
        out["runs"][label] = str(run_dir)
        out["stop_kind"] = result.stop_kind
        if not result.completed:
            out["problems"].append(f"{label} not completed: {result.stop_kind} {result.stop_reason or result.error or ''}"
                                   + ("" if result.end_state_known else " (end state unknown)"))
            if not result.end_state_known and has_torque:
                out["torque_state"] = "unknown"
        try:
            out.setdefault("metrics", {})[label] = _analyse(run_dir)
        except Exception as exc:  # noqa: BLE001 - the run is saved even when it cannot be analysed
            out.setdefault("analysis_errors", {})[label] = f"{type(exc).__name__}: {exc}"
        if interrupted is not None:
            raise interrupted
        return result

    awake = plan.ramp is None  # AWAKE start: the robot stands at INIT, the trial alone runs INIT -> INIT
    from reachy_motion.robot_r0 import AWAKE_INIT_TOL

    too_far = any(gap[k] > AWAKE_INIT_TOL[k] for k in AWAKE_INIT_TOL) if awake else (
        gap["rotation_deg"] > STATE_TOL["joint_deg"] or gap["translation_mm"] > 1.0
        or gap["antenna_deg"] > STATE_TOL["antenna_deg"] or gap["body_deg"] > STATE_TOL["joint_deg"])
    try:
        if too_far:
            out["problems"].append(f"the present state is not the approved start{' (INIT)' if awake else ''}: {gap}")
        else:
            out["stage"] = "still"
            still, drift, _ = watch_still(backend, ex.clock)
            if not still:
                out["problems"].append(f"the head moved {drift:.2f} deg (> {STILL_TOL_DEG:g}) before the "
                                       f"{'trial' if awake else 'ramp'}")
            elif awake:
                if part(plan.run, plan.trial.label, plan.trial, plan.check).completed:
                    completed = 3
                    out["run_dir"] = out["runs"][plan.trial.label]
                    out["stage"] = "at_init"
                    ex.clock.sleep_until(ex.clock.now_ns() + int(round(HOLD_S * 1e9)))  # settle before the end check
            elif part(plan.ramp, RAMP_LABEL, *_part_record(plan, plan.ramp, RAMP_LABEL)).completed:
                completed = 1
                ex.clock.sleep_until(ex.clock.now_ns() + int(round(HOLD_S * 1e9)))
                if part(plan.run, plan.trial.label, plan.trial, plan.check).completed:
                    completed = 2
                    out["run_dir"] = out["runs"][plan.trial.label]
                    if part(plan.back, RETURN_LABEL, *_part_record(plan, plan.back, RETURN_LABEL)).completed:
                        completed = 3
                        out["stage"] = "returned"
                        ex.clock.sleep_until(ex.clock.now_ns() + int(round(HOLD_S * 1e9)))  # settle before the end check
    except (Exception, KeyboardInterrupt) as exc:  # a stop request or an error: recorded, never "safe"
        stop = isinstance(exc, KeyboardInterrupt)
        out["problems"].append(f"{'stop requested' if stop else 'error'} during {out['stage']}: {exc!r}")
        if has_torque and sent and not stop:
            out["torque_state"] = "unknown"
    out.setdefault("run_dir", out["runs"].get(plan.trial.label))
    out["metrics_main"] = out.get("metrics", {}).get(plan.trial.label)
    out["main"] = out["metrics_main"]

    # ------------------------------------------------------------------ end of the run (module docstring)
    clean = completed == 3 or not sent  # nothing sent: the daemon still holds the pin where the head rested
    try:
        cur = backend.present()
        fresh = ex.clock.now_ns() - int(cur.t_ns) <= PRESENT_FRESH_S * 1e9
        out["end_gap_from_rest"] = _gap(cur, rest)
        out["end_gap"] = _gap(cur, plan.start)
    except Exception as exc:  # noqa: BLE001
        cur, fresh = None, False
        out["problems"].append(f"present state not readable at the end: {exc!r}")
    at_rest = cur is not None and fresh and all(out["end_gap_from_rest"][k] <= END_TOL[k] for k in END_TOL)
    out["ok"] = completed == 3 and not out["problems"]
    if not has_torque:
        pass
    elif awake and clean:  # back at INIT (or nothing sent): stay awake, torque on - the usual standby, no torque-off
        out["end_state"] = "standby at INIT, torque on"
        near = cur is not None and fresh and all(out["end_gap"][k] <= AWAKE_INIT_TOL[k] for k in AWAKE_INIT_TOL)
        if not near:
            out["ok"] = False
            out["problems"].append(f"not near INIT at the end (fresh={fresh}, gap {out.get('end_gap')})")
            out["operator_procedure"] = OPERATOR_PROCEDURE_KO
    elif clean and at_rest and out["torque_state"] == "on":
        out["torque_state"], out["torque_off_sent"] = torque_off(), True
        if out["torque_state"] == "off":
            out["off_drift_deg"] = _off_drift(backend, ex.clock)
            if out["off_drift_deg"] > OFF_DRIFT_WARN_DEG:
                out["warnings"].append(f"the head moved {out['off_drift_deg']:.2f} deg after the torque-off: it was "
                                       "not resting where it was switched off")
        else:
            out["ok"] = False
            out["problems"].append(f"the torque-off was not confirmed by the daemon ({out['torque_state']})")
            out["operator_procedure"] = OPERATOR_PROCEDURE_KO
    else:
        out["ok"] = False
        out["problems"].append("no automatic torque-off: " + ("the run did not end cleanly" if not clean else
                               f"the head is not back where it rested (fresh={fresh}, gap {out.get('end_gap_from_rest')})"))
        out["operator_procedure"] = OPERATOR_PROCEDURE_KO
        if confirm_torque_off is not None:
            try:
                yes = bool(confirm_torque_off(TORQUE_OFF_PROMPT_KO))
            except (Exception, KeyboardInterrupt):  # noqa: BLE001
                yes = False
            if yes:
                out["torque_state"], out["torque_off_sent"] = torque_off(), True
                out["torque_off_by"] = "operator (head supported)"
                if out["torque_state"] == "off":
                    out["operator_procedure"] = ""
    _write_json(Path(session_dir) / f"session_{plan.trial.label}.json", dict(out, generated_local=datetime.now().isoformat(
        timespec="seconds"), trial=plan.trial.label, robot_run_enabled=ROBOT_RUN_ENABLED,
        note="parts recorded in runs/ (data_kind per manifest); torque_state is what this PC knows at the end"))
    try:
        update_session_readme(session_dir)
    except Exception as exc:  # noqa: BLE001
        out.setdefault("analysis_errors", {})["session_readme"] = f"{type(exc).__name__}: {exc}"
    return out


def _off_drift(backend: Any, clock: Any, seconds: float = OFF_WATCH_S) -> float:
    """Largest head-joint change (deg) over ``seconds`` after the torque-off, against the first reading."""
    first = None
    worst = 0.0
    t_end = clock.now_ns() + int(round(seconds * 1e9))
    while True:
        p = backend.present()
        if p.joints is not None:
            q = np.asarray(p.joints, dtype=float)
            first = q if first is None else first
            worst = max(worst, math.degrees(float(np.max(np.abs(q - first)))))
        now = clock.now_ns()
        if now >= t_end:
            return worst
        clock.sleep_until(min(t_end, now + 20_000_000))


def _write_json(path: Path, body: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------- robot path
def operator_confirms_environment(assessment: dict[str, Any], input_fn: Callable[[str], str] | None = None,
                                  isatty_fn: Callable[[], bool] | None = None, out: Any = None) -> dict[str, Any]:
    """Right before a robot connection: the operator rows and the residual risks of the readiness, then the operator
    types ``ENV_TOKEN`` when they control the environment now. No terminal or anything else typed: not confirmed,
    no run. Returns the record (a confirmation, not a verification)."""
    import builtins
    import sys

    out = sys.stdout if out is None else out
    rows = assessment.get("rows") or []
    ops = [r for r in rows if r.get("category") == "operator"]
    res = [r for r in rows if r.get("category") == "residual"]
    rec: dict[str, Any] = {"confirmed": False, "at": datetime.now().isoformat(timespec="seconds"),
                           "operator_rows": [r.get("title") for r in ops], "residual": [r.get("key") for r in res],
                           "note": "the operator's confirmation, recorded; not a verification"}
    try:
        tty = bool((isatty_fn or sys.stdin.isatty)())
    except (AttributeError, OSError, ValueError):
        tty = False
    if not tty:
        out.write("operator environment NOT confirmed: stdin is not an interactive terminal (no run)\n")
        return rec
    bar = "=" * 78
    out.write("\n".join([bar, "운영 환경 확인 (연결 직전, 운영자)", *[f"  - {r.get('title')}: {r.get('detail')}" for r in ops],
                         "남는 위험 (확인·예방 불가, 이 실행 기록에 남음)",
                         *[f"  - {r.get('title')}: {r.get('residual')}" for r in res], bar]) + "\n")
    try:
        answer = (input_fn or builtins.input)(f"위 조건을 지금 모두 통제하고 있으면 '{ENV_TOKEN}'를 입력하세요. "
                                              "통제할 수 없으면 Enter (실행하지 않음): ")
    except (EOFError, KeyboardInterrupt):
        answer = ""
    rec["confirmed"] = isinstance(answer, str) and answer.strip().lower() == ENV_TOKEN
    out.write("operator environment confirmed (recorded; not a verification)\n" if rec["confirmed"]
              else "operator environment NOT confirmed: no run\n")
    return rec


def c1_stage(make_client: Callable[..., Any], r0_report: dict[str, Any], *, clock: Any, host: str, port: int,
             out_dir: Path, data_kind: str, what: dict[str, Any], operator_confirmed: bool = False,
             confirm_torque_off: Callable[[str], bool] | None = None) -> tuple[Any, Path]:
    """C1 (``robot_connect.run_c1``) from a fresh R0, then ``c1_report.json`` in ``out_dir`` with the data_kind of what
    produced it (``measured`` only for the SDK client on the robot), the residual risks and the torque state."""
    from reachy_motion.backends.robot_connect import run_c1

    rep = run_c1(make_client, r0_report, clock=clock, host=host, port=port, operator_confirmed=operator_confirmed,
                 confirm_torque_off=confirm_torque_off)
    body = {"stage": "C1", "generated_local": datetime.now().isoformat(timespec="seconds"), "data_kind": data_kind,
            "what": what, "ok": rep.ok, "ended_at": rep.stage, "torque_state": rep.torque_state,
            "residual_risks": rep.residual, "operator_confirmed": rep.operator_confirmed,
            "operator_procedure": rep.operator_procedure, "report": rep.to_dict(), "robot_run_enabled": ROBOT_RUN_ENABLED}
    return rep, _write_json(Path(out_dir) / "c1_report.json", body)


def _joints_deg(report: dict[str, Any] | None) -> dict[str, float]:
    j = (((report or {}).get("derived") or {}).get("joints") or {}).get("joints_deg") or {}
    return {k: float(v) for k, v in j.items() if isinstance(v, (int, float))}


def preflight(host: str, port: int, timeout_s: float, reference: dict[str, Any] | None = None,
              now: datetime | None = None, mode: str = "sleep") -> dict[str, Any]:
    """Fresh R0 (allowed GETs only) + readiness; with ``reference`` (the approved R0) the joints and antennas must
    not have moved more than ``STATE_TOL`` since (else: blocked)."""
    from reachy_motion.backends.daemon_client import DaemonClient
    from reachy_motion.r1_readiness import assess, is_loopback
    from reachy_motion.robot_r0 import run_r0

    report = run_r0(DaemonClient(host, port=port, timeout_s=timeout_s), timeout_s)
    if is_loopback(host):  # a local / fake daemon is never the robot: never "measured"
        report["data_kind"] = FAKE_KIND
        report.setdefault("derived", {})["data_kind"] = FAKE_KIND
        report["note"] = "loopback daemon: not the robot, not a measurement"
    result = assess(report, now, mode=mode)
    gaps: dict[str, float] = {}
    if reference is not None:
        ref, cur = r0_antennas(reference), r0_antennas(report)
        gaps["antenna_deg"] = math.nan if ref is None or cur is None else max(
            abs(math.degrees(a - b)) for a, b in zip(ref, cur))
        rj, cj = _joints_deg(reference), _joints_deg(report)
        gaps["joint_deg"] = max((abs(rj[k] - cj[k]) for k in rj if k in cj), default=math.nan) if rj and cj \
            else math.nan
        bad = [f"{k} {v:.2f}° (> {STATE_TOL[k]:g}°)" if math.isfinite(v) else f"{k} 읽기 없음"
               for k, v in gaps.items() if not v <= STATE_TOL[k]]  # NaN fails too
        if bad:
            result["rows"].append({"key": "state_since_approval", "category": "auto", "status": "blocked",
                                   "title": "승인한 R0 이후 상태", "detail": "승인한 R0와 다름: " + ", ".join(bad)
                                   + ". R0와 승인 양식을 다시 만드세요", "mitigation": "", "residual": ""})
            result["verdict"] = "blocked"
            result["verdict_ko"] = "차단: " + ", ".join(r["title"] for r in result["rows"] if r["status"] == "blocked")
    return {"report": report, "assessment": result, "gaps": gaps}


def write_preflight(out_dir: Path, pre: dict[str, Any], what: dict[str, Any], decision: str) -> Path:
    from reachy_motion.robot_r0 import write_report

    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(pre["report"], out_dir)
    p = out_dir / "preflight.json"
    p.write_text(json.dumps({"stage": "preflight (Step C preparation)", "generated_local": datetime.now().isoformat(
        timespec="seconds"), "data_kind": pre["report"].get("data_kind"), "what": what, "gaps": pre["gaps"],
        "assessment": pre["assessment"], "decision": decision, "robot_run_enabled": ROBOT_RUN_ENABLED},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return p


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = ["ROBOT_RUN_ENABLED", "PREFLIGHT_OUT", "FAKE_KIND", "C1_KEY", "STATE_TOL", "END_TOL", "OFF_WATCH_S",
           "ENV_TOKEN", "RAMP_LABEL", "RETURN_LABEL", "R2_DEFAULT_MOTIONS", "R3_BASE_S", "R3_BASE_A_DEG", "R3_LIMITS", "r3_base_name", "r3_metric_problems",
           "r3_next_step_problems", "r3_amplitude_step_problems", "R1Plan",
           "parse_r1", "load_r0", "r0_antennas", "r0_state", "build_plan", "build", "hold_antennas", "r2_plan",
           "r3_plan", "r3_step_problems", "prerequisite_problems", "check_prerequisites", "ramp_travel", "ramp_summary",
           "ramp_request", "trial_request", "trials_request", "c1_plan", "c1_request", "run_trial",
           "operator_confirms_environment", "c1_stage", "preflight", "write_preflight", "file_sha256"]
