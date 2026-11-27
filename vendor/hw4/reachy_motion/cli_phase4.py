"""Phase 4 preparation commands of ``reachy-motion`` (registered by ``cli.build_parser``). Neither moves anything.

    robot-status  R0: read-only REST GET checklist of a daemon, only the CLAUDE.md rule 3 GETs (``robot_r0``). No
                  ReachyMini client, no WebSocket, no motion. Exit 0 = no blocking finding, 2 = a blocking finding,
                  1 = the daemon did not answer. ``--host`` defaults to the project robot (``settings.ROBOT_HOST``,
                  confirmed by the user 2026-11-11); the requests are the same read-only GETs whatever the host.
    tracking      tracking metrics of recorded runs (``analysis.tracking``): one run folder (commands.csv +
                  present_<data_kind>.csv) or a sim-verify review folder (runs/*). Writes
                  results/phase4/tracking/<stamp>/; the data_kind is the input's (simulated for MuJoCo runs).
    robot-c1      C1 (``robot_r1.c1_stage`` / ``backends.robot_connect.run_c1``): torque on where the head rests,
                  hold, torque off (read back). A fresh R0 (GET only), the readiness, the state since the approved
                  R0, the approval gate - then STOP while ``robot_r1.ROBOT_RUN_ENABLED`` is False. Past it: the
                  operator confirms the controlled environment (else no run), C1 runs, c1_report.json.
    robot-r1      ONE R1 trial with its start-up and return ramps (``robot_r1``), each recorded as its own run:
                  ``--target mock`` runs it on a Mock that starts where R0 saw the robot (antennas held), data_kind
                  mock; ``--target robot`` = preflight + TWO approval gates (``--ramp-approval`` R1-0 ramps,
                  ``--approval`` trial), then STOP (``ROBOT_RUN_ENABLED``; past it the SdkBackend lock).
    sim-startup   the Phase 4 start-up procedure (``startup_procedure``) in a local MuJoCo daemon (127.0.0.1 only,
                  the sim interlocks of ``backends.sdk``), simulated data. ``--compare-wake-up`` then replays the
                  SDK ``wake_up`` goto sequence (reachy_mini.py:724-740, without the sound) from a fresh sleep start
                  and compares the measured joint speeds. Never the robot.

Units at this boundary: degrees, millimetres, milliseconds (CLAUDE.md code rules); the modules work in SI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from reachy_motion.settings import ROBOT_DAEMON_PORT, ROBOT_HOST

R0_OUT = Path("results") / "phase4" / "r0"
TRACKING_OUT = Path("results") / "phase4" / "tracking"
SETTLE_ABS_TOL = {"rad": math.radians(0.2), "m": 0.2e-3}  # 0.2 deg / 0.2 mm settling band floor


def _pins_ok() -> bool:
    from reachy_motion.sdk_env import pin_mismatches

    if pin_mismatches():
        print("SDK pins do not match; refusing to run (see `reachy-motion env`).", file=sys.stderr)
        return False
    return True


def cmd_robot_status(args: argparse.Namespace) -> int:
    """R0: the five read-only GETs, evaluated; report files under --out/<stamp>/."""
    from reachy_motion.backends.daemon_client import DaemonClient
    from reachy_motion.robot_r0 import run_r0, write_report

    if not _pins_ok():
        return 1
    client = DaemonClient(args.host, port=args.port, timeout_s=args.timeout)
    report = run_r0(client, args.timeout)
    out = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = write_report(report, out)
    d = report["derived"]
    print(f"R0 {report['daemon_url']} (GET only): data_kind {report['data_kind']}, version {d.get('daemon_version')}, "
          f"state {d.get('daemon_state')}, motors {d.get('motor_mode')}")
    for x in report["findings"]:
        print(f"  [{x['level']}] {x['code']}: {x['message']}")
    r1 = report["r1_entry"]
    print(f"R1 entry: {r1['verdict']} (automatic R0 checks {'passed' if r1['auto_met'] else 'NOT passed'}; never "
          "'verified')" + "".join(f"\n  [{c['category']}] {c['name']}: {c['status']}" for c in r1["conditions"]
                                  if c["status"] != "met"))
    print(paths[1])
    if any(x["code"] == "no_status" for x in report["findings"]):
        return 1
    return 2 if report["blocking"] else 0


class _GateRefused(Exception):
    def __init__(self, label: str, exc: Exception) -> None:
        super().__init__(f"{label}: {exc}")
        self.label, self.exc = label, exc


def _approval_gates(args: argparse.Namespace, gates: list[tuple[str, Any, str | None, str | None]]) -> dict[str, Any]:
    """Every approval of a robot run, in order (C1: one; R1: the R1-0 ramps, then the trial): armed + the approval
    file the user wrote + its id typed at a terminal, as the ``SdkBackend`` robot constructor checks it (safety/
    approval.py). Returns {label: SdkBackend, not connected}; raises ``_GateRefused`` at the first refusal."""
    from reachy_motion.backends.sdk import RobotApprovalError, SdkBackend
    from reachy_motion.safety.approval import ApprovalError, confirm_interactively, load_approval

    held: dict[str, Any] = {}
    for label, request, path, header in gates:
        if header:
            print(header)
        try:
            approval = load_approval(path) if path else None
            held[label] = SdkBackend("robot", armed=bool(args.armed), approval=approval, request=request,
                                     confirm=confirm_interactively)
        except (RobotApprovalError, ApprovalError, OSError, ValueError) as exc:
            raise _GateRefused(label, exc) from exc
    return held


def _robot_gate(args: argparse.Namespace, gates: list[tuple[str, Any, str | None, str | None]],
                ref: dict[str, Any] | None, what: dict[str, Any], out_dir: Path,
                confirm_env: Any = None) -> tuple[int | None, dict[str, Any] | None, dict[str, Any]]:
    """Shared robot path of robot-c1 / robot-r1: fresh R0 (GET only) + readiness + state since the approved R0, the
    approval gate of every part, STOP while ``ROBOT_RUN_ENABLED`` is False, then the operator's confirmation of the
    controlled environment (not confirmed: no run). Returns (exit code, None, {}) to stop, or (None, preflight, held
    backends) to go on."""
    from reachy_motion import robot_r1 as r1
    from reachy_motion.r1_readiness import CATEGORY_KO, STATUS_KO

    pre = r1.preflight(args.host, args.port, args.timeout, reference=ref, mode=getattr(args, "start", "sleep"))
    a = pre["assessment"]
    print(f"R0 {pre['report'].get('daemon_url')} (GET only), data_kind {pre['report'].get('data_kind')}, measured "
          f"{a['measured_at']}")
    for row in a["rows"]:
        print(f"  [{CATEGORY_KO[row['category']]}] {STATUS_KO[row['status']]}: {row['title']} - {row['detail']}"
              + (f" | 완화: {row['mitigation']} | 남는 것: {row['residual']}" if row["category"] in
                 ("connection", "residual") else ""))
    print(a["verdict_ko"])
    if a["verdict"] == "blocked":
        print(r1.write_preflight(out_dir, pre, what, "blocked before the approval gate"))
        return 2, None, {}
    if ref is None:
        print("refused: --r0 is required for the robot (the approval binds timelines planned from that R0)",
              file=sys.stderr)
        print(r1.write_preflight(out_dir, pre, what, "refused: no --r0"))
        return 2, None, {}
    try:
        held = _approval_gates(args, gates)
    except _GateRefused as exc:
        print(f"approval gate {exc}", file=sys.stderr)
        print(r1.write_preflight(out_dir, pre, what, f"refused at the {exc.label} approval gate: "
                                                     f"{type(exc.exc).__name__}"))
        return 4, None, {}
    if not r1.ROBOT_RUN_ENABLED:
        print("approvals confirmed, but the robot run is not enabled (robot_r1.ROBOT_RUN_ENABLED = False): no client "
              "was built, no connection made, nothing sent, nothing recorded as measured.")
        print(r1.write_preflight(out_dir, pre, what, "approvals confirmed; robot run disabled"))
        return 3, None, {}
    env = (confirm_env or r1.operator_confirms_environment)(a)
    what["operator_environment"] = env
    if not env.get("confirmed"):
        print(r1.write_preflight(out_dir, pre, what, "the operator did not confirm the controlled environment: no run"))
        return 5, None, {}
    return None, pre, held


def cmd_robot_c1(args: argparse.Namespace, *, make_client: Any = None, clock: Any = None, data_kind: str = "measured",
                 confirm_env: Any = None, confirm_off: Any = None) -> int:
    """C1: torque on where the head rests, hold, torque off. Preflight + approval gate, then STOP while
    ``ROBOT_RUN_ENABLED`` is False; past it the operator's confirmation and ``robot_r1.c1_stage`` (``run_c1``). The
    keyword arguments are for the offline harness: the client factory (default: the SDK's ``ReachyMini``), the clock,
    the data_kind of the record (``measured`` only for the SDK client on the robot) and the two prompts."""
    from reachy_motion import robot_r1 as r1

    if not _pins_ok():
        return 1
    try:
        ref = r1.load_r0(args.r0)
        request = r1.c1_request(ref, args.host)
    except (OSError, ValueError) as exc:
        print(f"C1: {exc}", file=sys.stderr)
        return 2
    prep = r1.c1_plan(ref)
    print(f"C1 ({r1.C1_KEY}): hold the state of R0 {args.r0} (sha256 {r1.file_sha256(args.r0)[:12]}...), "
          f"timeline {prep.timeline.digest()[:12]}..., policy {'pass' if prep.runnable else 'REFUSED'}")
    if not prep.runnable:
        print("refused: " + ", ".join(v.code for v in prep.report.violations), file=sys.stderr)
        return 2
    what = {"stage": "C1", "key": r1.C1_KEY, "r0": str(args.r0), "r0_sha256": r1.file_sha256(args.r0),
            "timeline_digest": prep.timeline.digest()}
    out_dir = Path(args.preflight_out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    code, pre, _ = _robot_gate(args, [("C1", request, args.approval, None)], ref, what, out_dir, confirm_env)
    if code is not None:
        return code
    from reachy_motion.backends.robot_connect import confirm_torque_off_interactively
    from reachy_motion.executor.clock import RealClock

    if make_client is None:  # the SDK client: network mode on the approved host (robot_connect.CONSTRUCT_KWARGS)
        from reachy_mini import ReachyMini

        make_client = ReachyMini
    rep, path = r1.c1_stage(make_client, pre["report"], clock=clock or RealClock(), host=args.host, port=args.port,
                            out_dir=out_dir, data_kind=data_kind, what=what, operator_confirmed=True,
                            confirm_torque_off=confirm_off or confirm_torque_off_interactively)
    print(f"C1 {'done' if rep.ok else 'NOT completed (' + rep.stage + ')'} ({data_kind}): torque {rep.torque_state}; "
          f"hold drift head {rep.hold_head_drift_deg} deg, antennas {rep.hold_antenna_drift_deg} deg; after the "
          f"torque-off {rep.off_head_drift_deg} deg; residual risks {', '.join(rep.residual) or '-'}")
    for line in rep.problems + rep.warnings:
        print("  " + line)
    if rep.operator_procedure:
        print("operator procedure: " + rep.operator_procedure)
    print(path)
    return 0 if rep.ok else 6


def _plan_lines(plan: Any) -> str:
    """Print a plan (computed) and return its R1-0 ramp summary."""
    from reachy_motion import robot_r1 as r1

    trial, m = plan.trial, plan.check.row()
    held = trial.condition.get("antennas_held_deg") or trial.condition.get("antennas_start_deg")
    print(f"{trial.title}: policy {m['policy']} (computed: Stewart speed ratio {m['stewart_speed_ratio']}, "
          f"min joint margin {m['min_margin_deg']} deg), record {plan.check.record_s:.1f} s"
          + (f", antennas held at {held} deg" if held else ""))
    for p in plan.parts:
        print(f"  {p.timeline.name}:{p.timeline.variant}  {p.timeline.duration:.1f} s  {p.timeline.digest()[:12]}...  "
              f"{'pass' if p.runnable else 'REFUSED ' + ', '.join(v.code for v in p.report.violations)}")
    summary = r1.ramp_summary(plan)
    print(summary)
    return summary


def _run_mock(args: argparse.Namespace, plans: list[Any], ref: dict[str, Any] | None,
              runner: str) -> tuple[int, list[dict[str, Any]]]:
    """Mock runs of the plans in order, one session folder; stops at the first one that does not end cleanly."""
    from reachy_motion import robot_r1 as r1
    from reachy_motion.backends.mock import MockBackend
    from reachy_motion.gui.lab import new_session_dir

    session = new_session_dir("mock", plans[0].trial.stage, args.out)
    outs: list[dict[str, Any]] = []
    for plan in plans:
        backend = MockBackend(structure="robot", start=plan.start)
        backend.connect()
        try:
            out = r1.run_trial(backend, plan, session, notes=args.notes, connection={
                "mode": "mock", "data_kind": "mock", "backend": f"mock ({runner})",
                "start": f"R0 {args.r0}" if ref else "REST", "runner": runner})
        finally:
            backend.close()
        outs.append(out)
        if not out["ok"]:
            print(f"{plan.trial.label} not completed ({out['stage']}): {out.get('problems') or out.get('stop_kind')}",
                  file=sys.stderr)
            return 3, outs
        mn = out.get("main") or {}
        print(f"done {plan.trial.label} (data_kind {out['data_kind']}, ended {out['stage']}; torque: "
              f"{out['torque_state']}): runs " + ", ".join(out["runs"].values()))
        if mn:
            print(f"  {mn['channel']}: RMSE {mn['rmse']:.3f}, Peak {mn['peak_error']:.3f}, lag {mn['lag_ms']:.0f} ms, "
                  f"amplitude ratio {mn['amplitude_ratio']:.3f}  (Mock model, not a measurement)")
    return 0, outs


def _run_robot(args: argparse.Namespace, plan: Any, ref: dict[str, Any] | None, summary: str,
               confirm_env: Any, runner: str) -> int:
    """The robot branch shared by robot-r1 / -r2 / -r3 (CLI, debugging / emergency; the GUI is the main path):
    preflight, the R1-0 ramp approval and the trial's own approval, STOP while ``ROBOT_RUN_ENABLED`` is False, the
    operator's confirmation, ``SdkBackend.connect`` (the connection sequence), then ``run_trial`` with the torque-off
    end path."""
    from reachy_motion import robot_r1 as r1

    trial, m = plan.trial, plan.check.row()
    stage = trial.stage
    start_mode = getattr(args, "start", "sleep")
    ramp_req = r1.ramp_request(plan, args.host) if plan.ramp is not None else None  # AWAKE: no ramps
    trial_req = r1.trial_request(plan, args.host)
    what = {"stage": stage, "trial": trial.label, "start": start_mode, "r0": str(args.r0) if args.r0 else None,
            "timelines": {f"{p.timeline.name}:{p.timeline.variant}": p.timeline.digest() for p in plan.parts},
            "approvals": {**({"R1-0": list(ramp_req.motions)} if ramp_req else {}), stage: list(trial_req.motions)},
            "ramp_travel": r1.ramp_travel(plan.ramp) if ramp_req else None, "ramp_summary": summary,
            "condition": trial.condition}
    out_dir = Path(args.preflight_out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    n = "2/2" if ramp_req else "1/1"
    gates = ([("R1-0", ramp_req, args.ramp_approval, "APPROVAL 1/2 - " + summary)] if ramp_req else []) + [
        (stage, trial_req, args.approval, f"APPROVAL {n} - {stage} trial {trial.title}: Stewart speed ratio "
                                          f"{m['stewart_speed_ratio']}, min joint margin {m['min_margin_deg']} deg "
                                          "(computed)")]
    code, pre, held = _robot_gate(args, gates, ref, what, out_dir, confirm_env)
    if code is not None:
        return code
    backend = held["R1-0" if ramp_req else stage]
    backend.operator_confirmed = True
    backend.start_mode = start_mode
    try:
        backend.connect()  # the robot connection sequence (CLAUDE.md rule 3, 2026-11-12)
    except NotImplementedError as exc:  # a backend that still refuses the robot: nothing built or sent
        print(f"{stage} stops at the SDK backend lock ({exc}): no client was built, nothing sent.")
        print(r1.write_preflight(out_dir, pre, what, f"robot run enabled; SdkBackend locked (Phase 2): no {stage} run"))
        return 3
    # Connected (the SdkBackend lock was lifted on 2026-11-12, user decision).
    from reachy_motion.backends.robot_connect import confirm_torque_off_interactively
    from reachy_motion.gui.lab import new_session_dir

    try:
        out = r1.run_trial(backend, plan, new_session_dir("robot", stage, args.out), notes=args.notes, connection=dict(
            backend.connection_info, mode="robot", runner=runner, operator_environment=what["operator_environment"]),
            torque_off=backend.disable_motors, confirm_torque_off=confirm_torque_off_interactively,
            before_part=lambda label: held[stage if label == trial.label else "R1-0"].recheck_approval())
    finally:
        backend.close()
    print(f"{stage} {'done' if out['ok'] else 'NOT completed (' + out['stage'] + ')'}: torque {out['torque_state']}; "
          "runs " + ", ".join(out["runs"].values()))
    for line in out["problems"] + out["warnings"]:
        print("  " + line)
    if out.get("operator_procedure"):
        print("operator procedure: " + out["operator_procedure"])
    return 0 if out["ok"] else 6


def cmd_robot_r1(args: argparse.Namespace, *, confirm_env: Any = None) -> int:
    """One R1 trial with its ramps: Mock run, or the robot preflight + the two approval gates (R1-0 ramps, R1 trial),
    then STOP (``ROBOT_RUN_ENABLED``; past it, the SDK backend lock)."""
    from reachy_motion import robot_r1 as r1

    if not _pins_ok():
        return 1
    try:
        ref = r1.load_r0(args.r0) if args.r0 else None
        plan = r1.build(args.r1, args.rep, ref, awake=args.start == "awake")
    except (OSError, ValueError) as exc:
        print(f"R1: {exc}", file=sys.stderr)
        return 2
    summary = _plan_lines(plan)
    if not plan.runnable:
        print("refused: a timeline is outside the (start-up) policy", file=sys.stderr)
        return 2
    if args.target == "mock":
        return _run_mock(args, [plan], ref, "robot-r1")[0]
    return _run_robot(args, plan, ref, summary, confirm_env, "robot-r1")


_ANTENNA_HINT = ("the antennas are never moved to their rest angle automatically; --antennas hold runs the head-only "
                 "diagnosis with the antennas held where R0 read them (antenna keys removed, labelled _antheld)")


def cmd_robot_r2(args: argparse.Namespace, *, confirm_env: Any = None) -> int:
    """R2: confirmed motions (E/N pairs). Mock: the selected set in order. Robot: one trial per run, after a clean
    measured C1 and R1; the trial approval may cover the whole set (approval-template --stage R2 --r2 ...)."""
    from reachy_motion import robot_r1 as r1

    if not _pins_ok():
        return 1
    motions = [x.strip() for x in str(args.motion).split(",") if x.strip()]
    variants = ("expressive", "neutral") if args.variant == "both" else (args.variant,)
    try:
        ref = r1.load_r0(args.r0) if args.r0 else None
        plans = [r1.r2_plan(mo, v, args.rep, ref, hold=args.antennas == "hold", awake=args.start == "awake")
                 for mo in motions for v in variants]
    except (OSError, ValueError) as exc:
        print(f"R2: {exc}", file=sys.stderr)
        if "antennas" in str(exc):
            print("  " + _ANTENNA_HINT, file=sys.stderr)
        return 2
    summaries = [_plan_lines(p) for p in plans]
    if not all(p.runnable for p in plans):
        print("refused: a timeline is outside the (start-up) policy", file=sys.stderr)
        return 2
    if args.target == "mock":
        return _run_mock(args, plans, ref, "robot-r2")[0]
    if len(plans) != 1:
        print("robot: one trial per run - give one --motion and --variant expressive|neutral (the approval may cover "
              "the whole set)", file=sys.stderr)
        return 2
    problems = r1.check_prerequisites(args.c1_report, args.r1_session, bool(plans[0].trial.spec.antennas),
                                      awake=args.start == "awake")
    if problems:
        print("refused before the preflight (C1 / R1 first):\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    return _run_robot(args, plans[0], ref, summaries[0], confirm_env, "robot-r2")


def cmd_robot_r3(args: argparse.Namespace, *, confirm_env: Any = None) -> int:
    """R3: one step of the duration ladder (pitch +A in D and back). The base step is D = 1 s; every shorter step
    needs --after <the previous step's run folder>, which must have ended within R3_LIMITS. Robot: after a clean
    measured C1 and R1, D >= 0.2 s only (gui.experiments.REAL_MIN_DURATION_S)."""
    from reachy_motion import robot_r1 as r1
    from reachy_motion.gui.experiments import R3_DEFAULT_DURATIONS

    if not _pins_ok():
        return 1
    try:
        ref = r1.load_r0(args.r0) if args.r0 else None
        plan = r1.r3_plan(args.amplitude, args.duration, args.rep, ref, hold=args.antennas == "hold",
                          awake=args.start == "awake")
    except (OSError, ValueError) as exc:
        print(f"R3: {exc}", file=sys.stderr)
        if "antennas" in str(exc):
            print("  " + _ANTENNA_HINT, file=sys.stderr)
        return 2
    summary = _plan_lines(plan)
    step = r1.r3_step_problems(plan, args.after, robot=args.target == "robot")
    if step or not plan.runnable:
        print("refused:\n  " + "\n  ".join(step or ["a timeline is outside the (start-up) policy"]), file=sys.stderr)
        return 2
    if args.target == "robot":
        problems = r1.check_prerequisites(args.c1_report, args.r1_session, False, awake=args.start == "awake")
        if problems:
            print("refused before the preflight (C1 / R1 first):\n  " + "\n  ".join(problems), file=sys.stderr)
            return 2
        return _run_robot(args, plan, ref, summary, confirm_env, "robot-r3")
    code, outs = _run_mock(args, [plan], ref, "robot-r3")
    shorter = [d for d in R3_DEFAULT_DURATIONS if d < float(args.duration) - 1e-9]
    if outs and outs[-1].get("run_dir") and shorter:
        nxt = r1.r3_plan(args.amplitude, shorter[0], args.rep, ref, hold=args.antennas == "hold",
                         awake=args.start == "awake")
        why = r1.r3_step_problems(nxt, outs[-1]["run_dir"], robot=False)
        print(f"next step D {shorter[0]:g} s: " + ("allowed with --after " + outs[-1]["run_dir"] if not why else
                                                  "NOT allowed - " + "; ".join(why)))
    return code


def _to_report_units(unit: str) -> tuple[float, str]:
    return (180.0 / math.pi, "deg") if unit == "rad" else (1000.0, "mm")


def cmd_tracking(args: argparse.Namespace) -> int:
    """Tracking metrics of one run folder or of every run of a review folder."""
    from reachy_motion.analysis.tracking import load_run, tracking_metrics
    from reachy_motion.sdk_env import environment_report

    if not _pins_ok():
        return 1
    src = Path(args.path)
    runs = sorted(p for p in (src / "runs").iterdir() if p.is_dir()) if (src / "runs").is_dir() else [src]
    out = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(rate_hz=args.rate, max_lag_s=args.max_lag, min_hold_s=args.min_hold, settle_skip_s=args.settle_skip)
    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    kinds: set[str] = set()
    note = ""
    print(f"{'run':30s} {'headline':10s} {'p2p':>6s} {'lag ms':>7s} {'r2':>5s} {'RMSE':>6s} {'RMSE-lc':>7s} "
          f"{'peak':>6s} {'amp':>5s} {'over':>5s} {'settle s':>9s} {'bias':>6s}   (deg)")
    for run in runs:
        try:
            data = load_run(run)
        except ValueError as exc:
            print(f"{run.name:30s} skipped: {exc}")
            continue
        kinds.add(data.data_kind)
        note = data.alignment_note
        tols = [SETTLE_ABS_TOL[u] for u in data.units]
        metrics = tracking_metrics(data.t_cmd, data.x_cmd, data.t_meas, data.x_meas, data.channels,
                                   settle_abs_tols=tols, **kw)
        joints = [c for c in data.channels if c.startswith("stewart_")]
        pool = joints or list(data.channels)
        head = max(pool, key=lambda c: metrics[c].command_p2p)
        results[run.name] = {"data_kind": data.data_kind, "headline_channel": head,
                             "channels": {c: dict(m.to_dict(), unit=u) for (c, m), u in zip(metrics.items(), data.units)}}
        for c, u in zip(data.channels, data.units):
            m, (k, unit) = metrics[c], _to_report_units(u)
            rows.append({"run": run.name, "channel": c, "unit": unit, "data_kind": data.data_kind,
                         "command_p2p": m.command_p2p * k, "lag_ms": m.lag_s * 1000, "lag_at_bound": m.lag_at_bound,
                         "lag_fit_r2": m.lag_fit_r2, "static_bias": m.static_bias * k,
                         "dynamic_rmse": m.dynamic_rmse * k,
                         "dynamic_rmse_lag_compensated": m.dynamic_rmse_lag_compensated * k,
                         "peak_error": m.peak_error * k, "amplitude_ratio": m.amplitude_ratio,
                         "overshoot": m.overshoot * k, "settling_time_s": m.settling_time_s,
                         "unsettled_steps": m.unsettled_steps, "steps": len(m.steps),
                         "duplicates_removed": m.duplicates_removed})
        m = metrics[head]
        k = 180.0 / math.pi if data.units[data.channels.index(head)] == "rad" else 1000.0
        print(f"{run.name:30s} {head:10s} {m.command_p2p * k:6.2f} {m.lag_s * 1000:7.0f} {m.lag_fit_r2:5.2f} "
              f"{m.dynamic_rmse * k:6.2f} {m.dynamic_rmse_lag_compensated * k:7.2f} {m.peak_error * k:6.2f} "
              f"{m.amplitude_ratio:5.2f} {m.overshoot * k:5.2f} {m.settling_time_s:6.2f}({m.unsettled_steps}) "
              f"{m.static_bias * k:6.2f}")
    if not results:
        print("no run folder with commands.csv + one present_*.csv", file=sys.stderr)
        return 1
    with open(out / "tracking.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    data_kind = kinds.pop() if len(kinds) == 1 else sorted(kinds)
    manifest = {"command": "python -m reachy_motion tracking " + str(src), "source": str(src),
                "generated_local": datetime.now().isoformat(timespec="seconds"), "data_kind": data_kind,
                "parameters": dict(kw, settle_abs_tol={"rad_deg": 0.2, "m_mm": 0.2}, settle_rel_tol=0.05),
                "time_alignment": note, "units": "tracking.csv: deg / mm / ms; tracking.json: SI (rad, m, s)",
                "environment": environment_report()}
    (out / "tracking.json").write_text(json.dumps({"manifest": manifest, "runs": results}, indent=2, default=str,
                                                  allow_nan=True), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"data_kind {data_kind}; time alignment: {note}")
    print(out / "tracking.csv")
    return 0


STARTUP_OUT = Path("results") / "phase4" / "startup_sim"


def _state_row(t_s: float, head: Any, antennas: Any, body_yaw: float, joints: Any = None) -> dict[str, Any]:
    from reachy_motion.library.pose import pose_from_matrix

    row: dict[str, Any] = {"t_s": t_s, **pose_from_matrix(head).to_deg_mm(),
                           "antenna_right_deg": math.degrees(antennas[0]), "antenna_left_deg": math.degrees(antennas[1]),
                           "body_yaw_deg": math.degrees(body_yaw)}
    if joints is not None:
        names = ("body_yaw",) + tuple(f"stewart_{i}" for i in range(1, 7))
        row.update({f"{n}_deg": math.degrees(float(q)) for n, q in zip(names, joints)})
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _distinct(samples: list[Any]) -> list[Any]:
    out, last = [], None
    for p in samples:
        if p.t_ns != last:
            out.append(p)
            last = p.t_ns
    return out


def _measured_motion(samples: list[Any]) -> dict[str, Any]:
    """Peak measured speeds between consecutive state pushes (simulated): Stewart joints, head rotation, antennas."""
    import numpy as np

    from reachy_motion.library.pose import rotation_distance

    s = [p for p in _distinct(samples) if p.joints is not None]
    if len(s) < 3:
        return {"samples": len(s)}
    t = np.array([p.t_ns for p in s], dtype=float) / 1e9
    q = np.degrees(np.array([p.joints for p in s], dtype=float))
    a = np.degrees(np.array([p.antennas for p in s], dtype=float))
    dt = np.diff(t)
    ok = dt > 1e-4
    v = np.abs(np.diff(q[:, 1:], axis=0))[ok] / dt[ok, None]
    rot = np.array([math.degrees(rotation_distance(s[i].head, s[i + 1].head)) for i in range(len(s) - 1)])[ok] / dt[ok]
    va = np.abs(np.diff(a, axis=0))[ok] / dt[ok, None]
    return {"samples": len(s), "span_s": float(t[-1] - t[0]), "median_push_interval_ms": float(np.median(dt) * 1e3),
            "peak_stewart_speed_dps": float(v.max()), "max_stewart_step_deg": float(np.abs(np.diff(q[:, 1:], axis=0)).max()),
            "peak_head_rotation_dps": float(rot.max()), "peak_antenna_speed_dps": float(va.max()),
            "peak_body_speed_dps": float((np.abs(np.diff(q[:, 0]))[ok] / dt[ok]).max())}


def _wake_up_replay(backend: Any) -> tuple[list[Any], list[tuple[str, float]], int]:
    """The SDK client's wake_up goto sequence (``sdk_wake_up.gotos``: reachy_mini.py:724-738, without the sound) on the
    sim backend, with the present state sampled from a second thread. Antennas are passed as their (unchanged) INIT
    values where wake_up passes none; body yaw 0.0 as wake_up's default (reachy_mini.py:680)."""
    import threading
    import time

    from reachy_motion.sdk_wake_up import PAUSE_S, gotos

    samples: list[Any] = []
    done = threading.Event()

    def sample() -> None:
        while not done.is_set():
            try:
                samples.append(backend.present())
            except Exception:  # noqa: BLE001 - a missed reading only thins the record
                pass
            time.sleep(0.004)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    marks: list[tuple[str, float]] = []
    t0 = time.perf_counter_ns()
    try:
        time.sleep(1.0)
        for i, (label, head, ant, d) in enumerate(gotos()):
            marks.append((label, (time.perf_counter_ns() - t0) / 1e9))
            backend.goto_target(head, ant, 0.0, d, "minjerk")
            if i == 0:
                time.sleep(PAUSE_S)
        time.sleep(1.0)
    finally:
        done.set()
        th.join(timeout=2.0)
    return samples, marks, t0


def _to_sdk_sleep_pose(backend: Any) -> dict[str, Any]:
    """MuJoCo setup only: bring the head to where the SDK's goto_sleep leaves it (SLEEP_HEAD_POSE, sleep antennas,
    daemon/backend/abstract.py:1330-1375) and wait until it is still. Needed because connecting already moved the
    MuJoCo head to REST: the connect's ``stop_head_tracking`` sets ``ik_required`` with no head target, and the IK of
    a missing target is the identity pose (abstract.py:555-560, 712-719) - MuJoCo motors are always on."""
    import numpy as np
    from reachy_mini.reachy_mini import SLEEP_ANTENNAS_JOINT_POSITIONS, SLEEP_HEAD_POSE

    from reachy_motion.sim_review import wait_still

    backend.goto_target(np.array(SLEEP_HEAD_POSE, dtype=float), [float(a) for a in SLEEP_ANTENNAS_JOINT_POSITIONS],
                        0.0, 3.0, "minjerk")
    return wait_still(backend, window_s=0.5, tol_deg=0.05, timeout_s=20.0)


def cmd_sim_startup(args: argparse.Namespace) -> int:
    """The start-up procedure in local MuJoCo, optionally against the SDK wake_up goto sequence (simulated)."""
    from reachy_motion.backends.sdk import SdkBackend
    from reachy_motion.executor.executor import Executor, StopPolicy
    from reachy_motion.sdk_env import environment_report
    from reachy_motion.startup_procedure import FK_JOINT_TOL_DEG, MIN_STARTUP_S, SLOW_CAPS, STILL_TOL_DEG, run_startup
    from reachy_motion.util.platform_timing import enable_high_resolution_timer

    if not _pins_ok():
        return 1
    enable_high_resolution_timer()
    out = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "command": "python -m reachy_motion sim-startup" + (" --compare-wake-up" if args.compare_wake_up else ""),
        "data_kind": "simulated", "note": "local MuJoCo (127.0.0.1), starts in the SDK sleep state; not the robot",
        "procedure": {"slow_caps": SLOW_CAPS, "min_s": MIN_STARTUP_S, "fk_joint_tol_deg": FK_JOINT_TOL_DEG,
                      "still_tol_deg": STILL_TOL_DEG, "stop_policy": StopPolicy().__dict__},
        "environment": environment_report()}
    from reachy_motion.sim_review import wait_still

    backend = SdkBackend("sim", port=args.port, viewer=args.viewer, log_dir=out / "daemon_startup")
    backend.connect()
    try:
        # MuJoCo only: the daemon starts the head in the SDK sleep joints, which its joint limits push back inside
        # (PHASE4_PLAN §0), so the head is still moving right after the start. Wait until it is still - on the
        # robot the procedure itself refuses a moving head (watch_still).
        summary["startup_settle"] = wait_still(backend, window_s=0.5, tol_deg=0.05, timeout_s=20.0)
        summary["startup_settle_at_sleep_pose"] = _to_sdk_sleep_pose(backend)
        ex = Executor(backend)
        report, result = run_startup(backend, ex)
        summary["startup"] = report.to_dict()
        if result is not None:
            t0 = result.started_ns
            _write_csv(out / "startup" / "commands.csv",
                       [_state_row((c.record.t_ns - t0) / 1e9, c.record.head, c.record.antennas, c.record.body_yaw)
                        for c in result.commands])
            _write_csv(out / "startup" / "present_simulated.csv",
                       [_state_row((p.t_ns - t0) / 1e9, p.head, p.antennas, p.body_yaw, p.joints)
                        for p in _distinct(result.present)])
            summary["startup_measured"] = _measured_motion(result.present)
    finally:
        backend.close()
    print(f"start-up procedure: {'OK' if report.ok else 'STOPPED at ' + report.stage} "
          f"{'; '.join(report.problems)}")
    if args.compare_wake_up:
        backend = SdkBackend("sim", port=args.port + 1, viewer=args.viewer, log_dir=out / "daemon_wake_up")
        backend.connect()
        try:
            wait_still(backend, window_s=0.5, tol_deg=0.05, timeout_s=20.0)
            settle = _to_sdk_sleep_pose(backend)
            samples, marks, t0 = _wake_up_replay(backend)
        finally:
            backend.close()
        s = _distinct(samples)
        if s:
            _write_csv(out / "wake_up" / "present_simulated.csv",
                       [_state_row((p.t_ns - t0) / 1e9, p.head, p.antennas, p.body_yaw, p.joints) for p in s])
        start_ns = t0 + int(marks[0][1] * 1e9) if marks else t0
        summary["wake_up_replay"] = {"settle": settle, "marks_s": marks,
                                     "measured": _measured_motion([p for p in s if p.t_ns >= start_ns]),
                                     "measured_before_first_goto": _measured_motion([p for p in s if p.t_ns < start_ns])}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    rows = [("start-up procedure", summary.get("startup_measured", {}))]
    if args.compare_wake_up:
        rows.append(("SDK wake_up gotos", summary["wake_up_replay"]["measured"]))
    print(f"{'measured (simulated)':22s} {'Stewart deg/s':>13s} {'max step':>9s} {'head deg/s':>10s} "
          f"{'antenna deg/s':>13s} {'span s':>7s}")
    for name, m in rows:
        if "peak_stewart_speed_dps" in m:
            print(f"{name:22s} {m['peak_stewart_speed_dps']:13.1f} {m['max_stewart_step_deg']:9.2f} "
                  f"{m['peak_head_rotation_dps']:10.1f} {m['peak_antenna_speed_dps']:13.1f} {m['span_s']:7.2f}")
    print(out / "summary.json")
    return 0 if report.ok else 2


def cmd_sim_stop(args: argparse.Namespace) -> int:
    """S1 / S2 stop experiments in local MuJoCo with position / velocity plots (``analysis.stop_sim``)."""
    from reachy_motion.analysis.stop_sim import run_scenarios
    from reachy_motion.backends.sdk import SdkBackend
    from reachy_motion.sim_review import wait_still
    from reachy_motion.util.platform_timing import enable_high_resolution_timer

    if not _pins_ok():
        return 1
    enable_high_resolution_timer()
    out = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    backend = SdkBackend("sim", port=args.port, viewer=args.viewer, log_dir=out / "daemon")
    backend.connect()
    try:
        wait_still(backend, window_s=0.5, tol_deg=0.05, timeout_s=20.0)
        summary = run_scenarios(backend, args.port, out)
    finally:
        backend.close()
    print(f"{'scenario':24s} {'stop':4s} {'cmds':>5s} {'known':>5s} {'v@stop':>7s} {'v+0.5s':>7s} {'v+1s':>6s} "
          f"{'gap@stop':>9s} {'gap 0.5s':>9s} {'gap 1s':>7s} {'(confirmed 1s)':>14s}   (deg, deg/s; simulated)")
    for name, r in summary["scenarios"].items():
        m = r["metrics"]
        print(f"{name:24s} {str(r['stop_kind']):4s} {r['commands']:5d} {str(r['end_state_known']):>5s} "
              f"{m.get('observed_speed_at_last_command_dps', math.nan):7.2f} "
              f"{m.get('observed_speed_0p5s_after_dps', math.nan):7.2f} {m.get('observed_speed_1s_after_dps', math.nan):6.2f} "
              f"{m.get('gap_to_last_target_at_stop_deg', math.nan):9.2f} {m.get('gap_to_last_target_0p5s_deg', math.nan):9.2f} "
              f"{m.get('gap_to_last_target_1s_deg', math.nan):7.2f} "
              f"{m.get('gap_to_last_confirmed_target_1s_deg', math.nan):14.2f}")
    print(out / "overview.png")
    return 0


def register(sub: Any) -> None:
    p = sub.add_parser("robot-status", help="R0: read-only REST GET checklist of a daemon (no motion, no client)")
    p.add_argument("--host", default=ROBOT_HOST, help=f"daemon host (default: the project robot {ROBOT_HOST})")
    p.add_argument("--port", type=int, default=ROBOT_DAEMON_PORT)
    p.add_argument("--timeout", type=float, default=3.0, help="seconds per GET")
    p.add_argument("--out", default=str(R0_OUT))
    p.set_defaults(func=cmd_robot_status)
    p = sub.add_parser("robot-r1", help="ONE R1 trial: Mock run, or the robot preflight + approval gate (Step B: "
                                        "no robot connection)")
    p.add_argument("--r1", required=True, help="AXIS:AMPLITUDE, e.g. pitch:+2, z:-1, antenna_right:+5 (deg / mm)")
    p.add_argument("--rep", type=int, default=1)
    p.add_argument("--r0", help="R0 report (r0_report.json): the antennas stay where it read them; required for the "
                                "robot, and the Mock starts there")
    p.add_argument("--target", choices=("mock", "robot"), default="mock")
    p.add_argument("--start", choices=("sleep", "awake"), default="sleep",
                   help="sleep = limp start with the start-up / return ramps; awake = woken the usual way and standing at INIT: no ramps, INIT -> trial -> INIT, the robot stays awake")
    p.add_argument("--host", default=ROBOT_HOST)
    p.add_argument("--port", type=int, default=ROBOT_DAEMON_PORT)
    p.add_argument("--timeout", type=float, default=3.0, help="seconds per R0 GET")
    p.add_argument("--armed", action="store_true", help="robot: arm the approval gate (CLAUDE.md rule 3)")
    p.add_argument("--ramp-approval", help="robot: the R1-0 approval file YOU wrote for the start-up and return ramps "
                                           "(approval-template --r1 ... --part ramp prints the form)")
    p.add_argument("--approval", help="robot: the R1 approval file YOU wrote for the trial (approval-template --r1 ... "
                                      "--part trial prints the form)")
    p.add_argument("--notes", default="", help="experiment notes saved with the run")
    p.add_argument("--out", default=str(Path("results") / "lab"), help="run sessions (the Motion Lab folder)")
    p.add_argument("--preflight-out", default=str(Path("results") / "phase4" / "r1_preflight"))
    p.set_defaults(func=cmd_robot_r1)
    for name, hlp in (("robot-r2", "R2: confirmed motions (E/N pairs) - Mock runs, or ONE robot trial per run (preflight "
                                   "+ two approvals, then the locks)"),
                      ("robot-r3", "R3: ONE step of the duration ladder (pitch +A in D and back); shorter steps need "
                                   "--after <previous step run folder>")):
        p = sub.add_parser(name, help=hlp)
        if name == "robot-r2":
            p.add_argument("--motion", default=",".join(("aha_recommend", "boredom", "success")),
                           help="confirmed motions, comma separated (default: the (A)/(B)/(C) pairs "
                                "aha_recommend,boredom,success)")
            p.add_argument("--variant", choices=("expressive", "neutral", "both"), default="both")
        else:
            p.add_argument("--amplitude", type=float, default=20.0, help="pitch amplitude A (deg), one per ladder")
            p.add_argument("--duration", type=float, required=True, help="one-way time D (s); base step 1.0")
            p.add_argument("--after", help="the previous step's run folder (required below the base step)")
        p.add_argument("--rep", type=int, default=1)
        p.add_argument("--r0", help="R0 report (r0_report.json) the run is planned from; required for the robot")
        p.add_argument("--antennas", choices=("designed", "hold"), default="designed",
                       help="designed = the trial's own start antennas (refused unless R0 reads them there; never "
                            "moved automatically); hold = keep them where R0 read them (head-only diagnosis)")
        p.add_argument("--target", choices=("mock", "robot"), default="mock")
        p.add_argument("--start", choices=("sleep", "awake"), default="sleep",
                       help="sleep = limp start with the start-up / return ramps; awake = woken the usual way and standing at INIT: no ramps, INIT -> trial -> INIT, the robot stays awake")
        p.add_argument("--host", default=ROBOT_HOST)
        p.add_argument("--port", type=int, default=ROBOT_DAEMON_PORT)
        p.add_argument("--timeout", type=float, default=3.0, help="seconds per R0 GET")
        p.add_argument("--armed", action="store_true", help="robot: arm the approval gate (CLAUDE.md rule 3)")
        p.add_argument("--ramp-approval", help="robot: the R1-0 approval file YOU wrote (same R0 -> same ramps)")
        p.add_argument("--approval", help="robot: the trial approval file YOU wrote (approval-template --part trial)")
        p.add_argument("--c1-report", help="robot: c1_report.json of a clean measured C1 (required)")
        p.add_argument("--r1-session", action="append", default=[],
                       help="robot: session_<trial>.json of a clean measured R1 run (repeatable; required)")
        p.add_argument("--notes", default="", help="experiment notes saved with the run")
        p.add_argument("--out", default=str(Path("results") / "lab"), help="run sessions (the Motion Lab folder)")
        p.add_argument("--preflight-out", default=str(Path("results") / "phase4" / f"{name[6:]}_preflight"))
        p.set_defaults(func=cmd_robot_r2 if name == "robot-r2" else cmd_robot_r3)
    p = sub.add_parser("robot-c1", help="C1: torque on where the head rests, hold, torque off - preflight + approval "
                                        "gate only (no robot connection before Step C)")
    p.add_argument("--r0", required=True, help="the R0 report (r0_report.json) the C1 approval was written for")
    p.add_argument("--host", default=ROBOT_HOST)
    p.add_argument("--port", type=int, default=ROBOT_DAEMON_PORT)
    p.add_argument("--timeout", type=float, default=3.0, help="seconds per R0 GET")
    p.add_argument("--armed", action="store_true", help="arm the approval gate (CLAUDE.md rule 3)")
    p.add_argument("--approval", help="the approval file YOU wrote (approval-template --c1 prints the form)")
    p.add_argument("--preflight-out", default=str(Path("results") / "phase4" / "c1_preflight"))
    p.set_defaults(func=cmd_robot_c1)
    p = sub.add_parser("tracking", help="tracking metrics of recorded runs (computed from their data)")
    p.add_argument("path", help="a run folder (commands.csv + present_*.csv) or a review folder with runs/")
    p.add_argument("--rate", type=float, default=200.0, help="analysis grid (Hz)")
    p.add_argument("--max-lag", type=float, default=0.5, help="lag search range (s)")
    p.add_argument("--min-hold", type=float, default=0.2, help="shortest hold used for bias / steps (s)")
    p.add_argument("--settle-skip", type=float, default=0.5, help="hold start excluded from the bias (s)")
    p.add_argument("--out", default=str(TRACKING_OUT))
    p.set_defaults(func=cmd_tracking)
    p = sub.add_parser("sim-startup", help="Phase 4 start-up procedure in local MuJoCo (simulated, never the robot)")
    p.add_argument("--port", type=int, default=8012, help="local sim daemon port (--compare-wake-up also uses +1)")
    p.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer")
    p.add_argument("--compare-wake-up", action="store_true",
                   help="then replay the SDK wake_up goto sequence from a fresh sleep start")
    p.add_argument("--out", default=str(STARTUP_OUT))
    p.set_defaults(func=cmd_sim_startup)
    p = sub.add_parser("sim-stop", help="S1 / S2 stop experiments in local MuJoCo with plots (simulated)")
    p.add_argument("--port", type=int, default=8014, help="local sim daemon port")
    p.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer")
    p.add_argument("--out", default=str(Path("results") / "phase4" / "stop_sim"))
    p.set_defaults(func=cmd_sim_stop)
