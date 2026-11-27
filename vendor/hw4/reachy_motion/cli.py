"""Command-line entry point: ``python -m reachy_motion <command>`` (or ``reachy-motion``).

Phase 1 commands - none of them talks to the real robot:
    env        print installed versions and check the SDK pins (exit 1 on mismatch)
    phase1     regenerate the Phase 1 SDK-analysis figures and tables (computed data, no daemon)
    sim-smoke  spawn a local MuJoCo daemon on 127.0.0.1 (robot_name=hw4_sim) and check the
               goto_target flow (simulated data); refuses unless the daemon reports
               simulation_enabled=True
    sim-api-check  spawn the same local sim daemon and record read-only REST responses
               (/api/daemon/status, /api/kinematics/info, /api/state/full with target flags)

Phase 2 commands (specs, check, play, sim-play, timing, export, approval-template, gui, phase2) live in
``reachy_motion.cli_phase2``; none of them can move the real robot.

Phase 4 preparation commands (robot-status = R0 read-only GETs, tracking = metrics of recorded runs) live in
``reachy_motion.cli_phase4``; neither moves anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_PHASE1_OUT = Path("results") / "phase1"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    def fmt(v: object) -> object:
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.6g}"
        return v

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(v) for k, v in row.items()})
    return path


def cmd_env(_args: argparse.Namespace) -> int:
    from reachy_motion.sdk_env import EXPECTED_PINS, environment_report, pin_mismatches

    print(json.dumps(environment_report(), indent=2))
    mismatches = pin_mismatches()
    if mismatches:
        for dist, (want, got) in mismatches.items():
            print(f"PIN MISMATCH {dist}: expected {want}, installed {got}", file=sys.stderr)
        return 1
    print("pins OK: " + ", ".join(f"{d}=={v}" for d, v in EXPECTED_PINS.items()))
    return 0


def ik_probe_poses() -> dict[str, object]:
    from reachy_motion.analysis.goto_replay import HeadTarget

    poses: dict[str, object] = {"neutral": HeadTarget()}
    for a in (10, 20, 30, 40):
        for sign in (1, -1):
            poses[f"pitch{sign * a:+d}"] = HeadTarget(pitch_deg=sign * a)
            poses[f"roll{sign * a:+d}"] = HeadTarget(roll_deg=sign * a)
    for a in (30, 60, 90):
        for sign in (1, -1):
            poses[f"yaw{sign * a:+d}"] = HeadTarget(yaw_deg=sign * a)
    for d in (10, 20):
        for sign in (1, -1):
            poses[f"z{sign * d:+d}mm"] = HeadTarget(z_mm=sign * d)
    for sign in (1, -1):
        poses[f"x{sign * 10:+d}mm"] = HeadTarget(x_mm=sign * 10)
        poses[f"y{sign * 10:+d}mm"] = HeadTarget(y_mm=sign * 10)
    return poses


def cmd_phase1(args: argparse.Namespace) -> int:
    from reachy_motion.analysis import figures as F
    from reachy_motion.analysis.goto_replay import (
        JOINT_NAMES,
        TASK_COORDS,
        GotoCase,
        HeadTarget,
        ik_sensitivity,
        ik_table,
        replay_goto,
    )
    from reachy_motion.analysis.interpolation import (
        METHODS,
        PLAY_MOVE_PERIOD_S,
        characterize,
        endpoint_shortfall,
        physical_peaks,
    )
    from reachy_motion.sdk_env import environment_report, installed_version, pin_mismatches

    if pin_mismatches() and not args.allow_version_mismatch:
        print("SDK pins do not match; run `reachy-motion env`. Refusing to label results "
              "as SDK 1.11.0 analysis (use --allow-version-mismatch to override).", file=sys.stderr)
        return 1

    out = Path(args.out)
    figs = out / "figures"
    written: list[Path] = []

    # 1. Normalised characteristics + physical peaks + endpoint shortfall
    chars = [characterize(m) for m in METHODS]
    written.append(_write_csv(out / "interpolation_characteristics.csv", [c.as_row() for c in chars]))
    peaks = [
        physical_peaks(c, a, d)
        for c in chars
        for a in (10.0, 20.0, 30.0)
        for d in (0.3, 0.5, 1.0, 2.0)
    ]
    written.append(_write_csv(out / "interpolation_peaks_physical.csv", peaks))
    # 10 ms = nominal play_move period (robot, Linux); 15.625 ms = idle-loop asyncio tick of a stub on
    # this Windows PC (time.monotonic resolution). A running daemon was not measured.
    shortfall = [
        {
            "method": m.value,
            "duration_s": d,
            "assumed_last_gap_s": dt,
            "where": ("robot daemon nominal period (100 Hz); real gaps are >= this"
                      if dt == PLAY_MOVE_PERIOD_S
                      else "Windows idle-loop stub tick (running daemon unmeasured)"),
            "shortfall_if_last_gap_eq_assumed": endpoint_shortfall(m, d, dt),
        }
        for dt in (PLAY_MOVE_PERIOD_S, 0.015625)
        for m in METHODS
        for d in (0.2, 0.3, 0.5, 1.0, 2.0, 3.0)
    ]
    written.append(_write_csv(out / "endpoint_shortfall.csv", shortfall))

    # 2. IK structure: joint angles of probe poses, local sensitivity at neutral
    written.append(_write_csv(out / "ik_pose_table.csv", ik_table(ik_probe_poses())))
    jac = ik_sensitivity()
    sens_rows = [
        {"joint": jn, **{f"d_{c}": float(jac[i, k]) for k, c in enumerate(TASK_COORDS)}}
        for i, jn in enumerate(JOINT_NAMES)
    ]
    written.append(_write_csv(out / "ik_sensitivity_neutral.csv", sens_rows))

    # 3. Joint-space image of one goto per method + pipeline delay of the command path
    amplitude, duration = 20.0, 1.0
    replays = {
        m: replay_goto(GotoCase(target=HeadTarget(pitch_deg=amplitude), duration=duration, method=m))
        for m in METHODS
    }
    joint_rows = []
    for m, r in replays.items():
        q = np.rad2deg(r.joints_dense)
        qd = np.rad2deg(r.joint_velocity_dense())
        for j, jn in enumerate(JOINT_NAMES):
            joint_rows.append({
                "method": m.value,
                "case": f"pitch 0->{amplitude:g} deg, D={duration:g} s",
                "joint": jn,
                "start_deg": q[0, j],
                "end_deg": q[-1, j],
                "min_deg": q[:, j].min(),
                "max_deg": q[:, j].max(),
                "peak_speed_deg_s": np.abs(qd[:, j]).max(),
                "ik_failures": r.ik_failures,
            })
    written.append(_write_csv(out / "goto_joint_peaks.csv", joint_rows))

    # The 100 Hz evaluation and the 50 Hz control loop are unsynchronised, so the control phase is
    # uniform; sample it densely (0.5 ms) instead of a coarse grid that misses the upper end.
    from reachy_motion.analysis.goto_replay import pipeline_lag

    delay_rows = []
    pipelines = {
        "robot: eval 100 Hz, ctrl 50 Hz, IK->write +1 tick": 1,
        "mujoco structure: eval 100 Hz, ctrl 50 Hz, same tick": 0,
    }
    for m, base in replays.items():  # replays: 100 Hz evaluation, computed above
        j = int(np.argmax(np.ptp(base.joints_dense[:, 1:], axis=0))) + 1
        for label, delay_ticks in pipelines.items():
            for k in range(40):  # control phase 0 ... 19.5 ms
                phase = k * 0.0005
                lag = pipeline_lag(base, j, phase, delay_ticks)
                delay_rows.append({"pipeline": label, "method": m.value, "control_phase_s": phase,
                                   "joint": JOINT_NAMES[j], "modelled_lag_s": lag})
    written.append(_write_csv(out / "pipeline_delay_model.csv", delay_rows))

    # 3b. Preliminary motor demand grid (IK-differentiated speeds vs datasheet, limit margins)
    from reachy_motion.analysis.motor_demand import demand_grid

    written.append(_write_csv(out / "motor_speed_grid.csv", demand_grid(METHODS)))

    # 4. Figures
    written += [
        F.fig_normalized_profiles(figs / "fig1_interpolation_normalized.png"),
        F.fig_physical_example(figs / "fig2_interpolation_physical.png", amplitude, duration),
        F.fig_joint_space(replays, figs / "fig3_goto_joint_space.png"),
        F.fig_endpoint_shortfall(figs / "fig4_endpoint_shortfall.png"),
        F.fig_pipeline_timing(figs / "fig5_command_pipeline.png"),
    ]

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_kind": "computed",
        "note": f"All values are computed by calling reachy_mini {installed_version('reachy-mini')} functions "
                "offline (time_trajectory, GotoMove, AnalyticalKinematics). No robot, no daemon, no MuJoCo.",
        "command": "python -m reachy_motion phase1 --out " + str(out)
                   + (" --allow-version-mismatch" if args.allow_version_mismatch else ""),
        "environment": environment_report(),
        "parameters": {
            "example_case": {"pitch_deg": amplitude, "duration_s": duration},
            "play_move_rate_hz": 100.0,
            "robot_control_rate_hz": 50.0,
            "dense_reference_rate_hz": 1000.0,
        },
        "files": [str(p.relative_to(out)).replace("\\", "/") for p in written],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for p in written:
        print(p)
    print(out / "manifest.json")
    return 0


def cmd_sim_smoke(args: argparse.Namespace) -> int:
    """Spawn a local MuJoCo daemon (127.0.0.1, robot_name=hw4_sim) and run one goto per method."""
    from reachy_motion.analysis.sim_smoke import run_sim_smoke
    from reachy_motion.sdk_env import pin_mismatches

    if pin_mismatches():
        print("SDK pins do not match; refusing to run (see `reachy-motion env`).", file=sys.stderr)
        return 1
    manifest = run_sim_smoke(Path(args.out), port=args.port, amplitude_deg=args.pitch_deg,
                             duration=args.duration)
    for s in manifest["summaries"]:
        print(json.dumps(s))
    print(Path(args.out) / "manifest.json")
    return 0


def cmd_sim_api_check(args: argparse.Namespace) -> int:
    """Read-only REST checks against a freshly spawned local sim daemon (no motion)."""
    from reachy_motion.analysis.sim_smoke import run_api_check
    from reachy_motion.sdk_env import pin_mismatches

    if pin_mismatches():
        print("SDK pins do not match; refusing to run (see `reachy-motion env`).", file=sys.stderr)
        return 1
    report = run_api_check(Path(args.out), port=args.port)
    for name, r in report["checks"].items():
        print(f"{name}: HTTP {r['http_status']} keys={r.get('response_keys', '-')}")
    print(Path(args.out) / "api_checks.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reachy-motion", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_env = sub.add_parser("env", help="print versions and check SDK pins")
    p_env.set_defaults(func=cmd_env)
    p1 = sub.add_parser("phase1", help="regenerate Phase 1 figures and tables (computed)")
    p1.add_argument("--out", default=str(DEFAULT_PHASE1_OUT), help="output directory")
    p1.add_argument("--allow-version-mismatch", action="store_true",
                    help="run even if installed SDK versions differ from the pins")
    p1.set_defaults(func=cmd_phase1)
    ps = sub.add_parser("sim-smoke", help="Phase 1 MuJoCo smoke check of goto_target (simulated, localhost only)")
    ps.add_argument("--out", default=str(Path("results") / "phase1" / "sim_smoke"))
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--pitch-deg", type=float, default=20.0)
    ps.add_argument("--duration", type=float, default=1.0)
    ps.set_defaults(func=cmd_sim_smoke)
    pa = sub.add_parser("sim-api-check", help="read-only REST checks against a local sim daemon (no motion)")
    pa.add_argument("--out", default=str(Path("results") / "phase1" / "sim_smoke"))
    pa.add_argument("--port", type=int, default=8000)
    pa.set_defaults(func=cmd_sim_api_check)
    from reachy_motion.cli_phase2 import register as register_phase2
    from reachy_motion.cli_phase4 import register as register_phase4

    register_phase2(sub)
    register_phase4(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
