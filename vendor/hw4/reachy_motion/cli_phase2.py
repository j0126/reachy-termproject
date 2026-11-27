"""Phase 2 commands of ``reachy-motion`` (registered by ``cli.build_parser``).

    specs              list the motion files (motions/*.yaml) with duration, keys, expected V/A, policy verdict
    check NAME         initial-policy report (+ neutral rules and expressive/neutral comparison) - computed
    play NAME          play on the Mock backend in real time (default) and save commands + tick timing;
                       --backend sim plays once in MuJoCo; --backend robot stops at the approval gate (Phase 4)
    sim-play NAME      MuJoCo viewer playback on 127.0.0.1 (interlocked); Enter = replay, r = reload file, v = other
                       variant, q = quit - for checking a design by eye (simulated data)
    sim-verify         Phase 3: play all motion files in one local MuJoCo session -> review tables (simulated)
    video              Phase 3: expressive-vs-neutral videos of every motion folder from an offline MuJoCo replay
                       (no daemon; H.264 MP4 via PyAV, or animated WebP; + index.html; simulated)
    timing             measure the 50 Hz executor's real period and jitter on this PC (Mock backend, pc_measured)
    export NAME        the Motion Editor's export bundle without the GUI
    approval-template  print (never write) the approval JSON a HUMAN fills in before Phase 4
    gui                open the Motion Editor (PySide6)
    phase2             regenerate the Phase 2 artifacts (results/phase2, docs/phase2/MOTION_SPECS.md) - computed

No command here can move the real robot: the robot backend refuses without armed + an approval file + a typed
confirmation, and in Phase 2 its connect() is not implemented (CLAUDE.md rules 3, 8).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reachy_motion.settings import ROBOT_HOST

PHASE2_OUT = Path("results") / "phase2"
MOTION_SPECS_MD = Path("docs") / "phase2" / "MOTION_SPECS.md"


# ---------------------------------------------------------------------- helpers
def _load_spec(args: argparse.Namespace, variant: str | None = None) -> Any:
    from reachy_motion.library.io import load_spec_file
    from reachy_motion.library.registry import motions_dir

    v = variant or args.variant
    if getattr(args, "file", None):
        spec = load_spec_file(Path(args.file))
    else:
        d = Path(args.dir) if getattr(args, "dir", None) else motions_dir()
        spec = load_spec_file(d / f"{args.name}.{v}.yaml")
    amp, tempo = getattr(args, "amplitude", 1.0), getattr(args, "tempo", 1.0)
    if amp != 1.0 or tempo != 1.0:
        spec = spec.scaled(amplitude=amp, tempo=tempo)
    return spec


def _counterpart(spec: Any, args: argparse.Namespace) -> Any | None:
    from reachy_motion.library.io import load_spec_file
    from reachy_motion.library.registry import motions_dir

    other = "neutral" if spec.variant == "expressive" else "expressive"
    d = Path(args.file).parent if getattr(args, "file", None) else (
        Path(args.dir) if getattr(args, "dir", None) else motions_dir())
    p = d / f"{spec.name}.{other}.yaml"
    return load_spec_file(p) if p.exists() else None


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, obj: Any) -> Path:
    from reachy_motion.gui.model import _json_default

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return path


def _print_report(report: Any, conformance: list[Any], comparison: dict[str, Any] | None) -> None:
    verdict = "WITHIN the initial policy" if report.passes_policy else "VIOLATES the initial policy"
    print(f"{report.name}:{report.variant}  {verdict}  (data_kind={report.data_kind})")
    print(f"  note: {report.disclaimer}")
    for v in report.violations:
        print(f"  VIOLATION {v.code}: {v.message}")
    for w in report.warnings:
        print(f"  warning   {w.code}: {w.message}")
    for i in conformance:
        print(f"  neutral   {i.code}: {i.message}")
    keys = ("duration_s", "min_stewart_margin_deg", "min_stewart_margin_joint", "peak_stewart_speed_ratio",
            "peak_body_speed_ratio", "peak_antenna_speed_ratio", "max_abs_roll_deg", "max_abs_pitch_deg",
            "max_abs_rel_yaw_deg", "max_auto_body_shift_deg", "max_stream_step_deg")
    for k in keys:
        if k in report.metrics:
            v = report.metrics[k]
            print(f"  {k:28s} {v:.3f}" if isinstance(v, float) else f"  {k:28s} {v}")
    if comparison is not None:
        issues = comparison.get("issues") or []
        print(f"  comparison with the other variant: {'OK' if not issues else f'{len(issues)} issue(s)'}")
        for i in issues:  # comparison_conditions returns JSON dicts (code, message, severity)
            print(f"    {i['code']}: {i['message']}")


# ---------------------------------------------------------------------- commands
def cmd_specs(args: argparse.Namespace) -> int:
    from reachy_motion.library.registry import default_library
    from reachy_motion.safety.checker import check_spec

    lib = default_library(Path(args.dir) if args.dir else None)
    print(f"{'motion':28s} {'dur s':>6s} {'keys h/a/b':>10s} {'V':>5s} {'A':>5s}  draft  policy")
    for s in lib:
        rep = check_spec(s)
        keys = f"{len(s.head)}/{len(s.antennas)}/{len(s.body)}"
        print(f"{s.key:28s} {s.duration:6.2f} {keys:>10s} {s.valence:5.2f} {s.arousal:5.2f}  "
              f"{'yes' if s.draft else 'no ':5s}  {'pass' if rep.passes_policy else 'FAIL'}")
    missing = lib.missing()
    if missing:
        print("missing: " + ", ".join(missing))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from reachy_motion.library.compare import comparison_conditions, neutral_conformance
    from reachy_motion.safety.checker import check_spec

    variants = ("expressive", "neutral") if args.variant == "both" else (args.variant,)
    ok = True
    out: dict[str, Any] = {}
    for v in variants:
        spec = _load_spec(args, v)
        report = check_spec(spec)
        conf = neutral_conformance(spec)
        other = _counterpart(spec, args)
        comp = None
        if other is not None:
            comp = comparison_conditions(spec, other) if spec.variant == "expressive" else comparison_conditions(other, spec)
        ok &= bool(report.passes_policy)
        if args.json:
            out[spec.key] = {"report": report.to_dict(), "neutral_conformance": [i.to_dict() for i in conf],
                             "comparison": comp}
        else:
            _print_report(report, conf, comp)
    if args.json:
        from reachy_motion.gui.model import _json_default

        print(json.dumps(out, indent=2, ensure_ascii=False, default=_json_default))
    return 0 if ok else 2


def _save_run(out: Path, spec: Any, result: Any, backend_kind: str, rate_hz: float, extra: dict[str, Any]) -> Path:
    from reachy_motion.gui.model import _write_rows, command_rows
    from reachy_motion.sdk_env import environment_report

    out.mkdir(parents=True, exist_ok=True)
    _write_rows(out / "commands.csv", command_rows(result))
    result.ticks.to_csv(out / "ticks.csv")
    summary = result.ticks.summary(int(round(1e9 / rate_hz)))
    present = getattr(result, "present", None) or []
    if present:
        from reachy_motion.analysis.goto_replay import JOINT_NAMES

        rows = []
        t0 = present[0].t_ns
        for p in present:
            row = {"t_s": (p.t_ns - t0) / 1e9, "antenna_right_rad": p.antennas[0], "antenna_left_rad": p.antennas[1],
                   "body_yaw_rad": p.body_yaw}
            if p.joints is not None:
                row.update({f"{n}_rad": float(q) for n, q in zip(JOINT_NAMES, p.joints)})
            rows.append(row)
        _write_rows(out / "present.csv", rows)
    _write_json(out / "manifest.json", {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "reachy-motion " + " ".join(sys.argv[1:]),
        "motion": spec.key, "draft": spec.draft, "spec_digest": spec.digest(),
        "completed": result.completed, "cancelled": result.cancelled, "error": result.error,
        "data_kind_by_file": {"commands.csv": "commanded (sent values)",
                              "present.csv": backend_kind, "ticks.csv": "pc_measured", "manifest.json": "-"},
        "backend_data_kind": backend_kind,
        "tick_timing_summary_pc_measured": summary,
        "environment": environment_report(), **extra,
    })
    return out


def cmd_play(args: argparse.Namespace) -> int:
    from reachy_motion.safety.checker import check_spec

    spec = _load_spec(args)
    report = check_spec(spec)
    if not report.passes_policy:
        _print_report(report, [], None)
        print("refused: the motion is outside the initial policy", file=sys.stderr)
        return 2
    if args.backend == "robot":
        return _robot_gate(args, spec)
    if args.backend == "sim":
        args.once = True
        return cmd_sim_play(args)
    from reachy_motion.backends.mock import MockBackend
    from reachy_motion.service import HandleState, MotionService
    from reachy_motion.util.platform_timing import enable_high_resolution_timer

    enable_high_resolution_timer()
    backend = MockBackend(structure="robot")
    backend.connect()
    service = MotionService(backend, record_present=True)
    service.start()
    try:
        h = service.play(spec, variant=spec.variant)
        h.wait(timeout=spec.duration + 30.0)
    finally:
        service.close()
        backend.close()
    print(f"{spec.key}: {h.state.value if hasattr(h.state, 'value') else h.state}"
          f"{' - ' + str(h.reason) if getattr(h, 'reason', None) else ''}")
    if h.state != HandleState.DONE or h.result is None:
        return 1
    out = Path(args.out) / f"{_stamp()}_{spec.name}_{spec.variant}_mock"
    _save_run(out, spec, h.result, "mock", 50.0, {"mock_command_count": backend.motor_command_count})
    s = h.result.ticks.summary(20_000_000)
    print(f"saved {out}  (mock commands {backend.motor_command_count}; ticks {s.get('n_ticks')}, "
          f"effective {s.get('effective_hz', float('nan')):.2f} Hz, interval p95 "
          f"{(s.get('interval_ms') or {}).get('p95', float('nan')):.2f} ms - pc_measured)")
    return 0


def _robot_gate(args: argparse.Namespace, spec: Any) -> int:
    """Phase 4 path: armed + approval file + typed confirmation; Phase 2 refuses at connect() regardless."""
    from reachy_motion.backends.sdk import SdkBackend
    from reachy_motion.planner import plan
    from reachy_motion.safety.approval import ApprovalError, ApprovalRequest, load_approval
    from reachy_motion.safety.checker import check_spec

    if not args.armed or not args.approval:
        print("refused: the robot backend needs --armed AND --approval <file written by the user> "
              "(CLAUDE.md rule 3). Phase 2 never drives the robot.", file=sys.stderr)
        return 3
    try:
        approval = load_approval(Path(args.approval))
        request = ApprovalRequest(host=args.host, stage=approval.stage, motions=(spec.key,),
                                  timeline_digests=(plan(spec).digest(),), reports=(check_spec(spec),))
        backend = SdkBackend("robot", armed=True, approval=approval, request=request)
        backend.connect()
    except ApprovalError as exc:
        print(f"refused by the approval gate: {exc}", file=sys.stderr)
        return 3
    except NotImplementedError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    print("unexpected: robot backend connected", file=sys.stderr)
    return 4


def _startup_to_rest(backend: Any, executor: Any) -> tuple[bool, Any]:
    """Ramp from the measured present pose (MuJoCo starts in the sleep pose) to REST under the start-up rules of
    ``safety.startup`` (per-joint "never closer to a limit than at the start", end pose inside the normal policy)."""
    from reachy_motion.library.pose import REST, RobotState, pose_from_matrix
    from reachy_motion.safety.startup import prepare_startup_ramp

    p = backend.present()
    start = RobotState(head=pose_from_matrix(p.head), antennas=tuple(p.antennas), body_yaw=float(p.body_yaw))
    prep = prepare_startup_ramp(start, REST)
    if not prep.runnable:
        print("refused: start-up ramp: " + "; ".join(v.message for v in prep.report.violations), file=sys.stderr)
        return False, prep
    executor.run(prep)
    return True, prep


def cmd_sim_play(args: argparse.Namespace) -> int:
    """MuJoCo viewer playback for checking a design by eye (127.0.0.1 only, interlocked; simulated data)."""
    from reachy_motion.backends.sdk import SdkBackend
    from reachy_motion.executor.executor import Executor
    from reachy_motion.safety.checker import check_spec, prepare
    from reachy_motion.sdk_env import pin_mismatches
    from reachy_motion.util.platform_timing import enable_high_resolution_timer

    if pin_mismatches():
        print("SDK pins do not match; refusing to run (see `reachy-motion env`).", file=sys.stderr)
        return 1
    spec = _load_spec(args)
    report = check_spec(spec)
    if not report.passes_policy:
        _print_report(report, [], None)
        print("refused: the motion is outside the initial policy", file=sys.stderr)
        return 2
    enable_high_resolution_timer()
    out_root = Path(args.out)
    log_dir = out_root / f"{_stamp()}_sim_daemon"
    backend = SdkBackend("sim", port=args.port, viewer=not args.headless, log_dir=log_dir)
    print(f"starting the local MuJoCo daemon on 127.0.0.1:{args.port} ({'headless' if args.headless else 'viewer'}) ...")
    backend.connect()
    runs = 0
    try:
        executor = Executor(backend)
        ok, _ramp = _startup_to_rest(backend, executor)
        if not ok:
            return 2
        print("at rest. ", end="")
        while True:
            prepared = prepare(spec)
            if not prepared.runnable:
                _print_report(prepared.report, [], None)
                print("refused: outside the initial policy")
            else:
                runs += 1
                print(f"playing {spec.key} ({spec.duration:.2f} s, digest {spec.digest()[:12]}) ...")
                result = executor.run(prepared, record_present=True)
                out = out_root / f"{_stamp()}_{spec.name}_{spec.variant}_sim{runs}"
                _save_run(out, spec, result, "simulated", executor.rate_hz,
                          {"sim_daemon_log": str(log_dir), "viewer": not args.headless,
                           "note": "MuJoCo is friction-dominated and runs ~0.84x real time on this PC; use it to check "
                                   "the design by eye, not as a prediction of robot tracking (Phase 1 5.5)"})
                print(f"  done ({'completed' if result.completed else result.error or 'stopped'}) -> {out}")
            if args.once or not sys.stdin.isatty():
                break
            cmd = input("[Enter] replay  [r] reload file  [v] other variant  [q] quit > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "r":
                spec = _load_spec(args, spec.variant)
            elif cmd == "v":
                spec = _load_spec(args, "neutral" if spec.variant == "expressive" else "expressive")
        return 0
    finally:
        backend.close()


def cmd_sim_verify(args: argparse.Namespace) -> int:
    """Phase 3 task 1: play all motion files in one local MuJoCo session and write the review material."""
    from reachy_motion.library.registry import default_library
    from reachy_motion.sdk_env import pin_mismatches
    from reachy_motion.sim_review import run_review
    from reachy_motion.util.platform_timing import enable_high_resolution_timer

    if pin_mismatches():
        print("SDK pins do not match; refusing to run (see `reachy-motion env`).", file=sys.stderr)
        return 1
    enable_high_resolution_timer()
    lib = default_library(Path(args.dir) if args.dir else None)
    specs = {(s.name, s.variant): s for s in lib}
    out = Path(args.out) / _stamp()
    print(f"local MuJoCo daemon on 127.0.0.1:{args.port} ({'viewer' if args.viewer else 'headless'}), "
          f"{len(specs)} motions ...")
    summary = run_review(out, specs, port=args.port, viewer=args.viewer, hold_s=args.hold, startup=_startup_to_rest)
    for key, m in summary["motions"].items():
        if "measured_end" in m:
            e = m["end_error"]
            print(f"  {key:26s} {'completed' if m['completed'] else m['error']}; end error {e['head_angle_deg']:.2f} deg / "
                  f"{e['head_trans_mm']:.2f} mm / antennas {e['antenna_deg']:.2f} deg; measured min margin "
                  f"{m['measured_margin']['min_margin_deg']:.1f} deg")
        else:
            print(f"  {key:26s} {m.get('skipped')}")
    print(out / "review.md")
    return 0


def cmd_video(args: argparse.Namespace) -> int:
    """Expressive-vs-neutral review videos: offline MuJoCo replay -> animated WebP + index.html (simulated)."""
    from dataclasses import asdict

    from reachy_motion import sim_video
    from reachy_motion.library.registry import default_library, motions_dir
    from reachy_motion.sdk_env import environment_report, pin_mismatches

    if pin_mismatches():
        print("SDK pins do not match; refusing to run (see `reachy-motion env`).", file=sys.stderr)
        return 1
    try:
        w, h = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        print(f"--size must be WIDTHxHEIGHT, got {args.size!r}", file=sys.stderr)
        return 2
    if not (16 <= w <= 1280 and 16 <= h <= 720):  # the scene's offscreen buffer is 1280x720 (scenes/empty.xml:9)
        print("--size must be within 16x16 .. 1280x720", file=sys.stderr)
        return 2
    has_mp4 = sim_video.mp4_available()
    formats = {"auto": ("mp4",) if has_mp4 else ("webp",), "mp4": ("mp4",), "webp": ("webp",),
               "both": ("mp4", "webp")}[args.format]
    if "mp4" in formats and not has_mp4:
        print("MP4 needs PyAV with libx264: uv pip install --offline --no-deps --python .venv/Scripts/python.exe "
              "av==19.0.1  (or use --format webp)", file=sys.stderr)
        return 2
    if args.format == "auto" and not has_mp4:
        print("note: PyAV not installed - writing animated WebP instead of MP4")
    base = motions_dir()
    if args.dir:
        dirs = [Path(d) for d in args.dir]
    else:
        cand = base / "candidates"
        dirs = [base] + (sorted(p for p in cand.iterdir() if p.is_dir()) if cand.is_dir() else [])
    names = args.name or ["aha_recommend", "boredom", "success", "failure"]
    jobs = []
    for d in dirs:
        lib = default_library(d)
        main = d.resolve() == base.resolve()
        single = len({s.name for s in lib}) == 1  # a candidate folder holds one motion: label it by the folder
        for name in names:
            specs = [lib.get(name, v) for v in ("expressive", "neutral") if f"{name}:{v}" in lib.keys()]
            if not specs:
                continue
            label = name if main else (d.name if single or d.name.startswith(name) else f"{d.name}_{name}")
            title = (f"{name}  (motions/)" if main else f"{name}  (candidate {d.name})") + "  -  " + \
                " vs ".join(s.variant for s in specs)
            jobs.append((label, d, title, specs))
    if not jobs:
        print("no motion files found", file=sys.stderr)
        return 1
    out = Path(args.out) / _stamp()
    print(f"{len(jobs)} video(s) -> {out}  (offline MuJoCo replay, simulated; not the robot)")
    entries, skipped = [], []
    for label, d, title, specs in jobs:
        try:
            e = sim_video.render_pair(specs, label, out, title=title, fps=args.fps, size=(w, h), scene=args.scene,
                                      lead_s=args.lead, tail_s=args.tail, formats=formats)
        except ValueError as exc:
            skipped.append({"label": label, "dir": str(d), "reason": str(exc)})
            print(f"  {label:16s} skipped: {exc}")
            continue
        e["dir"] = str(d)
        entries.append(e)
        print(f"  {label:16s} {e['video_s']:5.1f} s, {e['file']} {e['size_bytes'] / 1e6:4.1f} MB, end error "
              + " / ".join(f"{m['end_error']['head_angle_deg']:.2f} deg" for m in e["motions"]))
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "reachy-motion " + " ".join(sys.argv[1:]), "data_kind": sim_video.DATA_KIND,
        "note": sim_video.NOTE,
        "parameters": {"formats": list(formats), "fps": args.fps, "panel_size": [w, h], "scene": args.scene,
                       "lead_s": args.lead,
                       "tail_s": args.tail, "camera": asdict(sim_video.REVIEW_CAMERA),
                       "physics_dt_s": sim_video.PHYSICS_DT_S, "control_rate_hz": sim_video.CONTROL_RATE_HZ},
        "videos": entries, "skipped": skipped, "environment": environment_report()}
    index = sim_video.write_index(out, entries, manifest)
    print(index)
    return 0 if entries else 1


def cmd_timing(args: argparse.Namespace) -> int:
    """Executor tick timing on this PC with the Mock backend (pc_measured, not the robot)."""
    from reachy_motion.backends.mock import MockBackend
    from reachy_motion.executor.clock import RealClock
    from reachy_motion.executor.timing import measure_tick_timing
    from reachy_motion.sdk_env import environment_report
    from reachy_motion.util.platform_timing import enable_high_resolution_timer, high_resolution_timer_period_ms

    hires = enable_high_resolution_timer() if not args.no_hires_timer else False
    backend = MockBackend(structure="robot")
    backend.connect()
    try:
        log = measure_tick_timing(backend, duration_s=args.duration, rate_hz=args.rate, clock=RealClock(args.spin_us))
    finally:
        backend.close()
    period_ns = int(round(1e9 / args.rate))
    summary = log.summary(period_ns)
    out = Path(args.out) / f"{_stamp()}_rate{args.rate:g}_spin{args.spin_us}_{'hires' if hires else 'default'}"
    out.mkdir(parents=True, exist_ok=True)
    log.to_csv(out / "ticks.csv")
    _write_json(out / "summary.json", {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_kind": "pc_measured", "note": "executor tick timing on this PC with the Mock backend - not robot data",
        "command": "reachy-motion " + " ".join(sys.argv[1:]), "rate_hz": args.rate, "duration_s": args.duration,
        "spin_us": args.spin_us, "timeBeginPeriod_ms": high_resolution_timer_period_ms(),
        "summary": summary, "environment": environment_report()})
    _timing_figure(out / "ticks.csv", out / "intervals.png", args.rate, summary)
    print(json.dumps(summary, indent=2, default=str))
    print(out)
    return 0


def _timing_figure(csv_path: Path, png: Path, rate: float, summary: dict[str, Any]) -> None:
    import csv as _csv

    import numpy as np

    from reachy_motion.gui import plots

    with csv_path.open(encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    iv = np.array([float(r["interval_ms"]) for r in rows if r.get("interval_ms") not in ("", None, "nan")])
    late = np.array([float(r["late_ms"]) for r in rows if r.get("late_ms") not in ("", None)])
    fig = plots.new_figure(9.0, 3.6)
    ax1, ax2 = fig.subplots(1, 2)
    if iv.size:
        ax1.hist(iv, bins=60, color=plots.C3[0])
    ax1.axvline(1000.0 / rate, color=plots.INK, linewidth=1.0, linestyle="--", label=f"nominal {1000.0 / rate:g} ms")
    ax1.set_xlabel("tick interval (ms)")
    ax1.set_ylabel("ticks")
    ax1.legend(fontsize=7, frameon=False)
    if late.size:
        ax2.plot(late, color=plots.C3[1], linewidth=0.8)
    ax2.set_xlabel("tick")
    ax2.set_ylabel("wake-up lateness (ms)")
    iv_s = summary.get("interval_ms") or {}
    fig.suptitle(f"Executor tick timing, Mock backend, this PC (pc_measured): effective "
                 f"{summary.get('effective_hz', float('nan')):.2f} Hz, interval p95 {iv_s.get('p95', float('nan')):.2f} ms,"
                 f" max {iv_s.get('max', float('nan')):.2f} ms", fontsize=9, x=0.01, ha="left")
    fig.savefig(png, dpi=130)


def cmd_export(args: argparse.Namespace) -> int:
    from reachy_motion.gui.model import EditorModel

    model = EditorModel(Path(args.dir) if args.dir else None)
    if args.file:
        model.open_file(Path(args.file))
    else:
        model.open(args.name, args.variant)
    out = Path(args.out) / f"{_stamp()}_{model.spec.name}_{model.spec.variant}"
    files = model.export_bundle(out)
    print(f"exported {len(files)} files to {out}")
    return 0


def cmd_approval_template(args: argparse.Namespace) -> int:
    from reachy_motion.library.io import load_spec_file
    from reachy_motion.library.registry import motions_dir
    from reachy_motion.planner import plan
    from reachy_motion.safety.approval import ApprovalRequest, approval_template
    from reachy_motion.safety.checker import check_spec

    if args.r2 or args.r3:  # exactly what `robot-r2` / `robot-r3` would request (planned from --r0)
        from reachy_motion import robot_r1

        if args.motion or args.c1 or args.r1 or (args.r2 and args.r3) or args.part is None:
            print("--r2 / --r3 go alone (no --motion / --r1 / --c1) and with --part ramp|trial", file=sys.stderr)
            return 2
        try:
            ref = robot_r1.load_r0(args.r0) if args.r0 else None
            hold = args.antennas == "hold"
            if args.r2:
                variants = ("expressive", "neutral") if args.variants == "both" else (args.variants,)
                plans = [robot_r1.r2_plan(m.strip(), v, args.rep, ref, hold) for m in args.r2.split(",") if m.strip()
                         for v in variants]
            else:
                amp, _, dur = args.r3.partition(":")
                plans = [robot_r1.r3_plan(float(amp), float(dur), args.rep, ref, hold)]
            if args.part == "ramp":
                if args.stage != "R1":
                    raise ValueError("--part ramp is the R1-0 approval of the ramps: use --stage R1")
                if len({p.ramp.timeline.digest() for p in plans}) != 1:
                    raise ValueError("these trials do not share one start-up ramp")
                request = robot_r1.ramp_request(plans[0], args.host)
                print("# APPROVAL 1/2 - " + robot_r1.ramp_summary(plans[0]))
            else:
                stage = plans[0].trial.stage
                if args.stage != stage:
                    raise ValueError(f"--part trial of {stage} trials: use --stage {stage}")
                request = robot_r1.trials_request(plans, args.host)
                print(f"# APPROVAL 2/2 - {stage}: {len(plans)} trial(s); robot-{stage.lower()} runs ONE per run, each "
                      "covered by this approval")
                for p in plans:
                    row = p.check.row()
                    print(f"# {p.trial.title}: Stewart speed ratio {row['stewart_speed_ratio']:.4f}, min joint margin "
                          f"{row['min_margin_deg']:.1f} deg (computed)  {p.run.timeline.digest()}")
            if not args.r0:
                print("# WITHOUT --r0 the run starts at the SDK rest: not the robot's start")
        except (OSError, ValueError) as exc:
            print(f"--r2 / --r3: {exc}", file=sys.stderr)
            return 2
    elif args.r1 or args.c1:  # exactly what `robot-r1` / `robot-c1` would request (planned from --r0)
        from reachy_motion import robot_r1

        if args.stage != "R1" or args.motion or (args.r1 and args.c1) or (args.c1 and args.part):
            print("--r1 / --c1 go with --stage R1, alone and without --motion (--part only with --r1)", file=sys.stderr)
            return 2
        try:
            ref = robot_r1.load_r0(args.r0) if args.r0 else None
            if args.c1:
                if ref is None:
                    raise ValueError("--c1 needs --r0 (C1 holds the state that R0 read)")
                request = robot_r1.c1_request(ref, args.host)
                print(f"# C1: hold the state of R0 {args.r0} (sha256 {robot_r1.file_sha256(args.r0)}); nothing else is "
                      "streamed; torque on, hold, torque off")
            else:
                if args.part is None:
                    raise ValueError("--r1 needs --part ramp (approval 1/2: R1-0, the start-up and return ramps) or "
                                     "--part trial (approval 2/2: the R1 trial); each goes into its own file")
                plan = robot_r1.build(args.r1, args.rep, ref)
                if args.part == "ramp":
                    request = robot_r1.ramp_request(plan, args.host)
                    print("# APPROVAL 1/2 - " + robot_r1.ramp_summary(plan))
                    t = robot_r1.ramp_travel(plan.ramp)
                    print("# Stewart joint travel per joint (deg, computed): " + ", ".join(
                        f"{k} {v}" for k, v in t["per_joint_deg"].items()))
                else:
                    request = robot_r1.trial_request(plan, args.host)
                    row = plan.check.row()
                    print(f"# APPROVAL 2/2 - R1 trial {plan.trial.title}: Stewart speed ratio "
                          f"{row['stewart_speed_ratio']}, min joint margin {row['min_margin_deg']} deg (computed); "
                          "robot-r1 also needs the R1-0 ramp approval (--part ramp)")
                for prep in (plan.ramp, plan.back) if args.part == "ramp" else (plan.run,):
                    print(f"# {prep.timeline.name}:{prep.timeline.variant}  {prep.timeline.duration:.1f} s  "
                          f"{prep.timeline.digest()}")
                if not args.r0:
                    print("# WITHOUT --r0 the run starts at the SDK rest: not the robot's start")
        except (OSError, ValueError) as exc:
            print(f"--r1 / --c1: {exc}", file=sys.stderr)
            return 2
    else:
        specs = []
        for key in args.motion:
            name, _, variant = key.partition(":")
            specs.append(load_spec_file(motions_dir() / f"{name}.{variant or 'expressive'}.yaml"))
        request = ApprovalRequest(host=args.host, stage=args.stage, motions=tuple(s.key for s in specs),
                                  timeline_digests=tuple(plan(s).digest() for s in specs),
                                  reports=tuple(check_spec(s) for s in specs))
    template = approval_template(request)
    print("# Fill in and save this file YOURSELF (the assistant must not create approval files, CLAUDE.md rule 3).")
    print(json.dumps(template, indent=2, ensure_ascii=False, default=str))
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from reachy_motion.gui.app import main as gui_main
    except ImportError as exc:  # PySide6 missing
        print(f"the Motion Editor needs PySide6 (pip extra 'gui'): {exc}", file=sys.stderr)
        return 1
    return gui_main([sys.argv[0]], Path(args.dir) if args.dir else None)


def cmd_phase2(args: argparse.Namespace) -> int:
    """Regenerate the Phase 2 artifacts for every motion file (computed, no daemon)."""
    from reachy_motion.gui import plots
    from reachy_motion.gui.model import EditorModel, _write_rows
    from reachy_motion.library.compare import comparison_conditions
    from reachy_motion.library.registry import default_library
    from reachy_motion.sdk_env import environment_report

    out = Path(args.out)
    model = EditorModel(Path(args.dir) if args.dir else None)
    lib = default_library(model.dir)
    summary_rows: list[dict[str, Any]] = []
    written: list[str] = []
    md: list[str] = []
    for spec in lib:
        model.open(spec.name, spec.variant)
        b = model.analyze()
        d = out / "motions" / f"{spec.name}.{spec.variant}"
        d.mkdir(parents=True, exist_ok=True)
        _write_json(d / "policy_report.json", b.report.to_dict() if b.report else {"problems": b.problems})
        _write_rows(d / "executed_parameters.csv", b.segments)
        for fname, draw in (("motion.png", lambda f: plots.draw_motion(f, b)),
                            ("joints.png", lambda f: plots.draw_joints(f, b, model.policy))):
            fig = plots.new_figure()
            draw(fig)
            fig.savefig(d / fname, dpi=120)
        written += [str(d / x) for x in ("policy_report.json", "executed_parameters.csv", "motion.png", "joints.png")]
        m = b.report.metrics if b.report else {}
        summary_rows.append({
            "motion": spec.key, "draft": spec.draft, "duration_s": spec.duration,
            "head_keys": len(spec.head), "antenna_keys": len(spec.antennas), "body_keys": len(spec.body),
            "valence_expected": spec.valence, "arousal_expected": spec.arousal,
            "principles": "; ".join(p.principle.value for p in spec.principles),
            "passes_initial_policy": b.passes_policy,
            **{k: m.get(k) for k in ("min_stewart_margin_deg", "peak_stewart_speed_ratio", "peak_body_speed_ratio",
                                     "peak_antenna_speed_ratio", "max_abs_roll_deg", "max_abs_pitch_deg",
                                     "max_abs_rel_yaw_deg", "max_auto_body_shift_deg")},
            "violations": len(b.report.violations) if b.report else None,
            "warnings": "; ".join(w.code for w in b.report.warnings) if b.report else None,
            "neutral_rule_issues": "; ".join(i.code for i in b.conformance),
        })
        md.append(_spec_markdown(spec, b))
    comps = {}
    for name in sorted({s.name for s in lib}):
        try:
            e, n = lib.get(name, "expressive"), lib.get(name, "neutral")
        except KeyError:
            continue
        comps[name] = comparison_conditions(e, n)
        model.open(name, "expressive")
        be, bn = model.analyze(e), model.analyze(n)
        fig = plots.new_figure()
        plots.draw_compare(fig, be, bn)
        p = out / "comparison" / f"{name}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=120)
        written.append(str(p))
    _write_json(out / "comparison" / "comparison_conditions.json", comps)
    _write_rows(out / "motion_summary.csv", summary_rows)
    status_en, status_ko = _design_status(list(lib))
    _write_json(out / "manifest.json", {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python -m reachy_motion phase2 --out " + str(out), "data_kind": "computed",
        "note": f"{status_en}; planned values from the SDK's own interpolation / IK; policy = initial Phase 2 "
                "policy, not a mechanical-safety guarantee.",
        "motions_dir": str(model.dir), "files": written + [str(out / "motion_summary.csv")],
        "environment": environment_report()})
    MOTION_SPECS_MD.parent.mkdir(parents=True, exist_ok=True)
    header = ("# 모션 사양표 (자동 생성 — 직접 고치지 말 것)\n\n"
              f"생성: `python -m reachy_motion phase2` ({datetime.now():%Y-%m-%d %H:%M}). 원본: `motions/*.yaml`.\n"
              f"{status_ko} 실행 파라미터는 계획기(SDK 보간)로 계산한 값(computed)이고, "
              "V/A는 설계 목표(예상)다. 정책 판정은 초기 정책(관절 여유 10°, 무부하 속도 50%)이며 기구적 안전 보장이 아니다.\n\n")
    MOTION_SPECS_MD.write_text(header + "\n".join(md), encoding="utf-8")
    print(f"wrote {len(written)} figure/report files, {out / 'motion_summary.csv'}, {MOTION_SPECS_MD}")
    return 0


def _design_status(specs: list[Any]) -> tuple[str, str]:
    """(manifest note, MOTION_SPECS.md sentence) for the draft / signed-off state of ``specs``."""
    drafts = [s.key for s in specs if s.draft]
    if len(drafts) == len(specs):
        return ("Draft motions (to be designed and tuned by the student)",
                "모든 값은 **초안**이며 학생이 직접 설계·조정한다.")
    if not drafts:
        return ("Signed-off motions (draft: false, the student's final design)",
                "모든 모션은 학생이 확정한 설계다(`draft: false`).")
    return (f"Drafts: {', '.join(drafts)}; the other motions are signed off (draft: false)",
            f"초안(`draft: true`): {', '.join(drafts)}. 나머지는 학생이 확정한 설계다(`draft: false`).")


def _spec_markdown(spec: Any, b: Any) -> str:
    lines = [f"## `{spec.key}`{' (초안)' if spec.draft else ''}", "",
             f"- 의도: {spec.intent}", f"- 예상 V/A(설계 목표): valence {spec.valence:+.2f}, arousal {spec.arousal:+.2f}",
             f"- 길이 {spec.duration:.2f} s, 배율 amplitude ×{spec.amplitude_scale:g}, tempo ×{spec.tempo_scale:g}"
             + (f", 효과음 `{spec.sound.file}` @ {spec.sound.at_s:g} s(GUI가 로봇 스피커·노트북으로 재생, motions/sounds/)" if spec.sound else ", 효과음 없음"),
             "- 적용한 애니메이션 원리:"]
    lines += [f"  - **{p.principle.value}** — {p.how}" for p in spec.principles]
    lines += ["", "| 트랙 | # | 라벨 | 출발 s | 도착 s | 해제 s | duration s | pause s | 보간 | 목표값 | 공칭 피크 속도 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in b.segments:
        target = ", ".join(f"{k}={r[k]:.1f}" for k in ("x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg",
                                                       "right_deg", "left_deg") if k in r and isinstance(r[k], float))
        if r.get("track") == "body" and isinstance(r.get("yaw_deg"), float):
            target = f"yaw_deg={r['yaw_deg']:.1f}"
        speed = ", ".join(f"{label}={r[k]:.0f}" for k, label in (
            ("peak_rot_path_dps", "rot deg/s"), ("peak_trans_mms", "trans mm/s"), ("peak_antenna_dps", "ant deg/s"),
            ("peak_body_dps", "body deg/s")) if isinstance(r.get(k), float) and r[k] > 0)
        lines.append(f"| {r.get('track')} | {r.get('index')} | {r.get('label', '')} | {r.get('t_depart_s', 0):.2f} | "
                     f"{r.get('t_arrive_s', 0):.2f} | {r.get('t_release_s', 0):.2f} | {r.get('duration_s', 0):.2f} | "
                     f"{r.get('hold_s', 0):.2f} | {r.get('easing')} | {target} | {speed} |")
    if b.report is not None:
        m = b.report.metrics
        lines += ["", f"정책 판정: **{'통과' if b.report.passes_policy else '위반'}** — 최소 관절 여유 "
                      f"{m.get('min_stewart_margin_deg', float('nan')):.1f}°, Stewart 피크 속도 비 "
                      f"{m.get('peak_stewart_speed_ratio', float('nan')):.2f}, 안테나 피크 속도 비 "
                      f"{m.get('peak_antenna_speed_ratio', float('nan')):.2f} (computed)"]
        if b.report.warnings:
            lines.append("경고: " + "; ".join(f"{w.code}" for w in b.report.warnings))
    if b.conformance:
        lines.append("neutral 규칙: " + "; ".join(f"{i.code}" for i in b.conformance))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------- parser
def register(sub: Any) -> None:
    def motion_args(p: argparse.ArgumentParser, variant_choices: tuple[str, ...] = ("expressive", "neutral")) -> None:
        p.add_argument("name", nargs="?", default="aha_recommend", help="aha_recommend | boredom | success | failure")
        p.add_argument("--variant", default="expressive", choices=variant_choices)
        p.add_argument("--file", help="play/check this motion file instead of motions/<name>.<variant>.yaml")
        p.add_argument("--dir", help="motion folder (default: $REACHY_MOTION_DIR or <repo>/motions)")
        p.add_argument("--amplitude", type=float, default=1.0, help="extra amplitude scale on top of the file's")
        p.add_argument("--tempo", type=float, default=1.0, help="extra tempo scale on top of the file's")

    p = sub.add_parser("specs", help="list the motion files with a policy verdict (computed)")
    p.add_argument("--dir")
    p.set_defaults(func=cmd_specs)
    p = sub.add_parser("check", help="initial-policy report of a motion (computed)")
    motion_args(p, ("expressive", "neutral", "both"))
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("play", help="play a motion; default Mock backend (no hardware)")
    motion_args(p)
    p.add_argument("--backend", default="mock", choices=("mock", "sim", "robot"))
    p.add_argument("--out", default=str(PHASE2_OUT / "runs"))
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--armed", action="store_true", help="robot only (Phase 4)")
    p.add_argument("--approval", help="robot only: approval JSON written by the user (Phase 4)")
    p.add_argument("--host", default=ROBOT_HOST, help="robot only: daemon host named in the approval")
    p.set_defaults(func=cmd_play, once=True)
    p = sub.add_parser("sim-play", help="play a motion in the local MuJoCo viewer (127.0.0.1, simulated)")
    motion_args(p)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--headless", action="store_true", help="no viewer window")
    p.add_argument("--once", action="store_true", help="play once and exit (no prompt)")
    p.add_argument("--out", default=str(PHASE2_OUT / "sim_runs"))
    p.set_defaults(func=cmd_sim_play)
    p = sub.add_parser("sim-verify", help="play every motion file in local MuJoCo and write review tables (Phase 3)")
    p.add_argument("--dir", help="motion folder (default: $REACHY_MOTION_DIR or <repo>/motions)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--viewer", action="store_true", help="show the MuJoCo window while it runs")
    p.add_argument("--hold", type=float, default=1.0, help="seconds to hold at rest after each motion")
    p.add_argument("--out", default=str(Path("results") / "phase3_sim" / "review"))
    p.set_defaults(func=cmd_sim_verify)
    p = sub.add_parser("video", help="expressive-vs-neutral videos from an offline MuJoCo replay (simulated, WebP)")
    p.add_argument("--dir", action="append",
                   help="motion folder (repeatable); default: motions/ and every motions/candidates/*/")
    p.add_argument("--name", action="append", choices=("aha_recommend", "boredom", "success", "failure"),
                   help="only these motions (repeatable)")
    p.add_argument("--format", default="auto", choices=("auto", "mp4", "webp", "both"),
                   help="auto = MP4 (H.264) when PyAV is installed, else animated WebP")
    p.add_argument("--fps", type=int, default=25, choices=(10, 25, 50))
    p.add_argument("--size", default="560x420", help="size of each panel, WIDTHxHEIGHT (max 1280x720)")
    p.add_argument("--lead", type=float, default=0.5, help="seconds at the start pose before the motion")
    p.add_argument("--tail", type=float, default=1.5, help="seconds after the longer variant ends")
    p.add_argument("--scene", default="empty", choices=("empty", "minimal"))
    p.add_argument("--out", default=str(Path("results") / "phase3_sim" / "videos"))
    p.set_defaults(func=cmd_video)
    p = sub.add_parser("timing", help="measure the executor's real tick period / jitter on this PC (Mock)")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--rate", type=float, default=50.0)
    p.add_argument("--spin-us", type=int, default=0, help="busy-wait the last N microseconds of each tick")
    p.add_argument("--no-hires-timer", action="store_true", help="skip timeBeginPeriod(1)")
    p.add_argument("--out", default=str(PHASE2_OUT / "executor_timing"))
    p.set_defaults(func=cmd_timing)
    p = sub.add_parser("export", help="export a motion bundle (spec, report, parameters, CSV, PNG)")
    motion_args(p)
    p.add_argument("--out", default=str(PHASE2_OUT / "exports"))
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("approval-template", help="print the approval JSON a human fills in (Phase 4); writes nothing")
    p.add_argument("--stage", default="R1", choices=("R0", "R1", "R2", "R3"))
    p.add_argument("--host", default=ROBOT_HOST)
    p.add_argument("--motion", action="append", default=[], help="name:variant (repeatable)")
    p.add_argument("--r1", help="one R1 trial instead of motions: AXIS:AMPLITUDE (as robot-r1)")
    p.add_argument("--part", choices=("ramp", "trial"), help="with --r1: which of the two approvals - ramp = R1-0 "
                                                             "(start-up and return ramps), trial = R1 (the trial)")
    p.add_argument("--rep", type=int, default=1, help="with --r1")
    p.add_argument("--r0", help="with --r1 / --c1: the R0 report the run is planned from (as robot-r1 / robot-c1)")
    p.add_argument("--c1", action="store_true", help="the C1 step (torque on, hold, torque off) instead of motions")
    p.add_argument("--r2", help="R2 confirmed motions, comma separated (as robot-r2): --part trial = ONE approval for "
                                "the set, --part ramp = their R1-0 ramps (--stage R1)")
    p.add_argument("--variants", choices=("expressive", "neutral", "both"), default="both", help="with --r2")
    p.add_argument("--r3", help="one R3 step A:D, e.g. 20:0.6 (as robot-r3)")
    p.add_argument("--antennas", choices=("designed", "hold"), default="designed", help="with --r2 / --r3 (as robot-r2)")
    p.set_defaults(func=cmd_approval_template)
    p = sub.add_parser("gui", help="open the Reachy HW4 Motion Lab (PySide6): Motion / Experiment / Results, Mock and local MuJoCo only")
    p.add_argument("--dir")
    p.set_defaults(func=cmd_gui)
    p = sub.add_parser("phase2", help="regenerate Phase 2 artifacts (computed)")
    p.add_argument("--out", default=str(PHASE2_OUT))
    p.add_argument("--dir")
    p.set_defaults(func=cmd_phase2)
