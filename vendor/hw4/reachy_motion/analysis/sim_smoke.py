"""Phase 1 sim smoke check: does goto_target behave in MuJoCo the way the source analysis predicts?

data_kind = "simulated". A sanity check of the command flow - NOT the Phase 3 tracking study:
one goto per interpolation method, one amplitude, one duration, n = 1, on this Windows PC.

What is compared:
  reference  = IK of the SDK's own GotoMove, built exactly as Backend.goto_target does
               (start = present head pose / body yaw measured just before the call,
               daemon/backend/abstract.py:966-977), evaluated on a dense grid (computed);
  measured   = joint positions pushed by the sim daemon (JointPositionsMsg, one per 50 Hz
               control tick, daemon/backend/mujoco/backend.py:230-277), timestamped on arrival
               with perf_counter. The client keeps only the latest message
               (io/ws_client.py:69, 185-186); we detect new messages by object identity of that
               private attribute - acceptable for a diagnostic, not for library code.
Timing caveat (this PC): asyncio timers follow time.monotonic (15.625 ms). An idle loop ticks at
~60 Hz; a loop woken by other threads (the daemon broadcasts state to every /ws/sdk client each
control tick) ticks irregularly at ~100-145 Hz (stub results). The running daemon's play_move rate
was not measured. The robot's Linux daemon is not affected.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from reachy_mini.motion.goto import GotoMove
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import InterpolationTechnique
from reachy_mini.utils.rotation import Rotation as R

from reachy_motion.analysis.goto_replay import JOINT_NAMES, ik_joints, new_kinematics
from reachy_motion.analysis.interpolation import METHODS
from reachy_motion.analysis.metrics import estimate_lag, peak_error, rmse
from reachy_motion.backends.sim_daemon import (
    DEFAULT_PORT,
    SIM_HOST,
    SIM_ROBOT_NAME,
    SimDaemon,
    assert_local_sim,
    assert_motion_client_is_local_sim,
    get_status,
    port_answers,
)

NEUTRAL_ANTENNAS = [-0.1745, 0.1745]  # SDK INIT_ANTENNAS_JOINT_POSITIONS (reachy_mini.py:67)
SETTLE_WINDOW_S = 0.3  # start a run only after body + Stewart moved < SETTLE_TOL_DEG over this window
SETTLE_TOL_DEG = 0.03
SETTLE_TIMEOUT_S = 8.0
CSV_PRE_START_S = 1.0  # samples kept before t0 in samples_<method>.csv


class PushSampler(threading.Thread):
    """Records every JointPositionsMsg / HeadPoseMsg the client receives (arrival time)."""

    def __init__(self, client: object, poll_s: float = 0.0005):
        super().__init__(daemon=True)
        self.client = client
        self.poll_s = poll_s
        self.stop_event = threading.Event()
        self.joints: list[tuple[float, list[float], list[float]]] = []
        self.poses: list[tuple[float, np.ndarray]] = []

    def run(self) -> None:
        last_j = last_p = None
        while not self.stop_event.is_set():
            j = getattr(self.client, "_last_joint_positions", None)
            p = getattr(self.client, "_last_head_pose", None)
            now = time.perf_counter()
            if j is not None and j is not last_j:
                self.joints.append((now, list(j.head_joint_positions), list(j.antennas_joint_positions)))
                last_j = j
            if p is not None and p is not last_p:
                self.poses.append((now, np.array(p.head_pose, dtype=float).reshape(4, 4)))
                last_p = p
            time.sleep(self.poll_s)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=2.0)


def wait_until_settled(sampler: PushSampler, window_s: float = SETTLE_WINDOW_S, tol_deg: float = SETTLE_TOL_DEG,
                       timeout_s: float = SETTLE_TIMEOUT_S) -> dict:
    """Block until body + Stewart joints moved less than tol_deg over the last window_s, or time out."""
    t_start = time.perf_counter()
    spread = float("nan")
    while True:
        now = time.perf_counter()
        recent = [s for s in list(sampler.joints) if s[0] >= now - window_s]
        if len(recent) >= 5 and recent[0][0] <= now - 0.8 * window_s:
            spread = float(np.rad2deg(np.ptp(np.array([s[1] for s in recent]), axis=0).max()))
            if spread < tol_deg:
                return {"settled": True, "settle_wait_s": now - t_start, "settle_spread_deg": spread}
        if now - t_start > timeout_s:
            return {"settled": False, "settle_wait_s": now - t_start, "settle_spread_deg": spread}
        time.sleep(0.02)


@dataclass
class SmokeRun:
    method: InterpolationTechnique
    t0: float
    t_return: float
    start_pose: np.ndarray
    start_body_yaw: float
    samples: PushSampler
    start_joints: list[float] = field(default_factory=list)
    settle_info: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


def _pitch_deg(pose: np.ndarray) -> float:
    u, _, vt = np.linalg.svd(pose[:3, :3])  # daemon matrices are not exactly orthonormal
    return float(np.rad2deg(R.from_matrix(u @ vt).as_euler("xyz")[1]))


def _reference(run: SmokeRun, target: np.ndarray, duration: float, t: np.ndarray) -> np.ndarray:
    move = GotoMove(
        start_head_pose=run.start_pose,
        target_head_pose=target,
        start_antennas=np.array(NEUTRAL_ANTENNAS),
        target_antennas=None,
        start_body_yaw=run.start_body_yaw,
        target_body_yaw=0.0,
        duration=duration,
        method=run.method,
    )
    kin = new_kinematics()
    out = []
    for ti in t:
        head, _ant, body_yaw = move.evaluate(float(np.clip(ti, 0.0, duration)))
        out.append(ik_joints(kin, head, float(body_yaw)))
    return np.array(out)


def analyse(run: SmokeRun, target: np.ndarray, target_pitch_deg: float, duration: float,
            settle: float) -> dict:
    t = np.array([s[0] for s in run.samples.joints]) - run.t0
    q = np.array([s[1] for s in run.samples.joints])
    window = (t >= -0.05) & (t <= duration + settle)
    t, q = t[window], q[window]
    t_ref = np.linspace(-0.05, duration + settle, int((duration + settle + 0.05) * 1000) + 1)
    q_ref = _reference(run, target, duration, t_ref)
    j = int(np.argmax(np.ptp(q_ref[:, 1:], axis=0))) + 1
    moving = (t > 0.0) & (t < duration)
    dt_push = np.diff(t[moving]) if moving.sum() > 2 else np.array([np.nan])
    lag = estimate_lag(t_ref, q_ref[:, j], t, q[:, j], max_lag=0.25, step=0.001)
    q_goal = ik_joints(new_kinematics(), target, 0.0)
    tp = np.array([s[0] for s in run.samples.poses]) - run.t0
    pitch = np.array([_pitch_deg(s[1]) for s in run.samples.poses])
    in_move = (tp >= 0.0) & (tp <= duration + settle)
    final = tp >= duration + settle - 0.1
    return {
        "method": run.method.value,
        "joint": JOINT_NAMES[j],
        "settled_before_start": bool(run.settle_info.get("settled", False)),
        "settle_wait_s": float(run.settle_info.get("settle_wait_s", float("nan"))),
        "settle_spread_deg": float(run.settle_info.get("settle_spread_deg", float("nan"))),
        "pitch_start_deg": _pitch_deg(run.start_pose),
        "n_joint_msgs_in_move": int(moving.sum()),
        "push_interval_median_ms": float(np.nanmedian(dt_push) * 1000),
        "push_interval_p95_ms": float(np.nanpercentile(dt_push, 95) * 1000),
        "goto_call_blocked_s": run.t_return - run.t0,
        "lag_vs_reference_s": lag,
        "rmse_deg_no_shift": float(np.rad2deg(rmse(t_ref, q_ref[:, j], t, q[:, j]))),
        "peak_err_deg_no_shift": float(np.rad2deg(peak_error(t_ref, q_ref[:, j], t, q[:, j]))),
        "final_joint_err_deg": float(np.rad2deg(np.mean(q[t >= duration + settle - 0.1, j]) - q_goal[j])),
        "pitch_target_deg": target_pitch_deg,
        "pitch_max_deg": float(pitch[in_move].max()) if in_move.any() else float("nan"),
        "pitch_min_deg": float(pitch[in_move].min()) if in_move.any() else float("nan"),
        "pitch_final_deg": float(np.mean(pitch[final])) if final.any() else float("nan"),
    }


def run_sim_smoke(out_dir: Path, port: int = DEFAULT_PORT, amplitude_deg: float = 20.0,
                  duration: float = 1.0, settle: float = 0.6) -> dict:
    """Spawn the sim daemon, run one goto per method, write CSV/figure/manifest to out_dir."""
    from reachy_mini import ReachyMini

    out_dir.mkdir(parents=True, exist_ok=True)
    target = create_head_pose(pitch=amplitude_deg, degrees=True)
    neutral = np.eye(4)
    runs: list[SmokeRun] = []
    with SimDaemon(port=port, log_path=out_dir / "daemon.log") as _daemon:
        status = assert_local_sim(get_status(port))
        # For a non-default robot_name, localhost_only always dials "localhost" and ignores `host`
        # (reachy_mini/reachy_mini.py:471-476); "localhost" resolves to ::1 first on this PC and the
        # constructor already sends SetAutomaticBodyYawCmd (reachy_mini.py:161). Our daemon binds only
        # 127.0.0.1, so refuse if anything listens on [::1]:port. host=SIM_HOST is kept so that any
        # fallback would also be loopback, never the default "reachy-mini.local".
        if port_answers("::1", port):
            raise RuntimeError(f"something listens on [::1]:{port}; refusing to connect")
        mini = ReachyMini(robot_name=SIM_ROBOT_NAME, host=SIM_HOST, port=port, connection_mode="localhost_only",
                          spawn_daemon=False, media_backend="no_media", timeout=10.0)
        try:
            assert_local_sim(get_status(port))  # REST re-check right before the first motion command
            client_status = assert_motion_client_is_local_sim(mini)  # same check on the motion socket
            mini.goto_target(neutral, antennas=NEUTRAL_ANTENNAS, duration=2.0, body_yaw=0.0)
            time.sleep(settle)
            for method in METHODS:
                mini.goto_target(neutral, duration=1.0, body_yaw=0.0)
                sampler = PushSampler(mini.client)
                sampler.start()
                settle_info = wait_until_settled(sampler)
                # read the start state as late as possible: the daemon starts from the pose it
                # measures when the request arrives (daemon/backend/abstract.py:966-977)
                start_pose = np.asarray(mini.get_current_head_pose(), dtype=float)
                start_joints = [float(v) for v in mini.get_current_joint_positions()[0]]
                t0 = time.perf_counter()
                mini.goto_target(target, duration=duration, method=method, body_yaw=0.0)
                t_ret = time.perf_counter()
                time.sleep(settle)
                sampler.stop()
                runs.append(SmokeRun(method, t0, t_ret, start_pose, start_joints[0], sampler,
                                     start_joints=start_joints, settle_info=settle_info))
            mini.goto_target(neutral, duration=1.0, body_yaw=0.0)
        finally:
            mini.client.disconnect()

    summaries = [analyse(r, target, amplitude_deg, duration, settle) for r in runs]
    for r, s in zip(runs, summaries):
        r.summary = s
        with (out_dir / f"samples_{r.method.value}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_rel_s"] + [f"{n}_rad" for n in JOINT_NAMES] + ["antenna_right_rad", "antenna_left_rad"])
            for t, head, ant in r.samples.joints:
                if t - r.t0 >= -CSV_PRE_START_S:
                    w.writerow([f"{t - r.t0:.6f}"] + [f"{v:.6f}" for v in head] + [f"{v:.6f}" for v in ant])
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        for s in summaries:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in s.items()})
    _figure(runs, target, amplitude_deg, duration, settle, out_dir / "sim_smoke_goto.png")
    from reachy_motion.sdk_env import environment_report

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_kind": "simulated",
        "note": "Phase 1 smoke check of the goto_target command flow in MuJoCo (n=1 per method). "
                "Not the Phase 3 tracking study.",
        "daemon_status_at_start": status,
        "daemon_status_on_motion_socket": client_status,
        "case": {"pitch_deg": amplitude_deg, "duration_s": duration, "observe_after_s": settle,
                 "body_yaw": 0.0,
                 "start": (f"goto neutral 1 s, then wait until body + Stewart moved < {SETTLE_TOL_DEG} deg over "
                           f"{SETTLE_WINDOW_S} s (timeout {SETTLE_TIMEOUT_S} s); start pose read just before "
                           "goto_target and used for the reference")},
        "start_joints_rad": {r.method.value: r.start_joints for r in runs},
        "timing_caveat": "Windows asyncio timers follow time.monotonic (15.625 ms): idle loop ~60 Hz, loop "
                         "woken by per-tick WS broadcasts ~100-145 Hz irregular (stub results); the running "
                         "daemon's play_move rate was not measured. MuJoCo ran slower than real time "
                         "(push interval ~25 ms for 20 ms of sim time).",
        "environment": environment_report(),
        "summaries": summaries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


API_CHECK_QUERIES: tuple[tuple[str, str], ...] = (
    ("daemon_status", "/api/daemon/status"),
    ("kinematics_info", "/api/kinematics/info"),
    ("state_full_with_target_joints",
     "/api/state/full?with_head_joints=true&with_target_head_joints=true"
     "&with_target_body_yaw=true&with_target_antenna_positions=true"),
    ("state_full_with_target_head_pose", "/api/state/full?with_target_head_pose=true"),
)


def run_api_check(out_dir: Path, port: int = DEFAULT_PORT) -> dict:
    """Read-only GETs against our own sim daemon; no client connection, no motion command.

    Records what the 1.11.0 REST API really returns (e.g. whether /api/state/full exposes the
    commanded targets). Saves api_checks.json (data_kind = simulated) and the daemon log.
    """
    import urllib.error
    import urllib.request

    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    with SimDaemon(port=port, log_path=out_dir / "api_check_daemon.log") as _daemon:
        assert_local_sim(get_status(port))
        for name, path in API_CHECK_QUERIES:
            url = f"http://{SIM_HOST}:{port}{path}"
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                results[name] = {"request": f"GET {path}", "http_status": resp.status,
                                 "response_keys": sorted(body.keys()), "response": body}
            except urllib.error.HTTPError as e:
                results[name] = {"request": f"GET {path}", "http_status": e.code,
                                 "error_body": e.read().decode("utf-8", "replace")[:500]}
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                results[name] = {"request": f"GET {path}", "http_status": None,
                                 "transport_error": f"{type(e).__name__}: {e}"}
    from reachy_motion.sdk_env import environment_report

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_kind": "simulated",
        "note": "Read-only REST GETs against the local MuJoCo daemon spawned by SimDaemon (127.0.0.1, "
                "robot_name=hw4_sim). No ReachyMini client, no motion command.",
        "environment": environment_report(),
        "checks": results,
    }
    (out_dir / "api_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _figure(runs: list[SmokeRun], target: np.ndarray, amplitude_deg: float, duration: float, settle: float,
            path: Path) -> None:
    from reachy_motion.analysis import figures as F
    from reachy_motion.sdk_env import installed_version

    F._style()
    plt = F.plt
    fig, axes = plt.subplots(1, len(runs), figsize=(12.0, 3.8), sharey=True, constrained_layout=True)
    t_ref = np.linspace(-0.05, duration + settle, 1200)
    j = int(np.argmax(np.ptp(_reference(runs[0], target, duration, t_ref)[:, 1:], axis=0))) + 1
    for ax, r in zip(np.atleast_1d(axes), runs):
        t = np.array([s[0] for s in r.samples.joints]) - r.t0
        q = np.array([s[1] for s in r.samples.joints])
        q_ref = _reference(r, target, duration, t_ref)
        ax.plot(t_ref, np.rad2deg(q_ref[:, j]), color=F.INK_2, linewidth=F.LW, label="reference (computed)")
        ax.plot(t, np.rad2deg(q[:, j]), color=F.METHOD_COLOR[r.method], linewidth=F.LW,
                linestyle=F.METHOD_DASH[r.method], marker="o", markersize=2.2, label="MuJoCo (simulated)")
        ax.axvline(duration, color=F.AXIS, linewidth=0.8)
        ax.set_xlim(-0.05, duration + settle)
        ax.set_title(f"{r.method.value}  (lag {r.summary.get('lag_vs_reference_s', float('nan'))*1000:.0f} ms)")
        ax.set_xlabel("time since goto_target call (s)")
    np.atleast_1d(axes)[0].set_ylabel(f"{JOINT_NAMES[j]} (deg)")
    np.atleast_1d(axes)[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Sim smoke check: goto_target pitch → {amplitude_deg:g}° from the measured start pose "
                 f"(D = {duration:g} s), most-moving Stewart motor",
                 color=F.INK, fontsize=11, fontweight="bold", x=0.01, ha="left")
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, 0.04, 1.0, 0.96))
    fig.text(0.01, 0.006, f"simulated in MuJoCo {installed_version('mujoco')} via reachy_mini "
             f"{installed_version('reachy-mini')} daemon (Windows) - not robot data",
             color=F.MUTED, fontsize=7.5, ha="left", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
