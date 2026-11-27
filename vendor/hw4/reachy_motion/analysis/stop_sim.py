"""MuJoCo stop / hold experiments (Phase 4 preparation; data_kind simulated, never the robot).

Six scenarios, each from REST on a 20-degree pitch nod (0.5 s rest, 0.6 s up, 0.8 s hold, 0.6 s back, 0.5 s rest,
MIN_JERK), with the fault applied after sample ``FAULT_K`` (t = 0.8 s, the fastest instant of the move):

* ``S1_cancel``               cancel -> S1, slow down along the path, hold
* ``S2_send_failure``         ``set_target`` raises (``backends.faults.FaultInjector``) -> S2
* ``S2_slow_send``            one ``set_target`` takes 150 ms -> S2 after it
* ``S2_measurement_failure``  ``present()`` raises -> S2 once no fresh state for 0.2 s
* ``S2_measurement_stall``    ``present()`` repeats its last answer -> S2 the same way
* ``S2_link_loss``            the SDK client's WebSocket is closed (``mini.client.disconnect()``): the client then
                              drops every send without an error (io/ws_client.py:135-160: ``_is_alive`` stays True,
                              ``_ws`` is None) and receives nothing -> only the measurement watchdog notices (S2)

The head is observed independently of the motion client by a read-only REST poller (``DaemonClient``:
``/api/state/full?with_head_joints=true`` on 127.0.0.1, every 20 ms), so the hold is visible after a measurement
fault or a link loss too. Metrics per scenario (degrees, the Stewart joint that moves most): the observed speed when
the last command was sent and later, and the gap between the last commanded target (IK of the last command, as the
daemon computes it) and the observed joints at the stop, 0.5 s and 1.0 s later.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from reachy_motion.analysis.goto_replay import ik_joints, new_kinematics
from reachy_motion.library.pose import REST, Pose6, RobotState, pose_from_matrix
from reachy_motion.library.spec import Easing, HeadKey, MotionSpec, Principle, PrincipleUse

DATA_KIND = "simulated"
FAULT_K = 40
SCENARIOS: tuple[str, ...] = ("S1_cancel", "S2_send_failure", "S2_slow_send", "S2_measurement_failure",
                              "S2_measurement_stall", "S2_link_loss")
NAMES = ("body_yaw",) + tuple(f"stewart_{i}" for i in range(1, 7))


def nod_spec() -> MotionSpec:
    return MotionSpec(name="success", variant="expressive", intent="stop experiment", valence=0.0, arousal=0.0,
                      principles=(PrincipleUse(Principle.TIMING, "test"),),
                      head=(HeadKey(0.5, Pose6(), Easing.MIN_JERK, 0.0, "rest"),
                            HeadKey(0.6, Pose6.from_deg_mm(pitch_deg=20), Easing.MIN_JERK, 0.8, "up"),
                            HeadKey(0.6, Pose6(), Easing.MIN_JERK, 0.5, "back")), draft=False)


class RestObserver:
    """Polls the local sim daemon over REST (GET only) in a thread: (perf_counter_ns, 7 joints in rad)."""

    def __init__(self, port: int, period_s: float = 0.02) -> None:
        from reachy_motion.backends.daemon_client import DaemonClient

        self.client = DaemonClient("127.0.0.1", port=port, timeout_s=0.5)
        self.period_s = period_s
        self.samples: list[tuple[int, np.ndarray]] = []
        self.errors = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "RestObserver":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            t = time.perf_counter_ns()
            try:
                q = np.asarray(self.client.present_head_joints(), dtype=float)
                with self._lock:
                    self.samples.append((t, q))
            except Exception:  # noqa: BLE001 - counted; the daemon may be busy or already stopping
                self.errors += 1
            time.sleep(self.period_s)

    def window(self, t0_ns: int, t1_ns: int) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            s = [(t, q) for t, q in self.samples if t0_ns <= t <= t1_ns]
        if not s:
            return np.zeros(0), np.zeros((0, 7))
        return np.array([t for t, _ in s], dtype=float), np.array([q for _, q in s])


def _interp(t: np.ndarray, x: np.ndarray, at: float) -> float:
    return float(np.interp(at, t, x)) if t.size else math.nan


def stop_metrics(cmd_t: np.ndarray, cmd_q: np.ndarray, obs_t: np.ndarray, obs_q: np.ndarray, t_fault: int,
                 dom: int, confirmed: int | None = None) -> dict[str, Any]:
    """Metrics in degrees / deg/s (module docstring); times in ns on the PC clock. ``confirmed`` = index of the last
    command known to have reached the daemon (the executor's ``last_confirmed_k`` after a lost measurement)."""
    t_last = float(cmd_t[-1])
    q_last = cmd_q[-1]
    deg = 180.0 / math.pi
    out: dict[str, Any] = {"dominant_joint": NAMES[dom], "fault_to_last_command_ms": (t_last - t_fault) / 1e6}
    if obs_t.size < 5:
        out["observed_samples"] = int(obs_t.size)
        return out
    v = np.gradient(obs_q[:, dom], obs_t / 1e9) * deg
    out["observed_samples"] = int(obs_t.size)
    out["observed_speed_at_last_command_dps"] = abs(_interp(obs_t, v, t_last))
    for label, dt in (("0p5s", 0.5), ("1s", 1.0)):
        out[f"observed_speed_{label}_after_dps"] = abs(_interp(obs_t, v, t_last + dt * 1e9))
    late = (obs_t >= t_last + 0.3e9) & (obs_t <= t_last + 1.0e9)
    out["max_observed_speed_0p3_to_1s_after_dps"] = float(np.max(np.abs(v[late]))) if late.any() else math.nan
    after = obs_t >= t_last
    above = np.nonzero(after & (np.abs(v) >= 1.0))[0]
    out["observed_below_1dps_after_s"] = (float(obs_t[above[-1] + 1] - t_last) / 1e9
                                          if above.size and above[-1] + 1 < obs_t.size else 0.0)
    targets = [("last_target", q_last)]
    if confirmed is not None and confirmed != len(cmd_q) - 1:
        targets.append(("last_confirmed_target", cmd_q[confirmed]))
    for tag, qt in targets:
        for label, dt in (("at_stop", 0.0), ("0p5s", 0.5), ("1s", 1.0)):
            q = np.array([_interp(obs_t, obs_q[:, j], t_last + dt * 1e9) for j in range(7)])
            out[f"gap_to_{tag}_{label}_deg"] = float(np.max(np.abs(q[1:] - qt[1:]))) * deg
    q0 = _interp(obs_t, obs_q[:, dom], t_last)
    q1 = _interp(obs_t, obs_q[:, dom], t_last + 1.0e9)
    out["observed_travel_after_stop_deg"] = abs(q1 - q0) * deg
    cv = np.gradient(cmd_q[:, dom], cmd_t / 1e9) * deg
    out["commanded_speed_at_fault_dps"] = abs(_interp(cmd_t, cv, float(t_fault)))
    return out


def plot_scenario(ax_pos: Any, ax_vel: Any, name: str, cmd_t: np.ndarray, cmd_q: np.ndarray, obs_t: np.ndarray,
                  obs_q: np.ndarray, t_start: int, t_fault: int, dom: int, stop_kind: str | None) -> None:
    deg = 180.0 / math.pi
    ct, ot = (cmd_t - t_start) / 1e9, (obs_t - t_start) / 1e9
    ax_pos.plot(ct, cmd_q[:, dom] * deg, ls="--", lw=1.5, color="#8a8f98", label="commanded (IK of each command)")
    if ot.size:
        ax_pos.plot(ot, obs_q[:, dom] * deg, lw=1.8, color="#2a6fdb", label="observed (REST, 50 Hz)")
    if ct.size > 1:
        ax_vel.plot(ct, np.gradient(cmd_q[:, dom], ct) * deg, ls="--", lw=1.5, color="#8a8f98", label="commanded")
    if ot.size > 4:
        ax_vel.plot(ot, np.gradient(obs_q[:, dom], ot) * deg, lw=1.8, color="#2a6fdb", label="observed")
    for ax in (ax_pos, ax_vel):
        ax.axvline((t_fault - t_start) / 1e9, color="#d9480f", lw=1.2, label="fault / cancel")
        ax.axvline(ct[-1], color="#222222", lw=1.0, ls=":", label="last command")
        ax.grid(True, color="#e3e5e8", lw=0.6)
        ax.set_axisbelow(True)
    ax_pos.set_ylabel(f"{NAMES[dom]} (deg)")
    ax_vel.set_ylabel("velocity (deg/s)")
    ax_vel.set_xlabel("time from run start (s)")
    ax_pos.set_title(f"{name} -> {stop_kind}   (simulated, MuJoCo - not the robot)", fontsize=10)


def run_scenarios(backend: Any, port: int, out: Path, names: tuple[str, ...] = SCENARIOS) -> dict[str, Any]:
    """Run the scenarios in ``names`` on a connected sim backend (``SdkBackend("sim")``); writes CSV + PNG + JSON."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from reachy_motion.backends.faults import FaultInjector
    from reachy_motion.executor.executor import Executor, StopPolicy
    from reachy_motion.safety.checker import analyze_timeline, prepare
    from reachy_motion.safety.startup import prepare_startup_ramp
    from reachy_motion.sim_review import wait_still

    if backend.capabilities.real_hardware or backend.data_kind != DATA_KIND:
        raise ValueError("stop experiments run in MuJoCo only")
    out.mkdir(parents=True, exist_ok=True)
    prep = prepare(nod_spec())
    if not prep.runnable:
        raise ValueError("the test nod is outside the initial policy")
    plan_q = np.asarray(analyze_timeline(prep.timeline).joints, dtype=float)
    dom = 1 + int(np.argmax(np.ptp(plan_q[:, 1:], axis=0)))
    kin = new_kinematics()
    obs = RestObserver(port).start()
    results: dict[str, Any] = {}
    fig, axes = plt.subplots(len(names), 2, figsize=(13, 2.6 * len(names)), squeeze=False)
    try:
        for row, name in enumerate(names):
            p = backend.present()
            start = RobotState(head=pose_from_matrix(p.head), antennas=tuple(p.antennas), body_yaw=float(p.body_yaw))
            ramp = prepare_startup_ramp(start, REST)
            if ramp.runnable:
                Executor(backend).run(ramp)
            wait_still(backend, window_s=0.5, tol_deg=0.05, timeout_s=10.0)
            fi = FaultInjector(backend)
            cancel = threading.Event()
            marks: dict[str, int] = {}

            def fault(k: int, rec: Any, name: str = name, fi: Any = fi, cancel: threading.Event = cancel) -> None:
                if k != FAULT_K or "fault" in marks:
                    return
                marks["fault"] = time.perf_counter_ns()
                if name == "S1_cancel":
                    cancel.set()
                elif name == "S2_send_failure":
                    fi.fail_send = True
                elif name == "S2_slow_send":
                    fi.send_delay_s = 0.15
                elif name == "S2_measurement_failure":
                    fi.fail_present = True
                elif name == "S2_measurement_stall":
                    fi.freeze_present = True
                elif name == "S2_link_loss":
                    backend._mini.client.disconnect()  # the SDK client: no error, sends dropped from now on

            t_start = time.perf_counter_ns()
            result = Executor(fi).run(prep, cancel=cancel, on_tick=fault, record_present=True, stop=StopPolicy())
            fi.send_delay_s, fi.fail_send = 0.0, False
            time.sleep(1.5)  # watch the hold
            obs_t, obs_q = obs.window(t_start - 200_000_000, time.perf_counter_ns())
            cmd_t = np.array([c.record.t_ns for c in result.commands], dtype=float)
            cmd_q = np.array([ik_joints(kin, np.asarray(c.record.head), float(c.record.body_yaw))
                              for c in result.commands])
            known = (result.stop_detail or {}).get("known", {})
            confirmed = None
            if known.get("last_confirmed_k") is not None:
                ks = [c.k for c in result.commands]
                confirmed = max(i for i, k in enumerate(ks) if k <= known["last_confirmed_k"])
            m = stop_metrics(cmd_t, cmd_q, obs_t, obs_q, marks.get("fault", int(cmd_t[-1])), dom, confirmed)
            results[name] = {"stop_kind": result.stop_kind, "stop_reason": result.stop_reason,
                             "stop_commands": result.stop_commands, "commands": len(result.commands),
                             "completed": result.completed, "end_state_known": result.end_state_known,
                             "stop_detail": result.stop_detail, "metrics": m}
            np.savetxt(out / f"{name}_commands.csv", np.column_stack([(cmd_t - t_start) / 1e9, np.degrees(cmd_q)]),
                       delimiter=",", header="t_s," + ",".join(f"{n}_deg" for n in NAMES), comments="")
            np.savetxt(out / f"{name}_observed.csv", np.column_stack([(obs_t - t_start) / 1e9, np.degrees(obs_q)]),
                       delimiter=",", header="t_s," + ",".join(f"{n}_deg" for n in NAMES), comments="")
            f1, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
            plot_scenario(a1, a2, name, cmd_t, cmd_q, obs_t, obs_q, t_start, marks.get("fault", int(cmd_t[-1])),
                          dom, result.stop_kind)
            a1.legend(fontsize=8, loc="upper right")
            f1.tight_layout()
            f1.savefig(out / f"{name}.png", dpi=120)
            plt.close(f1)
            plot_scenario(axes[row][0], axes[row][1], name, cmd_t, cmd_q, obs_t, obs_q, t_start,
                          marks.get("fault", int(cmd_t[-1])), dom, result.stop_kind)
            if name == "S2_link_loss":
                break  # the motion client is gone; the caller closes the backend
        axes[0][0].legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(out / "overview.png", dpi=110)
    finally:
        plt.close(fig)
        obs.stop()
    summary = {"data_kind": DATA_KIND, "note": "local MuJoCo daemon (127.0.0.1); observer = REST GET "
               "/api/state/full?with_head_joints=true every 20 ms; not the robot", "fault_after_sample": FAULT_K,
               "observer_errors": obs.errors, "scenarios": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
