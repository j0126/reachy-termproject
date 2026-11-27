"""HW4 MuJoCo 시뮬레이터로 키오스크의 실제 SDK 모션 경로를 확인한다 (로봇 없이, 노트북에서).

HW4 venv의 `reachy-mini-daemon --sim`(MuJoCo 3.3.0)을 127.0.0.1에 띄우고, 키오스크와 같은 코드로 붙는다:
  RobotLink(localhost_only, robot_name=hw4_sim, no_media) → prepare_motion → wake_up
  → SharedMiniBackend + MotionPlayer (시선 레이어 + HW4 감정 모션)
HW4 안전 인터록(assert_local_sim, assert_motion_client_is_local_sim)을 그대로 써서 시뮬레이터에만 명령이 간다.
카메라·마이크는 시뮬레이터에 없으므로(no_media) 모션 경로만 본다.

  1 connect  daemon 시작, 연결, 준비(워블링·트래킹 끄기·모터), wake_up (MuJoCo는 잠든 자세에서 시작)
  2 gaze     화면·슬롯·손님(좌/우 극단) 시선: 명령 vs daemon 측정 머리 자세 오차, 몸통 yaw, 명령→측정 지연
  3 posture  자세·끄덕임·안테나 흔들기 → 측정 안테나·머리
  4 motions  감정 모션 4종 × expressive/neutral: 완료, 라벨 시각, 틱 지연, 끝 자세 오차
  5 preempt  boredom 도중 success (선점) → 명령 연속성 (한 틱 최대 변화)
  6 rest     stop(blend_to_rest) → REST 도착
  공통: 측정 관절로 계산한 Stewart 관절 속도·여유 (HW4 정책과 비교)

실행:
  .venv\\Scripts\\python -X utf8 tools\\sim_check.py            # headless
  .venv\\Scripts\\python -X utf8 tools\\sim_check.py --viewer   # MuJoCo 창으로 보기
결과: logs/sim_check_<시각>.json + 화면 요약
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reachy_cafe.config import load_config  # noqa: E402
from reachy_cafe.robot.connection import RobotLink  # noqa: E402
from reachy_cafe.robot.hwpaths import _resolve, ensure_hw_paths  # noqa: E402
from robot_measure import Recorder, direction_error_deg, gaze_lag  # noqa: E402  (tools/robot_measure.py)

MOTION_NAMES = ("aha_recommend", "boredom", "success", "failure", "greeting")  # greeting: 카페 모션 (motions/)


def deg(x: float) -> float:
    return round(math.degrees(x), 2)


class SimCheck:
    def __init__(self, cfg: dict[str, Any], args: argparse.Namespace):
        from reachy_motion.backends.sim_daemon import SIM_HOST, SIM_ROBOT_NAME, SimDaemon

        self.cfg = cfg
        self.args = args
        log_path = ROOT / "logs" / "sim_daemon.log"
        if importlib.util.find_spec("mujoco") is not None:
            # 이 venv에 MuJoCo가 있으면 (설치 때 sim 포함) HW4 SimDaemon 그대로: 이 venv의 reachy-mini-daemon,
            # 맥에서 --viewer면 mjpython (HW4 sim_daemon.py command)
            self.daemon = SimDaemon(port=args.port, log_path=log_path, headless=not args.viewer)
        else:
            hw4 = _resolve(cfg["hw"]["reachy_hw4"])
            exe = next((c for c in (hw4 / ".venv" / "Scripts" / "reachy-mini-daemon.exe",
                                    hw4 / ".venv" / "bin" / "reachy-mini-daemon") if c.exists()), None)
            if exe is None:
                raise FileNotFoundError("MuJoCo가 없습니다: 설치에서 sim을 포함하거나 (uv pip install -e \".[sim]\"), "
                                        "HW4 venv(../reachy_hw4/.venv)에 sim을 설치하세요")

            class HW4VenvSimDaemon(SimDaemon):  # HW4 SimDaemon 그대로, 실행 파일만 HW4 venv(MuJoCo 설치됨)의 것
                def command(self, platform: str | None = None) -> list[str]:
                    cmd = super().command(platform)
                    cmd[0] = str(exe)
                    return cmd

            self.daemon = HW4VenvSimDaemon(port=args.port, log_path=log_path, headless=not args.viewer)
        self.robot_cfg = {**cfg["robot"], "host": SIM_HOST, "port": args.port, "connection_mode": "localhost_only",
                          "robot_name": SIM_ROBOT_NAME, "media_backend": "no_media", "connect_timeout_s": 15.0}
        self.link = RobotLink(self.robot_cfg)
        self.rec = Recorder()
        self.report: dict[str, Any] = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}
        self.player: Any = None
        self.backend: Any = None
        self.events: list[dict[str, Any]] = []

    # ---- 공통 ----------------------------------------------------------------
    def step(self, name: str, fn) -> bool:
        print(f"\n[{name}] ...", flush=True)
        t0 = time.perf_counter()
        try:
            status, detail = fn()
        except Exception as exc:  # noqa: BLE001
            status, detail = "FAIL", {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)}
        detail["seconds"] = round(time.perf_counter() - t0, 1)
        self.report["steps"][name] = {"status": status, **detail}
        shown = {k: v for k, v in detail.items() if k != "trace"}
        print(f"[{name}] {status}  " + json.dumps(shown, ensure_ascii=False, default=str)[:900], flush=True)
        return status != "FAIL"

    def measured_now(self) -> tuple[np.ndarray, tuple[float, float], np.ndarray]:
        p = self.backend.present()
        return np.array(p.head), tuple(p.antennas), np.array(p.joints)

    # ---- 1 연결 -------------------------------------------------------------
    def connect(self) -> tuple[str, dict[str, Any]]:
        from reachy_motion.backends.sim_daemon import assert_motion_client_is_local_sim
        from reachy_cafe.robot.motion import MotionPlayer, SharedMiniBackend
        from reachy_cafe.robot.motion_library import cafe_library

        status = self.daemon.start()
        self.link.open()
        sim = assert_motion_client_is_local_sim(self.link.mini)  # 모션을 보내기 전에 시뮬레이터인지 다시 확인
        prepare = self.link.prepare_motion()
        self.backend = SharedMiniBackend(self.link.mini)
        self.backend.connect()
        head0, ant0, _j = self.measured_now()
        t0 = time.perf_counter()
        self.link.wake_up()  # goto INIT 2 s (+ 소리는 no_media라 경고만)
        wake_s = time.perf_counter() - t0
        head1, ant1, _j = self.measured_now()
        self.rec.wrap(self.backend)
        self.rec.sample(self.backend)
        self.player = MotionPlayer(self.backend, library=cafe_library(), motion_cfg=self.cfg["motion"],
                                   screen_point=self.cfg["robot"]["screen_point"],
                                   slot_point=self.cfg["robot"]["slot_point"])
        self.player.record_present = True
        self.player.on_event(self.events.append)
        self.player.preload(list(MOTION_NAMES), variants=("expressive", "neutral"))
        self.player.start()
        from reachy_motion.library.pose import pose_from_matrix

        p0, p1 = pose_from_matrix(head0), pose_from_matrix(head1)
        return "PASS", {"daemon": {k: status.get(k) for k in ("robot_name", "simulation_enabled", "version")},
                        "client_check": sim.get("robot_name"), "connect_s": self.link.info.get("connect_s"),
                        "prepare": prepare, "wake_s": round(wake_s, 2),
                        "start_pose": {"pitch_deg": deg(p0.pitch), "antennas": [round(a, 2) for a in ant0]},
                        "after_wake": {"pitch_deg": deg(p1.pitch), "yaw_deg": deg(p1.yaw),
                                       "antennas": [round(a, 2) for a in ant1]}}

    # ---- 2 시선 --------------------------------------------------------------
    def _hold(self, seconds: float) -> dict[str, Any]:
        from reachy_motion.library.pose import pose_from_matrix, rotation_distance

        time.sleep(seconds)
        cmd = self.player._last_state
        head, ant, joints = self.measured_now()
        m = pose_from_matrix(head)
        # 시선 방향 오차: 머리 정면 축(x)끼리의 각도. 전체 회전 오차(err_deg)에는 시선과 상관없는 roll도 들어간다
        dir_err = math.radians(direction_error_deg(cmd.head.matrix(), head))
        return {"cmd_yaw": deg(cmd.head.yaw), "cmd_pitch": deg(cmd.head.pitch), "meas_yaw": deg(m.yaw),
                "meas_pitch": deg(m.pitch), "meas_roll": deg(m.roll), "dir_err_deg": deg(dir_err),
                "err_deg": deg(rotation_distance(cmd.head.matrix(), head)),
                "body_yaw_deg": deg(float(joints[0])), "joint_margin_deg": round(self.player.guard.margin_deg(cmd), 1)}

    def _lag(self, t0: float, t1: float) -> float | None:
        return gaze_lag(self.rec, t0, t1)

    def gaze(self) -> tuple[str, dict[str, Any]]:
        out: dict[str, Any] = {}
        p = self.player
        p.set_gaze("wander")
        time.sleep(1.0)
        for name, setup in (("screen", lambda: p.set_gaze("screen")), ("slot", lambda: p.set_gaze("slot")),
                            ("user_center", lambda: (p.set_user_ray([1.0, 0.0, 0.0]), p.set_gaze("user"))),
                            ("user_right_far", lambda: (p.set_user_ray([0.4, -1.0, 0.1]), p.set_gaze("user"))),
                            ("user_left_up", lambda: (p.set_user_ray([0.8, 0.6, 0.35]), p.set_gaze("user")))):
            t0 = time.perf_counter()
            setup()
            ray = p._user_ray
            for _ in range(30):  # 카메라처럼 손님 방향을 계속 갱신. 시뮬은 추종이 느려(시간 상수 약 0.7 s) 3 s 기다림
                if name.startswith("user"):
                    p.set_user_ray(ray)
                time.sleep(0.1)
            out[name] = self._hold(0.2)
            out[name]["lag_s"] = self._lag(t0, time.perf_counter())
        dirs = [v["dir_err_deg"] for v in out.values()]
        errs = [v["err_deg"] for v in out.values()]
        bodies = [abs(v["body_yaw_deg"]) for v in out.values()]
        out["max_dir_err_deg"] = max(dirs)
        out["max_err_deg"] = max(errs)
        out["max_body_yaw_deg"] = max(bodies)
        ok = max(dirs) < 2.0 and max(bodies) < 3.0  # 시선 방향 2° 이내, 자동 몸통 회전이 끼어들지 않음
        return ("PASS" if ok else "FAIL"), out

    # ---- 3 자세·제스처 -----------------------------------------------------------
    def posture(self) -> tuple[str, dict[str, Any]]:
        from reachy_motion.library.pose import REST

        p = self.player
        p.set_user_ray([1.0, 0.0, -0.1])
        p.set_gaze("still")
        out: dict[str, Any] = {}
        for name in ("listening", "thinking", "perk_up", "tilt", "neutral"):
            p.set_posture(name)
            out[name] = self._hold(2.5)
            out[name]["meas_antennas"] = [round(a, 2) for a in self.measured_now()[1]]
        p.set_posture("neutral")
        time.sleep(0.8)
        t0 = time.perf_counter()
        p.gesture("antenna_wiggle")
        time.sleep(1.0)
        _cmds, meas = self.rec.window(t0, time.perf_counter())
        right = [m[2][0] for m in meas]
        wiggle = round(max(right) - min(right), 2) if right else None
        t0 = time.perf_counter()
        p.gesture("nod")
        time.sleep(0.8)
        from reachy_motion.library.pose import pose_from_matrix

        _cmds, meas = self.rec.window(t0, time.perf_counter())
        pitches = [math.degrees(pose_from_matrix(m[1]).pitch) for m in meas]
        nod = round(max(pitches) - min(pitches), 2) if pitches else None
        out["antenna_wiggle_range_rad"] = wiggle
        out["nod_range_deg"] = nod
        out["rest_antennas"] = [round(a, 2) for a in REST.antennas]
        ok = (wiggle or 0) > 0.2 and (nod or 0) > 3.0 and all(v["dir_err_deg"] < 2.0 for k, v in out.items()
                                                              if isinstance(v, dict))
        return ("PASS" if ok else "FAIL"), out

    # ---- 4 감정 모션 -------------------------------------------------------------
    def motions(self) -> tuple[str, dict[str, Any]]:
        from reachy_motion.library.pose import rotation_distance

        out: dict[str, Any] = {}
        all_ok = True
        for name in MOTION_NAMES:
            for variant in ("expressive", "neutral"):
                self.player.set_gaze("screen")  # 시선이 REST가 아닌 곳에서 시작 → 전이(transition)가 앞에 붙는 실제 상황
                time.sleep(1.0)
                h = self.player.play(name, variant)
                self.player.set_gaze("rest")  # 모션이 끝난 자리(REST)를 시선 레이어가 유지 → 끝 자세 오차를 잴 수 있게
                done = h.wait(15.0)
                time.sleep(2.0)
                started = next((e for e in self.events if e["kind"] == "started" and e["id"] == h.id), {})
                labels = [(e["label"], round(e["t"] - started.get("t", e["t"]) - started.get("offset_s", 0.0), 2))
                          for e in self.events if e["kind"] == "label" and e["id"] == h.id]
                summary = h.result.summary() if h.result is not None else {}
                ticks = summary.get("ticks") or {}
                end_err = None
                if h.result is not None and h.result.last_sample is not None:
                    end_err = deg(rotation_distance(np.asarray(h.result.last_sample[0]), self.measured_now()[0]))
                ok = done and h.state.value == "DONE"
                all_ok &= ok
                out[f"{name}:{variant}"] = {"state": h.state.value, "offset_s": started.get("offset_s"),
                                            "labels": labels, "late_ticks": ticks.get("late_ticks"),
                                            "effective_hz": None if ticks.get("effective_hz") is None
                                            else round(ticks["effective_hz"], 1),
                                            "n_present": summary.get("n_present"), "end_err_deg": end_err}
        return ("PASS" if all_ok else "FAIL"), out

    # ---- 5 선점 --------------------------------------------------------------
    def preempt(self) -> tuple[str, dict[str, Any]]:
        from reachy_motion.library.pose import rotation_distance
        from reachy_motion.service import Priority

        time.sleep(0.5)
        t0 = time.perf_counter()
        h1 = self.player.play("boredom", "expressive", priority=Priority.IDLE)
        time.sleep(2.5)
        h2 = self.player.play("success", "expressive", priority=Priority.NORMAL, interrupt=True)
        h2.wait(8.0)
        h1.wait(1.0)
        time.sleep(0.3)
        cmds, _meas = self.rec.window(t0, time.perf_counter())
        steps = [math.degrees(rotation_distance(a[1], b[1])) for a, b in zip(cmds, cmds[1:])]
        detail = {"first": h1.state.value, "second": h2.state.value, "max_step_deg": round(max(steps), 2) if steps
                  else None, "n_cmds": len(cmds)}
        ok = h1.state.value == "PREEMPTED" and h2.state.value == "DONE" and (detail["max_step_deg"] or 99) < 4.0
        return ("PASS" if ok else "FAIL"), detail

    # ---- 6 REST --------------------------------------------------------------
    def rest(self) -> tuple[str, dict[str, Any]]:
        from reachy_motion.library.pose import REST, rotation_distance

        self.player.set_gaze("screen")
        time.sleep(1.5)
        h = self.player.park()  # 키오스크 종료와 같은 경로 (RealRobot.close)
        if h is not None:
            h.wait(6.0)
        time.sleep(2.5)  # 시선 레이어가 다시 끌어가지 않는지 (park 전에는 화면 쪽으로 돌아갔다)
        head, ant, _j = self.measured_now()
        err = deg(rotation_distance(REST.head.matrix(), head))
        return ("PASS" if err < 1.5 else "FAIL"), {"rest_err_deg": err, "antennas": [round(a, 2) for a in ant],
                                                  "state": None if h is None else h.state.value}

    # ---- 공통: 측정 관절 ---------------------------------------------------------
    def joint_stats(self) -> dict[str, Any]:
        from reachy_motion.analysis.limits import STEWART_MOTOR, urdf_joint_limits
        from reachy_motion.safety.checker import STEWART_NAMES
        from reachy_motion.safety.policy import DEFAULT_POLICY

        with self.rec.lock:
            meas = list(self.rec.meas)
        if len(meas) < 10:
            return {}
        t = np.array([m[0] for m in meas])
        q = np.array([m[3][1:7] for m in meas])
        limits = urdf_joint_limits()
        lo = np.array([limits[n][0] for n in STEWART_NAMES])
        hi = np.array([limits[n][1] for n in STEWART_NAMES])
        dt = np.diff(t)
        keep = dt > 0.004
        speed = np.degrees(np.abs(np.diff(q, axis=0)))[keep] / dt[keep, None]
        return {"samples": len(meas), "min_margin_deg": round(float(np.degrees(np.minimum(q - lo, hi - q)).min()), 2),
                "policy_min_margin_deg": DEFAULT_POLICY.min_stewart_margin_deg,
                "max_stewart_speed_dps_p99": round(float(np.percentile(speed, 99)), 1),
                "policy_max_stewart_speed_dps": round(DEFAULT_POLICY.max_stewart_speed_ratio
                                                      * STEWART_MOTOR.no_load_deg_s_5v, 1)}

    # ---- 실행 ----------------------------------------------------------------
    def run(self) -> int:
        try:
            if self.step("connect", self.connect):
                for name in ("gaze", "posture", "motions", "preempt", "rest"):
                    self.step(name, getattr(self, name))
                self.report["joints_measured"] = self.joint_stats()
                print("\n[joints] " + json.dumps(self.report["joints_measured"], ensure_ascii=False))
        finally:
            self.finish()
        steps = self.report["steps"].values()
        return 0 if all(s["status"] == "PASS" for s in steps) else 1

    def finish(self) -> None:
        self.rec.stop()
        if self.player is not None:
            try:
                self.player.close()
            except Exception as exc:  # noqa: BLE001
                print(f"  player close: {exc}")
        try:
            self.link.close()
        finally:
            self.daemon.stop()
        out = ROOT / "logs" / f"sim_check_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("\n요약")
        for name, s in self.report["steps"].items():
            print(f"  {name:8s} {s['status']}")
        print(f"결과 파일: {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reachy Café 모션 경로 MuJoCo 시뮬 점검 (HW4 sim daemon)")
    parser.add_argument("--port", type=int, default=8020, help="시뮬 daemon 포트 (키오스크 서버 8000과 다르게)")
    parser.add_argument("--viewer", action="store_true", help="MuJoCo 창을 띄워 움직임을 본다")
    parser.add_argument("--config", action="append", default=[], help="덮어쓸 TOML")
    parser.add_argument("--daemon-only", action="store_true",
                        help="시뮬 daemon만 띄워 두고 기다린다 (키오스크 앱을 config/sim.toml로 붙여 끝까지 시험할 때)")
    args = parser.parse_args(argv)
    cfg = load_config(*args.config)
    ensure_hw_paths(cfg)
    check = SimCheck(cfg, args)
    if args.daemon_only:
        status = check.daemon.start()
        print(f"시뮬 daemon 실행 중: 127.0.0.1:{args.port} ({status.get('robot_name')}). Ctrl+C로 종료", flush=True)
        try:
            while check.daemon.proc is not None and check.daemon.proc.poll() is None:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            check.daemon.stop()
        return 0
    return check.run()


if __name__ == "__main__":
    raise SystemExit(main())
