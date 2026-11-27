"""2단계 연결 점검 (IMPLEMENTATION_PLAN.md 5.1): 데스크톱에서 실제 로봇으로 한 번 돌려 보고 결과를 JSON으로 남긴다.

연결 하나(ReachyMini, webrtc)를 카메라·오디오·모션이 함께 쓸 때 각각이 제대로 도는지 차례로 확인한다.

  1 connect  연결 시간, SDK 버전, 워블링·헤드트래킹 끄기 + 모터 켜기
  2 camera   5초 동안 프레임 fps, 해상도, 카메라 내부 파라미터(K)
  3 faces    5초 동안 얼굴 인식 (카메라 앞에 서 있기): 크기 h, yaw, pitch
  4 audio    마이크 입력, TTS 재생 (Supertonic, 캐시 없이 합성 시간 측정)
  5 asr      "아메리카노 주세요"라고 말하면 받아 적는지 (Whisper, GPU)
  6 motion   화면 쪽 시선 → Aha! 모션 + 효과음 + 동시에 TTS (카메라·오디오가 도는 중에), 라벨 시각·틱 지연
  7 aim      (--aim) 로봇이 화면 가운데·카드 슬롯을 보도록 목표점을 키보드로 옮김 → [robot] screen_point·slot_point
  8 calibrate (--calibrate) 로봇을 볼 때 / 화면을 볼 때 yaw·pitch → [vision] 부호·임계값 추천
  보정 값(aim·calibrate·모션 단계에서 잰 라벨 지연)은 config/calibration.toml에 합쳐 저장 (--no-save로 끔)

실행 (로봇 전원·네트워크 확인 후, 로봇이 움직이므로 주변을 비우고):
  .venv\\Scripts\\python -X utf8 tools\\spike_connection.py
  .venv\\Scripts\\python -X utf8 tools\\spike_connection.py --skip motion --skip asr
  .venv\\Scripts\\python -X utf8 tools\\spike_connection.py --calibrate
결과: logs/spike_<시각>.json (+ 화면 요약). 실패한 단계가 있어도 다음 단계를 계속 시도한다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reachy_cafe.bus import EventLog  # noqa: E402
from reachy_cafe.config import load_config, write_overlay  # noqa: E402
from reachy_cafe.robot.connection import RobotLink  # noqa: E402
from reachy_cafe.robot.hwpaths import ensure_hw_paths, find_face_model  # noqa: E402
from robot_measure import Recorder, gaze_lag  # noqa: E402  (tools/robot_measure.py)

STEPS = ("connect", "camera", "faces", "audio", "asr", "motion")


class SpikeBus:
    """KioskSpeech·FacePerception이 쓰는 버스 자리 (이벤트를 모으기만 한다)."""

    def __init__(self) -> None:
        self.log = EventLog(None)
        self.events: list[tuple[float, str, dict[str, Any]]] = []
        self._lock = threading.Lock()

    def publish_threadsafe(self, kind: str, **data: Any) -> None:
        with self._lock:
            self.events.append((time.perf_counter(), kind, data))

    log_threadsafe = publish_threadsafe

    def of(self, kind: str, since: float = 0.0) -> list[dict[str, Any]]:
        with self._lock:
            return [d for t, k, d in self.events if k == kind and t >= since]


class Spike:
    def __init__(self, cfg: dict[str, Any], args: argparse.Namespace):
        self.cfg = cfg
        self.args = args
        self.bus = SpikeBus()
        self.link = RobotLink(cfg["robot"])
        self.report: dict[str, Any] = {"host": cfg["robot"]["host"], "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       "steps": {}}
        self.speech: Any = None
        self.vision: Any = None
        self.player: Any = None
        self.backend: Any = None
        self.rec = Recorder()
        self.motion_events: list[dict[str, Any]] = []
        self.sound_log: list[tuple[float, str]] = []
        self.calibration: dict[str, dict[str, Any]] = {}  # config/calibration.toml에 합쳐 쓸 값

    # ---- 공통 ----------------------------------------------------------------
    def step(self, name: str, fn) -> None:
        if name in self.args.skip:
            self.report["steps"][name] = {"status": "SKIP"}
            return
        print(f"\n[{name}] ...")
        t0 = time.perf_counter()
        try:
            status, detail = fn()
        except Exception as exc:  # noqa: BLE001 - 다음 단계는 계속
            status, detail = "FAIL", {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=3)}
        detail["seconds"] = round(time.perf_counter() - t0, 1)
        self.report["steps"][name] = {"status": status, **detail}
        print(f"[{name}] {status}  " + json.dumps({k: v for k, v in detail.items() if k != "trace"},
                                                   ensure_ascii=False, default=str)[:400])

    def confirm(self, message: str) -> None:
        if not self.args.yes:
            input(f"  {message} (Enter로 계속, Ctrl+C로 중단) ")

    # ---- 1 연결 -------------------------------------------------------------
    def connect(self) -> tuple[str, dict[str, Any]]:
        self.confirm(f"로봇 {self.cfg['robot']['host']}에 연결하고 모터를 켭니다. 로봇 주변을 비워 주세요.")
        self.link.open()
        prepare = self.link.prepare_motion()
        if not self.args.no_wake:
            self.link.wake_up()
        ok = not any("FAILED" in s for s in prepare)
        return ("PASS" if ok else "WARN"), {**self.link.info, "prepare": prepare}

    # ---- 2 카메라 ------------------------------------------------------------
    def camera(self) -> tuple[str, dict[str, Any]]:
        mini = self.link.mini
        frames, shapes, t0 = 0, set(), time.perf_counter()
        while time.perf_counter() - t0 < 5.0:
            frame = mini.media.get_frame()
            if frame is not None:
                frames += 1
                shapes.add(tuple(frame.shape))
            else:
                time.sleep(0.005)
        fps = frames / 5.0
        model = self.link.camera_model()
        detail = {"fps": round(fps, 1), "shapes": sorted(shapes), "has_K": model is not None,
                  "resolution": None if model is None else model["resolution"]}
        return ("PASS" if fps >= 10 and model is not None else "FAIL"), detail

    # ---- 3 얼굴 --------------------------------------------------------------
    def _start_vision(self, on_ray=None) -> Any:
        from reachy_cafe.robot.vision import FacePerception

        if self.vision is None:
            model = find_face_model(self.cfg)
            if model is None:
                raise FileNotFoundError("face_landmarker.task 없음 (config [hw] face_model / models/ 폴더)")
            self.vision = FacePerception(self.cfg, self.link.mini, self.bus, str(model),
                                         camera_model=self.link.camera_model, on_user_ray=on_ray)
            self.vision.start()
        return self.vision

    def faces(self) -> tuple[str, dict[str, Any]]:
        self.confirm("카메라(로봇 머리) 앞 약 1 m에 서서 로봇을 바라봐 주세요.")
        self._start_vision()
        t0 = time.perf_counter()
        time.sleep(5.0)
        frames = self.bus.of("vision.faces", since=t0)
        seen = [f for d in frames for f in d["faces"]]
        detail: dict[str, Any] = {"frames": len(frames), "fps": round(len(frames) / 5.0, 1),
                                  "frames_with_face": sum(1 for d in frames if d["faces"])}
        if seen:
            detail.update(h=round(statistics.median(f["h"] for f in seen), 3),
                          yaw=round(statistics.median(f["yaw"] or 0.0 for f in seen), 1),
                          pitch=round(statistics.median(f["pitch"] or 0.0 for f in seen), 1),
                          ids=sorted({f["id"] for f in seen}))
        ok = detail["frames_with_face"] >= 0.6 * max(1, len(frames))
        return ("PASS" if ok else "FAIL"), detail

    # ---- 4·5 오디오·ASR ---------------------------------------------------------
    def _start_speech(self, listen: bool) -> Any:
        if self.speech is None:
            from reachy_s2s.robot.backends.reachy import ReachyMediaBackend

            from reachy_cafe.robot.speech import KioskSpeech, load_hw3_config

            hw3 = load_hw3_config(self.cfg)
            backend = ReachyMediaBackend(self.link.mini.media, hw3.audio.output_latency_hint_ms / 1000.0)
            self.speech = KioskSpeech(self.cfg, self.bus, backend, hw3=hw3,
                                      cache_dir=ROOT / "logs" / "spike_tts_cache")
            self.speech.start(listen=listen)
        return self.speech

    def audio(self) -> tuple[str, dict[str, Any]]:
        speech = self._start_speech(listen="asr" not in self.args.skip)
        time.sleep(2.0)
        stats = speech.audio.input_stats()
        level = speech.audio.level_dbfs(0.5)
        if not speech.ready["tts"]:
            t0 = time.perf_counter()
            while not speech.ready["tts"] and time.perf_counter() - t0 < 60:
                time.sleep(0.2)
        t0 = time.perf_counter()
        asyncio.run(speech.speak("안녕하세요! 리치 카페입니다. 소리가 잘 들리나요?", "ko"))
        tts = self.bus.of("speech.tts")
        detail = {"mic_chunks": stats.get("chunks_captured", stats), "mic_level_dbfs": round(level, 1),
                  "tts_ready": speech.ready["tts"], "speak_s": round(time.perf_counter() - t0, 2),
                  "tts": tts[-1] if tts else None}
        heard = input("  로봇 스피커에서 문장이 들렸나요? [y/n] ").strip().lower() if not self.args.yes else "y"
        detail["heard"] = heard.startswith("y")
        return ("PASS" if tts and detail["heard"] else "FAIL"), detail

    def asr(self) -> tuple[str, dict[str, Any]]:
        speech = self._start_speech(listen=True)
        t0 = time.perf_counter()
        while not speech.ready["asr"] and time.perf_counter() - t0 < 120:  # large-v3-turbo 로딩
            time.sleep(0.2)
        print("  지금 로봇에게 \"아메리카노 주세요\"라고 말해 주세요 (10초).")
        since = time.perf_counter()
        while time.perf_counter() - since < 10 and not self.bus.of("speech.utterance", since):
            time.sleep(0.1)
        utts = self.bus.of("speech.utterance", since)
        detail = {"asr_ready": speech.ready["asr"], "utterances": utts[:3],
                  "ignored": self.bus.of("speech.ignored", since)[:3]}
        ok = bool(utts) and "아메리카노" in utts[0]["text"].replace(" ", "")
        return ("PASS" if ok else "FAIL"), detail

    # ---- 6 모션 --------------------------------------------------------------
    def _motion_player(self) -> Any:
        """모션 단계와 보정 단계(--aim)가 같이 쓰는 MotionPlayer (효과음은 로봇 스피커, 명령·측정 기록)."""
        if self.player is not None:
            return self.player
        from reachy_motion.library.registry import motions_dir
        from reachy_motion.sound import RobotSpeaker

        from reachy_cafe.robot.motion import MotionPlayer, SharedMiniBackend
        from reachy_cafe.robot.motion_library import cafe_library

        from reachy_cafe.robot.real import upload_sounds

        speaker = RobotSpeaker(self.cfg["robot"]["host"], int(self.cfg["robot"].get("port", 8000)))
        uploaded: dict[str, str] = {}
        if self.cfg["motion"].get("sound", True):
            uploaded, problems = upload_sounds(speaker, motions_dir() / "sounds")
            for problem in problems:
                print(f"  효과음: {problem}")
        self.report["sound_files"] = sorted(uploaded)

        def play_sound(name: str) -> None:
            self.sound_log.append((time.perf_counter(), name))
            if name in uploaded:
                speaker.play(uploaded[name])

        backend = SharedMiniBackend(self.link.mini)
        backend.connect()
        self.rec.wrap(backend)
        self.rec.sample(backend)
        self.backend = backend
        self.player = MotionPlayer(backend, library=cafe_library(), motion_cfg=self.cfg["motion"],
                                   screen_point=self.cfg["robot"]["screen_point"],
                                   slot_point=self.cfg["robot"]["slot_point"], sound_player=play_sound)
        self.player.on_event(self.motion_events.append)
        self.player.preload(["aha_recommend", "success"])
        self.player.start()
        if self.vision is not None:  # 얼굴 단계가 돌았으면 손님 방향도 넣어 본다
            self.vision.on_user_ray = self.player.set_user_ray
        return self.player

    def motion(self) -> tuple[str, dict[str, Any]]:
        self.confirm("로봇이 화면 쪽을 본 뒤 Aha! 모션을 합니다.")
        player = self._motion_player()
        events = self.motion_events
        player.set_gaze("wander")
        time.sleep(1.5)
        lags = []
        for target in ("screen", "user" if self.vision is not None else "wander", "screen"):
            t0 = time.perf_counter()
            player.set_gaze(target)
            time.sleep(2.0)
            lag = gaze_lag(self.rec, t0, time.perf_counter())
            if lag is not None:
                lags.append(lag)
        speak_thread = None
        if self.speech is not None and self.speech.ready["tts"]:
            speak_thread = threading.Thread(target=lambda: asyncio.run(self.speech.speak(
                "이 메뉴 어떠세요? 상큼하고 달지 않아요.", "ko")), daemon=True)
        h = player.play("aha_recommend", self.cfg["motion"].get("variant", "expressive"))
        if speak_thread is not None:
            speak_thread.start()
        done = h.wait(10.0)
        ticks = h.result.summary().get("ticks") if h.result is not None else None
        started = next((e for e in events if e["kind"] == "started" and e["id"] == h.id), None)
        labels = [{"label": e["label"], "at_s": e["at_s"],
                   "measured_s": None if started is None else round(e["t"] - started["t"], 3)}
                  for e in events if e["kind"] == "label" and e["id"] == h.id]
        if speak_thread is not None:
            speak_thread.join(10.0)
        h2 = player.play("success", "expressive")
        h2.wait(6.0)
        detail = {"state": str(h.state.value), "offset_s": None if started is None else started.get("offset_s"),
                  "labels": labels, "late_ticks": (ticks or {}).get("late_ticks"),
                  "effective_hz": (ticks or {}).get("effective_hz"), "sounds": [n for _t, n in self.sound_log],
                  "success_state": str(h2.state.value), "gaze_steps": player.gaze_steps,
                  "tts_during_motion": speak_thread is not None, "gaze_lag_s": lags,
                  "joint_projections": player.guard.projections}
        if lags:
            # 명령 → 실제 머리 움직임 지연만큼 라벨을 늦추면 화면·음성이 '보이는' 동작과 맞는다 (HW4 실측 약 0.19 s)
            delay = round(min(0.3, max(0.0, statistics.median(lags))), 2)
            detail["suggest_label_delay_s"] = delay
            self.calibration.setdefault("motion", {})["label_delay_s"] = delay
        ok = done and h.state.value == "DONE" and h2.state.value == "DONE"
        return ("PASS" if ok else "FAIL"), detail

    def _assert_live(self, max_age_s: float = 1.0) -> None:
        """daemon 상태 수신이 끊기면 멈춘다 (연결이 끊긴 채 보정하면 값이 의미 없다)."""
        stamps = getattr(self.backend, "_stamps", None)
        last = None if stamps is None else stamps.joints_ns
        if last is None or (time.perf_counter_ns() - last) / 1e9 > max_age_s:
            raise RuntimeError("로봇 상태 수신이 끊김 (연결·daemon 확인)")

    # ---- 화면·슬롯 위치 맞추기 (--aim) ------------------------------------------------
    def aim(self) -> tuple[str, dict[str, Any]]:
        """로봇이 화면 가운데·카드 슬롯을 실제로 보도록 목표점을 옮긴다 (로봇 기준 m: x 앞, y 왼쪽, z 위)."""
        import math

        player = self._motion_player()
        keys = {"l": (1, 1), "r": (1, -1), "u": (2, 1), "d": (2, -1), "f": (0, 1), "b": (0, -1)}
        print("  로봇이 보는 곳이 목표(화면 가운데 / 카드 슬롯)가 될 때까지 옮깁니다.")
        print("  l/r = 로봇 기준 왼쪽/오른쪽, u/d = 위/아래, f/b = 앞/뒤 (2 cm, 대문자면 5 cm), Enter = 확정")
        out: dict[str, list[float]] = {}
        for name, label in (("screen", "화면 가운데"), ("slot", "카드 슬롯")):
            point = [float(v) for v in self.cfg["robot"][f"{name}_point"]]
            player.set_gaze(name)
            while True:
                want = player.set_point(name, point)
                time.sleep(1.0)
                self._assert_live()
                cmd = player._last_state
                clipped = (abs(cmd.head.yaw - want.yaw) > math.radians(1.5)
                           or abs(cmd.head.pitch - want.pitch) > math.radians(1.5))
                text = (f"  [{label}] x={point[0]:.2f} y={point[1]:.2f} z={point[2]:.2f} m -> 머리 yaw "
                        f"{math.degrees(cmd.head.yaw):.0f}, pitch {math.degrees(cmd.head.pitch):.0f} deg"
                        + (" (범위·관절 한계로 덜 돌아감)" if clipped else ""))
                key = input(text + "  > ").strip()
                if not key:
                    break
                axis = keys.get(key.lower())
                if axis is None:
                    print("  l/r/u/d/f/b 또는 Enter")
                    continue
                point[axis[0]] = round(point[axis[0]] + axis[1] * (0.05 if key.isupper() else 0.02), 3)
            out[name] = point
        player.set_gaze("wander")
        self.calibration.setdefault("robot", {}).update(screen_point=out["screen"], slot_point=out["slot"])
        return "PASS", {"points": out}

    # ---- 7 보정 --------------------------------------------------------------
    def calibrate(self) -> tuple[str, dict[str, Any]]:
        self._start_vision()
        phases = {}
        for name, prompt in (("robot", "로봇(카메라)을 바라보세요"), ("screen", "키오스크 화면을 바라보세요"),
                             ("away", "옆이나 뒤쪽(딴 곳)을 바라보세요")):
            self.confirm(prompt)
            time.sleep(1.0)
            since = time.perf_counter()
            time.sleep(4.0)
            seen = [f for d in self.bus.of("vision.faces", since) for f in d["faces"][:1]]
            phases[name] = {"n": len(seen),
                            "yaw": round(statistics.median(f["yaw"] or 0.0 for f in seen), 1) if seen else None,
                            "pitch": round(statistics.median(f["pitch"] or 0.0 for f in seen), 1) if seen else None,
                            "h": round(statistics.median(f["h"] for f in seen), 3) if seen else None}
            print(f"  {name}: {phases[name]}")
        r, s = phases["robot"], phases["screen"]
        if r["yaw"] is None or s["yaw"] is None:
            return "FAIL", {"phases": phases}
        dyaw, dpitch = s["yaw"] - r["yaw"], s["pitch"] - r["pitch"]
        suggest = {"screen_yaw_sign": 1.0 if dyaw >= 0 else -1.0, "screen_pitch_sign": 1.0 if dpitch >= 0 else -1.0,
                   "screen_yaw_min_deg": round(max(8.0, abs(dyaw) * 0.5), 1),
                   "screen_pitch_down_deg": round(max(8.0, abs(dpitch) * 0.5), 1),
                   "look_robot_deg": round(max(8.0, min(abs(dyaw), abs(dpitch) if abs(dpitch) > 5 else 99) * 0.4), 1)}
        print("  [vision] 추천 값 (config/calibration.toml에 저장):")
        for k, v in suggest.items():
            print(f"    {k} = {v}")
        self.calibration.setdefault("vision", {}).update(suggest)
        return "PASS", {"phases": phases, "suggest": suggest}

    # ---- 실행 ----------------------------------------------------------------
    def run(self) -> int:
        # finally: 안내대로 Ctrl+C로 끝내도 로봇을 REST로 보내 연결을 닫고 결과 파일을 남긴다 (예전에는 건너뜀)
        try:
            self.step("connect", self.connect)
            if self.report["steps"]["connect"]["status"] == "FAIL":
                return 2
            for name in STEPS[1:]:
                self.step(name, getattr(self, name))
            if self.args.aim:
                self.step("aim", self.aim)
            if self.args.calibrate:
                self.step("calibrate", self.calibrate)
            return 0 if all(s["status"] in ("PASS", "SKIP", "WARN") for s in self.report["steps"].values()) else 1
        finally:
            self.finish()

    def finish(self) -> None:
        for name, obj in (("vision", self.vision), ("speech", self.speech)):
            if obj is not None:
                try:
                    obj.close()
                except Exception as exc:  # noqa: BLE001
                    print(f"  {name} close: {exc}")
        if self.player is not None:
            handle = self.player.park()  # REST로 돌아가 머문다 (시선 레이어가 다시 끌어가지 않게)
            if handle is not None:
                handle.wait(5.0)
            time.sleep(0.5)
            self.player.close()
        self.rec.stop()
        self.link.close()
        if self.calibration and not self.args.no_save:
            path = write_overlay(self.calibration, *( [self.args.calibration_file] if self.args.calibration_file else []))
            self.report["calibration"] = {"file": str(path), **self.calibration}
            print(f"\n보정 값 저장: {path}")
            print("  키오스크 실행: .venv\\Scripts\\python -X utf8 -m reachy_cafe --config config/calibration.toml")
        out = ROOT / "logs" / f"spike_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("\n요약")
        for name, s in self.report["steps"].items():
            print(f"  {name:10s} {s['status']}")
        print(f"결과 파일: {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reachy Café 2단계 연결 점검 (실제 로봇)")
    parser.add_argument("--config", action="append", default=[], help="덮어쓸 TOML (여러 번 가능)")
    parser.add_argument("--robot-host", help="로봇 IP (기본: config robot.host / REACHY_HOST)")
    parser.add_argument("--skip", action="append", default=[], choices=STEPS[1:], help="건너뛸 단계")
    parser.add_argument("--calibrate", action="store_true", help="주의 방향(yaw/pitch) 보정 단계 추가")
    parser.add_argument("--aim", action="store_true", help="화면·카드 슬롯 위치를 로봇 시선으로 맞추는 단계 추가")
    parser.add_argument("--no-save", action="store_true", help="보정 값을 config/calibration.toml에 쓰지 않음")
    parser.add_argument("--calibration-file", default=None, help="보정 값을 쓸 파일 (기본 config/calibration.toml)")
    parser.add_argument("--no-wake", action="store_true", help="연결 후 wake_up 동작 생략")
    parser.add_argument("--yes", action="store_true", help="확인 질문 없이 진행 (소리 확인은 성공으로 간주)")
    args = parser.parse_args(argv)
    cfg = load_config(*args.config)
    if args.robot_host:
        cfg["robot"]["host"] = args.robot_host
    ensure_hw_paths(cfg)
    try:
        return Spike(cfg, args).run()
    except KeyboardInterrupt:
        print("\n중단됨")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
