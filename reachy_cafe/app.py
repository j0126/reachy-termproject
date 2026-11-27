"""앱 조립: 설정 → 버스 → 메뉴·NLU·FSM → 행동·로봇 → 서버. 이벤트 디스패처 하나와 10 Hz 티커가 돈다."""

from __future__ import annotations

import asyncio
import concurrent.futures
import socket
import time
from typing import Any

import uvicorn

from .behavior.manager import BehaviorManager
from .behavior.mock_robot import MockRobot
from .behavior.robot_api import RobotOutputs
from .bus import Event, EventBus, EventLog
from .config import CONFIG_DIR, ROOT
from .dialogue.fsm import InteractionManager, Ov
from .dialogue.menu import Menu, OrderCounter
from .dialogue.nlg import NLG
from .dialogue.nlu import NLU
from .kiosk_server.hub import Hub
from .kiosk_server.server import create_app
from .perception.human_state import Face, HumanStateEstimator
from .perception.sim import SimPerception

UI_DIR = ROOT / "kiosk_ui"
TICK_S = 0.1
FEATURE_KEYS = ("motion", "gaze", "status_strip", "slot_led")          # 조건 A/B로 함께 켜고 끄는 것
IMPROVEMENT_KEYS = ("invite", "attention", "backchannel", "context_talk", "auto_staff")  # 보완 기능 (유지/제외 결정 전)


def lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.0.1", 80))  # 패킷을 보내지 않고 라우팅용 로컬 주소만 얻는다
            return s.getsockname()[0]
    except OSError:
        return "localhost"


class App:
    def __init__(self, cfg: dict[str, Any], log_to_file: bool = True):
        self.cfg = cfg
        self._start_job: concurrent.futures.Future[Any] | None = None  # 실제 로봇 연결 스레드 (run의 정리용)
        self.log = EventLog(ROOT / cfg["logging"]["dir"] if log_to_file else None)
        self.bus = EventBus(self.log)
        self.hub = Hub()
        self.log.add_listener(self.hub.push_log)
        self.menu = Menu.load(CONFIG_DIR / "menu.json")
        self.nlg = NLG(self.menu)
        self.nlu = NLU(self.menu, min_score=float(cfg["speech"]["nlu_min_score"]))
        self.robot, self.robot_label = self._make_robot()
        self.features = {key: bool(value) for key, value in cfg["features"].items()}  # FSM·행동이 공유
        self.behavior = BehaviorManager(cfg, self.robot, self.hub, self.nlg, self.log, robot_label=self.robot_label,
                                        features=self.features)
        self.estimator = HumanStateEstimator(cfg["timing"], cfg["vision"])
        self.sim = SimPerception("none")
        self.use_sim_faces = self.robot_label == "mock"
        self.fsm = InteractionManager(cfg, self.menu, self.nlu, self.behavior,
                                      OrderCounter(ROOT / "state" / "order_counter.txt"), self.log,
                                      on_lock_primary=self.estimator.lock_primary, features=self.features)
        self._ticks = 0
        self._face_key: tuple[Any, ...] = ()
        self._cues: dict[str, Any] = self.estimator.cues()
        self.trial: dict[str, Any] = {}

    def _make_robot(self) -> tuple[RobotOutputs, str]:
        if self.cfg["robot"]["mode"] == "real":
            # HW2(카메라)·HW3(음성)·HW4(모션)를 공유 ReachyMini 연결 하나로 붙인 백엔드. 연결은 run()에서 연다.
            from .robot.real import RealRobot

            return RealRobot(self.cfg, self.bus, None if self.log.path is None else self.log.path.parent), "real"
        return MockRobot(self.log), "mock"

    async def _start_robot(self) -> None:
        """실제 로봇 연결. 연결 자체가 안 되면 mock으로 바꿔 화면·오퍼레이터 패널은 계속 쓸 수 있게 한다."""
        if self.robot_label != "real":
            return
        print(f"로봇 연결 중... ({self.cfg['robot']['host']})")
        # 연결 스레드를 직접 잡아 둔다: 기다리는 중에 Ctrl+C가 와도 스레드는 계속 돌므로, run()이 끝나기를 기다렸다가 닫는다
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="robot-start")
        self._start_job = executor.submit(self.robot.start, asyncio.get_running_loop(), self.menu,
                                          lambda: self.estimator.primary_id)
        executor.shutdown(wait=False)
        try:
            info = await asyncio.wrap_future(self._start_job)
        except Exception as exc:  # noqa: BLE001 - 연결 실패 (IP·전원·네트워크)
            self.log.log("error", where="robot.connect", error=f"{type(exc).__name__}: {exc}",
                         robot_host=self.cfg["robot"]["host"])
            print(f"  로봇 연결 실패: {exc}\n  → mock 로봇으로 계속합니다 (얼굴은 오퍼레이터 패널 프리셋 사용)")
            self.robot, self.robot_label = MockRobot(self.log), "mock (connect failed)"
            self.behavior.robot = self.robot
            self.behavior.status(robot=self.robot_label)
            self.use_sim_faces = True
            return
        self.use_sim_faces = not self.robot.camera_ok
        print("  " + ", ".join(f"{k}: {v}" for k, v in info.items() if k in ("connection", "motion", "speech",
                                                                            "camera", "connect_s")))
        self.behavior.set_lang(self.behavior.lang)  # 음성 인식 언어를 현재 세션 언어로

    # ---- 실행 ---------------------------------------------------------------
    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self.bus.bind_loop(loop)
        try:
            await self._start_robot()  # 실제 로봇 연결은 30초까지 걸린다
        except BaseException:
            # 시작 도중 Ctrl+C·예외: 열린 연결·모터를 정리하고 나간다 (예전에는 try 밖이라 정리를 건너뜀).
            # 연결 스레드가 아직 돌면 끝나기를 기다린다 (먼저 닫으면 그 뒤에 연결이 열리고 모터가 켜진 채로 남는다)
            job = self._start_job
            if job is not None and not job.done():
                print("  로봇 연결이 끝나기를 기다렸다가 정리합니다...")
                await asyncio.to_thread(concurrent.futures.wait, [job], 60)
            await self._close_robot()
            self.log.close()
            raise
        self.fsm.start(time.monotonic())
        self.behavior.status(human=self.estimator.state)
        server_cfg = self.cfg["server"]
        port = int(server_cfg["port"])
        server = uvicorn.Server(uvicorn.Config(create_app(self.hub, self.bus, UI_DIR), host=server_cfg["host"],
                                               port=port, log_level="warning", access_log=False))
        ip = lan_ip()
        self.log.log("app.start", robot=self.robot_label, robot_host=self.cfg["robot"]["host"], port=port,
                     log_file=str(self.log.path) if self.log.path else None)
        print(f"Reachy Café 실행 중 (로봇: {self.robot_label}, 로봇 IP 설정: {self.cfg['robot']['host']})")
        print(f"  키오스크 화면   http://{ip}:{port}/kiosk   (Pi 4의 BRAIN_HOST={ip})")
        print(f"  오퍼레이터 패널 http://localhost:{port}/operator")
        if self.log.path:
            print(f"  이벤트 로그     {self.log.path}")
        workers = [asyncio.create_task(self._dispatch()), asyncio.create_task(self._ticker())]
        try:
            await server.serve()
        finally:
            for task in workers:
                task.cancel()
            await self._close_robot()
            self.log.log("app.stop")
            self.log.close()

    async def _close_robot(self) -> None:
        close = getattr(self.robot, "close", None)
        if callable(close):
            try:
                await asyncio.to_thread(close)
            except Exception as exc:  # noqa: BLE001
                self.log.log("error", where="robot.close", error=repr(exc))

    async def _dispatch(self) -> None:
        while True:
            event = await self.bus.get()
            try:
                self.route(event, time.monotonic())
            except Exception as exc:
                self.log.log("error", where="dispatch", kind=event.kind, error=repr(exc))

    async def _ticker(self) -> None:
        while True:
            try:
                self.step(time.monotonic())
            except Exception as exc:
                self.log.log("error", where="ticker", error=repr(exc))
            await asyncio.sleep(TICK_S)

    # ---- 한 단계 (테스트에서도 직접 부른다) --------------------------------------
    def _ordering(self) -> bool:
        """주문이 진행 중인가 (세션이 있고 일시정지가 아님). 이때만 말소리를 손님이 있다는 근거로 쓴다."""
        return self.fsm.session is not None and self.fsm.overlay != Ov.PAUSED

    def route(self, event: Event, now: float) -> None:
        kind = event.kind
        if kind == "touch":
            self.estimator.note_input(now)
            self.behavior.on_touch()
            self.fsm.handle(event, now)
        elif kind == "card":
            self.estimator.note_input(now)
            self.fsm.handle(event, now)
        elif kind == "speech.start":
            # 말소리(VAD)만으로는 사람이 있다고 보지 않는다: 주문 중일 때만 '있음'으로 센다.
            # 예전에는 주변 잡담만으로 5초간 ENGAGED가 되어 빈 자리에 인사하고 끄덕였다 (발표장 관객 소리)
            if self._ordering():
                self.estimator.note_input(now)
            if self.fsm.session is not None or self.robot.is_speaking():
                self.behavior.on_user_speech_start()  # 바지인·백채널
        elif kind == "speech.utterance":
            self.fsm.handle(event, now)  # FSM이 아무도 없을 때의 알아들을 수 없는 말은 버린다
            if self._ordering():  # 받아들여진 말(세션이 시작됐거나 진행 중)만 '있음'·자막으로
                self.estimator.note_input(now)
                self.behavior.status(heard=str(event.get("text", "")), activity="thinking")
        elif kind == "vision.faces":
            faces = [Face.from_dict(raw) for raw in event.get("faces", [])]
            self.estimator.update_faces(faces, now)
        elif kind == "operator":
            self._operator(event.data, now)
        elif kind == "device":
            self.behavior.status(device=event.get("state"))
            self.fsm.on_device(event.get("state"), now)  # 결제 중에 카드 슬롯이 끊기면 바로 화면 결제 안내

    def step(self, now: float) -> None:
        if self.use_sim_faces:
            faces = self.sim.faces(now)
            if faces is not None:
                self.estimator.update_faces(faces, now)
        self.estimator.set_gaze_off_user(self.behavior.gaze_off_user)
        # 무반응(망설임·도움 요청) 시간은 로봇이 말을 끝낸 뒤부터 센다. 말하는 동안 쳐다보는 건 자연스러운 것
        awaiting = self.fsm.awaiting_since
        if awaiting is not None:
            awaiting = now if self.robot.is_speaking() else max(awaiting, self.behavior.speech_end_t)
        new_state = self.estimator.step(now, awaiting, use_attention=bool(self.features.get("attention")))
        cues = self.estimator.cues()
        if cues != self._cues:
            self._cues = cues
            self.log.log("human.cues", **cues)
            self.behavior.on_cues(cues)
            self.fsm.handle(Event("human.cues", dict(cues)), now)
        if new_state is not None:
            reason = self.estimator.reason if new_state == "NO_RESPONSE" else None
            self.log.log("human.state", state=new_state, reason=reason, faces=len(self.estimator.faces),
                         primary=self.estimator.primary_id, score=self.estimator.score)
            self.behavior.status(human=new_state)
            self.fsm.handle(Event("human.state", {"state": new_state, "reason": reason}), now)
        self._update_face_status()
        self.fsm.tick(now)
        self._ticks += 1
        if self._ticks % 5 == 0:
            self.push_snapshot(now)

    def _update_face_status(self) -> None:
        boxes = [{"cx": round(f.cx, 2), "cy": round(f.cy, 2), "h": round(f.h, 2),
                  "primary": f.id == self.estimator.primary_id} for f in self.estimator.faces]
        key = (tuple(tuple(sorted(b.items())) for b in boxes), self.estimator.score)
        if key != self._face_key:
            self._face_key = key
            self.behavior.status(faces=len(boxes), face_boxes=boxes, score=self.estimator.score)

    def push_snapshot(self, now: float) -> None:
        if self.hub.count("operator") == 0:
            return
        snapshot = self.fsm.snapshot(now)
        snapshot["perception"] = self.estimator.snapshot()
        snapshot["sim_preset"] = self.sim.preset if self.use_sim_faces else "camera"
        snapshot["features"] = dict(self.features)
        snapshot["improvements"] = list(IMPROVEMENT_KEYS)
        snapshot["trial"] = dict(self.trial)
        snapshot["robot"] = {"label": self.robot_label, "host": self.cfg["robot"]["host"],
                             "gaze": getattr(self.robot, "gaze_target", None),
                             "posture": getattr(self.robot, "posture_name", None),
                             "speaking": self.robot.is_speaking()}
        status = getattr(self.robot, "status", None)
        if callable(status):
            snapshot["robot"]["health"] = status()
        snapshot["clients"] = {"ui": self.hub.count("ui"), "device": self.hub.count("device")}
        self.hub.set_snapshot(snapshot)

    # ---- 오퍼레이터 패널 ---------------------------------------------------------
    def _operator(self, data: dict[str, Any], now: float) -> None:
        action = data.get("action")
        if action == "faces":
            preset = str(data.get("preset"))
            if preset == "camera":
                self.use_sim_faces = not getattr(self.robot, "camera_ok", False)
            elif self.sim.set(preset):
                self.use_sim_faces = True
        elif action == "say":
            text = str(data.get("text", "")).strip()
            if text:
                self.bus.publish("speech.start", source="operator")
                self.bus.publish("speech.utterance", text=text, conf=1.0, source="operator")
        elif action == "speech_start":
            self.bus.publish("speech.start", source="operator")
        elif action == "card" and data.get("state") in ("in", "out"):
            self.bus.publish("card", state=data["state"], source="operator")
        elif action == "touch" and data.get("touch"):
            touch = dict(data.get("params") or {})
            touch["action"] = data["touch"]
            self.bus.publish("touch", source="operator", **touch)
        elif action == "reset":
            self.fsm.reset(now)
        elif action == "staff_done":
            self.fsm.staff_done(now)  # 직원(오퍼레이터)이 손님 응대를 마침
        elif action == "force_state":
            self.fsm.force(str(data.get("state")), now)
        elif action == "stock" and data.get("item"):
            self.menu.set_stock(str(data["item"]), bool(data.get("in_stock")))
            self.fsm.refresh()
        elif action == "lang":
            self.fsm.set_lang(str(data.get("lang", "ko")), now)
        elif action == "feature" and data.get("key"):
            self.behavior.set_feature(str(data["key"]), bool(data.get("value")))
        elif action == "condition":
            on = str(data.get("condition", "B")).upper() == "B"
            for key in FEATURE_KEYS:
                self.behavior.set_feature(key, on)
            self.log.log("condition", condition="B" if on else "A")
        elif action == "neutral":
            self.robot.stop_motion(blend_to_rest=True)
            self.robot.posture("neutral")
        elif action == "context":
            self.fsm.set_context(str(data.get("weather", "normal")))
        elif action == "trial":
            # 평가 세션 표시: 참가자·조건·과제를 로그에 남겨 analyze_logs.py가 묶을 수 있게
            self.trial = {"participant": str(data.get("participant", "")).strip(),
                          "condition": str(data.get("condition", "")).strip(),
                          "task": str(data.get("task", "")).strip()}
            self.log.log("trial", **self.trial)
        self.push_snapshot(now)
