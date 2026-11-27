"""BehaviorManager: Act를 채널별 출력으로 바꾸고 타이밍을 맞춘다.

타이밍 규칙 (IMPLEMENTATION_PLAN.md 5.7):
- 감정 모션이 있으면 YAML 키프레임 라벨(예: Aha!의 present)이 시작될 때 화면과 음성을 띄운다 (Paper 1의 forethought).
  인사(greeting)는 고개를 숙이기 시작하는 bow 라벨에서 인사말을 시작한다. screen_first인 액트(터치 직후 인사)는
  화면을 바로 바꾸고 음성만 라벨을 기다린다.
- 글랜스가 있으면 시선이 0.3초 먼저 화면·슬롯으로 가고 그다음 말한다 ("저기 보세요"로 읽히게).
- 새 음성·모션 액트는 이전 액트를 끊는다. 음성이 있는 비선점 액트는 현재 액트가 끝난 뒤 말한다.
- 기능 플래그(motion·gaze·slot_led·status_strip)를 끄면 해당 채널만 빠진다 → with/without 비교.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..bus import EventLog
from ..dialogue.nlg import NLG
from ..kiosk_server.hub import Hub
from .acts import Act
from .motion_catalog import timeline
from .robot_api import RobotOutputs

STEP_OF_SCREEN = {"attract": -1, "welcome": -1, "menu": 0, "recommend": 0, "sold_out": 0,
                  "options": 1, "cart": 2, "payment": 3, "done": 4}


class BehaviorManager:
    def __init__(self, cfg: dict[str, Any], robot: RobotOutputs, hub: Hub, nlg: NLG, log: EventLog,
                 robot_label: str = "mock", features: dict[str, bool] | None = None):
        self.robot = robot
        self.hub = hub
        self.nlg = nlg
        self.log = log
        # FSM과 같은 사전을 공유한다 (오퍼레이터 패널에서 켜고 끄면 양쪽에 바로 반영)
        self.features = features if features is not None else {k: bool(v) for k, v in cfg["features"].items()}
        self.follow_cooldown = float(cfg["timing"].get("follow_glance_cooldown_s", 6.0))
        self._last_follow = -1e9
        self.speech_end_t = 0.0  # 로봇이 마지막으로 말을 끝낸 시각 (무반응 시간은 여기서부터 센다)
        motion_cfg = cfg.get("motion", {})
        self.variant = motion_cfg.get("variant", "expressive")
        self.sync_labels = dict(motion_cfg.get("sync_labels", {}))
        self.sync_timeout = float(motion_cfg.get("sync_label_timeout_s", 2.5))
        self.boredom_after = float(cfg["timing"]["boredom_after_s"])
        self.lang = cfg["session"]["default_lang"]
        self._task: asyncio.Task[Any] | None = None
        self._side: set[asyncio.Task[Any]] = set()
        self._side_speech: set[asyncio.Task[Any]] = set()
        self._gaze_off_until = 0.0
        self._idle_enabled = False
        self._plan: tuple[str, dict[str, Any]] | None = None  # 마지막 계획 (언어를 바꾸면 다시 씀)
        self._pending_screen: Act | None = None  # 모션 키프레임을 기다리는 중인 화면 (끊기면 바로 띄운다)
        # FSM이 마지막으로 요청한 자세·시선·LED. 기능을 껐다 켜면 이 값으로 되돌린다
        self._posture_target = "neutral"
        self._gaze_target = "wander"
        self._led_target = "off"
        self.view: dict[str, Any] = {"screen": "attract", "data": {}, "overlay": None, "lang": self.lang,
                                     "step": -1, "features": dict(self.features), "notice": None}
        self.status_fields: dict[str, Any] = {"activity": "idle", "heard": "", "said": "", "plan": "",
                                              "human": "ABSENT", "faces": 0, "score": 0.0, "led": "off",
                                              "robot": robot_label}

    # ---- FSM이 부르는 API -----------------------------------------------------
    def perform(self, act: Act) -> None:
        self.log.log("act", **act.describe())
        if act.clear_overlay and self.view.get("overlay") is not None:
            self.view["overlay"] = None
            self._push_view()
        if act.notice is not None:  # 직원 호출 알림 띠는 모션을 기다리지 않고 바로 (none = 지움)
            self.view["notice"] = None if act.notice == "none" else {"name": act.notice, "data": act.notice_data or {}}
            self._push_view()
        if act.reset_captions:
            self.status(heard="", said="")
        if act.plan:
            self._plan = (act.plan, dict(act.plan_slots))
            self.status(plan=self.nlg.plan(act.plan, self.lang, **act.plan_slots))
        if act.led:
            self._led_target = act.led
            self._set_led(act.led)
        if act.posture:
            self._posture_target = act.posture
            if self.features["motion"]:  # 조건 A(모션 끔)에서는 고개 자세(갸웃·고개 들기 등)도 움직이지 않는다
                self.robot.posture(act.posture)
        if act.gesture and self.features["motion"]:
            self.robot.gesture(act.gesture)
        if act.gaze:
            self._gaze_target = act.gaze
            if self.features["gaze"]:
                self.robot.gaze(act.gaze)
        if act.listen:
            self.robot.listen_context(act.listen)

        use_motion = bool(act.motion) and self.features["motion"]
        interrupts = act.interrupts()
        if interrupts:
            # 키프레임을 기다리던 이전 화면이 있으면 먼저 띄운다 (새 액트가 화면을 바꾸면 생략)
            self._cancel_current(flush_screen=not act.screen)
        if not use_motion or act.screen_first:
            self._apply_screen(act)  # 모션을 기다릴 필요가 없으면 화면은 바로 바꾼다

        if interrupts:
            self._task = asyncio.create_task(self._run(act, use_motion))
            return
        previous = self._task if act.say else None
        task = asyncio.create_task(self._run(act, use_motion, previous))
        self._side.add(task)
        task.add_done_callback(self._side.discard)
        if act.say:
            self._side_speech.add(task)
            task.add_done_callback(self._side_speech.discard)

    def set_idle(self, enabled: bool) -> None:
        self._idle_enabled = enabled
        name = "boredom" if enabled and self.features["motion"] else None
        self.robot.set_idle_motion(name, self.boredom_after)

    def set_lang(self, lang: str) -> None:
        if self.lang != lang or self.view.get("lang") != lang:
            self.lang = lang
            self.view["lang"] = lang
            self._push_view()
            if self._plan is not None:  # 상태 스트립의 계획 줄도 새 언어로 (메뉴 화면에서 English를 눌렀을 때)
                self.status(plan=self.nlg.plan(self._plan[0], lang, **self._plan[1]))
        set_robot_lang = getattr(self.robot, "set_lang", None)  # 실제 로봇: 음성 인식 언어도 바꾼다
        if callable(set_robot_lang):
            set_robot_lang(lang)

    # ---- 앱이 부르는 API -----------------------------------------------------
    def on_user_speech_start(self) -> bool:
        """손님이 말을 시작함. 로봇이 말하는 중이면 바지인: 음성을 끊고 모션을 중립으로.
        조용히 듣던 중이면 백채널(작은 끄덕임 + 듣는 자세)."""
        self.status(activity="listening")
        if self.robot.is_speaking():
            self.log.log("bargein")
            self._cancel_current()
            self.robot.stop_motion(blend_to_rest=True)
            return True
        if self.features.get("backchannel") and self.features["motion"] and not self.gaze_off_user:
            self.robot.gesture("nod")
            self.robot.posture("listening")
        return False

    def on_touch(self) -> None:
        """화면을 터치하면 로봇도 화면 쪽을 힐끗 본다 (터치를 로봇이 '본다'는 느낌)."""
        if (self.features.get("backchannel") and self.features["gaze"] and not self.gaze_off_user
                and not self.robot.is_speaking()):
            self._spawn_glance("screen", 0.6)

    def on_cues(self, cues: dict[str, Any]) -> None:
        """사람 단서(거리·접근·주의)가 바뀜: 상태 스트립에 보이고, 손님이 화면을 보면 로봇도 잠깐 따라 본다."""
        self.status(zone=cues.get("zone"), approaching=cues.get("approaching"), attention=cues.get("attention"))
        now = time.monotonic()
        if (cues.get("attention") == "screen" and self.features.get("attention") and self.features["gaze"]
                and not self.robot.is_speaking() and not self.gaze_off_user
                and now - self._last_follow >= self.follow_cooldown):
            self._last_follow = now
            self.log.log("joint_attention", target="screen")
            self._spawn_glance("screen", 0.8)

    def set_feature(self, key: str, value: bool) -> bool:
        if key not in self.features:
            return False
        self.features[key] = bool(value)
        self.view["features"] = dict(self.features)
        self._push_view()
        if key == "motion":
            if not value:
                self.robot.stop_motion(blend_to_rest=True)
            # 끄면 자세 오프셋도 없앤다 (조건 A), 다시 켜면 FSM이 마지막으로 요청한 자세로
            self.robot.posture(self._posture_target if value else "neutral")
            self.set_idle(self._idle_enabled)
        elif key == "gaze":
            # 끄면 정면(기본 자세)에 멈춘다. 예전 "still"은 마지막으로 보던 곳을 향한 채 멈춰 조건 A와 맞지 않았다
            self.robot.gaze(self._gaze_target if value else "rest")
        elif key == "slot_led":
            self._set_led(self._led_target if value else "off")
        self.log.log("feature", key=key, value=bool(value))
        return True

    def status(self, **fields: Any) -> None:
        changed = False
        for key, value in fields.items():
            if self.status_fields.get(key) != value:
                self.status_fields[key] = value
                changed = True
        if changed:
            self.hub.set_status(dict(self.status_fields))

    @property
    def gaze_off_user(self) -> bool:
        return time.monotonic() < self._gaze_off_until

    # ---- 내부 ---------------------------------------------------------------
    async def _run(self, act: Act, use_motion: bool, previous: asyncio.Task[Any] | None = None) -> None:
        try:
            if previous is not None and not previous.done():
                await asyncio.wait({previous})
            if use_motion:
                deferred = not act.screen_first and bool(act.screen or act.overlay)
                if deferred:
                    self._pending_screen = act  # 키프레임 전에 끊기면 _cancel_current가 바로 띄운다
                handle = self.robot.play_motion(act.motion or "", self.variant)
                duration, _labels = timeline(act.motion or "", self.variant)
                self._gaze_off(duration)
                label = act.sync_label or self.sync_labels.get(act.motion or "")
                if label:
                    await handle.wait_label(label, self.sync_timeout)
                if deferred and self._pending_screen is act:
                    self._pending_screen = None
                    self._apply_screen(act)
            if act.glance and self.features["gaze"]:
                self._gaze_off(act.glance_hold_s + 0.6)
                glance = asyncio.create_task(self.robot.glance(act.glance, act.glance_hold_s))
                self._side.add(glance)
                glance.add_done_callback(self._side.discard)
                await asyncio.sleep(0.3)
            if act.say:
                lang = act.lang or self.lang
                text = self.nlg.text(act.say, lang, **act.slots)
                self.status(activity="speaking", said=text)
                self.log.log("say", key=act.say, lang=lang, text=text)
                try:
                    await self.robot.speak(text, lang)
                finally:
                    self.speech_end_t = time.monotonic()
                self.status(activity="listening" if act.listen else "idle")
            elif act.listen:
                self.status(activity="listening")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # 한 액트의 오류가 키오스크 전체를 멈추지 않게
            self.log.log("error", where="behavior", act=act.name, error=repr(exc))

    def _apply_screen(self, act: Act) -> None:
        changed = False
        if act.screen:
            self.view["screen"] = act.screen
            self.view["data"] = act.data or {}
            self.view["step"] = STEP_OF_SCREEN.get(act.screen, -1)
            changed = True
        if act.overlay:
            self.view["overlay"] = {"name": act.overlay, "data": act.overlay_data or {}}
            changed = True
        if changed:
            self._push_view()

    def _cancel_current(self, flush_screen: bool = True) -> None:
        pending, self._pending_screen = self._pending_screen, None
        if pending is not None and flush_screen:
            # 모션 키프레임 전에 끊긴 액트의 화면: FSM은 이미 그 상태이므로 바로 띄운다.
            # (예전에는 Aha! 도중 되묻기가 끼면 FSM은 추천 중인데 화면은 메뉴인 채로 남았다)
            self._apply_screen(pending)
        if self._task is not None and not self._task.done():
            self._task.cancel()
        for task in list(self._side_speech):
            task.cancel()
        if self.robot.is_speaking():
            self.robot.stop_speech()

    def _spawn_glance(self, target: str, hold_s: float) -> None:
        self._gaze_off(hold_s + 0.6)
        task = asyncio.create_task(self.robot.glance(target, hold_s))
        self._side.add(task)
        task.add_done_callback(self._side.discard)

    def _set_led(self, mode: str) -> None:
        if mode != "off" and not self.features["slot_led"]:
            mode = "off"
        self.status(led=mode)
        self.hub.set_led(mode)

    def _gaze_off(self, seconds: float) -> None:
        self._gaze_off_until = max(self._gaze_off_until, time.monotonic() + seconds)

    def _push_view(self) -> None:
        self.hub.set_view(dict(self.view))
