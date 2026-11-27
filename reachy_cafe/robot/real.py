"""실제 로봇: robot_api.RobotOutputs 구현 (MockRobot과 같은 모양).

연결 하나(RobotLink) 위에 모션(MotionPlayer, HW4)·음성(KioskSpeech, HW3)·카메라(FacePerception, HW2)를 올린다.
세 부분은 따로 시작하고 따로 실패한다: 카메라가 안 되면 오퍼레이터의 얼굴 프리셋으로, 음성이 안 되면 화면과
오퍼레이터 텍스트 입력으로, 모션이 안 되면 음성·화면만으로 계속한다 (실패 처리, IMPLEMENTATION_PLAN.md 6장).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..behavior.mock_robot import MockRobot
from ..behavior.motion_catalog import MOTIONS
from ..dialogue.nlg import FIXED
from .connection import RobotLink
from .hwpaths import ensure_hw_paths, find_face_model

log = logging.getLogger(__name__)

TERMINAL_KINDS = ("finished", "cancelled", "preempted", "rejected", "failed")


def upload_sounds(speaker: Any, folder: Path) -> tuple[dict[str, str], list[str]]:
    """모션 효과음을 daemon에 올린다. daemon에 미디어가 있을 때만 (GET /api/media/status의 available).

    미디어가 없는 daemon에 올리면 GStreamer 검사에서 400이 나고, MuJoCo 시뮬 daemon(--no-media)은 몇 번째
    업로드에서 오류 메시지 없이 죽기도 했다 (tools/spike_connection.py 시뮬 리허설에서 발견).
    돌려주는 값: ({파일 이름: daemon 경로}, [문제 설명])
    """
    try:
        status = speaker.status()
    except Exception as exc:  # noqa: BLE001
        return {}, [f"media status unavailable: {type(exc).__name__}: {exc}"]
    if not status.get("available", False):
        return {}, [f"daemon media not available ({status}): motion sounds off"]
    uploaded: dict[str, str] = {}
    problems: list[str] = []
    for path in sorted(Path(folder).glob("*.wav")):
        try:
            uploaded[path.name] = speaker.upload(path)
        except Exception as exc:  # noqa: BLE001 - 효과음이 안 돼도 모션은 계속
            problems.append(f"{path.name}: {type(exc).__name__}: {exc}")
    return uploaded, problems


class RealMotionHandle:
    """HW4 MotionHandle을 asyncio에서 기다릴 수 있게 감싼다 (behavior.robot_api.MotionHandle)."""

    def __init__(self, name: str, variant: str, inner: Any, labels: set[str]):
        self.name = name
        self.variant = variant
        self.inner = inner
        self.labels = labels  # 이 모션 head 트랙에 실제로 있는 라벨 (neutral 변형에는 없는 라벨이 있다)
        self.is_idle = False
        self.state: str | None = None
        self._label_events = {label: asyncio.Event() for label in labels}
        self._done = asyncio.Event()

    def on_event(self, event: dict[str, Any]) -> None:  # 이벤트 루프 스레드에서 불린다
        kind = event.get("kind")
        if kind == "label":
            ev = self._label_events.get(str(event.get("label")))
            if ev is not None:
                ev.set()
        elif kind in TERMINAL_KINDS:
            self.state = kind
            for ev in self._label_events.values():
                ev.set()  # 끊기거나 거절돼도 기다리던 화면·음성이 멈추지 않게
            self._done.set()

    def done(self) -> bool:
        return self._done.is_set()

    async def wait_label(self, label: str, timeout: float) -> bool:
        ev = self._label_events.get(label)
        if ev is None:  # 이 변형에는 그 라벨이 없음 → 기다리지 않는다 (MockMotionHandle과 같음)
            return False
        try:
            await asyncio.wait_for(ev.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_done(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._done.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False


class RealRobot:
    def __init__(self, cfg: dict[str, Any], bus: Any, log_dir: Path | None = None):
        self.cfg = cfg
        self.bus = bus
        self.log_dir = log_dir
        self.link = RobotLink(cfg["robot"])
        self.player: Any = None
        self.speech: Any = None
        self.vision: Any = None
        self.speaker: Any = None
        self.sounds: dict[str, str] = {}
        self.health: dict[str, Any] = {"connection": "off", "motion": "off", "speech": "off", "camera": "off"}
        self.variant = cfg.get("motion", {}).get("variant", "expressive")
        self.gaze_target = "wander"
        self.posture_name = "neutral"
        self.context = "open"
        self._handles: dict[int, RealMotionHandle] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._fallback = MockRobot(bus.log)  # 음성 채널이 없을 때 말하는 시간 흉내

    # ---- 시작·종료 (블로킹: asyncio.to_thread로 부른다) ----------------------------------
    def start(self, loop: asyncio.AbstractEventLoop, menu: Any = None,
              primary_id: Callable[[], int | None] | None = None) -> dict[str, Any]:
        self._loop = loop
        added = ensure_hw_paths(self.cfg)
        t0 = time.perf_counter()
        self.link.open()  # 실패하면 예외 → App이 mock으로 전환
        self.health["connection"] = "ok"
        prepare = self.link.prepare_motion()
        if self.cfg["robot"].get("wake_up", True):
            try:
                self.link.wake_up()
            except Exception as exc:  # noqa: BLE001
                self._note("wake_up", exc)
        self._start_motion()
        self._start_speech(menu)
        self._start_vision(primary_id)
        info = {"connect_s": self.link.info.get("connect_s"), "sdk": self.link.info.get("sdk_version"),
                "prepare": prepare, "hw_paths": added, "start_s": round(time.perf_counter() - t0, 1),
                **self.health}
        self.bus.log_threadsafe("robot.start", **info)
        return info

    def _note(self, part: str, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        log.warning("%s failed: %s", part, message)
        self.bus.log_threadsafe("error", where=f"robot.{part}", error=message)

    def _start_motion(self) -> None:
        try:
            from reachy_motion.library.registry import motions_dir

            from .motion import MotionPlayer, SharedMiniBackend
            from .motion_library import cafe_library

            backend = SharedMiniBackend(self.link.mini)
            backend.connect()
            mcfg = self.cfg.get("motion", {})
            sound = self._play_sound if mcfg.get("sound", True) else None
            self.player = MotionPlayer(backend, library=cafe_library(), motion_cfg=mcfg,
                                       screen_point=self.cfg["robot"].get("screen_point", (0.20, 0.22, -0.12)),
                                       slot_point=self.cfg["robot"].get("slot_point", (0.22, 0.20, -0.28)),
                                       sound_player=sound)
            self.player.on_event(self._on_motion_event)
            self.player.preload(list(MOTIONS), variants=("expressive", "neutral"))
            self.player.start()
            self.health["motion"] = "ok"
            if sound is not None:
                self._upload_sounds(motions_dir() / "sounds")
        except Exception as exc:  # noqa: BLE001
            self.player = None
            self.health["motion"] = f"failed: {exc}"
            self._note("motion", exc)

    def _upload_sounds(self, folder: Path) -> None:
        """효과음을 미리 daemon에 올려 둔다 (업로드는 최대 5 s 걸릴 수 있어 큐 시점에 하면 늦다)."""
        from reachy_motion.sound import RobotSpeaker

        self.speaker = RobotSpeaker(self.cfg["robot"]["host"], int(self.cfg["robot"].get("port", 8000)))
        self.sounds, problems = upload_sounds(self.speaker, folder)
        for problem in problems:
            self.bus.log_threadsafe("error", where="robot.sound_upload", error=problem)
        self.health["sound"] = f"{len(self.sounds)} files" if self.sounds else "off"

    def _play_sound(self, filename: str) -> None:  # 모션 스레드에서 따로 띄운 스레드
        daemon_path = self.sounds.get(filename)
        if daemon_path is None or self.speaker is None:
            return
        self.speaker.play(daemon_path)
        self.bus.log_threadsafe("robot.sound", file=filename)

    def _start_speech(self, menu: Any) -> None:
        try:
            from reachy_s2s.robot.backends.reachy import ReachyMediaBackend

            from .speech import KioskSpeech, load_hw3_config, menu_prompt

            hw3 = load_hw3_config(self.cfg)
            backend = ReachyMediaBackend(self.link.mini.media, hw3.audio.output_latency_hint_ms / 1000.0)
            listen = self.cfg["speech"].get("asr", "whisper") == "whisper"
            prompts = {lang: menu_prompt(menu, lang) for lang in ("ko", "en")} if menu is not None else {}
            self.speech = KioskSpeech(self.cfg, self.bus, backend, hw3=hw3, prompts=prompts,
                                      events_path=None if self.log_dir is None else self.log_dir / "speech_events.jsonl")
            self.speech.start(listen=listen)
            if self.cfg["speech"].get("presynthesize", True):
                self.speech.presynthesize([(texts[lang], lang) for texts in FIXED.values() for lang in ("ko", "en")
                                           if texts.get(lang)])
            self.health["speech"] = "ok" if listen else "ok (output only)"
        except Exception as exc:  # noqa: BLE001
            self.speech = None
            self.health["speech"] = f"failed: {exc}"
            self._note("speech", exc)

    def _start_vision(self, primary_id: Callable[[], int | None] | None) -> None:
        try:
            from .vision import FacePerception

            self._wait_first_frame(float(self.cfg["vision"].get("first_frame_timeout_s", 5.0)))
            model = find_face_model(self.cfg)
            on_ray = self.player.set_user_ray if self.player is not None else None
            self.vision = FacePerception(self.cfg, self.link.mini, self.bus, None if model is None else str(model),
                                         camera_model=self.link.camera_model, on_user_ray=on_ray,
                                         primary_id=primary_id)
            self.vision.start()
            self.health["camera"] = "ok"
        except Exception as exc:  # noqa: BLE001
            self.vision = None
            self.health["camera"] = f"failed: {exc}"
            self._note("camera", exc)

    def _wait_first_frame(self, timeout: float) -> None:
        """카메라가 실제로 프레임을 주는지 확인한다. 안 주면 '카메라 실패'로 보고 얼굴 프리셋으로 진행
        (media 없이 연결된 경우, WebRTC 영상이 안 오는 경우)."""
        media = getattr(self.link.mini, "media", None)
        if getattr(media, "camera", None) is None:
            raise RuntimeError("camera not available (media_backend no_media or daemon without media)")
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if media.get_frame() is not None:
                return
            time.sleep(0.05)
        raise RuntimeError(f"no camera frame within {timeout:g} s")

    @property
    def camera_ok(self) -> bool:
        return self.vision is not None

    def close(self) -> None:
        for part in ("vision", "speech"):
            obj = getattr(self, part)
            if obj is not None:
                try:
                    obj.close()
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s close failed: %s", part, exc)
        if self.player is not None:
            try:
                handle = self.player.park()  # 끝날 때는 기본 자세로 (HW4 검사 전이 + 시선 레이어도 REST 유지)
                if handle is not None:
                    handle.wait(4.0)
                time.sleep(0.5)
                self.player.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("motion close failed: %s", exc)
        if self.cfg["robot"].get("sleep_on_exit", False) and self.link.mini is not None:
            try:
                self.link.goto_sleep()
            except Exception as exc:  # noqa: BLE001
                log.warning("goto_sleep failed: %s", exc)
        self.link.close()

    # ---- 모션 이벤트 (모션 스레드 → 이벤트 루프) ------------------------------------------
    def _on_motion_event(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        fields = {k: event.get(k) for k in ("key", "label", "at_s", "duration_s", "offset_s", "reason") if
                  event.get(k) is not None}
        self.bus.log_threadsafe(f"robot.motion_{kind}", **fields)
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._dispatch_motion_event, dict(event))

    def _dispatch_motion_event(self, event: dict[str, Any]) -> None:
        handle = self._handles.get(int(event.get("id", -1)))
        if handle is None:
            return
        handle.on_event(event)
        if event.get("kind") in TERMINAL_KINDS:
            self._handles.pop(int(event["id"]), None)

    # ---- RobotOutputs: 모션 ---------------------------------------------------------
    def play_motion(self, name: str, variant: str = "expressive") -> RealMotionHandle | Any:
        if self.player is None:  # 모션이 죽었으면 시간만 흉내 내서 화면·음성 동기화가 계속 돌게
            return self._fallback.play_motion(name, variant)
        from reachy_motion.service import Priority

        labels: set[str] = set()
        try:
            spec = self.player._library().get(name, variant)
            labels = {k.label for k in spec.effective().track("head") if k.label}
        except Exception:  # noqa: BLE001 - play()가 같은 이유로 거절하고 이벤트를 낸다
            pass
        inner = self.player.play(name, variant, priority=Priority.NORMAL, interrupt=True)
        handle = RealMotionHandle(name, variant, inner, labels)
        if inner.done:  # 바로 거절됨 (모션 이름 오류 등)
            handle.on_event({"kind": "rejected"})
        else:
            self._handles[inner.id] = handle
        return handle

    def stop_motion(self, blend_to_rest: bool = True) -> None:
        # 시선 레이어가 마지막 명령 자세에서 부드럽게 이어받으므로 REST까지 갔다 오지 않는다
        if self.player is not None:
            self.player.stop(blend_to_rest=False)
            self.bus.log.log("robot.motion_stop", blend_to_rest=blend_to_rest)

    def set_idle_motion(self, name: str | None, after_s: float = 20.0) -> None:
        if self.player is None:
            return
        if name:
            self.player.set_idle(name, self.variant, after_s)
        else:
            self.player.cancel_idle_motion()
        self.bus.log.log("robot.idle_motion", motion=name, after_s=after_s)

    # ---- RobotOutputs: 시선·자세 --------------------------------------------------------
    def gaze(self, target: str) -> None:
        if target != self.gaze_target:
            self.gaze_target = target
            self.bus.log.log("robot.gaze", target=target)
        if self.player is not None:
            self.player.set_gaze(target)

    async def glance(self, target: str, hold_s: float) -> None:
        self.bus.log.log("robot.glance", target=target, hold_s=hold_s)
        if self.player is not None:
            self.player.glance(target, hold_s)
        await asyncio.sleep(hold_s + 0.4)

    def posture(self, name: str) -> None:
        if name != self.posture_name:
            self.posture_name = name
            self.bus.log.log("robot.posture", posture=name)
        if self.player is not None:
            self.player.set_posture(name)

    def gesture(self, name: str) -> None:
        self.bus.log.log("robot.gesture", gesture=name)
        if self.player is not None:
            self.player.gesture(name)

    # ---- RobotOutputs: 음성 ---------------------------------------------------------
    async def speak(self, text: str, lang: str) -> None:
        if self.speech is None:
            await self._fallback.speak(text, lang)
            return
        await self.speech.speak(text, lang)

    def stop_speech(self) -> None:
        if self.speech is None:
            self._fallback.stop_speech()
        else:
            self.speech.stop_speech()

    def is_speaking(self) -> bool:
        return self.speech.is_speaking() if self.speech is not None else self._fallback.is_speaking()

    def listen_context(self, context: str) -> None:
        self.context = context
        if self.speech is not None:
            self.speech.listen_context(context)

    def set_lang(self, lang: str) -> None:
        if self.speech is not None:
            self.speech.set_lang(lang)

    def status(self) -> dict[str, Any]:
        out = dict(self.health)
        if self.speech is not None:
            out["asr_ready"] = self.speech.ready.get("asr")
            out["tts_ready"] = self.speech.ready.get("tts")
        if self.vision is not None:
            out["camera_fps"] = round(self.vision.fps_measured, 1)
        if self.player is not None:
            out["motion_running"] = self.player.status().get("running")
        return out
