from __future__ import annotations

from typing import Any

import pytest

from reachy_cafe.behavior.acts import Act
from reachy_cafe.bus import Event
from reachy_cafe.config import CONFIG_DIR, load_config
from reachy_cafe.dialogue.fsm import InteractionManager
from reachy_cafe.dialogue.menu import Menu, OrderCounter
from reachy_cafe.dialogue.nlu import NLU
from reachy_cafe.robot.hwpaths import ensure_hw_paths

# HW3·HW4 코드를 설치하지 않은 환경(데스크톱 묶음)에서도 config [hw] 폴더에서 불러와 로봇 모듈 테스트가 돌게 한다
ensure_hw_paths(load_config(env={}))


@pytest.fixture
def cfg() -> dict[str, Any]:
    return load_config("config/dev_laptop.toml", env={})


@pytest.fixture
def menu() -> Menu:
    return Menu.load(CONFIG_DIR / "menu.json")  # 테스트마다 새로 읽어 재고 변경이 섞이지 않게


@pytest.fixture
def nlu(menu: Menu) -> NLU:
    return NLU(menu)


class Recorder:
    """BehaviorManager 대신 FSM의 출력을 기록한다."""

    def __init__(self) -> None:
        self.acts: list[Act] = []
        self.idle: list[bool] = []
        self.langs: list[str] = []

    def perform(self, act: Act) -> None:
        self.acts.append(act)

    def set_idle(self, enabled: bool) -> None:
        self.idle.append(enabled)

    def set_lang(self, lang: str) -> None:
        self.langs.append(lang)

    def says(self) -> list[str]:
        return [act.say for act in self.acts if act.say]

    def motions(self) -> list[str]:
        return [act.motion for act in self.acts if act.motion]

    def last_say(self) -> Act:
        return next(act for act in reversed(self.acts) if act.say)


class Harness:
    def __init__(self, cfg: dict[str, Any], menu: Menu):
        self.now = 0.0
        self.out = Recorder()
        self.locked = 0
        self.fsm = InteractionManager(cfg, menu, NLU(menu), self.out, OrderCounter(None),
                                      on_lock_primary=self._lock)
        self.fsm.start(self.now)

    def _lock(self) -> None:
        self.locked += 1

    @property
    def state(self) -> str:
        return str(self.fsm.state)

    @property
    def overlay(self) -> str:
        return str(self.fsm.overlay)

    def advance(self, seconds: float, step: float = 0.1) -> None:
        end = self.now + seconds
        while self.now < end - 1e-9:
            self.now = round(self.now + step, 6)
            self.fsm.tick(self.now)

    def human(self, state: str, reason: str | None = None) -> None:
        self.fsm.handle(Event("human.state", {"state": state, "reason": reason}), self.now)

    def cues(self, zone: str = "none", approaching: bool = False, attention: str = "none") -> None:
        self.fsm.handle(Event("human.cues", {"zone": zone, "approaching": approaching, "attention": attention}),
                        self.now)

    def say(self, text: str) -> None:
        self.fsm.handle(Event("speech.utterance", {"text": text}), self.now)

    def touch(self, action: str, **data: Any) -> None:
        self.fsm.handle(Event("touch", {"action": action, **data}), self.now)

    def card(self, state: str) -> None:
        self.fsm.handle(Event("card", {"state": state}), self.now)


@pytest.fixture
def h(cfg: dict[str, Any], menu: Menu) -> Harness:
    return Harness(cfg, menu)
