"""멀티모달 액트: FSM은 "무엇을 할지"만 정하고, BehaviorManager가 채널별 실행과 타이밍을 맞춘다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Act:
    name: str
    say: str | None = None                  # NLG 문장 키
    slots: dict[str, Any] = field(default_factory=dict)
    lang: str | None = None                 # 이 말을 할 언어 (None = 지금 언어). 세션이 끝난 뒤의 작별 인사 등
    screen: str | None = None               # 키오스크 화면 이름
    data: dict[str, Any] | None = None
    overlay: str | None = None              # 화면 위 오버레이 (paused, one_at_a_time, candidates)
    overlay_data: dict[str, Any] | None = None
    clear_overlay: bool = False
    motion: str | None = None               # 모션 이름 (HW4: aha_recommend, boredom, success, failure / 카페: greeting)
    sync_label: str | None = None           # 이 키프레임이 시작될 때 화면·음성을 띄움 (None = 설정값)
    screen_first: bool = False              # 모션이 있어도 화면은 바로 바꾸고 음성만 키프레임을 기다림 (터치 직후 인사)
    glance: str | None = None               # 말하기 전에 잠깐 볼 곳: screen | slot
    glance_hold_s: float = 1.0
    gaze: str | None = None                 # 계속 볼 대상: user | wander
    posture: str | None = None              # neutral | listening | thinking | perk_up | tilt
    gesture: str | None = None              # 짧은 절차형 동작: antenna_wiggle | nod (MotionPlayer가 set_target으로 생성)
    led: str | None = None                  # on | off | blink (카드 슬롯 LED)
    plan: str | None = None                 # 상태 스트립에 보여줄 "로봇의 계획" (nlg.PLANS 키)
    plan_slots: dict[str, Any] = field(default_factory=dict)
    listen: str | None = None               # 음성 인식 문맥: open | yesno (상태별 endpoint)
    interrupt: bool | None = None           # None이면 음성·모션이 있을 때 이전 액트를 끊는다
    notice: str | None = None               # 화면 아래 알림 띠: staff(직원 호출) | none(지움) | None(그대로)
    notice_data: dict[str, Any] | None = None
    reset_captions: bool = False            # 상태 띠의 들은 말·로봇 말을 지움 (다음 손님에게 남지 않게)

    def interrupts(self) -> bool:
        return self.interrupt if self.interrupt is not None else bool(self.say or self.motion)

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"act": self.name}
        for key in ("say", "screen", "overlay", "motion", "gesture", "glance", "gaze", "posture", "led", "plan",
                    "listen", "notice"):
            value = getattr(self, key)
            if value:
                out[key] = value
        if self.slots:
            out["slots"] = self.slots
        if self.clear_overlay:
            out["clear_overlay"] = True
        if self.screen_first:
            out["screen_first"] = True
        return out
