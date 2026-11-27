"""BehaviorManager 타이밍: 모션 라벨에 맞춘 화면·음성, 기능 플래그, 바지인."""

from __future__ import annotations

import asyncio
from typing import Any

from reachy_cafe.behavior.acts import Act
from reachy_cafe.behavior.manager import BehaviorManager
from reachy_cafe.behavior.mock_robot import MockRobot
from reachy_cafe.bus import EventLog
from reachy_cafe.dialogue.menu import Menu
from reachy_cafe.dialogue.nlg import NLG

SCALE = 0.05  # mock 시간을 20배 빠르게


class FakeHub:
    def __init__(self) -> None:
        self.views: list[dict[str, Any]] = []
        self.statuses: list[dict[str, Any]] = []
        self.leds: list[str] = []

    def set_view(self, view: dict[str, Any]) -> None:
        self.views.append(view)

    def set_status(self, status: dict[str, Any]) -> None:
        self.statuses.append(status)

    def set_led(self, mode: str) -> None:
        self.leds.append(mode)


def make(cfg: dict[str, Any], menu: Menu) -> tuple[BehaviorManager, MockRobot, FakeHub, list[dict[str, Any]]]:
    log = EventLog(None)
    records: list[dict[str, Any]] = []
    log.add_listener(records.append)
    robot = MockRobot(log, time_scale=SCALE)
    hub = FakeHub()
    behavior = BehaviorManager(cfg, robot, hub, NLG(menu), log)  # type: ignore[arg-type]
    return behavior, robot, hub, records


def recommend_act() -> Act:
    return Act("recommend", motion="aha_recommend", screen="recommend", data={"item": "hibiscus_tea"},
               say="recommend", slots={"item": "hibiscus_tea", "reasons": ["less_sweet"]}, listen="yesno")


def test_screen_and_speech_wait_for_the_present_keyframe(cfg: dict[str, Any], menu: Menu) -> None:
    async def run() -> None:
        behavior, _robot, hub, records = make(cfg, menu)
        behavior.perform(recommend_act())
        await asyncio.sleep(0.5 * 1.7 * SCALE)              # present(1.7 s) 전
        assert not hub.views
        await asyncio.sleep(3.0 * SCALE)
        assert hub.views[-1]["screen"] == "recommend"
        kinds = [r["kind"] for r in records]
        assert kinds.index("robot.motion_label") < kinds.index("say")
        labels = [r["label"] for r in records if r["kind"] == "robot.motion_label"]
        assert "present" in labels
        await asyncio.sleep(10 * SCALE)
    asyncio.run(run())


def test_motion_flag_off_shows_screen_immediately(cfg: dict[str, Any], menu: Menu) -> None:
    async def run() -> None:
        behavior, _robot, hub, records = make(cfg, menu)
        behavior.set_feature("motion", False)
        behavior.perform(recommend_act())
        assert hub.views[-1]["screen"] == "recommend"         # 모션을 기다리지 않음
        await asyncio.sleep(0.01)
        assert not any(r["kind"] == "robot.motion" for r in records)
        await asyncio.sleep(10 * SCALE)
    asyncio.run(run())


def test_barge_in_cuts_speech_and_motion(cfg: dict[str, Any], menu: Menu) -> None:
    async def run() -> None:
        behavior, robot, _hub, records = make(cfg, menu)
        behavior.perform(Act("long", motion="failure", say="sold_out",
                             slots={"item": "einspanner", "alts": ["vanilla_latte", "cafe_mocha"]}))
        await asyncio.sleep(1.0 * SCALE)                       # slump(0.2 s) 뒤 말하는 중
        assert robot.is_speaking()
        assert behavior.on_user_speech_start() is True
        await asyncio.sleep(0.01)
        assert not robot.is_speaking()
        kinds = [r["kind"] for r in records]
        assert "bargein" in kinds and "robot.motion_cancel" in kinds
    asyncio.run(run())


def test_led_respects_flag(cfg: dict[str, Any], menu: Menu) -> None:
    async def run() -> None:
        behavior, _robot, hub, _records = make(cfg, menu)
        behavior.perform(Act("pay", led="blink"))
        behavior.set_feature("slot_led", False)
        behavior.perform(Act("pay", led="blink"))
        assert hub.leds == ["blink", "off", "off"]
    asyncio.run(run())


# ---- 인사 모션 (greeting) ----------------------------------------------------------

def greet_act(**kw: Any) -> Act:
    return Act("greet", say="greet", screen="welcome", data={}, motion="greeting", listen="open", **kw)


def timeline_order(records: list[dict[str, Any]]) -> list[str]:
    """모션 라벨·화면·말하기를 일어난 순서대로 (라벨 이름, 화면 이름, "say")."""
    return [r.get("label") or r.get("screen") or r["kind"] for r in records
            if r["kind"] in ("robot.motion_label", "view", "say")]


def test_greeting_line_and_welcome_screen_start_with_the_bow(cfg: dict[str, Any], menu: Menu) -> None:
    async def run() -> None:
        behavior, _robot, hub, records = make(cfg, menu)
        hub.set_view = lambda view: records.append({"kind": "view", "screen": view["screen"]})  # type: ignore
        behavior.perform(greet_act())
        assert not any(r["kind"] == "view" for r in records)  # 모션을 기다린다
        await asyncio.sleep(4.0 * SCALE)
        order = timeline_order(records)
        assert order.index("notice") < order.index("bow") < order.index("welcome") < order.index("say")
        await asyncio.sleep(10 * SCALE)
    asyncio.run(run())


def test_screen_first_greeting_shows_the_menu_at_once_but_waits_to_speak(cfg: dict[str, Any], menu: Menu) -> None:
    async def run() -> None:
        behavior, _robot, hub, records = make(cfg, menu)
        hub.set_view = lambda view: records.append({"kind": "view", "screen": view["screen"]})  # type: ignore
        behavior.perform(Act("menu", say="greet_menu", screen="menu", data={}, motion="greeting", screen_first=True))
        assert records[-1] == {"kind": "view", "screen": "menu"}  # 터치에 바로 반응
        await asyncio.sleep(4.0 * SCALE)
        order = timeline_order(records)
        assert order.index("menu") < order.index("bow") < order.index("say")
        assert order.count("menu") == 1                            # 라벨 뒤에 화면을 다시 그리지 않는다
        await asyncio.sleep(10 * SCALE)
    asyncio.run(run())


def test_greeting_without_motion_speaks_at_once(cfg: dict[str, Any], menu: Menu) -> None:
    async def run() -> None:
        behavior, _robot, hub, records = make(cfg, menu)
        behavior.set_feature("motion", False)                      # with/without 비교의 '모션 없음' 조건
        behavior.perform(greet_act())
        assert hub.views[-1]["screen"] == "welcome"
        await asyncio.sleep(0.01)
        assert any(r["kind"] == "say" for r in records)
        assert not any(r["kind"] == "robot.motion" for r in records)
        await asyncio.sleep(10 * SCALE)
    asyncio.run(run())


def test_screen_of_an_interrupted_motion_act_still_appears(cfg: dict[str, Any], menu: Menu) -> None:
    """Aha! 키프레임 전에 되묻기가 끼어도 추천 화면은 뜬다 (예전: FSM은 추천 중인데 화면은 메뉴)."""
    async def run() -> None:
        behavior, _robot, hub, _records = make(cfg, menu)
        behavior.perform(recommend_act())
        await asyncio.sleep(0.8 * SCALE)                           # present(1.7 s) 전
        behavior.perform(Act("repair", say="repair", posture="tilt", listen="open"))
        assert [v["screen"] for v in hub.views] == ["recommend"]
        await asyncio.sleep(6 * SCALE)
        assert [v["screen"] for v in hub.views] == ["recommend"]  # 나중에 다시 그리지 않는다
    asyncio.run(run())


def test_new_screen_replaces_a_pending_one_without_showing_it(cfg: dict[str, Any], menu: Menu) -> None:
    async def run() -> None:
        behavior, _robot, hub, _records = make(cfg, menu)
        behavior.perform(recommend_act())
        await asyncio.sleep(0.8 * SCALE)
        behavior.perform(Act("menu", say="menu_prompt", screen="menu", data={}))
        await asyncio.sleep(6 * SCALE)
        assert [v["screen"] for v in hub.views] == ["menu"]
    asyncio.run(run())


def test_idle_act_stops_speech_and_drops_the_pending_screen(cfg: dict[str, Any], menu: Menu) -> None:
    """리셋·손님이 떠남: 하던 말을 끊고, 대기 화면 위에 추천 화면이 나중에 덮이지 않는다."""
    async def run() -> None:
        behavior, robot, hub, _records = make(cfg, menu)
        behavior.perform(Act("explain", say="explain_item", slots={"item": "latte"}, listen="yesno"))
        await asyncio.sleep(0.3 * SCALE)
        assert robot.is_speaking()
        behavior.perform(recommend_act())
        await asyncio.sleep(0.5 * SCALE)
        behavior.perform(Act("idle", screen="attract", data={}, clear_overlay=True, posture="neutral",
                             gaze="wander", led="off", plan="waiting", interrupt=True))
        await asyncio.sleep(0.01)
        assert not robot.is_speaking()
        await asyncio.sleep(6 * SCALE)
        assert [v["screen"] for v in hub.views] == ["attract"]
    asyncio.run(run())


def test_condition_a_keeps_the_head_still_and_restores_on_return(cfg: dict[str, Any], menu: Menu) -> None:
    async def run() -> None:
        behavior, robot, _hub, _records = make(cfg, menu)
        behavior.set_feature("motion", False)                      # 조건 A
        behavior.set_feature("gaze", False)
        assert robot.posture_name == "neutral" and robot.gaze_target == "rest"
        behavior.perform(Act("offer_help", posture="tilt", gaze="user", gesture="nod"))
        assert robot.posture_name == "neutral" and robot.gaze_target == "rest"   # 예전: 갸웃은 그대로 움직임
        behavior.set_feature("motion", True)                       # 조건 B로 돌아오면 마지막 요청으로
        behavior.set_feature("gaze", True)
        assert robot.posture_name == "tilt" and robot.gaze_target == "user"
    asyncio.run(run())


def test_slot_led_comes_back_when_the_flag_is_turned_on(cfg: dict[str, Any], menu: Menu) -> None:
    async def run() -> None:
        behavior, _robot, hub, _records = make(cfg, menu)
        behavior.set_feature("slot_led", False)
        behavior.perform(Act("pay", led="blink"))
        behavior.set_feature("slot_led", True)
        assert hub.leds[-1] == "blink"
    asyncio.run(run())


def test_speech_uses_the_language_of_the_act(cfg: dict[str, Any], menu: Menu) -> None:
    """세션이 끝나 대기(한국어)로 돌아간 뒤에도 작별 인사는 손님의 언어로."""
    async def run() -> None:
        behavior, _robot, _hub, records = make(cfg, menu)
        behavior.perform(Act("cancelled", say="cancelled", lang="en"))
        await asyncio.sleep(0.01)
        say = next(r for r in records if r["kind"] == "say")
        assert say["lang"] == "en" and say["text"].startswith("Your order is cancelled")
        await asyncio.sleep(10 * SCALE)
    asyncio.run(run())


def test_staff_notice_shows_at_once_and_idle_clears_it_with_the_captions(cfg: dict[str, Any], menu: Menu) -> None:
    async def run() -> None:
        behavior, _robot, hub, _records = make(cfg, menu)
        behavior.status(heard="견과류 들어가요?", said="알레르기나 성분은…")
        behavior.perform(Act("staff", say="staff_allergy", notice="staff", notice_data={"reason": "allergy"}))
        assert hub.views[-1]["notice"] == {"name": "staff", "data": {"reason": "allergy"}}
        behavior.perform(Act("idle", screen="attract", data={}, notice="none", reset_captions=True, interrupt=True))
        assert hub.views[-1]["notice"] is None
        assert hub.statuses[-1]["heard"] == "" and hub.statuses[-1]["said"] == ""   # 다음 손님에게 남지 않게
        await asyncio.sleep(0.01)

    asyncio.run(run())


def test_cart_sentence_reads_notes_and_take_out(menu: Menu) -> None:
    nlg = NLG(menu)
    lines = [{"item": "latte", "qty": 1, "temp": "ice", "size": "regular", "notes": ["less_ice"]}]
    assert nlg.text("confirm_cart", "ko", lines=lines, total=4500, dine="togo") == \
        "아이스 카페라떼 한 잔(얼음 적게), 4,500원이에요. 포장으로 주문할까요?"
    assert nlg.text("confirm_cart", "en", lines=lines, total=4500, dine="togo") == \
        "One iced Caffè Latte (less ice). That's 4,500 won in total. Shall I place the order to go?"
    assert nlg.text("ask_temp_item", "ko", item="latte") == "카페라떼는 따뜻하게 드릴까요, 아이스로 드릴까요?"
    assert nlg.text("remove_which", "ko", item="latte") == "어떤 카페라떼를 뺄까요? 화면에서 ✕를 눌러 주셔도 돼요."
    assert nlg.text("sold_out_resume", "ko", item="einspanner", alts=["vanilla_latte", "cafe_mocha"]) ==         "비슷한 바닐라 라떼나 카페모카는 어떠세요?"


def test_plan_line_follows_the_language(cfg: dict[str, Any], menu: Menu) -> None:
    async def run() -> None:
        behavior, _robot, hub, _records = make(cfg, menu)
        behavior.perform(Act("menu", screen="menu", data={}, plan="browsing"))
        assert hub.statuses[-1]["plan"] == "메뉴 고르는 중"
        behavior.set_lang("en")                                    # 메뉴 화면에서 English를 누름
        assert hub.statuses[-1]["plan"] == "Customer is choosing"
        behavior.perform(Act("recommend", plan="recommend", plan_slots={"item": "latte"}))
        behavior.set_lang("ko")
        assert hub.statuses[-1]["plan"] == "추천: 카페라떼"
    asyncio.run(run())
