"""자연스러움 개선과 직원 안내 (11/26): 규칙으로 처리하고, 규칙 밖의 상황은 솔직하게 말한 뒤 직원을 부른다."""

from __future__ import annotations

from typing import Any

import pytest

from conftest import Harness


def start(h: Harness) -> None:
    h.human("ENGAGED")      # 인사
    h.touch("start")        # 메뉴


def to_cart_with_hot_latte(h: Harness) -> None:
    start(h)
    h.say("따뜻한 라떼 하나 주세요")
    h.say("레귤러요")
    assert h.state == "CART"


def lines(h: Harness) -> list[dict[str, Any]]:
    assert h.fsm.session is not None
    return [line.to_dict() for line in h.fsm.session.cart.lines]


# ---- 직원 안내 ---------------------------------------------------------------------------

def test_three_misheard_turns_call_staff_once_then_staff_resumes(h: Harness) -> None:
    """되묻기 → 화면 권유 → 세 번째에 직원 호출. 같은 말을 끝없이 되풀이하지 않는다."""
    start(h)
    for _ in range(3):
        h.say("음음")
    assert h.out.says()[-3:] == ["repair", "repair_touch", "staff_repair"]
    staff = h.out.acts[-1]
    assert staff.notice == "staff" and h.state == "MENU"           # 주문 상태는 그대로
    h.say("음음")
    assert h.out.says().count("staff_repair") == 1                 # 두 번 부르지 않는다
    h.fsm.staff_done(h.now)
    done, again = h.out.acts[-2:]
    assert done.say == "staff_thanks" and done.notice == "none"
    assert again.say == "menu_prompt" and again.interrupt is False          # 감사 인사가 끝난 뒤 하던 질문을 다시
    assert h.fsm.session is not None and not h.fsm.session.staff_called


def test_using_the_screen_resets_the_repair_ladder(h: Harness) -> None:
    start(h)
    h.say("음음")
    h.say("음음")
    h.touch("recommend")                                            # 화면을 쓰고 있다
    h.say("음음")
    assert h.out.says()[-1] == "repair" and "staff_repair" not in h.out.says()


def test_auto_staff_flag_off_keeps_suggesting_the_screen(h: Harness) -> None:
    h.fsm.features["auto_staff"] = False
    start(h)
    for _ in range(3):
        h.say("음음")
    assert h.out.says()[-1] == "repair_touch" and "staff_repair" not in h.out.says()


@pytest.mark.parametrize(("text", "say"), [
    ("직원 불러 주세요", "staff_request"),
    ("견과류 들어가요?", "staff_allergy"),
    ("현금으로 해도 돼요?", "staff_payment"),
    ("환불하고 싶어요", "staff_refund"),
    ("화장실 어디예요?", "staff_facility"),
    ("Does it have nuts?", "staff_allergy"),
])
def test_things_the_kiosk_must_not_answer_go_to_staff(h: Harness, text: str, say: str) -> None:
    """알레르기·현금·환불·매장 안내는 지어내서 답하지 않고 솔직하게 말한 뒤 직원을 부른다."""
    start(h)
    h.say(text)
    act = h.out.acts[-1]
    assert act.say == say and act.notice == "staff"
    assert h.state == "MENU"


def test_staff_button_and_second_request(h: Harness) -> None:
    start(h)
    h.touch("staff")
    assert h.out.acts[-1].say == "staff_request"
    h.say("직원 아직이에요?")
    assert h.out.acts[-1].say == "staff_already"


def test_customer_waiting_for_staff_is_not_nagged(h: Harness) -> None:
    start(h)
    h.touch("staff")
    h.human("NO_RESPONSE", "hesitation")
    assert h.state == "MENU" and "offer_help" not in h.out.says()


def test_chatter_mentioning_staff_with_nobody_present_is_ignored(h: Harness) -> None:
    before = len(h.out.acts)
    h.say("직원 불러 주세요")
    h.say("화장실 어디예요?")
    assert h.state == "IDLE" and len(h.out.acts) == before


def test_session_end_clears_the_staff_notice_and_captions(h: Harness) -> None:
    start(h)
    h.touch("staff")
    h.fsm.reset(h.now)
    idle = h.out.acts[-1]
    assert idle.name == "idle" and idle.notice == "none" and idle.reset_captions


# ---- 시작과 끝 ---------------------------------------------------------------------------

def test_hesitating_after_the_greeting_is_led_to_the_menu(h: Harness) -> None:
    h.cues(zone="personal")
    h.human("PRESENT")
    assert h.state == "GREET"
    h.advance(10.1)
    assert h.state == "MENU"
    act = h.out.last_say()
    assert act.say == "menu_lead" and act.glance == "screen"


def test_hesitation_event_in_greet_also_leads_to_the_menu(h: Harness) -> None:
    h.cues(zone="personal")
    h.human("PRESENT")
    h.human("NO_RESPONSE", "hesitation")
    assert h.state == "MENU" and h.out.last_say().say == "menu_lead"


def test_looking_at_the_robot_after_a_help_offer_starts_a_recommendation(h: Harness) -> None:
    start(h)
    h.human("NO_RESPONSE", "hesitation")
    assert h.state == "OFFER_HELP"
    h.human("ENGAGED")
    h.human("NO_RESPONSE", "seek_help")                            # "추천해 드릴까요?" 뒤 말없이 로봇을 봄
    assert h.state == "RECOMMEND" and h.out.last_say().say == "ask_preference"


# ---- 다시 말하기·추천 ------------------------------------------------------------------------

def test_repeat_says_the_last_question_again(h: Harness) -> None:
    start(h)
    h.say("라떼 주세요")
    assert h.out.last_say().say == "ask_temp"
    h.say("뭐라고요?")
    assert h.out.last_say().say == "ask_temp" and h.state == "OPTIONS"


def test_hello_in_the_middle_does_not_say_other_menu(h: Harness) -> None:
    start(h)
    h.say("안녕하세요")
    assert h.out.last_say().say == "menu_prompt"


def test_chip_taps_change_the_card_without_aha_and_speak_once(h: Harness) -> None:
    start(h)
    h.say("추천해 주세요")
    assert h.out.last_say().say == "ask_preference"
    motions, says = len(h.out.motions()), len(h.out.says())
    h.touch("pref", key="caffeine_free")
    h.touch("pref", key="less_sweet")
    assert len(h.out.motions()) == motions and len(h.out.says()) == says   # 카드만 바뀜
    assert h.out.acts[-1].screen == "recommend"
    h.advance(1.6)                                                   # 손이 멈춤 → 한 번 말함
    assert h.out.says()[says:] == ["recommend"] and len(h.out.motions()) == motions
    spoken = h.fsm.session.rec_item if h.fsm.session else None
    h.touch("pref", key="less_sweet")                                # 껐다가
    h.touch("pref", key="less_sweet")                                # 다시 켬 → 같은 메뉴
    h.advance(1.6)
    assert h.fsm.session is not None and h.fsm.session.rec_item == spoken
    assert h.out.says()[says:] == ["recommend"]                      # 같은 메뉴를 다시 말하지 않는다


def test_said_temperature_is_not_asked_again_in_options(h: Harness) -> None:
    start(h)
    h.say("따뜻한 거 추천해 주세요")
    assert h.state == "RECOMMEND"
    h.say("네")
    assert h.state == "OPTIONS" and h.out.last_say().say == "ask_size"
    assert h.fsm.session is not None and h.fsm.session.pending is not None
    assert h.fsm.session.pending.temp == "hot"


def test_that_one_orders_the_recommendation(h: Harness) -> None:
    start(h)
    h.say("커피 말고 상큼한 거요")
    rec = h.fsm.session.rec_item if h.fsm.session else None
    h.say("그거 주세요")
    assert h.fsm.session is not None and h.fsm.session.pending is not None
    assert h.fsm.session.pending.item_id == rec


def test_rejecting_the_recommended_drink_by_name_recommends_another(h: Harness, menu: Any) -> None:
    start(h)
    h.say("커피 말고 상큼한 거요")
    rec = h.fsm.session.rec_item if h.fsm.session else None
    h.say(f"{menu.get(rec).name['ko']} 말고요")
    assert h.state == "RECOMMEND" and h.fsm.session is not None
    assert h.fsm.session.rec_item not in (None, rec) and rec in h.fsm.session.rejected


def test_comparison_question_recommends_one_of_the_two(h: Harness) -> None:
    start(h)
    h.say("딸기 라떼랑 바닐라 라떼 중에 뭐가 나아요?")
    assert h.state == "RECOMMEND" and h.fsm.session is not None
    assert h.fsm.session.rec_item in ("strawberry_latte", "vanilla_latte")


# ---- 가리키는 말·순서 ------------------------------------------------------------------------

def test_ordinal_picks_a_sold_out_alternative(h: Harness) -> None:
    start(h)
    h.say("아인슈페너 주세요")
    assert h.state == "SOLD_OUT"
    alternatives = list(h.fsm.session.alternatives) if h.fsm.session else []
    h.say("두 번째 걸로")
    assert h.fsm.session is not None and h.fsm.session.pending is not None
    assert h.fsm.session.pending.item_id == alternatives[1]


def test_ordinal_picks_from_the_repair_candidates(h: Harness) -> None:
    start(h)
    h.say("바닐")
    assert h.overlay == "CANDIDATES"
    first = h.fsm.session.candidates[0] if h.fsm.session else None
    h.say("첫 번째요")
    assert h.fsm.session is not None and h.fsm.session.pending is not None
    assert h.fsm.session.pending.item_id == first


def test_not_this_but_that_orders_the_second_drink(h: Harness) -> None:
    start(h)
    h.say("아메리카노 말고 라떼요")
    assert h.fsm.session is not None and h.fsm.session.pending is not None
    assert h.fsm.session.pending.item_id == "latte" and "americano" in h.fsm.session.rejected


# ---- 여러 잔·요청사항·포장 --------------------------------------------------------------------

def test_two_drinks_in_one_sentence_are_both_ordered(h: Harness) -> None:
    start(h)
    h.say("따뜻한 라떼 하나랑 아이스 아메리카노 두 잔")
    ask = h.out.last_say()
    assert ask.say == "ask_size_item" and ask.slots == {"item": "latte"}   # 어느 잔인지 이름을 붙여 묻는다
    h.say("레귤러요")
    ask = h.out.last_say()
    assert ask.say == "ask_size_item" and ask.slots == {"item": "americano"}
    h.say("라지요")
    assert h.state == "CART"
    assert lines(h) == [{"item": "latte", "qty": 1, "temp": "hot", "size": "regular"},
                        {"item": "americano", "qty": 2, "temp": "ice", "size": "large"}]


def test_note_and_take_out_stay_on_the_order(h: Harness) -> None:
    start(h)
    h.say("아이스 라떼 얼음 적게 포장이요")
    h.say("레귤러요")
    assert h.state == "CART"
    assert lines(h) == [{"item": "latte", "qty": 1, "temp": "ice", "size": "regular", "notes": ["less_ice"]}]
    cart = h.out.last_say()
    assert cart.say == "confirm_cart" and cart.slots["dine"] == "togo"


def test_note_alone_is_acknowledged_then_the_question_continues(h: Harness) -> None:
    start(h)
    h.say("아이스 라떼 주세요")
    assert h.out.last_say().say == "ask_size"
    h.say("얼음 적게 해 주세요")
    assert h.out.says()[-2:] == ["note_ok", "ask_size"]
    assert h.out.acts[-1].interrupt is False                          # 받아 적었다는 말이 끝난 뒤 다시 묻는다
    assert h.fsm.session is not None and h.fsm.session.pending is not None
    assert h.fsm.session.pending.notes == ["less_ice"]


def test_to_go_is_not_a_removal_but_take_out_the_latte_is(h: Harness) -> None:
    to_cart_with_hot_latte(h)
    h.say("to go please")
    assert h.out.says()[-2:] == ["dine_togo", "confirm_cart"] and len(lines(h)) == 1
    h.say("take out the latte")
    assert h.state == "MENU" and h.out.last_say().say == "cart_empty"


# ---- 장바구니에서 이름으로 고치기 -------------------------------------------------------------

def test_change_by_name_edits_the_line_instead_of_adding(h: Harness) -> None:
    to_cart_with_hot_latte(h)
    h.say("라떼는 아이스로 바꿔 주세요")
    assert h.state == "CART" and lines(h) == [{"item": "latte", "qty": 1, "temp": "ice", "size": "regular"}]


def test_note_for_a_drink_in_the_cart_is_added_to_that_line(h: Harness) -> None:
    to_cart_with_hot_latte(h)
    h.say("라떼는 시럽 추가해 주세요")
    assert h.state == "CART"
    assert lines(h) == [{"item": "latte", "qty": 1, "temp": "hot", "size": "regular", "notes": ["extra_syrup"]}]


def test_one_more_adds_to_the_same_line(h: Harness) -> None:
    to_cart_with_hot_latte(h)
    h.say("라떼 하나 더")
    assert lines(h) == [{"item": "latte", "qty": 2, "temp": "hot", "size": "regular"}]


def test_replace_the_only_drink_by_name(h: Harness) -> None:
    to_cart_with_hot_latte(h)
    h.say("라떼 말고 아메리카노로 해 주세요")
    assert h.state == "CART" and lines(h) == [{"item": "americano", "qty": 1, "temp": "hot", "size": "regular"}]


def two_lattes(h: Harness) -> None:
    to_cart_with_hot_latte(h)
    h.say("아이스 라떼 하나 주세요")
    h.say("레귤러요")
    assert [line["temp"] for line in lines(h)] == ["hot", "ice"]


def test_remove_by_name_asks_which_when_two_lines_match(h: Harness) -> None:
    two_lattes(h)
    h.say("라떼 빼 주세요")
    assert h.out.last_say().say == "remove_which" and len(lines(h)) == 2   # 예전에는 둘 다 빠졌다
    h.say("아이스요")
    assert [line["temp"] for line in lines(h)] == ["hot"]


def test_quick_double_tap_on_remove_deletes_only_one_line(h: Harness) -> None:
    two_lattes(h)
    assert h.fsm.session is not None
    sign = h.fsm.session.cart.lines[0].signature()
    h.touch("remove", index=0, value=sign)
    h.touch("remove", index=0, value=sign)                           # 다시 그리기 전의 두 번째 터치
    assert [line["temp"] for line in lines(h)] == ["ice"]


# ---- 언어·카드 슬롯 ------------------------------------------------------------------------

def test_language_button_mid_order_repeats_the_question_in_the_new_language(h: Harness) -> None:
    start(h)
    h.touch("lang", lang="en")
    assert h.out.langs[-1] == "en" and h.fsm.lang == "en"
    assert h.out.last_say().say == "menu_prompt"


def test_card_slot_disconnect_offers_screen_payment_at_once(h: Harness) -> None:
    h.fsm.on_device("connected", h.now)
    to_cart_with_hot_latte(h)
    h.say("네")
    assert h.state == "PAY_WAIT_CARD" and h.out.last_say().say == "pay_request"
    h.fsm.on_device("disconnected", h.now)
    act = h.out.acts[-1]
    assert act.say == "pay_slot_down" and act.data is not None and act.data["fallback"] is True


def test_payment_after_the_slot_is_gone_starts_with_screen_payment(h: Harness) -> None:
    h.fsm.on_device("connected", h.now)
    h.fsm.on_device("disconnected", h.now)
    to_cart_with_hot_latte(h)
    h.say("네")
    act = h.out.last_say()
    assert act.say == "pay_slot_down" and act.data is not None and act.data["fallback"] is True
