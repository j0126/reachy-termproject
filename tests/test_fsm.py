"""데모 시나리오 S1–S3와 실패 처리 F1–F6를 가짜 시계로 끝까지 돌려 본다."""

from __future__ import annotations

from conftest import Harness


def test_s1_full_order(h: Harness) -> None:
    h.human("ENGAGED")
    assert h.state == "GREET" and h.out.says()[-1] == "greet"
    h.say("따뜻한 라떼 하나 주세요")
    assert h.state == "OPTIONS" and h.out.last_say().say == "ask_size"
    h.touch("set_option", key="size", value="regular")
    assert h.state == "CART" and h.out.last_say().say == "confirm_cart"
    assert h.out.last_say().slots["total"] == 4500
    h.say("네")
    assert h.state == "PAY_WAIT_CARD"
    pay = h.out.last_say()
    assert pay.led == "blink" and pay.glance == "slot"
    h.card("in")
    assert h.state == "PAY_PROCESSING"
    h.advance(2.1)
    assert h.state == "PAY_REMOVE_CARD"
    h.card("out")
    assert h.state == "DONE"
    done = h.out.last_say()
    assert done.motion == "success" and done.slots == {"number": 1}
    h.advance(8.1)
    assert h.state == "IDLE"


def test_s2_hesitation_recommendation_loop_in_english(h: Harness) -> None:
    h.human("ENGAGED")
    h.touch("lang", lang="en")
    assert h.state == "MENU" and h.fsm.lang == "en"
    h.human("NO_RESPONSE")
    assert h.state == "OFFER_HELP"
    h.say("Yes, something without coffee.")
    assert h.state == "RECOMMEND"
    first = h.out.last_say()
    assert first.motion == "aha_recommend" and first.slots["item"] == "honey_grapefruit_tea"
    h.say("Hmm, too sweet.")
    second = h.out.last_say()
    assert second.say == "recommend_again" and second.slots["item"] == "hibiscus_tea"
    assert second.slots["reasons"][0] == "less_sweet"
    h.say("Yes, I'll take it")
    assert h.state == "OPTIONS"
    h.say("iced, regular please")
    assert h.state == "CART"


def test_s3_sold_out_pause_resume_and_second_person(h: Harness) -> None:
    h.human("ENGAGED")
    h.say("아인슈페너 주세요")
    assert h.state == "SOLD_OUT"
    sold = h.out.last_say()
    assert sold.motion == "failure" and sold.slots["alts"] == ["vanilla_latte", "cafe_mocha"]

    h.human("ABSENT")                                   # F1: 손님이 떠남
    h.advance(2.1)
    assert h.overlay == "PAUSED" and h.out.last_say().say == "pause"
    h.human("ENGAGED")
    assert h.out.last_say().say == "resume_prompt"
    h.say("네")
    assert h.overlay == "NONE" and h.state == "SOLD_OUT"
    assert h.out.acts[-1].motion is None                # 다시 들어올 때는 Failure를 반복하지 않음

    h.human("AMBIGUOUS")                                # F2: 두 번째 사람
    busy = h.out.last_say()
    assert busy.say == "busy_with_other" and busy.interrupts() is False

    h.say("바닐라 라떼로 할게요")
    assert h.state == "OPTIONS" and h.out.last_say().motion == "success"


def test_s3_live_script_with_the_staff_scene(h: Harness) -> None:
    """S3 라이브 대본 (7장): 품절 → 자리 비움 → 대안 → 알레르기 질문 → 직원(오퍼레이터) 응대 → 결제 → 주문번호."""
    h.human("ENGAGED")
    h.say("아인슈페너 주세요")
    assert h.state == "SOLD_OUT" and h.out.last_say().motion == "failure"
    h.human("ABSENT")
    h.advance(2.1)
    assert h.overlay == "PAUSED"
    h.human("ENGAGED")
    h.say("네")
    assert h.state == "SOLD_OUT" and h.overlay == "NONE"
    assert h.out.last_say().say == "sold_out_resume"     # 돌아오면 사과 없이 대안만 다시
    h.say("바닐라 라떼로 할게요")
    assert h.state == "OPTIONS" and h.out.last_say().say == "ask_temp"
    h.say("혹시 견과류 들어가요?")                         # 로봇은 성분을 지어내지 않고 직원을 부른다
    staff = h.out.acts[-1]
    assert staff.say == "staff_allergy" and staff.notice == "staff" and h.state == "OPTIONS"
    h.advance(5.0)                                       # 직원 역할(오퍼레이터)이 답하는 동안 재촉하지 않음
    assert h.out.acts[-1] is staff
    h.fsm.staff_done(h.now)                              # 오퍼레이터 패널 "응대 완료"
    assert h.out.says()[-2:] == ["staff_thanks", "ask_temp"]
    h.say("아이스 레귤러요")
    assert h.state == "CART"
    h.say("네")
    h.card("in")
    h.advance(2.1)
    h.card("out")
    assert h.state == "DONE"
    assert h.out.motions() == ["greeting", "failure", "success", "success"]


def test_two_people_at_start_need_touch_confirmation(h: Harness) -> None:
    h.human("AMBIGUOUS")
    assert h.state == "GREET" and h.overlay == "ONE_AT_A_TIME"
    h.touch("start")
    assert h.overlay == "NONE" and h.state == "MENU" and h.locked == 1


def test_silence_offers_help_once_then_waits_quietly(h: Harness) -> None:   # F3
    h.human("ENGAGED")
    h.touch("start")
    h.human("NO_RESPONSE")
    assert h.state == "OFFER_HELP"
    h.human("ENGAGED")
    h.human("NO_RESPONSE")
    assert h.state == "MENU" and h.out.acts[-1].say is None
    h.human("ENGAGED")
    h.human("NO_RESPONSE")
    assert h.state == "MENU"                              # 두 번 제안하지 않는다
    assert h.out.says().count("offer_help") == 1


def test_misheard_speech_shows_candidates_then_touch(h: Harness) -> None:     # F4
    h.human("ENGAGED")
    h.touch("start")
    h.say("바닐")
    assert h.overlay == "CANDIDATES" and h.out.last_say().say == "repair_candidates"
    h.touch("candidate", item="vanilla_latte")
    assert h.overlay == "NONE" and h.state == "OPTIONS"


def test_repeated_misrecognition_falls_back_to_touch(h: Harness) -> None:
    h.human("ENGAGED")
    h.touch("start")
    h.say("음음")
    h.say("어어어 그거")
    assert h.out.last_say().say == "repair_touch"


def test_card_out_of_order_and_removed_during_processing(h: Harness) -> None:  # F5
    h.human("ENGAGED")
    h.touch("start")
    h.card("in")
    assert h.out.last_say().say == "not_payment_step" and h.state == "MENU"
    h.card("out")
    h.say("아메리카노 아이스 레귤러")
    assert h.state == "CART"
    h.say("네")
    h.card("in")
    h.advance(0.5)
    h.card("out")
    assert h.state == "PAY_WAIT_CARD"
    retry = h.out.last_say()
    assert retry.say == "card_reinsert" and retry.motion == "failure"


def test_screen_payment_fallback(h: Harness) -> None:
    h.human("ENGAGED")
    h.touch("start")
    h.touch("select_item", item="chamomile_tea")         # hot만 있음 → 사이즈만 묻는다
    assert h.state == "OPTIONS" and h.out.last_say().say == "ask_size"
    h.touch("options_done")
    h.touch("confirm")
    h.advance(20.1)
    assert h.out.last_say().say == "pay_repeat"
    h.touch("pay_fallback")
    h.advance(2.1)
    assert h.state == "DONE"


def test_no_pause_while_card_is_inserted(h: Harness) -> None:
    h.human("ENGAGED")
    h.say("라떼 따뜻하게 레귤러로")
    h.say("네")
    h.card("in")
    h.advance(2.1)
    assert h.state == "PAY_REMOVE_CARD"
    h.human("ABSENT")
    h.advance(5.0)
    assert h.overlay == "NONE"


def test_resume_window_expires(h: Harness) -> None:
    h.human("ENGAGED")
    h.touch("start")
    h.human("ABSENT")
    h.advance(2.1)
    assert h.overlay == "PAUSED"
    h.advance(60.1)
    assert h.state == "IDLE" and h.fsm.session is None


def test_cancel_by_voice(h: Harness) -> None:
    h.human("ENGAGED")
    h.say("라떼 하나")
    h.say("취소할게요")
    assert h.state == "IDLE" and "cancelled" in h.out.says()


def test_customer_who_just_ordered_is_not_greeted_again(h: Harness) -> None:
    """주문을 마친 손님이 그대로 서 있으면 다시 고개 숙여 인사하지 않는다 (예전: 14초 뒤 재인사, 평가 시행 사이에도).
    그 손님이 떠난 뒤 보이는 사람은 새 손님이라 인사한다."""
    test_s1_full_order(h)                                 # 끝나면 IDLE, 사람 상태는 ENGAGED 그대로
    assert h.state == "IDLE"
    h.advance(30.0)
    assert h.state == "IDLE" and h.out.motions().count("greeting") == 1
    h.human("ABSENT")
    h.cues(zone="personal")
    h.human("PRESENT")
    assert h.state == "GREET" and h.out.motions().count("greeting") == 2


def test_regreet_after_a_reset_when_someone_is_still_there(h: Harness) -> None:
    h.human("ENGAGED")
    assert h.state == "GREET"
    h.fsm.reset(h.now)                                    # 오퍼레이터 리셋: 주문을 마친 게 아니므로 다시 인사
    h.advance(6.1)
    assert h.state == "GREET"


def test_greet_leaves_back_to_idle(h: Harness) -> None:
    h.cues(zone="personal")
    h.human("PRESENT")
    assert h.state == "GREET"
    h.human("ABSENT")
    assert h.state == "IDLE"


# ---- 보완 기능 ---------------------------------------------------------------

def test_far_person_is_invited_then_greeted_when_approaching(h: Harness) -> None:
    h.cues(zone="social")
    h.human("PRESENT")
    assert h.state == "IDLE"                              # 멀리 있으면 말 걸지 않고
    invite = h.out.acts[-1]
    assert invite.name == "invite" and invite.gesture == "antenna_wiggle" and invite.say is None
    h.cues(zone="social", approaching=True)               # 다가오면 인사
    assert h.state == "GREET"


def test_passerby_is_invited_only_once_per_cooldown(h: Harness) -> None:
    h.cues(zone="far")
    h.human("PRESENT")
    h.advance(6.5)
    h.advance(1.0)
    assert [a.name for a in h.out.acts].count("invite") == 1
    h.human("ABSENT")
    assert h.out.acts[-1].name == "invite_end"
    assert h.state == "IDLE"


def test_without_invite_flag_present_greets_immediately(h: Harness) -> None:
    h.fsm.features["invite"] = False
    h.human("PRESENT")
    assert h.state == "GREET"


def test_looking_at_robot_gets_a_different_help_offer(h: Harness) -> None:
    h.human("ENGAGED")
    h.touch("start")
    h.human("NO_RESPONSE", reason="seek_help")
    assert h.state == "OFFER_HELP" and h.out.last_say().say == "offer_help_seek"


def test_explain_item_then_order_from_the_card(h: Harness) -> None:
    h.human("ENGAGED")
    h.touch("start")
    h.say("아인슈페너가 뭐예요?")
    assert h.overlay == "DETAIL"
    explain = h.out.last_say()
    assert explain.say == "explain_item" and explain.glance == "screen"
    h.say("바닐라 라떼는 무슨 맛이에요?")                     # 다른 메뉴 설명으로 바꿔도 된다
    assert h.fsm.session.explained_item == "vanilla_latte"
    h.say("네")
    assert h.overlay == "NONE" and h.state == "OPTIONS"


def test_context_talk_greeting_uses_weather(h: Harness) -> None:
    h.fsm.set_context("cold")
    h.human("ENGAGED")
    greet = h.out.last_say()
    assert greet.say == "greet" and greet.slots["weather"] == "cold"
    h.fsm.features["context_talk"] = False
    h.fsm.reset(h.now)
    h.human("ABSENT")
    h.human("ENGAGED")
    assert h.out.last_say().slots == {}


def test_change_option_in_cart_by_voice(h: Harness) -> None:
    h.human("ENGAGED")
    h.say("라떼 따뜻하게 레귤러로")
    assert h.state == "CART"
    h.say("아이스로 바꿔 주세요")
    assert h.state == "CART"
    assert h.fsm.session.cart.lines[-1].temp == "ice"


# ---- 인사 모션 (greeting) ----------------------------------------------------------

def test_arriving_customer_gets_a_bow_with_the_greeting_line(h: Harness) -> None:
    h.human("ENGAGED")
    greet = h.out.last_say()
    assert h.state == "GREET" and greet.say == "greet" and greet.motion == "greeting"
    assert greet.screen == "welcome" and not greet.screen_first  # 환영 화면도 인사말과 함께 bow 키프레임에
    assert greet.gesture is None                                  # 안테나 손 흔들기는 인사 모션 안에 있다


def test_touch_first_customer_gets_the_menu_at_once_and_a_bow(h: Harness) -> None:
    h.touch("start")                                    # 카메라가 알아채기 전에 화면부터 누름
    menu = h.out.last_say()
    assert h.state == "MENU" and menu.say == "greet_menu" and menu.motion == "greeting"
    assert menu.screen_first and menu.glance is None    # 메뉴는 바로, 인사말만 bow를 기다림


def test_language_change_after_the_greeting_does_not_bow_again(h: Harness) -> None:
    h.human("ENGAGED")
    h.touch("lang", lang="en")
    menu = h.out.last_say()
    assert h.state == "MENU" and menu.say == "greet_menu" and menu.motion is None
    assert h.out.motions().count("greeting") == 1


def test_customer_who_says_hello_first_gets_a_bow_back(h: Harness) -> None:
    h.say("안녕하세요")                                  # 카메라가 놓쳐도 먼저 말을 걸면
    menu = h.out.last_say()
    assert h.state == "MENU" and menu.say == "greet_menu" and menu.motion == "greeting"
    assert "greet" not in h.out.says()                    # 인사말을 두 번 하지 않는다


def test_customer_who_orders_first_gets_an_answer_without_a_bow(h: Harness) -> None:
    h.say("따뜻한 라떼 하나 주세요")
    assert h.state == "OPTIONS" and h.out.last_say().say == "ask_size"
    assert "greeting" not in h.out.motions()


def test_next_customer_after_done_is_greeted_with_a_bow_again(h: Harness) -> None:
    h.human("ENGAGED")
    h.fsm.force("DONE", h.now)
    h.advance(8.1)                                        # DONE → IDLE, 주문한 손님이 그대로 서 있음
    h.advance(6.1)
    assert h.state == "IDLE"                              # 같은 손님에게는 다시 인사하지 않음
    h.cues(zone="personal", approaching=True)             # 다음 손님이 다가옴
    assert h.state == "GREET"
    assert h.out.motions().count("greeting") == 2


# ---- 로봇의 계획 (상태 스트립·청중 화면의 '계획' 줄) ------------------------------------

def test_unclear_word_after_an_explanation_shows_the_repair_plan(h: Harness) -> None:
    h.human("ENGAGED")
    h.touch("start")
    h.say("아인슈페너가 뭐예요?")
    assert h.out.acts[-1].plan == "explain"
    h.say("바닐")                                          # 애매한 이름 → 후보 창 (예전: "메뉴 설명"이 남음)
    assert h.overlay == "CANDIDATES" and h.out.acts[-1].plan == "repair_candidates"


def test_closing_the_explanation_returns_to_the_menu_plan(h: Harness) -> None:
    h.human("ENGAGED")
    h.touch("start")
    h.say("아인슈페너가 뭐예요?")
    h.touch("dismiss")
    assert h.overlay == "NONE" and h.out.acts[-1].plan == "browsing"
    h.say("바닐라 라떼는 무슨 맛이에요?")
    h.say("아니요")                                        # 설명을 듣고 거절 → 메뉴 재질문
    assert h.out.acts[-1].say == "menu_again" and h.out.acts[-1].plan == "browsing"


def test_repair_plan_shows_once_then_the_state_plan_returns(h: Harness) -> None:
    h.human("ENGAGED")
    h.touch("start")
    h.say("음음")
    assert h.out.acts[-1].plan == "repair"
    h.say("다시 말해 주세요")                              # 되묻기가 아니라 그 앞의 실제 질문을 반복
    assert h.out.acts[-1].say == "menu_prompt" and h.out.acts[-1].plan == "browsing"


def test_invite_plan_ends_with_waiting_when_the_passerby_leaves(h: Harness) -> None:
    h.cues(zone="social")
    h.human("PRESENT")
    assert h.out.acts[-1].plan == "invite"
    h.human("ABSENT")
    assert h.out.acts[-1].name == "invite_end" and h.out.acts[-1].plan == "waiting"


def test_clearing_all_preferences_switches_the_plan_to_asking(h: Harness) -> None:
    h.human("ENGAGED")
    h.touch("start")
    h.say("카페인 없는 걸로 추천해 주세요")
    assert h.out.acts[-1].plan == "recommend"
    h.touch("pref", key="caffeine_free")                  # 하나뿐인 조건을 끄면 다시 취향을 묻는 화면
    assert h.out.acts[-1].name == "prefs" and h.out.acts[-1].plan == "ask_pref"


def test_every_act_carries_a_plan(h: Harness) -> None:
    h.human("ENGAGED")
    h.say("따뜻한 라떼 하나 주세요")
    h.say("음음")
    h.touch("set_option", key="size", value="regular")
    h.say("아니요")
    h.say("네")
    h.card("in")
    h.advance(2.1)
    h.card("out")
    assert h.state == "DONE"
    assert [a.name for a in h.out.acts if not a.plan] == []


# ---- 11/26 전체 검토에서 고친 것 ------------------------------------------------------------

def to_payment(h: Harness) -> None:
    h.human("ENGAGED")
    h.say("따뜻한 라떼 레귤러 하나 주세요")
    h.say("네")
    assert h.state == "PAY_WAIT_CARD"


def test_customer_looking_for_a_card_is_not_paused_quickly(h: Harness) -> None:
    to_payment(h)
    h.human("ABSENT")                                     # 지갑을 찾느라 카메라에서 사라짐
    h.advance(10.0)
    assert h.overlay == "NONE"                            # 예전: 2 s 만에 "잠시 기다릴게요"
    h.card("in")
    h.advance(2.1)
    assert h.state == "PAY_REMOVE_CARD"


def test_card_inserted_while_paused_clears_the_pause_and_pays(h: Harness) -> None:
    to_payment(h)
    h.human("ABSENT")
    h.advance(21.0)                                       # 정말 자리를 비움 → 일시정지
    assert h.overlay == "PAUSED"
    h.card("in")                                          # 돌아와 카드를 넣음 (예전: 결제가 끝나지 않고 60 s 뒤 취소)
    assert h.overlay == "NONE" and h.state == "PAY_PROCESSING"
    h.advance(2.1)
    h.card("out")
    assert h.state == "DONE"


def test_card_left_in_from_the_cart_step_is_used_for_payment(h: Harness) -> None:
    h.human("ENGAGED")
    h.say("따뜻한 라떼 레귤러 하나 주세요")
    h.card("in")                                          # 결제 전에 꽂음 → "아직 결제 단계가 아니에요"
    h.say("네")                                           # 빼지 않고 주문 확인 (예전: 카드를 넣으라며 무한 대기)
    assert h.state == "PAY_PROCESSING"
    h.advance(2.1)
    assert h.state == "PAY_REMOVE_CARD"


def test_cancel_is_ignored_once_payment_is_approved(h: Harness) -> None:
    to_payment(h)
    h.card("in")
    h.advance(2.1)
    h.say("아 취소할게요")                                 # 카드가 꽂힌 채 (예전: 주문이 사라짐)
    assert h.state == "PAY_REMOVE_CARD" and h.out.last_say().say == "remove_card"
    h.touch("cancel")
    assert h.state == "PAY_REMOVE_CARD"
    h.card("out")
    h.say("취소")                                          # 주문번호를 받은 뒤
    assert h.state == "DONE"


def test_resume_answer_continues_without_a_repair(h: Harness) -> None:
    h.human("ENGAGED")
    h.say("아인슈페너 주세요")
    h.human("ABSENT")
    h.advance(2.5)
    h.human("PRESENT")
    h.say("이어서 할게요")                                 # 예전: 이어 가자마자 "다시 말씀해 주시겠어요?"
    assert h.overlay == "NONE" and h.state == "SOLD_OUT"
    assert h.out.last_say().name == "sold_out"


def test_content_said_while_paused_is_handled_in_the_resumed_state(h: Harness) -> None:
    h.human("ENGAGED")
    h.say("라떼 하나 주세요")
    assert h.state == "OPTIONS"
    h.human("ABSENT")
    h.advance(2.5)
    h.human("PRESENT")
    h.say("아이스로요")
    assert h.state == "OPTIONS" and h.fsm.session.pending.temp == "ice"


def test_background_chatter_with_nobody_present_starts_nothing(h: Harness) -> None:
    h.say("음 그러니까 내일 수업이")                        # 관객 잡담
    assert h.state == "IDLE" and h.fsm.session is None and not h.out.says()
    h.say("라떼 하나 주세요")                              # 알아들은 주문은 카메라가 놓친 손님일 수 있다
    assert h.state == "OPTIONS"


def test_chatter_does_not_resume_a_paused_order(h: Harness) -> None:
    h.human("ENGAGED")
    h.touch("start")
    h.human("ABSENT")
    h.advance(2.5)
    assert h.overlay == "PAUSED"
    h.say("어제 그 영화 봤어?")
    assert h.overlay == "PAUSED"


def test_every_way_of_starting_turns_boredom_off(h: Harness) -> None:
    h.touch("start")                                      # 예전: 화면부터 누르면 주문 중에도 지루함 모션
    assert h.out.idle[-1] is False
    h.fsm.reset(h.now)
    assert h.out.idle[-1] is True
    h.touch("lang", lang="en")
    assert h.state == "MENU" and h.out.idle[-1] is False


def test_leaving_interrupts_and_goodbye_uses_the_session_language(h: Harness) -> None:
    h.human("ENGAGED")
    h.touch("lang", lang="en")
    h.say("cancel")
    idle = [a for a in h.out.acts if a.name == "idle"][-1]
    bye = h.out.acts[-1]
    assert idle.interrupt is True                         # 하던 말·키프레임을 기다리던 화면을 끊는다
    assert bye.name == "cancelled" and bye.lang == "en"   # 대기 상태(한국어)로 돌아간 뒤에도 영어로 작별
    assert h.out.acts.index(idle) < h.out.acts.index(bye)


def test_task_sentence_right_after_the_greeting_starts_a_recommendation(h: Harness) -> None:
    h.human("ENGAGED")
    h.say("카페인 없고 너무 달지 않은 음료 있어요?")          # 평가 과제 T2 (예전: 조건을 버리고 메뉴로)
    prefs = h.fsm.session.prefs
    assert h.state == "RECOMMEND" and prefs.caffeine is False and prefs.sweet_max == 1


def test_dislike_before_any_recommendation_is_a_preference(h: Harness) -> None:
    h.human("ENGAGED")
    h.touch("lang", lang="en")
    h.human("NO_RESPONSE")
    assert h.state == "OFFER_HELP"
    h.say("I don't like sweet drinks")                    # 예전: 조용히 메뉴로
    assert h.state == "RECOMMEND" and h.fsm.session.prefs.sweet_max == 1
