from __future__ import annotations

import pytest

from reachy_cafe.dialogue.nlu import NLU

CASES = [
    # (발화, 문맥, 기대 intent, 기대 슬롯)
    ("따뜻한 라떼 하나 주세요", "menu", "order_item", {"item": "latte", "temp": "hot", "qty": 1}),
    ("바닐라 라떼로 할게요", "sold_out", "order_item", {"item": "vanilla_latte"}),
    ("아인슈페너 주세요", "menu", "order_item", {"item": "einspanner"}),
    ("아아 두 잔", "menu", "order_item", {"item": "americano", "temp": "ice", "qty": 2}),
    ("딸기 라떼 주세요", "menu", "order_item", {"item": "strawberry_latte"}),
    ("바닐라 랏떼 하나요", "menu", "order_item", {"item": "vanilla_latte"}),
    ("two iced lattes please", "menu", "order_item", {"item": "latte", "temp": "ice", "qty": 2}),
    ("아니 그거 말고 라떼요", "options", "order_item", {"item": "latte"}),
    ("레귤러요", "options", "set_option", {"size": "regular"}),
    ("아이스로 주세요", "options", "set_option", {"temp": "ice"}),
    ("iced, regular please", "options", "set_option", {"temp": "ice", "size": "regular"}),
    ("네", "cart", "confirm", {}),
    ("네 맞아요", "cart", "confirm", {}),
    ("Yes, I'll take it", "recommend", "confirm", {}),
    ("결제할게요", "cart", "checkout", {}),
    ("다시 말해 주세요", "cart", "repeat", {}),
    ("괜찮아요", "offer_help", "deny", {}),
    ("아니요", "cart", "deny", {}),
    ("Yes, something without coffee.", "offer_help", "preference", {"prefs": {"caffeine": False}}),
    ("커피 말고 상큼한 거요", "offer_help", "preference", {"prefs": {"caffeine": False, "tags": ["refreshing"]}}),
    ("추천해 주세요", "menu", "ask_recommend", {}),
    ("What's good here?", "menu", "ask_recommend", {}),
    ("Hmm, too sweet.", "recommend", "reject_feedback", {"reason": "sweet"}),
    ("너무 달아요", "recommend", "reject_feedback", {"reason": "sweet"}),
    ("라떼 빼주세요", "cart", "remove_item", {"item": "latte"}),
    ("취소할게요", "menu", "cancel", {}),
    ("아인슈페너가 뭐예요?", "menu", "explain_item", {"item": "einspanner"}),
    ("What is Einspänner?", "menu", "explain_item", {"item": "einspanner"}),
    ("바닐라 라떼는 무슨 맛이에요?", "menu", "explain_item", {"item": "vanilla_latte"}),
    ("what's hibiscus tea like", "menu", "explain_item", {"item": "hibiscus_tea"}),
    # 11/26 검토에서 고친 것: 군말 "아", 예/아니오 순서, 단맛·카페인 부정, 이어서, 인사
    ("아, 아이스로요", "options", "set_option", {"temp": "ice", "item": None}),
    ("아 아니요", "cart", "deny", {"item": None}),
    ("아 네", "cart", "confirm", {}),
    ("No, that's not right", "cart", "deny", {}),
    ("That's not correct", "cart", "deny", {}),
    ("um I'm not sure", "cart", "unknown", {}),
    ("안 맞아요", "cart", "deny", {}),
    ("네 괜찮아요", "cart", "confirm", {}),
    ("카페인 없고 너무 달지 않은 음료 있어요?", "greet", "preference", {"prefs": {"caffeine": False, "sweet_max": 1}}),
    ("Something not too sweet", "recommend", "preference", {"prefs": {"sweet_max": 1}}),
    ("something without caffeine that's not too sweet", "menu", "preference",
     {"prefs": {"caffeine": False, "sweet_max": 1}}),
    ("커피가 아닌 거요", "recommend", "preference", {"prefs": {"caffeine": False}}),
    ("단 거 싫어요", "menu", "preference", {"prefs": {"sweet_max": 1}}),
    ("이어서 할게요", "paused", "confirm", {}),
    ("계속할게요", "paused", "confirm", {}),
    ("Continue", "paused", "confirm", {}),
    ("안녕하세요", "greet", "hello", {}),
    ("Hi there", "greet", "hello", {"lang": "en"}),
    ("아이스라떼 하나요", "menu", "order_item", {"item": "latte", "temp": "ice", "qty": 1}),
    ("따뜻한라떼하나주세요", "menu", "order_item", {"item": "latte", "temp": "hot", "qty": 1}),
    ("아아로 주세요", "menu", "order_item", {"item": "americano", "temp": "ice"}),
    # 11/26 자연스러움: 규칙 밖의 말(직원 안내), 요청사항·포장, "X 말고 Y", 여러 잔, 가리키는 말·순서, 장바구니 고치기
    ("직원 불러 주세요", "menu", "staff", {}),
    ("Can I talk to a staff member?", "menu", "staff", {}),
    ("견과류 들어가요?", "menu", "out_of_scope", {"reason": "allergy"}),
    ("현금으로 해도 돼요?", "pay", "out_of_scope", {"reason": "payment"}),
    ("환불하고 싶어요", "done", "out_of_scope", {"reason": "refund"}),
    ("화장실 어디예요?", "menu", "out_of_scope", {"reason": "facility"}),
    ("Do you take cash?", "pay", "out_of_scope", {"reason": "payment"}),
    ("두 사람이요", "menu", "unknown", {}),
    ("직원 추천 메뉴 뭐예요", "menu", "ask_recommend", {}),
    ("얼음 적게 해 주세요", "options", "note", {"notes": ["less_ice"]}),
    ("아이스 라떼 얼음 적게", "menu", "order_item", {"item": "latte", "temp": "ice", "notes": ["less_ice"]}),
    ("라떼 얼음 빼고 주세요", "cart", "order_item", {"item": "latte", "notes": ["no_ice"]}),
    ("latte no ice", "menu", "order_item", {"item": "latte", "temp": None, "notes": ["no_ice"]}),
    ("포장이요", "cart", "dine", {"dine": "togo"}),
    ("아이스 라떼 포장이요", "menu", "order_item", {"item": "latte", "temp": "ice", "dine": "togo"}),
    ("a latte to take out", "menu", "order_item", {"item": "latte", "dine": "togo"}),
    ("take out the latte", "cart", "remove_item", {"item": "latte"}),
    ("for here", "cart", "dine", {"dine": "here"}),
    ("아메리카노 말고 라떼요", "menu", "order_item", {"item": "latte", "rejected_items": ["americano"]}),
    ("latte instead of americano", "menu", "order_item", {"item": "latte", "rejected_items": ["americano"]}),
    ("아메리카노 말고요", "recommend", "reject_item", {"item": "americano"}),
    ("라떼는 싫어요", "recommend", "reject_item", {"item": "latte"}),
    ("그거 주세요", "recommend", "confirm", {}),
    ("I'll have that", "recommend", "confirm", {}),
    ("첫 번째 걸로", "sold_out", "choose", {"ordinal": 1}),
    ("the second one", "sold_out", "choose", {"ordinal": 2}),
    ("오른쪽 거요", "sold_out", "choose", {"ordinal": -1}),
    ("That's right", "cart", "confirm", {"ordinal": None}),
    ("라떼는 아이스로 바꿔 주세요", "cart", "order_item", {"item": "latte", "temp": "ice", "change": True}),
    ("라떼 하나 더", "cart", "order_item", {"item": "latte", "more": True}),
    ("아메리카노 하나 빼 주세요", "cart", "remove_item", {"item": "americano", "qty": 1}),
    ("딸기랑 바닐라 중에 뭐가 나아요?", "menu", "compare", {"candidates": ["strawberry_latte", "vanilla_latte"]}),
]


@pytest.mark.parametrize(("text", "context", "intent", "slots"), CASES)
def test_parse(nlu: NLU, text: str, context: str, intent: str, slots: dict) -> None:
    result = nlu.parse(text, context)
    assert result.name == intent, result.summary()
    for key, value in slots.items():
        assert getattr(result, key) == value, (key, result.summary())


def test_water_please_is_not_a_payment_confirmation(nlu: NLU) -> None:
    assert nlu.parse("물 주세요", "cart").name != "confirm"
    for text in ("Water please", "One moment please", "Can I get a straw please"):  # "please" 단독은 예가 아니다
        assert nlu.parse(text, "cart").name != "confirm", text


def test_filler_a_is_never_an_iced_americano(nlu: NLU) -> None:
    """두 글자 별칭("아아"·"아메")은 공백을 넘어 찾지 않는다: 말 앞의 "아"가 아메리카노 주문이 되던 문제."""
    for text in ("아, 아이스로요", "아 아니요", "아 메뉴 좀 볼게요", "아아아", "아 아까 그거요"):
        for context in ("menu", "options", "cart", "recommend"):
            assert nlu.parse(text, context).item != "americano", (text, context)


def test_too_sweet_complaint_is_still_a_rejection(nlu: NLU) -> None:
    """부정 처리 뒤에도 S2의 "너무 달아요"·"too sweet"은 거절로 남는다."""
    for text in ("Hmm, too sweet.", "너무 달아요", "이건 너무 달아요"):
        assert nlu.parse(text, "recommend").name == "reject_feedback", text


def test_short_fuzzy_match_only_suggests_candidates(nlu: NLU) -> None:
    result = nlu.parse("바닐", "menu")
    assert result.name == "unknown"
    assert "vanilla_latte" in result.candidates


def test_longest_alias_wins(nlu: NLU) -> None:
    assert nlu.parse("녹차 라떼 하나", "menu").item == "green_tea_latte"
    assert nlu.parse("카페모카", "menu").item == "cafe_mocha"


def test_several_drinks_in_one_sentence(nlu: NLU) -> None:
    result = nlu.parse("따뜻한 라떼 하나랑 아이스 아메리카노 두 잔", "menu")
    assert [(e["item"], e["qty"], e["temp"]) for e in result.items] == [("latte", 1, "hot"), ("americano", 2, "ice")]
    result = nlu.parse("two lattes and an iced americano", "menu")
    assert [(e["item"], e["qty"], e["temp"]) for e in result.items] == [("latte", 2, None), ("americano", None, "ice")]
    result = nlu.parse("아아랑 뜨아 하나씩", "menu")
    assert [(e["item"], e["temp"]) for e in result.items] == [("americano", "ice"), ("americano", "hot")]
    # 한 잔만 말한 문장·"주문하고 싶어요"의 '하고'는 여러 잔이 아니다
    for text in ("라떼 하나, 아이스로", "바닐라 라떼 주문하고 싶어요", "아메리카노 말고 라떼요"):
        assert nlu.parse(text, "menu").items == [], text


def test_language_detection(nlu: NLU) -> None:
    assert nlu.parse("one latte", "menu").lang == "en"
    assert nlu.parse("라떼 하나", "menu").lang == "ko"
