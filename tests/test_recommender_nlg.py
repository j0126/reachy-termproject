from __future__ import annotations

from reachy_cafe.dialogue.menu import Cart, CartLine, Menu
from reachy_cafe.dialogue.nlg import NLG, josa
from reachy_cafe.dialogue.recommender import Prefs, Recommender


def test_s2_recommendation_loop(menu: Menu) -> None:
    """S2: 커피 말고 → 자몽 허니 티, '너무 달아요' → 덜 단 히비스커스 티."""
    rec = Recommender(menu)
    prefs = Prefs()
    prefs.merge({"caffeine": False})
    first = rec.recommend(prefs)[0]
    assert first.id == "honey_grapefruit_tea"
    assert rec.reasons(first, prefs) == ["caffeine_free", "refreshing"]

    rec.on_reject(prefs, first, "sweet")
    second = rec.recommend(prefs, {first.id})[0]
    assert second.id == "hibiscus_tea"
    assert prefs.sweet_max == 2
    assert rec.reasons(second, prefs, focus=prefs.updated_by)[0] == "less_sweet"


def test_s3_alternatives_for_sold_out(menu: Menu) -> None:
    rec = Recommender(menu)
    alts = [item.id for item in rec.alternatives(menu.get("einspanner"))]
    assert alts == ["vanilla_latte", "cafe_mocha"]


def test_sold_out_items_are_never_recommended(menu: Menu) -> None:
    rec = Recommender(menu)
    assert "einspanner" not in [item.id for item in rec.recommend(Prefs())]


def test_sentences(menu: Menu) -> None:
    nlg = NLG(menu)
    assert nlg.text("recommend", "en", item="honey_grapefruit_tea", reasons=["caffeine_free", "refreshing"]) == \
        "How about Honey Grapefruit Tea? It's caffeine-free and refreshing."
    assert nlg.text("recommend", "ko", item="honey_grapefruit_tea", reasons=["caffeine_free", "refreshing"]) == \
        "자몽 허니 티는 어떠세요? 카페인이 없고 상큼한 메뉴예요."
    assert nlg.text("sold_out", "ko", item="einspanner", alts=["vanilla_latte", "cafe_mocha"]) == \
        "죄송해요, 아인슈페너는 오늘 품절이에요. 비슷한 바닐라 라떼나 카페모카는 어떠세요?"
    assert nlg.text("confirm_cart", "ko", lines=[{"item": "latte", "qty": 1, "temp": "hot", "size": "regular"}],
                    total=4500) == "따뜻한 카페라떼 한 잔, 4,500원이에요. 이대로 주문할까요?"
    assert nlg.text("done", "ko", number=27) == "주문번호는 27이에요. 감사합니다!"
    assert nlg.text("done", "ko", number=12) == "주문번호는 12예요. 감사합니다!"


def test_josa() -> None:
    assert josa("카페라떼", "은/는") == "카페라떼는"
    assert josa("아인슈페너", "을/를") == "아인슈페너를"
    assert josa("바닐라 라떼", "이나/나") == "바닐라 라떼나"
    assert josa("녹차", "으로/로") == "녹차로"
    assert josa("아메리카노 라지", "이에요/예요") == "아메리카노 라지예요"


def test_cart_merges_same_options_and_prices_large(menu: Menu) -> None:
    cart = Cart()
    cart.add(CartLine("latte", 1, "hot", "regular"))
    cart.add(CartLine("latte", 1, "hot", "regular"))
    cart.add(CartLine("americano", 1, "ice", "large"))
    assert len(cart.lines) == 2
    assert cart.count() == 3
    assert cart.total(menu) == 4500 * 2 + 4000 + 500
