"""문장 템플릿 (한국어·영어).

FSM은 문장 키와 슬롯(메뉴 id, 숫자)만 넘기고, 화면 자막과 TTS에 쓰는 문장은 여기서 만든다.
슬롯이 없는 문장(FIXED_KEYS)은 2단계에서 HW3 Supertonic으로 미리 합성해 TTS 캐시로 쓴다.
"""

from __future__ import annotations

from typing import Any, Callable

from .menu import Menu

_KO_QTY = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯"}
_EN_QTY = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
_BATCHIM_DIGITS = set("0136781")  # 영·일·삼·육·칠·팔 (받침 있음)

REASONS: dict[str, dict[str, Any]] = {
    "caffeine_free": {"ko": ("카페인이 없는", "카페인이 없고"), "en": "caffeine-free"},
    "less_sweet": {"ko": ("덜 단", "덜 달고"), "en": "less sweet"},
    "not_sweet": {"ko": ("달지 않은", "달지 않고"), "en": "not sweet at all"},
    "refreshing": {"ko": ("상큼한", "상큼하고"), "en": "refreshing"},
    "creamy": {"ko": ("부드러운", "부드럽고"), "en": "smooth and creamy"},
    "popular": {"ko": ("인기 있는", "인기 있고"), "en": "a popular pick"},
}

FIXED: dict[str, dict[str, str]] = {
    "greet": {"ko": "안녕하세요! 주문 도와드릴까요?", "en": "Hello! May I take your order?"},
    "greet_again": {"ko": "주문하실 때 말씀하시거나 화면을 눌러 주세요.",
                    "en": "Just speak or tap the screen whenever you're ready."},
    "greet_menu": {"ko": "안녕하세요! 드시고 싶은 메뉴를 말씀하시거나 화면에서 골라 주세요.",
                   "en": "Hello! Tell me what you'd like, or pick it on the screen."},
    "menu_prompt": {"ko": "드시고 싶은 메뉴를 말씀하시거나 화면에서 골라 주세요.",
                    "en": "Tell me what you'd like, or pick it on the screen."},
    "menu_again": {"ko": "다른 메뉴를 말씀하시거나 화면에서 골라 주세요.",
                   "en": "What would you like instead? You can say it or tap it."},
    "menu_ok": {"ko": "네, 천천히 골라 주세요.", "en": "Sure, take your time."},
    "offer_help": {"ko": "고르기 어려우시면 추천해 드릴까요?", "en": "Need a recommendation?"},
    "offer_help_seek": {"ko": "저를 보셨네요! 도와드릴까요? 추천도 해 드릴 수 있어요.",
                        "en": "You looked my way! Need some help? I can recommend something."},
    "ask_preference": {"ko": "어떤 음료가 좋으세요? 커피가 아닌 것, 덜 단 것처럼 말씀해 주세요.",
                       "en": "What do you feel like? For example, something without coffee, or not too sweet."},
    "recommend_reprompt": {"ko": "이 메뉴로 드릴까요?", "en": "Shall I get you this one?"},
    "no_recommendation": {"ko": "조건에 맞는 메뉴가 없어요. 화면에서 골라 주시겠어요?",
                          "en": "I couldn't find a match. Could you pick one on the screen?"},
    "rec_giveup": {"ko": "화면에서 직접 골라 보시겠어요?", "en": "Would you like to pick one on the screen?"},
    "sold_out_reprompt": {"ko": "화면의 메뉴 중에서 골라 주셔도 돼요.", "en": "You can also pick one of these on the screen."},
    "ask_temp": {"ko": "따뜻하게 드릴까요, 아이스로 드릴까요?", "en": "Hot or iced?"},
    "ask_size": {"ko": "사이즈는 레귤러와 라지 중 어떤 걸로 드릴까요?", "en": "Regular or large?"},
    "ask_change": {"ko": "어떤 걸 바꿀까요? 말씀하시거나 화면에서 고쳐 주세요.",
                   "en": "What would you like to change? Say it or edit it on the screen."},
    "add_more": {"ko": "더 주문하실 메뉴를 말씀해 주세요.", "en": "What else would you like?"},
    "cart_empty": {"ko": "장바구니가 비었어요. 메뉴를 골라 주세요.", "en": "Your cart is empty. Please choose a drink."},
    "pay_request": {"ko": "카드를 화면 아래 투입구에 넣어 주세요.",
                    "en": "Please insert your card into the slot below the screen."},
    "pay_repeat": {"ko": "카드를 아래 투입구에 넣어 주세요. 화면의 결제 버튼을 눌러도 돼요.",
                   "en": "Please insert your card below, or tap the pay button on the screen."},
    "processing": {"ko": "결제 중이에요. 잠시만 기다려 주세요.", "en": "Processing your payment. One moment, please."},
    "remove_card": {"ko": "결제가 끝났어요. 카드를 빼 주세요.", "en": "Payment complete. Please take your card."},
    "forgot_card": {"ko": "카드를 잊지 마세요!", "en": "Don't forget your card!"},
    "not_payment_step": {"ko": "아직 결제 단계가 아니에요. 카드를 빼 주세요.",
                         "en": "We're not at payment yet. Please remove your card."},
    "card_reinsert": {"ko": "결제 중에 카드가 빠졌어요. 카드를 다시 넣어 주세요.",
                      "en": "The card came out during payment. Please insert it again."},
    "pause": {"ko": "잠시 기다릴게요.", "en": "I'll wait for you."},
    "resume_prompt": {"ko": "주문을 이어서 하시겠어요?", "en": "Would you like to continue your order?"},
    "cancelled": {"ko": "주문을 취소했어요. 다음에 또 오세요.", "en": "Your order is cancelled. See you next time."},
    "one_at_a_time": {"ko": "한 분씩 도와드릴게요. 주문하실 분이 화면의 시작 버튼을 눌러 주세요.",
                      "en": "I'll help one person at a time. Whoever is ordering, please tap Start."},
    "busy_with_other": {"ko": "지금은 이분 주문을 먼저 도와드릴게요.", "en": "Let me finish this order first."},
    "repair": {"ko": "죄송해요, 다시 한 번 말씀해 주시겠어요?", "en": "Sorry, could you say that again?"},
    "repair_candidates": {"ko": "혹시 이 중에 있으세요? 화면에서 골라 주셔도 돼요.",
                          "en": "Is it one of these? You can also tap it on the screen."},
    "repair_touch": {"ko": "잘 못 알아들었어요. 화면에서 골라 주시겠어요?",
                     "en": "I didn't quite catch that. Could you choose on the screen?"},
    # 인사 뒤 망설이는 손님을 메뉴로 이끌기
    "menu_lead": {"ko": "메뉴를 보여 드릴게요. 천천히 보시고 말씀하시거나 골라 주세요.",
                  "en": "Here's our menu. Take your time, then tell me or tap what you'd like."},
    "no_match_chips": {"ko": "그 조건에 다 맞는 메뉴는 없어요. 조건을 하나 빼 보시겠어요?",
                       "en": "Nothing matches all of those. Could you remove one?"},
    # 규칙 밖의 상황 → 직원 안내. 키오스크가 모르는 것은 지어내지 않고 사람에게 넘긴다
    "staff_request": {"ko": "네, 직원을 불렀어요. 잠시만 기다려 주세요.",
                      "en": "Sure, I've called a staff member. Please wait a moment."},
    "staff_allergy": {"ko": "알레르기나 성분은 정확해야 해서 직원이 직접 알려 드릴게요. 직원을 불렀어요.",
                      "en": "Allergy and ingredient questions need an exact answer, so I've called a staff member."},
    "staff_payment": {"ko": "여기서는 카드 결제만 할 수 있어요. 직원을 불렀으니 잠시만 기다려 주세요.",
                      "en": "I can only take cards here. I've called a staff member to help you."},
    "staff_refund": {"ko": "그건 직원이 직접 도와드려야 해요. 직원을 불렀어요.",
                     "en": "A staff member needs to help with that. I've called them."},
    "staff_facility": {"ko": "그건 제가 안내해 드리기 어려워서 직원을 불렀어요. 잠시만 기다려 주세요.",
                       "en": "I can't help with that myself, so I've called a staff member."},
    "staff_repair": {"ko": "제가 잘 못 알아들어서 죄송해요. 직원을 불렀어요. 기다리시는 동안 화면으로 주문하셔도 돼요.",
                     "en": "Sorry, I keep missing that. I've called a staff member. You can keep ordering on the screen."},
    "staff_already": {"ko": "직원을 이미 불렀어요. 곧 올 거예요.", "en": "I've already called a staff member. They'll be right with you."},
    "staff_thanks": {"ko": "기다려 주셔서 감사해요. 이어서 도와드릴게요.", "en": "Thanks for waiting. Let's continue."},
    "dine_togo": {"ko": "포장으로 해 드릴게요.", "en": "Sure, to go."},
    "dine_here": {"ko": "매장에서 드시는 걸로 할게요.", "en": "Sure, for here."},
    "pay_slot_down": {"ko": "카드 투입구가 지금 연결되지 않았어요. 화면의 결제 버튼을 눌러 주세요.",
                      "en": "The card slot isn't connected right now. Please tap the pay button on the screen."},
}

# 요청사항 이름 (nlu.NOTES 키). 장바구니 확인 문장과 화면에 쓴다
NOTE_NAMES: dict[str, dict[str, str]] = {
    "less_ice": {"ko": "얼음 적게", "en": "less ice"},
    "no_ice": {"ko": "얼음 없이", "en": "no ice"},
    "extra_ice": {"ko": "얼음 많이", "en": "extra ice"},
    "extra_shot": {"ko": "샷 추가", "en": "extra shot"},
    "strong": {"ko": "진하게", "en": "stronger"},
    "weak": {"ko": "연하게", "en": "weaker"},
    "no_syrup": {"ko": "시럽 없이", "en": "no syrup"},
    "extra_syrup": {"ko": "시럽 추가", "en": "extra syrup"},
    "no_whip": {"ko": "휘핑 없이", "en": "no whipped cream"},
    "extra_whip": {"ko": "휘핑 추가", "en": "extra whipped cream"},
    "oat_milk": {"ko": "오트 밀크로", "en": "oat milk"},
    "soy_milk": {"ko": "두유로", "en": "soy milk"},
}

FIXED_KEYS = tuple(FIXED)

# 상태 스트립의 "로봇의 계획" (화면 전용, 말하지 않음)
PLANS: dict[str, dict[str, str]] = {
    "waiting": {"ko": "손님을 기다리는 중", "en": "Waiting for a customer"},
    "greeting": {"ko": "인사 → 주문 받기", "en": "Greeting → taking the order"},
    "browsing": {"ko": "메뉴 고르는 중", "en": "Customer is choosing"},
    "offer_help": {"ko": "망설임 감지 → 추천 제안", "en": "Hesitation detected → offering help"},
    "ask_pref": {"ko": "취향 묻는 중", "en": "Asking what you like"},
    "recommend": {"ko": "추천: {name}", "en": "Recommending {name}"},
    "sold_out": {"ko": "품절 → 비슷한 메뉴 제안", "en": "Sold out → suggesting similar drinks"},
    "options_temp": {"ko": "온도 묻는 중", "en": "Asking hot or iced"},
    "options_size": {"ko": "사이즈 묻는 중", "en": "Asking the size"},
    "confirm": {"ko": "주문 확인", "en": "Confirming the order"},
    "pay_insert": {"ko": "카드 투입 안내", "en": "Guiding card insertion"},
    "pay_retry": {"ko": "카드를 다시 넣도록 안내", "en": "Asking to insert the card again"},
    "processing": {"ko": "결제 처리 중", "en": "Processing payment"},
    "remove_card": {"ko": "카드 회수 안내", "en": "Reminding to take the card"},
    "done": {"ko": "주문 완료 #{number}", "en": "Order complete #{number}"},
    "paused": {"ko": "손님이 돌아오길 기다리는 중", "en": "Waiting for the customer to return"},
    "one_at_a_time": {"ko": "여러 명 감지 → 한 분씩", "en": "Several people → one at a time"},
    "invite": {"ko": "멀리 손님 발견 → 눈 맞추고 부르기", "en": "Someone nearby → inviting with a look"},
    "explain": {"ko": "메뉴 설명: {name}", "en": "Explaining {name}"},
    "seek_help": {"ko": "손님이 로봇을 봄 → 도움 제안", "en": "You looked at me → offering help"},
    "repair": {"ko": "잘 못 알아들음 → 다시 물어보기", "en": "Didn't catch that → asking again"},
    "repair_candidates": {"ko": "메뉴 이름이 애매함 → 후보 보여 주기", "en": "Not sure which drink → showing options"},
    "repair_touch": {"ko": "계속 못 알아들음 → 화면 터치 권유", "en": "Still unclear → suggesting the touch screen"},
    "staff": {"ko": "규칙 밖의 상황 → 직원 호출", "en": "Outside what I can do → calling staff"},
    "staff_done": {"ko": "직원 응대 끝 → 주문 이어 가기", "en": "Staff helped → continuing the order"},
}

_DONE_TAIL = {"ko": ("감사합니다!", "맛있게 드세요!", "좋은 하루 보내세요!"),
              "en": ("Thank you!", "Enjoy your drink!", "Have a great day!")}


def has_batchim(word: str) -> bool:
    """마지막 글자에 받침이 있는지 (한글·숫자). 그 밖의 글자는 받침 없음으로 본다."""
    for ch in reversed(word.strip()):
        if "가" <= ch <= "힣":
            return (ord(ch) - 0xAC00) % 28 != 0
        if ch.isdigit():
            return ch in _BATCHIM_DIGITS
        if ch.isalpha():
            return False
    return False


def josa(word: str, pair: str) -> str:
    """josa("카페모카", "은/는") -> "카페모카는"."""
    with_batchim, without = pair.split("/")
    if pair == "으로/로":
        last = word.strip()[-1:] or " "
        rieul = "가" <= last <= "힣" and (ord(last) - 0xAC00) % 28 == 8
        return word + (without if (rieul or not has_batchim(word)) else with_batchim)
    return word + (with_batchim if has_batchim(word) else without)


def qty_ko(n: int) -> str:
    return f"{_KO_QTY.get(n, str(n))} 잔"


def price_text(n: int, lang: str) -> str:
    return f"{n:,}원" if lang == "ko" else f"{n:,} won"


class NLG:
    def __init__(self, menu: Menu):
        self.menu = menu
        self._dynamic: dict[str, Callable[..., str]] = {
            "greet": self._greet,
            "explain_item": self._explain,
            "recommend": self._recommend,
            "recommend_again": lambda lang, **s: self._recommend(lang, again=True, **s),
            "sold_out": self._sold_out,
            "confirm_cart": self._confirm_cart,
            "done": self._done,
            "removed": self._removed,
            "remove_which": self._remove_which,
            "sold_out_resume": self._sold_out_resume,
            "note_ok": self._note_ok,
            "ask_temp_item": lambda lang, item="": (f"{josa(self.name(item, lang), '은/는')} 따뜻하게 드릴까요, 아이스로 드릴까요?"
                                                    if lang == "ko" else f"And the {self.name(item, lang)}: hot or iced?"),
            "ask_size_item": lambda lang, item="": (f"{self.name(item, lang)} 사이즈는 레귤러와 라지 중 어떤 걸로 드릴까요?"
                                                    if lang == "ko" else f"And the {self.name(item, lang)}: regular or large?"),
        }

    def text(self, key: str, lang: str = "ko", **slots: Any) -> str:
        lang = lang if lang in ("ko", "en") else "ko"
        if key in self._dynamic:
            return self._dynamic[key](lang, **slots)
        entry = FIXED.get(key)
        if entry is None:
            return key
        return entry.get(lang) or entry["ko"]

    def plan(self, key: str, lang: str = "ko", **slots: Any) -> str:
        entry = PLANS.get(key)
        if entry is None:
            return key
        template = entry.get(lang) or entry["ko"]
        if "item" in slots:
            slots["name"] = self.name(str(slots["item"]), lang)
        try:
            return template.format(**slots)
        except (KeyError, IndexError):
            return template

    # ---- 도우미 -------------------------------------------------------------
    def name(self, item_id: str, lang: str) -> str:
        item = self.menu.get(item_id)
        return item.label(lang) if item else item_id

    def reasons(self, keys: list[str], lang: str) -> str:
        keys = [k for k in keys if k in REASONS] or ["popular"]
        if lang == "ko":
            parts = [REASONS[k]["ko"][1] for k in keys[:-1]] + [REASONS[keys[-1]]["ko"][0]]
            return " ".join(parts)
        return " and ".join(REASONS[k]["en"] for k in keys)

    def notes(self, keys: list[str], lang: str) -> str:
        names = [NOTE_NAMES[k][lang] for k in keys if k in NOTE_NAMES]
        return ", ".join(names)

    def line(self, line: dict[str, Any], lang: str) -> str:
        name = self.name(line["item"], lang)
        qty = int(line.get("qty") or 1)
        temp, size = line.get("temp"), line.get("size")
        notes = self.notes(list(line.get("notes") or []), lang)
        if lang == "ko":
            words = [{"hot": "따뜻한", "ice": "아이스"}.get(temp or "", ""), name,
                     "라지" if size == "large" else "", qty_ko(qty) + (f"({notes})" if notes else "")]
            return " ".join(w for w in words if w)
        words = [_EN_QTY.get(qty, str(qty)), {"hot": "hot", "ice": "iced"}.get(temp or "", ""),
                 name + ("s" if qty > 1 else ""), "(large)" if size == "large" else "", f"({notes})" if notes else ""]
        return " ".join(w for w in words if w)

    # ---- 슬롯이 있는 문장 ----------------------------------------------------
    def _recommend(self, lang: str, item: str = "", reasons: list[str] | None = None, again: bool = False) -> str:
        name = self.name(item, lang)
        why = self.reasons(reasons or [], lang)
        if lang == "ko":
            return f"{'그럼 ' if again else ''}{josa(name, '은/는')} 어떠세요? {why} 메뉴예요."
        return f"{'Then how' if again else 'How'} about {name}? It's {why}."

    def _sold_out(self, lang: str, item: str = "", alts: list[str] | None = None) -> str:
        name = self.name(item, lang)
        names = [self.name(a, lang) for a in (alts or [])]
        if lang == "ko":
            head = f"죄송해요, {josa(name, '은/는')} 오늘 품절이에요."
            if len(names) >= 2:
                return f"{head} 비슷한 {josa(names[0], '이나/나')} {josa(names[1], '은/는')} 어떠세요?"
            if names:
                return f"{head} 비슷한 {josa(names[0], '은/는')} 어떠세요?"
            return f"{head} 다른 메뉴를 골라 주시겠어요?"
        head = f"Sorry, {name} is sold out today."
        if names:
            return f"{head} How about {' or '.join(names)} instead?"
        return f"{head} Could you pick something else?"

    def _confirm_cart(self, lang: str, lines: list[dict[str, Any]] | None = None, total: int = 0,
                      dine: str | None = None) -> str:
        lines = lines or []
        parts = [self.line(line, lang) for line in lines]
        price = price_text(total, lang)
        if lang == "ko":
            joined = ", ".join(parts)
            amount = f"모두 {price}" if len(parts) > 1 else price
            how = {"togo": "포장으로", "here": "매장에서 드시는 걸로"}.get(dine or "", "이대로")
            return f"{joined}, {amount}이에요. {how} 주문할까요?"
        joined = ", ".join(parts)
        if joined:
            joined = joined[0].upper() + joined[1:]
        how = {"togo": " to go", "here": " for here"}.get(dine or "", "")
        return f"{joined}. That's {price} in total. Shall I place the order{how}?"

    def _sold_out_resume(self, lang: str, item: str = "", alts: list[str] | None = None) -> str:
        """자리를 비웠다 돌아온 손님: 품절 사과는 이미 들었으니 대안만."""
        names = [self.name(a, lang) for a in (alts or [])]
        if lang == "ko":
            if len(names) >= 2:
                return f"비슷한 {josa(names[0], '이나/나')} {josa(names[1], '은/는')} 어떠세요?"
            return f"비슷한 {josa(names[0], '은/는')} 어떠세요?" if names else "다른 메뉴를 골라 주시겠어요?"
        return f"How about {' or '.join(names)} instead?" if names else "Could you pick something else?"

    def _remove_which(self, lang: str, item: str = "") -> str:
        name = self.name(item, lang)
        if lang == "ko":
            return f"어떤 {josa(name, '을/를')} 뺄까요? 화면에서 ✕를 눌러 주셔도 돼요."
        return f"Which {name} should I remove? You can also tap ✕ on the screen."

    def _note_ok(self, lang: str, notes: list[str] | None = None) -> str:
        text = self.notes(list(notes or []), lang)
        if lang == "ko":
            return f"{text}, 요청사항으로 적어 둘게요." if text else "요청사항으로 적어 둘게요."
        return f"Got it, I'll note that: {text}." if text else "Got it, I'll note that."

    def _done(self, lang: str, number: int = 0) -> str:
        tail = _DONE_TAIL[lang][number % 3]  # 주문번호마다 마무리 멘트를 바꿔 기계적으로 들리지 않게
        if lang == "ko":
            return f"주문번호는 {josa(str(number), '이에요/예요')}. {tail}"
        return f"Your order number is {number}. {tail}"

    def _greet(self, lang: str, weather: str | None = None, daypart: str | None = None) -> str:
        """날씨·시간대 인사 (context_talk). 슬롯이 없으면 기본 인사."""
        if lang == "ko":
            hello = "좋은 아침이에요!" if daypart == "morning" else "안녕하세요!"
            tail = {"cold": "오늘 꽤 춥죠? 따뜻한 음료로 도와드릴까요?",
                    "hot": "오늘 덥죠? 시원한 음료로 도와드릴까요?",
                    "rain": "비 오는 날엔 따뜻한 차도 좋아요. 주문 도와드릴까요?"}.get(weather or "", "주문 도와드릴까요?")
            return f"{hello} {tail}"
        hello = "Good morning!" if daypart == "morning" else "Hello!"
        tail = {"cold": "It's chilly today. Something warm, maybe?",
                "hot": "It's hot today. How about something iced?",
                "rain": "A warm tea is nice on a rainy day. May I take your order?"}.get(weather or "",
                                                                                      "May I take your order?")
        return f"{hello} {tail}"

    def _explain(self, lang: str, item: str = "") -> str:
        menu_item = self.menu.get(item)
        name = self.name(item, lang)
        desc = (menu_item.desc.get(lang) if menu_item else "") or ""
        sold = menu_item is not None and not menu_item.in_stock
        if lang == "ko":
            text = f"{josa(name, '은/는')} {josa(desc, '이에요/예요')}." if desc else f"{name}이에요."
            return text + (" 다만 오늘은 품절이에요." if sold else " 드셔 보시겠어요?")
        text = f"{name} is {desc[:1].lower() + desc[1:]}." if desc else f"That's {name}."
        return text + (" Sorry, it's sold out today." if sold else " Would you like to try it?")

    def _removed(self, lang: str, item: str = "") -> str:
        name = self.name(item, lang)
        return f"{josa(name, '을/를')} 뺐어요." if lang == "ko" else f"I removed the {name}."
