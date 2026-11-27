"""규칙 기반 NLU (한국어·영어).

메뉴 어휘가 닫혀 있으므로 별칭 사전 + 퍼지 매칭으로 충분하다. LLM은 쓰지 않는다
(제안서 원칙: Rules decide, LLM only phrases). ASR 결과 텍스트 한 줄을 Intent 하나로 바꾼다.

규칙 밖의 말은 두 가지로 나눈다 (11/26 결정: 규칙 + 직원 안내).
- 알아들었지만 키오스크가 하면 안 되거나 못 하는 일 (알레르기·성분, 현금·쿠폰, 환불, 화장실·와이파이,
  직원을 찾는 말): out_of_scope / staff → FSM이 솔직하게 말하고 직원을 부른다. 지어내서 답하지 않는다.
- 못 알아들은 말: unknown → 되묻기 → 화면 권유 → (세 번째) 직원 호출.
메뉴에 없는 음료 요청(얼음 적게, 샷 추가 등)은 정해 둔 목록만 요청사항(notes)으로 받아 주문에 붙인다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from rapidfuzz import fuzz

from .menu import Menu

_HANGUL = re.compile(r"[가-힣]")
_PUNCT = re.compile(r"[\"'“”‘’.,!?~…·:;()\[\]{}]")


def normalize(text: str) -> str:
    text = _PUNCT.sub(" ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def compact(text: str) -> str:
    return text.replace(" ", "")


def is_korean(text: str) -> bool:
    return bool(_HANGUL.search(text))


@dataclass
class Intent:
    name: str
    text: str = ""
    lang: str = "ko"
    item: str | None = None
    qty: int | None = None
    temp: str | None = None
    size: str | None = None
    prefs: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    score: float = 0.0
    candidates: list[str] = field(default_factory=list)
    items: list[dict[str, Any]] = field(default_factory=list)   # 한 번에 말한 여러 메뉴 (2개 이상일 때만)
    notes: list[str] = field(default_factory=list)              # 요청사항 키 (NOTES: less_ice, extra_shot …)
    dine: str | None = None                                     # togo | here
    rejected_items: list[str] = field(default_factory=list)     # "아메리카노 말고 라떼"의 아메리카노
    ordinal: int | None = None                                  # "첫 번째 걸로" = 1, "오른쪽 거" = -1 (마지막)
    change: bool = False                                        # "바꿔 주세요" (장바구니에서 이름으로 고치기)
    more: bool = False                                          # "하나 더"

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"intent": self.name}
        for key in ("item", "qty", "temp", "size", "reason", "dine", "ordinal"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        for key in ("prefs", "candidates", "notes", "rejected_items"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        if self.items:
            out["items"] = [entry["item"] for entry in self.items]
        if self.change:
            out["change"] = True
        if self.more:
            out["more"] = True
        if self.score:
            out["score"] = round(self.score, 1)
        return out


# ---- 단어 목록 -------------------------------------------------------------
# 한글 구절은 공백을 뺀 문자열에서 부분 일치, 영어 구절은 단어 경계로 찾는다.
CANCEL = ["취소", "처음부터", "그만할게", "안살게", "cancel", "start over", "never mind", "nevermind"]
# "to take out"(포장)과 "얼음 빼고"(요청사항)는 해석 전에 먼저 떼어 내므로 여기의 "take out"·"빼"에 걸리지 않는다
REMOVE = ["빼", "삭제", "지워", "remove", "delete", "take out"]
RECOMMEND = ["추천", "뭐가맛있", "뭐마실", "뭐가좋", "골라줘", "골라주", "모르겠", "recommend", "suggest",
             "what's good", "what is good", "something good", "popular", "help me choose"]
# 단맛 불만은 따로 둔다: "너무 달지 않은 거"·"not too sweet"처럼 부정이면 불만이 아니라 취향(LESS_SWEET)이다
REJECT_SWEET = ["너무달", "달아요", "too sweet"]
REJECT = REJECT_SWEET + ["너무써", "써요", "다른거", "다른걸", "별로", "too bitter", "something else", "another one",
                         "don't like", "do not like", "not that", "other one"]
CHECKOUT = ["결제", "계산", "그게다", "이게다", "다됐", "주문할게", "주문완료", "pay", "checkout", "check out",
            "that's all", "that is all", "that's it", "i'm done"]
ADD_MORE = ["더주문", "추가로", "추가할", "더시킬", "order more", "add another", "something more"]
EXPLAIN = ["뭐예요", "뭐에요", "뭔가요", "뭔데", "뭐야", "무슨맛", "어떤맛", "맛이어때", "설명", "들어가",
           "what is", "what s", "whats", "tell me about", "what does", "taste like", "how does it taste"]
REPEAT = ["다시말", "뭐라고", "못들었", "again", "repeat", "pardon"]
HELP = ["도와", "어떻게해", "어떻게하", "help", "how do i"]
# "please" 단독은 넣지 않는다: "Water please"·"One moment please"가 결제 승인이 되면 안 된다 ("yes please"는 yes로 잡힘)
YES_EN = ["yes", "yeah", "yep", "sure", "ok", "okay", "correct", "right", "sounds good",
          "i'll take it", "i will take it", "that one", "this one", "i'll have that", "i'll take that", "i'll have it",
          "that please", "go ahead", "continue", "resume", "go on", "keep going"]
# "주세요·할게요"는 넣지 않는다: "물 주세요"가 결제 승인이 되면 안 된다. 이어서·계속 = 일시정지에서 이어 가기.
# "그거 주세요·이걸로 할게요"(가리키는 말)는 예로 본다: 추천·설명·품절 대안처럼 가리킬 메뉴가 있을 때 FSM이 그 메뉴로 받는다
YES_KO = ["맞아", "좋아", "그래", "그럴게", "부탁", "그걸로", "그거로", "이걸로", "이거로", "저걸로", "그거요", "이거요",
          "저거요", "그거주", "이거주", "저거주", "그걸주", "이걸주", "그거할", "이거할", "저거할", "그거마실", "그거먹",
          "이어서", "계속"]
NO_EN = ["no", "nope", "nah", "not really", "no thanks", "no thank you"]
NO_KO = ["아니", "아뇨", "싫어", "됐어", "괜찮아", "괜찮습니다"]
# 예 단어가 들어 있어도 부정인 말: "That's not correct", "안 맞아요"
NEGATED = ["not right", "not correct", "not okay", "not ok", "isn't right", "is not right", "wrong", "incorrect",
           "안맞", "틀렸", "틀려", "아닌데"]
# 대답을 못 정한 말: 예/아니오로 보지 않는다 ("um I'm not sure")
UNSURE = ["not sure", "i don't know", "i do not know", "let me think", "글쎄"]
# 말 앞에 붙는 군말: 예/아니오를 볼 때 건너뛴다 ("아 네", "음 아니요")
FILLERS = {"아", "어", "음", "음음", "으음", "흠", "그", "저", "um", "uh", "hmm", "ah", "oh", "well", "so"}
HELLO = ["안녕하세요", "안녕", "hello", "hi", "hey", "good morning", "good afternoon", "good evening"]

# ---- 규칙 밖의 말 (직원 안내) ------------------------------------------------
# 직원을 찾는 말. "사람" 하나만으로는 찾지 않는다 ("두 사람이요")
STAFF = ["직원", "점원", "사장님", "매니저", "알바생", "사람불러", "사람좀불러", "사람있", "사람한테", "staff", "employee",
         "manager", "a person", "real person", "human", "someone who works", "talk to someone"]
# 키오스크가 답하면 안 되거나(알레르기: 틀리면 위험) 할 수 없는 일. 이유마다 FSM이 다른 말로 직원을 부른다
OUT_OF_SCOPE: dict[str, list[str]] = {
    "allergy": ["알레르기", "알러지", "견과", "땅콩", "유당", "글루텐", "성분", "우유들어가", "우유가들어가", "allergy", "allergic",
                "allergies", "nut", "nuts", "peanut", "peanuts", "lactose", "gluten", "dairy", "ingredient", "ingredients"],
    "payment": ["현금", "쿠폰", "포인트", "적립", "할인", "상품권", "기프티콘", "삼성페이", "애플페이", "카카오페이", "네이버페이",
                "cash", "coupon", "points", "discount", "voucher", "gift card", "apple pay", "samsung pay"],
    "refund": ["환불", "교환", "잘못나왔", "잘못나온", "안나왔", "안나와", "refund", "wrong drink", "wrong order",
               "never came", "complaint"],
    "facility": ["화장실", "와이파이", "비밀번호", "비번", "콘센트", "충전기", "restroom", "toilet", "bathroom", "wifi", "wi-fi",
                 "password", "outlet", "charger"],
}
# 메뉴 옵션에는 없지만 흔한 음료 요청: 정해 둔 것만 요청사항으로 받아 그 잔에 붙인다 (만들 때 직원이 봄)
NOTES: dict[str, list[str]] = {
    "less_ice": ["얼음적게", "얼음조금", "얼음은적게", "less ice", "light ice", "easy ice", "a little ice"],
    "no_ice": ["얼음빼", "얼음없이", "얼음은빼", "얼음없는", "no ice", "without ice"],
    "extra_ice": ["얼음많이", "얼음가득", "얼음추가", "extra ice", "more ice"],
    "extra_shot": ["샷추가", "샷하나추가", "샷더", "extra shot", "double shot", "add a shot"],
    "strong": ["진하게", "stronger", "strong please"],
    "weak": ["연하게", "weaker", "weak please"],
    "no_syrup": ["시럽빼", "시럽없이", "no syrup", "without syrup"],
    "extra_syrup": ["시럽추가", "시럽많이", "extra syrup", "more syrup"],
    "no_whip": ["휘핑빼", "휘핑없이", "휘핑크림빼", "no whip", "without whipped cream"],
    "extra_whip": ["휘핑추가", "휘핑많이", "extra whip"],
    "oat_milk": ["오트밀크", "귀리우유", "oat milk"],
    "soy_milk": ["두유로", "두유변경", "soy milk"],
}
# 영어 "take out the latte"는 메뉴 빼기라서, 포장은 "to take out"·"takeout"처럼 분명한 꼴만 받는다
TOGO = ["포장", "테이크아웃", "가져갈", "가지고갈", "들고갈", "to go", "to take out", "to take away", "for takeout",
        "for take out", "takeout", "take-out", "takeaway"]
DINE_HERE = ["매장에서", "먹고갈", "마시고갈", "여기서먹", "여기서마", "for here", "eat in", "dine in", "drink here"]
# 장바구니에서 이름으로 고치기 ("라떼는 아이스로 바꿔 주세요")
CHANGE = ["바꿔", "바꾸", "변경", "로해", "으로해", "로요", "change", "make it", "switch"]  # "라떼는 아이스로요"
# "X 말고 Y"·"X 대신 Y": X는 버리고 Y를 받는다. 메뉴 이름 바로 뒤(조사 허용)에 와야 한다
_NEG_AFTER_KO = re.compile(r"^(?:은|는|이|가|을|를|으로|로|요)?(?:말고|아니고|아니라|대신|싫)")
_NEG_BEFORE_EN = re.compile(r"(?:\bnot|\binstead of|\brather than|\bdon't want|\bdo not want)\s+(?:the\s+|a\s+|an\s+)?$")
# 요청사항·포장만 말했을 때 남는 군말 ("얼음 적게 해 주세요" → "해 주세요")
_FILLER_KO = re.compile(r"(?:으로요|로요|으로|로|이요|이에요|예요|에요|요|해요|해|주세요|줘|좀|부탁|드려|할게|할께|게|해주|"
                        r"이|가|은|는|도|만|랑|하고|그리고|네|아|음|그럼|저|그냥|하나|한잔|잔)+")
_FILLER_EN = {"please", "thanks", "thank", "you", "i", "id", "i'd", "want", "would", "like", "can", "could", "get",
              "make", "it", "to", "for", "and", "with", "a", "the", "my", "order", "drink", "that", "be", "also", "too",
              "oh", "um", "uh", "ok", "okay", "yes"}
_ORDINAL_KO = re.compile(r"(첫|두|세|네|[1-9])번째")
_ORDINAL_EN = re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\b")
_ORDINAL_WORDS = {"첫": 1, "두": 2, "세": 3, "네": 4, "first": 1, "second": 2, "third": 3, "fourth": 4,
                  "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
_SIDE = {"왼쪽": 1, "오른쪽": -1}  # 영어 "right"는 "That's right"(맞아요)와 겹쳐서 쓰지 않는다
# 여러 메뉴 나누기: 쉼표·그리고·and, 그리고 메뉴·수량 뒤에 붙은 "랑·하고·와·과"
_SPLIT_RE = re.compile(r"[,，/&+]|\band\b|\bplus\b|그리고")
_JOIN_SUFFIX = re.compile(r"^(.+?)(이랑|랑|하고|와|과)$")
_QTY_STEM = re.compile(r"^(?:한|두|세|네|다섯|하나|둘|셋|넷|[1-9])?(?:잔|개|컵)?$|^(?:하나|둘|셋|넷)$")
# 메뉴를 비교하는 질문("딸기랑 바닐라 중에 뭐가 나아요?")은 여러 잔 주문이 아니다
_QUESTION = ["뭐가", "뭐야", "뭐예요", "어떤게", "어떤거", "어때", "중에", "which", "what", "better"]

ICE = ["아이스", "차가운", "차갑게", "시원하게", "시원한", "iced", "ice", "cold"]
HOT = ["따뜻한", "따뜻하게", "따듯한", "따듯하게", "뜨거운", "뜨겁게", "핫", "hot", "warm"]
LARGE = ["라지", "큰거", "큰걸", "크게", "큰사이즈", "그란데", "벤티", "large", "big", "grande", "venti"]
REGULAR = ["레귤러", "작은", "작게", "보통", "기본", "톨", "regular", "small", "medium", "tall", "normal size"]

NO_CAFFEINE = ["커피말고", "커피아닌", "커피가아닌", "커피없는", "커피가없는", "커피안들어", "커피빼고", "커피는빼", "논커피",
               "카페인없", "카페인이없", "디카페인", "카페인빼", "without coffee", "no coffee", "not coffee",
               "non-coffee", "non coffee", "no caffeine", "without caffeine", "caffeine-free", "caffeine free",
               "decaf"]
LESS_SWEET = ["덜단", "덜달", "안단", "안달", "달지않", "단거말고", "단거빼고", "단거싫", "단게싫", "단건싫",
              "not sweet", "less sweet", "not too sweet", "not very sweet", "not so sweet", "not that sweet",
              "don't like sweet", "do not like sweet", "nothing sweet", "unsweetened", "no sugar", "sugar free",
              "sugar-free"]
WANT_SWEET = ["달달", "달콤", "단거", "sweet"]
REFRESHING = ["상큼", "청량", "과일", "새콤", "refreshing", "fruity", "light"]
MILKY = ["부드러", "우유", "고소", "milky", "creamy", "smooth"]
PREF_ICE = ["시원", "차가운", "아이스", "iced", "cold"]
PREF_HOT = ["따뜻", "따듯", "뜨거운", "warm", "hot"]

_KO_QTY = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4, "다섯": 5}
_EN_QTY = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_KO_QTY_RE = re.compile(r"(다섯|하나|한|두|둘|세|셋|네|넷|[1-9])(?:잔|개|컵)")
_KO_QTY_ALONE_RE = re.compile(r"(하나|둘|셋|넷)")
_EN_QTY_RE = re.compile(r"\b(one|two|three|four|five|[1-9])\b")
_YES_TOKEN_RE = re.compile(r"(네|예|응|넵|넹|옙)+요?")
_YES_FIRST_EN = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay"}
_NO_FIRST = re.compile(r"(no|nope|nah|아니.*|아뇨.*|아닙니다)")
# 두 글자 이하 한글 별칭("아아"·"아메"·"라떼")은 공백을 넘어 찾지 않는다: "아, 아이스로요"·"아 아니요"·"아 메뉴"가
# 아이스 아메리카노가 되던 문제. 별칭이 들어 있는 단어의 앞은 온도·크기 말만, 뒤는 조사·수량·요청 말만 허용한다
_ALIAS_TAIL = re.compile(r"(?:요|로|으로|를|을|은|는|이|가|도|만|랑|하고|하나|한|두|세|네|[1-9]|잔|개|컵|주세요|줘|좀)*")
_ALIAS_HEAD = ("아이스", "따뜻한", "따듯한", "뜨거운", "차가운", "시원한", "라지", "레귤러", "큰", "작은")


def _short_alias_ok(alias: str, tokens: list[str]) -> bool:
    for token in tokens:
        start = token.find(alias)
        while start != -1:
            head, tail = token[:start], token[start + len(alias):]
            if (not head or head.endswith(_ALIAS_HEAD)) and _ALIAS_TAIL.fullmatch(tail):
                return True
            start = token.find(alias, start + 1)
    return False


@lru_cache(maxsize=None)
def _pattern(phrase: str) -> tuple[bool, str | re.Pattern[str]]:
    """(한글인지, 비교용 문자열 또는 정규식). 구절도 입력과 똑같이 정규화한다."""
    norm = normalize(phrase)
    if is_korean(norm):
        return True, compact(norm)
    return False, re.compile(r"(?<![a-z])" + re.escape(norm) + r"(?![a-z])")


def _has(norm: str, comp: str, phrases: list[str]) -> bool:
    for phrase in phrases:
        korean, pat = _pattern(phrase)
        if korean:
            if pat in comp:  # type: ignore[operator]
                return True
        elif pat.search(norm):  # type: ignore[union-attr]
            return True
    return False


@lru_cache(maxsize=None)
def _spaced(compact_phrase: str) -> re.Pattern[str]:
    """공백 없는 한글 구절 → 글자 사이 공백을 허용하는 정규식 ("얼음적게" → "얼음 적게"도 찾음)."""
    return re.compile(r"\s*".join(re.escape(ch) for ch in compact_phrase))


def _strip_phrases(norm: str, phrases: list[str]) -> tuple[str, bool]:
    """구절을 문장에서 지운다. (남은 문장, 하나라도 찾았는지)."""
    found = False
    for phrase in phrases:
        korean, pat = _pattern(phrase)
        regex = _spaced(pat) if korean else pat  # type: ignore[arg-type]
        stripped = regex.sub(" ", norm)  # type: ignore[union-attr]
        if stripped != norm:
            found, norm = True, stripped
    return re.sub(r"\s+", " ", norm).strip(), found


def _rest_is_filler(norm: str, comp: str) -> bool:
    if not comp:
        return True
    if is_korean(comp):
        return bool(_FILLER_KO.fullmatch(comp))
    return all(token in _FILLER_EN for token in norm.split())


class NLU:
    def __init__(self, menu: Menu, min_score: float = 80.0):
        self.menu = menu
        self.min_score = min_score
        # (공백 없는 별칭, 메뉴 id, 원래 별칭) — 긴 별칭부터 검사해야 "바닐라 라떼"가 "라떼"보다 먼저 잡힌다
        self._aliases: list[tuple[str, str, str]] = []
        for item in menu:
            for alias in set(item.aliases) | set(item.name.values()):
                self._aliases.append((compact(normalize(alias)), item.id, alias))
        self._aliases.sort(key=lambda entry: len(entry[0]), reverse=True)

    # ---- 메뉴 찾기 ----------------------------------------------------------
    def find_item(self, comp: str, norm: str | None = None) -> tuple[str | None, float, list[str], dict[str, str]]:
        """(id, 점수, 후보 최대 3개, 별칭이 암시하는 옵션). 가장 긴 별칭의 부분 일치가 우선, 없으면 퍼지 매칭.

        norm(공백이 있는 정규화 문장)을 주면 두 글자 이하 한글 별칭은 단어 안에서만 찾는다 (_short_alias_ok)."""
        tokens = (norm or comp).split()
        for alias_comp, item_id, alias in self._aliases:
            if not alias_comp or alias_comp not in comp:
                continue
            if len(alias_comp) <= 2 and is_korean(alias_comp) and not _short_alias_ok(alias_comp, tokens):
                continue
            item = self.menu.get(item_id)
            options = item.alias_options.get(alias, {}) if item else {}
            return item_id, 100.0, [item_id], options
        if len(comp) < 2:
            return None, 0.0, [], {}
        best: dict[str, float] = {}
        for alias_comp, item_id, _alias in self._aliases:
            if len(alias_comp) < 3:  # 두 글자 별칭은 퍼지 매칭에서 오탐이 많다
                continue
            # 발화가 별칭보다 짧으면 전체 비교, 길면 발화 속 가장 비슷한 부분과 비교
            if len(comp) < len(alias_comp):
                score = fuzz.ratio(alias_comp, comp)
            else:
                score = fuzz.partial_ratio(alias_comp, comp)
            if score > best.get(item_id, 0.0):
                best[item_id] = score
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        candidates = [item_id for item_id, score in ranked[:3] if score >= 60]
        # 두 글자 발화("바닐")는 확정하지 않고 후보로만 보여준다
        if ranked and ranked[0][1] >= self.min_score and len(comp) >= 3:
            return ranked[0][0], ranked[0][1], candidates, {}
        return None, (ranked[0][1] if ranked else 0.0), candidates, {}

    def _mentions(self, comp: str, tokens: list[str]) -> list[tuple[int, int, str, str]]:
        """문장 속 모든 메뉴 언급 (시작, 끝, id, 별칭). 긴 별칭부터, 겹치지 않게 (comp 기준 위치)."""
        taken = [False] * len(comp)
        found: list[tuple[int, int, str, str]] = []
        for alias_comp, item_id, alias in self._aliases:
            if not alias_comp:
                continue
            if len(alias_comp) <= 2 and is_korean(alias_comp) and not _short_alias_ok(alias_comp, tokens):
                continue
            start = comp.find(alias_comp)
            while start != -1:
                end = start + len(alias_comp)
                if not any(taken[start:end]):
                    taken[start:end] = [True] * (end - start)
                    found.append((start, end, item_id, alias))
                start = comp.find(alias_comp, start + 1)
        found.sort()
        return found

    @staticmethod
    def _negated(norm: str, comp: str, mentions: list[tuple[int, int, str, str]]) -> set[int]:
        """'X 말고'·'X 대신'·'not X'·'instead of X'로 버린 언급의 번호."""
        index = [i for i, ch in enumerate(norm) if ch != " "]  # comp 위치 → norm 위치
        out: set[int] = set()
        for k, (start, end, _item, _alias) in enumerate(mentions):
            if _NEG_AFTER_KO.match(comp[end:]):
                out.add(k)
            elif start < len(index) and _NEG_BEFORE_EN.search(norm[:index[start]]):
                out.add(k)
        return out

    def _multi_items(self, text: str) -> list[dict[str, Any]]:
        """여러 메뉴를 한 번에 말했으면 메뉴마다 {item, qty, temp, size, notes}. 아니면 빈 목록.

        쉼표·그리고·and로, 그리고 메뉴·수량 뒤의 "랑·하고·와·과"로 조각을 나눈 뒤 조각마다 메뉴 하나를 찾는다.
        메뉴 둘을 견주는 질문("딸기랑 바닐라 중에 뭐가 나아요?")이나 한 조각에 메뉴가 둘이면("아메리카노 말고 라떼") 아니다.
        """
        whole = normalize(text)
        if "?" in text or _has(whole, compact(whole), _QUESTION):
            return []
        chunks: list[str] = []
        for part in _SPLIT_RE.split(text.lower()):
            current: list[str] = []
            for token in normalize(part).split():
                m = _JOIN_SUFFIX.match(token)
                stem = m.group(1) if m else ""
                if m and (_QTY_STEM.match(stem) or self._mentions(compact(stem), [stem])):
                    current.append(stem)
                    chunks.append(" ".join(current))
                    current = []
                else:
                    current.append(token)
            if current:
                chunks.append(" ".join(current))
        entries: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_norm, notes = self._notes(chunk)
            chunk_norm, _dine = self._dine(chunk_norm)
            comp = compact(chunk_norm)
            mentions = self._mentions(comp, chunk_norm.split())
            temp, size = self._temp(chunk_norm, comp), self._size(chunk_norm, comp)
            if not mentions:
                if entries:  # 메뉴 없는 조각("라떼 하나, 아이스로")은 앞 메뉴의 옵션으로
                    last = entries[-1]
                    last["temp"] = last["temp"] or temp
                    last["size"] = last["size"] or size
                    last["notes"] = last["notes"] + [n for n in notes if n not in last["notes"]]
                continue
            if len(mentions) > 1 or self._negated(chunk_norm, comp, mentions):
                return []
            _start, _end, item_id, alias = mentions[0]
            item = self.menu.get(item_id)
            alias_temp = item.alias_options.get(alias, {}).get("temp") if item else None
            entries.append({"item": item_id, "qty": self._qty(chunk_norm, comp), "temp": alias_temp or temp,
                            "size": size, "notes": notes})
        return entries if len(entries) >= 2 else []

    # ---- 슬롯 ---------------------------------------------------------------
    @staticmethod
    def _notes(norm: str) -> tuple[str, list[str]]:
        found: list[str] = []
        for key, phrases in NOTES.items():
            norm, hit = _strip_phrases(norm, phrases)
            if hit:
                found.append(key)
        return norm, found

    @staticmethod
    def _dine(norm: str) -> tuple[str, str | None]:
        norm, togo = _strip_phrases(norm, TOGO)
        norm, here = _strip_phrases(norm, DINE_HERE)
        return norm, ("togo" if togo else "here" if here else None)

    @staticmethod
    def _out_of_scope(norm: str, comp: str) -> str | None:
        for reason, phrases in OUT_OF_SCOPE.items():
            if _has(norm, comp, phrases):
                return reason
        return None

    @staticmethod
    def _ordinal(norm: str, comp: str) -> int | None:
        m = _ORDINAL_KO.search(comp)
        if m:
            token = m.group(1)
            return int(token) if token.isdigit() else _ORDINAL_WORDS[token]
        m = _ORDINAL_EN.search(norm)
        if m:
            return _ORDINAL_WORDS[m.group(1)]
        for word, n in _SIDE.items():
            if word in comp:
                return n
        return None

    @staticmethod
    def _more(norm: str) -> bool:
        for token in norm.split():
            if token in ("더", "하나더", "한잔더", "한개더", "잔더") or (
                    token.startswith("더") and token[1:] in ("요", "주세요", "줘", "할게요", "추가", "추가요")):
                return True
        return bool(re.search(r"\b(?:one more|another)\b", norm))

    @staticmethod
    def _temp(norm: str, comp: str) -> str | None:
        if _has(norm, comp, ICE):
            return "ice"
        if _has(norm, comp, HOT):
            return "hot"
        return None

    @staticmethod
    def _size(norm: str, comp: str) -> str | None:
        if _has(norm, comp, LARGE):
            return "large"
        if _has(norm, comp, REGULAR):
            return "regular"
        return None

    @staticmethod
    def _qty(norm: str, comp: str) -> int | None:
        m = _KO_QTY_RE.search(comp)
        if m:
            token = m.group(1)
            return int(token) if token.isdigit() else _KO_QTY[token]
        m = _KO_QTY_ALONE_RE.search(comp)
        if m:
            return _KO_QTY[m.group(1)]
        m = _EN_QTY_RE.search(norm)
        if m:
            token = m.group(1)
            return int(token) if token.isdigit() else _EN_QTY[token]
        return None

    @staticmethod
    def _prefs(norm: str, comp: str) -> dict[str, Any]:
        prefs: dict[str, Any] = {}
        tags: list[str] = []
        if _has(norm, comp, NO_CAFFEINE):
            prefs["caffeine"] = False
        if _has(norm, comp, LESS_SWEET):
            prefs["sweet_max"] = 1
        elif _has(norm, comp, WANT_SWEET) and not _has(norm, comp, REJECT):
            tags.append("sweet")
        if _has(norm, comp, REFRESHING):
            tags.append("refreshing")
        if _has(norm, comp, MILKY):
            tags.append("milky")
        if _has(norm, comp, PREF_ICE):
            prefs["temp"] = "ice"
        elif _has(norm, comp, PREF_HOT):
            prefs["temp"] = "hot"
        if tags:
            prefs["tags"] = tags
        return prefs

    @staticmethod
    def _answer(norm: str, comp: str) -> str | None:
        """예/아니오 판단: "confirm" | "deny" | "unsure" | None.

        순서: 군말을 건너뛴 첫 단어가 예·아니오면 그것 → 예 단어가 있어도 부정인 말 → 못 정한 말 → 예 단어 → 아니오 단어.
        예전에는 예 단어를 먼저 찾아 "No, that's not right"·"안 맞아요"도 결제 승인이 됐다.
        """
        tokens = norm.split()
        while tokens and tokens[0] in FILLERS:
            tokens.pop(0)
        if not tokens:
            return None
        first = tokens[0]
        if _YES_TOKEN_RE.fullmatch(first) or first in _YES_FIRST_EN:
            return "confirm"
        if _NO_FIRST.fullmatch(first):
            return "deny"
        if _has(norm, comp, NEGATED):
            return "deny"
        if _has(norm, comp, UNSURE):
            return "unsure"
        if _has(norm, comp, YES_EN) or _has(norm, comp, YES_KO):
            return "confirm"
        if _has(norm, comp, NO_EN) or _has(norm, comp, NO_KO):
            return "deny"
        return None

    @staticmethod
    def _reject_reason(norm: str, comp: str) -> str | None:
        if _has(norm, comp, ["달", "sweet"]):
            return "sweet"
        if _has(norm, comp, ["써", "쓴", "bitter"]):
            return "bitter"
        return None

    # ---- 해석 ---------------------------------------------------------------
    def parse(self, text: str, context: str = "menu") -> Intent:
        """context: greet | menu | offer_help | recommend | sold_out | options | cart | pay | paused | done."""
        norm = normalize(text)
        comp = compact(norm)
        lang = "ko" if is_korean(norm) else "en"
        if not comp:
            return Intent("unknown", text=text, lang=lang)

        # 0) 여러 메뉴를 한 번에 ("라떼 하나랑 아이스 아메리카노 두 잔") — 요청사항은 메뉴마다 따로 붙는다
        items = self._multi_items(text)
        # 요청사항·포장을 먼저 떼어 내고 나머지를 해석한다: "라떼 얼음 빼고"가 라떼 빼기가 되지 않게
        norm, notes = self._notes(norm)
        norm, dine = self._dine(norm)
        comp = compact(norm)
        extras: dict[str, Any] = {"notes": [] if items else notes, "dine": dine,
                                  "change": _has(norm, comp, CHANGE), "more": self._more(norm)}

        def make(name: str, **kw: Any) -> Intent:
            for key, value in extras.items():
                kw.setdefault(key, value)
            return Intent(name, text=text, lang=lang, **kw)

        if items:
            first = items[0]
            return make("order_item", item=first["item"], qty=first["qty"], temp=first["temp"], size=first["size"],
                        score=100.0, items=items)
        if (notes or dine) and _rest_is_filler(norm, comp):
            return make("note" if notes else "dine")  # "얼음 적게 해 주세요", "포장이요"

        # 1) 키오스크가 하면 안 되거나 못 하는 일, 직원을 찾는 말 → FSM이 직원을 부른다
        reason = self._out_of_scope(norm, comp)
        item_id, score, candidates, alias_opts = self.find_item(comp, norm)
        if reason:
            return make("out_of_scope", reason=reason, item=item_id)
        if _has(norm, comp, STAFF) and not _has(norm, comp, RECOMMEND):
            return make("staff")

        # 2) "아메리카노 말고 라떼"·"latte instead of americano": 버린 메뉴는 따로, 남은 메뉴로 해석
        mentions = self._mentions(comp, norm.split())
        negated = self._negated(norm, comp, mentions)
        distinct = list(dict.fromkeys(m[2] for m in mentions))
        if not negated and len(distinct) >= 2 and ("?" in text or _has(norm, comp, _QUESTION)):
            # "딸기랑 바닐라 중에 뭐가 나아요?": 주문이 아니라 둘 중 추천해 달라는 질문
            return make("compare", candidates=distinct, prefs=self._prefs(norm, comp))
        if negated:
            dropped = list(dict.fromkeys(mentions[k][2] for k in sorted(negated)))
            kept = [m for k, m in enumerate(mentions) if k not in negated and m[2] not in dropped]
            extras["rejected_items"] = dropped
            if not kept:
                return make("reject_item", item=dropped[0])  # "아메리카노 말고요", "라떼는 싫어요"
            _start, _end, item_id, alias = kept[0]
            kept_item = self.menu.get(item_id)
            alias_opts = kept_item.alias_options.get(alias, {}) if kept_item else {}
            score, candidates = 100.0, [item_id]

        temp = alias_opts.get("temp") or self._temp(norm, comp)
        size = self._size(norm, comp)
        qty = self._qty(norm, comp)
        prefs = self._prefs(norm, comp)
        option_context = context in ("options", "cart")
        # "너무 달지 않은 거"·"not too sweet": 단맛 불만 구절이 있어도 부정이면 불만이 아니다
        rejected = _has(norm, comp, REJECT[len(REJECT_SWEET):]) or (
            _has(norm, comp, REJECT_SWEET) and not _has(norm, comp, LESS_SWEET))

        # 3) 명시적인 동작이 먼저
        if item_id and not negated and _has(norm, comp, REMOVE + ["취소"]):
            return make("remove_item", item=item_id, score=score, qty=qty, temp=temp, size=size)
        if item_id is None and _has(norm, comp, CANCEL):
            return make("cancel")
        if rejected and not negated:
            return make("reject_feedback", reason=self._reject_reason(norm, comp), prefs=prefs)
        if _has(norm, comp, RECOMMEND):
            return make("ask_recommend", prefs=prefs)
        # 4) 메뉴 이름 + 질문이면 설명 ("아인슈페너가 뭐예요?"), 아니면 주문
        if item_id and not negated and _has(norm, comp, EXPLAIN):
            return make("explain_item", item=item_id, score=score)
        if item_id:
            return make("order_item", item=item_id, qty=qty, temp=temp, size=size, score=score,
                        candidates=candidates)
        # 5) "첫 번째 걸로", "오른쪽 거요": 화면에 보이는 후보·대안 중 몇 번째인지 (FSM이 목록에서 고른다)
        ordinal = self._ordinal(norm, comp)
        if ordinal is not None:
            return make("choose", ordinal=ordinal)
        # 6) 옵션을 묻는 중이면 온도·사이즈가 우선
        if option_context and (temp or size):
            return make("set_option", temp=temp, size=size)
        # 7) "네, 커피 말고요"처럼 취향이 들어 있으면 취향으로 본다
        if prefs and not option_context and context not in ("pay", "done"):
            return make("preference", prefs=prefs)
        # 8) 결제·다시 말하기·도움 요청이 예/아니오보다 먼저
        if _has(norm, comp, CHECKOUT):
            return make("checkout")
        if _has(norm, comp, REPEAT):
            return make("repeat")
        if _has(norm, comp, HELP):
            return make("help")
        answer = self._answer(norm, comp)
        if answer == "confirm":
            return make("confirm")
        if answer == "deny":
            return make("deny")
        if answer == "unsure":
            return make("unknown")
        if temp or size:
            return make("set_option", temp=temp, size=size)
        if _has(norm, comp, ADD_MORE):
            return make("add_more")
        if _has(norm, comp, HELLO):
            return make("hello")
        return make("unknown", score=score, candidates=candidates)
