"""상호작용 매니저 (주문 FSM).

- 주문 상태(IDLE … DONE)와 사람 상태(human_state)를 따로 추적하고, 둘을 합쳐 다음 행동을 정한다.
- 입력: human.state, touch, speech.utterance, card 이벤트 + 10 Hz tick(타이머).
- 출력: Act (BehaviorManager가 화면·음성·모션·시선·LED로 실행).
- 시간은 모두 인자로 받은 now(초)를 쓰므로 테스트에서 시계를 마음대로 돌릴 수 있다.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Protocol

from ..behavior.acts import Act
from ..bus import Event, EventLog
from ..perception.human_state import ABSENT, AMBIGUOUS, ENGAGED, NO_RESPONSE, PRESENT, UNCERTAIN
from .menu import Cart, CartLine, Menu, MenuItem, OrderCounter
from .nlu import NLU, Intent
from .recommender import Prefs, Recommender


class S(StrEnum):
    IDLE = "IDLE"
    GREET = "GREET"
    MENU = "MENU"
    OFFER_HELP = "OFFER_HELP"
    RECOMMEND = "RECOMMEND"
    SOLD_OUT = "SOLD_OUT"
    OPTIONS = "OPTIONS"
    CART = "CART"
    PAY_WAIT_CARD = "PAY_WAIT_CARD"
    PAY_PROCESSING = "PAY_PROCESSING"
    PAY_REMOVE_CARD = "PAY_REMOVE_CARD"
    DONE = "DONE"


class Ov(StrEnum):
    NONE = "NONE"
    PAUSED = "PAUSED"
    ONE_AT_A_TIME = "ONE_AT_A_TIME"
    CANDIDATES = "CANDIDATES"
    DETAIL = "DETAIL"


AWAITING = {S.GREET, S.MENU, S.OFFER_HELP, S.RECOMMEND, S.SOLD_OUT, S.OPTIONS, S.CART}
PAUSABLE = {S.MENU, S.OFFER_HELP, S.RECOMMEND, S.SOLD_OUT, S.OPTIONS, S.CART, S.PAY_WAIT_CARD, S.PAY_REMOVE_CARD}
CONTEXT = {S.IDLE: "greet", S.GREET: "greet", S.MENU: "menu", S.OFFER_HELP: "offer_help",
           S.RECOMMEND: "recommend", S.SOLD_OUT: "sold_out", S.OPTIONS: "options", S.CART: "cart",
           S.PAY_WAIT_CARD: "pay", S.PAY_PROCESSING: "pay", S.PAY_REMOVE_CARD: "pay", S.DONE: "done"}
HERE = {PRESENT, ENGAGED, NO_RESPONSE}
# 결제가 승인된 뒤에는 취소하지 않는다 (카드가 꽂혀 있거나 주문번호가 나간 뒤)
NO_CANCEL = {S.PAY_PROCESSING, S.PAY_REMOVE_CARD, S.DONE}
# 한 번만 보이고 그 상태의 기본 계획으로 돌아가는 계획 (호객, 되묻기). 나머지 계획은 그 상태의 기본 계획이 된다 (_act)
MOMENTARY_PLANS = frozenset({"invite", "repair", "repair_candidates", "repair_touch", "staff", "staff_done"})
# 직원을 부르는 이유 → 로봇이 하는 말. 못 하는 일은 지어내서 답하지 않고 솔직하게 말한 뒤 사람에게 넘긴다
STAFF_SAY = {"request": "staff_request", "allergy": "staff_allergy", "payment": "staff_payment",
             "refund": "staff_refund", "facility": "staff_facility", "repair": "staff_repair"}
# 못 알아들은 말이 이만큼 이어지면 직원을 부른다 (1번째 되묻기 → 2번째 화면 권유 → 3번째 직원)
STAFF_AFTER_REPAIRS = 3
# "다시 말해 주세요"로 반복할 때 바꿔 말하는 문장 (인사를 두 번 하지 않게)
REPEAT_AS = {"greet_menu": "menu_prompt"}


class Outputs(Protocol):
    def perform(self, act: Act) -> None: ...

    def set_idle(self, enabled: bool) -> None: ...

    def set_lang(self, lang: str) -> None: ...


@dataclass
class Session:
    id: int
    lang: str
    started: float
    lang_locked: bool = False
    cart: Cart = field(default_factory=Cart)
    pending: CartLine | None = None
    prefs: Prefs = field(default_factory=Prefs)
    rejected: set[str] = field(default_factory=set)
    rec_item: str | None = None
    rec_rejections: int = 0
    sold_out_item: str | None = None
    alternatives: list[str] = field(default_factory=list)
    help_offered: bool = False
    menu_prompted: bool = False
    greeted: bool = False  # 이 손님에게 인사(인사 모션 + 인사말)를 했는지: 두 번 절하지 않게
    repair_count: int = 0
    resume_state: S | None = None
    order_number: int | None = None
    fallback_pay: bool = False
    busy_notice_t: float = -math.inf
    explained_item: str | None = None
    candidates: list[str] = field(default_factory=list)       # 되묻기 창에 띄운 후보 ("첫 번째 걸로")
    queue: list[Intent] = field(default_factory=list)         # 한 번에 말한 여러 메뉴 중 아직 안 담은 것
    multi: bool = False                                       # 여러 메뉴 주문 중: 옵션을 물을 때 메뉴 이름을 붙인다
    pending_notes: list[str] = field(default_factory=list)    # 메뉴를 고르기 전에 말한 요청사항 (다음 잔에 붙임)
    dine: str | None = None                                   # togo | here
    rec_spoken: str | None = None                             # 마지막으로 말로 추천한 메뉴 (칩을 눌러도 같으면 다시 말하지 않음)
    staff_called: bool = False                                # 직원을 불렀고 아직 응대 전
    staff_reason: str | None = None
    rec_pool: set[str] = field(default_factory=set)           # 이 메뉴들 중에서만 추천 ("딸기랑 바닐라 중에 뭐가 나아요?")


class InteractionManager:
    def __init__(self, cfg: dict[str, Any], menu: Menu, nlu: NLU, out: Outputs, counter: OrderCounter,
                 log: EventLog | None = None, recommender: Recommender | None = None,
                 on_lock_primary: Callable[[], None] | None = None,
                 features: dict[str, bool] | None = None):
        self.t = cfg["timing"]
        # BehaviorManager와 같은 사전을 공유 (오퍼레이터 패널에서 켜고 끄면 바로 반영)
        self.features = features if features is not None else {k: bool(v) for k, v in cfg["features"].items()}
        self.default_lang = cfg["session"]["default_lang"]
        self.menu = menu
        self.nlu = nlu
        self.out = out
        self.counter = counter
        self.log = log or EventLog(None)
        self.rec = recommender or Recommender(menu)
        self.on_lock_primary = on_lock_primary
        self.state = S.IDLE
        self.overlay = Ov.NONE
        self.overlay_t = 0.0
        self.session: Session | None = None
        self.state_t = 0.0
        self.awaiting_since: float | None = None
        self.human = ABSENT
        self.human_t = 0.0
        self.card_in = False
        self.flags: dict[str, Any] = {}
        self.lang = self.default_lang
        self._session_seq = 0
        self.human_reason: str | None = None            # NO_RESPONSE 이유: hesitation | seek_help
        self.cues: dict[str, Any] = {"zone": "none", "approaching": False, "attention": "none"}
        self.context: dict[str, str] = {"weather": "normal"}  # 오퍼레이터가 설정하는 날씨 (context_talk)
        self.invite_t = -math.inf
        self.invited = False
        self._base_plan: tuple[str, dict[str, Any]] = ("waiting", {})  # 지금 상태의 기본 계획 (_act)
        # 주문을 마친 손님이 아직 앞에 있음: 같은 사람에게 다시 고개 숙여 인사하지 않는다.
        # 화면에서 사라지거나(ABSENT) 새로 다가오는 사람이 보이거나(approaching) 터치·말이 있으면 풀린다
        self.served = False
        self.device_lost = False  # 카드 슬롯 에이전트가 연결됐다가 끊김 (처음부터 없으면 오퍼레이터가 카드를 흉내 내는 개발 모드)
        self._last_say: dict[str, Any] | None = None  # "다시 말해 주세요"에 그대로 반복할 직전 말

    # ======================================================================
    # 공개 API
    # ======================================================================
    def start(self, now: float) -> None:
        self._enter(S.IDLE, now)

    def handle(self, event: Event, now: float) -> None:
        kind = event.kind
        if kind == "human.state":
            self.human_reason = event.get("reason")
            self._on_human(event.get("state"), now)
        elif kind == "human.cues":
            self._on_cues(event.data, now)
        elif kind == "touch":
            self._on_touch(event.data, now)
        elif kind == "speech.utterance":
            self._on_utterance(str(event.get("text", "")), now)
        elif kind == "card":
            self._on_card(event.get("state"), now)

    def tick(self, now: float) -> None:
        if self.state == S.IDLE:
            # 리셋·취소 뒤에도 사람이 계속 서 있으면 상태 변화 이벤트가 없으므로 다시 인사한다.
            # 단, 방금 주문을 마친 손님(served)에게는 다시 고개 숙여 인사하지 않는다 (_idle_presence)
            if self.human in HERE | {AMBIGUOUS} and now - self.state_t >= self.t.get("regreet_after_s", 6.0):
                self._idle_presence(now)
            return
        elapsed = now - self.state_t
        if self.overlay == Ov.PAUSED:
            if now - self.overlay_t >= self.t["resume_window_s"]:
                self._cancel(now, silent=True)
            return
        if (self.state in PAUSABLE and self.human == ABSENT and not self.card_in
                and now - self.human_t >= self._pause_after(self.state)):
            self._pause(now)
            return
        if self.state == S.GREET:
            # 인사 뒤 말없이 망설이면 같은 말을 되풀이하지 않고 로봇이 먼저 메뉴로 이끈다 (화면을 보며 "메뉴를 보여 드릴게요")
            if elapsed >= self.t["greet_repeat_s"] and self.overlay == Ov.NONE:
                self._lead_to_menu(now)
        elif self.state == S.RECOMMEND:
            # 취향 칩을 연달아 누르는 동안은 카드만 바꾸고, 손이 멈추면 한 번만 말한다 (Aha! 모션 없이)
            due = self.flags.get("rec_say_at")
            if due is not None and now >= due:
                self.flags.pop("rec_say_at", None)
                self._speak_recommendation()
        elif self.state == S.PAY_WAIT_CARD:
            if elapsed >= self.t["card_wait_s"] and not self.flags.get("card_reminded"):
                self.flags["card_reminded"] = True
                self._act("pay_repeat", say="pay_repeat", screen="payment",
                          data=self._pay_data("insert", fallback=True), led="blink", glance="slot")
        elif self.state == S.PAY_PROCESSING:
            if elapsed >= self.t["processing_s"]:
                session = self._s()
                if session.fallback_pay:
                    self._enter(S.DONE, now)
                elif self.card_in:
                    self._enter(S.PAY_REMOVE_CARD, now)
                else:
                    self._enter(S.PAY_WAIT_CARD, now, retry=True)
        elif self.state == S.PAY_REMOVE_CARD:
            if elapsed >= self.t["card_remove_s"] and not self.flags.get("forgot"):
                self.flags["forgot"] = True
                self._act("forgot_card", say="forgot_card", glance="slot", led="blink")
        elif self.state == S.DONE:
            if elapsed >= self.t["done_reset_s"]:
                self._enter(S.IDLE, now)

    def _pause_after(self, state: S) -> float:
        """세션 중 ABSENT가 이만큼 이어지면 일시정지한다.

        카드를 넣을 차례에는 손님이 지갑을 찾거나 화면 아래 투입구 쪽으로 몸을 숙여 얼굴이 카메라에서 잠깐 사라지는 게
        자연스럽다. 보통 상태처럼 몇 초 만에 멈추면 카드를 찾는 손님에게 "잠시 기다릴게요"가 뜨므로 훨씬 길게 기다린다.
        """
        if state == S.PAY_WAIT_CARD:
            return float(self.t.get("pay_pause_after_absent_s", 20.0))
        return float(self.t["pause_after_absent_s"])

    def reset(self, now: float) -> None:
        self.log.log("fsm.reset")
        self.overlay = Ov.NONE
        self._enter(S.IDLE, now)

    def force(self, name: str, now: float) -> bool:
        """오퍼레이터 패널: 특정 상태로 강제 이동 (필요한 장바구니·메뉴는 기본값으로 채운다)."""
        try:
            state = S(name)
        except ValueError:
            return False
        self.overlay = Ov.NONE
        if state == S.IDLE:
            self._enter(S.IDLE, now)
            return True
        session = self.session or self._new_session(now)
        if state in (S.CART, S.PAY_WAIT_CARD, S.PAY_PROCESSING, S.PAY_REMOVE_CARD, S.DONE) and session.cart.is_empty():
            session.cart.add(CartLine("latte", 1, "hot", "regular"))
        if state == S.OPTIONS and session.pending is None:
            session.pending = CartLine("latte")
        kw: dict[str, Any] = {}
        if state == S.SOLD_OUT:
            item = next((i for i in self.menu if not i.in_stock), None) or self.menu.get("einspanner")
            if item is None:
                return False
            kw["item"] = item
        self.log.log("fsm.force", state=str(state))
        self._enter(state, now, **kw)
        return True

    def set_lang(self, lang: str, now: float, locked: bool = True) -> None:
        if lang not in ("ko", "en"):
            return
        self.lang = lang
        if self.session is not None:
            self.session.lang = lang
            self.session.lang_locked = self.session.lang_locked or locked
        changed = lang != self.lang_spoken
        self.out.set_lang(lang)
        self.log.log("lang", lang=lang, locked=locked)
        if locked and self.state in (S.IDLE, S.GREET) and self.overlay == Ov.NONE:
            if self.session is None:
                self._new_session(now)
            self._enter(S.MENU, now, say="greet_menu")
        elif locked and changed and self.session is not None and self.state in AWAITING and self.overlay == Ov.NONE:
            # 주문 도중 화면의 언어 버튼: 지금 하던 질문을 새 언어로 다시 한다 (화면만 바뀌고 로봇은 조용하면 어색하다)
            self._reprompt(now)

    @property
    def lang_spoken(self) -> str:
        """로봇이 마지막으로 말한 언어 (언어 버튼을 눌렀을 때 실제로 바뀌었는지 보려고)."""
        return str((self._last_say or {}).get("lang") or self.default_lang)

    def staff_done(self, now: float) -> None:
        """오퍼레이터 패널(직원 역할): 직원이 손님 응대를 마침 → 알림을 내리고 주문을 이어 간다."""
        session = self.session
        if session is None or not session.staff_called:
            self._act("staff_done", notice="none", clear_overlay=False)
            return
        session.staff_called = False
        self.log.log("staff.done", reason=session.staff_reason, session=session.id, state=str(self.state))
        self._act("staff_done", say="staff_thanks", notice="none", gaze="user", clear_overlay=False, plan="staff_done")
        if self.overlay == Ov.NONE:
            # 직원을 부르기 전에 기다리던 질문을 이어서 다시 한다 ("…이어서 도와드릴게요. 따뜻하게 드릴까요, 아이스로…")
            self._reprompt(now, queued=True)
        if self.state in AWAITING:
            self.awaiting_since = now

    def on_device(self, state: str | None, now: float) -> None:
        """카드 슬롯 에이전트(Pi 4) 연결 상태. 결제 대기 중에 끊기면 바로 화면 결제를 안내한다."""
        if state in ("connected", "hello"):
            self.device_lost = False
        elif state == "disconnected":
            self.device_lost = True
            if self.state == S.PAY_WAIT_CARD and self.overlay == Ov.NONE:
                self._act("slot_down", say="pay_slot_down", screen="payment", data=self._pay_data("insert", fallback=True),
                          led="off", plan="pay_insert")

    def set_context(self, weather: str) -> bool:
        """오퍼레이터 패널: 오늘 날씨 (normal | cold | hot | rain). context_talk가 켜져 있으면 인사·추천에 반영."""
        if weather not in ("normal", "cold", "hot", "rain"):
            return False
        self.context["weather"] = weather
        self.log.log("context", weather=weather)
        return True

    def refresh(self) -> None:
        """재고가 바뀌었을 때 메뉴 화면만 다시 그린다 (음성 없음)."""
        if self.state in (S.MENU, S.OFFER_HELP) and self.session is not None:
            self._act("refresh", screen="menu", data=self._menu_data(), clear_overlay=False)

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        session = self.session
        out: dict[str, Any] = {
            "state": str(self.state), "overlay": str(self.overlay), "human": self.human, "lang": self.lang,
            "card_in": self.card_in, "awaiting": self.awaiting_since is not None,
            "stock": {item.id: item.in_stock for item in self.menu},
            "cues": dict(self.cues), "context": dict(self.context), "human_reason": self.human_reason,
            "session": None,
        }
        if session is not None:
            out["session"] = {
                "id": session.id, "lang": session.lang, "cart": session.cart.to_ui(self.menu),
                "pending": session.pending.to_dict() if session.pending else None,
                "prefs": session.prefs.to_ui(), "rejected": sorted(session.rejected), "rec_item": session.rec_item,
                "help_offered": session.help_offered, "repair_count": session.repair_count,
                "elapsed_s": round(now - session.started, 1) if now is not None else None,
                "staff": {"called": session.staff_called, "reason": session.staff_reason},
                "dine": session.dine, "queue": [it.item for it in session.queue],
            }
        out["served"] = self.served
        out["device_lost"] = self.device_lost
        return out

    # ======================================================================
    # 상태 진입
    # ======================================================================
    def _enter(self, state: S, now: float, **kw: Any) -> None:
        previous = self.state
        self.state = state
        self.state_t = now
        self.flags = {}
        self.awaiting_since = now if state in AWAITING else None
        if self.overlay in (Ov.CANDIDATES, Ov.DETAIL):
            self.overlay = Ov.NONE
        self.log.log("fsm.state", prev=str(previous), state=str(state),
                     session=self.session.id if self.session else None)
        getattr(self, f"_enter_{state.value.lower()}")(now, **kw)

    def _enter_idle(self, now: float, **kw: Any) -> None:
        self.session = None
        self.overlay = Ov.NONE
        self.lang = self.default_lang
        self.out.set_lang(self.lang)
        self._last_say = None
        # interrupt: 리셋·손님이 떠남·일시정지 만료 때 하던 말과 키프레임을 기다리던 화면을 끊는다
        # (안 끊으면 대기 화면이 뜬 뒤에 추천 화면이 덮어쓰고 로봇이 아무도 없는데 계속 말한다).
        # 상태 띠의 들은 말·로봇 말과 직원 호출 알림도 지운다: 다음 손님에게 이전 손님의 말이 보이지 않게
        self._act("idle", screen="attract", data={}, clear_overlay=True, posture="neutral", gaze="wander",
                  led="off", plan="waiting", interrupt=True, notice="none", reset_captions=True)
        self.out.set_idle(True)

    def _enter_greet(self, now: float, quiet: bool = False, **kw: Any) -> None:
        """손님이 오면 인사 모션(greeting: 고개 숙여 인사 + 안테나 손 흔들기)과 함께 인사말을 한다.
        환영 화면과 인사말은 모션의 bow 키프레임(고개를 숙이기 시작할 때)에 맞춰 나온다 (BehaviorManager).

        quiet: 손님이 먼저 말을 걸어서 들어온 경우. 그 말에 바로 답하므로 여기서는 말하지 않는다
        (그 말이 주문이 아니면 _say_greet가 인사 모션과 함께 greet_menu로 답한다)."""
        session = self.session or self._new_session(now)
        self.out.set_idle(False)
        self.invited = False
        if quiet:
            self._act("greet", screen="welcome", data={}, posture="perk_up", gaze="user", plan="greeting",
                      listen="open")
            return
        session.greeted = True
        self._act("greet", say="greet", slots=self._greet_slots(), screen="welcome", data={}, motion="greeting",
                  posture="perk_up", gaze="user", plan="greeting", listen="open")

    def _enter_menu(self, now: float, quiet: bool = False, say: str | None = None, resumed: bool = False,
                    **kw: Any) -> None:
        session = self._s()
        if say is None and not quiet:
            # "다른 메뉴를…"(menu_again)은 고르던 것을 그만뒀을 때만 부르는 쪽에서 직접 정한다.
            # 이어 가기·다시 묻기에서 쓰면 아무것도 안 고른 손님에게 "다른 메뉴"라고 하게 된다
            say = "menu_prompt"
        session.menu_prompted = True
        session.rec_pool = set()
        # 인사 없이 바로 메뉴로 온 경우(대기 화면 터치·언어 선택·손님이 먼저 인사)에도 인사 모션과 함께 인사한다.
        # 메뉴 화면은 터치에 바로 반응하도록 먼저 띄우고 인사말만 bow 키프레임을 기다린다.
        # (모션 중에는 시선 레이어가 멈춰 있어 화면 쪽 글랜스가 보이지 않으므로 이때는 글랜스를 뺀다)
        greet = say == "greet_menu" and not session.greeted
        if greet:
            session.greeted = True
        self._act("menu", say=say, screen="menu", data=self._menu_data(),
                  glance="screen" if say and not greet else None, motion="greeting" if greet else None,
                  screen_first=greet, gaze="user", posture="listening", plan="browsing", listen="open")

    def _enter_offer_help(self, now: float, seek: bool = False, **kw: Any) -> None:
        self._s().help_offered = True
        self._act("offer_help", say="offer_help_seek" if seek else "offer_help", screen="menu",
                  data=self._menu_data(highlight="recommend"), posture="tilt", gaze="user",
                  plan="seek_help" if seek else "offer_help", listen="yesno")

    def _enter_recommend(self, now: float, again: bool = False, resumed: bool = False, **kw: Any) -> None:
        if self._s().prefs.is_empty() and not self._s().rec_pool:
            self._act("ask_preference", say="ask_preference", screen="recommend", data=self._rec_data(None, []),
                      glance="screen", posture="listening", plan="ask_pref", listen="open")
        else:
            self._recommend(now, again=again, resumed=resumed)

    def _enter_sold_out(self, now: float, item: MenuItem | None = None, resumed: bool = False, **kw: Any) -> None:
        session = self._s()
        item = item or self.menu.get(session.sold_out_item)
        if item is None:
            self._enter(S.MENU, now)
            return
        alternatives = self.rec.alternatives(item, 2)
        session.sold_out_item = item.id
        session.alternatives = [alt.id for alt in alternatives]
        # 돌아온 손님에게는 사과를 처음부터 되풀이하지 않고 대안만 다시 말한다 (긴 품절 문장 약 8 s → 약 3 s)
        self._act("sold_out", motion=None if resumed else "failure", screen="sold_out",
                  data={"item": item.to_ui(), "alternatives": [alt.to_ui() for alt in alternatives]},
                  say="sold_out_resume" if resumed and session.alternatives else "sold_out",
                  slots={"item": item.id, "alts": session.alternatives}, plan="sold_out", listen="open")

    def _enter_options(self, now: float, reaction: str | None = None, **kw: Any) -> None:
        self._options_prompt(now, reaction=reaction)

    def _enter_cart(self, now: float, reaction: str | None = None, say: str | None = None, **kw: Any) -> None:
        session = self._s()
        if session.cart.is_empty():
            self._enter(S.MENU, now, say="cart_empty")
            return
        session.multi = False
        self._act("cart", motion=reaction, screen="cart", data=self._cart_data(), say=say or "confirm_cart",
                  slots=self._cart_slots(), glance="screen", plan="confirm", listen="yesno")

    def _enter_pay_wait_card(self, now: float, retry: bool = False, **kw: Any) -> None:
        if self.card_in and not retry:
            # 카드를 먼저 꽂아 둔 손님 (장바구니 단계에서 넣고 빼지 않은 채 "네", 또는 일시정지에서 돌아옴):
            # 카드를 넣으라고 하면 '넣음' 이벤트가 다시 오지 않아 결제가 끝나지 않는다 → 바로 결제
            self._enter(S.PAY_PROCESSING, now)
            return
        if retry:
            self._act("card_retry", motion="failure", screen="payment", data=self._pay_data("insert"),
                      say="card_reinsert", led="blink", plan="pay_retry")
        elif self.device_lost:
            # 카드 슬롯 에이전트가 끊겼다: 20초 동안 카드를 넣으라고 하지 않고 바로 화면 결제를 안내한다
            self._act("pay_request", screen="payment", data=self._pay_data("insert", fallback=True),
                      say="pay_slot_down", plan="pay_insert")
        else:
            self._act("pay_request", screen="payment", data=self._pay_data("insert"), led="blink",
                      glance="slot", glance_hold_s=1.5, say="pay_request", plan="pay_insert")

    def _enter_pay_processing(self, now: float, fallback: bool = False, **kw: Any) -> None:
        self._s().fallback_pay = fallback
        self._act("processing", screen="payment", data=self._pay_data("processing"), say="processing",
                  posture="thinking", led="on", plan="processing")

    def _enter_pay_remove_card(self, now: float, **kw: Any) -> None:
        self._act("remove_card", screen="payment", data=self._pay_data("remove"), say="remove_card",
                  glance="slot", glance_hold_s=1.2, led="blink", posture="neutral", plan="remove_card")

    def _enter_done(self, now: float, **kw: Any) -> None:
        session = self._s()
        number = self.counter.next()
        session.order_number = number
        self.served = True  # 이 손님이 그대로 서 있어도 대기 화면에서 다시 인사하지 않는다
        self.log.log("order.done", number=number, total=session.cart.total(self.menu),
                     lines=[line.to_dict() for line in session.cart.lines],
                     duration_s=round(now - session.started, 2), lang=session.lang, dine=session.dine)
        self._act("done", motion="success", screen="done",
                  data={"number": number, "cart": session.cart.to_ui(self.menu), "dine": session.dine},
                  say="done", slots={"number": number}, led="off", posture="neutral",
                  plan="done", plan_slots={"number": number})

    # ======================================================================
    # 이벤트 처리
    # ======================================================================
    def _on_human(self, state: str | None, now: float) -> None:
        if not state:
            return
        self.human = state
        self.human_t = now
        if state == ABSENT:
            self.served = False  # 주문을 마친 손님이 떠났다: 다음에 보이는 사람은 새 손님
        if state == UNCERTAIN:
            return  # 존재 판정 동결: 터치·음성으로 계속
        if self.state == S.IDLE:
            if state == ABSENT and self.invited:
                self.invited = False
                self._act("invite_end", gaze="wander", posture="neutral", clear_overlay=False)
                self.out.set_idle(True)
            self._idle_presence(now)
            return
        if self.overlay == Ov.PAUSED:
            if state in HERE | {AMBIGUOUS} and not self.flags.get("resume_asked"):
                self.flags["resume_asked"] = True
                self._act("resume_prompt", say="resume_prompt", overlay="paused",
                          overlay_data={"asked": True, "cart": self._s().cart.to_ui(self.menu)},
                          gaze="user", posture="perk_up", listen="yesno", clear_overlay=False)
            return
        if self.state == S.GREET:
            if state == ABSENT:
                self._enter(S.IDLE, now)
            elif state == AMBIGUOUS and self.overlay == Ov.NONE:
                self._set_overlay(Ov.ONE_AT_A_TIME, now)
            elif state == NO_RESPONSE:
                self._on_no_response(now)  # 인사 뒤 망설임 → 메뉴로 이끌기
            return
        if state == AMBIGUOUS:
            session = self.session
            if session is not None and now - session.busy_notice_t >= self.t["busy_notice_cooldown_s"]:
                session.busy_notice_t = now
                self._act("busy", say="busy_with_other", interrupt=False, clear_overlay=False)
            return
        if state == NO_RESPONSE:
            self._on_no_response(now)
            return
        if self.state == S.DONE and state == ABSENT:
            self._enter(S.IDLE, now)

    def _on_no_response(self, now: float) -> None:
        if self.overlay != Ov.NONE or self.session is None:
            return
        if self.session.staff_called:
            return  # 직원을 기다리는 손님은 재촉하지 않는다
        if self.state == S.GREET:
            self._lead_to_menu(now)  # 인사 뒤 망설임: 로봇이 먼저 메뉴로 이끈다
        elif self.state == S.MENU and not self.session.help_offered:
            self._enter(S.OFFER_HELP, now, seek=self.human_reason == "seek_help")
        elif self.state == S.OFFER_HELP:
            if self.human_reason == "seek_help":
                # "추천해 드릴까요?" 뒤에 말없이 로봇을 쳐다봄 = 도와 달라는 뜻. 예전에는 조용히 메뉴로 돌아갔다
                self._enter(S.RECOMMEND, now)
            else:
                self._enter(S.MENU, now, quiet=True)  # 무응답이면 재촉하지 않고 조용히 기다린다
        elif self.state in (S.RECOMMEND, S.OPTIONS, S.CART, S.SOLD_OUT) and not self.flags.get("reprompted"):
            self.flags["reprompted"] = True
            self._reprompt(now)

    def _on_touch(self, data: dict[str, Any], now: float) -> None:
        action = data.get("action")
        if not action:
            return
        if action == "lang":
            self.set_lang(str(data.get("lang", "ko")), now)
            return
        if self.state == S.IDLE:
            if action in ("start", "select_item", "recommend"):
                self.served = False
                self._new_session(now)
                self._enter(S.MENU, now, say="greet_menu")
            return
        if self.session is not None:
            self.session.repair_count = 0  # 화면을 쓰고 있으면 못 알아들은 말이 이어져도 직원까지 가지 않는다
        if action == "staff":
            self._call_staff("request", now)  # 화면의 직원 호출 버튼
            return

        # 오버레이가 떠 있으면 오버레이 버튼이 먼저
        if self.overlay == Ov.PAUSED:
            if action in ("resume_yes", "confirm"):
                self._resume(now)
            elif action in ("resume_no", "cancel"):
                self._cancel(now)
            return
        if self.overlay == Ov.ONE_AT_A_TIME:
            if action in ("start", "confirm"):
                self._clear_overlay()
                if self.on_lock_primary:
                    self.on_lock_primary()
                self._enter(S.MENU, now, say="menu_prompt")
            return
        if self.overlay == Ov.DETAIL:
            item = data.get("item") or self._s().explained_item
            self._clear_overlay()
            if action in ("confirm", "select_item") and item:
                self._order_item(Intent("order_item", item=str(item)), now)
            return
        if self.overlay == Ov.CANDIDATES:
            self._clear_overlay()
            if action in ("candidate", "select_item") and data.get("item"):
                self._order_item(Intent("order_item", item=str(data["item"])), now)
                return
            if action == "dismiss":
                return

        if action == "cancel" and self.state not in NO_CANCEL:
            self._cancel(now)
            return

        item_id = data.get("item")
        st = self.state
        if st == S.GREET:
            if action == "start":
                self._enter(S.MENU, now)
            elif action == "recommend":
                self._enter(S.RECOMMEND, now)
        elif st in (S.MENU, S.OFFER_HELP):
            if action == "select_item" and item_id:
                self._order_item(Intent("order_item", item=str(item_id)), now)
            elif action in ("recommend", "help") or (st == S.OFFER_HELP and action == "confirm"):
                self._enter(S.RECOMMEND, now)
            elif action == "dismiss" and st == S.OFFER_HELP:
                self._enter(S.MENU, now, say="menu_ok")
            elif action == "cart":
                self._enter(S.CART, now)
        elif st == S.RECOMMEND:
            session = self._s()
            if action == "pref" and data.get("key"):
                session.prefs.toggle(str(data["key"]))
                session.prefs.updated_by = None
                if session.prefs.is_empty():
                    self.flags.pop("rec_say_at", None)
                    session.rec_item = None
                    self._act("prefs", screen="recommend", data=self._rec_data(None, []), clear_overlay=False,
                              plan="ask_pref")
                else:
                    self._recommend(now, touch=True)  # 카드만 바꾸고 말은 손이 멈춘 뒤 한 번 (Aha! 반복 없음)
            elif action == "accept_rec" and session.rec_item:
                self._order_item(Intent("order_item", item=session.rec_item), now)
            elif action == "reject_rec" and session.rec_item:
                self._reject(None, now)
            elif action == "select_item" and item_id:
                self._order_item(Intent("order_item", item=str(item_id)), now)
            elif action == "back":
                self._enter(S.MENU, now, say="menu_again")
        elif st == S.SOLD_OUT:
            if action in ("accept_alt", "select_item") and item_id:
                self._order_item(Intent("order_item", item=str(item_id)), now)
            elif action == "back":
                self._enter(S.MENU, now, say="menu_again")
        elif st == S.OPTIONS:
            line = self._s().pending
            if line is None:
                self._enter(S.MENU, now)
            elif action == "set_option":
                key, value = data.get("key"), data.get("value")
                self._apply_options(line, value if key == "temp" else None, value if key == "size" else None)
                self._options_prompt(now)
            elif action == "options_done":
                self._fill_defaults(line)
                self._options_prompt(now)
            elif action == "back":
                self._s().pending = None
                self._enter(S.MENU, now, say="menu_again")
        elif st == S.CART:
            session = self._s()
            if action in ("confirm", "pay"):
                self._enter(S.PAY_WAIT_CARD, now)
            elif action == "add_more":
                self._enter(S.MENU, now, say="add_more")
            elif action == "remove" and data.get("index") is not None:
                index = int(data["index"])
                line = session.cart.lines[index] if 0 <= index < len(session.cart.lines) else None
                # ✕를 빠르게 두 번 누르면 두 번째 터치가 지운 줄 다음 줄을 가리킨다:
                # 화면이 보낸 줄 표식(메뉴/온도/사이즈/요청사항)이 지금 그 자리의 줄과 같을 때만 지운다
                sign = data.get("value")
                if line is not None and sign is not None and sign != line.signature():
                    self.log.log("touch.ignored", reason="stale_remove", index=index, line=sign)
                    return
                if session.cart.remove_index(index):
                    self.log.log("cart.remove", item=line.item_id if line else None, index=index)
                    self._enter(S.CART, now)
        elif st == S.PAY_WAIT_CARD:
            if action == "pay_fallback":
                self._enter(S.PAY_PROCESSING, now, fallback=True)
            elif action == "back":
                self._enter(S.CART, now)
        elif st == S.DONE:
            if action in ("start", "new_order"):
                lang = self.lang
                self._new_session(now)
                self._s().lang = lang
                self._enter(S.MENU, now, say="menu_prompt")

    def _on_utterance(self, text: str, now: float) -> None:
        if not text.strip():
            return
        if self.overlay in (Ov.PAUSED, Ov.ONE_AT_A_TIME):
            context = "paused"
        else:
            context = CONTEXT[self.state]
        intent = self.nlu.parse(text, context)
        self.log.log("nlu", text=text, context=context, state=str(self.state), **intent.summary())
        nobody = self.human not in HERE | {AMBIGUOUS, UNCERTAIN}
        if (intent.name in ("unknown", "staff", "out_of_scope", "note", "dine") and nobody
                and (self.state == S.IDLE or self.overlay == Ov.PAUSED)):
            # 아무도 안 보이는데 알아들을 수 없는 말 = 주변 잡담·잡음 (발표장 관객 등). 세션을 시작하거나
            # 일시정지를 풀지 않는다. 주문 말("라떼 주세요", "네")은 카메라가 놓친 손님일 수 있으니 받지만,
            # 잡담에 섞인 "직원"·"화장실" 같은 말로 직원을 부르지는 않는다
            self.log.log("speech.ignored", reason="nobody_present", text=text)
            return
        if self.state == S.IDLE:
            # 카메라가 놓쳤어도 알아들은 말을 걸면 손님이 있는 것. 그 말에 바로 답하므로 인사말은 따로 하지 않는다
            self.served = False
            self._enter(S.GREET, now, quiet=True)
        session = self.session
        if session is None:
            return
        if (not session.lang_locked and intent.name != "unknown" and intent.lang != session.lang
                and self.state in (S.GREET, S.MENU)):
            self.set_lang(intent.lang, now, locked=False)
        if intent.name != "unknown":
            session.repair_count = 0

        # 오버레이
        if self.overlay == Ov.PAUSED:
            if intent.name in ("deny", "cancel"):
                self._cancel(now)
                return
            self._resume(now)
            if intent.name in ("confirm", "unknown", "hello"):
                return  # "네·이어서 할게요" 또는 못 알아들은 말: 이어 가기만 하고 되묻지 않는다
            # 내용이 있는 말("아이스로요" 등)은 이어 간 상태의 문맥으로 다시 해석해 처리한다
            intent = self.nlu.parse(text, CONTEXT[self.state])
        elif self.overlay == Ov.ONE_AT_A_TIME:
            self._clear_overlay()
            if self.on_lock_primary:
                self.on_lock_primary()
            if intent.name in ("confirm", "unknown", "deny"):
                self._enter(S.MENU, now, say="menu_prompt")
                return
        elif self.overlay == Ov.CANDIDATES:
            candidates = list(session.candidates)
            self._clear_overlay()
            pick = None
            if intent.name == "choose":
                pick = self._pick(candidates, intent.ordinal)  # "첫 번째 걸로"
            elif intent.name == "confirm" and len(candidates) == 1:
                pick = candidates[0]  # 후보가 하나뿐일 때 "네·그거요"
            if pick is not None:
                self._order_item(Intent("order_item", item=pick, notes=intent.notes), now)
                return
        elif self.overlay == Ov.DETAIL:
            explained = session.explained_item
            self._clear_overlay()
            if intent.name == "confirm" and explained:
                self._order_item(Intent("order_item", item=explained), now)
                return
            if intent.name == "deny":
                if self.state == S.MENU:  # 설명한 메뉴를 거절: 이때는 "다른 메뉴를…"이 맞다
                    self._act("reprompt", say="menu_again", listen="open")
                    self.awaiting_since = now
                else:
                    self._reprompt(now)
                return

        # 어느 상태에서나
        # 규칙 밖의 말: 키오스크가 못 하는 일·직원을 찾는 말은 솔직하게 말하고 직원을 부른다 (주문 상태는 그대로)
        if intent.name in ("staff", "out_of_scope"):
            self._call_staff(intent.reason if intent.name == "out_of_scope" else "request", now)
            return
        if intent.dine:
            session.dine = intent.dine  # "아이스 라떼 포장이요": 포장은 조용히 기억하고 확인 문장에 넣는다
        if intent.name in ("note", "dine"):
            self._note_or_dine(intent, now)
            return
        if intent.notes and intent.name != "order_item":
            self._apply_notes(intent.notes)  # "아이스로 얼음 적게": 옵션과 함께 말한 요청사항
        if (intent.name == "reject_feedback" and intent.prefs
                and not (self.state == S.RECOMMEND and session.rec_item)):
            # "너무 달지 않은 거", "I don't like sweet drinks": 아직 추천한 메뉴가 없으면 거절이 아니라 취향이다
            intent = Intent("preference", text=intent.text, lang=intent.lang, prefs=intent.prefs)
        if intent.name == "explain_item" and intent.item and self.state in AWAITING:
            self._explain(intent.item, now)
            return
        if intent.name == "compare" and self.state in AWAITING:
            # "딸기랑 바닐라 중에 뭐가 나아요?" → 그 둘 중에서 취향(없으면 인기)으로 하나를 추천
            session.rec_pool = {c for c in intent.candidates if self.menu.get(c) is not None}
            if intent.prefs:
                session.prefs.merge(intent.prefs)
                session.prefs.updated_by = None
            self.log.log("rec.compare", items=sorted(session.rec_pool))
            if self.state == S.RECOMMEND:
                self._recommend(now)
            else:
                self._enter(S.RECOMMEND, now)
            return
        if intent.name == "cancel":
            if self.state not in NO_CANCEL:
                self._cancel(now)
            elif self.state == S.PAY_REMOVE_CARD:
                self._reprompt(now)  # 결제는 이미 끝났다: 취소 대신 카드를 빼 달라고 다시 안내
            return
        if intent.name == "repeat":
            self._repeat_last(now)  # 직전 말을 그대로 (예전에는 상태의 다음 질문 "다른 메뉴를…"을 했다)
            return
        if intent.name == "hello" and self.state != S.GREET:
            self._reprompt(now)
            return
        if intent.name == "reject_item" and self.state in AWAITING:
            self._reject_item(intent, now)
            return
        if intent.rejected_items:
            session.rejected.update(intent.rejected_items)  # "아메리카노 말고 라떼": 추천에서도 아메리카노는 뺀다
        handler = getattr(self, f"_say_{self.state.value.lower()}", None)
        if handler is not None:
            handler(intent, now)

    def _on_card(self, state: str | None, now: float) -> None:
        inserted = state == "in"
        self.card_in = inserted
        if inserted:
            if self.overlay == Ov.PAUSED:
                # 카드를 넣었다 = 손님이 와 있다 (지갑을 찾느라 카메라에서 잠깐 사라졌던 경우).
                # 결제 차례면 일시정지를 풀고 바로 결제, 아니면 주문을 이어 간다
                if self.state == S.PAY_WAIT_CARD:
                    self.overlay = Ov.NONE
                else:
                    self._resume(now)
            if self.state == S.PAY_WAIT_CARD:
                self._enter(S.PAY_PROCESSING, now)
            elif self.state in (S.PAY_PROCESSING, S.PAY_REMOVE_CARD, S.IDLE):
                return
            elif not self.flags.get("card_warned"):
                self.flags["card_warned"] = True
                self._act("not_payment_step", say="not_payment_step", glance="slot", interrupt=False,
                          clear_overlay=False)
            return
        if self.state == S.PAY_REMOVE_CARD:
            self._enter(S.DONE, now)
        elif self.state == S.PAY_PROCESSING:
            if self._s().fallback_pay or now - self.state_t < 0.3:
                return  # 바운스면 tick에서 카드 상태를 다시 본다
            self._enter(S.PAY_WAIT_CARD, now, retry=True)

    # ======================================================================
    # 상태별 발화 처리
    # ======================================================================
    def _say_greet(self, it: Intent, now: float) -> None:
        if it.name == "order_item":
            self._order_item(it, now)
        elif it.name in ("ask_recommend", "preference", "help"):
            self._start_recommend(it, now)
        elif it.name == "deny":
            self._act("greet_later", say="greet_again", listen="open")
        else:  # 네·안녕하세요·못 알아들은 말: 먼저 말을 건 손님에게는 아직 인사 전이므로 인사 모션과 함께 답한다
            self._enter(S.MENU, now, say=None if self._s().greeted else "greet_menu")

    def _say_menu(self, it: Intent, now: float) -> None:
        if it.name == "order_item":
            self._order_item(it, now)
        elif it.name in ("ask_recommend", "preference", "help", "reject_feedback"):
            self._start_recommend(it, now)
        elif it.name == "checkout":
            self._enter(S.CART, now)
        elif it.name == "remove_item":
            self._remove_item(it, now)
        elif it.name == "unknown":
            self._repair(it, now)
        else:
            self._reprompt(now)

    def _say_offer_help(self, it: Intent, now: float) -> None:
        if it.name in ("confirm", "preference", "ask_recommend", "help"):
            self._start_recommend(it, now)
        elif it.name == "deny":
            self._enter(S.MENU, now, say="menu_ok")
        elif it.name == "order_item":
            self._order_item(it, now)
        elif it.name == "unknown":
            self._repair(it, now)
        else:
            self._enter(S.MENU, now, quiet=True)

    def _say_recommend(self, it: Intent, now: float) -> None:
        session = self._s()
        if it.name == "preference" or (it.name == "ask_recommend" and it.prefs):
            session.prefs.merge(it.prefs)
            session.prefs.updated_by = None
            self._recommend(now)
        elif it.name == "ask_recommend":
            if session.rec_item:
                self._reject(None, now)
            else:
                self._reprompt(now)
        elif it.name == "confirm":
            if session.rec_item:
                self._order_item(Intent("order_item", item=session.rec_item), now)
            else:
                self._reprompt(now)
        elif it.name in ("reject_feedback", "deny"):
            if session.rec_item:
                self._reject(it.reason, now)
            else:
                self._enter(S.MENU, now, say="menu_ok")
        elif it.name == "order_item":
            self._order_item(it, now)
        elif it.name == "choose" and session.rec_item and it.ordinal in (1, -1):
            self._order_item(Intent("order_item", item=session.rec_item, notes=it.notes), now)  # 화면의 추천 하나
        elif it.name == "unknown":
            self._repair(it, now)
        else:
            self._reprompt(now)

    def _say_sold_out(self, it: Intent, now: float) -> None:
        session = self._s()
        if it.name == "order_item":
            self._order_item(it, now)
        elif it.name == "choose" and self._pick(session.alternatives, it.ordinal):
            self._order_item(Intent("order_item", item=self._pick(session.alternatives, it.ordinal),
                                    notes=it.notes), now)  # "두 번째 걸로"
        elif it.name == "confirm" and session.alternatives:
            # "네·그거 주세요": 로봇이 먼저 말한 대안 (화면 왼쪽)
            self._order_item(Intent("order_item", item=session.alternatives[0], notes=it.notes), now)
        elif it.name in ("deny", "reject_feedback"):
            self._enter(S.MENU, now, say="menu_again")
        elif it.name in ("ask_recommend", "preference"):
            self._start_recommend(it, now)
        elif it.name == "unknown":
            self._repair(it, now)
        else:
            self._reprompt(now)

    def _say_options(self, it: Intent, now: float) -> None:
        session = self._s()
        line = session.pending
        if line is None:
            self._enter(S.MENU, now)
            return
        if it.name == "set_option" or (it.name == "order_item" and it.item == line.item_id):
            self._apply_options(line, it.temp, it.size, it.qty if it.name == "order_item" else None)
            self._options_prompt(now)
        elif it.name == "order_item":
            self._order_item(it, now)  # 다른 메뉴로 바꿈
        elif it.name in ("confirm", "checkout") and line.missing() == ["size"]:
            line.size = "regular"
            self._options_prompt(now)
        elif it.name == "unknown":
            self._repair(it, now)
        else:
            self._reprompt(now)

    def _say_cart(self, it: Intent, now: float) -> None:
        session = self._s()
        removing = self.flags.pop("remove_item", None)  # "어떤 라떼를 뺄까요?"에 대한 대답
        if removing and it.name in ("set_option", "order_item") and self._remove_matching(
                Intent("remove_item", item=it.item or removing, temp=it.temp, size=it.size), now):
            return
        if it.name in ("confirm", "checkout"):
            self._enter(S.PAY_WAIT_CARD, now)
        elif it.name == "deny":
            self._act("ask_change", say="ask_change", listen="open")
        elif it.name == "order_item":
            if not self._cart_edit(it, now):
                self._order_item(it, now)
        elif it.name == "remove_item":
            self._remove_item(it, now)
        elif it.name == "set_option" and session.cart.lines:
            self._apply_options(session.cart.lines[-1], it.temp, it.size)
            self._enter(S.CART, now)
        elif it.name == "add_more":
            self._enter(S.MENU, now, say="add_more")
        elif it.name == "unknown":
            self._repair(it, now)
        else:
            self._reprompt(now)

    def _say_pay_wait_card(self, it: Intent, now: float) -> None:
        if it.name in ("help", "unknown", "confirm", "checkout"):
            self._reprompt(now)

    def _say_pay_remove_card(self, it: Intent, now: float) -> None:
        if it.name in ("help", "unknown", "confirm"):
            self._reprompt(now)

    def _say_done(self, it: Intent, now: float) -> None:
        if it.name not in ("order_item", "add_more", "ask_recommend", "preference"):
            return
        lang = self.lang
        self._new_session(now)
        self._s().lang = lang
        if it.name == "add_more":
            self._enter(S.MENU, now, say="menu_prompt")
            return
        self._enter(S.MENU, now, quiet=True)
        self._say_menu(it, now)

    # ======================================================================
    # 도우미
    # ======================================================================
    def _s(self) -> Session:
        if self.session is None:
            self._new_session(self.state_t)
        assert self.session is not None
        return self.session

    def _new_session(self, now: float) -> Session:
        self._session_seq += 1
        self.session = Session(id=self._session_seq, lang=self.lang, started=now)
        self.log.log("session.start", session=self.session.id, lang=self.lang)
        # 어떻게 시작했든(얼굴·터치·언어 버튼·말) 주문 중에는 지루함 모션을 끈다.
        # 예전에는 인사(GREET)로 들어올 때만 꺼서, 화면부터 누른 손님의 주문 중에 로봇이 두리번거렸다
        self.out.set_idle(False)
        return self.session

    def _act(self, name: str, **kw: Any) -> None:
        """액트를 내보낸다. 모든 액트에 '로봇의 계획'(상태 스트립·청중 화면)을 붙인다.

        계획을 정하지 않은 액트(되묻기·재질문·창 닫기 등)에는 지금 상태·오버레이의 계획을 붙여, 앞 액트의 계획이
        남지 않게 한다 (예: 메뉴 설명 창을 닫았는데 "메뉴 설명: 아인슈페너"가 남던 문제).
        오버레이 없이 정한 계획은 그 상태의 기본 계획으로 기억한다. MOMENTARY_PLANS는 한 번만 보이고 기본 계획으로 돌아간다.
        말이 있는 액트는 "다시 말해 주세요"에 그대로 반복할 수 있게 기억한다.
        """
        kw.setdefault("clear_overlay", self.overlay == Ov.NONE)
        if kw.get("plan") is None:
            kw["plan"], kw["plan_slots"] = self._context_plan()
        elif self.overlay == Ov.NONE and kw["plan"] not in MOMENTARY_PLANS:
            self._base_plan = (kw["plan"], dict(kw.get("plan_slots") or {}))
        if kw.get("say") and not str(kw["say"]).startswith("repair"):
            # 되묻기("다시 말씀해 주시겠어요?")는 기억하지 않는다: "뭐라고요?"에는 그 앞의 실제 질문을 반복해야 한다
            self._last_say = {"say": REPEAT_AS.get(kw["say"], kw["say"]), "slots": dict(kw.get("slots") or {}),
                              "listen": kw.get("listen"), "lang": kw.get("lang") or self.lang}
        self.out.perform(Act(name, **kw))

    def _context_plan(self) -> tuple[str, dict[str, Any]]:
        """지금 상태·오버레이에 맞는 계획 (계획 키, 슬롯)."""
        if self.overlay == Ov.DETAIL and self.session is not None and self.session.explained_item:
            return "explain", {"item": self.session.explained_item}
        if self.overlay == Ov.CANDIDATES:
            return "repair_candidates", {}
        if self.overlay == Ov.PAUSED:
            return "paused", {}
        if self.overlay == Ov.ONE_AT_A_TIME:
            return "one_at_a_time", {}
        return self._base_plan[0], dict(self._base_plan[1])

    def _on_cues(self, cues: dict[str, Any], now: float) -> None:
        """거리 구역·접근 여부·주의 방향이 바뀜. IDLE에서 다가오면 바로 인사한다."""
        self.cues.update({k: cues[k] for k in ("zone", "approaching", "attention") if k in cues})
        if cues.get("approaching"):
            self.served = False  # 새로 다가오는 사람 = 다음 손님 (주문을 마친 손님은 멀어진다)
        if self.state == S.IDLE and self.human in HERE:
            self._idle_presence(now)

    def _idle_presence(self, now: float) -> None:
        """IDLE에서 사람이 보일 때: 가까이 오거나 다가오면 인사, 멀리 있거나 지나가면 말없이 눈 맞추고 부르기.

        방금 주문을 마친 손님(served)이 그대로 서 있으면 아무것도 하지 않는다 (예전에는 14초 뒤 다시 고개 숙여
        인사했고, 평가 시행 사이에도 생겼다). 그 손님이 떠나거나 새 사람이 다가오거나 터치·말이 오면 다시 반응한다."""
        if self.served:
            return
        if self.human == AMBIGUOUS:
            self._enter(S.GREET, now)
            self._set_overlay(Ov.ONE_AT_A_TIME, now)
            return
        if self.human not in HERE:
            return
        near = (self.cues.get("zone") == "personal" or bool(self.cues.get("approaching"))
                or self.human in (ENGAGED, NO_RESPONSE))
        if near or not self.features.get("invite"):
            self._enter(S.GREET, now)
            return
        if now - self.invite_t >= self.t.get("invite_cooldown_s", 12.0):
            self.invite_t = now
            self.invited = True
            self.out.set_idle(False)
            self.log.log("invite", zone=self.cues.get("zone"))
            self._act("invite", gaze="user", gesture="antenna_wiggle", posture="perk_up", plan="invite",
                      clear_overlay=False)

    def _greet_slots(self) -> dict[str, Any]:
        if not self.features.get("context_talk"):
            return {}
        weather = self.context.get("weather")
        hour = time.localtime().tm_hour
        return {"weather": None if weather == "normal" else weather,
                "daypart": "morning" if 5 <= hour < 11 else None}

    def _explain(self, item_id: str, now: float) -> None:
        """"아인슈페너가 뭐예요?" → 화면에 메뉴 카드를 띄우고, 로봇이 그쪽을 보며 설명한다."""
        session = self._s()
        item = self.menu.get(item_id)
        if item is None:
            return
        session.explained_item = item.id
        self.overlay = Ov.DETAIL
        self.overlay_t = now
        self._act("explain", say="explain_item", slots={"item": item.id}, overlay="detail",
                  overlay_data={"item": item.to_ui()}, glance="screen", plan="explain",
                  plan_slots={"item": item.id}, listen="yesno", clear_overlay=False)
        self.awaiting_since = now

    def _order_item(self, it: Intent, now: float) -> None:
        session = self._s()
        if len(it.items) > 1:
            # 여러 메뉴를 한 번에 ("라떼 하나랑 아이스 아메리카노 두 잔"): 첫 메뉴부터 하나씩 담고,
            # 옵션이 빠진 메뉴만 이름을 붙여 차례로 묻는다 (예전에는 첫 메뉴만 받았다)
            entries = [Intent("order_item", text=it.text, lang=it.lang, item=e["item"], qty=e["qty"], temp=e["temp"],
                              size=e["size"], notes=list(e["notes"])) for e in it.items]
            session.multi = True
            session.queue = entries[1:]
            self.log.log("order.multi", items=[e.item for e in entries])
            it = entries[0]
        item = self.menu.get(it.item)
        if item is None:
            return
        session.rec_pool = set()
        if not item.in_stock:
            self._enter(S.SOLD_OUT, now, item=item)
            return
        reaction = "success" if self.state == S.SOLD_OUT and item.id in session.alternatives else None
        line = CartLine(item.id, qty=it.qty or 1)
        self._apply_options(line, it.temp, it.size)
        if line.temp is None and session.prefs.temp in item.temps:
            line.temp = session.prefs.temp  # "따뜻한 거"라고 이미 말했으면 옵션에서 온도를 다시 묻지 않는다
        line.notes = list(dict.fromkeys(session.pending_notes + list(it.notes)))
        session.pending_notes = []
        line.autofill(self.menu)
        session.pending = line
        self.log.log("order.item", from_state=str(self.state), **line.to_dict())
        if line.missing():
            self._enter(S.OPTIONS, now, reaction=reaction)
        else:
            self._commit_pending(now, reaction=reaction)

    def _apply_options(self, line: CartLine, temp: str | None, size: str | None, qty: int | None = None) -> None:
        item = self.menu.get(line.item_id)
        if item is None:
            return
        if temp in item.temps:
            line.temp = temp
        if size in item.sizes:
            line.size = size
        if qty:
            line.qty = qty

    def _fill_defaults(self, line: CartLine) -> None:
        item = self.menu.get(line.item_id)
        if item is None:
            return
        if line.temp is None:
            line.temp = item.temps[0]
        if line.size is None:
            line.size = "regular" if "regular" in item.sizes else item.sizes[0]

    def _options_prompt(self, now: float, reaction: str | None = None, reprompt: bool = False,
                        queued: bool = False) -> None:
        session = self._s()
        line = session.pending
        if line is None:
            self._enter(S.MENU, now)
            return
        missing = line.missing()
        if not missing:
            self._commit_pending(now, reaction=reaction)
            return
        asking = missing[0]
        # 여러 메뉴를 주문하는 중이면 어느 잔을 묻는지 이름을 붙인다 ("카페라떼는 따뜻하게 드릴까요…")
        named = session.multi
        say = ("ask_temp" if asking == "temp" else "ask_size") + ("_item" if named else "")
        extra: dict[str, Any] = {"interrupt": False} if queued else {}
        self._act("options", motion=reaction, screen="options", data=self._options_data(line, asking),
                  say=say, slots={"item": line.item_id} if named else {},
                  glance=None if reprompt else "screen", plan=f"options_{asking}", listen="open", **extra)
        self.awaiting_since = now

    def _commit_pending(self, now: float, reaction: str | None = None) -> None:
        session = self._s()
        if session.pending is not None:
            session.cart.add(session.pending)
            self.log.log("cart.add", **session.pending.to_dict())
            session.pending = None
        if session.queue:
            # 한 번에 말한 다음 메뉴: 옵션이 다 있으면 바로 담기고, 빠졌으면 그 메뉴의 옵션을 묻는다
            self._order_item(session.queue.pop(0), now)
            return
        self._enter(S.CART, now, reaction=reaction)

    def _remove_item(self, it: Intent, now: float) -> None:
        if not self._remove_matching(it, now):
            self._reprompt(now)

    def _remove_matching(self, it: Intent, now: float) -> bool:
        """이름으로 빼기. 같은 메뉴가 여러 줄이면 온도·사이즈로 고르고, 그래도 여럿이면 어느 것인지 묻는다
        (예전에는 같은 메뉴가 전부 빠졌다). "아메리카노 하나 빼 주세요"는 두 잔 중 한 잔만 뺀다."""
        session = self._s()
        lines = [line for line in session.cart.lines if line.item_id == it.item]
        if it.temp:
            lines = [line for line in lines if line.temp == it.temp] or lines
        if it.size:
            lines = [line for line in lines if line.size == it.size] or lines
        if not lines:
            return False
        if len(lines) > 1:
            self.flags["remove_item"] = it.item
            self._act("remove_which", say="remove_which", slots={"item": it.item}, listen="open", clear_overlay=False)
            return True
        line = lines[0]
        if it.qty and line.qty > it.qty:
            line.qty -= it.qty
        else:
            session.cart.lines.remove(line)
        self.log.log("cart.remove", item=line.item_id, qty=it.qty, left=session.cart.count())
        if session.cart.is_empty():
            self._enter(S.MENU, now, say="cart_empty")
        else:
            self._enter(S.CART, now)
        return True

    def _cart_edit(self, it: Intent, now: float) -> bool:
        """장바구니 단계에서 메뉴 이름으로 고치기. 고쳤으면 True, 새로 담을 주문이면 False.

        - "라떼 하나 더": 그 줄의 수량 +1
        - "라떼는 아이스로 바꿔 주세요"·"라떼 아이스로요": 그 줄의 옵션을 바꾼다 (예전에는 같은 메뉴가 한 줄 더 생겼다)
        - "라떼 말고 아메리카노로"·(한 줄뿐일 때) "아메리카노로 바꿔 주세요": 그 줄의 메뉴를 바꾼다
        """
        session = self._s()
        cart = session.cart
        lines = [line for line in cart.lines if line.item_id == it.item]
        if lines and it.more:
            target = self._closest(lines, it)
            target.qty += it.qty or 1
            self.log.log("cart.more", item=target.item_id, qty=target.qty)
            self._enter(S.CART, now)
            return True
        if lines and (it.change or (it.notes and not it.qty and not it.temp and not it.size)):
            # "아메리카노는 얼음 적게 해 주세요"처럼 요청사항만 붙이는 말도 그 줄을 고치는 것이다
            target = lines[-1]
            self._apply_options(target, it.temp, it.size, it.qty)
            target.notes = list(dict.fromkeys(target.notes + list(it.notes)))
            cart.merge()
            self.log.log("cart.change", **target.to_dict())
            self._enter(S.CART, now)
            return True
        replaced = [line for line in cart.lines if line.item_id in it.rejected_items]
        if not lines and (replaced or (it.change and len(cart.lines) == 1)):
            old = (replaced or cart.lines)[0]
            item = self.menu.get(it.item)
            if item is None or not item.in_stock:
                return False
            new = CartLine(item.id, qty=it.qty or old.qty, notes=list(dict.fromkeys(old.notes + list(it.notes))))
            self._apply_options(new, it.temp or old.temp, it.size or old.size)
            new.autofill(self.menu)
            cart.lines.remove(old)
            self.log.log("cart.replace", old=old.item_id, new=new.item_id)
            session.pending = new
            if new.missing():
                self._enter(S.OPTIONS, now)
            else:
                self._commit_pending(now)
            return True
        return False

    @staticmethod
    def _closest(lines: list[CartLine], it: Intent) -> CartLine:
        for line in reversed(lines):
            if (it.temp in (None, line.temp)) and (it.size in (None, line.size)):
                return line
        return lines[-1]

    @staticmethod
    def _pick(options: list[str], ordinal: int | None) -> str | None:
        """화면 목록에서 몇 번째 (1부터, -1은 마지막)."""
        if not options or ordinal is None:
            return None
        if ordinal == -1:
            return options[-1]
        return options[ordinal - 1] if 1 <= ordinal <= len(options) else None

    def _start_recommend(self, it: Intent, now: float) -> None:
        session = self._s()
        if it.prefs:
            session.prefs.merge(it.prefs)
            session.prefs.updated_by = None
        self._enter(S.RECOMMEND, now)

    def _recommend(self, now: float, again: bool = False, resumed: bool = False, touch: bool = False) -> None:
        """touch: 화면의 취향 칩. 카드만 바로 바꾸고 말은 손이 멈춘 뒤 한 번만 한다 (tick → _speak_recommendation).
        Aha! 모션은 첫 추천과 말로 바꾼 조건·거절 뒤의 재추천에만 쓴다 (칩마다 3.5초씩 반복되던 문제)."""
        session = self._s()
        weather = self.context.get("weather") if self.features.get("context_talk") else None
        items = self.rec.recommend(session.prefs, session.rejected, weather=weather, only=session.rec_pool or None)
        if touch:
            item = items[0] if items else None
            session.rec_item = item.id if item else None
            reasons = self.rec.reasons(item, session.prefs) if item else []
            self.log.log("rec.suggest", item=session.rec_item, reasons=reasons, prefs=session.prefs.to_ui(), touch=True)
            self._act("recommend_card", screen="recommend", data=self._rec_data(item, reasons), clear_overlay=False,
                      plan="recommend" if item else "ask_pref", plan_slots={"item": item.id} if item else {},
                      listen="yesno")
            self.flags["rec_say_at"] = now + float(self.t.get("chip_settle_s", 1.5))
            self.awaiting_since = now
            return
        if not items:
            session.rec_item = None
            self._enter(S.MENU, now, say="no_recommendation")
            return
        item = items[0]
        session.rec_item = item.id
        session.rec_spoken = item.id
        reasons = self.rec.reasons(item, session.prefs, focus=session.prefs.updated_by)
        self.log.log("rec.suggest", item=item.id, reasons=reasons, prefs=session.prefs.to_ui(), again=again)
        self._act("recommend", motion=None if resumed else "aha_recommend", screen="recommend",
                  data=self._rec_data(item, reasons), say="recommend_again" if again else "recommend",
                  slots={"item": item.id, "reasons": reasons}, plan="recommend", plan_slots={"item": item.id},
                  listen="yesno")
        self.awaiting_since = now

    def _reject(self, reason: str | None, now: float) -> None:
        session = self._s()
        item = self.menu.get(session.rec_item)
        if item is None:
            return
        session.rejected.add(item.id)
        session.rec_rejections += 1
        self.rec.on_reject(session.prefs, item, reason)
        self.log.log("rec.reject", item=item.id, reason=reason, prefs=session.prefs.to_ui())
        if session.rec_rejections > 2:
            session.rec_item = None
            self._enter(S.MENU, now, say="rec_giveup")
            return
        self._recommend(now, again=True)

    def _speak_recommendation(self) -> None:
        """칩을 다 누른 뒤 지금 카드의 메뉴를 한 번 말한다. 이미 말한 메뉴면 다시 말하지 않는다."""
        session = self._s()
        if session.rec_item is None:
            self._act("no_match", say="no_match_chips", listen="open", clear_overlay=False)
            return
        if session.rec_item == session.rec_spoken:
            return
        item = self.menu.get(session.rec_item)
        if item is None:
            return
        session.rec_spoken = item.id
        reasons = self.rec.reasons(item, session.prefs)
        self._act("recommend_say", say="recommend", slots={"item": item.id, "reasons": reasons}, listen="yesno",
                  clear_overlay=False, plan="recommend", plan_slots={"item": item.id})

    def _lead_to_menu(self, now: float) -> None:
        """인사 뒤 망설이는 손님: 로봇이 화면을 보며 "메뉴를 보여 드릴게요"라고 메뉴로 이끈다."""
        self.log.log("lead_to_menu", state=str(self.state))
        self._enter(S.MENU, now, say="menu_lead")

    def _call_staff(self, reason: str, now: float) -> None:
        """직원 호출 (규칙 밖의 상황). 주문 상태는 그대로 두고, 화면 알림과 오퍼레이터 패널 알림을 띄운다.
        직원(오퍼레이터)이 응대를 마치면 staff_done. 기다리는 동안에도 화면·음성으로 주문을 계속할 수 있다."""
        session = self._s()
        again = session.staff_called
        session.staff_called = True
        session.staff_reason = reason
        session.repair_count = 0
        self.log.log("staff.call", reason=reason, state=str(self.state), session=session.id, again=again)
        self._act("staff", say="staff_already" if again else STAFF_SAY.get(reason, "staff_request"),
                  notice="staff", notice_data={"reason": reason}, gaze="user", posture="neutral", plan="staff",
                  clear_overlay=False, listen="open")
        if self.state in AWAITING:
            self.awaiting_since = now

    def _apply_notes(self, notes: list[str]) -> None:
        """요청사항을 지금 고르는 잔(옵션 단계) → 장바구니 확인 중이면 마지막 잔 → 아니면 다음에 고를 잔에 붙인다."""
        session = self._s()
        target = session.pending
        if target is None and self.state == S.CART and session.cart.lines:
            target = session.cart.lines[-1]
        if target is None:
            session.pending_notes = list(dict.fromkeys(session.pending_notes + list(notes)))
            return
        target.notes = list(dict.fromkeys(target.notes + list(notes)))
        if target is not session.pending:
            session.cart.merge()
        self.log.log("order.note", item=target.item_id, notes=target.notes)

    def _note_or_dine(self, it: Intent, now: float) -> None:
        """"얼음 적게 해 주세요"·"포장이요"만 말했을 때: 받아 적었다고 말하고, 하던 질문으로 돌아간다."""
        if it.name == "note":
            self._apply_notes(it.notes)
            self._act("note", say="note_ok", slots={"notes": list(it.notes)}, clear_overlay=False)
        else:
            self.log.log("order.dine", dine=it.dine)
            self._act("dine", say="dine_togo" if it.dine == "togo" else "dine_here", clear_overlay=False)
        if self.state == S.GREET:
            self._enter(S.MENU, now, quiet=True)
            self._act("menu_prompt", say="menu_prompt", listen="open", interrupt=False)
        elif self.state == S.CART:
            self._act("cart_refresh", screen="cart", data=self._cart_data(), clear_overlay=False)
            self._reprompt(now, queued=True)  # 확인 문장에 요청사항·포장이 들어가도록 다시 확인
        elif self.state in (S.OPTIONS, S.RECOMMEND, S.SOLD_OUT, S.OFFER_HELP):
            self._reprompt(now, queued=True)

    def _repeat_last(self, now: float) -> None:
        last = self._last_say
        if last is None:
            self._reprompt(now)
            return
        self._act("repeat", say=last["say"], slots=dict(last["slots"]), listen=last.get("listen"), clear_overlay=False)
        if self.state in AWAITING:
            self.awaiting_since = now

    def _reject_item(self, it: Intent, now: float) -> None:
        """"아메리카노 말고요"·"라떼는 싫어요": 그 메뉴를 버린다 (추천에서도 뺀다)."""
        session = self._s()
        item_id = str(it.item)
        session.rejected.add(item_id)
        self.log.log("order.reject_item", item=item_id, state=str(self.state))
        st = self.state
        if st == S.RECOMMEND and session.rec_item == item_id:
            self._reject(None, now)
        elif st == S.OPTIONS and session.pending is not None and session.pending.item_id == item_id:
            session.pending = None
            self._enter(S.MENU, now, say="menu_again")
        elif st == S.CART and any(line.item_id == item_id for line in session.cart.lines):
            self._act("ask_change", say="ask_change", listen="open")
        elif st in (S.GREET, S.MENU, S.OFFER_HELP, S.SOLD_OUT):
            self._enter(S.MENU, now, say="menu_again")
        else:
            self._reprompt(now)

    def _repair(self, it: Intent, now: float) -> None:
        session = self._s()
        session.repair_count += 1
        if (session.repair_count >= STAFF_AFTER_REPAIRS and not session.staff_called
                and self.features.get("auto_staff", True)):
            # 되묻기 → 화면 권유에도 계속 못 알아들으면 사람에게 넘긴다 (같은 말을 무한히 되풀이하지 않는다)
            self._call_staff("repair", now)
            return
        candidates = [c for c in it.candidates if self.menu.get(c) is not None]
        session.candidates = candidates
        if session.repair_count >= 2:
            key = "repair_touch"
        else:
            key = "repair_candidates" if candidates else "repair"
        if candidates:
            self.overlay = Ov.CANDIDATES
            self.overlay_t = now
            self._act("repair", say=key, overlay="candidates",
                      overlay_data={"items": [self.menu.get(c).to_ui() for c in candidates]},  # type: ignore[union-attr]
                      posture="tilt", listen="open", clear_overlay=False, plan=key)
        else:
            self._act("repair", say=key, posture="tilt", listen="open", clear_overlay=False, plan=key)
        self.awaiting_since = now

    def _reprompt(self, now: float, queued: bool = False) -> None:
        """현재 상태의 질문을 다시 한다 (무응답·인사·언어 변경). queued: 지금 하는 말이 끝난 뒤에 (끊지 않음)."""
        session = self.session
        st = self.state
        extra: dict[str, Any] = {"interrupt": False} if queued else {}
        if st == S.GREET:
            self._act("reprompt", say="greet_again", listen="open", **extra)
        elif st == S.MENU:
            self._act("reprompt", say="menu_prompt", listen="open", **extra)
        elif st == S.OFFER_HELP:
            self._act("reprompt", say="offer_help", listen="yesno", **extra)
        elif st == S.RECOMMEND:
            key = "recommend_reprompt" if session and session.rec_item else "ask_preference"
            self._act("reprompt", say=key, listen="yesno", **extra)
        elif st == S.SOLD_OUT:
            self._act("reprompt", say="sold_out_reprompt", listen="open", **extra)
        elif st == S.OPTIONS:
            self._options_prompt(now, reprompt=True, queued=queued)
            return
        elif st == S.CART:
            self._act("reprompt", say="confirm_cart", slots=self._cart_slots(), listen="yesno", **extra)
        elif st == S.PAY_WAIT_CARD:
            self._act("reprompt", say="pay_repeat", glance="slot", **extra)
        elif st == S.PAY_REMOVE_CARD:
            self._act("reprompt", say="remove_card", glance="slot", **extra)
        if st in AWAITING:
            self.awaiting_since = now

    def _set_overlay(self, overlay: Ov, now: float) -> None:
        self.overlay = overlay
        self.overlay_t = now
        if overlay == Ov.ONE_AT_A_TIME:
            self._act("one_at_a_time", say="one_at_a_time", overlay="one_at_a_time", overlay_data={},
                      gaze="user", listen="yesno", clear_overlay=False, plan="one_at_a_time")

    def _clear_overlay(self) -> None:
        self.overlay = Ov.NONE
        self._act("clear_overlay", clear_overlay=True)  # 계획도 그 상태의 기본 계획으로 돌아간다

    def _pause(self, now: float) -> None:
        session = self._s()
        session.resume_state = self.state
        self.overlay = Ov.PAUSED
        self.overlay_t = now
        self.flags["resume_asked"] = False
        self.awaiting_since = None
        self.log.log("session.pause", state=str(self.state))
        self._act("pause", say="pause", overlay="paused",
                  overlay_data={"asked": False, "cart": session.cart.to_ui(self.menu)},
                  posture="neutral", gaze="wander", clear_overlay=False, plan="paused")

    def _resume(self, now: float) -> None:
        session = self._s()
        state = session.resume_state or S.MENU
        if state == S.OFFER_HELP:
            state = S.MENU
        self.overlay = Ov.NONE
        self.log.log("session.resume", state=str(state))
        self._enter(state, now, resumed=True)

    def _cancel(self, now: float, silent: bool = False) -> None:
        session = self.session
        self.log.log("session.cancel", silent=silent, session=session.id if session else None)
        self.overlay = Ov.NONE
        self._enter(S.IDLE, now)  # IDLE 액트가 하던 말을 끊으므로 작별 인사는 그 뒤에
        if not silent and session is not None:
            self._act("cancelled", say="cancelled", clear_overlay=True, lang=session.lang)  # 세션 언어로 작별 인사

    # ---- 화면 데이터 ----------------------------------------------------------
    def _menu_data(self, highlight: str | None = None) -> dict[str, Any]:
        session = self.session
        return {"menu": self.menu.to_ui(), "cart": session.cart.to_ui(self.menu) if session else None,
                "highlight": highlight}

    def _rec_data(self, item: MenuItem | None, reasons: list[str]) -> dict[str, Any]:
        session = self._s()
        return {"prefs": session.prefs.to_ui(), "item": item.to_ui() if item else None, "reasons": reasons,
                "rejected": sorted(session.rejected), "cart": session.cart.to_ui(self.menu)}

    def _options_data(self, line: CartLine, asking: str) -> dict[str, Any]:
        item = self.menu.get(line.item_id)
        return {"item": item.to_ui() if item else None, "line": line.to_dict(), "asking": asking,
                "unit_price": line.unit_price(self.menu), "large_upcharge": self.menu.large_upcharge}

    def _cart_data(self) -> dict[str, Any]:
        session = self._s()
        return {"cart": session.cart.to_ui(self.menu), "dine": session.dine}

    def _cart_slots(self) -> dict[str, Any]:
        session = self._s()
        return {"lines": [line.to_dict() for line in session.cart.lines], "total": session.cart.total(self.menu),
                "dine": session.dine}

    def _pay_data(self, step: str, fallback: bool = False) -> dict[str, Any]:
        session = self.session
        return {"step": step, "fallback": fallback,
                "total": session.cart.total(self.menu) if session else 0}
