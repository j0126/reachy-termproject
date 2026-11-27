"""Korean text -> what a local TTS should read aloud (speech only).

Measured on Supertonic 3 (synthesized speech re-transcribed by Whisper):
"16명" came back as "6명" / "전 6명" for 8 of 10 voices and "3시" as "살시"
for one, while plain sentences had 0 % CER. So numbers are spelled out in
Hangul with the reading the following counter needs:

* native numerals (하나, 둘 ... 아흔아홉; attributive 한/두/세/네/스무) before
  counters like 명, 개, 살, 시(1-12), 시간, 번, 번째 (1 -> 첫);
* Sino-Korean (일, 이, 삼 ...) for 분, 초, 월 (6 -> 유월, 10 -> 시월), 일,
  년, 원, %, units, numbers >= 100, decimals and bare numbers;
* "3:20" -> "세 시 이십 분", "3~5명" -> "세에서 다섯 명", "1,000" -> "천".

The conversation history, the GUI and the logs keep the original text; only
the TTS input changes. ``pronunciations`` replaces fixed strings first (e.g.
"Reachy Mini" -> "리치 미니").
"""

from __future__ import annotations

import re
from collections.abc import Iterable

SINO_DIGITS = "영일이삼사오육칠팔구"
_SINO_UNITS = ("", "십", "백", "천")
_SINO_GROUPS = ("", "만", "억", "조", "경")

_NATIVE_ONES = ("", "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉")
_NATIVE_ONES_ATTR = ("", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉")
_NATIVE_TENS = ("", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔")

# counter -> reading. Longest first when matching (시간 before 시, 번째 before 번, 개월 before 개).
NATIVE_COUNTERS = ("번째", "시간", "명", "개", "살", "시", "마리", "잔", "권", "장", "대", "번", "가지",
                   "군데", "병", "그릇", "켤레", "벌", "송이", "곳", "달", "사람")
SINO_COUNTERS = ("개월", "퍼센트", "%", "분", "초", "월", "일", "년", "원", "층", "도", "호", "학년", "회",
                 "점", "등", "배", "킬로미터", "킬로그램", "센티미터", "밀리미터", "미터", "km", "kg", "cm",
                 "mm", "°C", "℃")
_UNIT_WORDS = {"%": "퍼센트", "km": "킬로미터", "kg": "킬로그램", "cm": "센티미터", "mm": "밀리미터",
               "°C": "도", "℃": "도"}
_COUNTERS = sorted(NATIVE_COUNTERS + SINO_COUNTERS, key=len, reverse=True)
_COUNTER_RE = "|".join(re.escape(c) for c in _COUNTERS)
_NUMBER_RE = r"\d{1,3}(?:,\d{3})+|\d+"

_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])")
_RANGE = re.compile(rf"(?<![A-Za-z\d.])({_NUMBER_RE})\s*[~∼]\s*({_NUMBER_RE})\s*({_COUNTER_RE})")
_NUMBER = re.compile(rf"(?<![A-Za-z\d.])({_NUMBER_RE})(\.\d+)?(\s?)({_COUNTER_RE})?")


def sino(n: int) -> str:
    """Sino-Korean reading: 10 -> 십, 16 -> 십육, 10000 -> 만, 2026 -> 이천이십육."""
    if n < 0:
        return "마이너스 " + sino(-n)
    if n == 0:
        return "영"
    parts: list[str] = []
    for group in range(len(_SINO_GROUPS)):
        n, chunk = divmod(n, 10000)
        if chunk:
            text = ""
            for pos in range(3, -1, -1):
                d = chunk // 10 ** pos % 10
                if d:
                    text += ("" if d == 1 and pos else SINO_DIGITS[d]) + _SINO_UNITS[pos]
            if group == 1 and chunk == 1:
                text = ""  # 10000 is 만, not 일만 (but 1억 is 일억)
            parts.append(text + _SINO_GROUPS[group])
        if not n:
            break
    return "".join(reversed(parts))


def native(n: int, attributive: bool = True) -> str | None:
    """Native Korean numeral for 1-99 (attributive form before a counter: 한, 두, 스무 ...)."""
    if not 1 <= n <= 99:
        return None
    if attributive and n == 20:
        return "스무"
    tens, ones = divmod(n, 10)
    return _NATIVE_TENS[tens] + (_NATIVE_ONES_ATTR if attributive else _NATIVE_ONES)[ones]


def _digits(fraction: str) -> str:
    return "".join(SINO_DIGITS[int(c)] for c in fraction)


def read_number(n: int, counter: str | None) -> str:
    """Hangul reading of ``n`` followed by ``counter`` (counter itself not included)."""
    if counter is None:
        return sino(n)
    if counter == "월":
        return {6: "유", 10: "시"}.get(n, sino(n))
    if counter == "번째":
        return "첫" if n == 1 else (native(n) or sino(n))
    if counter == "시":
        return native(n) if 1 <= n <= 12 else sino(n)
    if counter == "시간":
        return native(n) if 1 <= n <= 12 else sino(n)
    if counter == "대" and n >= 10 and n % 10 == 0:
        return sino(n)  # 20대 = 이십 대 (age group), not 스무 대
    if counter in NATIVE_COUNTERS:
        return native(n) or sino(n)
    return sino(n)


def _counter_word(counter: str) -> str:
    return _UNIT_WORDS.get(counter, counter)


def _sep(counter: str) -> str:
    return "" if counter == "월" else " "  # 십일월, 유월 are single words; otherwise 열여섯 명


def _replace_number(m: re.Match) -> str:
    whole, fraction, _space, counter = m.group(1), m.group(2), m.group(3), m.group(4)
    n = int(whole.replace(",", ""))
    if fraction:
        spoken = f"{sino(n)} 점 {_digits(fraction[1:])}"
    else:
        spoken = read_number(n, counter)
    if counter is None:
        return spoken + _space
    sep = " " if fraction else _sep(counter)
    return f"{spoken}{sep}{_counter_word(counter)}"


def _replace_time(m: re.Match) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    text = f"{read_number(hour, '시')} 시"
    return text + (f" {sino(minute)} 분" if minute else "")


def _replace_range(m: re.Match) -> str:
    a, b, counter = int(m.group(1).replace(",", "")), int(m.group(2).replace(",", "")), m.group(3)
    return f"{read_number(a, counter)}에서 {read_number(b, counter)}{_sep(counter)}{_counter_word(counter)}"


def normalize_for_tts(text: str, pronunciations: Iterable[tuple[str, str]] = ()) -> str:
    """Spell out numbers (and fixed pronunciations) for a Korean TTS; other text is unchanged."""
    for written, spoken in pronunciations:
        text = text.replace(written, spoken)
    text = _TIME.sub(_replace_time, text)
    text = _RANGE.sub(_replace_range, text)
    return _NUMBER.sub(_replace_number, text)
