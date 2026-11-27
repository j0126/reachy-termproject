"""규칙 기반 추천. 손님의 피드백("너무 달아요")이 다음 추천을 바꾸는 closed loop의 핵심."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .menu import Menu, MenuItem

# 손님 취향 태그 → 메뉴 태그
TAG_MAP = {
    "refreshing": {"refreshing", "fruity"},
    "sweet": {"sweet"},
    "milky": {"milky", "creamy"},
}


@dataclass
class Prefs:
    caffeine: bool | None = None   # False = 카페인 없는 것
    sweet_max: int | None = None   # 당도 상한 (0–3)
    temp: str | None = None
    tags: set[str] = field(default_factory=set)
    updated_by: str | None = None  # 마지막으로 조건을 바꾼 피드백 (화면 표시용)

    def merge(self, raw: dict[str, Any]) -> None:
        if "caffeine" in raw:
            self.caffeine = raw["caffeine"]
        if "sweet_max" in raw:
            self.sweet_max = raw["sweet_max"] if self.sweet_max is None else min(self.sweet_max, raw["sweet_max"])
        if raw.get("temp"):
            self.temp = raw["temp"]
        self.tags |= set(raw.get("tags", []))

    def toggle(self, key: str) -> None:
        """화면의 취향 칩을 눌렀을 때."""
        if key == "caffeine_free":
            self.caffeine = None if self.caffeine is False else False
        elif key == "less_sweet":
            self.sweet_max = None if self.sweet_max is not None else 1
        elif key in ("hot", "ice"):
            self.temp = None if self.temp == key else key
        elif key in TAG_MAP:
            self.tags ^= {key}

    def is_empty(self) -> bool:
        return self.caffeine is None and self.sweet_max is None and self.temp is None and not self.tags

    def to_ui(self) -> dict[str, Any]:
        return {
            "caffeine_free": self.caffeine is False,
            "less_sweet": self.sweet_max is not None,
            "sweet_max": self.sweet_max,
            "refreshing": "refreshing" in self.tags,
            "milky": "milky" in self.tags,
            "sweet": "sweet" in self.tags,
            "hot": self.temp == "hot",
            "ice": self.temp == "ice",
            "updated_by": self.updated_by,
        }


class Recommender:
    def __init__(self, menu: Menu):
        self.menu = menu

    def _wanted_tags(self, prefs: Prefs) -> set[str]:
        wanted: set[str] = set()
        for tag in prefs.tags:
            wanted |= TAG_MAP.get(tag, {tag})
        return wanted

    def recommend(self, prefs: Prefs, rejected: set[str] | None = None, weather: str | None = None,
                  only: set[str] | None = None) -> list[MenuItem]:
        """weather(context_talk): 추운 날·비 오는 날은 따뜻하게 마실 수 있는 메뉴, 더운 날은 아이스 메뉴를 조금 더 앞에.
        only: 이 메뉴들 중에서만 ("딸기랑 바닐라 중에 뭐가 나아요?")."""
        rejected = rejected or set()
        wanted = self._wanted_tags(prefs)
        favored_temp = {"cold": "hot", "rain": "hot", "hot": "ice"}.get(weather or "")
        scored: list[tuple[float, MenuItem]] = []
        for item in self.menu:
            if not item.in_stock or item.id in rejected or (only is not None and item.id not in only):
                continue
            if prefs.caffeine is False and item.caffeine:
                continue
            if prefs.sweet_max is not None and item.sweetness > prefs.sweet_max:
                continue
            if prefs.temp and prefs.temp not in item.temps:
                continue
            score = item.popularity + 0.25 * len(wanted & set(item.tags))
            if favored_temp and favored_temp in item.temps and len(item.temps) == 1:
                score += 0.3   # 그 온도로만 나오는 메뉴 (캐모마일=따뜻, 딸기 라떼=아이스)
            elif favored_temp and favored_temp in item.temps:
                score += 0.1
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _score, item in scored]

    def on_reject(self, prefs: Prefs, item: MenuItem, reason: str | None) -> Prefs:
        """거절 이유를 다음 추천 조건에 반영한다."""
        if reason == "sweet":
            prefs.sweet_max = max(0, item.sweetness - 1)
            prefs.tags.discard("sweet")
            prefs.updated_by = "too_sweet"
        elif reason == "bitter":
            prefs.tags.add("sweet")
            prefs.updated_by = "too_bitter"
        else:
            prefs.updated_by = "rejected"
        return prefs

    def alternatives(self, item: MenuItem, n: int = 2) -> list[MenuItem]:
        """품절 메뉴와 비슷한 대안 (태그 겹침 + 당도 차이 + 인기도 + 같은 분류)."""
        base = set(item.tags)
        scored = []
        for other in self.menu:
            if other.id == item.id or not other.in_stock:
                continue
            score = (2.0 * len(base & set(other.tags))
                     - 0.5 * abs(other.sweetness - item.sweetness)
                     + other.popularity
                     + (0.5 if other.category == item.category else 0.0))
            scored.append((score, other))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [other for _score, other in scored[:n]]

    def reasons(self, item: MenuItem, prefs: Prefs, focus: str | None = None) -> list[str]:
        """추천 이유 키 (최대 2개). 방금 받은 피드백과 관련된 이유를 앞에 둔다."""
        sweet = None
        if prefs.sweet_max is not None:
            sweet = "not_sweet" if item.sweetness == 0 else "less_sweet"
        caffeine = "caffeine_free" if prefs.caffeine is False and not item.caffeine else None
        ordered = (sweet, caffeine) if focus == "too_sweet" else (caffeine, sweet)
        out: list[str] = [reason for reason in ordered if reason]
        if "refreshing" in item.tags and ("refreshing" in prefs.tags or len(out) < 2):
            out.append("refreshing")
        if {"milky", "creamy"} & set(item.tags) and "milky" in prefs.tags:
            out.append("creamy")
        if not out:
            out.append("popular")
        return out[:2]
