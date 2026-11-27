"""메뉴, 장바구니, 주문번호. 가격·재고·옵션은 전부 여기(데이터)에서 결정한다."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

TEMPS = ("hot", "ice")
SIZES = ("regular", "large")


@dataclass
class MenuItem:
    id: str
    name: dict[str, str]
    aliases: list[str]
    category: str
    price: int
    temps: list[str]
    sizes: list[str]
    sweetness: int
    caffeine: bool
    tags: list[str]
    popularity: float
    emoji: str = ""
    desc: dict[str, str] = field(default_factory=dict)
    in_stock: bool = True
    alias_options: dict[str, dict[str, str]] = field(default_factory=dict)

    def label(self, lang: str) -> str:
        return self.name.get(lang) or self.name.get("ko") or self.id

    def to_ui(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "price": self.price,
            "temps": self.temps,
            "sizes": self.sizes,
            "sweetness": self.sweetness,
            "caffeine": self.caffeine,
            "emoji": self.emoji,
            "desc": self.desc,
            "in_stock": self.in_stock,
        }


class Menu:
    def __init__(self, items: list[MenuItem], categories: list[dict[str, Any]], large_upcharge: int = 500):
        self.items = items
        self.categories = categories
        self.large_upcharge = large_upcharge
        self._by_id = {item.id: item for item in items}

    @classmethod
    def load(cls, path: str | Path) -> "Menu":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        items = [MenuItem(**entry) for entry in raw["items"]]
        return cls(items, raw.get("categories", []), raw.get("large_upcharge", 500))

    def __iter__(self) -> Iterator[MenuItem]:
        return iter(self.items)

    def get(self, item_id: str | None) -> MenuItem | None:
        return self._by_id.get(item_id) if item_id else None

    def set_stock(self, item_id: str, in_stock: bool) -> bool:
        item = self.get(item_id)
        if item is None:
            return False
        item.in_stock = in_stock
        return True

    def to_ui(self) -> dict[str, Any]:
        return {
            "categories": self.categories,
            "items": [item.to_ui() for item in self.items],
            "large_upcharge": self.large_upcharge,
        }


@dataclass
class CartLine:
    item_id: str
    qty: int = 1
    temp: str | None = None
    size: str | None = None
    notes: list[str] = field(default_factory=list)  # 요청사항 키 (nlu.NOTES: less_ice, extra_shot …)

    def autofill(self, menu: Menu) -> None:
        """선택지가 하나뿐인 옵션은 자동으로 채운다 (예: 캐모마일은 hot만)."""
        item = menu.get(self.item_id)
        if item is None:
            return
        if self.temp not in item.temps:
            self.temp = item.temps[0] if len(item.temps) == 1 else None
        if self.size not in item.sizes:
            self.size = item.sizes[0] if len(item.sizes) == 1 else None

    def missing(self) -> list[str]:
        out = []
        if self.temp is None:
            out.append("temp")
        if self.size is None:
            out.append("size")
        return out

    def unit_price(self, menu: Menu) -> int:
        item = menu.get(self.item_id)
        base = item.price if item else 0
        return base + (menu.large_upcharge if self.size == "large" else 0)

    def total(self, menu: Menu) -> int:
        return self.unit_price(menu) * self.qty

    def same_options(self, other: "CartLine") -> bool:
        return self.signature() == other.signature()

    def signature(self) -> str:
        """줄 표식: 메뉴/온도/사이즈/요청사항. 화면의 ✕ 버튼이 보내 와서 지울 줄이 맞는지 확인한다."""
        return "/".join([self.item_id, self.temp or "", self.size or "", "+".join(sorted(self.notes))])

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"item": self.item_id, "qty": self.qty, "temp": self.temp, "size": self.size}
        if self.notes:
            out["notes"] = list(self.notes)
        return out


class Cart:
    def __init__(self) -> None:
        self.lines: list[CartLine] = []

    def add(self, line: CartLine) -> None:
        for existing in self.lines:
            if existing.same_options(line):
                existing.qty += line.qty
                return
        self.lines.append(line)

    def remove_item(self, item_id: str) -> bool:
        before = len(self.lines)
        self.lines = [line for line in self.lines if line.item_id != item_id]
        return len(self.lines) != before

    def remove_index(self, index: int) -> bool:
        if 0 <= index < len(self.lines):
            del self.lines[index]
            return True
        return False

    def merge(self) -> None:
        """고친 뒤 옵션이 같아진 줄을 하나로 합친다 (라떼 아이스 1 + 라떼를 아이스로 바꿈 → 라떼 아이스 2)."""
        merged: list[CartLine] = []
        for line in self.lines:
            same = next((m for m in merged if m.same_options(line)), None)
            if same is None:
                merged.append(line)
            else:
                same.qty += line.qty
        self.lines = merged

    def is_empty(self) -> bool:
        return not self.lines

    def count(self) -> int:
        return sum(line.qty for line in self.lines)

    def total(self, menu: Menu) -> int:
        return sum(line.total(menu) for line in self.lines)

    def to_ui(self, menu: Menu) -> dict[str, Any]:
        lines = []
        for index, line in enumerate(self.lines):
            item = menu.get(line.item_id)
            lines.append({
                "index": index,
                "item": line.item_id,
                "name": item.name if item else {"ko": line.item_id, "en": line.item_id},
                "emoji": item.emoji if item else "",
                "qty": line.qty,
                "temp": line.temp,
                "size": line.size,
                "notes": list(line.notes),
                "sign": line.signature(),
                "price": line.total(menu),
            })
        return {"lines": lines, "total": self.total(menu), "count": self.count()}


class OrderCounter:
    """파일에 저장되는 주문번호 (재시작해도 이어지고 1–99를 순환)."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._value = 0
        if self.path and self.path.exists():
            try:
                self._value = int(self.path.read_text(encoding="utf-8").strip() or 0)
            except ValueError:
                self._value = 0

    def next(self) -> int:
        self._value = self._value % 99 + 1
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(str(self._value), encoding="utf-8")
        return self._value
