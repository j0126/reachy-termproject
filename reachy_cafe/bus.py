"""이벤트 버스와 JSONL 로그.

모든 입력(터치·카드·발화·얼굴·오퍼레이터)은 Event로 버스에 들어오고, 디스패처 한 곳에서 처리된다.
로그 형식은 HW3 EventLogger와 같다: {"t": 시작 후 경과 초, "kind": ..., ...}.
"""

from __future__ import annotations

import asyncio
import functools
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# 초당 여러 번 들어와서 파일에 남기지 않는 이벤트
QUIET_KINDS = frozenset({"vision.faces"})


@dataclass
class Event:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    t: float = field(default_factory=time.monotonic)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class EventLog:
    """세션 폴더(logs/<시각>/events.jsonl)에 기록하고, 리스너(오퍼레이터 패널)에 전달한다."""

    def __init__(self, directory: str | Path | None = None):
        self._t0 = time.monotonic()
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self.path: Path | None = None
        self._fh = None
        self._lock = threading.Lock()  # 카메라·오디오 스레드가 동시에 써도 줄이 섞이지 않게
        if directory is not None:
            folder = Path(directory) / time.strftime("%Y%m%d_%H%M%S")
            folder.mkdir(parents=True, exist_ok=True)
            self.path = folder / "events.jsonl"
            self._fh = self.path.open("a", encoding="utf-8")

    def add_listener(self, fn: Callable[[dict[str, Any]], None]) -> None:
        self._listeners.append(fn)

    def log(self, kind: str, t: float | None = None, **fields: Any) -> dict[str, Any]:
        now = time.monotonic() if t is None else t
        record = {"t": round(now - self._t0, 3), "kind": kind, **fields}
        with self._lock:
            if self._fh is not None:
                self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                self._fh.flush()
        for fn in list(self._listeners):
            try:
                fn(record)
            except Exception:  # 리스너 오류가 로깅을 막지 않게
                pass
        return record

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class EventBus:
    """asyncio 큐 기반 버스. 다른 스레드(카메라·오디오)에서는 publish_threadsafe를 쓴다."""

    def __init__(self, log: EventLog | None = None):
        self.log = log or EventLog(None)
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, kind: str, **data: Any) -> Event:
        event = Event(kind, data)
        if kind not in QUIET_KINDS:
            self.log.log(kind, t=event.t, **data)
        self._queue.put_nowait(event)
        return event

    def publish_threadsafe(self, kind: str, **data: Any) -> None:
        if self._loop is None:
            raise RuntimeError("EventBus.bind_loop()이 먼저 호출돼야 합니다")
        if self._loop.is_closed():  # 종료 중에 늦게 도착한 카메라·오디오 이벤트는 버린다
            return
        self._loop.call_soon_threadsafe(functools.partial(self.publish, kind, **data))

    def log_threadsafe(self, kind: str, **fields: Any) -> None:
        """다른 스레드에서 로그만 남길 때. 오퍼레이터 패널 전송(웹소켓)은 이벤트 루프에서 일어나야 한다."""
        t = time.monotonic()
        loop = self._loop
        if loop is None or loop.is_closed():
            self.log.log(kind, t=t, **fields)
            return
        try:
            if asyncio.get_running_loop() is loop:
                self.log.log(kind, t=t, **fields)
                return
        except RuntimeError:
            pass
        loop.call_soon_threadsafe(functools.partial(self.log.log, kind, t=t, **fields))

    async def get(self) -> Event:
        return await self._queue.get()

    def get_nowait(self) -> Event | None:
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
