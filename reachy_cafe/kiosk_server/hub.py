"""WebSocket 클라이언트 묶음과 마지막 상태 보관 (새로 접속한 화면이 바로 현재 상태를 그릴 수 있게)."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

CHANNELS = ("ui", "device", "operator")


class Hub:
    def __init__(self) -> None:
        self._clients: dict[str, set[Any]] = {channel: set() for channel in CHANNELS}
        self.view: dict[str, Any] = {}
        self.status: dict[str, Any] = {}
        self.snapshot: dict[str, Any] = {}
        self.led = "off"
        self.log_tail: deque[dict[str, Any]] = deque(maxlen=200)
        # Pi 키오스크 페이지(/ws/ui?device=pi)의 연결. /healthz의 pi_kiosk로 알려 start_kiosk.sh가 흰 화면을 알아챈다
        self.pi_kiosks: set[Any] = set()

    # ---- 연결 ---------------------------------------------------------------
    def connect(self, channel: str, ws: Any) -> None:
        self._clients[channel].add(ws)

    def disconnect(self, channel: str, ws: Any) -> None:
        self._clients[channel].discard(ws)

    def count(self, channel: str) -> int:
        return len(self._clients[channel])

    def send(self, channel: str, message: dict[str, Any]) -> None:
        clients = self._clients.get(channel)
        if not clients:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for ws in list(clients):
            loop.create_task(self._safe_send(channel, ws, message))

    async def _safe_send(self, channel: str, ws: Any, message: dict[str, Any]) -> None:
        try:
            await ws.send_json(message)
        except Exception:
            self.disconnect(channel, ws)

    # ---- 상태 ---------------------------------------------------------------
    def set_view(self, view: dict[str, Any]) -> None:
        self.view = view
        message = {"type": "view", **view}
        self.send("ui", message)
        self.send("operator", message)

    def set_status(self, status: dict[str, Any]) -> None:
        self.status = status
        message = {"type": "status", **status}
        self.send("ui", message)
        self.send("operator", message)

    def set_led(self, mode: str) -> None:
        self.led = mode
        self.send("device", {"type": "led", "mode": mode})

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot
        self.send("operator", {"type": "snapshot", **snapshot})

    def push_log(self, record: dict[str, Any]) -> None:
        self.log_tail.append(record)
        self.send("operator", {"type": "log", "record": record})

    def welcome(self, channel: str) -> list[dict[str, Any]]:
        """새로 접속한 클라이언트에게 보낼 현재 상태."""
        if channel == "ui":
            return [{"type": "view", **self.view}, {"type": "status", **self.status}]
        if channel == "device":
            return [{"type": "led", "mode": self.led}]
        return [{"type": "view", **self.view}, {"type": "status", **self.status},
                {"type": "snapshot", **self.snapshot}, {"type": "log_tail", "records": list(self.log_tail)}]
