"""FastAPI 앱: 정적 UI 파일과 WebSocket 3개.

- /ws/ui        Pi 4 키오스크 화면 (view·status 수신, touch 송신)
- /ws/device    Pi 4 카드 슬롯 에이전트 (card 송신, led 수신)
- /ws/operator  오퍼레이터 패널 겸 시뮬레이터 (operator 송신, snapshot·log 수신)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from ..bus import EventBus
from .hub import Hub

# 화면 파일은 매번 서버에 바뀌었는지 묻게 한다 (그대로면 304라 가볍다). 캐시 헤더가 없으면 Chromium이
# 몇 시간씩 옛 app.js·style.css를 써서, 두뇌 PC에서 화면을 고쳐도 Pi 키오스크에 바로 반영되지 않는다.
NO_CACHE = {"Cache-Control": "no-cache"}


class UIFiles(StaticFiles):
    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers.update(NO_CACHE)
        return response

INDEX_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reachy Café</title>
<style>body{font-family:system-ui,sans-serif;background:#16110d;color:#f6efe6;display:grid;place-items:center;
min-height:100vh;margin:0}a{display:block;margin:12px 0;padding:16px 24px;border-radius:14px;background:#2a211a;
color:#ffd8a8;text-decoration:none;font-size:20px}main{max-width:420px;padding:16px}</style></head>
<body><main><h1>☕ Reachy Café</h1>
<a href="/kiosk">키오스크 화면 (Pi 4 터치스크린)</a>
<a href="/operator">오퍼레이터 패널 · 시뮬레이터</a>
<a href="/stage">청중 화면 · Reachy의 생각 (프로젝터)</a>
</main></body></html>"""


def create_app(hub: Hub, bus: EventBus, ui_dir: Path) -> FastAPI:
    app = FastAPI(title="Reachy Café", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/ui", UIFiles(directory=str(ui_dir)), name="ui")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

    @app.get("/kiosk")
    async def kiosk() -> FileResponse:
        return FileResponse(ui_dir / "index.html", headers=NO_CACHE)

    @app.get("/operator")
    async def operator() -> FileResponse:
        return FileResponse(ui_dir / "operator.html", headers=NO_CACHE)

    @app.get("/stage")
    async def stage() -> FileResponse:
        return FileResponse(ui_dir / "stage.html", headers=NO_CACHE)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "ui": hub.count("ui"), "device": hub.count("device"), "operator": hub.count("operator"),
                "pi_kiosk": len(hub.pi_kiosks)}

    async def session(channel: str, ws: WebSocket, on_message: Callable[[dict[str, Any]], None],
                      on_open: Callable[[], None] | None = None,
                      on_close: Callable[[], None] | None = None) -> None:
        await ws.accept()
        hub.connect(channel, ws)
        if on_open:
            on_open()
        try:
            for message in hub.welcome(channel):
                await ws.send_json(message)
            while True:
                raw = await ws.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict):
                    on_message(message)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            bus.log.log("error", where=f"ws/{channel}", error=repr(exc))
        finally:
            hub.disconnect(channel, ws)
            if on_close:
                on_close()

    def strip_type(message: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in message.items() if key != "type"}

    def from_ui(message: dict[str, Any]) -> None:
        if message.get("type") == "touch" and message.get("action"):
            bus.publish("touch", **strip_type(message))

    def from_device(message: dict[str, Any]) -> None:
        if message.get("type") == "card" and message.get("state") in ("in", "out"):
            bus.publish("card", state=message["state"], source="device")
        elif message.get("type") == "hello":
            bus.publish("device", state="hello", **strip_type(message))

    def from_operator(message: dict[str, Any]) -> None:
        if message.get("type") == "operator" and message.get("action"):
            bus.publish("operator", **strip_type(message))

    @app.websocket("/ws/ui")
    async def ws_ui(ws: WebSocket) -> None:
        # Pi 키오스크 페이지는 ?device=pi로 붙는다. start_kiosk.sh가 /healthz의 pi_kiosk를 보고, 페이지가 안 붙으면
        # (11/27 첫 부팅 때 흰 화면으로 멈춤) Chromium을 다시 띄운다.
        if ws.query_params.get("device") == "pi":
            hub.pi_kiosks.add(ws)
        try:
            await session("ui", ws, from_ui)
        finally:
            hub.pi_kiosks.discard(ws)

    @app.websocket("/ws/device")
    async def ws_device(ws: WebSocket) -> None:
        await session("device", ws, from_device,
                      on_open=lambda: bus.publish("device", state="connected"),
                      on_close=lambda: bus.publish("device", state="disconnected"))

    @app.websocket("/ws/operator")
    async def ws_operator(ws: WebSocket) -> None:
        await session("operator", ws, from_operator)

    return app
