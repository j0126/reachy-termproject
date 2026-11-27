#!/usr/bin/env python3
"""Reachy Café 카드 슬롯 에이전트 (Raspberry Pi 4에서 실행).

- 마이크로스위치(GPIO17 ↔ GND, 내부 풀업)로 카드 in/out을 감지해 두뇌 PC(/ws/device)로 보낸다.
- 두뇌가 보내는 LED 명령(on/off/blink)을 슬롯 LED(GPIO27)에 반영한다.
- 연결이 끊기면 자동으로 다시 붙는다 (두뇌 PC를 나중에 켜도 됨).
- GPIO가 없는 PC에서는 --keyboard로 테스트: i + Enter = 카드 넣기, o + Enter = 카드 빼기.

설정은 pi_agent/kiosk.env(환경 변수)나 명령행 인자로 준다:
    BRAIN_HOST=192.168.0.10  BRAIN_PORT=8000  CARD_GPIO=17  LED_GPIO=27
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading

try:
    import websockets
except ImportError:  # pragma: no cover
    print("websockets가 없습니다: sudo apt install python3-websockets", file=sys.stderr)
    raise


class GpioIO:
    def __init__(self, card_pin: int, led_pin: int, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[str]):
        from gpiozero import LED, Button

        self.card = Button(card_pin, pull_up=True, bounce_time=0.05)
        self.led = LED(led_pin)
        self.card.when_pressed = lambda: loop.call_soon_threadsafe(queue.put_nowait, "in")
        self.card.when_released = lambda: loop.call_soon_threadsafe(queue.put_nowait, "out")

    def card_state(self) -> str:
        return "in" if self.card.is_pressed else "out"

    def set_led(self, mode: str) -> None:
        if mode == "on":
            self.led.on()
        elif mode == "blink":
            self.led.blink(on_time=0.3, off_time=0.3)
        else:
            self.led.off()


class KeyboardIO:
    """GPIO 없이 테스트할 때. 표준 입력에서 i/o를 읽는다."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[str]):
        self.state = "out"
        threading.Thread(target=self._read, args=(loop, queue), daemon=True).start()
        print("키보드 모드: i + Enter = 카드 넣기, o + Enter = 카드 빼기")

    def _read(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[str]) -> None:
        for line in sys.stdin:
            command = line.strip().lower()
            if command in ("i", "in"):
                self.state = "in"
            elif command in ("o", "out"):
                self.state = "out"
            else:
                continue
            loop.call_soon_threadsafe(queue.put_nowait, self.state)

    def card_state(self) -> str:
        return self.state

    def set_led(self, mode: str) -> None:
        print(f"[LED] {mode}")


def parse_args() -> argparse.Namespace:
    env = os.environ
    parser = argparse.ArgumentParser(description="Reachy Café 카드 슬롯 에이전트")
    parser.add_argument("--host", default=env.get("BRAIN_HOST", "192.168.0.10"), help="두뇌 PC IP")
    parser.add_argument("--port", type=int, default=int(env.get("BRAIN_PORT", "8000")))
    parser.add_argument("--card-gpio", type=int, default=int(env.get("CARD_GPIO", "17")))
    parser.add_argument("--led-gpio", type=int, default=int(env.get("LED_GPIO", "27")))
    parser.add_argument("--keyboard", action="store_true", help="GPIO 대신 키보드로 카드 입력 흉내")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str] = asyncio.Queue()
    io = KeyboardIO(loop, queue) if args.keyboard else GpioIO(args.card_gpio, args.led_gpio, loop, queue)
    uri = f"ws://{args.host}:{args.port}/ws/device"
    print(f"카드 슬롯 에이전트 → {uri}", flush=True)

    async for ws in websockets.connect(uri, open_timeout=5, ping_interval=10, ping_timeout=10):
        try:
            while not queue.empty():  # 끊겨 있던 동안 쌓인 이벤트는 버리고 현재 상태만 알린다
                queue.get_nowait()
            state = io.card_state()
            await ws.send(json.dumps({"type": "hello", "agent": "card_slot", "card": state}))
            if state == "in":
                await ws.send(json.dumps({"type": "card", "state": "in"}))
            print("두뇌 PC에 연결됨", flush=True)

            async def receive() -> None:
                async for raw in ws:
                    try:
                        message = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if message.get("type") == "led":
                        io.set_led(str(message.get("mode", "off")))

            async def transmit() -> None:
                while True:
                    card = await queue.get()
                    await ws.send(json.dumps({"type": "card", "state": card}))
                    print(f"카드 {card}", flush=True)

            await asyncio.gather(receive(), transmit())
        except websockets.ConnectionClosed:
            io.set_led("off")
            print("연결이 끊겼습니다. 다시 연결합니다…", flush=True)
            continue


def main() -> None:
    try:
        asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
