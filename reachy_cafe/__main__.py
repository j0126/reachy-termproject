"""실행: python -m reachy_cafe [--config config/dev_laptop.toml] [--port 8000] [--robot-host 192.168.0.15]"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .app import App
from .config import load_config


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except AttributeError:
            pass
    parser = argparse.ArgumentParser(prog="python -m reachy_cafe", description="Reachy Café 키오스크 두뇌")
    parser.add_argument("--config", action="append", default=[], metavar="TOML",
                        help="덮어쓰기 설정 파일 (여러 번 줄 수 있음). 예: config/dev_laptop.toml")
    parser.add_argument("--host", help="서버 수신 주소 (기본 0.0.0.0)")
    parser.add_argument("--port", type=int, help="서버 포트 (기본 8000)")
    parser.add_argument("--robot-host", help="로봇 IP (기본은 config의 robot.host 또는 REACHY_HOST)")
    args = parser.parse_args(argv)

    cfg = load_config(*args.config)
    if args.host:
        cfg["server"]["host"] = args.host
    if args.port:
        cfg["server"]["port"] = args.port
    if args.robot_host:
        cfg["robot"]["host"] = args.robot_host
    try:
        asyncio.run(App(cfg).run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
