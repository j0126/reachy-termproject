"""Print the robot daemon's journal (reachy-mini-daemon) over its logs websocket.

    python -m reachy_s2s.apps.daemon_logs                    # last 100 lines, then 3 s of new lines
    python -m reachy_s2s.apps.daemon_logs --follow 30        # keep streaming for 30 s
    python -m reachy_s2s.apps.daemon_logs --grep audio       # only lines containing "audio"

Useful to see what the robot did with our WebRTC audio, e.g.
"Setting up incoming audio playback for peer ..." / "Audio playback pipeline
started" or GStreamer errors. Read-only (``ws://<robot>:8000/logs/ws/daemon``,
Wireless daemon). Note: the robot's clock may be wrong without internet (NTP).
"""

from __future__ import annotations

import argparse
import sys
import time

from websockets.sync.client import connect

from ..config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", help="robot IP / hostname (default: REACHY_HOST or config)")
    parser.add_argument("--config", action="append",
                        help="TOML file overriding config/default.toml (repeatable, applied in order)")
    parser.add_argument("--follow", type=float, default=3.0, metavar="S", help="seconds to keep streaming")
    parser.add_argument("--grep", help="case-insensitive substring filter")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides={"robot": {"host": args.host}})
    url = f"ws://{cfg.robot.host}:{cfg.robot.port}/logs/ws/daemon"
    needle = args.grep.lower() if args.grep else None
    try:
        with connect(url, open_timeout=5) as ws:
            end = time.time() + args.follow
            while time.time() < end:
                try:
                    msg = ws.recv(timeout=max(0.05, end - time.time()))
                except TimeoutError:
                    break
                line = msg if isinstance(msg, str) else msg.decode("utf-8", "replace")
                if needle is None or needle in line.lower():
                    print(line.rstrip(), flush=True)
    except OSError as exc:
        print(f"[FAIL] cannot open {url}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
