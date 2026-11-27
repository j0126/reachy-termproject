"""Dashboard (PySide6): live mel spectrogram, record / play back, cascaded and cloud S2S.

    python -m reachy_s2s.apps.dashboard                    # then Connect / (Load engines) / Start
    python -m reachy_s2s.apps.dashboard --config config/demo.toml --system B --autoconnect   # team demo
    python -m reachy_s2s.apps.dashboard --auto-cascaded    # connect + load + start the cascade
    python -m reachy_s2s.apps.dashboard --auto-cloud       # connect + start the cloud system (Gemini Live)
    python -m reachy_s2s.apps.dashboard --mock some.wav --fake-engines --auto-cascaded   # no robot/GPU/API
    python -m reachy_s2s.apps.dashboard --mock some.wav --fake-engines --auto-cloud      # fake Live server

Scripted run (verification / screenshots):

    python -m reachy_s2s.apps.dashboard --autoconnect --auto-record 4 --auto-play \
        --screenshot dash.png --quit-after 14
"""

from __future__ import annotations

import argparse
import sys

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..config import load_config
from ..evaluation.session import Session, setup_console_logging
from ..gui.dashboard import DashboardWindow
from .common import add_common_args, prepare_runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--autoconnect", action="store_true", help="connect at startup")
    parser.add_argument("--auto-cascaded", action="store_true", help="connect, load engines and start cascaded")
    parser.add_argument("--auto-cloud", action="store_true", help="connect and start the cloud system (Gemini Live)")
    parser.add_argument("--system", choices=("A0", "A1", "B"),
                        help="system selected at start (default: the configured cascade, B with --auto-cloud)")
    parser.add_argument("--fake-engines", action="store_true",
                        help="stand-ins: ASR/LLM/TTS (no GPU, no API) and a fake Live server for the cloud mode")
    parser.add_argument("--echo-llm", action="store_true",
                        help="real VAD/ASR/TTS but the LLM is replaced by a deterministic echo (hardware tests)")
    parser.add_argument("--auto-record", type=float, metavar="S", help="after connecting, record S seconds")
    parser.add_argument("--auto-play", action="store_true", help="play the auto recording back on the robot")
    parser.add_argument("--screenshot", metavar="PNG", help="save a window screenshot before quitting")
    parser.add_argument("--quit-after", type=float, metavar="S", help="close the window after S seconds")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides={"robot": {"host": args.host}})
    setup_console_logging(cfg.logging.level)
    runtime = prepare_runtime(cfg)
    pg.setConfigOptions(antialias=False, imageAxisOrder="col-major")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    session = Session(cfg.logging.results_path, "dashboard")
    session.write_metadata(app="dashboard", args=vars(args), config=cfg.to_dict(), runtime=runtime)
    window = DashboardWindow(cfg, session, mock_wav=args.mock, fake_engines=args.fake_engines,
                             echo_llm=args.echo_llm)
    if args.system:
        window.set_system(args.system)
    if args.auto_cloud:
        window.set_mode("cloud")
    window.show()

    def after_connect() -> None:
        if window.audio is None:
            QtCore.QTimer.singleShot(200, after_connect)
            return
        if args.auto_record:
            window.toggle_recording()
            QtCore.QTimer.singleShot(int(args.auto_record * 1000), stop_auto_record)
        if args.auto_cascaded:
            start_when_loaded()
        if args.auto_cloud:
            window.toggle_cascaded()  # starts the system selected in the mode box (cloud)

    def start_when_loaded() -> None:
        if window.engines is None:
            QtCore.QTimer.singleShot(200, start_when_loaded)
            return
        window.toggle_cascaded()

    def stop_auto_record() -> None:
        window.toggle_recording()
        if args.auto_play:
            QtCore.QTimer.singleShot(500, play_when_saved)

    def play_when_saved() -> None:
        if not window.recordings:
            QtCore.QTimer.singleShot(200, play_when_saved)
            return
        window.play_selected()

    def finish() -> None:
        if args.screenshot:
            window.grab().save(args.screenshot)
            print(f"[..] screenshot: {args.screenshot}")
        window.close()

    if args.autoconnect or args.auto_cascaded or args.auto_cloud:
        QtCore.QTimer.singleShot(100, window.toggle_connection)
        QtCore.QTimer.singleShot(300, after_connect)
    if args.auto_cascaded:
        QtCore.QTimer.singleShot(150, window.load_engines)
    if args.quit_after:
        QtCore.QTimer.singleShot(int(args.quit_after * 1000), finish)

    code = app.exec()
    window.shutdown()
    window.session.close()
    print(f"[..] session: {window.session.dir}")
    return code


if __name__ == "__main__":
    sys.exit(main())
