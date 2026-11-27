"""Dashboard: connection, live mel spectrogram, record / play back (Phase 1), the
basic cascaded speech-to-speech system (Phase 2), the cloud system (Phase 3) and
the robot motion controls (Phase 4.5, Motion tab).

The Conversation tab runs ONE system at a time: A0 (basic cascade, Phase 2),
A1 (improved cascade, Phase 4: same modules + barge-in / merge) or B (Gemini Live).
The system can only change while nothing runs, so two never share the mic or the
speaker. All use the same panel (state, transcripts, latencies, history) and the
same Start / Reset / New session / Save buttons.

Motion (Phase 4.5): with ``[motion] auto_enable_motors`` (config/demo.toml) the motors
start right after connecting when it is safe; closing the window or disconnecting
always runs the normal shutdown first (sleep pose -> torque off).

Threading: robot I/O and model work never run on the Qt thread. Connecting,
loading engines, stopping the pipeline and saving run in worker threads;
capture, playback, mel, VAD, the cascaded turn worker and the cloud session
have their own threads. Worker threads talk to the GUI only through
``_Bridge`` signals.
"""

from __future__ import annotations

import logging
import threading
import tomllib
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ..apps.common import open_backend
from ..audio.convert import normalize_peak
from ..audio.mel import mel_band_edges
from ..audio.mel_monitor import MelMonitor
from ..audio.signals import tone
from ..config import AppConfig
from ..conversation.cloud_pipeline import CloudS2SPipeline
from ..conversation.manager import CascadedPipeline
from ..conversation.state import InteractionState, StateTracker
from ..engines.factory import Engines, build_engines
from ..evaluation.audio_stats import recording_stats
from ..evaluation.resource_monitor import ResourceMonitor
from ..evaluation.session import Session
from ..motion.controller import (ACTIVE, ESTOP, FAULT, HOLD, SHUTTING_DOWN, TORQUE_OFF, TORQUE_ON_MODES,
                                 MotionController)
from ..motion.sinks import RobotSink, SimulatedSink, simulated_like_robot
from ..robot.audio_manager import AudioManager, Recording
from .mel_view import MelSpectrogramView
from .motion_panel import MotionPanel

log = logging.getLogger(__name__)

STATE_COLORS = {
    InteractionState.IDLE: "#6b7280",
    InteractionState.LISTENING: "#16a34a",
    InteractionState.PROCESSING: "#d97706",
    InteractionState.SPEAKING: "#2563eb",
    InteractionState.INTERRUPTED: "#dc2626",
    InteractionState.ERROR: "#7f1d1d",
}
LEVEL_FLOOR_DB = -80.0
#: Conversation systems (docs/EVALUATION_PROTOCOL.md section 0). A0 and A1 share the local modules and differ only in
#: [cascaded] (config/a1.toml); B is the cloud system. Every turn row records which one ran (column "variant").
SYSTEMS = (("A0", "A0 — 기본 Cascaded (Phase 2 시스템)"),
           ("A1", "A1 — 개선 Cascaded (Phase 4: 끼어들기·정정)"),
           ("B", "B — Cloud S2S (Gemini Live)"))
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def cascaded_variants(cfg: AppConfig) -> dict[str, Any]:
    """[cascaded] settings of A0 and A1 on top of ``cfg``: A1 = config/a1.toml, A0 = the same keys taken from
    config/default.toml (the Phase 2 baseline). ASR, LLM, TTS and VAD are shared, so A0 vs A1 isolates the
    turn-taking changes."""
    def table(name: str) -> dict[str, Any]:
        path = CONFIG_DIR / name
        return tomllib.loads(path.read_text(encoding="utf-8")).get("cascaded", {}) if path.exists() else {}

    a1, base = table("a1.toml"), table("default.toml")
    cur = cfg.cascaded
    if not a1:  # config folder not found: only the configured variant
        return {cur.variant: cur}

    def build(values: dict[str, Any]) -> Any:
        out = {}
        for k, v in values.items():
            if hasattr(cur, k):
                old = getattr(cur, k)
                out[k] = type(old)(v) if old is not None and not isinstance(old, bool) else v
        return replace(cur, **out)

    return {"A0": build({k: base[k] for k in a1 if k in base}), "A1": build(a1)}
LATENCY_FIELDS = (
    ("VAD endpoint", "vad_endpoint_delay_ms", "speech end (VAD) -> endpoint decision"),
    ("ASR", "asr_ms", "faster-whisper decode"),
    ("LLM", "llm_ms", "LLM request -> text"),
    ("TTS", "tts_ms", "TTS request -> audio"),
    ("speech end -> push", "speech_end_to_push_ms", "VAD speech end -> first audio pushed to the robot (measured)"),
    ("total (est.)", "response_latency_est_ms", "speech end -> push + ASSUMED output latency (not measured)"),
)
CLOUD_LATENCY_FIELDS = (
    ("VAD endpoint (local)", "vad_endpoint_delay_ms",
     "local Silero: speech end -> endpoint (measurement only; the server decides the turn)"),
    ("uplink -> cloud audio", "uplink_to_cloud_audio_ms",
     "block with the speech end sent -> first response audio received (network + server VAD + model)"),
    ("speech end -> cloud audio", "speech_end_to_cloud_audio_ms",
     "local speech end estimate -> first response audio received"),
    ("cloud audio -> push", "cloud_audio_to_push_ms", "first audio received -> first audio pushed to the robot"),
    ("speech end -> push", "speech_end_to_push_ms",
     "local speech end estimate -> first audio pushed to the robot (same metric as the cascade)"),
    ("total (est.)", "response_latency_est_ms", "speech end -> push + ASSUMED output latency (not measured)"),
)


def variant_summary(cfg: Any, cloud: bool) -> str:
    """One line naming the evaluated variant and its turn-taking settings."""
    if cloud:
        c = cfg.cloud
        return (f"variant {c.variant}: server VAD silence {c.silence_duration_ms} ms, "
                f"{'server barge-in' if c.activity_handling == 'interrupts' else 'no barge-in'}")
    c = cfg.cascaded
    text = f"variant {c.variant}: VAD silence {cfg.vad.silence_timeout_ms:.0f} ms, "
    if cfg.tts.provider == "supertonic":  # baseline (0 = all cores) vs improved TTS threads (Phase 4.5)
        text += f"TTS threads {cfg.tts.threads or 'all cores'}, "
    if not c.barge_in:
        return text + "half duplex (speech while busy is ignored)"
    return (text + f"barge-in ≥{c.bargein_min_speech_ms:.0f} ms at p≥{c.bargein_threshold}, "
            f"interrupted answer: {c.interrupted_history}, "
            f"{'cancel + merge' if c.merge_cancelled_input else 'cancel'} while preparing")


def turn_taking_note(t: dict[str, Any]) -> str:
    """Short barge-in / premature-endpoint note for the turn line (both systems)."""
    parts = []
    if t.get("interrupted"):
        kind = t.get("interrupt_kind") or "server"
        bits = [f"barge-in ({kind})"]
        if t.get("bargein_detect_delay_ms") is not None:
            bits.append(f"detect {t['bargein_detect_delay_ms']:.0f} ms")
        if t.get("playback_stop_delay_ms") is not None:
            bits.append(f"stop {t['playback_stop_delay_ms']:.0f} ms")
        if t.get("heard_ratio_est") is not None:
            bits.append(f"heard ~{100 * t['heard_ratio_est']:.0f}%")
        if t.get("history_policy"):
            bits.append(f"history {t['history_policy']}")
        parts.append(", ".join(bits))
    elif t.get("user_interrupt_start") is not None:
        parts.append(f"user spoke while {t.get('user_interrupt_phase')}")
    if t.get("merged_from"):
        parts.append(f"merged turns {t['merged_from']}")
    if t.get("premature_suspect"):
        parts.append(f"premature endpoint? (spoke again {t.get('resume_after_endpoint_ms')} ms after it)")
    return ("  — " + "; ".join(parts)) if parts else ""


def turn_taking_counts(turns: list[Any]) -> str:
    """Live Phase 4 counters over the turn records of the running system."""
    variant = next((t.variant for t in reversed(turns) if getattr(t, "variant", "")), "-")
    interrupted = sum(1 for t in turns if t.interrupted)
    cancelled = sum(1 for t in turns if t.status == "cancelled")
    busy_speech = sum(1 for t in turns if t.user_interrupt_start is not None)
    premature = sum(1 for t in turns if t.premature_suspect)
    return (f"turn-taking [{variant}]: {len(turns)} turns, user speech while busy {busy_speech}, "
            f"interruptions {interrupted} (of which cancelled before playback {cancelled}), "
            f"premature endpoint suspected {premature}")


class _Bridge(QtCore.QObject):
    connected = QtCore.Signal(object)
    connect_failed = QtCore.Signal(str)
    disconnected = QtCore.Signal()
    event = QtCore.Signal(object)
    state = QtCore.Signal(str)
    saved = QtCore.Signal(object)
    engines_ready = QtCore.Signal(object)
    engines_failed = QtCore.Signal(str)
    pipeline_update = QtCore.Signal(str, object)
    pipeline_stopped = QtCore.Signal()
    session_switch = QtCore.Signal(bool)


class DashboardWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: AppConfig, session: Session, mock_wav: str | None = None,
                 fake_engines: bool = False, vad_model: Any = None, echo_llm: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.session = session
        self.mock_wav = mock_wav
        self.fake_engines = fake_engines
        self.echo_llm = echo_llm
        self.vad_model = vad_model  # tests inject a stand-in; None = Silero
        self._cascaded = cascaded_variants(cfg)  # A0 / A1 settings (the window offers both when config/ exists)
        self._auto_motor_pending = False  # automatic Motor ON after connecting is in progress
        self._disconnect_when_torque_off = False
        self.audio: AudioManager | None = None
        self.conn: Any = None
        self.monitor: MelMonitor | None = None
        self.engines: Engines | None = None
        self.pipeline: CascadedPipeline | CloudS2SPipeline | None = None
        self.recordings: list[Recording] = []
        self._rec_counter = 0
        self._busy = False
        self._engines_loading = False
        self.bridge = _Bridge()
        self.state: StateTracker | None = None
        self.motion: MotionController | None = None  # Phase 4.5: created by the first Motor ON
        self.resources: ResourceMonitor | None = None
        self._close_when_torque_off = False

        self._build_ui()
        self._update_title()
        self.bridge.connected.connect(self._on_connected)
        self.bridge.connect_failed.connect(self._on_connect_failed)
        self.bridge.disconnected.connect(self._on_disconnected)
        self.bridge.event.connect(self._append_event)
        self.bridge.state.connect(self._show_state)
        self.bridge.saved.connect(self._on_saved)
        self.bridge.engines_ready.connect(self._on_engines_ready)
        self.bridge.engines_failed.connect(self._on_engines_failed)
        self.bridge.pipeline_update.connect(self._on_pipeline_update)
        self.bridge.pipeline_stopped.connect(self._on_pipeline_stopped)
        self.bridge.session_switch.connect(self._on_session_switch)
        self._event_listener = self.bridge.event.emit  # keep one object so it can be removed
        self._attach_session(session)
        self._show_state(self.state.state.value)

        self._render_timer = QtCore.QTimer(self)
        self._render_timer.timeout.connect(self._render)
        self._render_timer.start(33)
        self._stats_timer = QtCore.QTimer(self)
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start(1000)
        self._motion_timer = QtCore.QTimer(self)
        self._motion_timer.timeout.connect(self._motion_tick)
        self._motion_timer.start(250)
        self._refresh_buttons()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        self.source_box = QtWidgets.QComboBox()
        self.source_box.addItem("Reachy Mini")
        if self.mock_wav:
            self.source_box.addItem("Mock WAV")
            self.source_box.setCurrentIndex(1)
        self.host_edit = QtWidgets.QLineEdit(self.cfg.robot.host)
        self.host_edit.setFixedWidth(140)
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.status_label = QtWidgets.QLabel("disconnected")
        # long status / error texts must not widen the window
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                        QtWidgets.QSizePolicy.Policy.Preferred)
        top.addWidget(QtWidgets.QLabel("Source"))
        top.addWidget(self.source_box)
        top.addWidget(QtWidgets.QLabel("Host"))
        top.addWidget(self.host_edit)
        top.addWidget(self.connect_btn)
        top.addWidget(self.status_label, 1)
        root.addLayout(top)

        bar = QtWidgets.QHBoxLayout()
        self.state_label = QtWidgets.QLabel()
        self.state_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.state_label.setFixedWidth(150)
        font = self.state_label.font()
        font.setPointSize(font.pointSize() + 3)
        font.setBold(True)
        self.state_label.setFont(font)
        self.level_bar = QtWidgets.QProgressBar()
        self.level_bar.setRange(0, int(-LEVEL_FLOOR_DB))
        self.level_bar.setTextVisible(False)
        self.level_label = QtWidgets.QLabel("   -- dBFS")
        self.level_label.setFixedWidth(80)
        self.channel_box = QtWidgets.QComboBox()
        self.channel_box.addItems(["left", "right", "mean"])
        self.channel_box.setCurrentText(self.cfg.audio.mic_channel)
        self.auto_levels = QtWidgets.QCheckBox("auto colour range")
        bar.addWidget(self.state_label)
        bar.addWidget(QtWidgets.QLabel("Mic level"))
        bar.addWidget(self.level_bar, 1)
        bar.addWidget(self.level_label)
        bar.addWidget(QtWidgets.QLabel("Display channel"))
        bar.addWidget(self.channel_box)
        bar.addWidget(self.auto_levels)
        root.addLayout(bar)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        left = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        mel = self.cfg.mel
        centers = mel_band_edges(mel.n_mels, mel.fmin_hz, mel.fmax_hz)[1:-1]
        self.mel_view = MelSpectrogramView(mel.n_mels, mel.display_seconds, centers,
                                           (mel.display_db_min, mel.display_db_max))
        left.addWidget(self.mel_view)
        left.addWidget(self._build_conversation_panel())
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 2)
        split.addWidget(left)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_audio_tab(), "Audio")
        tabs.addTab(self._build_cascaded_tab(), "Conversation")
        self.motion_panel = MotionPanel(self.cfg)
        self.motion_panel.provider = self._ensure_motion
        tabs.addTab(self.motion_panel, "Motion")
        tabs.setCurrentIndex(1)
        split.addWidget(tabs)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(400)
        self.log_view.setFixedHeight(130)
        self.log_view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        # The fixed-pitch font has no Hangul: the first Korean line (the first transcript) made Windows look
        # for a fallback font while the GUI thread held the GIL, stalling the mic capture and the motion loop
        # for 0.3-0.65 s once per process (Phase 4 / R0 logs, 2026-10-17). Do that lookup now, before connecting.
        self.log_view.appendPlainText("이벤트 로그 (한국어 표시 준비)")
        root.addWidget(self.log_view)
        self.setCentralWidget(central)
        self.resize(1280, 900)

        self.connect_btn.clicked.connect(self.toggle_connection)
        self.channel_box.currentTextChanged.connect(self._on_channel_changed)
        self.auto_levels.toggled.connect(self._on_auto_levels)

    def _build_audio_tab(self) -> QtWidgets.QWidget:
        side = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(side)
        self.record_btn = QtWidgets.QPushButton("Record")
        self.play_btn = QtWidgets.QPushButton("Play on robot")
        self.normalize_box = QtWidgets.QCheckBox("normalize playback (peak -3 dBFS, max +30 dB)")
        self.normalize_box.setChecked(True)
        self.normalize_box.setToolTip("Quiet-room recordings peak near -30 dBFS and are inaudible when replayed as-is.")
        self.stop_btn = QtWidgets.QPushButton("Stop playback")
        self.tone_btn = QtWidgets.QPushButton("Test tone 440 Hz")
        for w in (self.record_btn, self.play_btn, self.normalize_box, self.stop_btn, self.tone_btn):
            layout.addWidget(w)
        layout.addWidget(QtWidgets.QLabel("Recordings (saved to the session folder)"))
        self.rec_list = QtWidgets.QListWidget()
        layout.addWidget(self.rec_list, 1)
        self.stats_label = QtWidgets.QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.stats_label)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.play_btn.clicked.connect(self.play_selected)
        self.stop_btn.clicked.connect(self.stop_playback)
        self.tone_btn.clicked.connect(self.play_test_tone)
        return side

    def _build_cascaded_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addWidget(QtWidgets.QLabel("System"))
        self.mode_box = QtWidgets.QComboBox()
        for key, title in SYSTEMS:
            if key == "B" or key in self._cascaded:
                self.mode_box.addItem(title, key)
        start = self.cfg.cascaded.variant if self.cfg.cascaded.variant in self._cascaded else "B"
        self.mode_box.setCurrentIndex(max(0, self.mode_box.findData(start)))
        self.mode_box.setToolTip("한 번에 하나만 실행됩니다. 바꾸려면 먼저 Stop 하세요. A0/A1은 이 PC의 GPU·로컬 모델이 필요하고, "
                                 "B는 인터넷과 Gemini API 키만 있으면 됩니다.")
        layout.addWidget(self.mode_box)
        self.variant_label = QtWidgets.QLabel("")
        self.variant_label.setWordWrap(True)
        self.variant_label.setToolTip("선택한 시스템의 설정 요약. 시스템 이름(A0/A1/B)은 모든 턴 기록(turns.csv의 variant 열)에 남습니다.")
        layout.addWidget(self.variant_label)
        self.load_btn = QtWidgets.QPushButton("Load engines" + (" (fake)" if self.fake_engines else ""))
        self.cascade_btn = QtWidgets.QPushButton("Start cascaded")
        self.reset_btn = QtWidgets.QPushButton("Reset conversation")
        self.reconnect_btn = QtWidgets.QPushButton("Reconnect cloud (keep context)")
        self.reconnect_btn.setToolTip("Close the Live connection and resume it with the last session handle")
        self.new_session_btn = QtWidgets.QPushButton("New session")
        self.save_btn = QtWidgets.QPushButton("Save session log")
        self.reset_btn.setToolTip("대화 내용만 지웁니다(과제 사이에 사용). 한 번만 누르면 됩니다.")
        self.new_session_btn.setToolTip("새 결과 폴더를 만듭니다(확인 창). 과제 사이에는 Reset을 쓰세요.")
        self.open_results_btn = QtWidgets.QPushButton("Open results folder (결과 폴더 열기)")
        for w in (self.load_btn, self.cascade_btn, self.reset_btn, self.reconnect_btn, self.new_session_btn,
                  self.save_btn, self.open_results_btn):
            layout.addWidget(w)
        self.results_label = QtWidgets.QLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.results_label)
        layout.addWidget(QtWidgets.QLabel("Local VAD speech probability"))
        self.vad_bar = QtWidgets.QProgressBar()
        self.vad_bar.setRange(0, 100)
        layout.addWidget(self.vad_bar)
        self.cloud_label = QtWidgets.QLabel("cloud: not started")
        self.cloud_label.setWordWrap(True)
        self.cloud_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.cloud_label)
        self.engine_label = QtWidgets.QLabel("engines not loaded")
        self.engine_label.setWordWrap(True)
        self.engine_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.engine_label)
        layout.addStretch(1)
        self.mode_box.currentIndexChanged.connect(self._on_mode_changed)
        self.load_btn.clicked.connect(self.load_engines)
        self.cascade_btn.clicked.connect(self.toggle_cascaded)
        self.reset_btn.clicked.connect(self.reset_history)
        self.reconnect_btn.clicked.connect(self.reconnect_cloud)
        self.new_session_btn.clicked.connect(self._confirm_new_session)
        self.save_btn.clicked.connect(self.save_log)
        self.open_results_btn.clicked.connect(self.open_results_folder)
        self._on_mode_changed()
        return tab

    def reconnect_cloud(self) -> None:
        if isinstance(self.pipeline, CloudS2SPipeline) and self.pipeline.running:
            self.pipeline.reconnect()
            self.status_label.setText("cloud: reconnecting with the session handle (context kept)")

    @property
    def system(self) -> str:
        """A0 | A1 | B (the selected conversation system)."""
        return self.mode_box.currentData() or "B"

    @property
    def mode(self) -> str:
        return "cloud" if self.system == "B" else "cascaded"

    def set_mode(self, mode: str) -> None:
        """Select ``cloud`` (B) or ``cascaded`` (the configured A0 / A1 variant); ignored while a system runs."""
        variant = self.cfg.cascaded.variant if self.cfg.cascaded.variant in self._cascaded else "A0"
        self.set_system("B" if mode == "cloud" else variant)

    def set_system(self, system: str) -> None:
        """Select A0, A1 or B (ignored while a system runs)."""
        if self.pipeline is not None and self.pipeline.running:
            return
        index = self.mode_box.findData(system)
        if index >= 0:
            self.mode_box.setCurrentIndex(index)

    def system_cfg(self) -> AppConfig:
        """The configuration the selected system runs with (A0 and A1 differ only in [cascaded])."""
        if self.system in self._cascaded:
            return replace(self.cfg, cascaded=self._cascaded[self.system])
        return self.cfg

    def _on_mode_changed(self, *_: Any) -> None:
        cloud = self.mode == "cloud"
        fields = CLOUD_LATENCY_FIELDS if cloud else LATENCY_FIELDS
        self.latency_labels = {}
        for col, (title, key, tip) in enumerate(fields):
            self.latency_heads[col].setText(f"<b>{title}</b>")
            for widget in (self.latency_heads[col], self.latency_values[col]):
                widget.setToolTip(tip)
            self.latency_values[col].setText("—")
            self.latency_labels[key] = self.latency_values[col]
        self._latency_fields = fields
        self.user_title.setText("<b>사용자 (Cloud 자막)</b>" if cloud else "<b>사용자 (ASR)</b>")
        self.robot_title.setText("<b>로봇 (Cloud 자막)</b>" if cloud else "<b>로봇 (LLM)</b>")
        self.variant_label.setText(variant_summary(self.system_cfg(), cloud))
        self.cloud_label.setVisible(cloud)
        if cloud:
            c = self.cfg.cloud
            self.cloud_label.setText(
                f"cloud: not started — {c.model}, voice {c.voice or '(server default)'}, server VAD silence "
                f"{c.silence_duration_ms} ms, {'full duplex' if c.uplink_while_speaking == 'stream' else 'half duplex'}"
                + (" [FAKE server]" if self.fake_engines else ""))
        if hasattr(self, "connect_btn"):
            self._refresh_buttons()

    def _build_conversation_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(panel)
        big = QtGui.QFont(self.font())
        big.setPointSize(big.pointSize() + 3)
        self.user_text = QtWidgets.QLabel("—")
        self.robot_text = QtWidgets.QLabel("—")
        for label in (self.user_text, self.robot_text):
            label.setWordWrap(True)
            label.setFont(big)
            label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.user_title = QtWidgets.QLabel("<b>사용자 (ASR)</b>")
        self.robot_title = QtWidgets.QLabel("<b>로봇 (LLM)</b>")
        grid.addWidget(self.user_title, 0, 0)
        grid.addWidget(self.user_text, 0, 1, 1, 5)
        grid.addWidget(self.robot_title, 1, 0)
        grid.addWidget(self.robot_text, 1, 1, 1, 5)
        self.latency_labels: dict[str, QtWidgets.QLabel] = {}
        self.latency_heads: list[QtWidgets.QLabel] = []
        self.latency_values: list[QtWidgets.QLabel] = []
        self._latency_fields = LATENCY_FIELDS
        for col, (title, key, tip) in enumerate(LATENCY_FIELDS):  # relabelled per mode (_on_mode_changed)
            head = QtWidgets.QLabel(f"<b>{title}</b>")
            head.setToolTip(tip)
            value = QtWidgets.QLabel("—")
            value.setToolTip(tip)
            grid.addWidget(head, 2, col)
            grid.addWidget(value, 3, col)
            self.latency_heads.append(head)
            self.latency_values.append(value)
            self.latency_labels[key] = value
        self.turn_label = QtWidgets.QLabel("")
        grid.addWidget(self.turn_label, 4, 0, 1, 6)
        self.history_view = QtWidgets.QPlainTextEdit()
        self.history_view.setReadOnly(True)
        grid.addWidget(self.history_view, 5, 0, 1, 6)
        grid.setRowStretch(5, 1)
        return panel

    def _refresh_buttons(self) -> None:
        connected = self.audio is not None
        running = self.pipeline is not None and self.pipeline.running
        cloud = self.mode == "cloud"
        torque_on = self.motion is not None and self.motion.mode in TORQUE_ON_MODES
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.connect_btn.setEnabled(not self._busy and not running and not self._disconnect_when_torque_off)
        self.connect_btn.setToolTip("Disconnect: the robot first goes to its sleep pose and the torque goes off "
                                    "(normal shutdown)" if torque_on else "")
        self.source_box.setEnabled(not connected and not self._busy)
        self.host_edit.setEnabled(not connected and not self._busy)
        self.record_btn.setEnabled(connected)
        self.stop_btn.setEnabled(connected)
        self.tone_btn.setEnabled(connected and not running)
        self.play_btn.setEnabled(connected and bool(self.recordings) and not running)
        self.mode_box.setEnabled(not running and not self._busy)
        self.load_btn.setEnabled(not cloud and self.engines is None and not self._engines_loading)
        engines_ready = cloud or self.engines is not None  # the cloud system needs no local models
        self.cascade_btn.setEnabled(connected and engines_ready and not self._busy)
        self.cascade_btn.setText(("Stop " if running else "Start ") + ("cloud S2S (B)" if cloud else f"cascaded ({self.system})"))
        self.reset_btn.setEnabled(self.pipeline is not None)
        self.reconnect_btn.setVisible(cloud)
        self.reconnect_btn.setEnabled(cloud and running)
        self.save_btn.setEnabled(True)
        self.new_session_btn.setEnabled(not self._busy)

    def _update_title(self) -> None:
        self.setWindowTitle(f"Reachy S2S dashboard  [{self.session.id}]")
        if hasattr(self, "results_label"):
            self.results_label.setText(f"결과 폴더 (results folder): {self.session.dir}")

    def open_results_folder(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.session.dir)))

    def _confirm_new_session(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self, "New session", "새 결과 폴더(세션)를 만들까요?\n\n"
            "과제(대본) 사이에 대화만 지우려면 'Reset conversation'을 누르세요.")
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self.new_session()

    # ------------------------------------------------------------- session
    def _attach_session(self, session: Session) -> None:
        self.session = session
        session.events.add_listener(self._event_listener)
        initial = self.state.state if self.state is not None else InteractionState.IDLE
        self.state = StateTracker(session.events, initial=initial)
        self.state.add_listener(lambda _old, new, _rec: self.bridge.state.emit(new.value))
        if self.audio is not None:
            self.audio.set_event_logger(session.events)
        if self.motion is not None:  # same robot, new logs: follow the new state tracker and event log
            self.motion.attach(self.state, session.events)
            self.motion.set_session_dir(session.dir)
        if self.resources is not None:
            self.resources.set_event_logger(session.events)
        self._update_title()

    def new_session(self) -> None:
        """Close this session's logs and continue in a fresh results folder (models stay loaded)."""
        if self._busy:
            return
        was_running = self.pipeline is not None and self.pipeline.running
        pipeline, self.pipeline = self.pipeline, None
        self._busy = True
        self._refresh_buttons()

        def worker() -> None:  # stopping may wait for an in-flight API request
            if pipeline is not None:
                pipeline.stop()
            self.bridge.session_switch.emit(was_running)

        threading.Thread(target=worker, name="gui-new-session", daemon=True).start()

    def _on_session_switch(self, was_running: bool) -> None:
        old = self.session
        old.events.remove_listener(self._event_listener)
        old.close()
        self._attach_session(Session(self.cfg.logging.results_path, "dashboard"))
        self.session.write_metadata(app="dashboard", config=self.cfg.to_dict(),
                                    engines=self.engines.info if self.engines else None,
                                    previous_session=old.id)
        self.recordings.clear()
        self.rec_list.clear()
        self.history_view.clear()
        self.user_text.setText("—")
        self.robot_text.setText("—")
        self.status_label.setText(f"new session {self.session.id}")
        self._busy = False
        if was_running:
            self._start_pipeline()
        else:
            self._set_conversation_running(False)
        self._refresh_buttons()

    def save_log(self) -> None:
        if self.pipeline is not None:
            self.pipeline.save_conversation()
        self.session.write_metadata(saved_from_gui=True)
        self.session.events.log("session.saved")
        self.status_label.setText(f"saved: {self.session.dir}")

    # ----------------------------------------------------------- connection
    def toggle_connection(self) -> None:
        if self._busy:
            return
        if self.audio is None:
            self._connect()
        else:
            self._disconnect()

    def _connect(self) -> None:
        self._busy = True
        self._refresh_buttons()
        use_mock = self.source_box.currentText() == "Mock WAV"
        host = self.host_edit.text().strip()
        cfg = self.cfg
        if host and host != cfg.robot.host:
            cfg = replace(cfg, robot=replace(cfg.robot, host=host))
        self.status_label.setText(f"connecting to {'mock WAV' if use_mock else host} ...")
        events = self.session.events

        def worker() -> None:
            audio = conn = None
            try:
                backend, conn, info = open_backend(cfg, self.mock_wav if use_mock else None)
                audio = AudioManager(backend, cfg.audio, events)
                audio.start()
                self.bridge.connected.emit((audio, conn, info))
            except Exception as exc:
                log.exception("connect failed")
                if conn is not None:
                    conn.close()
                self.bridge.connect_failed.emit(repr(exc))

        threading.Thread(target=worker, name="gui-connect", daemon=True).start()

    def _on_connected(self, payload: tuple) -> None:
        self.audio, self.conn, info = payload
        self._busy = False
        self.audio.set_event_logger(self.session.events)
        self.monitor = MelMonitor(self.audio.subscribe("mel"), self.audio.input_rate, self.cfg.mel,
                                  self.channel_box.currentText())
        self.monitor.start()
        backend = self.audio.backend
        self.session.write_metadata(robot=info, backend=backend.describe(), config=self.cfg.to_dict())
        version = info.get("daemon_version") or info.get("mock_wav", "")
        self.status_label.setText(f"connected: {backend.name}  {backend.input_rate} Hz x "
                                  f"{backend.input_channels} ch  {version}")
        self.state.set(InteractionState.IDLE, "connected")
        conn = self.conn
        if conn is not None:  # REST torque off works even before a motion controller exists
            self.motion_panel.fallback_estop = lambda: conn.daemon.set_motor_mode("disabled", timeout_s=2.0)
        self.motion_panel.set_connected(True)
        if self.cfg.logging.resource_monitor and self.resources is None:
            self.resources = ResourceMonitor(self.session.events)
            self.resources.start()
        if self.cfg.motion.auto_enable_motors:  # demo default (config/demo.toml); experiments keep it off
            QtCore.QTimer.singleShot(0, self._auto_motor_on)
        self._refresh_buttons()

    def _on_connect_failed(self, message: str) -> None:
        self._busy = False
        self.status_label.setText(f"connect failed: {message}")
        self.state.set(InteractionState.ERROR, "connect failed", error=message)
        self._refresh_buttons()

    def _disconnect(self) -> None:
        if self.motion is not None and self.motion.mode in TORQUE_ON_MODES:
            # normal shutdown first (sleep pose -> torque off); _motion_tick disconnects when the torque is off
            if not self._disconnect_when_torque_off:
                self._disconnect_when_torque_off = True
                if self.motion.mode != SHUTTING_DOWN:
                    self.motion.motor_off()
                self.status_label.setText("연결 해제 전 정상 종료 중: 잠자기 자세 → 모터 OFF (normal shutdown)")
                self._refresh_buttons()
            return
        if self.motion_panel.torque_may_be_on():
            answer = QtWidgets.QMessageBox.warning(
                self, "Disconnect", f"Motion is in {self.motion.mode}: the torque may still be ON.\n"
                f"{self.motion.last_error or ''}\n\nDisconnect anyway (the robot keeps its last target)?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No)
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self._busy = True
        self._refresh_buttons()
        self.status_label.setText("disconnecting ...")
        monitor, audio, conn, pipeline = self.monitor, self.audio, self.conn, self.pipeline
        self.monitor = self.audio = self.conn = self.pipeline = None
        motion, resources = self._detach_motion()

        def worker() -> None:
            if pipeline is not None:
                pipeline.stop()
            self._close_motion(motion, audio, resources)
            self._shutdown(monitor, audio, conn, None)
            self.bridge.disconnected.emit()

        threading.Thread(target=worker, name="gui-disconnect", daemon=True).start()

    @staticmethod
    def _shutdown(monitor: MelMonitor | None, audio: AudioManager | None, conn: Any,
                  pipeline: CascadedPipeline | CloudS2SPipeline | None = None) -> None:
        if pipeline is not None:
            pipeline.stop()
        if monitor is not None:
            monitor.stop()
        if audio is not None:
            if audio.is_recording:
                audio.stop_recording()
            audio.stop()
        if conn is not None:
            conn.close()

    def _detach_motion(self) -> tuple[MotionController | None, ResourceMonitor | None]:
        motion, resources = self.motion, self.resources
        self.motion = self.resources = None
        self.motion_panel.set_controller(None)
        self.motion_panel.fallback_estop = None
        self.motion_panel.set_connected(False)
        self.motion_panel.set_running(False)
        return motion, resources

    @staticmethod
    def _close_motion(motion: MotionController | None, audio: AudioManager | None,
                      resources: ResourceMonitor | None) -> None:
        """Stop the motion threads. Never turns the torque off by itself (normal shutdown is explicit)."""
        if motion is not None:
            if audio is not None:
                audio.remove_playback_tap(motion.tap)
            if motion.mode in TORQUE_ON_MODES or motion.mode == FAULT:
                motion._log("exit_with_torque", mode=motion.mode, last_error=motion.last_error)
            motion.close()
        if resources is not None:
            resources.stop()

    # ------------------------------------------------------------------ motion
    def _motion_notice(self, text: str) -> None:
        self.motion_panel.notice = text
        self.motion_panel.error_label.setText(text)
        self.status_label.setText(text)

    def _auto_motor_on(self) -> None:
        """Demo default (config/demo.toml): Motor ON right after connecting, without a dialog, only when it is safe:
        joint guard loaded, motors reported off, head near the sleep / neutral pose (the controller checks the
        pose and refuses otherwise). If any check fails no motor command is sent: the conversation runs without motion."""
        if self.audio is None or self.motion_panel._locked_ui():
            return
        motion = self._ensure_motion()
        if motion is None:
            self._motion_notice("모션을 시작할 수 없습니다(컨트롤러 생성 실패). 음성 대화는 그대로 됩니다.")
            return
        if motion.mode != TORQUE_OFF:
            return
        if not motion.guard.available:
            self._motion_notice(f"관절 가드를 불러오지 못해 모터를 켜지 않았습니다({motion.guard.error}). 음성 대화만 됩니다.")
            return
        try:
            mode = motion.sink.motor_mode()
        except Exception as exc:  # noqa: BLE001 - reported; nothing is sent
            mode = f"unreadable: {exc!r}"
        if mode != "disabled":
            self._motion_notice(f"모터 상태가 '{mode}'라서 자동으로 켜지 않았습니다(다른 프로그램이 쓰는 중일 수 있음). "
                                "음성 대화만 됩니다.")
            return
        self._auto_motor_pending = True
        self.motion_panel.suppress_pose_prompt = True
        motion.motor_on()  # an unknown start pose is refused by the controller (no torque); see _motion_tick
        self.status_label.setText("모터 켜는 중: 중립 자세로 천천히 이동합니다 (약 6초)")

    def _ensure_motion(self) -> MotionController | None:
        """Create the motion controller (first Motor ON); the robot sink only for a real robot connection."""
        if self.motion is not None:
            return self.motion
        if self.audio is None:
            return None
        m = self.cfg.motion
        if self.conn is None:
            sink: Any = SimulatedSink()
        elif m.sink == "dry_run":  # real robot, simulated motors: start from its real pose (reads only)
            sink = simulated_like_robot(self.conn.mini)
        else:
            sink = RobotSink(self.conn.mini, self.conn.daemon)
        try:
            motion = MotionController(self.cfg, sink, self.session.events, self.session.dir)
        except Exception as exc:  # noqa: BLE001 - shown in the Motion tab; the conversation is unaffected
            log.exception("motion controller creation failed")
            self.motion_panel.error_label.setText(f"motion controller failed: {exc!r}")
            return None
        motion.attach(self.state, self.session.events)
        self.audio.add_playback_tap(motion.tap)
        motion.start()
        self.motion = motion
        self.motion_panel.set_controller(motion)
        self.session.write_metadata(motion={"sink": sink.name, "simulated": sink.simulated,
                                            "sim_origin": getattr(sink, "origin", None),
                                            "joint_guard": motion.guard.available, "guard_error": motion.guard.error})
        if self.pipeline is not None and self.pipeline.running:
            self._set_conversation_running(True)
        return motion

    def _set_conversation_running(self, running: bool) -> None:
        """Fix the motion condition while a conversation runs (experiments) and label its turns with it."""
        if self.motion is not None:
            self.motion.lock(running and self.cfg.motion.lock_during_conversation)
        if running and self.pipeline is not None:
            self.pipeline.motion_label = self.motion.label if self.motion is not None else None
        self.motion_panel.set_running(running)

    def _motion_tick(self) -> None:
        self.motion_panel.refresh()
        if self.motion is None:
            return
        self._refresh_buttons()
        m = self.motion
        if self._auto_motor_pending:
            if m.needs_confirmation:  # start pose outside the tolerance: no torque, no dialog in the automatic start
                m.needs_confirmation = None
                self._auto_motor_pending = False
                self.motion_panel.suppress_pose_prompt = False
                self._motion_notice("머리가 시작 허용 범위(잠자기·중립 자세 근처) 밖이라 모터를 켜지 않았습니다. 음성 대화만 됩니다. "
                                    "머리를 잠자기 자세 가까이 둔 뒤 Motion 탭에서 Motor ON을 누르세요.")
            elif m.mode in (HOLD, ACTIVE):
                self._auto_motor_pending = False
                self.motion_panel.suppress_pose_prompt = False
                self.status_label.setText("모션 준비 완료: " + ("Motion ON" if m.motion_enabled else "Motion OFF (중립 유지)"))
            elif m.mode in (FAULT, ESTOP):
                self._auto_motor_pending = False
                self.motion_panel.suppress_pose_prompt = False
                self._motion_notice(f"모터를 켜지 못했습니다: {m.last_error}. 음성 대화만 됩니다.")
        if self._disconnect_when_torque_off:
            if m.mode == TORQUE_OFF:
                self._disconnect_when_torque_off = False
                self._disconnect()
            elif m.mode == FAULT:
                self._disconnect_when_torque_off = False
                self._refresh_buttons()
                QtWidgets.QMessageBox.critical(
                    self, "Normal shutdown failed",
                    f"{m.last_error}\n\nThe torque was left ON. Fix the cause and retry, or use "
                    "EMERGENCY TORQUE OFF (Motion tab, F12).")
        if self._close_when_torque_off:
            if self.motion.mode == TORQUE_OFF:
                self._close_when_torque_off = False
                self.close()
            elif self.motion.mode == FAULT:
                self._close_when_torque_off = False
                QtWidgets.QMessageBox.critical(
                    self, "Normal shutdown failed",
                    f"{self.motion.last_error}\n\nThe torque was left ON. Fix the cause and retry, or use "
                    "EMERGENCY TORQUE OFF (Motion tab, F12).")

    def _on_disconnected(self) -> None:
        self._busy = False
        self.status_label.setText("disconnected")
        self.state.set(InteractionState.IDLE, "disconnected")
        self._refresh_buttons()

    # --------------------------------------------------------------- engines
    def load_engines(self) -> None:
        if self.engines is not None or self._engines_loading:
            return
        self._engines_loading = True
        self._refresh_buttons()
        self.engine_label.setText("loading ASR model (GPU) and API clients ...")
        cfg, events, fake, echo = self.cfg, self.session.events, self.fake_engines, self.echo_llm

        def worker() -> None:
            try:
                self.bridge.engines_ready.emit(build_engines(cfg, events, fake=fake, echo_llm=echo))
            except Exception as exc:
                log.exception("engine loading failed")
                self.bridge.engines_failed.emit(str(exc))

        threading.Thread(target=worker, name="gui-engines", daemon=True).start()

    def _on_engines_ready(self, engines: Engines) -> None:
        self.engines = engines
        self._engines_loading = False
        info = engines.info
        if info.get("fake"):
            text = "FAKE engines (no GPU / API) - for dry runs only"
        else:
            a, llm = info["asr"], info["llm"]
            if llm.get("provider") == "openai_compat":
                llm_text = f"{llm['model']} via {llm.get('base_url')} (temperature {llm.get('temperature')})"
            else:
                llm_text = f"{llm['model']} (thinking {llm.get('thinking_level')})"
            warm = llm.get("warmup") or {}
            if warm:
                llm_text += f", warm-up {warm.get('warmup_ms')} ms" + (
                    "" if warm.get("ok") else " FAILED → 로컬 LLM 서버가 꺼져 있으면 start_llm_server.bat를 먼저 실행하세요")
            text = (f"ASR: {a['model']} {a['device']} {a['compute_type']} - load {a['load_ms']} ms, "
                    f"warm-up {a['warmup_ms']} ms, GPU mem {a['gpu_mem_before_mb']}->{a['gpu_mem_after_mb']} MB\n"
                    f"LLM: {llm_text}\n"
                    f"TTS: {info['tts']['model']} voice {info['tts']['voice']} @ {info['tts']['sample_rate']} Hz")
        self.engine_label.setText(text)
        self.session.write_metadata(engines=info)
        self._refresh_buttons()

    def _on_engines_failed(self, message: str) -> None:
        self._engines_loading = False
        self.engine_label.setText(
            f"engine loading failed: {message}\n"
            "→ 이 PC에서 A0/A1(로컬 GPU 모델)을 실행할 수 없을 수 있습니다. check_setup.bat로 GPU·모델·LLM 서버를 "
            "확인하세요(TROUBLESHOOTING.md). B(Cloud)는 엔진 없이 실행됩니다.")
        self._refresh_buttons()

    # -------------------------------------------------------------- cascaded
    def toggle_cascaded(self) -> None:
        if self.pipeline is not None and self.pipeline.running:
            self._stop_pipeline()
        else:
            self._start_pipeline()

    def _start_pipeline(self) -> None:
        if self.audio is None:
            return
        cfg = self.system_cfg()
        self.session.write_metadata(last_system=self.system, last_system_cascaded=asdict(cfg.cascaded))
        if self.mode == "cloud":
            connector = self._cloud_connector()
            if connector is None:
                return
            if self.audio.is_recording:
                self.toggle_recording()
            self.pipeline = CloudS2SPipeline(self.audio, cfg, self.session, self.state, connector=connector,
                                             vad_model=self.vad_model)
        else:
            if self.engines is None:
                return
            if self.audio.is_recording:
                self.toggle_recording()
            e = self.engines
            self.pipeline = CascadedPipeline(self.audio, cfg, self.session, self.state, e.asr, e.llm, e.tts,
                                             vad_model=self.vad_model)
        self.pipeline.add_listener(self.bridge.pipeline_update.emit)
        self._set_conversation_running(True)
        self.pipeline.start()
        self.history_view.clear()
        self._refresh_buttons()

    def _cloud_connector(self) -> Any:
        if self.fake_engines:
            from ..engines.fake_live import FakeLiveConnector

            return FakeLiveConnector()
        from ..engines.gemini_client import api_key_present
        from ..engines.gemini_live import GeminiLiveConnector

        if not api_key_present():
            self.cloud_label.setText("cloud: GEMINI_API_KEY is not set (.env)")
            self.state.set(InteractionState.ERROR, "cloud: no API key")
            return None
        return GeminiLiveConnector()

    def _stop_pipeline(self) -> None:
        pipeline = self.pipeline
        self._busy = True
        self._refresh_buttons()
        self.status_label.setText("stopping (waits for an in-flight request / closes the cloud session) ...")

        def worker() -> None:
            pipeline.stop()
            self.bridge.pipeline_stopped.emit()

        threading.Thread(target=worker, name="gui-stop-cascaded", daemon=True).start()

    def _on_pipeline_stopped(self) -> None:
        self._busy = False
        self._set_conversation_running(False)
        self.status_label.setText(f"{self.mode} stopped")
        self._refresh_buttons()

    def reset_history(self) -> None:
        if self.pipeline is not None:
            dropped = self.pipeline.reset_history()
            extra = " (new cloud session, server context discarded)" if self.mode == "cloud" else ""
            self.history_view.clear()
            self.user_text.setText("—")
            self.robot_text.setText("—")
            self.status_label.setText(f"대화 초기화 완료 — conversation reset ({dropped} exchanges dropped){extra}")
            self.reset_btn.setEnabled(False)  # one press is enough; enabled again shortly
            QtCore.QTimer.singleShot(800, self._refresh_buttons)

    def _on_pipeline_update(self, kind: str, payload: dict) -> None:
        if kind == "history":
            lines = [f"{'사용자' if m['role'] == 'user' else '로봇'} [{m['turn_id']}]: {m['text']}"
                     for m in payload["transcript"]]
            self.history_view.setPlainText("\n".join(lines))
            self.history_view.verticalScrollBar().setValue(self.history_view.verticalScrollBar().maximum())
            return
        if kind == "status":  # cloud connection state
            p = payload
            extra = ", ".join(f"{k}={p[k]}" for k in ("attempt", "failures", "resumed") if p.get(k) is not None)
            self.cloud_label.setText(f"cloud: {p['status']} (connection {p.get('connection_id')}) — {p.get('model')}, "
                                     f"voice {p.get('voice')}" + (f" [{extra}]" if extra else "")
                                     + (" [FAKE server]" if self.fake_engines else ""))
            return
        t = payload
        self.user_text.setText(t.get("merged_text") or t.get("transcript") or "…")
        busy = t.get("status") in ("processing", "responding")
        self.robot_text.setText(t.get("response") or ("…" if busy else "—"))
        for _title, key, _tip in self._latency_fields:
            value = t.get(key)
            if key in self.latency_labels:
                self.latency_labels[key].setText("—" if value is None else f"{value:.0f} ms")
        status = t.get("status")
        detail = f"{t['error_stage']} {t['error_kind']} {t.get('error_status') or ''}" if t.get("error_stage") else ""
        motion = f" | motion {t['motion']}" if t.get("motion") not in (None, "", "none") else ""
        self.turn_label.setText(f"turn {t['turn_id']} [{t.get('variant') or '-'}{motion}]: {status} {detail}"
                                f"{turn_taking_note(t)}"
                                f"   (output latency {t.get('output_hint_ms') or '-'} ms is an assumption)")

    # ------------------------------------------------------ record / play
    def toggle_recording(self) -> None:
        if self.audio is None:
            return
        if self.audio.is_recording:
            rec = self.audio.stop_recording()
            self.record_btn.setText("Record")
            if self.pipeline is None or not self.pipeline.running:
                self.state.set(InteractionState.IDLE, "recording stopped")
            session = self.session

            def worker() -> None:
                paths = rec.save(session.audio_dir, session.t0_ns)
                self.bridge.saved.emit((rec, paths, recording_stats(rec.frames, rec.sample_rate)))

            threading.Thread(target=worker, name="gui-save", daemon=True).start()
        else:
            self._rec_counter += 1
            self.audio.start_recording(f"rec{self._rec_counter:02d}")
            self.record_btn.setText("Stop recording")
            if self.pipeline is None or not self.pipeline.running:
                self.state.set(InteractionState.LISTENING, "recording")

    def _on_saved(self, payload: tuple) -> None:
        rec, paths, stats = payload
        self.recordings.append(rec)
        item = QtWidgets.QListWidgetItem(
            f"{rec.name}  {rec.duration_s:5.1f} s  rms {stats.get('ch0_rms_dbfs')} dBFS  "
            f"peak {stats.get('ch0_peak_dbfs')} dBFS"
        )
        item.setToolTip(paths["wav"])
        item.setData(QtCore.Qt.ItemDataRole.UserRole, len(self.recordings) - 1)
        self.rec_list.addItem(item)
        self.rec_list.setCurrentItem(item)
        self.session.events.log("record.saved", name=rec.name, **paths, stats=stats)
        self._refresh_buttons()

    def play_selected(self) -> None:
        if self.audio is None or not self.recordings:
            return
        item = self.rec_list.currentItem()
        rec = self.recordings[item.data(QtCore.Qt.ItemDataRole.UserRole)] if item else self.recordings[-1]
        samples, gain_db = rec.mono(self.cfg.audio.mic_channel), 0.0
        if self.normalize_box.isChecked():
            samples, gain_db = normalize_peak(samples)
        self.audio.play(samples, rec.sample_rate, label=f"playback_{rec.name}",
                        meta={"gain_db": round(gain_db, 2)})
        self.state.set(InteractionState.SPEAKING, "playback", label=rec.name)

    def play_test_tone(self) -> None:
        if self.audio is None:
            return
        rate = self.audio.backend.output_rate
        self.audio.play(tone(440.0, 0.6, rate, amplitude=0.3), rate, label="test_tone_440Hz")
        self.state.set(InteractionState.SPEAKING, "test tone")

    def stop_playback(self) -> None:
        if self.audio is None:
            return
        self.audio.stop_playback("user")
        if self.state.state is InteractionState.SPEAKING and (self.pipeline is None or not self.pipeline.running):
            self.state.set(InteractionState.INTERRUPTED, "user stop")

    # ------------------------------------------------------------- timers
    def _render(self) -> None:
        if self.monitor is not None:
            self.mel_view.update_image(self.monitor.snapshot())
        if self.audio is not None:
            db = max(LEVEL_FLOOR_DB, self.audio.level_dbfs(0.1))
            self.level_bar.setValue(int(db - LEVEL_FLOOR_DB))
            self.level_label.setText(f"{db:6.1f} dBFS")
            cascaded = self.pipeline is not None and self.pipeline.running
            if not cascaded and self.state.state in (InteractionState.SPEAKING, InteractionState.INTERRUPTED) \
                    and not self.audio.is_speaking():
                nxt = InteractionState.LISTENING if self.audio.is_recording else InteractionState.IDLE
                self.state.set(nxt, "playback ended (estimated)")
        if self.pipeline is not None and self.pipeline.running:
            self.vad_bar.setValue(int(100 * self.pipeline.vad.last_prob))

    def _update_stats(self) -> None:
        if self.audio is None:
            return
        s = self.audio.input_stats()
        lines = [
            f"input: {s.get('audio_s', 0):.1f} s audio, {s.get('chunks')} chunks "
            f"({s.get('chunk_frames_min')}-{s.get('chunk_frames_max')} frames), gaps {s.get('gaps')}",
            f"measured rate: {s.get('measured_rate_hz', '-')} Hz, read errors {s.get('read_errors')}",
            f"queue drops: {s.get('subscriber_drops')}",
        ]
        if self.monitor is not None:
            lines.append(f"mel: {self.monitor.frames_total} frames, last chunk {self.monitor.compute_ms_last:.2f} ms")
        if self.pipeline is not None:
            lines.append(turn_taking_counts(list(self.pipeline.turns)))
        if self.motion is not None:
            st = self.motion.status()
            lines.append(f"motion: {st['mode']} ({st['label']}), sends {st['sends']}, faults {st['faults']}, "
                         f"guard rejections {st['guard_rejected']}, player tap errors {self.audio.player.tap_errors}")
        if self.resources is not None:
            lines.append(f"resource monitor: {self.resources.samples} samples"
                         + (f" ({self.resources.error})" if self.resources.error else ""))
        if isinstance(self.pipeline, CloudS2SPipeline):
            u = self.pipeline.uplink_stats
            lines.append(f"cloud uplink: {u['blocks']} blocks ({u['bytes'] / 32000:.1f} s), dropped "
                         f"{u['dropped_queue']} (queue) / {u['dropped_disconnected']} (disconnected), "
                         f"paused {u['paused_chunks']}")
        self.stats_label.setText("\n".join(lines))

    # ------------------------------------------------------------- slots
    def _append_event(self, record: dict) -> None:
        fields = ", ".join(f"{k}={v}" for k, v in record.items() if k not in ("t", "kind"))
        self.log_view.appendPlainText(f"{record['t']:9.3f}  {record['kind']:<24} {fields[:300]}")

    def _show_state(self, value: str) -> None:
        state = InteractionState(value)
        self.state_label.setText(state.value)
        self.state_label.setStyleSheet(
            f"background:{STATE_COLORS[state]}; color:white; border-radius:6px; padding:4px;"
        )

    def _on_channel_changed(self, channel: str) -> None:
        if self.monitor is not None:
            self.monitor.set_channel(channel)

    def _on_auto_levels(self, enabled: bool) -> None:
        self.mel_view.auto_levels = enabled
        if not enabled:
            self.mel_view.set_levels(self.cfg.mel.display_db_min, self.cfg.mel.display_db_max)

    # ------------------------------------------------------------- close
    def shutdown(self) -> None:
        """Synchronous cleanup (used at exit). Leaves the torque as it is (normal shutdown is explicit)."""
        self._render_timer.stop()
        self._stats_timer.stop()
        self._motion_timer.stop()
        self.session.events.remove_listener(self._event_listener)
        monitor, audio, conn, pipeline = self.monitor, self.audio, self.conn, self.pipeline
        self.monitor = self.audio = self.conn = self.pipeline = None
        motion, resources = self._detach_motion()
        if pipeline is not None:
            pipeline.stop()
        self._close_motion(motion, audio, resources)
        self._shutdown(monitor, audio, conn, None)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.motion is not None and self.motion.mode in TORQUE_ON_MODES:
            # torque on: normal shutdown first (sleep pose -> torque off); _motion_tick closes the window afterwards
            if not self._close_when_torque_off:
                self._close_when_torque_off = True
                if self.motion.mode != SHUTTING_DOWN:
                    self.motion.motor_off()
                self.status_label.setText("로봇을 잠자기 자세로 옮기고 모터를 끈 뒤 창을 닫습니다 (normal shutdown) ...")
            event.ignore()
            return
        if self.motion is not None and self.motion_panel.torque_may_be_on():  # FAULT: torque state unknown
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            box.setWindowTitle("Motion FAULT")
            box.setText(f"Motion is {self.motion.mode}: the torque may still be ON.\n{self.motion.last_error or ''}")
            retry = box.addButton("Normal shutdown (retry), then close", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            estop = box.addButton("Emergency Torque OFF and close", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is retry:
                self.motion.motor_off()
                self._close_when_torque_off = True
                event.ignore()
                return
            if box.clickedButton() is not estop:
                event.ignore()
                return
            self.motion.emergency_torque_off()
        self.shutdown()
        super().closeEvent(event)
