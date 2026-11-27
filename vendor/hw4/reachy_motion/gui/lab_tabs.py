"""PySide6 tabs of the Motion Lab: Experiment (R1 / R2 / R3 trials on MuJoCo or the real robot, live view) and
Results (tracking analysis, R3 and expressive-vs-neutral comparisons, one export).

Thin widgets over ``gui.experiments`` (trials, computed checks), ``gui.lab`` (one worker thread per backend: the GUI
never sends a command itself and never runs in the 50 Hz loop), ``gui.results`` and ``gui.lab_plots``.
Real robot (user decisions 2026-11-12, docs/lab/GUI_ROBOT.md): the "실제 로봇" mode shows ``gui.robot_panel`` (IP / port,
connect, wake-up, torque-off, disconnect) over ``gui.robot_session.RobotLabSession`` (one SDK client). Each click of
the person at the GUI is the approval of that one action (``safety.gui_approval``: one trial, never a batch). The Mock
stays as a hidden offline backend for the tests.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6 import QtCore, QtGui, QtWidgets

from reachy_motion.gui import experiments as ex
from reachy_motion.gui import guide as gd
from reachy_motion.gui import ko
from reachy_motion.gui import lab_plots
from reachy_motion.gui import results as res
from reachy_motion.gui.lab import LAB_ROOT, SIM_PORT, LabSession, new_session_dir
from reachy_motion.gui.robot_panel import STATE_SHORT, RobotPanel, ask_torque_off
from reachy_motion.gui.robot_session import R0_ROOT, RobotLabSession
from reachy_motion.library.pose import pose_from_matrix
from reachy_motion.library.spec import MOTION_NAMES

DEG = 57.29577951308232
KIND_COLOR = {"measured": "#1a7f37", "simulated": "#2a78d6", "mock": "#6b6a66", "computed": "#6b6a66"}
STATUS_COLOR = {"done": "#1a7f37", "anomaly": "#c0392b", "refused": "#c0392b", "error": "#c0392b",
                "running": "#2a78d6", "queued": "#2a78d6", "stopped": "#b26b00", "cancelled": "#6b6a66"}
LOCK_TEXT = "실제 로봇: 창 위쪽 로봇 상자에서 연결 → Wake-up → 시행. 누른 버튼의 동작만 실행됩니다."
TABLE_COLS = ("#", "시행", "속도 비", "관절 여유 deg", "정책", "상태")
STATUS_COL = TABLE_COLS.index("상태")
POLICY_COL = TABLE_COLS.index("정책")
NOTE_STYLE = "color:#6b6a66"


class EventBridge(QtCore.QObject):
    """Worker-thread callbacks -> queued signal on the GUI thread."""

    event = QtCore.Signal(object)


class LabSessions:
    """The Mock and MuJoCo sessions, created on first use and shared by the Motion and Experiment tabs."""

    def __init__(self, emit: Callable[[dict[str, Any]], None], root: Path = LAB_ROOT, sim_port: int = SIM_PORT,
                 backend_factories: dict[str, Callable[[], Any]] | None = None,
                 robot_options: dict[str, Any] | None = None) -> None:
        self._emit, self.root, self.sim_port = emit, Path(root), sim_port
        self.viewer = True
        self._factories = backend_factories or {}
        #: tests: link_factory / r0_reader / clock / speaker_factory of RobotLabSession
        self.robot_options = dict(robot_options or {})
        from reachy_motion.sound import SoundSettings

        #: the effect-sound setting every session of this window reads (off unless the GUI's main() turns it on, so
        #: tests and offline runs never send a sound; reachy_motion.sound)
        self.sound = SoundSettings()
        self._s: dict[str, LabSession] = {}

    def robot(self, host: str, port: int, r0_root: Path | str = R0_ROOT) -> RobotLabSession:
        """The one robot session (one SDK client). A new target replaces an idle, disconnected session only."""
        s = self._s.get("robot")
        if s is not None and s.state != "closed" and (s.host, s.port) != (host, int(port)) and not s.connected                 and not s.pending:
            s.close()
            s = None
        if s is None or s.state == "closed":
            s = RobotLabSession(self.root, self._emit, host, int(port), r0_root=r0_root, **self.robot_options)
            s.sound = self.sound
            self._s["robot"] = s
        return s  # type: ignore[return-value]

    def get(self, mode: str) -> LabSession:
        if mode == "robot":
            s = self._s.get("robot")
            if s is None or s.state == "closed":
                raise RuntimeError("the robot session is created with LabSessions.robot(host, port)")
            return s
        s = self._s.get(mode)
        if s is None or s.state == "closed":
            s = LabSession(mode, self.root, on_event=self._emit, port=self.sim_port, viewer=self.viewer,
                           backend_factory=self._factories.get(mode))
            s.sound = self.sound
            self._s[mode] = s
        return s

    def existing(self, mode: str) -> LabSession | None:
        return self._s.get(mode)

    def busy(self) -> bool:
        return any(s.pending for s in self._s.values())

    def stop_all(self) -> None:
        for s in self._s.values():
            s.stop()

    def close(self, mode: str | None = None) -> None:
        for m in ([mode] if mode else list(self._s)):
            s = self._s.pop(m, None)
            if s is not None:
                s.close()


class FigureBox(QtWidgets.QWidget):
    def __init__(self, figure: Any = None, parent: QtWidgets.QWidget | None = None, toolbar: bool = True) -> None:
        super().__init__(parent)
        from reachy_motion.gui.plots import new_figure

        self.figure = figure if figure is not None else new_figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if toolbar:
            lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas, 1)

    def redraw(self) -> None:
        self.canvas.draw_idle()


def _item(text: Any, color: str | None = None, bold: bool = False) -> QtWidgets.QTableWidgetItem:
    it = QtWidgets.QTableWidgetItem("" if text is None else str(text))
    if color:
        it.setForeground(QtGui.QColor(color))
    if bold:
        f = it.font()
        f.setBold(True)
        it.setFont(f)
    return it


def _short_path(p: Path | str | None) -> str:
    """A path for a label: relative to the working folder when inside it (labels must not force the panel wider)."""
    if p is None:
        return "-"
    try:
        return str(Path(p).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)


def _note(text: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(NOTE_STYLE)
    return lab


def _refusal(c: ex.TrialCheck) -> str:
    """Why a trial is outside the initial policy (one line)."""
    if c.report is not None and c.report.violations:
        return "; ".join(ko.violation_text(v) for v in c.report.violations[:2])
    return "; ".join(c.notes) or "초기 정책 밖"


def _f(v: Any, fmt: str = ".3f") -> str:
    return format(v, fmt) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else "-"


def sound_text(rec: dict[str, Any] | None) -> tuple[str, bool] | None:
    """One line on what an effect sound did (``reachy_motion.sound``: the cue's record) and whether it went well."""
    if not rec:
        return None
    from reachy_motion.sound import OUTPUT_KO

    head = f"효과음 {rec.get('file')} ({OUTPUT_KO.get(rec.get('output'), rec.get('output'))})"
    if rec.get("ok") is False:
        return f"{head} 실패: {rec.get('error')}", False
    if not rec.get("fired"):
        return f"{head}: 소리 시각 전에 끝나 재생 안 함", True
    ms = rec.get("request_ms")
    what = "로봇에 재생 요청" if rec.get("output") == "robot" else "재생"
    return f"{head} {what}" + (f" · 요청 {ms:.0f} ms" if isinstance(ms, (int, float)) else ""), True


def _channel_value(channel: str, pose: dict[str, float], antennas: Any) -> float:
    if channel in ("roll", "pitch", "yaw"):
        return pose[f"{channel}_deg"]
    if channel in ("x", "y", "z"):
        return pose[f"{channel}_mm"]
    if channel == "antenna_right":
        return float(antennas[0]) * DEG
    if channel == "antenna_left":
        return float(antennas[1]) * DEG
    return float("nan")


# ================================================================================================ Experiment tab
class ExperimentTab(QtWidgets.QWidget):
    """While a trial runs, the GUI thread does as little as possible (it shares the GIL with the 50 Hz executor
    thread): it drains the live queue every 100 ms, updates a few labels and redraws only the live LINES (blitting on
    fixed axes, 4 Hz, and only while this tab is visible); everything heavier waits for the end of the run."""

    run_saved = QtCore.Signal(str)
    run_loaded = QtCore.Signal(object)
    show_results = QtCore.Signal(object)  # run folders to look at in the Results tab
    robot_mode_changed = QtCore.Signal(bool)  # "실제 로봇" chosen as the backend (the window shows its robot box)

    def __init__(self, sessions: LabSessions, library: Callable[[], Any], policy: Any,
                 parent: QtWidgets.QWidget | None = None, r0_root: Path | str = R0_ROOT) -> None:
        super().__init__(parent)
        self.r0_root = Path(r0_root)
        self.sessions = sessions
        self._build_error: str | None = None
        self._sim_state = "idle"
        self._finished: dict[tuple[str, str], list[str]] = {}  # (mode, stage) -> run folders of this session
        self._batch: dict[str, str | None] = {}  # "run all": label -> run folder (None until it finished)
        self._guide_key: Any = None  # last applied button state (re-applied only when it changes)
        self._table_labels: list[str] = []  # trial labels in table order (the selection follows the trial)
        self._previewing: str | None = None  # the Motion tab's MuJoCo preview on the shared worker
        self._library = library
        self.policy = policy
        self.checks: list[ex.TrialCheck] = []
        self.status: dict[str, str] = {}
        self.session_dirs: dict[tuple[str, str], Path] = {}
        self.halted_at: str | None = None  # label of the trial whose anomaly halted the ladder
        #: robot R3: (ladder, fixed value) -> (the step where tracking broke down, why). ("duration", A): shorter D of
        #: that amplitude are off; ("amplitude", D): larger A at that D are off
        self._r3_end: dict[tuple[str, float], tuple[float, list[str]]] = {}
        self._panel_hosted = False  # the robot box lives in the main window (above the tabs), not in this tab
        self.active: dict[str, Any] | None = None
        self._active_started = 0.0
        self._check_cache: dict[tuple[str, float, float], ex.TrialCheck] = {}
        self._buf = {k: deque(maxlen=3000) for k in ("t", "cmd", "meas")}
        self._t0_ns: int | None = None
        self._last_draw = 0.0
        self._build()
        self._poll = QtCore.QTimer(self)
        self._poll.timeout.connect(self._drain_live)
        self._poll.start(100)
        self._guide_timer = QtCore.QTimer(self)
        self._guide_timer.timeout.connect(self._update_guide)
        self._guide_timer.start(400)
        self.rebuild_trials()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        self.control_box = self._control_bar()

        mode_box = QtWidgets.QGroupBox("백엔드")
        mv = QtWidgets.QVBoxLayout(mode_box)
        mrow = QtWidgets.QHBoxLayout()
        self.rb_mock = QtWidgets.QRadioButton("Mock")  # not shown: the offline backend of the tests
        self.rb_sim = QtWidgets.QRadioButton("MuJoCo (simulated)")
        self.rb_real = QtWidgets.QRadioButton("실제 로봇 (measured)")
        self.rb_mock.setVisible(False)
        self.rb_sim.setChecked(True)
        self.rb_sim.setToolTip(ko.glossary_text("MuJoCo", "data_kind"))
        self.rb_real.setToolTip(LOCK_TEXT)
        self.sim_close = QtWidgets.QPushButton("MuJoCo 닫기")
        self.sim_close.setToolTip("로컬 MuJoCo daemon과 viewer 창을 닫습니다 (다음 실행 때 다시 시작)")
        self.sim_close.clicked.connect(self.on_sim_close)
        for w in (self.rb_mock, self.rb_sim, self.rb_real):
            mrow.addWidget(w)
        mrow.addStretch(1)
        mrow.addWidget(self.sim_close)
        mv.addLayout(mrow)
        self.robot_panel = RobotPanel()
        self.robot_panel.setVisible(False)
        mv.addWidget(self.robot_panel)
        self.robot_panel.status_clicked.connect(self.on_robot_status)
        self.robot_panel.connect_clicked.connect(self.on_robot_connect)
        self.robot_panel.wake_clicked.connect(self.on_robot_wake)
        self.robot_panel.torque_off_clicked.connect(self.on_robot_torque_off)
        self.robot_panel.disconnect_clicked.connect(self.on_robot_disconnect)
        self.robot_panel.target_changed.connect(self._robot_target_changed)
        lv.addWidget(mode_box)

        stage_box = QtWidgets.QGroupBox("실험")
        sv = QtWidgets.QVBoxLayout(stage_box)
        srow = QtWidgets.QHBoxLayout()
        self.stage_buttons: dict[str, QtWidgets.QRadioButton] = {}
        for st in ("R1", "R2", "R3"):  # small motions, the confirmed motions, the limit ladder
            b = QtWidgets.QRadioButton(ko.GLOSSARY[st][0].lstrip("①②③④⑤⑥⑦⑧⑨ "))
            b.setToolTip(ko.glossary_text(st))
            self.stage_buttons[st] = b
            srow.addWidget(b)
        self.stage_buttons["R1"].setChecked(True)  # the order of the buttons and of TEAM_GUIDE §4
        srow.addStretch(1)
        sv.addLayout(srow)
        self.stage_pages = QtWidgets.QStackedWidget()
        self.stage_pages.addWidget(self._r1_page())
        self.stage_pages.addWidget(self._r2_page())
        self.stage_pages.addWidget(self._r3_page())
        sv.addWidget(self.stage_pages)
        lv.addWidget(stage_box)

        plan_box = QtWidgets.QGroupBox("시행 목록 (안전 검사: computed)")
        pv = QtWidgets.QVBoxLayout(plan_box)
        self.table = QtWidgets.QTableWidget()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)  # the "#" column numbers the trials
        self.table.itemSelectionChanged.connect(self._update_guide)
        pv.addWidget(self.table, 1)
        self.notes_edit = QtWidgets.QLineEdit()
        self.notes_edit.setPlaceholderText("메모 (선택, 다음 시행 기록에 저장): 예) 떨림 있음")
        pv.addWidget(self.notes_edit)
        lv.addWidget(plan_box, 1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        split.addWidget(scroll)
        split.addWidget(self._live_panel())
        split.setSizes([700, 800])
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.control_box)
        lay.addWidget(split, 1)

        for b in self.stage_buttons.values():
            b.toggled.connect(lambda on: on and self._stage_changed())
        for w in (self.rb_mock, self.rb_sim, self.rb_real):
            w.toggled.connect(lambda on: on and self._mode_changed())
        self._stage_changed()

    def _r1_page(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.r1_axis = QtWidgets.QComboBox()
        for axis in ex.R1_AXES:
            self.r1_axis.addItem(ko.label(ko.AXES, axis), axis)
        self.r1_amp = QtWidgets.QComboBox()  # the ladder steps, or a typed value (2026-11-13: the dead-band curve)
        self.r1_amp.setEditable(True)
        self.r1_amp.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.r1_amp.lineEdit().setValidator(QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^\d{0,3}([.]\d{0,2})?$"), self.r1_amp))
        self.r1_unit = QtWidgets.QLabel("")
        amp_row = QtWidgets.QHBoxLayout()
        amp_row.addWidget(self.r1_amp, 1)
        amp_row.addWidget(self.r1_unit)
        self.r1_sign = QtWidgets.QComboBox()
        self.r1_sign.addItems(["+", "-"])
        self.r1_reps = QtWidgets.QSpinBox()
        self.r1_reps.setRange(1, 10)
        f.addRow("축", self.r1_axis)
        f.addRow("진폭", amp_row)
        f.addRow("방향", self.r1_sign)
        f.addRow("반복", self.r1_reps)
        f.addRow(_note(f"rest → 목표 {ex.R1_MOVE_S:g} s → 정지 {ex.R1_HOLD_S:g} s → 복귀 {ex.R1_MOVE_S:g} s (minjerk). "
                       "진폭은 목록에서 고르거나 직접 입력 (축별 범위 안, Enter). 기본값은 가장 큰 단계: 실제 로봇에서 2°는 거의 "
                       "움직이지 않았다 (진폭비 0.1–0.2, 2026-11-12 실측). 1·3·7° 같은 중간값으로 데드밴드 곡선을 그릴 수 있다"))
        self.r1_axis.currentIndexChanged.connect(lambda _i: self._r1_axis_changed(self.r1_axis.currentData()))
        self.r1_amp.currentIndexChanged.connect(lambda *_: self.rebuild_trials())  # a ladder step picked
        self.r1_amp.lineEdit().editingFinished.connect(self.rebuild_trials)  # a typed value (Enter / focus out)
        for sig in (self.r1_sign.currentTextChanged, self.r1_reps.valueChanged):
            sig.connect(lambda *_: self.rebuild_trials())
        self._r1_axis_changed(self.r1_axis.currentData())
        return w

    def _r1_axis_changed(self, axis: str) -> None:
        """The axis' amplitude ladder. The selected step stays when the axis changes (step 3 of yaw -> step 3 of z; a
        typed value -> the old ladder's nearest step, as the units differ); the first fill selects the largest step:
        on the robot 2 deg barely moved the head (measured 2026-11-12)."""
        step = -1
        if self.r1_amp.count():
            old = [float(self.r1_amp.itemText(i)) for i in range(self.r1_amp.count())]
            try:
                typed = float(self.r1_amp.currentText())
                step = min(range(len(old)), key=lambda i: abs(math.log(old[i] / typed))) if typed > 0 else -1
            except ValueError:
                step = self.r1_amp.currentIndex()
        self.r1_amp.blockSignals(True)
        self.r1_amp.clear()
        _, unit, ladder, _ = ex.R1_AXES[axis]
        self.r1_amp.addItems([f"{a:g}" for a in ladder])
        self.r1_amp.setCurrentIndex(step if 0 <= step < len(ladder) else len(ladder) - 1)
        lo, hi = ex.R1_MIN_AMPLITUDE, ex.R1_MAX_AMPLITUDE[axis]
        self.r1_unit.setText(f"{unit}  (직접 입력 {lo:g}–{hi:g})")
        self.r1_amp.setToolTip(f"단위: {unit}. 목록에서 고르거나 {lo:g}–{hi:g} {unit} 사이 값을 입력하고 Enter. "
                               "실제 로봇에서 2°는 거의 움직이지 않았다 (2026-11-12 실측)")
        self.r1_amp.blockSignals(False)
        self.rebuild_trials()

    def _r2_page(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.r2_motion = QtWidgets.QComboBox()
        for n in MOTION_NAMES:
            self.r2_motion.addItem(ko.label(ko.MOTIONS, n), n)
        self.r2_motion.addItem("4개 모두", None)
        self.r2_variant = QtWidgets.QComboBox()
        self.r2_variant.addItems(["둘 다 (expressive vs neutral)", "expressive", "neutral"])
        self.r2_reps = QtWidgets.QSpinBox()
        self.r2_reps.setRange(1, 10)
        self.r2_reps.setValue(3)
        f.addRow("모션", self.r2_motion)
        f.addRow("변형", self.r2_variant)
        f.addRow("반복", self.r2_reps)
        f.addRow(_note(f"모션 파일 그대로, 앞 {ex.PRE_S:g} s · 뒤 {ex.POST_S:g} s 정지를 함께 기록. Expressive의 효과음은 "
                       "① 모션 탭 '효과음' 설정을 따른다"))
        for sig in (self.r2_motion.currentTextChanged, self.r2_variant.currentTextChanged, self.r2_reps.valueChanged):
            sig.connect(lambda *_: self.rebuild_trials())
        return w

    def _r3_page(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.r3_vary = QtWidgets.QComboBox()
        self.r3_vary.addItems(["시간 D (A 고정)", "진폭 A (D 고정)"])
        self.r3_vary.setToolTip(ko.glossary_text("D / A"))
        self.r3_fixed = QtWidgets.QDoubleSpinBox()
        self.r3_fixed.setDecimals(2)
        self.r3_values = QtWidgets.QLineEdit(", ".join(f"{d:g}" for d in ex.R3_DEFAULT_DURATIONS))
        self.r3_reps = QtWidgets.QSpinBox()
        self.r3_reps.setRange(1, 10)
        f.addRow("바꿀 값 (하나만)", self.r3_vary)
        f.addRow("고정값", self.r3_fixed)
        f.addRow("사다리 값", self.r3_values)
        f.addRow("반복", self.r3_reps)
        f.addRow(_note(f"pitch 끄덕임: rest → D 동안 +A → {ex.R3_HOLD_S:g} s 정지 → D 동안 복귀 (minjerk). "
                       "한 단계씩, 첫 이상에서 멈춤. 실제 로봇: D 사다리는 D ≥ 0.2 s, 진폭 사다리는 D ≥ 1 s에서 작은 A부터"))
        self.r3_vary.currentIndexChanged.connect(self._r3_vary_changed)
        for sig in (self.r3_fixed.valueChanged, self.r3_values.editingFinished, self.r3_reps.valueChanged):
            sig.connect(lambda *_: self.rebuild_trials())
        self._r3_vary_changed(0)
        return w

    def _r3_vary_changed(self, i: int) -> None:
        for wdg in (self.r3_fixed, self.r3_values):
            wdg.blockSignals(True)
        if i == 0:
            self.r3_fixed.setRange(1.0, 40.0)
            self.r3_fixed.setSuffix(" deg (A)")
            self.r3_fixed.setValue(ex.R3_DEFAULT_AMPLITUDE_DEG)
            self.r3_values.setText(", ".join(f"{d:g}" for d in ex.R3_DEFAULT_DURATIONS))
            self.r3_values.setToolTip("D = 편도 이동 시간 (s)")
        else:
            self.r3_fixed.setRange(0.1, 5.0)
            self.r3_fixed.setSuffix(" s (D)")
            self.r3_fixed.setValue(ex.R3_DEFAULT_FIXED_DURATION_S)
            self.r3_values.setText(", ".join(f"{a:g}" for a in ex.R3_DEFAULT_AMPLITUDES))
            self.r3_values.setToolTip("A = pitch 진폭 (deg)")
        for wdg in (self.r3_fixed, self.r3_values):
            wdg.blockSignals(False)
        self.rebuild_trials()

    def _live_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        fig, self.live = lab_plots.live_figure()
        self.live_box = FigureBox(fig, toolbar=False)
        self._bg: Any = None  # blit background of the live axes (valid for the current canvas size)
        self.live_box.canvas.mpl_connect("draw_event", self._cache_live_background)
        v.addWidget(self.live_box, 3)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        v.addWidget(self.log_view, 1)
        return w

    # ------------------------------------------------------------------ setup -> trials
    def mode(self) -> str:
        return "robot" if self.rb_real.isChecked() else "sim" if self.rb_sim.isChecked() else "mock"

    def _mode_changed(self) -> None:
        robot = self.mode() == "robot"
        if not self._panel_hosted:
            self.robot_panel.setVisible(robot)
        self.sim_close.setVisible(not robot)  # the robot has no automatic return: Wake-up or torque-off
        if robot:
            self.robot_panel.refresh_buttons()
        self.robot_mode_changed.emit(robot)
        self.rebuild_trials()

    def release_robot_panel(self) -> RobotPanel:
        """Hand the robot box to the main window, which shows it above the tabs (2026-11-13: connect, Wake-up and
        torque-off from every tab). The handlers stay here; only where the box is drawn changes."""
        if not self._panel_hosted:
            self._panel_hosted = True
            parent = self.robot_panel.parentWidget()
            if parent is not None and parent.layout() is not None:
                parent.layout().removeWidget(self.robot_panel)
            self.robot_panel.setParent(None)
        return self.robot_panel

    def stage(self) -> str:
        return next((s for s, b in self.stage_buttons.items() if b.isChecked()), "R1")

    def _stage_changed(self) -> None:
        idx = {"R1": 0, "R2": 1, "R3": 2}.get(self.stage(), 2)
        self.stage_pages.setCurrentIndex(idx)
        self.rebuild_trials()

    def planned_trials(self) -> list[ex.Trial]:
        st = self.stage()
        if st == "R1":
            sign = 1 if self.r1_sign.currentIndex() == 0 else -1
            axis, text = self.r1_axis.currentData(), self.r1_amp.currentText().strip()
            try:
                amp = float(text)
            except ValueError:
                raise ValueError(f"진폭 '{text}'을(를) 읽을 수 없습니다. 숫자를 넣으세요 (예: 3)") from None
            why = ex.r1_amplitude_problem(axis, amp)
            if why:
                raise ValueError(why)
            return [ex.r1_trial(axis, amp, sign, rep) for rep in range(1, self.r1_reps.value() + 1)]
        if st == "R2":
            lib = self._library()
            motions = list(MOTION_NAMES) if self.r2_motion.currentIndex() == len(MOTION_NAMES) else [
                self.r2_motion.currentData()]
            vi = self.r2_variant.currentIndex()
            variants = ("expressive", "neutral") if vi == 0 else (("expressive",) if vi == 1 else ("neutral",))
            return [ex.r2_trial(lib.get(m, v), rep)
                    for m in motions for v in variants for rep in range(1, self.r2_reps.value() + 1)]
        vary = "duration" if self.r3_vary.currentIndex() == 0 else "amplitude"
        values = ex.parse_values(self.r3_values.text())
        return [t for v in values for rep in range(1, self.r3_reps.value() + 1)
                for t in ex.r3_ladder(vary, [v], self.r3_fixed.value(), rep)]

    def _check(self, trial: ex.Trial) -> ex.TrialCheck:
        key = (trial.spec.digest(), trial.pre_s, trial.post_s)
        c = self._check_cache.get(key)
        if c is None:
            c = ex.check_trial(trial, self.policy)
            self._check_cache[key] = c
        return replace(c, trial=trial, real_scope=ex.real_scope(trial, c.runnable))

    def rebuild_trials(self) -> None:
        if not hasattr(self, "table"):
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            try:
                trials = self.planned_trials()
                self._build_error = None
            except ValueError as exc:
                self.checks = []
                self._build_error = (f"사다리 값 '{self.r3_values.text()}'을(를) 읽을 수 없습니다 ({exc}). 쉼표로 구분한 "
                                     "양수를 넣으세요 (예: 1.0, 0.6, 0.4)" if self.stage() == "R3" else str(exc))
                self._fill_table()
                self._update_guide()
                return
            except (KeyError, FileNotFoundError) as exc:
                self.checks = []
                self._build_error = f"모션 파일을 찾을 수 없습니다: {exc}"
                self._fill_table()
                self._update_guide()
                return
            self.checks = [self._check(t) for t in trials]
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._fill_table()
        if self.checks and not self.table.selectionModel().selectedRows():
            self.table.selectRow(self._next_row())
        self._update_guide()

    def _next_row(self) -> int:
        for i, c in enumerate(self.checks):
            if self.status.get(self._key(c.trial)) not in ("done", "anomaly"):
                return i
        return max(0, len(self.checks) - 1)

    def _key(self, trial: ex.Trial) -> str:
        return f"{self.mode()}:{trial.label}"

    def _fill_table(self) -> None:
        t = self.table
        sel = t.currentRow()
        prev = self._table_labels[sel] if 0 <= sel < len(self._table_labels) else None
        t.clear()
        t.setColumnCount(len(TABLE_COLS))
        t.setHorizontalHeaderLabels(list(TABLE_COLS))
        t.setRowCount(len(self.checks))
        for i, c in enumerate(self.checks):
            r = c.row()
            st = self.status.get(self._key(c.trial), "refused" if not c.runnable else "pending")
            block = self._r3_block(c.trial)
            if block and st == "pending":
                st = "refused"  # display only: the robot ladder ended at a longer step
            vals = [str(i + 1), c.trial.title, _f(r["stewart_speed_ratio"], ".2f"),
                    f"{_f(r['min_margin_deg'], '.1f')} ({r['min_margin_joint'] or '-'})", r["policy"],
                    ko.label(ko.STATUS, st)]
            for j, v in enumerate(vals):
                color = STATUS_COLOR.get(st) if j == STATUS_COL else (
                    "#c0392b" if j == POLICY_COL and not c.runnable else None)
                t.setItem(i, j, _item(v, color, bold=j == STATUS_COL))
            if not c.runnable:
                t.item(i, POLICY_COL).setToolTip(_refusal(c))
            if block:
                t.item(i, STATUS_COL).setToolTip(block)
        for j, term in ((2, "Speed Ratio"), (3, "Joint Margin"), (POLICY_COL, "Initial Policy")):
            t.horizontalHeaderItem(j).setToolTip(ko.glossary_text(term))
        t.resizeColumnsToContents()
        self._table_labels = [c.trial.label for c in self.checks]
        if prev in self._table_labels:  # same trial list: keep the user's choice
            t.selectRow(self._table_labels.index(prev))
        elif self.checks:  # new conditions: the first trial not run yet (R3: the safest step first)
            t.selectRow(self._next_row())

    def _selected(self) -> ex.TrialCheck | None:
        rows = self.table.selectionModel().selectedRows()
        return self.checks[rows[0].row()] if rows and rows[0].row() < len(self.checks) else None

    # ------------------------------------------------------------------ buttons and the status line
    def _control_bar(self) -> QtWidgets.QWidget:
        """Above the tab: the run buttons in one row and one status line under them."""
        box = QtWidgets.QFrame()
        box.setObjectName("controls")
        box.setStyleSheet("QFrame#controls { background:#f4f6fa; border:1px solid #d6dbe4; border-radius:6px }")
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(8, 6, 8, 6)
        btns = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("▶ 실행")
        self.run_all_btn = QtWidgets.QPushButton("▶▶ 모두 실행")
        self.stop_btn = QtWidgets.QPushButton("■ 정지")
        self.results_btn = QtWidgets.QPushButton("결과 보기 →")
        self.unlock_btn = QtWidgets.QPushButton("중단 해제")
        self.play_btn.setStyleSheet("font-weight:bold; padding:6px 14px")
        self.run_all_btn.setToolTip("표에서 아직 실행하지 않은 시행을 차례로 실행합니다 (R1 · R2, 실제 로봇 제외).")
        self.stop_btn.setToolTip(ko.glossary_text("S1/S2"))
        self.results_btn.setToolTip("이번에 끝난 시행을 결과 탭에서 봅니다.")
        self.unlock_btn.setToolTip("이상으로 멈춘 R3 사다리를, 결과를 확인한 뒤 다시 엽니다.")
        self.unlock_btn.setVisible(False)
        for b, slot in ((self.play_btn, self.on_play), (self.run_all_btn, self.on_run_all), (self.stop_btn, self.on_stop),
                        (self.results_btn, self.on_results), (self.unlock_btn, self.on_unlock)):
            b.clicked.connect(slot)
            btns.addWidget(b)
        btns.addStretch(1)
        v.addLayout(btns)
        self.status_line = QtWidgets.QLabel("")
        self.status_line.setWordWrap(True)
        self.status_line.setStyleSheet("border:none")
        v.addWidget(self.status_line)
        return box

    def _guide_state(self) -> gd.GuideState:
        c = self._selected()
        refused = [f"{x.trial.title}: " + ("; ".join(ko.violation_text(v) for v in x.report.violations[:1])
                                           if x.report is not None and x.report.violations else "; ".join(x.notes))
                   for x in self.checks if not x.runnable]
        reason = ""
        block = None if c is None else self._r3_block(c.trial)
        if c is not None and not c.runnable:
            reason = ("; ".join(ko.violation_text(v) for v in c.report.violations[:2])
                      if c.report is not None and c.report.violations else "; ".join(c.notes)) or "초기 정책 밖"
        elif block:
            reason = block
        s = self.sessions.existing("sim")
        sim_state = self._sim_state if s is not None else "idle"
        if s is not None and s.state == "ready":
            sim_state = "ready"
        return gd.GuideState(
            stage=self.stage(), mode=self.mode(), build_error=self._build_error, n_trials=len(self.checks),
            n_runnable=sum(1 for x in self.checks if x.runnable), refused=refused,
            selected=None if c is None else c.trial.title,
            selected_runnable=bool(c is not None and c.runnable and not block),
            selected_reason=reason, selected_done=bool(c is not None and self.status.get(self._key(c.trial)) == "done"),
            sim_state=sim_state, busy=self.sessions.busy(),
            running=None if self.active is None else self.active.get("title"), halted_at=self.halted_at,
            finished_any=bool(self._finished.get((self.mode(), self.stage()))), previewing=self._previewing,
            robot_state=self.robot_panel.state if self.mode() == "robot" else None)

    def _pending_runnable(self) -> list[ex.TrialCheck]:
        return [c for c in self.checks if c.runnable and self.status.get(self._key(c.trial)) not in ("done", "queued",
                                                                                                    "running")]

    def _update_guide(self) -> None:
        """The status line (every 400 ms and after each change) and the buttons (only when they change)."""
        if not hasattr(self, "table"):
            return
        st = self._guide_state()
        g = gd.guide(st, len(self._pending_runnable()))
        mode = self.mode()
        if mode == "robot":
            st_r = self.robot_panel.state
            where = f"실제 로봇 {self.robot_panel.host()}: {STATE_SHORT.get(st_r, st_r)}"
            self.robot_panel.set_busy(st.busy)
        elif mode == "mock":
            where = "Mock"
        else:
            where = {"ready": "MuJoCo 준비됨", "connecting": "MuJoCo 시작 중 (약 10초) ...",
                     "error": "MuJoCo 오류 (아래 기록)", "closed": "MuJoCo 닫힘 (실행하면 다시 시작)"}.get(
                st.sim_state, "MuJoCo 시작 안 함 (실행하면 자동 시작, 약 10초)")
        if st.busy and self.active is not None:
            el = time.perf_counter() - self._active_started
            now = f"실행 중: {self.active.get('title')} {_f(el, '.1f')} / {_f(self.active.get('duration_s'), '.1f')} s"
        elif st.busy:
            now = "준비 중 ..."
        elif not g.play_enabled and g.play_reason:
            now = g.play_reason
        else:
            now = f"선택: {st.selected}" if st.selected else "대기"
        text = f"{where}  ·  {now}"
        if text != self.status_line.text():
            self.status_line.setText(text)
        key = (g.play_enabled, g.play_reason, g.run_all_enabled, g.results_enabled, self.halted_at, st.stage, mode)
        if key == self._guide_key:
            return  # nothing else changed: no restyling while the executor runs
        self._guide_key = key
        # where the click runs, on the button itself: the robot box is visible from every tab now (2026-11-13)
        self.play_btn.setText({"robot": "▶ 실행 (실제 로봇)", "mock": "▶ 실행 (Mock)"}.get(mode, "▶ 실행 (MuJoCo)"))
        self.play_btn.setEnabled(g.play_enabled)
        self.play_btn.setToolTip(g.play_reason or "표에서 고른 시행 하나를 실행합니다.")
        self.run_all_btn.setVisible(st.stage != "R3" and mode != "robot")
        self.run_all_btn.setEnabled(g.run_all_enabled)
        self.results_btn.setEnabled(g.results_enabled)
        self.unlock_btn.setVisible(self.halted_at is not None)
        self.unlock_btn.setEnabled(self.halted_at is not None)

    def on_run_all(self) -> None:
        """R1 / R2 only: queue every runnable trial of the table that did not run yet (one worker runs them in order)."""
        if self.stage() == "R3":
            self.log("R3는 한 단계씩만 실행합니다")
            return
        if self.mode() == "robot":
            self.log("실제 로봇은 시행을 하나씩 실행합니다: '모두 실행'은 없습니다")
            return
        todo = self._pending_runnable()
        if not todo:
            self.log("실행할 시행이 없습니다")
            return
        if self.sessions.busy():
            self.log("작업이 아직 실행 중입니다. 정지하거나 기다리세요")
            return
        mode = self.mode()
        session = self.sessions.get(mode)
        folder = self._session_dir()
        notes = self._notes()
        self._batch = {c.trial.label: None for c in todo}
        for c in todo:
            self.status[self._key(c.trial)] = "queued"
            session.run_trial(c.trial, folder, notes)
        self._fill_table()
        self.log(f"{len(todo)}개 시행을 차례로 실행합니다 ({mode})")
        self._update_guide()

    def _batch_mark(self, label: str | None, run_dir: str) -> None:
        """One trial of a "run all" batch is over ('' = no usable run)."""
        if not label or label not in self._batch:
            return
        self._batch[label] = run_dir
        if all(v is not None for v in self._batch.values()):
            self._batch = {}

    def on_results(self) -> None:
        runs = self._finished.get((self.mode(), self.stage())) or [d for v in self._finished.values() for d in v]
        if runs:
            self.show_results.emit(list(runs))

    # ------------------------------------------------------------------ actions
    def on_sim_close(self) -> None:
        if self.sessions.existing("sim") is None:
            return
        self.status_line.setText("MuJoCo 닫는 중 ...")
        QtWidgets.QApplication.processEvents()
        self.sessions.close("sim")
        self._sim_state = "closed"
        self._update_guide()

    def _session_dir(self, mode: str | None = None, stage: str | None = None) -> Path:
        key = (mode or self.mode(), stage or self.stage())
        d = self.session_dirs.get(key)
        if d is None:
            d = new_session_dir(key[0], key[1], self.sessions.root)
            self.session_dirs[key] = d
            self.log(f"세션 폴더: {_short_path(d)}")
        return d

    def _notes(self) -> str:
        return self.notes_edit.text().strip()

    def on_play(self) -> None:
        c = self._selected()
        if c is None:
            self.log("먼저 시행을 고르세요")
            return
        if not c.runnable:
            self.log(f"거부: {c.trial.title}은(는) 초기 정책 밖이라 어디서도 실행하지 않습니다")
            return
        if self.halted_at is not None and self.stage() == "R3":
            self.log(f"{self.halted_at}의 이상으로 중단됨: 검토한 뒤 '중단 해제'를 누르세요")
            return
        if self.sessions.busy():
            self.log("작업이 아직 실행 중이거나 대기 중입니다. 정지하거나 기다리세요 (한 번에 한 시행)")
            return
        mode = self.mode()
        if mode == "robot":
            why = self._r3_block(c.trial) or self._robot_play(c)
            if why:
                self.robot_panel.show_message(why)  # also in the log: never only there
            return
        self.status[self._key(c.trial)] = "queued"
        self.sessions.get(mode).run_trial(c.trial, self._session_dir(), self._notes())
        self._fill_table()
        self.log(f"대기열에 넣음: {c.trial.title} ({mode})")

    def on_stop(self) -> None:
        self.sessions.stop_all()
        self.log("정지 요청 (다음 tick에서 제어 정지 S1)")

    def on_unlock(self) -> None:
        self.log(f"사용자가 중단을 해제함 ({self.halted_at}의 이상 검토 후)")
        self.halted_at = None
        self.unlock_btn.setEnabled(False)
        self._update_guide()

    def log(self, text: str) -> None:
        self.log_view.appendPlainText(f"[{datetime.now():%H:%M:%S}] {text}")

    # ------------------------------------------------------------------ real robot (gui.robot_panel / robot_session)
    def robot_session(self) -> RobotLabSession:
        rp = self.robot_panel
        return self.sessions.robot(rp.host(), rp.port(), self.r0_root)

    def _robot_target_changed(self, host: str, port: int) -> None:
        self.robot_panel.last_r0 = None
        self.robot_panel.refresh_buttons()

    def on_robot_status(self) -> None:
        self.robot_session().read_status()

    def on_robot_connect(self) -> None:
        s = self.robot_session()
        s.connect_robot()  # the click is the approval: the session reads the robot (GET) and connects
        self.log(f"실제 로봇 연결: {s.host}:{s.port}")

    def on_robot_wake(self) -> None:
        s = self.sessions.existing("robot")
        if s is None or not s.connected:
            self.log("Wake-up: 먼저 '연결'")
            return
        s.wake_up()
        self.log("실제 로봇 Wake-up")

    def on_robot_torque_off(self) -> None:
        s = self.sessions.existing("robot")
        if s is None or not s.connected:
            self.log("연결되어 있지 않습니다: 머리를 받치고 대시보드나 전원 스위치로 토크를 끄세요")
            return
        if not ask_torque_off(self):
            return
        s.torque_off(True)
        self.log("토크 OFF")

    def on_robot_disconnect(self) -> None:
        s = self.sessions.existing("robot")
        if s is None or not s.connected:
            return
        if self.robot_panel.state == "holding":
            ans = QtWidgets.QMessageBox.question(
                self, "연결 끊기", "로봇이 정지한 자세에서 토크가 켜진 채입니다. 연결을 끊어도 토크는 꺼지지 않습니다. 끊을까요?")
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        s.disconnect()

    def _robot_play(self, c: ex.TrialCheck) -> str | None:
        """One trial on the robot: the click is the approval (one trial, the timeline the table checked). Returns why
        it did not start (None = started)."""
        from reachy_motion.safety import gui_approval as ga

        s = self.sessions.existing("robot")
        if s is None or not s.connected or self.robot_panel.state != "at_init":
            why = "실제 로봇 시행: 먼저 '연결'(필요하면 'Wake-up')으로 INIT 대기 상태로"
            self.log(why)
            return why
        plan, probs = s.trial_plan(c.trial)
        req = None if plan is None else ga.trial_request(plan, s.host, s.port, probs)
        if req is None or not req.policy_ok:
            why = f"실행할 수 없음 {c.trial.title}: " + "; ".join(probs if req is None else req.problems)
            self.log(why)
            return why
        self.status[f"robot:{c.trial.label}"] = "queued"
        s.run_trial(c.trial, self._session_dir("robot", c.trial.stage), self._notes(), approval=ga.approve(req))
        self._fill_table()
        self.log(f"실제 로봇 시행: {c.trial.title}")
        return None

    @staticmethod
    def _r3_end_text(vary: str, x: float, why: list[str]) -> str:
        if vary == "amplitude":
            return f"R3 진폭 사다리 끝 — A {x:g}°에서 추종이 무너짐: " + "; ".join(why)
        return f"R3 사다리 끝 — D {x:g} s에서 추종이 무너짐: " + "; ".join(why)

    def _r3_block(self, trial: ex.Trial) -> str | None:
        """Why a robot R3 step cannot run because its ladder already broke down at an earlier step (a longer D, or a
        smaller A on the amplitude ladder). None: it may."""
        if self.mode() != "robot" or trial.stage != "R3":
            return None
        c = trial.condition
        a, d = float(c.get("amplitude_deg", 0.0)), float(c.get("duration_s", 0.0))
        if c.get("vary") == "amplitude":
            end = self._r3_end.get(("amplitude", d))
            beyond = end is not None and a > end[0] + 1e-9
        else:
            end = self._r3_end.get(("duration", a))
            beyond = end is not None and d < end[0] - 1e-9
        return self._r3_end_text(str(c.get("vary") or "duration"), end[0], end[1]) if beyond and end else None

    def play_motion_on_robot(self, name: str, variant: str) -> str | None:
        """The Motion tab's '▶ 실제 로봇 재생': one R2 trial of the motion file - the same check, click approval, record and
        Results entry as an R2 row run on the robot here. Returns why it did not start (None = started)."""
        if self.sessions.busy():
            return "다른 작업이 실행 중입니다. 끝나거나 정지한 뒤 다시 누르세요"
        try:
            spec = self._library().get(name, variant)
        except (KeyError, FileNotFoundError) as exc:
            return f"모션 파일을 찾을 수 없습니다: {exc}"
        c = self._check(ex.r2_trial(spec, 1))
        if not c.runnable:
            return f"초기 정책 밖이라 실행하지 않습니다: {_refusal(c)}"
        return self._robot_play(c)

    def _robot_event(self, e: dict[str, Any]) -> bool:
        """Robot-only events (True = handled); the trial events go through the common path."""
        kind, rp = e.get("kind"), self.robot_panel
        if kind == "robot_status":
            rp.show_status(e)
            if e.get("r0_error"):
                self.log(f"실제 로봇 R0 실패: {e['r0_error']}")
            return True
        if kind == "robot_state":
            rp.set_state(e.get("state", "disconnected"), e, e.get("operator_procedure") or "")
            if e.get("state") == "disconnected":
                rp.last_socket = None
                rp.refresh_buttons()
            self._update_guide()
            return True
        if kind == "connecting":
            rp.show_message(f"{e.get('host')}:{e.get('port')}에 연결하는 중 ...", error=False)
            return True
        if kind == "robot_connect_failed":  # something was (or may have been) sent: never "nothing sent"
            text = f"연결 실패 - {e.get('outcome_ko')} · 토크 {e.get('torque')} · " + "; ".join(e.get("reasons") or [])
            rp.show_message(text)
            self.log(f"실제 로봇: {text}")
            s = self.sessions.existing("robot")
            if s is not None:
                s.read_status()  # the motor mode again (GET only): ON / OFF instead of UNKNOWN if the daemon answers
            return True
        if kind == "trial_cancelled":
            label = e.get("label")
            if self.active is not None and self.active.get("label") == label:
                self.active = None
            self._set_status(label, "cancelled", "robot")
            rp.show_message(f"실행 취소: {e.get('reason')}", error=False)
            self.log(f"실제 로봇 실행 취소 {label}: {e.get('reason')} · 기록 {e.get('record')}")
            self._update_guide()
            return True
        if kind == "robot_connected":
            self._r3_end.clear()  # a new connection starts the R3 ladder again at D 1.0 s
            self._fill_table()
        if kind in ("robot_connected", "wake_up_finished", "robot_torque_off", "robot_disconnected"):
            text = {"robot_connected": "연결됨" + (" · 다음: Wake-up" if e.get("start_mode") == "sleep" else ""),
                    "robot_torque_off": f"토크 OFF: {e.get('torque')} · 연결 닫힘",
                    "robot_disconnected": "연결 끊음 (토크는 그대로)"}.get(kind)
            if kind == "wake_up_finished":
                text = "Wake-up 완료: INIT" if e.get("ok") else "Wake-up 이상: " + "; ".join(e.get("problems") or [])
            rp.show_message(str(text), error=kind == "wake_up_finished" and not e.get("ok"))
            self.log(f"실제 로봇: {text}")
            s = self.sessions.existing("robot")
            if s is not None and s.connected:
                s.read_status()  # what the robot reports now (GET + socket, nothing sent)
            return True
        if kind == "wake_up_started":
            self.log(f"실제 로봇 Wake-up 시작 ({_f(e.get('duration_s'), '.1f')} s)")
            return True
        if kind == "sound":  # the effect sound could not be prepared: the motion runs without it
            text = sound_text(e)
            if text is not None:
                rp.show_message(text[0], error=not text[1])
                self.log(text[0])
            return True
        if kind == "error":  # also logged by the common path; the box shows it next to the robot's state
            rp.show_message(f"오류 ({e.get('job')}): {e.get('error')}")
            return False
        if kind == "refused" and e.get("job") == "trial":  # the common path marks the row and logs it too
            rp.show_message("실행 안 함 (보낸 것 없음): " + "; ".join(e.get("reasons") or []))
            return False
        if kind == "refused" and e.get("job") != "trial":  # refused before anything was sent (GET at most)
            reasons = "; ".join(e.get("reasons") or [])
            rp.show_message(f"실행 안 함 (보낸 것 없음): {reasons}")
            self.log(f"실제 로봇 거부 ({e.get('job')}): {reasons}")
            return True
        if kind == "trial_finished" and e.get("robot"):
            r = e["robot"]
            ev = r.get("evidence") or {}
            self.log(f"실제 로봇 시행 {e.get('label')}: 명령 {ev.get('commands_sent')}개 전송 / 측정 {ev.get('present_samples')}개 "
                     f"수신 ({ev.get('data_kind_present')}), 주 채널 측정/명령 진폭비 "
                     f"{_f(ev.get('amplitude_ratio_measured_over_commanded'), '.2f')} · 상태 {r.get('state')}"
                     + (" · " + "; ".join(r.get("problems") or []) if r.get("problems") else ""))
            ratio = ev.get("amplitude_ratio_measured_over_commanded")
            snd = sound_text(e.get("sound"))
            if snd is not None:
                self.log(snd[0])
            if isinstance(ratio, (int, float)) and ratio == ratio:  # how much of the command the head really did
                small = ratio < 0.5
                rp.show_message(f"시행 끝 · 진폭비 {ratio:.2f}" + (" - 명령보다 훨씬 적게 움직였습니다: 진폭을 키워 보세요"
                                                              if small else "") + ("" if snd is None else f" · {snd[0]}"),
                                error=small or (snd is not None and not snd[1]))
            nxt = e.get("r3_next")
            if nxt is not None:  # R3: may the ladder go on? (robot_r1.r3_next_step_problems, computed by the worker)
                a, d, why = float(nxt["amplitude_deg"]), float(nxt["duration_s"]), list(nxt.get("problems") or [])
                vary = "amplitude" if nxt.get("vary") == "amplitude" else "duration"
                key, x = (("amplitude", d), a) if vary == "amplitude" else (("duration", a), d)
                step = f"A {a:g}°" if vary == "amplitude" else f"D {d:g} s"
                if why:
                    self._r3_end[key] = (x, why)
                    rp.show_message(self._r3_end_text(vary, x, why))
                    self.log(f"{self._r3_end_text(vary, x, why)} ({'D' if vary == 'amplitude' else 'A'} "
                             f"{d if vary == 'amplitude' else a:g} 고정)")
                else:
                    self._r3_end.pop(key, None)
                    rp.show_message(f"시행 끝 · R3 {step} 기준 안 · 다음 단계 가능", error=False)
        return False

    # ------------------------------------------------------------------ events (GUI thread)
    def on_lab_event(self, e: dict[str, Any]) -> None:
        kind, mode = e.get("kind"), e.get("mode")
        if mode == "sim" and kind in ("connecting", "ready", "closed", "error"):
            if kind != "error" or self._sim_state == "connecting":
                self._sim_state = {"connecting": "connecting", "ready": "ready", "closed": "closed"}.get(kind, "error")
            if kind == "ready":
                self.log(f"MuJoCo 준비됨 (127.0.0.1:{e.get('port')}, simulated)")
            self._update_guide()
        if kind in ("preview_started", "preview_finished") or (mode == "sim" and kind in ("closed", "error")):
            self._previewing = e.get("key") if kind == "preview_started" else None
            self._update_guide()
            if kind.startswith("preview"):
                return  # the Motion tab's preview
        if mode == "robot" and self._robot_event(e):
            return
        if kind == "status":
            self.log(f"{mode}: {e.get('text')}")
        elif kind == "trial_started":
            self.active = e
            self._active_started = time.perf_counter()
            self._t0_ns = None
            for b in self._buf.values():
                b.clear()
            self._set_cell_status(e["label"], "running", mode)  # one cell: no table rebuild while it runs
            self.live["title"].set_text(f"{e.get('title')}  ·  주 채널 {e.get('main_channel')}  ·  {e.get('data_kind')}")
            self._prepare_live_axes(e)
            self.log(f"시작: {e.get('title')} (기록 {e.get('duration_s', 0):.1f} s, 표본 {e.get('n_samples')}개)")
        elif kind == "trial_finished":
            self._finish(e)
        elif kind == "refused":
            if self.active is not None and self.active.get("label") == e.get("label"):
                self.active = None  # refused after it was announced (robot: nothing was sent)
            self._set_status(e.get("label"), "refused", mode)
            self.log(f"거부 {e.get('label')}: {e.get('reasons')}")
            self._batch_mark(e.get("label"), "")
        elif kind == "error":
            self.log(f"{mode} 오류 ({e.get('job')}): {e.get('error')}")
            if self.active is not None and e.get("job") == "trial":
                self._halt(self.active["label"], [f"오류: {e.get('error')}"], mode)
                self.active = None
            elif e.get("job") == "trial":
                if e.get("label"):
                    self.status[f"{mode}:{e['label']}"] = "error"
                else:
                    for k, v in list(self.status.items()):
                        if v == "queued" and k.startswith(f"{mode}:"):
                            self.status[k] = "error"
                self._fill_table()
            if e.get("job") == "trial":
                self._batch_mark(e.get("label"), "")
        elif kind == "reset_done":
            self.log(f"{mode}: rest로 돌아옴" if e.get("completed") else f"{mode}: 복귀 중단됨")
        elif kind == "sound":  # MuJoCo / Mock: the laptop sound could not be prepared (the robot's is handled above)
            snd = sound_text(e)
            if snd is not None:
                self.log(snd[0])
        elif kind == "skipped":
            self.log(f"{mode}: 대기 중이던 {e.get('job')} 취소됨" + (f" ({e['label']})" if e.get("label") else ""))
            if e.get("label"):  # back to "not run": it can be run again
                self.status.pop(f"{mode}:{e['label']}", None)
                self._fill_table()
                self._batch_mark(e["label"], "")

    def _set_status(self, label: str | None, status: str, mode: str | None = None) -> None:
        if label is None:
            return
        self.status[f"{mode or self.mode()}:{label}"] = status
        self._fill_table()

    def _set_cell_status(self, label: str, status: str, mode: str | None = None) -> None:
        self.status[f"{mode or self.mode()}:{label}"] = status
        for i, c in enumerate(self.checks):
            if c.trial.label == label:
                self.table.setItem(i, STATUS_COL, _item(ko.label(ko.STATUS, status), STATUS_COLOR.get(status),
                                                        bold=True))

    def _anomalies(self, e: dict[str, Any]) -> list[str]:
        out = []
        if e.get("error"):
            out.append(f"실행 오류: {e['error']}")
        if e.get("stop_kind") in ("S2", "S3"):
            out.append(f"즉시 정지 {e['stop_kind']}: {e.get('stop_reason')}")
        if not e.get("completed") and not e.get("cancelled") and not out:
            out.append("시행이 끝나지 않음")
        m = e.get("metrics") or {}
        if "analysis_error" in m:
            out.append(f"분석 실패: {m['analysis_error']}")
        return out

    def _halt(self, label: str, reasons: list[str], mode: str | None = None) -> None:
        self.status[f"{mode or self.mode()}:{label}"] = "anomaly"
        self.halted_at = label
        self.unlock_btn.setEnabled(True)
        self.log(f"이상 발생 ({label}): " + "; ".join(reasons) + " -> 중단: '중단 해제' 전에는 다음 단계를 실행하지 않음")
        self._fill_table()

    def _finish(self, e: dict[str, Any]) -> None:
        label = e.get("label")
        self.active = None
        t = e.get("ticks") or {}
        if t:
            self.log(f"실행기 주기 (pc_measured): tick {t.get('n_ticks')}개, {_f(t.get('effective_hz'), '.2f')} Hz, "
                     f"p95 {_f(t.get('interval_p95_ms'), '.1f')} ms, 최대 {_f(t.get('interval_max_ms'), '.1f')} ms")
        snd = sound_text(e.get("sound")) if e.get("mode") != "robot" else None  # the robot's is logged above
        if snd is not None:
            self.log(snd[0])
        problems = self._anomalies(e)
        mode = e.get("mode")
        if e.get("cancelled") and not problems:
            self._set_status(label, "stopped", mode)
            self.log(f"사용자가 정지함: {label} (저장: {e.get('run_dir')})")
        elif problems:
            self._halt(label, problems, mode)
        else:
            self._set_status(label, "done", mode)
            self.log(f"완료 {label}: " + _brief(e.get("metrics") or {}))
        if e.get("loaded") is not None:
            self.run_loaded.emit(e["loaded"])
        elif e.get("run_dir"):
            self.run_saved.emit(str(e["run_dir"]))
        if e.get("run_dir"):
            self._finished.setdefault((mode or self.mode(), e.get("stage") or self.stage()), []).append(
                str(e["run_dir"]))
        if problems and self._batch:  # the first anomaly also ends a "run all" batch: the rest is cancelled
            self.log("이상이 발생해 남은 일괄 실행을 취소합니다")
            s = self.sessions.existing(mode or self.mode())
            if s is not None:
                s.stop()
        self._batch_mark(label, str(e.get("run_dir") or ""))
        nxt = self._next_row()
        self.table.selectRow(nxt)  # selected, NOT started: every step needs its own Play
        self._fill_table()
        self._final_live_view()
        self._update_guide()

    # ------------------------------------------------------------------ live view (polled, GUI thread)
    def _drain_live(self) -> None:
        items = []
        for mode in ("mock", "sim", "robot"):
            s = self.sessions.existing(mode)
            if s is None:
                continue
            for _ in range(500):
                try:
                    items.append(s.live.get_nowait())
                except Exception:  # noqa: BLE001 - queue.Empty
                    break
        if not items:
            return
        main = (self.active or {}).get("main_channel", "pitch")
        for _label, _k, _n, t_ns, head, ant, _body, pres in items:
            if self._t0_ns is None:
                self._t0_ns = t_ns
            self._buf["t"].append((t_ns - self._t0_ns) / 1e9)
            self._buf["cmd"].append(_channel_value(main, pose_from_matrix(head).to_deg_mm(), ant))
            self._buf["meas"].append(float("nan") if pres is None else
                                     _channel_value(main, pose_from_matrix(pres[1]).to_deg_mm(), pres[2]))
        now = time.perf_counter()
        if now - self._last_draw >= 0.25 and self.live_box.isVisible():
            self._last_draw = now
            self._blit_live()

    def _planned(self, label: str, channel: str) -> list[float]:
        """Commanded values of ``channel`` in the trial's plan (computed): the fixed range of the live axes."""
        c = next((x for x in self.checks if x.trial.label == label), None)
        if c is None or c.prepared is None:
            return []
        tl = c.prepared.timeline
        return [_channel_value(channel, pose_from_matrix(tl.head[k]).to_deg_mm(), tl.antennas[k])
                for k in range(0, len(tl), 5)]

    def _prepare_live_axes(self, e: dict[str, Any]) -> None:
        """Fixed axes for the run (so only the lines are redrawn), one full draw now, lines animated."""
        main_ch = e.get("main_channel", "pitch")
        vals = self._planned(e.get("label", ""), main_ch)
        dur = float(e.get("duration_s") or 5.0)
        ax = self.live["ax"]
        lo, hi = (min(vals), max(vals)) if vals else (-5.0, 5.0)
        pad = max(1.0, 0.25 * (hi - lo))
        ax.set_xlim(-0.1, dur + 0.5)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(f"{main_ch} ({'mm' if main_ch in ('x', 'y', 'z') else 'deg'})", fontsize=8.5)
        for line in self.live["lines"].values():
            line.set_data([], [])
            line.set_animated(True)
        self._bg = None
        if self.live_box.isVisible():
            self.live_box.canvas.draw()  # draw_event caches the background without the animated lines

    def _cache_live_background(self, _event: Any) -> None:
        self._bg = self.live_box.canvas.copy_from_bbox(self.live["ax"].bbox)

    def _set_live_data(self) -> None:
        t = list(self._buf["t"])
        self.live["lines"]["cmd"].set_data(t, list(self._buf["cmd"]))
        self.live["lines"]["meas"].set_data(t, list(self._buf["meas"]))

    def _blit_live(self) -> None:
        self._set_live_data()
        c = self.live_box.canvas
        if self._bg is None:
            c.draw()
            if self._bg is None:
                return
        ax = self.live["ax"]
        c.restore_region(self._bg)
        for line in self.live["lines"].values():
            ax.draw_artist(line)
        c.blit(ax.bbox)

    def _final_live_view(self) -> None:
        """After the run: the whole recorded trace, autoscaled, drawn once."""
        self._set_live_data()
        for line in self.live["lines"].values():
            line.set_animated(False)
        self.live["ax"].relim()
        self.live["ax"].autoscale_view()
        self.live_box.redraw()


def _brief(m: dict[str, Any]) -> str:
    parts = []
    for ch, d in m.items():
        if isinstance(d, dict):
            parts.append(f"{ch}: RMSE {_f(d.get('dynamic_rmse'), '.2f')} {d.get('unit')}, Peak "
                         f"{_f(d.get('peak_error'), '.2f')}, 지연 {_f(d.get('lag_ms'), '.0f')} ms "
                         f"(r2 {_f(d.get('lag_fit_r2'), '.2f')}), 진폭비 {_f(d.get('amplitude_ratio'), '.2f')}")
    kind = m.get("data_kind")
    return (f"[{kind}] " if kind else "") + "; ".join(parts) if parts else str(m)


# ================================================================================================ Results tab
#: column -> glossary term (header tooltips)
COL_TIPS = {"dynamic_rmse": "RMSE", "dynamic_rmse_lag_compensated": "Lag-compensated RMSE", "peak_error": "Peak Error",
            "lag_ms": "Lag", "lag_fit_r2": "Lag", "amplitude_ratio": "Amplitude Ratio", "overshoot": "Overshoot",
            "overshoot_ratio": "Overshoot", "settling_time_s": "Settling Time", "static_bias": "Static Bias",
            "data_kind": "data_kind", "channel": "Trial axis", "role": "Trial axis", "stage": "R3",
            "vary": "D / A", "amplitude_deg": "D / A", "duration_s": "D / A", "shape_error_pct": "Lag-compensated RMSE",
            "overshoot_pct": "Overshoot", "hold_ripple": "Hold Ripple", "axis": "R1", "amplitude": "R1",
            "speed_ratio": "Speed Ratio"}
KEY_COLS = ["run", "data_kind", "dynamic_rmse", "peak_error", "lag_ms", "amplitude_ratio"]
KEY_HEAD = {"run": "시행", "dynamic_rmse": "RMSE (지연 포함)", "peak_error": "Peak (지연 포함)", "lag_ms": "지연 Lag (ms)",
            "amplitude_ratio": "진폭비"}
R1_TABLE_COLS = ["run", "data_kind", "axis", "amplitude", "unit", "sign", "rep", "amplitude_ratio", "lag_ms",
                 "lag_fit_r2", "static_bias", "hold_ripple"]
LIMIT_COLS = ["kind", "motion", "variant", "amplitude_deg", "duration_s", "speed_ratio", "accel_dps2", "runnable",
              "robot_scope", "data_kind"]
LAG_FIT_MIN = 0.9  # a lag whose pure-delay fit is weaker is shown with its r2 (results.describe: same threshold)


def _default_library() -> Any:
    from reachy_motion.library.registry import default_library

    return default_library(Path("motions"))


def _key_row(r: dict[str, Any]) -> dict[str, Any]:
    """One row of the key-metrics table: the unit on RMSE / Peak, a lag with a weak fit marked by its r2."""
    unit = r.get("unit") or ""

    def with_unit(v: Any) -> str:
        t = _f(v, ".3g")
        return t if t == "-" else f"{t} {unit}".strip()

    lag = _f(r.get("lag_ms"), ".0f")
    r2 = r.get("lag_fit_r2")
    if lag != "-" and isinstance(r2, float) and r2 == r2 and r2 < LAG_FIT_MIN:
        lag += f" (r² {r2:.2f})"
    return {**r, "dynamic_rmse": with_unit(r.get("dynamic_rmse")), "peak_error": with_unit(r.get("peak_error")),
            "lag_ms": lag}


class ResultsTab(QtWidgets.QWidget):
    """The key numbers of each run's main channel (RMSE, Peak Error, Lag, amplitude ratio), three figures (main
    channel, R3 ladder, expressive vs neutral) and one export with everything else (all metrics, 6-axis and joint
    figures, summaries). Runs are grouped by session and every row names its data_kind, so conditions and data sources
    are not mixed by mistake. Redraws only when visible and no trial is running (``busy``): the figures are too heavy
    to draw while the executor streams."""

    _loaded = QtCore.Signal(object)

    def __init__(self, root: Path = LAB_ROOT, parent: QtWidgets.QWidget | None = None,
                 busy: Callable[[], bool] | None = None, extra_roots: tuple[Path, ...] = (
                     Path("results") / "phase3_sim" / "review",), library: Callable[[], Any] | None = None) -> None:
        super().__init__(parent)
        self.root = Path(root)
        self.extra_roots = tuple(Path(p) for p in extra_roots)
        self._library = library or _default_library  # the motion files of the limit map
        self._limit_dirty = True  # the limit map is drawn when its tab is shown (it computes ~1 s the first time)
        self.runs: list[res.LoadedRun] = []
        self._items: dict[str, QtWidgets.QTreeWidgetItem] = {}  # run key -> its tree item
        self._sessions: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self._focus: set[str] = set()
        self._busy = busy or (lambda: False)
        self._dirty = True
        self._last_export: Path | None = None
        self._session_pinned = False  # the user picked a session in the list: keep it on refresh
        self._pick_view = False  # after loading a session: open the view that fits it (R3 / E-N / main)
        self._loading: set[str] = set()
        self._loaded.connect(self._on_loaded)
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        self._build()
        self.refresh_sessions()
        self.refresh(force=True)

    @staticmethod
    def _key(run_dir: Any) -> str:
        return str(Path(run_dir).resolve())

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        row = QtWidgets.QHBoxLayout()
        self.session_combo = QtWidgets.QComboBox()
        self.session_combo.setToolTip("저장된 세션(실험 묶음). 맨 위가 가장 최근입니다.")
        self.session_combo.activated.connect(lambda _i: setattr(self, "_session_pinned", True))
        self.load_session_btn = QtWidgets.QPushButton("불러오기")
        self.load_session_btn.setStyleSheet("font-weight:bold")
        self.load_session_btn.clicked.connect(self.on_load_session)
        row.addWidget(self.session_combo, 1)
        row.addWidget(self.load_session_btn)
        lv.addLayout(row)
        other = QtWidgets.QPushButton("다른 폴더 불러오기 ...")
        other.setToolTip("시행 폴더나 세션 폴더 (다른 조원이 보낸 폴더도 됩니다)")
        other.clicked.connect(self.on_load_folder)
        lv.addWidget(other)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["세션 / 시행", "data_kind", "단계", "조건", "완료"])
        self.tree.headerItem().setToolTip(1, ko.glossary_text("data_kind"))
        self.tree.itemChanged.connect(lambda *_: self._refresh_timer.start(200))
        lv.addWidget(self.tree, 1)
        brow = QtWidgets.QHBoxLayout()
        for text, slot, tip in (("이 세션만 보기", self.on_only_session, "선택한 세션의 시행만 체크합니다."),
                                ("선택 항목 빼기", self.on_remove, "목록에서만 뺍니다 (파일은 지우지 않음)."),
                                ("모두 비우기", self.on_clear, "목록을 비웁니다 (파일은 지우지 않음).")):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            brow.addWidget(b)
        lv.addLayout(brow)
        self.load_state = QtWidgets.QLabel("")
        self.load_state.setWordWrap(True)
        lv.addWidget(self.load_state)
        split.addWidget(left)

        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        top = QtWidgets.QWidget()
        tv = QtWidgets.QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        self.banner = QtWidgets.QLabel()
        self.banner.setWordWrap(True)
        tv.addWidget(self.banner)
        self.warn_label = QtWidgets.QLabel()
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet("padding:4px; background:#fde8e4; color:#7a1f12")
        tv.addWidget(self.warn_label)
        head = QtWidgets.QLabel("핵심 지표 (각 시행의 주 채널)")
        head.setStyleSheet("font-weight:bold")
        head.setToolTip(ko.glossary_text("RMSE", "Peak Error", "Lag", "Amplitude Ratio", "Trial axis"))
        tv.addWidget(head)
        self.key_table = QtWidgets.QTableWidget()
        tv.addWidget(self.key_table, 1)
        right.addWidget(top)

        bottom = QtWidgets.QWidget()
        bv = QtWidgets.QVBoxLayout(bottom)
        bv.setContentsMargins(0, 0, 0, 0)
        self.tabs = QtWidgets.QTabWidget()
        self.fig_main = FigureBox()
        r3_page = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.fig_r3 = FigureBox()
        self.r3_table = QtWidgets.QTableWidget()
        r3_page.addWidget(self.fig_r3)
        r3_page.addWidget(self.r3_table)
        r3_page.setSizes([600, 160])
        pair_page = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.fig_pair = FigureBox()
        self.pair_table = QtWidgets.QTableWidget()
        role_row = QtWidgets.QWidget()
        rv_ = QtWidgets.QVBoxLayout(role_row)
        rv_.setContentsMargins(0, 0, 0, 0)
        rl = QtWidgets.QHBoxLayout()
        rv_.addLayout(rl)
        en_note = QtWidgets.QLabel(res.EN_METRIC_NOTE)
        en_note.setWordWrap(True)
        en_note.setStyleSheet("padding:4px; background:#fbf3df; color:#5c4400")
        rv_.addWidget(en_note)
        role_row.setMinimumHeight(84)  # the selector + up to three wrapped lines of the note
        rl.addWidget(QtWidgets.QLabel("비교 기준 채널:"))
        self.pair_role = QtWidgets.QComboBox()
        self.pair_role.addItem("주 머리 축 (핵심 지표 표와 같은 채널)", res.ROLE_HEAD)
        self.pair_role.addItem("대표 관절 (명령 진폭이 가장 큰 Stewart 관절)", res.ROLE_JOINT)
        self.pair_role.setToolTip("두 변형 모두 같은 채널로 비교합니다 (Expressive 시행의 채널).\n\n"
                                  + ko.glossary_text("Trial axis", "Stewart"))
        self.pair_role.currentIndexChanged.connect(lambda *_: self.refresh(force=True))
        rl.addWidget(self.pair_role)
        rl.addStretch(1)
        pair_page.addWidget(role_row)
        pair_page.addWidget(self.fig_pair)
        pair_page.addWidget(self.pair_table)
        pair_page.setSizes([90, 600, 160])
        r1_page = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.fig_r1 = FigureBox()
        self.r1_table = QtWidgets.QTableWidget()
        r1_page.addWidget(self.fig_r1)
        r1_page.addWidget(self.r1_table)
        r1_page.setSizes([600, 160])
        limit_page = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.fig_limit = FigureBox()
        self.limit_table = QtWidgets.QTableWidget()
        limit_page.addWidget(self.fig_limit)
        limit_page.addWidget(self.limit_table)
        limit_page.setSizes([640, 140])
        self._tab_pages = {}
        for key, w, title, tip in (
                ("main", self.fig_main, "주 채널",
                 "각 시행의 주 채널: 명령값(점선)과 측정값(실선). 축 단위와 data_kind는 그림에 표시됩니다."),
                ("r1", r1_page, "R1 진폭비", ko.glossary_text("R1", "Amplitude Ratio")),
                ("r3", r3_page, "R3 단계 비교", ko.glossary_text("R3", "D / A", "Lag-compensated RMSE", "Hold Ripple")),
                ("limit", limit_page, "한계 지도",
                 "모션 8개와 R3 사다리가 요구하는 최고 관절 속도·가속 (computed). 체크한 시행 중 실제 로봇에서 실행한 것(원)과 "
                 "R3에서 추종이 무너진 단계(X)를 겹쳐 보여 줍니다."),
                ("pair", pair_page, "Expressive vs Neutral", ko.glossary_text("Expressive", "Neutral"))):
            self.tabs.setTabToolTip(self.tabs.addTab(w, title), tip)
            self._tab_pages[key] = w
        self.tabs.currentChanged.connect(lambda _i: self._draw_limit_if_shown())
        bv.addWidget(self.tabs, 1)
        erow = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton("보고서 자료 내보내기 (PNG · CSV · Markdown)")
        self.export_btn.setStyleSheet("font-weight:bold; padding:6px 10px")
        self.export_btn.setToolTip("체크한 시행의 지표 전체(CSV · JSON), 그림(머리 6축 · 관절 포함), 요약(Markdown)을 "
                                   "results/lab/exports/ 아래 새 폴더에 저장합니다.")
        self.export_btn.clicked.connect(self.on_export_report)
        self.open_btn = QtWidgets.QPushButton("폴더 열기")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.on_open_folder)
        self.export_state = QtWidgets.QLabel("")
        self.export_state.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.export_state.setWordWrap(True)
        erow.addWidget(self.export_btn)
        erow.addWidget(self.open_btn)
        erow.addWidget(self.export_state, 1)
        bv.addLayout(erow)
        right.addWidget(bottom)
        right.setSizes([260, 680])
        split.addWidget(right)
        split.setSizes([400, 1100])
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(split)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        self.refresh_sessions()
        if self._dirty:
            self._refresh_timer.start(50)

    # ------------------------------------------------------------------ sessions and loading
    def refresh_sessions(self, prefer: Path | None = None) -> None:
        """The saved sessions, newest first: Motion Lab sessions, then older sim-verify reviews. The newest is
        selected unless the user picked one (or ``prefer``, the session just shown, is given)."""
        current = self.session_combo.currentData() if self._session_pinned else None
        if prefer is not None:
            current = str(Path(prefer))
        found: list[tuple[str, Path, str]] = []
        if self.root.is_dir():
            for p in self.root.iterdir():
                if p.is_dir() and (p / "runs").is_dir():
                    n = sum(1 for x in (p / "runs").iterdir() if x.is_dir())
                    kind = next((k for tag, k in (("_mock_", "mock"), ("_sim_", "simulated"), ("_robot_", "measured"))
                                 if tag in p.name), "")
                    found.append((p.name, p, f"{p.name}  [{kind or '?'}]  시행 {n}개"))
        for root in self.extra_roots:
            if root.is_dir():
                for p in root.iterdir():
                    if p.is_dir() and (p / "runs").is_dir():
                        n = sum(1 for x in (p / "runs").iterdir() if x.is_dir())
                        found.append(("0" + p.name, p, f"{p.name}  [simulated, sim-verify]  시행 {n}개"))
        found.sort(key=lambda x: x[0], reverse=True)
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        for _, p, text in found:
            self.session_combo.addItem(text, str(p))
        if not found:
            self.session_combo.addItem("(저장된 세션 없음: 실험 탭에서 시행하세요)", None)
        idx = -1
        if current:
            want = Path(current).resolve()
            idx = next((i for i in range(self.session_combo.count()) if self.session_combo.itemData(i)
                        and Path(self.session_combo.itemData(i)).resolve() == want), -1)
        self.session_combo.setCurrentIndex(max(0, idx))
        self.session_combo.blockSignals(False)
        self.load_session_btn.setEnabled(bool(found))

    def on_load_session(self) -> None:
        d = self.session_combo.currentData()
        if d:
            self._pick_view = True
            n = self.load_folder(Path(d))
            self.load_state.setText(f"세션 {Path(d).name}: 시행 {n}개를 불러옵니다 ...")

    def add_loaded(self, run: res.LoadedRun) -> None:
        """A run the Experiment tab's worker already analysed (no reload, no recomputation)."""
        key = self._key(run.run_dir)
        if key in self._items:
            return
        self._on_loaded((key, run, None))

    def add_run(self, run_dir: str | Path) -> None:
        key = self._key(run_dir)
        if key in self._loading or key in self._items:
            return
        if self._busy():  # loading computes IK in Python: wait until no trial runs
            QtCore.QTimer.singleShot(500, lambda p=run_dir: self.add_run(p))
            self.load_state.setText("실행 중인 시행이 끝나면 불러옵니다 ...")
            return
        self._loading.add(key)
        self.load_state.setText(f"시행 {len(self._loading)}개 불러오는 중 ...")

        def work() -> None:
            try:
                self._loaded.emit((key, res.load_result(run_dir), None))
            except Exception as exc:  # noqa: BLE001 - shown in the list state
                self._loaded.emit((key, None, f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=work, name="results-load", daemon=True).start()

    def _session_item(self, run: res.LoadedRun) -> QtWidgets.QTreeWidgetItem:
        s = res.session_of(run)
        key = str(s.resolve())
        it = self._sessions.get(key)
        if it is None:
            it = QtWidgets.QTreeWidgetItem([s.name, "", "", "", ""])
            it.setToolTip(0, str(s))
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsAutoTristate)
            f = it.font(0)
            f.setBold(True)
            it.setFont(0, f)
            self.tree.addTopLevelItem(it)
            it.setExpanded(True)
            self._sessions[key] = it
        return it

    def _on_loaded(self, payload: tuple[str, Any, str | None]) -> None:
        key, run, err = payload
        self._loading.discard(key)
        if run is None:
            self.load_state.setText(f"불러오지 못함 {Path(key).name}: {err}")
        elif key not in self._items:
            self.runs.append(run)
            self.tree.blockSignals(True)
            parent = self._session_item(run)
            it = QtWidgets.QTreeWidgetItem([run.label, run.data_kind, run.stage, res.condition_text(run.condition),
                                            {True: "예", False: "아니오", None: "-"}[run.completed]])
            it.setForeground(1, QtGui.QBrush(QtGui.QColor(KIND_COLOR.get(run.data_kind, "#000000"))))
            it.setToolTip(0, str(run.run_dir))
            it.setData(0, QtCore.Qt.ItemDataRole.UserRole, key)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            checked = (not self._focus) or key in self._focus
            it.setCheckState(0, QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
            parent.insertChild(self._child_index(parent, run), it)  # reading order, not arrival order
            self._items[key] = it
            self.tree.blockSignals(False)
            for c in range(5):
                self.tree.resizeColumnToContents(c)
            self.load_state.setText(f"시행 {len(self.runs)}개 불러옴" + (f", {len(self._loading)}개 불러오는 중"
                                                                       if self._loading else ""))
        self._dirty = True
        self._refresh_timer.start(150)

    def _child_index(self, parent: QtWidgets.QTreeWidgetItem, run: res.LoadedRun) -> int:
        """Where ``run`` goes among its session's runs: ``res.run_order`` (the R3 ladder, expressive first)."""
        by_key = {self._key(r.run_dir): r for r in self.runs}
        try:
            order = res.run_order(run)
            for i in range(parent.childCount()):
                other = by_key.get(parent.child(i).data(0, QtCore.Qt.ItemDataRole.UserRole))
                if other is not None and res.run_order(other) > order:
                    return i
        except TypeError:  # runs of different kinds in one folder: arrival order
            pass
        return parent.childCount()

    def on_load_folder(self) -> None:
        start = self.root if self.root.exists() else Path.cwd()
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "시행 폴더 또는 세션 폴더 (다른 조원이 보낸 폴더도 됩니다)",
                                                       str(start))
        if d:
            self.load_folder(Path(d))

    def load_folder(self, d: Path) -> int:
        found = res.find_runs(d)
        if not found:
            self.load_state.setText(f"{d} 아래에 시행 폴더(commands.csv + present_*.csv)가 없습니다")
        for p in found:
            self.add_run(p)
        return len(found)

    def focus(self, run_dirs: list[str]) -> None:
        """Show exactly these runs (the Experiment tab's 'results' button): others stay loaded but unchecked."""
        self._focus = {self._key(d) for d in run_dirs}
        self.tree.blockSignals(True)
        for key, it in self._items.items():
            it.setCheckState(0, QtCore.Qt.CheckState.Checked if key in self._focus else QtCore.Qt.CheckState.Unchecked)
        self.tree.blockSignals(False)
        for d in run_dirs:
            if self._key(d) not in self._items:
                self.add_run(d)
        self._dirty = True
        self.refresh(force=True)
        self.show_best_view()
        first = Path(run_dirs[0]) if run_dirs else None
        if first is not None:
            self.refresh_sessions(prefer=first.parent.parent if first.parent.name == "runs" else first.parent)

    def show_best_view(self) -> None:
        runs = self.checked_runs()
        page = "r3" if res.r3_rows(runs) else ("pair" if res.pair_rows(runs) else ("r1" if res.r1_rows(runs) else "main"))
        self.tabs.setCurrentWidget(self._tab_pages[page])

    def on_only_session(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        it = items[0]
        sess = it if it.parent() is None else it.parent()
        self._focus = set()
        self.tree.blockSignals(True)
        for key, item in self._items.items():
            item.setCheckState(0, QtCore.Qt.CheckState.Checked if item.parent() is sess else
                               QtCore.Qt.CheckState.Unchecked)
        self.tree.blockSignals(False)
        self.refresh(force=True)

    def on_remove(self) -> None:
        keys = set()
        for it in self.tree.selectedItems():
            kids = [it] if it.parent() is not None else [it.child(i) for i in range(it.childCount())]
            keys.update(k.data(0, QtCore.Qt.ItemDataRole.UserRole) for k in kids)
        for key in keys:
            it = self._items.pop(key, None)
            if it is not None and it.parent() is not None:
                it.parent().removeChild(it)
        self.runs = [r for r in self.runs if self._key(r.run_dir) not in keys]
        for skey, sit in list(self._sessions.items()):
            if sit.childCount() == 0:
                self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(sit))
                del self._sessions[skey]
        self.refresh(force=True)

    def on_clear(self) -> None:
        self.tree.clear()
        self.runs.clear()
        self._items.clear()
        self._sessions.clear()
        self._focus = set()
        self.refresh(force=True)

    def checked_runs(self) -> list[res.LoadedRun]:
        """The checked runs in reading order (session, stage, R3 ladder / expressive before neutral)."""
        return sorted((r for r in self.runs if self._items.get(self._key(r.run_dir)) is not None
                       and self._items[self._key(r.run_dir)].checkState(0) == QtCore.Qt.CheckState.Checked),
                      key=res.run_order)

    # ------------------------------------------------------------------ views
    def refresh(self, force: bool = False) -> None:
        if not force and (not self.isVisible() or self._busy()):
            self._dirty = True
            if self.isVisible():
                self._refresh_timer.start(500)  # a trial runs: draw when it is over
            return
        self._dirty = False
        runs = self.checked_runs()
        n = res.kind_counts(runs)
        self.banner.setText(res.source_banner(runs))
        self.banner.setStyleSheet("padding:4px; background:#e7f3ea; color:#1a3d26" if n["measured"] else
                                  "padding:4px; background:#fbf3df; color:#5c4400")
        warns = res.mixing_warnings(runs)
        self.warn_label.setText("주의: " + "  /  ".join(warns) if warns else "")
        self.warn_label.setVisible(bool(warns))
        _fill(self.key_table, [_key_row(r) for r in res.main_metrics_rows(runs)], KEY_COLS, heads=KEY_HEAD)
        lab_plots.draw_main_channel(self.fig_main.figure, runs)
        self.fig_main.redraw()
        r1 = res.r1_rows(runs)
        lab_plots.draw_r1(self.fig_r1.figure, r1)
        self.fig_r1.redraw()
        _fill(self.r1_table, r1, R1_TABLE_COLS)
        r3 = res.r3_rows(runs)
        lab_plots.draw_r3(self.fig_r3.figure, r3)
        self.fig_r3.redraw()
        _fill(self.r3_table, r3, ["run", "data_kind", "vary", "amplitude_deg", "duration_s", "rep", "channel", "role",
                                  "unit", "lag_ms", "lag_fit_r2", "shape_error_pct", "overshoot_pct", "amplitude_ratio",
                                  "hold_ripple", "dynamic_rmse", "dynamic_rmse_lag_compensated", "peak_error",
                                  "overshoot", "settling_time_s"])
        self._limit_dirty = True
        self._draw_limit_if_shown()
        pairs = res.pair_rows(runs)
        lab_plots.draw_pairs(self.fig_pair.figure, pairs, role=self.pair_role.currentData())
        self.fig_pair.redraw()
        _fill(self.pair_table, pairs, ["motion", "data_kind", "variant", "run", "channel", "role", "unit",
                                       "command_p2p", "dynamic_rmse", "peak_error", "lag_ms", "amplitude_ratio",
                                       "overshoot", "settling_time_s"])
        if self._pick_view and not self._loading:
            self._pick_view = False
            self.show_best_view()

    def _limit_view(self, fig: Any, runs: list[res.LoadedRun]) -> list[dict[str, Any]]:
        """Draw the limit map of ``runs`` into ``fig``; returns its computed points (empty: no motions found)."""
        try:
            points = ex.limit_points(self._library())
        except Exception as exc:  # noqa: BLE001 - shown in the figure
            lab_plots._empty(fig, f"한계 지도를 계산하지 못했습니다: {type(exc).__name__}: {exc}")
            return []
        lab_plots.draw_limit_map(fig, points, lab_plots.limit_overlay(runs, res.r3_rows(runs)))
        return points

    def _draw_limit_if_shown(self) -> None:
        """The limit map, only while its tab is shown (the first time it computes the motions, about 1 s)."""
        if not self._limit_dirty or self.tabs.currentWidget() is not self._tab_pages.get("limit"):
            return
        self._limit_dirty = False
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            points = self._limit_view(self.fig_limit.figure, self.checked_runs())
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.fig_limit.redraw()
        yes_no = {True: "예", False: "아니오"}
        _fill(self.limit_table, [{**p, "runnable": yes_no.get(p.get("runnable"), "-"),
                                  "robot_scope": yes_no.get(p.get("robot_scope"), "-")} for p in points], LIMIT_COLS,
              heads={"duration_s": "모션 길이 / R3 D (s)", "amplitude_deg": "R3 A (deg)"})

    # ------------------------------------------------------------------ export
    def on_export_report(self) -> None:
        """One click, no dialog: everything for the report into results/lab/exports/<time>_report/."""
        if not self.checked_runs():
            self.export_state.setText("내보낼 시행이 없습니다: 왼쪽에서 세션을 불러오고 시행을 체크하세요.")
            return
        out = self.export_all(self.root / "exports" / f"{datetime.now():%Y%m%d_%H%M%S}_report")
        self._last_export = out[0].parent if out else None
        self.open_btn.setEnabled(self._last_export is not None)
        self.export_state.setText(f"파일 {len(out)}개 → {_short_path(self._last_export)}")

    def on_open_folder(self) -> None:
        if self._last_export is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self._last_export.resolve())))

    def export_all(self, out: Path) -> list[Path]:
        """Key metrics and every metric (CSV, JSON), the figures (PNG: the three on screen plus the 6-axis pose and the
        Stewart joints, drawn for the export only), the report summary and the plain-language facts (Markdown), and a
        manifest of the runs and their data_kind."""
        import json

        from reachy_motion.gui.plots import new_figure

        self.refresh(force=True)
        runs = self.checked_runs()
        out.mkdir(parents=True, exist_ok=True)
        files = [res.write_rows_csv(out / "key_metrics.csv", res.main_metrics_rows(runs))]
        files += [res.write_rows_csv(out / f"metrics_{w}.csv", res.metric_rows(runs, w))
                  for w in ("representative", "pose", "joints")]
        files.append(res.write_metrics_json(out / "metrics.json", runs))
        if res.r1_rows(runs):
            files.append(res.write_rows_csv(out / "r1_amplitude.csv", res.r1_rows(runs)))
        if res.r3_rows(runs):
            files.append(res.write_rows_csv(out / "r3_ladder.csv", res.r3_rows(runs)))
        if res.pair_rows(runs):
            files.append(res.write_rows_csv(out / "expressive_vs_neutral.csv", res.pair_rows(runs)))
        pose, joints, limit = new_figure(12.0, 8.0), new_figure(12.0, 8.0), new_figure(11.0, 7.5)
        lab_plots.draw_tracking(pose, runs, res.POSE_CHANNELS, "머리 자세: 명령값 vs 측정값")
        lab_plots.draw_tracking(joints, runs, res.JOINT_CHANNELS, "Stewart 관절: 명령 자세의 IK vs 측정 관절")
        points = self._limit_view(limit, runs)
        if points:
            files.append(res.write_rows_csv(out / "limit_map.csv", points))
        for name, fig in (("main_channel.png", self.fig_main.figure), ("pose.png", pose), ("joints.png", joints),
                          ("r1_amplitude.png", self.fig_r1.figure), ("r3_ladder.png", self.fig_r3.figure),
                          ("limit_map.png", limit), ("expressive_vs_neutral.png", self.fig_pair.figure)):
            fig.savefig(out / name, dpi=130)
            files.append(out / name)
        p = out / "summary.md"
        p.write_text(res.summary_markdown(runs), encoding="utf-8")
        files.append(p)
        p = out / "auto_summary.md"
        p.write_text("# 자동 요약 (계산된 사실만)\n\n## 쉬운 요약\n\n" + "\n".join(f"- {x}" for x in res.describe_simple(runs))
                     + "\n\n## 자세한 요약\n\n" + "\n".join(f"- {x}" for x in res.describe(runs)) + "\n",
                     encoding="utf-8")
        files.append(p)
        p = out / "manifest.json"
        p.write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                 "tool": "reachy-motion Motion Lab results export",
                                 "data_kinds": res.kind_counts(runs),
                                 "runs": [{"run": r.label, "run_dir": str(r.run_dir), "session": res.session_of(r).name,
                                           "data_kind": r.data_kind, "stage": r.stage,
                                           "condition": r.condition} for r in runs],
                                 "note": "figures and metrics are computed from the listed runs; their data_kind is "
                                         "the run's own (mock / simulated / measured)"}, indent=2, ensure_ascii=False),
                     encoding="utf-8")
        files.append(p)
        return files


def _fill(table: QtWidgets.QTableWidget, rows: list[dict[str, Any]], cols: list[str],
          heads: dict[str, str] | None = None) -> None:
    table.clear()
    table.setColumnCount(len(cols))
    table.setHorizontalHeaderLabels([(heads or {}).get(c) or res.METRIC_LABELS.get(c) or ko.label(ko.COLUMNS, c)
                                     for c in cols])
    for j, c in enumerate(cols):
        term = COL_TIPS.get(c)
        if term:
            table.horizontalHeaderItem(j).setToolTip(ko.glossary_text(term))
    table.setRowCount(len(rows))
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = r.get(c)
            text = _f(v, ".3g") if isinstance(v, float) else ("" if v is None else str(v))
            color = KIND_COLOR.get(v) if c == "data_kind" else None
            table.setItem(i, j, _item(text, color, bold=c == "data_kind"))
    table.resizeColumnsToContents()
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)


__all__ = ["LabSessions", "ExperimentTab", "ResultsTab", "EventBridge", "FigureBox", "LOCK_TEXT", "STATUS_COL"]
