"""PySide6 window of the Reachy HW4 Motion Lab (``reachy-motion gui``): Motion / Experiment / Results tabs.

Motion tab: the motion files as they are (``motions/*.yaml``; the design is the student's, so there is no editing
here), what each one is for, its planned position / velocity (computed), expressive vs neutral, its settings and a
side-by-side video, and playback in the local MuJoCo through ``gui.lab.LabSession`` (the daemon's viewer window shows
it; nothing is recorded). Experiment and Results tabs: ``gui.lab_tabs`` (MuJoCo or the real robot; every robot action
is one click of the person at the GUI: CLAUDE.md rule 3, docs/lab/GUI_ROBOT.md).
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from reachy_motion.gui import ko, plots
from reachy_motion.gui.lab_tabs import EventBridge, ExperimentTab, LabSessions, ResultsTab, sound_text
from reachy_motion.gui.model import EditorModel
from reachy_motion.gui.robot_panel import STATE_SHORT
from reachy_motion.gui.video_tab import MotionVideoTab
from reachy_motion.library.spec import MOTION_NAMES

DOCS = Path(__file__).resolve().parents[2] / "docs"  # <repo>/docs (the project is installed editable)
GUIDE = DOCS / "TEAM_GUIDE.md"
README = DOCS.parent / "README.md"  # the help where docs/ is not shipped (the professor's source zip)
TRACK_KO = {"head": "머리", "antennas": "안테나", "body": "몸통"}
PLAY_TIP = "고른 변형을 로컬 MuJoCo에서 재생합니다 (viewer 창, 처음에는 시작에 약 10초). 기록은 남기지 않습니다."
ROBOT_TIP = ("고른 모션·변형을 실제 로봇에서 1번 실행합니다: R2 시행(INIT → 모션 → INIT)과 같은 검사·기록, 결과는 ③ 결과 탭에. "
             "먼저 창 위쪽 로봇 상자에서 연결 → Wake-up. 누르면 바로 움직입니다 (클릭 = 승인).")
ROBOT_BOX_TIP = ("로봇 상자(IP · 연결 · Wake-up · 토크 OFF · 연결 끊기)를 탭 위에 보이거나 숨깁니다. 처음에는 보이고, 연결된 "
                 "동안과 ② 실험 탭이 '실제 로봇'일 때는 숨길 수 없습니다. 버튼 클릭 = 그 동작의 승인.")
PAIR_TIP = "같은 모션의 Expressive와 Neutral을 MuJoCo에서 이어서 재생합니다 (기록 안 함)."
SOUND_TIP = ("Expressive 4개의 효과음을 어디서 낼지 고릅니다 (aha·success는 밝게 올라가는 소리, failure·boredom은 낮게 "
             "내려가는 소리). Neutral은 항상 소리 없음.\n"
             "로봇 스피커: 실제 로봇 재생 때, 시행을 시작하기 전에 음원을 로봇에 올리고(연결마다 한 번) 설계 시각에 로봇에 "
             "재생을 요청합니다 (SDK daemon의 /api/media/sounds/upload · play_sound). 실제 로봇에서 네 소리 모두 재생을 "
             "확인했습니다 (2026-11-14). 소리가 안 나면 노트북 스피커로 바꾸세요.\n"
             "노트북 스피커: 이 PC에서 냅니다 (MuJoCo 재생에서도).\n끄기: 소리 없음.")
VIDEO_TIP = "두 변형을 한 화면에 나란히 (MuJoCo 오프라인 재생, simulated). 처음에는 약 10–15초 걸립니다."


class FigureTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = plots.new_figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas, 1)

    def redraw(self) -> None:
        self.canvas.draw_idle()


def help_start() -> Path:
    """What '? 사용법' opens: the team guide in the repository, README.md where docs/ is not shipped."""
    return GUIDE if GUIDE.is_file() else README


class HelpDialog(QtWidgets.QDialog):
    """The usage guide in the app (``help_start``). Links to other .md files open here; pictures and other files
    open in the system's viewer."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, start: Path | None = None) -> None:
        super().__init__(parent)
        start = help_start() if start is None else start
        self.setWindowTitle("사용법")
        self.resize(940, 820)
        v = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        back = QtWidgets.QPushButton("← 뒤로")
        home = QtWidgets.QPushButton("처음으로")
        row.addWidget(back)
        row.addWidget(home)
        row.addStretch(1)
        v.addLayout(row)
        self.view = QtWidgets.QTextBrowser()
        self.view.setOpenLinks(False)
        self.view.anchorClicked.connect(self.follow)
        v.addWidget(self.view, 1)
        back.clicked.connect(self.view.backward)
        home.clicked.connect(lambda: self.show_doc(start))
        self.show_doc(start)

    def show_doc(self, path: Path | str) -> None:
        if not Path(path).is_file():
            self.view.setPlainText(f"문서를 찾을 수 없습니다: {path}")
            return
        self.view.setSource(QtCore.QUrl.fromLocalFile(str(Path(path).resolve())),
                            QtGui.QTextDocument.ResourceType.MarkdownResource)

    def follow(self, url: QtCore.QUrl) -> None:
        if url.isRelative() and not url.path() and url.hasFragment():
            self.view.scrollToAnchor(url.fragment())
            return
        target = self.view.source().resolved(url) if url.isRelative() else url
        if target.isLocalFile() and target.toLocalFile().lower().endswith(".md"):
            self.show_doc(target.toLocalFile())
        else:
            QtGui.QDesktopServices.openUrl(target)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, model: EditorModel | None = None, initial: tuple[str, str] | None = None,
                 sessions: LabSessions | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Reachy HW4 Motion Lab")
        self.resize(1560, 940)
        self.model = model or EditorModel()
        self._signed_digests = self._signed_off_digests()
        ko.use_korean_qt(QtWidgets.QApplication.instance())
        self.lab_bridge = EventBridge()
        self.lab_bridge.event.connect(self.on_lab_event)
        self.sessions = sessions or LabSessions(self.lab_bridge.event.emit)
        if sessions is not None:
            sessions._emit = self.lab_bridge.event.emit
        self._play_started: float | None = None
        self._play_duration = 0.0
        self._loading = False
        self._spec_ok = False  # the shown motion passes the initial policy
        self._robot_key: Any = None  # last applied state of the robot button (re-applied only when it changes)
        self._help: HelpDialog | None = None
        from reachy_motion.sound import LaptopSpeaker

        self.laptop = LaptopSpeaker()  # '♪ 소리 듣기' (this PC only)
        self._build()
        self._progress = QtCore.QTimer(self)
        self._progress.timeout.connect(self._tick_progress)
        self._robot_timer = QtCore.QTimer(self)
        self._robot_timer.timeout.connect(self._refresh_robot_play)
        self._robot_timer.start(500)
        name, variant = initial or self._first_available()
        self.open_motion(name, variant)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        tb = QtWidgets.QToolBar("모션")
        tb.setMovable(False)
        tb.addWidget(QtWidgets.QLabel(" 모션: "))
        self.motion_combo = QtWidgets.QComboBox()
        for n in MOTION_NAMES:
            self.motion_combo.addItem(ko.label(ko.MOTIONS, n), n)
        tb.addWidget(self.motion_combo)
        self.rb_expr = QtWidgets.QRadioButton("Expressive (표현)")
        self.rb_neut = QtWidgets.QRadioButton("Neutral (중립)")
        self.rb_expr.setChecked(True)
        tb.addWidget(self.rb_expr)
        tb.addWidget(self.rb_neut)
        self.rb_expr.setToolTip(ko.glossary_text("Expressive"))
        self.rb_neut.setToolTip(ko.glossary_text("Neutral"))
        tb.addSeparator()
        export = QtGui.QAction("내보내기...", self)
        export.setToolTip("이 모션의 사양·안전 검사·그림을 폴더로 내보냅니다 (computed)")
        export.triggered.connect(self.on_export)
        tb.addAction(export)
        self.motion_combo.currentIndexChanged.connect(lambda _i: self._switch())
        self.rb_expr.toggled.connect(lambda on: on and self._switch())
        self.rb_neut.toggled.connect(lambda on: on and self._switch())

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.info_view = QtWidgets.QTextBrowser()
        splitter.addWidget(self.info_view)
        self.tabs = QtWidgets.QTabWidget()
        self.fig_motion = FigureTab()
        self.fig_compare = FigureTab()
        self.compare_view = QtWidgets.QTextBrowser()
        self.params_table = QtWidgets.QTableWidget()
        self.params_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        compare_box = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        compare_box.addWidget(self.compare_view)
        compare_box.addWidget(self.fig_compare)
        compare_box.setSizes([260, 600])
        self.video_tab = MotionVideoTab()
        for w, title, tip in ((self.fig_motion, "위치 / 속도", "설계한 모션의 머리·안테나·몸통 위치와 속도 (computed: 계획값, 측정 아님)"),
                              (compare_box, "Expressive vs Neutral", ko.glossary_text("Expressive", "Neutral")),
                              (self.video_tab, "E/N 나란히 영상", VIDEO_TIP),
                              (self.params_table, "설정값", "키프레임마다 시간·보간·값 (모션 파일 그대로)")):
            self.tabs.setTabToolTip(self.tabs.addTab(w, title), tip)
        splitter.addWidget(self.tabs)
        splitter.setSizes([420, 1080])
        motion_page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(motion_page)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(tb)
        lay.addWidget(splitter, 1)
        lay.addWidget(self._build_playback())
        from reachy_motion.library.registry import default_library

        self.experiment_tab = ExperimentTab(self.sessions, lambda: default_library(self.model.dir), self.model.policy)
        self.results_tab = ResultsTab(self.sessions.root, busy=self.sessions.busy,
                                      library=lambda: default_library(self.model.dir))
        self.experiment_tab.run_saved.connect(self.results_tab.add_run)
        self.experiment_tab.run_loaded.connect(self.results_tab.add_loaded)
        self.experiment_tab.show_results.connect(self.on_show_results)
        self.motion_page = motion_page
        self.main_tabs = QtWidgets.QTabWidget()
        self.main_tabs.addTab(motion_page, "① 모션")
        self.main_tabs.addTab(self.experiment_tab, "② 실험")
        self.main_tabs.addTab(self.results_tab, "③ 결과")
        corner = QtWidgets.QWidget()
        cl = QtWidgets.QHBoxLayout(corner)
        cl.setContentsMargins(0, 0, 2, 0)
        self.robot_toggle = QtWidgets.QToolButton()
        self.robot_toggle.setText("로봇 상자")
        self.robot_toggle.setCheckable(True)
        self.robot_toggle.setChecked(True)  # the box is shown from the start (user decision 2026-11-13)
        self.robot_toggle.setToolTip(ROBOT_BOX_TIP)
        self.robot_toggle.toggled.connect(lambda _on: self._refresh_robot_box())
        self.help_btn = QtWidgets.QToolButton()
        self.help_btn.setText("? 사용법")
        self.help_btn.setToolTip("사용법을 창 안에서 엽니다: 설치 · 실행 · 실험 · 결과 · 문제 해결")
        self.help_btn.clicked.connect(self.on_help)
        cl.addWidget(self.robot_toggle)
        cl.addWidget(self.help_btn)
        self.main_tabs.setCornerWidget(corner, QtCore.Qt.Corner.TopRightCorner)
        # the robot box above the tabs (2026-11-13): connect / Wake-up / torque-off from every tab, one box only
        self.robot_box = self.experiment_tab.release_robot_panel()
        self.robot_box.use_wide_layout()
        self.experiment_tab.robot_mode_changed.connect(lambda _on: self._refresh_robot_box())
        central = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(central)
        cv.setContentsMargins(4, 2, 4, 0)
        cv.setSpacing(2)
        cv.addWidget(self.robot_box)
        cv.addWidget(self.main_tabs, 1)
        self.setCentralWidget(central)
        self.status = self.statusBar()
        self._refresh_robot_box()

    def _build_playback(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("재생 (MuJoCo는 기록 안 함 · 실제 로봇은 R2 시행으로 기록)")
        outer = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("▶ MuJoCo 재생")
        self.pair_btn = QtWidgets.QPushButton("▶ Expressive → Neutral")
        self.video_btn = QtWidgets.QPushButton("▶ E/N 나란히 영상")
        self.stop_btn = QtWidgets.QPushButton("■ 정지")
        self.robot_play_btn = QtWidgets.QPushButton("▶ 실제 로봇 재생")
        for w in (self.play_btn, self.pair_btn, self.video_btn):
            w.setStyleSheet("font-weight:bold; padding:6px 12px")
        self.robot_play_btn.setStyleSheet("QPushButton { font-weight:bold; padding:6px 12px; color:#8a2b0f } "
                                          "QPushButton:disabled { color:#a3a19c }")
        self.robot_play_btn.setToolTip(ROBOT_TIP)
        self.robot_play_btn.setEnabled(False)
        self.stop_btn.setToolTip("MuJoCo 재생과 실제 로봇 시행을 멈춥니다 (실제 로봇: 0.3초 감속 뒤 그 자리, 토크는 켜진 채)")
        self.robot_state = QtWidgets.QLabel("")
        self.robot_state.linkActivated.connect(lambda _link: self.go_robot_box())
        self.play_btn.clicked.connect(lambda: self.on_preview_sim(pair=False))
        self.pair_btn.clicked.connect(lambda: self.on_preview_sim(pair=True))
        self.video_btn.clicked.connect(self.on_video)
        self.stop_btn.clicked.connect(self.on_stop)
        self.robot_play_btn.clicked.connect(self.on_robot_play)
        for w in (self.play_btn, self.pair_btn, self.video_btn, self.stop_btn):
            row.addWidget(w)
        row.addSpacing(24)
        row.addWidget(self.robot_play_btn)
        row.addWidget(self.robot_state)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addLayout(self._build_sound_row())
        state_row = QtWidgets.QHBoxLayout()
        self.policy_badge = QtWidgets.QLabel("")
        self.policy_badge.setWordWrap(True)
        self.policy_badge.setToolTip(ko.glossary_text("Initial Policy"))
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.play_state = QtWidgets.QLabel("대기")
        state_row.addWidget(self.policy_badge, 2)
        state_row.addWidget(self.progress, 1)
        state_row.addWidget(self.play_state, 1)
        outer.addLayout(state_row)
        return box

    def _build_sound_row(self) -> QtWidgets.QHBoxLayout:
        """The effect sounds (reachy_motion.sound): where they come out, a timing offset, and a laptop listen button."""
        from reachy_motion.sound import OFFSET_RANGE_MS, OUTPUT_KO, SOUND_OUTPUTS

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("효과음:"))
        self.sound_combo = QtWidgets.QComboBox()
        for key in SOUND_OUTPUTS:
            self.sound_combo.addItem(OUTPUT_KO[key], key)
        self.sound_combo.setToolTip(SOUND_TIP)
        self.sound_offset = QtWidgets.QSpinBox()
        self.sound_offset.setRange(*OFFSET_RANGE_MS)
        self.sound_offset.setSingleStep(10)
        self.sound_offset.setSuffix(" ms")
        self.sound_offset.setToolTip("소리를 모션의 설계 시각보다 늦게(+) 또는 일찍(-) 냅니다. 실제 로봇은 머리가 명령보다 약 "
                                     "0.19 s 늦게 움직이고(2026-11-12 실측), 스피커 쪽 지연은 아직 재지 않았습니다: 한 번 "
                                     "보고 맞추세요.")
        self.sound_listen = QtWidgets.QPushButton("♪ 소리 듣기")
        self.sound_listen.setToolTip("이 모션의 효과음을 노트북 스피커로 한 번 들어 봅니다 (로봇에는 아무것도 보내지 않음).")
        self.sound_label = QtWidgets.QLabel("")
        self.sound_label.setStyleSheet("color:#52514e")
        self.sound_combo.currentIndexChanged.connect(lambda _i: self._sound_changed())
        self.sound_offset.valueChanged.connect(lambda _v: self._sound_changed())
        self.sound_listen.clicked.connect(self.on_sound_listen)
        for w in (self.sound_combo, QtWidgets.QLabel("시각 보정"), self.sound_offset, self.sound_listen):
            row.addWidget(w)
        row.addSpacing(12)
        row.addWidget(self.sound_label, 1)
        self.sessions.sound.sounds_dir = Path(self.model.dir) / "sounds"
        self._sync_sound_controls()
        return row

    def _sync_sound_controls(self) -> None:
        """The widgets from the shared setting (``main()`` turns the robot speaker on)."""
        s = self.sessions.sound
        for w in (self.sound_combo, self.sound_offset):
            w.blockSignals(True)
        self.sound_combo.setCurrentIndex(max(0, self.sound_combo.findData(s.output)))
        self.sound_offset.setValue(int(round(s.offset_ms)))
        for w in (self.sound_combo, self.sound_offset):
            w.blockSignals(False)
        self._refresh_sound_label()

    def _sound_changed(self) -> None:
        self.sessions.sound.output = str(self.sound_combo.currentData())
        self.sessions.sound.offset_ms = float(self.sound_offset.value())
        self._refresh_sound_label()

    def _refresh_sound_label(self) -> None:
        """What the shown motion will sound like, and where."""
        if not hasattr(self, "sound_label"):
            return
        try:
            spec = self.model.spec
        except RuntimeError:  # no motion open yet (the window is being built)
            spec = None
        cue = None if spec is None else spec.effective().sound
        path = None if cue is None else Path(self.sessions.sound.sounds_dir) / cue.file
        self.sound_listen.setEnabled(bool(path is not None and path.is_file()))
        out = self.sessions.sound.output
        if cue is None:
            text = "이 모션은 효과음 없음" + (" (Neutral은 소리 없음)" if spec is not None and spec.variant == "neutral" else "")
        elif path is None or not path.is_file():
            text = f"{cue.file} @ {cue.at_s:g} s · 음원 파일 없음 ({path})"
        elif out == "off":
            text = f"{cue.file} @ {cue.at_s:g} s · 끔"
        elif out == "robot":
            text = (f"{cue.file} @ {cue.at_s:g} s · 실제 로봇 재생 때 로봇 스피커로 (MuJoCo 재생은 노트북 스피커일 때만)")
        else:
            text = f"{cue.file} @ {cue.at_s:g} s · 노트북 스피커로 (MuJoCo·실제 로봇 재생 모두)"
        self.sound_label.setText(text)

    def on_sound_listen(self) -> None:
        """The shown motion's sound once on this PC (nothing sent anywhere)."""
        spec = self.model.spec
        cue = None if spec is None else spec.effective().sound
        if cue is None:
            return
        try:
            self.laptop.play(Path(self.sessions.sound.sounds_dir) / cue.file)
            self.play_state.setText(f"노트북 스피커: {cue.file}")
        except Exception as exc:  # noqa: BLE001 - shown
            self.play_state.setText(f"소리를 낼 수 없음: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ motion / variant
    def _signed_off_digests(self) -> dict[str, str]:
        """Digests of the signed-off motion files (draft: false): the card's Korean summary describes exactly those;
        another file shows its own YAML intent instead."""
        from reachy_motion.library.registry import default_library

        try:
            return {s.key: s.digest() for s in default_library(self.model.dir) if not s.draft}
        except Exception:  # noqa: BLE001 - no summary then, never a crash
            return {}

    def _first_available(self) -> tuple[str, str]:
        avail = self.model.available()
        return avail[0] if avail else (MOTION_NAMES[0], "expressive")

    def _variant(self) -> str:
        return "neutral" if self.rb_neut.isChecked() else "expressive"

    def _switch(self) -> None:
        if self._loading:
            return
        self.open_motion(self.motion_combo.currentData(), self._variant())

    def open_motion(self, name: str, variant: str) -> None:
        self._loading = True
        try:
            self.motion_combo.setCurrentIndex(max(0, self.motion_combo.findData(name)))
            (self.rb_neut if variant == "neutral" else self.rb_expr).setChecked(True)
            try:
                self.model.open(name, variant)
            except (FileNotFoundError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(self, "모션 열기", f"{name}:{variant}\n{exc}")
                return
        finally:
            self._loading = False
        self.refresh_analysis()

    def on_export(self) -> None:
        spec = self.model.spec
        default = Path("results") / "phase2" / "exports" / f"{datetime.now():%Y%m%d_%H%M%S}_{spec.name}_{spec.variant}"
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "내보낼 폴더 (그 안에 새 폴더를 만듭니다)", str(
            default.parent if default.parent.exists() else Path.cwd()))
        if not path:
            return
        out = Path(path) / default.name
        try:
            files = self.model.export_bundle(out)
        except Exception as exc:  # noqa: BLE001 - show, do not crash
            QtWidgets.QMessageBox.warning(self, "내보내기", f"{type(exc).__name__}: {exc}")
            return
        self.log(f"파일 {len(files)}개를 {out}에 내보냄")

    # ------------------------------------------------------------------ views
    def refresh_analysis(self, cursor_s: float | None = None) -> None:
        bundle = self.model.analyze()
        plots.draw_motion(self.fig_motion.figure, bundle, cursor_s)
        self.fig_motion.redraw()
        self._fill_params(bundle)
        self._refresh_compare(bundle)
        self._refresh_card(bundle)
        self.setWindowTitle(f"Reachy HW4 Motion Lab - {bundle.spec.key}")

    def _refresh_card(self, bundle: Any) -> None:
        """The motion card, the policy badge and which play buttons can run (and why not)."""
        spec = bundle.spec
        ok = bundle.passes_policy
        other = self.model.counterpart()
        other_ok = other is not None and self.model.analyze(other).passes_policy
        if ok:
            badge = ("<span style='color:#1a7f37'><b>초기 정책 통과</b></span> "
                     "<span style='color:#6b6a66'>(computed · 기구적 안전 보장 아님)</span>")
        else:
            rep = bundle.report
            why = "; ".join(ko.violation_text(v) for v in (rep.violations[:2] if rep is not None else [])) or \
                "; ".join(bundle.problems[:2]) or (bundle.error or "초기 정책 밖")
            badge = f"<span style='color:#c0392b'><b>초기 정책 밖 - 재생 안 함</b></span> {why}"
        self.policy_badge.setText(badge)
        self.play_btn.setEnabled(ok)
        self.play_btn.setToolTip(PLAY_TIP if ok else "이 변형은 초기 정책 밖이라 재생하지 않습니다.")
        self._spec_ok = ok
        self._refresh_robot_play()
        pair_why = "" if ok and other_ok else ("다른 변형 파일이 없습니다" if other is None else
                                                "두 변형 중 하나가 초기 정책 밖입니다")
        for b, tip in ((self.pair_btn, PAIR_TIP), (self.video_btn, VIDEO_TIP)):
            b.setEnabled(not pair_why)
            b.setToolTip(pair_why or tip)
        pair = None if other is None else ((spec, other) if spec.variant == "expressive" else (other, spec))
        self.video_tab.set_specs(*(pair or (None, None)), edited=False, why_not=pair_why)
        kind = "Expressive (표현)" if spec.variant == "expressive" else "Neutral (중립, 비교 기준)"
        signed = self._signed_digests.get(spec.key) == spec.digest()
        what = (ko.MOTION_SUMMARY.get((spec.name, spec.variant)) if signed else None) or spec.intent or "-"
        lecture = ko.MOTION_PRINCIPLES.get((spec.name, spec.variant)) if signed else None
        principles = lecture or ", ".join(p.principle.value.replace("_", " ") for p in spec.principles) or "-"
        sound = "" if spec.sound is None else f" · 효과음 {spec.sound.file} @ {spec.sound.at_s:g} s (아래 '효과음' 설정)"
        self._refresh_sound_label()
        v_word, a_word = ko.va_words(spec.valence, spec.arousal)
        self.info_view.setHtml(
            f"<h3>{ko.label(ko.MOTIONS, spec.name)} - {kind}</h3>"
            f"<p>{what}</p>"
            f"<p><b>길이</b> {spec.duration:.2f} s{sound}</p>"
            f"<p><b>애니메이션 원리{' (강의 용어)' if lecture else ''}</b> {principles}</p>"
            f"<p><b>기대하는 느낌 (설계 목표, 측정 아님)</b><br>{v_word} (Valence {spec.valence:+.2f}), "
            f"{a_word} (Arousal {spec.arousal:.2f})</p>")

    def _refresh_compare(self, bundle: Any) -> None:
        try:
            comp = self.model.comparison()
        except Exception as exc:  # noqa: BLE001
            self.compare_view.setPlainText(f"비교 실패: {exc}")
            return
        if comp is None:
            self.compare_view.setPlainText("이 모션의 다른 변형 파일이 아직 없습니다.")
            self.fig_compare.figure.clear()
            self.fig_compare.redraw()
            return
        self.compare_view.setHtml(_comparison_html(comp))
        other = self.model.counterpart()
        ob = self.model.analyze(other)
        a, b = (bundle, ob) if self.model.spec.variant == "expressive" else (ob, bundle)
        plots.draw_compare(self.fig_compare.figure, a, b)
        self.fig_compare.redraw()

    def _fill_params(self, bundle: Any) -> None:
        """Each key of the motion as the assignment asks: duration, amplitude (change), peak speed, pause,
        interpolation (computed from the motion file; the full table is in the export bundle)."""
        rows = bundle.segments or []
        cols = ["트랙", "#", "이름", "출발 s", "도착 s", "이동 s", "정지 s", "보간", "목표 자세", "변화량 (진폭)", "최고 속도"]
        tips = {7: "보간법 (SDK 4종: linear, minjerk, ease_in_out, cartoon)",
                9: "앞 키에서 바뀐 양", 10: "공칭 최고 속도 = 보간 피크 계수 × 평균 속도 (computed)"}
        t = self.params_table
        t.clear()
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels(cols)
        for j, tip in tips.items():
            t.horizontalHeaderItem(j).setToolTip(tip)
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [TRACK_KO.get(r.get("track"), str(r.get("track"))), str(r.get("index", "")), str(r.get("label") or ""),
                    _num(r.get("t_depart_s")), _num(r.get("t_arrive_s")), _num(r.get("duration_s")), _num(r.get("hold_s")),
                    str(r.get("easing") or ""), _pose_text(r, ""), _pose_text(r, "delta_", signed=True), _peak_text(r)]
            for j, text in enumerate(vals):
                t.setItem(i, j, QtWidgets.QTableWidgetItem(text))
        t.resizeColumnsToContents()

    def on_help(self) -> None:
        if self._help is None:
            self._help = HelpDialog(self)
        self._help.show()
        self._help.raise_()

    def on_show_results(self, run_dirs: list[str]) -> None:
        self.main_tabs.setCurrentWidget(self.results_tab)
        self.results_tab.focus(list(run_dirs))
        self.log(f"결과 탭: 시행 {len(run_dirs)}개를 표시합니다")

    # ------------------------------------------------------------------ MuJoCo preview
    def on_preview_sim(self, pair: bool = False) -> None:
        """Play the motion (or expressive then neutral) in the local MuJoCo. Nothing is recorded."""
        spec = self.model.spec
        specs = [spec]
        if pair:
            other = self.model.counterpart()
            if other is None:
                self.play_state.setText("다른 변형 파일이 없습니다")
                return
            specs = [spec, other] if spec.variant == "expressive" else [other, spec]
        bad = [s.key for s in specs if not self.model.analyze(s).passes_policy]
        if bad:  # the worker refuses them too; this is the GUI-side gate
            self.play_state.setText(f"재생 거부: 초기 정책 밖 ({', '.join(bad)})")
            return
        if self.sessions.busy():
            self.play_state.setText("MuJoCo / 실험 작업이 아직 실행 중입니다. 끝나거나 '정지'한 뒤 다시 누르세요")
            return
        sim = self.sessions.get("sim")
        for sp in specs:
            sim.preview(sp)
        first = sim.state != "ready"
        self.play_state.setText(("MuJoCo: 시작하는 중 (약 10초" + (", viewer 창이 열립니다" if self.sessions.viewer else "")
                                 + ") ...") if first else f"MuJoCo: 재생 대기열 {len(specs)}개")
        self.log("MuJoCo 미리보기 " + " -> ".join(sp.key for sp in specs) + " (simulated, 저장 안 함)")

    def on_video(self) -> None:
        """The side-by-side video of this motion (shown at once when it was made before)."""
        self.tabs.setCurrentWidget(self.video_tab)
        if self.video_tab.found is None and not self.video_tab.rendering:
            self.video_tab.on_make()

    def on_stop(self) -> None:
        self.sessions.stop_all()  # MuJoCo and the robot (S1: 0.3 s along the path, then it holds)
        self.log("정지 요청")

    # ------------------------------------------------------------------ real robot (one R2 trial per click)
    def _refresh_robot_play(self) -> None:
        """The robot button runs only at INIT (connected, woken up), idle, for a motion inside the initial policy."""
        if not hasattr(self, "experiment_tab"):
            return
        self._refresh_robot_box()
        st = self.experiment_tab.robot_panel.state
        s = self.sessions.existing("robot")
        connected = bool(s is not None and s.connected)
        busy = self.sessions.busy()
        ready = connected and st == "at_init" and not busy and self._spec_ok
        if not connected:
            text = "실제 로봇 연결 안 됨 · <a href='robot'>위쪽 로봇 상자에서 연결</a>"
        elif st != "at_init":
            text = f"실제 로봇 {STATE_SHORT.get(st, st)} · <a href='robot'>위쪽 로봇 상자</a>"
        else:
            text = "실제 로봇 실행 중" if busy else "실제 로봇 INIT 대기"
        if (ready, text) == self._robot_key:
            return
        self._robot_key = (ready, text)
        self.robot_play_btn.setEnabled(ready)
        self.robot_state.setText(text)

    def _refresh_robot_box(self) -> None:
        """The robot box above the tabs: shown from the start (the corner button is checked), always while the robot is
        connected or the Experiment tab is on '실제 로봇' (those two cannot hide it); otherwise the person may hide it with
        the corner button, and the Motion tab's link shows it again."""
        if not hasattr(self, "robot_box"):
            return
        s = self.sessions.existing("robot")
        forced = bool(s is not None and s.connected) or self.robot_box.state != "disconnected" \
            or self.experiment_tab.mode() == "robot"
        if forced and not self.robot_toggle.isChecked():
            self.robot_toggle.blockSignals(True)
            self.robot_toggle.setChecked(True)
            self.robot_toggle.blockSignals(False)
        if self.robot_toggle.isEnabled() == forced:
            self.robot_toggle.setEnabled(not forced)
        show = forced or self.robot_toggle.isChecked()
        if self.robot_box.isHidden() == show:
            self.robot_box.setVisible(show)
            if show:
                self.robot_box.refresh_buttons()

    def go_robot_box(self) -> None:
        """The Motion tab's link: the robot box above the tabs (the tab stays)."""
        self.robot_toggle.setChecked(True)
        self._refresh_robot_box()

    def on_robot_play(self) -> None:
        """This motion once on the robot: an R2 trial (the same check, click approval and record as the R2 table)."""
        spec = self.model.spec
        why = self.experiment_tab.play_motion_on_robot(spec.name, spec.variant)
        self.play_state.setText(why or f"실제 로봇: {spec.key} 준비 중 ...")
        self.log(why or f"실제 로봇 재생 {spec.key} (R2 시행, measured)")
        self._refresh_robot_play()

    @QtCore.Slot(object)
    def on_lab_event(self, event: dict[str, Any]) -> None:
        """Motion Lab worker events (queued to the GUI thread): MuJoCo preview here, trials in the Experiment tab."""
        kind = event.get("kind")
        if event.get("mode") == "sim" and kind in ("connecting", "ready", "preview_started", "preview_finished",
                                                    "error", "closed"):
            text = {"connecting": "MuJoCo: 시작 중 (약 10초" + (", viewer 창이 열림" if event.get("viewer") else "")
                                  + ") ...", "ready": "MuJoCo: 준비됨",
                    "preview_started": f"MuJoCo: 재생 중 {event.get('key')}",
                    "preview_finished": f"MuJoCo: {'완료' if event.get('completed') else '정지'}",
                    "error": f"MuJoCo 오류: {event.get('error')}", "closed": "MuJoCo: 닫힘"}[kind]
            snd = sound_text(event.get("sound")) if kind == "preview_finished" else None
            self.play_state.setText(text + ("" if snd is None else f" · {snd[0]}"))
            if kind == "preview_started":  # progress of the MuJoCo preview
                self._play_started = time.perf_counter()
                self._play_duration = max(float(event.get("duration_s") or 0.0), 1e-3)
                self.progress.setValue(0)
                self._progress.start(50)
            elif kind in ("preview_finished", "error", "closed"):
                self._play_started = None
                if kind == "preview_finished" and event.get("completed"):
                    self.progress.setValue(1000)
        if event.get("mode") == "robot" and (kind in ("trial_started", "trial_finished", "trial_cancelled") or (
                kind == "refused" and event.get("job") == "trial")):
            if kind == "trial_started":  # a robot trial (from here or the Experiment tab): its progress here too
                self.play_state.setText(f"실제 로봇: 실행 중 {event.get('title') or ''}")
                self._play_started = time.perf_counter()
                self._play_duration = max(float(event.get("duration_s") or 0.0), 1e-3)
                self.progress.setValue(0)
                self._progress.start(50)
            else:
                self._play_started = None
                snd = sound_text(event.get("sound")) if kind == "trial_finished" else None
                self.play_state.setText({"trial_finished": "실제 로봇: 완료 (③ 결과 탭)" if event.get("completed") else
                                         "실제 로봇: 정지",
                                         "trial_cancelled": "실제 로봇: 실행 취소 (보낸 것 없음)",
                                         "refused": "실제 로봇: 실행 안 함 - " + "; ".join(event.get("reasons") or [])}[kind]
                                        + ("" if snd is None else f" · {snd[0]}"))
                if kind == "trial_finished" and event.get("completed"):
                    self.progress.setValue(1000)
        if kind == "sound":  # the effect sound could not be prepared (the motion runs without it)
            snd = sound_text(event)
            if snd is not None:
                self.play_state.setText(snd[0])
        self.experiment_tab.on_lab_event(event)
        self._refresh_robot_play()

    def _tick_progress(self) -> None:
        if self._play_started is None:
            self._progress.stop()
            return
        frac = (time.perf_counter() - self._play_started) / self._play_duration
        self.progress.setValue(int(min(1.0, frac) * 1000))

    def log(self, text: str) -> None:
        self.status.showMessage(text, 8000)

    # ------------------------------------------------------------------ close
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt naming
        self.video_tab.shutdown()  # a render still running is killed (its folder stays as it is)
        self.sessions.close()  # stops a MuJoCo daemon and its viewer; a robot connection is closed (torque unchanged)
        event.accept()


def _num(v: Any, fmt: str = ".2f") -> str:
    return format(v, fmt) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else ""


_POSE = (("roll_deg", "roll", "°"), ("pitch_deg", "pitch", "°"), ("yaw_deg", "yaw", "°"), ("x_mm", "x", " mm"),
         ("y_mm", "y", " mm"), ("z_mm", "z", " mm"))


def _pose_text(r: dict[str, Any], prefix: str, signed: bool = False) -> str:
    """The key's target (prefix "") or change (prefix "delta_") in words: only the non-zero axes."""
    sign = "+" if signed else ""
    if r.get("track") == "antennas":
        right, left = r.get(f"{prefix}right_deg"), r.get(f"{prefix}left_deg")
        return "" if right is None else f"오른쪽 {right:{sign}.0f}°, 왼쪽 {left:{sign}.0f}°"
    if r.get("track") == "body":
        v = r.get(f"{prefix}yaw_deg")
        return "" if v is None else f"yaw {v:{sign}.0f}°"
    parts = [f"{name} {r[f'{prefix}{key}']:{sign}.0f}{unit}" for key, name, unit in _POSE
             if isinstance(r.get(f"{prefix}{key}"), (int, float)) and abs(r[f"{prefix}{key}"]) >= 0.05]
    return ", ".join(parts) or ("변화 없음" if prefix else "rest (정면)")


def _peak_text(r: dict[str, Any]) -> str:
    if r.get("track") == "antennas":
        return f"{_num(r.get('peak_antenna_dps'), '.0f')} °/s"
    if r.get("track") == "body":
        return f"{_num(r.get('peak_body_dps'), '.0f')} °/s"
    out = f"{_num(r.get('peak_rot_dps'), '.0f')} °/s"
    trans = r.get("peak_trans_mms")
    return out + (f", {trans:.0f} mm/s" if isinstance(trans, (int, float)) and trans >= 0.5 else "")


def _comparison_html(comp: dict[str, Any]) -> str:
    """The expressive / neutral pair in a few Korean lines (computed from the motion files; the full record is in the
    export bundle)."""
    e, n = "expressive", "neutral"

    def both(key: str, fmt: str, unit: str) -> str:
        d = comp.get(key) or {}
        return f"E {_num(d.get(e), fmt)}{unit} · N {_num(d.get(n), fmt)}{unit}"

    def diff(d: dict[str, Any] | None) -> str:
        d = d or {}
        return (f"회전 {_num(d.get('head_rotation_deg'), '.2f')}°, 이동 {_num(d.get('head_translation_mm'), '.2f')} mm, "
                f"안테나 {_num(d.get('antennas_max_deg'), '.2f')}°")

    speeds = comp.get("speeds") or {}
    rot = {v: ((speeds.get(v) or {}).get("head_rotation_dps") or {}) for v in (e, n)}
    used = comp.get("antennas_used") or {}
    easings = comp.get("easings") or {}
    head_n = (easings.get(n) or {}).get("head") or []
    sound = comp.get("sound") or {}
    rows = [
        ("같은 시작 자세", "예" if comp.get("same_start") else f"아니오 ({diff(comp.get('start_difference'))})"),
        ("같은 끝 자세", "예" if comp.get("same_end") else f"아니오 ({diff(comp.get('end_difference'))})"),
        ("길이", both("duration_s", ".2f", " s")),
        ("머리 최대 회전", both("max_head_rotation_deg", ".1f", "°")),
        ("머리 최대 이동", both("max_head_translation_mm", ".1f", " mm")),
        ("머리 회전 속도 (평균 / 최고)",
         f"E {_num(rot[e].get('overall_avg'), '.0f')} / {_num(rot[e].get('max_nominal_peak'), '.0f')} °/s · "
         f"N {_num(rot[n].get('overall_avg'), '.0f')} / {_num(rot[n].get('max_nominal_peak'), '.0f')} °/s"),
        ("머리 보간", f"E {', '.join((easings.get(e) or {}).get('head') or []) or '-'} · N {', '.join(head_n) or '-'}"
                    + (" (일정 속도)" if head_n and set(head_n) <= {"linear"} else "")),
        ("안테나", f"E {'움직임' if used.get(e) else '없음'} · N {'움직임' if used.get(n) else '없음'}"),
        ("소리", f"E {(sound.get(e) or {}).get('file') or '없음'} · N {(sound.get(n) or {}).get('file') or '없음'}"),
    ]
    issues = comp.get("issues") or []
    issue_html = "".join(
        f"<li>{getattr(i, 'code', i.get('code') if isinstance(i, dict) else i)}: "
        f"{getattr(i, 'message', i.get('message') if isinstance(i, dict) else '')}</li>" for i in issues)
    head = ("<p style='color:#1a7f37'><b>비교 조건 충족</b> - 같은 시작·끝 자세, Neutral은 일정 속도·안테나·소리 없음</p>"
            if not issues else
            f"<p style='color:#c0392b'><b>비교 조건 문제 {len(issues)}개</b></p><ul>{issue_html}</ul>")
    return (head + "<table>" + "".join(f"<tr><td><b>{k}</b>&nbsp;&nbsp;</td><td>{v}</td></tr>" for k, v in rows)
            + "</table><p style='color:#6b6a66'>computed: 모션 파일의 설계값 (측정 아님)</p>")


def main(argv: list[str] | None = None, motions_dir: Path | None = None) -> int:
    # The executor thread wakes every 20 ms; a 1 ms GIL switch interval bounds how long GUI drawing can delay it.
    sys.setswitchinterval(0.001)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv if argv is None else argv)
    ko.use_korean_qt(app)
    model = EditorModel(motions_dir)
    win = MainWindow(model)
    # the effect sounds on the robot speaker (user decision 2026-11-14); every other entry (tests, offline runs) keeps
    # the sessions' default "off"
    win.sessions.sound.output = "robot"
    win._sync_sound_controls()
    win.show()
    return int(app.exec())
