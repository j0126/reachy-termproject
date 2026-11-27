"""Motion tab page: expressive and neutral side by side in one video (simulated: offline MuJoCo replay).

The video is rendered by ``gui.video`` in a separate process (``QProcess``), so neither MuJoCo's OpenGL nor the
rendering work runs in this process. Playback is ``QMovie`` on the animated WebP (Qt's own image plugins; no
multimedia module). A finished video of the same two specs (by digest) is shown at once, without rendering again.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from reachy_motion.gui import ko
from reachy_motion.gui import video as vd

SPEEDS = (("1×", 100), ("0.5×", 50), ("0.25×", 25))
NOTE = ("simulated: daemon 없이 MuJoCo 모델로 50 Hz 명령을 다시 재생해 그린 영상입니다 (로봇 아님). "
        "왼쪽 Expressive, 오른쪽 Neutral, 같은 시간축. 표의 V/A는 설계 목표(예상)이며 측정값이 아닙니다. "
        "끝 오차·명령-시뮬 차는 명령을 따라간 정도일 뿐, 감정 표현력이나 의도 인식률이 아닙니다.")


def _f(v: Any, fmt: str) -> str:
    return format(v, fmt) if isinstance(v, (int, float)) else "-"


class MotionVideoTab(QtWidgets.QWidget):
    """Shows the video of the current motion's two variants; ``set_specs`` follows the Motion tab's selection."""

    def __init__(self, root: Path | str = vd.VIDEO_ROOT, search_roots: tuple[Path | str, ...] = vd.SEARCH_ROOTS,
                 fps: int = 25, size: tuple[int, int] = (560, 420), parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.root = Path(root)
        roots = [self.root] + [Path(r) for r in search_roots]
        self.search_roots = tuple(dict.fromkeys(p.resolve() for p in roots))
        self.fps, self.size = fps, size
        self._specs: tuple[Any, Any] | None = None
        self._key: tuple[str, str] | None = None
        self._edited = False
        self._why_not = "모션을 고르세요"
        self.found: vd.FoundVideo | None = None
        self.movie: QtGui.QMovie | None = None
        self._frame = QtCore.QSize()
        self._proc: QtCore.QProcess | None = None
        self._job: vd.VideoJob | None = None
        self._job_t0 = 0.0
        self._resume = False
        self._build()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._progress)
        self._update_buttons()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        self.make_btn = QtWidgets.QPushButton("영상 만들기 (MuJoCo 오프라인, 약 10–15초)")
        self.make_btn.setStyleSheet("font-weight:bold; padding:6px 10px")
        self.make_btn.clicked.connect(self.on_make)
        self.play_btn = QtWidgets.QPushButton("Ⅱ 일시정지")
        self.play_btn.clicked.connect(self.toggle_play)
        self.restart_btn = QtWidgets.QPushButton("|◀ 처음부터")
        self.restart_btn.clicked.connect(self.restart)
        self.step_btn = QtWidgets.QPushButton("▶| 한 프레임")
        self.step_btn.setToolTip("멈추고 한 프레임씩 넘깁니다 (키 자세 확인용)")
        self.step_btn.clicked.connect(self.step)
        self.speed = QtWidgets.QComboBox()
        for text, pct in SPEEDS:
            self.speed.addItem(text, pct)
        self.speed.setToolTip("재생 속도")
        self.speed.currentIndexChanged.connect(self._apply_speed)
        self.open_btn = QtWidgets.QPushButton("브라우저로 열기 (MP4 · 표)")
        self.open_btn.setToolTip("만든 폴더의 index.html을 기본 브라우저로 엽니다 (발표·공유용)")
        self.open_btn.clicked.connect(self.open_html)
        self.folder_btn = QtWidgets.QPushButton("폴더 열기")
        self.folder_btn.clicked.connect(self.open_folder)
        for w in (self.make_btn, self.play_btn, self.restart_btn, self.step_btn, QtWidgets.QLabel("속도"), self.speed,
                  self.open_btn, self.folder_btn):
            row.addWidget(w)
        row.addStretch(1)
        v.addLayout(row)
        self.state = QtWidgets.QLabel("")
        self.state.setWordWrap(True)
        v.addWidget(self.state)
        note = QtWidgets.QLabel(NOTE)
        note.setWordWrap(True)
        note.setStyleSheet("color:#52514e")
        note.setToolTip(ko.glossary_text("MuJoCo", "Expressive", "Neutral", "Valence", "Arousal"))
        v.addWidget(note)
        self.view = QtWidgets.QLabel()
        self.view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        self.view.setMinimumSize(320, 160)
        self.view.setWordWrap(True)
        self.view.setStyleSheet("background:#181e2a; color:#e8ecf3; border-radius:6px; padding:12px")
        v.addWidget(self.view, 1)
        self.table = QtWidgets.QTableWidget()
        self.table.setMaximumHeight(110)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        v.addWidget(self.table)

    # ------------------------------------------------------------------ selection
    def set_specs(self, expressive: Any, neutral: Any, edited: bool = False, why_not: str = "") -> None:
        """The Motion tab's current pair (``why_not`` = why it cannot be rendered, e.g. outside the policy)."""
        self._edited = bool(edited)
        if expressive is None or neutral is None:
            self._specs, self._why_not = None, why_not or "다른 변형 파일이 없습니다"
            key = None
        else:
            self._specs, self._why_not = (expressive, neutral), why_not
            key = (expressive.digest(), neutral.digest())
        if key != self._key:
            self._key = key
            self._show(None if key is None else vd.find_video(key, self.search_roots))
        self._update_buttons()

    @property
    def rendering(self) -> bool:
        return self._proc is not None

    def _update_buttons(self) -> None:
        busy = self.rendering
        can = self._specs is not None and not self._why_not and not busy
        self.make_btn.setEnabled(can)
        self.make_btn.setText("영상 다시 만들기 (약 10–15초)" if self.found is not None else
                              "영상 만들기 (MuJoCo 오프라인, 약 10–15초)")
        self.make_btn.setToolTip("영상을 만드는 중입니다" if busy else (self._why_not or (
            "편집 중인 사본(저장 전)으로 만듭니다" if self._edited else "저장된 모션 파일로 만듭니다")))
        has = self.movie is not None
        for b in (self.play_btn, self.restart_btn, self.step_btn, self.speed):
            b.setEnabled(has)
        for b in (self.open_btn, self.folder_btn):
            b.setEnabled(self.found is not None)

    # ------------------------------------------------------------------ showing a video
    def _stop_movie(self) -> None:
        """Stop and drop the player now and close its WebP file (Windows keeps an open file locked): the reader is
        pointed at no file, the label lets go of the movie (PySide keeps a reference for ``setMovie``), and the
        movie has no Qt parent, so the last reference deletes it at once."""
        if self.movie is not None:
            m, self.movie = self.movie, None
            m.stop()
            m.setFileName("")
            self.view.setMovie(None)
            self.view.clear()
            del m

    def _show(self, found: vd.FoundVideo | None) -> None:
        self._stop_movie()
        self.found = found
        self.table.clear()
        self.table.setRowCount(0)
        if found is None:
            self.view.clear()
            self.view.setText("이 모션(Expressive · Neutral)의 나란히 영상이 아직 없습니다.\n'영상 만들기'를 누르면 "
                              "약 10–15초 뒤 여기서 재생됩니다." if self._specs is not None else self._why_not)
            if not self.rendering:
                self.state.setText("")
            self._update_buttons()
            return
        self.movie = QtGui.QMovie(str(found.webp))  # no parent: see _stop_movie
        self.movie.setCacheMode(QtGui.QMovie.CacheMode.CacheNone)  # every frame cached would be ~2 MB each
        self._frame = QtGui.QImageReader(str(found.webp)).size()
        self.view.setText("")
        self.view.setMovie(self.movie)
        self._rescale()
        self._apply_speed()
        self.movie.stateChanged.connect(self._movie_state)
        self.movie.start()
        if not self.isVisible():
            self.movie.setPaused(True)
            self._resume = True
        self._fill_table(found.entry)
        e = found.entry
        self.state.setText(f"영상: {found.folder.name}  |  simulated, {_f(e.get('video_s'), '.1f')} s, "
                           f"{e.get('fps', '-')} fps" + ("  |  편집 중인 사본으로 만든 영상" if "edited" in
                                                         str(e.get("title", "")) else ""))
        self._update_buttons()

    def _fill_table(self, entry: dict[str, Any]) -> None:
        cols = ["변형", "길이", "예상 Valence / Arousal (설계 목표)", "끝 오차 (sim)", "최대 명령-시뮬 차 (sim)", "의도 (intent)"]
        rows = vd.table_rows(entry)
        t = self.table
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [r["variant"], f"{_f(r['duration_s'], '.2f')} s", f"{_f(r['valence'], '+.2f')} / {_f(r['arousal'], '.2f')}",
                    f"{_f(r['end_error_deg'], '.2f')}° / {_f(r['end_error_mm'], '.2f')} mm", f"{_f(r['max_gap_deg'], '.1f')}°",
                    r["intent"]]
            for j, val in enumerate(vals):
                t.setItem(i, j, QtWidgets.QTableWidgetItem(val))
        tips = {2: ko.glossary_text("Valence", "Arousal"),
                3: "영상 끝에서 시뮬레이션 머리 자세와 계획한 마지막 자세의 차이 (simulated)",
                4: "움직이는 동안 시뮬레이션 머리 자세가 명령 자세에서 가장 많이 벗어난 각도 (simulated)"}
        for j, tip in tips.items():
            t.horizontalHeaderItem(j).setToolTip(tip)
        t.resizeColumnsToContents()

    def _movie_state(self, state: QtGui.QMovie.MovieState) -> None:
        self.play_btn.setText("Ⅱ 일시정지" if state == QtGui.QMovie.MovieState.Running else "▶ 재생")

    def _rescale(self) -> None:
        if self.movie is None or not self._frame.isValid():
            return
        box = self.view.contentsRect().size()
        if box.width() < 10 or box.height() < 10:
            return
        self.movie.setScaledSize(self._frame.scaled(box, QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    def _apply_speed(self, *_: Any) -> None:
        if self.movie is not None:
            self.movie.setSpeed(int(self.speed.currentData() or 100))

    # ------------------------------------------------------------------ player buttons
    def toggle_play(self) -> None:
        m = self.movie
        if m is None:
            return
        if m.state() == QtGui.QMovie.MovieState.Running:
            m.setPaused(True)
        elif m.state() == QtGui.QMovie.MovieState.Paused:
            m.setPaused(False)
        else:
            m.start()

    def restart(self) -> None:
        if self.movie is not None:
            self.movie.stop()
            self.movie.start()

    def step(self) -> None:
        m = self.movie
        if m is None:
            return
        if m.state() == QtGui.QMovie.MovieState.Running:
            m.setPaused(True)
        if m.state() == QtGui.QMovie.MovieState.NotRunning:
            m.start()
            m.setPaused(True)
        if not m.jumpToNextFrame():  # past the last frame: back to the start
            m.stop()
            m.start()
            m.setPaused(True)

    def open_html(self) -> None:
        if self.found is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.found.index_html.resolve())))

    def open_folder(self) -> None:
        if self.found is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.found.folder.resolve())))

    # ------------------------------------------------------------------ rendering (separate process)
    def on_make(self) -> None:
        if self.rendering or self._specs is None or self._why_not:
            return
        e, n = self._specs
        try:
            job = vd.prepare_job(e, n, self.root, edited=self._edited, fps=self.fps, size=self.size)
        except (OSError, ValueError) as exc:
            self.state.setText(f"영상을 준비하지 못했습니다: {exc}")
            return
        cmd = job.command()
        proc = QtCore.QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(Path.cwd()))
        proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        proc.finished.connect(self._done)
        proc.errorOccurred.connect(self._failed_to_start)
        self._proc, self._job, self._job_t0 = proc, job, time.perf_counter()
        proc.start()
        self._timer.start(500)
        self._progress()
        self._update_buttons()

    def _progress(self) -> None:
        if self._job is not None:
            self.state.setText(f"{self._job.name}: Expressive · Neutral 영상을 만드는 중 ... "
                               f"{time.perf_counter() - self._job_t0:.0f} s (별도 프로세스, MuJoCo 오프라인 재생)")

    def _failed_to_start(self, err: QtCore.QProcess.ProcessError) -> None:
        if err == QtCore.QProcess.ProcessError.FailedToStart:
            self._finish_job(False, "영상 프로세스를 시작하지 못했습니다")

    def _done(self, code: int, _status: QtCore.QProcess.ExitStatus) -> None:
        out = bytes(self._proc.readAll()).decode("utf-8", errors="replace") if self._proc is not None else ""
        tail = " / ".join(line for line in out.strip().splitlines()[-2:] if line.strip())
        self._finish_job(code == 0, tail)

    def _finish_job(self, ok: bool, text: str) -> None:
        self._timer.stop()
        job, proc = self._job, self._proc
        self._proc, self._job = None, None
        if proc is not None:
            proc.deleteLater()
        if job is None:
            return
        found = vd.find_video(job.digests, [job.out_dir.parent]) if ok else None
        if found is None or found.folder.resolve() != job.out_dir.resolve():
            if job.out_dir.is_dir() and {p.name for p in job.out_dir.iterdir()} <= {"specs"}:
                shutil.rmtree(job.out_dir, ignore_errors=True)  # only our own spec copies: nothing was rendered
            self.state.setText(f"영상을 만들지 못했습니다: {text or '알 수 없는 오류'}")
            self._update_buttons()
            return
        seconds = time.perf_counter() - self._job_t0
        if self._key == job.digests:
            self._show(found)
            self.state.setText(self.state.text() + f"  |  {seconds:.0f} s 걸림")
        else:
            self.state.setText(f"{job.name} 영상을 만들었습니다 ({found.folder.name}). 지금 고른 모션과 달라 표시하지 "
                               "않았습니다.")
        self._update_buttons()

    def shutdown(self) -> None:
        """Window closing: stop the player and a running render (its partial folder is left as it is)."""
        self._stop_movie()
        if self._proc is not None:
            self._proc.kill()
            self._proc.waitForFinished(3000)
            self._proc = None
        self._timer.stop()

    # ------------------------------------------------------------------ Qt events
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._rescale()

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._rescale)
        if self._resume and self.movie is not None:
            self.movie.setPaused(False)
        self._resume = False

    def hideEvent(self, event: QtGui.QHideEvent) -> None:  # noqa: N802
        super().hideEvent(event)
        if self.movie is not None and self.movie.state() == QtGui.QMovie.MovieState.Running:
            self.movie.setPaused(True)  # no decoding while nobody looks
            self._resume = True


__all__ = ["MotionVideoTab", "SPEEDS"]
