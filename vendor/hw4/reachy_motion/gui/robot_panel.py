"""Real-robot box (PySide6): IP / port, the robot actions, one status line. One click per action (user decision
2026-11-12): the click is the approval; the work runs in ``gui.robot_session.RobotLabSession``. Since 2026-11-13 the
main window shows it above the tabs (``use_wide_layout``: one row), so connect / Wake-up / torque-off are at hand in
every tab; the Experiment tab keeps the handlers.
"""

from __future__ import annotations

import html
from typing import Any

from PySide6 import QtCore, QtWidgets

from reachy_motion.safety import gui_approval as ga
from reachy_motion.settings import ROBOT_DAEMON_PORT, ROBOT_HOST

STATE_SHORT = {"disconnected": "연결 안 됨", "rest_torque_on": "연결됨 · Wake-up 필요", "at_init": "INIT 대기 · 시행 가능",
               "holding": "정지됨 · Wake-up 또는 토크 OFF"}
STATE_COLOR = {"disconnected": "#6b6a66", "rest_torque_on": "#b26b00", "at_init": "#1a7f37", "holding": "#c0392b"}
TORQUE_COLOR = {"ON": "#b26b00", "OFF": "#1a7f37", "UNKNOWN": "#c0392b"}
HINT = "연결 = 토크 켜짐 · 연결 끊기 = 토크 유지 · 다른 앱·CLI와 동시에 쓰지 않기"


def switches_off() -> list[str]:
    """The robot switches that are off now (empty in normal use)."""
    import inspect

    from reachy_motion import robot_r1
    from reachy_motion.backends import sdk

    out = []
    if not ga.enabled():
        out.append("GUI_APPROVAL_ENABLED=False")
    if robot_r1.ROBOT_RUN_ENABLED is not True:
        out.append("ROBOT_RUN_ENABLED=False")
    if "NotImplementedError(ROBOT_NOT_IN_PHASE2)" in inspect.getsource(sdk.SdkBackend.connect):
        out.append("SdkBackend.connect 잠김")
    return out


def ask_torque_off(parent: QtWidgets.QWidget | None) -> bool:
    """One question: the head is supported (the torque-off lets it fall)."""
    ans = QtWidgets.QMessageBox.question(parent, "토크 OFF", "머리를 손으로 받쳤나요? 토크를 끄면 머리가 내려갑니다.")
    return ans == QtWidgets.QMessageBox.StandardButton.Yes


def _link_short(link: str) -> str:
    for key in ("열림", "응답 없음", "닫힘", "없음"):
        if key in link:
            return key
    return link


class RobotPanel(QtWidgets.QGroupBox):
    """IP / port, the robot actions, the robot's state in one line. Emits what was clicked."""

    status_clicked = QtCore.Signal()
    connect_clicked = QtCore.Signal()
    wake_clicked = QtCore.Signal()
    torque_off_clicked = QtCore.Signal()
    disconnect_clicked = QtCore.Signal()
    target_changed = QtCore.Signal(str, int)

    def __init__(self, host: str = ROBOT_HOST, port: int = ROBOT_DAEMON_PORT,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("실제 로봇", parent)
        self.state = "disconnected"
        self.view: dict[str, Any] = {"torque": "UNKNOWN", "torque_basis": "아직 읽지 않음", "link": "없음"}
        self.procedure = ""
        self.busy = False
        self.last_socket: dict[str, Any] | None = None
        self.last_r0: dict[str, Any] | None = None
        v = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.host_edit = QtWidgets.QLineEdit(host)
        self.host_edit.setFixedWidth(130)
        self.port_edit = QtWidgets.QSpinBox()
        self.port_edit.setRange(1, 65535)
        self.port_edit.setValue(int(port))
        self.connect_btn = QtWidgets.QPushButton("연결")
        self.wake_btn = QtWidgets.QPushButton("Wake-up")
        self.off_btn = QtWidgets.QPushButton("토크 OFF")
        self.disc_btn = QtWidgets.QPushButton("연결 끊기")
        self.status_btn = QtWidgets.QPushButton("새로고침")
        self.connect_btn.setToolTip("로봇 상태를 읽고 연결합니다. 토크가 꺼져 있으면 지금 자세에서 토크를 켭니다 (움직이지 않음).")
        self.wake_btn.setToolTip("SDK Wake-up: INIT까지 2초, roll 20° 갔다 오기. 정지 뒤 INIT로 돌아갈 때도 씁니다.")
        self.off_btn.setToolTip("머리를 받친 뒤 토크를 끄고 연결을 닫습니다.")
        self.disc_btn.setToolTip("연결만 닫습니다. 로봇은 토크·자세를 그대로 유지합니다.")
        self.status_btn.setToolTip("로봇 상태만 다시 읽습니다 (명령 없음).")
        for w in (QtWidgets.QLabel("IP"), self.host_edit, QtWidgets.QLabel("포트"), self.port_edit):
            row.addWidget(w)
        row.addStretch(1)
        v.addLayout(row)
        self._row = row
        self._brow = brow = QtWidgets.QHBoxLayout()  # the buttons on their own row (a narrow column); see use_wide_layout
        for b, sig in ((self.connect_btn, self.connect_clicked), (self.wake_btn, self.wake_clicked),
                       (self.off_btn, self.torque_off_clicked), (self.disc_btn, self.disconnect_clicked),
                       (self.status_btn, self.status_clicked)):
            b.clicked.connect(sig.emit)
            brow.addWidget(b)
        brow.addStretch(1)
        v.addLayout(brow)
        self.state_label = QtWidgets.QLabel()
        v.addWidget(self.state_label)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:#52514e")
        self.info.setVisible(False)  # shown once there is something to say (no empty lines in a small window)
        v.addWidget(self.info)
        self.message = QtWidgets.QLabel("")
        self.message.setWordWrap(True)
        self.message.setVisible(False)
        v.addWidget(self.message)
        hint = QtWidgets.QLabel(HINT)
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8a5a00")
        hint.setToolTip("토크가 꺼진 로봇은 연결하면 지금 자세에서 토크가 켜집니다(움직이지 않음). 연결 끊기와 창 닫기는 토크를 "
                        "끄지 않습니다. 같은 로봇을 대시보드 앱·다른 PC·CLI로 동시에 조작하지 마세요.")
        v.addWidget(hint)
        self.switch_warning = QtWidgets.QLabel("")
        self.switch_warning.setStyleSheet("color:#c0392b")
        v.addWidget(self.switch_warning)
        self.host_edit.editingFinished.connect(self._target_edited)
        self.port_edit.valueChanged.connect(lambda *_: self._target_edited())
        self.set_state("disconnected")

    def use_wide_layout(self) -> None:
        """One row: IP, port, the buttons and the robot's state (the box above the main window's tabs)."""
        row, v = self._row, self.layout()
        last = row.itemAt(row.count() - 1)
        if last is not None and last.spacerItem() is not None:
            row.takeAt(row.count() - 1)
        row.addSpacing(10)
        for b in (self.connect_btn, self.wake_btn, self.off_btn, self.disc_btn, self.status_btn):
            self._brow.removeWidget(b)
            row.addWidget(b)
        v.removeItem(self._brow)
        v.removeWidget(self.state_label)
        row.addSpacing(10)
        row.addWidget(self.state_label)
        row.addStretch(1)
        v.setContentsMargins(8, 2, 8, 4)
        v.setSpacing(2)

    # ------------------------------------------------------------------ data
    def host(self) -> str:
        return self.host_edit.text().strip()

    def port(self) -> int:
        return int(self.port_edit.value())

    def _target_edited(self) -> None:
        self.target_changed.emit(self.host(), self.port())

    def set_state(self, state: str, view: dict[str, Any] | None = None, procedure: str | None = None) -> None:
        """``view``: the session's torque_view (torque ON / OFF / UNKNOWN, why, the SDK link)."""
        self.state = state
        if view and view.get("torque"):
            self.view = {k: str(view.get(k) or "-") for k in ("torque", "torque_basis", "link")}
        if procedure is not None:
            self.procedure = procedure
        v = self.view
        self.state_label.setText(
            f"<b style='color:{STATE_COLOR.get(state, '#6b6a66')}'>{STATE_SHORT.get(state, state)}</b>"
            f" &nbsp;·&nbsp; 토크 <b style='color:{TORQUE_COLOR.get(v['torque'], '#c0392b')}'>{v['torque']}</b>"
            f" &nbsp;·&nbsp; 연결 {_link_short(v['link'])}")
        self.state_label.setToolTip(f"토크: {v['torque_basis']}\nSDK 연결: {v['link']}")
        if self.procedure:
            self.show_message("머리를 손으로 받치고 '토크 OFF' (안 되면 전원 스위치). 자세한 절차: 마우스를 올려 보세요.")
            self.message.setToolTip(self.procedure)
        self.refresh_buttons()

    def set_busy(self, busy: bool) -> None:
        if busy != self.busy:
            self.busy = busy
            self.refresh_buttons()

    def refresh_buttons(self) -> None:
        st, busy = self.state, self.busy
        connected = st != "disconnected"
        self.host_edit.setEnabled(not connected)
        self.port_edit.setEnabled(not connected)
        self.status_btn.setEnabled(not busy)
        self.connect_btn.setEnabled(not busy and not connected)
        self.wake_btn.setEnabled(not busy and st in ("rest_torque_on", "holding"))
        self.off_btn.setEnabled(not busy and connected)
        self.disc_btn.setEnabled(not busy and connected)
        off = switches_off()
        self.switch_warning.setText("실제 로봇 경로 꺼짐: " + ", ".join(off) if off else "")
        self.switch_warning.setVisible(bool(off))

    @staticmethod
    def _num(v: Any, fmt: str = ".1f") -> str:
        if isinstance(v, (list, tuple)):
            return " / ".join(RobotPanel._num(x, fmt) for x in v)
        return format(v, fmt) if isinstance(v, (int, float)) and not isinstance(v, bool) else "-"

    def show_status(self, e: dict[str, Any]) -> None:
        """A ``robot_status`` event: what the robot reported (GET) and, while connected, the SDK socket."""
        n = self._num
        if e.get("r0") is not None or e.get("r0_error"):
            self.info.setVisible(True)
        if e.get("r0") is not None:
            self.last_r0 = r = e["r0"]
            hp = r.get("head_pose") or {}
            reasons = list(r.get("block_reasons") or (["R0 결과에 차단 표시가 있습니다"] if r.get("blocking") else []))
            self.info.setText(f"daemon {r.get('daemon_state')} v{r.get('daemon_version')} · 모터 {r.get('motor_mode')} · "
                              f"머리 pitch {n(hp.get('pitch_deg'))}° roll {n(hp.get('roll_deg'))}° yaw "
                              f"{n(hp.get('yaw_deg'))}° · 안테나 {n(r.get('antennas_deg'))}°"
                              + (f" · <span style='color:#c0392b'>연결 불가: {html.escape(reasons[0])}</span>"
                                 if reasons else "")
                              + (f" · <span style='color:#b9770e'>머리가 관절 한계를 {n(r.get('limp_past_limits_deg'))}° "
                                 "넘어 쉬는 중: 연결할 수 있음 (토크가 켜질 때 조금 움직일 수 있으니 머리를 받친 채)</span>"
                                 if r.get("limp_past_limits_deg") is not None and not reasons else ""))
            self.info.setToolTip("\n".join(reasons))
        elif e.get("r0_error"):
            self.info.setText(f"<span style='color:#c0392b'>상태를 읽지 못함: {e['r0_error']}</span>")
        sock = e.get("socket")
        if sock is not None:
            self.last_socket = sock
        elif self.state == "disconnected":
            self.last_socket = None
        if e.get("torque"):
            self.set_state(e.get("state") or self.state, e, e.get("operator_procedure"))
        else:
            self.refresh_buttons()

    def show_message(self, text: str, error: bool = True) -> None:
        self.message.setStyleSheet(f"color:{'#c0392b' if error else '#1a7f37'}")
        self.message.setText(text)
        self.message.setVisible(bool(text))
        self.message.setToolTip("")


__all__ = ["RobotPanel", "ask_torque_off", "switches_off", "HINT"]
