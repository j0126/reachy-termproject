"""Motion tab of the dashboard (Phase 4.5): motors, motion condition, safety stops.

Three separate controls (docs/PHASE4_5_DESIGN.md section 9):

* Motor ON / Normal shutdown - torque on + ramp to neutral / neutral -> sleep pose -> torque off
  (never cuts the torque if a check fails; the controller goes to FAULT instead);
* Motion ON / OFF - expressive motion vs. the neutral hold (torque stays on): the Phase 5 condition;
* Motion Stop (hold the last pose, torque on) and EMERGENCY TORQUE OFF (REST, its own thread,
  independent of the motion queue; F12).

The controller is created only when Motor ON is pressed, so sessions that never use the motors
are exactly the Phase 4 sessions. Settings are locked while a conversation runs.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ..config import AppConfig
from ..motion.controller import (
    ACTIVE,
    ESTOP,
    FAULT,
    HOLD,
    MOVING_MODES,
    SHUTTING_DOWN,
    STARTING,
    STOPPED,
    TORQUE_OFF,
    TORQUE_ON_MODES,
    MotionController,
)

log = logging.getLogger(__name__)

MODE_COLORS = {
    TORQUE_OFF: "#6b7280", STARTING: "#d97706", HOLD: "#0d9488", ACTIVE: "#16a34a", STOPPED: "#ca8a04",
    SHUTTING_DOWN: "#d97706", FAULT: "#b91c1c", ESTOP: "#7f1d1d",
}
SPEAKING_ITEMS = (("custom", "Custom (PCM envelope, our limits)"), ("sdk", "SDK wobbling (comparison only)"),
                  ("off", "OFF"))
GESTURE_ITEMS = (("nod", "Nod"), ("antenna", "Antenna"), ("off", "OFF"))
RATE_ITEMS = ((50.0, "50 Hz"), (25.0, "25 Hz"))


class MotionPanel(QtWidgets.QWidget):
    estop_finished = QtCore.Signal(str)

    def __init__(self, cfg: AppConfig, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        m = cfg.motion
        self.controller: MotionController | None = None
        #: creates the controller on demand (dashboard); None while disconnected
        self.provider: Callable[[], MotionController | None] | None = None
        #: torque off over REST when no controller exists yet (robot connected); None otherwise
        self.fallback_estop: Callable[[], Any] | None = None
        self.connected = False
        self.running = False
        self.lock_when_running = m.lock_during_conversation  # experiments: settings fixed during a conversation
        self.suppress_pose_prompt = False  # automatic Motor ON (dashboard): an unknown pose is reported, not asked
        self.notice = ""  # why the motors were not started automatically (kept until the next Motor ON)
        self._prompting = False
        self._pending: dict[str, Any] = {}
        self._mode_style = ""

        layout = QtWidgets.QVBoxLayout(self)
        self.mode_label = QtWidgets.QLabel()
        self.mode_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = self.mode_label.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        self.mode_label.setFont(font)
        layout.addWidget(self.mode_label)
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.info_label)
        self.error_label = QtWidgets.QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color:#b91c1c;")
        self.error_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.error_label)

        motors = QtWidgets.QGroupBox("Motors")
        ml = QtWidgets.QVBoxLayout(motors)
        self.motor_on_btn = QtWidgets.QPushButton("Motor ON (torque on → neutral)")
        self.motor_on_btn.setToolTip("Turns the torque on (the daemon holds the present pose), then ramps to the "
                                     "neutral pose within the speed limits. Asks for confirmation first.")
        self.motor_off_btn = QtWidgets.QPushButton("Normal shutdown (neutral → sleep → torque off)")
        self.motor_off_btn.setToolTip("Torque goes off only after the head is verified at the sleep pose; "
                                      "on any failure the torque stays ON and the controller reports FAULT.")
        ml.addWidget(self.motor_on_btn)
        ml.addWidget(self.motor_off_btn)
        layout.addWidget(motors)

        cond = QtWidgets.QGroupBox("Motion condition (fixed while a conversation runs)" if m.lock_during_conversation
                                   else "Motion condition")
        form = QtWidgets.QFormLayout(cond)
        self.motion_box = QtWidgets.QCheckBox("Motion ON")
        self.motion_box.setChecked(m.enabled)
        self.motion_box.setToolTip("Unchecked = Motion OFF: torque ON, neutral pose held (Phase 5 control "
                                   "condition). Not the same as Torque OFF.")
        self.intensity_spin = QtWidgets.QDoubleSpinBox()
        self.intensity_spin.setRange(0.0, 1.5)
        self.intensity_spin.setSingleStep(0.05)
        self.intensity_spin.setValue(m.intensity)
        self.speaking_box = QtWidgets.QComboBox()
        for key, title in SPEAKING_ITEMS:
            self.speaking_box.addItem(title, key)
        if not m.allow_sdk_wobble:  # bypasses the controller's limits: separate approval
            item = self.speaking_box.model().item(1)
            item.setEnabled(False)
            item.setToolTip("[motion] allow_sdk_wobble = false")
        self.speaking_box.setCurrentIndex(next(i for i, (k, _t) in enumerate(SPEAKING_ITEMS)
                                               if k == m.speaking_motion))
        self.gesture_box = QtWidgets.QComboBox()
        for key, title in GESTURE_ITEMS:
            self.gesture_box.addItem(title, key)
        self.gesture_box.setCurrentIndex(next(i for i, (k, _t) in enumerate(GESTURE_ITEMS)
                                              if k == m.speech_started_gesture))
        self.rate_box = QtWidgets.QComboBox()
        for hz, title in RATE_ITEMS:
            self.rate_box.addItem(title, hz)
        self.rate_box.setCurrentIndex(0 if m.rate_hz >= 37.5 else 1)
        form.addRow(self.motion_box)
        form.addRow("Intensity", self.intensity_spin)
        form.addRow("Speaking motion", self.speaking_box)
        form.addRow("User speech start", self.gesture_box)
        form.addRow("Control rate", self.rate_box)
        layout.addWidget(cond)

        safety = QtWidgets.QGroupBox("Safety")
        sl = QtWidgets.QVBoxLayout(safety)
        row = QtWidgets.QHBoxLayout()
        self.stop_btn = QtWidgets.QPushButton("Motion Stop (hold pose) [F11]")
        self.stop_btn.setToolTip("Stops sending immediately; the robot holds its last target with the torque ON.")
        self.resume_btn = QtWidgets.QPushButton("Resume")
        row.addWidget(self.stop_btn)
        row.addWidget(self.resume_btn)
        sl.addLayout(row)
        self.estop_btn = QtWidgets.QPushButton("EMERGENCY TORQUE OFF [F12]")
        self.estop_btn.setMinimumHeight(44)
        self.estop_btn.setStyleSheet("QPushButton{background:#b91c1c;color:white;font-weight:bold;}"
                                     "QPushButton:disabled{background:#d1d5db;color:#6b7280;}")
        self.estop_btn.setToolTip("Torque off now over REST (WebSocket if REST fails), from its own thread, "
                                  "independent of the motion queue. The head drops to its rest position.")
        sl.addWidget(self.estop_btn)
        layout.addWidget(safety)
        layout.addStretch(1)

        self.motor_on_btn.clicked.connect(self.request_motor_on)
        self.motor_off_btn.clicked.connect(self.request_motor_off)
        self.motion_box.toggled.connect(lambda on: self._setting("motion", on))
        self.intensity_spin.valueChanged.connect(lambda v: self._setting("intensity", float(v)))
        self.speaking_box.currentIndexChanged.connect(lambda _i: self._setting("speaking",
                                                                               self.speaking_box.currentData()))
        self.gesture_box.currentIndexChanged.connect(lambda _i: self._setting("gesture",
                                                                              self.gesture_box.currentData()))
        self.rate_box.currentIndexChanged.connect(lambda _i: self._setting("rate", self.rate_box.currentData()))
        self.stop_btn.clicked.connect(self.motion_stop)
        self.resume_btn.clicked.connect(self.resume)
        self.estop_btn.clicked.connect(self.emergency_torque_off)
        self.estop_finished.connect(self._on_estop_finished)
        self._shortcuts = []
        for key, slot in (("F11", self.motion_stop), ("F12", self.emergency_torque_off)):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)
        self.refresh()

    # ------------------------------------------------------------- external
    def set_controller(self, controller: MotionController | None) -> None:
        self.controller = controller
        if controller is not None:
            for name, value in self._pending.items():
                self._apply(controller, name, value)
        self._pending.clear()
        self.refresh()

    def set_connected(self, connected: bool) -> None:
        self.connected = connected
        self.refresh()

    def set_running(self, running: bool) -> None:
        self.running = running
        self.refresh()

    def _locked_ui(self) -> bool:
        return self.running and self.lock_when_running

    def torque_may_be_on(self) -> bool:
        c = self.controller
        return c is not None and (c.mode in TORQUE_ON_MODES or (c.mode == FAULT and c.motor_mode_seen != "disabled"))

    # -------------------------------------------------------------- actions
    def request_motor_on(self) -> None:
        if not self.connected or self._locked_ui():
            return
        self.notice = ""
        sim = self.controller.sink.simulated if self.controller is not None else None
        target = "SIMULATED sink (nothing is sent)" if sim else "the ROBOT"
        intensity = self.intensity_spin.value()
        answer = QtWidgets.QMessageBox.question(
            self, "Motor ON",
            f"Turn the motor torque ON on {target}?\n\n"
            "The head moves from its present pose to the neutral pose within the speed limits "
            "(about 6 s from the sleep pose).\n"
            f"Motion: {'ON' if self.motion_box.isChecked() else 'OFF (neutral hold)'}, intensity {intensity:g}.\n\n"
            "Keep hands and objects away from the head. F11 = Motion Stop, F12 = Emergency Torque OFF.")
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        controller = self.controller
        if controller is None and self.provider is not None:
            controller = self.provider()
        if controller is None:
            self.error_label.setText("motion controller unavailable (not connected?)")
            return
        controller.motor_on()
        self.refresh()

    def request_motor_off(self) -> None:
        if self.controller is not None:
            self.controller.motor_off()
            self.refresh()

    def motion_stop(self) -> None:
        if self.controller is not None:
            self.controller.motion_stop()
            self.refresh()

    def resume(self) -> None:
        if self.controller is not None and not self._locked_ui():
            self.controller.resume()
            self.refresh()

    def emergency_torque_off(self) -> None:
        controller, fallback = self.controller, self.fallback_estop
        if controller is None and fallback is None:
            self.error_label.setText("Emergency Torque OFF: not connected to a robot")
            return
        self.error_label.setText("EMERGENCY TORQUE OFF requested ...")

        def worker() -> None:
            if controller is not None:
                result = controller.emergency_torque_off()
            else:
                try:
                    fallback()
                    result = "ok via rest (no motion controller)"
                except Exception as exc:  # noqa: BLE001
                    result = f"FAILED: {exc!r}"
            self.estop_finished.emit(result)

        threading.Thread(target=worker, name="gui-estop", daemon=True).start()

    def _on_estop_finished(self, result: str) -> None:
        self.error_label.setText(f"Emergency Torque OFF: {result}")
        self.refresh()

    def _setting(self, name: str, value: Any) -> None:
        if self.controller is None:
            self._pending[name] = value
            return
        try:
            self._apply(self.controller, name, value)
        except Exception as exc:  # noqa: BLE001 - locked / refused: show it, keep the old value
            self.error_label.setText(str(exc))
        self.refresh()

    @staticmethod
    def _apply(c: MotionController, name: str, value: Any) -> None:
        {"motion": c.set_motion, "intensity": c.set_intensity, "speaking": c.set_speaking_motion,
         "gesture": c.set_speech_gesture, "rate": c.set_rate}[name](value)

    # -------------------------------------------------------------- display
    @staticmethod
    def _set_text(label: QtWidgets.QLabel, text: str) -> None:
        if label.text() != text:  # the panel refreshes 4x per second: no relayout when nothing changed
            label.setText(text)

    def refresh(self) -> None:
        c = self.controller
        mode = c.mode if c is not None else TORQUE_OFF
        title = mode if c is not None else "TORQUE_OFF (no controller)"
        if c is not None and c.sink.simulated:
            title += "  [SIMULATED]"
        self._set_text(self.mode_label, title)
        style = f"background:{MODE_COLORS.get(mode, '#6b7280')}; color:white; border-radius:6px; padding:4px;"
        if style != self._mode_style:  # re-styling re-polishes the widget: only when the colour changes
            self._mode_style = style
            self.mode_label.setStyleSheet(style)
        if c is None:
            self._set_text(self.info_label, "Motors untouched (Phase 4 condition). Motor ON creates the controller.")
        else:
            s = c.status()
            self._set_text(self.info_label,
                f"state {s['state']}  •  label {s['label']}  •  motor mode (REST) {s['motor_mode'] or '-'}\n"
                f"sink {s['sink']}  •  {s['rate_hz']:g} Hz  •  sends {s['sends']}  •  faults {s['faults']}  •  "
                f"guard {'on' if s['guard'] else 'OFF'}  •  wobbling {'on' if s['wobbling'] else 'off'}  •  "
                f"torso yaw {s['torso_yaw_deg']:g}°")
            err = s["last_error"] or s["health_error"] or ""
            if s["estop_result"]:
                err = f"Emergency Torque OFF: {s['estop_result']}" + (f"\n{err}" if err else "")
            if err:
                self._set_text(self.error_label, err)
            elif mode not in (FAULT, ESTOP) and not self.error_label.text().startswith("EMERGENCY"):
                self._set_text(self.error_label, self.notice)
        robot_or_ctl = self.connected or c is not None
        self.motor_on_btn.setEnabled(self.connected and not self._locked_ui() and mode in (TORQUE_OFF, FAULT, ESTOP))
        self.motor_off_btn.setEnabled(c is not None and mode in (STARTING, HOLD, ACTIVE, STOPPED, FAULT))
        for w in (self.motion_box, self.intensity_spin, self.speaking_box, self.gesture_box, self.rate_box):
            w.setEnabled(not self._locked_ui())
        self.stop_btn.setEnabled(c is not None and mode in MOVING_MODES)
        self.resume_btn.setEnabled(c is not None and mode == STOPPED and not self._locked_ui())
        self.estop_btn.setEnabled(robot_or_ctl and (c is not None or self.fallback_estop is not None))
        if c is not None and c.needs_confirmation and not self._prompting and not self.suppress_pose_prompt:
            QtCore.QTimer.singleShot(0, self._confirm_unknown_pose)

    def _confirm_unknown_pose(self) -> None:
        c = self.controller
        if c is None or not c.needs_confirmation or self._prompting:
            return
        self._prompting = True
        detail, c.needs_confirmation = c.needs_confirmation, None
        try:
            answer = QtWidgets.QMessageBox.warning(
                self, "Motor ON: unknown initial pose",
                f"The head is not near the sleep or the neutral pose:\n{detail}\n\n"
                "Turn the torque on anyway? The head then ramps from this pose to neutral.",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No)
            if answer == QtWidgets.QMessageBox.StandardButton.Yes:
                c.motor_on(force=True)
        finally:
            self._prompting = False
