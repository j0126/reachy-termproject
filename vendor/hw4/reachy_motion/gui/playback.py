"""Qt-free playback controller of the Motion tab's Mock preview.

A ``MotionService`` owns the backend on its own worker thread; the GUI thread only calls the non-blocking
``play`` / ``stop`` / ``reset`` here and receives events through ``on_event`` (called on the worker thread - the Qt
layer re-emits them as queued signals). Command execution and log collection therefore never run on the GUI thread.

Backends listed in the Motion tab: ``mock`` (this controller, the default); ``sim`` (the local MuJoCo, played by
``gui.lab.LabSession`` - not by this controller); ``robot`` is listed disabled: it needs the Phase 4 approval gate
and is never offered by the GUI.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from reachy_motion.library.pose import REST
from reachy_motion.library.spec import MotionSpec
from reachy_motion.safety.policy import DEFAULT_POLICY, SafetyPolicy

#: backend name -> (label, enabled in Phase 2, reason when disabled)
BACKENDS: dict[str, tuple[str, bool, str]] = {
    "mock": ("Mock (하드웨어 없음)", True, ""),
    "sim": ("MuJoCo (로컬, viewer 창)", True, ""),
    "robot": ("실제 로봇 (로봇 상자에서)", False, "실제 로봇은 창 위쪽 로봇 상자에서 연결하고, 버튼 하나가 동작 하나의 승인입니다"),
}


class PlaybackController:
    def __init__(self, backend_name: str = "mock", policy: SafetyPolicy = DEFAULT_POLICY,
                 on_event: Callable[[dict[str, Any]], None] | None = None, backend: Any = None) -> None:
        label, enabled, reason = BACKENDS.get(backend_name, (backend_name, False, "unknown backend"))
        if not enabled:
            raise ValueError(f"backend {label!r} is not available: {reason}")
        if backend_name != "mock":
            raise ValueError("PlaybackController drives the Mock only; MuJoCo playback goes through gui.lab.LabSession")
        from reachy_motion.service import MotionService

        if backend is None:
            from reachy_motion.backends.mock import MockBackend

            backend = MockBackend(structure="robot")
        self.backend = backend
        self.policy = policy
        self._on_event = on_event
        self._lock = threading.Lock()
        self.current: Any = None
        self.last_result: Any = None
        self.backend.connect()
        self.service = MotionService(self.backend, library=None, policy=policy, record_present=True)
        self.service.on_event(self._forward)
        self.service.start()

    # ------------------------------------------------------------------ actions (non-blocking)
    def play(self, spec: MotionSpec) -> Any:
        """Queue ``spec`` (interrupting what runs); a spec outside the initial policy comes back REJECTED."""
        handle = self.service.play(spec, variant=spec.variant, interrupt=True)
        with self._lock:
            self.current = handle
        return handle

    def stop(self) -> None:
        """Stop the running motion at the next tick (the head holds the last commanded pose)."""
        with self._lock:
            h = self.current
        if h is not None:
            h.cancel()

    def reset(self) -> None:
        """Stop everything and blend back to the rest pose (a checked transition)."""
        self.service.stop(blend_to_rest=True)

    def close(self) -> None:
        try:
            self.service.close()
        finally:
            self.backend.close()

    # ------------------------------------------------------------------ state
    @property
    def command_count(self) -> int:
        return int(getattr(self.backend, "motor_command_count", len(getattr(self.backend, "commands", []))))

    @property
    def rest(self) -> Any:
        return REST

    def _forward(self, event: dict[str, Any]) -> None:
        handle = event.get("handle")
        if event.get("kind") in ("finished", "cancelled", "preempted", "failed") and handle is not None:
            self.last_result = getattr(handle, "result", None)
        if self._on_event is not None:
            self._on_event(event)
