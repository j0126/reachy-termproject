"""Interaction state shared by both speech systems, the GUI and the motion module.

Phase 1 only *tracks* states (recording -> LISTENING, playback -> SPEAKING, ...).
The transition policy (VAD, barge-in, ...) is added by the engines in Phase 2-4;
every transition is logged as a ``state`` event so timing can be analysed.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from enum import Enum
from typing import Any

from ..evaluation.event_logger import EventLogger


class InteractionState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    ERROR = "ERROR"


StateListener = Callable[[InteractionState, InteractionState, dict[str, Any]], None]


class StateTracker:
    def __init__(self, events: EventLogger | None = None, initial: InteractionState = InteractionState.IDLE) -> None:
        self._state = initial
        self._events = events
        self._lock = threading.Lock()
        self._listeners: list[StateListener] = []

    @property
    def state(self) -> InteractionState:
        with self._lock:
            return self._state

    def set(self, new: InteractionState, reason: str = "", t_ns: int | None = None, **fields: Any) -> bool:
        """Switch state; returns ``False`` (and logs nothing) if already in ``new``.

        ``t_ns`` timestamps the transition when it is logged after the fact
        (e.g. SPEAKING at the exact first push to the robot).
        """
        with self._lock:
            old = self._state
            if old is new:
                return False
            self._state = new
            listeners = list(self._listeners)
        record = {"from": old.value, "to": new.value, "reason": reason, **fields}
        if self._events is not None:
            record = self._events.log("state", t_ns=t_ns, **record)
        for listener in listeners:
            listener(old, new, record)
        return True

    def add_listener(self, listener: StateListener) -> None:
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: StateListener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
