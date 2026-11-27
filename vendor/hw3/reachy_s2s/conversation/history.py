"""Client-side conversation history (the server never stores the conversation).

Each exchange keeps the exact request step for the user turn and the model
steps exactly as the API returned them (Gemini Interactions API: thinking
signatures must be resent unchanged), plus plain text, which is what
OpenAI-style chat APIs get (``chat_messages``) and what is displayed / logged.
Trimming always drops a whole exchange so user and model turns stay paired.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Exchange:
    turn_id: int
    user_text: str
    model_text: str
    user_step: dict[str, Any]
    model_steps: list[dict[str, Any]] = field(default_factory=list)


class ConversationHistory:
    def __init__(self, max_exchanges: int = 20) -> None:
        self.max_exchanges = max_exchanges
        self._exchanges: list[Exchange] = []
        self._lock = threading.Lock()
        self.resets = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._exchanges)

    def api_input(self) -> list[dict[str, Any]]:
        with self._lock:
            steps: list[dict[str, Any]] = []
            for ex in self._exchanges:
                steps.append(ex.user_step)
                steps.extend(ex.model_steps)
            return steps

    def chat_messages(self) -> list[dict[str, Any]]:
        """Plain-text history for OpenAI-style chat APIs (provider independent)."""
        with self._lock:
            messages: list[dict[str, Any]] = []
            for ex in self._exchanges:
                messages.append({"role": "user", "content": ex.user_text})
                messages.append({"role": "assistant", "content": ex.model_text})
            return messages

    def add(self, exchange: Exchange) -> int:
        """Append an exchange; returns how many old exchanges were trimmed."""
        with self._lock:
            self._exchanges.append(exchange)
            overflow = max(0, len(self._exchanges) - self.max_exchanges)
            del self._exchanges[:overflow]
            return overflow

    def pop_last(self, turn_id: int) -> bool:
        """Undo the exchange of ``turn_id`` (e.g. TTS failed, the user never heard it)."""
        with self._lock:
            if self._exchanges and self._exchanges[-1].turn_id == turn_id:
                self._exchanges.pop()
                return True
            return False

    def replace_last(self, turn_id: int, model_text: str, model_steps: list[dict[str, Any]] | None = None) -> bool:
        """Rewrite the robot side of the last exchange (an answer the user interrupted).

        ``model_steps`` replaces the provider steps too (Gemini); when omitted a plain text
        step is stored so no provider ever re-sends the unspoken part.
        """
        with self._lock:
            if not self._exchanges or self._exchanges[-1].turn_id != turn_id:
                return False
            ex = self._exchanges[-1]
            ex.model_text = model_text
            ex.model_steps = model_steps if model_steps is not None else [
                {"type": "model_output", "content": [{"type": "text", "text": model_text}]}]
            return True

    def reset(self) -> int:
        with self._lock:
            dropped = len(self._exchanges)
            self._exchanges.clear()
            self.resets += 1
            return dropped

    def transcript(self) -> list[dict[str, Any]]:
        with self._lock:
            out: list[dict[str, Any]] = []
            for ex in self._exchanges:
                out.append({"turn_id": ex.turn_id, "role": "user", "text": ex.user_text})
                out.append({"turn_id": ex.turn_id, "role": "assistant", "text": ex.model_text})
            return out
