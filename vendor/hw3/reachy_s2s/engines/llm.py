"""Text LLM: Gemini via the Interactions API with client-managed history.

Request (documented pattern for client-side state): ``store=False``,
``input`` = all previous steps (user steps + model steps as returned) + the
new user step, ``system_instruction``, ``generation_config`` with
``thinking_level`` (gemini-3.8-flash: low|medium|high) and
``max_output_tokens``. Gemini 3.6+ ignores temperature/top_p/top_k, so they
are not sent. The history is only extended after a successful response.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import LlmConfig
from ..conversation.history import ConversationHistory
from .base import Attempt, EngineError, call_with_retries

log = logging.getLogger(__name__)


@dataclass
class LlmResult:
    text: str
    status: str | None
    model: str
    params: dict[str, Any]
    t_start_ns: int
    t_end_ns: int
    user_step: dict[str, Any]
    model_steps: list[dict[str, Any]]
    usage: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    input_steps: int = 0

    @property
    def duration_ms(self) -> float:
        return (self.t_end_ns - self.t_start_ns) / 1e6


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    keys = ("total_input_tokens", "total_output_tokens", "total_thought_tokens", "total_tokens",
            "total_cached_tokens")
    return {k: getattr(usage, k, None) for k in keys if getattr(usage, k, None) is not None}


def user_step(text: str) -> dict[str, Any]:
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


class GeminiLLM:
    name = "gemini"

    def __init__(self, cfg: LlmConfig, client: Any = None) -> None:
        self.cfg = cfg
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from .gemini_client import make_client

            self._client = make_client("llm")
        return self._client

    @property
    def params(self) -> dict[str, Any]:
        return {"thinking_level": self.cfg.thinking_level or None, "max_output_tokens": self.cfg.max_output_tokens,
                "store": False, "timeout_s": self.cfg.timeout_s, "max_retries": self.cfg.max_retries,
                "retry_on_timeout": self.cfg.retry_on_timeout}

    def warmup(self) -> dict[str, Any]:
        """One tiny request before the conversation (first calls showed 17-27 s cold starts).

        Never raises: a failed warm-up is reported, the conversation can still start.
        """
        t0 = time.perf_counter_ns()
        try:
            result = self.respond(ConversationHistory(), "안녕")
            return {"ok": True, "warmup_ms": round(result.duration_ms, 1), "attempts": result.attempts}
        except EngineError as exc:
            return {"ok": False, "warmup_ms": round((time.perf_counter_ns() - t0) / 1e6, 1), **exc.to_dict()}

    def _generation_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {"max_output_tokens": self.cfg.max_output_tokens}
        if self.cfg.thinking_level:  # Gemma 4 rejects thinking_level (HTTP 400): leave it empty there
            config["thinking_level"] = self.cfg.thinking_level
        return config

    def respond(self, history: ConversationHistory, text: str,
                on_attempt: Callable[[Attempt], None] | None = None) -> LlmResult:
        cfg = self.cfg
        step = user_step(text)
        steps = history.api_input() + [step]

        def call() -> Any:
            return self.client.interactions.create(
                model=cfg.model,
                store=False,
                system_instruction=cfg.system_prompt.strip(),
                input=steps,
                generation_config=self._generation_config(),
                timeout=cfg.timeout_s,
            )

        interaction, attempts = call_with_retries("llm", call, cfg.max_retries, on_attempt,
                                                  retry_timeouts=cfg.retry_on_timeout)
        status = getattr(interaction, "status", None)
        reply = (getattr(interaction, "output_text", None) or "").strip()
        model_steps = []
        for s in getattr(interaction, "steps", None) or []:
            if getattr(s, "type", None) == "user_input":
                continue
            model_steps.append(s.model_dump() if hasattr(s, "model_dump") else dict(s))
        if not reply:
            errors = getattr(interaction, "errors", None)
            raise EngineError("llm", "empty", f"no text in response (status={status}, errors={errors})",
                              attempts=len(attempts))
        if status not in (None, "completed"):
            log.warning("LLM status %s; using the text returned", status)
        return LlmResult(
            text=reply, status=status, model=cfg.model, params=self.params,
            t_start_ns=attempts[0].t_start_ns, t_end_ns=attempts[-1].t_end_ns,
            user_step=step, model_steps=model_steps,
            usage=_usage_dict(getattr(interaction, "usage", None)), attempts=len(attempts),
            input_steps=len(steps),
        )
