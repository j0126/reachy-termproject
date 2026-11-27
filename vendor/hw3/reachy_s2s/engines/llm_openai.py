"""Text LLM over an OpenAI-compatible ``/chat/completions`` endpoint.

One adapter for every server that speaks this API: a local llama.cpp
``llama-server`` (Gemma 4 GGUF from Hugging Face), Ollama, Groq, the Hugging
Face router or OpenAI itself. Only ``base_url``, ``model`` and the name of the
API-key environment variable change (config ``[llm]``).

* Plain ``httpx`` (already a dependency), no SDK: no hidden retries, the
  project's retry policy in :func:`engines.base.call_with_retries` applies.
* History is rebuilt from the plain text of each exchange (system + user /
  assistant messages), so it does not depend on which provider made it.
* Only standard request fields are sent (``model``, ``messages``,
  ``max_tokens``, ``temperature``, optionally ``reasoning_effort``); server
  specific sampling / thinking switches belong to the server's own config
  (see ``[llm_server]`` and ``apps.llm_server``).
* llama.cpp reports its own prompt / generation timings; they are kept in
  ``usage["server_timings"]`` so LLM latency can be split into prompt
  processing and decoding.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any

from ..config import LlmConfig
from ..conversation.history import ConversationHistory
from .base import Attempt, EngineError, call_with_retries
from .llm import LlmResult

log = logging.getLogger(__name__)

FINISH_TO_STATUS = {"stop": "completed", "length": "incomplete"}


class HttpStatusError(Exception):
    """Non-2xx answer; attribute names match what ``classify_exception`` reads."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.message = f"HTTP {status_code}"
        self.body = body[:300]


def chat_user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _usage_dict(data: dict[str, Any]) -> dict[str, Any]:
    usage = data.get("usage") or {}
    out: dict[str, Any] = {}
    mapping = {"prompt_tokens": "total_input_tokens", "completion_tokens": "total_output_tokens",
               "total_tokens": "total_tokens"}
    for src, dst in mapping.items():
        if usage.get(src) is not None:
            out[dst] = usage[src]
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reasoning:
        out["total_thought_tokens"] = reasoning
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is not None:
        out["total_cached_tokens"] = cached
    timings = data.get("timings")  # llama.cpp only
    if isinstance(timings, dict):
        keys = ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_second")
        out["server_timings"] = {k: round(v, 2) if isinstance(v, float) else v
                                 for k, v in timings.items() if k in keys}
    return out


class OpenAICompatLLM:
    name = "openai-compat"

    def __init__(self, cfg: LlmConfig, http: Any = None) -> None:
        self.cfg = cfg
        self._http = http

    # ------------------------------------------------------------ plumbing
    @property
    def base_url(self) -> str:
        return self.cfg.base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key_env:
            key = os.environ.get(self.cfg.api_key_env)
            if not key:
                raise EngineError("llm", "config", f"{self.cfg.api_key_env} is not set (add it to .env)")
            headers["Authorization"] = f"Bearer {key}"  # never logged
        return headers

    @property
    def http(self) -> Any:
        if self._http is None:
            import httpx

            if not self.base_url:
                raise EngineError("llm", "config", "[llm] base_url is empty (needed for provider openai_compat)")
            self._http = httpx.Client(headers=self._headers(), timeout=self.cfg.timeout_s)
        return self._http

    @property
    def params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"provider": "openai_compat", "base_url": self.base_url,
                                  "max_tokens": self.cfg.max_output_tokens, "temperature": self.cfg.temperature,
                                  "timeout_s": self.cfg.timeout_s, "max_retries": self.cfg.max_retries,
                                  "retry_on_timeout": self.cfg.retry_on_timeout}
        if self.cfg.thinking_level:
            params["reasoning_effort"] = self.cfg.thinking_level
        return params

    def _payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.cfg.model, "messages": messages,
                                   "max_tokens": self.cfg.max_output_tokens, "temperature": self.cfg.temperature,
                                   "stream": False}
        if self.cfg.thinking_level:
            payload["reasoning_effort"] = self.cfg.thinking_level
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.http.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.cfg.timeout_s)
        if response.status_code >= 400:
            raise HttpStatusError(response.status_code, response.text)
        return response.json()

    # ------------------------------------------------------------- probing
    def server_info(self) -> dict[str, Any]:
        """Best effort: what the server says it serves (``/models``; llama.cpp ``/props``). Never raises."""
        info: dict[str, Any] = {}
        try:
            r = self.http.get(f"{self.base_url}/models", timeout=5.0)
            if r.status_code < 400:
                info["models"] = [m.get("id") for m in (r.json().get("data") or [])][:10]
            else:
                info["models_status"] = r.status_code
        except Exception as exc:  # noqa: BLE001 - informational only
            info["error"] = f"{type(exc).__name__}: {exc}"[:200]
            return info
        root = self.base_url.removesuffix("/v1")
        try:
            r = self.http.get(f"{root}/props", timeout=5.0)
            if r.status_code < 400:
                props = r.json()
                settings = props.get("default_generation_settings") or {}
                sampling = settings.get("params") or settings
                info["llama_cpp"] = {
                    "model_path": props.get("model_path"),
                    "build_info": props.get("build_info"),
                    "n_ctx": settings.get("n_ctx"),
                    "total_slots": props.get("total_slots"),
                    "sampling": {k: sampling.get(k) for k in ("temperature", "top_k", "top_p", "min_p")
                                 if k in sampling},
                }
        except Exception:  # noqa: BLE001 - not a llama.cpp server
            pass
        return info

    def check_server(self) -> dict[str, Any]:
        """Raise a clear ``config`` error when the server is not reachable (called at engine load)."""
        info = self.server_info()
        if "error" in info:
            raise EngineError("llm", "config", f"LLM server not reachable at {self.base_url} ({info['error']}). "
                              "For the local model start it first: python -m reachy_s2s.apps.llm_server")
        return info

    def warmup(self) -> dict[str, Any]:
        """One tiny request before the conversation (loads weights / fills caches). Never raises."""
        t0 = time.perf_counter_ns()
        try:
            result = self.respond(ConversationHistory(), "안녕")
            return {"ok": True, "warmup_ms": round(result.duration_ms, 1), "attempts": result.attempts}
        except EngineError as exc:
            return {"ok": False, "warmup_ms": round((time.perf_counter_ns() - t0) / 1e6, 1), **exc.to_dict()}

    # ------------------------------------------------------------- respond
    def messages(self, history: ConversationHistory, text: str) -> list[dict[str, Any]]:
        system = self.cfg.system_prompt.strip()
        head = [{"role": "system", "content": system}] if system else []
        return head + history.chat_messages() + [chat_user_message(text)]

    def respond(self, history: ConversationHistory, text: str,
                on_attempt: Callable[[Attempt], None] | None = None) -> LlmResult:
        cfg = self.cfg
        messages = self.messages(history, text)
        payload = self._payload(messages)
        data, attempts = call_with_retries("llm", lambda: self._post(payload), cfg.max_retries, on_attempt,
                                           retry_timeouts=cfg.retry_on_timeout)
        choices = data.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        finish = choices[0].get("finish_reason") if choices else None
        reply = (message.get("content") or "").strip()
        usage = _usage_dict(data)
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            usage["reasoning_chars"] = len(reasoning)  # never spoken, only counted
        if not reply:
            raise EngineError("llm", "empty", f"no text in response (finish_reason={finish})",
                              attempts=len(attempts))
        status = FINISH_TO_STATUS.get(finish, finish)
        if status not in (None, "completed"):
            log.warning("LLM finish_reason %s; using the text returned", finish)
        return LlmResult(
            text=reply, status=status, model=data.get("model") or cfg.model, params=self.params,
            t_start_ns=attempts[0].t_start_ns, t_end_ns=attempts[-1].t_end_ns,
            user_step=chat_user_message(text), model_steps=[{"role": "assistant", "content": reply}],
            usage=usage, attempts=len(attempts), input_steps=len(messages),
        )
