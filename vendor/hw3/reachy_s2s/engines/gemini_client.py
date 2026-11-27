"""Gemini API client factory (google-genai >= 2.25, Interactions API).

* The API key is read from ``GEMINI_API_KEY`` (``.env`` is loaded by
  ``load_config``). It is passed to the SDK only and never logged.
* SDK-level retries are disabled. In google-genai 2.31 the Interactions client
  otherwise retries 408/409/429/5XX with backoff (3 times by default), and
  ``HttpRetryOptions(attempts=0 or 1)`` still maps to one retry (checked in the
  installed SDK). Hidden retries would mask free-tier limits (429) and inflate
  latency, so retries are done explicitly in ``engines.base.call_with_retries``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import EngineError

log = logging.getLogger(__name__)


def api_key_present() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def disable_sdk_retries(client: Any) -> bool:
    cfg = getattr(getattr(client, "interactions", None), "sdk_configuration", None)
    if cfg is None or not hasattr(cfg, "retry_config"):
        log.warning("could not disable google-genai retries (SDK internals changed?)")
        return False
    cfg.retry_config = None
    return True


def make_client(stage: str = "llm") -> Any:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise EngineError(stage, "config", "GEMINI_API_KEY is not set (copy .env.example to .env and fill it in)")
    from google import genai

    client = genai.Client(api_key=key)
    disable_sdk_retries(client)
    return client
