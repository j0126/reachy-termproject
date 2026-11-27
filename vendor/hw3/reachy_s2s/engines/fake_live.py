"""A scripted stand-in for the Gemini Live server (tests and dry runs: no network, no key, no cost).

It speaks the same SDK types as the real service (``google.genai.types.LiveServerMessage``)
so the client code is exercised end to end:

* energy-based "server VAD" on the uploaded 16 kHz PCM: speech, then
  ``silence_duration_ms`` of silence (from the session config) ends the user turn;
* a reply = input transcription, 24 kHz audio chunks (faster than real time, like the
  real service), output transcription, ``generation_complete``, then ``turn_complete``
  after the reply's play-out time (the real server also waits for playback);
* user speech during a reply -> ``interrupted`` + ``turn_complete`` (barge-in);
* conversation context per session: resumed with the last handle it handed out,
  forgotten when a session starts without one (what Reset relies on);
* failure injection: refused connects, a dropped connection after N replies, GoAway.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..audio.signals import tone
from .gemini_live import OUTPUT_RATE, float_to_pcm16, pcm16_to_float

_CLOSE = object()


class FakeLiveError(Exception):
    """Mimics google.genai.errors.APIError raised for a closed WebSocket (``code`` = close code)."""

    def __init__(self, code: int, details: str) -> None:
        super().__init__(f"{code} {details}")
        self.code = code
        self.details = details


@dataclass
class FakeLiveScript:
    reply: Callable[[str, int], str] = lambda user, n: f"{n}번째 대화예요."  # (user text, user turns in context)
    user_texts: list[str] = field(default_factory=list)  # transcription of each detected user turn
    reply_s: float = 0.6  # reply audio length
    chunk_ms: int = 40
    chunk_interval_s: float = 0.005  # emission pace (much faster than real time)
    speech_dbfs: float = -40.0
    min_speech_ms: int = 100
    emit_voice_activity: bool = False  # the real service may or may not send voice_activity
    respond: bool = True  # False: detect turns but never answer (no_response tests)
    refuse_connects: int = 0  # first N connects fail
    refuse_error: tuple[int, str] = (1011, "Internal error (fake)")
    drop_after_replies: int | None = None  # close the connection abnormally after N replies (per connection)
    go_away_after_replies: int | None = None


class FakeLiveSession:
    def __init__(self, server: "FakeLiveConnector", config: Any, context: list[tuple[str, str]],
                 connection: int) -> None:
        self.server = server
        self.script = server.script
        self.config = config
        self.context = context
        self.connection = connection
        self._out: asyncio.Queue = asyncio.Queue()
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._in_speech = False
        self._reply_task: asyncio.Task | None = None
        self._replying = False
        self._replies = 0
        self.audio_bytes = 0
        self.stream_ends = 0
        aad = getattr(getattr(config, "realtime_input_config", None), "automatic_activity_detection", None)
        self.silence_ms = getattr(aad, "silence_duration_ms", None) or 800
        handling = getattr(getattr(config, "realtime_input_config", None), "activity_handling", None)
        self.interrupts = str(getattr(handling, "value", handling) or "").upper() != "NO_INTERRUPTION"

    # ----------------------------------------------------------- client side
    async def send_realtime_input(self, *, audio: Any = None, audio_stream_end: bool | None = None,
                                  **_: Any) -> None:
        if audio_stream_end:
            self.stream_ends += 1
            await self._end_of_speech_if_any()
            return
        data = audio.data if hasattr(audio, "data") else audio["data"]
        self.audio_bytes += len(data)
        x = pcm16_to_float(data)
        ms = 1000.0 * x.shape[0] / 16000
        loud = x.size and 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12) > self.script.speech_dbfs
        if loud:
            self._speech_ms += ms
            self._silence_ms = 0.0
            if not self._in_speech and self._speech_ms >= self.script.min_speech_ms:
                self._in_speech = True
                if self.script.emit_voice_activity:
                    await self._put(voice_activity="ACTIVITY_START")
                if self._replying and self.interrupts:
                    await self._interrupt()
        elif self._in_speech:
            self._silence_ms += ms
            if self._silence_ms >= self.silence_ms:
                await self._end_of_speech_if_any()
        else:
            self._speech_ms = 0.0

    async def _end_of_speech_if_any(self) -> None:
        if not self._in_speech:
            return
        self._in_speech = False
        self._speech_ms = self._silence_ms = 0.0
        if self.script.emit_voice_activity:
            await self._put(voice_activity="ACTIVITY_END")
        n = len(self.server.user_turns)
        user = self.script.user_texts[n] if n < len(self.script.user_texts) else f"(가짜 사용자 발화 {n + 1})"
        self.server.user_turns.append(user)
        if self.script.respond and not self._replying:
            self._reply_task = asyncio.create_task(self._reply(user))

    async def receive(self) -> AsyncIterator[Any]:
        """Like the SDK: yields messages up to and including the next turn_complete."""
        while True:
            msg = await self._out.get()
            if msg is _CLOSE:
                raise FakeLiveError(1006, "abnormal closure (fake drop)")
            yield msg
            sc = getattr(msg, "server_content", None)
            if sc is not None and sc.turn_complete:
                return

    async def close(self) -> None:
        if self._reply_task is not None:
            self._reply_task.cancel()

    # ----------------------------------------------------------- server side
    async def _put(self, **fields: Any) -> None:
        from google.genai import types

        if "voice_activity" in fields:
            msg = types.LiveServerMessage(voice_activity=types.VoiceActivity(voice_activity_type=fields["voice_activity"]))
        elif "resumption" in fields:
            msg = types.LiveServerMessage(session_resumption_update=types.LiveServerSessionResumptionUpdate(
                new_handle=fields["resumption"], resumable=True))
        elif "go_away" in fields:
            msg = types.LiveServerMessage(go_away=types.LiveServerGoAway(time_left=fields["go_away"]))
        elif "usage" in fields:
            msg = types.LiveServerMessage(usage_metadata=types.UsageMetadata(**fields["usage"]))
        elif "setup" in fields:
            msg = types.LiveServerMessage(setup_complete=types.LiveServerSetupComplete())
        else:
            msg = types.LiveServerMessage(server_content=types.LiveServerContent(**fields))
        await self._out.put(msg)

    async def _reply(self, user: str) -> None:
        from google.genai import types

        self._replying = True
        try:
            self.context.append(("user", user))
            text = self.script.reply(user, sum(1 for role, _ in self.context if role == "user"))
            await self._put(input_transcription=types.Transcription(text=user, finished=True))
            audio = tone(300.0, self.script.reply_s, OUTPUT_RATE, amplitude=0.3)
            step = int(OUTPUT_RATE * self.script.chunk_ms / 1000)
            await self._put(output_transcription=types.Transcription(text=text))
            for i in range(0, audio.shape[0], step):
                blob = types.Blob(data=float_to_pcm16(audio[i:i + step]), mime_type=f"audio/pcm;rate={OUTPUT_RATE}")
                await self._put(model_turn=types.Content(role="model", parts=[types.Part(inline_data=blob)]))
                await asyncio.sleep(self.script.chunk_interval_s)
            await self._put(generation_complete=True)
            await asyncio.sleep(self.script.reply_s)  # the server waits for the client's play-out
            self.context.append(("model", text))
            await self._put(turn_complete=True)
            await self._put(usage={"prompt_token_count": 40 * len(self.context), "response_token_count": 30,
                                   "total_token_count": 40 * len(self.context) + 30})
            self.server.handle_counter += 1
            handle = f"fake-handle-{self.server.handle_counter}"
            self.server.contexts[handle] = self.context
            await self._put(resumption=handle)
            self._replies += 1
            if self.script.go_away_after_replies == self._replies:
                await self._put(go_away="5s")
            if self.script.drop_after_replies == self._replies:
                await self._out.put(_CLOSE)
        except asyncio.CancelledError:
            pass
        finally:
            self._replying = False

    async def _interrupt(self) -> None:
        if self._reply_task is not None:
            self._reply_task.cancel()
        self._replying = False
        await self._put(interrupted=True)
        await self._put(turn_complete=True)


class FakeLiveConnector:
    """Drop-in for ``GeminiLiveConnector`` (records configs and sessions for assertions)."""

    name = "fake-live"

    def __init__(self, script: FakeLiveScript | None = None) -> None:
        self.script = script or FakeLiveScript()
        self.configs: list[Any] = []
        self.models: list[str] = []
        self.sessions: list[FakeLiveSession] = []
        self.contexts: dict[str, list[tuple[str, str]]] = {}
        self.user_turns: list[str] = []
        self.handle_counter = 0
        self._connects = itertools.count(1)
        self.refused = 0

    @contextlib.asynccontextmanager
    async def connect(self, model: str, config: Any) -> AsyncIterator[FakeLiveSession]:
        n = next(self._connects)
        self.models.append(model)
        self.configs.append(config)
        if self.refused < self.script.refuse_connects:
            self.refused += 1
            raise FakeLiveError(*self.script.refuse_error)
        handle = getattr(getattr(config, "session_resumption", None), "handle", None)
        context = self.contexts.get(handle, []) if handle else []
        session = FakeLiveSession(self, config, context, n)
        self.sessions.append(session)
        await session._put(setup=True)
        try:
            yield session
        finally:
            await session.close()
