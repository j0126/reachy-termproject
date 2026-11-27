"""App.route: 말소리만으로는 손님이 있다고 보지 않는다 (발표장 관객 잡담에 인사·끄덕임하던 문제)."""

from __future__ import annotations

import asyncio
import math
from typing import Any

from reachy_cafe.app import App
from reachy_cafe.bus import Event


def test_voices_alone_do_not_make_someone_present(cfg: dict[str, Any]) -> None:
    async def run() -> None:
        app = App(cfg, log_to_file=False)
        nods: list[str] = []
        app.robot.gesture = nods.append  # type: ignore[method-assign]
        app.route(Event("speech.start"), 10.0)                        # 대기 중, 아무도 안 보임
        assert app.estimator.last_input_t == -math.inf and not nods
        app.route(Event("speech.utterance", {"text": "음 그러니까 내일 수업이"}), 10.5)
        assert app.fsm.session is None and app.estimator.last_input_t == -math.inf
        assert app.behavior.status_fields["heard"] == ""              # 잡담은 자막에도 올리지 않는다
        app.route(Event("speech.utterance", {"text": "라떼 하나 주세요"}), 11.0)
        assert app.fsm.session is not None and app.estimator.last_input_t == 11.0
        app.route(Event("speech.start"), 12.0)                        # 주문 중에는 말소리가 '있음'의 근거
        assert app.estimator.last_input_t == 12.0
    asyncio.run(run())
