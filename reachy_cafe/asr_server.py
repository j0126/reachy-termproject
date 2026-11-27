"""데스크톱(NVIDIA GPU)에서 음성 인식만 맡는 서버 — 현장 두뇌 PC에 GPU가 없을 때의 예비 경로.

현장 노트북은 config에 [speech] asr = "remote", asr_url = "http://<이 PC IP>:8001"을 두고 손님 발화를 보낸다.
인식 설정은 키오스크와 똑같이 이 PC의 config(HW3 large-v3-turbo, cuda, beam 5)를 쓰고, 언어·메뉴 이름 프롬프트는
요청마다 노트북이 보낸다.

    python -m reachy_cafe.asr_server                          # 0.0.0.0:8001
    python -m reachy_cafe.asr_server --port 8001 --config config/sim.toml

GET /health = 모델·장치, POST /transcribe = 본문 16 kHz mono int16 PCM → text·no_speech_prob·avg_logprob·asr_ms.
REACHY_ASR_TOKEN 환경 변수를 정하면 같은 값을 X-Token 헤더로 보낸 요청만 받는다 (노트북에도 같은 값).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import threading
from dataclasses import replace
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request

from .robot.remote_asr import TOKEN_ENV, decode_pcm


def create_app(engine: Any, token: str = "") -> FastAPI:
    """engine = HW3 WhisperASR처럼 cfg(AsrConfig)·model·transcribe(audio, rate)를 가진 것."""
    app = FastAPI(title="Reachy Café ASR", docs_url=None, redoc_url=None, openapi_url=None)
    lock = threading.Lock()  # GPU 모델 하나: 한 번에 한 발화
    base = engine.cfg

    def check(x_token: str | None) -> None:
        if token and x_token != token:
            raise HTTPException(status_code=401, detail="bad token")

    @app.get("/health")
    async def health(x_token: str | None = Header(default=None)) -> dict[str, Any]:
        check(x_token)
        return {"ok": engine.model is not None, "model": base.model, "device": base.device,
                "compute_type": base.compute_type}

    def run(audio: Any, rate: int, language: str, beam_size: int, prompt: str, hotwords: str) -> Any:
        with lock:
            engine.cfg = replace(base, language=language or base.language, beam_size=beam_size or base.beam_size,
                                 initial_prompt=prompt, hotwords=hotwords)
            return engine.transcribe(audio, rate)

    @app.post("/transcribe")
    async def transcribe(request: Request, sample_rate: int = 16000, language: str = "", beam_size: int = 0,
                         initial_prompt: str = "", hotwords: str = "",
                         x_token: str | None = Header(default=None)) -> dict[str, Any]:
        check(x_token)
        body = await request.body()
        if not body or len(body) % 2:
            raise HTTPException(status_code=400, detail="body must be 16-bit mono PCM")
        result = await asyncio.to_thread(run, decode_pcm(body), sample_rate, language, beam_size, initial_prompt,
                                         hotwords)
        return {"text": result.text, "language": result.language, "language_probability": result.language_probability,
                "no_speech_prob": result.no_speech_prob, "avg_logprob": result.avg_logprob, "segments": result.segments,
                "audio_s": round(result.audio_s, 3), "asr_ms": round(result.duration_ms, 1)}

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reachy_cafe.asr_server",
                                     description="Reachy Café 음성 인식 서버 (데스크톱 GPU → 현장 노트북)")
    parser.add_argument("--config", action="append", default=[], metavar="TOML", help="덮어쓸 설정 파일 (여러 번 가능)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)

    import uvicorn

    from .config import load_config
    from .robot.hwpaths import ensure_hw_paths

    cfg = load_config(*args.config)
    ensure_hw_paths(cfg)
    from reachy_s2s.engines.asr import WhisperASR

    from .robot.speech import load_hw3_config, prepare_cuda_libraries

    hw3 = load_hw3_config(cfg)
    if hw3.asr.device == "cuda":
        prepare_cuda_libraries()
    engine = WhisperASR(hw3.asr)
    info = engine.load()
    print(f"음성 인식 서버: {info.get('model')} ({info.get('device')}, {info.get('compute_type')}), "
          f"로딩 {info.get('load_ms')} ms + 웜업 {info.get('warmup_ms')} ms")
    print(f"  현장 노트북 config: [speech] asr = \"remote\", asr_url = \"http://<이 PC IP>:{args.port}\"")
    token = os.environ.get(TOKEN_ENV, "")
    if token:
        print(f"  {TOKEN_ENV}가 설정됨: 노트북에도 같은 값을 넣어야 합니다")
    uvicorn.run(create_app(engine, token), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
