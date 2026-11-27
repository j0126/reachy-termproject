"""Run the local LLM server (llama.cpp ``llama-server``) described in ``[llm_server]``.

    python -m reachy_s2s.apps.llm_server                       # Gemma 4 E4B (default.toml)
    python -m reachy_s2s.apps.llm_server --config config/llm_local_gemma4_e2b.toml
    python -m reachy_s2s.apps.llm_server --dry-run             # print the command only

The GGUF file is taken from the Hugging Face cache (``--download`` fetches it
once if missing). The server exposes an OpenAI-compatible API at
``http://<host>:<port>/v1`` that ``[llm] provider = "openai_compat"`` uses.
Keep this running in its own terminal; Ctrl+C stops it.

Logged to ``results/sessions/<id>_llm_server/``: the exact command, model file,
GPU memory before / after loading, load time, the buffer sizes llama.cpp
reports (where the weights / KV cache live) and the full server log.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..config import AppConfig, load_config
from ..engines.asr import gpu_memory_used_mb
from ..evaluation.session import Session, setup_console_logging

BUFFER_LINE = re.compile(r"(buffer size|offloaded|KV|n_ctx|CUDA0|model type|file type|build)", re.I)


def resolve_exe(cfg: AppConfig) -> Path:
    exe = Path(os.path.expandvars(cfg.llm_server.exe))
    if not exe.is_file():
        raise SystemExit(f"llama-server not found: {exe}\n"
                         "Install the official llama.cpp Windows build there (see README, section Phase 2 local LLM) "
                         "or set [llm_server] exe.")
    return exe


def resolve_model(cfg: AppConfig, download: bool) -> Path:
    from huggingface_hub import hf_hub_download

    s = cfg.llm_server
    revision = s.model_revision or None
    try:
        return Path(hf_hub_download(s.model_repo, s.model_file, revision=revision, local_files_only=True))
    except Exception as exc:  # noqa: BLE001 - not cached
        if not download:
            raise SystemExit(f"{s.model_repo}/{s.model_file} is not in the Hugging Face cache ({exc}).\n"
                             "Run again with --download to fetch it once.") from exc
    return Path(hf_hub_download(s.model_repo, s.model_file, revision=revision))


def build_command(cfg: AppConfig, exe: Path, model: Path) -> list[str]:
    s = cfg.llm_server
    return [str(exe), "-m", str(model), "--alias", s.alias, "--host", s.host, "--port", str(s.port),
            "-c", str(s.ctx_size), "-ngl", str(s.n_gpu_layers), "-np", str(s.parallel),
            *[str(a) for a in s.extra_args]]


def wait_ready(proc: subprocess.Popen, url: str, timeout_s: float) -> tuple[bool, str]:
    import httpx

    deadline = time.monotonic() + timeout_s
    last = "no answer"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False, f"server exited with code {proc.returncode}"
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                return True, "ok"
            last = f"HTTP {r.status_code}"  # 503 while the model loads
        except Exception as exc:  # noqa: BLE001 - not listening yet
            last = type(exc).__name__
        time.sleep(0.25)
    return False, f"not ready after {timeout_s:.0f} s ({last})"


def server_report(log_path: Path) -> list[str]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [ln.strip() for ln in lines if BUFFER_LINE.search(ln)][:80]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append",
                        help="TOML file overriding config/default.toml (repeatable, applied in order)")
    parser.add_argument("--download", action="store_true", help="fetch the GGUF if it is not cached yet")
    parser.add_argument("--dry-run", action="store_true", help="print the command and exit")
    parser.add_argument("--ngl", type=int, help="override [llm_server] n_gpu_layers")
    parser.add_argument("--ctx", type=int, help="override [llm_server] ctx_size")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides={"llm_server": {"n_gpu_layers": args.ngl, "ctx_size": args.ctx}})
    setup_console_logging(cfg.logging.level)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # the readiness poll would log every request
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # status lines show up at once when redirected

    exe = resolve_exe(cfg)
    model = resolve_model(cfg, args.download)
    cmd = build_command(cfg, exe, model)
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    if args.dry_run:
        return 0

    s = cfg.llm_server
    url = f"http://{s.host}:{s.port}"
    session = Session(cfg.logging.results_path, "llm_server")
    log_path = session.path("llama_server.log")
    mem_before = gpu_memory_used_mb()
    meta: dict[str, Any] = {"llm_server": {**cfg.to_dict()["llm_server"], "exe": str(exe)}, "command": cmd,
                            "model_path": str(model), "model_bytes": model.stat().st_size, "url": f"{url}/v1",
                            "gpu_mem_before_mb": mem_before}
    session.write_metadata(**meta)
    t0 = time.perf_counter_ns()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_fh:
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, cwd=str(exe.parent),
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        session.events.log("llm_server.started", pid=proc.pid)
        ok, why = wait_ready(proc, url, s.startup_timeout_s)
        load_ms = round((time.perf_counter_ns() - t0) / 1e6, 1)
        mem_after = gpu_memory_used_mb()
        report = server_report(log_path)
        session.write_metadata(ready=ok, ready_detail=why, load_ms=load_ms, gpu_mem_after_mb=mem_after,
                               gpu_mem_delta_mb=(mem_after - mem_before) if None not in (mem_after, mem_before)
                               else None, server_report=report)
        session.events.log("llm_server.ready" if ok else "llm_server.failed", load_ms=load_ms, detail=why,
                           gpu_mem_before_mb=mem_before, gpu_mem_after_mb=mem_after)
        for line in report:
            if "buffer size" in line or "offloaded" in line:
                print("   ", line)
        if not ok:
            print(f"[!!] llama-server failed: {why}. Log: {log_path}")
            if proc.poll() is None:
                proc.terminate()
            session.close()
            return 1
        print(f"[ok] {s.alias} ready at {url}/v1 in {load_ms / 1000:.1f} s; "
              f"GPU memory {mem_before} -> {mem_after} MiB. Ctrl+C stops it. Log: {log_path}")
        try:
            code = proc.wait()
            session.events.log("llm_server.exited", returncode=code)
            print(f"[!!] llama-server exited with code {code}")
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            session.events.log("llm_server.stopped", reason="Ctrl+C")
        finally:
            session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
