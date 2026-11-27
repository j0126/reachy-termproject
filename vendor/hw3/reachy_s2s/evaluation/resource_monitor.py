"""Process / thread CPU usage as ``system.resources`` events (Phase 4.5 measurements).

Every ``period_s`` one event: process CPU % (100 = one core), system CPU %, resident memory, and
the CPU % of our named Python threads (``audio-capture``, ``audio-player``, ``motion``, ``vad`` ...).
Enabled with ``[logging] resource_monitor = true``; off by default so the Phase 4 sessions stay
unchanged.

The monitor must not disturb what it measures: psutil's per-thread snapshot (``Process.threads()``)
takes about 23 ms on Windows and holds the GIL meanwhile, which delayed every Python thread (mic
capture, VAD, the 50 Hz motion loop) once a second in the R0 dashboard sessions (2026-10-17).
So the named threads are read with ``GetThreadTimes`` (microseconds, no GIL stall) every period, and
the full snapshot that also shows native threads (ONNX Runtime, CTranslate2, GStreamer:
``native:<id>``) is taken only every ``native_every_s`` (an extra ``native_threads`` field).

Garbage collection pauses stop every Python thread at once, so the monitor also records
``system.gc`` events: every generation-2 collection and any collection of 10 ms or more (duration,
objects collected, thread). The gc callback only appends to a deque; the monitor thread logs
(logging inside the callback could deadlock on the event log's lock).
"""

from __future__ import annotations

import gc
import logging
import sys
import threading
import time
from collections import deque
from typing import Any

from .event_logger import EventLogger

log = logging.getLogger(__name__)
GC_LOG_MS = 10.0


def _thread_cpu_reader():
    """``f(native_id) -> CPU seconds or None`` without a process snapshot (Windows); None elsewhere."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenThread.restype = wintypes.HANDLE
    kernel32.OpenThread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetThreadTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    query_limited = 0x0800  # THREAD_QUERY_LIMITED_INFORMATION

    def read(native_id: int) -> float | None:
        handle = kernel32.OpenThread(query_limited, False, native_id)
        if not handle:
            return None
        try:
            c, e, k, u = (wintypes.FILETIME() for _ in range(4))
            if not kernel32.GetThreadTimes(handle, ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u)):
                return None
            ticks = (k.dwHighDateTime << 32 | k.dwLowDateTime) + (u.dwHighDateTime << 32 | u.dwLowDateTime)
            return ticks / 1e7  # 100 ns units
        finally:
            kernel32.CloseHandle(handle)

    return read


class ResourceMonitor:
    def __init__(self, events: EventLogger, period_s: float = 1.0, top_threads: int = 8,
                 native_every_s: float = 30.0) -> None:
        self.events = events
        self.period_s = period_s
        self.top_threads = top_threads
        self.native_every_s = native_every_s
        self.samples = 0
        self.snapshots = 0
        self.error: str | None = None
        self.gc_pauses: deque[tuple[int, float, int, str]] = deque(maxlen=1000)  # (generation, ms, collected, thread)
        self._gc_t0: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _on_gc(self, phase: str, info: dict[str, Any]) -> None:  # any thread, inside the collector
        if phase == "start":
            self._gc_t0 = time.perf_counter_ns()
            return
        t0, self._gc_t0 = self._gc_t0, None
        if t0 is None:
            return
        ms = (time.perf_counter_ns() - t0) / 1e6
        gen = int(info.get("generation", -1))
        if gen == 2 or ms >= GC_LOG_MS:
            self.gc_pauses.append((gen, ms, int(info.get("collected", 0)), threading.current_thread().name))

    def set_event_logger(self, events: EventLogger) -> None:
        self.events = events

    def start(self) -> bool:
        try:
            import psutil  # noqa: F401 - installed with reachy-mini
        except ImportError as exc:
            self.error = f"psutil not available: {exc}"
            log.warning(self.error)
            return False
        self._stop.clear()
        gc.callbacks.append(self._on_gc)
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        self._log_gc()

    def _log_gc(self) -> None:
        while self.gc_pauses:
            try:
                gen, ms, collected, thread = self.gc_pauses.popleft()
            except IndexError:
                return
            self.events.log("system.gc", generation=gen, ms=round(ms, 1), collected=collected, thread=thread)

    def _python_threads(self, read: Any) -> dict[str, float]:
        out: dict[str, float] = {}
        for t in threading.enumerate():
            if t.native_id is None:
                continue
            cpu = read(t.native_id) if read is not None else None
            if cpu is not None:
                out[f"{t.name}#{t.native_id}"] = cpu
        return out

    def _run(self) -> None:
        import psutil

        proc = psutil.Process()
        proc.cpu_percent(None)
        psutil.cpu_percent(None)
        read = _thread_cpu_reader()
        last_named = self._python_threads(read)
        last_t = time.perf_counter()
        last_snapshot = {t.id: t.user_time + t.system_time for t in proc.threads()}
        last_snapshot_t = last_t
        while not self._stop.wait(self.period_s):
            try:
                now = time.perf_counter()
                wall = max(1e-6, now - last_t)
                named = self._python_threads(read)
                per_thread = []
                for key, cpu_s in named.items():
                    pct = 100.0 * (cpu_s - last_named.get(key, cpu_s)) / wall
                    if pct > 0.05:
                        per_thread.append((key.rsplit("#", 1)[0], round(pct, 1)))
                per_thread.sort(key=lambda x: -x[1])
                mem = proc.memory_info()
                record: dict[str, Any] = {
                    "process_cpu_pct": round(proc.cpu_percent(None), 1),
                    "system_cpu_pct": round(psutil.cpu_percent(None), 1),
                    "rss_mb": round(mem.rss / 2**20, 1),
                    "threads": dict(per_thread[: self.top_threads])}
                if now - last_snapshot_t >= self.native_every_s:  # ~23 ms GIL stall: rarely
                    names = {t.native_id: t.name for t in threading.enumerate() if t.native_id is not None}
                    current = {t.id: t.user_time + t.system_time for t in proc.threads()}
                    span = max(1e-6, now - last_snapshot_t)
                    native = sorted(((names.get(tid, f"native:{tid}"),
                                      round(100.0 * (cpu - last_snapshot.get(tid, cpu)) / span, 1))
                                     for tid, cpu in current.items()), key=lambda x: -x[1])
                    record.update(threads_total=len(current), native_threads=dict(native[: self.top_threads]),
                                  native_span_s=round(span, 1))
                    last_snapshot, last_snapshot_t = current, now
                    self.snapshots += 1
                self.events.log("system.resources", **record)
                self.samples += 1
                last_named, last_t = named, now
                self._log_gc()
            except Exception as exc:  # noqa: BLE001 - measurement only
                self.error = repr(exc)
                log.debug("resource sample failed: %r", exc)


def resource_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean / max process CPU and the mean CPU of each named thread over ``system.resources`` events."""
    rows = [r for r in records if r.get("kind") == "system.resources"]
    if not rows:
        return {}
    cpu = [r["process_cpu_pct"] for r in rows]
    threads: dict[str, list[float]] = {}
    for r in rows:
        for name, pct in (r.get("threads") or {}).items():
            if not name.startswith("native:"):
                threads.setdefault(name, []).append(pct)
    gcs = [r for r in records if r.get("kind") == "system.gc"]
    return {"samples": len(rows), "process_cpu_mean": round(sum(cpu) / len(cpu), 1), "process_cpu_max": max(cpu),
            "thread_cpu_mean": {k: round(sum(v) / len(rows), 2) for k, v in sorted(threads.items())},
            "gc_pauses": len(gcs), "gc_ms_max": max((g["ms"] for g in gcs), default=None)}
