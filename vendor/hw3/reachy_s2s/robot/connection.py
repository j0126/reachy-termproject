"""Connection to a Reachy Mini daemon (SDK client + media + REST helpers).

One ``ReachyConnection`` owns one ``ReachyMini`` instance. Audio and (from
Phase 6) motion both use it, so they share a single daemon session.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import requests

from ..config import RobotConfig
from .backends.reachy import ReachyMediaBackend

log = logging.getLogger(__name__)

#: XVF3800 parameters recorded with every session (mic DSP state matters for
#: VAD/ASR results). Read-only GETs through the daemon.
AUDIO_PARAMS_TO_RECORD = (
    "VERSION",
    "AUDIO_MGR_OP_L",
    "AUDIO_MGR_OP_R",
    "AUDIO_MGR_MIC_GAIN",
    "AUDIO_MGR_REF_GAIN",
    "AUDIO_MGR_SYS_DELAY",
    "AEC_ASROUTONOFF",
    "AEC_AECCONVERGED",
    "PP_AGCONOFF",
    "PP_AGCMAXGAIN",
    "PP_AGCGAIN",
    "PP_ECHOONOFF",
    "PP_NLATTENONOFF",
    "PP_MIN_NS",
    "PP_MIN_NN",
    "PP_LIMITONOFF",
)


def sdk_version() -> str | None:
    try:
        return version("reachy_mini")
    except PackageNotFoundError:
        return None


class DaemonClient:
    """Small REST client for the daemon's FastAPI server (no WebRTC, no WebSocket)."""

    def __init__(self, host: str, port: int = 8000, timeout_s: float = 5.0) -> None:
        self.base_url = f"http://{host}:{port}"
        self.timeout_s = timeout_s

    def get(self, path: str, timeout_s: float | None = None) -> Any:
        resp = requests.get(f"{self.base_url}{path}", timeout=timeout_s or self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, json: Any = None, timeout_s: float | None = None) -> Any:
        resp = requests.post(f"{self.base_url}{path}", json=json, timeout=timeout_s or self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def status(self) -> dict[str, Any]:
        return self.get("/api/daemon/status")

    # -- motors (Phase 4.5): REST, independent of the SDK WebSocket the motion commands use --
    def motor_mode(self, timeout_s: float = 2.0) -> str | None:
        """``enabled`` | ``disabled`` | ``gravity_compensation`` (GET /api/motors/status)."""
        return self.get("/api/motors/status", timeout_s=timeout_s).get("mode")

    def set_motor_mode(self, mode: str, timeout_s: float = 2.0) -> Any:
        """POST /api/motors/set_mode/{mode}; used for Emergency Torque OFF (``disabled``)."""
        if mode not in ("enabled", "disabled", "gravity_compensation"):
            raise ValueError(f"unknown motor mode {mode!r}")
        return self.post(f"/api/motors/set_mode/{mode}", timeout_s=timeout_s)

    def control_loop_stats(self, timeout_s: float = 2.0) -> dict[str, Any] | None:
        """The daemon's 50 Hz control-loop statistics (mean frequency, max interval, errors)."""
        backend = self.get("/api/daemon/status", timeout_s=timeout_s).get("backend_status") or {}
        stats = backend.get("control_loop_stats")
        return dict(stats) if stats else None

    def volumes(self) -> dict[str, Any]:
        return {
            "speaker": self.get("/api/volume/current").get("volume"),
            "microphone": self.get("/api/volume/microphone/current").get("volume"),
        }

    def audio_params(self, names: tuple[str, ...] = AUDIO_PARAMS_TO_RECORD) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name in names:
            try:
                params[name] = self.get(f"/api/audio/config/parameter/{name}").get("values")
            except requests.RequestException as exc:
                params[name] = f"unavailable: {exc}"
        return params

    def doa(self) -> Any:
        return self.get("/api/state/doa")

    def summary(self) -> dict[str, Any]:
        """Robot facts worth storing with an experiment (best effort)."""
        info: dict[str, Any] = {"daemon_url": self.base_url}
        try:
            st = self.status()
            info.update(
                robot_name=st.get("robot_name"),
                daemon_version=st.get("version"),
                wireless=st.get("wireless_version"),
                state=st.get("state"),
                wlan_ip=st.get("wlan_ip"),
                no_media=st.get("no_media"),
                motor_control_mode=(st.get("backend_status") or {}).get("motor_control_mode"),
            )
        except requests.RequestException as exc:
            info["status_error"] = str(exc)
            return info
        try:
            info["volume"] = self.volumes()
        except requests.RequestException as exc:
            info["volume"] = f"unavailable: {exc}"
        info["xvf3800"] = self.audio_params()
        return info


class ReachyConnection:
    def __init__(self, cfg: RobotConfig, log_level: str = "INFO") -> None:
        self.cfg = cfg
        self.log_level = log_level
        self.mini: Any = None
        self.daemon = DaemonClient(cfg.host, cfg.port)

    def open(self) -> "ReachyConnection":
        from reachy_mini import ReachyMini  # heavy import (GStreamer); keep it lazy

        log.info(
            "Connecting to Reachy Mini at %s:%s (connection_mode=%s, media_backend=%s)",
            self.cfg.host, self.cfg.port, self.cfg.connection_mode, self.cfg.media_backend,
        )
        self.mini = ReachyMini(
            host=self.cfg.host,
            port=self.cfg.port,
            connection_mode=self.cfg.connection_mode,
            media_backend=self.cfg.media_backend,
            timeout=self.cfg.connect_timeout_s,
            log_level=self.log_level,
        )
        return self

    @property
    def media(self) -> Any:
        if self.mini is None:
            raise RuntimeError("not connected")
        return self.mini.media

    def audio_backend(self, output_latency_hint_s: float) -> ReachyMediaBackend:
        return ReachyMediaBackend(self.media, output_latency_hint_s)

    def describe(self) -> dict[str, Any]:
        info = self.daemon.summary()
        info.update(
            sdk_version=sdk_version(),
            connection_mode=getattr(self.mini, "connection_mode", None),
            media_backend=self.media.backend.value if self.mini is not None else None,
        )
        return info

    def close(self) -> None:
        if self.mini is None:
            return
        mini, self.mini = self.mini, None
        try:
            mini.__exit__(None, None, None)  # closes media + websocket like `with ReachyMini()`
        except Exception:
            log.exception("error while closing the Reachy Mini connection")
