"""Project settings: ``config/default.toml`` + optional override file + env + CLI.

Precedence (low -> high): default.toml, ``--config`` file, environment
(``REACHY_HOST``), explicit overrides passed by an app (e.g. ``--host``).
Secrets (API keys) are never stored here; they come from ``.env`` / env vars.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.toml"


@dataclass(frozen=True)
class RobotConfig:
    host: str
    port: int
    connection_mode: str
    media_backend: str
    connect_timeout_s: float

    @property
    def daemon_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class AudioConfig:
    mic_channel: str
    ring_seconds: float
    subscriber_queue_chunks: int
    first_input_timeout_s: float
    playback_chunk_ms: int
    playback_lead_ms: int
    output_latency_hint_ms: float
    output_gain_db: float
    gap_warn_ms: float


@dataclass(frozen=True)
class TimingConfig:
    high_resolution_timer: bool


@dataclass(frozen=True)
class MelConfig:
    n_fft: int
    win_ms: float
    hop_ms: float
    n_mels: int
    fmin_hz: float
    fmax_hz: float
    floor_db: float
    display_seconds: float
    display_db_min: float
    display_db_max: float


@dataclass(frozen=True)
class LoggingConfig:
    results_dir: str
    level: str
    resource_monitor: bool

    @property
    def results_path(self) -> Path:
        path = Path(self.results_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class VadConfig:
    threshold: float
    neg_threshold: float
    min_speech_ms: float
    silence_timeout_ms: float
    min_pause_ms: float
    pre_roll_ms: float
    post_roll_ms: float
    max_utterance_s: float


@dataclass(frozen=True)
class AsrConfig:
    model: str
    device: str
    compute_type: str
    language: str
    beam_size: int
    initial_prompt: str
    hotwords: str
    warmup: bool


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    model: str
    base_url: str
    api_key_env: str
    thinking_level: str
    max_output_tokens: int
    temperature: float
    timeout_s: float
    max_retries: int
    retry_on_timeout: bool
    warmup: bool
    max_history_exchanges: int
    system_prompt: str


@dataclass(frozen=True)
class LlmServerConfig:
    exe: str
    model_repo: str
    model_file: str
    model_revision: str
    alias: str
    host: str
    port: int
    ctx_size: int
    n_gpu_layers: int
    parallel: int
    extra_args: list
    startup_timeout_s: float


@dataclass(frozen=True)
class TtsConfig:
    provider: str
    model: str
    voice: str
    sample_rate: int
    style: str
    timeout_s: float
    max_retries: int
    retry_on_timeout: bool
    lang: str
    speed: float
    total_steps: int
    gain_db: float
    threads: int
    normalize_numbers: bool
    pronunciations: list


@dataclass(frozen=True)
class CascadedConfig:
    error_tone: bool
    save_turn_audio: bool
    variant: str
    barge_in: bool
    bargein_threshold: float
    bargein_min_speech_ms: float
    bargein_reset_silence_ms: float
    interrupted_history: str
    merge_cancelled_input: bool
    premature_window_ms: float


@dataclass(frozen=True)
class CloudConfig:
    variant: str
    model: str
    voice: str
    silence_duration_ms: int
    prefix_padding_ms: int
    start_sensitivity: str
    end_sensitivity: str
    activity_handling: str
    uplink_while_speaking: str
    uplink_chunk_ms: int
    uplink_queue_ms: int
    input_transcription: bool
    output_transcription: bool
    session_resumption: bool
    context_compression: bool
    connect_timeout_s: float
    reconnect_attempts: int
    reconnect_backoff_s: float
    response_timeout_s: float
    network_probe: bool
    save_turn_audio: bool


@dataclass(frozen=True)
class MotionConfig:
    enabled: bool
    auto_enable_motors: bool
    sink: str
    intensity: float
    speaking_motion: str
    allow_sdk_wobble: bool
    speech_gain: float
    speech_sync_offset_ms: float
    antenna_beat_rad: float
    speech_started_gesture: str
    nod_deg: float
    rate_hz: float
    deadband_deg: float
    deadband_mm: float
    deadband_rad: float
    keepalive_s: float
    startup_ramp_s: float
    shutdown_ramp_s: float
    max_pitch_deg: float
    max_roll_deg: float
    max_yaw_deg: float
    max_offset_mm: float
    max_antenna_rad: float
    max_head_speed_dps: float
    max_head_accel_dps2: float
    max_linear_speed_mms: float
    max_linear_accel_mms2: float
    max_antenna_speed_rps: float
    max_antenna_accel_rps2: float
    joint_guard: bool
    joint_max_change_deg: float
    joint_limit_margin_deg: float
    pose_tolerance_deg: float
    pose_tolerance_mm: float
    log_commands: bool
    thread_priority: str
    lock_during_conversation: bool


@dataclass(frozen=True)
class AppConfig:
    robot: RobotConfig
    audio: AudioConfig
    timing: TimingConfig
    mel: MelConfig
    logging: LoggingConfig
    vad: VadConfig
    asr: AsrConfig
    llm: LlmConfig
    llm_server: LlmServerConfig
    tts: TtsConfig
    cascaded: CascadedConfig
    cloud: CloudConfig
    motion: MotionConfig

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _dataclass_to_dict(getattr(self, f.name)) for f in fields(self)}


def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj


def _deep_merge(base: dict[str, Any], extra: dict[str, Any], where: str = "") -> dict[str, Any]:
    """Merge ``extra`` into a copy of ``base``; unknown keys are an error (typo guard)."""
    merged = dict(base)
    for key, value in extra.items():
        if key not in base:
            raise KeyError(f"Unknown config key '{where}{key}'")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"Config key '{where}{key}' must be a table")
            merged[key] = _deep_merge(base[key], value, f"{where}{key}.")
        else:
            merged[key] = value
    return merged


def _build(cls: type, table: dict[str, Any], section: str) -> Any:
    missing = [f.name for f in fields(cls) if f.name not in table]
    if missing:
        raise KeyError(f"Config section [{section}] is missing keys: {missing}")
    values = {}
    for f in fields(cls):
        value = table[f.name]
        if f.type in ("float", float) and isinstance(value, int):
            value = float(value)
        values[f.name] = value
    return cls(**values)


def load_config(
    path: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load settings.

    Args:
        path: optional TOML file (or several, applied in order) whose keys override
            ``config/default.toml``; e.g. a system config plus an experiment variant
            (``local_e4b_supertonic.toml`` + ``a1.toml``).
        overrides: nested dict applied last, e.g. ``{"robot": {"host": ip}}``.
            ``None`` values are ignored so argparse defaults can be passed as-is.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    with DEFAULT_CONFIG_PATH.open("rb") as fh:
        data = tomllib.load(fh)
    paths = [] if path is None else list(path) if isinstance(path, (list, tuple)) else [path]
    for p in paths:
        with Path(p).open("rb") as fh:
            data = _deep_merge(data, tomllib.load(fh))

    env_host = os.environ.get("REACHY_HOST")
    if env_host:
        data = _deep_merge(data, {"robot": {"host": env_host}})

    if overrides:
        cleaned = {
            section: {k: v for k, v in values.items() if v is not None}
            for section, values in overrides.items()
        }
        data = _deep_merge(data, cleaned)

    sections = {f.name: f.type for f in fields(AppConfig)}
    unknown = set(data) - set(sections)
    if unknown:
        raise KeyError(f"Unknown config sections: {sorted(unknown)}")
    return AppConfig(**{name: _build(_SECTION_TYPES[name], data[name], name) for name in sections})


_SECTION_TYPES: dict[str, type] = {
    "robot": RobotConfig,
    "audio": AudioConfig,
    "timing": TimingConfig,
    "mel": MelConfig,
    "logging": LoggingConfig,
    "vad": VadConfig,
    "asr": AsrConfig,
    "llm": LlmConfig,
    "llm_server": LlmServerConfig,
    "tts": TtsConfig,
    "cascaded": CascadedConfig,
    "cloud": CloudConfig,
    "motion": MotionConfig,
}
