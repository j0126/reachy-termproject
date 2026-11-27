"""설정 로딩: config/default.toml 위에 덮어쓰기 파일을 차례로 합치고, 환경 변수를 적용한다."""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DEFAULT_CONFIG = CONFIG_DIR / "default.toml"


def deep_merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return ROOT / p


def load_config(*overlays: str | Path, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """기본 설정 + 덮어쓰기 파일들 + REACHY_HOST 환경 변수."""
    env = os.environ if env is None else env
    with DEFAULT_CONFIG.open("rb") as f:
        cfg: dict[str, Any] = tomllib.load(f)
    for overlay in overlays:
        with _resolve(overlay).open("rb") as f:
            cfg = deep_merge(cfg, tomllib.load(f))
    host = env.get("REACHY_HOST")
    if host:
        cfg["robot"]["host"] = host
    return cfg


CALIBRATION_FILE = CONFIG_DIR / "calibration.toml"
_CALIBRATION_HEADER = ("# 데스크톱에서 실제 로봇으로 잰 값 (tools/spike_connection.py가 씀, 손으로 고쳐도 됨)\n"
                       "# 실행: .venv\\Scripts\\python -X utf8 -m reachy_cafe --config config/calibration.toml\n")


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(round(float(value), 4)) if isinstance(value, float) else str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    raise TypeError(f"unsupported TOML value: {value!r}")


def write_overlay(updates: Mapping[str, Mapping[str, Any]], path: str | Path = CALIBRATION_FILE) -> Path:
    """덮어쓰기 TOML([섹션] 키 = 값, 한 단계만)에 updates를 합쳐 쓴다. 기존 값은 유지하고 같은 키만 바꾼다."""
    target = Path(path)
    current: dict[str, Any] = {}
    if target.is_file():
        with target.open("rb") as f:
            current = tomllib.load(f)
    merged = deep_merge(current, {k: dict(v) for k, v in updates.items()})
    lines = [_CALIBRATION_HEADER.rstrip("\n")]
    for section, values in merged.items():
        if not isinstance(values, Mapping):
            raise TypeError(f"top-level key {section!r} must be a section")
        lines.append(f"\n[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_value(value)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
