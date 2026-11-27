"""보정 값 저장(config.write_overlay): 기존 값은 두고 같은 키만 바꾸며, load_config로 다시 읽힌다."""

from __future__ import annotations

import tomllib

from reachy_cafe.config import load_config, write_overlay


def test_overlay_merges_and_loads(tmp_path):
    path = tmp_path / "calibration.toml"
    write_overlay({"robot": {"screen_point": [0.2, 0.25, -0.1]}, "motion": {"label_delay_s": 0.18}}, path)
    write_overlay({"robot": {"slot_point": [0.22, 0.2, -0.3]}, "vision": {"screen_yaw_sign": -1.0}}, path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    assert raw["robot"] == {"screen_point": [0.2, 0.25, -0.1], "slot_point": [0.22, 0.2, -0.3]}
    assert raw["motion"]["label_delay_s"] == 0.18
    cfg = load_config(path, env={})
    assert cfg["robot"]["screen_point"] == [0.2, 0.25, -0.1]
    assert cfg["vision"]["screen_yaw_sign"] == -1.0
    assert cfg["robot"]["host"] == "192.168.0.15"  # 나머지는 기본값 그대로
    assert path.read_text(encoding="utf-8").startswith("# 데스크톱에서 실제 로봇으로 잰 값")
