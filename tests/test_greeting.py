"""손님 인사 모션(greeting): HW4 형식·안전 검사, mock 시간표와 YAML 일치, MotionPlayer 라벨 시각.

greeting은 카페에서 더한 모션이라 termproject/motions/에 있다 (HW4 폴더는 그대로).
YAML을 고친 뒤에는 이 테스트와 tools/sim_check.py(MuJoCo)로 확인한다.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("reachy_motion")
pytest.importorskip("yaml")

from reachy_cafe.behavior.motion_catalog import MOTIONS, timeline  # noqa: E402
from reachy_cafe.robot.motion_library import CAFE_MOTION_NAMES, cafe_library  # noqa: E402

HW4_MOTIONS = ("aha_recommend", "boredom", "success", "failure")


@pytest.fixture(scope="module")
def library() -> Any:
    return cafe_library()


def head_labels(spec: Any) -> dict[str, float]:
    eff = spec.effective()
    return {k.label: round(t0, 6) for (t0, _arrive, _release), k in zip(eff.key_times("head"), eff.track("head"))
            if k.label}


def test_library_has_hw4_and_cafe_motions(library: Any) -> None:
    keys = set(library.keys())
    for name in (*HW4_MOTIONS, *CAFE_MOTION_NAMES):
        for variant in ("expressive", "neutral"):
            assert f"{name}:{variant}" in keys


def test_catalog_matches_yaml(library: Any) -> None:
    """mock 시간표(motion_catalog)가 YAML head 트랙과 같아야 화면·음성 동기화가 mock과 실제 로봇에서 같다."""
    for spec in library:
        assert spec.variant in MOTIONS.get(spec.name, {}), f"{spec.key} missing from motion_catalog"
        duration, labels = timeline(spec.name, spec.variant)
        assert duration == pytest.approx(spec.duration, abs=1e-6), spec.key
        assert labels == pytest.approx(head_labels(spec), abs=1e-6), spec.key


def test_sync_labels_exist_in_the_motions(cfg: dict[str, Any], library: Any) -> None:
    """config [motion.sync_labels]의 라벨이 YAML에 있어야 화면·음성이 그 키프레임을 기다린다 (없으면 바로 나옴)."""
    sync = cfg["motion"]["sync_labels"]
    assert sync["greeting"] == "bow"
    for name, label in sync.items():
        assert label in head_labels(library.get(name, "expressive")), (name, label)
    assert sync["greeting"] in head_labels(library.get("greeting", "neutral"))


@pytest.mark.parametrize("variant", ["expressive", "neutral"])
def test_greeting_passes_the_hw4_policy(library: Any, variant: str) -> None:
    from reachy_motion.safety.checker import prepare
    from reachy_motion.safety.policy import DEFAULT_POLICY

    prepared = prepare(library.get("greeting", variant), DEFAULT_POLICY)
    assert prepared.runnable, [v.code for v in prepared.report.violations]
    # HW4 정책 하한은 10°. 시선 레이어와 같은 14° 이상을 지킨다 (가장자리에서는 추종 오차가 커짐, MuJoCo 점검)
    assert prepared.report.metrics["min_stewart_margin_deg"] >= 14.0


def test_greeting_neutral_follows_the_hw4_neutral_rules(library: Any) -> None:
    """neutral = 같은 시작·끝 자세, 직선(linear) 속도, 안테나·소리 없음 (HW4 library.compare)."""
    from reachy_motion.library.compare import comparison_issues

    issues = comparison_issues(library.get("greeting", "expressive"), library.get("greeting", "neutral"))
    assert [i.code for i in issues if i.severity == "warning"] == []


def test_greeting_plays_with_labels_at_yaml_times(library: Any) -> None:
    """HW4 MockBackend 위에서 실제 시간으로: 전이 뒤 라벨이 YAML 시각에 나오고 끝까지 돈다."""
    import time

    from reachy_motion.backends.mock import MockBackend
    from reachy_motion.service import HandleState

    from reachy_cafe.robot.motion import MotionPlayer

    backend = MockBackend(structure="robot")
    backend.connect()
    events: list[dict[str, Any]] = []
    player = MotionPlayer(backend, library=library, motion_cfg={"label_delay_s": 0.0})
    player.on_event(events.append)
    player.preload(["greeting"])
    player.start()
    try:
        time.sleep(0.3)  # 시선이 REST에서 벗어나 있어 전이가 앞에 붙는 실제 상황
        h = player.play("greeting", "expressive")
        assert h.wait(8.0)
        assert h.state == HandleState.DONE
        started = next(e for e in events if e["kind"] == "started" and e["id"] == h.id)
        got = [(e["label"], e["t"] - started["t"] - started["offset_s"]) for e in events
               if e["kind"] == "label" and e["id"] == h.id]
        _duration, expected = timeline("greeting", "expressive")
        assert [label for label, _t in got] == list(expected)
        for label, t in got:
            assert abs(t - expected[label]) < 0.15, (label, t)
    finally:
        player.close()
