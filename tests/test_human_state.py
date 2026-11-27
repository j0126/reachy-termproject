from __future__ import annotations

from typing import Any

from reachy_cafe.perception.human_state import (ABSENT, AMBIGUOUS, ENGAGED, NO_RESPONSE, PRESENT, UNCERTAIN, Face,
                                                HumanStateEstimator)

NEAR = [Face(1, 0.30)]
FAR = [Face(1, 0.10)]
TWO = [Face(1, 0.30, 0.4), Face(2, 0.26, 0.66)]


class Sim:
    def __init__(self, cfg: dict[str, Any]):
        self.est = HumanStateEstimator(cfg["timing"], cfg["vision"])
        self.now = 0.0
        self.changes: list[str] = []

    def run(self, seconds: float, faces: list[Face] | None, awaiting: float | None = None,
            gaze_off: bool = False) -> str:
        end = self.now + seconds
        while self.now < end - 1e-9:
            self.now = round(self.now + 0.1, 6)
            if faces is not None:
                self.est.update_faces(faces, self.now)
            self.est.set_gaze_off_user(gaze_off)
            new = self.est.step(self.now, awaiting)
            if new:
                self.changes.append(new)
        return self.est.state


def test_approach_and_engage(cfg: dict[str, Any]) -> None:
    sim = Sim(cfg)
    assert sim.run(1.0, []) == ABSENT
    assert sim.run(0.7, FAR) == PRESENT
    assert sim.run(1.2, NEAR) == ENGAGED


def test_no_response_after_hesitation(cfg: dict[str, Any]) -> None:
    sim = Sim(cfg)
    sim.run(1.5, NEAR)
    awaiting = sim.now
    assert sim.run(7.5, NEAR, awaiting=awaiting) == ENGAGED
    assert sim.run(1.0, NEAR, awaiting=awaiting) == NO_RESPONSE
    sim.est.note_input(sim.now)
    assert sim.run(0.2, NEAR, awaiting=awaiting) == ENGAGED


def test_two_people_is_ambiguous(cfg: dict[str, Any]) -> None:
    sim = Sim(cfg)
    assert sim.run(1.5, TWO) == AMBIGUOUS


def test_stale_frames_are_uncertain_not_absent(cfg: dict[str, Any]) -> None:
    sim = Sim(cfg)
    sim.run(1.5, NEAR)
    assert sim.run(1.5, None) == UNCERTAIN


def test_face_loss_hysteresis_then_absent(cfg: dict[str, Any]) -> None:
    sim = Sim(cfg)
    sim.run(1.5, NEAR)
    assert sim.run(1.5, []) == ENGAGED       # 잠깐 고개를 돌려도 유지
    assert sim.run(1.0, []) == PRESENT
    assert sim.run(1.5, []) == ABSENT


def test_recent_input_counts_as_presence(cfg: dict[str, Any]) -> None:
    sim = Sim(cfg)
    sim.run(1.0, [])
    sim.est.note_input(sim.now)
    assert sim.run(0.5, []) == ENGAGED       # 카드를 넣느라 얼굴이 안 보여도
    assert sim.run(5.0, []) in (PRESENT, ABSENT)


def test_glance_freezes_judgement(cfg: dict[str, Any]) -> None:
    sim = Sim(cfg)
    sim.run(1.5, NEAR)
    assert sim.run(3.0, [], gaze_off=True) == ENGAGED


# ---- 보완 기능: 거리·접근·주의 -------------------------------------------------

def test_distance_zone_and_approach_intention(cfg: dict[str, Any]) -> None:
    est = HumanStateEstimator(cfg["timing"], cfg["vision"])
    est.update_faces([Face(1, 0.06)], 0.0)
    assert est.cues()["zone"] == "far"
    for i, h in enumerate([0.08, 0.10, 0.12, 0.14]):     # 0.8초 동안 얼굴이 커짐
        est.update_faces([Face(1, h)], 0.2 * (i + 1))
    assert est.cues()["zone"] == "social" and est.cues()["approaching"] is True
    for i in range(8):                                     # 멈춰 서면 접근 아님
        est.update_faces([Face(1, 0.14)], 1.0 + 0.2 * (i + 1))
    assert est.cues()["approaching"] is False
    est.update_faces([Face(1, 0.25)], 3.0)
    assert est.cues()["zone"] == "personal"


def test_attention_classification(cfg: dict[str, Any]) -> None:
    est = HumanStateEstimator(cfg["timing"], cfg["vision"])
    est.update_faces([Face(1, 0.3, yaw=3, pitch=2)], 0.0)
    assert est.attention == "robot"
    est.update_faces([Face(1, 0.3, yaw=35, pitch=18)], 0.1)
    assert est.attention == "screen"
    est.update_faces([Face(1, 0.3, yaw=-55, pitch=0)], 0.2)
    assert est.attention == "away"


def test_staring_at_robot_is_a_help_request(cfg: dict[str, Any]) -> None:
    sim = Sim(cfg)
    looking = [Face(1, 0.30, yaw=0, pitch=0)]
    sim.run(1.5, looking)
    awaiting = sim.now
    est = sim.est
    end = sim.now + 3.0
    while sim.now < end:
        sim.now = round(sim.now + 0.1, 6)
        est.update_faces(looking, sim.now)
        est.step(sim.now, awaiting, use_attention=True)
    assert est.state == NO_RESPONSE and est.reason == "seek_help"


def test_looking_away_does_not_trigger_hesitation(cfg: dict[str, Any]) -> None:
    sim = Sim(cfg)
    away = [Face(1, 0.30, yaw=-55, pitch=0)]
    sim.run(1.5, [Face(1, 0.30)])
    sim.est.note_input(sim.now)
    awaiting = sim.now
    est = sim.est
    end = sim.now + 10.0
    while sim.now < end:
        sim.now = round(sim.now + 0.1, 6)
        est.update_faces(away, sim.now)
        est.step(sim.now, awaiting, use_attention=True)
    assert est.state != NO_RESPONSE


def test_reading_the_screen_counts_as_engaged(cfg: dict[str, Any]) -> None:
    sim = Sim(cfg)
    reading = [Face(1, 0.30, yaw=35, pitch=18)]
    end = 2.0
    while sim.now < end:
        sim.now = round(sim.now + 0.1, 6)
        sim.est.update_faces(reading, sim.now)
        sim.est.step(sim.now, None, use_attention=True)
    assert sim.est.state == ENGAGED
