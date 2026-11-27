"""사람 상태 추정기 (주문 상태와 분리해서 추적).

입력: 얼굴 목록(크기·위치·yaw·pitch), 터치·발화·카드 입력 시각, 로봇이 손님에게서 시선을 뗐는지 여부,
      FSM이 입력을 기다리기 시작한 시각.
출력:
- 상태 (바뀔 때만): ABSENT / PRESENT / ENGAGED / NO_RESPONSE / AMBIGUOUS / DETECTION_UNCERTAIN
- 단서(cues): 거리 구역(far/social/personal), 다가오는 중인지(intention), 주의 방향(robot/screen/away)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

ABSENT = "ABSENT"
PRESENT = "PRESENT"
ENGAGED = "ENGAGED"
NO_RESPONSE = "NO_RESPONSE"
AMBIGUOUS = "AMBIGUOUS"
UNCERTAIN = "DETECTION_UNCERTAIN"
STATES = (ABSENT, PRESENT, ENGAGED, NO_RESPONSE, AMBIGUOUS, UNCERTAIN)


@dataclass
class Face:
    id: int
    h: float                    # 얼굴 높이 / 프레임 높이 (가까울수록 큼)
    cx: float = 0.5             # 얼굴 중심 (0–1, 이미지 좌표)
    cy: float = 0.5
    yaw: float | None = 0.0     # 도 단위, 0 = 카메라(로봇) 정면. 모르면 None
    pitch: float | None = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "h": round(self.h, 3), "cx": round(self.cx, 3), "cy": round(self.cy, 3),
                "yaw": None if self.yaw is None else round(self.yaw, 1),
                "pitch": None if self.pitch is None else round(self.pitch, 1)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Face":
        return cls(id=int(raw.get("id", 0)), h=float(raw.get("h", 0.0)), cx=float(raw.get("cx", 0.5)),
                   cy=float(raw.get("cy", 0.5)), yaw=raw.get("yaw", 0.0), pitch=raw.get("pitch", 0.0))


class HumanStateEstimator:
    def __init__(self, timing: dict[str, float], vision: dict[str, float]):
        self.t = timing
        self.v = vision
        self.state = ABSENT
        self.reason: str | None = None      # NO_RESPONSE 이유: hesitation | seek_help
        self.faces: list[Face] = []
        self.primary_id: int | None = None
        self.score = 0.0
        self.gaze_off_user = False
        self.use_attention = False
        self.last_frame_t = -math.inf
        self.last_face_t = -math.inf
        self.last_input_t = -math.inf
        self.face_since: float | None = None
        self.engaged_since: float | None = None
        self.release_since: float | None = None
        self.multi_since: float | None = None
        # 단서
        self.zone = "none"
        self.approaching = False
        self.attention = "none"
        self.attention_since = -math.inf
        self._history: deque[tuple[float, int, float]] = deque()

    # ---- 입력 ---------------------------------------------------------------
    def update_faces(self, faces: list[Face], now: float) -> None:
        self.last_frame_t = now
        self.faces = list(faces)
        if faces:
            self.last_face_t = now
            if self.face_since is None:
                self.face_since = now
        else:
            self.face_since = None
        self._update_primary()
        self._update_cues(now)

    def note_input(self, now: float) -> None:
        self.last_input_t = now

    def set_gaze_off_user(self, off: bool) -> None:
        self.gaze_off_user = off

    def lock_primary(self) -> None:
        """'한 분씩' 상황에서 터치로 확인한 사람을 주 사용자로 고정 (가장 큰 얼굴)."""
        if self.faces:
            self.primary_id = max(self.faces, key=lambda f: f.h).id

    # ---- 단서: 거리·접근·주의 ----------------------------------------------------
    def cues(self) -> dict[str, Any]:
        return {"zone": self.zone, "approaching": self.approaching, "attention": self.attention}

    def _update_cues(self, now: float) -> None:
        v = self.v
        primary = self.primary()
        if primary is None:
            self.zone, self.approaching = "none", False
            self._set_attention("none", now)
            self._history.clear()
            return
        if primary.h >= v["engaged_face_h"]:
            self.zone = "personal"
        elif primary.h >= v.get("social_face_h", 0.08):
            self.zone = "social"
        else:
            self.zone = "far"

        # 다가오는 의도: 같은 사람의 얼굴이 짧은 시간에 눈에 띄게 커지면
        window = v.get("approach_window_s", 1.2)
        self._history.append((now, primary.id, primary.h))
        while self._history and (now - self._history[0][0] > window or self._history[0][1] != primary.id):
            self._history.popleft()
        oldest = self._history[0][2] if self._history else primary.h
        self.approaching = (len(self._history) >= 3 and primary.h >= oldest * (1 + v.get("approach_gain", 0.25))
                            and primary.h - oldest >= 0.02)

        # 주의 방향: 로봇(카메라) 정면 / 화면 쪽(옆·아래) / 딴 곳
        yaw = primary.yaw if primary.yaw is not None else 0.0
        pitch = primary.pitch if primary.pitch is not None else 0.0
        look = v.get("look_robot_deg", 15.0)
        side = yaw * v.get("screen_yaw_sign", 1.0)
        down = pitch * v.get("screen_pitch_sign", 1.0)
        if abs(yaw) <= look and abs(pitch) <= look:
            attention = "robot"
        elif (v.get("screen_yaw_min_deg", 15.0) <= side <= v.get("screen_yaw_max_deg", 70.0)
              or down >= v.get("screen_pitch_down_deg", 15.0)):
            attention = "screen"
        else:
            attention = "away"
        self._set_attention(attention, now)

    def _set_attention(self, attention: str, now: float) -> None:
        if attention != self.attention:
            self.attention = attention
            self.attention_since = now

    # ---- 판정 ---------------------------------------------------------------
    def primary(self) -> Face | None:
        for face in self.faces:
            if face.id == self.primary_id:
                return face
        return None

    def _update_primary(self) -> None:
        if self.primary() is not None or not self.faces:
            return
        # 가장 크고(가깝고) 가운데에 가까운 얼굴
        best = max(self.faces, key=lambda f: f.h - 0.2 * abs(f.cx - 0.5))
        self.primary_id = best.id

    def step(self, now: float, awaiting_since: float | None = None, use_attention: bool = False) -> str | None:
        self.use_attention = use_attention
        new = self._compute(now, awaiting_since)
        primary = self.primary()
        if primary is not None:
            closeness = min(1.0, primary.h / max(self.v["engaged_face_h"], 1e-6))
            frontal = 1.0 if primary.yaw is None else max(0.0, 1.0 - abs(primary.yaw) / 90.0)
            self.score = round(closeness * frontal, 2)
        else:
            self.score = 0.0
        if new != self.state:
            self.state = new
            return new
        return None

    def _compute(self, now: float, awaiting_since: float | None) -> str:
        t, v = self.t, self.v

        # 로봇이 화면·슬롯을 보거나 감정 모션 중이면 카메라가 손님을 못 본다 → 판정 동결
        if self.gaze_off_user:
            self.last_face_t = now
            if self.state == NO_RESPONSE and now - self.last_input_t <= t["recent_input_s"]:
                return ENGAGED  # 방금 대답했으면 시선을 뗀 동안에도 "망설임"은 풀어 준다
            return self.state
        # 프레임이 끊기면 이탈로 판단하지 않고 불확실 상태로 (터치·음성으로 계속)
        if now - self.last_frame_t > t["stale_frame_s"]:
            self.last_face_t = now
            return UNCERTAIN

        big = [f for f in self.faces if f.h >= v["ambiguous_face_h"]]
        if len(big) >= 2:
            if self.multi_since is None:
                self.multi_since = now
        else:
            self.multi_since = None
        if self.multi_since is not None and now - self.multi_since >= t["ambiguous_hold_s"]:
            return AMBIGUOUS

        primary = self.primary()
        facing = primary is not None and (primary.yaw is None or abs(primary.yaw) <= v["max_yaw_deg"])
        if self.use_attention and self.attention == "screen":
            facing = True  # 화면을 읽고 있는 것도 키오스크에 참여 중인 것
        close_engaged = primary is not None and primary.h >= v["engaged_face_h"] and facing
        if close_engaged:
            if self.engaged_since is None:
                self.engaged_since = now
            self.release_since = None
        else:
            self.engaged_since = None
            if self.release_since is None:
                self.release_since = now

        engaged = False
        if self.engaged_since is not None and now - self.engaged_since >= t["engaged_hold_s"]:
            engaged = True
        elif (self.state in (ENGAGED, NO_RESPONSE) and self.release_since is not None
              and now - self.release_since < t["engaged_release_s"]):
            engaged = True  # 히스테리시스: 잠깐 고개를 돌려도 유지
        if now - self.last_input_t <= t["recent_input_s"]:
            engaged = True  # 다중 모달 존재 융합: 최근에 터치·발화·카드가 있으면 있는 것

        if engaged:
            if awaiting_since is not None:
                idle = now - max(self.last_input_t, awaiting_since)
                if self.use_attention and primary is not None:
                    if self.attention == "away":
                        return ENGAGED  # 딴 곳(동행과 대화 등)을 보면 재촉하지 않는다
                    if (self.attention == "robot" and now - self.attention_since >= t.get("seek_help_s", 2.5)
                            and idle >= t.get("seek_help_s", 2.5)):
                        self.reason = "seek_help"  # 말없이 로봇을 쳐다보면 도움을 청하는 것으로 본다
                        return NO_RESPONSE
                if idle >= t["hesitation_s"]:
                    self.reason = "hesitation"
                    return NO_RESPONSE
            return ENGAGED

        if primary is not None and self.face_since is not None and now - self.face_since >= t["present_confirm_s"]:
            return PRESENT
        if self.state != ABSENT and now - self.last_face_t < t["absent_s"]:
            return PRESENT  # 잠깐 놓친 얼굴은 바로 이탈로 보지 않는다
        return ABSENT

    def snapshot(self) -> dict[str, Any]:
        return {"state": self.state, "reason": self.reason, "faces": [f.to_dict() for f in self.faces],
                "primary": self.primary_id, "score": self.score, "gaze_off_user": self.gaze_off_user,
                **self.cues()}
