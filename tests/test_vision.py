"""FacePerception: 얼굴 상자 → 추적 id·크기·yaw/pitch, 주 손님 눈높이 → 월드 방향 벡터 (가짜 landmarker로)."""

from __future__ import annotations

import math
import threading
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")  # 로봇 의존성(.[robot])이 없는 노트북 환경에서는 건너뜀
pytest.importorskip("reachy_mini")

from reachy_mini.vision.look_at import default_head_to_camera_transform  # noqa: E402

from reachy_cafe.config import load_config  # noqa: E402
from reachy_cafe.robot.vision import FacePerception, FaceTracker, head_angles_deg, iou  # noqa: E402

CFG = load_config(env={})
MODEL = {"K": np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]), "D": np.zeros(5),
         "resolution": (640, 480), "T_head_cam": default_head_to_camera_transform()}


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    m = np.eye(4)
    m[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    return m


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    m = np.eye(4)
    m[:3, :3] = [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]
    return m


def face(x0: float, y0: float, x1: float, y1: float) -> list[SimpleNamespace]:
    return [SimpleNamespace(x=x0, y=y0), SimpleNamespace(x=x1, y=y1), SimpleNamespace(x=(x0 + x1) / 2, y=y1)]


class FakeLandmarker:
    def __init__(self) -> None:
        self.next: tuple[list, list] = ([], [])
        self.timestamps: list[int] = []

    def detect_for_video(self, _image, ts: int):
        self.timestamps.append(ts)
        landmarks, matrices = self.next
        return SimpleNamespace(face_landmarks=landmarks, facial_transformation_matrixes=matrices)


class FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self._lock = threading.Lock()

    def publish_threadsafe(self, kind: str, **data) -> None:
        with self._lock:
            self.events.append((kind, data))

    log_threadsafe = publish_threadsafe


def make(primary=None):
    bus, lm, rays = FakeBus(), FakeLandmarker(), []
    perception = FacePerception(CFG, mini=None, bus=bus, model_path=None, camera_model=lambda: MODEL,
                                on_user_ray=rays.append, primary_id=primary, landmarker=lm)
    perception._mp_image = staticmethod(lambda rgb: rgb)  # mediapipe.Image 없이
    return bus, lm, rays, perception


def test_head_angles_from_transformation_matrix():
    yaw, pitch = head_angles_deg(rot_y(30.0))
    assert yaw == pytest.approx(30.0, abs=0.1) and pitch == pytest.approx(0.0, abs=0.1)
    yaw, pitch = head_angles_deg(rot_x(20.0).reshape(-1))  # MediaPipe는 16개 값으로 줄 수도 있다
    assert yaw == pytest.approx(0.0, abs=0.1) and pitch == pytest.approx(20.0, abs=0.1)


def test_tracker_keeps_ids_while_faces_move_and_forgets_after_missing():
    tracker = FaceTracker(max_missing=2)
    a, b = (0.1, 0.1, 0.3, 0.4), (0.6, 0.2, 0.8, 0.5)
    first = tracker.assign([a, b])
    moved = tracker.assign([(0.62, 0.21, 0.82, 0.51), (0.12, 0.1, 0.32, 0.4)])  # 순서가 바뀌어도
    assert moved == [first[1], first[0]]
    for _ in range(3):
        tracker.assign([])
    assert tracker.assign([a])[0] not in first  # 오래 사라졌다 돌아오면 새 사람
    assert iou(a, a) == pytest.approx(1.0)


def test_process_publishes_faces_and_aims_at_the_eyes():
    bus, lm, rays, perception = make()
    lm.next = ([face(0.4, 0.3, 0.6, 0.7)], [rot_y(10.0)])
    faces = perception.process(np.zeros((480, 640, 3), np.uint8), np.eye(4))
    assert len(faces) == 1 and faces[0].h == pytest.approx(0.4) and faces[0].yaw == pytest.approx(10.0)
    kind, data = bus.events[-1]
    assert kind == "vision.faces" and data["faces"][0]["cx"] == pytest.approx(0.5)
    ray = rays[-1]
    assert ray[0] > 0.99 and abs(ray[1]) < 1e-6 and ray[2] > 0.0  # 정면, 눈높이라 살짝 위
    lm.next = ([face(0.05, 0.3, 0.25, 0.7)], [rot_y(0.0)])  # 이미지 왼쪽 = 로봇의 왼쪽(+y)
    perception.process(np.zeros((480, 640, 3), np.uint8), np.eye(4))
    assert rays[-1][1] > 0.3
    assert lm.timestamps == sorted(set(lm.timestamps))  # VIDEO 모드: 시각이 계속 늘어야 한다


def test_gaze_follows_the_primary_customer_not_the_biggest_face():
    holder = {"id": None}
    _bus, lm, rays, perception = make(primary=lambda: holder["id"])
    big, small = face(0.05, 0.2, 0.35, 0.8), face(0.7, 0.4, 0.8, 0.6)
    lm.next = ([big, small], [rot_y(0.0), rot_y(0.0)])
    faces = perception.process(np.zeros((480, 640, 3), np.uint8), np.eye(4))
    assert rays[-1][1] > 0  # 주 손님이 없으면 가장 큰 얼굴 (왼쪽)
    holder["id"] = faces[1].id
    perception.process(np.zeros((480, 640, 3), np.uint8), np.eye(4))
    assert rays[-1][1] < 0  # 추정기가 고른 주 손님 (오른쪽)
