"""카메라 → 얼굴 (IMPLEMENTATION_PLAN.md 5.2절). HW2의 MediaPipe FaceLandmarker(VIDEO 모드, 변환 행렬로 yaw)를
여러 명으로 넓히고 pitch·크기·추적 id를 더했다.

- 프레임: 공유 ReachyMini의 media.get_frame() (BGR). 카메라는 30 fps, 처리는 [vision] fps (기본 15)
- 얼굴마다: 크기 h(거리 단서), 중심(cx, cy), yaw·pitch(주의 방향 단서), 프레임 사이 IoU로 같은 id 유지
- vision.faces 이벤트를 버스로 → HumanStateEstimator (프레임마다 보내므로 얼굴이 없어도 '프레임은 살아 있음'을 안다)
- 주 손님(추정기의 primary, 없으면 가장 큰 얼굴)의 눈높이 픽셀을 '프레임을 찍은 순간의 머리 자세'로 월드 방향
  벡터로 바꿔 MotionPlayer 시선 목표로 넘긴다. 머리가 움직이는 중에도 목표가 흔들리지 않는다.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..perception.human_state import Face
from .connection import pixel_ray_world

log = logging.getLogger(__name__)

EYE_LEVEL = 0.42  # 얼굴 상자 위에서부터 눈까지 비율 (시선은 눈을 향해야 '쳐다본다'로 읽힌다)


def head_angles_deg(matrix: Any) -> tuple[float, float]:
    """MediaPipe facial transformation matrix → (yaw, pitch) 도. yaw는 HW2와 같은 식 atan2(R02, R22).

    부호(어느 쪽이 화면 쪽인지, 숙이면 +인지)는 배치마다 [vision] screen_yaw_sign / screen_pitch_sign으로 보정한다.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.size == 16:
        m = m.reshape(4, 4)
    r = m[:3, :3]
    yaw = math.degrees(math.atan2(r[0, 2], r[2, 2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r[1, 2]))))
    return yaw, pitch


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = w * h
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 1e-12 else 0.0


class FaceTracker:
    """프레임 사이 얼굴 상자를 IoU로 이어 같은 사람에게 같은 id를 준다 (잠깐 놓쳐도 max_missing 프레임까지 유지)."""

    def __init__(self, iou_min: float = 0.2, max_missing: int = 8):
        self.iou_min = iou_min
        self.max_missing = max_missing
        self.tracks: dict[int, tuple[tuple[float, float, float, float], int]] = {}
        self._next = 1

    def assign(self, boxes: list[tuple[float, float, float, float]]) -> list[int]:
        pairs = sorted(((iou(box, tbox), i, tid) for i, box in enumerate(boxes)
                        for tid, (tbox, _miss) in self.tracks.items()), reverse=True)
        ids: list[int | None] = [None] * len(boxes)
        used: set[int] = set()
        for score, i, tid in pairs:
            if score < self.iou_min:
                break
            if ids[i] is None and tid not in used:
                ids[i] = tid
                used.add(tid)
        for i, box in enumerate(boxes):
            if ids[i] is None:
                ids[i] = self._next
                self._next += 1
        new_tracks = {tid: (boxes[i], 0) for i, tid in enumerate(ids)}
        for tid, (tbox, miss) in self.tracks.items():
            if tid not in new_tracks and miss + 1 <= self.max_missing:
                new_tracks[tid] = (tbox, miss + 1)
        self.tracks = new_tracks
        return [int(t) for t in ids]


class FacePerception:
    def __init__(self, cfg: dict[str, Any], mini: Any, bus: Any, model_path: str | None,
                 camera_model: Callable[[], dict[str, Any] | None] | None = None,
                 on_user_ray: Callable[[np.ndarray | None], None] | None = None,
                 primary_id: Callable[[], int | None] | None = None, landmarker: Any = None):
        vision = cfg["vision"]
        self.mini = mini
        self.bus = bus
        self.model_path = model_path
        self.fps = float(vision.get("fps", 15.0))
        self.num_faces = int(vision.get("num_faces", 3))
        self.camera_model = camera_model
        self.on_user_ray = on_user_ray
        self.primary_id = primary_id
        self.landmarker = landmarker
        self.tracker = FaceTracker()
        self._last_ts_ms = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._model: dict[str, Any] | None = None
        self.frames = 0
        self.no_frame = 0
        self.proc_ms = 0.0
        self.fps_measured = 0.0
        self._stats_t = time.monotonic()
        self._stats_frames = 0

    # ---- 시작·종료 -------------------------------------------------------------
    def start(self) -> None:
        if self.landmarker is None:
            self.landmarker = self._create_landmarker()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="vision", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        close = getattr(self.landmarker, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass

    def _create_landmarker(self) -> Any:
        import mediapipe as mp

        if not self.model_path:
            raise FileNotFoundError("face_landmarker.task not found (config [hw] face_model)")
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=self.num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # ---- 루프 ----------------------------------------------------------------
    def _loop(self) -> None:
        period = 1.0 / self.fps
        next_t = time.monotonic()
        while not self._stop.is_set():
            try:
                frame = self.mini.media.get_frame()
                head_pose = np.asarray(self.mini.get_current_head_pose(), dtype=np.float64).reshape(4, 4)
            except Exception as exc:  # noqa: BLE001 - 카메라 오류가 스레드를 죽이지 않게 (추정기가 끊김을 감지)
                log.warning("camera read failed: %s", exc)
                frame, head_pose = None, None
            if frame is None:
                self.no_frame += 1
                time.sleep(0.02)
                continue
            try:
                self.process(frame, head_pose)
            except Exception as exc:  # noqa: BLE001
                log.warning("face processing failed: %s", exc)
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.monotonic()  # 늦으면 따라잡으려 몰아서 처리하지 않는다

    def _timestamp_ms(self) -> int:
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts_ms:  # MediaPipe VIDEO 모드는 시각이 계속 늘어야 한다 (HW2)
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def process(self, frame: np.ndarray, head_pose: np.ndarray | None) -> list[Face]:
        """프레임 하나 → 얼굴 목록 (버스로 보내고 시선 목표도 갱신)."""
        t0 = time.perf_counter()
        height, width = frame.shape[:2]
        rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR → RGB (OpenCV 없이)
        result = self.landmarker.detect_for_video(self._mp_image(rgb), self._timestamp_ms())
        boxes: list[tuple[float, float, float, float]] = []
        angles: list[tuple[float | None, float | None]] = []
        matrices = list(getattr(result, "facial_transformation_matrixes", None) or [])
        for i, landmarks in enumerate(getattr(result, "face_landmarks", None) or []):
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            boxes.append((max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys))))
            angles.append(head_angles_deg(matrices[i]) if i < len(matrices) else (None, None))
        ids = self.tracker.assign(boxes)
        faces = [Face(id=fid, h=box[3] - box[1], cx=(box[0] + box[2]) / 2.0, cy=(box[1] + box[3]) / 2.0,
                      yaw=None if ang[0] is None else round(ang[0], 1),
                      pitch=None if ang[1] is None else round(ang[1], 1))
                 for fid, box, ang in zip(ids, boxes, angles)]
        self.bus.publish_threadsafe("vision.faces", faces=[f.to_dict() for f in faces], source="camera")
        self._update_gaze(faces, boxes, head_pose, width, height)
        self.proc_ms = (time.perf_counter() - t0) * 1000.0
        self.frames += 1
        self._stats(len(faces))
        return faces

    @staticmethod
    def _mp_image(rgb: np.ndarray) -> Any:
        import mediapipe as mp

        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def _update_gaze(self, faces: list[Face], boxes: list[tuple[float, float, float, float]],
                     head_pose: np.ndarray | None, width: int, height: int) -> None:
        if self.on_user_ray is None or head_pose is None or not faces:
            return
        model = self._model if self._model is not None else (self.camera_model() if self.camera_model else None)
        if model is None:
            return
        self._model = model
        want = self.primary_id() if self.primary_id is not None else None
        index = next((i for i, f in enumerate(faces) if f.id == want), None)
        if index is None:
            index = max(range(len(faces)), key=lambda i: faces[i].h)
        x0, y0, x1, y1 = boxes[index]
        cal_w, cal_h = model.get("resolution", (width, height))
        u = (x0 + x1) / 2.0 * cal_w  # K는 캘리브레이션 해상도 기준 → 정규화 좌표에서 바로 그 해상도로
        v = (y0 + EYE_LEVEL * (y1 - y0)) * cal_h
        self.on_user_ray(pixel_ray_world(u, v, model, head_pose))

    def _stats(self, n_faces: int) -> None:
        self._stats_frames += 1
        now = time.monotonic()
        if now - self._stats_t >= 10.0:
            fps = self._stats_frames / (now - self._stats_t)
            self.fps_measured = fps
            self.bus.log_threadsafe("vision.stats", fps=round(fps, 1), proc_ms=round(self.proc_ms, 1),
                                    faces=n_faces, no_frame=self.no_frame)
            self._stats_t, self._stats_frames, self.no_frame = now, 0, 0
