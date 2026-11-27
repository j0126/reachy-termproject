"""Offline MuJoCo replay of planned motions and expressive-vs-neutral review videos (simulated; no daemon, no robot).

Each motion is replayed through the same MuJoCo model and command law as the SDK 1.11.0 simulation backend
(site-packages ``reachy_mini/daemon/backend/mujoco/backend.py``), in simulated time, and rendered offscreen:

* model ``descriptions/reachy_mini/mjcf/scenes/<scene>.xml`` (backend.py:83-88); physics step 2 ms and a control
  update every 10 steps = 50 Hz (backend.py:90-91);
* at every control update ``ctrl[:7]`` = IK joints [body_yaw, stewart_1..6] and ``ctrl[-2:] = -antennas``
  (backend.py:257-260). The IK is the daemon's (AnalyticalKinematics, automatic_body_yaw=True) through
  ``safety.checker.analyze_timeline``; a sample without an IK solution keeps the previous joint command, as the
  daemon's control loop does (abstract.py:586-588);
* start: unnamed collision geoms off, joints and ctrl set, 100 steps, collisions on, 100 steps
  (backend.py:112-122, 188-210) - here at the motion's start state instead of the sleep pose - then ``settle_s``;
* head pose for the checks: site ``head`` with z - 0.177 m (backend.py:294-308).

Differences from ``sim-play`` / ``sim-verify`` (daemon): simulated time instead of the wall clock, so no real-time
factor (the daemon's MuJoCo runs at about 0.8-0.86x real time on this PC), no PC timing jitter and no network
path - the video shows the motion at its designed speed. Everything here is data_kind ``simulated``: MuJoCo is
friction-dominated and was far more optimistic than the robot in Phase 1, so a video is a design-review aid, not a
prediction of the robot. Only motions inside the initial policy are replayed (``prepare(spec).runnable``), as for
every other player. Videos are H.264 MP4 through PyAV's bundled libx264 (optional package ``av==19.0.1``, installed
offline from the uv cache; the SDK only imports it in ``reachy_mini/media/central_consumer.py``, which nothing
imports) and/or animated WebP (Pillow).
"""

from __future__ import annotations

import functools
import html
import json
import math
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reachy_motion.library.pose import rotation_distance, translation_distance
from reachy_motion.library.spec import MotionSpec

__all__ = [
    "PHYSICS_DT_S", "DECIMATION", "CONTROL_RATE_HZ", "ACTUATORS", "DATA_KIND", "CameraView", "REVIEW_CAMERA",
    "VIEWER_CAMERA", "VIDEO_FORMATS", "scene_path", "SimReplay", "ReplayResult", "replay", "compose_frame",
    "save_webp", "save_mp4", "mp4_available", "render_pair", "write_index",
]

PHYSICS_DT_S = 0.002  # backend.py:90
DECIMATION = 10  # backend.py:91: a control update every 10 physics steps
CONTROL_RATE_HZ = 1.0 / (PHYSICS_DT_S * DECIMATION)  # 50 Hz
HEAD_SITE = "head"  # backend.py:95-99
HEAD_SITE_Z_OFFSET_M = 0.177  # backend.py:307
ACTUATORS = ("yaw_body", "stewart_1", "stewart_2", "stewart_3", "stewart_4", "stewart_5", "stewart_6",
             "right_antenna", "left_antenna")  # MJCF actuator order (checked at load)
DATA_KIND = "simulated"
VIDEO_FORMATS = ("mp4", "webp")
NOTE ="simulated: offline MuJoCo replay of the 50 Hz command stream (SDK sim model) - not the robot"


@dataclass(frozen=True)
class CameraView:
    """MuJoCo free camera (MjvCamera) parameters."""

    distance: float = 0.68
    azimuth: float = 170.0
    elevation: float = -8.0
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.21)


REVIEW_CAMERA = CameraView()  # near-frontal, whole robot with the antennas in frame
VIEWER_CAMERA = CameraView(0.8, 160.0, -20.0, (0.0, 0.0, 0.15))  # the daemon viewer's camera (backend.py:175-179)


def scene_path(scene: str = "empty") -> Path:
    """The SDK's MuJoCo scene file (backend.py:83-88)."""
    import reachy_mini

    p = Path(str(files(reachy_mini).joinpath("descriptions/reachy_mini/mjcf/scenes"))) / f"{scene}.xml"
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


class SimReplay:
    """One MuJoCo world driven like the SDK simulation backend, advanced in simulated time."""

    def __init__(self, scene: str = "empty") -> None:
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene_path(scene)))
        self.model.opt.timestep = PHYSICS_DT_S
        self.data = mujoco.MjData(self.model)
        names = tuple(self.model.actuator(k).name for k in range(self.model.nu))
        if names != ACTUATORS:
            raise RuntimeError(f"unexpected MuJoCo actuators {names}; expected {ACTUATORS}")
        self.qpos_addr = np.array([int(np.ravel(self.model.joint(n).qposadr)[0]) for n in names])
        self.head_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, HEAD_SITE)
        if self.head_site < 0:
            raise RuntimeError(f"site {HEAD_SITE!r} not in the model")
        self._joints = np.zeros(7)
        self.time_s = 0.0

    def command(self, joints: Sequence[float], antennas: Sequence[float]) -> None:
        """Set the actuator targets as backend.py:257-260 (non-finite joints keep the previous command)."""
        j = np.asarray(joints, dtype=float)
        if j.shape != (7,):
            raise ValueError("joints must be [body_yaw, stewart_1..6]")
        if np.all(np.isfinite(j)):
            self._joints = j
        self.data.ctrl[:7] = self._joints
        self.data.ctrl[-2:] = -np.asarray(antennas, dtype=float)

    def start(self, joints: Sequence[float], antennas: Sequence[float], settle_s: float = 1.0) -> None:
        """Start-up of backend.py:112-122, 188-210 at the given state, then ``settle_s`` more; time restarts at 0."""
        mj, m = self._mj, self.model
        off = [i for i, ct in enumerate(m.geom_contype)
               if ct != 0 and mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, i) is None]
        for i in off:
            m.geom_contype[i] = 0
            m.geom_conaffinity[i] = 0
        self.command(joints, antennas)
        self.data.qpos[self.qpos_addr] = self.data.ctrl[:]
        mj.mj_forward(m, self.data)
        for _ in range(100):
            mj.mj_step(m, self.data)
        for i in off:
            m.geom_contype[i] = 1
            m.geom_conaffinity[i] = 1
        for _ in range(100 + int(round(settle_s / PHYSICS_DT_S))):
            mj.mj_step(m, self.data)
        self.time_s = 0.0

    def tick(self) -> None:
        """One 50 Hz control period (10 physics steps)."""
        for _ in range(DECIMATION):
            self._mj.mj_step(self.model, self.data)
        self.time_s += PHYSICS_DT_S * DECIMATION

    def head_pose(self) -> np.ndarray:
        """Simulated head pose (4x4, m) as backend.py:294-308."""
        m = np.eye(4)
        m[:3, :3] = self.data.site_xmat[self.head_site].reshape(3, 3)
        m[:3, 3] = self.data.site_xpos[self.head_site]
        m[2, 3] -= HEAD_SITE_Z_OFFSET_M
        return m

    def joints(self) -> np.ndarray:
        """Simulated head joints [body_yaw, stewart_1..6] (rad)."""
        return np.array(self.data.qpos[self.qpos_addr[:7]], dtype=float)

    def antennas(self) -> np.ndarray:
        """Simulated antennas [right, left] in the SDK convention (inverse of the command mapping)."""
        return -np.array(self.data.qpos[self.qpos_addr[-2:]], dtype=float)


@dataclass
class ReplayResult:
    """One motion replayed in MuJoCo (data_kind simulated). Frame k shows video time k / fps."""

    key: str
    spec_digest: str
    duration_s: float  # motion length
    lead_s: float  # rest before the motion in the video
    total_s: float  # video length
    fps: int
    frames: list[np.ndarray] = field(default_factory=list)  # RGB uint8 (empty without rendering)
    frame_info: list[dict[str, Any]] = field(default_factory=list)
    end_error: dict[str, float] = field(default_factory=dict)  # simulated end vs planned end, after the tail
    max_command_gap_deg: float = 0.0  # largest |simulated head - commanded head| angle during the motion
    data_kind: str = DATA_KIND

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("frames")
        d.pop("frame_info")
        return d


def _label(labels: list[str], idx: int) -> str:
    if not labels:
        return "(none)"
    return labels[idx] if 0 <= idx < len(labels) else "-"


def replay(spec: MotionSpec, *, scene: str = "empty", lead_s: float = 0.5, tail_s: float = 1.0,
           total_s: float | None = None, fps: int = 25, render_size: tuple[int, int] | None = None,
           camera: CameraView = REVIEW_CAMERA, settle_s: float = 1.0) -> ReplayResult:
    """Replay ``spec`` (policy-gated) in MuJoCo; render frames when ``render_size`` = (width, height) is given.

    The video holds the start state for ``lead_s``, plays the motion, then holds its last command until
    ``total_s`` (default lead + duration + tail; a pair passes a common length).
    """
    from reachy_motion.safety.checker import analyze_timeline, prepare

    if fps <= 0 or CONTROL_RATE_HZ % fps:
        raise ValueError(f"fps must divide {CONTROL_RATE_HZ:g} (10, 25 or 50)")
    prep = prepare(spec)
    if not prep.runnable:
        codes = sorted({v.code for v in prep.report.violations})
        raise ValueError(f"{spec.key}: outside the initial policy ({', '.join(codes)}); not replayed")
    tl = prep.timeline
    if abs(tl.rate_hz - CONTROL_RATE_HZ) > 1e-9:
        raise ValueError(f"timeline rate {tl.rate_hz} Hz != control rate {CONTROL_RATE_HZ:g} Hz")
    joints = analyze_timeline(tl).joints
    eff = spec.effective()
    head_labels, ant_labels = [k.label for k in eff.head], [k.label for k in eff.antennas]
    n = len(tl.t)
    n_lead = int(round(lead_s * CONTROL_RATE_HZ))
    total = lead_s + tl.duration + tail_s if total_s is None else total_s
    n_total = max(int(round(total * CONTROL_RATE_HZ)), n_lead + n)
    every = int(CONTROL_RATE_HZ // fps)

    sim = SimReplay(scene)
    sim.start(joints[0], tl.antennas[0], settle_s)
    renderer = cam = None
    if render_size is not None:
        mj = sim._mj
        renderer = mj.Renderer(sim.model, height=int(render_size[1]), width=int(render_size[0]))
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = camera.distance, camera.azimuth, camera.elevation
        cam.lookat[:] = camera.lookat
    res = ReplayResult(key=spec.key, spec_digest=spec.digest(), duration_s=float(tl.duration), lead_s=lead_s,
                       total_s=n_total / CONTROL_RATE_HZ, fps=fps)
    try:
        for k in range(n_total):
            i = min(max(k - n_lead, 0), n - 1)
            if k % every == 0:
                phase = "start" if k < n_lead else ("end" if k >= n_lead + n else "move")
                res.frame_info.append({
                    "video_t": k / CONTROL_RATE_HZ, "motion_t": min(max((k - n_lead) / CONTROL_RATE_HZ, 0.0),
                                                                   float(tl.duration)),
                    "phase": phase,
                    "head_key": _label(head_labels, int(tl.head_key[i])) if phase == "move" else phase,
                    "antenna_key": _label(ant_labels, int(tl.antenna_key[i])) if phase == "move" else phase})
                if renderer is not None:
                    renderer.update_scene(sim.data, camera=cam)
                    res.frames.append(renderer.render().copy())
            sim.command(joints[i], tl.antennas[i])
            sim.tick()
            if n_lead <= k < n_lead + n:
                gap = math.degrees(rotation_distance(sim.head_pose(), tl.head[i]))
                res.max_command_gap_deg = max(res.max_command_gap_deg, gap)
    finally:
        if renderer is not None:
            renderer.close()
    end = sim.head_pose()
    res.end_error = {
        "head_angle_deg": math.degrees(rotation_distance(end, tl.head[-1])),
        "head_trans_mm": 1000.0 * translation_distance(end, tl.head[-1]),
        "antenna_deg": float(np.degrees(np.max(np.abs(sim.antennas() - tl.antennas[-1])))),
        "body_yaw_deg": float(np.degrees(abs(sim.joints()[0] - joints[-1][0]))),
    }
    return res


# --------------------------------------------------------------------------------------------- frames and files
_BG = (24, 30, 42)
_FG = (232, 236, 243)
_DIM = (160, 170, 185)
_ACCENT = {"expressive": (110, 168, 254), "neutral": (190, 196, 206)}


@functools.lru_cache(maxsize=None)
def _font(size: int) -> Any:
    from PIL import ImageFont

    return ImageFont.load_default(size=size)


def compose_frame(panels: Sequence[tuple[np.ndarray, dict[str, Any], str, float]], title: str,
                  note: str = NOTE) -> Any:
    """One video frame: a title bar, then the panels side by side, each with a footer (time, current keys, bar).

    ``panels`` = [(rgb, frame_info, variant, motion_duration_s), ...].
    """
    from PIL import Image, ImageDraw

    h = max(p[0].shape[0] for p in panels)
    widths = [p[0].shape[1] for p in panels]
    head_h, foot_h, gap = 52, 50, 4
    img = Image.new("RGB", (sum(widths) + gap * (len(panels) - 1), head_h + h + foot_h), _BG)
    draw = ImageDraw.Draw(img)
    draw.text((12, 6), title, fill=_FG, font=_font(20))
    draw.text((12, 31), note, fill=_DIM, font=_font(14))
    x = 0
    for (rgb, info, variant, dur), w in zip(panels, widths):
        img.paste(Image.fromarray(rgb), (x, head_h))
        accent = _ACCENT.get(variant, _FG)
        tag = f"{variant.upper()}  {dur:.2f} s"
        tw = int(draw.textlength(tag, font=_font(18)))
        draw.rectangle((x + 8, head_h + 8, x + 20 + tw, head_h + 34), fill=(16, 20, 28))
        draw.text((x + 14, head_h + 11), tag, fill=accent, font=_font(18))
        fy = head_h + h
        draw.text((x + 10, fy + 5), f"t {info['motion_t']:5.2f} / {dur:.2f} s", fill=_FG, font=_font(15))
        draw.text((x + 10, fy + 25), f"head: {info['head_key']}   antennas: {info['antenna_key']}", fill=_DIM,
                  font=_font(15))
        frac = 0.0 if dur <= 0 else min(max(info["motion_t"] / dur, 0.0), 1.0)
        draw.rectangle((x, fy + foot_h - 4, x + w - 1, fy + foot_h - 1), fill=(48, 56, 72))
        draw.rectangle((x, fy + foot_h - 4, x + int((w - 1) * frac), fy + foot_h - 1), fill=accent)
        x += w + gap
    return img


def save_webp(images: Sequence[Any], path: Path, fps: int, quality: int = 80) -> Path:
    """Animated WebP, looping (Pillow)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    first, rest = images[0], list(images[1:])
    first.save(path, format="WEBP", save_all=True, append_images=rest, duration=int(round(1000 / fps)), loop=0,
               quality=quality, method=4)
    return path


def mp4_available() -> bool:
    """True when PyAV with an H.264 encoder (libx264) is importable."""
    try:
        import av

        av.codec.Codec("libx264", "w")
    except Exception:  # ImportError, or a PyAV build without libx264
        return False
    return True


def save_mp4(images: Any, path: Path, fps: int, crf: int = 20) -> Path:
    """H.264 MP4 (yuv420p, faststart) with PyAV's libx264; ``images`` may be a generator (frames are streamed).

    Odd frame sizes are cropped by one pixel (yuv420p needs even sizes).
    """
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = None
    with av.open(str(path), mode="w", options={"movflags": "faststart"}) as container:
        for k, img in enumerate(images):
            if stream is None:
                w, h = img.size[0] - img.size[0] % 2, img.size[1] - img.size[1] % 2
                stream = container.add_stream("libx264", rate=fps, options={"crf": str(crf), "preset": "medium"})
                stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
            if img.size != (stream.width, stream.height):
                img = img.crop((0, 0, stream.width, stream.height))
            frame = av.VideoFrame.from_image(img.convert("RGB"))
            frame.pts = k
            for packet in stream.encode(frame):
                container.mux(packet)
        if stream is None:
            raise ValueError("no frames to encode")
        for packet in stream.encode():
            container.mux(packet)
    return path


def render_pair(specs: Sequence[MotionSpec], label: str, out_dir: Path, *, title: str | None = None,
                fps: int = 25, size: tuple[int, int] = (560, 420), camera: CameraView = REVIEW_CAMERA,
                scene: str = "empty", lead_s: float = 0.5, tail_s: float = 1.0,
                formats: Sequence[str] = ("webp",)) -> dict[str, Any]:
    """Replay the specs (expressive first) on a common timeline and write ``<label>.<format>``; returns a summary.

    ``formats`` from VIDEO_FORMATS; the first one is the primary file shown by ``index.html``.
    """
    if not specs:
        raise ValueError("no motion to render")
    if not formats or any(f not in VIDEO_FORMATS for f in formats):
        raise ValueError(f"formats must be a non-empty subset of {VIDEO_FORMATS}")
    total = lead_s + max(s.duration for s in specs) + tail_s
    runs = [replay(s, scene=scene, lead_s=lead_s, total_s=total, fps=fps, render_size=size, camera=camera)
            for s in specs]
    n = min(len(r.frames) for r in runs)
    head = title or f"{specs[0].name}  [{label}]  -  " + " vs ".join(s.variant for s in specs)

    def frames() -> Any:
        for k in range(n):
            yield compose_frame([(r.frames[k], r.frame_info[k], s.variant, r.duration_s)
                                 for r, s in zip(runs, specs)], head)

    written: dict[str, Path] = {}
    if "webp" in formats:
        images = list(frames())
        written["webp"] = save_webp(images, out_dir / f"{label}.webp", fps)
        if "mp4" in formats:
            written["mp4"] = save_mp4(images, out_dir / f"{label}.mp4", fps)
        del images
    else:
        written["mp4"] = save_mp4(frames(), out_dir / f"{label}.mp4", fps)
    for r in runs:
        r.frames.clear()
    primary = written[formats[0]]
    return {"label": label, "file": primary.name, "files": {f: p.name for f, p in written.items()},
            "title": head, "fps": fps, "frames": n, "video_s": n / fps, "size_bytes": primary.stat().st_size,
            "data_kind": DATA_KIND,
            "motions": [{**r.summary(), "intent": s.intent, "expected": {"valence": s.valence, "arousal": s.arousal},
                         "draft": s.draft} for r, s in zip(runs, specs)]}


def write_index(out_dir: Path, entries: Sequence[dict[str, Any]], manifest: dict[str, Any]) -> Path:
    """``index.html`` (every video on one page, relative links) and ``manifest.json`` in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    cards = []
    for e in entries:
        rows = "".join(
            f"<tr><td>{html.escape(m['key'].split(':')[1])}</td><td>{m['duration_s']:.2f} s</td>"
            f"<td>{m['expected']['valence']:+.1f} / {m['expected']['arousal']:+.1f}</td>"
            f"<td>{m['end_error']['head_angle_deg']:.2f}&deg; / {m['end_error']['head_trans_mm']:.2f} mm</td>"
            f"<td>{m['max_command_gap_deg']:.1f}&deg;</td><td>{html.escape(m['intent'])}</td></tr>"
            for m in e["motions"])
        src = html.escape(e["file"])
        media = (f"<video src=\"{src}\" controls autoplay loop muted playsinline preload=\"metadata\" "
                 f"title=\"{html.escape(e['title'])}\"></video>" if e["file"].endswith(".mp4") else
                 f"<img src=\"{src}\" alt=\"{html.escape(e['title'])}\" loading=\"lazy\">")
        cards.append(
            f"<section><h2>{html.escape(e['label'])}</h2>{media}<table><tr><th>variant</th><th>length</th>"
            f"<th>expected V / A (design target)</th><th>end error (sim)</th><th>max sim-command gap</th>"
            f"<th>intent</th></tr>{rows}</table></section>")
    page = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Motion videos</title><style>"
        ":root{color-scheme:light dark;--bg:#f6f7f9;--fg:#1d2433;--muted:#5b6578;--card:#fff;--line:#d9dee7}"
        "@media (prefers-color-scheme:dark){:root{--bg:#141922;--fg:#e8ecf3;--muted:#9aa5b8;--card:#1d2433;"
        "--line:#323b4d}}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);"
        "font:15px/1.45 system-ui,sans-serif}main{max-width:1180px;margin:0 auto}h1{font-size:22px;margin:4px 0}"
        "p.note{color:var(--muted);margin:4px 0 16px}section{background:var(--card);border:1px solid var(--line);"
        "border-radius:10px;padding:12px;margin:0 0 18px}h2{font-size:18px;margin:0 0 8px}"
        "img,video{display:block;width:100%;height:auto;border-radius:6px;background:#000}"
        "table{border-collapse:collapse;width:100%;"
        "margin-top:8px;font-size:13px}th,td{text-align:left;padding:4px 6px;border-top:1px solid var(--line);"
        "vertical-align:top}th{color:var(--muted);font-weight:600}</style></head><body><main>"
        f"<h1>Motion videos (expressive vs neutral)</h1><p class=\"note\">{html.escape(NOTE)}. "
        f"Generated {html.escape(str(manifest.get('generated_utc', '')))}. Values are design targets and simulated "
        "checks, not measurements of the robot. Antenna direction and roll sign are unverified on the robot.</p>"
        + "".join(cards) + "</main></body></html>")
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path
