"""Expressive-vs-neutral side-by-side video for the Motion tab (Qt-free; data_kind simulated, never the robot).

The video itself is ``sim_video.render_pair`` - the same offline MuJoCo replay of the 50 Hz command stream as the
``reachy-motion video`` command (no daemon, simulated time). The Motion tab runs it in a separate Python process
(``python -m reachy_motion.gui.video ...``): MuJoCo's offscreen OpenGL context stays out of the Qt process and the
rendering never holds the GUI process's GIL (the 50 Hz executor thread lives there).

A job folder ``results/lab/videos/<time>_<motion>/`` holds ``specs/`` (the exact two motion files that were
rendered - the working copies when they were edited), the video (animated WebP for the GUI, H.264 MP4 too when
PyAV is installed), ``index.html`` (the video with its table) and ``manifest.json``. A finished video is found again
by the digests of the two specs, so a motion that did not change is not rendered twice.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

VIDEO_ROOT = Path("results") / "lab" / "videos"
SEARCH_ROOTS = (VIDEO_ROOT, Path("results") / "phase3_sim" / "videos")  # GUI videos first, then `video` CLI runs
TOOL = "reachy-motion Motion Lab (E/N side-by-side video)"
LEAD_S, TAIL_S = 0.5, 1.5  # the `video` command's defaults: rest before, hold after the longer variant


@dataclass
class VideoJob:
    name: str
    digests: tuple[str, str]  # (expressive, neutral) spec digests
    out_dir: Path
    spec_files: tuple[Path, Path]
    title: str
    edited: bool
    fps: int = 25
    size: tuple[int, int] = (560, 420)

    def command(self, python: str | None = None) -> list[str]:
        return [python or sys.executable, "-m", "reachy_motion.gui.video", "--out", str(self.out_dir),
                "--title", self.title, "--fps", str(self.fps), "--size", f"{self.size[0]}x{self.size[1]}",
                str(self.spec_files[0]), str(self.spec_files[1])]


@dataclass
class FoundVideo:
    folder: Path
    webp: Path
    entry: dict[str, Any] = field(default_factory=dict)  # render_pair summary (motions, fps, frames, ...)

    @property
    def index_html(self) -> Path:
        return self.folder / "index.html"

    @property
    def mp4(self) -> Path | None:
        f = (self.entry.get("files") or {}).get("mp4")
        return self.folder / f if f and (self.folder / f).is_file() else None


def title_for(name: str, edited: bool) -> str:
    """Video title (ASCII: the frame font has no Hangul)."""
    return f"{name}  ({'edited working copy' if edited else 'motions/'})  -  expressive vs neutral"


def prepare_job(expressive: Any, neutral: Any, root: Path | str = VIDEO_ROOT, *, edited: bool = False,
                fps: int = 25, size: tuple[int, int] = (560, 420), stamp: str | None = None) -> VideoJob:
    """Create the job folder and write the two specs into ``specs/`` (what will be rendered, exactly)."""
    from reachy_motion.library.io import save_spec_file

    if expressive.variant != "expressive" or neutral.variant != "neutral" or expressive.name != neutral.name:
        raise ValueError("need the expressive and the neutral variant of one motion")
    out = Path(root) / f"{stamp or datetime.now().strftime('%Y%m%d_%H%M%S')}_{expressive.name}"
    spec_dir = out / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    files = (save_spec_file(expressive, spec_dir), save_spec_file(neutral, spec_dir))
    return VideoJob(name=expressive.name, digests=(expressive.digest(), neutral.digest()), out_dir=out,
                    spec_files=files, title=title_for(expressive.name, edited), edited=edited, fps=fps, size=size)


def _webp_of(folder: Path, entry: dict[str, Any]) -> Path | None:
    files = entry.get("files") or {}
    name = files.get("webp") or (entry.get("file") if str(entry.get("file", "")).endswith(".webp") else None)
    p = folder / name if name else None
    return p if p is not None and p.is_file() else None


def find_video(digests: Sequence[str], roots: Sequence[Path | str] = SEARCH_ROOTS) -> FoundVideo | None:
    """Newest finished video (with a WebP the GUI can play) of exactly these two spec digests."""
    want = sorted(digests)
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for folder in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
            m = folder / "manifest.json"
            if not m.is_file():
                continue
            try:
                manifest = json.loads(m.read_text(encoding="utf-8"))
            except ValueError:
                continue
            for entry in manifest.get("videos") or []:
                got = sorted(str(x.get("spec_digest")) for x in entry.get("motions") or [])
                webp = _webp_of(folder, entry)
                if got == want and webp is not None:
                    return FoundVideo(folder=folder, webp=webp, entry=entry)
    return None


def table_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per variant, from the render summary: length, design-target V/A, simulated end error and gap."""
    rows = []
    for m in entry.get("motions") or []:
        e = m.get("end_error") or {}
        rows.append({"variant": str(m.get("key", ":")).split(":")[-1], "duration_s": m.get("duration_s"),
                     "valence": (m.get("expected") or {}).get("valence"),
                     "arousal": (m.get("expected") or {}).get("arousal"),
                     "end_error_deg": e.get("head_angle_deg"), "end_error_mm": e.get("head_trans_mm"),
                     "max_gap_deg": m.get("max_command_gap_deg"), "intent": m.get("intent", "")})
    return rows


# ------------------------------------------------------------------------------------------------ runner process
def main(argv: Sequence[str] | None = None) -> int:
    from dataclasses import asdict

    from reachy_motion import sim_video
    from reachy_motion.library.io import load_spec_file
    from reachy_motion.safety.checker import prepare
    from reachy_motion.sdk_env import environment_report, pin_mismatches

    ap = argparse.ArgumentParser(prog="python -m reachy_motion.gui.video",
                                 description="Render one expressive-vs-neutral video (offline MuJoCo, simulated).")
    ap.add_argument("expressive")
    ap.add_argument("neutral")
    ap.add_argument("--out", required=True, help="job folder (the video, index.html and manifest.json go here)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--fps", type=int, default=25, choices=(10, 25, 50))
    ap.add_argument("--size", default="560x420", help="panel size WIDTHxHEIGHT (max 1280x720)")
    args = ap.parse_args(argv)
    if pin_mismatches():
        print("SDK pins do not match; refusing to run (see `reachy-motion env`).", file=sys.stderr)
        return 1
    try:
        w, h = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        print(f"--size must be WIDTHxHEIGHT, got {args.size!r}", file=sys.stderr)
        return 2
    if not (16 <= w <= 1280 and 16 <= h <= 720):  # the scene's offscreen buffer is 1280x720 (scenes/empty.xml:9)
        print("--size must be within 16x16 .. 1280x720", file=sys.stderr)
        return 2
    try:
        specs = [load_spec_file(args.expressive), load_spec_file(args.neutral)]
    except (OSError, ValueError) as exc:
        print(f"cannot read the motion files: {exc}", file=sys.stderr)
        return 2
    if [s.variant for s in specs] != ["expressive", "neutral"] or specs[0].name != specs[1].name:
        print("need the expressive and the neutral file of one motion, in that order", file=sys.stderr)
        return 2
    for s in specs:  # the same gate as every other player (render_pair refuses them too)
        prep = prepare(s)
        if not prep.runnable:
            codes = ", ".join(sorted({v.code for v in prep.report.violations}))
            print(f"{s.key}: outside the initial policy ({codes}); not rendered", file=sys.stderr)
            return 3
    formats = ("mp4", "webp") if sim_video.mp4_available() else ("webp",)
    out = Path(args.out)
    entry = sim_video.render_pair(specs, specs[0].name, out, title=args.title or title_for(specs[0].name, False),
                                  fps=args.fps, size=(w, h), lead_s=LEAD_S, tail_s=TAIL_S, formats=formats)
    entry["dir"] = str(Path(args.expressive).parent)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": TOOL,
        "command": "python -m reachy_motion.gui.video " + " ".join(sys.argv[1:] if argv is None else argv),
        "data_kind": sim_video.DATA_KIND, "note": sim_video.NOTE,
        "spec_files": [str(Path(args.expressive)), str(Path(args.neutral))],
        "parameters": {"formats": list(formats), "fps": args.fps, "panel_size": [w, h], "scene": "empty",
                       "lead_s": LEAD_S, "tail_s": TAIL_S, "camera": asdict(sim_video.REVIEW_CAMERA),
                       "physics_dt_s": sim_video.PHYSICS_DT_S, "control_rate_hz": sim_video.CONTROL_RATE_HZ},
        "videos": [entry], "skipped": [], "environment": environment_report()}
    sim_video.write_index(out, [entry], manifest)
    print(json.dumps({"ok": True, "folder": str(out), "files": entry.get("files"), "video_s": entry.get("video_s")}))
    return 0


__all__ = ["VIDEO_ROOT", "SEARCH_ROOTS", "VideoJob", "FoundVideo", "title_for", "prepare_job", "find_video",
           "table_rows", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
