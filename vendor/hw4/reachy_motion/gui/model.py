"""Qt-free editing model of the Motion Editor: files, key edits, analysis, comparison, export.

The model never touches a backend - it cannot move anything (docs/VERIFICATION.md V2.11). Edits are immutable
``dataclasses.replace`` updates of ``MotionSpec`` objects; analysis is the motion engine's own planner and initial-policy
checker, cached by spec digest. Values are edited in human units (mm, deg) and stored in SI inside the spec.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from reachy_motion.library.pose import Pose6
from reachy_motion.library.spec import (
    MIN_KEY_DURATION_S,
    MOTION_NAMES,
    TRACKS,
    VARIANTS,
    AntennaKey,
    BodyKey,
    Easing,
    HeadKey,
    MotionSpec,
    SoundCue,
)
from reachy_motion.safety.policy import DEFAULT_POLICY, SafetyPolicy, SafetyReport

DEG = math.pi / 180.0
#: (human field, Pose6 field, SI per human unit)
HEAD_FIELDS: tuple[tuple[str, str, float], ...] = (
    ("x_mm", "x", 1e-3), ("y_mm", "y", 1e-3), ("z_mm", "z", 1e-3),
    ("roll_deg", "roll", DEG), ("pitch_deg", "pitch", DEG), ("yaw_deg", "yaw", DEG),
)
ANTENNA_FIELDS: tuple[str, ...] = ("right_deg", "left_deg")
BODY_FIELDS: tuple[str, ...] = ("yaw_deg",)
TRACK_FIELDS: dict[str, tuple[str, ...]] = {
    "head": tuple(f for f, _, _ in HEAD_FIELDS), "antennas": ANTENNA_FIELDS, "body": BODY_FIELDS,
}
#: Editor ranges (human units). A little wider than the initial policy envelope on purpose: values outside the
#: policy can be tried and are then flagged by the checker (and cannot be played).
FIELD_RANGES: dict[str, tuple[float, float]] = {
    "x_mm": (-30.0, 40.0), "y_mm": (-50.0, 50.0), "z_mm": (-50.0, 35.0),
    "roll_deg": (-45.0, 45.0), "pitch_deg": (-45.0, 45.0), "yaw_deg": (-90.0, 90.0),
    "right_deg": (-180.0, 180.0), "left_deg": (-180.0, 180.0), "body_yaw_deg": (-160.0, 160.0),
    "duration_s": (MIN_KEY_DURATION_S, 10.0), "hold_s": (0.0, 10.0),
    "amplitude": (0.0, 1.5), "tempo": (0.25, 3.0), "valence": (-1.0, 1.0), "arousal": (-1.0, 1.0),
}
DEFAULT_NEW_KEY_DURATION_S = 0.5


@dataclass
class AnalysisBundle:
    """Everything the editor shows for one spec (computed by the motion engine)."""

    spec: MotionSpec
    problems: list[str]  # spec.validate() problems; analysis is empty when there are any
    timeline: Any = None  # reachy_motion.timeline.Timeline at the executor rate
    analysis: Any = None  # reachy_motion.safety.checker.PlanAnalysis (dense samples)
    report: SafetyReport | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)  # planner.segment_table (executed parameters)
    conformance: list[Any] = field(default_factory=list)  # compare.Issue (neutral rules)
    error: str | None = None  # unexpected engine error text

    @property
    def passes_policy(self) -> bool:
        return self.report is not None and bool(self.report.passes_policy) and not self.problems and not self.error


def key_value_human(track: str, key: Any) -> dict[str, float]:
    if track == "head":
        return {f: getattr(key.pose, attr) / k for f, attr, k in HEAD_FIELDS}
    if track == "antennas":
        return {"right_deg": key.antennas[0] / DEG, "left_deg": key.antennas[1] / DEG}
    return {"yaw_deg": key.yaw / DEG}


def _with_value(track: str, key: Any, field_name: str, value: float) -> Any:
    if track == "head":
        for f, attr, k in HEAD_FIELDS:
            if f == field_name:
                return replace(key, pose=replace(key.pose, **{attr: float(value) * k}))
    elif track == "antennas" and field_name in ANTENNA_FIELDS:
        r, l = key.antennas
        return replace(key, antennas=(float(value) * DEG, l) if field_name == "right_deg" else (r, float(value) * DEG))
    elif track == "body" and field_name == "yaw_deg":
        return replace(key, yaw=float(value) * DEG)
    raise KeyError(f"{track} has no field {field_name!r}")


class EditorModel:
    """Working copies of motion specs keyed by (name, variant), plus analysis and export."""

    def __init__(self, motions_dir: Path | None = None, policy: SafetyPolicy = DEFAULT_POLICY) -> None:
        from reachy_motion.library.registry import motions_dir as default_dir

        self.dir = Path(motions_dir) if motions_dir is not None else default_dir()
        self.policy = policy
        self._working: dict[tuple[str, str], MotionSpec] = {}
        self._saved: dict[tuple[str, str], str] = {}  # digest at load / save
        self._paths: dict[tuple[str, str], Path] = {}
        self.current: tuple[str, str] | None = None
        self._cache: dict[str, AnalysisBundle] = {}
        self._listeners: list[Callable[[str], None]] = []

    # ------------------------------------------------------------------ events
    def subscribe(self, fn: Callable[[str], None]) -> None:
        self._listeners.append(fn)

    def _notify(self, what: str) -> None:
        for fn in list(self._listeners):
            fn(what)

    # ------------------------------------------------------------------ files
    def path_for(self, name: str, variant: str) -> Path:
        return self.dir / f"{name}.{variant}.yaml"

    def available(self) -> list[tuple[str, str]]:
        out = []
        for n in MOTION_NAMES:
            for v in VARIANTS:
                if self.path_for(n, v).exists() or (n, v) in self._working:
                    out.append((n, v))
        return out

    def open(self, name: str, variant: str) -> MotionSpec:
        """Select (name, variant): the working copy if one exists, else load the file."""
        k = (name, variant)
        if k not in self._working:
            self._load_into(k, self.path_for(name, variant))
        self.current = k
        self._notify("file")
        return self._working[k]

    def open_file(self, path: Path) -> MotionSpec:
        from reachy_motion.library.io import load_spec_file

        spec = load_spec_file(Path(path))
        k = (spec.name, spec.variant)
        self._working[k] = spec
        self._saved[k] = spec.digest()
        self._paths[k] = Path(path)
        self.current = k
        self._notify("file")
        return spec

    def _load_into(self, k: tuple[str, str], path: Path) -> None:
        from reachy_motion.library.io import load_spec_file

        spec = load_spec_file(path)
        if (spec.name, spec.variant) != k:
            raise ValueError(f"{path} holds {spec.key}, expected {k[0]}:{k[1]}")
        self._working[k] = spec
        self._saved[k] = spec.digest()
        self._paths[k] = path

    def revert(self) -> MotionSpec:
        k = self._require()
        self._working.pop(k, None)
        self._load_into(k, self._paths.get(k, self.path_for(*k)))
        self._notify("spec")
        return self._working[k]

    def save(self, path: Path | None = None) -> Path:
        """Write the current working copy (YAML by default, JSON for a .json path)."""
        from reachy_motion.library.io import save_spec_file

        k = self._require()
        target = Path(path) if path is not None else self._paths.get(k, self.path_for(*k))
        save_spec_file(self._working[k], target)
        if target.suffix.lower() in (".yaml", ".yml") and target.parent.resolve() == self.dir.resolve():
            self._paths[k] = target
        self._saved[k] = self._working[k].digest()
        self._notify("file")
        return target

    def current_path(self) -> Path:
        """The file ``save()`` without a path would write."""
        k = self._require()
        return self._paths.get(k, self.path_for(*k))

    @staticmethod
    def is_signed_off_file(path: Path) -> bool:
        """True when ``path`` holds a motion the student signed off (``draft: false``): the GUI never overwrites it."""
        from reachy_motion.library.io import load_spec_file

        p = Path(path)
        if not p.is_file():
            return False
        try:
            return not load_spec_file(p).draft
        except (OSError, ValueError):
            return False

    @property
    def spec(self) -> MotionSpec:
        return self._working[self._require()]

    def working(self, name: str, variant: str) -> MotionSpec | None:
        """Working copy if edited / opened this session, else the file on disk, else None."""
        k = (name, variant)
        if k in self._working:
            return self._working[k]
        p = self.path_for(name, variant)
        if p.exists():
            self._load_into(k, p)
            return self._working[k]
        return None

    def dirty(self, k: tuple[str, str] | None = None) -> bool:
        k = k or self.current
        return k is not None and k in self._working and self._working[k].digest() != self._saved.get(k)

    def dirty_keys(self) -> list[tuple[str, str]]:
        return [k for k in self._working if self.dirty(k)]

    def _require(self) -> tuple[str, str]:
        if self.current is None:
            raise RuntimeError("no motion selected")
        return self.current

    # ------------------------------------------------------------------ edits (immutable)
    def set_spec(self, spec: MotionSpec) -> None:
        k = self._require()
        if (spec.name, spec.variant) != k:
            raise ValueError("set_spec cannot change name or variant")
        self._working[k] = spec
        self._notify("spec")

    def keys(self, track: str) -> tuple[Any, ...]:
        return self.spec.track(track)

    def key_rows(self, track: str) -> list[dict[str, Any]]:
        """Designed keys with their (designed, tempo 1.0) times - what the table shows and edits."""
        rows, t = [], 0.0
        for i, k in enumerate(self.keys(track)):
            row = {"index": i, "label": k.label, "t_depart_s": t, "t_arrive_s": t + k.duration,
                   "t_release_s": t + k.duration + k.hold, "duration_s": k.duration, "hold_s": k.hold,
                   "easing": k.easing.value}
            row.update(key_value_human(track, k))
            rows.append(row)
            t += k.duration + k.hold
        return rows

    def _replace_key(self, track: str, index: int, new_key: Any) -> None:
        keys = list(self.keys(track))
        keys[index] = new_key
        self.set_spec(replace(self.spec, **{track: tuple(keys)}))

    def set_key_value(self, track: str, index: int, field_name: str, value: float) -> None:
        self._replace_key(track, index, _with_value(track, self.keys(track)[index], field_name, value))

    def set_key_timing(self, track: str, index: int, duration_s: float | None = None, hold_s: float | None = None,
                       t_arrive_s: float | None = None) -> None:
        k = self.keys(track)[index]
        if t_arrive_s is not None:
            depart = self.key_rows(track)[index]["t_depart_s"]
            duration_s = t_arrive_s - depart
        changes: dict[str, float] = {}
        if duration_s is not None:
            if not math.isfinite(duration_s) or duration_s < MIN_KEY_DURATION_S:
                raise ValueError(f"duration must be >= {MIN_KEY_DURATION_S} s")
            changes["duration"] = float(duration_s)
        if hold_s is not None:
            if not math.isfinite(hold_s) or hold_s < 0.0:
                raise ValueError("hold (pause) must be >= 0 s")
            changes["hold"] = float(hold_s)
        self._replace_key(track, index, replace(k, **changes))

    def set_key_easing(self, track: str, index: int, easing: Easing | str) -> None:
        self._replace_key(track, index, replace(self.keys(track)[index], easing=Easing(easing)))

    def set_key_label(self, track: str, index: int, label: str) -> None:
        self._replace_key(track, index, replace(self.keys(track)[index], label=str(label)))

    def add_key(self, track: str, after: int | None = None) -> int:
        """Insert a key after ``after`` (default: at the end) holding the same value; returns its index."""
        keys = list(self.keys(track))
        pos = len(keys) if after is None else after + 1
        easing = Easing.LINEAR if self.spec.variant == "neutral" else Easing.MIN_JERK
        if keys:
            src = keys[min(max(pos - 1, 0), len(keys) - 1)]
            new = replace(src, duration=DEFAULT_NEW_KEY_DURATION_S, hold=0.0, easing=easing, label="")
        elif track == "head":
            new = HeadKey(DEFAULT_NEW_KEY_DURATION_S, self.spec.start.head, easing)
        elif track == "antennas":
            new = AntennaKey(DEFAULT_NEW_KEY_DURATION_S, self.spec.start.antennas, easing)
        else:
            new = BodyKey(DEFAULT_NEW_KEY_DURATION_S, self.spec.start.body_yaw, easing)
        keys.insert(pos, new)
        self.set_spec(replace(self.spec, **{track: tuple(keys)}))
        return pos

    def delete_key(self, track: str, index: int) -> None:
        keys = list(self.keys(track))
        del keys[index]
        self.set_spec(replace(self.spec, **{track: tuple(keys)}))

    def move_key(self, track: str, index: int, delta: int) -> int:
        keys = list(self.keys(track))
        j = index + delta
        if not 0 <= j < len(keys):
            return index
        keys[index], keys[j] = keys[j], keys[index]
        self.set_spec(replace(self.spec, **{track: tuple(keys)}))
        return j

    def set_scale(self, amplitude: float | None = None, tempo: float | None = None) -> None:
        changes: dict[str, float] = {}
        if amplitude is not None:
            changes["amplitude_scale"] = float(amplitude)
        if tempo is not None:
            if tempo <= 0:
                raise ValueError("tempo must be > 0")
            changes["tempo_scale"] = float(tempo)
        self.set_spec(replace(self.spec, **changes))

    def set_meta(self, intent: str | None = None, description: str | None = None, valence: float | None = None,
                 arousal: float | None = None, draft: bool | None = None) -> None:
        changes: dict[str, Any] = {}
        for name, v in (("intent", intent), ("description", description), ("valence", valence),
                        ("arousal", arousal), ("draft", draft)):
            if v is not None:
                changes[name] = v
        self.set_spec(replace(self.spec, **changes))

    def set_sound(self, file: str | None, at_s: float = 0.0) -> None:
        self.set_spec(replace(self.spec, sound=None if not file else SoundCue(str(file), float(at_s))))

    def retime_uniform(self, rot_speed_dps: float, trans_speed_mms: float | None = None) -> None:
        """Neutral helper: constant speed (LINEAR) with the same angular speed on every moving head key."""
        from reachy_motion.library.compare import retime_uniform_speed

        self.set_spec(retime_uniform_speed(self.spec, rot_speed_dps, trans_speed_mms, set_linear=True))

    # ------------------------------------------------------------------ analysis
    def analyze(self, spec: MotionSpec | None = None) -> AnalysisBundle:
        spec = spec if spec is not None else self.spec
        d = spec.digest()
        if d in self._cache:
            return self._cache[d]
        bundle = AnalysisBundle(spec=spec, problems=spec.validate())
        if not bundle.problems:
            try:
                from reachy_motion.library.compare import neutral_conformance
                from reachy_motion.planner import plan, segment_table
                from reachy_motion.safety.checker import analyze_spec, check_spec

                bundle.timeline = plan(spec, self.policy.stream_rate_hz)
                bundle.analysis = analyze_spec(spec, self.policy.check_rate_hz)
                bundle.report = check_spec(spec, self.policy)
                bundle.segments = segment_table(spec)
                bundle.conformance = neutral_conformance(spec)
            except Exception as exc:  # noqa: BLE001 - shown in the GUI instead of crashing it
                bundle.error = f"{type(exc).__name__}: {exc}"
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[d] = bundle
        return bundle

    def counterpart(self) -> MotionSpec | None:
        name, variant = self._require()
        other = "neutral" if variant == "expressive" else "expressive"
        return self.working(name, other)

    def comparison(self) -> dict[str, Any] | None:
        """``compare.comparison_conditions`` of the expressive / neutral pair of the current motion."""
        from reachy_motion.library.compare import comparison_conditions

        other = self.counterpart()
        if other is None:
            return None
        expressive, neutral = (self.spec, other) if self.spec.variant == "expressive" else (other, self.spec)
        return comparison_conditions(expressive, neutral)

    # ------------------------------------------------------------------ export
    def export_bundle(self, out_dir: Path, figures: bool = True, playback: Any = None) -> list[Path]:
        """Spec (YAML + JSON), policy report, executed parameters, dense trajectory, comparison, figures, manifest.

        ``playback`` (optional ``ExecutionResult``) adds the sent commands and the tick timing of a Mock run.
        """
        from reachy_motion.library.io import save_spec_file
        from reachy_motion.sdk_env import environment_report

        spec = self.spec
        bundle = self.analyze(spec)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{spec.name}.{spec.variant}"
        written: list[Path] = []
        kinds: dict[str, str] = {}

        def add(p: Path, kind: str) -> None:
            written.append(p)
            kinds[p.name] = kind

        save_spec_file(spec, out / f"{stem}.yaml")
        add(out / f"{stem}.yaml", "design")
        save_spec_file(spec, out / f"{stem}.json")
        add(out / f"{stem}.json", "design")
        if bundle.report is not None:
            p = out / "policy_report.json"
            p.write_text(json.dumps(bundle.report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
            add(p, "computed")
        if bundle.segments:
            add(_write_rows(out / "executed_parameters.csv", bundle.segments), "computed")
        if bundle.analysis is not None:
            add(_write_rows(out / "trajectory_dense.csv", analysis_rows(bundle.analysis)), "computed")
        if bundle.timeline is not None:
            add(_write_rows(out / "stream_samples.csv", timeline_rows(bundle.timeline)), "computed")
        comp = self.comparison()
        if comp is not None:
            p = out / "comparison.json"
            p.write_text(json.dumps(comp, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
            add(p, "computed")
        if figures and bundle.analysis is not None:
            from reachy_motion.gui import plots

            for fname, draw in (("motion.png", lambda f: plots.draw_motion(f, bundle)),
                                ("joints.png", lambda f: plots.draw_joints(f, bundle, self.policy))):
                fig = plots.new_figure()
                draw(fig)
                fig.savefig(out / fname, dpi=130)
                add(out / fname, "computed")
            other = self.counterpart()
            if other is not None:
                ob = self.analyze(other)
                if ob.analysis is not None:
                    fig = plots.new_figure()
                    a, b = (bundle, ob) if spec.variant == "expressive" else (ob, bundle)
                    plots.draw_compare(fig, a, b)
                    fig.savefig(out / "compare.png", dpi=130)
                    add(out / "compare.png", "computed")
        if playback is not None:
            add(_write_rows(out / "playback_commands.csv", command_rows(playback)), "mock")
            ticks = playback.ticks
            p = out / "playback_ticks.csv"
            ticks.to_csv(p)
            add(p, "pc_measured")
            p = out / "playback_timing.json"
            period_ns = int(round(1e9 / self.policy.stream_rate_hz))
            p.write_text(json.dumps(ticks.summary(period_ns), indent=2, default=_json_default), encoding="utf-8")
            add(p, "pc_measured")
        manifest = {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool": "reachy-motion gui export",
            "motion": spec.key, "draft": spec.draft, "intent": spec.intent,
            "expected_valence_arousal": {"valence": spec.valence, "arousal": spec.arousal,
                                         "note": "design targets, not measured"},
            "principles": [{"principle": p.principle.value, "how": p.how} for p in spec.principles],
            "scale": {"amplitude": spec.amplitude_scale, "tempo": spec.tempo_scale},
            "spec_digest": spec.digest(),
            "timeline_digest": None if bundle.timeline is None else bundle.timeline.digest(),
            "passes_initial_policy": bundle.passes_policy,
            "policy": self.policy.to_dict(),
            "data_kind_by_file": kinds,
            "data_kind_note": "design = edited input; computed = SDK functions offline; mock = Mock backend model; "
                              "pc_measured = executor timing measured on this PC (not the robot)",
            "environment": environment_report(),
        }
        p = out / "manifest.json"
        p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        written.append(p)
        return written


# ---------------------------------------------------------------------- row helpers (CSV)
def analysis_rows(a: Any) -> list[dict[str, float]]:
    """Dense planned trajectory in human units + IK joints and speeds (computed)."""
    from reachy_motion.analysis.goto_replay import JOINT_NAMES

    rows = []
    t = np.asarray(a.t)
    pose = np.asarray(a.pose6)
    ant = np.asarray(a.antennas)
    body = np.asarray(a.body)
    q = np.asarray(a.joints)
    qd = np.asarray(a.joint_velocity)
    for i in range(t.shape[0]):
        row = {"t_s": float(t[i])}
        for j, (f, _, k) in enumerate(HEAD_FIELDS):
            row[f] = float(pose[i, j] / k)
        row["antenna_right_deg"] = float(ant[i, 0] / DEG)
        row["antenna_left_deg"] = float(ant[i, 1] / DEG)
        row["body_cmd_yaw_deg"] = float(body[i] / DEG)
        for j, name in enumerate(JOINT_NAMES):
            row[f"ik_{name}_deg"] = float(q[i, j] / DEG)
            row[f"ik_{name}_speed_dps"] = float(qd[i, j] / DEG)
        rows.append(row)
    return rows


def timeline_rows(tl: Any) -> list[dict[str, float]]:
    """The exact samples an executor sends (head pose converted to human units)."""
    from reachy_motion.library.pose import pose_from_matrix

    rows = []
    for k in range(len(tl)):
        p = pose_from_matrix(tl.head[k]).to_deg_mm()
        rows.append({"k": k, "t_s": float(tl.t[k]), **p, "antenna_right_deg": float(tl.antennas[k, 0] / DEG),
                     "antenna_left_deg": float(tl.antennas[k, 1] / DEG), "body_yaw_deg": float(tl.body_yaw[k] / DEG)})
    return rows


def command_rows(result: Any) -> list[dict[str, Any]]:
    from reachy_motion.library.pose import pose_from_matrix

    rows = []
    cmds = list(result.commands)
    t0 = cmds[0][1].t_ns if cmds and isinstance(cmds[0], tuple) else (cmds[0].t_ns if cmds else 0)
    for item in cmds:
        k, c = item if isinstance(item, tuple) else (None, item)
        p = pose_from_matrix(c.head).to_deg_mm()
        rows.append({"k": k, "t_s": (c.t_ns - t0) / 1e9, "kind": c.kind, **p,
                     "antenna_right_deg": c.antennas[0] / DEG, "antenna_left_deg": c.antennas[1] / DEG,
                     "body_yaw_deg": c.body_yaw / DEG})
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in r.items()})
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


__all__ = ["EditorModel", "AnalysisBundle", "HEAD_FIELDS", "TRACK_FIELDS", "FIELD_RANGES", "TRACKS",
           "analysis_rows", "timeline_rows", "command_rows", "key_value_human"]
