"""Human-editable motion files: ``MotionSpec`` <-> YAML / JSON, schema ``reachy-motion/v1``. SDK-free.

Motion files live in ``motions/<name>.<variant>.yaml`` (``library.registry.motions_dir``). They use human
units with the unit in every key name - degrees (``*_deg``), millimetres (``*_mm``), seconds (``*_s``) - and
are converted to the SI ``MotionSpec`` (m, rad, s; CLAUDE.md code rules) only here, at the file boundary.
The shipped motions are DRAFTS (``draft: true``): the student designs and tunes the final ones.

Example (block or flow style both work; comments are allowed anywhere)::

    schema: reachy-motion/v1
    name: aha_recommend
    variant: expressive
    draft: true                                  # true until the student signs the design off
    intent: what the motion should communicate
    description: free text
    expected: {valence: 0.7, arousal: 0.6}       # design targets on [-1, 1], NOT measured
    principles:
    - principle: anticipation                    # a Principle value
      how: small dip before the lift
    scale: {amplitude: 1.0, tempo: 1.0}          # amplitude about the start state; durations, holds / tempo
    start: rest                     # or {head: {...}, antennas_deg: {right: .., left: ..}, body_yaw_deg: ..}
    tracks:
      head:
      - label: dip
        duration_s: 0.25                         # travel time from the previous key (or the start)
        hold_s: 0.0                              # pause after arriving
        easing: minjerk                          # linear | minjerk | ease_in_out | cartoon
        pose: {x_mm: 0.0, y_mm: 0.0, z_mm: -4.0, roll_deg: 0.0, pitch_deg: 6.0, yaw_deg: 0.0}
      antennas:
      - {label: perk, duration_s: 0.2, hold_s: 0.3, easing: ease_in_out, right_deg: 0.0, left_deg: 0.0}
      - {label: settle, duration_s: 0.4, hold_s: 0.0, easing: minjerk, right_deg: rest, left_deg: rest}
      body:
      - {label: turn, duration_s: 1.0, hold_s: 0.0, easing: minjerk, yaw_deg: 10.0}
    sound: null                                  # or {file: aha.wav, at_s: 0.3}

Semantics (same as ``library.spec``):

* Head pose values are ABSOLUTE, relative to the neutral head pose (SDK INIT_HEAD_POSE = identity), not to
  the previous key; an omitted pose component is 0. ``pose: {}`` is the neutral pose.
* Antenna angles are absolute ``[right, left]`` angles in degrees, or ``rest`` = that antenna's SDK rest angle
  ``INIT_ANTENNAS_JOINT_POSITIONS = [-0.1745, 0.1745]`` rad exactly (reachy_mini/reachy_mini.py:67, copied as
  ``pose.INIT_ANTENNAS``); an angle equal to that constant is written back as ``rest``. Writing -10 / +10 deg
  instead gives -/+0.174533 rad, 0.0019 deg away from the SDK constant: below one XL330 position tick
  (360/4096 = 0.088 deg), and ``library.compare`` calls such start / end states "the same" (0.01 deg tolerance
  of ``same_start`` / ``same_end``), but its Neutral rule "no antenna motion" is exact (1e-9 rad), so a key at
  -10 / +10 deg in a neutral file is reported as antenna motion: leave a neutral antenna track empty (or use
  ``rest``). Body yaw is absolute.
* ``start`` omitted, ``null`` or ``rest`` = ``pose.REST``: neutral head, antennas at rest, body 0. A start
  mapping fills omitted parts with those REST values.
* Key defaults: ``hold_s`` 0, ``easing`` minjerk, ``label`` "". Required: ``duration_s`` and the value
  (``pose`` / ``right_deg`` + ``left_deg`` / ``yaw_deg``). Tracks may be omitted or empty.

Strictness (files are edited by hand): unknown keys -> ``ValueError`` naming the key path, the allowed keys
and a close match; missing required keys, wrong types (``true`` is not a number), non-finite or huge numbers,
unknown easing / principle names and DUPLICATE keys (PyYAML silently keeps the last one) are all refused. So
are the YAML 1.1 number forms that silently change a hand-typed value - a leading zero (``010`` is octal 8),
hex / binary, base 60 (``1:30`` is 90) and ``_`` separators: numbers are plain decimals. Every error is a
``ValueError`` (``load_spec_file`` prefixes the file name; a file that is not UTF-8 says so).
Physical / semantic checks (durations >= MIN_KEY_DURATION_S, valence range, known motion names, ...) stay in
``MotionSpec.validate()`` and the safety checker, so an unfinished design can still be loaded and shown.

Numbers are written as the shortest decimal that converts back to exactly the same SI value (searched among
the doubles next to the converted value), so spec -> file -> spec is exact for values that came from a file
(within one ulp for other values) and a saved file reads back to the same text when saved again.

YAML is read with a ``yaml.SafeLoader`` subclass (adds the duplicate-key and number checks) and written with
a ``yaml.SafeDumper`` subclass (layout only: number-only mappings on one line, multi-line text as literal
blocks, no anchors); both use standard YAML tags only, so the files also load with ``yaml.safe_load``.
"""

from __future__ import annotations

import difflib
import json
import math
import numbers
import os
import re
from collections.abc import Callable, Hashable, Mapping
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError

from reachy_motion.library.pose import INIT_ANTENNAS, REST, Pose6, RobotState
from reachy_motion.library.spec import (
    TRACKS,
    AntennaKey,
    BodyKey,
    Easing,
    HeadKey,
    MotionSpec,
    Principle,
    PrincipleUse,
    SoundCue,
)

__all__ = [
    "SCHEMA", "START_REST", "ANTENNA_REST", "FILE_EXTENSIONS", "YAML_EXTENSIONS", "JSON_EXTENSIONS", "HEAD_FIELDS",
    "spec_from_dict", "spec_to_dict", "key_from_dict", "key_to_dict", "pose_from_dict", "pose_to_dict",
    "dumps_yaml", "dumps_json", "loads_spec", "file_header", "load_spec_file", "save_spec_file",
    "motion_file_name", "parse_motion_file_name",
]

SCHEMA = "reachy-motion/v1"
START_REST = "rest"  # ``start: rest`` = pose.REST (SDK rest antennas exactly)
ANTENNA_REST = "rest"  # ``right_deg: rest`` / ``left_deg: rest`` = that antenna's SDK rest angle (INIT_ANTENNAS)
YAML_EXTENSIONS: tuple[str, ...] = (".yaml", ".yml")
JSON_EXTENSIONS: tuple[str, ...] = (".json",)
FILE_EXTENSIONS: tuple[str, ...] = YAML_EXTENSIONS + JSON_EXTENSIONS

#: Head pose components in file order (= ``Pose6.from_deg_mm`` keyword names).
HEAD_FIELDS: tuple[str, ...] = ("x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg")

_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")  # name / variant end up in the file name; always used with fullmatch

_TOP_KEYS = ("schema", "name", "variant", "draft", "intent", "description", "expected", "principles", "scale",
             "start", "tracks", "sound")
_TOP_REQUIRED = ("schema", "name", "variant", "intent", "expected", "principles", "tracks")
_EXPECTED_KEYS = ("valence", "arousal")
_PRINCIPLE_KEYS = ("principle", "how")
_SCALE_KEYS = ("amplitude", "tempo")
_START_KEYS = ("head", "antennas_deg", "body_yaw_deg")
_SIDE_KEYS = ("right", "left")
_KEY_COMMON = ("label", "duration_s", "hold_s", "easing")
_KEY_FIELDS: dict[str, tuple[str, ...]] = {
    "head": _KEY_COMMON + ("pose",),
    "antennas": _KEY_COMMON + ("right_deg", "left_deg"),
    "body": _KEY_COMMON + ("yaw_deg",),
}
_KEY_REQUIRED: dict[str, tuple[str, ...]] = {
    "head": ("duration_s", "pose"),
    "antennas": ("duration_s", "right_deg", "left_deg"),
    "body": ("duration_s", "yaw_deg"),
}
_SOUND_KEYS = ("file", "at_s")

#: A decimal number as text (what a number in quotes, or one YAML 1.1 left as text, looks like).
_DECIMAL_TEXT = re.compile(r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?")
#: The only integer form the YAML reader accepts (YAML 1.1 also reads 010 as octal, 0x0A, 0b11, 1:30, 1_000).
_YAML_PLAIN_INT = re.compile(r"[-+]?(?:0|[1-9][0-9]*)")
#: Doubles tried on each side of the converted value when writing a number (see :func:`_human`).
_NEIGHBOUR_ULPS = 8


# ====================================================================== unit conversion (file boundary)
def _mm_to_m(v: float) -> float:
    return v / 1000.0  # same expression as Pose6.from_deg_mm


def _m_to_mm(v: float) -> float:
    return v * 1000.0


def _human(value: float, path: str, to_human: Callable[[float], float], from_human: Callable[[float], float]) -> float:
    """``to_human(value)`` written as the shortest decimal that converts back to ``value`` exactly.

    Candidates are the doubles within ``_NEIGHBOUR_ULPS`` of ``to_human(value)``: a decimal x read from a file
    lies within a few ulps of ``to_human(from_human(x))`` (two roundings), so x and every other exact candidate
    are among them. The winner converts back closest to ``value`` (exactly whenever any candidate does), then
    has the shortest ``repr``, then is the smallest - a fixed rule, so a file saved again is written the same way.
    """
    v = _finite(value, path)
    h = to_human(v)
    if not math.isfinite(h):
        raise ValueError(f"{path}: cannot store {v!r}: too large")
    candidates = [h]
    lo = hi = h
    for _ in range(_NEIGHBOUR_ULPS):
        lo, hi = math.nextafter(lo, -math.inf), math.nextafter(hi, math.inf)
        candidates += (lo, hi)
    best = min(candidates, key=lambda x: (abs(from_human(x) - v), len(repr(x)), x))
    return best + 0.0  # + 0.0 turns -0.0 into 0.0


def _deg(value: float, path: str) -> float:
    return _human(value, path, math.degrees, math.radians)


def _mm(value: float, path: str) -> float:
    return _human(value, path, _m_to_mm, _mm_to_m)


def _finite(value: Any, path: str) -> float:
    """A spec value as a plain finite ``float`` (any real number type, e.g. numpy scalars; not bool)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{path}: expected a number, got {_describe(value)}")
    try:
        v = float(value)
    except OverflowError:
        raise ValueError(f"{path}: cannot store {_describe(value)}: too large") from None
    if not math.isfinite(v):
        raise ValueError(f"{path}: cannot store the non-finite value {value!r}")
    return v


# ====================================================================== parsing helpers (strict, with key paths)
def _where(path: str) -> str:
    return path if path else "top level"


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _describe(value: Any) -> str:
    if value is None:
        return "nothing (null)"
    if isinstance(value, bool):
        return f"the boolean {str(value).lower()}"
    if isinstance(value, (int, float)):
        try:
            text = repr(value)
        except ValueError:  # an int with more digits than Python converts (sys.get_int_max_str_digits)
            return "a number with thousands of digits"
        return f"the number {text}" if len(text) <= 40 else f"a {len(text)}-character number"
    if isinstance(value, str):
        return f"the text {value!r}"
    if isinstance(value, Mapping):
        return "a mapping"
    if isinstance(value, list):
        return "a list"
    return f"a {type(value).__name__}"


def _did_you_mean(word: Any, allowed: tuple[str, ...] | list[str]) -> str:
    close = difflib.get_close_matches(str(word), list(allowed), n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _mapping(value: Any, path: str, allowed: tuple[str, ...], required: tuple[str, ...] = ()) -> Mapping[str, Any]:
    """``value`` as a mapping with only ``allowed`` keys and every ``required`` key."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{_where(path)}: expected a mapping with keys {', '.join(allowed)}; got {_describe(value)}")
    for k in value:
        if not isinstance(k, str):
            raise ValueError(f"{_where(path)}: key {k!r} is not text; allowed keys: {', '.join(allowed)}")
        if k not in allowed:
            raise ValueError(f"{_where(path)}: unknown key {k!r}{_did_you_mean(k, allowed)}; "
                             f"allowed keys: {', '.join(allowed)}")
    missing = [k for k in required if k not in value]
    if missing:
        raise ValueError(f"{_where(path)}: missing key{'s' if len(missing) > 1 else ''} "
                         f"{', '.join(repr(k) for k in missing)} (required: {', '.join(required)})")
    return value


def _text_number_hint(text: str) -> str:
    """Why a number-like text is not a number, e.g. YAML 1.1 reads ``1e-3``, ``08`` and ``-.5`` as text."""
    t = text.strip()
    body = t.lstrip("+-")
    if body.lower() in ("inf", "infinity", "nan", ".inf", ".nan"):
        return " (only finite numbers are allowed)"
    if not _DECIMAL_TEXT.fullmatch(t):
        return ""
    sign = t[: len(t) - len(body)]
    if re.fullmatch(r"0[0-9]+", body):
        return f" (write {sign}{body.lstrip('0')}, not {t}: no leading zeros)"
    if body.startswith("."):
        return f" (write {sign}0{body}, not {t})"
    if "e" in body.lower():
        return " (YAML reads exponents without a dot and a sign as text: write 1.0e-3, not 1e-3)"
    return " (write it without quotes)"


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        hint = _text_number_hint(value) if isinstance(value, str) else ""
        raise ValueError(f"{path}: expected a number, got {_describe(value)}{hint}")
    try:
        v = float(value)
    except OverflowError:  # an int beyond the float range
        raise ValueError(f"{path}: {_describe(value)} is too large") from None
    if not math.isfinite(v):
        raise ValueError(f"{path}: {value!r} is not a finite number")
    return v


def _text(value: Any, path: str) -> str:
    """Text; ``null`` (an empty YAML value) reads as ""."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected text, got {_describe(value)} (put it in quotes)")
    return value


def _ident(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise ValueError(f"{path}: expected a name made of letters, digits, '_' or '-' (it is part of the file "
                         f"name), got {_describe(value)}")
    return value


def _easing(value: Any, path: str) -> Easing:
    allowed = [e.value for e in Easing]
    if isinstance(value, str) and value in allowed:
        return Easing(value)
    raise ValueError(f"{path}: unknown easing {value!r}{_did_you_mean(value, allowed)}; allowed: {', '.join(allowed)}")


def _principle(value: Any, path: str) -> Principle:
    allowed = [p.value for p in Principle]
    if isinstance(value, str) and value in allowed:
        return Principle(value)
    raise ValueError(f"{path}: unknown principle {value!r}{_did_you_mean(value, allowed)}; "
                     f"allowed: {', '.join(allowed)}")


def _list(value: Any, path: str, what: str) -> list[Any]:
    """A YAML list; ``null`` (e.g. ``body:`` with nothing after it) reads as an empty list."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a list of {what}, got {_describe(value)}")
    return value


def _antenna_from(value: Any, path: str, side: int) -> float:
    """One antenna angle of a file (degrees, or ``rest`` = the SDK rest angle of that side) in rad."""
    if isinstance(value, str) and value == ANTENNA_REST:
        return INIT_ANTENNAS[side]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        hint = ((_text_number_hint(value) or _did_you_mean(value.lower(), (ANTENNA_REST,)))
                if isinstance(value, str) else "")
        raise ValueError(f"{path}: expected a number (deg) or {ANTENNA_REST!r} (the SDK rest angle), got "
                         f"{_describe(value)}{hint}")
    return math.radians(_number(value, path))


def _antenna_to(value: float, path: str, side: int) -> float | str:
    """One antenna angle for a file: ``rest`` for exactly the SDK rest angle of that side, else degrees."""
    v = _finite(value, path)
    return ANTENNA_REST if v == INIT_ANTENNAS[side] else _deg(v, path)


# ====================================================================== dict -> spec
def pose_from_dict(d: Any, path: str = "pose") -> Pose6:
    """``{x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg}`` (each optional, default 0) -> SI ``Pose6``."""
    m = _mapping(d, path, HEAD_FIELDS)
    return Pose6.from_deg_mm(**{k: _number(m[k], _join(path, k)) for k in HEAD_FIELDS if k in m})


def _key_common(m: Mapping[str, Any], path: str) -> dict[str, Any]:
    return {
        "duration": _number(m["duration_s"], _join(path, "duration_s")),
        "hold": _number(m["hold_s"], _join(path, "hold_s")) if "hold_s" in m else 0.0,
        "easing": _easing(m["easing"], _join(path, "easing")) if "easing" in m else Easing.MIN_JERK,
        "label": _text(m.get("label"), _join(path, "label")),
    }


def key_from_dict(track: str, d: Any, path: str = "") -> HeadKey | AntennaKey | BodyKey:
    """One key of ``track`` ("head" | "antennas" | "body") from its human-unit mapping."""
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}; allowed: {', '.join(TRACKS)}")
    path = path or f"tracks.{track}[]"
    m = _mapping(d, path, _KEY_FIELDS[track], _KEY_REQUIRED[track])
    common = _key_common(m, path)
    if track == "head":
        return HeadKey(pose=pose_from_dict(m["pose"], _join(path, "pose")), **common)
    if track == "antennas":
        right = _antenna_from(m["right_deg"], _join(path, "right_deg"), 0)
        left = _antenna_from(m["left_deg"], _join(path, "left_deg"), 1)
        return AntennaKey(antennas=(right, left), **common)
    return BodyKey(yaw=math.radians(_number(m["yaw_deg"], _join(path, "yaw_deg"))), **common)


def _start_from(value: Any, path: str = "start") -> RobotState:
    if value is None or (isinstance(value, str) and value == START_REST):
        return REST
    if isinstance(value, str):
        raise ValueError(f"{path}: expected {START_REST!r} or a mapping with keys {', '.join(_START_KEYS)}; "
                         f"got {_describe(value)}")
    m = _mapping(value, path, _START_KEYS)
    head = pose_from_dict(m["head"], _join(path, "head")) if "head" in m else REST.head
    antennas = REST.antennas
    if "antennas_deg" in m:
        a = _mapping(m["antennas_deg"], _join(path, "antennas_deg"), _SIDE_KEYS)
        antennas = tuple(
            _antenna_from(a[side], _join(path, f"antennas_deg.{side}"), i) if side in a else INIT_ANTENNAS[i]
            for i, side in enumerate(_SIDE_KEYS))
    body = (math.radians(_number(m["body_yaw_deg"], _join(path, "body_yaw_deg"))) if "body_yaw_deg" in m
            else REST.body_yaw)
    return RobotState(head=head, antennas=antennas, body_yaw=body)


def spec_from_dict(d: Any) -> MotionSpec:
    """Parse one motion (schema ``reachy-motion/v1``, human units) into an SI ``MotionSpec``.

    Strict about structure and types (``ValueError`` with the key path); does NOT call
    ``MotionSpec.validate()`` - the registry and the GUI do, so an unfinished design still loads.
    """
    if d is None:
        raise ValueError("the motion file is empty")
    m = _mapping(d, "", _TOP_KEYS, _TOP_REQUIRED)
    if m["schema"] != SCHEMA:
        raise ValueError(f"schema: expected {SCHEMA!r}, got {_describe(m['schema'])}")
    draft = m.get("draft", True)
    if not isinstance(draft, bool):
        raise ValueError(f"draft: expected true or false, got {_describe(draft)}")

    exp = _mapping(m["expected"], "expected", _EXPECTED_KEYS, _EXPECTED_KEYS)
    principles = []
    for i, p in enumerate(_list(m["principles"], "principles", "{principle, how} mappings")):
        path = f"principles[{i}]"
        pm = _mapping(p, path, _PRINCIPLE_KEYS, _PRINCIPLE_KEYS)
        principles.append(PrincipleUse(principle=_principle(pm["principle"], f"{path}.principle"),
                                       how=_text(pm["how"], f"{path}.how")))

    amplitude = tempo = 1.0
    if m.get("scale") is not None:
        sc = _mapping(m["scale"], "scale", _SCALE_KEYS)
        amplitude = _number(sc["amplitude"], "scale.amplitude") if "amplitude" in sc else 1.0
        tempo = _number(sc["tempo"], "scale.tempo") if "tempo" in sc else 1.0

    tracks = _mapping(m["tracks"], "tracks", TRACKS)
    keys = {t: tuple(key_from_dict(t, k, f"tracks.{t}[{i}]")
                     for i, k in enumerate(_list(tracks.get(t), f"tracks.{t}", f"{t} keys")))
            for t in TRACKS}

    sound = None
    if m.get("sound") is not None:
        sm = _mapping(m["sound"], "sound", _SOUND_KEYS, _SOUND_KEYS)
        sound = SoundCue(file=_text(sm["file"], "sound.file"), at_s=_number(sm["at_s"], "sound.at_s"))

    return MotionSpec(
        name=_ident(m["name"], "name"), variant=_ident(m["variant"], "variant"),
        intent=_text(m["intent"], "intent"),
        valence=_number(exp["valence"], "expected.valence"), arousal=_number(exp["arousal"], "expected.arousal"),
        principles=tuple(principles), head=keys["head"], antennas=keys["antennas"], body=keys["body"],
        start=_start_from(m.get("start")), description=_text(m.get("description"), "description"),
        amplitude_scale=amplitude, tempo_scale=tempo, sound=sound, draft=draft,
    )


# ====================================================================== spec -> dict
def pose_to_dict(pose: Pose6, path: str = "pose") -> dict[str, float]:
    """All six components, human units (mm, deg), shortest exact decimals."""
    return {"x_mm": _mm(pose.x, _join(path, "x_mm")), "y_mm": _mm(pose.y, _join(path, "y_mm")),
            "z_mm": _mm(pose.z, _join(path, "z_mm")), "roll_deg": _deg(pose.roll, _join(path, "roll_deg")),
            "pitch_deg": _deg(pose.pitch, _join(path, "pitch_deg")), "yaw_deg": _deg(pose.yaw, _join(path, "yaw_deg"))}


def key_to_dict(track: str, key: HeadKey | AntennaKey | BodyKey, path: str = "") -> dict[str, Any]:
    """Human-unit mapping of one key with every field written explicitly (label, timing, easing, value).

    An antenna angle equal to the SDK rest angle is written as ``rest`` (exact; the degrees would read -9.998...).
    """
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}; allowed: {', '.join(TRACKS)}")
    path = path or f"tracks.{track}[]"
    out: dict[str, Any] = {
        "label": str(key.label), "duration_s": _finite(key.duration, _join(path, "duration_s")),
        "hold_s": _finite(key.hold, _join(path, "hold_s")), "easing": Easing(key.easing).value,
    }
    if track == "head":
        out["pose"] = pose_to_dict(key.pose, _join(path, "pose"))
    elif track == "antennas":
        out["right_deg"] = _antenna_to(key.antennas[0], _join(path, "right_deg"), 0)
        out["left_deg"] = _antenna_to(key.antennas[1], _join(path, "left_deg"), 1)
    else:
        out["yaw_deg"] = _deg(key.yaw, _join(path, "yaw_deg"))
    return out


def spec_to_dict(spec: MotionSpec) -> dict[str, Any]:
    """Human-unit, JSON-able mapping of ``spec`` in file order (inverse of :func:`spec_from_dict`).

    Every head component and both antenna angles are written explicitly, empty tracks as ``[]``, the
    REST start as ``start: rest``. Raises ``ValueError`` for values a file could not hold (non-finite or
    huge numbers, names that cannot be part of a file name).
    """
    _ident(spec.name, "name")
    _ident(spec.variant, "variant")
    s0 = spec.start
    start: str | dict[str, Any]
    if s0 == REST:
        start = START_REST
    else:
        start = {"head": pose_to_dict(s0.head, "start.head"),
                 "antennas_deg": {"right": _antenna_to(s0.antennas[0], "start.antennas_deg.right", 0),
                                  "left": _antenna_to(s0.antennas[1], "start.antennas_deg.left", 1)},
                 "body_yaw_deg": _deg(s0.body_yaw, "start.body_yaw_deg")}
    return {
        "schema": SCHEMA,
        "name": spec.name,
        "variant": spec.variant,
        "draft": bool(spec.draft),
        "intent": str(spec.intent),
        "description": str(spec.description),
        "expected": {"valence": _finite(spec.valence, "expected.valence"),
                     "arousal": _finite(spec.arousal, "expected.arousal")},
        "principles": [{"principle": Principle(p.principle).value, "how": str(p.how)} for p in spec.principles],
        "scale": {"amplitude": _finite(spec.amplitude_scale, "scale.amplitude"),
                  "tempo": _finite(spec.tempo_scale, "scale.tempo")},
        "start": start,
        "tracks": {t: [key_to_dict(t, k, f"tracks.{t}[{i}]") for i, k in enumerate(spec.track(t))] for t in TRACKS},
        "sound": None if spec.sound is None else {"file": str(spec.sound.file),
                                                  "at_s": _finite(spec.sound.at_s, "sound.at_s")},
    }


# ====================================================================== YAML / JSON text
def _yaml_number_hint(text: str) -> str:
    """What to write instead of a YAML 1.1 number form that is not a plain decimal."""
    body = text.strip().lstrip("+-")
    if ":" in body:
        return "YAML 1.1 reads it as a base-60 number (1:30 = 90); write a plain decimal number"
    if body[:2].lower() in ("0x", "0b"):
        return (f"YAML 1.1 reads it as a {'hexadecimal' if body[1] in 'xX' else 'binary'} number; write a plain "
                f"decimal number")
    if "_" in body:
        return "write numbers without '_' separators"
    if re.fullmatch(r"0[0-9]+", body):
        sign = text.strip()[: len(text.strip()) - len(body)]
        octal = (f" (YAML 1.1 reads {text.strip()} as octal {sign}{int(body, 8)})"
                 if re.fullmatch(r"0[0-7]+", body) else "")
        return f"write {sign}{body.lstrip('0') or '0'}, without leading zeros{octal}"
    return "write a plain decimal number"


class _StrictLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that refuses duplicate mapping keys (PyYAML would silently keep the last one) and the
    YAML 1.1 number forms that silently change a hand-typed value: ``010`` (octal 8), ``0x0A``, ``0b11``,
    ``1:30`` (base 60 = 90) and ``1_000``. Plain decimals (``10``, ``-2.5``, ``1.0e-3``) are unchanged."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        if isinstance(node, yaml.MappingNode):
            seen: set[Any] = set()
            for key_node, _value_node in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    continue
                key = self.construct_object(key_node, deep=deep)
                if isinstance(key, Hashable):
                    if key in seen:
                        raise ConstructorError(
                            "while constructing a mapping", node.start_mark,
                            f"found duplicate key {key!r} (each key may appear only once)", key_node.start_mark)
                    seen.add(key)
        return super().construct_mapping(node, deep=deep)

    def construct_yaml_int(self, node: yaml.ScalarNode) -> int:
        text = self.construct_scalar(node)
        if not _YAML_PLAIN_INT.fullmatch(text):
            raise ConstructorError(None, None, f"found {text!r}: {_yaml_number_hint(text)}", node.start_mark)
        try:
            return int(text)
        except ValueError as exc:  # more digits than Python converts (sys.get_int_max_str_digits)
            raise ConstructorError(None, None, f"found a number with {len(text)} digits: too large",
                                   node.start_mark) from exc

    def construct_yaml_float(self, node: yaml.ScalarNode) -> float:
        text = self.construct_scalar(node)
        if "_" in text or ":" in text:
            raise ConstructorError(None, None, f"found {text!r}: {_yaml_number_hint(text)}", node.start_mark)
        try:
            return super().construct_yaml_float(node)
        except (ValueError, IndexError) as exc:  # only reachable through an explicit !!float tag
            raise ConstructorError(None, None, f"found {text!r}: write a plain decimal number",
                                   node.start_mark) from exc


# SafeConstructor registered its own functions for these tags; register the overrides for the subclass.
_StrictLoader.add_constructor("tag:yaml.org,2002:int", _StrictLoader.construct_yaml_int)
_StrictLoader.add_constructor("tag:yaml.org,2002:float", _StrictLoader.construct_yaml_float)


class _MotionDumper(yaml.SafeDumper):
    """``yaml.SafeDumper`` with the motion-file layout; standard tags only (loads with ``yaml.safe_load``)."""

    def ignore_aliases(self, data: Any) -> bool:
        return True  # never emit &anchors / *aliases in a file meant for hand editing


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _represent_dict(dumper: yaml.SafeDumper, data: dict[Any, Any]) -> yaml.MappingNode:
    # Value-only mappings (poses, expected, scale, antennas_deg) on one line; everything else in block style.
    flow = bool(data) and all(_is_number(v) or (isinstance(v, str) and v == ANTENNA_REST) for v in data.values())
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=flow)


#: YAML 1.1 line breaks besides \n and \r. PyYAML writes them raw in plain, single-quoted and block scalars, and
#: the reader then folds them into a space; double quotes escape them (\N, \L, \P).
_RAW_YAML_BREAKS = ("\x85", " ", " ")


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    if any(c in data for c in _RAW_YAML_BREAKS):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    if "\n" in data:  # multi-line text as a literal block (the emitter falls back to quotes if it cannot)
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_MotionDumper.add_representer(dict, _represent_dict)
_MotionDumper.add_representer(str, _represent_str)


def file_header(spec: MotionSpec) -> str:
    """Short ``#`` comment block written at the top of a YAML motion file (draft notice, units, hints)."""
    lines = [
        f"Reachy Mini HW4 motion '{spec.name}' ({spec.variant}) - schema {SCHEMA}. Edit the values and reload.",
        ("DRAFT: initial values for the student to design and tune; set draft: false once signed off."
         if spec.draft else "Signed-off design (draft: false)."),
        "Units: angles in degrees (*_deg), head translation in millimetres (*_mm), times in seconds (*_s).",
        "Write plain decimal numbers (10, -2.5, 1.0e-3); 010, 0x0A, 1:30 and 1_000 are refused.",
        "Each key moves its track from the previous value (or the start) in duration_s with its easing",
        "(linear | minjerk | ease_in_out | cartoon), then pauses hold_s. Tracks head/antennas/body run in parallel.",
        "Head pose values are absolute (relative to the neutral head pose, omitted component = 0), not deltas.",
        "Antennas: absolute right_deg/left_deg; 'rest' = the SDK rest angle exactly (-0.1745/+0.1745 rad, i.e.",
        "-9.998/+9.998 deg; -10/+10 deg is 0.0019 deg off). start: rest = neutral head, antennas at rest, body 0.",
        "scale.amplitude scales every key about the start; scale.tempo divides every duration and hold.",
        "expected valence/arousal are design targets, not measured results.",
    ]
    if spec.variant == "neutral":
        lines.append("Neutral variant: same start/end as expressive, linear easing, no sound and no antenna motion "
                     "(leave tracks.antennas empty).")
    return "".join(f"# {line}\n" for line in lines)


def dumps_yaml(spec: MotionSpec, header: bool = True) -> str:
    """YAML text of ``spec`` (optionally with :func:`file_header`), as written by :func:`save_spec_file`."""
    body = yaml.dump(spec_to_dict(spec), Dumper=_MotionDumper, sort_keys=False, allow_unicode=True,
                     default_flow_style=False, width=120)
    return (file_header(spec) if header else "") + body


def dumps_json(spec: MotionSpec) -> str:
    """JSON text of ``spec`` (same mapping as the YAML; JSON has no comments, so no header)."""
    return json.dumps(spec_to_dict(spec), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r} (each key may appear only once)")
        out[k] = v
    return out


def _parse_text(text: str, fmt: str) -> Any:
    if fmt == "json":
        try:
            return json.loads(text, object_pairs_hook=_json_pairs)
        except (ValueError, RecursionError) as exc:  # JSONDecodeError, duplicate keys, ints beyond the digit limit
            raise ValueError(f"invalid JSON: {exc}") from exc
    if fmt != "yaml":
        raise ValueError(f"unknown format {fmt!r}; allowed: yaml, json")
    try:
        return yaml.load(text, Loader=_StrictLoader)
    except (yaml.YAMLError, ValueError, TypeError, AttributeError, IndexError, RecursionError) as exc:
        # YAMLError for ordinary mistakes; the others only from malformed explicit tags (e.g. "!!timestamp x")
        # or absurd nesting, which PyYAML's safe constructors do not turn into YAMLError.
        raise ValueError(f"invalid YAML: {exc}") from exc


def loads_spec(text: str, fmt: str = "yaml") -> MotionSpec:
    """Parse YAML (``fmt="yaml"``) or JSON (``fmt="json"``) text into a ``MotionSpec`` (strict)."""
    if text.startswith("﻿"):  # byte-order mark written by some Windows editors
        text = text[1:]
    return spec_from_dict(_parse_text(text, fmt))


# ====================================================================== files
def motion_file_name(name: str, variant: str, ext: str = ".yaml") -> str:
    """``<name>.<variant><ext>``, the only file name :func:`load_spec_file` accepts for that motion."""
    if ext.lower() not in FILE_EXTENSIONS:
        raise ValueError(f"extension {ext!r} not in {FILE_EXTENSIONS}")
    return f"{_ident(name, 'name')}.{_ident(variant, 'variant')}{ext}"


def parse_motion_file_name(path: str | os.PathLike[str]) -> tuple[str, str, str]:
    """``(name, variant, ext)`` from ``<name>.<variant>.<yaml|yml|json>``; ``ValueError`` for anything else."""
    fname = Path(path).name
    ext = Path(fname).suffix.lower()
    if ext not in FILE_EXTENSIONS:
        raise ValueError(f"{fname}: a motion file must end in {', '.join(FILE_EXTENSIONS)}")
    parts = fname[: -len(ext)].split(".")
    if len(parts) != 2 or not all(_NAME_RE.fullmatch(p) for p in parts):
        raise ValueError(f"{fname}: a motion file must be named <name>.<variant>{ext}, "
                         f"e.g. aha_recommend.expressive{ext}")
    return parts[0], parts[1], ext


def load_spec_file(path: str | os.PathLike[str]) -> MotionSpec:
    """Load ``<name>.<variant>.yaml|.yml|.json``; the name and variant inside must match the file name.

    Every problem with the content is a ``ValueError`` that starts with the file name (a file that is not
    UTF-8 included); a missing or unreadable file raises the usual ``OSError``.
    """
    p = Path(path)
    name, variant, ext = parse_motion_file_name(p)
    try:
        text = p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{p.name}: not UTF-8 text (save the file as UTF-8): {exc}") from exc
    try:
        spec = loads_spec(text, "json" if ext in JSON_EXTENSIONS else "yaml")
    except ValueError as exc:
        raise ValueError(f"{p.name}: {exc}") from exc
    if (spec.name, spec.variant) != (name, variant):
        raise ValueError(f"{p.name}: the file contains name {spec.name!r}, variant {spec.variant!r}; "
                         f"rename it to {motion_file_name(spec.name, spec.variant, ext)} or fix the content")
    return spec


def save_spec_file(spec: MotionSpec, path: str | os.PathLike[str]) -> Path:
    """Write ``spec`` as YAML (with :func:`file_header`) or JSON, chosen by the extension; returns the path.

    ``path`` may be a directory (the file is then ``<name>.<variant>.yaml`` inside it); a file path must be
    named ``<name>.<variant>.<ext>`` for this spec so that it loads back. The write is atomic (temporary
    file + ``os.replace``), so an editor or the registry never sees a half-written file.
    """
    p = Path(path)
    if p.is_dir():
        p = p / motion_file_name(spec.name, spec.variant)
    name, variant, ext = parse_motion_file_name(p)
    if (name, variant) != (spec.name, spec.variant):
        raise ValueError(f"{p.name}: file name does not match the spec {spec.name}.{spec.variant}; "
                         f"use {motion_file_name(spec.name, spec.variant, ext)}")
    text = dumps_json(spec) if ext in JSON_EXTENSIONS else dumps_yaml(spec)
    tmp = p.with_name(p.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()
    return p
