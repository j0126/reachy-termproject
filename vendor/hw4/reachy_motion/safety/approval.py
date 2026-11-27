"""Robot approval gate: nothing reaches the real motors without a SEPARATE, explicit human approval.

CLAUDE.md rule 3 (user instruction): every real-robot experiment needs the user's explicit approval. This
module turns the rule into code for Phase 4 (``backends.sdk.SdkBackend(target="robot")`` refuses to exist
unless all four hold):

1. ``armed=True`` - the caller deliberately asks for the robot;
2. an approval FILE written and saved by the HUMAN supervisor and read with ``load_approval``. It names the
   stage, the robot host, the exact motions, the exact command timelines (``Timeline.digest()``), the
   safety policy the plans were checked with (``policy_name`` + ``policy_sha256`` = ``policy_digest``),
   a validity window and the speed / margin caps the human accepts. ``load_approval`` records where the
   approval came from (``RobotApproval.source``: resolved path + SHA-256 of the file bytes). An approval
   built in code - or copied with ``dataclasses.replace`` - has no source, and the gate refuses it, as it
   refuses a file that changed after it was loaded (``source_problems``);
3. ``validate_approval`` - the approval covers this request, and every plan was checked under exactly the
   approved policy, passes it with no violation listed (``data_kind == "computed"``) and meets the caps
   (``SafetyReport.metrics`` keys ``peak_stewart_speed_ratio`` / ``min_stewart_margin_deg``);
4. ``confirm_interactively`` - a person at a terminal types the approval id back (refused without a TTY).
   The prompt has no time limit, so the gate checks everything again AFTER the answer (``gate_problems``)
   and the robot backend checks again before connecting.

In Phase 2 even a fully approved robot backend refuses to connect: nothing may reach the robot yet.

The assistant and other agents must NEVER create, edit or save approval files and never answer the
confirmation prompt. ``approval_template`` only RETURNS a dict for the human to fill in and save; this
module never writes files. An approval written by an agent is not an approval (CLAUDE.md rule 3: approval
comes from the user, per experiment). Code cannot tell who wrote a file: ``approved_by`` is screened for
template placeholders and assistant / agent names, but it is a signature, not a proof of identity - the
id typed back by a person at a terminal is the real check.

Stages (docs/DEVELOPMENT_PLAN.md Phase 4, docs/VERIFICATION.md):
  R0  REST GET only (daemon status, kinematics info, state, motors status). Even constructing
      ``ReachyMini`` sends commands to the daemon (SetAutomaticBodyYawCmd, reachy_mini/reachy_mini.py:161;
      media release for media_backend="no_media", reachy_mini.py:350-357), so R0 never allows a client,
      let alone motion: an R0 approval never validates here. R0 GETs need no approval file at all.
  R1  small amplitude, slow motions        R2  the designed motions        R3  sweeps

Times: ``approved_on`` is an ISO date; the approval is valid from local midnight of that day. ``expires``
is an ISO datetime WITH a time, either timezone-aware ("2026-11-12T18:00:00+09:00") or naive, which is
read as local time of this PC. An approval may span at most ``MAX_VALIDITY_DAYS``.

File format: one JSON object, every key of ``APPROVAL_KEYS`` exactly once (duplicate keys are refused:
JSON parsers silently keep the last one), finite numbers only (no NaN / Infinity), and printable text only:
control / format characters (ESC, CR, newline, tab, bidi overrides, ...) are refused because the strings
are shown to the person who confirms. ``approval_id`` is 4-64 printable ASCII characters without spaces
(it is typed back exactly).
"""

from __future__ import annotations

import builtins
import difflib
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, fields
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from reachy_motion.safety.policy import DEFAULT_POLICY, SafetyPolicy, SafetyReport

if TYPE_CHECKING:  # pragma: no cover
    from reachy_motion.timeline import PreparedMotion

STAGES: tuple[str, ...] = ("R0", "R1", "R2", "R3")
MOTION_STAGES: frozenset[str] = frozenset({"R1", "R2", "R3"})
STAGE_DESCRIPTIONS: dict[str, str] = {
    "R0": "REST GET only - no ReachyMini client, no motion",
    "R1": "small amplitude, slow motions",
    "R2": "the designed motions",
    "R3": "parameter sweeps",
}
MAX_VALIDITY_DAYS = 7
MIN_APPROVAL_ID_LEN = 4  # the id is typed back at the terminal; a 1-character id would make that trivial
MAX_APPROVAL_ID_LEN = 64
MAX_APPROVAL_FILE_BYTES = 1 << 20  # an approval file is a few hundred bytes
METRIC_SPEED_RATIO = "peak_stewart_speed_ratio"  # SafetyReport.metrics key (x XL330-M288-T no-load speed)
METRIC_MARGIN_DEG = "min_stewart_margin_deg"  # SafetyReport.metrics key (distance to the URDF limit)
REPORT_DATA_KIND = "computed"  # a plan check: SDK IK on the planned commands, nothing simulated or measured

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")  # Timeline.digest() / policy_digest(): SHA-256 hex
# Printable ASCII without spaces: typed back exactly, so no look-alikes (NBSP, Cyrillic, ...) and no spaces.
_APPROVAL_ID_RE = re.compile(r"[!-~]{%d,%d}" % (MIN_APPROVAL_ID_LEN, MAX_APPROVAL_ID_LEN))
_WORD_RE = re.compile(r"[^\W_]+")  # letter / digit runs (any script)
# Whole-field placeholders (case-insensitive); "<...>" filling the whole field is one too.
_PLACEHOLDERS = frozenset({
    "", "-", "--", "?", "...", "none", "null", "nil", "n/a", "na", "todo", "tbd", "fixme", "xxx", "name",
    "your name", "full name", "test", "tester", "unknown", "anonymous", "someone", "me", "user", "the user",
    "human", "a human", "supervisor", "the supervisor", "admin", "operator",
})
# Words (case-insensitive, matched as whole tokens) that name an assistant / agent rather than a person.
_AGENT_WORDS = frozenset({
    "ai", "assistant", "agent", "agents", "subagent", "bot", "chatbot", "llm", "model", "claude", "claudecode",
    "anthropic", "gpt", "chatgpt", "openai", "copilot", "gemini", "codex", "automation", "autopilot", "script",
    "workflow", "orchestrator", "harness",
})
_AGENT_PREFIXES = ("gpt", "llm")  # gpt4, gpt4o, llms


class ApprovalError(RuntimeError):
    """An approval is missing, malformed, does not cover the request, or was not confirmed."""


# ---------------------------------------------------------------------------------------------- helpers
def _now() -> datetime:
    """Current local time (naive). Module level so that tests can move the clock (patch ``_now``)."""
    return datetime.now()


def _is_control(ch: str) -> bool:
    """Unicode category C*: control (ESC, CR, LF, TAB), format (bidi overrides, zero-width), unassigned..."""
    return unicodedata.category(ch).startswith("C")


def _escape(ch: str) -> str:
    return ch.encode("unicode_escape").decode("ascii")


def _control_chars(text: str) -> list[str]:
    """Escaped forms of the control / format characters in ``text`` (empty = printable)."""
    return sorted({_escape(ch) for ch in text if _is_control(ch)})


def _printable(text: str) -> str:
    """``text`` with control / format characters shown as escapes: safe to print on a terminal."""
    return "".join(_escape(ch) if _is_control(ch) else ch for ch in text)


def _short(value: Any, limit: int = 60) -> str:
    """Bounded ``repr`` for error messages."""
    try:
        text = repr(value)
    except ValueError:  # an int longer than sys.get_int_max_str_digits()
        return f"<{type(value).__name__} too long to show>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------------------------- parsing
def _parse_date(value: Any, key: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            pass
    raise ApprovalError(f"{key}: expected an ISO date like '2026-11-12', got {_short(value)}")


def _parse_datetime(value: Any, key: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        s = value.strip()
        if "T" not in s and " " not in s:
            raise ApprovalError(f"{key}: needs a date AND a time, e.g. '2026-11-12T18:00:00+09:00' "
                                f"(naive = local time of this PC), got {_short(value)}")
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            pass
    raise ApprovalError(f"{key}: expected an ISO datetime like '2026-11-12T18:00:00+09:00', got {_short(value)}")


def _str_tuple(value: Any, key: str, lower: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):  # not a str (iterates characters), dict (keys) or set (order)
        raise ApprovalError(f"{key}: expected a list of strings, got {type(value).__name__}")
    out = []
    for v in value:
        if not isinstance(v, str):
            raise ApprovalError(f"{key}: every entry must be a string, got {_short(v)}")
        v = v.strip()
        out.append(v.lower() if lower else v)
    return tuple(out)


def _number(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApprovalError(f"{key}: expected a number, got {_short(value)}")
    try:
        return float(value)
    except OverflowError:
        raise ApprovalError(f"{key}: the number is too large") from None


def _aware(dt: datetime) -> datetime:
    """Timezone-aware copy; a naive datetime is read as local time of this PC."""
    if dt.tzinfo is not None and dt.utcoffset() is not None:
        return dt
    try:
        return dt.astimezone()
    except (OverflowError, OSError, ValueError):
        # Windows cannot convert naive times near / before 1970 or far in the future (OSError 22):
        # use the current local UTC offset instead.
        return dt.replace(tzinfo=datetime.now().astimezone().tzinfo)


def _norm_host(host: str) -> str:
    return host.strip().lower().rstrip(".")


# ---------------------------------------------------------------------------------------------- policy
def policy_digest(policy: SafetyPolicy) -> str:
    """SHA-256 hex of a policy's name and every threshold (``SafetyPolicy.to_dict()``, canonical JSON).

    An approval binds this digest (``policy_sha256``): a plan checked under any other policy - a wider
    envelope, a disabled detector, unlimited speeds - is not covered even when it "passes" that policy.
    """
    if not isinstance(policy, SafetyPolicy):
        raise ApprovalError(f"expected a SafetyPolicy, got {type(policy).__name__}")
    blob = json.dumps(policy.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=repr)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def policy_relaxations(policy: SafetyPolicy, reference: SafetyPolicy = DEFAULT_POLICY) -> list[str]:
    """Thresholds where ``policy`` is looser than ``reference`` ("key value (default value)").

    ``min_*`` lower, ``max_*`` higher, ``*_range_mm`` wider, ``check_rate_hz`` lower or a NaN threshold
    (every comparison with NaN is False, i.e. the check never fires) count as looser; any other changed
    value (e.g. ``stream_rate_hz``) is listed as a difference. ``name`` and ``notes`` are ignored.
    """
    if not isinstance(policy, SafetyPolicy) or not isinstance(reference, SafetyPolicy):
        raise ApprovalError("policy_relaxations needs two SafetyPolicy objects")
    ref, cur = reference.to_dict(), policy.to_dict()
    out: list[str] = []
    for key, r in ref.items():
        if key in ("name", "notes"):
            continue
        c = cur.get(key)
        if c == r:
            continue
        num = (int, float)
        if isinstance(r, num) and isinstance(c, num) and not isinstance(c, bool):
            if math.isnan(c):
                looser = True
            elif key.startswith("min_") or key == "check_rate_hz":
                looser = c < r
            elif key.startswith("max_"):
                looser = c > r
            else:
                looser = True  # no direction known: list the difference
            if looser:
                out.append(f"{key} {c:g} (default {r:g})")
        elif (key.endswith("_range_mm") and isinstance(r, (list, tuple)) and isinstance(c, (list, tuple))
              and len(r) == len(c) == 2 and all(isinstance(v, num) for v in (*r, *c))):
            if not (c[0] >= r[0] and c[1] <= r[1]):  # also True for NaN bounds
                out.append(f"{key} [{c[0]:g}, {c[1]:g}] (default [{r[0]:g}, {r[1]:g}])")
        else:
            out.append(f"{key} {_short(c)} (default {_short(r)})")
    return out


def _is_default_policy(name: str, sha256: str) -> bool:
    return name == DEFAULT_POLICY.name and sha256 == policy_digest(DEFAULT_POLICY)


# ---------------------------------------------------------------------------------------------- types
@dataclass(frozen=True)
class ApprovalSource:
    """Where ``load_approval`` read an approval: binds it to the exact bytes of the human-written file."""

    path: str  # resolved absolute path of the approval file
    sha256: str  # SHA-256 hex of the file bytes as loaded
    size: int  # bytes
    loaded_at: datetime  # local, timezone-aware


@dataclass(frozen=True)
class RobotApproval:
    """One human-written approval (``load_approval``). Strings are normalised; dates are parsed.

    ``source`` is set only by ``load_approval`` (it is not a constructor argument, not compared, and
    ``dataclasses.replace`` drops it): an approval without a source was not read from a human-written file
    and the robot gate refuses it (``source_problems``).
    """

    approval_id: str  # chosen by the human; typed back at the terminal to confirm
    stage: str  # "R0" | "R1" | "R2" | "R3" (R0 never allows motion)
    approved_by: str  # the human supervisor's name (a signature: code cannot verify it)
    approved_on: date  # ISO date; valid from local midnight of that day
    expires: datetime  # ISO datetime, aware or naive local
    host: str  # robot daemon host the approval applies to (e.g. "192.168.0.15")
    motions: tuple[str, ...]  # "name:variant" covered
    timeline_digests: tuple[str, ...]  # Timeline.digest() values covered (exact commands)
    max_stewart_speed_ratio: float  # cap on report metric peak_stewart_speed_ratio
    min_stewart_margin_deg: float  # floor on report metric min_stewart_margin_deg
    policy_name: str  # SafetyPolicy.name the plans were checked with
    policy_sha256: str  # policy_digest() of that policy: binds every threshold
    notes: str = ""
    source: ApprovalSource | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        for key in ("approval_id", "stage", "approved_by", "host", "policy_name", "policy_sha256", "notes"):
            if not isinstance(getattr(self, key), str):
                raise ApprovalError(f"{key}: expected a string, got {_short(getattr(self, key))}")
        object.__setattr__(self, "stage", self.stage.strip().upper())
        object.__setattr__(self, "policy_name", self.policy_name.strip())
        object.__setattr__(self, "policy_sha256", self.policy_sha256.strip().lower())
        object.__setattr__(self, "approved_on", _parse_date(self.approved_on, "approved_on"))
        object.__setattr__(self, "expires", _parse_datetime(self.expires, "expires"))
        object.__setattr__(self, "motions", _str_tuple(self.motions, "motions"))
        object.__setattr__(self, "timeline_digests", _str_tuple(self.timeline_digests, "timeline_digests", True))
        object.__setattr__(self, "max_stewart_speed_ratio",
                           _number(self.max_stewart_speed_ratio, "max_stewart_speed_ratio"))
        object.__setattr__(self, "min_stewart_margin_deg",
                           _number(self.min_stewart_margin_deg, "min_stewart_margin_deg"))

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form with the same keys as the approval file (for manifests and logs; no ``source``)."""
        values = {
            "approval_id": self.approval_id, "stage": self.stage, "approved_by": self.approved_by,
            "approved_on": self.approved_on.isoformat(), "expires": self.expires.isoformat(),
            "host": self.host, "motions": list(self.motions), "timeline_digests": list(self.timeline_digests),
            "max_stewart_speed_ratio": self.max_stewart_speed_ratio,
            "min_stewart_margin_deg": self.min_stewart_margin_deg, "policy_name": self.policy_name,
            "policy_sha256": self.policy_sha256, "notes": self.notes,
        }
        return {k: values[k] for k in APPROVAL_KEYS}

    def summary_lines(self) -> list[str]:
        """What the person confirming reads. Control / format characters are shown escaped, never raw."""
        p = _printable
        digests = ", ".join(p(d[:12]) + "..." for d in self.timeline_digests[:4])
        more = f" (+{len(self.timeline_digests) - 4} more)" if len(self.timeline_digests) > 4 else ""
        which = ("the default Phase 2 policy" if _is_default_policy(self.policy_name, self.policy_sha256)
                 else f"NOT the default policy {DEFAULT_POLICY.name}: check its thresholds")
        src = self.source
        origin = (f"{p(src.path)} (sha256 {src.sha256[:12]}..., loaded {src.loaded_at.isoformat(timespec='seconds')})"
                  if src is not None else "(none - not loaded from a file; the robot gate refuses it)")
        return [
            f"  approval id : {p(self.approval_id)}",
            f"  stage       : {p(self.stage)} ({STAGE_DESCRIPTIONS.get(self.stage, 'unknown stage')})",
            f"  approved by : {p(self.approved_by)} on {self.approved_on.isoformat()}",
            f"  expires     : {self.expires.isoformat()}",
            f"  robot host  : {p(self.host)}",
            f"  motions     : {', '.join(p(m) for m in self.motions) or '(none)'}",
            f"  timelines   : {len(self.timeline_digests)} digest(s): {digests or '(none)'}{more}",
            f"  policy      : {p(self.policy_name) or '(none)'} (sha256 {p(self.policy_sha256[:12])}...; {which})",
            f"  caps        : peak Stewart speed <= {self.max_stewart_speed_ratio:g} x no-load speed, "
            f"Stewart margin >= {self.min_stewart_margin_deg:g} deg",
            f"  file        : {origin}",
            f"  notes       : {p(self.notes) or '-'}",
        ]


APPROVAL_KEYS: tuple[str, ...] = tuple(f.name for f in fields(RobotApproval) if f.init)
_STR_KEYS = ("approval_id", "stage", "approved_by", "approved_on", "expires", "host", "policy_name",
             "policy_sha256", "notes")
_LIST_KEYS = ("motions", "timeline_digests")
_NUM_KEYS = ("max_stewart_speed_ratio", "min_stewart_margin_deg")


@dataclass(frozen=True)
class ApprovalRequest:
    """What the caller wants to run on the robot: the plans' identities and their safety reports."""

    host: str  # robot daemon host
    stage: str  # "R1" | "R2" | "R3"
    motions: tuple[str, ...]  # "name:variant"
    timeline_digests: tuple[str, ...]  # Timeline.digest() of every timeline that would be sent
    reports: tuple[SafetyReport, ...] = ()  # one SafetyReport per timeline

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not isinstance(self.stage, str):
            raise ApprovalError("host and stage must be strings")
        object.__setattr__(self, "stage", self.stage.strip().upper())
        object.__setattr__(self, "motions", _str_tuple(self.motions, "motions"))
        object.__setattr__(self, "timeline_digests", _str_tuple(self.timeline_digests, "timeline_digests", True))
        if isinstance(self.reports, SafetyReport) or not isinstance(self.reports, Iterable):
            raise ApprovalError("reports: expected a sequence of SafetyReport")
        object.__setattr__(self, "reports", tuple(self.reports))

    @classmethod
    def from_prepared(cls, host: str, stage: str, prepared: Iterable["PreparedMotion"]) -> "ApprovalRequest":
        """Request for the given ``PreparedMotion`` objects (digests recomputed from their timelines)."""
        motions: list[str] = []
        digests: list[str] = []
        reports: list[SafetyReport] = []
        for p in prepared:
            key = f"{p.timeline.name}:{p.timeline.variant}"
            digest = p.timeline.digest()
            if key not in motions:
                motions.append(key)
            if digest not in digests:
                digests.append(digest)
            reports.append(p.report)
        return cls(host=host, stage=stage, motions=tuple(motions), timeline_digests=tuple(digests),
                   reports=tuple(reports))


# ---------------------------------------------------------------------------------------------- loading
def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook``: refuse duplicate keys (``json`` would silently keep the last value)."""
    obj: dict[str, Any] = {}
    dups: list[str] = []
    for k, v in pairs:
        if k in obj and k not in dups:
            dups.append(k)
        obj[k] = v
    if dups:
        raise ApprovalError("duplicate key(s) " + ", ".join(repr(k) for k in dups)  # repr escapes control chars
                            + ": every key must appear exactly once (JSON would silently keep the last value)")
    return obj


def _no_constant(name: str) -> Any:
    """``parse_constant``: refuse NaN / Infinity / -Infinity (not JSON numbers; caps must be finite)."""
    raise ApprovalError(f"{name} is not allowed: every number must be finite")


def load_approval(path: str | Path) -> RobotApproval:
    """Read a human-written approval JSON file. Strict: every key exactly once, no unknown keys.

    Raises ApprovalError listing every problem found (duplicate / missing / unknown keys, wrong types,
    non-finite numbers, bad dates, control characters). Semantic checks (expiry, coverage, policy, caps)
    are ``validate_approval``'s job. The returned approval carries its ``source`` (resolved path, SHA-256
    of the bytes read, load time) - only this function sets it.
    """
    p = Path(path)
    try:
        resolved = p.resolve(strict=True)
        size = resolved.stat().st_size
        if size > MAX_APPROVAL_FILE_BYTES:
            raise ApprovalError(f"approval file {p} has {size} bytes: an approval is a small JSON object "
                                f"(limit {MAX_APPROVAL_FILE_BYTES} bytes)")
        raw = resolved.read_bytes()
    except OSError as e:
        raise ApprovalError(f"cannot read approval file {p}: {e}") from e
    try:
        text = raw.decode("utf-8-sig")  # tolerate the BOM that Windows editors add
    except UnicodeDecodeError as e:
        raise ApprovalError(f"cannot read approval file {p} as UTF-8 text: {e}") from e
    try:
        data = json.loads(text, object_pairs_hook=_unique_keys, parse_constant=_no_constant)
    except ApprovalError as e:
        raise ApprovalError(f"invalid approval file {p}: {e}") from None
    except json.JSONDecodeError as e:
        raise ApprovalError(f"{p}: not valid JSON (line {e.lineno}, column {e.colno}): {e.msg}") from e
    except (ValueError, RecursionError) as e:  # e.g. an integer beyond sys.get_int_max_str_digits(), deep nesting
        raise ApprovalError(f"{p}: not a valid approval JSON ({type(e).__name__}: {str(e)[:120]})") from e
    if not isinstance(data, dict):
        raise ApprovalError(f"{p}: the top level must be a JSON object with keys {', '.join(APPROVAL_KEYS)}")

    errors: list[str] = []
    for k in data:
        if k not in APPROVAL_KEYS:
            hint = difflib.get_close_matches(str(k), APPROVAL_KEYS, n=1)
            errors.append(f"unknown key {k!r}" + (f" (did you mean {hint[0]!r}?)" if hint else ""))
    missing = [k for k in APPROVAL_KEYS if k not in data]
    if missing:
        errors.append("missing keys: " + ", ".join(missing))
    for k in _STR_KEYS:
        if k in data and not isinstance(data[k], str):
            errors.append(f"{k}: expected a string, got {_short(data[k])}")
        elif isinstance(data.get(k), str) and _control_chars(data[k]):
            errors.append(f"{k}: control / format characters are not allowed ({', '.join(_control_chars(data[k]))}):"
                          " write plain one-line text")
    for k in _LIST_KEYS:
        v = data.get(k)
        if k not in data:
            continue
        if not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
            errors.append(f"{k}: expected a list of strings, got {_short(v)}")
            continue
        bad = sorted({c for x in v for c in _control_chars(x)})
        if bad:
            errors.append(f"{k}: control / format characters are not allowed ({', '.join(bad)})")
    for k in _NUM_KEYS:
        if k not in data:
            continue
        v = data[k]
        if v is None:
            errors.append(f"{k}: fill in a number")
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"{k}: expected a number, got {_short(v)}")
        else:
            try:
                finite = math.isfinite(float(v))
            except OverflowError:
                finite = False
            if not finite:
                errors.append(f"{k}: expected a finite number of a sensible size, got {_short(v)}")
    for k, parse in (("approved_on", _parse_date), ("expires", _parse_datetime)):
        if isinstance(data.get(k), str) and not _control_chars(data[k]):
            try:
                parse(data[k], k)
            except ApprovalError as e:
                errors.append(str(e))
    if errors:
        raise ApprovalError(f"invalid approval file {p}:\n  - " + "\n  - ".join(errors))
    try:
        approval = RobotApproval(**{k: data[k] for k in APPROVAL_KEYS})
    except (ApprovalError, ValueError, OverflowError, TypeError) as e:  # pragma: no cover - mirrored above
        raise ApprovalError(f"invalid approval file {p}: {e}") from e
    source = ApprovalSource(path=str(resolved), sha256=hashlib.sha256(raw).hexdigest(), size=len(raw),
                            loaded_at=_aware(_now()))
    object.__setattr__(approval, "source", source)
    return approval


# ---------------------------------------------------------------------------------------------- checks
def _approver_problem(name: str) -> str | None:
    """Placeholder or assistant / agent signature (a screen, not identity verification: see module doc)."""
    n = name.strip()
    if not n:
        return "approved_by is empty: an approval must name the human supervisor who gave it"
    if n.lower() in _PLACEHOLDERS or (n.startswith("<") and n.endswith(">")):
        return f"approved_by {name!r} is a placeholder, not a person: the human supervisor signs with their name"
    for token in _WORD_RE.findall(n.lower()):
        if token in _AGENT_WORDS or token.startswith(_AGENT_PREFIXES):
            return (f"approved_by {name!r} names an assistant / agent ({token!r}), not a person: only the human "
                    "supervisor signs (CLAUDE.md rule 3)")
    return None


def _metric(report: SafetyReport, key: str) -> float | None:
    try:
        v = float(report.metrics[key])
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None
    return v if math.isfinite(v) else None


def _violation_codes(report: SafetyReport) -> list[str] | None:
    try:
        return sorted({str(getattr(v, "code", v)) for v in report.violations})
    except TypeError:  # not iterable
        return None


def _report_problems(r: SafetyReport, key: str, approval: RobotApproval, cap: float, floor: float) -> list[str]:
    """Everything wrong with one plan's report: consistency, data kind, policy binding, caps."""
    out: list[str] = []
    policy = r.policy if isinstance(r.policy, SafetyPolicy) else None
    pname = policy.name if policy is not None else "?"
    codes = _violation_codes(r)
    if codes is None:
        out.append(f"report {key}: violations is not a sequence (malformed report)")
    if r.passes_policy is not True:
        out.append(f"report {key} does not pass policy {pname!r} ({', '.join(codes or []) or 'no violation listed'})")
    elif codes:
        out.append(f"report {key} claims to pass policy {pname!r} but lists violation(s) {', '.join(codes)}: "
                   "an inconsistent report is refused")
    if not (isinstance(r.data_kind, str) and r.data_kind == REPORT_DATA_KIND):
        out.append(f"report {key} has data_kind {_short(r.data_kind)}: only a plan check (data_kind "
                   f"{REPORT_DATA_KIND!r}) can be approved")
    try:
        digest = policy_digest(policy) if policy is not None else None
    except (TypeError, ValueError) as e:  # a policy whose fields cannot be serialised
        digest = None
        out.append(f"report {key}: its policy cannot be digested ({type(e).__name__})")
    if policy is None:
        out.append(f"report {key} names no SafetyPolicy: the approved policy cannot be verified")
    elif digest is not None:
        if digest != approval.policy_sha256:
            out.append(f"report {key} was checked under policy {pname!r} (sha256 {digest[:12]}...), not the "
                       f"approved policy {approval.policy_name!r} (sha256 {approval.policy_sha256[:12]}...)")
        elif policy.name != approval.policy_name:
            out.append(f"approval policy_name {approval.policy_name!r} is not the name of the policy with that "
                       f"digest ({pname!r})")
    speed = _metric(r, METRIC_SPEED_RATIO)
    if speed is None:
        out.append(f"report {key} has no finite {METRIC_SPEED_RATIO}: the speed cap cannot be checked")
    elif speed < 0.0:
        out.append(f"report {key}: {METRIC_SPEED_RATIO} {speed:.4g} is negative (not a speed ratio)")
    elif speed > cap:
        out.append(f"report {key}: {METRIC_SPEED_RATIO} {speed:.4g} is above the approval cap {cap:g}")
    margin = _metric(r, METRIC_MARGIN_DEG)
    if margin is None:
        out.append(f"report {key} has no finite {METRIC_MARGIN_DEG}: the margin floor cannot be checked")
    elif margin < floor:
        out.append(f"report {key}: {METRIC_MARGIN_DEG} {margin:.4g} is below the approval floor {floor:g}")
    return out


def validate_approval(approval: RobotApproval, request: ApprovalRequest, now: datetime | None = None) -> list[str]:
    """Every reason why ``approval`` does not allow ``request`` (empty list = allowed by content).

    ``now`` (a datetime) defaults to the current local time; naive datetimes are read as local time.
    Pure: no file access. Whether the approval came from an unchanged human-written file is
    ``source_problems``; the robot gate runs both (``gate_problems``).
    """
    if not isinstance(approval, RobotApproval):
        return [f"approval must be a RobotApproval (load_approval), got {type(approval).__name__}"]
    if not isinstance(request, ApprovalRequest):
        return [f"request must be an ApprovalRequest, got {type(request).__name__}"]
    problems: list[str] = []

    # --- who / which approval
    who = _approver_problem(approval.approved_by)
    if who:
        problems.append(who)
    aid = approval.approval_id
    if not _APPROVAL_ID_RE.fullmatch(aid):
        problems.append(f"approval_id {aid!r} must be {MIN_APPROVAL_ID_LEN}-{MAX_APPROVAL_ID_LEN} printable ASCII "
                        "characters without spaces (it is typed back exactly at the terminal)")
    for key in ("approved_by", "host", "policy_name", "notes"):
        bad = _control_chars(getattr(approval, key))
        if bad:
            problems.append(f"{key} contains control / format characters ({', '.join(bad)})")
    bad = sorted({c for m in approval.motions for c in _control_chars(m)})
    if bad:
        problems.append(f"motions contain control / format characters ({', '.join(bad)})")

    # --- stage and host
    if approval.stage not in STAGES:
        problems.append(f"unknown approval stage {approval.stage!r} (expected one of {', '.join(STAGES)})")
    elif approval.stage not in MOTION_STAGES:
        problems.append("stage R0 is REST GET only (CLAUDE.md rule 3): it never allows a robot client or "
                        "motion - motion needs an R1, R2 or R3 approval")
    if request.stage != approval.stage:
        problems.append(f"stage mismatch: request {request.stage!r}, approval {approval.stage!r}")
    if not approval.host.strip():
        problems.append("approval host is empty")
    elif _norm_host(request.host) != _norm_host(approval.host):
        problems.append(f"host mismatch: request {request.host!r}, approval {approval.host!r}")

    # --- validity window
    if now is not None and not isinstance(now, datetime):
        problems.append(f"now must be a datetime, got {type(now).__name__}: the validity window cannot be checked")
    else:
        try:
            now_a = _aware(now if now is not None else _now())
            valid_from = _aware(datetime.combine(approval.approved_on, dtime.min))
            expires = _aware(approval.expires)
            if expires <= valid_from:
                problems.append(f"expires {approval.expires.isoformat()} is not after approved_on "
                                f"{approval.approved_on.isoformat()}")
            elif expires - valid_from > timedelta(days=MAX_VALIDITY_DAYS):
                problems.append(f"approval spans more than {MAX_VALIDITY_DAYS} days (approved_on "
                                f"{approval.approved_on.isoformat()}, expires {approval.expires.isoformat()})")
            if now_a < valid_from:
                problems.append(f"approval not yet valid: approved_on {approval.approved_on.isoformat()} is in "
                                f"the future (now {now_a.isoformat(timespec='seconds')})")
            if now_a >= expires:
                problems.append(f"approval expired at {approval.expires.isoformat()} "
                                f"(now {now_a.isoformat(timespec='seconds')})")
        except (OverflowError, OSError, ValueError) as e:  # dates at the edge of the datetime range
            problems.append(f"validity window cannot be evaluated ({type(e).__name__}: {e})")

    # --- caps chosen by the human
    cap = approval.max_stewart_speed_ratio
    if not (math.isfinite(cap) and 0.0 < cap <= 1.0):
        problems.append(f"max_stewart_speed_ratio {cap!r} must be in (0, 1] (fraction of the no-load speed)")
    floor = approval.min_stewart_margin_deg
    if not (math.isfinite(floor) and floor >= 0.0):
        problems.append(f"min_stewart_margin_deg {floor!r} must be a finite number >= 0")

    # --- the policy the human approved
    if not approval.policy_name:
        problems.append("approval names no safety policy (policy_name): it must name the policy the plans "
                        "were checked with")
    if not _DIGEST_RE.fullmatch(approval.policy_sha256):
        problems.append(f"approval policy_sha256 {approval.policy_sha256[:70]!r} is not a 64-character SHA-256 "
                        "hex digest (policy_digest of the policy)")

    # --- what the approval covers
    for m in approval.motions:
        name, sep, _variant = m.partition(":")
        if not sep or not name or ":" in _variant:
            problems.append(f"approval motion {m!r} is not 'name:variant'")
    for d in approval.timeline_digests:
        if not _DIGEST_RE.fullmatch(d):
            problems.append(f"approval timeline digest {d[:70]!r} is not a 64-character SHA-256 hex digest")
    if not request.motions:
        problems.append("request names no motions")
    if not request.timeline_digests:
        problems.append("request lists no timeline digests: an approval must bind the exact commands")
    for m in request.motions:
        if m not in approval.motions:
            problems.append(f"motion {m} is not covered by approval {approval.approval_id!r}")
    for d in request.timeline_digests:
        if d not in approval.timeline_digests:
            problems.append(f"timeline digest {d[:12]}... is not covered by approval {approval.approval_id!r}")

    # --- the plans' safety reports
    if not request.reports:
        problems.append("request has no safety reports: every timeline must be checked before approval")
    report_digests: set[str] = set()
    report_keys: set[str] = set()
    for r in request.reports:
        if not isinstance(r, SafetyReport):
            problems.append(f"request report {_short(r)} is not a SafetyReport")
            continue
        key = f"{r.name}:{r.variant}"
        report_keys.add(key)
        report_digests.add(str(r.timeline_digest))
        if key not in request.motions:
            problems.append(f"report {key} is not one of the requested motions")
        if r.timeline_digest not in request.timeline_digests:
            problems.append(f"report {key} belongs to timeline {str(r.timeline_digest)[:12] or '(none)'}..., "
                            "which is not in the request")
        problems.extend(_report_problems(r, key, approval, cap, floor))
    for d in request.timeline_digests:
        if request.reports and d not in report_digests:
            problems.append(f"no safety report for timeline digest {d[:12]}...")
    for m in request.motions:
        if request.reports and m not in report_keys:
            problems.append(f"no safety report for motion {m}")
    return problems


def source_problems(approval: RobotApproval) -> list[str]:
    """Why ``approval`` is not an unchanged human-written file (empty list = it is).

    Requires the ``source`` that only ``load_approval`` sets, and re-reads the file: it must still hash to
    the SHA-256 recorded at load time (an edit after loading means the file no longer says what was loaded).
    """
    src = getattr(approval, "source", None)
    if not isinstance(src, ApprovalSource):
        return ["approval was not loaded from a file (load_approval): the robot accepts only an approval file "
                "written and saved by the human supervisor - approvals built or copied in code are refused"]
    try:
        digest = hashlib.sha256(Path(src.path).read_bytes()).hexdigest()
    except OSError as e:
        return [f"approval file {src.path} cannot be read any more ({type(e).__name__}): load it again"]
    if digest != src.sha256:
        return [f"approval file {src.path} changed since it was loaded (sha256 {src.sha256[:12]}... -> "
                f"{digest[:12]}...): load it again"]
    return []


def gate_problems(approval: RobotApproval, request: ApprovalRequest, now: datetime | None = None) -> list[str]:
    """Everything the robot gate checks: ``validate_approval`` + ``source_problems`` (empty list = allowed)."""
    problems = validate_approval(approval, request, now)
    if isinstance(approval, RobotApproval):
        problems.extend(source_problems(approval))
    return problems


def format_problems(problems: Sequence[str], header: str = "robot approval refused:") -> str:
    """One message for an exception / the terminal; control characters from any input are shown escaped."""
    return "\n  - ".join(_printable(str(line)) for line in (header, *problems))


def require_approval(approval: RobotApproval, request: ApprovalRequest, now: datetime | None = None) -> RobotApproval:
    """``validate_approval`` or raise ApprovalError listing every problem; returns the approval.

    Content only: the robot gate also requires an unchanged approval file (``gate_problems``).
    """
    problems = validate_approval(approval, request, now)
    if problems:
        raise ApprovalError(format_problems(problems))
    return approval


# ---------------------------------------------------------------------------------------------- human steps
def _stdin_isatty() -> bool:
    """True only when stdin is an interactive terminal.

    On Windows ``isatty()`` is also True for the NUL device (measured here: ``python < NUL`` -> True), which
    is how non-interactive tool shells attach stdin, so a console handle is required as well:
    ``GetConsoleMode`` succeeds only on console handles (0 for NUL and pipes, measured). Git Bash's mintty
    gives a pipe, not a console: confirm from PowerShell / cmd / Windows Terminal (or ``winpty``).
    """
    stdin = sys.stdin
    try:
        if stdin is None or not stdin.isatty():
            return False
        if sys.platform != "win32":
            return True
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(stdin.fileno())
        mode = ctypes.c_uint32()
        return bool(ctypes.windll.kernel32.GetConsoleMode(ctypes.c_void_p(handle), ctypes.byref(mode)))
    except (AttributeError, OSError, ValueError):
        return False


def confirm_interactively(approval: RobotApproval, input_fn: Callable[[str], str] | None = None,
                          isatty_fn: Callable[[], bool] | None = None, out: TextIO | None = None) -> bool:
    """A person at a terminal confirms by typing the approval id exactly. Returns True only then.

    Refuses (False, without prompting) when stdin is not a TTY: scripts, pipes and agents cannot confirm.
    ``input_fn`` / ``isatty_fn`` / ``out`` default to ``builtins.input`` / the console check / ``sys.stdout``,
    all looked up at call time (so a patched ``builtins.input`` is the one used). The prompt has no time
    limit: callers must validate again after it returns (``gate_problems``).
    """
    out = sys.stdout if out is None else out
    check_tty = _stdin_isatty if isatty_fn is None else isatty_fn
    ask = builtins.input if input_fn is None else input_fn
    try:
        tty = bool(check_tty())
    except (AttributeError, OSError, ValueError):
        tty = False
    if not tty:
        out.write("Robot approval NOT confirmed: stdin is not an interactive terminal. A person must type the "
                  "approval id at a terminal; scripts and agents cannot confirm.\n")
        return False
    if not isinstance(approval, RobotApproval) or not _APPROVAL_ID_RE.fullmatch(approval.approval_id):
        out.write("Robot approval NOT confirmed: missing approval or an approval id that cannot be typed back.\n")
        return False
    bar = "=" * 78
    out.write("\n".join([bar, "REAL ROBOT: commands would go to the PHYSICAL motors.", *approval.summary_lines(),
                         "Keep the emergency stop within reach.", bar]) + "\n")
    flush = getattr(out, "flush", None)
    if callable(flush):
        flush()
    try:
        answer = ask("Type the approval id exactly to confirm (anything else cancels): ")
    except (EOFError, KeyboardInterrupt):
        out.write("\nRobot approval NOT confirmed.\n")
        return False
    ok = isinstance(answer, str) and answer.rstrip("\r\n") == approval.approval_id
    out.write("Robot approval confirmed.\n" if ok else "Robot approval NOT confirmed: the typed id does not match.\n")
    return ok


def worst_metrics(reports: Iterable[SafetyReport]) -> dict[str, float | None]:
    """Worst ``peak_stewart_speed_ratio`` (max) and ``min_stewart_margin_deg`` (min) over the reports."""
    speeds, margins = [], []
    for r in reports:
        s, m = _metric(r, METRIC_SPEED_RATIO), _metric(r, METRIC_MARGIN_DEG)
        if s is not None:
            speeds.append(s)
        if m is not None:
            margins.append(m)
    return {METRIC_SPEED_RATIO: max(speeds) if speeds else None,
            METRIC_MARGIN_DEG: min(margins) if margins else None}


def approval_template(request: ApprovalRequest) -> dict[str, Any]:
    """A fill-in template for the HUMAN supervisor; returned only, never written to disk.

    Request-derived fields (stage, host, motions, digests, and the policy the reports were checked with:
    ``policy_name`` / ``policy_sha256``) are pre-filled; everything the human decides stays empty
    (approval_id, approved_by, approved_on, expires) or None (both caps), so the template itself never
    loads as an approval. ``notes`` carries the computed worst values and, when the policy is looser than
    the default policy, every relaxed threshold. All reports must share one policy (one approval binds one).
    The assistant / agents must not fill in, save or create approval files (CLAUDE.md rule 3).
    """
    if not isinstance(request, ApprovalRequest):
        raise ApprovalError(f"request must be an ApprovalRequest, got {type(request).__name__}")
    if request.stage not in MOTION_STAGES:
        raise ApprovalError(f"stage {request.stage!r}: approval files exist only for R1, R2 and R3 "
                            "(R0 is REST GET only and needs no approval file)")
    policies: dict[str, SafetyPolicy] = {}
    for r in request.reports:
        if isinstance(r, SafetyReport) and isinstance(r.policy, SafetyPolicy):
            policies.setdefault(policy_digest(r.policy), r.policy)
    if len(policies) > 1:
        names = ", ".join(sorted(f"{_printable(p.name)} ({d[:12]}...)" for d, p in policies.items()))
        raise ApprovalError(f"the plans were checked under {len(policies)} different safety policies ({names}): "
                            "one approval binds one policy - check every plan with the same policy")
    digest, policy = next(iter(policies.items()), ("", None))
    worst = worst_metrics(request.reports)
    ref = ", ".join(f"{k}={v:.4g}" if v is not None else f"{k}=n/a" for k, v in worst.items())
    name = _printable(policy.name) if policy is not None else ""
    relaxed = policy_relaxations(policy) if policy is not None else []
    notes = (f"computed worst over {len(request.reports)} plan(s) (data_kind=computed, policy {name or 'n/a'}): "
             f"{ref}."
             + (f" Policy {name} is LOOSER than the default {DEFAULT_POLICY.name} in: {'; '.join(relaxed)}."
                if relaxed else "")
             + " Caps and every empty field are the human supervisor's decision.")
    values: dict[str, Any] = {
        "approval_id": "",
        "stage": request.stage,
        "approved_by": "",
        "approved_on": "",
        "expires": "",
        "host": request.host,
        "motions": list(request.motions),
        "timeline_digests": list(request.timeline_digests),
        "max_stewart_speed_ratio": None,
        "min_stewart_margin_deg": None,
        "policy_name": policy.name if policy is not None else "",
        "policy_sha256": digest,
        "notes": notes,
    }
    return {k: values[k] for k in APPROVAL_KEYS}
