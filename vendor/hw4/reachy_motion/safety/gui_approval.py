"""GUI approval of ONE robot action in the Motion Lab - in force since 2026-11-12 (user decision, CLAUDE.md rule 3:
``GUI_APPROVAL_ENABLED = True``).

Why: the Motion Lab is the HW4 experiment interface; a person connects, wakes up and runs the R1 / R2 / R3 trials from
the GUI. The terminal path (a human-written approval FILE plus its approval ID typed at a terminal, ``safety.approval``)
stays for debugging and emergencies. Rule 3 exists to stop an AI agent from moving the robot on its own; the GUI
approval keeps that purpose with a different input:

* one dialog per action (connection, wake-up, ONE trial; every R3 step separately), opened only by a person's click;
  it shows what will run - trial, target amplitude, duration, the computed policy check and the required speed
  (Stewart speed ratio, joint margin) - and the person clicks its approve button (not the default button: Enter
  cancels). There is no code to type (user decision 2026-11-12): a code shown in the dialog proved nothing - it can be
  typed without reading and read by automation - and only added a step to every trial;
* a ``GuiApproval`` binds the action kind, host / port, the trial label and the digest of the exact timeline
  (``timeline.Timeline.digest``), is valid for ``MOTION_VALID_S`` and is used ONCE (the session keeps the used ids);
  nothing outside it runs: no batch, no automatic next step, no automatic return;
* a refused policy check cannot be approved (the dialog's approve button stays off and ``approve`` raises).

What it does NOT do (docs/lab/GUI_APPROVAL_POLICY.md): a terminal prompt with a TTY check stops scripts and agents that
have no terminal; a GUI dialog can be driven by UI automation (an agent with screen control, pywinauto, a test
harness that calls ``approve``). No input can tell a person from automation in either case; the guard against an
AI agent remains the development rule (the agent never builds approvals, never clicks robot actions, never runs the
GUI against the robot), plus the operator at the robot with the power switch.

With ``GUI_APPROVAL_ENABLED`` False every check below refuses with ``POLICY_NOT_CONFIRMED`` and ``SdkBackend`` does not
accept a GUI approval (the switch to stop the GUI robot path again). Nothing here sends anything; every number in a
request is ``computed``.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

#: The policy switch: True since 2026-11-12 (user decision, CLAUDE.md rule 3). False: file + terminal ID only.
GUI_APPROVAL_ENABLED = True
POLICY_NOT_CONFIRMED = ("GUI 승인이 꺼져 있습니다 (GUI_APPROVAL_ENABLED=False): GUI 승인으로는 실제 로봇을 구동하지 않습니다 "
                        "(CLAUDE.md 규칙 3, docs/lab/GUI_APPROVAL_POLICY.md)")
KINDS = ("connect", "wake_up", "trial")
MOTION_VALID_S = 300.0  # an approved wake-up / trial must start within 5 min of the approval
CONNECT_VALID_S = 4 * 3600.0  # a connection approval covers one connection (at most 4 h)
#: What the operator ticks in the connection dialog (robot_r0.R1_OPERATOR, split into checkable rows).
OPERATOR_ROWS_KO = (
    "이 PC만 로봇에 연결되어 있다: 대시보드 앱·데모·다른 PC의 SDK 스크립트·CLI(robot-r1 등)를 쓰지 않는다",
    "트래킹·시작 앱(startup app)이 꺼져 있다",
    "머리·안테나 주위에 아무것도 없고, 머리를 받칠 사람이 옆에 있다",
    "전원 스위치에 손이 닿는다 (이상하면 바로 끈다)",
)


class GuiApprovalError(ValueError):
    """The dialog input does not approve the request (refused check, missing operator row)."""


@dataclass(frozen=True)
class GuiRequest:
    """What one dialog shows and what an approval of it binds (computed; nothing sent)."""

    kind: str  # connect | wake_up | trial
    host: str
    port: int
    stage: str  # CONNECT | WAKE_UP | R1 | R2 | R3
    label: str  # trial label, "connect" or "sdk_wake_up"
    title: str  # for the person
    digest: str  # timeline digest of the trial / wake-up model; "" for a connection
    rows: tuple[tuple[str, str], ...]  # (name, value) shown in the dialog, in order
    policy_ok: bool  # the computed check passes (False: cannot be approved)
    problems: tuple[str, ...] = ()  # why not (shown in red)
    start_mode: str = ""  # connect: "sleep" | "awake" (from the R0 the dialog was built from)
    start_joints_deg: tuple[float, ...] = ()  # wake-up: present joints the model starts from (re-checked at run)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    data_kind: str = "computed"

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown GUI approval kind {self.kind!r} (one of {', '.join(KINDS)})")


@dataclass(frozen=True)
class GuiApproval:
    """A person approved ``request`` in the GUI dialog (or something that drove the dialog: module docstring)."""

    request: GuiRequest
    approval_id: str
    approved_utc: str
    approved_ns: int  # time.monotonic_ns() at the approval
    valid_s: float
    operator_rows: tuple[str, ...] = ()  # connect: the rows the operator ticked
    source: str = "gui_dialog"

    def age_s(self, now_ns: int | None = None) -> float:
        return ((time.monotonic_ns() if now_ns is None else now_ns) - self.approved_ns) / 1e9

    def record(self) -> dict[str, Any]:
        """For manifests and session records."""
        r = asdict(self.request)
        return {"approval_id": self.approval_id, "source": self.source, "approved_utc": self.approved_utc,
                "valid_s": self.valid_s, "operator_rows": list(self.operator_rows), "request": r,
                "policy": "GUI approval (proposal, docs/lab/GUI_APPROVAL_POLICY.md); GUI_APPROVAL_ENABLED="
                          f"{enabled()}"}


def enabled() -> bool:
    """The flag, read at call time (tests patch the module attribute only)."""
    return GUI_APPROVAL_ENABLED is True


def approve(request: GuiRequest, operator_rows: Sequence[str] = (), now_ns: int | None = None) -> GuiApproval:
    """The dialog's approve button: the check must pass and a connection needs every operator row ticked. Raises
    GuiApprovalError otherwise. Works with the flag off too (the GUI can show and record the dialog); the flag is
    checked where the approval would be used (``problems``)."""
    if not request.policy_ok:
        raise GuiApprovalError("계산 검사(computed)가 거부한 동작은 승인할 수 없습니다: " + "; ".join(request.problems))
    rows = tuple(operator_rows)
    if request.kind == "connect" and set(rows) != set(OPERATOR_ROWS_KO):
        raise GuiApprovalError("연결 전에 운영자 확인 항목을 모두 체크해야 합니다")
    return GuiApproval(request, uuid.uuid4().hex[:12], datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       time.monotonic_ns() if now_ns is None else int(now_ns),
                       CONNECT_VALID_S if request.kind == "connect" else MOTION_VALID_S, rows)


def problems(approval: Any, *, kind: str, host: str, port: int, label: str | None = None, digest: str | None = None,
             used: Iterable[str] = (), now_ns: int | None = None) -> list[str]:
    """Everything that stops ``approval`` from covering this action now ([] = covered). The policy flag first."""
    out: list[str] = []
    if not enabled():
        out.append(POLICY_NOT_CONFIRMED)
    if not isinstance(approval, GuiApproval):
        return out + ["이 동작의 GUI 승인이 없습니다 (사람이 대화상자에서 승인해야 함)"]
    r = approval.request
    if r.kind != kind:
        out.append(f"승인 종류가 다릅니다: {r.kind} (필요: {kind})")
    if (r.host, int(r.port)) != (host, int(port)):
        out.append(f"승인한 대상 {r.host}:{r.port} ≠ 지금 대상 {host}:{port}")
    if label is not None and r.label != label:
        out.append(f"승인한 시행 {r.label} ≠ 실행하려는 시행 {label}")
    if digest is not None and r.digest != digest:
        out.append("승인한 타임라인과 실행하려는 타임라인이 다릅니다 (다이제스트 불일치)")
    if approval.approval_id in set(used):
        out.append("이미 사용한 승인입니다 (승인 1개 = 동작 1개)")
    age = approval.age_s(now_ns)
    if not (math.isfinite(age) and 0.0 <= age <= approval.valid_s):
        out.append(f"승인이 만료되었습니다 ({age:.0f} s > {approval.valid_s:.0f} s): 다시 승인하세요")
    if not r.policy_ok:
        out.append("계산 검사가 거부한 요청입니다")
    return out


# ---------------------------------------------------------------------------------------------- requests
def _f(v: Any, fmt: str = ".3f") -> str:
    return format(v, fmt) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else "-"


def trial_request(plan: Any, host: str, port: int, extra_problems: Sequence[str] = ()) -> GuiRequest:
    """One R1 / R2 / R3 trial of an AWAKE robot (``robot_r1.build_plan(trial, None, awake=True)``: INIT -> trial ->
    INIT): the rows the person checks before clicking approve."""
    t, c = plan.trial, plan.check
    m = c.metrics
    cond = t.condition
    if t.stage == "R1":
        amp, dur = f"{t.condition['axis']} {cond['amplitude']:+g} {cond['unit']}", (
            f"이동 {cond['move_s']:g} s, 정지 {cond['hold_s']:g} s, 복귀 {cond['move_s']:g} s")
    elif t.stage == "R3":
        amp, dur = f"pitch {cond['amplitude_deg']:g}°", f"D {cond['duration_s']:g} s (편도), 정지 {cond['hold_s']:g} s"
    else:
        peak = max((abs(math.degrees(getattr(k.pose, ax))) for k in t.spec.effective().head
                    for ax in ("roll", "pitch", "yaw")), default=0.0)
        amp = f"{cond['motion']} {cond['variant']} (설계 머리 최대 {peak:.1f}°, 확정 모션 파일 그대로)"
        dur = f"모션 {_f(cond.get('duration_s'), '.2f')} s"
    rows = (("시행", f"{t.title}  [{t.label}]"), ("단계", t.stage), ("목표 진폭", amp), ("Duration", dur),
            ("기록 길이", f"{_f(c.record_s, '.1f')} s (앞 {t.pre_s:g} s · 뒤 {t.post_s:g} s 정지 포함)"),
            ("시작 · 끝", "INIT → 시행 → INIT (깨어 있는 로봇, 램프 없음)"),
            ("정책 검사 (computed)", "통과 (초기 정책)" if c.runnable else "거부"),
            ("예상 필요 속도", f"Stewart 속도 비 {_f(m.get('peak_stewart_speed_ratio'))} (한계 0.5), 안테나 "
                         f"{_f(m.get('peak_antenna_speed_ratio'))}, 몸통 {_f(m.get('peak_body_speed_ratio'))}"),
            ("최소 관절 여유", f"{_f(m.get('min_stewart_margin_deg'), '.1f')}° ({m.get('min_stewart_margin_joint') or '-'}, "
                         "최소 10°)"),
            ("실제 로봇 범위", c.real_scope or "-"), ("대상", f"{host}:{port}"),
            ("타임라인 다이제스트", plan.run.timeline.digest()[:16]),
            ("승인 범위", f"이 시행 1회만, {MOTION_VALID_S / 60:.0f}분 안에 시작. 다음 시행·복귀는 따로 승인"))
    probs = tuple(list(c.notes if not c.runnable else []) + list(extra_problems))
    return GuiRequest("trial", host, int(port), t.stage, t.label, t.title, plan.run.timeline.digest(), rows,
                      bool(plan.runnable) and not extra_problems, probs)


def wake_up_request(check: Any, host: str, port: int, start_joints_deg: Sequence[float]) -> GuiRequest:
    """The SDK wake-up (``sdk_wake_up.check_wake_up``: computed model of its gotos from the present state)."""
    rows = (("동작", "SDK 1.11.0 Wake-up: ReachyMini.wake_up()과 같은 goto 3개 (reachy_mini.py:724-738)"),
            ("순서", "INIT까지 2.0 s (minjerk, 안테나도 INIT로) → 0.1 s → roll 20° 0.2 s → INIT 0.2 s"),
            ("시작 자세", check.start_text), ("최대 관절 이동", f"{_f(check.max_joint_travel_deg, '.1f')}° "
                                                       f"({check.max_travel_joint})"),
            ("정책 검사 (computed)", "통과" if check.runnable else "거부"),
            ("예상 필요 속도", f"Stewart 속도 비 {_f(check.stewart_speed_ratio)} (한계 0.5), 안테나 "
                         f"{_f(check.antenna_speed_ratio)}"),
            ("최소 관절 여유", f"{_f(check.min_margin_deg, '.1f')}°"),
            ("시작 앱", "daemon의 wake_up(REST /api/move/play/wake_up)이 아니므로 시작 앱 콜백이 실행되지 않음 "
                     "(daemon/backend/abstract.py:1284-1323)"),
            ("소리", "no_media 클라이언트라 재생 안 함 (media/media_manager.py:264-278)"),
            ("대상", f"{host}:{port}"),
            ("승인 범위", f"Wake-up 1회만, {MOTION_VALID_S / 60:.0f}분 안에 시작"))
    return GuiRequest("wake_up", host, int(port), "WAKE_UP", "sdk_wake_up", "SDK Wake-up", check.digest, rows,
                      bool(check.runnable), tuple(check.problems),
                      start_joints_deg=tuple(float(v) for v in start_joints_deg))


def connect_request(r0: dict[str, Any] | None, host: str, port: int) -> GuiRequest:
    """One connection of this PC's SDK client to the robot daemon, built from the latest R0 (GET only)."""
    from reachy_motion import r1_readiness as rr

    mode = rr.start_mode(r0)
    probs: list[str] = []
    if not isinstance(r0, dict):
        probs.append("R0(상태 읽기) 결과가 없습니다: 먼저 '상태 읽기'")
    else:
        if r0.get("blocking", True):  # say which and what to do (the 11/12 GUI has no R0 box any more)
            probs.extend("로봇 상태(R0): " + t for t in rr.r0_block_reasons(r0))
        if r0.get("data_kind") != "measured":
            probs.append(f"R0 data_kind {r0.get('data_kind')!r}: 실제 로봇이 아닙니다")
        url_host = str(r0.get("daemon_url", "")).split("//")[-1].split(":")[0]
        if url_host and url_host != host:
            probs.append(f"R0는 {url_host}에서 읽었습니다 (지금 대상 {host})")
    d = (r0 or {}).get("derived") or {}
    what = ("토크가 꺼진 로봇: 현재 자세로 목표를 맞춘 뒤 전체 enable_motors()로 토크를 켜서 그 자리에 고정 (움직임 없음). "
            "그다음 Wake-up을 따로 승인" if mode == "sleep" else
            "깨어 있는 로봇(토크 켜짐): 토크·목표를 바꾸지 않고 INIT 근처·정지 상태를 확인 (움직임 없음)")
    torque = ("이 연결은 토크를 켭니다: 연결 순서 마지막에 전체 enable_motors()를 보냅니다 (SetTorqueCmd on, "
              "robot/backend.py:352-377). 연결 자체가 모터 동작입니다 - 현재 자세에 고정되며 움직이지 않는 것이 정상"
              if mode == "sleep" else
              "이미 켜져 있음: 이 연결은 토크를 바꾸지 않습니다 (enable_motors 없음)")
    rows = (("토크", torque),
            ("대상", f"{host}:{port} (SDK 클라이언트 1개: network 모드, spawn_daemon=False, no_media)"),
            ("R0 (GET)", f"{(r0 or {}).get('generated_local', '-')}: 모터 {d.get('motor_mode', '-')}, "
                         f"rest까지 {_f((d.get('distance_to_rest') or {}).get('rotation_deg'), '.1f')}°"),
            ("시작 상태", rr.MODE_KO.get(mode, mode)), ("연결이 하는 일", what),
            ("연결이 보내는 명령", "ReachyMini 생성(WebSocket 연결, SetAutomaticBodyYaw, daemon이 미디어를 가지고 있으면 "
                                "POST /api/media/release), stop_head_tracking, disable_wobbling" +
             (", set_target(현재 자세, 토크 꺼진 채), enable_motors() ← 여기서 토크 켜짐" if mode == "sleep" else "")),
            ("다른 클라이언트", "SDK에는 제어 잠금이 없다: 다른 앱·CLI·PC가 동시에 명령하면 막을 수 없고, 정지 확인·"
                          "추종 오차로 뒤늦게 보일 뿐이다 (잔여 위험, 기록됨)"),
            ("승인 범위", "이 연결 1개 (모션은 따로 승인)"))
    past = next((x for x in (r0 or {}).get("findings") or [] if x.get("code") == "start_beyond_limits_limp"), None)
    if past is not None and mode == "sleep":  # user decision 2026-11-17 (robot_r0.LIMP_BEYOND_TOL_DEG)
        beyond = ((d.get("joints") or {}).get("max_beyond_deg"))
        rows += (("한계 밖에서 쉬는 머리", f"토크가 꺼진 머리가 관절 한계를 {_f(beyond, '.1f')}° 넘어 쉬고 있습니다(앱으로 재운 뒤·"
                  "전원을 켠 뒤 흔함, 10° 이내만 허용). 연결은 그 자리에서 토크를 켭니다(공식 앱의 Wake up과 같은 전체 "
                  "enable). 이때 그 관절이 한계 쪽으로 조금 움직일 수 있으니 머리를 받친 채 누르세요. 이어서 Wake-up이 "
                  "INIT으로 옮깁니다"),)
    title = f"{host} 연결 = 토크 켜기 (enable_motors)" if mode == "sleep" else f"{host} 연결 (토크 유지)"
    return GuiRequest("connect", host, int(port), "CONNECT", "connect", title, "", rows, not probs,
                      tuple(probs), start_mode=mode)


__all__ = ["GUI_APPROVAL_ENABLED", "POLICY_NOT_CONFIRMED", "KINDS", "MOTION_VALID_S", "CONNECT_VALID_S",
           "OPERATOR_ROWS_KO", "GuiApprovalError", "GuiRequest", "GuiApproval", "enabled", "approve", "problems",
           "trial_request", "wake_up_request", "connect_request"]
