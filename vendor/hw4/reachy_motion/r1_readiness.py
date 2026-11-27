"""R1 entry, sorted for people: what R0 checked, what the connection itself establishes, what only the operator
controls, what nobody can verify, and what blocks (Step B 2026-11-12; categories as ``robot_r0.r1_entry``, user
decision 2026-11-12). Pure function of an R0 report (``robot_r0``); no request, no motor command.

Rows (docs/phase4/STARTUP_STOP.md §8-8.2, §11-12):

* ``auto``       - read by R0's allowed GETs and judged by code: ``ok`` / ``attention`` or ``blocked``.
* ``connection`` - not readable, but the connection sequence (``backends.robot_connect``) establishes it with SDK calls
                   and re-checks it; a failed re-check stops the run: ``by_connection``.
* ``operator``   - only a person at the robot controls it: ``operator_confirms`` - confirmed at the terminal right
                   before the connection (``robot-c1`` / ``robot-r1``), otherwise no run. A confirmation is recorded as
                   such and never turns into "verified".
* ``residual``   - SDK 1.11.0 does not let anyone read or prevent it: ``residual`` risk, with the mitigation and what is
                   left over. Recorded with every run; an operator's word does not change it.
* the verdict    - ``blocked`` when any automatic row blocks; otherwise ``conditional``. Never "verified".
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

EXPECTED_VERSION = "1.11.0"
MAX_AGE_S = 600.0  # an R0 older than this is history, not the robot's present state
LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")
ANTENNA_LIMIT_DEG = math.degrees(3.05)  # SafetyPolicy.max_antenna_abs_rad (the SDK sleep angle)
ANTENNA_REST_DEG = (-10.0, 10.0)  # SDK INIT_ANTENNAS_JOINT_POSITIONS (reachy_mini.py:67), right / left
ANTENNA_WRAP_WARN_DEG = 30.0  # a reading this close to +-180 deg may be the other side of the wrap

CATEGORY_KO = {"auto": "자동 확인 (R0)", "connection": "연결 과정에서 처리 (SDK)",
               "operator": "운영자가 통제 (실행 직전 확인)", "residual": "남는 위험 (확인·예방 불가)"}
STATUS_KO = {"ok": "통과", "blocked": "차단", "attention": "주의", "by_connection": "연결 순서가 처리",
             "operator_confirms": "실행 직전 운영자 확인", "residual": "남는 위험"}


MODE_KO = {"awake": "AWAKE: 깨어 있음 (모터 켜짐)", "sleep": "SLEEP: 토크 꺼짐"}

#: What blocks an R0 (``robot_r0.evaluate`` "block" findings) and what the person does next, for the robot box
#: (2026-11-17: the refusal said only "R0 상자 참고", a box the 11/12 GUI no longer shows). The SDK 1.11.0 web
#: dashboard is replaced by the "Reachy Mini Control" desktop app (daemon/app/main.py:446-470); its Wake up is the
#: daemon's wake_up (routers/move.py:152). The wireless robot boots asleep (launcher.sh:18 --no-wake-up-on-start).
WAKE_UP_FIRST_KO = ("Reachy Mini Control 앱에서 로봇을 깨운(Wake up) 뒤 앱을 닫고 다시 '연결'하세요. 깨어 있는 로봇(토크 켜짐, "
                    "INIT)은 토크를 그대로 둔 채 연결됩니다")
R0_BLOCK_KO = {
    "no_status": "로봇이 응답하지 않습니다. IP·포트(8000), 같은 Wi-Fi, 로봇 전원을 확인하세요 (macOS: 시스템 설정 → 개인정보 "
                 "보호 및 보안 → 로컬 네트워크에서 터미널을 허용)",
    "version": "로봇 daemon 버전이 1.11.0이 아닙니다. 이 프로그램은 1.11.0 전용입니다(로봇을 업데이트하지 마세요)",
    "state": "로봇 daemon이 실행 중이 아닙니다. 로봇 전원을 다시 켜거나 Reachy Mini Control 앱에서 daemon을 켠 뒤 다시 하세요",
    "daemon_error": "로봇 daemon이 오류를 보고합니다. 로봇을 다시 켠 뒤 다시 하세요",
    "head_tracking_active": "헤드 트래킹이 켜져 있습니다(다른 앱이 로봇을 쓰는 중). 다른 앱을 끈 뒤 다시 하세요",
    "motors_unknown": "모터 상태(토크)를 읽지 못했습니다. 잠시 뒤 '새로고침' 후 다시 하세요",
    "no_pose": "머리 자세를 읽지 못했습니다. 잠시 뒤 '새로고침' 후 다시 하세요",
    "no_body_yaw": "몸통 각도를 읽지 못했습니다. 잠시 뒤 '새로고침' 후 다시 하세요",
    "fk_joint_mismatch": "로봇이 알려 준 머리 자세와 관절값이 맞지 않습니다. " + WAKE_UP_FIRST_KO,
    "ik_nan": "로봇이 알려 준 머리 자세를 계산할 수 없습니다. " + WAKE_UP_FIRST_KO,
    "start_beyond_limits": "머리가 관절 한계를 10°보다 많이 넘어 있거나, 토크가 켜진 채 한계 밖에 있습니다. 이 자세에서는 "
                           "토크를 켜지 않습니다. 머리를 손으로 받쳐 조금 들어 올린 뒤 '새로고침' → '연결'하거나, "
                           + WAKE_UP_FIRST_KO,
}


def r0_block_reasons(report: dict[str, Any] | None) -> list[str]:
    """Each blocking R0 finding in plain Korean with the next step (empty when nothing blocks)."""
    out = []
    for x in (report or {}).get("findings") or []:
        if x.get("level") == "block":
            code = str(x.get("code"))
            out.append(R0_BLOCK_KO.get(code, f"{code}: {x.get('message', '')}".strip()))
    if not out and (report or {}).get("blocking", True):
        out.append(R0_BLOCK_KO["no_status"] if not report else "R0 결과에 차단 표시가 있습니다")
    return out


def start_mode(report: dict[str, Any] | None) -> str:
    """Which start an R0 report describes: "awake" when R0 read the motors on (woken the usual way), else "sleep".
    For the display (GUI R0 box); the CLI takes the start explicitly (--start)."""
    return "awake" if ((report or {}).get("derived") or {}).get("motor_mode") == "enabled" else "sleep"


def host_of(report: dict[str, Any]) -> str:
    return str(report.get("daemon_url", "")).split("//")[-1].rsplit(":", 1)[0]


def is_loopback(host: str) -> bool:
    return host.strip().lower() in LOOPBACK


def report_time(report: dict[str, Any]) -> datetime | None:
    try:
        return datetime.fromisoformat(str(report.get("generated_local")))
    except (TypeError, ValueError):
        return None


def age_s(report: dict[str, Any], now: datetime | None = None) -> float | None:
    t = report_time(report)
    if t is None:
        return None
    now = now or (datetime.now(t.tzinfo) if t.tzinfo else datetime.now())
    return (now - t).total_seconds()


def age_text(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "측정 시각 알 수 없음"
    if seconds < 0:
        return "측정 시각이 이 PC 시계보다 뒤"
    if seconds < 60:
        return f"{seconds:.0f}초 전"
    if seconds < 3600:
        return f"{seconds / 60:.0f}분 전"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}시간 전"
    return f"{seconds / 86400:.1f}일 전"


def _row(key: str, category: str, status: str, title: str, detail: str, mitigation: str = "",
         residual: str = "") -> dict[str, str]:
    return {"key": key, "category": category, "status": status, "title": title, "detail": detail,
            "mitigation": mitigation, "residual": residual}


def assess(report: dict[str, Any] | None, now: datetime | None = None, max_age_s: float = MAX_AGE_S,
           mode: str = "sleep") -> dict[str, Any]:
    """Rows and verdict for one R0 report (see the module docstring). ``mode`` "awake": the robot was woken the
    usual way - the motors must be ON and the measured pose near INIT (``robot_r0.AWAKE_INIT_TOL``)."""
    if not isinstance(report, dict):
        rows = [_row("r0_read", "auto", "blocked", "R0 보고서", "R0 보고서가 없습니다. R0를 실행하세요 (GET만).")]
        return {"rows": rows, "verdict": "blocked", "verdict_ko": "차단: R0 결과 없음", "age_s": None,
                "measured_at": None, "host": ""}
    d = report.get("derived") or {}
    codes = {f.get("code"): f for f in report.get("findings") or [] if isinstance(f, dict)}
    host = host_of(report)
    age = age_s(report, now)
    rows: list[dict[str, str]] = []

    def auto(key: str, ok: bool, title: str, detail_ok: str, detail_bad: str, attention: bool = False) -> None:
        rows.append(_row(key, "auto", "ok" if ok and not attention else ("attention" if ok else "blocked"), title,
                         detail_ok if ok else detail_bad))

    # ------------------------------------------------------------------ automatic (R0 GET)
    auto("r0_read", "no_status" not in codes, "daemon 응답", f"{report.get('daemon_url')} 응답함",
         str((codes.get("no_status") or {}).get("message", "응답 없음")))
    auto("r0_fresh", age is not None and 0 <= age <= max_age_s, "측정 시각",
         f"{report.get('generated_local')} ({age_text(age)})",
         f"{report.get('generated_local')} ({age_text(age)}): 지금 상태가 아닙니다. 실제 로봇 시험 직전에 R0를 다시 실행하세요")
    real = report.get("data_kind") == "measured" and not is_loopback(host)
    auto("real_robot", real, "실제 로봇", f"{host}, data_kind measured",
         f"{host or '-'}, data_kind {report.get('data_kind')!r}: 실제 로봇 측정이 아닙니다 (가짜·로컬 daemon은 실제 로봇이 아님)")
    v = d.get("daemon_version")
    auto("version", v == EXPECTED_VERSION and "version" not in codes, "daemon 버전", f"{v}",
         f"{v!r} (필요: {EXPECTED_VERSION})")
    st = d.get("daemon_state")
    auto("state", st == "running" and not ({"state", "daemon_error"} & set(codes)), "daemon 상태", f"{st}, 오류 없음",
         f"{st!r}: " + "; ".join(str(codes[c]["message"]) for c in ("state", "daemon_error") if c in codes))
    mm = d.get("motor_mode")
    if mode == "awake":
        from reachy_motion.robot_r0 import AWAKE_INIT_TOL, init_pose_error

        auto("motors_enabled", mm == "enabled", "모터", "켜짐 (평소 방식으로 깨어 있음)",
             f"{mm!r}: AWAKE 시작은 평소 방식으로 깨운 로봇(토크 켜짐)에서만 합니다")
        err = init_pose_error(report)
        txt = "읽지 못함" if err is None else (f"회전 {err['rotation_deg']:.1f}°, 이동 {err['translation_mm']:.1f} mm, "
                                           f"안테나 {err['antenna_deg']:.1f}°, 몸통 {err['body_deg']:.1f}°")
        auto("init_pose", err is not None and all(err[k] <= AWAKE_INIT_TOL[k] for k in AWAKE_INIT_TOL), "INIT 자세",
             f"INIT까지 {txt} (허용 {AWAKE_INIT_TOL['rotation_deg']:g}°·{AWAKE_INIT_TOL['translation_mm']:g} mm, 처짐 포함)",
             f"INIT 근처가 아님: {txt}. 평소 방식으로 깨운 뒤 R0를 다시 실행하세요")
    else:
        auto("motors_disabled", mm == "disabled", "모터", "꺼짐 (토크 없음)",
             f"{mm!r}: 연결 순서는 모터가 꺼진 로봇에서만 시작합니다 (깨어 있는 로봇은 --start awake)")
    j = d.get("joints") or {}
    margin = j.get("min_stewart_margin_deg")
    auto("joints_limits", "start_beyond_limits" not in codes, "관절 여유",
         f"최소 {margin}°" + (" (10° 미만, 주의)" if "start_near_limits" in codes else ""),
         str((codes.get("start_beyond_limits") or {}).get("message", "관절이 한계 밖")),
         attention="start_near_limits" in codes)
    gap = j.get("ik_of_pose_vs_joints_max_deg")
    auto("pose_joints", not ({"fk_joint_mismatch", "ik_nan", "no_pose"} & set(codes)) and gap is not None,
         "자세-관절 일치", f"{gap}° (≤ 1°)",
         "; ".join(str(codes[c]["message"]) for c in ("fk_joint_mismatch", "ik_nan", "no_pose") if c in codes)
         or "관절 읽기 없음")
    auto("body_yaw", "no_body_yaw" not in codes, "몸통 yaw 읽기", f"{d.get('body_yaw_deg')}°",
         str((codes.get("no_body_yaw") or {}).get("message", "")))
    auto("tracking_flag", "head_tracking_active" not in codes, "face_target",
         "기본값 (켜짐의 증거 없음; 꺼짐의 증명도 아님 - 아래 미검증 위험)",
         str((codes.get("head_tracking_active") or {}).get("message", "트래킹이 켜져 있음")))
    other_blocks = [f for c, f in codes.items() if f.get("level") == "block" and c not in {
        "no_status", "version", "state", "daemon_error", "start_beyond_limits", "fk_joint_mismatch", "ik_nan",
        "no_pose", "no_body_yaw", "head_tracking_active", "motors_unknown"}]
    if "motors_unknown" in codes:
        other_blocks.append(codes["motors_unknown"])
    auto("other_findings", not other_blocks, "그 밖의 R0 차단", "없음",
         "; ".join(f"{f.get('code')}: {f.get('message')}" for f in other_blocks))
    ant = d.get("antennas_deg")
    if isinstance(ant, list) and len(ant) == 2 and all(isinstance(a, (int, float)) for a in ant):
        inside = all(abs(a) <= ANTENNA_LIMIT_DEG for a in ant)
        far = [abs(a - r) for a, r in zip(ant, ANTENNA_REST_DEG)]
        near_wrap = [a for a in ant if 180.0 - abs(a) < ANTENNA_WRAP_WARN_DEG]
        auto("antenna_start", inside, "안테나 시작 위치",
             f"오른쪽 {ant[0]:.1f}°, 왼쪽 {ant[1]:.1f}° (rest와 {far[0]:.0f}°, {far[1]:.0f}° 차이). 기동은 안테나를 "
             "움직이지 않고 이 위치에 둡니다 (rest 이동 보류)",
             f"오른쪽 {ant[0]:.1f}°, 왼쪽 {ant[1]:.1f}°: 정책 범위 ±{ANTENNA_LIMIT_DEG:.2f}° 밖",
             attention=bool(near_wrap))
    else:
        auto("antenna_start", False, "안테나 시작 위치", "", "안테나 각도를 읽지 못했습니다")

    # ------------------------------------------------------------------ the connection establishes these (SDK calls)
    status = ((report.get("answers") or {}).get("status") or {})
    media = ("이 daemon은 미디어 서버가 있어(no_media=False) 앱이 트래킹을 켤 수 있다. "
             if status.get("no_media") is False else "")
    rows += [
        _row("head_tracking", "connection", "by_connection", "머리 트래킹 끄기",
             media + "켜짐 플래그는 허용 GET에 없고 face_target 기본값은 꺼짐을 증명하지 못한다 (abstract.py:683-699, "
             "729). 그래서 확인 대신 연결이 끈다",
             "연결 3단계: 토크가 꺼진 채 stop_head_tracking()·disable_wobbling() (abstract.py:702-719, 이후 "
             "step_head_tracking은 바로 반환 729-730). 6단계: face_target 다시 확인, 켜져 있으면 중단 (실제 SDK 하네스: 머리 0°)",
             "6단계 뒤 다른 클라이언트가 다시 켜는 것은 막지 못한다 → 아래 '다른 클라이언트' 남는 위험"),
        _row("partial_torque", "connection", "by_connection", "일부 모터만 켜진 상태",
             "모터별 enable은 보고되는 모드를 바꾸지 않는다 (robot/backend.py:646-667). 전역 플래그가 없으면 제어 루프는 "
             "목표를 쓰지 않는다 (203)",
             "연결 7단계: 전체 enable_motors()가 다시 적용하고 모든 목표를 현재 관절로 고정한 뒤 모두 켠다 "
             "(614-626, 352-376), 8단계 정지 확인 (하네스 0°)",
             "연결 전에 이미 힘을 쓰던 모터가 있었는지는 알 수 없다 (위치는 R0 관절로 확인됨)"),
    ]
    if mode == "awake":  # the robot is already awake: nothing is re-pinned, tracking running refuses the run
        rows[-2:] = [
            _row("head_tracking", "connection", "by_connection", "머리 트래킹",
                 media + "face_target이 설정돼 있으면(트래킹 실행 중) 시행하지 않는다. 기본값은 꺼짐의 증명이 아니다",
                 "연결 때 토크가 켜진 채 stop_head_tracking()·disable_wobbling(): IK 입력이 INIT 목표(없으면 단위행렬 = "
                 "INIT)라 움직임이 없다(실제 SDK 하네스 확인). 뒤에 정지 확인",
                 "그 뒤 다른 클라이언트가 다시 켜는 것은 막지 못한다 → '다른 클라이언트' 남는 위험"),
            _row("partial_torque", "connection", "by_connection", "일부 모터만 꺼진 상태",
                 "모터별 토크 변경은 보고되는 모드에 나타나지 않는다 (robot/backend.py:646-667)",
                 "INIT 오차 확인과 정지 확인이 처지는 축을 잡는다 (AWAKE 연결은 다시 고정하지 않는다)",
                 "작은 처짐은 놓칠 수 있다"),
        ]

    # ------------------------------------------------------------------ operator (confirmed right before the run)
    ant_txt = (f"오른쪽 {ant[0]:.1f}°, 왼쪽 {ant[1]:.1f}°" if isinstance(ant, list) and len(ant) == 2 else "R0 값")
    for key, title, detail in (
            ("op_no_other_client", "실험 PC 하나만 연결",
             "Reachy 대시보드·앱, 다른 노트북의 SDK·스크립트를 모두 끄고 이 PC만 로봇에 연결 (가능하면 로봇과 이 PC만 있는 "
             "네트워크). 통제할 수 없으면 실행하지 않는다"),
            ("op_no_tracking_app", "트래킹·시작 앱 끄기", "머리 트래킹을 켜는 앱·데모와 시작 앱(startup app: 깨어날 때 "
                                                    "자동 실행될 수 있음)을 끄고 대시보드에서 확인"),
            ("op_physical", "주변·전원", "안정된 바닥, 머리·안테나 주변에 걸릴 것 없음, 전원 스위치를 손이 닿는 곳에, "
                                      "머리를 받칠 사람"),
            ("op_antenna_visual", "안테나 실제 위치", f"안테나가 R0 값({ant_txt})과 맞아 보이는지 눈으로 확인")):
        rows.append(_row(key, "operator", "operator_confirms", title,
                         detail + ". 연결 직전 터미널에서 운영자가 확인 (기록됨, 검증으로 바뀌지 않음)"))

    # ------------------------------------------------------------------ residual (SDK 1.11.0 analysis)
    rows += [
        _row("other_client", "residual", "residual", "다른 클라이언트의 제어",
             "SDK에 제어권 잠금이 없고, 허용 GET은 클라이언트 목록을 주지 않는다 (직접 SDK 연결은 앱 잠금도 우회)",
             "운영자가 환경을 통제 (실행 직전 확인) + 연결 4·8단계 정지 확인으로 감지 (하네스: 움직임 7.5° 뒤 중단)",
             "예방할 수 없다. 시행 중 다른 클라이언트가 명령하면 머리가 움직일 수 있고, 감지는 움직인 뒤다. 모든 실행 기록에 남김"),
        _row("antenna_convention", "residual", "residual", "안테나 영점·부호·래핑",
             "각도 변환은 컴파일된 모터 컨트롤러(reachy_mini_motor_controller 1.5.6) 안에 있어 소스로 확인할 수 없다. "
             "SDK는 값을 그대로 쓰고(abstract.py:657-667), goto는 래핑 없이 직선 보간한다(motion/goto.py:65-67)",
             "기동은 안테나를 움직이지 않음(현재 위치 유지). R1 안테나 시행은 한쪽씩 ±5°부터",
             "첫 안테나 시행 전까지 실제 회전 방향을 모른다. 오른쪽 157°는 SDK sleep 값(-174.75°)과 부호가 반대"),
    ]
    blocked = [r for r in rows if r["status"] == "blocked"]
    n_conn = sum(1 for r in rows if r["status"] == "by_connection")
    n_res = sum(1 for r in rows if r["status"] == "residual")
    n_op = sum(1 for r in rows if r["status"] == "operator_confirms")
    if blocked:
        verdict, verdict_ko = "blocked", "차단: " + ", ".join(r["title"] for r in blocked)
    else:
        verdict, verdict_ko = "conditional", (f"조건부: 자동 확인 통과. 연결 과정 처리 {n_conn}개, 실행 직전 운영자 확인 "
                                              f"{n_op}개, 남는 위험 {n_res}개 (검증 완료가 아님). 운영 환경을 통제할 수 "
                                              "없으면 실행하지 않음")
    return {"rows": rows, "verdict": verdict, "verdict_ko": verdict_ko, "age_s": age,
            "measured_at": report.get("generated_local"), "host": host}


__all__ = ["assess", "age_s", "age_text", "report_time", "host_of", "is_loopback", "start_mode", "CATEGORY_KO",
           "STATUS_KO", "MODE_KO", "MAX_AGE_S", "EXPECTED_VERSION", "R0_BLOCK_KO", "WAKE_UP_FIRST_KO", "r0_block_reasons"]
