"""Step guide of the Experiment tab (Qt-free): which of the five steps the user is at, what to press next, and why
the run button is disabled.

조건 설정 -> 안전 검사 -> 실험 선택 -> 실행 -> 결과 확인. The checks are the motion engine's own (computed when the
conditions change); this module only turns the tab's state into words. It never enables anything the engine or the
backend lock refuses. Real robot (mode "robot", gui.robot_session): a trial may only be started from INIT (connected
and woken up), one approved trial at a time, never "run all".
"""

from __future__ import annotations

from dataclasses import dataclass, field

STEPS = ("조건 설정", "안전 검사", "실험 선택", "실행", "결과 확인")


@dataclass
class GuideState:
    stage: str  # "R1" | "R2" | "R3"
    mode: str  # "mock" | "sim" | "robot"
    build_error: str | None = None  # the conditions cannot make trials (wrong number, missing motion ...)
    n_trials: int = 0
    n_runnable: int = 0
    refused: list[str] = field(default_factory=list)  # "trial: why" for the trials outside the policy
    selected: str | None = None  # title of the selected trial
    selected_runnable: bool = False
    selected_reason: str = ""
    selected_done: bool = False  # the selected trial already ran in this session
    sim_state: str = "idle"  # idle | connecting | ready | closed | error
    busy: bool = False
    running: str | None = None
    halted_at: str | None = None
    finished_any: bool = False  # a trial of the current conditions has finished
    previewing: str | None = None  # the Motion tab's MuJoCo preview (same worker) is playing this motion
    robot_state: str | None = None  # mode "robot": disconnected | rest_torque_on | at_init | holding


@dataclass
class Guide:
    step: int  # 0..4 (index into STEPS)
    next_action: str
    play_enabled: bool
    play_reason: str  # empty when enabled
    run_all_enabled: bool
    run_all_reason: str
    results_enabled: bool


def guide(s: GuideState, n_pending_runnable: int = 0) -> Guide:
    """``n_pending_runnable``: runnable trials of the table that did not run yet (for "run all")."""
    run_all_ok, run_all_why = False, ""
    if s.mode == "robot":
        run_all_why = "실제 로봇은 시행을 하나씩 실행합니다"
    elif s.stage == "R3":
        run_all_why = "R3는 위험 단계가 자동으로 이어지지 않도록 한 단계씩만 실행합니다"
    elif n_pending_runnable == 0:
        run_all_why = "아직 실행하지 않은 실행 가능 시행이 없습니다"
    results_ok = s.finished_any
    if s.build_error:
        return Guide(0, f"조건을 고치세요: {s.build_error}", False, s.build_error, False, s.build_error, results_ok)
    if s.n_trials == 0:
        why = "조건에 맞는 시행이 없습니다"
        return Guide(0, "실험 조건을 설정하세요.", False, why, False, why, results_ok)
    if s.busy:
        if s.previewing and not s.running:
            why = f"모션 탭의 MuJoCo 미리보기({s.previewing})가 재생 중입니다"
            return Guide(2 if s.selected else 0, why + ". 끝나면 실행할 수 있습니다.", False, why, False, why,
                         results_ok)
        why = f"실행 중입니다: {s.running or '준비 중'}"
        nxt = (f"실행 중: {s.running}. 끝나면 '결과 보기'가 켜집니다. 멈추려면 '정지'." if s.running else
               ("MuJoCo를 시작하는 중입니다 (약 10초)." if s.sim_state == "connecting" and s.mode == "sim"
                else "실행을 준비하는 중입니다 (시작 자세로 이동, 정지 확인)."))
        return Guide(3, nxt, False, why, False, why, results_ok)
    if s.halted_at and s.stage == "R3":
        why = f"R3 사다리가 {s.halted_at}의 이상으로 멈췄습니다. 결과를 확인한 뒤 '중단 해제'를 누르세요"
        return Guide(4, why + ".", False, why, False, why, True)
    if s.n_runnable == 0:
        first = s.refused[0] if s.refused else "초기 정책 밖"
        why = f"모든 시행이 초기 정책 밖입니다 ({first})"
        return Guide(1, f"{why}. 조건(예: D를 늘리거나 A를 줄이기)을 바꾸세요.", False, why, False, why, results_ok)
    if s.selected is None:
        why = "시행 표에서 시행을 하나 고르세요"
        return Guide(2, why + ".", False, why, s.mode != "robot" and s.stage != "R3" and n_pending_runnable > 0,
                     run_all_why, results_ok)
    if not s.selected_runnable:
        why = f"선택한 시행은 실행할 수 없습니다: {s.selected_reason}"
        return Guide(2, why + ". 다른 시행을 고르세요.", False, why,
                     s.mode != "robot" and s.stage != "R3" and n_pending_runnable > 0,
                     run_all_why, results_ok)
    if s.mode == "robot" and s.robot_state != "at_init":
        why = {"disconnected": "로봇 상자에서 '연결'을 누르세요",
               "rest_torque_on": "'Wake-up'을 누르세요",
               "holding": "정지됨: 'Wake-up'으로 INIT에 돌아가거나, 머리를 받치고 '토크 OFF'"}.get(
            s.robot_state or "disconnected", "실제 로봇이 INIT 대기 상태가 아닙니다")
        return Guide(3, why + ".", False, why, False, run_all_why, results_ok)
    run_all_ok = s.stage != "R3" and n_pending_runnable > 0 and s.mode != "robot"
    if s.mode == "robot":
        return Guide(3, f"'선택한 시행 실행'을 누르면 '{s.selected}'이(가) 실제 로봇에서 실행됩니다.", True, "", False,
                     run_all_why, results_ok)
    start_note = " (MuJoCo가 먼저 자동으로 시작됩니다, 약 10초)" if s.mode == "sim" and s.sim_state != "ready" else ""
    if s.finished_any:
        if s.selected_done:
            nxt = ("선택한 시행은 이미 실행했습니다 (다시 실행하면 새 기록이 추가됩니다). '결과 보기'로 결과를 "
                   "확인하세요.")
        else:
            nxt = (f"다음 시행 '{s.selected}'이(가) 선택돼 있습니다. '선택한 시행 실행'으로 계속하거나{start_note} "
                   "'결과 보기'로 지금까지의 결과를 확인하세요.")
        return Guide(4, nxt, True, "", run_all_ok, run_all_why, True)
    again = " 이미 실행한 시행이며, 다시 실행하면 새 기록이 추가됩니다." if s.selected_done else ""
    return Guide(3, f"'선택한 시행 실행'을 누르세요{start_note}.{again} 조건은 왼쪽 2번 상자에서, 다른 시행은 3번 표에서 "
                    "고를 수 있습니다.", True, "", run_all_ok, run_all_why, results_ok)


__all__ = ["STEPS", "GuideState", "Guide", "guide"]
