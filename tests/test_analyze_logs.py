from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import analyze_logs  # noqa: E402


def write(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")


def test_sessions_are_grouped_with_trial_and_metrics(tmp_path: Path) -> None:
    log = tmp_path / "run1" / "events.jsonl"
    log.parent.mkdir()
    write(log, [
        {"t": 0.0, "kind": "trial", "participant": "P01", "condition": "A", "task": "T1"},
        {"t": 0.5, "kind": "condition", "condition": "A"},  # 조건은 실제로 켠 기능으로 정한다 (기록 칸 값이 아니라)
        {"t": 1.0, "kind": "session.start", "session": 1, "lang": "ko"},
        {"t": 2.0, "kind": "speech.utterance", "text": "라떼"},
        {"t": 3.2, "kind": "say", "key": "ask_temp"},
        {"t": 4.0, "kind": "act", "act": "repair"},
        {"t": 6.0, "kind": "act", "act": "pay_request"},
        {"t": 9.5, "kind": "card", "state": "in"},
        {"t": 14.0, "kind": "order.done", "number": 1, "total": 4500, "lines": [{}]},
        {"t": 20.0, "kind": "session.start", "session": 2, "lang": "en"},
        {"t": 25.0, "kind": "session.cancel"},
    ])
    sessions = analyze_logs.analyze_file(log)
    assert [s["outcome"] for s in sessions] == ["done", "cancelled"]
    first = sessions[0]
    assert first["participant"] == "P01" and first["condition"] == "A"
    assert first["duration_s"] == 13.0 and first["repairs"] == 1
    assert first["card_insert_s"] == 3.5 and first["median_reply_s"] == 1.2
    assert first["condition_mismatch"] is False
    summary = analyze_logs.summarize(sessions)
    assert summary[0]["completion_rate"] == 0.5


def analyze(tmp_path: Path, records: list[dict]) -> list[dict]:
    log = tmp_path / "run" / "events.jsonl"
    log.parent.mkdir(exist_ok=True)
    write(log, records)
    return analyze_logs.analyze_file(log)


def test_condition_comes_from_the_features_that_were_really_on(tmp_path: Path) -> None:
    sessions = analyze(tmp_path, [
        {"t": 0.0, "kind": "trial", "participant": "P02", "condition": "B", "task": "T1"},  # 기록 칸은 B (기본값)
        {"t": 0.5, "kind": "condition", "condition": "A"},                                    # 실제로는 A로 진행
        {"t": 1.0, "kind": "session.start", "session": 1},
        {"t": 5.0, "kind": "order.done", "total": 4000, "lines": [{}]},
        {"t": 6.0, "kind": "feature", "key": "motion", "value": True},                        # 기능 하나만 켬
        {"t": 7.0, "kind": "session.start", "session": 2},
        {"t": 9.0, "kind": "order.done", "total": 4000, "lines": [{}]},
    ])
    assert sessions[0]["condition"] == "A" and sessions[0]["condition_mismatch"] is True
    assert sessions[1]["condition"] == "mixed"


def test_operator_help_and_preview_touches_are_counted_as_assisted(tmp_path: Path) -> None:
    sessions = analyze(tmp_path, [
        {"t": 1.0, "kind": "session.start", "session": 1},
        {"t": 2.0, "kind": "operator", "action": "say", "text": "라떼 주세요"},
        {"t": 2.0, "kind": "speech.utterance", "text": "라떼 주세요", "source": "operator"},
        {"t": 3.0, "kind": "touch", "action": "set_option", "source": "preview"},   # 패널 미리보기에서 누름
        {"t": 4.0, "kind": "touch", "action": "confirm"},                            # 참가자가 누름
        {"t": 4.5, "kind": "operator", "action": "stock", "item": "latte"},          # 설정 동작은 개입이 아님
        {"t": 5.0, "kind": "order.done", "total": 4500, "lines": [{}]},
    ])
    assert sessions[0]["assisted"] == 2 and sessions[0]["touches"] == 2


def test_input_that_starts_the_session_belongs_to_it(tmp_path: Path) -> None:
    sessions = analyze(tmp_path, [
        {"t": 0.9, "kind": "speech.utterance", "text": "라떼 하나 주세요", "latency_s": 0.8},  # 세션 시작보다 먼저 기록됨
        {"t": 1.0, "kind": "session.start", "session": 1, "lang": "ko"},
        {"t": 1.5, "kind": "say", "key": "ask_temp"},
        {"t": 3.0, "kind": "lang", "lang": "en"},
        {"t": 8.0, "kind": "order.done", "total": 4500, "lines": [{}]},
    ])
    s = sessions[0]
    assert s["utterances"] == 1 and s["median_reply_s"] == 0.6 and s["median_asr_s"] == 0.8
    assert s["lang"] == "en"                                                      # 세션 중 바꾼 마지막 언어


def test_session_ends_when_the_kiosk_returns_to_idle(tmp_path: Path) -> None:
    sessions = analyze(tmp_path, [
        {"t": 1.0, "kind": "session.start", "session": 1},
        {"t": 4.0, "kind": "fsm.state", "prev": "GREET", "state": "IDLE"},          # 손님이 떠남
        {"t": 60.0, "kind": "session.start", "session": 2},
        {"t": 70.0, "kind": "order.done", "total": 4500, "lines": [{}]},
    ])
    assert sessions[0]["outcome"] == "abandoned" and sessions[0]["duration_s"] == 3.0


def test_recovery_times_and_wrong_step_cards(tmp_path: Path) -> None:
    sessions = analyze(tmp_path, [
        {"t": 1.0, "kind": "session.start", "session": 1},
        {"t": 2.0, "kind": "fsm.state", "prev": "MENU", "state": "SOLD_OUT"},
        {"t": 6.5, "kind": "fsm.state", "prev": "SOLD_OUT", "state": "OPTIONS"},
        {"t": 8.0, "kind": "act", "act": "repair"},
        {"t": 9.0, "kind": "fsm.state", "prev": "OPTIONS", "state": "CART"},
        {"t": 9.5, "kind": "act", "act": "not_payment_step"},
        {"t": 12.0, "kind": "order.done", "total": 5000, "lines": [{}]},
    ])
    s = sessions[0]
    assert s["sold_out_recovery_s"] == 4.5 and s["repair_recovery_s"] == 1.0 and s["wrong_step_cards"] == 1


def test_unanswered_utterance_is_not_paired_with_a_late_reminder(tmp_path: Path) -> None:
    sessions = analyze(tmp_path, [
        {"t": 1.0, "kind": "session.start", "session": 1},
        {"t": 2.0, "kind": "speech.utterance", "text": "음"},
        {"t": 22.0, "kind": "say", "key": "pay_repeat"},                            # 20 s 뒤 카드 재안내
        {"t": 30.0, "kind": "order.done", "total": 4500, "lines": [{}]},
    ])
    assert sessions[0]["median_reply_s"] is None


def test_staff_calls_are_counted_once_per_call_with_reasons(tmp_path: Path) -> None:
    log = tmp_path / "run1" / "events.jsonl"
    log.parent.mkdir()
    write(log, [
        {"t": 1.0, "kind": "session.start", "session": 1, "lang": "ko"},
        {"t": 2.0, "kind": "staff.call", "reason": "allergy", "again": False},
        {"t": 3.0, "kind": "staff.call", "reason": "request", "again": True},   # "직원 아직이에요?": 새 호출이 아님
        {"t": 9.0, "kind": "staff.done", "reason": "request"},
        {"t": 12.0, "kind": "staff.call", "reason": "repair", "again": False},
        {"t": 20.0, "kind": "order.done", "number": 1, "total": 4500, "lines": [{}]},
    ])
    sessions = analyze_logs.analyze_file(log)
    assert sessions[0]["staff_calls"] == 2 and sessions[0]["staff_reasons"] == "allergy+repair"
    assert analyze_logs.summarize(sessions)[0]["staff_sessions"] == 1


def test_cli_writes_csv(tmp_path: Path) -> None:
    log = tmp_path / "run" / "events.jsonl"
    log.parent.mkdir()
    write(log, [{"t": 0.0, "kind": "session.start", "session": 1},
                {"t": 5.0, "kind": "order.done", "total": 4000, "lines": [{}]}])
    out = tmp_path / "summary.csv"
    assert analyze_logs.main([str(tmp_path), "--out", str(out)]) == 0
    assert "duration_s" in out.read_text(encoding="utf-8-sig").splitlines()[0]
