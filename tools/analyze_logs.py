"""이벤트 로그(logs/*/events.jsonl)를 세션 단위로 묶어 평가 지표를 계산한다.

사용법:
    python tools/analyze_logs.py                      # logs/ 아래 전부
    python tools/analyze_logs.py logs/20261128_140000 --out results.csv

오퍼레이터 패널의 "평가 기록"(참가자·과제)을 누른 뒤 진행한 세션은 그 값으로 묶인다.
조건(A/B)은 기록 칸에 입력한 값이 아니라 실제로 켜져 있던 기능(condition·feature 이벤트)으로 정한다.
입력값과 다르면 condition_mismatch에 표시한다. 지표는 IMPLEMENTATION_PLAN.md 8장 평가 계획과 맞춰 두었다.

- assisted: 세션 중 오퍼레이터가 대신 한 입력 수 (발화 주입·터치 대신 누르기·카드·얼굴 프리셋·상태 강제 이동,
  패널의 키오스크 미리보기 터치). 평가 프로토콜 8절대로 도움받은 시행을 따로 보기 위한 값.
- median_asr_s: 손님 말이 끝난 뒤 글자가 나올 때까지 (실제 음성 인식만, 오퍼레이터 입력은 없음).
- median_reply_s: 글자가 나온 뒤 로봇이 말을 시작할 때까지. 조건 B에는 의도한 대기(시선이 먼저 가는 0.3 s,
  인사 0.4 s, Aha!의 present 1.7 s)가 들어가므로 A/B 반응 속도 비교에는 쓰지 말고 참고로만 본다.
- sold_out_recovery_s: 품절 화면에서 옵션·장바구니로 넘어갈 때까지 (H2).
- repair_recovery_s: 처음 되물은 뒤 다음 상태로 넘어갈 때까지 (H2).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent

COLUMNS = ["file", "session", "participant", "condition", "trial_condition", "condition_mismatch", "task", "lang",
           "outcome", "duration_s", "utterances", "touches", "assisted", "repairs", "bargeins", "pauses",
           "help_offers", "seek_help", "recommendations", "rejections", "explains", "card_insert_s",
           "card_retries", "wrong_step_cards", "sold_out_recovery_s", "repair_recovery_s", "median_asr_s",
           "median_reply_s", "staff_calls", "staff_reasons", "total_won", "items"]
FEATURE_KEYS = ("motion", "gaze", "status_strip", "slot_led")  # 조건 A는 넷 다 끔, B는 다 켬 (app.py와 같음)
# 오퍼레이터가 손님·센서 대신 한 입력 (설정 동작인 reset·trial·condition·feature·stock·context는 제외)
ASSIST_ACTIONS = {"say", "speech_start", "card", "touch", "faces", "force_state", "lang"}
REPLY_WINDOW_S = 10.0     # 이 안에 로봇이 말하지 않으면 그 발화는 대답 없음으로 본다
START_INPUT_WINDOW_S = 2.0  # 세션 시작 직전의 터치·발화는 그 세션의 첫 입력이다 (세션 시작보다 먼저 기록됨)


def read_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def condition_of(features: dict[str, bool]) -> str:
    values = [features.get(k, True) for k in FEATURE_KEYS]
    if all(values):
        return "B"
    if not any(values):
        return "A"
    return "mixed"


def new_session(file: str, rec: dict[str, Any], trial: dict[str, str], condition: str) -> dict[str, Any]:
    typed = trial.get("condition", "")
    return {"file": file, "session": rec.get("session"), "participant": trial.get("participant", ""),
            "condition": condition, "trial_condition": typed,
            "condition_mismatch": bool(typed) and typed != condition, "task": trial.get("task", ""),
            "lang": rec.get("lang", ""), "outcome": "incomplete", "start_t": rec["t"], "end_t": rec["t"],
            "utterances": 0, "touches": 0, "assisted": 0, "repairs": 0, "bargeins": 0, "pauses": 0,
            "help_offers": 0, "seek_help": 0, "recommendations": 0, "rejections": 0, "explains": 0,
            "card_retries": 0, "wrong_step_cards": 0, "card_insert_s": None, "sold_out_recovery_s": None,
            "repair_recovery_s": None, "total_won": None, "items": None, "staff_calls": 0, "staff_reasons": "",
            "_pay_t": None, "_utter_t": None, "_replies": [], "_asr": [], "_sold_out_t": None, "_repair_t": None}


def finish(session: dict[str, Any], outcome: str, t: float) -> dict[str, Any]:
    session["outcome"] = outcome
    session["end_t"] = t
    session["duration_s"] = round(t - session["start_t"], 2)
    replies, asr = session.pop("_replies"), session.pop("_asr")
    session["median_reply_s"] = round(statistics.median(replies), 2) if replies else None
    session["median_asr_s"] = round(statistics.median(asr), 2) if asr else None
    for key in ("_pay_t", "_utter_t", "_sold_out_t", "_repair_t", "start_t", "end_t"):
        session.pop(key, None)
    return session


def count_input(session: dict[str, Any], rec: dict[str, Any], t: float) -> None:
    """손님 입력 하나 (세션 안의 것, 또는 세션을 시작한 첫 입력)."""
    kind = rec.get("kind")
    if kind == "speech.utterance":
        session["utterances"] += 1
        session["_utter_t"] = t
        latency = rec.get("latency_s")
        if isinstance(latency, (int, float)) and rec.get("source") != "operator":
            session["_asr"].append(float(latency))
    elif kind == "touch":
        session["touches"] += 1
    if rec.get("source") == "preview":  # 패널의 키오스크 미리보기 터치: operator 레코드가 따로 없어 여기서 센다
        session["assisted"] += 1


def analyze_file(path: Path) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    trial: dict[str, str] = {}
    features = {k: True for k in FEATURE_KEYS}
    label = path.parent.name
    last_t = 0.0
    pending_input: tuple[dict[str, Any], float] | None = None  # 세션 밖에서 들어온 마지막 입력
    for rec in read_records(path):
        kind, t = rec.get("kind"), float(rec.get("t", 0.0))
        last_t = t
        if kind == "trial":
            trial = {k: str(rec.get(k, "")) for k in ("participant", "condition", "task")}
            continue
        if kind == "condition":
            on = str(rec.get("condition", "B")).upper() == "B"
            features = {k: on for k in FEATURE_KEYS}
            continue
        if kind == "feature" and rec.get("key") in FEATURE_KEYS:
            features[str(rec["key"])] = bool(rec.get("value"))
            if current is not None:
                current["condition"] = condition_of(features)  # 세션 중에 바뀌면 마지막 상태로
                typed = current["trial_condition"]
                current["condition_mismatch"] = bool(typed) and typed != current["condition"]
            continue
        if kind == "session.start":
            if current is not None:
                sessions.append(finish(current, "abandoned", t))
            current = new_session(label, rec, trial, condition_of(features))
            if pending_input is not None and t - pending_input[1] <= START_INPUT_WINDOW_S:
                count_input(current, *pending_input)
            pending_input = None
            continue
        if current is None:
            if kind in ("speech.utterance", "touch"):
                pending_input = (rec, t)
            continue
        if kind in ("speech.utterance", "touch"):
            count_input(current, rec, t)
        elif kind == "operator" and rec.get("action") in ASSIST_ACTIONS:
            current["assisted"] += 1
        elif kind == "say" and current["_utter_t"] is not None:
            gap = t - current["_utter_t"]
            if gap <= REPLY_WINDOW_S:
                current["_replies"].append(gap)
            current["_utter_t"] = None
        elif kind == "lang" and rec.get("lang"):
            current["lang"] = rec["lang"]  # 세션 중 언어를 바꾸면 마지막 언어
        elif kind == "bargein":
            current["bargeins"] += 1
        elif kind == "session.pause":
            current["pauses"] += 1
        elif kind == "fsm.state":
            state = rec.get("state")
            if state == "OFFER_HELP":
                current["help_offers"] += 1
            if state == "SOLD_OUT" and current["_sold_out_t"] is None:
                current["_sold_out_t"] = t
            elif (state in ("OPTIONS", "CART") and current["_sold_out_t"] is not None
                  and current["sold_out_recovery_s"] is None):
                current["sold_out_recovery_s"] = round(t - current["_sold_out_t"], 2)
            if current["_repair_t"] is not None and current["repair_recovery_s"] is None:
                current["repair_recovery_s"] = round(t - current["_repair_t"], 2)
            if state == "IDLE":  # 주문번호 없이 대기로 돌아감 (손님이 떠남·리셋): 여기서 끝난 것으로 본다
                sessions.append(finish(current, "abandoned", t))
                current = None
        elif kind == "staff.call" and not rec.get("again"):
            # 규칙 밖의 상황으로 직원을 부름 (이유: request·allergy·payment·refund·facility·repair)
            current["staff_calls"] += 1
            reasons = [r for r in current["staff_reasons"].split("+") if r] + [str(rec.get("reason", ""))]
            current["staff_reasons"] = "+".join(reasons)
        elif kind == "human.state" and rec.get("reason") == "seek_help":
            current["seek_help"] += 1
        elif kind == "rec.suggest":
            current["recommendations"] += 1
        elif kind == "rec.reject":
            current["rejections"] += 1
        elif kind == "act":
            name = rec.get("act")
            if name == "repair":
                current["repairs"] += 1
                if current["_repair_t"] is None:
                    current["_repair_t"] = t
            elif name == "explain":
                current["explains"] += 1
            elif name == "card_retry":
                current["card_retries"] += 1
            elif name == "not_payment_step":
                current["wrong_step_cards"] += 1
            elif name == "pay_request" and current["_pay_t"] is None:
                current["_pay_t"] = t
        elif kind == "card" and rec.get("state") == "in":
            if current["_pay_t"] is not None and current["card_insert_s"] is None:
                current["card_insert_s"] = round(t - current["_pay_t"], 2)
        elif kind == "order.done":
            current["total_won"] = rec.get("total")
            current["items"] = len(rec.get("lines", []))
            sessions.append(finish(current, "done", t))
            current = None
        elif kind == "session.cancel":
            sessions.append(finish(current, "cancelled", t))
            current = None
    if current is not None:
        sessions.append(finish(current, "incomplete", last_t))
    return sessions


def summarize(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for s in sessions:
        groups.setdefault(s["condition"] or "?", []).append(s)
    out = []
    for condition, items in sorted(groups.items()):
        done = [s for s in items if s["outcome"] == "done"]

        def med(key: str, pool: list[dict[str, Any]] = done) -> float | None:
            values = [s[key] for s in pool if s.get(key) is not None]
            return round(statistics.median(values), 2) if values else None

        def mean(key: str) -> float | None:
            values = [s[key] for s in items if s.get(key) is not None]
            return round(statistics.mean(values), 2) if values else None

        out.append({"condition": condition, "sessions": len(items), "completed": len(done),
                    "completion_rate": round(len(done) / len(items), 2) if items else None,
                    "assisted_sessions": sum(1 for s in items if s.get("assisted")),
                    "mismatched": sum(1 for s in items if s.get("condition_mismatch")),
                    "staff_sessions": sum(1 for s in items if s.get("staff_calls")),
                    "median_duration_s": med("duration_s"), "mean_repairs": mean("repairs"),
                    "median_card_insert_s": med("card_insert_s"), "mean_help_offers": mean("help_offers"),
                    "median_sold_out_recovery_s": med("sold_out_recovery_s", items),
                    "median_repair_recovery_s": med("repair_recovery_s", items),
                    "median_asr_s": med("median_asr_s", items), "median_reply_s": med("median_reply_s", items)})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reachy Café 이벤트 로그 → 세션별 평가 지표")
    parser.add_argument("paths", nargs="*", help="로그 폴더 또는 events.jsonl (기본: logs/ 전부)")
    parser.add_argument("--out", default=str(ROOT / "logs" / "summary_sessions.csv"), help="세션별 CSV 경로")
    parser.add_argument("--only-trials", action="store_true", help="참가자 ID가 기록된 세션만")
    args = parser.parse_args(argv)

    files: list[Path] = []
    targets = [Path(p) for p in args.paths] or [ROOT / "logs"]
    for target in targets:
        if target.is_file():
            files.append(target)
        else:
            files.extend(sorted(target.glob("**/events.jsonl")))
    sessions: list[dict[str, Any]] = []
    for path in files:
        sessions.extend(analyze_file(path))
    if args.only_trials:
        sessions = [s for s in sessions if s["participant"]]
    if not sessions:
        print("세션이 없습니다.", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:  # 엑셀에서 한글이 깨지지 않게 BOM
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for s in sessions:
            writer.writerow({key: s.get(key) for key in COLUMNS})

    print(f"세션 {len(sessions)}개 → {out}")
    mismatched = [s for s in sessions if s.get("condition_mismatch")]
    if mismatched:
        print(f"  주의: 기록 칸의 조건과 실제 조건이 다른 세션 {len(mismatched)}개 (condition_mismatch)")
    for row in summarize(sessions):
        print("  " + ", ".join(f"{k}={v}" for k, v in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
