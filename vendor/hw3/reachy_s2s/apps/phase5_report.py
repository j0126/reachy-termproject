"""Phase 5 results: label every turn with the scripted task, score what can be scored automatically, and
write the comparison tables, graphs and a sheet for the judgements a person has to make by listening.

    python -m reachy_s2s.apps.phase5_report                          # uses results/phase5/manifest.csv
    python -m reachy_s2s.apps.phase5_report path/to/manifest.csv

manifest.csv (one row per session; edit it to add runs):
    session,variant,condition,round,note
    20261017_073221_dashboard,A0,noise,1,
  - session: folder name under results/sessions/. Several sessions with the same variant/condition/round are merged
    in time order (e.g. after "New session" by mistake). A session that holds two systems can be listed twice.
  - condition: quiet | noise | motion_on | exclude (exclude = ignored)

Outputs next to the manifest: turns_labeled.csv, task_scores.csv, judgment_sheet.csv, summary.md,
latency.svg, success.svg, report.html (open it in a browser; the graphs are plain SVG, no extra packages).

Turns are matched to the script (TEST_GUIDE.md section 3) by transcript similarity inside each block
between "Reset conversation" presses. When a sentence was said again (misheard, redone), the last try counts and
the number of tries is kept. Automatic scores are keyword checks; the robot's actual speech is what counts, so
check the judgement sheet by listening (Cloud B's transcripts can differ from what it said).
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ script (TEST_GUIDE.md section 3)
#: (task, item, sentence, other wordings that count as the same sentence)
SCRIPT: list[tuple[str, str, str, tuple[str, ...]]] = [
    ("T1", "T1-1", "Reachy Mini가 뭐야?", ()),
    ("T1", "T1-2", "대구에서 가볼 만한 곳 하나 추천해 줘.", ()),
    ("T1", "T1-3", "물은 몇 도에서 끓어?", ()),
    ("T2", "T2-1", "내 이름은 준수야.", ()),
    ("T2", "T2-2", "오늘 저녁 메뉴 하나 추천해 줘.", ()),
    ("T2", "T2-3", "아까 내 이름이 뭐라고 했지?", ()),
    ("T3", "T3-1", "내가 좋아하는 로봇은 Reachy야.", ()),
    ("T3", "T3-2", "내일 회의는 오후 2시에 해.", ()),
    ("T4A", "T4A-1", "나 내일 마트에서 사과 3개 살 거야. 기억해 줘.", ("장바구니에 사과 3개 담아 줘.",)),
    ("T4A", "T4A-2", "3개 아니 2개로 바꿀게.", ("3개 아니 2개로 해 줘.",)),
    ("T4A", "T4A-3", "내가 사과 몇 개 산다고 했지?", ("사과 몇 개 담았어?",)),
    ("T4B", "T4B-1", "한국의 사계절을 자세히 설명해 줘.", ()),
    ("T4B", "T4B-2", "잠깐, 다른 질문할게. 1 더하기 1은 뭐야?", ()),
    ("T5", "T5-1", "회의는 11월 13일 오후 3시 20분에 DGIST에서 하고, 14명이 참석해.", ()),
    ("T5", "T5-2", "방금 말한 회의 정보를 다시 정리해 줘.", ()),
    ("CS", "CS-1", "오늘 meeting은 3시에 Zoom으로 하고, deadline은 다음 주 Friday야.", ()),
    ("CS", "CS-2", "meeting이랑 deadline 다시 말해 줘.", ()),
    ("CS", "CS-3", "'나는 로봇을 좋아해'를 영어로 말해 줘.", ()),
]
TASKS = ["T1", "T2", "T3", "T4A", "T4B", "T5", "CS"]
TASK_TITLES = {"T1": "T1 일반 대화", "T2": "T2 맥락 유지", "T3": "T3 멈춤 처리", "T4A": "T4A 발화 중 정정",
               "T4B": "T4B 끼어들기", "T5": "T5 핵심 정보", "CS": "한영 혼용"}
VARIANTS = ["A0", "A1", "B"]
CONDITIONS = ["quiet", "noise", "motion_on"]
COND_TITLES = {"quiet": "조용함", "noise": "소음", "motion_on": "Motion ON"}
MATCH_MIN = 0.45  # transcript similarity to count a turn as one of the script sentences
TASK_MIN = 0.35   # best similarity in a block below this -> the block is not a scripted task

#: spoken / transcribed forms -> one form, so that "리치미니" matches "Reachy Mini" and "세 개" matches "3개"
ALIASES = [
    (r"reachy|리치|레치|리찌", "리치"), (r"mini|미니", "미니"), (r"meeting|미팅", "미팅"), (r"zoom|줌", "줌"),
    (r"deadline|데드라인", "데드라인"), (r"friday|프라이데이", "프라이데이"), (r"dgist|디지스트", "디지스트"),
    (r"세\s*개", "3개"), (r"두\s*개", "2개"), (r"한\s*개", "1개"), (r"일\s*더하기\s*일", "1더하기1"),
    (r"열\s*네\s*명", "14명"), (r"두\s*시", "2시"), (r"세\s*시", "3시"),
]


def norm(text: str) -> str:
    t = (text or "").lower()
    for pattern, repl in ALIASES:
        t = re.sub(pattern, repl, t)
    return re.sub(r"[\s\W_]+", "", t)


def similarity(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def has(text: str, *patterns: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in patterns)


# ------------------------------------------------------------------ loading
@dataclass
class Turn:
    session: str
    order: float
    row: dict[str, str]
    block: int = 0
    item: str = ""
    sim: float = 0.0
    continuation: bool = False

    @property
    def text(self) -> str:  # what the system understood (A1 merged text when it joined two utterances)
        return self.row.get("merged_text") or self.row.get("transcript") or ""

    @property
    def response(self) -> str:
        return self.row.get("response") or ""

    def num(self, key: str) -> float | None:
        try:
            return float(self.row.get(key) or "")
        except ValueError:
            return None

    @property
    def spoke(self) -> bool:  # robot audio was pushed for this turn
        return bool(self.row.get("queue_first_push"))


@dataclass
class Run:  # one variant x condition x round (one or more sessions)
    variant: str
    condition: str
    round: str
    sessions: list[str] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)


def load_runs(manifest: Path, sessions_root: Path) -> list[Run]:
    runs: dict[tuple[str, str, str], Run] = {}
    with manifest.open(encoding="utf-8-sig") as fh:
        for rec in csv.DictReader(fh):
            cond = (rec.get("condition") or "").strip()
            if not rec.get("session") or cond in ("", "exclude"):
                continue
            key = (rec["variant"].strip(), cond, (rec.get("round") or "1").strip())
            run = runs.setdefault(key, Run(*key))
            run.sessions.append(rec["session"].strip())
    block = 0
    for run in runs.values():
        for sid in sorted(run.sessions):  # session folders are named by start time
            d = sessions_root / sid
            if not (d / "turns.csv").exists():
                print(f"[!!] {sid}: no turns.csv, skipped", file=sys.stderr)
                continue
            with (d / "turns.csv").open(encoding="utf-8") as fh:
                rows = [r for r in csv.DictReader(fh) if r.get("variant") == run.variant]
            resets = []
            if (d / "events.jsonl").exists():
                with (d / "events.jsonl").open(encoding="utf-8") as fh:
                    resets = [json.loads(line)["t"] for line in fh if '"conversation.reset"' in line]
            block += 1  # a new session starts a new block
            turns = sorted((Turn(sid, float(r.get("speech_start_est") or r.get("endpoint") or 0), r) for r in rows),
                           key=lambda t: t.order)
            for t in turns:
                t.block = block + sum(1 for x in resets if x < t.order) * 1  # resets in between -> later block
            block += len(resets) + 1
            run.turns.extend(turns)
    return [runs[k] for k in sorted(runs, key=lambda k: (VARIANTS.index(k[0]) if k[0] in VARIANTS else 9,
                                                          CONDITIONS.index(k[1]) if k[1] in CONDITIONS else 9, k[2]))]


# ------------------------------------------------------------------ labelling
def label(run: Run) -> dict[str, list[Turn]]:
    """Assign every turn to its script sentence (or to the sentence before it: the part after a pause, a
    misheard repeat). Returns task -> its turns from the LAST block that holds the task (a redo replaces)."""
    blocks: dict[int, list[Turn]] = {}
    for t in run.turns:
        blocks.setdefault(t.block, []).append(t)
    item_task = {item: task for task, item, _s, _a in SCRIPT}
    chosen: dict[str, list[Turn]] = {}
    for _b, turns in sorted(blocks.items()):
        current = ""
        per_task: dict[str, list[Turn]] = {}
        for t in turns:
            best_item, best = "", 0.0
            for task, item, sentence, alts in SCRIPT:
                s = max(similarity(t.text, x) for x in (sentence, *alts))
                if current and task == item_task[current]:
                    s += 0.05  # near tie: stay in the task being done (a missing Reset is still handled)
                if s > best:
                    best_item, best = item, s
            if best >= MATCH_MIN:
                current = best_item
                t.item, t.sim = best_item, min(best, 1.0)
            elif current:
                t.item, t.continuation = current, True
            else:
                t.item = "?"
                continue
            per_task.setdefault(item_task[t.item], []).append(t)
        chosen.update(per_task)  # a later block with the same task (a redo) replaces the earlier one
    return chosen


def by_item(turns: list[Turn]) -> dict[str, list[Turn]]:
    out: dict[str, list[Turn]] = {}
    for t in turns:
        if t.item and t.item != "?":
            out.setdefault(t.item, []).append(t)
    return out


# ------------------------------------------------------------------ automatic scores
NOT_UNDERSTOOD = (r"알아듣지\s*못", r"다시\s*(한\s*번\s*)?말씀", r"잘\s*못\s*알아")
SLOTS_T5 = {"date": (r"11\s*월\s*13\s*일", r"십일월\s*십삼일"), "time": (r"3\s*시\s*20\s*분", r"세\s*시\s*이십\s*분", r"15\s*시\s*20"),
            "place": (r"dgist", r"디지스트"), "people": (r"14\s*명", r"열\s*네\s*명", r"열네\s*분", r"14\s*분")}
TERMS_CS = {"meeting": (r"meeting", r"미팅"), "zoom": (r"zoom", r"줌"), "deadline": (r"deadline", r"데드라인", r"마감"),
            "friday": (r"friday", r"프라이데이", r"금요일"), "3pm": (r"3\s*시", r"세\s*시", r"15\s*시")}
RECALL_CS = {"3pm": TERMS_CS["3pm"], "zoom": TERMS_CS["zoom"], "friday": TERMS_CS["friday"]}


def answered(t: Turn | None) -> bool:
    return t is not None and t.row.get("status") in ("done", "interrupted") and bool(t.response) \
        and not has(t.response, *NOT_UNDERSTOOD)


def count_slots(text: str, slots: dict[str, tuple[str, ...]]) -> int:
    return sum(1 for pats in slots.values() if has(text, *pats))


def score_task(task: str, turns: list[Turn]) -> dict[str, Any]:
    it = by_item(turns)
    last = {k: v[-1] for k, v in it.items()}
    first = {k: v[0] for k, v in it.items()}
    # every turn is one user utterance: more than one per sentence = the user said it again (misheard, cut off).
    # T3 is left out: there the part after the pause is meant to be a second turn (it has its own metrics)
    s: dict[str, Any] = {"turns": len(turns),
                         "retries": 0 if task == "T3" else sum(max(0, len(v) - 1) for v in it.values())}
    if task == "T1":
        ok = {"T1-1": lambda r: has(r, r"로봇") and not has(r, r"모르겠", *NOT_UNDERSTOOD),
              "T1-2": lambda r: not has(r, r"수\s*가?\s*없", r"없어요", r"모르겠", *NOT_UNDERSTOOD),
              "T1-3": lambda r: has(r, r"100", r"백\s*도")}
        s["correct"] = sum(1 for i, f in ok.items() if answered(last.get(i)) and f(last[i].response))
        s["of"] = len(ok)
    elif task == "T2":
        if "T2-3" in last:
            heard = re.search(r"이름은\s*([^\s.,!?]+?)(?:이야|야|이에요|예요|입니다)?[\s.,!?]*$", first["T2-1"].text) \
                if "T2-1" in first else None
            s["recall_exact"] = has(last["T2-3"].response, r"준수")
            s["recall_heard"] = bool(heard) and heard.group(1) in last["T2-3"].response  # what it heard in turn 1
            s["name_heard"] = heard.group(1) if heard else ""
    elif task == "T3":
        for item, keys in (("T3-1", (r"리치", r"레치", r"reachy")), ("T3-2", (r"2\s*시", r"두\s*시", r"14\s*시"))):
            ts = it.get(item, [])
            if not ts:
                continue
            first_text = ts[0].row.get("transcript") or ""
            s[f"{item}_premature"] = not has(first_text, *keys)  # the turn ended in the pause (no second half)
            s[f"{item}_fragment_spoken"] = s[f"{item}_premature"] and ts[0].spoke  # robot answered the half sentence
            s[f"{item}_merged"] = any(t.row.get("merged_text") for t in ts)
            s[f"{item}_final_ok"] = has(ts[-1].text, *keys)  # what the system finally understood has the second half
            s[f"{item}_pause_ms"] = ";".join(t.row.get("pauses_ms") or "" for t in ts if t.row.get("pauses_ms"))
    elif task == "T4A":
        s["correct_2"] = has(last["T4A-3"].response, r"2\s*개", r"두\s*개") if "T4A-3" in last else None
    elif task == "T4B":
        t1 = first.get("T4B-1")
        s["bargein"] = bool(t1) and t1.row.get("status") == "interrupted"
        s["detect_ms"] = t1.num("bargein_detect_delay_ms") if t1 else None
        s["stop_ms"] = t1.num("playback_stop_delay_ms") if t1 else None
        s["heard_ratio"] = t1.num("heard_ratio_est") if t1 else None
        q = it.get("T4B-2", [])
        s["new_question_captured"] = bool(q)
        s["new_question_ok"] = any(has(t.response, r"(^|\D)2(\D|$)", r"이예요", r"이에요", r"이입니다", r"둘") for t in q)
    elif task == "T5":
        s["asr_slots"] = count_slots(first["T5-1"].text, SLOTS_T5) if "T5-1" in first else 0
        s["recall_slots"] = count_slots(last["T5-2"].response, SLOTS_T5) if "T5-2" in last else 0
    elif task == "CS":
        text = first["CS-1"].text if "CS-1" in first else ""
        s["asr_terms"] = count_slots(text, TERMS_CS)
        s["latin_kept"] = len(re.findall(r"meeting|zoom|deadline|friday", text.lower()))
        s["recall_terms"] = count_slots(last["CS-2"].response, RECALL_CS) if "CS-2" in last else 0
        r3 = last["CS-3"].response if "CS-3" in last else ""
        s["english_out"] = ("latin" if has(r3, r"i\s+like\s+robot") else
                            "hangul" if has(r3, r"아이\s*(엠\s*)?라이크", r"아임\s*라이크") else "")
    return s


# ------------------------------------------------------------------ summaries
def pct(values: list[float], q: float) -> float:
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def latency(turns: list[Turn]) -> dict[str, Any]:
    vals = [t.num("speech_end_to_push_ms") for t in turns]
    vals = [v for v in vals if v is not None and v > 0]
    out: dict[str, Any] = {"n": len(vals)}
    if vals:
        out.update(median=statistics.median(vals), p25=pct(vals, 0.25), p75=pct(vals, 0.75))
    for key in ("asr_ms", "llm_ms", "tts_ms", "vad_endpoint_delay_ms"):
        v = [t.num(key) for t in turns]
        v = [x for x in v if x is not None and x > 0]
        if v:
            out[key] = statistics.median(v)
    return out


def fmt(v: Any, digits: int = 0) -> str:
    if v is None or v == "":
        return "–"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


# ------------------------------------------------------------------ graphs (plain SVG, no packages)
COLORS = {"A0": "#8c96a8", "A1": "#3b6fd8", "B": "#e07a2e"}


def bar_svg(title: str, groups: list[str], series: list[str], values: dict[tuple[str, str], tuple[float, float, float] | None],
            y_label: str, y_max: float) -> str:
    """Grouped bars; values[(group, series)] = (value, low, high) or None."""
    w, h, left, bottom, top = 760, 360, 64, 56, 40
    plot_w, plot_h = w - left - 20, h - top - bottom
    gw = plot_w / max(1, len(groups))
    bw = gw * 0.7 / max(1, len(series))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
           f'font-family="sans-serif" font-size="12">',
           f'<rect width="{w}" height="{h}" fill="white"/>',
           f'<text x="{w / 2}" y="22" text-anchor="middle" font-size="15" font-weight="bold">{html.escape(title)}</text>']
    for i in range(6):
        y_val = y_max * i / 5
        y = top + plot_h - plot_h * i / 5
        out.append(f'<line x1="{left}" x2="{w - 20}" y1="{y:.1f}" y2="{y:.1f}" stroke="#ddd"/>')
        out.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{y_val:g}</text>')
    out.append(f'<text x="14" y="{top + plot_h / 2}" transform="rotate(-90 14 {top + plot_h / 2})" '
               f'text-anchor="middle">{html.escape(y_label)}</text>')
    for gi, g in enumerate(groups):
        x0 = left + gi * gw + gw * 0.15
        out.append(f'<text x="{left + gi * gw + gw / 2:.1f}" y="{h - bottom + 18}" text-anchor="middle">'
                   f'{html.escape(g)}</text>')
        for si, s in enumerate(series):
            v = values.get((g, s))
            if v is None:
                continue
            val, lo, hi = v
            x = x0 + si * bw
            bh = plot_h * min(val, y_max) / y_max
            out.append(f'<rect x="{x:.1f}" y="{top + plot_h - bh:.1f}" width="{bw * 0.9:.1f}" height="{bh:.1f}" '
                       f'fill="{COLORS.get(s, "#999")}"><title>{html.escape(s)}: {val:g}</title></rect>')
            if hi > lo:
                y_lo, y_hi = top + plot_h - plot_h * min(lo, y_max) / y_max, top + plot_h - plot_h * min(hi, y_max) / y_max
                cx = x + bw * 0.45
                out.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y_lo:.1f}" y2="{y_hi:.1f}" stroke="#222"/>')
            out.append(f'<text x="{x + bw * 0.45:.1f}" y="{top + plot_h - bh - 4:.1f}" text-anchor="middle" '
                       f'font-size="10">{val:.2g}</text>' if y_max <= 1.0 else
                       f'<text x="{x + bw * 0.45:.1f}" y="{top + plot_h - bh - 4:.1f}" text-anchor="middle" '
                       f'font-size="10">{val:.0f}</text>')
    lx = left
    for s in series:
        out.append(f'<rect x="{lx}" y="{h - 22}" width="12" height="12" fill="{COLORS.get(s, "#999")}"/>'
                   f'<text x="{lx + 16}" y="{h - 12}">{html.escape(s)}</text>')
        lx += 70
    out.append("</svg>")
    return "\n".join(out)


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", nargs="?", default="results/phase5/manifest.csv")
    parser.add_argument("--sessions", default="results/sessions", help="folder that holds the session folders")
    args = parser.parse_args(argv)
    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"[FAIL] manifest not found: {manifest} (see the docstring: python -m reachy_s2s.apps.phase5_report -h)")
        return 1
    out_dir = manifest.parent
    runs = load_runs(manifest, Path(args.sessions))
    if not runs:
        print("[FAIL] no sessions in the manifest")
        return 1

    labelled_rows, score_rows, judge_rows = [], [], []
    scores: dict[tuple[str, str, str], dict[str, Any]] = {}
    lat: dict[tuple[str, str], list[Turn]] = {}
    for run in runs:
        chosen = label(run) if run.condition in ("quiet", "noise") else {}  # motion_on: latency only
        lat.setdefault((run.variant, run.condition), []).extend(run.turns)
        for t in run.turns:
            labelled_rows.append({"variant": run.variant, "condition": run.condition, "round": run.round,
                                  "session": t.session, "turn_id": t.row.get("turn_id"), "block": t.block,
                                  "item": t.item, "continuation": t.continuation, "similarity": round(t.sim, 2),
                                  "status": t.row.get("status"), "transcript": t.row.get("transcript"),
                                  "merged_text": t.row.get("merged_text", ""), "response": t.response,
                                  "speech_end_to_push_ms": t.row.get("speech_end_to_push_ms")})
        for task in TASKS if chosen else []:
            if task not in chosen:
                score_rows.append({"variant": run.variant, "condition": run.condition, "round": run.round,
                                   "task": task, "done": False})
                continue
            s = score_task(task, chosen[task])
            scores[(run.variant, run.condition, run.round, task)] = s
            score_rows.append({"variant": run.variant, "condition": run.condition, "round": run.round,
                               "task": task, "done": True, **{k: v for k, v in s.items()}})
            for item, ts in by_item(chosen[task]).items():
                final = ts[-1]
                judge_rows.append({
                    "variant": run.variant, "condition": run.condition, "round": run.round, "item": item,
                    "turns_for_item": len(ts),
                    "understood_text": final.text, "robot_text": final.response,
                    "robot_audio": f"results/sessions/{final.session}/audio/turn_{int(final.row['turn_id']):04d}_robot.wav",
                    "user_audio": f"results/sessions/{final.session}/audio/turn_{int(final.row['turn_id']):04d}_user.wav",
                    "success_0_1": "", "appropriateness_1_5": "" if item.startswith("T1") else "n/a", "note": ""})

    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        keys: list[str] = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with (out_dir / name).open("w", newline="", encoding="utf-8-sig") as fh:  # BOM: Excel opens Korean correctly
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv("turns_labeled.csv", labelled_rows)
    write_csv("task_scores.csv", score_rows)
    write_csv("judgment_sheet.csv", judge_rows)

    # ---- summary tables
    conds = [c for c in CONDITIONS if any(r.condition == c for r in runs)]
    lines = ["# Phase 5 결과 요약", "",
             f"자동 생성: `python -m reachy_s2s.apps.phase5_report {manifest.as_posix()}`. 세션 목록은 `{manifest.name}`.", "",
             "## 1. 응답 지연 (사용자 발화 종료 추정 → 로봇 첫 오디오 전송)", "",
             "스피커에서 실제로 들리기까지는 출력 지연이 더해진다(추정 약 0.4 s, 실측 아님).", "",
             "| 변형 | 조건 | n | 중앙값 (ms) | 25–75 % (ms) | ASR | LLM | TTS | VAD 종료 판정 |", "|---|---|---|---|---|---|---|---|---|"]
    lat_vals: dict[tuple[str, str], tuple[float, float, float] | None] = {}
    for v in VARIANTS:
        for c in conds:
            if (v, c) not in lat:
                continue
            L = latency(lat[(v, c)])
            lat_vals[(COND_TITLES[c], v)] = (L["median"], L["p25"], L["p75"]) if L.get("median") else None
            lines.append(f"| {v} | {COND_TITLES[c]} | {L['n']} | {fmt(L.get('median'))} | "
                         f"{fmt(L.get('p25'))}–{fmt(L.get('p75'))} | {fmt(L.get('asr_ms'))} | {fmt(L.get('llm_ms'))} | "
                         f"{fmt(L.get('tts_ms'))} | {fmt(L.get('vad_endpoint_delay_ms'))} |")
    lines += ["", "단계 시간은 A(Cascaded)만 있다. B는 서버 내부 단계를 알 수 없다.", "",
              "## 2. 과제별 자동 채점 (키워드 기준, 회차별 마지막 시도)", ""]
    header = "| 과제 | 지표 | " + " | ".join(f"{v} {COND_TITLES[c]}" for c in conds for v in VARIANTS) + " |"
    lines += [header, "|" + "---|" * (2 + len(conds) * len(VARIANTS))]
    metrics = [("T1", "correct", "정답 (/3)"), ("T2", "recall_exact", "이름 회상(정확히 '준수')"),
               ("T2", "recall_heard", "이름 회상(들은 이름)"),
               ("T3", "T3-1_premature", "①멈춤에서 끊김"), ("T3", "T3-1_fragment_spoken", "①반쪽에 대답 시작"),
               ("T3", "T3-1_final_ok", "①뒷말까지 이해"), ("T3", "T3-2_premature", "②멈춤에서 끊김"),
               ("T3", "T3-2_fragment_spoken", "②반쪽에 대답 시작"), ("T3", "T3-2_final_ok", "②뒷말까지 이해"),
               ("T4A", "correct_2", "정정 반영(2개)"), ("T4B", "bargein", "끼어들기 멈춤"),
               ("T4B", "stop_ms", "소리 정지 (ms)"), ("T4B", "new_question_ok", "새 질문 답(2)"),
               ("T5", "asr_slots", "인식 슬롯 (/4)"), ("T5", "recall_slots", "응답 슬롯 (/4)"),
               ("CS", "asr_terms", "인식 용어 (/5)"), ("CS", "latin_kept", "영어 철자 유지 (/4)"),
               ("CS", "recall_terms", "응답 용어 (/3)"), ("CS", "english_out", "영어 번역 출력"),
               ("*", "retries", "다시 말한 횟수(과제 합, T3 제외)")]
    for task, key, title in metrics:
        cells = []
        for c in conds:
            for v in VARIANTS:
                vals = []
                for (vv, cc, _r, tt), s in scores.items():
                    if vv == v and cc == c and (tt == task or task == "*") and key in s:
                        vals.append(s[key])
                if not vals:
                    cells.append("–")
                elif task == "*":
                    cells.append(str(sum(vals)))
                elif all(isinstance(x, bool) for x in vals):
                    cells.append(f"{sum(vals)}/{len(vals)}" if len(vals) > 1 else fmt(vals[0]))
                elif all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vals if x is not None):
                    nums = [x for x in vals if x is not None]
                    cells.append(fmt(statistics.mean(nums), 0 if key.endswith("_ms") else 1) if nums else "–")
                else:
                    cells.append(", ".join(str(x) or "–" for x in vals))
        lines.append(f"| {TASK_TITLES.get(task, '전체')} | {title} | " + " | ".join(cells) + " |")
    lines += ["", "✓/✗는 회차가 하나일 때, `k/n`은 여러 회차일 때다. 숫자는 회차 평균이다. 빈칸(–)은 그 과제를 하지 않은 경우다.",
              "A0는 끼어들기가 없으므로 T4B 멈춤이 ✗인 것이 정상이다.",
              "자동 채점은 글자 기준이다. 최종 판정은 `judgment_sheet.csv`에 녹음을 듣고 적는다(B 자막은 실제 음성과 다를 수 있다).", ""]

    # ---- graphs
    groups = [COND_TITLES[c] for c in conds]
    y_max = max([x[2] for x in lat_vals.values() if x] + [1000.0])
    y_max = float(int(y_max / 1000) + 1) * 1000
    (out_dir / "latency.svg").write_text(bar_svg("응답 지연 중앙값 (막대 = 25–75 %)", groups, VARIANTS, lat_vals,
                                                 "ms (발화 종료 → 첫 오디오 전송)", y_max), encoding="utf-8")
    succ_metrics = [("T1", "correct", 3), ("T2", "recall_exact", 1), ("T3", "T3-1_final_ok", 1), ("T4A", "correct_2", 1),
                    ("T4B", "new_question_ok", 1), ("T5", "recall_slots", 4), ("CS", "recall_terms", 3)]
    scored_conds = [c for c in conds if c in ("quiet", "noise")]  # motion_on runs give latency only
    for c in scored_conds:
        vals: dict[tuple[str, str], tuple[float, float, float] | None] = {}
        for task, key, denom in succ_metrics:
            for v in VARIANTS:
                xs = [float(s[key]) / denom for (vv, cc, _r, tt), s in scores.items()
                      if vv == v and cc == c and tt == task and s.get(key) is not None]
                vals[(task, v)] = (statistics.mean(xs), 0.0, 0.0) if xs else None
        (out_dir / f"success_{c}.svg").write_text(
            bar_svg(f"과제 성공률 — {COND_TITLES[c]} (자동 채점)", [t for t, _k, _d in succ_metrics],
                    VARIANTS, vals, "성공 비율 (0–1)", 1.0), encoding="utf-8")
    lines += ["## 3. 그래프", "", "![응답 지연](latency.svg)", ""]
    lines += [f"![과제 성공률 {COND_TITLES[c]}](success_{c}.svg)" for c in scored_conds]
    lines += ["", "성공률 그래프의 지표: T1 정답/3, T2 이름 회상(정확), T3 ①뒷말까지 이해, T4A 정정 반영, T4B 새 질문 답, T5 응답 슬롯/4, CS 응답 용어/3."]
    lines += ["", "## 4. 세션", "", "| 변형 | 조건 | 회차 | 세션 |", "|---|---|---|---|"]
    lines += [f"| {r.variant} | {COND_TITLES.get(r.condition, r.condition)} | {r.round} | {', '.join(sorted(r.sessions))} |"
              for r in runs]
    summary = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")

    # ---- one page for the browser (tables + graphs)
    body: list[str] = []
    in_table = False
    for line in lines:
        if line.startswith("|"):
            cells = [html.escape(x.strip()) for x in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-") for c in cells):
                continue  # markdown separator row
            tag = "td" if in_table else "th"
            if not in_table:
                body.append("<table>")
                in_table = True
            body.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table>")
            in_table = False
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("![") and "](" in line:
            body.append((out_dir / line.split("](", 1)[1].rstrip(")")).read_text(encoding="utf-8"))
        elif line.strip():
            body.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        body.append("</table>")
    page = ("<!doctype html><meta charset='utf-8'><title>Phase 5 결과</title><style>"
            "body{font-family:sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}"
            "table{border-collapse:collapse;margin:8px 0 16px}td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}"
            "th{background:#f3f3f3}svg{max-width:100%;height:auto;margin:8px 0}</style>" + "\n".join(body))
    (out_dir / "report.html").write_text(page, encoding="utf-8")
    print(summary)
    print(f"[OK] wrote {out_dir / 'report.html'}, summary.md, task_scores.csv, turns_labeled.csv, judgment_sheet.csv, *.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
