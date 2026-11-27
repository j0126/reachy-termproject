"use strict";
// 청중용 "Reachy의 생각" 화면 (프로젝터). /ws/operator의 스냅숏·상태·로그를 읽기만 한다.
(() => {
  const HUMAN = { ABSENT: "아무도 없음", PRESENT: "손님 발견", ENGAGED: "주문 중", NO_RESPONSE: "반응 없음",
                  AMBIGUOUS: "여러 명", DETECTION_UNCERTAIN: "카메라 불확실" };
  const REASON = { hesitation: "망설이는 중", seek_help: "로봇을 쳐다봄 → 도움 요청" };
  const ZONE = { none: "-", far: "멀리", social: "사회적 거리 (~2.5 m)", personal: "개인 거리 (~1 m)" };
  const ATTN = { none: "-", robot: "로봇을 봄", screen: "화면을 봄", away: "딴 곳을 봄" };
  const ACTIVITY = { idle: "대기 중", listening: "듣는 중", thinking: "생각 중", speaking: "말하는 중" };
  const GAZE = { user: "손님", screen: "화면", slot: "카드 슬롯", wander: "둘러보기", still: "고정" };
  const LED = { off: "꺼짐", on: "켜짐", blink: "깜빡임" };
  const STEP_OF = {
    "speech.start": "sense", "speech.utterance": "sense", touch: "sense", card: "sense", "human.cues": "sense",
    nlu: "interp", "human.state": "interp",
    "fsm.state": "decide", "rec.suggest": "decide", "rec.reject": "decide", "order.item": "decide", "cart.add": "decide",
    act: "act", say: "act", "robot.motion": "act", "robot.gesture": "act", "robot.glance": "act",
  };

  const $ = (id) => document.getElementById(id);
  let ws = null, retry = 0, faces = [], primary = null, observeTimer = null;

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/operator`);
    ws.onopen = () => { retry = 0; $("conn").style.display = "none"; };
    ws.onclose = () => { $("conn").style.display = "block"; setTimeout(connect, Math.min(5000, 500 * 2 ** retry++)); };
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "snapshot") snapshot(msg);
      else if (msg.type === "status") status(msg);
      else if (msg.type === "log") record(msg.record);
    };
  }

  function light(step) {
    document.querySelectorAll(".step").forEach((el) => el.classList.toggle("on", el.dataset.step === step));
    clearTimeout(observeTimer);
    if (step === "act") observeTimer = setTimeout(() => light("observe"), 2500);
  }

  function pulse(id) {
    const el = $(id);
    el.classList.remove("pulse");
    void el.offsetWidth;
    el.classList.add("pulse");
  }

  function snapshot(s) {
    $("state").textContent = s.state || "-";
    $("overlay").textContent = s.overlay && s.overlay !== "NONE" ? `오버레이 ${s.overlay}` : "";
    $("overlay").style.display = $("overlay").textContent ? "" : "none";
    $("lang").textContent = s.lang === "en" ? "English" : "한국어";
    const f = s.features || {};
    const condB = f.motion && f.gaze && f.status_strip && f.slot_led;
    $("condition").textContent = condB ? "조건 B · Reachy Café" : "조건 A · Voice + Screen";
    const tr = s.trial || {};
    $("trial").textContent = tr.participant ? `${tr.participant} · ${tr.task}` : "";
    $("trial").style.display = tr.participant ? "" : "none";
    const p = s.perception || {};
    faces = p.faces || [];
    primary = p.primary;
    $("human").textContent = HUMAN[s.human] || s.human || "-";
    $("humanSub").textContent = s.human === "NO_RESPONSE" ? (REASON[s.human_reason] || "") : `${faces.length}명 감지`;
    const cues = s.cues || {};
    $("zone").textContent = `${ZONE[cues.zone] || "-"}${cues.approaching ? " · 다가오는 중 ↗" : ""}`;
    $("attention").textContent = ATTN[cues.attention] || "-";
    $("score").textContent = p.score ?? "-";
    const sess = s.session;
    if (sess) {
      const lines = (sess.cart.lines || []).map((l) => `${l.emoji}${l.name.ko} ×${l.qty}`);
      $("cart").textContent = lines.length ? `${lines.join(", ")} · ${sess.cart.total.toLocaleString()}원` : "(비어 있음)";
      const pr = sess.prefs || {};
      const names = { caffeine_free: "카페인 없음", less_sweet: "덜 달게", refreshing: "상큼", milky: "부드럽게", hot: "따뜻", ice: "시원" };
      const on = Object.keys(names).filter((k) => pr[k]).map((k) => names[k]);
      $("prefs").textContent = `${on.join(" + ") || "-"}${pr.updated_by === "too_sweet" ? " (↻ '너무 달아요' 반영)" : ""}`;
      $("session").textContent = `#${sess.id} · ${sess.elapsed_s ?? "-"}초`;
    } else {
      $("cart").textContent = "-"; $("prefs").textContent = "-"; $("session").textContent = "(손님 없음)";
    }
    const r = s.robot || {};
    if (r.gaze) $("gaze").textContent = GAZE[r.gaze] || r.gaze;
    draw();
  }

  function status(st) {
    $("heard").textContent = st.heard || "";
    $("said").textContent = st.said || "";
    $("plan").textContent = st.plan || "-";
    $("activity").textContent = ACTIVITY[st.activity] || st.activity || "-";
    $("led").textContent = LED[st.led] || st.led || "-";
    if (st.activity === "listening") light("observe");
  }

  function record(rec) {
    const step = STEP_OF[rec.kind];
    if (step) light(step);
    if (rec.kind === "nlu") {
      const slots = ["item", "qty", "temp", "size", "reason"].filter((k) => rec[k] != null).map((k) => `${k}=${rec[k]}`);
      if (rec.prefs) slots.push(`prefs=${JSON.stringify(rec.prefs)}`);
      $("intent").textContent = `의도: ${rec.intent}(${slots.join(", ")})`;
      pulse("cardTalk");
    } else if (rec.kind === "fsm.state") {
      pulse("cardDecide");
    } else if (rec.kind === "robot.motion") {
      $("motion").textContent = rec.motion;
      pulse("cardAct");
    } else if (rec.kind === "robot.motion_label") {
      $("motion").textContent = `${rec.motion} · ${rec.label}`;
    } else if (rec.kind === "robot.motion_done" || rec.kind === "robot.motion_cancel") {
      $("motion").textContent = `${rec.motion} (끝)`;
    } else if (rec.kind === "robot.gesture") {
      $("gesture").textContent = rec.gesture;
      pulse("cardAct");
    } else if (rec.kind === "robot.glance") {
      $("gaze").textContent = `${GAZE[rec.target] || rec.target} 힐끗 → 손님`;
    } else if (rec.kind === "robot.gaze") {
      $("gaze").textContent = GAZE[rec.target] || rec.target;
    } else if (rec.kind === "human.state" || rec.kind === "human.cues") {
      pulse("cardSee");
    }
  }

  function draw() {
    const cv = $("cv");
    const ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    ctx.fillStyle = "#0b0806";
    ctx.fillRect(0, 0, W, H);
    for (const f of faces) {
      const size = Math.max(12, f.h * H * 1.15);
      const x = f.cx * W - size * 0.4, y = f.cy * H - size / 2;
      ctx.strokeStyle = f.id === primary ? "#e8772e" : "#8b7d70";
      ctx.lineWidth = f.id === primary ? 5 : 3;
      ctx.strokeRect(x, y, size * 0.8, size);
      if (f.yaw != null) {  // 고개 방향 화살표
        ctx.beginPath();
        ctx.moveTo(f.cx * W, f.cy * H);
        ctx.lineTo(f.cx * W + Math.sin(f.yaw * Math.PI / 180) * size * 0.7, f.cy * H + (f.pitch || 0) * 0.6);
        ctx.stroke();
      }
    }
  }

  light("observe");
  connect();
})();
