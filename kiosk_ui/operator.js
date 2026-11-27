"use strict";
// 오퍼레이터 패널: 개발 중에는 로봇 대신 얼굴·발화·카드를 흉내 내고, 라이브 데모에서는 인식이 실패했을 때 시나리오를 이어 준다.
(() => {
  const FACES = [
    ["none", "없음"], ["approach", "다가옴"], ["passing", "지나감"], ["one_far", "1명 멀리"],
    ["one_near", "가까이·로봇 봄"], ["one_near_screen", "가까이·화면 봄"], ["one_away", "가까이·딴 곳 봄"],
    ["two_near", "2명"], ["stale", "프레임 끊김"], ["camera", "카메라"],
  ];
  const PHRASES = {
    "S1 전체 주문 (한국어)": ["따뜻한 라떼 하나 주세요", "레귤러요", "네"],
    "S2 추천 (영어)": ["Yes, something without coffee.", "Hmm, too sweet.", "Yes, I'll take it", "iced, regular please", "Yes"],
    "S3 품절 + 직원 안내": ["아인슈페너 주세요", "네", "바닐라 라떼로 할게요", "혹시 견과류 들어가요?", "아이스 레귤러요", "네"],
    "메뉴 설명": ["아인슈페너가 뭐예요?", "What is Einspänner?", "바닐라 라떼는 무슨 맛이에요?", "네"],
    "실패·기타": ["바닐", "음음", "다시 말해 주세요", "아이스로 바꿔 주세요", "라떼 빼주세요", "추천해 주세요", "취소할게요"],
    "자연스러운 말": ["라떼 하나랑 아이스 아메리카노 두 잔", "아메리카노 말고 라떼요", "그거 주세요", "첫 번째 걸로",
                 "얼음 적게 해 주세요", "포장이요", "라떼는 아이스로 바꿔 주세요", "라떼 하나 더"],
    "직원 안내 (규칙 밖)": ["직원 불러 주세요", "견과류 들어가요?", "현금으로 해도 돼요?", "화장실 어디예요?", "환불하고 싶어요"],
  };
  const STAFF_REASONS = { request: "손님이 직원을 찾음", allergy: "알레르기·성분 문의", payment: "카드 외 결제 (현금·쿠폰)",
                          refund: "환불·교환·잘못 나온 음료", facility: "화장실·와이파이 등 매장 안내",
                          repair: "세 번 연속 못 알아들음" };
  // 직원 역할(오퍼레이터)이 할 일. 손님 쪽을 보며 한 줄로 답하고 "응대 완료" → 로봇이 하던 질문으로 이어 간다
  const STAFF_HINTS = {
    request: "손님에게 무엇이 필요한지 묻고 도운 뒤 응대 완료",
    allergy: "S3 대사: \"확인해 드릴게요. 바닐라 라떼에는 견과류가 들어가지 않아요.\" → 응대 완료",
    payment: "\"현금은 카운터에서 받아 드릴게요. 카드로 하셔도 돼요.\" → 응대 완료",
    refund: "주문번호를 확인해 처리하겠다고 답한 뒤 응대 완료",
    facility: "위치·비밀번호를 알려 준 뒤 응대 완료",
    repair: "손님 옆에서 화면으로 주문을 도와준 뒤 응대 완료",
  };
  const STATES = ["IDLE", "GREET", "MENU", "OFFER_HELP", "RECOMMEND", "SOLD_OUT", "OPTIONS", "CART",
                  "PAY_WAIT_CARD", "PAY_PROCESSING", "PAY_REMOVE_CARD", "DONE"];
  const FEATURES = [["motion", "모션"], ["gaze", "시선"], ["status_strip", "상태 스트립"], ["slot_led", "슬롯 LED"]];
  const IMPROVEMENTS = [["invite", "호객·접근 의도"], ["attention", "주의 방향"], ["backchannel", "백채널"],
                        ["context_talk", "날씨 인사"], ["auto_staff", "반복 실패 시 직원 호출"]];

  let ws = null;
  let retry = 0;
  let snapshot = {};

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/operator`);
    ws.onopen = () => { retry = 0; setConn(true); };
    ws.onclose = () => { setConn(false); setTimeout(connect, Math.min(5000, 500 * 2 ** retry++)); };
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "snapshot") { snapshot = msg; renderSnapshot(); }
      else if (msg.type === "log") appendLog(msg.record);
      else if (msg.type === "log_tail") { $("log").innerHTML = ""; (msg.records || []).forEach(appendLog); }
    };
  }

  function setConn(ok) {
    const el = $("conn");
    el.textContent = ok ? "연결됨" : "연결 끊김";
    el.className = `pill ${ok ? "ok" : "bad"}`;
  }

  function op(action, extra = {}) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "operator", action, ...extra }));
  }

  // ---- 정적 버튼 만들기 ---------------------------------------------------------
  $("faces").innerHTML = FACES.map(([id, label]) => `<button data-op="faces" data-preset="${id}">${esc(label)}</button>`).join("");
  $("phrases").innerHTML = Object.entries(PHRASES).map(([group, items]) =>
    `<h3>${esc(group)}</h3><div class="btns">${items.map((t) => `<button data-say="${esc(t)}">${esc(t)}</button>`).join("")}</div>`).join("");
  $("forceSel").innerHTML = STATES.map((s) => `<option>${s}</option>`).join("");

  document.addEventListener("click", (event) => {
    const el = event.target.closest("button");
    if (!el) return;
    const d = el.dataset;
    if (d.op === "faces") op("faces", { preset: d.preset });
    else if (d.op === "card") op("card", { state: d.state });
    else if (d.op === "condition") op("condition", { condition: d.condition });
    else if (d.op === "neutral") op("neutral");
    else if (d.op === "feature") op("feature", { key: d.key, value: d.value === "1" });
    else if (d.op === "context") op("context", { weather: d.weather });
    else if (d.say) op("say", { text: d.say });
    else if (d.touch) op("touch", { touch: d.touch, params: d.lang ? { lang: d.lang } : {} });
  });
  document.addEventListener("change", (event) => {
    const el = event.target;
    if (el.dataset && el.dataset.stock) op("stock", { item: el.dataset.stock, in_stock: el.checked });
  });
  $("sayBtn").onclick = () => { const t = $("sayText").value.trim(); if (t) { op("say", { text: t }); $("sayText").value = ""; } };
  $("sayText").addEventListener("keydown", (e) => { if (e.key === "Enter") $("sayBtn").click(); });
  $("bargeBtn").onclick = () => op("speech_start");
  $("staffDoneBtn").onclick = () => op("staff_done");
  $("resetBtn").onclick = () => op("reset");
  $("forceBtn").onclick = () => op("force_state", { state: $("forceSel").value });
  $("trialBtn").onclick = () => op("trial", { participant: $("trialP").value, condition: $("trialC").value,
                                             task: $("trialT").value });

  // ---- 스냅숏 ---------------------------------------------------------------
  function renderSnapshot() {
    const s = snapshot;
    $("state").textContent = s.state || "-";
    $("overlay").textContent = s.overlay || "-";
    const p = s.perception || {};
    const cues = s.cues || {};
    $("human").textContent = `${s.human || "-"}${s.human_reason && s.human === "NO_RESPONSE" ? `(${s.human_reason})` : ""}` +
      ` · 얼굴 ${(p.faces || []).length}개 · ${cues.zone || "-"}${cues.approaching ? " · 다가옴" : ""} · 주의 ${cues.attention || "-"}` +
      `${p.gaze_off_user ? " · 시선 이동 중" : ""}`;
    const r = s.robot || {};
    $("robot").textContent = `로봇 ${r.label || "-"} (${r.host || "-"})`;
    $("robotState").textContent = `시선 ${r.gaze || "-"} · 자세 ${r.posture || "-"}${r.speaking ? " · 말하는 중" : ""}`;
    // 실제 로봇: 채널별 상태 (하나가 죽어도 나머지로 계속 진행하므로 어느 것이 죽었는지 바로 보이게)
    const h = r.health;
    if (h) {
      const mark = (v) => (typeof v === "string" && v.startsWith("ok") ? "✓" : "✗");
      const extra = [];
      if (h.asr_ready === false) extra.push("ASR 로딩 중");
      if (h.tts_ready === false) extra.push("TTS 로딩 중");
      if (h.camera_fps) extra.push(`카메라 ${h.camera_fps} fps`);
      $("robotHealth").textContent = `모션 ${mark(h.motion)} · 음성 ${mark(h.speech)} · 카메라 ${mark(h.camera)}`
        + (extra.length ? ` (${extra.join(", ")})` : "");
      $("robotHealth").title = JSON.stringify(h);
    } else {
      $("robotHealth").textContent = "mock";
    }
    const c = s.clients || {};
    $("clients").textContent = `화면 ${c.ui ?? 0} · 카드 슬롯 ${c.device ?? 0}`;
    $("lang").textContent = s.lang || "-";
    $("cardIn").textContent = s.card_in ? "꽂혀 있음" : "없음";
    const sess = s.session;
    // 직원 호출 알림: 직원 역할(오퍼레이터)이 손님에게 가고, 끝나면 "응대 완료"
    const staff = sess && sess.staff && sess.staff.called ? sess.staff : null;
    $("staffAlert").hidden = !staff;
    if (staff) {
      $("staffReason").textContent = `${STAFF_REASONS[staff.reason] || staff.reason} · ${s.state}`;
      $("staffHint").textContent = STAFF_HINTS[staff.reason] || "";
    }
    if (sess) {
      const lines = (sess.cart.lines || []).map((l) =>
        `${l.name.ko} ${l.temp || "?"}/${l.size || "?"} ×${l.qty}${(l.notes || []).length ? ` [${l.notes.join(", ")}]` : ""}`);
      $("cart").textContent = (lines.length ? `${lines.join(", ")} = ${sess.cart.total.toLocaleString()}원` : "(비어 있음)")
        + (sess.dine ? ` · ${sess.dine === "togo" ? "포장" : "매장"}` : "") + ((sess.queue || []).length ? ` · 대기 ${sess.queue.join(", ")}` : "");
      const pr = sess.prefs || {};
      const on = Object.entries(pr).filter(([k, v]) => v === true).map(([k]) => k);
      $("prefs").textContent = `${on.join(", ") || "-"}${pr.updated_by ? ` (↻ ${pr.updated_by})` : ""}${sess.rec_item ? ` → 추천 ${sess.rec_item}` : ""}`;
      $("session").textContent = `#${sess.id} · ${sess.elapsed_s ?? "-"}s · 도움 제안 ${sess.help_offered ? "함" : "안 함"} · 재질문 ${sess.repair_count}`;
    } else {
      $("cart").textContent = "-"; $("prefs").textContent = "-"; $("session").textContent = "(세션 없음)";
    }
    document.querySelectorAll("[data-op=faces]").forEach((b) => b.classList.toggle("on", b.dataset.preset === s.sim_preset));
    const f = s.features || {};
    const toggle = ([k, label]) =>
      `<button data-op="feature" data-key="${k}" data-value="${f[k] ? "0" : "1"}" class="${f[k] ? "on" : ""}">${esc(label)} ${f[k] ? "켬" : "끔"}</button>`;
    $("features").innerHTML = FEATURES.map(toggle).join("");
    $("improvements").innerHTML = IMPROVEMENTS.map(toggle).join("");
    const weather = (s.context || {}).weather || "normal";
    document.querySelectorAll("[data-op=context]").forEach((b) => b.classList.toggle("on", b.dataset.weather === weather));
    const tr = s.trial || {};
    $("trialNow").textContent = tr.participant ? `기록 중: ${tr.participant} · 조건 ${tr.condition} · ${tr.task}` : "기록 중인 시행 없음";
    const stock = s.stock || {};
    const box = $("stock");
    if (box.childElementCount !== Object.keys(stock).length) {
      box.innerHTML = Object.keys(stock).map((id) =>
        `<label><input type="checkbox" data-stock="${esc(id)}"> ${esc(id)}</label>`).join("");
    }
    box.querySelectorAll("input[data-stock]").forEach((el) => { el.checked = !!stock[el.dataset.stock]; });
  }

  // ---- 로그 -----------------------------------------------------------------
  function appendLog(record) {
    const log = $("log");
    const kind = record.kind || "";
    const cls = kind.startsWith("fsm") ? "k-fsm" : kind.startsWith("robot") ? "k-robot" : kind.startsWith("human") ? "k-human"
      : `k-${kind.split(".")[0]}`;
    const { t, kind: _k, ...rest } = record;
    const line = document.createElement("div");
    line.className = cls;
    line.textContent = `${String(t).padStart(8)}  ${kind}  ${JSON.stringify(rest)}`;
    const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
    log.appendChild(line);
    while (log.childElementCount > 400) log.removeChild(log.firstChild);
    if (atBottom) log.scrollTop = log.scrollHeight;
  }

  connect();
})();
