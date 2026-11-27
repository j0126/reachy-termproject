"use strict";
// Reachy Café 키오스크 화면. 두뇌(PC)가 보내는 view·status를 그대로 그리고, 터치는 touch 이벤트로 보낸다.
// 화면 상태의 주인은 서버(FSM)다. 여기서는 메뉴 탭 같은 화면 내부 상태만 가진다.
(() => {
  const T = {
    ko: {
      activity: { idle: "대기 중", listening: "듣고 있어요", thinking: "생각 중", speaking: "말하는 중" },
      human: { ABSENT: "아무도 없음", PRESENT: "손님 발견", ENGAGED: "주문 중", NO_RESPONSE: "망설이는 중",
               AMBIGUOUS: "여러 명", DETECTION_UNCERTAIN: "카메라 확인 중" },
      heard: "들은 말", steps: ["메뉴", "옵션", "확인", "결제", "완료"], people: (n) => `${n}명`,
      attractTitle: "Reachy Café", attractSub: "말을 걸거나 화면을 눌러 주문하세요", start: "주문 시작",
      welcomeTitle: "안녕하세요!", welcomeSub: "말로 주문하거나 화면을 눌러 주세요",
      recommendBtn: "🤖 추천받기", cartBtn: (n, total) => `🛒 ${n}잔 · ${total}`,
      soldOut: "품절", caffeine: "카페인", noCaffeine: "카페인 없음", sweet: "당도",
      recTitle: "Reachy의 추천", recAsk: "원하는 조건을 말하거나 눌러 주세요", thinkTitle: "Reachy 생각",
      accept: "이걸로 할게요", another: "다른 거 추천", back: "메뉴로",
      prefs: { caffeine_free: "카페인 없이", less_sweet: "덜 달게", refreshing: "상큼하게", milky: "부드럽게",
               hot: "따뜻하게", ice: "시원하게" },
      updated: { too_sweet: "방금 '너무 달아요'를 반영했어요", too_bitter: "방금 '너무 써요'를 반영했어요",
                 rejected: "다른 메뉴를 다시 찾았어요" },
      reasons: { caffeine_free: "카페인 없음", less_sweet: "덜 달아요", not_sweet: "달지 않아요",
                 refreshing: "상큼해요", creamy: "부드러워요", popular: "인기 메뉴" },
      soldTitle: (name) => `${name} 품절`, similar: "비슷한 메뉴는 어떠세요?", choose: "이걸로",
      temp: "온도", hot: "따뜻하게", ice: "아이스", size: "사이즈", regular: "레귤러", large: "라지",
      addToCart: "담기", cartTitle: "주문 확인", total: "합계", pay: "결제하기", more: "더 주문하기",
      qty: (n) => `${n}잔`,
      payInsert: "카드를 아래 투입구에 넣어 주세요", payProcessing: "결제 중이에요", payRemove: "카드를 빼 주세요",
      payFallback: "화면으로 결제 (데모)", doneLabel: "주문번호", thanks: "감사합니다! 음료가 준비되면 불러 드릴게요",
      pausedWait: "잠시 기다릴게요", pausedSub: "돌아오시면 이어서 주문할 수 있어요",
      resumeQ: "주문을 이어서 하시겠어요?", yes: "네, 이어서", no: "아니요, 취소",
      oneTitle: "한 분씩 도와드릴게요", oneSub: "주문하실 분이 시작을 눌러 주세요",
      candTitle: "혹시 이 메뉴인가요?", sayAgain: "다시 말할게요",
      orderThis: "이걸로 할게요", close: "닫기",
      attention: { robot: "로봇 봄", screen: "화면 봄", away: "딴 곳 봄" },
      approaching: "다가오는 중",
      won: (n) => `${Number(n || 0).toLocaleString("ko-KR")}원`,
      staffBtn: "🙋 직원 호출", staffCalled: "🙋 직원을 불렀어요 · 잠시만 기다려 주세요",
      staffSub: "기다리는 동안 화면으로 계속 주문하셔도 돼요", otherLang: "English",
      dine: { togo: "포장", here: "매장" },
      notes: { less_ice: "얼음 적게", no_ice: "얼음 없이", extra_ice: "얼음 많이", extra_shot: "샷 추가", strong: "진하게",
               weak: "연하게", no_syrup: "시럽 없이", extra_syrup: "시럽 추가", no_whip: "휘핑 없이", extra_whip: "휘핑 추가",
               oat_milk: "오트 밀크", soy_milk: "두유" },
    },
    en: {
      activity: { idle: "Idle", listening: "Listening", thinking: "Thinking", speaking: "Speaking" },
      human: { ABSENT: "Nobody", PRESENT: "Someone's here", ENGAGED: "Ordering", NO_RESPONSE: "Hesitating",
               AMBIGUOUS: "Several people", DETECTION_UNCERTAIN: "Checking camera" },
      heard: "You said", steps: ["Menu", "Options", "Review", "Pay", "Done"],
      people: (n) => `${n} ${n === 1 ? "person" : "people"}`,
      attractTitle: "Reachy Café", attractSub: "Talk to me or tap the screen to order", start: "Start order",
      welcomeTitle: "Hello!", welcomeSub: "Order by voice or tap the screen",
      recommendBtn: "🤖 Recommend", cartBtn: (n, total) => `🛒 ${n} · ${total}`,
      soldOut: "Sold out", caffeine: "Caffeine", noCaffeine: "No caffeine", sweet: "Sweet",
      recTitle: "Reachy recommends", recAsk: "Say or tap what you'd like", thinkTitle: "Reachy is thinking",
      accept: "I'll take it", another: "Something else", back: "Back to menu",
      prefs: { caffeine_free: "No caffeine", less_sweet: "Less sweet", refreshing: "Refreshing", milky: "Creamy",
               hot: "Hot", ice: "Iced" },
      updated: { too_sweet: "Updated after \"too sweet\"", too_bitter: "Updated after \"too bitter\"",
                 rejected: "Found another one" },
      reasons: { caffeine_free: "Caffeine-free", less_sweet: "Less sweet", not_sweet: "Not sweet",
                 refreshing: "Refreshing", creamy: "Creamy", popular: "Popular" },
      soldTitle: (name) => `${name} is sold out`, similar: "How about something similar?", choose: "Choose",
      temp: "Temperature", hot: "Hot", ice: "Iced", size: "Size", regular: "Regular", large: "Large",
      addToCart: "Add", cartTitle: "Your order", total: "Total", pay: "Pay", more: "Order more",
      qty: (n) => `× ${n}`,
      payInsert: "Please insert your card below", payProcessing: "Processing payment", payRemove: "Please take your card",
      payFallback: "Pay on screen (demo)", doneLabel: "Order number", thanks: "Thank you! We'll call you when it's ready",
      pausedWait: "I'll wait for you", pausedSub: "Come back to continue your order",
      resumeQ: "Continue your order?", yes: "Yes, continue", no: "No, cancel",
      oneTitle: "One person at a time", oneSub: "Whoever is ordering, please tap Start",
      candTitle: "Did you mean one of these?", sayAgain: "I'll say it again",
      orderThis: "I'll have this", close: "Close",
      attention: { robot: "looking at me", screen: "reading the screen", away: "looking away" },
      approaching: "approaching",
      won: (n) => `₩${Number(n || 0).toLocaleString("en-US")}`,
      staffBtn: "🙋 Call staff", staffCalled: "🙋 A staff member is on the way",
      staffSub: "You can keep ordering on the screen meanwhile", otherLang: "한국어",
      dine: { togo: "To go", here: "For here" },
      notes: { less_ice: "less ice", no_ice: "no ice", extra_ice: "extra ice", extra_shot: "extra shot", strong: "stronger",
               weak: "weaker", no_syrup: "no syrup", extra_syrup: "extra syrup", no_whip: "no whip", extra_whip: "extra whip",
               oat_milk: "oat milk", soy_milk: "soy milk" },
    },
  };

  let view = { screen: "attract", data: {}, overlay: null, lang: "ko", step: -1, features: { status_strip: true } };
  let status = {};
  let category = "coffee";
  let ws = null;
  let retry = 0;

  // Pi 키오스크(start_kiosk.sh가 ?device=pi로 연다)에서는 커서를 숨긴다. Pi는 터치를 마우스로 바꿔 보내서
  // 누른 자리에 커서가 남는다. 노트북 미리보기에서는 그대로 보인다. 서버에도 ?device=pi로 붙어서,
  // start_kiosk.sh가 /healthz의 pi_kiosk로 페이지가 떴는지(흰 화면이 아닌지) 확인한다.
  const ON_PI = new URLSearchParams(location.search).get("device") === "pi";
  if (ON_PI) document.documentElement.classList.add("on-pi");

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const L = () => T[view.lang] || T.ko;
  const nm = (name) => esc((name && (name[view.lang] || name.ko)) || "");
  const price = (n) => L().won(n);

  // ---- 연결 ----------------------------------------------------------------
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/ui${ON_PI ? "?device=pi" : ""}`);
    ws.onopen = () => { retry = 0; $("conn").classList.add("hidden"); };
    ws.onclose = () => {
      $("conn").classList.remove("hidden");
      setTimeout(connect, Math.min(5000, 500 * 2 ** retry++));
    };
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "view") { view = msg; render(); }
      else if (msg.type === "status") { status = msg; renderStatus(); }
    };
  }

  // 오퍼레이터 패널 안의 미리보기(iframe)에서 누른 터치는 표시해서 보낸다: 평가 집계에서 참가자 터치와 구분 (assisted)
  const SOURCE = window.self !== window.top ? { source: "preview" } : {};

  function send(action, extra = {}) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "touch", action, ...extra, ...SOURCE }));
  }

  document.addEventListener("click", (event) => {
    const el = event.target.closest("[data-action],[data-local]");
    if (!el || el.disabled) return;
    if (el.dataset.local === "category") { category = el.dataset.value; render(); return; }
    const extra = {};
    for (const key of ["item", "key", "value", "lang"]) if (el.dataset[key] !== undefined) extra[key] = el.dataset[key];
    if (el.dataset.index !== undefined) extra.index = Number(el.dataset.index);
    send(el.dataset.action, extra);
    // ✕는 한 번만: 서버가 장바구니를 다시 그리기 전에 두 번째로 누르면 다음 줄이 지워지던 문제 (서버도 줄 표식을 확인)
    if (el.dataset.action === "remove") el.disabled = true;
  });

  // ---- 상태 스트립 ------------------------------------------------------------
  function renderStatus() {
    const t = L();
    const activity = status.activity || "idle";
    $("robotFace").className = `robot-face ${activity}`;
    const label = $("activity");
    label.className = `activity ${activity}`;
    label.textContent = t.activity[activity] || activity;
    $("plan").textContent = status.plan || "";
    $("heardLabel").textContent = t.heard;
    $("heard").textContent = status.heard || "";
    $("said").textContent = status.said || "";
    const cue = status.approaching ? t.approaching : (t.attention[status.attention] || "");
    $("visionLabel").textContent = [t.people(status.faces || 0), t.human[status.human] || status.human || "", cue]
      .filter(Boolean).join(" · ");
    drawVision();
  }

  function drawVision() {
    const canvas = $("visionCanvas");
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = "#0d0a08";
    ctx.fillRect(0, 0, W, H);
    if (status.human === "DETECTION_UNCERTAIN") {
      ctx.fillStyle = "#c9b8a8";
      ctx.font = "bold 40px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("?", W / 2, H / 2 + 14);
      return;
    }
    for (const box of status.face_boxes || []) {
      const size = Math.max(10, box.h * H * 1.15);
      const x = box.cx * W - size * 0.4, y = box.cy * H - size / 2;
      ctx.strokeStyle = box.primary ? "#e8772e" : "#8b7d70";
      ctx.lineWidth = box.primary ? 4 : 2;
      ctx.strokeRect(x, y, size * 0.8, size);
    }
  }

  function renderSteps() {
    const steps = $("steps");
    if (view.step < 0) { steps.classList.add("hidden"); return; }
    steps.classList.remove("hidden");
    steps.innerHTML = L().steps.map((name, i) =>
      `<li class="${i < view.step ? "done" : i === view.step ? "now" : ""}">${esc(name)}</li>`).join("");
  }

  // ---- 화면 ----------------------------------------------------------------
  const screens = {
    attract: () => hero(L().attractTitle, L().attractSub, "☕"),
    welcome: () => hero(L().welcomeTitle, L().welcomeSub, "👋"),
    menu: renderMenu,
    recommend: renderRecommend,
    sold_out: renderSoldOut,
    options: renderOptions,
    cart: renderCart,
    payment: renderPayment,
    done: renderDone,
  };

  function hero(title, sub, logo) {
    const t = L();
    return `<div class="hero"><div class="logo">${logo}</div><h1 class="title">${esc(title)}</h1>
      <p class="subtitle">${esc(sub)}</p>
      <button class="btn primary block" data-action="start">${esc(t.start)}</button>
      <div class="lang-row">
        <button class="btn ${view.lang === "ko" ? "on" : ""}" data-action="lang" data-lang="ko">한국어</button>
        <button class="btn ${view.lang === "en" ? "on" : ""}" data-action="lang" data-lang="en">English</button>
      </div></div>`;
  }

  function badges(item) {
    const t = L();
    const dots = "●".repeat(item.sweetness) + "○".repeat(3 - item.sweetness);
    return `<div class="badges"><span>${esc(t.sweet)} <span class="dots">${dots}</span></span>
      <span>${esc(item.caffeine ? t.caffeine : t.noCaffeine)}</span></div>`;
  }

  // 그림과 가격(품절이면 품절 표시)을 한 줄에 둔다: 720×1280(Pi 터치 화면)에서 메뉴 세 줄이 스크롤 없이 다 보이게.
  // Pi는 터치를 마우스 클릭으로 보내서 손가락으로 밀어도 스크롤되지 않는다.
  function card(item, action) {
    const t = L();
    const sold = !item.in_stock;
    const corner = sold ? `<span class="tag-soldout">${esc(t.soldOut)}</span>` : `<span class="price">${price(item.price)}</span>`;
    return `<button class="card ${sold ? "soldout" : ""}" data-action="${action}" data-item="${esc(item.id)}" ${sold ? "disabled" : ""}>
      <span class="card-top"><span class="emoji">${item.emoji}</span>${corner}</span>
      <span class="name">${nm(item.name)}</span>${badges(item)}</button>`;
  }

  function renderMenu(d) {
    const t = L();
    const menu = d.menu || { categories: [], items: [] };
    const tabs = menu.categories.map((c) =>
      `<button class="tab ${c.id === category ? "on" : ""}" data-local="category" data-value="${esc(c.id)}">${nm(c.name)}</button>`).join("");
    const items = menu.items.filter((item) => item.category === category).map((item) => card(item, "select_item")).join("");
    const cart = d.cart && d.cart.count > 0
      ? `<button class="btn primary" data-action="cart">${esc(t.cartBtn(d.cart.count, price(d.cart.total)))}</button>` : "";
    // 주문 도중에도 언어를 바꿀 수 있게 (예전에는 첫 화면에만 언어 버튼이 있었다)
    const lang = `<button class="tab lang" data-action="lang" data-lang="${view.lang === "ko" ? "en" : "ko"}">${esc(t.otherLang)}</button>`;
    return `<div class="tabs">${tabs}${lang}</div><div class="scroll"><div class="grid">${items}</div></div>
      <div class="bottom-bar"><button class="btn accent ${d.highlight === "recommend" ? "pulse" : ""}" data-action="recommend">${esc(t.recommendBtn)}</button>${cart}</div>`;
  }

  function renderRecommend(d) {
    const t = L();
    const p = d.prefs || {};
    const keys = ["caffeine_free", "less_sweet", "refreshing", "milky", "hot", "ice"];
    const chips = keys.map((k) => `<button class="chip ${p[k] ? "on" : ""}" data-action="pref" data-key="${k}">${esc(t.prefs[k])}</button>`).join("");
    const active = keys.filter((k) => p[k]).map((k) => esc(t.prefs[k]));
    const think = `<div class="think-box"><b>${esc(t.thinkTitle)}</b> · ${active.length ? active.join(" + ") : esc(t.recAsk)}
      ${p.updated_by ? `<br>↻ ${esc(t.updated[p.updated_by] || "")}` : ""}</div>`;
    const item = d.item;
    const body = item
      ? `<div class="feature"><div class="emoji">${item.emoji}</div><div><div class="name">${nm(item.name)}</div>
           <div class="price">${price(item.price)}</div>
           <div class="chips">${(d.reasons || []).map((r) => `<span class="chip on">${esc(t.reasons[r] || r)}</span>`).join("")}</div></div></div>
         <div class="row"><button class="btn primary" data-action="accept_rec">${esc(t.accept)}</button>
           <button class="btn" data-action="reject_rec">${esc(t.another)}</button></div>`
      : `<p class="subtitle">${esc(t.recAsk)}</p>`;
    return `<h1 class="title">${esc(t.recTitle)}</h1><div class="chips">${chips}</div>${think}${body}
      <div class="spacer"></div><button class="btn ghost block" data-action="back">${esc(t.back)}</button>`;
  }

  function renderSoldOut(d) {
    const t = L();
    const item = d.item || {};
    return `<div class="banner"><span class="emoji">${item.emoji || ""}</span><span>${esc(t.soldTitle(nm(item.name)))}</span></div>
      <h2 class="subtitle">${esc(t.similar)}</h2>
      <div class="grid">${(d.alternatives || []).map((alt) => card(alt, "accept_alt")).join("")}</div>
      <div class="spacer"></div><button class="btn ghost block" data-action="back">${esc(t.back)}</button>`;
  }

  function renderOptions(d) {
    const t = L();
    const item = d.item || { temps: [], sizes: [], name: {}, desc: {} };
    const line = d.line || {};
    const temps = item.temps.map((v) =>
      `<button class="btn ${line.temp === v ? "on" : ""}" data-action="set_option" data-key="temp" data-value="${v}">${esc(t[v])}</button>`).join("");
    const sizes = item.sizes.map((v) =>
      `<button class="btn ${line.size === v ? "on" : ""}" data-action="set_option" data-key="size" data-value="${v}">${esc(t[v])}${v === "large" ? ` +${price(d.large_upcharge)}` : ""}</button>`).join("");
    return `<div class="item-head"><div class="emoji">${item.emoji || ""}</div><div><div class="name">${nm(item.name)}</div>
        <div class="desc">${nm(item.desc)}</div></div></div>
      <div class="opt-group ${d.asking === "temp" ? "asking" : ""}"><h3>${esc(t.temp)}</h3><div class="row">${temps}</div></div>
      <div class="opt-group ${d.asking === "size" ? "asking" : ""}"><h3>${esc(t.size)}</h3><div class="row">${sizes}</div></div>
      <div class="spacer"></div>
      <div class="row"><button class="btn ghost" data-action="back">${esc(t.back)}</button>
        <button class="btn primary" data-action="options_done">${esc(t.addToCart)} · ${price((d.unit_price || 0) * (line.qty || 1))}</button></div>`;
  }

  function optionText(line) {
    const t = L();
    const notes = (line.notes || []).map((k) => t.notes[k] || k);
    return [line.temp ? t[line.temp] : "", line.size ? t[line.size] : "", ...notes].filter(Boolean).map(esc).join(" · ");
  }

  function renderCart(d) {
    const t = L();
    const cart = d.cart || { lines: [], total: 0 };
    // data-value = 줄 표식: 서버가 지금 그 자리의 줄과 같을 때만 지운다 (✕ 두 번 누름 방지)
    const lines = cart.lines.map((line) => `<div class="line"><span class="emoji">${line.emoji}</span>
        <div class="info"><div class="name">${nm(line.name)} <small>${esc(t.qty(line.qty))}</small></div>
        <div class="opts">${optionText(line)}</div></div><div class="price">${price(line.price)}</div>
        <button class="remove" data-action="remove" data-index="${line.index}" data-value="${esc(line.sign || "")}" aria-label="remove">✕</button></div>`).join("");
    const dine = d.dine ? `<span class="dine">${esc(t.dine[d.dine] || d.dine)}</span>` : "";
    return `<h1 class="title">${esc(t.cartTitle)}</h1><div class="scroll"><div class="lines">${lines}</div></div>
      <div class="total"><span>${esc(t.total)} ${dine}</span><span>${price(cart.total)}</span></div>
      <div class="row"><button class="btn" data-action="add_more">${esc(t.more)}</button>
        <button class="btn primary" data-action="confirm">${esc(t.pay)}</button></div>`;
  }

  function renderPayment(d) {
    const t = L();
    if (d.step === "processing") {
      return `<div class="pay"><div class="amount">${price(d.total)}</div><div class="spinner"></div>
        <h1 class="title">${esc(t.payProcessing)}</h1></div>`;
    }
    const remove = d.step === "remove";
    return `<div class="pay"><div class="amount">${price(d.total)}</div>
      <div class="card-anim ${remove ? "remove" : "insert"}"><div class="cardpic"></div><div class="slot blink"></div></div>
      <h1 class="title">${esc(remove ? t.payRemove : t.payInsert)}</h1><div class="arrow">${remove ? "↑" : "↓"}</div>
      ${d.fallback && !remove ? `<button class="btn" data-action="pay_fallback">${esc(t.payFallback)}</button>` : ""}</div>`;
  }

  function renderDone(d) {
    const t = L();
    const dine = d.dine ? `<span class="dine">${esc(t.dine[d.dine] || d.dine)}</span>` : "";
    return `<div class="done"><div class="label">${esc(t.doneLabel)} ${dine}</div><div class="number">${esc(d.number)}</div>
      <div class="thanks">${esc(t.thanks)}</div></div>`;
  }

  // 직원 호출: 화면 위 알림 띠 + 주문 화면 아래의 호출 버튼 (부르면 버튼 대신 띠가 보인다)
  const STAFF_SCREENS = new Set(["menu", "recommend", "sold_out", "options", "cart", "payment"]);

  function renderNotice() {
    const el = $("notice");
    const t = L();
    if (view.notice && view.notice.name === "staff") {
      el.innerHTML = `<b>${esc(t.staffCalled)}</b><span>${esc(t.staffSub)}</span>`;
      el.classList.remove("hidden");
    } else {
      el.classList.add("hidden");
      el.innerHTML = "";
    }
  }

  function renderOverlay() {
    const el = $("overlay");
    const ov = view.overlay;
    if (!ov) { el.classList.add("hidden"); el.innerHTML = ""; return; }
    const t = L();
    const d = ov.data || {};
    let html = "";
    if (ov.name === "paused") {
      html = d.asked
        ? `<div class="big">${esc(t.resumeQ)}</div>`
        : `<div class="big">${esc(t.pausedWait)}</div><p class="subtitle">${esc(t.pausedSub)}</p>`;
      html += `<div class="row"><button class="btn" data-action="resume_no">${esc(t.no)}</button>
        <button class="btn primary" data-action="resume_yes">${esc(t.yes)}</button></div>`;
    } else if (ov.name === "one_at_a_time") {
      html = `<div class="big">${esc(t.oneTitle)}</div><p class="subtitle">${esc(t.oneSub)}</p>
        <button class="btn primary block" data-action="start">${esc(t.start)}</button>`;
    } else if (ov.name === "detail") {
      const item = d.item || { name: {}, desc: {} };
      html = `<div class="feature"><div class="emoji">${item.emoji || ""}</div><div style="text-align:left">
          <div class="name">${nm(item.name)}</div><div class="price">${price(item.price)}</div>
          <p class="subtitle">${nm(item.desc)}</p>${badges(item)}</div></div>
        ${item.in_stock === false ? `<div class="banner">${esc(t.soldOut)}</div>` : ""}
        <div class="row"><button class="btn" data-action="dismiss">${esc(t.close)}</button>
          <button class="btn primary" data-action="confirm" ${item.in_stock === false ? "disabled" : ""}>${esc(t.orderThis)}</button></div>`;
    } else if (ov.name === "candidates") {
      html = `<div class="big">${esc(t.candTitle)}</div>
        <div class="grid">${(d.items || []).map((item) => card(item, "candidate")).join("")}</div>
        <button class="btn ghost block" data-action="dismiss">${esc(t.sayAgain)}</button>`;
    }
    el.innerHTML = `<div class="sheet">${html}</div>`;
    el.classList.remove("hidden");
  }

  function render() {
    document.documentElement.lang = view.lang;
    $("app").classList.toggle("no-strip", view.features && view.features.status_strip === false);
    const name = screens[view.screen] ? view.screen : "attract";
    const staff = STAFF_SCREENS.has(name) && !view.notice
      ? `<button class="staff-link" data-action="staff">${esc(L().staffBtn)}</button>` : "";
    $("screen").className = `screen screen-${name}`;  // 화면별 간격 조정용 (style.css의 .screen-menu)
    $("screen").innerHTML = screens[name](view.data || {}) + staff;
    renderNotice();
    renderOverlay();
    renderSteps();
    renderStatus();
  }

  render();
  connect();
})();
