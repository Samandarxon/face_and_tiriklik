// Real-time yuz autentifikatsiyasi — WebSocket oqimi.
//
// Mijoz ATAYLAB "ahmoq" qilib yozilgan: u faqat kadr yuboradi va hodisa oladi.
// Barcha qarorlar (qachon to'liq tekshiruv boshlanadi, chegaralar, tanaffuslar)
// serverda — `internal/delivery/httpapi/stream_handler.go` da. Shu tufayli
// mobil yoki kiosk mijozi ham xuddi shu mantiqni takrorlamaydi.
//
// Oqim:  brauzer --WebSocket(JPEG)--> Golang --gRPC bidi--> Python

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("canvas"), overlay = $("overlay"), stage = $("stage");
const ctx = overlay.getContext("2d");

const MIN_FRAME_GAP_MS = 60;    // ketma-ket kadrlar orasidagi minimal tanaffus
const FRAME_WIDTH = 480;        // yuborilayotgan kadr eni (kichikroq => tezroq)
const FRAME_QUALITY = 0.7;
const RECOGNIZED_MS = 6000;     // tanilgan ism ekranda qancha turadi

// Ramka harakati sozlamalari.
//   SMOOTH_TAU_MS — eksponensial silliqlash vaqt doimiysi. Kichikroq => tezroq
//   ergashadi, lekin sakrashlar ko'proq seziladi.
//   PREDICT_MAX_MS — kechikish kompensatsiyasi chegarasi. Server javobi kelguncha
//   yuz allaqachon siljigan bo'ladi, shuning uchun oxirgi ikki o'lchov bo'yicha
//   tezlikni hisoblab, ramkani oldinga surib qo'yamiz.
const SMOOTH_TAU_MS = 50;
const PREDICT_MAX_MS = 220;

let stream = null, socket = null;

// --- Kadr yuborish holati (pipelined: bittadan ortiq kadr "havoda" bo'lmaydi) ---
let inFlight = false;
let lastSentAt = 0;
let latencyMs = 200;            // o'lchangan aylanma vaqt (silliqlangan)
let eventsPerSec = 0;

// --- Overlay holati ---
let targetBox = null, targetAt = 0;   // serverdan kelgan oxirgi o'lchov
let prevBox = null, prevAt = 0;       // undan oldingisi (tezlik uchun)
let renderBox = null;                 // ekranda ko'rinayotgan (silliqlangan)
let boxScale = 1;                     // kadr -> video koordinata koeffitsiyenti
let boxState = "checking";            // checking | real | spoof | known
let boxLabel = "";
let recognizedUntil = 0;
let lastDrawAt = 0;

const COLORS = {
  checking: "#d29922",
  real: "#3fb950",
  spoof: "#f85149",
  known: "#2f81f7",
};

// ==================================================================== kamera

$("startCam").onclick = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 960 }, facingMode: "user" },
    });
    video.srcObject = stream;
    await video.play().catch(() => {});

    stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    $("startCam").disabled = true;
    $("stopCam").disabled = false;
    $("btnRegister").disabled = false;
    $("btnVerify").disabled = false;

    // connect() endi async (avval ticket oladi). Kamerani ko'rsatishni
    // kutib turmaymiz — ticket kelguncha video allaqachon oqib turadi.
    void connect();
    requestAnimationFrame(drawLoop);
  } catch (err) {
    show("verifyResult", "err", "Kamerani ochib bo'lmadi", String(err) +
      "\n\nEslatma: brauzer kamerani faqat https:// yoki localhost'da beradi.");
  }
};

$("stopCam").onclick = () => {
  disconnect();
  stream?.getTracks().forEach((t) => t.stop());
  video.srcObject = stream = null;
  clearBox();
  lastDrawAt = 0;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  $("startCam").disabled = false;
  ["stopCam", "btnRegister", "btnVerify"].forEach((id) => ($(id).disabled = true));
  setLive("idle", "Kamera o'chiq");
};

// ================================================================== API kalit
//
// Servis YOPIQ: `/health` dan tashqari hamma narsa kalit talab qiladi. Bu
// sahifa avval kalitsiz ishlardi va kirish nazorati qo'shilgandan keyin
// jimgina buzilib qolgan edi — WebSocket 401 olardi, sabab esa ko'rinmasdi.
//
// sessionStorage, localStorage EMAS: kalit tab yopilishi bilan o'chadi.
// Umumiy kompyuterda bu farq muhim.

const KEY_STORAGE = "face_api_key";

const apiKey = () => sessionStorage.getItem(KEY_STORAGE) || "";

function renderKeyStatus() {
  const has = Boolean(apiKey());
  $("keyStatus").textContent = has
    ? "Kalit saqlangan (faqat shu tabda). Kamerani yoqishingiz mumkin."
    : "Kalit kiritilmagan — amallar ishlamaydi.";
  $("keyStatus").style.color = has ? "" : "#f59e0b";
}

$("saveKey").onclick = () => {
  const v = $("apiKey").value.trim();
  if (!v) return;
  sessionStorage.setItem(KEY_STORAGE, v);
  $("apiKey").value = "";
  renderKeyStatus();
};

$("clearKey").onclick = () => {
  sessionStorage.removeItem(KEY_STORAGE);
  renderKeyStatus();
};

renderKeyStatus();

// ================================================================= WebSocket

// Ticket oladi. WebSocket handshake sarlavha ko'tara olmaydi, shuning uchun
// kredensial URL'da ketishi kerak — va URL loglarga tushadi. Ticket aynan shu
// sababdan bir martalik va 60 soniyalik: logga tushgan paytda u allaqachon
// yaroqsiz bo'ladi.
async function fetchTicket() {
  const key = apiKey();
  if (!key) return null;
  try {
    const res = await fetch("/api/v1/stream/tickets", {
      method: "POST",
      headers: { "X-API-Key": key },
    });
    if (!res.ok) return null;
    return (await res.json()).ticket || null;
  } catch {
    return null;
  }
}

async function connect() {
  const ticket = await fetchTicket();
  if (!ticket) {
    setLive("idle", apiKey()
      ? "Ticket olinmadi — kalit noto'g'ri yoki muddati o'tgan"
      : "API kalit kiritilmagan");
    return;
  }

  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${scheme}//${location.host}/api/v1/stream?ticket=${encodeURIComponent(ticket)}`);
  socket.binaryType = "arraybuffer";

  socket.onopen = () => {
    setLive("checking", "Ulandi — tiriklik tekshirilmoqda...");
    inFlight = false;
    pushFrame();
  };

  socket.onmessage = (msg) => {
    handleEvent(JSON.parse(msg.data));
    // PIPELINE: javob kelishi bilan keyingi kadrni yuboramiz. Qat'iy
    // interval o'rniga shu usul kadr tezligini serverning HAQIQIY tezligiga
    // moslashtiradi — sekin mashinada navbat o'smaydi, tezida esa ramka
    // ancha ravon harakatlanadi.
    inFlight = false;
    scheduleNextFrame();
  };

  socket.onclose = () => {
    inFlight = false;
    if (stream) {
      setLive("checking", "Ulanish uzildi — qayta ulanmoqda...");
      // Har urinishda YANGI ticket: ular bir martalik, eskisini qayta
      // ishlatish ikkinchi urinishda albatta yiqilardi.
      setTimeout(() => stream && void connect(), 2000);
    }
  };

  socket.onerror = () => setLive("checking", "Ulanish xatosi");
}

function disconnect() {
  inFlight = false;
  if (socket) {
    socket.onclose = null;   // qayta ulanishni o'chiramiz
    socket.close();
    socket = null;
  }
}

function scheduleNextFrame() {
  const wait = Math.max(0, MIN_FRAME_GAP_MS - (performance.now() - lastSentAt));
  setTimeout(pushFrame, wait);
}

// Kadrni oqimga yuboradi. Bufer to'lib ketsa kadrni tashlab yuboramiz —
// tarmoq sekin bo'lsa navbat cheksiz o'smasligi uchun.
async function pushFrame() {
  if (!socket || socket.readyState !== WebSocket.OPEN || inFlight) return;
  if (socket.bufferedAmount > 512 * 1024) return;

  try {
    const frame = await grabFrame(FRAME_WIDTH, FRAME_QUALITY);
    boxScale = video.videoWidth / frame.width;
    inFlight = true;
    lastSentAt = performance.now();
    socket.send(await frame.blob.arrayBuffer());
  } catch {
    inFlight = false;   // kamera hali tayyor emas — keyingi urinishda
    setTimeout(pushFrame, 200);
  }
}

// ---------------------------------------------------------- server hodisalari

function handleEvent(evt) {
  const recognized = Date.now() < recognizedUntil;

  // Aylanma vaqtni o'lchaymiz — bashorat aynan shunga tayanadi.
  if (lastSentAt) {
    const rtt = performance.now() - lastSentAt;
    latencyMs = latencyMs * 0.7 + rtt * 0.3;   // silliqlangan o'rtacha
    eventsPerSec = 1000 / Math.max(rtt, 1);
  }

  setTargetBox(normalizeBBox(evt.bbox));

  switch (evt.type) {
    case "ready":
      setLive("checking", "Tiriklik tekshirilmoqda...");
      break;

    case "liveness": {
      const p = evt.liveness?.real_probability ?? 0;
      if (!recognized) {
        boxState = "real";
        boxLabel = `Tirik · ${(p * 100).toFixed(0)}%`;
      }
      setLive("real", "Tirik inson", p);
      break;
    }

    case "spoof":
      recognizedUntil = 0;
      boxState = "spoof";
      boxLabel = "SOXTA";
      setLive("spoof", "Soxta yuz (foto/ekran)", evt.liveness?.real_probability ?? 0);
      show("verifyResult", "err", "🚫 SOXTA YUZ ANIQLANDI — bazaga so'rov yuborilmadi",
        JSON.stringify(evt, null, 2));
      break;

    case "recognized":
      boxState = "known";
      boxLabel = `${evt.name} · ${(evt.score * 100).toFixed(1)}%`;
      recognizedUntil = Date.now() + RECOGNIZED_MS;
      setLive("real", `Tanildi: ${evt.name}`, 1);
      show("verifyResult", "ok",
        `✅ ${evt.name} (${evt.person_id}) — score ${evt.score.toFixed(4)}`,
        JSON.stringify(evt, null, 2));
      break;

    case "not_found":
      boxState = "checking";
      boxLabel = "Notanish";
      setLive("checking", "Bazadan topilmadi");
      show("verifyResult", "warn", "⚠️ Bazadan inson topilmadi", JSON.stringify(evt, null, 2));
      break;

    case "error":
      if (evt.code === "no_face_detected") {
        clearBox();
        setLive("checking", "Yuz ko'rinmayapti");
      } else {
        setLive("checking", evt.message || "Kadr yaroqsiz");
      }
      break;
  }
}

// ====================================================== ramka harakati

function normalizeBBox(bbox) {
  if (Array.isArray(bbox) && bbox.length === 4) return bbox.slice();
  if (bbox && typeof bbox === "object" && (bbox.x2 || bbox.y2)) {
    return [bbox.x1, bbox.y1, bbox.x2, bbox.y2];
  }
  return null;
}

function setTargetBox(bbox) {
  if (!bbox) return;
  prevBox = targetBox;
  prevAt = targetAt;
  targetBox = bbox;
  targetAt = performance.now();
  if (!renderBox) renderBox = bbox.slice();   // birinchi kadr — darhol joyiga
}

function clearBox() {
  targetBox = prevBox = renderBox = null;
}

// aimBox — server javobi kelguncha yuz allaqachon siljigan bo'ladi. Oxirgi
// ikki o'lchov bo'yicha tezlikni hisoblab, ramkani AYLANMA VAQT qadar oldinga
// suramiz. Bu ko'zga tashlanadigan kechikishni sezilarli kamaytiradi.
function aimBox() {
  if (!targetBox) return null;
  if (!prevBox || targetAt <= prevAt) return targetBox;

  const dt = targetAt - prevAt;
  const ahead = Math.min(PREDICT_MAX_MS, latencyMs);
  const k = ahead / dt;

  return targetBox.map((v, i) => {
    const velocity = v - prevBox[i];
    // Juda katta sakrash (boshqa yuzga o'tish) bashoratni buzmasligi uchun cheklaymiz.
    const step = Math.max(-80, Math.min(80, velocity * k));
    return v + step;
  });
}

// ============================================================ overlay chizish

function drawLoop(now) {
  if (!stream) return;
  const dt = lastDrawAt ? Math.min(100, now - lastDrawAt) : 16;
  lastDrawAt = now;

  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const aim = aimBox();
  if (aim && renderBox) {
    // Eksponensial silliqlash: kadr tezligidan QAT'I NAZAR bir xil tezlikda
    // yaqinlashadi (60 Hz da ham, 30 Hz da ham bir xil his).
    const alpha = 1 - Math.exp(-dt / SMOOTH_TAU_MS);
    for (let i = 0; i < 4; i++) {
      renderBox[i] += (aim[i] - renderBox[i]) * alpha;
    }

    // Video CSS'da ko'zguga olingan (scaleX(-1)), shuning uchun X'ni qo'lda
    // aylantiramiz — aks holda ramka yuzning teskari tomonida turadi.
    // Yozuv esa ko'zguga olinmaydi, o'qilishi kerak.
    const s = boxScale;
    const x1 = overlay.width - renderBox[2] * s;
    const x2 = overlay.width - renderBox[0] * s;
    drawBox(x1, renderBox[1] * s, x2 - x1, (renderBox[3] - renderBox[1]) * s,
            COLORS[boxState], boxLabel);
  }

  requestAnimationFrame(drawLoop);
}

function drawBox(x, y, w, h, color, label) {
  const corner = Math.max(16, Math.min(w, h) * 0.22);
  ctx.lineWidth = Math.max(3, overlay.width / 320);
  ctx.strokeStyle = color;
  ctx.lineCap = "round";

  // Burchakli "vizir" ramka — to'liq to'rtburchakdan ko'ra yuzni yaxshi ochib beradi.
  const segments = [
    [[x, y + corner], [x, y], [x + corner, y]],
    [[x + w - corner, y], [x + w, y], [x + w, y + corner]],
    [[x + w, y + h - corner], [x + w, y + h], [x + w - corner, y + h]],
    [[x + corner, y + h], [x, y + h], [x, y + h - corner]],
  ];
  for (const points of segments) {
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
    ctx.stroke();
  }

  ctx.globalAlpha = 0.35;
  ctx.strokeRect(x, y, w, h);
  ctx.globalAlpha = 1;

  if (!label) return;

  const fontSize = Math.max(15, overlay.width / 36);
  ctx.font = `600 ${fontSize}px -apple-system, "Segoe UI", Roboto, sans-serif`;
  const padX = fontSize * 0.55, padY = fontSize * 0.38;
  const boxW = ctx.measureText(label).width + padX * 2;
  const boxH = fontSize + padY * 2;
  const labelX = Math.max(4, Math.min(x, overlay.width - boxW - 4));
  const labelY = y - boxH - 6 >= 0 ? y - boxH - 6 : y + 6;

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(labelX, labelY, boxW, boxH, 6);
  ctx.fill();

  ctx.fillStyle = "#08121f";
  ctx.textBaseline = "middle";
  ctx.fillText(label, labelX + padX, labelY + boxH / 2);
}

// ================================================================= yordamchi

function grabFrame(maxWidth = 0, quality = 0.92) {
  return new Promise((resolve, reject) => {
    if (!video.videoWidth) return reject(new Error("Kamera hali tayyor emas."));
    let w = video.videoWidth, h = video.videoHeight;
    if (maxWidth && w > maxWidth) {
      h = Math.round((h * maxWidth) / w);
      w = maxWidth;
    }
    canvas.width = w;
    canvas.height = h;
    canvas.getContext("2d").drawImage(video, 0, 0, w, h);
    canvas.toBlob(
      (b) => (b ? resolve({ blob: b, width: w }) : reject(new Error("Kadr olinmadi."))),
      "image/jpeg", quality);
  });
}

function setLive(state, text, probability = null) {
  const el = $("liveStatus");
  el.className = `live ${state}`;
  const pct = probability === null ? "" : ` ${(probability * 100).toFixed(0)}%`;
  // Tezlik ko'rsatkichi — ramka sekin harakatlansa sababi shu yerda ko'rinadi.
  const rate = socket && socket.readyState === WebSocket.OPEN && eventsPerSec
    ? `<span class="rate">${eventsPerSec.toFixed(1)} fps · ${latencyMs.toFixed(0)} ms</span>`
    : "";
  el.innerHTML = `<span>${text}${pct}</span>` +
    (probability === null ? "" : `<span class="bar"><i style="width:${probability * 100}%"></i></span>`) +
    rate;
}

function show(target, kind, title, body) {
  $(target).innerHTML =
    `<div class="result ${kind}"><b>${title}</b>${body ? `<pre>${body}</pre>` : ""}</div>`;
}

async function post(url, form, button) {
  button.disabled = true;
  const original = button.textContent;
  button.textContent = "Tekshirilmoqda...";
  try {
    const key = apiKey();
    const res = await fetch(url, {
      method: "POST",
      headers: key ? { "X-API-Key": key } : {},
      body: form,
    });
    return { res, data: await res.json() };
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

// ============================== qo'lda chaqiriladigan REST endpointlar =======
// Oqim real-time uchun; bu ikkitasi bir martalik amallar uchun.

$("btnRegister").onclick = async () => {
  const name = $("personName").value.trim();
  if (!name) return show("registerResult", "err", "Ism to'ldirilishi shart");

  try {
    const frame = await grabFrame();
    const form = new FormData();
    form.append("name", name);
    form.append("image", frame.blob, "frame.jpg");
    // person_id yuborilmaydi — backend UUIDv4 generatsiya qiladi.

    const { res, data } = await post("/api/v1/persons/register", form, $("btnRegister"));
    show("registerResult", res.ok ? "ok" : "err",
      res.ok ? `✅ ${data.name} — ${data.person_id}` : `❌ ${data.message}`,
      JSON.stringify(data, null, 2));
  } catch (err) {
    show("registerResult", "err", "Xatolik", String(err));
  }
};

$("btnVerify").onclick = async () => {
  try {
    const frame = await grabFrame();
    boxScale = video.videoWidth / frame.width;

    const form = new FormData();
    form.append("image", frame.blob, "frame.jpg");

    const { res, data } = await post("/api/v1/persons/verify", form, $("btnVerify"));
    if (res.ok) {
      show("verifyResult", "ok", `✅ ${data.name} (${data.person_id}) — score ${data.score}`,
        JSON.stringify(data, null, 2));
    } else if (data.code === "spoof_detected") {
      show("verifyResult", "err", "🚫 SOXTA YUZ ANIQLANDI — bazaga so'rov yuborilmadi",
        JSON.stringify(data, null, 2));
    } else if (data.code === "person_not_found") {
      show("verifyResult", "warn", "⚠️ Bazadan inson topilmadi", JSON.stringify(data, null, 2));
    } else {
      show("verifyResult", "err", `❌ ${data.message}`, JSON.stringify(data, null, 2));
    }
  } catch (err) {
    show("verifyResult", "err", "Xatolik", String(err));
  }
};
