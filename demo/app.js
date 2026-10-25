/* PrimeTTS v2.1 Nano demo client.
 * POSTs text to the public gateway (OpenAI-shaped /v1/audio/speech) and plays the
 * streamed PCM16 16 kHz sentences back-to-back via Web Audio as they arrive.
 *
 * If the tunnel restarted and this URL 404s/refuses, update it (see demo/README.md).
 */
const GATEWAY_URL = "https://variation-joan-requirement-delivers.trycloudflare.com";

const SR = 16000;
const EXAMPLES = [
  "您好，歡迎使用語音助理。",
  "您好,幫您轉接給 Jason 王經理,他的分機是 2580。",
  "今天天氣很好，我們一起去公園散步吧。公園裡有很多花，非常漂亮。",
  "My phone number is 0912-345-678. Please call me back tomorrow.",
  "您好,歡迎使用 PrimeTTS。Thank you for calling.",
];

const $ = (id) => document.getElementById(id);
const textEl = $("text"), goBtn = $("go"), stopBtn = $("stop"), errEl = $("err");
let ctx = null, ctrl = null, playing = false;

for (const ex of EXAMPLES) {
  const c = document.createElement("span");
  c.className = "chip";
  c.textContent = ex.length > 26 ? ex.slice(0, 26) + "…" : ex;
  c.title = ex;
  c.onclick = () => { textEl.value = ex; };
  $("chips").appendChild(c);
}

async function health() {
  $("health").textContent = "checking…";
  try {
    const r = await fetch(GATEWAY_URL + "/health");
    const h = await r.json();
    const n = h.nano || {};
    $("health").innerHTML =
      `gateway <b>ok</b> · Nano <b>${n.status || "?"}</b> · ` +
      `${n.ready ?? "?"}/${n.workers ?? "?"} workers · GPU ${n.gpu_mhz ?? "?"} MHz · queue ${n.queued ?? "?"}`;
    $("backend").textContent = n.status === "ok" ? "Nano online" : "Nano busy";
  } catch (e) {
    $("health").textContent = "backend unreachable — " + e;
    $("backend").textContent = "offline";
  }
}
$("retry").onclick = health;
health();

function showErr(m) { errEl.style.display = "block"; errEl.textContent = m; }
function clearErr() { errEl.style.display = "none"; errEl.textContent = ""; }

function wavBlob(pcmBytes) {
  const n = pcmBytes.length, buf = new ArrayBuffer(44 + n), v = new DataView(buf);
  const wstr = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  wstr(0, "RIFF"); v.setUint32(4, 36 + n, true); wstr(8, "WAVEfmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, SR, true); v.setUint32(28, SR * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); wstr(36, "data");
  v.setUint32(40, n, true);
  new Uint8Array(buf, 44).set(pcmBytes);
  return new Blob([buf], { type: "audio/wav" });
}

goBtn.onclick = speak;

async function speak() {
  const text = textEl.value.trim();
  if (!text || playing) return;
  clearErr();
  $("dl").style.display = "none";
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
  if (ctx.state === "suspended") await ctx.resume();
  playing = true; goBtn.disabled = true; stopBtn.disabled = false;
  $("first").textContent = "…"; $("nc").textContent = "0"; $("audio").textContent = "0.0s";
  $("bar").style.width = "0";
  const t0 = performance.now();
  const outRate = ctx.sampleRate, ratio = SR / outRate;  // resample only if the browser ignored 16 kHz
  const IN_CHUNK = 4000;  // 0.25 s of input samples per scheduled piece: gapless queue
  let firstMs = null, nChunks = 0, audioS = 0, fq = new Float32Array(0), playAt = ctx.currentTime + 0.05;
  const full = [];
  ctrl = new AbortController();
  const pushBytes = (bytes) => {  // s16le -> float with SIGN extension (unsigned decode distorts)
    const n = bytes.length >> 1, f = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let v = (bytes[2 * i + 1] << 8) | bytes[2 * i];
      if (v >= 0x8000) v -= 0x10000;
      f[i] = v / 32768;
    }
    const m = new Float32Array(fq.length + n);
    m.set(fq); m.set(f, fq.length);
    fq = m;
  };
  const playIn = (nIn) => {  // schedule nIn input samples (linear resample to the context rate)
    const outN = Math.max(1, Math.round(nIn / ratio));
    const buf = ctx.createBuffer(1, outN, outRate), ch = buf.getChannelData(0);
    for (let i = 0; i < outN; i++) {
      const pos = i * ratio, i0 = Math.floor(pos), fr = pos - i0;
      const a = fq[i0], b = fq[Math.min(i0 + 1, fq.length - 1)];
      ch[i] = a + (b - a) * fr;
    }
    fq = fq.slice(nIn);
    const src = ctx.createBufferSource();
    src.buffer = buf; src.connect(ctx.destination);
    playAt = Math.max(playAt, ctx.currentTime + 0.02);
    src.start(playAt); playAt += outN / outRate;
    nChunks++; audioS += nIn / SR;
    $("nc").textContent = nChunks; $("audio").textContent = audioS.toFixed(1) + "s";
    $("bar").style.width = Math.min(100, audioS / Math.max(1, text.length / 5) * 100) + "%";
  };
  const timer = setInterval(() => { while (fq.length >= IN_CHUNK) playIn(IN_CHUNK); }, 100);
  try {
    const resp = await fetch(GATEWAY_URL + "/v1/audio/speech", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "primetts-v21", input: text, voice: $("voice").value,
        response_format: "pcm", stream_format: "audio",
        speed: parseFloat($("speed").value),
      }),
      signal: ctrl.signal,
    });
    if (!resp.ok) {
      let msg = "HTTP " + resp.status;
      try { msg = (await resp.json()).error.message || msg; } catch (_) {}
      throw new Error(msg);
    }
    const reader = resp.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (firstMs === null) { firstMs = performance.now() - t0; $("first").textContent = firstMs.toFixed(0) + " ms"; }
      full.push(value);
      pushBytes(value);
    }
    await new Promise((r) => setTimeout(r, 150));
    if (fq.length > 0) playIn(fq.length);  // flush the tail
    const total = full.reduce((a, b) => a + b.length, 0), pcm = new Uint8Array(total);
    let off = 0;
    for (const b of full) { pcm.set(b, off); off += b.length; }
    const url = URL.createObjectURL(wavBlob(pcm));
    const dl = $("dl");
    dl.href = url; dl.style.display = "inline";
  } catch (e) {
    if (e.name !== "AbortError") showErr("Synthesis failed: " + e.message);
  } finally {
    clearInterval(timer);
    playing = false; goBtn.disabled = false; stopBtn.disabled = true;
  }
}

stopBtn.onclick = () => { if (ctrl) ctrl.abort(); };
textEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); goBtn.click(); }
});
