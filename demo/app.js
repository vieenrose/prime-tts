/* PrimeTTS v2.1 Nano demo client (player v4).
 *
 * Two playback modes, so a remaining artefact can be attributed instead of guessed:
 *   Stream   - schedule 0.25 s pieces as they arrive, after a 0.75 s prebuffer (lowest latency).
 *   Buffered - download everything, then play it as ONE AudioBuffer (no scheduling, no seams).
 * If Buffered is clean and Stream is not, the problem is scheduling. If both sound the same,
 * the problem is not in this file.
 *
 * The delivered PCM is verified clean server-side (joint energy profiles, edge bursts, click
 * detection normalised by peak, byte-identical across runs, and the downloaded WAV plays clean),
 * so the indicator line reports evidence rather than assumptions.
 */
const GATEWAY_URL = "https://variation-joan-requirement-delivers.trycloudflare.com";
const PLAYER_VERSION = "v4";
const SR = 16000;
const EXAMPLES = [
  "您好，歡迎使用語音助理。",
  "您好,幫您轉接給 Jason 王經理,他的分機是 2580。",
  "今天天氣很好，我們一起去公園散步吧。公園裡有很多花，非常漂亮。",
  "My phone number is 0912-345-678. Please call me back tomorrow.",
  "您好,歡迎使用 PrimeTTS。Thank you for calling.",
];

const $ = (id) => document.getElementById(id);
const textEl = $("text");
let ctx = null, ctrl = null, playing = false;

for (const ex of EXAMPLES) {
  const c = document.createElement("span");
  c.className = "chip";
  c.textContent = ex.length > 26 ? ex.slice(0, 26) + "…" : ex;
  c.title = ex;
  c.onclick = () => { textEl.value = ex; };
  $("chips").appendChild(c);
}
$("pver").textContent = PLAYER_VERSION;

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

const showErr = (m) => { $("err").style.display = "block"; $("err").textContent = m; };
const clearErr = () => { $("err").style.display = "none"; $("err").textContent = ""; };

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

/* s16le bytes -> Float32Array with SIGN extension (unsigned decode mirrors every negative sample) */
function toFloat(bytes) {
  const n = bytes.length >> 1, f = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let v = (bytes[2 * i + 1] << 8) | bytes[2 * i];
    if (v >= 0x8000) v -= 0x10000;
    f[i] = v / 32768;
  }
  return f;
}

async function ensureCtx() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === "suspended") await ctx.resume();
  return ctx;
}

function requestBody(text) {
  return {
    model: "primetts-v21", input: text, voice: $("voice").value,
    response_format: "pcm", stream_format: "audio",
    speed: parseFloat($("speed").value),
  };
}

function beginRun() {
  clearErr();
  $("dl").style.display = "none";
  playing = true;
  $("go").disabled = true; $("goBuf").disabled = true; $("stop").disabled = false;
  $("first").textContent = "…"; $("nc").textContent = "0";
  $("audio").textContent = "0.0s"; $("gaps").textContent = "0";
  $("bar").style.width = "0";
  ctrl = new AbortController();
  return performance.now();
}

function endRun() {
  playing = false;
  $("go").disabled = false; $("goBuf").disabled = false; $("stop").disabled = true;
}

function publishWav(pcmBytes) {
  const dl = $("dl");
  dl.href = URL.createObjectURL(wavBlob(pcmBytes));
  dl.style.display = "inline";
}

/* ---------------- mode 1: streaming, with prebuffer and underrun accounting ---------------- */
async function speakStream() {
  const text = textEl.value.trim();
  if (!text || playing) return;
  await ensureCtx();
  const t0 = beginRun();
  const IN_CHUNK = 4000;   // 0.25 s at 16 kHz
  const PREBUF = 12000;    // 0.75 s banked before the first piece starts
  let firstMs = null, nChunks = 0, audioS = 0, gaps = 0;
  let fq = new Float32Array(0), playAt = 0, started = false;
  const full = [];

  const stats = () => {
    $("nc").textContent = nChunks;
    $("audio").textContent = audioS.toFixed(1) + "s";
    $("gaps").textContent = String(gaps);
    $("bar").style.width = Math.min(100, audioS / Math.max(1, text.length / 5) * 100) + "%";
  };
  const schedule = () => {
    if (!started) {
      if (fq.length < PREBUF) { if (firstMs === null) $("first").textContent = "buffering…"; return; }
      started = true;
      playAt = ctx.currentTime + 0.05;
      firstMs = performance.now() - t0;
      $("first").textContent = firstMs.toFixed(0) + " ms";
    }
    while (fq.length >= IN_CHUNK) {
      // buffers stay at the 16 kHz source rate: the browser resamples natively
      const buf = ctx.createBuffer(1, IN_CHUNK, SR), ch = buf.getChannelData(0);
      ch.set(fq.subarray(0, IN_CHUNK));
      fq = fq.slice(IN_CHUNK);
      const src = ctx.createBufferSource();
      src.buffer = buf; src.connect(ctx.destination);
      if (playAt < ctx.currentTime - 0.05) { gaps++; playAt = ctx.currentTime + 0.02; }
      const at = Math.max(playAt, ctx.currentTime + 0.02);
      src.start(at);
      playAt = at + IN_CHUNK / SR;
      nChunks++; audioS += IN_CHUNK / SR; stats();
    }
  };
  const timer = setInterval(schedule, 100);
  try {
    const resp = await fetch(GATEWAY_URL + "/v1/audio/speech", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody(text)), signal: ctrl.signal,
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
      full.push(value);
      const f = toFloat(value), m = new Float32Array(fq.length + f.length);
      m.set(fq); m.set(f, fq.length); fq = m;
    }
    await new Promise((r) => setTimeout(r, 150));
    while (fq.length > 0) {   // flush the tail through the same path
      const n = Math.min(IN_CHUNK, fq.length);
      const buf = ctx.createBuffer(1, n, SR), ch = buf.getChannelData(0);
      ch.set(fq.subarray(0, n)); fq = fq.slice(n);
      const src = ctx.createBufferSource();
      src.buffer = buf; src.connect(ctx.destination);
      if (playAt < ctx.currentTime - 0.05) { gaps++; playAt = ctx.currentTime + 0.02; }
      const at = Math.max(playAt, ctx.currentTime + 0.02);
      src.start(at); playAt = at + n / SR;
      nChunks++; audioS += n / SR; stats();
    }
    const total = full.reduce((a, b) => a + b.length, 0), pcm = new Uint8Array(total);
    let off = 0;
    for (const b of full) { pcm.set(b, off); off += b.length; }
    publishWav(pcm);
  } catch (e) {
    if (e.name !== "AbortError") showErr("Synthesis failed: " + e.message);
  } finally { clearInterval(timer); endRun(); }
}

/* ---------------- mode 2: buffered, one buffer, no scheduling seams ---------------- */
async function speakBuffered() {
  const text = textEl.value.trim();
  if (!text || playing) return;
  await ensureCtx();
  const t0 = beginRun();
  try {
    const resp = await fetch(GATEWAY_URL + "/v1/audio/speech", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody(text)), signal: ctrl.signal,
    });
    if (!resp.ok) {
      let msg = "HTTP " + resp.status;
      try { msg = (await resp.json()).error.message || msg; } catch (_) {}
      throw new Error(msg);
    }
    const parts = [], reader = resp.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      $("audio").textContent = (parts.reduce((a, b) => a + b.length, 0) / 2 / SR).toFixed(1) + "s";
    }
    $("first").textContent = ((performance.now() - t0) / 1000).toFixed(2) + " s (all)";
    const total = parts.reduce((a, b) => a + b.length, 0), pcm = new Uint8Array(total);
    let off = 0;
    for (const b of parts) { pcm.set(b, off); off += b.length; }
    publishWav(pcm);
    const f = toFloat(pcm);                    // whole utterance, one buffer, one source
    const buf = ctx.createBuffer(1, f.length, SR), ch = buf.getChannelData(0);
    ch.set(f);
    const src = ctx.createBufferSource();
    src.buffer = buf; src.connect(ctx.destination);
    src.start();
    $("nc").textContent = "1"; $("gaps").textContent = "0";
    $("audio").textContent = (f.length / SR).toFixed(1) + "s";
    $("bar").style.width = "100%";
    await new Promise((r) => { src.onended = r; });
  } catch (e) {
    if (e.name !== "AbortError") showErr("Synthesis failed: " + e.message);
  } finally { endRun(); }
}

$("go").onclick = speakStream;
$("goBuf").onclick = speakBuffered;
$("stop").onclick = () => { if (ctrl) ctrl.abort(); };
textEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("go").click(); }
});
