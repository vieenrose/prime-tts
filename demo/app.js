/* PrimeTTS v2.1 Nano demo client.
 *
 * The Android/iOS-style flow: text -> public gateway (zh-TW/en frontend) -> ONE request on the
 * Jetson Nano, which synthesizes the whole utterance once (single encoder pass, decoder streamed
 * in windows) and streams PCM16 16 kHz back. Playback goes through an AudioWorklet ring buffer, so
 * the audio is one continuous stream with no per-buffer seams.
 */
const GATEWAY_URL = "https://variation-joan-requirement-delivers.trycloudflare.com";
const SR = 16000;
const PREBUF = 6400;   // 0.4 s banked before playback starts (network jitter only)

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

async function health() {
  $("health").textContent = "checking…";
  try {
    const h = await (await fetch(GATEWAY_URL + "/health")).json();
    const n = h.nano || {};
    $("health").innerHTML =
      `Jetson Nano <b>${n.status || "?"}</b> · ${n.ready ?? "?"}/${n.workers ?? "?"} workers · ` +
      `GPU ${n.gpu_mhz ?? "?"} MHz · queue ${n.queued ?? "?"}`;
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

function wavBlob(pcm) {
  const n = pcm.length, buf = new ArrayBuffer(44 + n), v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + n, true); w(8, "WAVEfmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, SR, true); v.setUint32(28, SR * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data");
  v.setUint32(40, n, true);
  new Uint8Array(buf, 44).set(pcm);
  return new Blob([buf], { type: "audio/wav" });
}

/* s16le -> Float32Array with sign extension */
function toFloat(bytes) {
  const n = bytes.length >> 1, f = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let v = (bytes[2 * i + 1] << 8) | bytes[2 * i];
    if (v >= 0x8000) v -= 0x10000;
    f[i] = v / 32768;
  }
  return f;
}

/* Fallback for browsers that refuse a 16 kHz context: one buffer, browser-side conversion. */
async function speakBuffered(text) {
  playing = true;
  $("go").disabled = true; $("stop").disabled = false;
  $("first").textContent = "…"; $("gaps").textContent = "0";
  ctrl = new AbortController();
  const t0 = performance.now();
  try {
    const resp = await fetch(GATEWAY_URL + "/v1/audio/speech", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "primetts-v21", input: text, voice: $("voice").value,
        speed: parseFloat($("speed").value), response_format: "pcm", stream_format: "audio",
      }),
      signal: ctrl.signal,
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
    const total = parts.reduce((a, b) => a + b.length, 0), pcm = new Uint8Array(total);
    let off = 0;
    for (const b of parts) { pcm.set(b, off); off += b.length; }
    $("dl").href = URL.createObjectURL(wavBlob(pcm));
    $("dl").style.display = "inline";
    const f = toFloat(pcm);
    const buf = ctx.createBuffer(1, f.length, SR), ch = buf.getChannelData(0);
    ch.set(f);
    const src = ctx.createBufferSource();
    src.buffer = buf; src.connect(ctx.destination);
    src.start();
    $("first").textContent = ((performance.now() - t0) / 1000).toFixed(2) + " s (all)";
    $("audio").textContent = (f.length / SR).toFixed(1) + "s";
    $("bar").style.width = "100%";
    await new Promise((r) => { src.onended = r; });
  } catch (e) {
    if (e.name !== "AbortError") showErr("Synthesis failed: " + e.message);
  } finally {
    playing = false; $("go").disabled = false; $("stop").disabled = true;
  }
}

async function speak() {
  const text = textEl.value.trim();
  if (!text || playing) return;
  clearErr();
  $("dl").style.display = "none";
  if (!ctx) {
    // Open the context AT THE SOURCE RATE: the browser then does the device conversion with its
    // own resampler, and the worklet is a pure copy (see player.worklet.js).
    const AC = window.AudioContext || window.webkitAudioContext;
    try { ctx = new AC({ sampleRate: SR }); } catch (_) { ctx = new AC(); }
    if (ctx.sampleRate !== SR) {
      // This browser would not open at 16 kHz. Playback is then one whole buffer so the BROWSER
      // does the rate conversion (the worklet must never resample: measured -14 dB images).
      $("health").textContent += ` · browser audio ${ctx.sampleRate} Hz (buffered playback)`;
      return speakBuffered(text);
    }
  }
  if (ctx.state === "suspended") await ctx.resume();
  if (!ctx.audioWorklet) { showErr("This browser has no AudioWorklet support."); return; }

  playing = true;
  $("go").disabled = true; $("stop").disabled = false;
  $("first").textContent = "…"; $("audio").textContent = "0.0s"; $("gaps").textContent = "0";
  $("bar").style.width = "0";
  ctrl = new AbortController();
  const t0 = performance.now();
  let firstMs = null, audioS = 0, started = false, gaps = 0;
  const parts = [];

  await ctx.audioWorklet.addModule("./player.worklet.js");
  const node = new AudioWorkletNode(ctx, "primetts-player", {
    numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1],
    processorOptions: { inRate: SR },
  });
  node.connect(ctx.destination);
  node.port.onmessage = (e) => {
    if (e.data && e.data.underrunSamples) {   // network starvation, in ms of inserted silence
      gaps = Math.round((e.data.underrunSamples / SR) * 1000);
      $("gaps").textContent = String(gaps);
    }
  };

  try {
    const resp = await fetch(GATEWAY_URL + "/v1/audio/speech", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "primetts-v21", input: text, voice: $("voice").value,
        speed: parseFloat($("speed").value), response_format: "pcm", stream_format: "audio",
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
      parts.push(value);
      node.port.postMessage(toFloat(value));      // one continuous ring: no buffer seams
      audioS += value.length / 2 / SR;
      $("audio").textContent = audioS.toFixed(1) + "s";
      $("bar").style.width = Math.min(100, audioS / Math.max(1, text.length / 5) * 100) + "%";
      if (!started && audioS * SR >= PREBUF) {
        started = true;
        node.port.postMessage("start");
        firstMs = performance.now() - t0;
        $("first").textContent = firstMs.toFixed(0) + " ms";
      }
    }
    if (!started) {                                // shorter than the prebuffer
      node.port.postMessage("start");
      started = true;
      firstMs = performance.now() - t0;
      $("first").textContent = firstMs.toFixed(0) + " ms";
    }
    const total = parts.reduce((a, b) => a + b.length, 0), pcm = new Uint8Array(total);
    let off = 0;
    for (const b of parts) { pcm.set(b, off); off += b.length; }
    $("dl").href = URL.createObjectURL(wavBlob(pcm));
    $("dl").style.display = "inline";
    const waitMs = audioS * 1000 - (performance.now() - t0) + 400;
    if (waitMs > 0) await new Promise((r) => setTimeout(r, waitMs));
  } catch (e) {
    if (e.name !== "AbortError") showErr("Synthesis failed: " + e.message);
  } finally {
    try { node.port.postMessage("stop"); node.disconnect(); } catch (_) {}
    playing = false; $("go").disabled = false; $("stop").disabled = true;
  }
}

$("go").onclick = speak;
$("stop").onclick = () => { if (ctrl) ctrl.abort(); };
textEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("go").click(); }
});
