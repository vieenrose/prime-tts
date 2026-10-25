// Regression test for the demo player (no browser needed):
//   node demo/test/playback_e2e.mjs demo/app.js demo/player.worklet.js <gateway-url>
// Runs the shipped makeS16Decoder() + player.worklet.js on a simulated audio clock, with synthetic
// long streams and a live gateway stream (real chunk sizes and arrival times). Every played sample
// must equal the whole-buffer decode of the downloaded bytes, with zero gaps and a drain at the end.
// End-to-end playback check of the shipped client code (decoder + worklet), no browser needed:
//  1. synthetic long streams delivered faster than real time (FIFO growth, no overwrite/skip)
//  2. live tunnel stream: real chunk sizes AND real arrival times, decoded + played on a simulated
//     128-sample audio clock exactly as the page does (start after 0.4 s banked, then "end").
//  Pass = every played sample equals the whole-buffer decode, zero gaps, drain reported at the end.
import fs from 'fs';
const [appPath, wkPath, url] = process.argv.slice(2);
const app = fs.readFileSync(appPath, 'utf8');
const toFloat = new Function(app.match(/function toFloat\(bytes\) \{[\s\S]*?\n\}/)[0] + '; return toFloat;')();
const makeS16Decoder = new Function(app.match(/function makeS16Decoder\(\) \{[\s\S]*?\n\}/)[0] + '; return makeS16Decoder;')();
let Proc;
new Function('sampleRate', 'AudioWorkletProcessor', 'registerProcessor', fs.readFileSync(wkPath, 'utf8'))(
  16000, class { constructor() { this.port = { onmessage: null, msgs: [], postMessage(m) { this.msgs.push(m); } }; } },
  (n, c) => { Proc = c; });
const SR = 16000, PREBUF = 6400, BLK = 128;

// events: [{t: seconds, f: Float32Array}] ; returns played samples + worklet messages
function play(events, total) {
  const p = new Proc(); const out = [new Float32Array(BLK)]; const played = [];
  let k = 0, banked = 0, started = false, t = 0, ended = false;
  while (t < 300) {
    while (k < events.length && events[k].t <= t) { p.port.onmessage({ data: events[k].f }); banked += events[k].f.length; k++; }
    if (!started && (banked >= PREBUF || k === events.length)) { p.port.onmessage({ data: 'start' }); started = true; }
    if (!ended && k === events.length) { p.port.onmessage({ data: 'end' }); ended = true; }
    if (started) { p.process([], [out]); for (const v of out[0]) played.push(v); }
    if (p.port.msgs.some((m) => m && m.drained)) break;
    t += BLK / SR;
  }
  return { played: Float32Array.from(played), msgs: p.port.msgs };
}
function verdict(name, ref, { played, msgs }) {
  const gaps = Math.max(0, ...msgs.filter((m) => m && m.underrunSamples).map((m) => m.underrunSamples));
  let bad = 0; for (let i = 0; i < ref.length; i++) if (played[i] !== ref[i]) bad++;
  const drained = msgs.some((m) => m && m.drained);
  console.log(`${name.padEnd(42)} ${bad === 0 ? 'EXACT' : bad + ' wrong'} | gaps ${(1000 * gaps / SR).toFixed(0)} ms | drained ${drained} | played ${(played.length / SR).toFixed(2)}s of ${(ref.length / SR).toFixed(2)}s`);
}
for (const [secs, deliver] of [[5, 1.2], [10.93, 2.6], [30, 7], [120, 25]]) {
  const N = Math.round(secs * SR), src = new Float32Array(N);
  for (let i = 0; i < N; i++) src[i] = Math.fround(Math.sin(i * 0.05) * 0.3 + 0.001 * (i % 97));
  const ev = []; for (let s = 0; s < N; s += 4096) ev.push({ t: (s / N) * deliver, f: src.slice(s, s + 4096) });
  verdict(`synthetic ${secs} s delivered in ${deliver} s`, src, play(ev, N));
}
for (const text of ['您好,幫您轉接給 Jason 王經理,他的分機是 2580。',
                    '今天天氣很好，我們一起去公園散步吧。公園裡有很多花，非常漂亮。下午我們可以去喝咖啡，然後回家休息。晚上再一起吃晚餐。']) {
  const t0 = performance.now();
  const r = await fetch(url + '/v1/audio/speech', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'primetts-v21', input: text, voice: '0', response_format: 'pcm', stream_format: 'audio' }) });
  const reader = r.body.getReader(), raw = [], ev = [], dec = makeS16Decoder();
  for (;;) { const { done, value } = await reader.read(); if (done) break; raw.push(value); ev.push({ t: (performance.now() - t0) / 1000, f: dec(value) }); }
  const all = new Uint8Array(raw.reduce((a, c) => a + c.length, 0)); let o = 0; for (const c of raw) { all.set(c, o); o += c.length; }
  const odd = raw.filter((c) => c.length % 2).length;
  verdict(`live ${text.length}ch (${raw.length} chunks, ${odd} odd)`, toFloat(all), play(ev, all.length / 2));
}
