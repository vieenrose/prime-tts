/* Continuous 16 kHz -> device-rate player for the PrimeTTS Nano demo.
 *
 * Why this exists: scheduling many small AudioBufferSourceNodes makes the browser resample EACH
 * buffer independently (16 kHz -> 44.1/48 kHz), and its resampler restarts at every buffer edge.
 * Every cut therefore lands mid-speech as a discontinuity (4/s at 0.25 s pieces). One big buffer
 * has no such seams -- which is why "buffered" mode sounds clean -- but it kills streaming.
 *
 * This worklet keeps a ring buffer of input-rate samples and interpolates CONTINUOUSLY across
 * chunk boundaries, so the output stream has no internal edges, and playback can still begin
 * after a short prebuffer. Catmull-Rom cubic interpolation keeps the imaging of a 3x upsample
 * well below audibility (linear interpolation would add a faint HF sheen).
 */
class PrimeTTSPlayer extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = (options && options.processorOptions) || {};
    this.inRate = o.inRate || 16000;
    this.ratio = this.inRate / sampleRate;   // input samples consumed per output sample
    this.cap = 1 << 17;                      // 131072 input samples ~ 8.2 s at 16 kHz
    this.buf = new Float32Array(this.cap);
    this.w = 0;        // absolute write index (monotonic)
    this.pos = 0;      // absolute fractional read position
    this.started = false;
    this.underrunSamples = 0;
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d === "start") { this.started = true; return; }
      if (d === "stop") { this.started = false; return; }
      const f = d;     // Float32Array at the input rate; grows the read pointer if it laps us
      for (let i = 0; i < f.length; i++) this.buf[(this.w + i) % this.cap] = f[i];
      this.w += f.length;
      if (this.w - this.pos > this.cap) this.pos = this.w - this.cap;   // overflow guard
    };
  }

  at(i) {
    const cap = this.cap;
    return this.buf[((i % cap) + cap) % cap];
  }

  process(inputs, outputs) {
    const out = outputs[0][0];
    if (!this.started) {
      out.fill(0);
      return true;
    }
    let starved = 0;
    for (let i = 0; i < out.length; i++) {
      if (this.w - this.pos < 3) {           // no lookahead left: silence, count it
        out[i] = 0;
        starved++;
        continue;
      }
      const i0 = Math.floor(this.pos), t = this.pos - i0;
      const p0 = this.at(i0 - 1), p1 = this.at(i0), p2 = this.at(i0 + 1), p3 = this.at(i0 + 2);
      out[i] = p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 +
                   t * (3 * (p1 - p2) + p3 - p0)));
      this.pos += this.ratio;
    }
    if (starved) {
      this.underrunSamples += starved;
      this.port.postMessage({ underrunSamples: this.underrunSamples });
    }
    return true;
  }
}

registerProcessor("primetts-player", PrimeTTSPlayer);
