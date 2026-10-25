/* Streaming PCM player for the PrimeTTS Nano demo.
 *
 * The page opens its AudioContext at the 16 kHz source rate (and falls back to whole-buffer
 * playback if the browser refuses), so this processor never resamples: it is a FIFO that copies
 * samples 1:1 to the output, and the browser converts to the device rate.
 *
 * The FIFO grows instead of wrapping. The Nano delivers audio several times faster than real time
 * (a 10.9 s paragraph arrives in ~2.6 s); the earlier fixed 8.2 s ring overwrote audio that had not
 * been played yet (measured: 78% of a 10.9 s paragraph wrong).
 */
class PrimeTTSPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(1 << 16);
    this.r = 0;                 // read index: buf[r, w) is queued audio
    this.w = 0;                 // write index
    this.started = false;
    this.ended = false;         // the stream is complete: running dry is the end, not a gap
    this.drainSent = false;
    this.underrunSamples = 0;
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d === "start") this.started = true;
      else if (d === "end") this.ended = true;
      else if (d === "stop") this.started = false;
      else this.push(d);
    };
  }

  push(f) {
    if (this.w + f.length > this.buf.length) {
      const queued = this.w - this.r;
      let cap = this.buf.length;
      while (cap < queued + f.length) cap *= 2;
      if (cap === this.buf.length) {
        this.buf.copyWithin(0, this.r, this.w);          // compact in place
      } else {
        const nb = new Float32Array(cap);
        nb.set(this.buf.subarray(this.r, this.w));       // grow, keeping every queued sample
        this.buf = nb;
      }
      this.w = queued;
      this.r = 0;
    }
    this.buf.set(f, this.w);
    this.w += f.length;
  }

  process(inputs, outputs) {
    const out = outputs[0][0];
    if (!this.started) {
      out.fill(0);
      return true;
    }
    const n = Math.min(this.w - this.r, out.length);
    out.set(this.buf.subarray(this.r, this.r + n));
    this.r += n;
    if (n < out.length) {
      out.fill(0, n);
      if (!this.ended) {          // ran dry mid-stream: report the inserted silence
        this.underrunSamples += out.length - n;
        this.port.postMessage({ underrunSamples: this.underrunSamples });
      }
    }
    if (this.ended && this.r === this.w && !this.drainSent) {
      this.drainSent = true;      // last sample played: the page may tear the node down now
      this.port.postMessage({ drained: true });
    }
    return true;
  }
}

registerProcessor("primetts-player", PrimeTTSPlayer);
