#!/usr/bin/env python3
"""Build the demo input wav: zh-TW TTS output from PrimeTTS v1b_8k, resampled to 16 kHz.

Source: tests/golden/zh_greeting/wav.npy — the float32 8 kHz output of the golden
onnxruntime run of model/v1b_8k (same weights the Nano runtime loads; bench/cpu/RESULTS.md
§1 shows the Nano C runtime reproduces this wav to 2.1e-5 relative error, so this is
real zh-TW TTS speech, not synthetic tones). 2.608 s, "您好，歡迎…".

Resampling: 2× linear interpolation 8 kHz -> 16 kHz (no anti-alias filter needed for
up-sampling; the 8–16 kHz band is simply interpolated). Peak is scaled to 0.5 (-6 dBFS)
so the file looks like a phone-mic level signal for the ASR leg.

Run: .venv/bin/python server/make_input_wav.py
"""
from __future__ import annotations

import argparse
import wave

import numpy as np

SRC_DEFAULT = "tests/golden/zh_greeting/wav.npy"
OUT_DEFAULT = "server/input_zh_greeting_16k.wav"
SR_IN = 8000
SR_OUT = 16000
PEAK = 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    w = np.load(args.src).reshape(-1).astype(np.float64)
    n_out = int(round(w.size * SR_OUT / SR_IN))
    up = np.interp(np.linspace(0, w.size - 1, n_out), np.arange(w.size), w)
    peak = float(np.max(np.abs(up)))
    up = up * (PEAK / peak) if peak > 0 else up
    pcm = np.clip(up * 32767.0, -32768, 32767).astype("<i2").tobytes()
    with wave.open(args.out, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR_OUT)
        f.writeframes(pcm)
    print(
        f"{args.src} ({w.size} samples @ {SR_IN} Hz, peak {peak:.3f}) -> {args.out} "
        f"({n_out} samples @ {SR_OUT} Hz, {n_out / SR_OUT:.3f} s, peak {PEAK})"
    )


if __name__ == "__main__":
    main()
