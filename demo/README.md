---
title: PrimeTTS v2.1 live on Jetson Nano
emoji: 🔊
colorFrom: green
colorTo: indigo
sdk: static
pinned: false
short_description: zh-TW+English TTS synthesized live on a Jetson Nano Gen1
models:
  - Luigi/PrimeTTS
---

# 🔊 PrimeTTS v2.1 · live on a Jetson Nano Gen1

Web demo for the custom C/CUDA PrimeTTS v2.1 runtime in
[vieenrose/prime-tts](https://github.com/vieenrose/prime-tts/tree/custom-runtime)
(`runtime/v21/`, see `docs/server.md`). Type zh-TW / English text, hear speech.

- **Demo site + client here** (static page, Web Audio streaming playback).
- **Text frontend** on the gateway host (g2pw bopomofo + g2p_en, same `model/scripts/`
  frontend the training used).
- **Waveforms synthesized remotely on one Jetson Nano** (`pt_v21_serve`, 8 warm workers,
  resident weights, 1 CPU core). One request = one continuous utterance: the encoder runs once and
  the decoder is streamed in overlapping windows, so prosody never restarts mid-sentence and audio
  still starts early. The page reassembles the PCM byte stream and plays it through an AudioWorklet FIFO at 16 kHz,
  verified in Chrome to render exactly the samples of the downloadable WAV.

Inspired by the [PrimeTTS-Streaming](https://huggingface.co/spaces/Luigi/PrimeTTS-Streaming)
demo — except inference runs on real Jetson hardware instead of in the browser.

## Latency (measured, 1 core, gateway in the loop)

| request | first audio | total |
|---|---|---|
| 12-char sentence | 0.44 s | 0.64 s |
| 58-char paragraph (10.9 s of audio) | 1.45 s | 2.61 s |
| 4 concurrent paragraphs | 1.5-5.1 s | 6.5 s wall (6.7x realtime) |

## If the demo is offline

The page calls a tunnel URL (`GATEWAY_URL` in `app.js`) that rotates whenever the tunnel
restarts. The gateway host and the Nano must both be up. To re-point the demo, edit `app.js`
and push — no rebuild needed.
