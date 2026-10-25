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
  resident weights, 1 CPU core) and streamed back sentence by sentence as 16 kHz PCM.

Inspired by the [PrimeTTS-Streaming](https://huggingface.co/spaces/Luigi/PrimeTTS-Streaming)
demo — except inference runs on real Jetson hardware instead of in the browser.

## If the demo is offline

The page calls a tunnel URL (`GATEWAY_URL` in `app.js`) that rotates whenever the tunnel
restarts. The gateway host and the Nano must both be up. To re-point the demo, edit `app.js`
and push — no rebuild needed.
