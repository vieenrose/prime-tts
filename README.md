# Gemini Live + on-device PrimeTTS on Jetson Nano

Goal: a real-time zh-TW/English voice agent where the caller's speech goes to Gemini and the
assistant's voice comes from a **local** TTS model on a Jetson Nano, at much lower cost than Gemini's
native speech-to-speech, serving **8 concurrent conversations**.

## Two premises had to be corrected (both verified, not assumed)

1. **"PrimeTTS v2.1 8k" does not exist.** The HF repo has `v21_mbistft_16k` (37.9M, 16 kHz) and 8 kHz
   only in the v1 family (`v1b_8k`, 4.09M). This project therefore builds both tiers:
   `v1b_8k` (8 kHz, CPU, the one that can actually do 8 streams) and `v2.1 16k` (GPU, quality tier;
   8 kHz delivery = downsample on the server, compute unchanged).
2. **`gemini-3.8-live` cannot output text.** Native-audio Live models accept only the AUDIO response
   modality, and they always generate their spoken reply - which is billed as audio output even when
   you throw it away. Measured: a "speech-in / text-out" leg through Live costs **more** than native
   speech-to-speech ($0.173 vs $0.027 per caller-minute). The cheap path is audio-in straight to a
   text model: **$0.0017-0.0026 per caller-minute, 10-16x cheaper than native** (see `docs/cost.md`).

## What is built and measured

Phase 1 - v1b_8k from-scratch CPU runtime (aarch64 NEON assembly), verified on the Nano:
- parity vs onnxruntime, 6 cases (zh-TW / en / code-mix / entity / random), rel err 1.8e-5..2.0e-4,
  layer-by-layer checks 1e-6..6e-5; streaming synthesis is bit-identical to one-shot
  (`tests/run_tests.sh`, verified on a clean build).
- kernels: 4x16/8x12 NEON tiles, **9.6 GFLOP/s in L1** (82% of the A57 11.8 GFLOP/s core peak),
  6.6-8.6 GFLOP/s on real vocoder shapes, 13-17 GFLOP/s on 4 threads.
- single stream RTF **0.19** (1 thread) / **0.13-0.19** (4 threads) vs the published onnxruntime 0.35.
- **8 concurrent streams: 13.7-16.7 audio-seconds per wall-second**, worst per-stream RTF 0.56-0.71,
  worst lateness +42 ms, peak RSS **79.8 MB** (weights mmap'd and shared by all streams).
  Details: `bench/cpu/RESULTS.md`.

Phase 2 - GPU roofline for v2.1 (measured, `bench/gpu/RESULTS.md`):
- fp32 tiled GEMM **62 GFLOP/s**, fp16-storage+fp32-acc **70 GFLOP/s**, issue ceilings
  88 GFLOP/s fp32 and **168 GFLOP/s half2**; cooperative launch unavailable; launch overhead 26 us.
- v2.1 costs 11.9 GFLOP per audio-second, so **fp32 tops out at ~5 streams; 8 streams need a
  half2-compute runtime** (target ~110 GFLOP/s -> aggregate RTF 0.87).
- Not built yet: the CUDA C runtime for v2.1 (`runtime/kern/*_cuda.cu`), the export/golden for v2.1.

Gemini leg (measured with real API calls, `server/`, `docs/cost.md`, `docs/proto.md`):
- `server/live_asr.py` (Live input transcription), `server/reply_gen.py` (3.8 Flash streaming),
  `server/audio_in_flash.py` (audio-in straight to a text model - the winner),
  `server/tts_client.py` + `server/mock_tts.py` (the Nano TTS wire protocol), `server/pipeline.py`.
- Latency: 3.5 Flash-Lite audio-in -> reply first token **1.3 s**; Nano TTS first audio byte
  **63-140 ms** (1 stream, 4 threads). Live's turn closure needs ~0.8 s of trailing silence.

## Known gaps

- The text frontend (g2pw bopomofo + polyphone BERT, g2p_en, entity normalization) is **not ported**;
  the runtime takes phone/tone/lang ids and the tests used hand-built ids. It must run on the server
  host (a BERT forward pass is not a Nano-scale operation) and the ids travel in the TTS request.
- v2.1 GPU runtime (the quality tier) is unbuilt; roofline says it needs half2 compute + CUDA Graphs
  with T-buckets to reach 8 streams.
- First-audio latency for v1b_8k is bounded by the decoder's bidirectional GRU (whole-sentence mel):
  split Gemini replies into short sentences, or move the decoder to the GPU.
- The Nano is a shared box with ~185 MB free RAM and no sudo; a 204 MB benchmark pushed it into swap
  for ~10 minutes. Keep RSS bounded (the 8-stream server is at 79.8 MB) and add a guard.

## Layout

```
runtime/       C + aarch64 asm runtime (Phase 1 = v1b_8k; Phase 2 = v2.1 CUDA, not started)
tools/         probe_shapes.py (ONNX shape/MAC probe), export_ptk.py (weights -> .ptk)
tests/         golden generation (host, onnxruntime) + kernel/e2e tests (run on the Nano)
bench/cpu/     Nano CPU benchmarks + RESULTS.md        bench/gpu/ GPU roofline + RESULTS.md
server/        Gemini legs + TTS client/mock + pipeline docs/  cost.md, proto.md
spec/ ref/     layer specs, MAC tables, parity oracle   workflows/  subagent lane scripts
```

## Reproduce

```sh
# host (x86, .venv with onnxruntime)
.venv/bin/python tools/export_ptk.py                 # build/v1b8k.ptk + spec/v1b8k_layers.json
.venv/bin/python tests/gen_golden.py                 # tests/golden/*  (onnxruntime goldens)
.venv/bin/python bench/cpu/make_requests.py          # bench/cpu/requests.txt
# Nano (repo copied to ~/primetts)
make -C runtime -j4 && tests/run_tests.sh
B=build/aarch64/bench_cpu; M=build/v1b8k.ptk; Q=bench/cpu/requests.txt
$B $M $Q single ; $B $M $Q mt ; $B $M $Q conc8 ; $B $M $Q admit
# GPU roofline (Nano)
/usr/local/cuda/bin/nvcc -O3 -arch=sm_53 -o sgemm bench/gpu/sgemm.cu && ./sgemm
# Gemini legs (host; key comes from gemini-live-credential.txt, never printed)
set -a; source ./gemini-live-credential.txt; set +a
.venv/bin/python server/audio_in_flash.py --wav server/input_zh_greeting_16k.wav
.venv/bin/python server/pipeline.py --wav server/input_zh_greeting_16k.wav
```

Device access: `ssh picard@100.84.151.116` (no sudo, no apt; CUDA 10.2 at /usr/local/cuda/bin/nvcc).
