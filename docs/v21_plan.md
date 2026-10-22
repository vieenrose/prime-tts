# v2.1 on Jetson Nano — measured feasibility + runtime plan

Target (user directive): PrimeTTS **v2.1** (`model/v21_mbistft_16k/primetts_v21_3voice.onnx`,
16 kHz, 4066 nodes, 29.08M initializer params + 1924 Constant nodes), **dependency-free C + asm +
CUDA C runtime** (no onnxruntime, no ggml, no cuBLAS/cuDNN), **8 parallel real-time requests**,
lowest latency, streaming input and output.

## Measured constraints (all on this Nano, `bench/gpu/RESULTS.md`, `bench/gpu/hgemm.cu`)

| quantity | measured |
|---|---|
| v2.1 compute cost | 19.605 GMAC per T=81 utterance (3.312 s audio) = **11.9 GFLOP per audio-second** |
| 8-stream requirement | **95 GFLOP/s sustained** |
| fp32 tiled GEMM | 62 GF/s (1024³), 62 GF/s at 512x4000x512 |
| fp32 FMA issue peak | 88 GF/s |
| **half2 (fp16 storage, `__hfma2` multiply+accumulate)** | **121 GF/s** (81x256x1536), 151 (324x512x1024), **161 (648x512x1024)**, **168 (1296x512x1024)**, 171 (1024³), 165 (512x4000x512) |
| fp16 storage + fp32 accumulate | 54–74 GF/s |
| half2 accumulate accuracy (K=128, inputs ±0.2) | max 1.43e-3, rms 1.80e-4 |
| kernel launch overhead | 26 µs |
| cooperative launch | unavailable (`cooperativeLaunch = 0`) |
| device bandwidth | ~8.3 GB/s (not binding at these shapes) |
| free RAM | ~165 MB (weights: 117 MB fp32, **75.8 MB fp16**) |

fp32 math cannot reach 95 GFLOP/s. half2 can. So the runtime must compute in half2, and fp16
quality must be measured, not assumed.

## Measured fp16 sensitivity (`tools/fp16_probe.py`, deterministic `noise_scale=0`, T=40, 26880 samples)

The model is stochastic at `noise_scale>0` — comparisons must use `noise_scale=0` or they measure
the sampler, not the quantization (I made that mistake first: the "fp16" number was sampling noise).

| variant (weights rounded to fp16 values) | relL2 | SNDR |
|---|---|---|
| **fp16 activations on conv outputs, fp32 weights** | 4.0e-5 | **87.9 dB** |
| decoder convs only (`g:dec`) | 4.2e-4 | **67.4 dB** |
| encoder convs only (`g:enc`) | 1.9e-2 | 34.3 dB |
| all conv/matmul weights | 2.4e-2 | 32.2 dB |
| every initializer | 7.6e-2 | 22.4 dB |

So: **activations in fp16 are free; the decoder tolerates fp16 weights (67 dB); the encoder does
not (34 dB).** Weight params are ~50/50 enc/dec (14.77M / 13.70M), so the sensitive half is half
the model — keeping the encoder in fp32 is not affordable:

    enc fp32 + dec half2:  11.9*(0.5/62 + 0.5/168) = 0.132 s per audio-second -> 8 streams = 1.06 (misses real time)
    enc HiLo half2 + dec half2: 11.9*(0.5/80 + 0.5/160) = 0.112 -> 8 streams = 0.90 (fits, ~11% headroom)

**Decision: split-fp16 (HiLo) for the encoder, plain half2 for the decoder.** W = W_hi + W_lo with
both halves in fp16 (Markidis-style 2-term split): two half2 GEMMs recover ~fp32 weight precision,
which is what the 34 dB -> ~fp32 measurement predicts. fp16 activations throughout (87.9 dB).
Chunked-K promotion (accumulate 16–32 taps in half2, then into an fp32 accumulator) keeps the
half2 accumulate error from growing with K; its speed cost is being measured (target ≥130 GF/s,
which raises the headroom to ~1.5x).

## Runtime shape

1. **Batching = the 8 streams share every kernel.** The GEMM/conv M dimension becomes
   `T x streams` (measured shapes 648/1296 hit 161–168 GF/s, i.e. batching *is* the efficiency:
   81-row GEMMs alone only reach 121 GF/s). One kernel launch per op for all 8 requests, streams
   packed per T-bucket (T buckets 32/64/96/128/… padded) so graphs stay reusable.
2. **CUDA Graphs per bucket** (26 µs launch overhead x ~200 ops = 5.2 ms per inference is
   acceptable, but graphs remove it entirely; capture at startup, replay per bucket — cooperative
   launch is unavailable so graph capture is how the whole network stays one submission).
3. **fp16 weights, fp32 for embeddings/LayerNorm/softmax/flow scales/iSTFT magnitude head**
   (the 22 dB vs 32 dB gap says these must stay fp32; they are <1% of MACs so cost is time, not FLOPs).
4. **Streaming output**: the model is sentence-based, so synthesize per *sentence/chunk* and emit
   40 ms PCM frames as soon as the iSTFT head produces them (protocol in `docs/proto.md` is
   unchanged; the client already starts playback on the first frame). First-audio latency target
   = first chunk (~20-24 frames => ~0.32-0.38 s audio) at RTF 0.112 => ~40-50 ms compute.
5. **Streaming input**: text arrives incrementally from the caller's conversation turn. The
   sentence-segmenter feeds chunks; the encoder is bidirectional over a chunk, so chunk boundaries
   are sentence boundaries (no cross-chunk state in the encoder) and the duration allocator works
   per chunk. Acoustic cross-chunk state is limited to the iSTFT overlap window (keep 2 frames of
   history per stream) and the stream's speaker embedding.
6. **Memory budget** (Nano has ~165 MB free, no sudo): 75.8 MB fp16 weights mmap'd + HiLo encoder
   half (fp16, +14.8 MB) + per-stream activations. Target: RSS < 130 MB at 8 streams; the fp32
   container (117 MB) will not fit with 8 streams.

## Milestones (each gated by measurement)

| # | deliverable | gate |
|---|---|---|
| M1 | `tools/export_v21.py` -> `.ptk` (fp16 + HiLo + fp32 groups) + goldens + layer spec | unpack/round-trip check == ONNX weights within 1e-6; layer-by-layer intermediates vs onnxruntime |
| M2 | CUDA C single-stream runtime | audio parity vs golden: SNDR >= 60 dB (measured fp16 ceiling for dec-fp16 is 67 dB) |
| M3 | 8-stream batching + CUDA graphs | aggregate RTF <= 0.9, RSS <= 130 MB, first-audio < 200 ms |
| M4 | streaming sentence/chunk pipeline | first-audio latency and per-chunk continuity (no clicks at chunk joins) |
| M5 | admission/latency policy | worst lateness over a full run with Poisson-arriving requests; buffer size recommendation |

## Known risks

- The encoder is the accuracy-critical half *and* the expensive half; if HiLo proves insufficient
  (measured by re-running `fp16_probe.py --variant g:enc` style checks against the runtime), the
  fallback is 6 streams (measured: 11.9 GFLOP/audio-s x 6 / 132 GF/s = 0.54 aggregate RTF) or
  encoder-side structure work (attention sparsity, T-bucket padding).
- `noise_scale>0` makes output stochastic: parity tests must use `noise_scale=0`.
- v2.1 has 1924 `Constant` nodes: constants must be exported into the container too, not just
  initializers (the v1 exporter only walks initializers).
- Sustained-clock behaviour under 8-stream load is not yet measured (a burst GEMM rate of 168 GF/s
  is not a promise); lane measurement in progress.
