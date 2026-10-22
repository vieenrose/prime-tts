# PrimeTTS v1b_8k — from-scratch CPU runtime on Jetson Nano: measured results

Everything below was measured on the target unless marked "host". Raw logs: `bench/cpu/logs/`.

**Device:** Jetson Nano gen-1 (`ssh picard@100.84.151.116`).
- 4× Cortex-A57 ARMv8.0, fp32 NEON, L4T R32.5.1.
- `schedutil` governor, MAXN power mode.
- gcc 8.4.
- About 180 MB RAM available while testing; other services (jtop, a python server) were running.
- CPU 37 °C after the full run (`logs/thermal_after.txt`).

**Model:** `model/v1b_8k` (encoder + host regulator + decoder with a bidirectional GRU + Snake-HiFiGAN), 8 kHz, 4.09 M params.
- Exported once to `build/v1b8k.ptk` (16.4 MB fp32, mmap'd and shared by all streams).

**Baseline:** onnxruntime is **not installable** on this Nano (Python 3.6, no wheels, no sudo). The reference point is therefore the model card's published number: **RTF 0.35, 1 thread, onnxruntime, same Nano**. No onnxruntime run was made here.

## Commands

```sh
# dev host (x86, .venv with onnxruntime): export weights, golden + numpy vectors, requests file
.venv/bin/python tools/export_ptk.py                 # -> build/v1b8k.ptk + spec/v1b8k_layers.json
.venv/bin/python tests/gen_golden.py                 # -> tests/golden/<case>/*.npy (onnxruntime outputs)
.venv/bin/python tests/gen_kernel_vectors.py         # -> tests/kern/*  (numpy kernel references)
.venv/bin/python bench/cpu/make_requests.py          # -> bench/cpu/requests.txt
# copy repo to the Nano (~/primetts), then on the Nano:
make -C runtime -j4                                  # -> build/aarch64/{test_kernels,test_e2e,bench_cpu}
tests/run_tests.sh                                   # kernels (numpy + reference) and end-to-end parity
B=build/aarch64/bench_cpu; M=build/v1b8k.ptk; Q=bench/cpu/requests.txt
$B $M $Q kernels ; PT_PROF=1 $B $M $Q single ; $B $M $Q mt ; $B $M $Q conc ; $B $M $Q conc8 ; $B $M $Q admit
```

Benchmark requests (`bench/cpu/requests.txt`): all six golden-test inputs, built by hand from the symbol table.
- zh-TW greeting, English "thank you for calling", zh/en code-mix booking, zh phone number (speaker 1), and two random sequences (1.55 s to 8.48 s of audio).
- The concurrency runs use the first five.

## 1. Parity vs onnxruntime (on the Nano, asm kernels) — `logs/run_tests_nano.txt`

Waveform error is measured against onnxruntime's fp32 output on x86. "rel" means max|err| / max|ref|.

| case | frames | samples | max abs err | rel (max) | rel (L2) | streamed vs single chunk |
|---|---|---|---|---|---|---|
| zh_greeting (zh-TW) | 163 | 20864 | 8.3e-06 | 2.1e-05 | 1.8e-05 | bit-identical |
| en_thanks (English) | 97 | 12416 | 1.2e-05 | 2.9e-05 | 2.2e-05 | bit-identical |
| mix_booking (code-mix) | 149 | 19072 | 1.2e-05 | 2.4e-05 | 1.7e-05 | bit-identical |
| zh_phone_sid1 (speaker 1) | 138 | 17664 | 1.3e-05 | 4.8e-05 | 2.5e-05 | bit-identical |
| random_T90 | 530 | 67840 | 7.0e-05 | 2.0e-04 | 6.9e-05 | bit-identical |
| random_T23 | 141 | 18048 | 1.8e-05 | 5.5e-05 | 2.3e-05 | bit-identical |

All 6 cases pass the target of < 1e-3, in both single-chunk and streaming mode. Durations, and therefore frame counts, match exactly.

Worst per-layer relative error over all cases: encoder blocks ≤ 9e-7, conditioned 4.9e-7, decoder blocks ≤ 3.9e-6, GRU 6.9e-5, mel 7.2e-6, vocoder stage outputs ≤ 1.6e-4.

All kernels also pass against the numpy vectors (11 real layer shapes plus snake, SiLU, tanh, GLU, depthwise and LayerNorm+residual; rel ≤ 1.4e-6) and against the C reference/direct-loop tests (`logs/test_kernels.txt`).

## 2. Speed, single stream — `logs/single.txt`, `logs/mt.txt`

| mode | RTF (all 6 requests, 19.5 s audio) | vs published ORT 0.35 | first 48 ms chunk (short sentences) | first chunk, 8.5 s request |
|---|---|---|---|---|
| 1 thread | **0.186** | **1.88× faster** | 100–140 ms | 414 ms |
| 4 threads (intra-op split) | **0.088** | 4.0× | 63–72 ms | 175 ms |

- **Streaming overhead** (chunked vs one-chunk synthesis): −0.3 % to −6 % with 1 thread, i.e. faster. Carried left context means nothing is recomputed, and the smaller windows stay in cache.
- **First chunk vs mel:** the first audio arrives about 21 ms after the mel is ready.

Phase profile, 1 thread, PT_PROF=1:

| phase | share |
|---|---|
| vocoder conv1 (+ fused snake) | 34.2 % |
| vocoder conv2 (+ fused residual) | 31.4 % |
| decoder 6 blocks | 13.1 % |
| vocoder snake1 passes | 5.3 % |
| GRU | 3.5 % |
| mel head + postnet | 3.2 % |
| encoder | 2.8 % |
| ConvT | 2.3 % |
| decoder pre-nets | 2.2 % |
| everything else | < 2 % |

## 3. Concurrency — `logs/conc.txt` (N = 1..8 in one process), `logs/conc8_rss.txt` (N = 8, fresh process)

Setup: each stream is its own request, and worker threads are pinned round-robin to the 4 cores. There are 3 requests per stream slot, all arriving at t = 0, with no admission throttling. There are 4 shared encoder/decoder workspaces, and vocoder chunks start at 3 frames and double up to 32.

"Lateness" assumes playback starts at the first chunk; it is the worst time any chunk arrived after it was due (≤ 0 means no underrun).

| N streams | aggregate audio-s / wall-s | mean RTF per request | max RTF per request | mean first chunk | max first chunk | worst lateness |
|---|---|---|---|---|---|---|
| 1 | 5.26 | 0.191 | 0.196 | 126 ms | 143 ms | 0 ms |
| 2 | 8.47 | 0.195 | 0.202 | 184 ms | 428 ms | 0 ms |
| 3 | 12.31 | 0.204 | 0.212 | 173 ms | 443 ms | 0 ms |
| 4 | 14.28 | 0.222 | 0.270 | 210 ms | 489 ms | 0 ms |
| 5 | 14.85 | 0.261 | 0.426 | 279 ms | 1033 ms | 0 ms |
| 6 | 13.78 | 0.363 | 0.538 | 371 ms | 1160 ms | +31 ms |
| 7 | 16.60 | 0.381 | 0.513 | 397 ms | 1082 ms | 0 ms |
| **8** | **16.66** | **0.441** | **0.556** | 434 ms | 1106 ms | +3 ms |
| 8 (fresh process) | 14.82 | 0.447 | 0.623 | 436 ms | 939 ms | 0 ms |

- **8 streams are real time:** every request in the 8-stream runs finished at RTF ≤ 0.62, a ≥ 1.6× margin. Aggregate throughput is 14.8–16.7 audio-s/s, against 8 needed.
- **Burst first-chunk latency:** with all 8 requests arriving in the same instant, the first chunk takes up to about 1.1 s, because 8 encoders and decoders queue behind each other.
- **Underruns:** the worst lateness seen was +31 ms (once, at N = 6), so a client pre-roll of about 40 ms covers every case measured.
- **Peak RSS of the 8-stream server: 79.8 MB.** That is the 16.4 MB model mmap, 8 × 1.85 MB stream state and 4 × 9.0 MB workspaces. Before the workspace pool it was 204 MB and swapped at N ≥ 5.

**Admission policy** (`logs/admit.txt`):
- Capacity is 1 / RTF_single = 5.41 audio-s/s per core, with safety factor 0.85 and a cap of 8 streams.
- 16 requests arriving every 100 ms gave 14.79 audio-s/s, max RTF 0.478, mean first chunk 361 ms (max 931 ms), worst lateness +1 ms, max queue wait 245 ms, and 16/16 completed.

## 4. Kernel throughput (GFLOP/s, fp32, epilogue included) — `logs/kernels.txt`

| layer shape (real model layers) | 4×16, no packing | 8×12, no packing | **8×12 + packed B (default)** | default, 4 threads |
|---|---|---|---|---|
| voc stage0 48×48 K11 d5 + snake | 6.67 | 6.60 | **7.28** | 21.2 |
| voc stage0 48×48 K3 | 6.65 | 6.76 | **7.82** | 23.1 |
| voc stage1 24×24 K7 d3 + snake | 6.52 | 6.68 | **6.62** | 21.3 |
| voc stage2 12×12 K11 | 8.98 | 8.61 | **8.63** | 27.0 |
| voc stage3 6×6 K7 d3 + snake | 4.31 | 4.27 | **4.39** | 13.1 |
| voc ups0 (8 phases) 384×96 K2 | 6.36 | 6.62 | **8.18** | 20.2 |
| voc conv_pre 96×80 K7 + snake | 6.56 | 6.59 | **7.36** | 14.1 |
| dec ff0 504×168 + SiLU | 4.42 | 4.55 | **6.28** | 19.5 |
| dec ff3 168×504 | 4.84 | 4.57 | **7.39** | 17.2 |
| dec GRU input proj 504×168 | 5.37 | 5.53 | **8.25** | 24.5 |
| dec postnet 168×168 K5 + tanh | 5.19 | 5.39 | **6.90** | 16.3 |
| enc ff0 672×168 (T = 40) | 4.39 | 4.27 | **5.52** | 15.7 |

- **Micro-kernels with operands in L1:** k4x16 at 9.64 and k8x12 at 9.10 GFLOP/s. The A57 peak is 8 flop/cycle, 11.8 GFLOP/s at 1.479 GHz, so that is 82 %.
- **Compared with the earlier naive 4×4 NEON GEMM** (3.74 GFLOP/s on 1 core, 9.78 on 4): the conv path now runs 4.4–8.6 GFLOP/s on 1 core and 13–27 on 4.
- **Activations (asm):** snake 212, SiLU 85 and tanh 57 Melem/s.

## Design notes (what made the difference)

- **One tap-major implicit-GEMM conv kernel.** Each (tap, input channel) pair is one byte-offset entry, so dilation, padding and channel slices cost nothing. Linear layers are K = 1 convs, and each ConvTranspose is split into S phases of a 2-tap conv. Bias is applied in-kernel; activation, scale and residual are applied to the 4/8-row strip while it is still in L1.
- **Packed B panels** (contiguous tile-major) fix DTLB thrash on the 168–504-channel linears (+40–50 %) and help the wide vocoder convs (+10–24 %).
- **Exact streaming vocoder.**
  - Each layer computes only the new columns of a chunk, and the left context it needs is carried as per-tensor history (`vhist`).
  - Columns outside [0, N) are forced to zero, matching onnxruntime's zero padding.
  - The result is bit-identical to one-chunk synthesis and does no recompute. An earlier overlap-recompute design computed 1.9–2.4× the stage-0 columns (host measurement).
- **local_ctx MLP at token rate.** It runs on T token columns instead of F frame columns (about 6× fewer FLOPs, same arithmetic per column), because it is a per-column function of `repeat(...)`.
- **Memory:** encoder/decoder workspaces are pooled (one per core); a stream keeps only its mel and vocoder state (1.85 MB).
