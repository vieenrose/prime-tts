# v1b_16k (16 kHz) on the from-scratch CPU runtime - measurements

All numbers here are measured on the Jetson Nano (`picard@100.84.151.116`, repo copy `~/primetts`):
4 x Cortex-A57 (`ARMv8 Processor rev 1`) at 1479 MHz, governor `schedutil`, kernel 4.9.201-tegra,
`make -C runtime -j4` (native aarch64 NEON kernels). MemTotal 4059 MB; MemAvailable 182-202 MB during
the runs; SwapTotal 10175 MB with **~1230 MB already used by other processes before any run** (this box
is shared), so the memory guard below tracks *our* swap delta as well as the absolute value.

One binary serves both variants: the stage channels, strides, pads, kernels, hop size, sample rate and
frame limits all come from the container (`voc.geom`, `model.meta`). Nothing in the runtime branches on
"16k". Repro commands at the end; raw logs in `bench/cpu/logs/`.

## 1. Packing gate (host, `.venv/bin/python tools/export_ptk.py ...`)

The gate unpacks the packed ConvTranspose tensors (`s` phases x 2 taps, rows phase-major
`(phase*Cout4 + c)`, taps in the packed `K` axis) and compares against the original ONNX weight+bias.
Strict asserts: `kernel == 2*stride`, `2*pad == stride`, `ups.0 Cin == conv_pre Cout`,
`ups.i Cin == Cout(i-1)`. Gate result = **max|W - unpack| = 0 and max|B - unpack| = 0 for all 8 stages**.

```
packing gate for v1b_8k:
  convT gate voc.ups.0    Cin= 96 Cout= 48 stride=8 K=16 pad=4  max|W-unpack|=0.0e+00 max|B-unpack|=0.0e+00  OK
  convT gate voc.ups.1    Cin= 48 Cout= 24 stride=4 K=8 pad=2  max|W-unpack|=0.0e+00 max|B-unpack|=0.0e+00  OK
  convT gate voc.ups.2    Cin= 24 Cout= 12 stride=2 K=4 pad=1  max|W-unpack|=0.0e+00 max|B-unpack|=0.0e+00  OK
  convT gate voc.ups.3    Cin= 12 Cout=  6 stride=2 K=4 pad=1  max|W-unpack|=0.0e+00 max|B-unpack|=0.0e+00  OK
packing gate for v1b_16k:
  convT gate voc.ups.0    Cin=160 Cout= 80 stride=8 K=16 pad=4  max|W-unpack|=0.0e+00 max|B-unpack|=0.0e+00  OK
  convT gate voc.ups.1    Cin= 80 Cout= 40 stride=8 K=16 pad=4  max|W-unpack|=0.0e+00 max|B-unpack|=0.0e+00  OK
  convT gate voc.ups.2    Cin= 40 Cout= 20 stride=2 K=4 pad=1  max|W-unpack|=0.0e+00 max|B-unpack|=0.0e+00  OK
  convT gate voc.ups.3    Cin= 20 Cout= 10 stride=2 K=4 pad=1  max|W-unpack|=0.0e+00 max|B-unpack|=0.0e+00  OK
```
So the packed ConvTranspose layout was already stride-generic; the factor-of-2 was runtime-side
(hardcoded stage rates + hop). `model.meta` in the container carries `[sample_rate, hop_size,
abs_frame_bins, max_frames]`; the runtime asserts `hop_size == product of the four strides`.

## 2. Parity vs onnxruntime (Nano, `tests/run_tests.sh`)

Worst per-layer relative error over all taps of all 6 cases, and the end-to-end waveform numbers.
Test gate is `rel_max < 1e-3 && rel_l2 < 1e-3` plus the streamed == single-chunk check.

| container | cases | worst per-layer rel | end-to-end rel_max (min..max over cases) | rel_l2 (max) | streamed vs single-chunk |
|---|---|---|---|---|---|
| `build/v1b8k.ptk` vs `tests/golden` | 6/6 PASS | 1.638e-04 | 2.07e-05 .. 2.00e-04 | 6.85e-05 | 0.000e+00 (bit-identical) |
| `build/v1b16k.ptk` vs `tests/golden16` | 6/6 PASS | 1.271e-04 | 5.36e-05 .. 1.21e-04 | 3.32e-05 | 0.000e+00 (bit-identical) |

Per case (16 kHz, single chunk and streamed, identical numbers):

| case | F | samples | maxabs | meanabs | rel_max | rel_l2 | status |
|---|---|---|---|---|---|---|---|
| zh_greeting | 168 | 43008 | 2.912e-05 | 6.370e-07 | 6.437e-05 | 3.330e-05 | PASS |
| en_thanks | 96 | 24576 | 5.473e-05 | 7.246e-07 | 1.209e-04 | 3.094e-05 | PASS |
| mix_booking | 146 | 37376 | 2.722e-05 | 9.303e-07 | 5.358e-05 | 2.651e-05 | PASS |
| zh_phone_sid1 | 125 | 32000 | 2.907e-05 | 1.045e-06 | 8.905e-05 | 2.813e-05 | PASS |
| random_T90 | 544 | 139264 | 3.485e-05 | 1.163e-06 | 7.911e-05 | 3.318e-05 | PASS |
| random_T23 | 154 | 39424 | 2.481e-05 | 1.018e-06 | 6.149e-05 | 2.911e-05 | PASS |

16 kHz layer ladder for `zh_greeting` (single chunk): `enc_blk*` 1.6e-07..3.4e-07, `dec_blk*`
7.1e-07..1.9e-06, `mel` 2.83e-06, `voc_ups0` 3.19e-06, `voc_stage0` 1.51e-05, `voc_ups1` 1.95e-05,
`voc_stage1` 4.37e-05, `voc_ups2` 3.98e-05, `voc_stage2` 6.43e-05, `voc_ups3` 6.67e-05,
`voc_stage3` 6.41e-05. The tap lengths match the goldens exactly (stage rates 8/64/128/256 per frame).

## 3. v1b_8k regression check (same host state, measured twice)

```
build/aarch64/bench_cpu build/v1b8k.ptk bench/cpu/requests.txt single    # run a, then run b
build/aarch64/bench_cpu build/v1b8k.ptk bench/cpu/requests.txt mt        # 4 threads
build/aarch64/bench_cpu build/v1b8k.ptk bench/cpu/requests.txt conc8
```

| section | measured | README baseline (pre-change) |
|---|---|---|
| single, 1 thread, RTF | **0.185** (a) / **0.187** (b) | 0.19 |
| single, 4 threads, RTF | **0.088** | 0.13-0.19 |
| conc8 audio-s per wall-s | **14.41** | 13.7-16.7 |
| conc8 max per-stream RTF | **0.712** | 0.56-0.71 |
| conc8 worst lateness | **+0 ms** | +42 ms |
| conc8 peak RSS | **79.4 MB** | 79.8 MB |

**Verdict: no v1b_8k regression.** The ConvTranspose packing was not changed by this fix - it stays
`s` phases x 2 taps (`K=2`), which the 8-phase stage-0 case already exercised at both variants, so the
micro-kernel call sequence in the vocoder is byte-for-byte the same code path as before. The "0.225"
8k single-stream figure that motivated this investigation is not reproducible on this build (0.185 /
0.187 on two runs, 1.1% run-to-run spread), and it came from a general `K = 2*stride` packing
experiment, not from the geometry-driven runtime.

## 4. v1b_16k kernel throughput (geometry-driven rows, `bench/cpu/logs/kernels16.txt`)

Vocoder rows are generated from the container (reference length 1000 mel frames; stage `s` has
`1000*rate` columns, its ConvT phase grid `1000*rate/stride`):

| layer shape | n cols | 4x16, 1 thr | 8x12, 1 thr, no packing | 8x12+packing K=1 only | 8x12+packing incl. K>1, 4 thr | **8x12+packing incl. K>1, 1 thr** |
|---|---|---|---|---|---|---|
| voc conv_pre 160x80 K7 +snake | 1000 | 6.45 | 6.45 | 6.41 | 20.53 | **7.50** |
| voc ups0 ConvT (8 phases x 2 taps) 640x160 K16 | 1000 | 4.96 | 4.81 | 4.81 | 24.38 | **8.19** |
| voc stage0 resblock conv 80x80 K11 d5 +snake | 8000 | 6.37 | 6.30 | 6.33 | 19.19 | **7.21** |
| voc ups1 ConvT (8 phases x 2 taps) 320x80 K16 | 8000 | 6.08 | 6.06 | 6.08 | 29.19 | **8.26** |
| voc stage1 resblock conv 40x40 K11 d5 +snake | 64000 | 7.35 | 7.04 | 7.03 | 24.04 | **6.96** |
| voc ups2 ConvT (2 phases x 2 taps) 40x40 K4 | 64000 | 6.13 | 5.49 | 5.47 | 23.57 | **6.75** |
| voc stage2 resblock conv 20x20 K11 d5 +snake | 128000 | 7.37 | 7.01 | 7.00 | 23.06 | **6.53** |
| voc ups3 ConvT (2 phases x 2 taps) 24x20 K4 | 128000 | 5.86 | 4.65 | 4.65 | 19.15 | **5.77** |
| voc stage3 resblock conv 12x12 K11 d5 +snake | 256000 | 6.70 | 6.25 | 6.20 | 22.49 | **6.22** |
| dec ff0 504x168 (linear) +silu | 500 | 4.44 | 4.51 | 6.31 | 19.52 | **6.27** |
| dec ff3 168x504 (linear) | 500 | 4.79 | 4.50 | 7.23 | 17.27 | **7.30** |
| dec gru input proj 504x168 | 500 | 5.41 | 5.49 | 8.29 | 25.24 | **8.30** |
| dec postnet 168x168 K5 +tanh | 500 | 5.27 | 5.57 | 5.58 | 16.33 | **6.88** |
| enc ff0 672x168 (linear, T=40) | 40 | 4.45 | 4.43 | 5.61 | 15.49 | **5.61** |

micro-kernel peak (operands L1-resident, nk=256): k4x16 9.76 GFLOP/s, k8x12 9.31 GFLOP/s
(A57 peak 11.8 GFLOP/s at 1.479 GHz). Activations: snake 214 Melem/s, silu 86, tanh 58.

## 5. v1b_16k single stream

`build/aarch64/bench_cpu build/v1b16k.ptk bench/cpu/requests.txt single` (chunks: 3 frames, doubling to 32):

| request | phones | audio s | synth s (best of 3) | RTF | mel ready ms | first chunk ms | non-streamed synth s | streaming overhead |
|---|---|---|---|---|---|---|---|---|
| zh_greeting | 21 | 2.69 | 1.698 | 0.632 | 118.7 | 176.2 | 1.776 | -4.4% |
| en_thanks | 19 | 1.54 | 0.987 | 0.643 | 72.7 | 130.7 | 1.023 | -3.5% |
| mix_booking | 24 | 2.34 | 1.490 | 0.638 | 109.9 | 167.6 | 1.552 | -4.0% |
| zh_phone_sid1 | 29 | 2.00 | 1.283 | 0.642 | 100.3 | 158.4 | 1.334 | -3.8% |
| random_T90 | 90 | 8.70 | 5.461 | 0.627 | 395.7 | 453.5 | 5.730 | -4.7% |
| random_T23 | 23 | 2.46 | 1.564 | 0.635 | 111.4 | 169.2 | 1.631 | -4.1% |
| **all** | | 19.73 | 12.483 | **0.633** | | | | |

stream arena 14.2 MB (limits 400 phones / 1000 frames = 16 s). `mt` (1 stream, 4 threads):
RTF **0.276**, first chunk 76-206 ms.
Peak RSS in these two sections is 143.7 / 145.7 MB because `bench_cpu single|mt` also allocates a
whole-utterance stream (chunk = 1000 frames = 256000 sample windows); the server-shaped numbers are
the concurrency section below.

Phase profile, 16k vs 8k (same 6 requests, `PT_PROF=1`): encoder+decoder identical (6.4 s both),
vocoder 75.9 s (16k) vs 19.2 s (8k) = 3.96x. Shares at 16k: resblock conv1 43.3%, conv2 40.6%,
snake1 copies 4.8%, ConvT+interleave 2.7%, encoder 0.8%, decoder blocks 3.7%, rest 4.9%.

## 6. v1b_16k concurrency (chunk 32 frames, 4 shared encoder/decoder workspaces, pinned 1/core)

Guard before every sweep (`MemAvailable` and system swap used from `/proc/meminfo`); all sweeps ran,
none was aborted:

| sweep | MemAvailable before | swap used before -> after |
|---|---|---|
| conc1 | 195 MB | 1232 -> 1234 MB |
| conc2 | 194 MB | 1234 -> 1235 MB |
| conc4 | 195 MB | 195 MB avail, no swap growth |
| conc6 | 163 MB | 1232 -> 1238 MB |
| conc8 | 202 MB | 1238 -> 1240 MB |

| N streams | requests | audio s | wall s | audio-s per wall-s | mean RTF | max RTF | mean first-chunk ms | max first-chunk ms | worst lateness ms | max queue wait ms | ok | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 6.6 | 4.22 | **1.56** | 0.643 | 0.647 | 162 | 183 | +52 | 2711 | 3/3 | 35.7 MB |
| 2 | 6 | 20.0 | 8.37 | **2.38** | 0.679 | 0.697 | 226 | 481 | +78 | 3196 | 6/6 | 49.9 MB |
| 4 | 12 | 38.8 | 9.59 | **4.04** | 0.786 | 0.969 | 260 | 666 | +310 | 4553 | 12/12 | 78.7 MB |
| 6 | 18 | 58.4 | 14.75 | **3.96** | 1.360 | 1.790 | 472 | 1367 | +4382 | 8829 | 18/18 | 89.8 MB |
| 8 | 24 | 77.6 | 19.68 | **3.94** | 1.549 | 2.063 | 529 | 1351 | +4422 | 10973 | 24/24 | 102.4 MB |

Reading:
* **Throughput ceiling is CPU, not RAM.** Aggregate saturates at ~4.0 audio-s per wall-s; at N=6/8 the
  mean per-stream RTF (1.36 / 1.55) exceeds 1.0, i.e. 8 streams at 16 kHz are **not** real-time
  (worst lateness +4.4 s), and no sweep pushed the box into swap (peak RSS 102.4 MB, swap delta <= 8 MB,
  MemAvailable stayed 163-202 MB).
* **Largest real-time 16 kHz concurrency = 4 streams** (max per-stream RTF 0.969, worst lateness
  +310 ms -> a client needs ~0.35 s pre-roll). 2 streams is comfortable (max RTF 0.697, lateness +78 ms).
* Memory headroom is fine up to 8 streams: 102.4 MB peak RSS vs 182-202 MB available, weights mmap'd and
  shared (container 20.0 MB), 14.2 MB of arena per stream, 26.8 MB per encoder/decoder workspace
  (4 pooled).
* 8 kHz comparison, same host state: conc8 14.41 audio-s per wall-s, worst RTF 0.712, RSS 79.4 MB.
  16 kHz delivers ~3.9-4.0 audio-s per wall-s = about **0.5x real-time for 8 streams**, 4.0x for 4.

## 7. Window slack arithmetic (`V_SLACK`, computed from geometry, not a constant)

`pt_conv` rounds every conv to whole 16-column tiles, so a window must hold `n + 15 + (K-1)*dil`
columns; the runtime sets `V_SLACK = 16 + max tap span`, max span over the graph:

| layer | K | dil | tap span (K-1)*dil |
|---|---|---|---|
| resblock convs1, d=5 (K11) | 11 | 5 | **50** (max) |
| resblock convs1, d=3 (K7) | 7 | 3 | 18 |
| resblock convs2 (K11) | 11 | 1 | 10 |
| ConvTranspose (packed K = 2*stride) | 16 | 1 | 15 (15 at stride 8 in both variants) |
| conv_pre / conv_post | 7 | 1 | 6 |
| tile rounding | - | - | 15 (writes up to `n+15`) |

Required = 15 (tile) + 50 (span) = 65; `V_SLACK = 16 + 50 = 66` -> 1 column of margin.
The old constant was 96 (over-provisioned); 66 is exact by construction: window `ld = r16(cols + 66)`,
usable `cap = ld - 66`, so reads/writes at `cols + 65` land inside `ld`.
`RB_REACH = 60 = sum_k (d_k + 1)*(K_k - 1)/2` (2+12+30) is the left/right reach of a resblock group,
independent of the stage rate; `MELPAD = 64` covers the stage-0 look-back (10 frames at 8 kHz,
9 frames at 16 kHz).

## Repro

Host (x86, `.venv` with onnxruntime; host python is `.venv/bin/python`, never system python3):

```sh
.venv/bin/python tools/export_ptk.py --model-dir model/v1b_8k  --out build/v1b8k.ptk  --layers-json spec/v1b8k_layers.json
.venv/bin/python tools/export_ptk.py --model-dir model/v1b_16k --out build/v1b16k.ptk --layers-json spec/v1b16k_layers.json
.venv/bin/python tests/gen_golden.py --model-dir model/v1b_8k  --out tests/golden
.venv/bin/python tests/gen_golden.py --model-dir model/v1b_16k --out tests/golden16
.venv/bin/python bench/cpu/make_requests.py
rsync -a --relative runtime/src runtime/include runtime/kern runtime/Makefile \
        tests/test_e2e.c tests/test_kernels.c tests/run_tests.sh \
        bench/cpu/bench_cpu.c build/v1b8k.ptk build/v1b16k.ptk picard@100.84.151.116:primetts/
```

Nano (`~/primetts`):

```sh
make -C runtime -j4                                     # clean, only pre-existing test_e2e snprintf warnings
PTK=build/v1b8k.ptk  GOLD=tests/golden  tests/run_tests.sh   # must stay green
PTK=build/v1b16k.ptk GOLD=tests/golden16 tests/run_tests.sh
B=build/aarch64/bench_cpu; Q=bench/cpu/requests.txt
$B build/v1b8k.ptk  $Q single ; $B build/v1b8k.ptk  $Q mt ; $B build/v1b8k.ptk  $Q conc8
$B build/v1b16k.ptk $Q kernels ; $B build/v1b16k.ptk $Q single ; $B build/v1b16k.ptk $Q mt
for N in 1 2 4 6 8; do awk '/MemAvailable/{print $2/1024" MB"}' /proc/meminfo; $B build/v1b16k.ptk $Q conc$N; done
PT_PROF=1 $B build/v1b16k.ptk $Q single     # phase profile
```
Raw logs: `bench/cpu/logs/{e2e8,e2e16,single8_a,single8_b,mt8,conc8_8k,kernels16,single16,single16_prof,
mt16,conc16_1,conc16_2,conc16_4,conc16_6,conc16_8}.txt`.

## Residual risks

* 16 kHz single-stream RTF 0.633 leaves no CPU headroom: 8 streams need the GPU tier (v2.1) or the
  intra-op 4-thread path on a per-core budget; the 8-stream product target stays an 8 kHz delivery.
* `bench_cpu single|mt` peak RSS (143-146 MB) includes the whole-utterance stream used for the
  non-streamed comparison; do not read it as a server number (server-shaped: 102.4 MB at 8 streams).
* Worst lateness at 4 streams is +310 ms, so a 16 kHz server needs ~0.35 s pre-roll, and admission must
  cap 16 kHz streams at 4 (capacity 1/0.633 = 1.58 audio-s/s per core, `pt_v1b8k_sched_cfg.capacity_rt`).
* `V_SLACK` margin is 1 column by construction; a future vocoder with `K > 11` at `d = 5` would need the
  max-span term recomputed (it is read from the container kernels for ConvT, from `RB_K`/`RB_D` for resblocks).
* Duration rounding (the only non-deterministic part of parity) has margins 0.0102-0.0504 at 16 kHz
  (`tests/golden16/MANIFEST.json`), same order as 8 kHz: no case sits on a rounding boundary.
