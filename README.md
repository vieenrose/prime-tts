# PrimeTTS v2.1 on the Jetson Nano (Gen1) — custom C/CUDA runtime

A from-scratch runtime that runs **PrimeTTS v2.1** (`v21_mbistft_16k`, 37.9M params, 16 kHz,
iSTFT decoder) on an original **Jetson Nano** (Tegra X1: 4× Cortex-A57 + 128-core Maxwell,
`sm_53`, 4 GB of RAM shared between CPU and GPU).

Everything is **C (gnu11) + CUDA C + pthreads**. No PyTorch, no ONNX Runtime, no TensorRT, no
framework at runtime. Host-side Python is used only to export the model into a container, generate
the graph tables, and check parity against goldens.

**Why a custom runtime:** on this chip the graph interpreter *is* the product. v2.1 is a 4066-node
graph over 41 ONNX op kinds; the model's own GEMM/conv work is only ~11.9 GFLOP per audio-second,
so a runtime that serializes launches, copies activations over PCIe per node, and walks index
spaces element-by-element loses most of the machine. This branch is the work of removing that
overhead while keeping the output **bit-for-bit inside the validated parity envelope**.

## Current state (measured on the Nano)

Target: **4 concurrent real-time requests** with quality preserved. Metric is the wall time of
four concurrent requests (`n4_wall_ms`), not one fast request.

| step | N=4 wall | 1 stream | SNDR |
|---|---|---|---|
| baseline: resident GPU weights, per-stream arenas, per-stream CUDA streams, async staging | 10308 ms | 3863 ms | 70.08 dB |
| contiguous / scalar-broadcast fp32 elementwise paths (ADD, SUB, MUL, DIV, MAX, MIN) | 10239 | 3690 | 70.08 |
| fp16→fp32 container weight conversions shared across contexts (refcounted, ctab-keyed) | 10070 | 3613 | 70.08 |
| iSTFT einsums (`bkt,kn->bnt`) as a specialized batched matmul | 9867 | 3294 | 70.08 |
| vectorizable LEAKYRELU / RELU / NEG / ABS paths | 9865 | 3212 | 70.08 |
| conv bias resident on the GPU + async input staging | 9638 | 3142 | 70.08 |
| conv kernel: cooperative shared-memory weight staging | 8787 | 2807 | 70.08 |
| conv_transpose kernel: shared-memory phase-slice staging (barrier-safe) | **8325** | **2714** | 70.08 |

- **-19.2 % at N=4** and **-29.7 % single stream** from the session baseline, with **every kept
  change producing bit-identical audio** (`max|d|` 2.247e-04 on T81, identical SNDR).
- 4 streams × 3.62 s of audio = 14.46 s of speech in **8.33 s** of wall time → aggregate RTF 0.58,
  per-stream RTF 1.78. True real-time for four streams needs ≈ 6.0 s, so the remaining gap is ~28 %.
- Quality gate throughout: **SNDR ≥ 60 dB** on the Nano for T20 + T81 on the GPU path and T81 on the
  CPU path, checked on every run by `.auto/checks.sh`, plus the per-node gate
  (`tests/v21_parity_c.py --from-dumps <Nano dumps>` → 0 nodes over gate).

## Layout

```
runtime/v21/src/pt_v21_host.c   portable C graph interpreter (4066 nodes, 41 op kinds) + CPU kernels
runtime/v21/src/pt_v21_cuda.cu  sm_53 kernels: conv, conv_transpose, matmul; resident weight cache,
                                per-stream arena, per-stream CUDA streams
runtime/v21/src/pt_v21.h        shared types, pt_ctx, kernel signatures
runtime/v21/src/main_v21.c      single-stream driver (parity)
runtime/v21/src/main_v21_par.c  multi-stream driver (N threads: own ctx/stream/arena, shared weights)
runtime/v21/gen/pt_graph.c      generated graph tables (from spec/v21_layers.json + the ONNX model)
tools/gen_v21_graph.py          regenerates those tables; legitimate place to bake in optimizations
tools/export_v21.py             ONNX -> build/v21.ptk container
tests/golden_v21/**             goldens, feeds, per-node probe dump (the ground truth)
tests/v21_parity_c.py           per-node parity gate against the Nano's dumps
bench/gpu/RESULTS.md            measured Nano roofline
.auto/                          the latency loop: playbook, measure.sh, checks.sh, ideas, run log
```

`model/`, `build/*.ptk`, `*.wav`, `*.npy` are git-ignored. Fetch the PrimeTTS v2.1 model into
`model/v21_mbistft_16k/primetts_v21_3voice.onnx`, then
`.venv/bin/python tools/export_v21.py` (writes `build/v21.ptk`; `--model` overrides the path),
then `make -f runtime/v21/Makefile gen` to build the graph tables for it.

## Build and run on the Nano

The Nano builds natively (gcc 8.4, `/usr/local/cuda/bin/nvcc`, `-arch=sm_53`). No root needed.

```bash
make -f runtime/v21/Makefile nano     # single-stream binary, CPU + GPU paths in one binary
make -f runtime/v21/Makefile par      # multi-stream driver
make -f runtime/v21/Makefile host     # CPU-only build for the x86 parity box
make -f runtime/v21/Makefile gen      # regenerate runtime/v21/gen/pt_graph.c

# one request, GPU path, parity against the golden
./build/pt_v21_nano --container build/v21.ptk --case tests/golden_v21/T81 --cuda
# four concurrent requests
./build/pt_v21_par --container build/v21.ptk --case tests/golden_v21/T81 --cuda --parallel 4
```

Profiling knobs (no recompile): `PT_V21_EKIND=1` prints accumulated time per op kind plus the
GPU/host split (kernels, staging copies, arena, alloc); `PT_V21_EKIND=2` adds the top-20 per-node
GPU times; `PT_V21_DUMP_T=<node>` writes `/tmp/pt_dump_<node>.bin` (raw fp32) for byte-exact A/B of
a kernel variant against the CPU reference; `PT_V21_GPU=conv|convT|matmul` restricts the GPU to one
kind; `PT_V21_ARENA_MB` sizes the per-stream device arena.

The measurement harness is `./.auto/measure.sh` (syncs to the Nano, rebuilds, samples free RAM
while running N=4 and N=1, prints `METRIC name=value` lines plus the phase split) and
`./.auto/checks.sh` (integrity manifest + SNDR gates). Both take `NANO=user@host`.

## What the Nano actually gives you (measured, `bench/gpu/RESULTS.md`)

- fp32 tiled GEMM **62 GFLOP/s**; fp16-storage + fp32-acc **70 GFLOP/s**; issue ceilings
  88 GFLOP/s fp32 and **168 GFLOP/s half2** (a half2-compute runtime is the only way past ~5 streams).
- **2 SMs**, 64 KB shared per SM, launch overhead ~26 µs, cooperative launch unavailable,
  device memcpy 5-5.9 GB/s, and GPU memory (NvMap) counts against system RAM — so RSS understates
  real memory pressure and free-RAM sampling is part of every measurement.
- v2.1 costs 11.9 GFLOP per audio-second: fp32 tops out at ~5 concurrent streams.

## Kernel rules this branch learned

1. **Stage weights in shared memory, cooperatively, once per block.** The naive conv kernel gave
   each thread the whole `K × cin` weight row for its group of 4 output channels, so all 256 threads
   in a block re-read the same 12 KB from L2. Staging that slice once per block: conv 2526 → 1855 ms,
   conv_transpose 359 → 249 ms. Bit-identical, because the summation order and every fp32 product
   are untouched.
2. **Do not tile the input with per-chunk barriers.** A variant that cooperatively staged input
   tiles 16 channels at a time was **2.7× slower** (conv 2.5 s → 6.9 s): ~48 `__syncthreads` per
   block per launch dominate on a 2-SM part. Read activations straight from global, coalesced.
3. **`__syncthreads()` must be reached by every thread in the block.** A `continue` that can skip a
   barrier is undefined behaviour. It produced a 6.6 % faster conv_transpose kernel that was
   deterministically wrong on exactly the nodes where the staged path was active (SNDR 6.47 dB on
   T20), and bit-exact on the fallback nodes. Rewriting the loop with a predicate instead of
   `continue` made it bit-identical and kept the speed.
4. **Accumulation order is a contract.** fp32 sums are not associative: seed accumulators with the
   bias like the reference does, keep the same tap/channel order. Reordering shows up as SNDR drops,
   not as a clean relative-error number.
5. **Keep weights, biases and host-side weight conversions resident and shared.** The 39 fp16
   decoder records are 13.6M elements: each stream used to build its own 54.5 MB fp32 copy. Sharing
   them (ctab-keyed, refcounted, mutex-guarded, container-path guarded) removed ~164 MB at N=4 and
   the per-call `cudaMalloc`+copy for every conv's bias.
6. **Host-side wins do not move the N=4 wall.** At 4 streams the wall is GPU-contention-bound;
   host wins (einsum, elementwise) show up in single-stream latency and free CPU cycles. To move
   the primary metric, attack GPU kernels.
7. **A buffer pool for activation tensors is not a win here**: no measurable latency gain, and
   ~200 MB of retained memory at N=4 (glibc was already reusing the freed blocks).

## Open work

- Encoder convs (12 at ~42 ms each, `cin=768`): their 4-channel weight slice is 36 KB, above the
  shared-memory gate, so they fall back to the unstaged kernel. A 2-lane (`float2`) slice halves it
  to 18 KB and fits.
- Matmul kernels (50 calls per request): same weight-staging idea.
- Host side, biggest remaining: `SLICE` 205 ms over 214 calls (memcpy-bound, scalar), `ADD` 105 ms,
  `TANH` 81 ms (libm per element), `PAD` 76 ms. Shape-only ops (Reshape/Transpose/Copy) still
  materialize; carrying a strided view through the interpreter is unbuilt.
- fp16 activations in the decoder (its weights are already fp16; measured rel 1.9e-3 today) to halve
  activation traffic in the deepest group — hard-gated on SNDR and the per-node gate.
- Half2-compute path if the target is more than 5 concurrent streams.

## Reproduction and integrity

`.auto/checks.sh` verifies by md5 manifest that the container (`build/v21.ptk`), the spec, the
golden audio, the golden feeds, the probe dump, the reference oracle and the parity gate's
tolerances (rel 1e-3 fp32 groups, 2e-3 decoder, SNDR ≥ 60 dB) are the exact bytes they were when
the latency session started. Parity runs use `noise_scale = 0.0` from the golden feeds (the model is
stochastic; never compare runs at `noise_scale > 0`). `runtime/v21/gen/pt_graph.c` is generated —
regenerate it with `make -f runtime/v21/Makefile gen`, never hand-edit it.

## Device notes

Two original Nanos are in use: a primary measurement box and a second, measurably faster and
quieter box used as an A/B confirmation target (a change that only wins on one of the two is not a
win). Both are unprivileged targets: no `sudo`/`apt`, guard ≥ 60 MB free system RAM before every
sweep, never kill a co-tenant server on the box. Host names live in the harness's `NANO`
environment variable, not in the code.

The broader product context (Gemini Live on the text leg, the v1b_8k CPU runtime, cost per
caller-minute) is documented in `docs/` and on `main`.
