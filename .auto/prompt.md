# Autoresearch: v2.1 TTS latency on the Jetson Nano (quality preserved)

## Objective
Make PrimeTTS v2.1 (`v21_mbistft_16k`, 16 kHz) produce speech with the **lowest possible latency**
on a Jetson Nano (4x A57 + sm_53 Maxwell, 4 GB shared memory), while keeping the audio bit-for-bit
in the validated parity envelope. The operator's deployment target is **4 parallel real-time
requests**, so the metric is the wall time of four concurrent requests, not one fast request.

Starting point (measured, session baseline): N=4 concurrent T81 requests in 9.9 s wall for 14.46 s
of audio (1.46x realtime, slowest stream RTF 2.51). One stream alone is 4.5 s for 3.62 s of audio
(RTF 1.25). Real-time for four streams means roughly 2.8x more throughput than that.

## Metrics
- **Primary**: `n4_wall_ms` (milliseconds, **lower is better**) — wall time of the multi-stream
  driver running 4 concurrent T81 requests on the Nano with `--cuda`.
- **Secondary** (tradeoff monitors, not objectives):
  - `n1_stream_ms` — single-stream latency (a fix that only helps throughput at N=4 is a partial win).
  - `min_sndr_db` — worst SNDR across the four streams; the quality gate is >= 60 dB.
  - `min_free_mb` — lowest free system RAM sampled during the run; Nano memory is a hard constraint
    (GPU/NvMap memory counts against system RAM and is invisible in RSS).

## How to Run
`./.auto/measure.sh` — syncs `runtime/` + `build/v21.ptk` to the Nano, rebuilds, samples free RAM,
runs N=4 and N=1, and emits `METRIC name=value` lines plus the GPU/host phase split.

`./.auto/checks.sh` — runs after every passing benchmark: integrity manifest (container, goldens,
feeds, spec, gate untouched), and SNDR >= 60 dB on the Nano for T20 + T81 on the GPU path and T81 on
the CPU path. A run that fails checks cannot be kept.

## Files in Scope
- `runtime/v21/src/pt_v21_host.c` — the portable C graph interpreter (4066 nodes, 41 op kinds).
  Most host-side time lives here: elementwise ops, shape ops, staging copies, allocator.
- `runtime/v21/src/pt_v21_cuda.cu` — conv / conv-transpose / matmul kernels (sm_53), resident weight
  cache, per-stream arena, per-stream CUDA streams.
- `runtime/v21/src/pt_v21.h` — shared types, `pt_ctx` (device arena/stream/weight fields), kernel
  signatures (conv/convT carry the weight ctab indices so streams share one upload).
- `runtime/v21/src/main_v21_par.c` — multi-stream driver (N threads: own context, own stream, own
  arena, shared weights). `main_v21.c` is the single-stream parity driver.
- `runtime/v21/Makefile` — `host`, `nano`, `par` targets.
- `tools/gen_v21_graph.py` — regenerates the C graph tables (`runtime/v21/gen/pt_graph.c`) from
  `spec/v21_layers.json` + the ONNX model. Legitimate place to bake in optimizations (constant
  folding, liveness, fusion metadata).
- `.auto/*` — this playbook, measure/check scripts, ideas backlog.

## Off Limits
- `build/v21.ptk`, `spec/v21_layers.json`, `model/*.onnx`, `tests/golden_v21/**` (goldens, feeds,
  probe dump), `tests/v21_parity_c.py`, `tests/v21_reference.py`, `tools/export_v21.py`,
  `tests/gen_golden_v21.py` — the benchmark and its ground truth. Re-exporting the container to make
  a number look better is cheating; the md5 manifest enforces this.
- The parity gate's tolerances (rel 1e-3 fp32 groups, 2e-3 decoder, SNDR >= 60 dB) are fixed.
- The v1.8 runtime tree (`runtime/src`, `runtime/kern`, `runtime/include`, `bench/`) is a reference,
  not the optimization target.

## Constraints
- **Quality is preserved**: every kept change must pass `.auto/checks.sh`, i.e. SNDR >= 60 dB on the
  Nano for T20 + T81 (GPU) and T81 (CPU). Parity runs use `noise_scale = 0.0` from the golden feeds;
  the sampler node is evaluated as zeros. The model is stochastic — never compare runs at
  `noise_scale > 0`.
- **No overfitting, no cheating**: don't special-case the T81 shape, don't key optimizations to the
  golden's phoneme count, don't skip layers, don't drop the conv-transpose tail flush, don't widen
  the parity tolerance, don't measure a smaller N than asked, don't time the build, don't disable
  the parity driver to make the loop look faster. The gate is the arbiter; when in doubt, run the
  full per-node gate (`tests/v21_parity_c.py --from-dumps <Nano dumps>`) and expect 0 nodes over gate.
- Every kept change must also hold on the **other golden cases** (T20, T40, T120, sid1_T29) — spot
  check them with `CASE=... ./.auto/measure.sh` after structural changes.
- Device rules for the Nano: `ssh picard@100.84.151.116`, CUDA at `/usr/local/cuda/bin/nvcc`
  (sm_53, gcc 8.4, python 3.6), no sudo/apt. Guard free RAM >= 60 MB before every sweep (measure.sh
  does this). Never kill/restart `fwd35_serve` (the voicechat LLM server sharing the box). Never
  print, echo, copy, or log `gemini-live-credential.txt`.
- No new dependencies: C (gnu11) + CUDA C + pthreads only. Host Python is `.venv/bin/python`.
- `runtime/v21/gen/pt_graph.c` is generated; regenerate it with `make -f runtime/v21/Makefile gen`
  rather than hand-editing.

## Two Nanos are available
- `picard@100.84.151.116` (host `nano`, gcc 8.4, ~1700 MB free) - the **primary** measurement box,
  the one the session baseline was established on. Keep the primary metric here so deltas compare.
- `pesi@100.70.245.36` (host `iPhO`, gcc 7.5, CUDA 10.2, ~767 MB free) - a second Nano, measurably
  faster and quieter: current code runs single-stream T81 in 3.43 s vs 4.44 s on picard, identical
  parity (70.08 dB, max|d| 2.247e-04). Use it as the **A/B confirmation box** when a delta is inside
  picard's ~4% noise: `NANO=pesi@100.70.245.36 ./.auto/measure.sh`. It is a real deployment target,
  not a benchmark toy - a change that only wins on one of the two boxes is not a win. Its memory
  headroom is half of picard's, so watch `min_free_mb` there.
Both boxes share the same rules: no sudo/apt, guard >= 60 MB free RAM, never kill a co-tenant server,
never touch `gemini-live-credential.txt`.

## Measurement discipline
- One change per experiment. If a change is structural (touches the graph loop), verify parity on
  T20 + T81 before spending a measurement.
- Noise floor: N=4 wall varies ~2-4% run to run on the Nano (thermal, co-tenant LLM server). Treat
  improvements under ~4% as noise: re-run before keeping. Confidence >= 2.0x = real.
- Log the phase split every time (`measure.sh` prints it): GPU kernels, staging copies, arena, host
  graph work. That is where the next idea comes from.

## What's Been Tried
Baseline (this commit): **N=4 wall 9.9 s**, N=1 4.5 s, SNDR 70.08 dB, min free ~1714 MB.
- **Phase 1 (runs 1-272): 9.9 s -> ~2.34 s (4.3x)**, quality proved (best 2288 then). Full record in `.auto/ideas.md`
  (convergence ledger + kill sections). Key pieces: resident shared weights, per-stream arena and
  streams, NEON elementwise ops, fp16 LUT decode, GEMM-tiled conv (ig8), chained conv on device,
  activation tables, GPU clock warmup kernel (the single biggest win, -148 ms cold).
- **Phase 2 (runs 273-281): parallel container record pass KEPT (-22 ms, 35 matched pairs, bit-exact,
  `PT_V21_OPEN_PAR=0` = serialized bisection).** Killed: CPU affinity pinning (2 ms tie), arena-size
  tuning (no gain, +4 MB free). pesi is offline for measurement: its disk is 100% full (operator data,
  nothing deleted).
- **Phase 3 (runs 282+): startup de-serialization, three keeps -- official best now 2232 ms.**
  (a) parallel device weight upload (-50 ms, 4/4 pairs, `PT_V21_UP_PAR`); (b) prewarm context +
  clock ramp before the record pass (-25 ms, 17 pairs, `PT_V21_WARM_PRE`); (c) the record pass
  itself (phase 2). Killed: overlapping the ramp with the upload (the low-clock copy penalty cancels
  the gain). Pattern: at N=4 a serialized shared-work phase costs the wall through the slowest
  stream's start, so splitting it wins even when the work is bandwidth-bound. Serialized startup is
  now ~150 ms (was ~390).
- Known bug patterns, do not reintroduce: conv/convT lanes are **Cout channels** (all lanes multiply
  the same input column); the `cout4 != cout` trim copies per batch at stride `cout4*len`, so kernels
  write the padded row layout; bias has `cout` entries (per-lane bound); conv reads the zero-padded
  scratch at `t*stride + k*dil` (no negative indexing); fp16 subnormals decode exactly (141k of them);
  batch is a real dimension on the dec/iSTFT conv path (four frames at once); **shared-payload work
  must be checked with peak RSS, not SNDR** (a dropped ctab claim duplicates payloads silently).
- Resident GPU weights (uploaded once, ctab-indexed, refcounted, shared read-only) + per-stream 8 MB
  bump arena + per-stream non-blocking CUDA streams + async staging copies: alloc 222 -> 0 ms,
  free 65 -> 0 ms, copies 153 -> 40 ms, single-stream T81 5.38 -> 4.44 s. Kept (commit `e072b0b`).
- Known bug patterns, do not reintroduce: conv/convT lanes are **Cout channels** (all lanes multiply
  the same input column); the `cout4 != cout` trim copies per batch at stride `cout4*len`, so kernels
  write the padded row layout; bias has `cout` entries (per-lane bound); conv reads the zero-padded
  scratch at `t*stride + k*dil` (no negative indexing); fp16 subnormals decode exactly (141k of them);
  batch is a real dimension on the dec/iSTFT conv path (four frames at once).

## Ideas backlog
See `.auto/ideas.md`.
