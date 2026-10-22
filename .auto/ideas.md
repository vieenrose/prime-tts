# Ideas backlog (v2.1 Nano latency)

Ordered by expected payoff per unit of risk. Measure the phase split before and after each.

- **CUDA Graphs per stream.** ~180 launches (124 conv + 5 convT + 50 matmul) each with per-call
  copy/launch/sync; capture the per-request launch sequence once per shape bucket and replay with
  `cudaGraphLaunch`. Needs stable device addresses (the arena gives that) and a fixed shape bucket
  per T. Biggest remaining structural win; but the graph is issued from the interpreter, so capture
  needs one full warm run per shape first.
- **Batch the four streams per layer.** One launch per conv covering all four requests
  (grid.z = stream count, per-stream row stride). Removes 3/4 of the launches and amortizes weight
  reads. Requires the driver to interleave four graph loops in lockstep (shared node loop, per-stream
  pt_ctx), not four independent threads.
- **Pinned host staging buffers** (`cudaMallocHost`) for conv input/output staging: async copies get
  roughly 2x bandwidth. NOTE: `cudaHostRegister` on the whole mmapped container fails on this box
  ("operation not supported", tested); only freshly malloc'd staging buffers are candidates.
- **Device-resident residuals (conv -> Add -> LeakyReLU -> conv).** Tried and reverted: the Add's
  second operand is not device-resident, so falling back to `bin_op` reads the stale host buffer of
  the chained conv (SNDR 0 dB). Needs both operands resident before it can work.


## Learned this session (do not relearn)
- `__syncthreads()` must be reached by every thread in the block.
- Staging weights cooperatively into shared memory is the big conv/convT lever. Staging the *input*
  tile too (per-chunk barriers) was 2.7x SLOWER.
- Host-side wins under ~100 ms per stream do not move the N=4 wall; they show in n1_stream_ms.
- Buffer pooling of activation tensors: no gain, costs ~200 MB.
- gcc 8 does NOT auto-vectorize control flow: the `a[i] >= 0 ? a[i] : alpha*a[i]` LeakyReLU loop is
  scalar even at -O3. NEON intrinsics (vcgeq/vbslq) are needed, and they halve the time.
- The GEMM conv is register-bound at 64 registers / 4 blocks per SM: 8x8 tiles (104 regs) and
  register prefetch (84 regs) both lose. `-maxrregcount` 48/56 also loses badly (measured).
- A cold-start GPU clock ramp (76.8 MHz until ~0.6 s in) inflates the first process after a build;
  that is why the official measure.sh N=4 run and warm paired runs disagree.
- Diagnostics: `PT_V21_EKIND=2` prints per-node times; `PT_V21_DUMP_T=<node>` dumps raw fp32.

## Killed by analysis #5 (run 78): ig8 tail waste is real but unrecoverable
MAC-weighted tail waste across the 124 convs is 8.5% of conv GPU (~36 ms/stream, ~140 ms wall at
x4): Tout=904 (43% of all MACs) wastes 11.7% (8 blocks, tail covers 8 cols), Tout=226 wastes 11.7%,
Tout=81 wastes 36.7%. BUT a skinny tail block runs the same kc reduction loop and takes the same
time whether 256 or 32 threads wide -- the waste only converts to wall savings through SM packing
with other streams' blocks, and at N=4 the scheduler already backfills tails with other streams'
full waves, so most of it is hidden in practice. The alternatives are worse: `ig` (64-col) has less
tail waste but a much slower per-element rate (0.66 vs 0.88 effective -- ig8 still wins); padding
Tout to 128 and trimming needs padded intermediates that break the conv->conv chain. Do not pursue
without evidence from a packing-aware measurement, not a single-stream model.

## Killed by analysis #7 (run 86): parallel record pass + N=1 upload/inference overlap
- **Parallel record pass (per-record claiming instead of the open_lock).** The ~99 ms serialized pass
  moves ~165 MB (84 MB memcpy + 27 MB fp16 reads + 54 MB LUT writes): bandwidth floor ~33 ms + loop
  overhead ~10 ms, so even perfect 4-way claiming saves ~54 ms wall -- below the ~93 ms bar and
  unconfirmable, while replacing the coarse lock that fixed run #56 with fine-grained claiming in
  the exact #55-race zone (double-convert leaks, refcount skew). Rejected on size AND risk.
- **Overlapping stream-0 early inference with the upload tail.** Saves ~50 ms of N=1 only (the wall
  is the slowest stream, which waited for all weights) and needs per-layer upload ordering + events
  in the same race zone. Secondary-only gain for correctness risk: rejected.

## Killed by measurement #6 (run 82): pre-faulting the container mmap
`MAP_POPULATE` (+`WILLNEED`, env-gated `PT_V21_POPULATE`, page cache hot post-rsync) on the 111 MB
container mapping. Record-pass+map, alternating 3 pairs: 99.0/98.7/99.2 vs 101.9/102.1/101.0 --
pre-faulting is ~3 ms worse-or-neutral, 3/3. Faults are free (fault-around + hot page cache); the
99 ms is compute (84 MB memcpy + LUT fp16 convert, already optimized in runs 65/70). No wall A/B was
needed: a zero mechanism contribution bounds the wall effect at zero. Mapping tuning is exhausted.

## Killed by measurement #5 (run 81): PreferL1 cache config for the GEMM convs
`cudaFuncSetCacheConfig(ig/ig8, PreferL1)`, env-gated, wsm/plain/convT/matmul untouched (wsm needs
up to 48 KB shared). N=1 kernel+sync, alternating 3 pairs: 395.8/396.3/397.6 vs 401.9/396.9/402.6
(0/3 wins, directionally worse). The kernels' working sets already live in shared tiles by design;
the remaining global traffic is streaming with low reuse, so a bigger L1 cannot help. Same family
as the __ldg discard (same cache hierarchy). L1/shared tuning for these kernels is exhausted.

## Killed as stale or under-bar (run 80, by code reading + measured bounds, no Nano run spent)
- **Constant folding in the generator: STALE PREMISE.** `PT_CONST` (host.c:2192) aliases the container
  or int pool and never copies; CONST totals 3.1 ms/stream. There is no per-request materialization
  to fold away.
- **Encoder-conv float2 slice: STALE.** The wsm gate is 48 KB now, so the 36 KB cin=768/K=3 slices
  already take the weight-staged kernel with 512-thread blocks. Nothing falls back.
- **Pinned staging for conv I/O: killed by run 77.** The engine does ~0.8 GB/s from pageable memory
  no matter the call pattern, and pinned ADDS its own memcpy -- and per-stream copy volume (~130 MB)
  hides behind kernel time on the copy engine anyway, so even free copies would not move the wall.
- **Matmul launch coalescing: under bar.** 50 calls total 11-14 ms; batching same-shape grids saves
  only launch overhead (~50 x ~10 us = 0.5 ms) while adding booking complexity.
- **Shape-only op elision: under bar + risky.** Transpose/Reshape/etc. total ~15 ms of hidden host
  work; views would disturb the alive/dev liveness flags the chain and trim paths rely on.
- **-mcpu=cortex-a57 / -flto: not worth a loop.** Would only reschedule scalar double code (~25 ms
  of hidden host work); numerics-neutral but provably under bar.

## Killed by measurement #4 (run 77)
- **Async and pinned-staging variants of the 111 MB weight upload.** With the clock ramped (run 75),
  N=1 upload probes: sync 140.9/138.2 ms vs async+one-sync 139.4/136.0 ms (identical -- per-call
  overhead is NOT the bottleneck) vs pinned double-buffered staging 152.7/153.5 ms (WORSE: the extra
  111 MB host memcpy outweighs any DMA gain). The engine does ~0.8 GB/s H2D from pageable memory no
  matter the call pattern. Correction: the "H2D runs at ~3 GB/s" claim in the transfer-budget note
  below was a misread -- the copy split measures async ISSUE time; completion hides in kernel+sync.
  The upload path is bandwidth-bound and exhausted (only fewer bytes, e.g. fp16 upload + on-GPU
  convert, would help: ~60 ms for real complexity and risk -- below the bar). Do not touch it again.

## Killed by measurement #3 (run 76)
- **Overlapping CUDA context creation + clock warmup with the container record pass** on a dedicated
  prewarm thread (`pt_v21_prewarm` + `prewarm_run`, `PT_V21_PREWARM`). 6 matched cold pairs with
  order reversal: 2/6 wins, mean +34 ms. The record pass is serialized and CPU-bound, so the 5th
  thread contends on 4x A57 cores (open phase +12 ms) while stream 0 blocks on the prewarm condvar
  anyway. The prior 3/3 wins were spread luck. Correct primitive (condvar-guarded once-per-process
  prewarm), wrong bottleneck -- do not retry without a core to spare.

## Killed by measurement #2 (run 74)
- **Folding a single-consumer activation into its producer when the producer is a Slice.** The
  activation windows (32 Tanh/Sigmoid pairs) are fed by a Slice whose output tensor is still
  *unmaterialised* at the call site (n=0): the node's own op_bind_shapes pass is what computes its
  dims, so a loop over `out[0]->n` at that point iterates zero times while the rest of the graph
  still consumes real values (SNDR -27.8 dB). Separately, the fold does not even save work: SLICE
  10.8 -> 12.1 ms, TANH/SIGMOID unchanged, because those Slices are small. Do not retry.
- **Add->LeakyReLU fold** (same-shape fp32 add path): correct and bit-exact (LEAKYRELU 34.2 ->
  16.8 ms) but ADD 29.0 -> 30.1, i.e. net ~0. Not available for the 12 residual Adds (their tensor
  has 2 consumers) nor the 14 lrelus fed by Conv/ConvTranspose/Div (2 or 4 consumers).

## Killed by measurement (do not retry without new evidence)
- **fp16 conv activations**: SNDR 43.73 dB, far below the gate.
- 4 output columns per thread (16 accumulators): 39% slower.
- 2-channel conv blocks: 52% slower.
- Launch-shape heuristic (block size from output length): +10.5% at N=4.
- Conv reduction chunk 16 -> 32: barriers are not the limiter; 16 KB tiles cost more.
- Small 32x32 tiles for short outputs: arithmetic intensity matters more than block count.
- Padé-approximant / coarse-table tanh+sigmoid: SNDR 67.4 / 64.3 dB. The 65536-point interpolated
  table keeps 70.09 dB and is kept (TANH 62 -> 19 ms).

## Run 75: the GPU clock, without root -- INVALIDATES two earlier rollback reasons
`nvhost_podgov` idles the GPU at 76.8 MHz and ramps only on sustained SM load. A fixed 64k-iteration
ALU kernel launched right after `cudaFree(0)` (25.9 ms, once per process, `PT_V21_WARM=0` to disable)
reaches the same place `jetson_clocks` would:
- weight upload 226 -> 138 ms (0.46 GB/s at the idle clock was never a copy-path limit);
- matched cold A/B on picard: 2454/2511/2481 -> 2329/2316/2358 (mean -148 ms, -6%);
- second box pesi: 2516/2496 -> 2337/2378 (mean -148 ms).
**Rollback reasons now obsolete:** #50 "stop pursuing startup, it is off-objective" and #57
"warm-instrumented startup costs do not transfer to the cold measurement". The startup WAS on the
primary metric and its root cause was the clock. Anything that touches the cold first ~600 ms
should be re-examined now that the clock ramps first (the pooled cudaMallocs of #57 may now show up,
for instance).

## Measurement noise floor (run 73, important)
The SAME binary draws 2428 / 2436 / 2449 / 2479 / 2502 / 2521 ms on n4_wall -- a ~4% spread on this
box, worse than the 2-4% assumed earlier. A single draw cannot separate anything below ~100 ms.
Use the matched alternating procedure (>= 3 base/patch pairs, compare means) for every keep decision;
the "best" draw seen so far is the low end of the spread, not a record.

## Host budget (run 73) -- nothing above 31 ms/stream
ADD 30.9 (107 calls, incl. the 12 large residual adds), conv wrapper 26, TANH 19.1, LEAKYRELU 17.0,
SIGMOID 13.4, MATMUL 12.4, SLICE 11.2 (214 calls), EINSUM 9.3 (2 calls, double-precision
accumulation in the iSTFT basis contraction), SOFTMAX 7.5, MUL 6.9, PAD 6.2, EXP 4.8, DIV 4.8,
SIN 4.7, TRANSPOSE 3.7, RESHAPE 3.6, WHERE 3.3, CONST 3.1 (1924 nodes!), LAYERNORM 2.8, COS 2.2.
Largest remaining single mechanism: fold the pointwise activations into their producer's output loop
(every one of the 16 Tanh and 16 Sigmoid nodes is fed by a Slice, so that is 32 Slice->act pairs =
~30 ms/stream of pure duplicate memory traffic).

## Graph op dispatch (run 72)
A warm `PT_V21_EKIND=2` profile is the right tool: it surfaces nodes whose *measured* time is out of
line with their FLOPs. Node 4065 (the final cout=1, cin=4, K=63 output projection) was 11th at
17.3 ms for 0.01 GMAC because the 16-channel-group GEMM tile computed and discarded 15/16 of every
block. Generalising: always check `cout4/4 >= 16` before choosing a 16-group tile.
Next out-of-line host node: `/dec/reflection_pad` 13.4 ms -- `op_pad`'s row fast path covers only
constant mode, so reflect mode falls through to the per-element div/mod decode.

## Startup budget (measured with temporary timestamps, run 71)
open/record pass 100 ms (one serialized pass; later contexts alias the shared payloads and wait),
CUDA context 85 ms, device weight upload 230 ms for 111 MB. The upload is 0.46 GB/s only because the
GPU is still at 76.8 MHz: an identical second upload immediately after runs at 0.66 GB/s. So it is the
clock ramp, not the copy path. `cudaHostRegister` on the container mmap returns "operation not
supported" on this driver (tried twice: file mmap and MAP_POPULATE anon mmap), and a pinned
cudaMallocHost staging buffer bought only 240 -> 230 ms. Overlapping the 230 ms upload with the
100 ms CPU record pass is the only remaining startup win (~4% of the primary) and it would put CUDA
calls inside pt_v21_open, next to where run #55's race lived -- not worth the correctness risk.

## Transfer budget during inference (measured, run 71)
Conv activations move 67.5 MB out + 63.2 MB in per stream, but at the ramped clock H2D runs at
~3 GB/s (the 63.2 MB shows up as the 21 ms `copy` split), so all PCIe traffic is only ~43 ms of the
~380 ms GPU block. Transfers are NOT a lever; the GPU compute (34.9 GFLOP at ~44% of fp32 peak) is.

## GPU saturation trace (run 87): no idle left during inference
`nvprof` cannot hook on this box ("No kernels were profiled", twice -- Tegra/CUPTI limitation, do
not retry). Fallback that worked: sampling `/sys/.../57000000.gpu/load` + `cur_freq` at ~30 ms
during an N=4 run (80 samples over 2.66 s). Result: 12 low-load samples, ALL in the first ~0.8 s
(startup, expected); 58/58 inference-window samples show load >= 200, mostly 700-999. The GPU is
essentially saturated for the whole inference phase -- there is no idle time for a scheduler to
reclaim. Correction to the run-79 note: per-stream cost falls with N (better statistical
multiplexing of tails), but at N=4 the SMs are already full; the remaining per-stream overhead is
congestion/throughput (kernels at local optima), not schedulable idle. The ~250 ms "packing
ceiling" was a model artifact -- the measured ceiling is ~0. Do not propose graphs/full-residency
for idle-reclamation; its only remaining argument would be launch overhead (~1-2 ms), which is
far below bar.

## Scaling probe (run 79): the wall is GPU-total, and packing improves with N
N=1 inference-only 638 ms (= 583 graph + ~55 SNDR/close; open/cuda are before t0) vs N=2 wall 1551
(slowest 1139) vs N=4 wall ~2330 (slowest ~1920) vs N=8 wall 3961. Per-stream wall cost (wall-400)/N:
N=2 575 ms < N=4 482 ms < N=8 445 ms -- FALLS with N, i.e. SM packing improves as more streams share
the 2 SMs (tail waves get backfilled). Consequences: (1) the wall really is ~ startup_serialized
(~400: open 100 + cuda 300) + GPU_total_packed + small host tail -- confirmed arithmetically at four
different N; (2) at N=4 there is ~60-80 ms/stream of imperfect packing (~250 ms wall) that only a
perfect scheduler reclaims -- this QUANTIFIES the ceiling of the CUDA-graphs/full-residency project
(~250 ms), the only known >100 ms lever left, and it is multi-experiment project-scale, not one
change; (3) N=8 stays healthy (min_free 2104 MB, peak_rss 321 MB), so the box has headroom beyond
the operator's 4-stream target. Integrity note: moving close()/SNDR out of the timed region, or
pre-warming the GPU from a side process, would be timer-scope gaming -- explicitly rejected, the
timer stays as the driver defines it.

## Ledger (run 78): session best confirmed
picard official draws on identical code: 2328 (run 75), 2333 (run 77), 2315, 2330 (run 78) --
mean ~2326, tight cluster, no drift. All 5 golden cases bit-identical GPU vs CPU (worst SNDR 67.45
dB sid1_T29). pesi health check: 2427 (+2.9% over its earlier mean, within noise; identical parity,
min_free 250 MB, no crash). Standing: 9.9 s -> ~2.33 s (4.25x) with quality preserved.

## Run 70 findings (measurement budget: what is left)
MAC-weighted conv wave efficiency is already 0.921, so batching the four streams for wave
quantisation buys at most ~8% of the GPU total. No conv has cout % 4 != 0 and none runs with
batch > 1, so there is no padded-channel or batch waste to remove. Against 4 x ~380 ms of GPU work
(~1520 ms) the N=4 wall is ~1960 ms per stream, i.e. the aggregate is close to
max(GPU total, host total) + startup: cutting host work alone no longer moves it much.
FLOP share by kernel: K=11 30%, K=7 21%, K=3 15%, K=5 8%, dilated K=11 15%, dilated K=7 9%.
Winograd F(2,3) on the K=3 convs is the only remaining GPU FLOP reduction (15% of MACs, ~2.25x
fewer multiplies there -> ~8% of GPU work) but it changes numerics and must clear the per-node gate.
