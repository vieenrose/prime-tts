# Ideas backlog (v2.1 Nano latency)

## SESSION STATUS (run 125): converged -- see 'Convergence ledger' at the end for the full record
- Best: **n4_wall 2288 ms** (cluster 2288-2375 over 30+ official draws, mean ~2335, stdev ~1.7%,
  zero drift, draws independent). Baseline was 9900 ms: **4.3x faster**.
- Runs 118-125 (no code changes except run 120's 1-word CONVLOG label fix): box-parity
  correction (old pesi speed gap was pre-warmup throttling, boxes are peers now), exact
  residual traffic 60.4 MB/stream (~70 ms net best case, declined on multi-part + race risk),
  conv dispatch audit (122 ig8 + 2 wsm, plain kernel intentionally retained as bisection
  fallback), pesi health re-check (2341, headroom stable-low), disk hygiene (all clean),
  T120/sid1 re-verification (identical SNDR, no drift).
- Quality: per-node gate 0 over gate (unoptimized build) + SNDR 70.09 dB on all 5 cases, both
  boxes, both paths. Worst case sid1_T29/T120 at 67-68 dB (>= 60 gate).
- What remains, all needing operator approval: (1) batch-4 lockstep driver; (2) full device
  residency + CUDA graphs (~190 ms ceiling at ~1.4 ms/MB, run 88); (3) server-mode weights;
  (4) Winograd F(2,3) on the K=3 convs (15% of MACs, ~2.25x fewer multiplies there, ~80-130 ms
  net — the only remaining GPU FLOP reduction, run 70 — but numerics-changing, needs the
  per-node gate + a new kernel family + dispatch, i.e. project-scale, run 135).
- Everything else: banked, falsified with data/bounds, or explicitly out of bounds (see kill
  sections below). Next single-experiment step, if any: none with positive risk-adjusted EV.
  Recommended: autoresearch-finalize.

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

## Run 88 CORRECTION: copies are ~20-30% exposed -- device residency reopens (project-scale)
A doubled-copy probe (every conv H2D+D2H issued twice, identical bytes, env-gated, reverted):
off 2433/2340/2373 vs on 2562/2480/2505 (+129/+140/+132, 3/3, mean +134 ms for +94 MB/stream).
That is ~1.4 ms/MB-per-stream; the kept chain corroborates independently (~36 MB -> ~89 ms).
Run 85's "copy-removal saves ~0" is FALSIFIED -- revised to: copies mostly hidden, ~20-30%
exposed/congested. What survives of run 85: GPU-work additions still hurt (#55, untouched); host-only
removals still ~0. Consequence: remaining ~94 MB/stream of conv traffic x ~2 ms/MB ~= up to ~190 ms
wall IF fully chained -- the only known >100 ms single-direction lever. But no single-link
extension clears the bar alone (residual bundle: ~43 MB -> ~60-100 ms saved minus ~20 ms added GPU,
net ~40-80 ms, borderline; convT/matmul links small). It needs the bundled project-scale residency
work (device Add + device softmax + chain liveness for both), with operator approval -- the same
bucket as batch-4/graphs, now with the best ceiling (~190 ms) and a measured price.

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

## Gate methodology, final (run 96): the unoptimized-gate + optimized-SNDR split is CORRECT
Run 95 recommended teaching ck_save to D2H dev_resident tensors. That idea is WRONG -- analysis of
the fusion block shows why it cannot work in general: (1) epi-fused convs never materialize their
pre-epi output anywhere (the kernel accumulates post-epi directly), so no D2H can recover the
probe-expected pre-epi tensor; (2) chained + tv_alloc_nz (run 69: no zero-fill) means stale host
buffers are uninitialized mmap zeros, not even a stable wrong answer -- confirmed by the failure
pattern (rel exactly 1.0/0.725, i.e. zeros-vs-truth, varying with value scale); (3) skipped (fused)
nodes DO run ck_save on their aliased V entries (host.c:2179), so dumps are taken, just of garbage.
Therefore the per-node gate REQUIRES unoptimized execution (CHAIN=0 FUSE=0); the SNDR gate covers
the optimized path (every real corruption in this session showed up in SNDR instantly, while
chained-buffer staleness never affects inference -- SNDR 70.09 IS the proof the chain liveness is
correct). The split is not a stopgap, it is the only correct methodology. Do NOT implement
ck_save-D2H; do NOT gate the optimized path per-node.

## Full per-node gate (run 95): 0 over gate -- and what the 45 on-state fails mean
Ran the prompt's arbiter (`v21_parity_c.py --from-dumps`) on T81 Nano dumps: with chaining+fusion
DISABLED (env only): 4010 nodes compared, 2 exempt, **0 over gate, M2 parity PASS** (dec worst
1.898e-3 vs 2e-3 tol on a LeakyReLU output = accumulated upstream table error; dp/enc/flow ~1e-6,
root 9.8e-4). With chaining+fusion ON: 45 fails including rel=1.0 -- proven to be dump artifacts
(stale host buffers left deliberately by the chain + fused intermediates that never exist as
computed), NOT numerics bugs: same SNDR 70.09 either way, and every real corruption incident in
this session showed up in SNDR immediately. Consequence: the per-node gate cannot validate
chained/fused execution as-is; coverage = unoptimized-path gate + optimized-path SNDR x5 cases +
CPU/GPU cross-check. Recommended (not in-loop): teach ck_save to D2H dev_resident tensors so
future runs gate the optimized path too -- diagnostic-only, include in finalize review.

## Killed by analysis #7 (run 86): parallel record pass + N=1 upload/inference overlap
- **Parallel record pass (per-record claiming instead of the open_lock).** The ~99 ms serialized pass
  moves ~165 MB (84 MB memcpy + 27 MB fp16 reads + 54 MB LUT writes): bandwidth floor ~33 ms + loop
  overhead ~10 ms, so even perfect 4-way claiming saves ~54 ms wall -- below the ~93 ms bar and
  unconfirmable, while replacing the coarse lock that fixed run #56 with fine-grained claiming in
  the exact #55-race zone (double-convert leaks, refcount skew). Rejected on size AND risk.
- **Overlapping stream-0 early inference with the upload tail.** Saves ~50 ms of N=1 only (the wall
  is the slowest stream, which waited for all weights) and needs per-layer upload ordering + events
  in the same race zone. Secondary-only gain for correctness risk: rejected.

## Killed by analysis #8 (run 92): arena suballoc, pinned host arena, ctx duplication
- **Arena suballocation (1x32 MB carved 4 ways vs 4x8 MB cudaMalloc):** run 57 prices cudaMalloc at
  ~0.3-0.8 ms each; 3 saved allocs ~= 1-2.5 ms wall. Below even matched-pair practicality.
- **Pinned host arena (cudaMallocHost for the fp32 payloads):** would dodge run 77's extra staging
  memcpy by converting directly into pinned memory. Killed by run 77's own numbers: they diagnose
  the copy ENGINE at ~0.8 GB/s (direct 138 + staging memcpy ~22 = ~160 predicted vs 153 measured),
  and unified memory has no separate DRAM to DMA between, so pinned uploads at the same 139 ms.
- **Context-creation duplication:** would show as ~600 ms+ startup; measured ~400 ms with PHASE
  showing waits not work. Later cudaFree(0) calls attach fast. Confirmed no duplication.

## Killed by analysis #8 (run 91): staggered stream starts + bar-semantics correction
- **Staggered starts are pointless here, not just neutral.** Stream inference phases are ~1900 ms;
  natural offsets from serialized open are already ~25-100 ms. An artificial stagger D adds ~3D to
  the wall via the last stream while shifting phase alignment by D/1900 (2.6% for D=50 ms costing
  150 ms). Desyncing 1900 ms phases needs ~500 ms+ offsets (+1500 ms wall). The run-54 dips come
  from within-phase structure common to all streams, not cross-stream sync. Decisive, no silicon.
- **Bar-semantics correction to run 89.** "Below the bar" was overstated for one case: with n=8-10
  matched pairs (SE = sigma_pair/sqrt(n), sigma_pair ~25 ms after common-mode rejection) a 30-60 ms
  effect IS confirmable (~3-6 sigma). The bar governs single draws, not matched designs. This does
  NOT revive the residual bundle: it is 3-4 interlocking changes (device Add + chain liveness +
  convT-output residency + release discipline) in the exact #55-race / 0-dB-incident zone, i.e. it
  fails one-change-per-experiment and risk-adjusted EV independent of size. Rejected finally, for
  the right reasons this time: multi-part + race risk, not size.

## Critical-path timestamps (run 106): the 3-slow-1-fast upload pattern explained
Absolute per-stream timestamps (temporary instrumentation, reverted): all 4 opens start at ~t+0
and each reports ~107 ms (serialized record pass, ~105 ms total, not 4x). cuda phase ~282 ms per
stream: waiters do stream/arena creation concurrently with the single warmup+upload (different
locks) and block only on the remaining upload. Inference starts ~t+390 ms. The soak test's
3-slow-1-fast upload numbers were lock-wait accounting (waiters report waits), not three uploads;
the real upload happens once. No new lever; the startup model stands at higher resolution.

## Killed by measurement #7 (run 90): shortening the GPU warmup 64k -> 16k
N=1 warmup+upload totals, 3 alternating pairs: 64k 279.0/288.6/259.4 (mean 275.7) vs 16k
235.0/246.5/308.9 (mean 263.5). Tied on means, but pair 3 inverted (+49) on a 230 ms upload blowup
under a degraded box state (warmups at 3-5x normal, one thermal zone pinned at 100 C): 16k iters do
not ramp the clock when the box is already slow, and that is exactly when the ramp matters most.
The 64k size is robustness margin for worst-case thermal state, not average-case optimal. (Separately,
the best-case saving ~19 ms is below the confirmability bar anyway.) Do not shorten without a
load-aware ramp (e.g. loop until cur_freq reads max, which needs no new privileges but adds
unpredictable startup time).

## Killed by measurement #6 (run 82): pre-faulting the container mmap
`MAP_POPULATE` (+`WILLNEED`, env-gated `PT_V21_POPULATE`, page cache hot post-rsync) on the 111 MB
container mapping. Record-pass+map, alternating 3 pairs: 99.0/98.7/99.2 vs 101.9/102.1/101.0 --
pre-faulting is ~3 ms worse-or-neutral, 3/3. Faults are free (fault-around + hot page cache); the
99 ms is compute (84 MB memcpy + LUT fp16 convert, already optimized in runs 65/70). No wall A/B was
needed: a zero mechanism contribution bounds the wall effect at zero. Mapping tuning is exhausted.

## Desync mechanism (run 94): why host work structurally cannot move the wall
Stream start offsets from serialized open are ~25-100 ms (PHASE data); per-op granularity is
~1-3 ms (conv ~3 ms, elementwise ~1 ms, softmax 7.5 ms over 6 calls, einsum 4.7 ms x2). The
offset/granularity ratio is 25-100x, so the 4 streams' op phases are effectively RANDOM relative to
each other and the GPU idles only if all 4 are simultaneously host-bound -- which the offsets make
near-impossible except in synchronized startup/shutdown. Hence sub-100 ms host optimizations are
hidden STRUCTURALLY, not just empirically (this upgrades run 70's observation to a mechanism, and
reconciles run 63's -32 ms with run 68's ~0 as ~zero true effect + noise). Only synchronized phases
expose host/driver work: open, upload, close/SNDR tail. The remaining wall tail (~330 ms) is
serialized startup + SNDR/close + residual packing gaps (~250 ms ceiling stands, project-scale).

## Dispatch audit (run 121): the plain conv kernel is intentionally retained
All 124 T81 convs dispatch as: 122x ig8 (GEMM 4x8 tile), 2x wsm (narrow: cout=1 final
projection + one more), 0x plain `pt_conv_kernel`. The plain kernel is dead in practice but is
kept ON PURPOSE: it is the only fallback when bisection env vars disable the tiled paths
(PT_V21_IG_MINT/IG8_MINT, tiny WSM_KB). Do not remove it as dead code; its value is as a
bisection endpoint, and deleting it saves nothing at runtime.

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

## Convergence ledger (runs 78-127, kept current): session best confirmed
picard official draws, identical code: cluster 2288-2375 over 30+ draws, mean ~2335, stdev ~1.7%,
zero drift, draws independent (lag-1 -0.11). Best 2288 (run 87). All 5 golden cases bit-identical
GPU vs CPU on both boxes (worst SNDR 67.45 dB sid1_T29); full per-node gate 0 over gate on
unoptimized build (run 95); pesi 2341-2427 stable with headroom (peer of picard since run 119).
Standing: 9.9 s -> ~2.33 s (4.3x) with quality preserved and proved. Session converged; only
project-scale items remain (see SESSION STATUS at top).
- Runs 118-127: box-parity correction, CONVLOG label fix (1-word) + exact residual traffic
  60.4 MB (~70 ms net, declined multi-part + race risk), dispatch audit (plain kernel retained
  as bisection fallback), pesi health (2341) + pesi T120 N=4 (3219, 82 MB headroom), disk
  hygiene (all clean), T120/sid1 re-verification (identical SNDR, no drift).
- Wall budget, refreshed run 133 (replaces run 85's version, falsified by run 88 on copies):
  wall ~2335 = serialized startup ~390 (open/record 105 + cuda/warmup/upload 285)
  + packed GPU kernels ~1600 (4 streams x ~400, SMs saturated) + exposed copies ~100
  (~1.4 ms/MB effective) + SM packing gaps ~200 + close/SNDR tail ~45. Lever status per
  component: startup CLOSED (run 108 ceilings); kernels at local optima (only Winograd K=3
  ~130 ms, numerics-changing, needs gate + approval); copies + packing ~300 only via the
  project-scale residency/scheduling bundle (needs operator approval); tail sub-bar.

## Startup endgame: exact ceilings (run 108) -- all remaining startup ideas closed
- fp16 upload of the 39 fp16 records + same-table GPU convert (bit-exact by construction):
  ships 83.8 MB not 111 MB, saves 27.2 MB @ 0.8 GB/s = 34 ms minus ~8 ms convert = NET ~25 ms max,
  and touches weights_upload (the run #55 race zone). Below bar + risk: NO.
- Parallel record pass: 165 MB @ ~5 GB/s = 33 ms floor + ~10 ms loop = ~43 ms vs serialized 99 ms,
  so saves ~54 ms max, needs per-record claiming in the g_hcache lock zone. Below bar + risk: NO.
- Combined ceiling for every remaining startup idea: ~80 ms across 2 risky changes; no single
  change clears the bar and the combination is not one experiment. Startup is CLOSED as a lever.

## Run 70 findings (measurement budget: what is left)
MAC-weighted conv wave efficiency is already 0.921, so batching the four streams for wave
quantisation buys at most ~8% of the GPU total. No conv has cout % 4 != 0 and none runs with
batch > 1, so there is no padded-channel or batch waste to remove. Against 4 x ~380 ms of GPU work
(~1520 ms) the N=4 wall is ~1960 ms per stream, i.e. the aggregate is close to
max(GPU total, host total) + startup: cutting host work alone no longer moves it much.
FLOP share by kernel: K=11 30%, K=7 21%, K=3 15%, K=5 8%, dilated K=11 15%, dilated K=7 9%.
Winograd F(2,3) on the K=3 convs is the only remaining GPU FLOP reduction (15% of MACs, ~2.25x
fewer multiplies there -> ~8% of GPU work) but it changes numerics and must clear the per-node gate.
