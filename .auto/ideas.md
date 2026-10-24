# Ideas backlog (v2.1 Nano latency)

## KEPT (run 1574, PHASE 8): coalesced X staging in the ig8 conv -- the kernel axis was NOT closed
`PT_V21_IG8_COAL=1` (default; `=0` = old quad staging). Thread lq stages columns lq, lq+16, ..., lq+112
(contiguous 64 B per half-warp per load) instead of 4 consecutive columns via 4 scalar loads 16 B apart.
sm_53 does not cache plain global loads in L1, so the old pattern fetched every 32 B L2 sector for 2
useful floats. Same values in the same Xs slots -> bit-exact on all 5 cases (identical max|d|/SNDR).
64 regs / 0 spills needs `__launch_bounds__(256,4)` + 32-bit column math (an int64 draft hit 66 regs =
3 blocks/SM). Results: picard T81 N=4 -162 ms (6/6 pairs), official 1996 (below all 551 prior phase-8
draws, 2122-2323); N=1 stream ~640 -> ~595; CONV 424 -> 388 ms/stream. Other cases (N=4, 2 pairs each,
8/8 wins): T20 -39 (warm pair), T40 -79, T120 -236, sid1_T29 -70. pesi T81 -179 ms (4/4 pairs).
LESSON: the "closed from 8 directions" list varied tile shape, timing, cache config and occupancy but
never the per-instruction global ACCESS PATTERN. Audit every staging load for sector efficiency.
DONE (run 1575): convT ig8 coalesced too (PT_V21_CONVT_COAL) -- -39 ms T81 N=4 (8/8 pairs), 8/8 on
the other cases; the old convT ig8 was 84 regs = 3 blocks/SM (run 348's occupancy table said 4: re-check
any table built from assumptions with `nvcc -Xptxas -v`). DEAD: the ig8 Ws load is already sector-
efficient (lanes lr=2w/2w+1 fetch adjacent float4s of one group = one full sector).
KEPT (runs 1578, 1580): magic-number division in conv ig8 staging (PT_V21_IG8_MAG; carrying the index
instead spilled and regressed T20, run 1577) and a narrow-conv grid (pt_conv_kernel_igt, PT_V21_IG8_TAIL=2:
only for convs narrower than one 128-col tile; splitting WIDE convs' tails into a 2nd launch is +30 ms on
T81 because block duration is kc-bound and the extra launch serializes). T20 -48 ms. DISCARDED: block
raster swap (run 1576, not DRAM-bound), convT magic division (run 1579, below floor).
OLD NEXT LIST: (1) same column-per-thread staging in pt_convT_kernel_ig8 (same Xs quad pattern); (2) the Ws
weight load `w4[lg*kc + r]` puts lanes kc float4s apart (16 B useful per 32 B sector) -- a [r][g]
weight layout at upload time (or a transposed staging) would coalesce it; (3) ig/wsm/matmul_t staging.

- **Keep taxonomy: CPU-side vs GPU-side (runs 417, 442, 454).** The startup bundle (-77 quiet
  official / -64 quiet warm) compresses to -45 ms pooled over 10 pairs at ~2 busy cores: splitting
  serialized startup work needs free cores. The convT ig8 keep (-13 picard / -20 pesi warm, -70
  official) is GPU SM time and measures a +11 ms tie under 4 contention pairs (n=4, pair SE ~40 ms),
  i.e. masked not reversed. No keep regresses in any regime; report each keep with its regime
  condition.

## ig88p CLOSED BY SASS + MEASUREMENT (run 1772, phase 8): the per-chunk overhead is real but irreducible at the C level
Read the ig88p SASS (cuobjdump -sass, 4401 instr total): the main loop [0xba0,0x3fd8) is 1253 instr =
1024 FFMA (82%, the 16 rr x 64 float4 FMAs) + 64 LDS (5%) + ~165 per-chunk overhead (W/X address
forests with XMAD.PSL.CBCC 64-bit mul chains, edge-path predicates, STS addressing). A SASS-guided C
rewrite (run 1772: hoisted 64-bit W/X row bases, 32-bit folded row math, fetch_full/fetch_tail split
with the tail chunk hoisted out of the loop and zeroed tail rows) compiled bit-exact but to 1246
instr -- a wash -- and measured neutral-to-WORSE over 6 order-alternating official pairs (1799.3 vs
1759.8 mean, CONV +3.2 ms/stream 327.1 vs 323.9). MECHANISM: ig88p's W loads are inherently mul.wide
(64-bit pointer + int index); the ORIGINAL w4[g0*kc + r] 32-bit-folded index is the better
formulation -- my hoisted 64-bit wb0/wb1 pointers added a mul.wide+add.s64 per chunk per load, and
ptxas (which rematerializes addresses instead of keeping them live at the 128-reg ceiling) had
already balanced the tradeoff. Remaining theoretical slice: ~2-3% of conv (~25-30 ms N=4) and only a
hand-SASS rewrite of the 1250-instr loop could get it -- below the confirmability bar for the effort.
The kernel axis is now closed BY MEASUREMENT, not just by reading: FFMA 82% + LDS 5% is the floor of
this tile on sm_53.

## Contention-sensitivity map (runs 434-448) -- which regimes the co-tenant CPU cost hits
Loaded vs quiet official numbers on the primary box (loaded = ~1-2 busy cores from the operator's
jtop GPU monitors, load 1.0-1.9): N=1 1200/1262 = **-5%** (single stream is co-tenant-proof: one
thread on 4 cores); N=2 T81 1486/1341 = **+10.8%** (the most sensitive regime: 21%
serialized startup, little GPU to hide it); T81 N=4 warm band 2234/2092 = +6.8%, official means
2209/2176 = +1.3% (cold-draw variance masks it); T20 N=4 1020/1002 = +1.8%; T40 1363/1323 = +3.0%;
T120 2967/2974 = -0.2% (GPU-dominated, immune). The shift scales with the CPU-side share of the
regime, i.e. startup and host-op time, matching the startup-keep mechanism and the +2.9%/busy-core
law (run 371). Production guidance: give the TTS process CPU priority at the container/cgroup level
(in-process affinity is a tie, run 378); the low-concurrency regime is where the co-tenant hurts most.

## SESSION STATUS (run 413, PHASE 8): converged -- wall fully accounted for; monitoring + deployment evidence complete
- **Official cluster (primary box, shipped tree):** 201 N=4 T81 draws, mean **2209 ms**, stdev 2.1%,
  SE 5.4, range 2122-2307; warm soak mean 2092 quiet / ~2235 loaded; cold-after-idle ~2190-2230.
  Draws at load <0.5 average ~2176, at load ~1.0-1.2 average ~2247 (the contention law, run 371).
  Best draw **2128 ms, now reached twice** (runs 432, 641) with a fully nominal profile both times.
  End-to-end vs the session-start tree measured the same day: **4.74x** (10390 -> 2193).
  Second box pesi (30 draws): mean ~2233, i.e. +1.1% over picard, stable all phase-8.
- **Keeps quantified on the official procedure with bisection endpoints:** startup bundle -84 ms on
  picard (3.8%, 4/4 pairs) and -125 on pesi; re-refreshed run 632 at -49.5 ms (2/2 pairs, mid-loaded
  box) -- the series -77 (quiet) / -64 (warm) / -49.5 (loaded) follows the core-availability law.
  convT ig8 -70 ms quiet / -62.5 ms re-refreshed run 633 (2/2 pairs) / -13..-20 ms warm, both boxes.
- **Harness:** measure.sh now supports NANO=, TMPFS=1 (second box, disk-full-safe), CASE=, N4=,
  ARENA_MB=, ENVS= (bisection A/B in the official procedure); default path verified unchanged.
- **Phase-8 case table, primary box (official cold draws, all checks PASS):** T20 1002, T40 1323,
  T81 2193 (31 draws; warm 2064-2123), T120 2974, sid1_T29 1242. Second box official (TMPFS=1):
  1086 / 1402 / 2244 / 2957 / 1250 -- the box gap scales with the startup share (+8.4% T20 -> +0.6%
  T120). SNDR identical on both boxes (69.70 / 69.93 / 70.09 / 67.90 / 67.45), max|d| identical, and
  the full per-node gate on the phase-8 tree: 4010 nodes, 0 over gate, M2 PASS (run 359), and on the
  pesi toolchain too with identical per-group worsts (run 373). Startup share of the wall: 29% (T20),
  22% (T40), 23% (sid1_T29), 13% (T81), 10% (T120) -- the deployment (server mode) amortizes it.
- **N-scaling on the phase-8 binary (official):** N=1 1252-1272 (stream ~640-760), N=2 1341, N=4 2193,
  N=8 3806 (5-run soak). The 2nd stream is nearly free (+69 ms: it fills the SM's idle thread slots),
  streams 3-4 cost ~426 ms each -- the 1-SM saturation signature. Occupancy is not the limiter at N=1
  either (run 409: forcing 62.5%-occupancy conv is a tie at N=1).
- **Inference-phase clock measured (run 358):** 69% of an N=4 run is at the 921.6 MHz max; the rest is
  the startup ramp + ~210 ms of 691 MHz podgov dips. Governor knobs are root-owned (no sudo), max_freq
  already reads the max, so there is no clock lever left in inference. Idle clock is 76.8 MHz.
- **Phase-8 kills:** fp32 container aliasing (bit-exact, +10.6 ms, RSS flat -- reverted, run 346/347);
  fp16 weight upload (arithmetic: 32.5 ms startup vs +10-20 ms in-kernel converts, run 349);
  occupancy/carveout (zero of 11 kernels changes blocks/SM, run 348); kernel chunk-width/register-
  pipelining variants (+9% on one shape, -2..-26% elsewhere, run 355); the LDS-ceiling theory of
  run 352 falsified by its own ablation (the tile is staging/barrier bound); pinning under contention
  (tie, run 378); N=1 occupancy (tie, run 409).
- **Recommendation: autoresearch-finalize.** The wall is fully explained by measured components, each
  with its own closure; the approval list is unchanged (batch-4 lockstep +390 ms measured, CUDA graphs
  ~1-2 ms launch, server-mode weights now quantified at +300-340 ms in the multi-process shape, run
  398, Winograd 1.6x more ops).

## Wall budget, phase-8 form (1 SM) -- every term has a measured closure
T81 official 2193 = startup 287 + 4 streams x ~440 ms of SM-serialized work + tail ~45. Decomposition
of one stream's `kernel+sync` 397.7 ms (N=1): kernels ~250 (at the isolated clone rates) + wrapper ~43
(copies 21.6, zero-pad scratch memcpy ~10, 179 launch/sync ~12) + mid-run clock ~75 (the 384-921 MHz
governor ramp; 69% of a run is at max, run 358) + wave tails ~21 (8.5% MAC-weighted, run 78) = ~389.
Closures per term: kernels -- 9 measured directions (ig/ig8/ig16, window-direct, templated-K, FFT,
Winograd arithmetic, block count, occupancy/carveout, chunk width, N=1 occupancy); wrapper -- device
residency +261 ms (run 305); clock -- root-owned governor + 5 measured startup attempts; tails --
hidden by cross-stream backfill (runs 78/87, zero SM idle in inference).

## Deployment-regime findings (phase 8, runs 367-371) -- report content, not benchmark levers
- **Cold vs warm (run 367):** first N=4 request after 150 s idle 2232 ms mean vs sustained warm 2091 ms
  mean = +141 ms (6.7%) cold penalty; the clock-ramp kernel removes ~73 ms of it (re-verified in run 369
  with 4 cold pairs: ramp OFF 2317.5 vs ON 2244.8, 3/4 wins). A warm server-mode process pays the warm
  number and the 3 ms adaptive ramp, not the cold number.
- **Co-tenant CPU sensitivity (run 371, disposable spinners; fwd35_serve never touched):** 0 busy cores
  2243, 1 busy core 2307 (+2.9%), 2 busy cores 2446 (+9.0%). SNDR 70.09 in all 9 runs. This CORRECTS
  run 94: host work is hidden by stream desync only when cores are free. Per stream carries ~250 ms of
  host work (conv wrapper staging + zero-pad memcpys + NEON elementwise), ~1000 ms at N=4 on 4 cores.
  In production (TTS + LLM server on the same 4 A57) expect +3-9% TTS latency during LLM generation.
  Consequence for the operator's scope list: host-op reductions (PAD reflect fast path 6.4 ms/stream on
  T81 / 10.4 on T120; ADD 30.7; the run-74 Add->LeakyReLU fold; -mcpu=cortex-a57) are ~0 on the quiet
  benchmark box but worth ~20-60 ms in the contended deployment -- they cannot be KEPT by the primary
  metric, so they belong in the approval list, not the loop. CPU affinity pinning (killed quiet-box as a
  2 ms tie, run 274) is worth re-testing in the contended regime under operator approval.

- **N=8 soak (run 376):** 3771-3830 (mean 3806, 1.6% spread), SNDR 70.09 x5, MemAvailable flat 3262 MB
  -> 30x realtime aggregate, i.e. 2x the target concurrency with no leak.
- **T20 cold/warm (run 375):** 1020 cold / 927 warm (+93 ms, 10.1%) -- the cold penalty is ~100-140 ms
  absolute on every case, so it is the largest FRACTION on the shortest (startup-dominated) requests.
- **Cross-toolchain node gate (run 373):** the pesi build (gcc 7.5 + CUDA 10.2) passes the full per-node
  gate with 0 over gate and per-group worsts IDENTICAL to picard's -- the two toolchains are bit-identical.

## Deployment-regime evidence, phase 8 (runs 367-381) -- report section, benchmark-neutral
- Fairness at N=4 (run 380): per-stream spread 44-61 ms (2.5-3.5%) -> no stream starved; last user's
  latency = startup 287 + stream ~1780-1900 + tail 45.
- Mixed-length load (run 381): 4 processes x {T20,T40,T81,T120} = 11.54 s audio in 2.23-2.27 s (5.2x
  realtime), per-case SNDR/max|d| identical to standalone -> quality is load-independent; the gap to the
  shared-weights driver (6.8x realtime) is per-process startup, i.e. the server-mode item's value.
- Pinning under contention (run 378): tie (+7 ms) -> no affinity work; the +203 ms of 2 busy cores is
  intrinsic host throughput (run 371: +2.9% at 1 busy core, +9.0% at 2).

## Official-procedure keep-stack table (runs 390-391) -- the report's headline keep numbers
Measured with the ENVS= bisection knob through the OFFICIAL procedure (cold process, primary box,
2 order-alternating pairs each, SNDR 70.09 in every arm, checks green):
- Startup de-serialization bundle (PT_V21_OPEN_PAR/UP_PAR/WARM_PRE/WARM_ADAPT all =0): OFF 2287 vs
  ON 2210 = **-77 ms (3.4%), 2/2 pairs**; matches the warm pairs (-64 ms run 339) and the per-case
  table (-75 to -141 ms).
- convT ig8 (PT_V21_CONVT_IG8=0): OFF 2230 vs ON 2160 = **-70 ms (3.1%), 2/2 pairs**; warm pairs say
  -13/-20 ms, and the SM-serialization model predicts ~15 ms at N=4, so quote "15-70 ms, sign certain".
- Method: official per-draw noise is ~40 ms (stdev 1.7%), so official A/B needs >= 4 pairs to resolve
  20-30 ms; the profile line in measure.sh is from the nanoprof run, which does not receive ENVS.
- Harness now: NANO=, TMPFS=1 (pesi), CASE=, N4=, ARENA_MB=, ENVS= -- all verified green; the default
  path is unchanged (verified by draws 386/389).

## Two-box OFFICIAL case table, phase 8 (runs 360-396, all checks PASS, SNDR identical per case)
| case | picard (primary) | pesi (TMPFS=1) | pesi gap | startup share |
|---|---|---|---|---|
| T20 | 1002 | 1086 | +8.4% | 29% |
| T40 | 1323 | 1402 | +6.0% | 22% |
| T81 | 2192 +/- 8 (21 draws) | 2232-2296 | +2.5% | 13% |
| T120 | 2974 | 2957 | +0.6% | 10% |
| sid1_T29 | 1242 | 1250 | +0.6% | 23% |
The box gap scales with the startup share of the wall, confirming GPU-level parity (1 SM, 922 MHz,
bit-identical kernels on both toolchains) with a CPU/driver-side startup difference. SNDR per case:
69.70 / 69.93 / 70.09 / 67.90 / 67.45 on both boxes; max|d| identical; per-node gate 0 over gate on
both toolchains (runs 359/373).

## Loaded-box case refresh + per-stream RTF (runs 432-437) -- the deployment-realtime table
Box state matters: the primary box now carries ~1-2 busy cores (operator's jtop GPU monitors, load
~1.0-1.9) and the contention law (run 371, +2.9%/busy core) shows up in every regime. Official
primary-box numbers, loaded vs quiet: T20 1020/1002, T40 1363/1323, T81 2204/2176, T120 2967/2974,
sid1_T29 1267/1242; SNDR identical in all ten draws (69.70 / 69.93 / 70.09 / 67.90 / 67.45). The
GPU-dominated long case is insensitive to CPU contention (-0.2%); the short/mid cases shift 2-3%
(and the T81 warm band shifts +6.8%: 2092 quiet -> 2234 loaded, run 428).
Per-stream RTF from the driver's own L (T20 L=14848 = 0.928 s; T81 3.62 s; T120 5.24 s):
**T20 1.10 cold / 1.00 warm at N=4** (the binding realtime case, because 287 ms of serialized
startup is 28% of that request), T40 0.68, sid1_T29 0.75, T81 0.61, T120 0.57; single-stream N=1
T20 0.355, T81 0.18. Consequence: server-mode weight/context amortization is worth ~-260 ms
per request on T20-class utterances (1.10 -> ~0.78 RTF), the largest remaining deployment win.

## End-to-end speedup measured in ONE box state (run 403) -- the report's headline
Rebuilt the session-start tree (20f4a82, run-1 baseline) in a Nano temp dir with today's container and
goldens and measured both trees the same day: **10390 ms (old) vs 2193 ms (shipped) = 4.74x** (4.92x
vs the warm band). Quality: SNDR 70.08 -> 70.09 dB, max|d| 2.247e-04 -> 2.258e-04 -- 35 optimizations
moved the output by 5% of the existing deviation, far inside the 60 dB gate.

- **Keep value is core-availability dependent (runs 417, 442):** the startup bundle is -77 ms (official,
  quiet box) / -64 ms (warm, quiet) and -45 ms pooled over 10 pairs at ~2 busy cores (7/10 wins;
  batches -18 and -85 ms, pair spread ~90 ms at load). Splitting serialized startup CPU work needs
  free cores. Keeps never regress, so they stand; on a loaded box the deployment sees the smaller
  gain. Contention law (run 371): +2.9% per busy core at N=4, reproduced exactly (+203 ms at 2 cores).

## SESSION STATUS (run 348, PHASE 8): converged -- and the hardware model is corrected (1 SM)
- **THE BOX HAS ONE SM** (run 348, `cudaGetDeviceProperties` on BOTH Nanos: Tegra X1,
  `multiProcessorCount = 1`, 2048 threads/SM, 64 KB shared/SM, 65536 regs/SM, 922 MHz, 256 KB L2).
  Every earlier budget note said "4 streams share 2 SMs" (runs 79/87/328) -- that is wrong, and the
  fix makes the arithmetic cleaner: fp32 peak = 1 x 128 lanes x 2 x 0.922 GHz = **235.9 GFLOP/s
  (118 GFMA/s)**, and the measured pure-FMA peak (`.auto/peak_test.cu`, 97.6 GFMA/s) is 83% of THAT.
  Wall model, 1-SM form: **startup ~287 + 4 streams' GPU work serialized on one SM (4 x ~405 = ~1620)
  + tail ~45 ~= 1950-2200** (measured 2056-2318). Consequences: there is no second SM and no SM idle
  to reclaim (runs 79/87 confirmed at higher resolution), so ONLY fewer GPU cycles (kernels, closed
  from 7 directions) or less GPU work (device residency, closed at +261 ms) can move the wall.
  Both deployment boxes are GPU-peers, so pesi's 3-7% edge is CPU/driver/memory, not GPU width.
- **Occupancy is NOT the kernels' limiter** (run 348, `.auto/occ_probe.cu`, occupancy API on the real
  kernels): ig8 = 64 regs + 12288 B static shared -> 4 blocks/SM = 1024/2048 threads (50%), and it is
  **register**-limited, not shared-limited. `cudaFuncAttributePreferredSharedMemoryCarveout = 100`
  changes blocks/SM for **zero of 11 kernels** (4->4 conv_ig8, 4->4 convT_ig8, 5->5 conv_ig/matmul_t,
  1->1 wsm at 48 KB dynamic), so the untested direction of run 81's PreferL1 kill is closed with data.
  Two measurements bracket occupancy as irrelevant: conv_ig runs at 62.5% and is **+73 ms worse**
  (run 337); PreferL1 halved occupancy for only **+1.5%** (run 81). Register reduction (5 blocks needs
  <= 51 regs) is conv_ig's territory = measured-bad, and `-maxrregcount 48/56` already lost.
- **The kernels are staging/barrier-latency bound, and the tile is NOT where the runtime's 3x lives**
  (runs 352-355, `.auto/ig8_bench.cu` ablation + `.auto/ig8_v2_bench.cu` candidates):
  per-node efficiency (N=1, T81, exact shapes from the probe dump) puts every dominant conv at
  **21.6-24.3 GFMA/s** = 19% of the 1-SM peak, uniform across K=11/7/3 and cin=cout=128/256.
  The same tile structure in a bench runs **43-69 GFMA/s**. Ablation of the tile: removing per-chunk
  X staging +45%, W staging +49%, barriers +64%, and *halving* FMA-per-LDS made it **faster** (+72%)
  -- which FALSIFIES run 352's LDS-bandwidth-ceiling theory (67% of peak): the tile is not LDS-bound.
  Candidate fixes measured: register-pipelined staging +9% on the biggest node / -2..-3% elsewhere;
  RK=32 chunks (half the barriers) -16..-26%. Kernel axis is therefore closed from **8 directions**.
  **BENCHMARK LESSON #3: an isolated kernel's rate can be 2.5-3x better than the same kernel in the
  runtime**; the difference is the per-node wrapper (host zero-pad scratch memcpy + H2D/D2H staging,
  priced at ~1.4 ms/MB by run 88) and the mid-run clock (~700 of 921 MHz), not the tile. The only
  lever that attacks wrapper traffic is device residency, closed by measurement (+261 ms, run 305).
- **fp16 weight upload is closed by arithmetic** (run 349): the 39 fp16 container records are
  26.0 MiB stored vs 52.0 MiB expanded (container parse: raw 84.9 MiB, fp32 110.9 MiB), so uploading
  as-stored is bit-exact and saves 32.5 ms of copy-engine time -- against +10-20 ms of in-kernel
  fp16->fp32 converts in the staging loop of every decoder conv. Net 0 to +20 ms in the refcount zone.
- **Record pass / container aliasing is closed from 4 directions** (runs 326, 331, 332, 346): pointing
  the shared ctab at the read-only container mapping (PT_V21_ALIAS) is bit-exact on all 5 cases but
  +10.6 ms at N=4 (5 warm pairs) and peak RSS is FLAT (276 -> 275 MiB: MAP_PRIVATE file pages are
  page-cache pages, so the 84 MB moves bucket instead of shrinking). Reverted (run 347).
  Side product kept as knowledge: `pt_v21_close` munmaps the container map while other streams' aliased
  payloads point into it -- any future in-place container aliasing needs a live-context refcount.
- **Phase-8 official draws so far:** 2128 (baseline), 2193 (post-revert, checks PASS) -- in the phase-7
  cluster (mean 2219.6, warm ~2075). Kernel/scheduling axis closed from 7 directions (column tiling,
  window-direct, templated-K, FFT, Winograd, block count, occupancy/carveout).
- **Recommendation stands: autoresearch-finalize.** Approval list is unchanged (batch-4 lockstep, CUDA
  graphs, server-mode weights, Winograd -- all closed by measurement/arithmetic except the last two
  out-of-scope/below-bar ones), plus the new finalize-report content: the 1-SM correction, the
  occupancy table, and the record-pass closure.

## SESSION STATUS (run 343, PHASE 6): converged -- one keep, and every lever class closed with a mechanism
- **KEPT in phase 6: convT ig8** (run 325). convT 37.1 -> 33.4 ms/stream, -13 ms on picard (7 pairs),
  **-20 ms on pesi** (5 pairs), bit-exact on all 5 cases, full per-node gate 0 over gate (run 327).
  Official best draw 2155.
- **Phase-6 official cluster:** 2203 (baseline), 2155, 2194, 2318, 2228 -> mean 2219.6, stdev ~1.5%.
  Warm paired cluster on the same binary: 2056-2102, mean ~2075. The gap is the cold-process clock ramp
  plus first-touch cost (run 312's finding, unchanged).
- **Startup is CLOSED WITH A MECHANISM** (runs 330-332): serialized startup is 287 ms (open 71 +
  cuda_init 216, dominated by the 139-170 ms weight upload at 0.8 GB/s), and the upload runs at
  384-768 MHz because the governor needs ~600 ms of SM load to reach 921 MHz. Three attacks, all
  measured, all neutral-or-worse: SM-hold during the copy (+45 ms, 6/6), zero-copy mapped slab
  (+15 ms, 5/6 -- the copies were paying for the ramp), coalescing/pinned sources (no lever).
  **Any change that removes GPU work from the startup window makes the wall worse.**
- **Kernels are CLOSED from six directions:** ig8 (kept) beats ig 64-column (+73 ms), ig16 256-column
  (+10-31% in bench), register-window direct (+640 ms at N=4 despite being 1.9x faster in isolation),
  FFT (10x slower), Winograd (1.6x more ops for 1-D), wsm/plain fallbacks. convT and conv both run at
  21-35 GFMA/s against the box's ~97 GFMA/s pure-FMA peak, and the gap is the shared-memory/occupancy
  regime, not a bug.
- **Sync mode closed from all three directions:** yield (default) < spin (+89 ms, run 343) < blocking
  (+118 ms, run 293).
- **Re-verified on today's box state (run 339):** the three-item startup bundle is still worth
  -64 ms (5/5 warm pairs, 2068 vs 2132); the parallel upload alone is now a tie, so its original
  -50 ms was box-state dependent (it stays: harmless, and the bundle mechanism is what pays).
- **Evidence complete:** 5 golden cases x 2 boxes parity identical (SNDR 70.09/69.70/69.93/67.90/67.45,
  max|d| 2.258e-04/1.072e-04/3.813e-04/4.887e-04/1.726e-04); per-node gate 0 over gate; N=1/2/8
  scaling; CPU-path cross-check (T20 6777 ms, T81 27016 ms, parity identical); picard soak 2056-2102
  (10 runs, RAM flat 1455 MB); pesi soak 2099-2143 (8 runs, RAM 292-295 MB); integrity manifest clean.
- **Recommendation: autoresearch-finalize.** Remaining approval list: CUDA graphs (~1-2 ms launch
  overhead, below bar) and server-mode weights (out of benchmark scope). Everything else is closed
  with data.

## SESSION STATUS (run 287, PHASE 4 relaunched): converged -- see 'Convergence ledger' at the end
- Phase 4 (run 287+): baseline re-set to the startup-de-serialized cluster (official draws 2232, 2247
  => compare to ~2240). Phase 3 kept three startup de-serialization items (-22 record pass, -50 upload
  split, -25 ramp prewarm); serialized startup is now ~150 ms (was ~390). Phase-4 targets are the
  GPU-bound components: packed kernels ~1600 ms, exposed copies ~100 ms, packing gaps ~200 ms.
  Untried single-experiment levers: none with positive risk-adjusted EV. Spend loop on anti-overfitting
  robustness draws (N=8, T20/T40/T120/sid1 at N=4, pesi when disk frees, CPU path) + keep the 5-item
  operator-approval list current.
- Best: **n4_wall 2232 ms**. Baseline was 9900 ms: **4.4x faster**.
- Phase 2 (runs 273-281): affinity pinning killed (#274), arena tuning killed (#272), pesi headroom
  alarm (#271), parallel record pass kept (#278). pesi is offline for measurement (disk 100% full,
  operator data, nothing deleted).
- Phase 1 numbers below (2314-2375 cluster) are superseded by the phase-3 keeps: expect ~2240 mean now.
- Runs 118-125 (no code changes except run 120's 1-word CONVLOG label fix): box-parity
  correction (old pesi speed gap was pre-warmup throttling, boxes are peers now), exact
  residual traffic 60.4 MB/stream (~70 ms net best case, declined on multi-part + race risk),
  conv dispatch audit (122 ig8 + 2 wsm, plain kernel intentionally retained as bisection
  fallback), pesi health re-check (2341, headroom stable-low), disk hygiene (all clean),
  T120/sid1 re-verification (identical SNDR, no drift).
- Runs 270-272: ledger draws; pesi health re-check -- parity identical (2414, SNDR 70.09) but
  min_free 66 MB = the guard boundary (co-tenant growth, not our code); arena 8->4 MiB killed
  (see 'Killed by measurement #9'), full per-stream memory map measured and banked.
- Quality: per-node gate 0 over gate (unoptimized build) + SNDR 70.09 dB on all 5 cases, both
  boxes, both paths. Worst case sid1_T29/T120 at 67-68 dB (>= 60 gate).
- What remains, all needing operator approval: (1) batch-4 lockstep driver; (2) full device
  residency + CUDA graphs (~190 ms ceiling at ~1.4 ms/MB, run 88); (3) server-mode weights;
  (4) Winograd F(2,3) on the K=3 convs (15% of MACs, ~2.25x fewer multiplies there, ~80-130 ms
  net — the only remaining GPU FLOP reduction, run 70 — but numerics-changing, needs the
  per-node gate + a new kernel family + dispatch, i.e. project-scale, run 135); (5) NEW run 272 --
  per-record host liveness for the 84 MiB container copy (weight records feed only the one-time
  upload): the only real memory lever, now important because pesi bottoms at 66 MB free in N=4.
- Everything else: banked, falsified with data/bounds, or explicitly out of bounds (see kill
  sections below). Next single-experiment step, if any: none with positive risk-adjusted EV.
  Recommended: autoresearch-finalize.

Ordered by expected payoff per unit of risk. Measure the phase split before and after each.

- **Phase-2/3 REVIVAL NOTE (runs 278, 283): the "below the ~90 ms bar" kill argument is too
  conservative.** Two startup items killed on that argument were kept at -22 and -50 ms using
  order-alternating matched pairs (15-35 pairs for -22 ms, 4 pairs for -50 ms). Anything 20-50 ms
  is testable; anything touching shared startup work is worth a matched-pair probe.

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

## KEPT in phase 4 (run 300): adaptive clock ramp (sysfs cur/max gate)
Reads the world-readable devfreq sysfs (`cur_freq`, `max_freq`, no root) at prewarm: if the GPU is
already at max clock, the ramp kernel is cut from 64k to 8k iterations (~3 ms maintenance burst).
**-25 ms on pesi (8 pairs, 7/8 wins, t=-2.5)** and **-22 ms on picard (6 pairs, t=-1.0)**; pooled 14
pairs -24 ms, 10/14 wins. Official 2220, all 5 golden cases bit-identical (clocks-only change),
checks PASS. `PT_V21_WARM_ADAPT=0` = bisection endpoint.
Why run 90's kill of a shorter ramp does NOT apply: that kill measured 16k unconditionally, which
fails to ramp the clock when the box is already slow. This gate shortens only when the box is
measurably NOT slow, and never *skips* the ramp: run 90's real failure mode is the governor dropping
the clock while the CPU-side record pass runs, so the 8k burst stays as insurance (~3 ms).
Production value > benchmark value: a server-mode process under continuous load has cur_freq at max and
pays 3 ms instead of 26 ms per request. Startup chain is now context 85 (ramp overlapped) + upload
share ~40 -> ~130-150 ms serialized startup, from ~390 at the start of phase 2.

## pesi is measurable again: build and run in tmpfs (run 298) -- startup keeps confirmed on both boxes
pesi's disk is 100% full (operator data: vc 1496 MB, b2 587 MB, exp 141 MB; nothing deleted), so
rsync/build there fails. Workaround that touches NO disk: `mkdir -p /dev/shm/pt && rsync -a runtime/
pesi:/dev/shm/pt/runtime/ && cd /dev/shm/pt && make -f runtime/v21/Makefile par nano`, then run with
`--container ~/primetts/build/v21.ptk --case ~/primetts/tests/golden_v21/<case>` (read-only from disk).
/tmpfs is 1983 MB; the build costs 8 MB. pesi N=4 headroom is healthy again (min_free 232-260 MB, not
the 66 MB of run 271); pesi GPU idles at 921 MHz vs picard's 768 MHz.
RESULT: the three phase-3 startup keeps confirmed on pesi too -- 12 order-alternating pairs,
2210 (ON) vs 2242 (OFF) = **-32 ms, 10/12 wins, t=-2.5**, SNDR 70.09 in every arm. Smaller than
picard's -95 ms because pesi's startup is already cheaper (faster CPU/driver, higher idle clock).

## Phase-4 scaling + control probes (runs 292-294): the final code's shape/N envelope
Scaling table (T81, picard, final code): N=1 stream ~630 ms, N=2 1375, N=4 ~2230 (best draw 2173),
N=8 3932. Per-stream wall cost (wall - 225)/N = 575 / 501 / 473 ms: packing improves with N.
Startup-keep transfer by N: N=2 -176 ms (11.3%), N=4 -95 ms (4.2%), N=8 -30 ms (~0.8%) -- biggest
relative effect at low N, exactly the operator's deployment regime.
Control probe (run 293): PT_V21_SYNC=1 (blocking) = +118 ms vs the yield default, 6/6 pairs -> the
defect-isolated default is validated, not accidental.
Critical-path budget after all keeps: context 85 + upload 139 + inference ~1900 + tail ~45 = 2169 vs
measured ~2230 -> ~60 ms of packing/variance left. The session is at the architectural floor of the
current design; the only levers above that floor are the 5 project-scale items.

## Phase-4 generality proof of the startup keeps (runs 288-289) + fp16-upload arithmetic kill
Generality table (order-alternating pairs, keeps ON vs all three OFF, N=4, picard):
- T20: 1027 vs 1102 = **-75 ms (7.3%)**, 5/8 wins, t=-2.1, SNDR 69.70 identical.
- T81: -95 ms (4.2%) (the phase-3 confirmations).
- T120: 2953 vs 3086 = **-133 ms (4.3%)**, **6/6 wins, t=-4.7**, SNDR 67.90 identical.
- sid1_T29: 1236 vs 1368 = **-132 ms (10.7%)**, 6/6 wins, SNDR 67.45 identical.
- T40: 1382 vs 1523 = **-141 ms (9.3%)**, 6/6 wins, SNDR 69.93 identical.
Table total: 29/31 pair wins, zero SNDR / max|d| change in any arm. Largest RELATIVE gains are on the
shortest requests (7-11%), the operator's latency-critical regime. Wall per case with keeps ON:
T20 1027, T40 1382, T81 2232-2236, T120 2953, sid1_T29 1236.
**fp16 weight upload KILLED BY ARITHMETIC (run 287, revived legitimately then closed):** ship the 39
fp16 records as stored (83.8 MB not 111 MB) and decode on device. The split-half trick removes the
staging buffer the run-108 kill assumed was needed (copy the fp16 into the first half of the record's
fp32-sized slab region, decode into the disjoint second half -> safe, parallel, same slab footprint),
so the assumption changed. But the device decode moves 83.8 read + 167.6 write = 251 MB of device
memory traffic ~= 21-31 ms at the Nano's ~8-12 GB/s, against a 27.2 MB copy saving of ~34 ms. Net ~0
to +10 ms for a change in the #55 refcount zone. Closed for bandwidth arithmetic, not staging.

## KEPT in phase 3 (run 285): prewarm context + clock ramp before the record pass
`pt_v21_cuda_prewarm()` (called by both drivers before `pt_v21_open` when `--cuda`): set the sync
flags, `cudaFree(0)` (context, ~85 ms) and launch the clock-ramp kernel on its own non-blocking
stream; `pt_v21_cuda_init` joins the ramp after the upload. The ramp (GPU SM work) now overlaps the
container record pass (CPU/memory work). `PT_V21_WARM_PRE=0` = serialized bisection endpoint.
**-25 ms** (17 order-alternating pairs: 2213 vs 2238, 11/17 wins, t~-2.0), official **2232 = new
session best**, all 5 cases bit-identical (the ramp changes clocks only), checks PASS.
Run 76's 5-thread prewarm died on core contention; this revives it correctly because the ramp adds
NO thread (GPU work launched by the thread that already exists). Startup chain now: context 85 ms
(overlapped by ramp 26 + record-pass share 21) -> upload share ~40 ms; serialized startup ~150 ms,
down from ~390 at the start of phase 2.
BUILD HAZARD (run 285): a python cleanup regex silently swallowed the ramp implementation and the
build failed with "referenced but not defined", so the first measurements were taken on a STALE
binary. After scripted edits to sources, always rebuild and confirm the build succeeded. -22 (278),
-50 (283), -25 (285): de-serializing shared startup work is the session's highest-yield lever.

## Killed by measurement #11 (run 284, phase 3): overlapping the clock ramp with the upload
Ran the 26 ms ramp kernel on its own non-blocking stream during the parallel weight upload (SM work
vs copy-engine work): 4 order-alternating pairs, 2293 vs 2297 = tie (t=-0.2). Mechanism: overlapping
only wins if the copies keep their ramped rate, but the ramped-rate share of the upload shrinks by
exactly the overlap window -- the low-clock copy penalty cancels the saved ramp time. Ramping in
front of the upload is the arithmetically correct order. Code reverted (simpler = better).
Startup is now: context 85 ms (one-time, can't overlap: run 76's 5th-thread prewarm loses cores, and
moving it in front of the record pass is arithmetically neutral for the slowest stream) + ramp 26 ms
(enabler of fast copies) + record pass ~21 ms/stream share + upload ~40 ms/stream share.

## KEPT in phase 3 (run 283): parallel device weight upload
`PT_V21_UP_PAR=1` (default; `=0` = serialized bisection): all 4 stream threads claim container
records with an atomic counter and copy their claims into the shared weight slab with
`cudaMemcpyAsync` on their own CUDA streams, sync their own stream, then wait at a barrier. Setup
(slab + offsets) stays under g_wlock; the copies run outside it (they must, or the barrier deadlocks).
Result **-50 ms** (4 order-alternating pairs: 2292 vs 2342, 4/4 wins, t=-4.2); official draw **2260
= new session best**; n1 638 neutral; SNDR 70.09, checks PASS; max|d| identical on all 5 cases.
The copy engine does NOT get faster (upload work stays ~165 ms, ~0.8 GB/s): the win is the critical
path -- stream 0 no longer queues behind a single copier (pays ~1/4 of the upload), the last-arriving
stream pays ~0 (it finds the copies done, 19 ms reported).
MECHANISM PATTERN (now twice proved, runs 278 + 283): at N=4, a serialized shared-work phase costs
the wall through the slowest stream's start, so splitting it wins even when the work is bandwidth-
bound and the total time is unchanged. Startup is now ~285 ms of the ~2260 ms wall (was ~390).
Corollary: the remaining serialized items are the one-time CUDA context creation (~85 ms, cudaFree(0),
unavoidable) and the 26 ms clock warmup. NEXT CANDIDATE: run the warmup kernel on a temporary stream
so it overlaps the uploads (SM work vs copy-engine work do not contend): overlapped cost ~134 ms vs
sequential 26+138 = 164, i.e. ~-30 ms, since copies issued during the ramp are partly at low clock.

## KEPT in phase 2 (run 278): parallel container record pass
`PT_V21_OPEN_PAR=1` (now the default, PT_V21_OPEN_PAR=0 = serialized bisection endpoint) splits the
one-pass container materialization (84 MB memcpy + 27 MB fp16 LUT decode) across the stream threads:
prepare (record->ctab map, payload arena, cache pre-size, fp16 LUT) single-threaded under open_lock;
then atomic record claiming, arena bump under a short lock, completion barrier, per-context aliasing.
Result: **-22 ms on the N=4 wall** (35 order-alternating pairs, means 2322 vs 2344, 24/35 pair wins,
t ~ -2.2), N=1 neutral (648 vs 652), peak_rss +10 MB, SNDR 70.09 and checks PASS. Bit-exact proof:
all 5 golden cases at N=4 give identical max|d| (1.072e-04, 3.813e-04, 2.258e-04, 4.887e-04,
1.726e-04) and identical SNDR (69.70/69.93/70.09/67.90/67.45) on both arms. Realized gain is ~1/3 of
the 60 ms ceiling: the pass is LPDDR2-bandwidth-bound (4 cores buy ~1.3x, not 4x). N=1 is neutral by
construction (one thread: nothing to split).
SILENT-BUG LESSON (the reason RSS is a first-class metric): the first version dropped claims for
ctab indices beyond the lazily-grown g_hcache_n -- host_cont_shared_claim checks k < g_hcache_n and
returns void -- so every context fell back to a PRIVATE copy: peak RSS 468 MB (4 payloads), wall
+62 ms, SNDR still 70.09. Fixed by hcache_prealloc(n_cont) in prepare + par_claim() that reports
failure. SNDR cannot see duplicated payloads; RSS is the sharing proof.
Startup components now: ~85 ms record pass (parallel), ~85 ms CUDA context, ~26 ms clock warmup,
~138 ms weight upload. Remaining: project-scale only. Prewarm-overlap stays killed (run 76): with the
pass using all 4 cores, a 5th CUDA thread contends worse.

## pesi is OFFLINE for measurement (run 278): disk 100% full
`df` on pesi: 14 GB root, 13 GB used, 0 available; rsync of runtime/ fails with "No space left on
device". Largest home entries: vc 1.5 GB, b2 587 MB, exp 141 MB, primetts 135 MB, tailscale tarballs
~107 MB -- all operator/co-tenant data, NOT ours, so nothing was deleted. pesi confirmation of the
kept change is pending until disk frees; pesi min_free was already at the 60 MB guard (66 MB, run 271).

## Killed by measurement #10 (run 274, phase 2): CPU affinity pinning
`PT_V21_PIN=1` pins each stream thread to its own A57 core (1:1, numerics-neutral, driver-only).
Order-alternating 4 pairs: pinned 2426 vs unpinned 2424 -> a 2 ms tie. Mechanism: the streams are
GPU-bound and desynchronized (run 94), so the 4 host halves are rarely core-hungry at the same time
and Linux already gives them near-exclusive cores; pinning also removes the scheduler's freedom to
dodge the co-tenant, so it never wins under load. Host/OS scheduling is CLOSED (with run 76 prewarm).

## Killed by measurement #9 (run 272): arena-size tuning + the real memory map
Per-stream memory, measured with PT_V21_MEM on all 5 golden cases (N=1 and N=4, picard):
host container_fp32 84 MiB SHARED across streams (one record pass), host activation_peak 2/5/10/15/4 MiB
per stream (T20/T40/T81/T120/sid1_T29), device weights 110 MiB SHARED (one upload), device arena
HIGH-WATER 0/0/1/2/0 MiB (allocated 8), `dev_bytes_stray` is a cumulative traffic counter, NOT a live
footprint (do not read 60 MiB as memory).
Shrinking the arena 8 -> 4 MiB (2x margin over the worst case): order-alternating 4 pairs (8 runs):
arena8 mean 2349 vs arena4 2388 (no gain), min_free 1821 vs 1825 MB (+4 MB, not the expected +16):
Tegra CUDA suballocates from a lazily-grown heap pool, so arena size does not convert to system RAM.
Arena tuning CLOSED. The only real footprint lever left is per-record host liveness (84 MiB container
copy whose weight records are needed only for the one-time upload) -- project-scale, #55/#56 lock zone.
Memory context: pesi min_free is now 66 MB during an N=4 run (767 MB at session start, 82-90 MB in
run 127) from co-tenant growth; picard is healthy (1820-2043 MB). Headroom, not latency, is the
binding constraint on pesi.
METHOD NOTE (run 272): the 1st run of an A/B pair is ~30-50 ms faster (clock/thermal asymmetry). A
fixed-order harness produced a spurious -146 ms for arena=4; ALWAYS alternate pair order (session
rule, re-proved).

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

## KEPT in phase 5 (runs 304-305): device-residency scaffold (inert) + the project CLOSED BY MEASUREMENT
Implemented the full residual bundle: device Add kernel (fused LeakyReLU epilogue), device LeakyReLU kernel,
per-stream residency sub-arena (best-fit free list + coalescing, blocks tracked by off+size, released at
tv_free = last use, operand uploads reclaimed at chain-end syncs), eligibility rules
(resident(tensor) = 1..4 readers all conv-in0/Add/LeakyReLU AND producer can publish it: conv = free base
case, Add/Leaky = one resident operand; recursion walks producers so it terminates), conv-site keep
extension, chain terminators (result read by host op -> device kernel + one D2H), a generic host-reader
safety sweep (any host op reading a device-stale tensor restores its host copy first), and
PT_V21_DEVADD_LIMIT link-cap bisection.
NUMERICS: two bugs found and fixed (operand A uploaded from B's host copy -> Add computed B+B; host readers
of stale tensors) -> unlimited DEVADD=1 is bit-exact on all 5 golden cases at N=1 and N=4 (SNDR 69.70 /
69.93 / 70.09 / 67.90 / 67.45, max|d| 2.258e-04).
PERFORMANCE: +261 ms at N=4 (6/6 order-alternating pairs, mean 2325 ON vs 2064 OFF). Mechanism: the
staging copies it removes were hidden behind GPU work (run 85/88), while the 57 added device passes over
~120 MB are real SM time on a saturated GPU, and every chain terminator ends in cudaStreamSynchronize,
destroying the conv chain's async issue-ahead. => DEVICE RESIDENCY IS CLOSED BY MEASUREMENT (not by risk).
Approval list is now 4 items: batch-4 lockstep, CUDA graphs, server-mode weights, Winograd F(2,3) K=3.
Session best 2172 ms (run 305, default path).

## Phase 5 closures (runs 305-314): every remaining project lever closed by measurement or arithmetic
- **Device residency (PT_V21_DEVADD)**: implemented fully (device Add + fused epilogues, device
  LeakyReLU, sub-arena allocator with last-use release, producer-direction eligibility, chain
  terminators, generic host-reader safety sweep). BIT-EXACT on all 5 cases (N=1 and N=4) but
  **+261 ms at N=4** (6/6 pairs, 2325 vs 2064). Copies were hidden behind GPU work; the added device
  passes are not. Scaffold stays committed inert (default 0). Runs 304-305.
- **Winograd F(2,3) on K=3**: killed by arithmetic, no silicon. These are 1-D convs: direct = 5
  ops/output; F(2,3) = 3 input-transform adds + 2 mults + 3 output-transform adds = 8 ops/output
  (1.6x MORE work). The 2.25x figure is the 2-D number (transforms amortize over rows). Run 306.
- **Batch-4 lockstep driver**: killed by measurement before implementation. A per-node barrier probe
  (PT_V21_BAR=1) costs **+390 ms** (4/4 pairs, 2462 vs 2072): lockstep forces streams that are
  naturally desynchronized by 25-100 ms to align at every point, making each node max-over-streams.
  The tail/packing gains batch-4 targets are exactly what natural multiplexing already harvests
  (run 87: zero SM idle during inference). Run 314.
- **Upload claim-order rebalance**: killed by mechanism -- one serialized copy engine means engine
  completion (~139 ms) is partition-invariant. Run 311.
- **CUDA graphs**: launch overhead ~1-2 ms, below bar (runs 87/108). **Server-mode weights**: out of
  benchmark scope (timer-scope gaming, run 79 integrity note).
- Deployment-quality (run 312): 12 consecutive N=4 runs -- cold draw 2293, then stable 2057-2091
  (mean 2075), SNDR 70.09 every run, RAM flat (no leak), no thermal drift. Sustained-load regime is
  ~165 ms faster than cold single-shot: the clock ramp + first-touch cost of a cold process.
- Case coverage complete on BOTH boxes with the shipped binary (runs 308, 313): parity identical
  everywhere; pesi 3-7% faster on 4/5 cases.
- **The session is at the floor of this architecture.** Phase-5 cluster: 2284, 2273, 2172, 2242, 2290
  (mean ~2252). Recommended: autoresearch-finalize with the two closure proofs and the soak evidence.

## Phase 5 (runs 316-318): GPU kernel levers closed from three directions
- **Window-direct conv (PT_V21_WIN)**: implemented, integrated, debugged, and reverted. Bench (identical
  math, bit-exact vs a padded reference): 39.8 ms on cin=cout=256, K=11, T=904 vs ~45 ms for the ig8
  kernel on the same node => **parity (+12%, below bar)**; in the runtime the win path cost +50% conv
  time (944 vs 615 ms). Killed by measurement. Bench artifacts kept: `.auto/bench_fft.cu`,
  `.auto/stage_test.cu`, `.auto/win_test.cu` (the last two localized the bug: the write index used the
  output-loop counter as the channel, scattering one thread's 8 time accumulators over 8 channels).
- **FFT overlap-add conv**: killed by measurement (221 ms = 10x slower than direct at 2.4 GFLOP/s,
  despite the predicted 2.5x op reduction; 32-bin per-thread state spills).
- **Winograd F(2,3)**: killed by arithmetic (1-D convs: 8 ops/output vs 5 direct = 1.6x MORE work).
- **BENCHMARK LESSON (write it down)**: the "ig8 is 2.3x off achievable" reading came from a bench
  kernel whose grid covered ~2.3x more output slots than the useful outputs (over-coverage = wasted
  threads = inflated rate). Normalize grid coverage against the useful output count before believing
  any kernel speed ratio.
- GPU component status: saturated SMs (run 87), wave tails hidden by cross-stream backfill (run 78),
  scheduling lockstep costs +390 ms (run 314), data placement costs +261 ms (run 305), algorithm
  variants at parity or worse (runs 316-317). The kernels are at their local optimum for this model.
- Phase-5 cluster: 2284, 2273, 2172, 2242, 2290, 2169, 2242(probe), 2191 (mean ~2233).

## PHASE 6 (runs 324+, 1009-iteration budget): one keep, and startup closed with a mechanism
- **KEPT (run 325): convT ig8.** `pt_convT_kernel_ig8` = the conv kernel's 128-column tiling applied to
  the convT phase formulation (8 q-values x 4 channels per thread, Xs[16][32], grid.x = ceil(nq/128)).
  convT 37.1 -> 33.4 ms/stream, -13 ms over 7 clean N=4 matched pairs, **bit-exact** on all 5 cases
  (same accumulation order per output), official draw 2155 = session best at the time.
  `PT_V21_CONVT_IG8=0` = bisection endpoint. Validated by the full per-node gate: 4010 nodes, 0 over
  gate, M2 PASS (run 327).
- **The per-node gate recipe (run 327), write it down:** on the Nano, dump with `PT_V21_CHAIN=0
  PT_V21_FUSE=0 ./build/pt_v21_nano --cuda --checkpoints /tmp/names.txt --ckdir /tmp/ck81 --out ...`
  (names = one line per graph output from the ONNX, 4071 of them), then a SECOND plain run for the
  driver's parity line, write it to `<ckdir>/_parity.txt`, rsync the dump dir to the host and run
  `.venv/bin/python tests/v21_parity_c.py --from-dumps <dir>`. 247 MB of dumps, ~1 min per case.
- **The 5th-thread prewarm is a tie (run 326)** and the mechanism is why: `pt_v21_open`'s prepare step
  is guarded by `open_lock`, not by the CUDA context, so streams 1..N already start the record pass at
  t~1 ms while stream 0 is inside `cudaFree(0)`. Startup is max(context 85, record pass ~71) + upload.
- **ig16 (256-column conv tiles) KILLED in the kernel bench** (run 328): +10-31% on the conv total.
  Xs[16][64] = 20 KB of shared halves blocks/SM (50% occupancy on a 2-SM part) and the weight
  re-read saving is an order of magnitude smaller. Column tiling is now: ig 64 (old) -> ig8 128 (kept)
  -> ig16 256 (killed).
- **Templated-K window conv KILLED at N=4 (run 329) even though it is 1.9x FASTER in the isolated
  bench** (19.4 vs 36.7 ms on node 3854, bit-exact): N=4 wall 2700 vs 2066 ms, 4/4 pairs. Mechanism:
  the register-window design issues K+7 scalar global loads per thread per input channel, so it is
  load-issue bound -- invisible on an idle GPU with warm L1, fatal when 4 streams share 2 SMs.
  **BENCHMARK LESSON #2: a single-kernel bench cannot predict the N=4 regime for anything that changes
  the instruction mix; only wall-time matched pairs can.** (Lesson #1 was grid over-coverage.)
- **The `PT_V21_IG8_MINT` knob is closed (run 329)**: forcing the 64-column kernel for Tout < 256 is
  +20 ms at N=4 on T81 (5/5 pairs) and neutral on T20. ig8 everywhere is correct.
- **STARTUP IS CLOSED WITH A MECHANISM (runs 330-332).** Per-stream phase timings at N=4 (temporary
  `PT_V21_PHASE`): open 71, `pt_v21_cuda_init` 216, run 1750, tail 3 -- so serialized startup is
  **287 ms**, not the ~140 the phase-4 budget assumed. The upload's engine time is 139-170 ms at
  **0.8 GB/s** while the same copy reaches 2.4 GB/s at max clock, and the clock trace
  (768 -> 384 -> 691 -> 768 -> 844 -> 921 over ~0.6 s) shows why: the copies run at 384-768 MHz.
  Three ways to attack it, all measured, all neutral-or-worse:
  (a) hold the clock with a 100 ms SM kernel joined after the upload: **+45 ms, 6/6** (heat/power);
  (b) zero-copy slab (cudaHostAlloc mapped, CPU fill at 3.75 GB/s = 19-30 ms instead of 139-170):
  **+15 ms, 5/6** -- the copies were inadvertently PAYING FOR THE RAMP, so removing them moves the
  ~600 ms ramp into the inference phase (+10 ms/stream of kernels at the lower clock);
  (c) coalescing the copies (336 records over 4 streams 47.8 ms == one coalesced copy 47.4 ms at max
  clock) and pinning the source (mmap/POPULATE, MADV_HUGEPAGE, cudaMallocHost, malloc all 2.0-2.4
  GB/s at max clock; cudaHostRegister on an anonymous mmap = "operation not supported"): no lever.
  **Consequence: the serialized 287 ms is bounded below by the governor's clock ramp. Any change that
  removes GPU work from the startup window makes the wall worse, and any change that adds SM load to
  hold the clock costs more in heat/power than it saves.**
- **Benchmark artifacts kept:** `.auto/bench_fft.cu` (FFT / window / templated-K kernels),
  `.auto/stage_test.cu`, `.auto/win_test.cu` (unit tests that localized two real bugs),
  `.auto/peak_test.cu` (the box's real fp32 rate: 97.6 GFMA/s at max clock, and it needs ~500 blocks
  before the clock ramps), `.auto/h2d_test*.cu` (upload-rate table), `.auto/zcopy_test.cu`.
  These are standalone (never linked into the runtime).
- Phase-6 cluster (official draws): 2203 (baseline), 2155 (convT ig8). Warm-pair cluster ~2065-2090.
