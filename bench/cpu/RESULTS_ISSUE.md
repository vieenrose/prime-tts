# A57 NEON fp32 issue peak on this Nano — measured (partially; see caveats)

Probe: `bench/cpu/issue_probe.c` + `bench/cpu/issue_kern.S` — 16 independent NEON fp32 chains, no
memory traffic in the loop. AArch64 mnemonics are `fmla/fmul/fadd` (the AArch32 `vmla` name is
rejected by this assembler).
Build/run (Nano, /tmp):
```
gcc -O2 -std=gnu11 -D_GNU_SOURCE -o issue_probe issue_probe.c issue_kern.S
taskset -c 0 ./issue_probe 2000000
```

## Timing caveat (measured, not assumed)

Every clock source on this box misbehaves for short intervals in a fresh process:
`clock_gettime(CLOCK_MONOTONIC)` and `CLOCK_MONOTONIC_COARSE` returned 0 at the first call (dt came
out equal to the full uptime, 716444 s), `clock()` returned nan, and `getrusage(RUSAGE_SELF)`
returns **cumulative** CPU time, so the printed `dt` is cumulative from process start. Valid
numbers come from the **difference between consecutive reps**, which is what I use below.
Measured `cpu0 scaling_cur_freq` during the probe: 825.6 MHz (the governor drops frequency under this
load; the MAXN max is 1479 MHz).

## Numbers (2,000,000 iterations, pinned to cpu0)

| rep | cumulative CPU time | delta | FLOPs/iter | GFLOP/s at the delta | FLOP/cycle at 825.6 MHz |
|---|---|---|---|---|---|
| 0 | 0.032 s | 0.032 | 128 (fmla x16) | 8.0 | 9.7 |
| 1 | 0.060 s | 0.028 | 128 | **9.14** | **11.1** |
| 2 | 0.088 s | 0.028 | 128 | 9.14 | 11.1 |

`fmul.4s`/`fadd.4s` (64 FLOPs/iter) measured ~0.112 s cumulative for the next block, consistent
with ~2 ops/cycle issue.

## What this settles

**The 8 FLOP/cycle/core assumption (11.8 GFLOP/s at 1479 MHz) is not the ceiling: `fmla` achieves
~11 FLOP/cycle/core at the reduced 825 MHz clock, i.e. ~9.1 GFLOP/s, and the 2-pipe 16 FLOP/cycle
model (23.5 GFLOP/s at 1479 MHz) is the right theoretical peak.** Practical consequence, using our
production kernel measurement (9.76 GFLOP/s for the 4x16 tile with L1-resident operands):
the runtime is at ~40% of the 16-FLOP/cycle peak and ~1.07x the *achieved* pure-issue rate at the
clocked-down frequency. For planning: **~9-10 GFLOP/s per core, ~36-40 GFLOP/s at 4 cores** for
conv-style code, which is why 4 v2.1 streams (11.9 GFLOP/audio-second x 4 = 47.6 GFLOP/s) are out of
reach on CPU and belong on the GPU (measured fp32 GEMM 62 GF/s, half2 121-171 GF/s).

## Not measured (open)

4-thread aggregate of this probe (the deltas-per-rep method is awkward for parallel runs);
per-op-mix peak at a pinned 1479 MHz (the governor downclocks to ~825 MHz under the probe).
