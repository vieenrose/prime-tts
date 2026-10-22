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
- **Constant folding in the generator.** 1924 of 4066 nodes are `Constant`; they are materialized as
  fresh tensors every request. Bake them into read-only pooled tables with no per-run copy.
- **Shape-only op elision.** Reshape/Transpose/Squeeze/Unsqueeze/Expand that only rename a view
  should not memcpy; carry a strided view and materialize only at kernels needing contiguity.
- **Encoder convs (cin=768):** the 4-channel weight slice is 36 KB so they fall back to the unsliced
  kernel. A float2-lane slice halves that to 18 KB and fits shared.

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

## Killed by measurement (do not retry without new evidence)
- **fp16 conv activations**: SNDR 43.73 dB, far below the gate.
- 4 output columns per thread (16 accumulators): 39% slower.
- 2-channel conv blocks: 52% slower.
- Launch-shape heuristic (block size from output length): +10.5% at N=4.
- Conv reduction chunk 16 -> 32: barriers are not the limiter; 16 KB tiles cost more.
- Small 32x32 tiles for short outputs: arithmetic intensity matters more than block count.
- Padé-approximant / coarse-table tanh+sigmoid: SNDR 67.4 / 64.3 dB. The 65536-point interpolated
  table keeps 70.09 dB and is kept (TANH 62 -> 19 ms).
