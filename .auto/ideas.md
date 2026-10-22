# Ideas backlog (v2.1 Nano latency)

Ordered by expected payoff per unit of risk. Measure the phase split before and after each.

- **Host/GPU overlap.** Single stream is ~4.44 s: ~2.16 s GPU kernels, ~2.2 s host graph work
  serialized against it. The graph is a strict chain, but each conv's *input staging copy* and the
  host elementwise ops that feed the next node can run while the current kernel is in flight
  (record event per op, defer sync to the point where the host actually needs the result).
- **CUDA Graphs per stream.** ~180 launches (124 conv + 5 convT + 50 matmul) each with per-call
  copy/launch/sync; capture the per-request launch sequence once per shape bucket and replay with
  `cudaGraphLaunch`. Needs stable device addresses (the arena gives that) and a fixed shape bucket
  per T.
- **Batch the four streams per layer.** One launch per conv covering all four requests
  (grid.z = stream count, per-stream row stride). Removes 3/4 of the launches and amortizes weight
  reads. Requires the driver to interleave four graph loops in lockstep (shared node loop, per-stream
  pt_ctx), not four independent threads.
- **Pinned host staging buffers** (`cudaMallocHost`) for conv input/output staging: async copies get
  roughly 2x bandwidth, and overlap becomes possible.
- **Vectorize the host elementwise ops** on aarch64 (NEON fp32): the interpreter's
  `bin_op`/`un_op`/`Cast`/`Concat`/`Transpose`/`Expand`/`Gather` loops are scalar. Measured A57 NEON
  headroom is large (memcpy ~5-5.9 GB/s, fp32 NEON GEMM ~3.74 GFLOP/s/core).
- **Constant folding in the generator.** 1924 of 4066 nodes are `Constant`; they are materialized as
  fresh tensors every request. Bake them into read-only pooled tables with no per-run copy, and drop
  chains of Constants feeding shape ops.
- **Shape-only op elision.** Reshape/Transpose/Squeeze/Unsqueeze/Expand/Copy that only rename a view
  should not memcpy; carry a strided view (offsets + strides) through the interpreter and materialize
  only at kernels that need contiguity.
- **Kernel tuning** (`PT_V21_GPU=conv` to isolate): split cin across threads with shared-memory
  reduction for the big encoder convs (cin=192..768, K=1..3), threads 128 vs 256, `__ldg`, float4
  loads on 4-aligned padded rows, occupancy check at T81 shapes.
- **fp16 activations in the decoder** (its weights are already fp16; measured rel 1.9e-3 today):
  halves activation traffic in the deepest group. Gate hard on SNDR and the per-node gate.
- **Reduce staging volume**: keep decoder activations resident on device across the whole dec group
  so the iSTFT path does not round-trip over PCIe per node.
- **Arena sizing/alignment**: 32 MB per stream is a guess; measure the true high-water mark at T120
  and cache by size class so the bump pointer never fragments.
- **Launch coalescing for tiny ops**: 26 dynamic matmuls + 50 matmul calls are latency-bound; batch
  same-shape ones into one grid.
