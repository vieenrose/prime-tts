# Jetson Nano GPU roofline (measured) - does v2.1 at 8 streams fit?

Device: Tegra X1 (`device=NVIDIA Tegra X1 cc=5.3 SMs=1 clock=921MHz mem=3964MB L2=256KB coop=0`).
`coop=0` means **cooperative launch is not available** on this device/driver, so a single-kernel
whole-network design with `grid.sync()` is off the table.
Commands: `picard@100.84.151.116:/tmp`, `/usr/local/cuda/bin/nvcc -O3 -arch=sm_53 -o sgemm sgemm.cu && ./sgemm`
(source `bench/gpu/sgemm.cu`; the program self-checks its own tiles against a CPU reference:
`selfcheck fp32 128^3 max|gpu-cpu| = 5.75e-06`, `selfcheck fp16 128^3 = 5.28e-03`).

## Pure-issue peak (register-resident FMA chains, no memory)

| kernel | measured |
|---|---|
| fp32 `fmaf` chains | **88.4 GFLOP/s** |
| `__hfma2` (half2) chains | **168.0 GFLOP/s** |

So sm_53 doubles fp32 issue in half2 (Maxwell half2 FMA). The 236 GFLOP/s "paper peak" is not
reachable by any kernel: issue tops out at 88 fp32 / 168 half2.

## Tiled GEMM (my kernel: 256 threads, 64x64 tile, 4x4 per thread)

| M x N x K | fp32 accumulate, fp32 store | fp32 accumulate, fp16 store | bytes/s |
|---|---|---|---|
| 256x256x256 | 10.0 GF/s | 29.1 GF/s | 0.24-0.46 GB/s |
| 512x512x512 | 36.4 GF/s | 64.5 GF/s | 0.43-0.50 GB/s |
| 1024x1024x1024 | 62.4 GF/s | 71.3 GF/s | 0.28-0.37 GB/s |
| **512x4000x512** (v2.1 decoder resblock shape) | **62.4 GF/s** | **70.0 GF/s** | 0.43-0.52 GB/s |
| 256x2000x256 | 60.2 GF/s | 65.3 GF/s | 0.80-1.00 GB/s |
| 192x2000x192 (encoder shape) | 57.6 GF/s | 61.7 GF/s | 1.00-1.26 GB/s |

Notes: memory bandwidth is never the binding constraint at these shapes (1-5% of the ~8 GB/s
device bandwidth); the constraint is fp32 issue. fp16 storage buys ~13% here because the kernel
converts to fp32 and accumulates in fp32. A kernel that accumulates in **half2** is the only way
toward the 168 GFLOP/s ceiling.

Other measured facts from the same board: tiny-kernel launch **26 us**; axpy-style device bandwidth
5.5 GB/s counted as 2 accesses per element = **8.3 GB/s** real.

## v2.1 compute cost (measured with tools/probe_shapes.py, T=81 -> 3.312 s of 16 kHz audio)

19.605 GMAC per utterance = **5.93 GMAC per audio-second = 11.9 GFLOP per audio-second**.
By op: Conv 16.05 GMAC, ConvTranspose 3.48 GMAC, everything else < 0.1 GMAC.
Biggest nodes: `/dec/ups.0/ConvTranspose` 1736 M, `/dec/ups.1/ConvTranspose` 1736 M, and twelve
decoder resblock Convs at ~597 M each.

## Feasibility arithmetic (measured rate vs measured cost)

| streams | GFLOP/s needed | fits at 62 GF/s (fp32) | fits at ~110 GF/s (half2, target) |
|---|---|---|---|
| 1 | 11.9 | yes, RTF 0.19 | yes, RTF 0.11 |
| 4 | 47.6 | yes, RTF 0.77 | yes, RTF 0.43 |
| **8** | **95.2** | **no** (max ~5 streams, RTF 1.54 aggregate) | yes, RTF 0.87 aggregate |
| 10 | 119 | no | marginal |

Verdict: **v2.1 at 8 real-time streams requires a half2-compute CUDA runtime** (fp16 weights and
activations, `__hfma2` inner loops, fp32 accumulate only where parity demands it). With the fp32
kernels measured here the ceiling is ~5 streams. The published ggml-CUDA number for this model on
this board is RTF 0.42 single-stream; a 62 GF/s fp32 runtime predicts 0.19, so the gain over that
runtime is fusion + batching (26 us per launch means kernel count, not FLOPs, dominates small
kernels), and CUDA Graphs with T-buckets on top.

Design consequences, from the numbers above:
1. Fuse to a small kernel count and batch all 8 streams inside each kernel (the N dimension is
   `T x streams`), otherwise 26 us x ~200 kernels = 5 ms per inference floor.
2. fp16 storage is free bandwidth/footprint (weights 117 MB -> 58 MB, which matters against ~185 MB
   free RAM for 8 streams); fp16 **compute** is what buys the 8th stream.
3. No cooperative launch: use CUDA Graphs (CUDA 10.2 `cudaStreamBeginCapture` ->
   `cudaGraphInstantiate` at startup only) per T-bucket, not per request.
4. The two ConvTranspose upsamplers are 18% of all compute; a space-to-depth (pixel-shuffle)
   formulation turns them into a GEMM with K = Cin*16 and removes the zero-fill waste.
