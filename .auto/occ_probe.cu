/* Occupancy probe (standalone diagnostic; NEVER linked into the runtime).
 *
 * Why: ptxas says pt_conv_kernel_ig8 uses 64 registers + 12288 B static shared, and cudaGetDeviceProperties
 * reports this Tegra X1 as ONE SM with 2048 threads/SM, 64 KB shared/SM and 65536 registers/SM. The resident
 * blocks/SM is therefore min(regs: 4, threads: 8, shared: 5 by raw size) -- and the shared term depends on the
 * *shared/L1 carveout*, which nothing in the runtime sets. Run 81 measured cudaFuncCachePreferL1 (a smaller
 * shared carveout) and generalized "L1/shared tuning is exhausted"; that kill is explained by exactly this
 * occupancy term, and the opposite direction has never been measured. This probe prints the API's answer for
 * the real kernels, default vs carveout=100%, so the kernel work is closed (or opened) with data.
 *
 * Build on the Nano:
 *   /usr/local/cuda/bin/nvcc -arch=sm_53 -O3 .auto/occ_probe.cu -I runtime/v21/src -I runtime/v21/gen -o /tmp/occ_probe -lcuda
 * Run: /tmp/occ_probe
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

/* bring in the real kernel definitions (the .cu has no main) */
#include "pt_v21_cuda.cu"

struct Q { const char *name; const void *f; int threads; size_t dyn; };

static void probe(Q q) {
  cudaFuncAttributes a;
  cudaError_t e = cudaFuncGetAttributes(&a, q.f);
  if (e != cudaSuccess) { printf("%-24s attrs: %s\n", q.name, cudaGetErrorString(e)); return; }

  int def = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&def, q.f, q.threads, q.dyn);


  /* ask for the maximum shared carveout and re-query */
  cudaError_t se = cudaFuncSetAttribute(const_cast<void *>(q.f),
                                        cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  int after = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&after, q.f, q.threads, q.dyn);

  printf("%-24s regs %3d  static_smem %6zu B  probe_dyn %6zu B  "
         "blocks/SM: default %d -> carveout100 %d   threads/SM %d -> %d of %d   set:%s\n",
         q.name, a.numRegs, a.sharedSizeBytes, q.dyn, def, after,
         def * q.threads, after * q.threads, 2048,
         se == cudaSuccess ? "ok" : cudaGetErrorString(se));
}

int main() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s  SMs %d  threads/SM %d  regs/SM %d  shared/SM %zu  shared/block %zu  clock %.0f MHz\n"
         "fp32 lane FMA peak = %d SM x %d lanes x 2 x %.3f GHz = %.1f GFLOP/s (%.1f GFMA/s)\n\n",
         p.name, p.multiProcessorCount, p.maxThreadsPerMultiProcessor, p.regsPerMultiprocessor,
         p.sharedMemPerMultiprocessor, p.sharedMemPerBlock, p.clockRate / 1000.0,
         p.multiProcessorCount, p.maxThreadsPerMultiProcessor * 2 / 32 /*lanes/SM ~ 128*/,
         p.clockRate / 1e6,
         p.multiProcessorCount * 128 * 2 * (p.clockRate / 1e6),
         p.multiProcessorCount * 128 * (p.clockRate / 1e6));

  Q qs[] = {
    { "conv_ig8",       (const void *) pt_conv_kernel_ig8,       256, 0 },
    { "convT_ig8",      (const void *) pt_convT_kernel_ig8,      256, 0 },
    { "conv_ig",        (const void *) pt_conv_kernel_ig,        256, 0 },
    { "convT_ig",       (const void *) pt_convT_kernel_ig,       256, 0 },
    { "conv_wsm",       (const void *) pt_conv_kernel_wsm,       512, 49152 },
    { "convT_wsm",      (const void *) pt_convT_kernel_wsm,      256, 49152 },
    { "conv_plain",     (const void *) pt_conv_kernel,           256, 0 },
    { "convT_plain",    (const void *) pt_convT_kernel,          256, 0 },
    { "convT_ph",       (const void *) pt_convT_kernel_ph,       256, 0 },
    { "matmul_tiled",   (const void *) pt_matmul_kernel_t,       256, 0 },
    { "matmul_plain",   (const void *) pt_matmul_kernel,         256, 0 },
  };
  for (size_t i = 0; i < sizeof(qs) / sizeof(qs[0]); i++)
    probe(qs[i]);

  /* dynamic-shared launch limits for the wsm family (they ask for up to 48 KB) */
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess)
    printf("last error: %s\n", cudaGetErrorString(e));
  return 0;
}
