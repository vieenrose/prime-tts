#include <cstdio>
#include <cuda_runtime.h>
/* calibrate the box: what fp32 FMA rate does a pure-register kernel actually reach?
 * Each thread issues 8 independent FMAs per iteration (the ig8 accumulator pattern), no memory. */
__global__ void __launch_bounds__(256) fma_peak(float *out, int iters) {
  float a0 = threadIdx.x * 1e-6f, a1 = a0 + 1e-6f, a2 = a0 + 2e-6f, a3 = a0 + 3e-6f;
  float a4 = a0 + 4e-6f, a5 = a0 + 5e-6f, a6 = a0 + 6e-6f, a7 = a0 + 7e-6f;
  float w = 1.0000001f, v = 1e-7f;
  for (int i = 0; i < iters; i++) {
    a0 = fmaf(w, v, a0); a1 = fmaf(w, v, a1); a2 = fmaf(w, v, a2); a3 = fmaf(w, v, a3);
    a4 = fmaf(w, v, a4); a5 = fmaf(w, v, a5); a6 = fmaf(w, v, a6); a7 = fmaf(w, v, a7);
  }
  if (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 == 12345.f)
    out[threadIdx.x] = a0;
}
int main() {
  int blocks = 32, threads = 256, iters = 20000;
  float *d;
  cudaMalloc(&d, 256 * 4);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  fma_peak<<<blocks, threads>>>(d, 1000);   /* warm */
  cudaDeviceSynchronize();
  cudaEventRecord(e0);
  fma_peak<<<blocks, threads>>>(d, iters);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  double fma = (double) blocks * threads * iters * 8;
  printf("%d blocks x %d threads: %.3f ms  %.1f GFMA/s  %.1f GFLOP/s\n", blocks, threads, ms,
         fma / (ms * 1e6), 2.0 * fma / (ms * 1e6));
  int fb[4] = {16, 64, 128, 512};
  for (int i = 0; i < 4; i++) {
    fma_peak<<<fb[i], threads>>>(d, 1000);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    fma_peak<<<fb[i], threads>>>(d, iters);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    double f2 = (double) fb[i] * threads * iters * 8;
    printf("%4d blocks: %.3f ms  %.1f GFMA/s\n", fb[i], ms, f2 / (ms * 1e6));
  }
  return 0;
}
