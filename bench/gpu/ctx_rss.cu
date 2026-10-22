// RSS cost of a CUDA context on Tegra (unified memory): VmRSS/VmHWM before/after cudaFree(0) and a 64 MB alloc.
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
static long kb(const char *key) {
  FILE *f = fopen("/proc/self/status", "r"); char l[256]; long v = -1;
  while (fgets(l, sizeof l, f)) if (!strncmp(l, key, strlen(key))) v = atol(l + strlen(key));
  fclose(f); return v;
}
int main() {
  printf("start          VmRSS %6ld kB  VmHWM %6ld kB\n", kb("VmRSS:"), kb("VmHWM:"));
  cudaFree(0);
  printf("after ctx      VmRSS %6ld kB  VmHWM %6ld kB\n", kb("VmRSS:"), kb("VmHWM:"));
  size_t fr, tot; cudaMemGetInfo(&fr, &tot);
  printf("cudaMemGetInfo free %zu MB total %zu MB\n", fr >> 20, tot >> 20);
  void *p; cudaMalloc(&p, 64 << 20); cudaMemset(p, 0, 64 << 20); cudaDeviceSynchronize();
  printf("after 64MB dev VmRSS %6ld kB  VmHWM %6ld kB\n", kb("VmRSS:"), kb("VmHWM:"));
  cudaMemGetInfo(&fr, &tot);
  printf("cudaMemGetInfo free %zu MB\n", fr >> 20);
  int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrCooperativeLaunch, 0); printf("cudaDevAttrCooperativeLaunch=%d\n", v);
  cudaDeviceGetAttribute(&v, cudaDevAttrConcurrentKernels, 0); printf("ConcurrentKernels=%d\n", v);
  cudaDeviceGetAttribute(&v, cudaDevAttrCanMapHostMemory, 0); printf("CanMapHostMemory=%d\n", v);
  cudaDeviceGetAttribute(&v, cudaDevAttrIntegrated, 0); printf("Integrated=%d\n", v);
  cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlock, 0); printf("smem/block=%d\n", v);
  cudaDeviceGetAttribute(&v, cudaDevAttrMaxRegistersPerMultiprocessor, 0); printf("regs/SM=%d\n", v);
  cudaDeviceGetAttribute(&v, cudaDevAttrClockRate, 0); printf("clock=%d kHz\n", v);
  cudaDeviceGetAttribute(&v, cudaDevAttrMemoryClockRate, 0); printf("memclock=%d kHz\n", v);
  cudaDeviceGetAttribute(&v, cudaDevAttrGlobalMemoryBusWidth, 0); printf("buswidth=%d\n", v);
  return 0;
}
