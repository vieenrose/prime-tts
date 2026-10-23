#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <time.h>
#include <cuda_runtime.h>
static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
int main() {
  size_t bytes = 111u * 1048576u;
  void *src = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  memset(src, 7, bytes);
  /* 1) CPU memcpy into a plain anon buffer (baseline host-to-host rate) */
  void *dstp = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  double best = 1e9;
  for (int i = 0; i < 3; i++) {
    double t = now_ms();
    memcpy(dstp, src, bytes);
    double d = now_ms() - t;
    if (d < best) best = d;
  }
  printf("CPU memcpy -> anon       %7.1f ms  %.2f GB/s\n", best, bytes / (best * 1e6));
  /* 2) CPU memcpy into a cudaHostAlloc(mapped) buffer: the zero-copy weight slab */
  void *mp = 0;
  if (cudaHostAlloc(&mp, bytes, cudaHostAllocMapped) != cudaSuccess) {
    printf("cudaHostAlloc(mapped) FAILED: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
  }
  best = 1e9;
  for (int i = 0; i < 3; i++) {
    double t = now_ms();
    memcpy(mp, src, bytes);
    double d = now_ms() - t;
    if (d < best) best = d;
  }
  printf("CPU memcpy -> mapped     %7.1f ms  %.2f GB/s\n", best, bytes / (best * 1e6));
  void *devp = 0;
  cudaError_t e = cudaHostGetDevicePointer(&devp, mp, 0);
  printf("HostGetDevicePointer: %s (host %p dev %p same=%d)\n", cudaGetErrorString(e), mp, devp,
         mp == devp);
  /* 3) GPU read bandwidth from the mapped buffer vs a device buffer */
  cudaMalloc(&dstp, bytes);
  cudaMemcpy(dstp, src, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  return 0;
}
