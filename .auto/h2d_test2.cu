#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <cuda_runtime.h>
/* Interleaved (order-alternating) H2D rate for each host source kind: the first measurement in a
 * process runs at a lower GPU clock, so every kind is measured 3 times in rotating order. */
int main(int argc, char **argv) {
  size_t bytes = argc > 1 ? (size_t) atol(argv[1]) * 1048576u : 111u * 1048576u;
  float *dst;
  cudaMalloc(&dst, bytes);
  const int NK = 4;
  const char *nm[NK] = {"mmap+POPULATE", "mmap+HUGEPAGE", "cudaMallocHost", "malloc(pageable)"};
  void *src[NK];
  src[0] = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  src[1] = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  madvise(src[1], bytes, MADV_HUGEPAGE);
  cudaMallocHost(&src[2], bytes);
  src[3] = malloc(bytes);
  for (int k = 0; k < NK; k++)
    memset(src[k], k + 1, bytes);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  for (int k = 0; k < NK; k++)
    cudaMemcpy(dst, src[k], bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  double best[NK];
  for (int k = 0; k < NK; k++)
    best[k] = 1e9;
  for (int rep = 0; rep < 3; rep++)
    for (int j = 0; j < NK; j++) {
      int k = (j + rep) % NK;   /* rotate the order every repetition */
      cudaEventRecord(e0);
      cudaMemcpy(dst, src[k], bytes, cudaMemcpyHostToDevice);
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      if (ms < best[k]) best[k] = ms;
    }
  for (int k = 0; k < NK; k++)
    printf("%-16s best %7.1f ms  %.2f GB/s\n", nm[k], best[k], bytes / (best[k] * 1e6));
  return 0;
}
