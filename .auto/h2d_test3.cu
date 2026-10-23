#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <cuda_runtime.h>
/* Price the per-copy overhead of the weight upload: 111 MB as ONE copy vs as N chunked copies of the
 * same bytes (the runtime issues ~336 per-record cudaMemcpyAsync calls, ~0.8 GB/s end to end). */
int main(int argc, char **argv) {
  size_t bytes = 111u * 1048576u;
  int nch = argc > 1 ? atoi(argv[1]) : 336;
  float *dst;
  cudaMalloc(&dst, bytes);
  void *src = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  memset(src, 7, bytes);
  cudaStream_t st;
  cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  double best1 = 1e9, bestN = 1e9;
  for (int rep = 0; rep < 3; rep++) {
    for (int order = 0; order < 2; order++) {
      cudaEventRecord(e0, st);
      if (order == 0) {
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, st);
      } else {
        size_t chunk = bytes / nch;
        for (int i = 0; i < nch; i++) {
          size_t off = (size_t) i * chunk;
          size_t len = (i == nch - 1) ? bytes - off : chunk;
          cudaMemcpyAsync((char *) dst + off, (char *) src + off, len, cudaMemcpyHostToDevice, st);
        }
      }
      cudaEventRecord(e1, st);
      cudaStreamSynchronize(st);
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      if (order == 0 && ms < best1) best1 = ms;
      if (order == 1 && ms < bestN) bestN = ms;
    }
  }
  printf("%d chunks:  one copy %6.1f ms (%.2f GB/s)   chunked %6.1f ms (%.2f GB/s)   overhead %+.1f ms\n",
         nch, best1, bytes / (best1 * 1e6), bestN, bytes / (bestN * 1e6), bestN - best1);
  return 0;
}
