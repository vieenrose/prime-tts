#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <cuda_runtime.h>
/* What is the H2D upload rate for each host source kind on this box?
 * The runtime's 111 MB weight upload measures ~0.8 GB/s from the mmap'd payload arena. */
static float bench(void *src, size_t bytes, float *dst) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);   /* warm */
  cudaDeviceSynchronize();
  cudaEventRecord(e0);
  cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  return ms;
}
int main(int argc, char **argv) {
  size_t bytes = argc > 1 ? (size_t) atol(argv[1]) * 1048576u : 111u * 1048576u;
  const char *file = argc > 2 ? argv[2] : 0;
  float *dst;
  cudaMalloc(&dst, bytes);
  void *a = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  memset(a, 1, bytes);
  float ms = bench(a, bytes, dst);
  printf("anon mmap+POPULATE  %7.1f ms  %.2f GB/s\n", ms, bytes / (ms * 1e6));
  void *ah = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  madvise(ah, bytes, MADV_HUGEPAGE);
  memset(ah, 4, bytes);
  ms = bench(ah, bytes, dst);
  printf("anon mmap+HUGEPAGE  %7.1f ms  %.2f GB/s\n", ms, bytes / (ms * 1e6));
  void *ap = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  memset(ap, 5, bytes);
  ms = bench(ap, bytes, dst);
  printf("anon mmap (no pop)  %7.1f ms  %.2f GB/s\n", ms, bytes / (ms * 1e6));
  void *p = 0;
  if (cudaMallocHost(&p, bytes) == cudaSuccess) {
    memset(p, 2, bytes);
    ms = bench(p, bytes, dst);
    printf("cudaMallocHost      %7.1f ms  %.2f GB/s\n", ms, bytes / (ms * 1e6));
  } else {
    printf("cudaMallocHost      FAILED\n");
  }
  void *h = malloc(bytes);
  memset(h, 3, bytes);
  ms = bench(h, bytes, dst);
  printf("malloc (pageable)   %7.1f ms  %.2f GB/s\n", ms, bytes / (ms * 1e6));
  if (file) {
    FILE *f = fopen(file, "rb");
    if (f) {
      int fd = fileno(f);
      void *m = mmap(0, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
      if (m != MAP_FAILED) {
        ms = bench(m, bytes, dst);
        printf("file mmap (ptk)     %7.1f ms  %.2f GB/s\n", ms, bytes / (ms * 1e6));
        if (cudaHostRegister(m, bytes, cudaHostRegisterDefault) == cudaSuccess) {
          ms = bench(m, bytes, dst);
          printf("file mmap+Register  %7.1f ms  %.2f GB/s\n", ms, bytes / (ms * 1e6));
        } else {
          printf("file mmap+Register  FAILED (%s)\n", cudaGetErrorString(cudaGetLastError()));
        }
      }
      fclose(f);
    }
  }
  if (a && cudaHostRegister(a, bytes, cudaHostRegisterDefault) == cudaSuccess) {
    ms = bench(a, bytes, dst);
    printf("anon mmap+Register  %7.1f ms  %.2f GB/s\n", ms, bytes / (ms * 1e6));
  } else {
    printf("anon mmap+Register  FAILED (%s)\n", cudaGetErrorString(cudaGetLastError()));
  }
  return 0;
}
