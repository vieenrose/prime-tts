#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <cuda_runtime.h>
/* The runtime's actual upload shape: ~336 records (~330 KB each) issued as cudaMemcpyAsync from
 * 4 host threads on 4 non-blocking streams. Compare with coalesced variants. */
static double best = 1e9;
static void run(int variant, void *src, float *dst, size_t bytes, int nch, cudaStream_t *st, int ns) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  size_t chunk = bytes / nch;
  for (int rep = 0; rep < 3; rep++) {
    for (int i = 0; i < ns; i++)
      cudaStreamSynchronize(st[i]);
    for (int i = 0; i < ns; i++)
      cudaEventRecord(e0, st[i]);
    if (variant == 0) {   /* per-record copies, round robin over ns streams */
      for (int i = 0; i < nch; i++) {
        size_t off = (size_t) i * chunk;
        size_t len = (i == nch - 1) ? bytes - off : chunk;
        cudaMemcpyAsync((char *) dst + off, (char *) src + off, len, cudaMemcpyHostToDevice,
                        st[i % ns]);
      }
    } else {   /* one coalesced copy per stream */
      size_t per = bytes / ns;
      for (int i = 0; i < ns; i++) {
        size_t off = (size_t) i * per;
        size_t len = (i == ns - 1) ? bytes - off : per;
        cudaMemcpyAsync((char *) dst + off, (char *) src + off, len, cudaMemcpyHostToDevice, st[i]);
      }
    }
    double ms = 0;
    for (int i = 0; i < ns; i++) {
      cudaEventRecord(e1, st[i]);
      cudaStreamSynchronize(st[i]);
      float m;
      cudaEventElapsedTime(&m, e0, e1);
      if (m > ms) ms = m;
    }
    if (ms < best) best = ms;
  }
}
int main() {
  size_t bytes = 111u * 1048576u;
  int nch = 336;
  float *dst;
  cudaMalloc(&dst, bytes);
  void *src = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  memset(src, 7, bytes);
  cudaStream_t st[4];
  for (int i = 0; i < 4; i++)
    cudaStreamCreateWithFlags(&st[i], cudaStreamNonBlocking);
  const char *nm[4] = {"336 recs / 1 stream", "336 recs / 4 streams", "coalesced / 1 stream",
                       "coalesced / 4 streams"};
  for (int v = 0; v < 4; v++) {
    best = 1e9;
    int ns = (v % 2) ? 4 : 1;
    for (int w = 0; w < 2; w++) {   /* two passes, keep the best (clock ramp) */
      double prev = best;
      run(v, src, dst, bytes, nch, st, ns);
      if (best > prev) best = prev;
    }
    printf("%-22s %7.1f ms  %.2f GB/s\n", nm[v], best, bytes / (best * 1e6));
  }
  return 0;
}
