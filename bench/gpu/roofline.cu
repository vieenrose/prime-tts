// roofline.cu - Jetson Nano (Tegra X1, sm_53, CUDA 10.2) GPU roofline for the PrimeTTS v2.1 runtime.
//   /usr/local/cuda/bin/nvcc -O3 -arch=sm_53 -std=c++11 bench/gpu/roofline.cu -o build/roofline && ./build/roofline
// Sections: peak FMA (fp32 / half2), bandwidth (fp32 / fp16), tiled SGEMM (plain + implicit-conv, fp32 and
// fp16-storage), launch overhead, CUDA Graphs, cooperative launch, zero-copy / memcpy, cudaMalloc,
// 8 streams of small kernels vs 1 batched kernel.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <sys/time.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
static double now() { struct timeval t; gettimeofday(&t, 0); return t.tv_sec + t.tv_usec * 1e-6; }

struct Timer {
  cudaEvent_t a, b;
  Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
  void start(cudaStream_t s = 0) { cudaEventRecord(a, s); }
  float stop(cudaStream_t s = 0) { cudaEventRecord(b, s); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms; }
};

// warm the GPU clock (nvhost_podgov devfreq starts at 76.8 MHz)
__global__ void spin(float *p, int n) { float x = p[threadIdx.x]; for (int i = 0; i < n; i++) x = x * 0.999f + 0.001f; p[threadIdx.x] = x; }
static void warm(double sec) {
  float *p; CK(cudaMalloc(&p, 4096));
  double t0 = now();
  while (now() - t0 < sec) { spin<<<64, 256>>>(p, 20000); CK(cudaDeviceSynchronize()); }
  cudaFree(p);
}

// ---------------------------------------------------------------- peak FMA
__global__ void fma_f32(float *out, int iters) {
  float a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
  const float b = 0.9999f, c = 0.0001f;
  for (int i = 0; i < iters; i++) {
    a0 = fmaf(a0, b, c); a1 = fmaf(a1, b, c); a2 = fmaf(a2, b, c); a3 = fmaf(a3, b, c);
    a4 = fmaf(a4, b, c); a5 = fmaf(a5, b, c); a6 = fmaf(a6, b, c); a7 = fmaf(a7, b, c);
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
__global__ void fma_h2(half2 *out, int iters) {
  half2 a0 = __float2half2_rn(threadIdx.x * 1e-3f), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
  const half2 b = __float2half2_rn(0.999f), c = __float2half2_rn(0.001f);
  for (int i = 0; i < iters; i++) {
    a0 = __hfma2(a0, b, c); a1 = __hfma2(a1, b, c); a2 = __hfma2(a2, b, c); a3 = __hfma2(a3, b, c);
    a4 = __hfma2(a4, b, c); a5 = __hfma2(a5, b, c); a6 = __hfma2(a6, b, c); a7 = __hfma2(a7, b, c);
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = __hadd2(__hadd2(__hadd2(a0, a1), __hadd2(a2, a3)), __hadd2(__hadd2(a4, a5), __hadd2(a6, a7)));
}

// ---------------------------------------------------------------- bandwidth
__global__ void copy4(const float4 *__restrict__ a, float4 *__restrict__ b, long n) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) b[i] = a[i];
}
__global__ void scale_h(const half2 *__restrict__ a, half2 *__restrict__ b, long n) {
  const half2 s = __float2half2_rn(1.0001f);
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) b[i] = __hmul2(a[i], s);
}
__global__ void read_sum(const float4 *__restrict__ a, long n, float *out) {
  float s = 0;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) { float4 v = a[i]; s += v.x + v.y + v.z + v.w; }
  if (s == 12345.f) out[0] = s;
}

// ---------------------------------------------------------------- tiled GEMM: C[M][N] = A[M][K] * B[K][N]
// IMPL=0: B is a plain [K][N] matrix.  IMPL=1: implicit conv: k = tap*Cin + ci, B[k][n] = X[ci][n + tap*dil]
// (X already shifted by -pad; ldx = row stride).  T = float or __half storage; math always fp32.
template <int BM, int BN, int BK, int TM, int TN, typename T, int IMPL>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
gemm(int M, int N, int K, const T *__restrict__ A, const T *__restrict__ B, int ldb, int Cin, int dil, float *__restrict__ C, int ldc) {
  constexpr int NT = (BM / TM) * (BN / TN);
  __shared__ float As[BK][BM + 4];
  __shared__ float Bs[BK][BN + 4];
  const int tid = threadIdx.x, tx = tid % (BN / TN), ty = tid / (BN / TN);
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int l = tid; l < BM * BK; l += NT) {     // A tile, coalesced along k
      int mm = l / BK, kk = l % BK;
      int gm = m0 + mm, gk = k0 + kk;
      As[kk][mm] = (gm < M && gk < K) ? (float)A[(long)gm * K + gk] : 0.f;
    }
#pragma unroll
    for (int l = tid; l < BK * BN; l += NT) {     // B tile, coalesced along n
      int kk = l / BN, nn = l % BN;
      int gk = k0 + kk, gn = n0 + nn;
      float v = 0.f;
      if (gk < K && gn < N) {
        if (IMPL == 0) v = (float)B[(long)gk * ldb + gn];
        else { int tap = gk / Cin, ci = gk - tap * Cin; v = (float)B[(long)ci * ldb + gn + tap * dil]; }
      }
      Bs[kk][nn] = v;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; kk++) {
      float a[TM], b[TN];
#pragma unroll
      for (int i = 0; i < TM; i++) a[i] = As[kk][ty + i * (BM / TM)];
#pragma unroll
      for (int j = 0; j < TN; j++) b[j] = Bs[kk][tx + j * (BN / TN)];
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) {
      int gm = m0 + ty + i * (BM / TM), gn = n0 + tx + j * (BN / TN);
      if (gm < M && gn < N) C[(long)gm * ldc + gn] = acc[i][j];
    }
}

// half2 math variant (fp16 storage, half2 FMA, fp16 accumulate) to measure what half2 buys and costs.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN) / 2)
gemm_h2(int M, int N, int K, const __half *__restrict__ A, const __half *__restrict__ B, int ldb, float *__restrict__ C, int ldc) {
  constexpr int NT = (BM / TM) * (BN / TN) / 2;   // each thread: TM x (2*TN) via half2 pairs along n
  constexpr int BN2 = BN / 2;
  __shared__ __half As[BK][BM + 8];
  __shared__ half2 Bs[BK][BN2 + 4];
  const int tid = threadIdx.x, tx = tid % (BN2 / TN), ty = tid / (BN2 / TN);
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  half2 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = __float2half2_rn(0.f);
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int l = tid; l < BM * BK; l += NT) {
      int mm = l / BK, kk = l % BK, gm = m0 + mm, gk = k0 + kk;
      As[kk][mm] = (gm < M && gk < K) ? A[(long)gm * K + gk] : __float2half(0.f);
    }
    for (int l = tid; l < BK * BN2; l += NT) {
      int kk = l / BN2, nn = l % BN2, gk = k0 + kk, gn = n0 + 2 * nn;
      half2 v = __float2half2_rn(0.f);
      if (gk < K && gn + 1 < N) v = *(const half2 *)&B[(long)gk * ldb + gn];
      Bs[kk][nn] = v;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; kk++) {
      half2 a[TM], b[TN];
#pragma unroll
      for (int i = 0; i < TM; i++) a[i] = __half2half2(As[kk][ty + i * (BM / TM)]);
#pragma unroll
      for (int j = 0; j < TN; j++) b[j] = Bs[kk][tx + j * (BN2 / TN)];
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = __hfma2(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) {
      int gm = m0 + ty + i * (BM / TM), gn = n0 + 2 * (tx + j * (BN2 / TN));
      if (gm < M && gn + 1 < N) { float2 f = __half22float2(acc[i][j]); C[(long)gm * ldc + gn] = f.x; C[(long)gm * ldc + gn + 1] = f.y; }
    }
}

template <typename T> __global__ void fill(T *p, long n, unsigned seed) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)i * 2654435761u + seed; h ^= h >> 13; h *= 0x5bd1e995; h ^= h >> 15;
    p[i] = (T)(((h & 0xffff) / 65536.0f - 0.5f) * 0.5f);
  }
}

struct Shape { const char *what; int M, N, K, Cin, dil, impl; };

template <int BM, int BN, int BK, int TM, int TN, typename T, int IMPL>
static double run_gemm(const Shape &s, T *A, T *B, float *C, int reps) {
  dim3 grid((s.N + BN - 1) / BN, (s.M + BM - 1) / BM), blk((BM / TM) * (BN / TN));
  int ldb = IMPL ? s.N + 64 : s.N;
  gemm<BM, BN, BK, TM, TN, T, IMPL><<<grid, blk>>>(s.M, s.N, s.K, A, B, ldb, s.Cin, s.dil, C, s.N);
  CK(cudaGetLastError());
  Timer t; t.start();
  for (int r = 0; r < reps; r++) gemm<BM, BN, BK, TM, TN, T, IMPL><<<grid, blk>>>(s.M, s.N, s.K, A, B, ldb, s.Cin, s.dil, C, s.N);
  float ms = t.stop();
  return 2.0 * s.M * s.N * (double)s.K * reps / (ms * 1e-3) / 1e9;
}

__global__ void empty_k() {}
__global__ void tiny_k(float *p) { p[threadIdx.x] += 1.f; }

int main() {
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  printf("# roofline: %s cc %d.%d, %d SM, clock %d MHz (max), L2 %d KB, CUDA runtime ", pr.name, pr.major, pr.minor,
         pr.multiProcessorCount, pr.clockRate / 1000, pr.l2CacheSize / 1024);
  int rv; cudaRuntimeGetVersion(&rv); printf("%d\n", rv);
  warm(1.5);

  // ---- peak FMA
  {
    float *o; CK(cudaMalloc(&o, 1 << 22));
    int blocks = 64, thr = 256, it = 4096;
    fma_f32<<<blocks, thr>>>(o, it); CK(cudaDeviceSynchronize());
    Timer t; t.start(); for (int r = 0; r < 10; r++) fma_f32<<<blocks, thr>>>(o, it); float ms = t.stop();
    double gf = 2.0 * 8 * it * (double)blocks * thr * 10 / (ms * 1e-3) / 1e9;
    fma_h2<<<blocks, thr>>>((half2 *)o, it); CK(cudaDeviceSynchronize());
    t.start(); for (int r = 0; r < 10; r++) fma_h2<<<blocks, thr>>>((half2 *)o, it); float ms2 = t.stop();
    double gh = 2.0 * 2 * 8 * it * (double)blocks * thr * 10 / (ms2 * 1e-3) / 1e9;
    printf("\n## peak FMA\n| fp32 FMA | half2 HFMA2 |\n|---|---|\n| %.1f GFLOP/s | %.1f GFLOP/s |\n", gf, gh);
    cudaFree(o);
  }
  // ---- bandwidth
  {
    long n = 16 << 20;  // 64 MB floats
    float *a, *b; CK(cudaMalloc(&a, n * 4)); CK(cudaMalloc(&b, n * 4));
    fill<<<256, 256>>>(a, n, 1); CK(cudaDeviceSynchronize());
    copy4<<<512, 256>>>((float4 *)a, (float4 *)b, n / 4); CK(cudaDeviceSynchronize());
    Timer t; t.start(); for (int r = 0; r < 10; r++) copy4<<<512, 256>>>((float4 *)a, (float4 *)b, n / 4); float ms = t.stop();
    double bw = 2.0 * n * 4 * 10 / (ms * 1e-3) / 1e9;
    t.start(); for (int r = 0; r < 10; r++) read_sum<<<512, 256>>>((float4 *)a, n / 4, b); float msr = t.stop();
    double bwr = 1.0 * n * 4 * 10 / (msr * 1e-3) / 1e9;
    t.start(); for (int r = 0; r < 10; r++) scale_h<<<512, 256>>>((half2 *)a, (half2 *)b, n / 2); float msh = t.stop();
    double bwh = 2.0 * n * 2 * 10 / (msh * 1e-3) / 1e9;   /* n/2 half2 = n halves read + written */
    printf("\n## DRAM bandwidth (64 MB buffers)\n| copy fp32 (r+w) | read-only fp32 | half2 scale (r+w) | elements/s fp16 vs fp32 |\n|---|---|---|---|\n"
           "| %.2f GB/s | %.2f GB/s | %.2f GB/s | %.2fx |\n", bw, bwr, bwh, (bwh / 2) / (bw / 4));
    printf("(\"elements/s\" = values moved per second; fp16 halves the bytes per value)\n");
    cudaFree(a); cudaFree(b);
  }
  // ---- GEMMs
  {
    Shape S[] = {
      {"square 256^3", 256, 256, 256, 1, 1, 0},
      {"square 512^3", 512, 512, 512, 1, 1, 0},
      {"512x512xK=4096", 512, 512, 4096, 1, 1, 0},
      {"enc linear 192x192 (N=81)", 192, 81, 192, 1, 1, 0},
      {"enc FFN 768x192xK3 conv (N=81)", 768, 81, 192 * 3, 192, 1, 1},
      {"enc FFN x8 streams (N=648)", 768, 648, 192 * 3, 192, 1, 1},
      {"dec conv_pre 512x192xK7 (N=207)", 512, 207, 192 * 7, 192, 1, 1},
      {"dec ups.0 4 phases 1024x(512x4) (N=207)", 1024, 207, 512 * 4, 512, 1, 1},
      {"dec stage0 resblock 256x256xK11 d5 (N=828)", 256, 828, 256 * 11, 256, 5, 1},
      {"dec stage0 resblock x8 streams (N=6624)", 256, 6624, 256 * 11, 256, 5, 1},
      {"dec ups.1 4 phases 512x(256x4) (N=828)", 512, 828, 256 * 4, 256, 1, 1},
      {"dec stage1 resblock 128x128xK11 d3 (N=3312)", 128, 3312, 128 * 11, 128, 3, 1},
      {"dec stage1 resblock x8 streams (N=26496)", 128, 26496, 128 * 11, 128, 3, 1},
      {"dec stage1 resblock 128x128xK3 (N=3312)", 128, 3312, 128 * 3, 128, 1, 1},
    };
    int ns = sizeof S / sizeof S[0];
    size_t maxA = 0, maxB = 0, maxC = 0;
    for (int i = 0; i < ns; i++) {
      maxA = std::max(maxA, (size_t)S[i].M * S[i].K);
      size_t b = S[i].impl ? (size_t)S[i].Cin * (S[i].N + 64) + 64 : (size_t)S[i].K * S[i].N;
      maxB = std::max(maxB, b);
      maxC = std::max(maxC, (size_t)S[i].M * S[i].N);
    }
    float *A, *B, *C; __half *Ah, *Bh;
    CK(cudaMalloc(&A, maxA * 4)); CK(cudaMalloc(&B, maxB * 4)); CK(cudaMalloc(&C, maxC * 4));
    CK(cudaMalloc(&Ah, maxA * 2)); CK(cudaMalloc(&Bh, maxB * 2));
    fill<<<256, 256>>>(A, maxA, 3); fill<<<256, 256>>>(B, maxB, 5); fill<<<256, 256>>>(Ah, maxA, 3); fill<<<256, 256>>>(Bh, maxB, 5);
    CK(cudaDeviceSynchronize());
    printf("\n## tiled GEMM GFLOP/s (fp32 math; A = weights [M][K], B = activations; implicit = conv taps read in-kernel)\n");
    printf("| shape | 64x64 tile, 4x4/thr fp32 | 128x64, 8x4/thr fp32 | 128x128, 8x8/thr fp32 | 128x64 fp16 storage (fp32 math) | half2 math, fp16 accumulate (plain GEMM only) |\n|---|---|---|---|---|---|\n");
    for (int i = 0; i < ns; i++) {
      const Shape &s = S[i];
      double fl = 2.0 * s.M * s.N * s.K;
      int reps = fl > 1e10 ? 3 : fl > 1e9 ? 10 : 50;
      double g1, g2, g3, g4, g5 = -1;
      if (s.impl) {
        g1 = run_gemm<64, 64, 8, 4, 4, float, 1>(s, A, B, C, reps);
        g2 = run_gemm<128, 64, 8, 8, 4, float, 1>(s, A, B, C, reps);
        g3 = run_gemm<128, 128, 8, 8, 8, float, 1>(s, A, B, C, reps);
        g4 = run_gemm<128, 64, 8, 8, 4, __half, 1>(s, Ah, Bh, C, reps);
      } else {
        g1 = run_gemm<64, 64, 8, 4, 4, float, 0>(s, A, B, C, reps);
        if (s.N % 2) { /* half2 B loads need an even row stride */
          g2 = run_gemm<128, 64, 8, 8, 4, float, 0>(s, A, B, C, reps);
          g3 = run_gemm<128, 128, 8, 8, 8, float, 0>(s, A, B, C, reps);
          g4 = run_gemm<128, 64, 8, 8, 4, __half, 0>(s, Ah, Bh, C, reps);
          printf("| %s | %.1f | %.1f | %.1f | %.1f | n/a (odd N) |\n", s.what, g1, g2, g3, g4);
          continue;
        }
        g2 = run_gemm<128, 64, 8, 8, 4, float, 0>(s, A, B, C, reps);
        g3 = run_gemm<128, 128, 8, 8, 8, float, 0>(s, A, B, C, reps);
        g4 = run_gemm<128, 64, 8, 8, 4, __half, 0>(s, Ah, Bh, C, reps);
        dim3 grid((s.N + 127) / 128, (s.M + 127) / 128), blk(128);
        gemm_h2<128, 128, 8, 8, 8><<<grid, blk>>>(s.M, s.N, s.K, Ah, Bh, s.N, C, s.N);
        CK(cudaGetLastError());
        Timer t; t.start();
        for (int r = 0; r < reps; r++) gemm_h2<128, 128, 8, 8, 8><<<grid, blk>>>(s.M, s.N, s.K, Ah, Bh, s.N, C, s.N);
        g5 = fl * reps / (t.stop() * 1e-3) / 1e9;
      }
      if (g5 >= 0) printf("| %s | %.1f | %.1f | %.1f | %.1f | %.1f |\n", s.what, g1, g2, g3, g4, g5);
      else printf("| %s | %.1f | %.1f | %.1f | %.1f | n/a |\n", s.what, g1, g2, g3, g4);
    }
    // accuracy of fp16 accumulate vs fp32 on the K=4096 GEMM
    {
      const Shape &s = S[2];
      float *C2; CK(cudaMalloc(&C2, (size_t)s.M * s.N * 4));
      dim3 g1((s.N + 63) / 64, (s.M + 127) / 128), b1(256);
      gemm<128, 64, 8, 8, 4, __half, 0><<<g1, b1>>>(s.M, s.N, s.K, Ah, Bh, s.N, 1, 1, C, s.N);
      dim3 g2((s.N + 127) / 128, (s.M + 127) / 128), b2(128);
      gemm_h2<128, 128, 8, 8, 8><<<g2, b2>>>(s.M, s.N, s.K, Ah, Bh, s.N, C2, s.N);
      CK(cudaDeviceSynchronize());
      std::vector<float> h1((size_t)s.M * s.N), h2(h1.size());
      cudaMemcpy(h1.data(), C, h1.size() * 4, cudaMemcpyDeviceToHost);
      cudaMemcpy(h2.data(), C2, h2.size() * 4, cudaMemcpyDeviceToHost);
      double mx = 0, ma = 0;
      for (size_t i = 0; i < h1.size(); i++) { mx = fmax(mx, fabs(h1[i] - h2[i])); ma = fmax(ma, fabs(h1[i])); }
      printf("\nhalf2/fp16-accumulate error on 512x512xK4096 (same fp16 inputs, vs fp32 accumulate): max abs %.3e, rel %.3e\n", mx, mx / ma);
      cudaFree(C2);
    }
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(Ah); cudaFree(Bh);
  }
  // ---- launch overhead + CUDA graphs
  {
    float *p; CK(cudaMalloc(&p, 4096)); CK(cudaMemset(p, 0, 4096));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    for (int i = 0; i < 100; i++) tiny_k<<<1, 32, 0, st>>>(p);
    CK(cudaStreamSynchronize(st));
    printf("\n## launch overhead and CUDA Graphs (tiny 1-block kernel)\n");
    printf("| N kernels | individual launches: host issue us/kernel | individual: end-to-end us/kernel | graph capture ms | graph instantiate ms | graph replay end-to-end us/kernel | replay speedup |\n|---|---|---|---|---|---|---|\n");
    int Ns[] = {10, 100, 500};
    for (int k = 0; k < 3; k++) {
      int N = Ns[k], reps = 20;
      double h0 = now();
      for (int r = 0; r < reps; r++) for (int i = 0; i < N; i++) tiny_k<<<1, 32, 0, st>>>(p);
      double h1 = now();
      CK(cudaStreamSynchronize(st));
      double h2 = now();
      double issue = (h1 - h0) / (reps * N) * 1e6, e2e = (h2 - h0) / (reps * N) * 1e6;
      cudaGraph_t g; cudaGraphExec_t ge;
      double c0 = now();
      CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
      for (int i = 0; i < N; i++) tiny_k<<<1, 32, 0, st>>>(p);
      CK(cudaStreamEndCapture(st, &g));
      double c1 = now();
      CK(cudaGraphInstantiate(&ge, g, 0, 0, 0));
      double c2 = now();
      CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
      double g0 = now();
      for (int r = 0; r < reps; r++) CK(cudaGraphLaunch(ge, st));
      CK(cudaStreamSynchronize(st));
      double g1 = now();
      double rep = (g1 - g0) / (reps * N) * 1e6;
      printf("| %d | %.1f | %.1f | %.2f | %.2f | %.1f | %.2fx |\n", N, issue, e2e, (c1 - c0) * 1e3, (c2 - c1) * 1e3, rep, e2e / rep);
      cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    }
    // empty-kernel GPU-side cost via events around 1000 back-to-back launches
    Timer t; t.start(st); for (int i = 0; i < 1000; i++) empty_k<<<1, 32, 0, st>>>(); float ms = t.stop(st);
    printf("\nempty kernel, 1000 back-to-back in one stream (event-timed): %.1f us/kernel\n", ms);
    cudaFree(p);
    // cooperative launch
    int coop = 0; cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0);
    printf("cudaDevAttrCooperativeLaunch = %d -> grid.sync() persistent kernels %s on this device/driver\n", coop,
           coop ? "available" : "NOT available (cudaLaunchCooperativeKernel would fail); use graphs / fused kernels instead");
  }
  // ---- zero-copy, memcpy, malloc
  {
    size_t sz = 4 << 20;
    float *hp, *dp, *hm, *dmap; void *pg = malloc(sz);
    double t0 = now(); CK(cudaHostAlloc(&hp, sz, cudaHostAllocMapped)); double t1 = now();
    CK(cudaMalloc(&dp, sz));
    CK(cudaHostGetDevicePointer(&dmap, hp, 0));
    hm = hp; memset(hm, 0, sz);
    Timer t; float ms;
    cudaMemcpy(dp, hp, sz, cudaMemcpyHostToDevice);
    t.start(); for (int r = 0; r < 10; r++) cudaMemcpy(dp, hp, sz, cudaMemcpyHostToDevice); ms = t.stop();
    double h2d_p = sz * 10 / (ms * 1e-3) / 1e9;
    t.start(); for (int r = 0; r < 10; r++) cudaMemcpy(pg, dp, sz, cudaMemcpyDeviceToHost); ms = t.stop();
    double d2h_pg = sz * 10 / (ms * 1e-3) / 1e9;
    t.start(); for (int r = 0; r < 10; r++) cudaMemcpy(hp, dp, sz, cudaMemcpyDeviceToHost); ms = t.stop();
    double d2h_p = sz * 10 / (ms * 1e-3) / 1e9;
    read_sum<<<128, 256>>>((float4 *)dmap, sz / 16, dp); CK(cudaDeviceSynchronize());
    t.start(); for (int r = 0; r < 10; r++) read_sum<<<128, 256>>>((float4 *)dmap, sz / 16, dp); ms = t.stop();
    double zc = sz * 10 / (ms * 1e-3) / 1e9;
    t.start(); for (int r = 0; r < 10; r++) read_sum<<<128, 256>>>((float4 *)dp, sz / 16, dp + 1); ms = t.stop();
    double dv = sz * 10 / (ms * 1e-3) / 1e9;
    // small transfer latency (one 40 ms chunk of 16 kHz audio = 640 floats)
    t.start(); for (int r = 0; r < 100; r++) cudaMemcpy(hp, dp, 640 * 4, cudaMemcpyDeviceToHost); ms = t.stop();
    printf("\n## host<->device on unified Tegra DRAM (4 MB)\n| cudaHostAlloc(mapped) 4 MB | H2D pinned | D2H pageable | D2H pinned | kernel read of zero-copy mapped | kernel read of cudaMalloc | D2H 2.5 KB (one 40 ms chunk) |\n|---|---|---|---|---|---|---|\n"
           "| %.2f ms | %.2f GB/s | %.2f GB/s | %.2f GB/s | %.2f GB/s | %.2f GB/s | %.1f us |\n", (t1 - t0) * 1e3, h2d_p, d2h_pg, d2h_p, zc, dv, ms * 10);
    cudaFreeHost(hp); cudaFree(dp); free(pg);
    printf("\n## cudaMalloc / cudaFree cost\n| size | cudaMalloc us | cudaFree us |\n|---|---|---|\n");
    size_t szs[] = {4096, 1 << 20, 16 << 20, 64 << 20};
    for (int i = 0; i < 4; i++) {
      void *q; double a = now(); CK(cudaMalloc(&q, szs[i])); double b = now(); CK(cudaFree(q)); double c = now();
      printf("| %zu KB | %.0f | %.0f |\n", szs[i] >> 10, (b - a) * 1e6, (c - b) * 1e6);
    }
  }
  // ---- 8 streams of small kernels vs 1 batched kernel (conv-as-GEMM, stage0 resblock K3, 40 ms chunk each)
  {
    Shape s = {"", 256, 160, 256 * 3, 256, 1, 1};  // one stream, 160 columns = a 40 ms-ish chunk at stage 0
    int NS = 8;
    long ldb = s.N * NS + 64;
    float *A, *B, *C; CK(cudaMalloc(&A, (size_t)s.M * s.K * 4)); CK(cudaMalloc(&B, (size_t)s.Cin * ldb * 4 + 64)); CK(cudaMalloc(&C, (size_t)s.M * s.N * NS * 4));
    fill<<<256, 256>>>(A, (long)s.M * s.K, 1); fill<<<256, 256>>>(B, (long)s.Cin * ldb, 2); CK(cudaDeviceSynchronize());
    cudaStream_t sts[8]; for (int i = 0; i < NS; i++) CK(cudaStreamCreate(&sts[i]));
    dim3 blk(256), g1((s.N + 63) / 64, (s.M + 127) / 128), g8((s.N * NS + 63) / 64, (s.M + 127) / 128);
    int reps = 50;
    for (int i = 0; i < NS; i++) gemm<128, 64, 8, 8, 4, float, 1><<<g1, blk, 0, sts[i]>>>(s.M, s.N, s.K, A, B + i * s.N, (int)ldb, s.Cin, 1, C + i * s.N, s.N * NS);
    CK(cudaDeviceSynchronize());
    double t0 = now();
    for (int r = 0; r < reps; r++) for (int i = 0; i < NS; i++)
      gemm<128, 64, 8, 8, 4, float, 1><<<g1, blk, 0, sts[i]>>>(s.M, s.N, s.K, A, B + i * s.N, (int)ldb, s.Cin, 1, C + i * s.N, s.N * NS);
    CK(cudaDeviceSynchronize());
    double t1 = now();
    for (int r = 0; r < reps; r++) gemm<128, 64, 8, 8, 4, float, 1><<<g8, blk>>>(s.M, s.N * NS, s.K, A, B, (int)ldb, s.Cin, 1, C, s.N * NS);
    CK(cudaDeviceSynchronize());
    double t2 = now();
    // same, but each stream gets a tiny kernel (encoder-like 192x192, N=20)
    Shape e = {"", 192, 20, 192 * 3, 192, 1, 1};
    dim3 ge1((e.N + 63) / 64, (e.M + 127) / 128), ge8((e.N * NS + 63) / 64, (e.M + 127) / 128);
    double t3 = now();
    for (int r = 0; r < reps; r++) for (int i = 0; i < NS; i++)
      gemm<128, 64, 8, 8, 4, float, 1><<<ge1, blk, 0, sts[i]>>>(e.M, e.N, e.K, A, B + i * e.N, (int)ldb, e.Cin, 1, C + i * e.N, e.N * NS);
    CK(cudaDeviceSynchronize());
    double t4 = now();
    for (int r = 0; r < reps; r++) gemm<128, 64, 8, 8, 4, float, 1><<<ge8, blk>>>(e.M, e.N * NS, e.K, A, B, (int)ldb, e.Cin, 1, C, e.N * NS);
    CK(cudaDeviceSynchronize());
    double t5 = now();
    double f1 = 2.0 * s.M * s.N * s.K * NS * reps, f2 = 2.0 * e.M * e.N * e.K * NS * reps;
    printf("\n## 8 streams: 8 per-stream kernels on 8 cudaStreams vs 1 kernel batched along N (implicit-conv GEMM)\n"
           "| work per stream | 8 streams x 1 kernel: ms per round | GFLOP/s | 1 batched kernel: ms | GFLOP/s | batching speedup |\n|---|---|---|---|---|---|\n"
           "| dec stage0 conv 256x256xK3, N=160 | %.3f | %.1f | %.3f | %.1f | %.2fx |\n"
           "| enc conv 192x192xK3, N=20 | %.3f | %.1f | %.3f | %.1f | %.2fx |\n",
           (t1 - t0) / reps * 1e3, f1 / (t1 - t0) / 1e9, (t2 - t1) / reps * 1e3, f1 / (t2 - t1) / 1e9, (t1 - t0) / (t2 - t1),
           (t4 - t3) / reps * 1e3, f2 / (t4 - t3) / 1e9, (t5 - t4) / reps * 1e3, f2 / (t5 - t4) / 1e9, (t4 - t3) / (t5 - t4));
  }
  return 0;
}
