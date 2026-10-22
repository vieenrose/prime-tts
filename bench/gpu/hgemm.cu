// Half2 GEMM feasibility on Jetson Nano (Tegra X1, sm_53, 1 SM, 921 MHz).
//
// Why this matters: v2.1 costs 11.9 GFLOP per audio-second (measured, tools/probe_shapes.py), so
// eight real-time streams need 95 GFLOP/s sustained. Measured fp32 tiled GEMM reaches 62 GFLOP/s
// and fp32 FMA issue peaks at 88 GFLOP/s, so fp32 math cannot serve 8 streams. sm_53 doubles
// throughput in half2 (measured __hfma2 issue peak 168 GFLOP/s), so half2 compute is the only
// route to 95+. This program measures the two kernels that decide it:
//   h_fma  : peak __hfma2 issue (register-resident ceiling)
//   hgemm  : tiled GEMM, fp16 storage, half2 multiply AND half2 accumulate  -> the 8-stream number
//   sgemm2 : tiled GEMM, fp16 storage, fp32 accumulate  -> what the accuracy-safe variant costs
// plus a batched shape (M = T x streams), which is how the runtime runs 8 conversations in one
// kernel launch. GFLOP/s = 2*M*N*K / time.
//
// half2 pairing convention (Maxwell, no tensor cores): pairs are taken along K.
//   A2[m][k/2]   = (A[m][k],   A[m][k+1])     A is [M][K] row-major, K even
//   B2[k/2][n]   = (B[k][n],   B[k+1][n])     B is stored [K/2][N] (pairs contiguous)
//   hfma2(a,b,acc) accumulates lane0 = A[m][2i]*B[2i][n], lane1 = A[m][2i+1]*B[2i+1][n],
//   both lanes are partial sums of the SAME output element; the epilogue adds the lanes.
//
// Build: /usr/local/cuda/bin/nvcc -O3 -arch=sm_53 -o hgemm hgemm.cu && ./hgemm
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CK(x)                                                                                  \
  do {                                                                                         \
    cudaError_t _e = (x);                                                                      \
    if (_e != cudaSuccess) {                                                                   \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(_e), __LINE__);                 \
      exit(1);                                                                                 \
    }                                                                                          \
  } while (0)

#define TM 64      // rows (M) per block
#define TN 64      // cols (N) per block
#define KT 32      // K scalars per stage
#define KP 16      // K pairs per stage
#define PAD 8

__global__ void h_fma(float *out, int iters) {
  half2 a = __float2half2_rn(1.0001f), b = __float2half2_rn(0.9999f), c = __float2half2_rn(0.5f);
  half2 d = __float2half2_rn(0.25f), e = __float2half2_rn(0.75f), f = __float2half2_rn(0.125f);
  half2 g = __float2half2_rn(0.6f), h = __float2half2_rn(0.3f);
  for (int i = 0; i < iters; i++) {
    a = __hfma2(a, b, c);
    d = __hfma2(d, e, f);
    g = __hfma2(g, h, a);
    b = __hfma2(b, c, d);
    c = __hfma2(c, d, e);
    e = __hfma2(e, f, g);
    f = __hfma2(f, g, h);
    h = __hfma2(h, a, b);
  }
  out[threadIdx.x + blockIdx.x * blockDim.x] =
      __low2float(a) + __low2float(d) + __low2float(g) + __low2float(h);
}

// fp16 storage, half2 multiply, half2 accumulate (4x4 per thread over 16x16 threads)
__global__ void hgemm(const half2 *__restrict__ A2, const half2 *__restrict__ B2, half *__restrict__ C,
                      int M, int N, int K) {
  __shared__ half2 As[KP][TM + PAD];
  __shared__ half2 Bs[KP][TN + PAD];
  const int m0 = blockIdx.x * TM, n0 = blockIdx.y * TN;
  const int mrow = m0 + threadIdx.x / 16 * 4, ncol = n0 + threadIdx.x % 16 * 4;
  const half2 zero = __float2half2_rn(0.0f);
  const int kh = K / 2;

  half2 acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) acc[i][j] = zero;

  for (int k0 = 0; k0 < kh; k0 += KP) {
    for (int idx = threadIdx.x; idx < KP * TM; idx += 256) {
      int p = idx / TM, m = idx % TM;
      As[p][m] = (k0 + p < kh && m0 + m < M) ? A2[(m0 + m) * kh + k0 + p] : zero;
    }
    for (int idx = threadIdx.x; idx < KP * TN; idx += 256) {
      int p = idx / TN, n = idx % TN;
      Bs[p][n] = (k0 + p < kh && n0 + n < N) ? B2[(k0 + p) * N + n0 + n] : zero;
    }
    __syncthreads();
#pragma unroll
    for (int p = 0; p < KP; p++) {
      half2 a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; i++) a[i] = As[p][threadIdx.x / 16 * 4 + i];
#pragma unroll
      for (int j = 0; j < 4; j++) b[j] = Bs[p][threadIdx.x % 16 * 4 + j];
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = __hfma2(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int r = mrow + i, c = ncol + j;
      if (r < M && c < N) {
        half s = __hadd(acc[i][j].x, acc[i][j].y);
        C[r * N + c] = s;
      }
    }
}

// fp16 storage, fp32 accumulate: the accuracy-safe variant (per-element fp32 math)
__global__ void sgemm2(const half *__restrict__ A, const half *__restrict__ B, float *__restrict__ C,
                       int M, int N, int K) {
  __shared__ float As[KT][TM + PAD];
  __shared__ float Bs[KT][TN + PAD];
  const int m0 = blockIdx.x * TM, n0 = blockIdx.y * TN;
  const int mrow = m0 + threadIdx.x / 16 * 4, ncol = n0 + threadIdx.x % 16 * 4;
  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += KT) {
    for (int idx = threadIdx.x; idx < KT * TM; idx += 256) {
      int k = idx / TM, m = idx % TM;
      As[k][m] = (k0 + k < K && m0 + m < M) ? __half2float(A[(m0 + m) * K + k0 + k]) : 0.0f;
    }
    for (int idx = threadIdx.x; idx < KT * TN; idx += 256) {
      int k = idx / TN, n = idx % TN;
      Bs[k][n] = (k0 + k < K && n0 + n < N) ? __half2float(B[k0 + k >= K ? 0 : ((k0 + k) / 2) * N + n0 + n]) : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < KT; k++) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; i++) a[i] = As[k][threadIdx.x / 16 * 4 + i];
#pragma unroll
      for (int j = 0; j < 4; j++) b[j] = Bs[k][threadIdx.x % 16 * 4 + j];
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++)
      if (mrow + i < M && ncol + j < N) C[(mrow + i) * N + ncol + j] = acc[i][j];
}

static double gflops(int M, int N, int K) { return 2.0 * (double)M * N * K; }

int main(void) {
  int dev;
  cudaDeviceProp p;
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&p, dev));
  printf("device=%s cc=%d.%d SMs=%d clock=%dMHz mem=%zuMB\n", p.name, p.major, p.minor,
         p.multiProcessorCount, p.clockRate, p.totalGlobalMem >> 20);

  {   // peak half2 issue
    int blocks = p.multiProcessorCount * 32, threads = 256, iters = 300000;
    float *out;
    CK(cudaMalloc(&out, (size_t)blocks * threads * sizeof(float)));
    h_fma<<<blocks, threads>>>(out, 1000);
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    h_fma<<<blocks, threads>>>(out, iters);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    printf("peak __hfma2 issue: %.2f GFLOP/s\n\n", (double)blocks * threads * iters * 8 * 2 / ms / 1e6);
    cudaFree(out);
  }

  // v2.1-relevant shapes: K = Cin*kernel of the decoder convs, N = Cout, M = T (x streams batched)
  int shapes[7][3] = {{81, 256, 1536}, {81, 512, 1024}, {324, 512, 1024}, {648, 512, 1024},
                      {1296, 512, 1024}, {1024, 1024, 1024}, {512, 4000, 512}};
  printf("%-22s %14s %14s   (8-stream need: 95 GFLOP/s)\n", "shape MxNxK", "half2 acc", "fp32 acc");
  for (int s = 0; s < 7; s++) {
    int M = shapes[s][0], N = shapes[s][1], K = shapes[s][2];
    M = (M / TM) ? M : TM;   // keep at least one tile
    size_t an = (size_t)M * K / 2, bn = (size_t)K / 2 * N, cn = (size_t)M * N;
    half2 *A2, *B2;
    half *Ch;
    float *Cf;
    CK(cudaMalloc(&A2, an * sizeof(half2)));
    CK(cudaMalloc(&B2, bn * sizeof(half2)));
    CK(cudaMalloc(&Ch, cn * sizeof(half)));
    CK(cudaMalloc(&Cf, cn * sizeof(float)));
    cudaMemset(A2, 0x3c, an * sizeof(half2));
    cudaMemset(B2, 0x3c, bn * sizeof(half2));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    dim3 grid((M + TM - 1) / TM, (N + TN - 1) / TN);
    int reps = (int)(4e9 / gflops(M, N, K)) + 3;

    hgemm<<<grid, 256>>>(A2, B2, Ch, M, N, K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    cudaEventRecord(e0);
    for (int i = 0; i < reps; i++) hgemm<<<grid, 256>>>(A2, B2, Ch, M, N, K);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms1;
    cudaEventElapsedTime(&ms1, e0, e1);
    ms1 /= reps;

    sgemm2<<<grid, 256>>>((const half *)A2, (const half *)B2, Cf, M, N, K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    cudaEventRecord(e0);
    for (int i = 0; i < reps; i++) sgemm2<<<grid, 256>>>((const half *)A2, (const half *)B2, Cf, M, N, K);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms2;
    cudaEventElapsedTime(&ms2, e0, e1);
    ms2 /= reps;

    printf("%dx%dx%d  ->  half2 %.2f GF/s, fp32-acc %.2f GF/s\n", M, N, K, gflops(M, N, K) / ms1 / 1e6,
           gflops(M, N, K) / ms2 / 1e6);
    cudaFree(A2);
    cudaFree(B2);
    cudaFree(Ch);
    cudaFree(Cf);
  }

  {   // correctness self-check of hgemm against an fp32-accumulate reference on fp16 inputs
    int M = 128, N = 128, K = 128;
    size_t an = (size_t)M * K / 2, bn = (size_t)K / 2 * N, cn = (size_t)M * N;
    half2 *A2, *B2;
    half *Ch;
    CK(cudaMalloc(&A2, an * sizeof(half2)));
    CK(cudaMalloc(&B2, bn * sizeof(half2)));
    CK(cudaMalloc(&Ch, cn * sizeof(half)));
    half2 *hA = (half2 *)malloc(an * sizeof(half2));
    half2 *hB = (half2 *)malloc(bn * sizeof(half2));
    srand(7);
    for (size_t i = 0; i < an; i++)
      hA[i] = __floats2half2_rn((rand() / (float)RAND_MAX - 0.5f) * 0.4f, (rand() / (float)RAND_MAX - 0.5f) * 0.4f);
    for (size_t i = 0; i < bn; i++)
      hB[i] = __floats2half2_rn((rand() / (float)RAND_MAX - 0.5f) * 0.4f, (rand() / (float)RAND_MAX - 0.5f) * 0.4f);
    CK(cudaMemcpy(A2, hA, an * sizeof(half2), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(B2, hB, bn * sizeof(half2), cudaMemcpyHostToDevice));
    dim3 grid(M / TM, N / TN);
    hgemm<<<grid, 256>>>(A2, B2, Ch, M, N, K);
    CK(cudaDeviceSynchronize());
    half *hC = (half *)malloc(cn * sizeof(half));
    CK(cudaMemcpy(hC, Ch, cn * sizeof(half), cudaMemcpyDeviceToHost));
    double maxe = 0, rms = 0;
    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++) {
        double r = 0;
        for (int k = 0; k < K / 2; k++) {
          r += (double)__low2float(hA[m * (K / 2) + k]) * __low2float(hB[k * N + n]);
          r += (double)__high2float(hA[m * (K / 2) + k]) * __high2float(hB[k * N + n]);
        }
        double d = fabs(r - __half2float(hC[m * N + n]));
        if (d > maxe) maxe = d;
        rms += d * d;
      }
    printf("\nselfcheck hgemm 128^3 (fp16 storage, half2 accumulate): max|ref-gpu| = %.3e, rms = %.3e\n", maxe,
           sqrt(rms / cn));
    free(hA);
    free(hB);
    free(hC);
  }
  return 0;
}
