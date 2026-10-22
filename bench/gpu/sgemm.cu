// sgemm.cu - fp32 tiled GEMM + half2 GEMM roofline for Tegra X1 (sm_53, CUDA 10.2).
// Build: /usr/local/cuda/bin/nvcc -O3 -arch=sm_53 -o sgemm sgemm.cu
// Run:   ./sgemm            (prints all tables)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA err %d %s @line %d\n",(int)e,cudaGetErrorString(e),__LINE__); exit(1);} }while(0)

__global__ void sgemm_tile(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                           int M, int N, int K) {
  // 256 threads (16x16), 64x64 output tile, 4x4 per thread, k-chunk 16.
  __shared__ float As[16][64], Bs[16][64];
  const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;
  const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  float acc[4][4];
  #pragma unroll
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
  for (int t = 0; t < K; t += 16) {
    #pragma unroll
    for (int s = 0; s < 4; s++) {           // fill As[16][64] and Bs[16][64] with 4 loads each
      int idx = tid + s * 256;
      int k = idx >> 6, c = idx & 63;
      int ar = m0 + c, bc = n0 + c, bk = t + k;
      As[k][c] = (ar < M && bk < K) ? A[(size_t)ar * K + bk] : 0.f;
      Bs[k][c] = (bk < K && bc < N) ? B[(size_t)bk * N + bc] : 0.f;
    }
    __syncthreads();
    const int am = ty * 4, bn = tx * 4;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
      float a[4], b[4];
      #pragma unroll
      for (int i = 0; i < 4; i++) a[i] = As[k][am + i];
      #pragma unroll
      for (int j = 0; j < 4; j++) b[j] = Bs[k][bn + j];
      #pragma unroll
      for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < 4; i++)
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      int r = m0 + ty * 4 + i, c = n0 + tx * 4 + j;
      if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j];
    }
}

// fp16 STORAGE + fp32 accumulate GEMM: halves the bytes a conv-as-GEMM must move (the real win on a
// 5.5-8 GB/s device), while the math stays fp32. This is the realistic v2.1 decoder kernel shape.
__global__ void hgemm32_tile(const __half* __restrict__ A, const __half* __restrict__ B, float* __restrict__ C,
                             int M, int N, int K) {
  __shared__ __half As[16][64], Bs[16][64];
  const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;
  const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  float acc[4][4];
  #pragma unroll
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
  for (int t = 0; t < K; t += 16) {
    #pragma unroll
    for (int s = 0; s < 4; s++) {
      int idx = tid + s * 256;
      int k = idx >> 6, c = idx & 63;
      int ar = m0 + c, bc = n0 + c, bk = t + k;
      As[k][c] = (ar < M && bk < K) ? A[(size_t)ar * K + bk] : __float2half(0.f);
      Bs[k][c] = (bk < K && bc < N) ? B[(size_t)bk * N + bc] : __float2half(0.f);
    }
    __syncthreads();
    const int am = ty * 4, bn = tx * 4;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
      float a[4], b[4];
      #pragma unroll
      for (int i = 0; i < 4; i++) a[i] = __half2float(As[k][am + i]);
      #pragma unroll
      for (int j = 0; j < 4; j++) b[j] = __half2float(Bs[k][bn + j]);
      #pragma unroll
      for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < 4; i++)
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      int r = m0 + ty * 4 + i, c = n0 + tx * 4 + j;
      if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j];
    }
}

// Pure-issue peak: register-resident FMA chains, fp32 (float4) vs half2. Bounds what any kernel of
// this device can do; the GEMM tiles above show what a real kernel achieves.
__global__ void peak_fp32(float* out, int iters) {
  float4 a = {1.f, 2.f, 3.f, 4.f}, b = {0.5f, 0.5f, 0.5f, 0.5f}, c = {1.1f, 1.2f, 1.3f, 1.4f};
  float4 d = {2.f, 3.f, 4.f, 5.f}, e = {0.25f, 0.25f, 0.25f, 0.25f}, f = {1.5f, 1.6f, 1.7f, 1.8f};
  for (int i = 0; i < iters; i++) {
    a.x = fmaf(a.x, b.x, c.x); a.y = fmaf(a.y, b.y, c.y); a.z = fmaf(a.z, b.z, c.z); a.w = fmaf(a.w, b.w, c.w);
    d.x = fmaf(d.x, e.x, f.x); d.y = fmaf(d.y, e.y, f.y); d.z = fmaf(d.z, e.z, f.z); d.w = fmaf(d.w, e.w, f.w);
  }
  out[threadIdx.x + blockIdx.x * blockDim.x] = a.x + a.y + a.z + a.w + d.x + d.y + d.z + d.w;
}
__global__ void peak_half2(float* out, int iters) {
  __half2 a = __floats2half2_rn(1.f, 2.f), b = __floats2half2_rn(0.5f, 0.5f), c = __floats2half2_rn(1.1f, 1.2f);
  __half2 d = __floats2half2_rn(2.f, 3.f), e = __floats2half2_rn(0.25f, 0.25f), f = __floats2half2_rn(1.5f, 1.6f);
  __half2 g = __floats2half2_rn(1.f, 1.f), h = __floats2half2_rn(0.1f, 0.1f), k = __floats2half2_rn(0.3f, 0.4f);
  for (int i = 0; i < iters; i++) {
    a = __hfma2(a, b, c); d = __hfma2(d, e, f); g = __hfma2(g, h, k); a = __hfma2(a, e, d);
  }
  float2 s = __half22float2(__hadd2(__hadd2(a, d), __hadd2(g, a)));
  out[threadIdx.x + blockIdx.x * blockDim.x] = s.x + s.y;
}

static float bench(void (*fn)(int, int, int), int M, int N, int K, int reps) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  fn(M, N, K);  // warmup
  CK(cudaEventRecord(a));
  for (int i = 0; i < reps; i++) fn(M, N, K);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b));
  return ms / reps;
}

static float *A, *B, *C;
static __half2 *A2, *B2;
static int gM, gN, gK;

static void run_s(int M, int N, int K) {
  dim3 blk(16, 16), grd((N + 63) / 64, (M + 63) / 64);
  sgemm_tile<<<grd, blk>>>(A, B, C, M, N, K);
  CK(cudaGetLastError());
}
static void run_h(int M, int N, int K) {
  dim3 blk(16, 16), grd((N + 63) / 64, (M + 63) / 64);
  hgemm32_tile<<<grd, blk>>>((const __half*)A2, (const __half*)B2, C, M, N, K);
  CK(cudaGetLastError());
}

int main() {
  CK(cudaSetDevice(0));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device=%s cc=%d.%d SMs=%d clock=%dMHz mem=%zuMB L2=%dKB coop=%d\n",
         p.name, p.major, p.minor, p.multiProcessorCount, p.clockRate / 1000, p.totalGlobalMem >> 20,
         p.l2CacheSize >> 10, p.cooperativeLaunch);
  // correctness self-check (fp32 tile and fp16-storage tile) on 128^3
  {
    int M = 128, N = 128, K = 128;
    size_t na = (size_t)M * K, nb = (size_t)K * N, nc = (size_t)M * N;
    float *hA = (float*)malloc(na * 4), *hB = (float*)malloc(nb * 4), *hC = (float*)malloc(nc * 4);
    float *dA, *dB, *dC; __half *dH, *dHb;
    srand(7);
    for (size_t i = 0; i < na; i++) hA[i] = (rand() % 200 - 100) / 100.f;
    for (size_t i = 0; i < nb; i++) hB[i] = (rand() % 200 - 100) / 100.f;
    CK(cudaMalloc(&dA, na * 4)); CK(cudaMalloc(&dB, nb * 4)); CK(cudaMalloc(&dC, nc * 4)); CK(cudaMalloc(&dH, na * 2));
    CK(cudaMalloc(&dHb, nb * 2));
    CK(cudaMemcpy(dA, hA, na * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, hB, nb * 4, cudaMemcpyHostToDevice));
    A = dA; B = dB; C = dC;
    run_s(M, N, K);
    CK(cudaMemcpy(hC, dC, nc * 4, cudaMemcpyDeviceToHost));
    double mx = 0;
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
      double s = 0;
      for (int k = 0; k < K; k++) s += (double)hA[i * K + k] * hB[k * N + j];
      double d = fabs(s - hC[i * N + j]);
      if (d > mx) mx = d;
    }
    printf("selfcheck fp32 128^3 max|gpu-cpu| = %.3e\n", mx);
    // fp16-storage path: store A,B as half
    __half* hH = (__half*)malloc(na * 2);
    for (size_t i = 0; i < na; i++) hH[i] = __float2half(hA[i]);
    CK(cudaMemcpy(dH, hH, na * 2, cudaMemcpyHostToDevice));
    for (size_t i = 0; i < nb; i++) hH[i] = __float2half(hB[i]);
    CK(cudaMemcpy(dHb, hH, nb * 2, cudaMemcpyHostToDevice));
    A2 = (__half2*)dH; B2 = (__half2*)dHb;
    run_h(M, N, K);
    CK(cudaMemcpy(hC, dC, nc * 4, cudaMemcpyDeviceToHost));
    mx = 0;
    for (int i = 0; i < 64; i++) for (int j = 0; j < N; j++) {
      double s = 0;
      for (int k = 0; k < K; k++) s += (double)hA[i * K + k] * hB[k * N + j];
      double d = fabs(s - hC[i * N + j]);
      if (d > mx) mx = d;
    }
    printf("selfcheck fp16 128^3 (rows 0..63) max|gpu-cpu| = %.3e\n", mx);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dH); cudaFree(dHb); free(hA); free(hB); free(hC); free(hH);
  }
  int shapes[6][3] = {{256,256,256},{512,512,512},{1024,1024,1024},{512,4000,512},{256,2000,256},{192,2000,192}};
  for (int i = 0; i < 6; i++) {
    int M = shapes[i][0], N = shapes[i][1], K = shapes[i][2];
    size_t na = (size_t)M * K, nb = (size_t)K * N, nc = (size_t)M * N;
    CK(cudaMalloc(&A, na * 4)); CK(cudaMalloc(&B, nb * 4)); CK(cudaMalloc(&C, nc * 4));
    CK(cudaMalloc(&A2, na * 2)); CK(cudaMalloc(&B2, nb * 2));
    CK(cudaMemset(A, 0x3c, na * 4)); CK(cudaMemset(B, 0x3c, nb * 4));
    CK(cudaMemset(A2, 0x3c, na * 2)); CK(cudaMemset(B2, 0x3c, nb * 2));
    double fl = 2.0 * M * N * K;
    float ms = bench(run_s, M, N, K, 20);
    float mh = bench(run_h, M, N, K, 20);
    double gbs = (na * 4.0 + nb * 4.0 + nc * 4.0) / (ms / 1e3) / 1e9;
    double gbs2 = (na * 2.0 + nb * 2.0 + nc * 4.0) / (mh / 1e3) / 1e9;
    printf("M=%-5d N=%-5d K=%-5d fp32 %6.2f GF/s (%7.2f ms, %5.2f GB/s)  fp16store+fp32acc %6.2f GF/s (%7.2f ms, %5.2f GB/s)\n",
           M, N, K, fl / (ms / 1e3) / 1e9, ms, gbs, fl / (mh / 1e3) / 1e9, mh, gbs2);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(A2); cudaFree(B2);
  }
  // pure-issue peak
  float* o; int blocks = p.multiProcessorCount * 8, tpb = 256, iters = 200000;
  CK(cudaMalloc(&o, blocks * tpb * 4));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  peak_fp32<<<blocks, tpb>>>(o, iters); CK(cudaGetLastError());
  CK(cudaEventRecord(e0));
  for (int i = 0; i < 5; i++) peak_fp32<<<blocks, tpb>>>(o, iters);
  CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  float t32; CK(cudaEventElapsedTime(&t32, e0, e1)); t32 /= 5;
  peak_half2<<<blocks, tpb>>>(o, iters);
  CK(cudaEventRecord(e0));
  for (int i = 0; i < 5; i++) peak_half2<<<blocks, tpb>>>(o, iters);
  CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  float th2; CK(cudaEventElapsedTime(&th2, e0, e1)); th2 /= 5;
  double thr = (double)blocks * tpb;
  printf("peak FMA issue: fp32(float4x2 chains=8 lanes) %6.2f GFLOP/s | half2(4 hfma2 x2 lanes = 8) %6.2f GFLOP/s\n",
         thr * 8.0 * iters / (t32 / 1e3) / 1e9, thr * 8.0 * iters / (th2 / 1e3) / 1e9);
  cudaFree(o);
  return 0;
}
