/* ig8 v2 bench (standalone; NEVER linked into the runtime).
 *
 * Follows .auto/ig8_bench.cu, whose ablation showed the shipped ig8 tile is staging/barrier bound,
 * NOT LDS-bandwidth bound (removing per-chunk X staging +45%, W staging +49%, barriers +64%; and
 * HALVING the FMA-per-LDS ratio made it FASTER -- 74.3 vs 43.2 GFMA/s). This bench measures the
 * candidate fixes on the runtime's fast staging path (zero-padded scratch, one bound check):
 *   RK        chunk width in rank rows (16 = shipped; 32/8 = fewer/more barrier rounds)
 *   PIPE      1 = software pipeline: the next chunk's loads are issued before the current compute
 *             (register-triple-buffered staging), so the ~400-cycle global latency hides behind FMA.
 * Numerics are validated against a naive fp32 conv before any rate is trusted.
 *
 * Build on the Nano:
 *   /usr/local/cuda/bin/nvcc -arch=sm_53 -O3 .auto/ig8_v2_bench.cu -o /tmp/ig8v2 && /tmp/ig8v2
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

/* ---- v2 family: the runtime's FAST staging path (zero-padded scratch, one bound check per
   quad), template chunk width RK, optional software pipeline PIPE (the next chunk's global loads
   are issued before the current compute so their latency hides behind the FMAs). ---- */
template<int RK>
__device__ __forceinline__ void stage_row(const float *__restrict__ x, int64_t ldx, int cin, int K,
                                          int kc, int nblk, int pad, int64_t Tout,
                                          const float4 *__restrict__ w4, int r0, int row, int lq,
                                          int64_t tc0, int64_t tc1,
                                          float4 &w, float4 &q0, float4 &q1) {
  int rr = r0 + row;
  int lg = blockIdx.y * 16 + lq;
  w = (rr < kc && lg < nblk) ? w4[(int64_t)lg * kc + rr] : make_float4(0.f, 0.f, 0.f, 0.f);
  q0 = make_float4(0.f, 0.f, 0.f, 0.f);
  q1 = q0;
  if (rr < kc) {
    int k = rr / cin, ci = rr - k * cin;
    const float *r = x + (int64_t)ci * ldx + k;
    
    if (tc0 + 1 < Tout) q0.y = r[tc0 + 1];
    if (tc0 + 2 < Tout) q0.z = r[tc0 + 2];
    if (tc0 + 3 < Tout) q0.w = r[tc0 + 3];
    if (tc1 + 0 < Tout) q1.x = r[tc1 + 0];
    if (tc1 + 1 < Tout) q1.y = r[tc1 + 1];
    if (tc1 + 2 < Tout) q1.z = r[tc1 + 2];
    if (tc1 + 3 < Tout) q1.w = r[tc1 + 3];
  }
}

template<int RK>
__device__ __forceinline__ void compute_tile(float4 *a, const float4 (*Ws)[16],
                                             const float4 (*Xs)[32], int ty, int tx) {
#pragma unroll
  for (int rr = 0; rr < RK; rr++) {
    float4 w = Ws[rr][ty], v0 = Xs[rr][tx], v1 = Xs[rr][tx + 16];
    a[0].x += w.x * v0.x; a[0].y += w.y * v0.x; a[0].z += w.z * v0.x; a[0].w += w.w * v0.x;
    a[1].x += w.x * v0.y; a[1].y += w.y * v0.y; a[1].z += w.z * v0.y; a[1].w += w.w * v0.y;
    a[2].x += w.x * v0.z; a[2].y += w.y * v0.z; a[2].z += w.z * v0.z; a[2].w += w.w * v0.z;
    a[3].x += w.x * v0.w; a[3].y += w.y * v0.w; a[3].z += w.z * v0.w; a[3].w += w.w * v0.w;
    a[4].x += w.x * v1.x; a[4].y += w.y * v1.x; a[4].z += w.z * v1.x; a[4].w += w.w * v1.x;
    a[5].x += w.x * v1.y; a[5].y += w.y * v1.y; a[5].z += w.z * v1.y; a[5].w += w.w * v1.y;
    a[6].x += w.x * v1.z; a[6].y += w.y * v1.z; a[6].z += w.z * v1.z; a[6].w += w.w * v1.z;
    a[7].x += w.x * v1.w; a[7].y += w.y * v1.w; a[7].z += w.z * v1.w; a[7].w += w.w * v1.w;
  }
}

template<int RK, int PIPE>
__global__ void __launch_bounds__(256) ig8_v2(const float *__restrict__ x, int64_t ldx, int cin,
                                              int K, float *__restrict__ y, int64_t Tout, int cout,
                                              const float4 *__restrict__ w4,
                                              const float *__restrict__ bias, int pad) {
  __shared__ float4 Ws[RK][16];
  __shared__ float4 Xs[RK][32];
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int nblk = (cout + 3) / 4, kc = K * cin;
  const int64_t t0 = (int64_t)blockIdx.x * 128;
  const int co0 = (blockIdx.y * 16 + ty) * 4;
  float4 bz = make_float4(co0 + 0 < cout ? bias[co0 + 0] : 0.f,
                          co0 + 1 < cout ? bias[co0 + 1] : 0.f,
                          co0 + 2 < cout ? bias[co0 + 2] : 0.f,
                          co0 + 3 < cout ? bias[co0 + 3] : 0.f);
  float4 a[8];
#pragma unroll
  for (int j = 0; j < 8; j++) a[j] = bz;
  const int lr = tid >> 4, lq = tid & 15;
  const int64_t tc0 = t0 + lq * 4, tc1 = t0 + (lq + 16) * 4;

  if (PIPE) {
    float4 wn, xn0, xn1;
    stage_row<RK>(x, ldx, cin, K, kc, nblk, pad, Tout, w4, 0, lr, lq, tc0, tc1, wn, xn0, xn1);
    for (int r0 = 0; r0 < kc; r0 += RK) {
      Ws[lr][lq] = wn; Xs[lr][lq] = xn0; Xs[lr][lq + 16] = xn1;
      if (RK > 16) stage_row<RK>(x, ldx, cin, K, kc, nblk, pad, Tout, w4, r0 + 16, lr, lq, tc0, tc1,
                                 wn, xn0, xn1);
      __syncthreads();
      if (r0 + RK < kc)
        stage_row<RK>(x, ldx, cin, K, kc, nblk, pad, Tout, w4, r0 + RK, lr, lq, tc0, tc1, wn, xn0, xn1);
      compute_tile<RK>(a, Ws, Xs, ty, tx);
      __syncthreads();
    }
  } else {
    for (int r0 = 0; r0 < kc; r0 += RK) {
      float4 w, q0, q1;
      stage_row<RK>(x, ldx, cin, K, kc, nblk, pad, Tout, w4, r0, lr, lq, tc0, tc1, w, q0, q1);
      Ws[lr][lq] = w; Xs[lr][lq] = q0; Xs[lr][lq + 16] = q1;
      if (RK > 16) {
        float4 w2, q2, q3;
        stage_row<RK>(x, ldx, cin, K, kc, nblk, pad, Tout, w4, r0 + 16, lr, lq, tc0, tc1, w2, q2, q3);
        Ws[lr + 16][lq] = w2; Xs[lr + 16][lq] = q2; Xs[lr + 16][lq + 16] = q3;
      }
      __syncthreads();
      compute_tile<RK>(a, Ws, Xs, ty, tx);
      __syncthreads();
    }
  }

#pragma unroll
  for (int j = 0; j < 8; j++) {
    int64_t col = t0 + ((j < 4) ? 4 * tx + j : 64 + 4 * tx + (j - 4));
    float *src = &a[j].x;
#pragma unroll
    for (int ln = 0; ln < 4; ln++)
      if (co0 + ln < cout && col < Tout) y[(int64_t)(co0 + ln) * Tout + col] = src[ln];
  }
}

static float *d_x, *d_y, *d_w, *d_b;
static int64_t ldxg;

template<int RK, int PIPE>
static double run2(int cin, int cout, int K, int64_t T, int pad, int reps) {
  int nblk = (cout + 3) / 4;
  dim3 g((unsigned)((T + 127) / 128), (unsigned)((nblk + 15) / 16), 1);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int i = 0; i < 3; i++)
    ig8_v2<RK, PIPE><<<g, 256>>>(d_x, ldxg, cin, K, d_y, T, cout, (const float4 *)d_w, d_b, pad);
  cudaEventRecord(e0);
  for (int i = 0; i < reps; i++)
    ig8_v2<RK, PIPE><<<g, 256>>>(d_x, ldxg, cin, K, d_y, T, cout, (const float4 *)d_w, d_b, pad);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms / reps;
}

int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  double peak = p.multiProcessorCount * 128 * (p.clockRate / 1e6);
  printf("device %s: %d SM %.0f MHz  fp32 peak %.1f GFMA/s   (ig8_bench ablation: shipped tile = staging/barrier bound)\n\n",
         p.name, p.multiProcessorCount, p.clockRate / 1000.0, peak);

  /* ---- numerics first: padded scratch, naive reference ---- */
  {
    int cin = 8, cout = 8, K = 5; int64_t T = 257; int pad = (K - 1) / 2;
    ldxg = T + 2 * pad;
    int nblk = (cout + 3) / 4; int64_t kc = (int64_t)K * cin;
    std::vector<float> hx(cin * ldxg, 0.f), hw(nblk * kc * 4), hb(cout), ref(cout * T, 0.f), got(cout * T, 0.f);
    for (int ci = 0; ci < cin; ci++)
      for (int64_t t = 0; t < T; t++) hx[ci * ldxg + pad + t] = (float)(((ci * 977 + t) * 31 % 71) - 35) / 35.f;
    for (size_t i = 0; i < hw.size(); i++) hw[i] = (float)((i * 17 % 43) - 21) / 43.f;
    for (int i = 0; i < cout; i++) hb[i] = (float)((i % 7) - 3) / 7.f;
    for (int co = 0; co < cout; co++)
      for (int64_t t = 0; t < T; t++) {
        double acc = hb[co];
        for (int k = 0; k < K; k++)
          for (int ci = 0; ci < cin; ci++)
            acc += (double)hw[((co / 4) * kc + (int64_t)k * cin + ci) * 4 + (co % 4)] * hx[(int64_t)ci * ldxg + t + k];
        ref[co * T + t] = (float)acc;
      }
    cudaMalloc(&d_x, hx.size() * 4); cudaMalloc(&d_y, got.size() * 4);
    cudaMalloc(&d_w, hw.size() * 4); cudaMalloc(&d_b, hb.size() * 4);
    cudaMemcpy(d_x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice);
    dim3 g((unsigned)((T + 127) / 128), (unsigned)((nblk + 15) / 16), 1);
    printf("numerics (cin=8 cout=8 K=5 T=257, naive fp32 reference):\n");
    (void)cudaGetLastError();
    float mxs[3] = {0, 0, 0};
    const char *lbl[3] = {"RK=16 PIPE=0", "RK=16 PIPE=1", "RK=32 PIPE=0 (rows 16+ uninitialized: timing-only)"};
    ig8_v2<16,0><<<g,256>>>(d_x,ldxg,cin,K,d_y,T,cout,(const float4*)d_w,d_b,pad); cudaMemcpy(got.data(),d_y,got.size()*4,cudaMemcpyDeviceToHost);
    printf("   launch err: %s\n", cudaGetErrorString(cudaGetLastError()));
    for (size_t i=0;i<got.size();i++) mxs[0]=std::max(mxs[0],std::fabs(got[i]-ref[i]));
    ig8_v2<16,1><<<g,256>>>(d_x,ldxg,cin,K,d_y,T,cout,(const float4*)d_w,d_b,pad); cudaMemcpy(got.data(),d_y,got.size()*4,cudaMemcpyDeviceToHost);
    for (size_t i=0;i<got.size();i++) mxs[1]=std::max(mxs[1],std::fabs(got[i]-ref[i]));
    ig8_v2<32,0><<<g,256>>>(d_x,ldxg,cin,K,d_y,T,cout,(const float4*)d_w,d_b,pad); cudaMemcpy(got.data(),d_y,got.size()*4,cudaMemcpyDeviceToHost);
    for (size_t i=0;i<got.size();i++) mxs[2]=std::max(mxs[2],std::fabs(got[i]-ref[i]));
    for (int v = 0; v < 3; v++)
      printf("   %-13s max|d| = %.3e  %s\n", lbl[v], mxs[v], mxs[v] < 2e-3 ? "OK" : "MISMATCH");
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_w); cudaFree(d_b);
  }

  struct Sh { int cin, cout, K; int64_t T; int pad; const char *tag; } sh[] = {
    {256, 256, 11, 828, 5, "K=11 256x256 T=828  (runtime node 22.4 GFMA/s)"},
    {128, 128, 11, 3312, 5, "K=11 128x128 T=3312 (runtime node 23.2 GFMA/s)"},
    {128, 128, 7, 3312, 3, "K=7  128x128 T=3312 (runtime node 21.6 GFMA/s)"},
    {128, 128, 3, 3312, 1, "K=3  128x128 T=3312 (runtime node 17.9 GFMA/s)"},
  };
  printf("\nrates (GFMA/s, %%; of the 1-SM peak; A = shipped tile in .auto/ig8_bench.cu)\n");
  for (unsigned s2 = 0; s2 < sizeof(sh) / sizeof(sh[0]); s2++) {
    int cin = sh[s2].cin, cout = sh[s2].cout, K = sh[s2].K, pad = sh[s2].pad;
    int64_t T = sh[s2].T; ldxg = T + 2 * pad;
    int nblk = (cout + 3) / 4; int64_t kc = (int64_t)K * cin;
    std::vector<float> hx(cin * ldxg, 0.f), hw(nblk * kc * 4), hb(cout);
    for (int ci = 0; ci < cin; ci++)
      for (int64_t t = 0; t < T; t++) hx[ci * ldxg + pad + t] = (float)(((ci * 977 + t) * 31 % 71) - 35) / 35.f;
    for (size_t i = 0; i < hw.size(); i++) hw[i] = (float)((i * 17 % 43) - 21) / 43.f;
    for (int i = 0; i < cout; i++) hb[i] = (float)((i % 7) - 3) / 7.f;
    cudaMalloc(&d_x, hx.size() * 4); cudaMalloc(&d_y, (size_t)cout * T * 4);
    cudaMalloc(&d_w, hw.size() * 4); cudaMalloc(&d_b, hb.size() * 4);
    cudaMemcpy(d_x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice);
    double macs = (double)cout * cin * K * T;
    printf("%s\n", sh[s2].tag);
    double base = 0;
    const int REPS = 30;
    double m160 = run2<16, 0>(cin, cout, K, T, pad, REPS);
    double m161 = run2<16, 1>(cin, cout, K, T, pad, REPS);
    double m320 = run2<32, 0>(cin, cout, K, T, pad, REPS);
    double m321 = run2<32, 1>(cin, cout, K, T, pad, REPS);
    base = m160;
    printf("   RK16 P0 %6.2f ms %6.1f GFMA/s (shipped tile, fast path) | RK16 P1 (pipelined) %6.2f ms %6.1f GFMA/s %+.0f%% | RK32 P0 %6.1f GFMA/s %+.0f%% | RK32 P1 %6.1f GFMA/s %+.0f%%\n",
           m160, macs / (m160 / 1e3) / 1e9,
           m161, macs / (m161 / 1e3) / 1e9, 100.0 * (m160 / m161 - 1),
           macs / (m320 / 1e3) / 1e9, 100.0 * (m160 / m320 - 1),
           macs / (m321 / 1e3) / 1e9, 100.0 * (m160 / m321 - 1));
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_w); cudaFree(d_b);
  }
  return 0;
}
