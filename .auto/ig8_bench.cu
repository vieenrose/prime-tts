/* ig8 ablation bench (standalone; NEVER linked into the runtime).
 *
 * Why: the shipped conv kernels run at 22.4 GFMA/s on the dominant nodes (measured: node 3854,
 * cin=cout=256 K=11 T=828, 597 MMAC in 26.7 ms) on a 1-SM part whose fp32 peak is 118 GFMA/s.
 * The tile's shared-bandwidth ceiling is 78.6 GFMA/s (3 LDS.128 per 32 FMAs, 128 B/clk), so the
 * kernels sit 3.5x below their own ceiling and 5.3x below FMA peak. Every structural variant
 * (ig 64-col, ig16 256-col, window-direct, templated-K, FFT, Winograd) has been measured; none of
 * them isolated WHICH TERM costs the 3.5x. This bench ablates the shipped inner loop term by term.
 *
 * Variants (timing-only ablations are marked; they compute wrong values on purpose):
 *   A  exact ig8 clone                         -> reproduce the runtime's rate (the calibration run)
 *   B  X staged once, not per chunk (stale)    -> removes the per-chunk global X staging cost
 *   C  1 quad per thread (16 FMA per 3 LDS)    -> tests the LDS-bound model (ceiling 50% vs 67%)
 *   D  no __syncthreads per chunk              -> isolates the barrier cost
 *   E  no Ws staging (Ws filled once, stale)   -> removes the per-chunk weight staging cost
 *
 * Build on the Nano:
 *   /usr/local/cuda/bin/nvcc -arch=sm_53 -O3 .auto/ig8_bench.cu -o /tmp/ig8b && /tmp/ig8b
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define IG_RK 16

/* A: the shipped kernel's inner structure, faithfully (stride 1, dil 1, pad_in = pad). */
template<int V>
__global__ void __launch_bounds__(256) ig8_bench(const float *__restrict__ x, int64_t ldx, int cin,
                                                 int K, float *__restrict__ y, int64_t Tout, int cout,
                                                 const float4 *__restrict__ w4,
                                                 const float *__restrict__ bias, int pad) {
  __shared__ float4 Ws[IG_RK][16];
  __shared__ float4 Xs[IG_RK][32];
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int nblk = (cout + 3) / 4;
  const int g = blockIdx.y * 16 + ty;
  const int64_t t0 = (int64_t) blockIdx.x * 128;
  const int kc = K * cin;
  const int co0 = g * 4;
  float4 bz = make_float4(bias[co0], bias[co0 + 1], bias[co0 + 2], bias[co0 + 3]);
  float4 a[8];
#pragma unroll
  for (int j = 0; j < 8; j++) a[j] = bz;
  const int lr = tid >> 4, lq = tid & 15;
  const int NQ = (V == 3) ? 1 : 2;          /* C: one quad per thread */
  int lg = blockIdx.y * 16 + lq;
  for (int r0 = 0; r0 < kc; r0 += IG_RK) {
    int r = r0 + lr;
    if (V != 5)
      Ws[lr][lq] = (r < kc && lg < nblk) ? w4[(int64_t) lg * kc + r] : make_float4(0.f, 0.f, 0.f, 0.f);
    int k = r / cin, ci = r - k * cin;
    const float *row = x + (int64_t) ci * ldx;
#pragma unroll
    for (int h = 0; h < NQ; h++) {
      int qq = lq + 16 * h;
      float4 xv = make_float4(0.f, 0.f, 0.f, 0.f);
      if (r < kc && !(V == 2 && r0 > 0)) {   /* B: stage only in the first chunk */
        int64_t tc = t0 + qq * 4;
        int64_t s0 = tc + k - pad;
        if (tc + 0 < Tout && s0 >= 0 && s0 < ldx) xv.x = row[s0];
        int64_t s1 = s0 + 1;
        if (tc + 1 < Tout && s1 >= 0 && s1 < ldx) xv.y = row[s1];
        int64_t s2 = s1 + 1;
        if (tc + 2 < Tout && s2 >= 0 && s2 < ldx) xv.z = row[s2];
        int64_t s3 = s2 + 1;
        if (tc + 3 < Tout && s3 >= 0 && s3 < ldx) xv.w = row[s3];
      }
      Xs[lr][qq] = xv;
    }
    if (V != 4) __syncthreads();
#pragma unroll
    for (int rr = 0; rr < IG_RK; rr++) {
      float4 w = Ws[rr][ty];
      float4 v0 = Xs[rr][tx];
      float4 v1 = (V == 3) ? v0 : Xs[rr][tx + 16];
      a[0].x += w.x * v0.x; a[0].y += w.y * v0.x; a[0].z += w.z * v0.x; a[0].w += w.w * v0.x;
      a[1].x += w.x * v0.y; a[1].y += w.y * v0.y; a[1].z += w.z * v0.y; a[1].w += w.w * v0.y;
      a[2].x += w.x * v0.z; a[2].y += w.y * v0.z; a[2].z += w.z * v0.z; a[2].w += w.w * v0.z;
      a[3].x += w.x * v0.w; a[3].y += w.y * v0.w; a[3].z += w.z * v0.w; a[3].w += w.w * v0.w;
      a[4].x += w.x * v1.x; a[4].y += w.y * v1.x; a[4].z += w.z * v1.x; a[4].w += w.w * v1.x;
      a[5].x += w.x * v1.y; a[5].y += w.y * v1.y; a[5].z += w.z * v1.y; a[5].w += w.w * v1.y;
      a[6].x += w.x * v1.z; a[6].y += w.y * v1.z; a[6].z += w.z * v1.z; a[6].w += w.w * v1.z;
      a[7].x += w.x * v1.w; a[7].y += w.y * v1.w; a[7].z += w.z * v1.w; a[7].w += w.w * v1.w;
    }
    if (V != 4) __syncthreads();
  }
  /* write the tile: a[j].component ln = (column j, lane co0+ln), columns 0..3 at 4*tx and
     4..7 at 64+4*tx -- the shipped kernel's mapping. */
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
static int64_t ldx;

static double run(int Vsel, int cin, int cout, int K, int64_t Tout, int pad) {
  int nblk = (cout + 3) / 4;
  dim3 g((unsigned)((Tout + 127) / 128), (unsigned)((nblk + 15) / 16), 1);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  float4 *w4 = (float4 *) d_w;
  #define L(V) ig8_bench<V><<<g, 256>>>(d_x, ldx, cin, K, d_y, Tout, cout, w4, d_b, pad)
  for (int i = 0; i < 3; i++) {
    switch (Vsel) { case 0: L(0); break; case 2: L(2); break; case 3: L(3); break;
                    case 4: L(4); break; case 5: L(5); break; }
  }
  cudaEventRecord(e0);
  const int REPS = 20;
  for (int i = 0; i < REPS; i++) {
    switch (Vsel) { case 0: L(0); break; case 2: L(2); break; case 3: L(3); break;
                    case 4: L(4); break; case 5: L(5); break; }
  }
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms / REPS;
}

int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("device %s: %d SM, %.0f MHz, fp32 peak %.1f GFMA/s, shared ceiling (3 LDS.128 : 32 FMA) %.1f GFMA/s\n\n",
         p.name, p.multiProcessorCount, p.clockRate / 1000.0,
         p.multiProcessorCount * 128 * (p.clockRate / 1e6),
         p.multiProcessorCount * 128 * (p.clockRate / 1e6) * (2.0 / 3.0));

  struct Sh { int cin, cout, K; int64_t T; int pad; const char *tag; } sh[] = {
    {256, 256, 11, 828, 5, "resblocks.2 K=11 T=828 (26.7 ms in runtime = 22.4 GFMA/s)"},
    {128, 128, 11, 3312, 5, "resblocks.5 K=11 T=3312 (25.7 ms in runtime = 23.2 GFMA/s)"},
    {128, 128, 7, 3312, 3, "resblocks.5 K=7  T=3312 (17.6 ms in runtime = 21.6 GFMA/s)"},
    {128, 128, 3, 3312, 1, "resblocks.3 K=3  T=3312 (9.1 ms in runtime = 17.9 GFMA/s)"},
  };
  const char *vn[] = {"A exact-clone", "B X-staged-once", "C 1-quad(16FMA/3LDS)", "D no-barrier", "E W-staged-once"};
  const int vs[] = {0, 2, 3, 4, 5};

  for (unsigned s = 0; s < sizeof(sh) / sizeof(sh[0]); s++) {
    int cin = sh[s].cin, cout = sh[s].cout, K = sh[s].K, pad = sh[s].pad;
    int64_t T = sh[s].T;
    ldx = T;
    int nblk = (cout + 3) / 4;
    int64_t kc = (int64_t)K * cin;
    size_t xb = (size_t)cin * T * 4, yb = (size_t)cout * T * 4,
           wb = (size_t)nblk * kc * 16, bb = (size_t)cout * 4;
    std::vector<float> hx(xb / 4), hy(yb / 4), hw(wb / 4), hb(cout);
    for (size_t i = 0; i < hx.size(); i++) hx[i] = (float)((i * 37 % 101) - 50) / 50.f;
    for (size_t i = 0; i < hw.size(); i++) hw[i] = (float)((i * 53 % 97) - 48) / 96.f;
    for (int i = 0; i < cout; i++) hb[i] = (float)((i % 11) - 5) / 10.f;
    cudaMalloc(&d_x, xb); cudaMalloc(&d_y, yb); cudaMalloc(&d_w, wb); cudaMalloc(&d_b, bb);
    cudaMemcpy(d_x, hx.data(), xb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, hw.data(), wb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, hb.data(), bb, cudaMemcpyHostToDevice);
    double macs = (double)cout * cin * K * T;
    printf("shape %s: %d blocks, %.0f MMAC\n", sh[s].tag,
           (int)((T + 127) / 128) * ((nblk + 15) / 16), macs / 1e6);
    for (unsigned v = 0; v < sizeof(vs) / sizeof(vs[0]); v++) {
      double ms = run(vs[v], cin, cout, K, T, pad);
      printf("   %-20s %7.2f ms  %6.1f GFMA/s\n", vn[v], ms, macs / (ms / 1000.0) / 1e9);
    }
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_w); cudaFree(d_b);
  }

  /* numeric validation of the exact clone on a small case (naive reference, pad=K-1/2, stride 1) */
  {
    int cin = 8, cout = 8, K = 5; int64_t T = 257; int pad = (K - 1) / 2;
    ldx = T;
    int nblk = (cout + 3) / 4; int64_t kc = (int64_t)K * cin;
    std::vector<float> hx(cin * T), hw(nblk * kc * 4), hb(cout), ref(cout * T, 0.f), got(cout * T);
    for (size_t i = 0; i < hx.size(); i++) hx[i] = (float)((i * 31 % 71) - 35) / 35.f;
    for (size_t i = 0; i < hw.size(); i++) hw[i] = (float)((i * 17 % 43) - 21) / 43.f;
    for (int i = 0; i < cout; i++) hb[i] = (float)((i % 7) - 3) / 7.f;
    for (int co = 0; co < cout; co++)
      for (int64_t t = 0; t < T; t++) {
        double acc = hb[co];
        for (int k = 0; k < K; k++)
          for (int ci = 0; ci < cin; ci++) {
            int64_t s = t + k - pad;
            if (s >= 0 && s < T) acc += (double)hw[((co / 4) * kc + (int64_t)k * cin + ci) * 4 + (co % 4)] * hx[(int64_t)ci * T + s];
          }
        ref[co * T + t] = (float)acc;
      }
    cudaMalloc(&d_x, hx.size() * 4); cudaMalloc(&d_y, got.size() * 4);
    cudaMalloc(&d_w, hw.size() * 4); cudaMalloc(&d_b, hb.size() * 4);
    cudaMemcpy(d_x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice);
    dim3 g((unsigned)((T + 127) / 128), (unsigned)((nblk + 15) / 16), 1);
    ig8_bench<0><<<g, 256>>>(d_x, ldx, cin, K, d_y, T, cout, (const float4 *)d_w, d_b, pad);
    cudaMemcpy(got.data(), d_y, got.size() * 4, cudaMemcpyDeviceToHost);
    double mx = 0;
    for (size_t i = 0; i < got.size(); i++) mx = std::max(mx, (double)std::fabs(got[i] - ref[i]));
    printf("\nclone numeric check (cin=8 cout=8 K=5 T=257): max|d| = %.3e  -> %s\n", mx,
           mx < 2e-3 ? "MATCHES the naive conv (the clone is faithful)" : "MISMATCH (clone is not faithful; rate numbers are not comparable)");
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_w); cudaFree(d_b);
  }
  return 0;
}

/* ---- v2 family: fast staging path (zero-padded scratch, 1 bound check like the runtime's
   pad_in<0 path), configurable chunk width RK, optional register-pipelined prefetch. ---- */
template<int RK, int PIPE>
__global__ void __launch_bounds__(256) ig8_v2(const float *__restrict__ x, int64_t ldx, int cin,
                                              int K, float *__restrict__ y, int64_t Tout, int cout,
                                              const float4 *__restrict__ w4,
                                              const float *__restrict__ bias, int pad) {
  __shared__ float4 Ws[RK][16];
  __shared__ float4 Xs[RK][32];
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int nblk = (cout + 3) / 4;
  const int g = blockIdx.y * 16 + ty;
  const int64_t t0 = (int64_t) blockIdx.x * 128;
  const int kc = K * cin;
  const int co0 = g * 4;
  float4 bz = make_float4(bias[co0], bias[co0 + 1], bias[co0 + 2], bias[co0 + 3]);
  float4 a[8];
#pragma unroll
  for (int j = 0; j < 8; j++) a[j] = bz;
  const int lr = tid >> 4, lq = tid & 15;
  const int lg = blockIdx.y * 16 + lq;
  const int r = lr;                       /* staged rank row for this thread */
  const int64_t tc0 = t0 + lq * 4, tc1 = t0 + (lq + 16) * 4;

/* stage chunk (rank rows r0..r0+RK-1): 1 weight quad + 2 input quads, ONE bound check each */
#define STAGE(r0) do { \
    int rr = (r0) + r; \
    Ws[r][lq] = (rr < kc && lg < nblk) ? w4[(int64_t) lg * kc + rr] : make_float4(0.f, 0.f, 0.f, 0.f); \
    if (rr < kc) { \
      int k = rr / cin, ci = rr - k * cin; \
      const float *row = x + (int64_t) ci * ldx + k; \
      float4 q0 = make_float4(0.f, 0.f, 0.f, 0.f), q1 = q0; \
      if (tc0 + 3 < Tout) q0 = *(const float4 *)(row + (tc0 - pad)); \
      else { if (tc0 + 0 < Tout) q0.x = row[tc0 - pad]; if (tc0 + 1 < Tout) q0.y = row[tc0 - pad + 1]; \
             if (tc0 + 2 < Tout) q0.z = row[tc0 - pad + 2]; if (tc0 + 3 < Tout) q0.w = row[tc0 - pad + 3]; } \
      if (tc1 + 3 < Tout) q1 = *(const float4 *)(row + (tc1 - pad)); \
      else { if (tc1 + 0 < Tout) q1.x = row[tc1 - pad]; if (tc1 + 1 < Tout) q1.y = row[tc1 - pad + 1]; \
             if (tc1 + 2 < Tout) q1.z = row[tc1 - pad + 2]; if (tc1 + 3 < Tout) q1.w = row[tc1 - pad + 3]; } \
      Xs[r][lq] = q0; Xs[r][lq + 16] = q1; \
    } else { Xs[r][lq] = make_float4(0.f,0.f,0.f,0.f); Xs[r][lq + 16] = Xs[r][lq]; } \
  } while (0)

#define COMPUTE() do { \
    __syncthreads(); \
    _Pragma("unroll") \
    for (int rr = 0; rr < RK; rr++) { \
      float4 w = Ws[rr][ty]; float4 v0 = Xs[rr][tx]; float4 v1 = Xs[rr][tx + 16]; \
      a[0].x += w.x*v0.x; a[0].y += w.y*v0.x; a[0].z += w.z*v0.x; a[0].w += w.w*v0.x; \
      a[1].x += w.x*v0.y; a[1].y += w.y*v0.y; a[1].z += w.z*v0.y; a[1].w += w.w*v0.y; \
      a[2].x += w.x*v0.z; a[2].y += w.y*v0.z; a[2].z += w.z*v0.z; a[2].w += w.w*v0.z; \
      a[3].x += w.x*v0.w; a[3].y += w.y*v0.w; a[3].z += w.z*v0.w; a[3].w += w.w*v0.w; \
      a[4].x += w.x*v1.x; a[4].y += w.y*v1.x; a[4].z += w.z*v1.x; a[4].w += w.w*v1.x; \
      a[5].x += w.x*v1.y; a[5].y += w.y*v1.y; a[5].z += w.z*v1.y; a[5].w += w.w*v1.y; \
      a[6].x += w.x*v1.z; a[6].y += w.y*v1.z; a[6].z += w.z*v1.z; a[6].w += w.w*v1.z; \
      a[7].x += w.x*v1.w; a[7].y += w.y*v1.w; a[7].z += w.z*v1.w; a[7].w += w.w*v1.w; \
    } \
  } while (0)

  if (PIPE) {                     /* prefetch chunk 0, then prefetch chunk i+1 while computing chunk i */
    STAGE(0);
    for (int r0 = 0; r0 < kc; r0 += RK) {
      if (r0 + RK < kc) {
        float4 wsave = Ws[r][lq]; float4 x0 = Xs[r][lq], x1 = Xs[r][lq + 16];
        COMPUTE();
        __syncthreads();
        (void) wsave; (void) x0; (void) x1;
        STAGE(r0 + RK);
      } else { COMPUTE(); __syncthreads(); }
    }
  } else {
    for (int r0 = 0; r0 < kc; r0 += RK) { STAGE(r0); COMPUTE(); __syncthreads(); }
  }
#undef STAGE
#undef COMPUTE

#pragma unroll
  for (int j = 0; j < 8; j++) {
    int64_t col = t0 + ((j < 4) ? 4 * tx + j : 64 + 4 * tx + (j - 4));
    float *src = &a[j].x;
#pragma unroll
    for (int ln = 0; ln < 4; ln++)
      if (co0 + ln < cout && col < Tout) y[(int64_t)(co0 + ln) * Tout + col] = src[ln];
  }
}
