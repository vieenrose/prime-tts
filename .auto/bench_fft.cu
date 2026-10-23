/* Micro-benchmark for the FFT (overlap-add) conv idea on the Jetson Nano (sm_53).
 *
 * Motivation (measured, session runs 315+): the non-dilated K=11 convs (cin=cout=256, T=904 and
 * cin=cout=128, T=3616) run at ~5% of fp32 peak with the GEMM-tiled kernels (node 3854 = 51.8 ms for
 * 1.29 GFLOP). A 1-D FFT conv with block N=32 and L=22 needs 32 complex multiplies per 22 outputs per
 * channel pair = 8.7 flops/output/cin against 22 for direct K=11, i.e. ~2.5x fewer ops. This file
 * prices it: a direct tiled kernel (calibrated to the runtime's measured time), an FFT kernel, and a
 * naive reference for correctness. Nothing here touches the runtime; it is a standalone measurement.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x)                                                                                  \
  do {                                                                                         \
    cudaError_t e_ = (x);                                                                      \
    if (e_ != cudaSuccess) {                                                                   \
      fprintf(stderr, "cuda error %s at line %d\n", cudaGetErrorString(e_), __LINE__);        \
      exit(1);                                                                                 \
    }                                                                                          \
  } while (0)

/* ---------------- naive reference: y[c][n] = sum_ci sum_k w[c][ci][k] x[ci][n-k] (zero outside) */
__global__ void ref_conv(const float *x, const float *w, float *y, int cin, int cout, int K,
                         long long T) {
  long long c = blockIdx.y * blockDim.y + threadIdx.y;   /* cout */
  long long n = blockIdx.x * blockDim.x + threadIdx.x;   /* time */
  if (c >= cout || n >= T)
    return;
  float acc = 0.f;
  for (int ci = 0; ci < cin; ci++)
    for (int k = 0; k < K; k++) {
      long long t = n - k;
      if (t >= 0)
        acc += w[(c * cin + ci) * K + k] * x[ci * T + t];
    }
  y[c * T + n] = acc;
}

/* ---------------- direct tiled (mirrors the runtime's ig8 shape: 4 cout x 8 outputs per thread,
 * shared W[ci 16][cout 8][K], 128 threads, register window of K+7 input values per ci) */
#define FB_N 32
#define FB_L 22
__global__ void direct_conv(const float *__restrict__ x, const float *__restrict__ w,
                            float *__restrict__ y, int cin, int cout, int K, long long T) {
  __shared__ float ws[16][8][11];
  const int cg = blockIdx.y * 8;               /* cout base */
  const int t0 = blockIdx.x * 128;             /* time base: 16 groups x 8 outputs */
  int tid = threadIdx.x;                       /* 0..127 */
  int cl = tid & 7;                            /* local cout (8 channels per block) */
  int tg = tid >> 3;                           /* output group 0..15 */
  const int base = t0 + tg * 8;
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; i++)
    acc[i] = 0.f;
  for (int cb = 0; cb < cin; cb += 16) {
    for (int i = tid; i < 16 * 8 * 11; i += 128) {
      int k = i % 11, c = (i / 11) & 7, ci = i / (11 * 8);
      ws[ci][c][k] = w[((cg + c) * cin + (cb + ci)) * K + k];
    }
    __syncthreads();
#pragma unroll
    for (int ci = 0; ci < 16; ci++) {
      float xv[18];                            /* x[base-10 .. base+7] */
#pragma unroll
      for (int j = 0; j < 18; j++) {
        long long t = base - 10 + j;
        xv[j] = (t >= 0 && t < T) ? x[(cb + ci) * T + t] : 0.f;
      }
#pragma unroll
      for (int k = 0; k < 11; k++) {
        float wv = ws[ci][cl][k];
#pragma unroll
        for (int i = 0; i < 8; i++)
          acc[i] += wv * xv[10 - k + i];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 8; i++) {
    long long n = base + i;
    if (n < T)
      y[(cg + cl) * T + n] = acc[i];
  }
}

/* ---------------- FFT conv: N=32, L=22, block = 16 cout x 8 tiles, thread = (cout, tile)
 * W_spec[cout][cin][32] complex pairs are precomputed on the host and uploaded.
 * Each thread accumulates Y_spec (32 complex) over cin, does an in-register inverse FFT,
 * writes L=22 outputs, and atomically adds the 10-value tail into the next tile's head. */
#define FB_N 32
#define FB_L 22

/* Packed radix-2 twiddles: butterfly j of a stage whose half-length is h uses W_{2h}^j stored in slot
 * j + h - 1; the slots of the five stages (h = 16,8,4,2,1) tile [0,32) exactly. Forward = e^{-2pi i
 * j/2h}, inverse = conjugate. Stages: forward decimation-in-frequency (natural input, bit-reversed
 * output), inverse decimation-in-frequency run backwards (bit-reversed input, natural output). The
 * convolution theorem is insensitive to the shared bit-reversal as long as the weight spectrum is
 * stored in the same (bit-reversed) bin order. */
__device__ __forceinline__ void fill_tw(float *twr, float *twi) {
  int tid = threadIdx.x;
  for (int i = tid; i < 31; i += blockDim.x) {
    int h, j;
    if (i >= 15) { h = 16; j = i - 15; }
    else if (i >= 7) { h = 8; j = i - 7; }
    else if (i >= 3) { h = 4; j = i - 3; }
    else if (i >= 1) { h = 2; j = i - 1; }
    else { h = 1; j = 0; }
    float ang = -6.2831853071795864f * (float)j / (float)(2 * h);
    twr[i] = cosf(ang);
    twi[i] = sinf(ang);
  }
  __syncthreads();
}

template <int H>
__device__ __forceinline__ void wstage(float *re, float *im, const float *twr, const float *twi,
                                       int inv) {
  const int S = (H == 16) ? 15 : (H == 8) ? 7 : (H == 4) ? 3 : (H == 2) ? 1 : 0;
#pragma unroll
  for (int g = 0; g < 32 / (2 * H); g++) {
#pragma unroll
    for (int j = 0; j < H; j++) {
      int a = g * 2 * H + j, b = a + H;
      float ar = re[a], ai = im[a];
      float br = re[b], bi = im[b];
      re[a] = ar + br;
      im[a] = ai + bi;
      float dr = ar - br, di = ai - bi;
      float wr = twr[S + j], wi = inv ? -twi[S + j] : twi[S + j];
      re[b] = dr * wr - di * wi;
      im[b] = dr * wi + di * wr;
    }
  }
}

__device__ __forceinline__ void fft32(float *re, float *im, const float *twr, const float *twi,
                                      int inv) {
  if (!inv) {
    wstage<16>(re, im, twr, twi, 0);
    wstage<8>(re, im, twr, twi, 0);
    wstage<4>(re, im, twr, twi, 0);
    wstage<2>(re, im, twr, twi, 0);
    wstage<1>(re, im, twr, twi, 0);
  } else {
    wstage<1>(re, im, twr, twi, 1);
    wstage<2>(re, im, twr, twi, 1);
    wstage<4>(re, im, twr, twi, 1);
    wstage<8>(re, im, twr, twi, 1);
    wstage<16>(re, im, twr, twi, 1);
  }
}

__global__ void fft_conv(const float *__restrict__ x, const float *__restrict__ wspec,
                         float *__restrict__ y, int cin, int cout, int tiles, long long T) {
  __shared__ float xs[8][8][FB_N][2];    /* [tile][ci][bin] for the current ci chunk */
  __shared__ float ws[8][8][FB_N][2];    /* [cout][ci][bin] weight spectrum, staged */
  __shared__ float xtime[8][8][FB_L];    /* time-domain tiles */
  __shared__ float twr[32], twi[32];
  const int cb = blockIdx.y * 8;         /* cout base */
  const int tb = blockIdx.x * 8;         /* tile base */
  int tid = threadIdx.x;                 /* 0..63 */
  int tl = tid >> 3;                     /* tile 0..7 */
  int cl = tid & 7;                      /* local cout */
  fill_tw(twr, twi);
  float yr[FB_N], yi[FB_N];
#pragma unroll
  for (int i = 0; i < FB_N; i++) {
    yr[i] = 0.f;
    yi[i] = 0.f;
  }
  for (int cb2 = 0; cb2 < cin; cb2 += 8) {
    /* stage the time-domain tiles: 8 tiles x 8 ci x 22 samples */
    for (int i = tid; i < 8 * 8 * FB_L; i += 64) {
      int st = i % FB_L, ci = (i / FB_L) & 7, t = i / (FB_L * 8);
      long long n = (long long)(tb + t) * FB_L + st;
      xtime[t][ci][st] = (n < T) ? x[(cb2 + ci) * T + n] : 0.f;
    }
    /* stage the weight spectrum: 8 cout x 8 ci x 32 bins (complex) */
    for (int i = tid; i < 8 * 8 * FB_N * 2; i += 64) {
      int b = i % (FB_N * 2);
      int ci = (i / (FB_N * 2)) & 7, c = i / (FB_N * 2 * 8);
      ws[c][ci][b / 2][b % 2] = wspec[((cb + c) * cin + (cb2 + ci)) * FB_N * 2 + b];
    }
    __syncthreads();
    /* forward transforms: 64 (tile, ci) pairs, one per thread */
    {
      int t = tid >> 3, ci = tid & 7;
      float re[FB_N], im[FB_N];
#pragma unroll
      for (int i = 0; i < FB_L; i++) {
        re[i] = xtime[t][ci][i];
        im[i] = 0.f;
      }
#pragma unroll
      for (int i = FB_L; i < FB_N; i++) {
        re[i] = 0.f;
        im[i] = 0.f;
      }
      fft32(re, im, twr, twi, 0);
#pragma unroll
      for (int i = 0; i < FB_N; i++) {
        xs[t][ci][i][0] = re[i];
        xs[t][ci][i][1] = im[i];
      }
    }
    __syncthreads();
    /* elementwise products over the 8 ci: thread (tile tl, cout cl) */
#pragma unroll
    for (int ci = 0; ci < 8; ci++) {
#pragma unroll
      for (int b = 0; b < FB_N; b++) {
        float xr = xs[tl][ci][b][0], xi = xs[tl][ci][b][1];
        float wr = ws[cl][ci][b][0], wi = ws[cl][ci][b][1];
        yr[b] += xr * wr - xi * wi;
        yi[b] += xr * wi + xi * wr;
      }
    }
    __syncthreads();
  }
  /* inverse transform (bit-reversed input -> natural output), scaled by 1/N */
  fft32(yr, yi, twr, twi, 1);
  long long base = (long long)(tb + tl) * FB_L;
#pragma unroll
  for (int i = 0; i < FB_N; i++) {
    float v = yi[i] * 0.03125f;
    long long n = base + i;
    atomicAdd(&y[(cb + cl) * (T + FB_N) + n], v);
  (void)T;
  }
}

/* --- the runtime's window kernel, verbatim conventions: float4-grouped weights [cout4/4][K*cin],
 * padded-in-kernel with pad_in shift, ldx == Tout --- */
__global__ void __launch_bounds__(128) win_conv(const float *__restrict__ x, int64_t ldx, int cin,
                                                int K, int pad_in, float *__restrict__ y, int64_t ldy,
                                                int64_t Tout, int cout,
                                                const float *__restrict__ wp) {
  __shared__ float Ws[16][8][11];
  const int tid = threadIdx.x;
  const int co0 = blockIdx.y * 8;
  const int64_t t0 = (int64_t) blockIdx.x * 128 + (tid >> 3) * 8;
  const float *xb = x;
  const float4 *w4 = (const float4 *) wp;
  const int kc = K * cin;
  const int g0 = (co0 >> 2) * kc;
  const int g1 = ((co0 >> 2) + 1) * kc;
  const int cl = tid & 7;
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; j++)
    acc[j] = 0.f;
  const int64_t base = (pad_in >= 0) ? (t0 - pad_in) : t0;
  for (int cb = 0; cb < cin; cb += 16) {
    for (int i = tid; i < 16 * 8 * 11; i += 128) {
      int k = i % 11, c = (i / 11) & 7, ci = i / (11 * 8);
      float v = 0.f;
      if (k < K && cb + ci < cin && co0 + c < cout) {
        const float4 *w = ((c & 4) ? w4 + g1 : w4 + g0) + k * cin + (cb + ci);
        v = (&w->x)[c & 3];
      }
      Ws[ci][c][k] = v;
    }
    __syncthreads();
#pragma unroll
    for (int ci = 0; ci < 16; ci++) {
      if (cb + ci >= cin)
        break;
      float xv[18];
      const float *row = xb + (int64_t) (cb + ci) * ldx;
#pragma unroll
      for (int j = 0; j < 18; j++) {
        int64_t sq = base + j;
        xv[j] = (sq >= 0 && sq < ldx) ? row[sq] : 0.f;
      }
      for (int k = 0; k < K; k++) {
        float wv = Ws[ci][cl][k];
#pragma unroll
        for (int i = 0; i < 8; i++)
          acc[i] += wv * xv[i + k];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int64_t t = t0 + j;
    if (t >= Tout || co0 + j >= cout)
      continue;
    y[(int64_t) (co0 + cl) * ldy + t] = acc[j];
  }
}

/* reference with the same convention: y[co][n] = sum_ci sum_k w4g[co][ci][k] * xp[n - k] where
 * xp[j] = x[ci][j - pad_in] valid for j in [0, ldx) */
__global__ void ref_conv_pad(const float *__restrict__ x, const float *__restrict__ w4, int cin,
                             int cout, int K, int pad_in, float *__restrict__ y, int64_t ldx,
                             int64_t Tout) {
  int co = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t n = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (co >= cout || n >= Tout)
    return;
  float acc = 0.f;
  for (int ci = 0; ci < cin; ci++)
    for (int k = 0; k < K; k++) {
      int64_t j = n + k - pad_in;
      if (j < 0 || j >= ldx)
        continue;
      const float4 *w = (const float4 *)w4 + (co >> 2) * K * cin + k * cin + ci;
      acc += (&w->x)[co & 3] * x[(int64_t) ci * ldx + j];
    }
  y[(int64_t) co * Tout + n] = acc;
}

/* --- SAME algorithm as win_conv but TEMPLATED on K: the k-loop bound is a compile-time constant so
 * the 18-value register window and the Ws indices stay constant-indexed. This is the
 * variable-K-vs-constant-K hypothesis for win_conv's 39.5 ms vs direct_conv's 21.9 ms. */
template <int TK>
__global__ void __launch_bounds__(128) winT_conv(const float *__restrict__ x, int64_t ldx, int cin,
                                                 int pad_in, float *__restrict__ y, int64_t ldy,
                                                 int64_t Tout, int cout,
                                                 const float *__restrict__ wp) {
  __shared__ float Ws[16][8][TK];
  const int tid = threadIdx.x;
  const int co0 = blockIdx.y * 8;
  const int64_t t0 = (int64_t) blockIdx.x * 128 + (tid >> 3) * 8;
  const float4 *w4 = (const float4 *)wp;
  const int cl = tid & 7;
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; j++)
    acc[j] = 0.f;
  const int64_t base = (pad_in >= 0) ? (t0 - pad_in) : t0;
  for (int cb = 0; cb < cin; cb += 16) {
    for (int i = tid; i < 16 * 8 * TK; i += 128) {
      int k = i % TK, c = (i / TK) & 7, ci = i / (TK * 8);
      const float4 *w = w4 + (int64_t) ((co0 >> 2) + ((c & 4) ? 1 : 0)) * TK * cin + k * cin + (cb + ci);
      Ws[ci][c][k] = (cb + ci < cin && co0 + c < cout) ? (&w->x)[c & 3] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int ci = 0; ci < 16; ci++) {
      if (cb + ci >= cin)
        break;
      float xv[TK + 7];
      const float *row = x + (int64_t) (cb + ci) * ldx;
#pragma unroll
      for (int j = 0; j < TK + 7; j++) {
        int64_t sq = base + j;
        xv[j] = (sq >= 0 && sq < ldx) ? row[sq] : 0.f;
      }
#pragma unroll
      for (int k = 0; k < TK; k++) {
        float wv = Ws[ci][cl][k];
#pragma unroll
        for (int i = 0; i < 8; i++)
          acc[i] += wv * xv[i + k];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int64_t t = t0 + j;
    if (t < Tout)
      y[(int64_t) (co0 + cl) * ldy + t] = acc[j];
  }
}

__device__ __forceinline__ float bench_epi(float v, int op, float a) {
  if (op == 1) return v >= 0.0f ? v : a * v;
  if (op == 2) return v > 0.0f ? v : 0.0f;
  if (op == 3) return v * a;
  if (op == 4) return v + a;
  return v;
}
/* the runtime kernel verbatim */
template <int TK>
__global__ void __launch_bounds__(128) winR_conv(
    const float *__restrict__ x, int64_t ldx, int cin, float *__restrict__ y, int64_t ldy,
    int64_t Tout, int cout, const float *__restrict__ wp, const float *__restrict__ bias,
    int64_t bsx, int64_t bsy, int epi0, int epi1, float ea0, float ea1, int64_t pad_in) {
  __shared__ float Ws[16][8][TK];
  const int tid = threadIdx.x;
  const int co0 = blockIdx.y * 8;
  const int64_t b = blockIdx.z;
  const int64_t t0 = (int64_t) blockIdx.x * 128 + (tid >> 3) * 8;
  const float4 *w4 = (const float4 *) wp;
  const float *xb = x + b * bsx;
  const int kc = TK * cin;
  const int cl = tid & 7;
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; j++)
    acc[j] = (bias && co0 + cl < cout) ? bias[co0 + cl] : 0.f;
  const int64_t base = (pad_in >= 0) ? (t0 - pad_in) : t0;
  for (int cb = 0; cb < cin; cb += 16) {
    for (int i = tid; i < 16 * 8 * TK; i += 128) {
      int k = i % TK, c = (i / TK) & 7, ci = i / (TK * 8);
      const float4 *w = w4 + (int64_t) ((co0 >> 2) + ((c & 4) ? 1 : 0)) * kc + k * cin + (cb + ci);
      Ws[ci][c][k] = (cb + ci < cin && co0 + c < cout) ? (&w->x)[c & 3] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int ci = 0; ci < 16; ci++) {
      if (cb + ci >= cin)
        break;
      float xv[TK + 7];
      const float *row = xb + (int64_t) (cb + ci) * ldx;
#pragma unroll
      for (int j = 0; j < TK + 7; j++) {
        int64_t sq = base + j;
        xv[j] = (sq >= 0 && sq < ldx) ? row[sq] : 0.f;
      }
#pragma unroll
      for (int k = 0; k < TK; k++) {
        float wv = Ws[ci][cl][k];
#pragma unroll
        for (int i = 0; i < 8; i++)
          acc[i] += wv * xv[i + k];
      }
    }
    __syncthreads();
  }
  float *yb = y + b * bsy;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int64_t t = t0 + j;
    if (t >= Tout || co0 + cl >= cout)
      continue;
    float v = acc[j];
    if (epi0 || epi1)
      v = bench_epi(bench_epi(v, epi0, ea0), epi1, ea1);
    yb[(int64_t) (co0 + cl) * ldy + t] = v;
  }
}

static float frand(void) {
  return (float)((rand() % 2001) - 1000) / 1000.f;
}

int main(int argc, char **argv) {
  int cin = argc > 1 ? atoi(argv[1]) : 256;
  int cout = argc > 2 ? atoi(argv[2]) : 256;
  int K = argc > 3 ? atoi(argv[3]) : 11;
  long long T = argc > 4 ? atoll(argv[4]) : 904;
  int iters = argc > 5 ? atoi(argv[5]) : 30;
  int tiles = (int)((T + FB_L - 1) / FB_L);

  float *hx = new float[cin * T], *hw = new float[cout * cin * K], *hy1 = new float[cout * T],
        *hy2 = new float[cout * T], *hy3 = new float[cout * T];
  float *hws = new float[cout * cin * FB_N * 2];
  for (long long i = 0; i < cin * T; i++)
    hx[i] = frand();
  for (long long i = 0; i < (long long)cout * cin * K; i++)
    hw[i] = frand() * 0.05f;
  /* host DFT of the zero-padded kernel */
  for (int c = 0; c < cout; c++)
    for (int ci = 0; ci < cin; ci++)
      for (int b = 0; b < FB_N; b++) {
        double sr = 0, si = 0;
        for (int k = 0; k < K; k++) {
          double a = -6.283185307179586 * (double)k * (double)b / (double)FB_N;
          sr += (double)hw[(c * cin + ci) * K + k] * cos(a);
          si += (double)hw[(c * cin + ci) * K + k] * sin(a);
        }
        int rb = ((b & 1) << 4) | ((b & 2) << 2) | (b & 4) | ((b & 8) >> 2) | ((b & 16) >> 4);
        hws[((c * cin + ci) * FB_N + rb) * 2 + 0] = (float)sr;
        hws[((c * cin + ci) * FB_N + rb) * 2 + 1] = (float)si;
      }
  float *dx, *dw, *dws, *dy;
  CK(cudaMalloc(&dx, sizeof(float) * cin * T));
  CK(cudaMalloc(&dw, sizeof(float) * cout * cin * K));
  CK(cudaMalloc(&dws, sizeof(float) * (long long)cout * cin * FB_N * 2));
  CK(cudaMalloc(&dy, sizeof(float) * cout * (T + FB_N)));
  CK(cudaMemcpy(dx, hx, sizeof(float) * cin * T, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dw, hw, sizeof(float) * cout * cin * K, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dws, hws, sizeof(float) * (long long)cout * cin * FB_N * 2, cudaMemcpyHostToDevice));

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float ms;
  double gflop_direct = 2.0 * (double)T * cout * cin * K / 1e9;
  double gflop_fft = 6.0 * (double)tiles * cout * cin * FB_N / 1e9;

  /* correctness: reference */
  CK(cudaMemset(dy, 0, sizeof(float) * cout * (T + FB_N)));
  {
    dim3 b(16, 16);
    dim3 g((unsigned)((T + 15) / 16), (unsigned)((cout + 15) / 16));
    ref_conv<<<g, b>>>(dx, dw, dy, cin, cout, K, T);
  }
  CK(cudaGetLastError());
  CK(cudaMemcpy(hy1, dy, sizeof(float) * cout * T, cudaMemcpyDeviceToHost));

  /* direct */
  CK(cudaMemset(dy, 0, sizeof(float) * cout * (T + FB_N)));
  {
    dim3 b(128);
    dim3 g((unsigned)((T + 127) / 128), (unsigned)((cout + 7) / 8));
    direct_conv<<<g, b>>>(dx, dw, dy, cin, cout, K, T);
  }
  CK(cudaGetLastError());
  CK(cudaMemcpy(hy2, dy, sizeof(float) * cout * T, cudaMemcpyDeviceToHost));

  /* fft */
  CK(cudaMemset(dy, 0, sizeof(float) * cout * (T + FB_N)));
  {
    dim3 b(64);
    dim3 g((unsigned)((tiles + 7) / 8), (unsigned)((cout + 7) / 8));
    fft_conv<<<g, b>>>(dx, dws, dy, cin, cout, tiles, T);
  }
  CK(cudaGetLastError());
  for (int c = 0; c < cout; c++)
    CK(cudaMemcpy(hy3 + c * T, dy + (long long)c * (T + FB_N), sizeof(float) * T,
                  cudaMemcpyDeviceToHost));

  double md1 = 0, md2 = 0, scale = 0;
  for (long long i = 0; i < cout * T; i++) {
    double a = hy1[i];
    if (fabs(a) > scale)
      scale = fabs(a);
    double d1 = fabs(hy2[i] - a), d2 = fabs(hy3[i] - a);
    if (d1 > md1)
      md1 = d1;
    if (d2 > md2)
      md2 = d2;
  }

  /* window kernel vs its own padded reference (runtime conventions) */
  {
    int pad_in = getenv("PADIN") ? atoi(getenv("PADIN")) : (K - 1) / 2;
    CK(cudaMemset(dy, 0, sizeof(float) * cout * (T + FB_N)));
    {
      dim3 b(16, 16);
      dim3 g((unsigned)((T + 15) / 16), (unsigned)((cout + 15) / 16));
      ref_conv_pad<<<g, b>>>(dx, dw, cin, cout, K, pad_in, dy, T, T);
    }
    CK(cudaGetLastError());
    float *hr = new float[cout * T];
    for (int c = 0; c < cout; c++)
      CK(cudaMemcpy(hr + c * T, dy + (long long)c * T, sizeof(float) * T, cudaMemcpyDeviceToHost));
    CK(cudaMemset(dy, 0, sizeof(float) * cout * (T + FB_N)));
    {
      dim3 b(128);
      dim3 g((unsigned)((T + 127) / 128), (unsigned)((cout + 7) / 8), 1);
      win_conv<<<g, b>>>(dx, T, cin, K, pad_in, dy, T, T, cout, dw);
    }
    CK(cudaGetLastError());
    for (int c = 0; c < cout; c++)
      CK(cudaMemcpy(hy3 + c * T, dy + (long long)c * T, sizeof(float) * T, cudaMemcpyDeviceToHost));
    double mw = 0, sg = 0;
    for (long long i = 0; i < cout * T; i++) {
      double d = fabs(hy3[i] - hr[i]);
      if (d > mw) mw = d;
      if (fabs(hr[i]) > sg) sg = fabs(hr[i]);
    }
    if (K == 3 || K == 5 || K == 7 || K == 11) {
      dim3 bw(128);
      dim3 gw((unsigned)((T + 127) / 128), (unsigned)((cout + 7) / 8), 1);
      CK(cudaMemset(dy, 0, sizeof(float) * cout * (T + FB_N)));
#define LAUNCH_WINT                                                        \
      do {                                                                 \
        if (K == 11) winT_conv<11><<<gw, bw>>>(dx, T, cin, pad_in, dy, T, T, cout, dw); \
        else if (K == 7) winT_conv<7><<<gw, bw>>>(dx, T, cin, pad_in, dy, T, T, cout, dw); \
        else if (K == 5) winT_conv<5><<<gw, bw>>>(dx, T, cin, pad_in, dy, T, T, cout, dw); \
        else winT_conv<3><<<gw, bw>>>(dx, T, cin, pad_in, dy, T, T, cout, dw); \
      } while (0)
      LAUNCH_WINT;
      CK(cudaGetLastError());
      for (int c = 0; c < cout; c++)
        CK(cudaMemcpy(hy3 + c * T, dy + (long long)c * T, sizeof(float) * T, cudaMemcpyDeviceToHost));
      double mwt = 0;
      for (long long i = 0; i < cout * T; i++) {
        double d = fabs(hy3[i] - hr[i]);
        if (d > mwt) mwt = d;
      }
      cudaEventRecord(e0);
      for (int i = 0; i < iters; i++) LAUNCH_WINT;
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      cudaEventElapsedTime(&ms, e0, e1);
      printf("winT   %.3f ms  (%.1f GFLOP/s eff)   max|d| winT-vs-padref %.3e\n", ms / iters,
             gflop_direct * 1000 * iters / ms, mwt);
#undef LAUNCH_WINT
    }
    if (K == 11) {
      dim3 bw2(128);
      dim3 gw2((unsigned)((T + 127) / 128), (unsigned)((cout + 7) / 8), 1);
      CK(cudaMemset(dy, 0, sizeof(float) * cout * (T + FB_N)));
      winR_conv<11><<<gw2, bw2>>>(dx, T, cin, dy, T, T, cout, dw, (const float *)0, (int64_t)(cin * T),
                                  (int64_t)(cout * T), 0, 0, 0.f, 0.f, (int64_t) pad_in);
      CK(cudaGetLastError());
      for (int c = 0; c < cout; c++)
        CK(cudaMemcpy(hy3 + c * T, dy + (long long)c * T, sizeof(float) * T, cudaMemcpyDeviceToHost));
      double mwr = 0;
      for (long long i = 0; i < cout * T; i++) {
        double d = fabs(hy3[i] - hr[i]);
        if (d > mwr) mwr = d;
      }
      cudaEventRecord(e0);
      for (int i = 0; i < iters; i++)
        winR_conv<11><<<gw2, bw2>>>(dx, T, cin, dy, T, T, cout, dw, (const float *)0, (int64_t)(cin * T),
                                    (int64_t)(cout * T), 0, 0, 0.f, 0.f, (int64_t) pad_in);
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      cudaEventElapsedTime(&ms, e0, e1);
      printf("winR   %.3f ms  (%.1f GFLOP/s eff)   max|d| winR-vs-padref %.3e\n", ms / iters,
             gflop_direct * 1000 * iters / ms, mwr);
    }
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) {
      dim3 b(128);
      dim3 g((unsigned)((T + 127) / 128), (unsigned)((cout + 7) / 8), 1);
      win_conv<<<g, b>>>(dx, T, cin, K, pad_in, dy, T, T, cout, dw);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    if (getenv("BDBG")) {
      printf("co0: ref ");
      for (int i = 0; i < 10; i++) printf("%.4f ", hr[i]);
      printf("  win ");
      for (int i = 0; i < 10; i++) printf("%.4f ", hy3[i]);
      printf("\nco1: ref ");
      for (int i = 0; i < 10; i++) printf("%.4f ", hr[T + i]);
      printf("  win ");
      for (int i = 0; i < 10; i++) printf("%.4f ", hy3[T + i]);
      printf("\nco8: ref ");
      for (int i = 0; i < 10; i++) printf("%.4f ", hr[8 * T + i]);
      printf("  win ");
      for (int i = 0; i < 10; i++) printf("%.4f ", hy3[8 * T + i]);
      printf("\n");
    }
    printf("win    %.3f ms  (%.1f GFLOP/s eff)   max|d| win-vs-padref %.3e (signal %.3e)\n",
           ms / iters, gflop_direct * 1000 * iters / ms, mw, sg);
    delete[] hr;
  }

  /* timings */
  CK(cudaMemset(dy, 0, sizeof(float) * cout * (T + FB_N)));
  cudaEventRecord(e0);
  for (int i = 0; i < iters; i++) {
    dim3 b(128);
    dim3 g((unsigned)((T + 127) / 128), (unsigned)((cout + 7) / 8));
    direct_conv<<<g, b>>>(dx, dw, dy, cin, cout, K, T);
  }
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  cudaEventElapsedTime(&ms, e0, e1);
  double tdir = ms / iters;

  CK(cudaMemset(dy, 0, sizeof(float) * cout * (T + FB_N)));
  cudaEventRecord(e0);
  for (int i = 0; i < iters; i++) {
    dim3 b(64);
    dim3 g((unsigned)((tiles + 7) / 8), (unsigned)((cout + 7) / 8));
    fft_conv<<<g, b>>>(dx, dws, dy, cin, cout, tiles, T);
  }
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  cudaEventElapsedTime(&ms, e0, e1);
  double tfft = ms / iters;

  printf("shape cin=%d cout=%d K=%d T=%lld tiles=%d iters=%d\n", cin, cout, K, T, tiles, iters);
  printf("direct %.3f ms  (%.1f GFLOP/s eff, %.3f GFLOP)\n", tdir, gflop_direct * 1000 / tdir,
         gflop_direct);
  printf("fft    %.3f ms  (%.1f GFLOP/s eff, %.3f GFLOP)  ratio %.2fx\n", tfft,
         gflop_fft * 1000 / tfft, gflop_fft, tdir / tfft);
  printf("max|d| direct-vs-ref %.3e  fft-vs-ref %.3e  (signal max %.3e)\n", md1, md2, scale);
  return 0;
}
