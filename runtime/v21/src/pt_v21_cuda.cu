/* pt_v21_cuda.cu -- conv / conv-transpose / matmul for v2.1 on the Jetson Nano (sm_53).
 *
 * Same packed layouts as the host path and the numpy oracle (docs/v21_layers.md):
 *   conv   W[cout4/4][K][cin][4]  X[batch][cin][ldx]  Y[batch][cout4][ldy]
 *   convT  W[s][cout4/4][ntap][cin][4],  X[batch][cin][tin]  Y[batch][cout4][Lout]
 * A thread owns one row-block and one 4-wide slot, i.e. four channels per lane -- the same
 * decomposition the aarch64 NEON kernels use, so both paths consume identical tensors.
 *
 * M2 shape is deliberately the simple one: cudaMalloc per call and activations round-tripping over
 * PCIe. It proves the kernels and gives a GPU-time number; the resident arena, CUDA Graphs and
 * 4-stream batching (M3) change the plumbing, not the kernels.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>

#include "pt_v21.h"

#define PTCHK(x)                                                                                 \
  do {                                                                                           \
    cudaError_t _e = (x);                                                                        \
    if (_e != cudaSuccess) {                                                                     \
      fprintf(stderr, "pt_v21 cuda: %s at %s:%d\n", cudaGetErrorString(_e), __FILE__,           \
              __LINE__);                                                                         \
      return -1;                                                                                 \
    }                                                                                            \
  } while (0)

/* Per-opeswitches for bisection: PT_V21_GPU=conv,convT,matmul (all by default). */
static int gpu_on(const char *name) {
  const char *e = getenv("PT_V21_GPU");
  if (!e || !*e)
    return 1;
  return strstr(e, name) != NULL;
}

int pt_v21_cuda_init(pt_ctx *c, int verbose) {
  if (verbose)
    fprintf(stderr, "pt_v21 cuda: device 0 (sm_53)\n");
  PTCHK(cudaFree(0)); /* force context creation */
  c->use_cuda = 1;
  return 0;
}

void pt_v21_cuda_shutdown(pt_ctx *c) { c->use_cuda = 0; }

/* ------------------------------------------------------------------- conv1d
 * thread = (row block, output column) -> four channel outputs at one column.
 * x is the zero-padded scratch the host builds: rows of ldx floats, real input starting at
 * column pad, so column (t*stride + k*dil - pad) is a plain offset. */
__global__ void pt_conv_kernel(const float *__restrict__ x, int64_t ldx, int cin, int K, int dil,
                               int stride, int pad, float *__restrict__ y, int64_t ldy,
                               int64_t Tout, int cout, const float *__restrict__ wp,
                               const float *__restrict__ bias, int64_t bsx, int64_t bsy) {
  int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
  int blk = (int) blockIdx.y;
  int64_t b = (int64_t) blockIdx.z;
  if (t >= Tout)
    return;
  const float4 *w4 = (const float4 *) wp + (int64_t) blk * K * cin;
  float4 acc = bias ? ((const float4 *) bias)[blk] : make_float4(0.f, 0.f, 0.f, 0.f);
  const float *xb = x + b * bsx;
  for (int k = 0; k < K; k++) {
    const float *xs = xb + (int64_t) (k * dil) + t * stride;   /* padded rows: column j is input j-pad */
    for (int ci = 0; ci < cin; ci++) {
      float4 w = w4[(int64_t) k * cin + ci];
      const float *r = xs + (int64_t) ci * ldx;
      acc.x += w.x * r[0];
      acc.y += w.y * r[1];
      acc.z += w.z * r[2];
      acc.w += w.w * r[3];
    }
  }
  float *yb = y + b * bsy;
  int co = blk * 4;
  if (co + 0 < cout)
    yb[(int64_t) (co + 0) * ldy + t] = acc.x;
  if (co + 1 < cout)
    yb[(int64_t) (co + 1) * ldy + t] = acc.y;
  if (co + 2 < cout)
    yb[(int64_t) (co + 2) * ldy + t] = acc.z;
  if (co + 3 < cout)
    yb[(int64_t) (co + 3) * ldy + t] = acc.w;
}

int pt_v21_conv(pt_ctx *c, const float *X, int64_t ldx, int64_t cin, float *Y, int64_t ldy,
                int64_t n, int cout, int K, int stride, int pad, int dil, const float *wp,
                const float *bias, int64_t batch) {
  if (!c->use_cuda || !gpu_on("conv"))
    return pt_v21_conv_cpu(c, X, ldx, cin, Y, ldy, n, cout, K, stride, pad, dil, wp, bias, batch);
  c->n_conv++;
  int cout4 = (cout + 3) / 4 * 4;
  int64_t Tout = n;   /* graph-computed output length, same convention as the CPU kernel */
  float *dx, *dw, *dy, *db = NULL;
  size_t xb = (size_t) batch * cin * ldx * 4;
  size_t yb = (size_t) batch * cout * Tout * 4;
  size_t wb = (size_t) (cout4 / 4) * K * cin * 4 * 4;
  PTCHK(cudaMalloc(&dx, xb));
  PTCHK(cudaMalloc(&dw, wb));
  PTCHK(cudaMalloc(&dy, yb));
  PTCHK(cudaMemcpy(dx, X, xb, cudaMemcpyHostToDevice));
  PTCHK(cudaMemcpy(dw, wp, wb, cudaMemcpyHostToDevice));
  if (bias) {
    PTCHK(cudaMalloc(&db, 16));
    PTCHK(cudaMemcpy(db, bias, 16, cudaMemcpyHostToDevice));
  }
  int threads = 256;
  dim3 grid((unsigned) ((Tout + threads - 1) / threads), (unsigned) (cout4 / 4), (unsigned) batch);
  pt_conv_kernel<<<grid, threads>>>(dx, ldx, cin, K, dil, stride, pad, dy, /*ldy=*/Tout,
                                    /*Tout=*/Tout, cout, dw, db, (int64_t) cin * ldx,
                                    (int64_t) cout * Tout);
  PTCHK(cudaMemcpy(Y, dy, yb, cudaMemcpyDeviceToHost));
  cudaFree(dx);
  cudaFree(dw);
  cudaFree(dy);
  if (db)
    cudaFree(db);
  return 0;
}

/* -------------------------------------------------------- conv_transpose1d
 * Out[co][oi] = bias[co] + sum over (phase r, tap t, channel ci) of W[r][blk][t][ci][p] *
 * X[ci][q - t], where q is the input step feeding output column oi: q*s + r - pad == oi.
 * One thread per (row block, output column); threads read every tap, so no atomics. */
__global__ void pt_convT_kernel(const float *__restrict__ x, int64_t ldx, int cin, int64_t tin,
                                float *__restrict__ y, int64_t ldy, int64_t Lout, int cout, int s,
                                int pad, int ntap, const float *__restrict__ wp,
                                const float *__restrict__ bias, int64_t bsx, int64_t bsy) {
  int64_t oi = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
  int blk = (int) blockIdx.y;
  int64_t b = (int64_t) blockIdx.z;
  if (oi >= Lout)
    return;
  int cout4 = (cout + 3) / 4 * 4;
  float4 acc = bias ? ((const float4 *) bias)[blk] : make_float4(0.f, 0.f, 0.f, 0.f);
  const float *xb = x + b * bsx;
  for (int r = 0; r < s; r++) {
    int64_t num = oi + pad - r;
    if (num < 0 || num % s)
      continue;
    int64_t q = num / s;
    const float4 *w4 = (const float4 *) wp + ((int64_t) r * (cout4 / 4) + blk) * ntap * cin;
    for (int t = 0; t < ntap; t++) {
      int64_t tq = q - t;
      if (tq < 0 || tq >= tin)
        continue;
      const float4 *wt = w4 + (int64_t) t * cin;
      const float *xs = xb + tq;
      for (int ci = 0; ci < cin; ci++) {
        float4 w = wt[ci];
        const float *r0 = xs + (int64_t) ci * ldx;
        acc.x += w.x * r0[0];
        acc.y += w.y * r0[1];
        acc.z += w.z * r0[2];
        acc.w += w.w * r0[3];
      }
    }
  }
  float *yb = y + b * bsy;
  int co = blk * 4;
  if (co + 0 < cout)
    yb[(int64_t) (co + 0) * ldy + oi] = acc.x;
  if (co + 1 < cout)
    yb[(int64_t) (co + 1) * ldy + oi] = acc.y;
  if (co + 2 < cout)
    yb[(int64_t) (co + 2) * ldy + oi] = acc.z;
  if (co + 3 < cout)
    yb[(int64_t) (co + 3) * ldy + oi] = acc.w;
}

int pt_v21_convT(pt_ctx *c, const float *X, int64_t ldx, int64_t cin, int64_t tin, float *Y,
                 int64_t ldy, int cout, int s, int pad, int K, const float *wp, const float *bias,
                 int64_t batch) {
  if (!c->use_cuda || !gpu_on("convT"))
    return pt_v21_convT_cpu(c, X, ldx, cin, tin, Y, ldy, cout, s, pad, K, wp, bias, batch);
  c->n_convT++;
  int ntap = (K + s - 1) / s;
  int cout4 = (cout + 3) / 4 * 4;
  int64_t Lout = (tin - 1) * s - 2 * pad + K;
  float *dx, *dw, *dy, *db = NULL;
  size_t xb = (size_t) batch * cin * ldx * 4;
  size_t yb = (size_t) batch * cout * Lout * 4;
  size_t wb = (size_t) s * (cout4 / 4) * ntap * cin * 4 * 4;
  PTCHK(cudaMalloc(&dx, xb));
  PTCHK(cudaMalloc(&dw, wb));
  PTCHK(cudaMalloc(&dy, yb));
  PTCHK(cudaMemcpy(dx, X, xb, cudaMemcpyHostToDevice));
  PTCHK(cudaMemcpy(dw, wp, wb, cudaMemcpyHostToDevice));
  if (bias) {
    PTCHK(cudaMalloc(&db, 16));
    PTCHK(cudaMemcpy(db, bias, 16, cudaMemcpyHostToDevice));
  }
  int threads = 256;
  dim3 grid((unsigned) ((Lout + threads - 1) / threads), (unsigned) (cout4 / 4), (unsigned) batch);
  pt_convT_kernel<<<grid, threads>>>(dx, ldx, cin, tin, dy, /*ldy=*/Lout, /*Lout=*/Lout, cout, s,
                                     pad, ntap, dw, db, (int64_t) cin * ldx,
                                     (int64_t) cout * Lout);
  PTCHK(cudaMemcpy(Y, dy, yb, cudaMemcpyDeviceToHost));
  cudaFree(dx);
  cudaFree(dw);
  cudaFree(dy);
  if (db)
    cudaFree(db);
  return 0;
}

/* ------------------------------------------------------------------- matmul
 * thread = (row i, four columns of C); A row-major [m][lda], B row-major [k][ldb]. */
__global__ void pt_matmul_kernel(const float *__restrict__ A, int64_t m, int64_t k, int64_t lda,
                                 const float *__restrict__ B, int64_t ldb, int64_t nn,
                                 float *__restrict__ C, int64_t ldc) {
  int64_t i = (int64_t) blockIdx.y * blockDim.y + threadIdx.y;
  int64_t j4 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m)
    return;
  int64_t j0 = j4 * 4;
  if (j0 >= nn)
    return;
  const float *ar = A + i * lda;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int64_t p = 0; p < k; p++) {
    float a = ar[p];
    const float *bp = B + (int64_t) p * ldb + j0;
    acc.x += a * bp[0];
    if (j0 + 1 < nn)
      acc.y += a * bp[1];
    if (j0 + 2 < nn)
      acc.z += a * bp[2];
    if (j0 + 3 < nn)
      acc.w += a * bp[3];
  }
  float *cr = C + i * ldc + j0;
  cr[0] = acc.x;
  if (j0 + 1 < nn)
    cr[1] = acc.y;
  if (j0 + 2 < nn)
    cr[2] = acc.z;
  if (j0 + 3 < nn)
    cr[3] = acc.w;
}

int pt_v21_matmul(pt_ctx *c, const float *A, int64_t m, int64_t k, int64_t lda, const float *B,
                  int64_t ldb, int64_t nn, float *C, int64_t ldc) {
  if (!c->use_cuda || !gpu_on("matmul"))
    return pt_v21_matmul_cpu(c, A, m, k, lda, B, ldb, nn, C, ldc);
  c->n_matmul++;
  float *dA, *dB, *dC;
  size_t ab = (size_t) m * lda * 4;
  size_t bb = (size_t) k * ldb * 4;
  size_t cb = (size_t) m * ldc * 4;
  PTCHK(cudaMalloc(&dA, ab));
  PTCHK(cudaMalloc(&dB, bb));
  PTCHK(cudaMalloc(&dC, cb));
  PTCHK(cudaMemcpy(dA, A, ab, cudaMemcpyHostToDevice));
  PTCHK(cudaMemcpy(dB, B, bb, cudaMemcpyHostToDevice));
  int tx = 32, ty = 8;
  dim3 block(tx, ty);
  dim3 grid((unsigned) ((nn / 4 + tx - 1) / tx), (unsigned) ((m + ty - 1) / ty));
  pt_matmul_kernel<<<grid, block>>>(dA, m, k, lda, dB, ldb, nn, dC, ldc);
  PTCHK(cudaMemcpy(C, dC, cb, cudaMemcpyDeviceToHost));
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}
