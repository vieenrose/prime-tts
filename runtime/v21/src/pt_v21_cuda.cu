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
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>

#include "pt_v21.h"
#include <stdlib.h>

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

static double gpu_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double) ts.tv_sec * 1000.0 + 1e-6 * (double) ts.tv_nsec;
}
static int gpu_prof(void) {
  static int v = -1;
  if (v < 0) v = getenv("PT_V21_CUDAPROF") ? 1 : 0;
  return v;
}


/* ------------------------------------------------- resident weights + arena (M3)
 * Weights are the same for every request and every stream, so they are uploaded once per process
 * and shared: a sorted (host ptr -> device ptr) table over the container's fp32 payloads, looked
 * up by bsearch on the pointer the graph hands us. Activations come from a bump arena so the
 * steady state has no cudaMalloc/cudaFree at all (measured: 222 ms alloc + 65 ms free per T81 run,
 * and the per-call weight copy is inside the 153 ms copy split). */
static void **g_wdev;      /* ctab index -> device pointer, read-only across streams */
static int g_wdev_n, g_wref;

static size_t g_wdev_bytes;

static int weights_upload(pt_ctx *c, int verbose) {
  int n = pt_v21_n_cont(c);
  if (++g_wref > 1)
    return 0;   /* already uploaded: the weights are identical for every stream */
  g_wdev = (void **) calloc((size_t) n, sizeof(void *));
  if (!g_wdev) {
    g_wref = 0;
    return -1;
  }
  g_wdev_n = n;
  size_t total = 0;
  for (int i = 0; i < n; i++) {
    const void *h = pt_v21_cont(c, i);
    size_t nb = (size_t) pt_v21_cont_bytes(c, i);
    if (!h || nb == 0)
      continue;
    g_wdev_bytes += nb;
    if (cudaMalloc(&g_wdev[i], nb) != cudaSuccess ||
        cudaMemcpy(g_wdev[i], h, nb, cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "pt_v21 cuda: weight upload failed for ctab %d (%zu B)\n", i, nb);
      g_wref = 0;
      return -1;
    }
    total += nb;
  }
  if (verbose)
    fprintf(stderr, "pt_v21 cuda: %d container records resident on the GPU (%.1f MiB)\n", n,
            (double) total / 1048576.0);
  return 0;
}

int pt_v21_conv_pads_in_kernel(pt_ctx *c) {
  /* only when the GEMM kernel takes *every* conv (PT_V21_IG_MINT unset or 0): the other kernels
   * read the padded row layout */
  (void) c;
  const char *e = getenv("PT_V21_IG_MINT");
  return !e || atol(e) == 0;
}

double pt_v21_cuda_weights_mib(pt_ctx *c) { (void) c; return (double) (g_wdev_bytes >> 20); }

static void *weight_ctab(int idx) {
  if (idx < 0 || idx >= g_wdev_n)
    return NULL;
  return g_wdev[idx];
}

static void weights_release(void) {
  if (--g_wref <= 0) {
    for (int i = 0; i < g_wdev_n; i++)
      if (g_wdev[i])
        cudaFree(g_wdev[i]);
    free(g_wdev);
    g_wdev = NULL;
    g_wdev_n = 0;
    g_wref = 0;
  }
}

/* bump arena with a mark/release discipline: the graph issues conv kernels one at a time, so each
 * call marks the top, allocates what it needs, and releases back to the mark */
static int arena_init(pt_ctx *c, size_t bytes) {
  if (cudaMalloc(&c->g_arena, bytes) != cudaSuccess)
    return -1;
  c->g_arena_bytes = bytes;
  c->dev_bytes_arena = bytes;
  c->dev_bytes_weights = g_wdev_bytes;   /* shared across streams, reported by each */
  c->g_arena_used = 0;
  return 0;
}
static void *arena_alloc(pt_ctx *c, size_t bytes) {
  size_t off = (c->g_arena_used + 255) & ~(size_t) 255;
  if (off + bytes > c->g_arena_bytes)
    return NULL;
  c->g_arena_used = off + bytes;
  if ((size_t) off + bytes > c->dev_bytes_arena)
    c->dev_bytes_arena = (size_t) off + bytes;   /* high-water mark, for the memory report */
  c->arena_alloc_bytes = (size_t) off + bytes;
  return (char *) c->g_arena + off;
}
static void *arena_mark(pt_ctx *c) { return (char *) c->g_arena + c->g_arena_used; }
static void arena_release(pt_ctx *c, void *mark) {
  c->g_arena_used = (size_t) ((char *) mark - (char *) c->g_arena);
}

int pt_v21_cuda_init(pt_ctx *c, int verbose) {
  if (verbose)
    fprintf(stderr, "pt_v21 cuda: device 0 (sm_53)\n");
  /* Yielding sync instead of spin-wait: with N streams on a 4-core A57, N threads spinning in
   * cudaStreamSynchronize steal the cores the other streams need for their host graph work. Must
   * precede context creation; a context that already exists (another stream) keeps its flags. */
  {
    const char *e = getenv("PT_V21_SYNC");
    int mode = e ? atoi(e) : 2;   /* 0 = driver default (spin), 1 = blocking, 2 = yield (measured best) */
    if (mode == 1) cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    else if (mode == 2) cudaSetDeviceFlags(cudaDeviceScheduleYield);
    cudaGetLastError();   /* setting flags on an active context is a harmless error */
  }
  PTCHK(cudaFree(0)); /* force context creation */
  c->use_cuda = 1;
  if (cudaStreamCreateWithFlags((cudaStream_t *) &c->g_stream, cudaStreamNonBlocking)) {
    fprintf(stderr, "pt_v21 cuda: stream create failed\n");
    return -1;
  }
  size_t aw = getenv("PT_V21_ARENA_MB") ? (size_t) atoi(getenv("PT_V21_ARENA_MB")) : 64u;
  if (arena_init(c, aw * 1048576u)) {
    fprintf(stderr, "pt_v21 cuda: arena of %zu MiB failed\n", aw);
    return -1;
  }
  if (weights_upload(c, verbose))
    return -1;
  if (verbose)
    fprintf(stderr, "pt_v21 cuda: resident weights + %zu MiB arena ready\n", aw);
  return 0;
}

void pt_v21_cuda_shutdown(pt_ctx *c) {
  if (c->g_stream) {
    cudaStreamSynchronize((cudaStream_t) c->g_stream);
    cudaStreamDestroy((cudaStream_t) c->g_stream);
    c->g_stream = NULL;
  }
  weights_release();
  if (c->g_arena) {
    cudaFree(c->g_arena);
    c->g_arena = NULL;
    c->g_arena_bytes = c->g_arena_used = 0;
  }
  c->use_cuda = 0;
}

/* ------------------------------------------------------------------- conv1d
 * thread = (row block, output column) -> four channel outputs at one column.
 * x is the zero-padded scratch the host builds: rows of ldx floats, real input starting at
 * column pad, so column (t*stride + k*dil - pad) is a plain offset. */
/* Epilogue fused into the conv kernels: the same fp32 arithmetic the interpreter's elementwise
 * nodes do, applied to the accumulated output before it is written. */
__device__ __forceinline__ float pt_epi_dev(float v, int op, float a) {
  if (op == 1)
    return v >= 0.0f ? v : a * v;   /* LeakyReLU */
  if (op == 2)
    return v > 0.0f ? v : 0.0f;     /* ReLU */
  if (op == 3)
    return v * a;                   /* multiply by scalar */
  if (op == 4)
    return v + a;                   /* add scalar */
  return v;
}

__global__ void pt_conv_kernel(const float *__restrict__ x, int64_t ldx, int cin, int K, int dil,
                               int stride, int pad, float *__restrict__ y, int64_t ldy,
                               int64_t Tout, int cout, const float *__restrict__ wp,
                               const float *__restrict__ bias, int64_t bsx, int64_t bsy, int epi0, int epi1,
                                   float ea0, float ea1) {
  int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
  int blk = (int) blockIdx.y;
  int64_t b = (int64_t) blockIdx.z;
  if (t >= Tout)
    return;
  const float4 *w4 = (const float4 *) wp + (int64_t) blk * K * cin;
  int co0 = blk * 4;
  /* the bias has exactly cout entries, so each lane takes its own and the padded lanes get 0 */
  float4 acc = make_float4(bias && co0 + 0 < cout ? bias[co0 + 0] : 0.f,
                           bias && co0 + 1 < cout ? bias[co0 + 1] : 0.f,
                           bias && co0 + 2 < cout ? bias[co0 + 2] : 0.f,
                           bias && co0 + 3 < cout ? bias[co0 + 3] : 0.f);
  const float *xb = x + b * bsx;
  for (int k = 0; k < K; k++) {
    const float *xs = xb + (int64_t) (k * dil) + t * stride;   /* padded rows: column j is input j-pad */
    for (int ci = 0; ci < cin; ci++) {
      float4 w = w4[(int64_t) k * cin + ci];
      /* one input value, four channel outputs: the lanes are the four Cout channels of this row
       * block, NOT four input columns, so every lane multiplies the same x column */
      float xv = xs[(int64_t) ci * ldx];
      acc.x += w.x * xv;
      acc.y += w.y * xv;
      acc.z += w.z * xv;
      acc.w += w.w * xv;
    }
  }
  if (epi0 || epi1) {
    for (int lane = 0; lane < 4; lane++) {
      float v = (&acc.x)[lane];
      v = pt_epi_dev(v, epi0, ea0);
      v = pt_epi_dev(v, epi1, ea1);
      (&acc.x)[lane] = v;
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

/* Weight-staged conv: the weights for a block's 4 output channels are cooperatively loaded into
 * shared memory once at block start, so a block reads its 12 KB weight slice from memory once
 * instead of every thread re-reading it. The input is still read straight from global in the
 * compute loop - consecutive threads read consecutive columns, so those loads are coalesced and no
 * tile staging or __syncthreads is needed in the inner loop. Summation order and every fp32 product
 * are identical to the reference kernel, so results are bit-identical.
 */
/* Weight-staged conv: the weights for a block's 4 output channels are cooperatively loaded into
 * shared memory once at block start, so a block reads its 12 KB weight slice from memory once
 * instead of every thread re-reading it. The input is still read straight from global in the
 * compute loop - consecutive threads read consecutive columns, so those loads are coalesced and no
 * tile staging or __syncthreads is needed in the inner loop. Summation order and every fp32 product
 * are identical to the reference kernel, so results are bit-identical.
 */
__global__ void pt_conv_kernel_wsm(const float *__restrict__ x, int64_t ldx, int cin, int K,
                                   int dil, int stride, int pad, float *__restrict__ y, int64_t ldy,
                                   int64_t Tout, int cout, const float *__restrict__ wp,
                                   const float *__restrict__ bias, int64_t bsx, int64_t bsy, int epi0, int epi1,
                                   float ea0, float ea1) {
  extern __shared__ float4 wsm[];   /* [k*cin + ci] -> this block's 4 channel lanes */
  const int NT = blockDim.x;
  int64_t t = (int64_t) blockIdx.x * NT + threadIdx.x;
  int blk = (int) blockIdx.y;
  int64_t b = (int64_t) blockIdx.z;
  const int kc = K * cin;
  const float4 *w4 = (const float4 *) wp + (int64_t) blk * kc;
  for (int idx = threadIdx.x; idx < kc; idx += NT)
    wsm[idx] = w4[idx];
  __syncthreads();
  if (t >= Tout)
    return;
  int co0 = blk * 4;
  float4 acc = make_float4(bias && co0 + 0 < cout ? bias[co0 + 0] : 0.f,
                           bias && co0 + 1 < cout ? bias[co0 + 1] : 0.f,
                           bias && co0 + 2 < cout ? bias[co0 + 2] : 0.f,
                           bias && co0 + 3 < cout ? bias[co0 + 3] : 0.f);
  const float *xb = x + b * bsx;
  for (int k = 0; k < K; k++) {
    const float *xs = xb + (int64_t) (k * dil) + t * stride;
    for (int ci = 0; ci < cin; ci++) {
      float4 w = wsm[(int64_t) k * cin + ci];
      float xv = xs[(int64_t) ci * ldx];
      acc.x += w.x * xv;
      acc.y += w.y * xv;
      acc.z += w.z * xv;
      acc.w += w.w * xv;
    }
  }
  if (epi0 || epi1) {
    for (int lane = 0; lane < 4; lane++) {
      float v = (&acc.x)[lane];
      v = pt_epi_dev(v, epi0, ea0);
      v = pt_epi_dev(v, epi1, ea1);
      (&acc.x)[lane] = v;
    }
  }
  float *yb = y + b * bsy;
  if (co0 + 0 < cout)
    yb[(int64_t) (co0 + 0) * ldy + t] = acc.x;
  if (co0 + 1 < cout)
    yb[(int64_t) (co0 + 1) * ldy + t] = acc.y;
  if (co0 + 2 < cout)
    yb[(int64_t) (co0 + 2) * ldy + t] = acc.z;
  if (co0 + 3 < cout)
    yb[(int64_t) (co0 + 3) * ldy + t] = acc.w;
}


/* Weight-staged conv with two output channels per block.
 *
 * The 4-channel blocks above need K*cin*16 bytes of shared for their weight slice, which is 36 KB
 * for the flow-encoder convs (cin=768, K=3): too much to leave room for several blocks per SM on the
 * Nano's 64 KB. Halving the channel group halves the slice (18 KB) at the cost of twice as many
 * blocks. Each lane reads the matching component pair of the packed float4 weight, so every output
 * channel's accumulation chain is exactly the reference chain: bit-identical.
 */

/* Implicit-GEMM conv: block tile = 64 output channels x 64 output columns, 256 threads, each thread
 * owning 4 channels (one packed float4 group) x 4 consecutive columns. The reduction index
 * r = k*cin + ci is walked in chunks of 16 through two small shared tiles (weights [16][16 groups],
 * inputs [16][64 cols]); each input element is fetched from global once per block instead of once
 * per 4-channel block. Every thread accumulates its outputs over r ascending onto the bias - the
 * same chain as the staged kernel - so results are bit-identical. Zero-filled tails add exact +0. */
#define IG_RK 16
__global__ void __launch_bounds__(256) pt_conv_kernel_ig(
    const float *__restrict__ x, int64_t ldx, int cin, int K, int dil, int stride,
    float *__restrict__ y, int64_t ldy, int64_t Tout, int cout, const float *__restrict__ wp,
    const float *__restrict__ bias, int64_t bsx, int64_t bsy, int epi0, int epi1, float ea0,
    float ea1, int64_t pad_in) {
  __shared__ float4 Ws[IG_RK][16];
  __shared__ float4 Xs[IG_RK][16];   /* [r][column quad] */
  const int tid = threadIdx.x;
  const int tx = tid & 15;   /* column quad */
  const int ty = tid >> 4;   /* channel group within the tile */
  const int nblk = (cout + 3) / 4;
  const int g = blockIdx.y * 16 + ty;   /* global channel group */
  const int64_t t0 = (int64_t) blockIdx.x * 64;
  const int64_t b = blockIdx.z;
  const int kc = K * cin;
  const float *xb = x + b * bsx;
  const float4 *w4 = (const float4 *) wp;
  int co0 = g * 4;
  float4 bz = make_float4(bias && co0 + 0 < cout ? bias[co0 + 0] : 0.f,
                          bias && co0 + 1 < cout ? bias[co0 + 1] : 0.f,
                          bias && co0 + 2 < cout ? bias[co0 + 2] : 0.f,
                          bias && co0 + 3 < cout ? bias[co0 + 3] : 0.f);
  float4 a0 = bz, a1 = bz, a2 = bz, a3 = bz;   /* columns 4*tx + 0..3 */
  /* loader roles: weights - thread -> (rr = tid/16, group tx); inputs - thread -> (rr = tid/16,
   * column quad tx), 4 columns each */
  const int lr = tid >> 4, lq = tid & 15;
  for (int r0 = 0; r0 < kc; r0 += IG_RK) {
    int r = r0 + lr;
    int lg = blockIdx.y * 16 + lq;
    Ws[lr][lq] = (r < kc && lg < nblk) ? w4[(int64_t) lg * kc + r] : make_float4(0.f, 0.f, 0.f, 0.f);
    float4 xv = make_float4(0.f, 0.f, 0.f, 0.f);
    if (r < kc) {
      int k = r / cin, ci = r - k * cin;
      int64_t tc = t0 + lq * 4;
      if (pad_in < 0) {
        const float *row = xb + (int64_t) ci * ldx + (int64_t) k * dil;
        if (tc + 0 < Tout) xv.x = row[(tc + 0) * stride];
        if (tc + 1 < Tout) xv.y = row[(tc + 1) * stride];
        if (tc + 2 < Tout) xv.z = row[(tc + 2) * stride];
        if (tc + 3 < Tout) xv.w = row[(tc + 3) * stride];
      } else {
        /* unpadded row of length ldx: positions outside [0, ldx) are the zero padding */
        const float *row = xb + (int64_t) ci * ldx;
        int64_t s0 = tc * stride + (int64_t) k * dil - pad_in;
        if (tc + 0 < Tout && s0 >= 0 && s0 < ldx) xv.x = row[s0];
        int64_t s1 = s0 + stride;
        if (tc + 1 < Tout && s1 >= 0 && s1 < ldx) xv.y = row[s1];
        int64_t s2 = s1 + stride;
        if (tc + 2 < Tout && s2 >= 0 && s2 < ldx) xv.z = row[s2];
        int64_t s3 = s2 + stride;
        if (tc + 3 < Tout && s3 >= 0 && s3 < ldx) xv.w = row[s3];
      }
    }
    Xs[lr][lq] = xv;
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < IG_RK; rr++) {
      float4 w = Ws[rr][ty];
      float4 v = Xs[rr][tx];
      a0.x += w.x * v.x; a0.y += w.y * v.x; a0.z += w.z * v.x; a0.w += w.w * v.x;
      a1.x += w.x * v.y; a1.y += w.y * v.y; a1.z += w.z * v.y; a1.w += w.w * v.y;
      a2.x += w.x * v.z; a2.y += w.y * v.z; a2.z += w.z * v.z; a2.w += w.w * v.z;
      a3.x += w.x * v.w; a3.y += w.y * v.w; a3.z += w.z * v.w; a3.w += w.w * v.w;
    }
    __syncthreads();
  }
  if (g >= nblk)
    return;
  float4 *acc[4] = {&a0, &a1, &a2, &a3};
  float *yb = y + b * bsy;
  for (int j = 0; j < 4; j++) {
    int64_t t = t0 + tx * 4 + j;
    if (t >= Tout)
      continue;
    float4 a = *acc[j];
    if (epi0 || epi1) {
      a.x = pt_epi_dev(pt_epi_dev(a.x, epi0, ea0), epi1, ea1);
      a.y = pt_epi_dev(pt_epi_dev(a.y, epi0, ea0), epi1, ea1);
      a.z = pt_epi_dev(pt_epi_dev(a.z, epi0, ea0), epi1, ea1);
      a.w = pt_epi_dev(pt_epi_dev(a.w, epi0, ea0), epi1, ea1);
    }
    if (co0 + 0 < cout) yb[(int64_t) (co0 + 0) * ldy + t] = a.x;
    if (co0 + 1 < cout) yb[(int64_t) (co0 + 1) * ldy + t] = a.y;
    if (co0 + 2 < cout) yb[(int64_t) (co0 + 2) * ldy + t] = a.z;
    if (co0 + 3 < cout) yb[(int64_t) (co0 + 3) * ldy + t] = a.w;
  }
}


/* GEMM conv with a 4-channel x 8-column register tile per thread: block = 64 channels x 128
 * columns, 256 threads. 32 FMAs per (one weight + two input) 128-bit shared loads, versus 16 per
 * two in pt_conv_kernel_ig. Same per-output chain (r = k*cin + ci ascending onto the bias), same
 * in-kernel padding: bit-identical. */
__global__ void __launch_bounds__(256) pt_conv_kernel_ig8(
    const float *__restrict__ x, int64_t ldx, int cin, int K, int dil, int stride,
    float *__restrict__ y, int64_t ldy, int64_t Tout, int cout, const float *__restrict__ wp,
    const float *__restrict__ bias, int64_t bsx, int64_t bsy, int epi0, int epi1, float ea0,
    float ea1, int64_t pad_in) {
  __shared__ float4 Ws[IG_RK][16];
  __shared__ float4 Xs[IG_RK][32];   /* [r][column quad], 32 quads = 128 columns */
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int nblk = (cout + 3) / 4;
  const int g = blockIdx.y * 16 + ty;
  const int64_t t0 = (int64_t) blockIdx.x * 128;
  const int64_t b = blockIdx.z;
  const int kc = K * cin;
  const float *xb = x + b * bsx;
  const float4 *w4 = (const float4 *) wp;
  int co0 = g * 4;
  float4 bz = make_float4(bias && co0 + 0 < cout ? bias[co0 + 0] : 0.f,
                          bias && co0 + 1 < cout ? bias[co0 + 1] : 0.f,
                          bias && co0 + 2 < cout ? bias[co0 + 2] : 0.f,
                          bias && co0 + 3 < cout ? bias[co0 + 3] : 0.f);
  float4 a[8];
#pragma unroll
  for (int j = 0; j < 8; j++) a[j] = bz;
  const int lr = tid >> 4, lq = tid & 15;
  for (int r0 = 0; r0 < kc; r0 += IG_RK) {
    int r = r0 + lr;
    int lg = blockIdx.y * 16 + lq;
    Ws[lr][lq] = (r < kc && lg < nblk) ? w4[(int64_t) lg * kc + r] : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int h = 0; h < 2; h++) {
      int qq = lq + 16 * h;   /* column quad 0..31 */
      float4 xv = make_float4(0.f, 0.f, 0.f, 0.f);
      if (r < kc) {
        int k = r / cin, ci = r - k * cin;
        int64_t tc = t0 + qq * 4;
        if (pad_in < 0) {
          const float *row = xb + (int64_t) ci * ldx + (int64_t) k * dil;
          if (tc + 0 < Tout) xv.x = row[(tc + 0) * stride];
          if (tc + 1 < Tout) xv.y = row[(tc + 1) * stride];
          if (tc + 2 < Tout) xv.z = row[(tc + 2) * stride];
          if (tc + 3 < Tout) xv.w = row[(tc + 3) * stride];
        } else {
          const float *row = xb + (int64_t) ci * ldx;
          int64_t s0 = tc * stride + (int64_t) k * dil - pad_in;
          if (tc + 0 < Tout && s0 >= 0 && s0 < ldx) xv.x = row[s0];
          int64_t s1 = s0 + stride;
          if (tc + 1 < Tout && s1 >= 0 && s1 < ldx) xv.y = row[s1];
          int64_t s2 = s1 + stride;
          if (tc + 2 < Tout && s2 >= 0 && s2 < ldx) xv.z = row[s2];
          int64_t s3 = s2 + stride;
          if (tc + 3 < Tout && s3 >= 0 && s3 < ldx) xv.w = row[s3];
        }
      }
      Xs[lr][qq] = xv;
    }
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < IG_RK; rr++) {
      float4 w = Ws[rr][ty];
      float4 v0 = Xs[rr][tx];        /* columns 4*tx .. 4*tx+3 */
      float4 v1 = Xs[rr][tx + 16];   /* columns 64+4*tx .. */
      a[0].x += w.x * v0.x; a[0].y += w.y * v0.x; a[0].z += w.z * v0.x; a[0].w += w.w * v0.x;
      a[1].x += w.x * v0.y; a[1].y += w.y * v0.y; a[1].z += w.z * v0.y; a[1].w += w.w * v0.y;
      a[2].x += w.x * v0.z; a[2].y += w.y * v0.z; a[2].z += w.z * v0.z; a[2].w += w.w * v0.z;
      a[3].x += w.x * v0.w; a[3].y += w.y * v0.w; a[3].z += w.z * v0.w; a[3].w += w.w * v0.w;
      a[4].x += w.x * v1.x; a[4].y += w.y * v1.x; a[4].z += w.z * v1.x; a[4].w += w.w * v1.x;
      a[5].x += w.x * v1.y; a[5].y += w.y * v1.y; a[5].z += w.z * v1.y; a[5].w += w.w * v1.y;
      a[6].x += w.x * v1.z; a[6].y += w.y * v1.z; a[6].z += w.z * v1.z; a[6].w += w.w * v1.z;
      a[7].x += w.x * v1.w; a[7].y += w.y * v1.w; a[7].z += w.z * v1.w; a[7].w += w.w * v1.w;
    }
    __syncthreads();
  }
  if (g >= nblk)
    return;
  float *yb = y + b * bsy;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int64_t t = t0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + (j - 4));
    if (t >= Tout)
      continue;
    float4 v = a[j];
    if (epi0 || epi1) {
      v.x = pt_epi_dev(pt_epi_dev(v.x, epi0, ea0), epi1, ea1);
      v.y = pt_epi_dev(pt_epi_dev(v.y, epi0, ea0), epi1, ea1);
      v.z = pt_epi_dev(pt_epi_dev(v.z, epi0, ea0), epi1, ea1);
      v.w = pt_epi_dev(pt_epi_dev(v.w, epi0, ea0), epi1, ea1);
    }
    if (co0 + 0 < cout) yb[(int64_t) (co0 + 0) * ldy + t] = v.x;
    if (co0 + 1 < cout) yb[(int64_t) (co0 + 1) * ldy + t] = v.y;
    if (co0 + 2 < cout) yb[(int64_t) (co0 + 2) * ldy + t] = v.z;
    if (co0 + 3 < cout) yb[(int64_t) (co0 + 3) * ldy + t] = v.w;
  }
}

int pt_v21_conv(pt_ctx *c, const float *X, int64_t ldx, int64_t cin, float *Y, int64_t ldy,
                int64_t n, int cout, int K, int stride, int pad, int dil, const float *wp,
                const float *bias, int64_t batch, int wtab, int btab, int epi0, int epi1, float ea0,
                float ea1) {
  if (!c->use_cuda || !gpu_on("conv"))
    return pt_v21_conv_cpu(c, X, ldx, cin, Y, ldy, n, cout, K, stride, pad, dil, wp, bias, batch,
                           wtab, btab, epi0, epi1, ea0, ea1);
  c->n_conv++;
  int cout4 = (cout + 3) / 4 * 4;
  int64_t Tout = n;   /* graph-computed output length, same convention as the CPU kernel */
  float *dx = NULL, *dw = NULL, *dy = NULL, *db = NULL;
  int dw_own = 0, db_own = 0, dx_own = 0, dy_own = 0;
  size_t xb = (size_t) batch * cin * ldx * 4;
  /* the output buffer is the *padded* row layout [batch][cout4][Tout]: the host trims by copying
   * cout rows at a per-batch stride of cout4*Tout, so the kernel must use that stride too */
  size_t yb = (size_t) batch * cout4 * Tout * 4;
  size_t wb = (size_t) (cout4 / 4) * K * cin * 4 * 4;
  void *mark = arena_mark(c);
  double tA = gpu_prof() ? gpu_now() : 0.0;
  dx = (float *) arena_alloc(c, xb);
  dy = (float *) arena_alloc(c, yb);
  if (gpu_prof()) c->g_alloc_ms += gpu_now() - tA;
  if (!dx || !dy) {
    /* same as the conv path: a fallback allocation must be owned and freed, and dy must exist too */
    if (!dx) {
      PTCHK(cudaMalloc(&dx, xb));
      dx_own = 1;
    }
    if (!dy) {
      PTCHK(cudaMalloc(&dy, yb));
      dy_own = 1;
    }
    c->dev_bytes_stray += xb + yb;
  }   /* arena full: fall back, still correct */
  c->dev_bytes_stray += xb;
  dw = (float *) weight_ctab(wtab);   /* resident weights, shared across streams */
  if (!dw) {
    PTCHK(cudaMalloc(&dw, wb));
    c->dev_bytes_stray += wb;
    PTCHK(cudaMemcpy(dw, wp, wb, cudaMemcpyHostToDevice));
    dw_own = 1;
  }
  tA = gpu_prof() ? gpu_now() : 0.0;
  /* async on the stream: the copy queues behind the previous kernel instead of blocking the host */
  PTCHK(cudaMemcpyAsync(dx, X, xb, cudaMemcpyHostToDevice, (cudaStream_t) c->g_stream));
  if (gpu_prof()) c->g_copy_ms += gpu_now() - tA;
  if (bias) {
    /* the bias is a container record: keep it resident like the weights, so a conv does not pay a
     * cudaMalloc + a synchronous copy for it on every call (124 convs per stream) */
    db = (float *) weight_ctab(btab);
    if (!db) {
      PTCHK(cudaMalloc(&db, (size_t) cout * 4));
      c->dev_bytes_stray += (size_t) cout * 4;
      PTCHK(cudaMemcpy(db, bias, (size_t) cout * 4, cudaMemcpyHostToDevice));
      db_own = 1;
    }
  }
  int threads = 256;
  /* stage the weight slice in shared memory when it is small enough to leave room for a few blocks
   * per SM (the Nano has 64 KB of shared per SM and 2 SMs) */
  size_t wsm_bytes = (size_t) K * cin * sizeof(float4);
  /* shared-memory gate for the staged weight slice: sm_53 allows 48 KB per block */
  static long wsm_gate = -1;
  if (wsm_gate < 0) wsm_gate = getenv("PT_V21_WSM_KB") ? atol(getenv("PT_V21_WSM_KB")) * 1024 : 49152;   /* 48 KB: measured -12% at N=4 vs the old 16 KB gate */
  int wsm = wsm_bytes > 0 && (long) wsm_bytes <= wsm_gate;
  /* Shared memory, not threads, caps residency for fat slices: at >16 KB only one or two blocks fit
   * per SM, so a 256-thread block leaves the SM with 8-16 warps. Bigger blocks for those slices
   * keep the same shared footprint and put more warps in flight. Grid shape only: bit-identical. */
  static int big_threads = -1;
  if (big_threads < 0)
    big_threads = getenv("PT_V21_BIG_THREADS") ? atoi(getenv("PT_V21_BIG_THREADS")) : 512;
  if (wsm && wsm_bytes > 16384 && big_threads > 0)
    threads = big_threads;
  dim3 grid((unsigned) ((Tout + threads - 1) / threads), (unsigned) (cout4 / 4), (unsigned) batch);
  tA = gpu_prof() ? gpu_now() : 0.0;
  static long ig_min = -2;
  if (ig_min == -2) ig_min = getenv("PT_V21_IG_MINT") ? atol(getenv("PT_V21_IG_MINT")) : 0;
  if (ig_min >= 0 && Tout >= ig_min) {
    static long ig8_min = -2;
    if (ig8_min == -2) ig8_min = getenv("PT_V21_IG8_MINT") ? atol(getenv("PT_V21_IG8_MINT")) : 0;
    if (ig8_min >= 0 && Tout >= ig8_min) {
      dim3 g8((unsigned) ((Tout + 127) / 128), (unsigned) ((cout4 / 4 + 15) / 16), (unsigned) batch);
      pt_conv_kernel_ig8<<<g8, 256, 0, (cudaStream_t) c->g_stream>>>(
          dx, ldx, cin, K, dil, stride, dy, /*ldy=*/Tout, /*Tout=*/Tout, cout, dw, db,
          (int64_t) cin * ldx, (int64_t) cout4 * Tout, epi0, epi1, ea0, ea1, c->conv_pad_in);
    } else {
    dim3 gig((unsigned) ((Tout + 63) / 64), (unsigned) ((cout4 / 4 + 15) / 16), (unsigned) batch);
    pt_conv_kernel_ig<<<gig, 256, 0, (cudaStream_t) c->g_stream>>>(
        dx, ldx, cin, K, dil, stride, dy, /*ldy=*/Tout, /*Tout=*/Tout, cout, dw, db,
        (int64_t) cin * ldx, (int64_t) cout4 * Tout, epi0, epi1, ea0, ea1, c->conv_pad_in);
    }
  } else if (wsm) {
    pt_conv_kernel_wsm<<<grid, threads, wsm_bytes, (cudaStream_t) c->g_stream>>>(
        dx, ldx, cin, K, dil, stride, pad, dy, /*ldy=*/Tout, /*Tout=*/Tout, cout, dw, db,
        (int64_t) cin * ldx, (int64_t) cout4 * Tout, epi0, epi1, ea0, ea1);
  } else {
    pt_conv_kernel<<<grid, threads, 0, (cudaStream_t) c->g_stream>>>(dx, ldx, cin, K, dil, stride,
                                                                      pad, dy, /*ldy=*/Tout,
                                                                      /*Tout=*/Tout, cout, dw, db,
                                                                      (int64_t) cin * ldx,
                                                                      (int64_t) cout4 * Tout, epi0,
                                                                      epi1, ea0, ea1);
  }
  /* a launch that fails (e.g. too many registers for the block size) must not pass silently */
  PTCHK(cudaGetLastError());
  PTCHK(cudaMemcpyAsync(Y, dy, yb, cudaMemcpyDeviceToHost, (cudaStream_t) c->g_stream));
  PTCHK(cudaStreamSynchronize((cudaStream_t) c->g_stream));
  if (gpu_prof()) {
    cudaDeviceSynchronize();
    c->g_kernel_ms += gpu_now() - tA;
  }
  tA = gpu_prof() ? gpu_now() : 0.0;
  if (dw_own)
    cudaFree(dw);
  if (db_own)
    cudaFree(db);
  if (dx_own)
    cudaFree(dx);
  if (dy_own)
    cudaFree(dy);
  arena_release(c, mark);   /* weights are resident: only the activation scratch is released */
  if (gpu_prof()) c->g_free_ms += gpu_now() - tA;
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
  int co0 = blk * 4;
  float4 acc = make_float4(bias && co0 + 0 < cout ? bias[co0 + 0] : 0.f,
                           bias && co0 + 1 < cout ? bias[co0 + 1] : 0.f,
                           bias && co0 + 2 < cout ? bias[co0 + 2] : 0.f,
                           bias && co0 + 3 < cout ? bias[co0 + 3] : 0.f);
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
        /* same lane rule as conv: one input column, four channel outputs */
        float xv = xs[(int64_t) ci * ldx];
        acc.x += w.x * xv;
        acc.y += w.y * xv;
        acc.z += w.z * xv;
        acc.w += w.w * xv;
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


/* Weight-staged conv_transpose1d.
 *
 * Each thread of the reference kernel walks every (phase, tap, channel) term for its 4 output
 * channels and re-reads that phase's slice of the weights (ntap*cin float4) from L2 per thread. Here
 * the phase slices are staged cooperatively into shared memory with double buffering, so a block
 * touches each slice in memory once.
 *
 * The staging loop is written so that every thread of the block reaches both __syncthreads()
 * unconditionally: a barrier inside divergent control flow is undefined behaviour, and the first
 * version of this kernel used `continue` to skip the per-phase work, which corrupted a few percent
 * of the outputs deterministically (bit-exact on the fallback and cin=1 nodes, SNDR 6.47 dB on T20).
 *
 * Accumulation order (phase, tap, channel, all ascending) and every fp32 product are unchanged, and
 * the accumulator is seeded with the bias exactly as the reference does, so a correct run is
 * bit-identical.
 */
__global__ void pt_convT_kernel_wsm(const float *__restrict__ x, int64_t ldx, int cin, int64_t tin,
                                    float *__restrict__ y, int64_t ldy, int64_t Lout, int cout, int s,
                                    int pad, int ntap, const float *__restrict__ wp,
                                    const float *__restrict__ bias, int64_t bsx, int64_t bsy,
                                    int dbuf) {
  /* dbuf=1: two phase slices, next phase prefetched while this one computes.
   * dbuf=0: one slice, reloaded per phase behind a barrier - half the shared memory, so the
   * cin=512 upsampler (32 KB per phase) fits under the 48 KB per-block limit. Same arithmetic
   * order either way: bit-identical. */
  extern __shared__ float4 wbuf[];   /* [phase & 1][t*cin + ci], or one slice when dbuf=0 */
  const int NT = blockDim.x;
  const int kc = ntap * cin;
  const int nblk = ((cout + 3) / 4 * 4) / 4;
  int64_t oi = (int64_t) blockIdx.x * NT + threadIdx.x;
  int blk = (int) blockIdx.y;
  int64_t b = (int64_t) blockIdx.z;
  int co0 = blk * 4;
  float4 acc = make_float4(bias && co0 + 0 < cout ? bias[co0 + 0] : 0.f,
                           bias && co0 + 1 < cout ? bias[co0 + 1] : 0.f,
                           bias && co0 + 2 < cout ? bias[co0 + 2] : 0.f,
                           bias && co0 + 3 < cout ? bias[co0 + 3] : 0.f);
  const float *xb = x + b * bsx;
  const float4 *wbase = (const float4 *) wp + (int64_t) blk * kc;
  int live = oi < Lout;
  if (dbuf)
    for (int idx = threadIdx.x; idx < kc; idx += NT)
      wbuf[idx] = wbase[idx];   /* prime phase 0 */
  for (int r = 0; r < s; r++) {
    int cur = dbuf ? (r & 1) : 0;
    if (!dbuf) {
      /* the trailing barrier of the previous phase guarantees nobody still reads the slice */
      const float4 *nx = wbase + (int64_t) r * nblk * kc;
      for (int idx = threadIdx.x; idx < kc; idx += NT)
        wbuf[idx] = nx[idx];
    } else if (r + 1 < s) {
      const float4 *nx = wbase + (int64_t) (r + 1) * nblk * kc;
      for (int idx = threadIdx.x; idx < kc; idx += NT)
        wbuf[(cur ^ 1) * kc + idx] = nx[idx];
    }
    __syncthreads();
    int64_t num = oi + pad - r;
    if (live && num >= 0 && num % s == 0) {
      int64_t q = num / s;
      const float4 *ws = wbuf + cur * kc;
      for (int t = 0; t < ntap; t++) {
        int64_t tq = q - t;
        if (tq < 0 || tq >= tin)
          continue;
        const float4 *wt = ws + (int64_t) t * cin;
        const float *xs = xb + tq;
        for (int ci = 0; ci < cin; ci++) {
          float4 w = wt[ci];
          float xv = xs[(int64_t) ci * ldx];
          acc.x += w.x * xv;
          acc.y += w.y * xv;
          acc.z += w.z * xv;
          acc.w += w.w * xv;
        }
      }
    }
    __syncthreads();
  }
  if (!live)
    return;
  float *yb = y + b * bsy;
  if (co0 + 0 < cout)
    yb[(int64_t) (co0 + 0) * ldy + oi] = acc.x;
  if (co0 + 1 < cout)
    yb[(int64_t) (co0 + 1) * ldy + oi] = acc.y;
  if (co0 + 2 < cout)
    yb[(int64_t) (co0 + 2) * ldy + oi] = acc.z;
  if (co0 + 3 < cout)
    yb[(int64_t) (co0 + 3) * ldy + oi] = acc.w;
}


/* Phase-major conv-transpose: blockIdx.z = batch*s + phase, and thread i of the block computes the
 * output oi = q*s + r - pad of that phase. The time-major kernel above lets every block cover all s
 * phases and keeps only the threads whose output belongs to the current phase busy, so (s-1)/s of
 * the threads idle in every phase (3/4 for the s=4 upsamplers). Here no thread idles, and a block
 * stages only its own phase slice, once. Each output still sums taps t ascending and channels ci
 * ascending onto the bias: the reference chain, bit-identical. */
__global__ void pt_convT_kernel_ph(const float *__restrict__ x, int64_t ldx, int cin, int64_t tin,
                                   float *__restrict__ y, int64_t ldy, int64_t Lout, int cout, int s,
                                   int pad, int ntap, const float *__restrict__ wp,
                                   const float *__restrict__ bias, int64_t bsx, int64_t bsy) {
  extern __shared__ float4 wph[];
  const int NT = blockDim.x;
  const int kc = ntap * cin;
  const int nblk = ((cout + 3) / 4 * 4) / 4;
  int r = (int) (blockIdx.z % s);
  int64_t b = (int64_t) (blockIdx.z / s);
  int blk = (int) blockIdx.y;
  const float4 *wsrc = (const float4 *) wp + (int64_t) blk * kc + (int64_t) r * nblk * kc;
  for (int idx = threadIdx.x; idx < kc; idx += NT)
    wph[idx] = wsrc[idx];
  __syncthreads();
  /* first q whose output index is >= 0 */
  int64_t q0 = (pad - r + s - 1) / s;
  if (pad - r < 0) q0 = 0;
  int64_t q = q0 + (int64_t) blockIdx.x * NT + threadIdx.x;
  int64_t oi = q * s + r - pad;
  if (oi < 0 || oi >= Lout)
    return;
  int co0 = blk * 4;
  float4 acc = make_float4(bias && co0 + 0 < cout ? bias[co0 + 0] : 0.f,
                           bias && co0 + 1 < cout ? bias[co0 + 1] : 0.f,
                           bias && co0 + 2 < cout ? bias[co0 + 2] : 0.f,
                           bias && co0 + 3 < cout ? bias[co0 + 3] : 0.f);
  const float *xb = x + b * bsx;
  for (int t = 0; t < ntap; t++) {
    int64_t tq = q - t;
    if (tq < 0 || tq >= tin)
      continue;
    const float4 *wt = wph + (int64_t) t * cin;
    const float *xs = xb + tq;
    for (int ci = 0; ci < cin; ci++) {
      float4 w = wt[ci];
      float xv = xs[(int64_t) ci * ldx];
      acc.x += w.x * xv;
      acc.y += w.y * xv;
      acc.z += w.z * xv;
      acc.w += w.w * xv;
    }
  }
  float *yb = y + b * bsy;
  if (co0 + 0 < cout)
    yb[(int64_t) (co0 + 0) * ldy + oi] = acc.x;
  if (co0 + 1 < cout)
    yb[(int64_t) (co0 + 1) * ldy + oi] = acc.y;
  if (co0 + 2 < cout)
    yb[(int64_t) (co0 + 2) * ldy + oi] = acc.z;
  if (co0 + 3 < cout)
    yb[(int64_t) (co0 + 3) * ldy + oi] = acc.w;
}


/* GEMM-tiled phase-major conv-transpose: the same tiling as pt_conv_kernel_ig, applied to one
 * phase r (blockIdx.z = batch*s + r), where output q of the phase is sum_t sum_ci
 * W[r][t][ci] x[ci][q - t]. Each thread walks idx = t*cin + ci ascending onto its bias, the chain
 * of pt_convT_kernel_ph: bit-identical. Out-of-range taps contribute exact zeros. */
__global__ void __launch_bounds__(256) pt_convT_kernel_ig(
    const float *__restrict__ x, int64_t ldx, int cin, int64_t tin, float *__restrict__ y,
    int64_t ldy, int64_t Lout, int cout, int s, int pad, int ntap, const float *__restrict__ wp,
    const float *__restrict__ bias, int64_t bsx, int64_t bsy, int64_t q0) {
  __shared__ float4 Ws[IG_RK][16];
  __shared__ float4 Xs[IG_RK][16];
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int nblk = (cout + 3) / 4;
  const int r = (int) (blockIdx.z % s);
  const int64_t b = (int64_t) (blockIdx.z / s);
  const int g = blockIdx.y * 16 + ty;
  const int64_t qb = q0 + (int64_t) blockIdx.x * 64;
  const int kc = ntap * cin;
  const float *xb = x + b * bsx;
  const float4 *w4 = (const float4 *) wp + (int64_t) r * nblk * kc;
  int co0 = g * 4;
  float4 bz = make_float4(bias && co0 + 0 < cout ? bias[co0 + 0] : 0.f,
                          bias && co0 + 1 < cout ? bias[co0 + 1] : 0.f,
                          bias && co0 + 2 < cout ? bias[co0 + 2] : 0.f,
                          bias && co0 + 3 < cout ? bias[co0 + 3] : 0.f);
  float4 a0 = bz, a1 = bz, a2 = bz, a3 = bz;
  const int lr = tid >> 4, lq = tid & 15;
  for (int i0 = 0; i0 < kc; i0 += IG_RK) {
    int i = i0 + lr;
    int lg = blockIdx.y * 16 + lq;
    Ws[lr][lq] = (i < kc && lg < nblk) ? w4[(int64_t) lg * kc + i] : make_float4(0.f, 0.f, 0.f, 0.f);
    float4 xv = make_float4(0.f, 0.f, 0.f, 0.f);
    if (i < kc) {
      int t = i / cin, ci = i - t * cin;
      const float *row = xb + (int64_t) ci * ldx;
      int64_t qc = qb + lq * 4 - t;
      if (qc + 0 >= 0 && qc + 0 < tin) xv.x = row[qc + 0];
      if (qc + 1 >= 0 && qc + 1 < tin) xv.y = row[qc + 1];
      if (qc + 2 >= 0 && qc + 2 < tin) xv.z = row[qc + 2];
      if (qc + 3 >= 0 && qc + 3 < tin) xv.w = row[qc + 3];
    }
    Xs[lr][lq] = xv;
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < IG_RK; rr++) {
      float4 w = Ws[rr][ty];
      float4 v = Xs[rr][tx];
      a0.x += w.x * v.x; a0.y += w.y * v.x; a0.z += w.z * v.x; a0.w += w.w * v.x;
      a1.x += w.x * v.y; a1.y += w.y * v.y; a1.z += w.z * v.y; a1.w += w.w * v.y;
      a2.x += w.x * v.z; a2.y += w.y * v.z; a2.z += w.z * v.z; a2.w += w.w * v.z;
      a3.x += w.x * v.w; a3.y += w.y * v.w; a3.z += w.z * v.w; a3.w += w.w * v.w;
    }
    __syncthreads();
  }
  if (g >= nblk)
    return;
  float4 *acc[4] = {&a0, &a1, &a2, &a3};
  float *yb = y + b * bsy;
  for (int j = 0; j < 4; j++) {
    int64_t q = qb + tx * 4 + j;
    int64_t oi = q * s + r - pad;
    if (oi < 0 || oi >= Lout)
      continue;
    float4 a = *acc[j];
    if (co0 + 0 < cout) yb[(int64_t) (co0 + 0) * ldy + oi] = a.x;
    if (co0 + 1 < cout) yb[(int64_t) (co0 + 1) * ldy + oi] = a.y;
    if (co0 + 2 < cout) yb[(int64_t) (co0 + 2) * ldy + oi] = a.z;
    if (co0 + 3 < cout) yb[(int64_t) (co0 + 3) * ldy + oi] = a.w;
  }
}

int pt_v21_convT(pt_ctx *c, const float *X, int64_t ldx, int64_t cin, int64_t tin, float *Y,
                 int64_t ldy, int cout, int s, int pad, int K, const float *wp, const float *bias,
                 int64_t batch, int wtab, int btab) {
  if (!c->use_cuda || !gpu_on("convT"))
    return pt_v21_convT_cpu(c, X, ldx, cin, tin, Y, ldy, cout, s, pad, K, wp, bias, batch, wtab,
                            btab);
  c->n_convT++;
  int ntap = (K + s - 1) / s;
  int cout4 = (cout + 3) / 4 * 4;
  int64_t Lout = (tin - 1) * s - 2 * pad + K;
  float *dx = NULL, *dw = NULL, *dy = NULL, *db = NULL;
  int dw_own = 0, db_own = 0, dx_own = 0, dy_own = 0;
  size_t xb = (size_t) batch * cin * ldx * 4;
  size_t yb = (size_t) batch * cout4 * Lout * 4;   /* padded rows, matching the host's trim stride */
  size_t wb = (size_t) s * (cout4 / 4) * ntap * cin * 4 * 4;
  void *mark = arena_mark(c);
  dx = (float *) arena_alloc(c, xb);
  dy = (float *) arena_alloc(c, yb);
  if (!dx || !dy) {
    /* same as the conv path: a fallback allocation must be owned and freed, and dy must exist too */
    if (!dx) {
      PTCHK(cudaMalloc(&dx, xb));
      dx_own = 1;
    }
    if (!dy) {
      PTCHK(cudaMalloc(&dy, yb));
      dy_own = 1;
    }
    c->dev_bytes_stray += xb + yb;
  }
  dw = (float *) weight_ctab(wtab);
  if (!dw) {
    PTCHK(cudaMalloc(&dw, wb));
    c->dev_bytes_stray += wb;
    PTCHK(cudaMemcpy(dw, wp, wb, cudaMemcpyHostToDevice));
    dw_own = 1;
  }
  PTCHK(cudaMemcpyAsync(dx, X, xb, cudaMemcpyHostToDevice, (cudaStream_t) c->g_stream));
  if (bias) {
    db = (float *) weight_ctab(btab);
    if (!db) {
      PTCHK(cudaMalloc(&db, (size_t) cout * 4));
      c->dev_bytes_stray += (size_t) cout * 4;
      PTCHK(cudaMemcpy(db, bias, (size_t) cout * 4, cudaMemcpyHostToDevice));
      db_own = 1;
    }
  }
  int threads = 256;
  /* prefer the double-buffered slice; fall back to one slice per phase when two do not fit */
  const char *gk = getenv("PT_V21_CONVT_KB");
  size_t gate = gk ? (size_t) atol(gk) * 1024 : 49152;
  size_t one = (size_t) ntap * cin * sizeof(float4);
  int dbuf = getenv("PT_V21_CONVT_SB") ? 0 : 1;
  if (2 * one > gate) dbuf = 0;
  size_t wbuf_bytes = dbuf ? 2 * one : one;
  int wsm = one > 0 && wbuf_bytes <= gate && getenv("PT_V21_CONVT_WSM") == NULL;
  dim3 grid((unsigned) ((Lout + threads - 1) / threads), (unsigned) (cout4 / 4), (unsigned) batch);
  int ph = getenv("PT_V21_CONVT_PH") ? atoi(getenv("PT_V21_CONVT_PH")) : 1;
  static int tig = -1;
  if (tig < 0) tig = getenv("PT_V21_CONVT_IG") ? atoi(getenv("PT_V21_CONVT_IG")) : 1;
  if (tig) {
    /* q runs over every q whose output index can land in [0, Lout) for some phase */
    int64_t qlo = (pad - (s - 1) + s - 1) / s;   /* ceil((pad - (s-1)) / s), clamp below */
    if (pad - (s - 1) <= 0) qlo = 0;
    int64_t qhi = (Lout - 1 + pad) / s;          /* inclusive */
    int64_t nq = qhi - qlo + 1;
    dim3 gig((unsigned) ((nq + 63) / 64), (unsigned) ((cout4 / 4 + 15) / 16),
             (unsigned) (batch * s));
    pt_convT_kernel_ig<<<gig, 256, 0, (cudaStream_t) c->g_stream>>>(
        dx, ldx, cin, tin, dy, /*ldy=*/Lout, /*Lout=*/Lout, cout, s, pad, ntap, dw, db,
        (int64_t) cin * ldx, (int64_t) cout4 * Lout, qlo);
  } else if (wsm && ph) {
    /* outputs per phase: at most ceil(Lout/s) + 1 */
    int64_t nq = (Lout + pad) / s + 2;
    dim3 gph((unsigned) ((nq + threads - 1) / threads), (unsigned) (cout4 / 4),
             (unsigned) (batch * s));
    pt_convT_kernel_ph<<<gph, threads, one, (cudaStream_t) c->g_stream>>>(
        dx, ldx, cin, tin, dy, /*ldy=*/Lout, /*Lout=*/Lout, cout, s, pad, ntap, dw, db,
        (int64_t) cin * ldx, (int64_t) cout4 * Lout);
  } else if (wsm) {
    pt_convT_kernel_wsm<<<grid, threads, wbuf_bytes, (cudaStream_t) c->g_stream>>>(
        dx, ldx, cin, tin, dy, /*ldy=*/Lout, /*Lout=*/Lout, cout, s, pad, ntap, dw, db,
        (int64_t) cin * ldx, (int64_t) cout4 * Lout, dbuf);
  } else {
    pt_convT_kernel<<<grid, threads, 0, (cudaStream_t) c->g_stream>>>(dx, ldx, cin, tin, dy,
                                                                       /*ldy=*/Lout, /*Lout=*/Lout,
                                                                       cout, s, pad, ntap, dw, db,
                                                                       (int64_t) cin * ldx,
                                                                       (int64_t) cout4 * Lout);
  }
  PTCHK(cudaMemcpyAsync(Y, dy, yb, cudaMemcpyDeviceToHost, (cudaStream_t) c->g_stream));
  PTCHK(cudaStreamSynchronize((cudaStream_t) c->g_stream));
  if (dw_own)
    cudaFree(dw);
  if (db_own)
    cudaFree(db);
  if (dx_own)
    cudaFree(dx);
  if (dy_own)
    cudaFree(dy);
  arena_release(c, mark);
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
  float *dA = NULL, *dB = NULL, *dC = NULL;
  size_t ab = (size_t) m * lda * 4;
  size_t bb = (size_t) k * ldb * 4;
  size_t cb = (size_t) m * ldc * 4;
  void *mark = arena_mark(c);
  dA = (float *) arena_alloc(c, ab);
  dB = (float *) arena_alloc(c, bb);
  dC = (float *) arena_alloc(c, cb);
  int d_own = 0;
  if (!dA || !dB || !dC) {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    PTCHK(cudaMalloc(&dA, ab));
    PTCHK(cudaMalloc(&dB, bb));
    PTCHK(cudaMalloc(&dC, cb));
    d_own = 1;
    c->dev_bytes_stray += ab + bb + cb;
  }
  PTCHK(cudaMemcpyAsync(dA, A, ab, cudaMemcpyHostToDevice, (cudaStream_t) c->g_stream));
  PTCHK(cudaMemcpyAsync(dB, B, bb, cudaMemcpyHostToDevice, (cudaStream_t) c->g_stream));
  int tx = 32, ty = 8;
  dim3 block(tx, ty);
  dim3 grid((unsigned) ((nn / 4 + tx - 1) / tx), (unsigned) ((m + ty - 1) / ty));
  pt_matmul_kernel<<<grid, block, 0, (cudaStream_t) c->g_stream>>>(dA, m, k, lda, dB, ldb, nn, dC,
                                                                    ldc);
  PTCHK(cudaMemcpyAsync(C, dC, cb, cudaMemcpyDeviceToHost, (cudaStream_t) c->g_stream));
  PTCHK(cudaStreamSynchronize((cudaStream_t) c->g_stream));
  if (d_own) {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
  }
  arena_release(c, mark);
  return 0;
}
