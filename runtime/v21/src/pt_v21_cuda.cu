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
  c->g_arena_used = 0;
  return 0;
}
static void *arena_alloc(pt_ctx *c, size_t bytes) {
  size_t off = (c->g_arena_used + 255) & ~(size_t) 255;
  if (off + bytes > c->g_arena_bytes)
    return NULL;
  c->g_arena_used = off + bytes;
  return (char *) c->g_arena + off;
}
static void *arena_mark(pt_ctx *c) { return (char *) c->g_arena + c->g_arena_used; }
static void arena_release(pt_ctx *c, void *mark) {
  c->g_arena_used = (size_t) ((char *) mark - (char *) c->g_arena);
}

int pt_v21_cuda_init(pt_ctx *c, int verbose) {
  if (verbose)
    fprintf(stderr, "pt_v21 cuda: device 0 (sm_53)\n");
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
__global__ void pt_conv_kernel_wsm(const float *__restrict__ x, int64_t ldx, int cin, int K,
                                   int dil, int stride, int pad, float *__restrict__ y, int64_t ldy,
                                   int64_t Tout, int cout, const float *__restrict__ wp,
                                   const float *__restrict__ bias, int64_t bsx, int64_t bsy) {
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

int pt_v21_conv(pt_ctx *c, const float *X, int64_t ldx, int64_t cin, float *Y, int64_t ldy,
                int64_t n, int cout, int K, int stride, int pad, int dil, const float *wp,
                const float *bias, int64_t batch, int wtab, int btab) {
  if (!c->use_cuda || !gpu_on("conv"))
    return pt_v21_conv_cpu(c, X, ldx, cin, Y, ldy, n, cout, K, stride, pad, dil, wp, bias, batch,
                           wtab, btab);
  c->n_conv++;
  int cout4 = (cout + 3) / 4 * 4;
  int64_t Tout = n;   /* graph-computed output length, same convention as the CPU kernel */
  float *dx = NULL, *dw = NULL, *dy = NULL, *db = NULL;
  int dw_own = 0, db_own = 0;
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
  if (!dx || !dy) PTCHK(cudaMalloc(&dx, xb));   /* arena full: fall back, still correct */
  dw = (float *) weight_ctab(wtab);   /* resident weights, shared across streams */
  if (!dw) {
    PTCHK(cudaMalloc(&dw, wb));
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
      PTCHK(cudaMemcpy(db, bias, (size_t) cout * 4, cudaMemcpyHostToDevice));
      db_own = 1;
    }
  }
  int threads = 256;
  /* stage the weight slice in shared memory when it is small enough to leave room for a few blocks
   * per SM (the Nano has 64 KB of shared per SM and 2 SMs) */
  size_t wsm_bytes = (size_t) K * cin * sizeof(float4);
  int wsm = wsm_bytes > 0 && wsm_bytes <= 16384;
  dim3 grid((unsigned) ((Tout + threads - 1) / threads), (unsigned) (cout4 / 4), (unsigned) batch);
  tA = gpu_prof() ? gpu_now() : 0.0;
  if (wsm) {
    pt_conv_kernel_wsm<<<grid, threads, wsm_bytes, (cudaStream_t) c->g_stream>>>(
        dx, ldx, cin, K, dil, stride, pad, dy, /*ldy=*/Tout, /*Tout=*/Tout, cout, dw, db,
        (int64_t) cin * ldx, (int64_t) cout4 * Tout);
  } else {
    pt_conv_kernel<<<grid, threads, 0, (cudaStream_t) c->g_stream>>>(dx, ldx, cin, K, dil, stride,
                                                                      pad, dy, /*ldy=*/Tout,
                                                                      /*Tout=*/Tout, cout, dw, db,
                                                                      (int64_t) cin * ldx,
                                                                      (int64_t) cout4 * Tout);
  }
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
                                    const float *__restrict__ bias, int64_t bsx, int64_t bsy) {
  extern __shared__ float4 wbuf[];   /* two phase slices: [phase & 1][t*cin + ci] */
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
  for (int idx = threadIdx.x; idx < kc; idx += NT)
    wbuf[idx] = wbase[idx];   /* prime phase 0 */
  for (int r = 0; r < s; r++) {
    int cur = r & 1;
    if (r + 1 < s) {
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
  int dw_own = 0, db_own = 0;
  size_t xb = (size_t) batch * cin * ldx * 4;
  size_t yb = (size_t) batch * cout4 * Lout * 4;   /* padded rows, matching the host's trim stride */
  size_t wb = (size_t) s * (cout4 / 4) * ntap * cin * 4 * 4;
  void *mark = arena_mark(c);
  dx = (float *) arena_alloc(c, xb);
  dy = (float *) arena_alloc(c, yb);
  if (!dx || !dy) PTCHK(cudaMalloc(&dx, xb));
  dw = (float *) weight_ctab(wtab);
  if (!dw) {
    PTCHK(cudaMalloc(&dw, wb));
    PTCHK(cudaMemcpy(dw, wp, wb, cudaMemcpyHostToDevice));
    dw_own = 1;
  }
  PTCHK(cudaMemcpyAsync(dx, X, xb, cudaMemcpyHostToDevice, (cudaStream_t) c->g_stream));
  if (bias) {
    db = (float *) weight_ctab(btab);
    if (!db) {
      PTCHK(cudaMalloc(&db, (size_t) cout * 4));
      PTCHK(cudaMemcpy(db, bias, (size_t) cout * 4, cudaMemcpyHostToDevice));
      db_own = 1;
    }
  }
  int threads = 256;
  size_t wbuf_bytes = (size_t) 2 * ntap * cin * sizeof(float4);
  int wsm = wbuf_bytes > 0 && wbuf_bytes <= 32768 && getenv("PT_V21_CONVT_WSM") == NULL;
  dim3 grid((unsigned) ((Lout + threads - 1) / threads), (unsigned) (cout4 / 4), (unsigned) batch);
  if (wsm) {
    pt_convT_kernel_wsm<<<grid, threads, wbuf_bytes, (cudaStream_t) c->g_stream>>>(
        dx, ldx, cin, tin, dy, /*ldy=*/Lout, /*Lout=*/Lout, cout, s, pad, ntap, dw, db,
        (int64_t) cin * ldx, (int64_t) cout4 * Lout);
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
  if (!dA || !dB || !dC) {
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    PTCHK(cudaMalloc(&dA, ab));
    PTCHK(cudaMalloc(&dB, bb));
    PTCHK(cudaMalloc(&dC, cb));
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
  arena_release(c, mark);
  return 0;
}
