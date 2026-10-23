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
#include <pthread.h>
#include <sched.h>
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
static void *g_wbase;      /* single pooled allocation backing every resident record */
static int g_wdev_n, g_wref;

static size_t g_wdev_bytes;

/* Streams open concurrently. The refcount check and the whole upload run under one lock, so a
 * stream that arrives while another is uploading waits until every record is on the device.
 * Without it, a non-atomic '++g_wref > 1' let a late stream return immediately and run inference
 * with cudaMalloc'd pointers whose cudaMemcpy had not happened yet: garbage weights, observed as
 * one wrong stream of four on a cold run (SNDR -1.98 dB, length off by 256 samples). */
static pthread_mutex_t g_wlock = PTHREAD_MUTEX_INITIALIZER;

static int weights_upload_locked(pt_ctx *c, int verbose);
static void *arena_alloc(pt_ctx *c, size_t bytes);
static void *arena_mark(pt_ctx *c);

/* One-time setup of the pooled device slab: offsets + pointers only, no copies. Split out so the
 * parallel upload can run its copies outside g_wlock (the lock must be free while the other stream
 * threads come in to copy their share). */
static volatile int g_up_next, g_up_done, g_up_n, g_up_fail;
static int g_setup;
static size_t *g_up_offs;

static int weights_setup_locked(pt_ctx *c, int verbose) {
  if (g_setup)
    return 0;
  int n = pt_v21_n_cont(c);
  /* Pool every record into one device allocation: 336 cudaMalloc calls cost ~280 ms of startup
   * (measured) and serialize across streams. A single allocation with 256-byte sub-alignment keeps
   * the same per-record pointers the ctab hands out, so kernels and refcounting are unchanged. */
  g_wdev = (void **) calloc((size_t) n, sizeof(void *));
  g_up_offs = (size_t *) calloc((size_t) n, sizeof(size_t));
  if (!g_wdev || !g_up_offs) {
    g_wref = 0;
    return -1;
  }
  g_wdev_n = n;
  size_t total = 0;
  for (int i = 0; i < n; i++) {
    const void *h = pt_v21_cont(c, i);
    size_t nb = (size_t) pt_v21_cont_bytes(c, i);
    if (!h || nb == 0) {
      g_up_offs[i] = 0;
      continue;
    }
    g_up_offs[i] = total;
    total += (nb + 255) & ~(size_t) 255;
    g_wdev_bytes += nb;
  }
  if (cudaMalloc(&g_wbase, total ? total : 1) != cudaSuccess) {
    fprintf(stderr, "pt_v21 cuda: pooled weight allocation failed (%zu B)\n", total);
    g_wref = 0;
    return -1;
  }
  for (int i = 0; i < n; i++) {
    const void *h = pt_v21_cont(c, i);
    size_t nb = (size_t) pt_v21_cont_bytes(c, i);
    if (!h || nb == 0)
      continue;
    g_wdev[i] = (char *) g_wbase + g_up_offs[i];
  }
  g_up_n = n;
  g_up_next = 0;
  g_up_done = 0;
  g_up_fail = 0;
  g_setup = 1;
  (void) verbose;
  return 0;
}

static int weights_upload(pt_ctx *c, int verbose) {
  static int par = -1;
  if (par < 0)
    par = getenv("PT_V21_UP_PAR") ? atoi(getenv("PT_V21_UP_PAR")) : 1;
  if (!par) {
    pthread_mutex_lock(&g_wlock);
    int r = weights_upload_locked(c, verbose);
    pthread_mutex_unlock(&g_wlock);
    return r;
  }
  /* The 110 MB weight upload is the largest single item of the serialized startup (~138-165 ms at
   * the Nano's ~0.8 GB/s H2D). Default ON (PT_V21_UP_PAR=0 = serialized bisection endpoint): the
   * copy is issued by one thread while the other three wait on g_wlock, so every stream's startup
   * pays the whole upload. Every record is an independent range of the shared slab and the host
   * payload is the same bytes in every context, so the copy is claimed the way the container record
   * pass is: each stream thread copies the records it claims with cudaMemcpyAsync on its own CUDA
   * stream, syncs its own stream, then waits for everyone's share. A late-arriving stream finds the
   * copies done instead of queueing behind them. Numerics are untouched: identical bytes, identical
   * destinations (proved bit-identical max|d| on all 5 golden cases, both arms). */
  pthread_mutex_lock(&g_wlock);
  g_wref++;
  int r = weights_setup_locked(c, verbose);
  int n = g_up_n;
  pthread_mutex_unlock(&g_wlock);
  if (r)
    return r;
  for (;;) {
    int i = __sync_fetch_and_add(&g_up_next, 1);
    if (i >= n)
      break;
    const void *h = pt_v21_cont(c, i);
    size_t nb = (size_t) pt_v21_cont_bytes(c, i);
    if (g_wdev[i] && h && nb)
      cudaMemcpyAsync(g_wdev[i], h, nb, cudaMemcpyHostToDevice, (cudaStream_t) c->g_stream);
    __sync_fetch_and_add(&g_up_done, 1);
  }
  if (cudaStreamSynchronize((cudaStream_t) c->g_stream) != cudaSuccess)
    __sync_bool_compare_and_swap(&g_up_fail, 0, 1);
  while (__sync_add_and_fetch(&g_up_done, 0) < n)
    sched_yield();
  if (g_up_fail) {
    fprintf(stderr, "pt_v21 cuda: parallel weight upload failed\n");
    return -1;
  }
  return 0;
}

static int weights_upload_locked(pt_ctx *c, int verbose) {
  int n = pt_v21_n_cont(c);
  if (++g_wref > 1)
    return 0;   /* already uploaded: the weights are identical for every stream */
  if (weights_setup_locked(c, verbose))
    return -1;
  for (int i = 0; i < n; i++) {
    const void *h = pt_v21_cont(c, i);
    size_t nb = (size_t) pt_v21_cont_bytes(c, i);
    if (!h || nb == 0 || !g_wdev[i])
      continue;
    if (cudaMemcpy(g_wdev[i], h, nb, cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "pt_v21 cuda: weight upload failed for ctab %d (%zu B)\n", i, nb);
      g_wref = 0;
      return -1;
    }
  }
  if (verbose)
    fprintf(stderr, "pt_v21 cuda: %d container records resident on the GPU (%.1f MiB)\n", n,
            (double) g_wdev_bytes / 1048576.0);
  return 0;
}

/* --------------------------------------------- device-resident residual Add (PT_V21_DEVADD)
 * The decoder/iSTFT tail moves ~121 MB per stream through the 22 big Adds: each link copies its two
 * operands D2H, adds on the host, then copies the result back H2D for the next conv. When the
 * operands are already on the device (a chained conv output, an earlier device Add) and the result's
 * only reader is the next conv, all three copies disappear: one kernel pass over the tensor, result
 * left in the arena. The arithmetic is the same fp32 add (+ the fused LeakyReLU, the same fp32 math
 * as the host un_op), so results are bit-identical to the host path.
 * Liveness: allocations join the conv chain's arena mark and are released when the chain ends, so
 * nothing is freed while a reader can still see it. */
__global__ void pt_devleaky_kernel(const float *__restrict__ a, float *__restrict__ y, long long n,
                                  float alpha) {
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long st = (long long) gridDim.x * blockDim.x;
  for (; i < n; i += st) {
    float v = a[i];
    y[i] = v >= 0.0f ? v : alpha * v;   /* the same fp32 math as the host un_op */
  }
}

int pt_v21_devleaky_run(pt_ctx *c, const float *a, float *y, int64_t n, float alpha) {
  long long blocks = (n + 255) / 256;
  if (blocks > 4096)
    blocks = 4096;
  pt_devleaky_kernel<<<(unsigned) blocks, 256, 0, (cudaStream_t) c->g_stream>>>(a, y, n, alpha);
  if (cudaGetLastError() != cudaSuccess)
    return -1;
  return 0;
}

__global__ void pt_devadd_kernel(const float *__restrict__ a, const float *__restrict__ b,
                                float *__restrict__ y, long long n, int epi, float alpha) {
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long st = (long long) gridDim.x * blockDim.x;
  for (; i < n; i += st) {
    float v = a[i] + b[i];
    if (epi == 1)
      v = v >= 0.0f ? v : alpha * v;
    y[i] = v;
  }
}

/* ------------------------------ device-residency sub-arena (PT_V21_DEVADD)
 * The conv chain's bump arena releases wholesale at the chain end, which is correct only while every
 * kept tensor dies with the chain. The residual Add tree shares intermediate tensors between two
 * readers, so those tensors outlive any single chain and need a real allocator: best-fit over a free
 * list with coalescing, one slab per stream, created on first use, blocks tracked by offset+size.
 * Result tensors are released by the interpreter at their last use (tv_free); operand uploads are
 * released at the chain end, where the stream is synchronized and the kernels that read them are
 * provably done. */
typedef struct { size_t off, size; } pt_devrng;
typedef struct {
  void *base;
  size_t size;
  size_t used;
  int nalloc;
  int nfree;
  int npend;
  int oom;
  pt_devrng alloc[128];
  pt_devrng free[128];
  size_t pend[64];
} pt_devres;

void *pt_v21_dev_alloc(pt_ctx *c, size_t bytes) {
  pt_devres *d = (pt_devres *) c->devres;
  size_t nb = (bytes + 255) & ~(size_t) 255;
  if (!d) {
    d = (pt_devres *) calloc(1, sizeof *d);
    if (!d)
      return nullptr;
    long mb = getenv("PT_V21_DEVRES_MB") ? atol(getenv("PT_V21_DEVRES_MB")) : 24;
    d->size = (size_t) mb << 20;
    if (cudaMalloc(&d->base, d->size) != cudaSuccess) {
      cudaGetLastError();
      free(d);
      c->devres_oom = 1;
      return nullptr;
    }
    c->devres = d;
    c->dev_bytes_res = (double) d->size;
  }
  if (d->oom || d->nalloc >= 128)
    return nullptr;
  int best = -1;
  for (int i = 0; i < d->nfree; i++)
    if (d->free[i].size >= nb && (best < 0 || d->free[i].size < d->free[best].size))
      best = i;
  size_t off;
  if (best >= 0) {
    off = d->free[best].off;
    size_t left = d->free[best].size - nb;
    if (left >= 256) {
      d->free[best].off += nb;
      d->free[best].size = left;
    } else {
      d->free[best] = d->free[d->nfree - 1];
      d->nfree--;
    }
  } else {
    off = d->used;
    d->used += nb;
    if (d->used > d->size) {
      d->oom = 1;
      return nullptr;
    }
  }
  d->alloc[d->nalloc].off = off;
  d->alloc[d->nalloc].size = nb;
  d->nalloc++;
  c->dev_bytes_stray += nb;
  return (char *) d->base + off;
}

void pt_v21_dev_free(pt_ctx *c, void *p) {
  pt_devres *d = (pt_devres *) c->devres;
  if (!d || !p)
    return;
  if ((char *) p < (char *) d->base)
    return;   /* not ours (the conv chain's bump arena): its own discipline owns it */
  size_t base = (size_t) ((char *) p - (char *) d->base);
  if (base >= d->size)
    return;
  int hit = -1;
  for (int i = 0; i < d->nalloc; i++)
    if (d->alloc[i].off == base) {
      hit = i;
      break;
    }
  if (hit < 0 || d->nfree >= 128)
    return;   /* not an allocation of ours, or the free table is full (leak, never corrupt) */
  pt_devrng r = d->alloc[hit];
  d->alloc[hit] = d->alloc[d->nalloc - 1];
  d->nalloc--;
  for (int i = 0; i < d->npend; i++)
    if (d->pend[i] == base) {
      d->pend[i] = d->pend[d->npend - 1];
      d->npend--;
      break;
    }
  d->free[d->nfree++] = r;
  for (int i = 0; i < d->nfree; i++)
    for (int j = i + 1; j < d->nfree; j++)
      if (d->free[i].off + d->free[i].size == d->free[j].off) {
        d->free[i].size += d->free[j].size;
        d->free[j] = d->free[d->nfree - 1];
        d->nfree--;
        j--;
      } else if (d->free[j].off + d->free[j].size == d->free[i].off) {
        d->free[j].size += d->free[i].size;
        d->free[i] = d->free[d->nfree - 1];
        d->nfree--;
        i--;
      }
  d->oom = 0;
}

/* Release the operand uploads queued since the last chain end. Called where the stream is
 * synchronized, so the kernels that read them have completed. */
void pt_v21_dev_reclaim(pt_ctx *c) {
  pt_devres *d = (pt_devres *) c->devres;
  if (!d)
    return;
  while (d->npend > 0) {
    size_t off = d->pend[d->npend - 1];
    int hit = -1;
    for (int i = 0; i < d->nalloc; i++)
      if (d->alloc[i].off == off) {
        hit = i;
        break;
      }
    if (hit < 0) {
      d->npend--;
      continue;
    }
    pt_devrng r = d->alloc[hit];
    d->alloc[hit] = d->alloc[d->nalloc - 1];
    d->nalloc--;
    d->npend--;
    if (d->nfree < 128)
      d->free[d->nfree++] = r;
  }
  d->oom = 0;
}

void pt_v21_dev_sync_host(pt_ctx *c, void *host, const void *dev, int64_t n) {
  if (n <= 0)
    return;
  cudaMemcpyAsync(host, dev, (size_t) n * 4, cudaMemcpyDeviceToHost, (cudaStream_t) c->g_stream);
  cudaStreamSynchronize((cudaStream_t) c->g_stream);
}

float *pt_v21_dev_upload(pt_ctx *c, const void *h, int64_t n) {
  size_t nb = (size_t) n * 4;
  float *d = (float *) pt_v21_dev_alloc(c, nb);
  if (!d)
    return nullptr;
  if (cudaMemcpyAsync(d, h, nb, cudaMemcpyHostToDevice, (cudaStream_t) c->g_stream) != cudaSuccess) {
    cudaGetLastError();
    pt_v21_dev_free(c, d);
    return nullptr;   /* a failed upload: the caller falls back to the host path */
  }
  pt_devres *dr = (pt_devres *) c->devres;
  if (dr && dr->npend < 64)
    dr->pend[dr->npend++] = (size_t) ((char *) d - (char *) dr->base);
  return d;
}

int pt_v21_devadd_run(pt_ctx *c, const float *a, const float *b, float *y, int64_t n, int epi,
                      float alpha) {
  long long blocks = (n + 255) / 256;
  if (blocks > 4096)
    blocks = 4096;
  pt_devadd_kernel<<<(unsigned) blocks, 256, 0, (cudaStream_t) c->g_stream>>>(a, b, y, n, epi, alpha);
  if (cudaGetLastError() != cudaSuccess)
    return -1;   /* launch failure: fall back to the host path */
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
  pthread_mutex_lock(&g_wlock);
  if (--g_wref <= 0) {
    if (g_wbase)
      cudaFree(g_wbase);
    g_wbase = NULL;
    free(g_wdev);
    g_wdev = NULL;
    free(g_up_offs);
    g_up_offs = NULL;
    g_setup = 0;
    g_wdev_n = 0;
    g_wref = 0;
  }
  pthread_mutex_unlock(&g_wlock);
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


/* ------------------------------------------------------- GPU clock warmup (nvhost_podgov)
 * nvhost_podgov idles the GPU at 76.8 MHz and only ramps once the SMs carry sustained load -- there
 * is no root on this box to pin clocks (jetson_clocks / min_freq are mode 644 root-only). The cost of
 * that idle clock is visible in two places: the 111 MB resident-weight upload runs at only ~0.46 GB/s
 * (230 ms) while an identical second upload immediately after runs 1.4x faster (0.66 GB/s), and the
 * first process after a build measures 700-850 ms slower single-stream than any warm one. A short
 * fixed amount of real arithmetic right after the context comes up makes the governor ramp *before*
 * the upload and the first request, which is work the process owes anyway.
 * Fixed iteration count (not a time target), so its own cost is bounded and observable. */
__global__ void pt_warmup_kernel(int *sink, int iters) {
  int v = sink[threadIdx.x];
  for (int i = 0; i < iters; i++) {
    v = v * 1664525 + 1013904223;   /* ALU chain: no memory traffic, no denormals */
    v ^= v >> 13;
  }
  sink[threadIdx.x] = v;
}

static pthread_mutex_t g_warm_lock = PTHREAD_MUTEX_INITIALIZER;

/* Runs at most once per process; returns the wall cost of the warmup itself. */
static double gpu_warm_once(void) {
  static int done;
  pthread_mutex_lock(&g_warm_lock);
  if (done) {
    pthread_mutex_unlock(&g_warm_lock);
    return 0.0;
  }
  done = 1;
  pthread_mutex_unlock(&g_warm_lock);

  const char *e = getenv("PT_V21_WARM");
  int iters = e ? atoi(e) : 64000;
  if (iters <= 0)
    return 0.0;
  int *sink = NULL;
  if (cudaMalloc((void **) &sink, 256 * sizeof(int)) != cudaSuccess) {
    cudaGetLastError();
    return 0.0;
  }
  cudaMemset(sink, 0, 256 * sizeof(int));
  double t0 = gpu_now();
  pt_warmup_kernel<<<32, 256>>>(sink, iters);
  if (cudaDeviceSynchronize() != cudaSuccess)
    cudaGetLastError();
  double cost = gpu_now() - t0;
  cudaFree(sink);
  return cost;
}

static int g_prewarm;
static void gpu_warm_launch(void);
static double gpu_warm_finish(void);

/* The clock ramp: nvhost_podgov idles the GPU at 76.8 MHz and ramps only on sustained SM load, so a
 * fixed ALU kernel is what reaches full clock without root (run 75: upload 226 -> 138 ms, matched N=4
 * -148 ms). Launched on its own non-blocking stream so it can overlap the CPU-side container record
 * pass; gpu_warm_finish joins it before any inference can start. */
static cudaStream_t g_warm_stream;
static int *g_warm_sink;
static double g_warm_t0;
static double g_warm_cost = -1.0;

/* Load-aware ramp gate: the governor's idle clock is box-dependent (picard idles at 768 MHz, pesi at
 * the 921 MHz max), and a process that arrives while the GPU is already at max owes no ramp at all.
 * cur_freq/max_freq are world-readable devfreq sysfs, no root needed. Unknown sysfs -> ramp
 * (conservative: an unmeasured box gets the ramp, which is what run 90's robustness finding wants). */
static int gpu_clock_is_max(void) {
  static int state = -1;   /* -1 unknown, 0 below max, 1 at max */
  if (state >= 0)
    return state;
  const char *base = "/sys/devices/platform/host1x/57000000.gpu/devfreq/57000000.gpu/";
  char path[128], buf[32];
  long cur = -1, max = -1;
  snprintf(path, sizeof path, "%scur_freq", base);
  FILE *fc = fopen(path, "r");
  if (fc && fgets(buf, sizeof buf, fc))
    cur = atol(buf);
  if (fc)
    fclose(fc);
  snprintf(path, sizeof path, "%smax_freq", base);
  FILE *fm = fopen(path, "r");
  if (fm && fgets(buf, sizeof buf, fm))
    max = atol(buf);
  if (fm)
    fclose(fm);
  if (cur <= 0 || max <= 0)
    return -1;
  state = cur >= max ? 1 : 0;
  return state;
}

static void gpu_warm_launch(void) {
  static int ticket;
  pthread_mutex_lock(&g_warm_lock);
  if (ticket) {
    pthread_mutex_unlock(&g_warm_lock);
    return;
  }
  ticket = 1;
  pthread_mutex_unlock(&g_warm_lock);
  const char *e = getenv("PT_V21_WARM");
  int iters = e ? atoi(e) : 64000;
  if (iters <= 0)
    return;
  if (getenv("PT_V21_WARM_ADAPT") ? atoi(getenv("PT_V21_WARM_ADAPT")) : 1) {
    if (gpu_clock_is_max() == 1) {
      /* Already at max: no ramp is owed. Keep a short maintenance burst instead of skipping entirely
       * (run 90's failure mode: the governor drops the clock again while this thread runs its
       * CPU-side record pass, and copies issued at the dropped clock cost far more than the burst).
       * 8k iters is ~3 ms of SM work -- an insurance premium, not a ramp. */
      if (iters > 8000)
        iters = 8000;
      if (getenv("PT_V21_WARMLOG"))
        fprintf(stderr, "pt_v21 cuda: clock at max, maintenance burst only\n");
    }
  }
  if (cudaMalloc((void **) &g_warm_sink, 256 * sizeof(int)) != cudaSuccess) {
    cudaGetLastError();
    g_warm_sink = NULL;
    return;
  }
  cudaMemset(g_warm_sink, 0, 256 * sizeof(int));
  if (cudaStreamCreateWithFlags(&g_warm_stream, cudaStreamNonBlocking)) {
    g_warm_stream = NULL;
    cudaFree(g_warm_sink);
    g_warm_sink = NULL;
    return;
  }
  g_warm_t0 = gpu_now();
  pt_warmup_kernel<<<32, 256, 0, g_warm_stream>>>(g_warm_sink, iters);
}

static double gpu_warm_finish(void) {
  double cost = 0.0;
  pthread_mutex_lock(&g_warm_lock);
  if (g_warm_stream) {
    if (cudaStreamSynchronize(g_warm_stream) != cudaSuccess)
      cudaGetLastError();
    cost = gpu_now() - g_warm_t0;
    cudaStreamDestroy(g_warm_stream);
    g_warm_stream = NULL;
    if (g_warm_sink) {
      cudaFree(g_warm_sink);
      g_warm_sink = NULL;
    }
    g_warm_cost = cost;
  } else {
    cost = g_warm_cost >= 0.0 ? g_warm_cost : 0.0;
  }
  pthread_mutex_unlock(&g_warm_lock);
  return cost;
}

int pt_v21_cuda_prewarm(void) {
  static int ticket;
  pthread_mutex_lock(&g_warm_lock);
  if (ticket) {
    pthread_mutex_unlock(&g_warm_lock);
    return 0;
  }
  ticket = 1;
  pthread_mutex_unlock(&g_warm_lock);
  const char *e = getenv("PT_V21_WARM_PRE");
  if (e && !atoi(e))
    return 0;   /* serialized bisection endpoint: cuda_init does the ramp in its own time */
  /* The context must exist before the ramp kernel can run, and creating it is the first serialized
   * item of every process (~85 ms). The ramp kernel is pure SM work; the container record pass that
   * the caller runs next is pure CPU/memory work, so launching the ramp here lets both proceed at
   * once, and the weight copies that follow are still issued after the ramp completes (at the
   * ramped clock, where H2D is 0.8 not 0.46 GB/s). This is not a thread: the ramp costs the CPU
   * nothing, so no core is taken from the 4 streams -- which is what killed the 5-thread prewarm. */
  {
    const char *s = getenv("PT_V21_SYNC");
    int mode = s ? atoi(s) : 2;
    if (mode == 1) cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    else if (mode == 2) cudaSetDeviceFlags(cudaDeviceScheduleYield);
    cudaGetLastError();
  }
  if (cudaFree(0) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  gpu_warm_launch();
  g_prewarm = 1;
  return 0;
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
  double tw0 = gpu_now();
  if (!g_prewarm) {
    gpu_warm_launch();
    gpu_warm_finish();   /* serialized ramp (prewarm off): the ramped clock must front the copies */
  }
  double twarm = gpu_warm_once();
  if (verbose || getenv("PT_V21_WARMLOG"))
    fprintf(stderr, "pt_v21 cuda: clock warmup %.1f ms\n", twarm);
  (void) tw0;
  c->use_cuda = 1;
  if (cudaStreamCreateWithFlags((cudaStream_t *) &c->g_stream, cudaStreamNonBlocking)) {
    fprintf(stderr, "pt_v21 cuda: stream create failed\n");
    return -1;
  }
  size_t aw = getenv("PT_V21_ARENA_MB") ? (size_t) atoi(getenv("PT_V21_ARENA_MB")) : 8u;
  if (arena_init(c, aw * 1048576u)) {
    fprintf(stderr, "pt_v21 cuda: arena of %zu MiB failed\n", aw);
    return -1;
  }
  {
    double tu0 = gpu_now();
    int ur = weights_upload(c, verbose);
    double twarm = gpu_warm_finish();   /* ramp in flight since prewarm: join before inference */
    if (verbose || getenv("PT_V21_WARMLOG"))
      fprintf(stderr, "pt_v21 cuda: weight upload %.1f ms (ramp %.1f ms)\n", gpu_now() - tu0, twarm);
    if (ur)
      return -1;
  }
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
                                   float ea0, float ea1, int64_t pad_in) {
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
  if (pad_in >= 0) {
    /* unpadded row of length ldx: the tap is the zero padding when it falls outside, and whether it
     * does depends only on (k, t) -- so the test hoists out of the ci loop. A masked tap still
     * performs the same `acc += w*0` add (exactly zero) that the reference kernel does, so the
     * summation chain is unchanged. */
    for (int k = 0; k < K; k++) {
      int64_t s = (int64_t) t * stride + (int64_t) (k * dil) - pad_in;
      int ok = (s >= 0 && s < ldx);
      for (int ci = 0; ci < cin; ci++) {
        float4 w = wsm[(int64_t) k * cin + ci];
        float xv = ok ? xb[(int64_t) ci * ldx + s] : 0.0f;
        acc.x += w.x * xv;
        acc.y += w.y * xv;
        acc.z += w.z * xv;
        acc.w += w.w * xv;
      }
    }
  } else {
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
  const float *din = c->conv_in_dev;   /* chained input already on the device */
  int keep = c->conv_keep;
  c->conv_out_dev = NULL;
  dx = din ? (float *) din : (float *) arena_alloc(c, xb);
  /* a kept output that outlives the chain (the residual Add tree) comes from the residency sub-arena */
  dy = (keep && c->conv_keep_res) ? (float *) pt_v21_dev_alloc(c, yb) : (float *) arena_alloc(c, yb);
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
  if (!din)
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
  /* The GEMM kernels tile channels in 16 groups of 4. Every wide conv in this model has a channel
   * count that is an exact multiple of 64, so the tile is always full; a narrow conv (e.g. the final
   * cout=1 output projection) would leave 15 of the 16 groups computing and discarding results, so
   * it goes to the column-per-thread kernel instead, which has no channel padding. */
  int wide_tile = (cout4 / 4) >= 16 || getenv("PT_V21_IG_ANY");
  if (ig_min >= 0 && Tout >= ig_min && wide_tile) {
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
        (int64_t) cin * ldx, (int64_t) cout4 * Tout, epi0, epi1, ea0, ea1, c->conv_pad_in);
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
  if (keep && !dy_own && !dx_own && !dw_own && !db_own) {
    /* chain link: the next conv reads dy straight from the arena. No copy back, no sync; the arena
     * range stays allocated until the chain's last conv releases it. */
    if (!c->chain_mark)
      c->chain_mark = mark;
    c->conv_out_dev = dy;
    return 0;
  }
  PTCHK(cudaMemcpyAsync(Y, dy, yb, cudaMemcpyDeviceToHost, (cudaStream_t) c->g_stream));
  PTCHK(cudaStreamSynchronize((cudaStream_t) c->g_stream));
  pt_v21_dev_reclaim(c);   /* chain end: the operand uploads' kernels are provably done */
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
  if (c->chain_mark) {
    arena_release(c, c->chain_mark);   /* end of a conv chain: release every link */
    c->chain_mark = NULL;
  } else {
    arena_release(c, mark);   /* weights are resident: only the activation scratch is released */
  }
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


/* 128-column version of the GEMM-tiled phase-major conv-transpose: the pt_conv_kernel_ig8 column
 * tiling applied to the convT phase formulation (blockIdx.z = batch*s + r). Each thread owns 8
 * consecutive q values of one 4-channel group (32 accumulators) and the block stages 128 q values.
 * Per output the accumulation walks the kc reductions in the same ascending order as
 * pt_conv_kernel_ig, so results are bit-identical to that kernel; the win is 2x the columns per
 * block, i.e. half the index/loop overhead per element. */
__global__ void __launch_bounds__(256) pt_convT_kernel_ig8(
    const float *__restrict__ x, int64_t ldx, int cin, int64_t tin, float *__restrict__ y,
    int64_t ldy, int64_t Lout, int cout, int s, int pad, int ntap, const float *__restrict__ wp,
    const float *__restrict__ bias, int64_t bsx, int64_t bsy, int64_t q0) {
  __shared__ float4 Ws[IG_RK][16];
  __shared__ float4 Xs[IG_RK][32]; /* [r][column quad], 32 quads = 128 columns */
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int nblk = (cout + 3) / 4;
  const int r = (int) (blockIdx.z % s);
  const int64_t b = (int64_t) (blockIdx.z / s);
  const int g = blockIdx.y * 16 + ty;
  const int64_t qb = q0 + (int64_t) blockIdx.x * 128;
  const int kc = ntap * cin;
  const float *xb = x + b * bsx;
  const float4 *w4 = (const float4 *) wp + (int64_t) r * nblk * kc;
  int co0 = g * 4;
  float4 bz = make_float4(bias && co0 + 0 < cout ? bias[co0 + 0] : 0.f,
                          bias && co0 + 1 < cout ? bias[co0 + 1] : 0.f,
                          bias && co0 + 2 < cout ? bias[co0 + 2] : 0.f,
                          bias && co0 + 3 < cout ? bias[co0 + 3] : 0.f);
  float4 a[8];
#pragma unroll
  for (int j = 0; j < 8; j++)
    a[j] = bz;
  const int lr = tid >> 4, lq = tid & 15;
  for (int i0 = 0; i0 < kc; i0 += IG_RK) {
    int i = i0 + lr;
    int lg = blockIdx.y * 16 + lq;
    Ws[lr][lq] = (i < kc && lg < nblk) ? w4[(int64_t) lg * kc + i] : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int h = 0; h < 2; h++) {
      int qq = lq + 16 * h; /* column quad 0..31 */
      float4 xv = make_float4(0.f, 0.f, 0.f, 0.f);
      if (i < kc) {
        int t = i / cin, ci = i - t * cin;
        const float *row = xb + (int64_t) ci * ldx;
        int64_t qc = qb + qq * 4 - t;
        if (qc + 0 >= 0 && qc + 0 < tin)
          xv.x = row[qc + 0];
        if (qc + 1 >= 0 && qc + 1 < tin)
          xv.y = row[qc + 1];
        if (qc + 2 >= 0 && qc + 2 < tin)
          xv.z = row[qc + 2];
        if (qc + 3 >= 0 && qc + 3 < tin)
          xv.w = row[qc + 3];
      }
      Xs[lr][qq] = xv;
    }
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < IG_RK; rr++) {
      float4 w = Ws[rr][ty];
      float4 v0 = Xs[rr][tx];      /* columns 4*tx .. 4*tx+3 */
      float4 v1 = Xs[rr][tx + 16]; /* columns 64+4*tx .. */
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
    int64_t q = qb + ((j < 4) ? tx * 4 + j : 64 + tx * 4 + (j - 4));
    int64_t oi = q * s + r - pad;
    if (oi < 0 || oi >= Lout)
      continue;
    float4 acc = a[j];
    if (co0 + 0 < cout)
      yb[(int64_t) (co0 + 0) * ldy + oi] = acc.x;
    if (co0 + 1 < cout)
      yb[(int64_t) (co0 + 1) * ldy + oi] = acc.y;
    if (co0 + 2 < cout)
      yb[(int64_t) (co0 + 2) * ldy + oi] = acc.z;
    if (co0 + 3 < cout)
      yb[(int64_t) (co0 + 3) * ldy + oi] = acc.w;
  }
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
    /* 128-column convT tiles: bit-exact vs pt_conv_kernel_ig (same accumulation order per output),
     * convT 37.1 -> 33.0 ms/stream, N=4 wall -13 ms over 7 clean matched pairs (5/7 wins).
     * PT_V21_CONVT_IG8=0 is the bisection endpoint (64-column kernel). */
    static int tig8 = -1;
    if (tig8 < 0)
      tig8 = getenv("PT_V21_CONVT_IG8") ? atoi(getenv("PT_V21_CONVT_IG8")) : 1;
    if (tig8) {
      dim3 g8t((unsigned) ((nq + 127) / 128), (unsigned) ((cout4 / 4 + 15) / 16),
               (unsigned) (batch * s));
      pt_convT_kernel_ig8<<<g8t, 256, 0, (cudaStream_t) c->g_stream>>>(
          dx, ldx, cin, tin, dy, /*ldy=*/Lout, /*Lout=*/Lout, cout, s, pad, ntap, dw, db,
          (int64_t) cin * ldx, (int64_t) cout4 * Lout, qlo);
    } else {
    dim3 gig((unsigned) ((nq + 63) / 64), (unsigned) ((cout4 / 4 + 15) / 16),
             (unsigned) (batch * s));
    pt_convT_kernel_ig<<<gig, 256, 0, (cudaStream_t) c->g_stream>>>(
        dx, ldx, cin, tin, dy, /*ldy=*/Lout, /*Lout=*/Lout, cout, s, pad, ntap, dw, db,
        (int64_t) cin * ldx, (int64_t) cout4 * Lout, qlo);
    }
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


/* Tiled matmul: block = 64 rows x 64 columns, 256 threads, 4x4 outputs per thread, p walked in
 * chunks of 16 through shared memory. Each output still accumulates p = 0..k-1 in order onto 0,
 * the chain of pt_matmul_kernel: bit-identical. */
__global__ void __launch_bounds__(256) pt_matmul_kernel_t(
    const float *__restrict__ A, int64_t m, int64_t k, int64_t lda, const float *__restrict__ B,
    int64_t ldb, int64_t nn, float *__restrict__ C, int64_t ldc) {
  __shared__ float4 As[16][16];   /* [p][row group] = 4 rows */
  __shared__ float4 Bs[16][16];   /* [p][column quad] */
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int64_t i0 = (int64_t) blockIdx.y * 64, j0 = (int64_t) blockIdx.x * 64;
  float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;   /* rows 4ty+0..3 */
  const int lr = tid >> 4, lq = tid & 15;
  for (int64_t p0 = 0; p0 < k; p0 += 16) {
    int64_t p = p0 + lr;
    float4 av = make_float4(0.f, 0.f, 0.f, 0.f), bv = av;
    if (p < k) {
      int64_t r = i0 + 4 * lq;
      if (r + 0 < m) av.x = A[(r + 0) * lda + p];
      if (r + 1 < m) av.y = A[(r + 1) * lda + p];
      if (r + 2 < m) av.z = A[(r + 2) * lda + p];
      if (r + 3 < m) av.w = A[(r + 3) * lda + p];
      int64_t cidx = j0 + 4 * lq;
      const float *bp = B + p * ldb;
      if (cidx + 0 < nn) bv.x = bp[cidx + 0];
      if (cidx + 1 < nn) bv.y = bp[cidx + 1];
      if (cidx + 2 < nn) bv.z = bp[cidx + 2];
      if (cidx + 3 < nn) bv.w = bp[cidx + 3];
    }
    As[lr][lq] = av;
    Bs[lr][lq] = bv;
    __syncthreads();
    int pn = (int) ((k - p0) < 16 ? (k - p0) : 16);
    for (int rr = 0; rr < pn; rr++) {
      float4 a = As[rr][ty];
      float4 b = Bs[rr][tx];
      a0.x += a.x * b.x; a0.y += a.x * b.y; a0.z += a.x * b.z; a0.w += a.x * b.w;
      a1.x += a.y * b.x; a1.y += a.y * b.y; a1.z += a.y * b.z; a1.w += a.y * b.w;
      a2.x += a.z * b.x; a2.y += a.z * b.y; a2.z += a.z * b.z; a2.w += a.z * b.w;
      a3.x += a.w * b.x; a3.y += a.w * b.y; a3.z += a.w * b.z; a3.w += a.w * b.w;
    }
    __syncthreads();
  }
  float4 *rows[4] = {&a0, &a1, &a2, &a3};
  for (int j = 0; j < 4; j++) {
    int64_t i = i0 + 4 * ty + j;
    if (i >= m)
      continue;
    int64_t c0 = j0 + 4 * tx;
    float *cr = C + i * ldc;
    float4 v = *rows[j];
    if (c0 + 0 < nn) cr[c0 + 0] = v.x;
    if (c0 + 1 < nn) cr[c0 + 1] = v.y;
    if (c0 + 2 < nn) cr[c0 + 2] = v.z;
    if (c0 + 3 < nn) cr[c0 + 3] = v.w;
  }
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
  static int mt = -1;
  if (mt < 0) mt = getenv("PT_V21_MM_TILED") ? atoi(getenv("PT_V21_MM_TILED")) : 1;
  if (mt) {
    dim3 gt((unsigned) ((nn + 63) / 64), (unsigned) ((m + 63) / 64));
    pt_matmul_kernel_t<<<gt, 256, 0, (cudaStream_t) c->g_stream>>>(dA, m, k, lda, dB, ldb, nn, dC,
                                                                     ldc);
  } else {
  int tx = 32, ty = 8;
  dim3 block(tx, ty);
  dim3 grid((unsigned) ((nn / 4 + tx - 1) / tx), (unsigned) ((m + ty - 1) / ty));
  pt_matmul_kernel<<<grid, block, 0, (cudaStream_t) c->g_stream>>>(dA, m, k, lda, dB, ldb, nn, dC,
                                                                    ldc);
  }
  PTCHK(cudaGetLastError());
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
