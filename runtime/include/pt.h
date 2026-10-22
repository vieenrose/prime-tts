/* pt.h - PrimeTTS Nano runtime core types.
 * Design notes (measured on the target, Jetson Nano gen-1):
 *  - 4x Cortex-A57, ARMv8.0: NEON fp32 only. int8/fp16 give nothing here, so fp32 is the format.
 *  - fp32 NEON GEMM: 3.74 GFLOP/s per core, 9.78 GFLOP/s at 4 threads (measured, naive 4x4 tile).
 *  - memcpy ~5-5.9 GB/s; Tegra X1 CUDA axpy 5.5 GB/s; kernel launch ~26 us -> kernel COUNT dominates.
 *  - ~185 MB free RAM: weights are mmap'd read-only and shared by all 8 streams; per-stream cost is
 *    activations only.
 *
 * Activation layout everywhere: channel-major [C][ld] fp32, time contiguous, rows 16-byte aligned.
 * Channel counts are padded to a multiple of 4 (padded rows hold exact zeros).
 */
#ifndef PT_H_
#define PT_H_
#include <stdint.h>
#include <stddef.h>

#define PT_MAGIC 0x50544B31u /* "PTK1" */
#define PT_MAX_TENSOR 4096
#define PT_MAX_NAME 64
#define PT_MAX_STREAM 8

typedef struct {
  char name[PT_MAX_NAME];
  int32_t ndim;
  int64_t dim[6];
  int64_t nelem;
  int64_t off;            /* byte offset of fp32 payload from the (64-byte aligned) data start */
} pt_tensor;

typedef struct {
  uint32_t magic, version, n_tensor;
  int64_t data_bytes;
  pt_tensor t[PT_MAX_TENSOR];
  int fd;                 /* mmap source */
  void *map;              /* whole-file mapping */
  void *base;             /* data start inside the mapping */
  size_t maplen;
} pt_weights;

/* Load a .ptk container (weights exported from ONNX by tools/export_ptk.py). */
int pt_weights_open(pt_weights *w, const char *path);
void pt_weights_close(pt_weights *w);
const pt_tensor *pt_find(const pt_weights *w, const char *name);
const float *pt_f32(const pt_weights *w, const char *name); /* NULL if missing */

/* Scratch arena: one per stream, reused across requests (no malloc in the hot path). */
typedef struct {
  void *p;
  size_t cap, used, peak;
} pt_arena;

int pt_arena_init(pt_arena *a, size_t bytes);
void pt_arena_free(pt_arena *a);
static inline void pt_arena_reset(pt_arena *a) { a->used = 0; }
void *pt_arena_alloc(pt_arena *a, size_t bytes);

/* ---------------- micro-kernels: runtime/kern/ (asm) on aarch64, runtime/src/kern_ref.c elsewhere -------------- */
/* Conv / GEMM micro-kernel, 4 output rows x (16*ntile) columns:
 *   Y[i][t] = bias[i] + sum_{j<nk} A[j*4+i] * X[off[j]/4 + t]        i < 4,  t < 16*ntile
 * A: packed weights of one 4-row block ([K][Cin][4] -> j = k*Cin + ci); off: BYTE offsets into X
 * (ci*ldx + k*dil)*4; ldy: BYTE stride between output rows. Stores (does not accumulate). */
void pt_k4x16(float *Y, long ldy, const float *A, const float *X, const long *off, long nk, long ntile,
              const float *bias);
/* Same contract, 8 rows (two consecutive 4-row blocks: A0 then A1 = A0 + 4*nk) x (12*ntile) columns. */
void pt_k8x12(float *Y, long ldy, const float *A, const float *X, const long *off, long nk, long ntile,
              const float *bias);

/* In-place activations on one row segment, n % 4 == 0. */
void pt_snake_row(float *y, long n, float a, float ia);   /* y += sin(a*y)^2 * ia      (Snake, BigVGAN) */
void pt_silu_row(float *y, long n);                      /* y = y * sigmoid(y)                          */
void pt_tanh_row(float *y, long n);                      /* y = tanh(y)                                 */
void pt_sigmul_row(float *y, const float *g, long n);    /* y = y * sigmoid(g)        (GLU)             */

/* Depthwise conv row: y[t] = b + sum_k w[k] * x[t + k]  (x already points at column -pad), n % 4 == 0 */
void pt_dw_row(float *y, const float *x, long n, const float *w, long K, float b);

/* LayerNorm over channels for a channel-major block, fused with an optional residual add:
 *   if r: x[c][t] += r[c][t];  y[c][t] = (x[c][t]-mean_t)/sqrt(var_t+eps)*g[c] + b[c]
 * C rows, row stride ld (floats), n columns (n % 4 == 0). */
void pt_ln_res(float *x, const float *r, float *y, long C, long ld, long n, const float *g, const float *b,
               float eps);

/* Portable reference versions (always available, used by tests and non-aarch64 builds). */
void pt_ref_k4x16(float *Y, long ldy, const float *A, const float *X, const long *off, long nk, long ntile,
                  const float *bias);
void pt_ref_k8x12(float *Y, long ldy, const float *A, const float *X, const long *off, long nk, long ntile,
                  const float *bias);
void pt_ref_snake_row(float *y, long n, float a, float ia);
void pt_ref_silu_row(float *y, long n);
void pt_ref_tanh_row(float *y, long n);
void pt_ref_sigmul_row(float *y, const float *g, long n);
void pt_ref_dw_row(float *y, const float *x, long n, const float *w, long K, float b);
void pt_ref_ln_res(float *x, const float *r, float *y, long C, long ld, long n, const float *g, const float *b,
                   float eps);

/* ---------------- conv driver (runtime/src/conv.c) ---------------- */
enum { PT_ACT_NONE = 0, PT_ACT_SILU, PT_ACT_TANH, PT_ACT_SNAKE };

typedef struct {
  const float *wp, *b;    /* packed [Cout4/4][K][Cin][4], bias [Cout4] */
  int cout, cout4, cin, K, dil;
} pt_conv_w;

typedef struct {
  int act;                /* PT_ACT_* applied to (acc + bias) */
  const float *a, *ia;    /* snake params per output row (PT_ACT_SNAKE) */
  float scale;            /* y = act(acc) * scale  (1.0f = none) */
  const float *res;       /* y = res + y  (may alias Y: computed tile-by-tile) */
  long ldres;
} pt_epi;

int pt_conv_bind(pt_conv_w *c, const pt_weights *w, const char *wp_name, const char *b_name, int cout, int dil);
/* Y[co][t] = epi( b[co] + sum_k sum_ci W[co][ci][k] * X[ci][t + k*dil] ),  t in [0, n).
 * X points at the input column that output column 0's first tap reads (i.e. already shifted by -pad).
 * Columns are processed in 16-wide tiles: Y and X must have >= 16 columns of slack past n (+ the tap span),
 * extra columns receive garbage that callers never read. */
void pt_conv(const pt_conv_w *c, const float *X, long ldx, float *Y, long ldy, long n, const pt_epi *e);

/* Thread pool used to split big convs (rows) when a single stream wants low latency. */
void pt_pool_init(int nthreads);
void pt_pool_shutdown(void);
int pt_pool_threads(void);
void pt_pool_set_local(int n);   /* per-calling-thread: how many pool threads pt_conv may use (default 1) */
int pt_pool_get_local(void);
void pt_conv_set_kernel(int use8x12);
void pt_conv_set_pack(int on);           /* 0 off, 1: pack B panels for wide K=1 linears, 2 (default): also K>1 convs with >= 16 output rows */

/* ---------------- tiny .npy I/O (tests / tools) ---------------- */
typedef struct {
  int ndim;
  long dim[8];
  long n;
  char dtype;   /* 'f' float32, 'i' int64, 'b' bool */
  void *data;
} pt_npy;
int pt_npy_load(pt_npy *a, const char *path);
void pt_npy_free(pt_npy *a);
int pt_npy_save_f32(const char *path, const float *d, int ndim, const long *dim);

static inline double pt_now(void);
#include <time.h>
static inline double pt_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

#endif /* PT_H_ */
