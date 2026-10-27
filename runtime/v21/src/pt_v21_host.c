/* pt_v21_host.c - v2.1 graph interpreter over the generated tables (dependency-free C).
 *
 * Reads build/v21.ptk (mmap, fp16 payloads converted once), materialises every tensor the graph
 * produces, frees it at its last use (liveness computed by tools/gen_v21_graph.py), and evaluates
 * the 41 op kinds v2.1 uses with ONNX opset-17 semantics.  The conv/conv-transpose/matmul paths go
 * through pt_v21_conv / pt_v21_convT / pt_v21_matmul, which the CUDA build replaces with kernels;
 * the packed layouts they consume are the container's own layouts, so the parity run exercises the
 * layout contract the CUDA runtime is built on (docs/v21_layers.md).
 */
#include "pt_v21.h"
#include <pthread.h>
#include <sched.h>
#include "pt_graph_tab.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

static float tanh_tab65536[65537];
static int tanh_tab65536_ready = 0;
static void tanh_tab65536_init(void) {
  if (tanh_tab65536_ready)
    return;
  for (int i = 0; i <= 65536; i++) {
    double x = -8.0 + (double) i * (16.0 / 65536.0);
    tanh_tab65536[i] = (float) tanh(x);
  }
  tanh_tab65536_ready = 1;
}
static inline float tanh_tab65536_eval(float x) {
  if (x >= 8.0f)
    return 1.0f;
  if (x <= -8.0f)
    return -1.0f;
  float scaled = (x + 8.0f) * (65536.0f / 16.0f);
  int i = (int) scaled;
  if (i < 0)
    i = 0;
  if (i > 65535)
    i = 65535;
  float frac = scaled - (float) i;
  return tanh_tab65536[i] + frac * (tanh_tab65536[i + 1] - tanh_tab65536[i]);
}
/* Batched table evaluation, bit-identical to tanh_tab65536_eval (verified over 42M inputs incl.
 * +-8, NaN, +-inf, +-0): the index/fraction math and the interpolation run 4-wide in NEON, the
 * indices go through a small stack array so the table is gathered with plain integer-indexed
 * loads (moving NEON lanes to integer registers is what made a naive version barely faster), and
 * the +-8 saturation is a select. 1.5x faster than the scalar loop on the A57 (7.8 vs 11.9 ns).
 * sig = 1 evaluates the sigmoid form 0.5 * (tanh(x * 0.5) + 1), as un_op does. */
static void tanh_tab65536_eval_n(const float *a, float *o, int64_t n, int sig) {
  int64_t k = 0;
#if defined(__aarch64__)
  const float32x4_t c8 = vdupq_n_f32(8.0f), cs = vdupq_n_f32(65536.0f / 16.0f), m8 = vdupq_n_f32(-8.0f);
  const float32x4_t p1 = vdupq_n_f32(1.0f), n1 = vdupq_n_f32(-1.0f), half = vdupq_n_f32(0.5f);
  const int32x4_t z = vdupq_n_s32(0), mx = vdupq_n_s32(65535);
  for (; k + 16 <= n; k += 16) {
    int32_t idx[16] __attribute__((aligned(16)));
    float lo[16] __attribute__((aligned(16))), hi[16] __attribute__((aligned(16)));
    float32x4_t fr[4], x[4];
    for (int j = 0; j < 4; j++) {
      x[j] = vld1q_f32(a + k + 4 * j);
      if (sig) x[j] = vmulq_f32(x[j], half);
      float32x4_t sc = vmulq_f32(vaddq_f32(x[j], c8), cs);
      int32x4_t i = vminq_s32(vmaxq_s32(vcvtq_s32_f32(sc), z), mx);
      fr[j] = vsubq_f32(sc, vcvtq_f32_s32(i));
      vst1q_s32(idx + 4 * j, i);
    }
    for (int j = 0; j < 16; j++) {
      const float *t = tanh_tab65536 + idx[j];
      lo[j] = t[0];
      hi[j] = t[1];
    }
    for (int j = 0; j < 4; j++) {
      float32x4_t l = vld1q_f32(lo + 4 * j), h = vld1q_f32(hi + 4 * j);
      float32x4_t r = vfmaq_f32(l, fr[j], vsubq_f32(h, l));
      r = vbslq_f32(vcgeq_f32(x[j], c8), p1, r);
      r = vbslq_f32(vcleq_f32(x[j], m8), n1, r);
      if (sig) r = vmulq_f32(half, vaddq_f32(r, p1));
      vst1q_f32(o + k + 4 * j, r);
    }
  }
#endif
  for (; k < n; k++)
    o[k] = sig ? 0.5f * (tanh_tab65536_eval(a[k] * 0.5f) + 1.0f) : tanh_tab65536_eval(a[k]);
}

#define PT_MAGIC_V21 0x50544B31u   /* "PTK1", same magic as runtime/include/pt.h */

#pragma pack(push, 1)
typedef struct {
  uint32_t magic, version, n_tensor, reserved;
  int64_t data_bytes;
} v21_hdr;
#pragma pack(pop)
#define V21_HDR_SIZE 24
#define V21_REC_SIZE 136   /* "<64si4x6qqq": name[64], ndim i32, pad 4, dim[6] i64, nelem i64, off i64 */

struct pt_ctx *g_ctx;
int g_node = -1, g_op = -1;
int g_trace = -1;
/* How many nodes read each tensor, and the node index of the sole reader: the fusion decision
 * needs "this conv's output goes to exactly one elementwise node, and nothing else touches it". */
static int g_n_cons[PT_N_TENSOR];
/* every consumer of a tensor (up to PT_V21_MAX_CONS) and whether all of them are ops the device
 * residency project can chain through (Add / a conv reading in[0]). Multi-consumer tensors are fine
 * on the device -- readers do not conflict -- the constraint is liveness, which the interpreter
 * already tracks (pt_graph_last_use), so the residency sub-arena releases at tv_free. */
#define PT_V21_MAX_CONS 4
static int g_cons_list[PT_N_TENSOR][PT_V21_MAX_CONS];
static signed char g_cons_dev[PT_N_TENSOR];
static int g_cons_node[PT_N_TENSOR];
static int g_prod[PT_N_TENSOR];   /* producing node: the residency recursion walks producers */
static int g_cons_built;
static int g_dump_tensor = -1;
void pt_v21_trace(int idx) { g_trace = idx; }
FILE *g_shapes = 0;
void pt_v21_shape_log(FILE *f) { g_shapes = f; }



/* struct pt_ctx lives in pt_v21.h (the CUDA kernels read it too) */


#define V21TV struct pt_ctx
static int dt_sz(int dt) {
  switch (dt) {
    case DT_F32: return 4;
    case DT_I64: return 8;
    case DT_I32: return 4;
    case DT_F64: return 8;
    case DT_F16: return 2;
    case DT_BOOL: return 1;
    default: return 0;
  }
}

static int64_t tv_n(const pt_tv *t) {
  int64_t n = 1;
  for (int i = 0; i < t->ndim; i++) n *= t->d[i];
  return n;
}

/* ---- tensor buffer pool -----------------------------------------------------------------------
 * Every op allocates its output and the interpreter frees each tensor at its last use, so a request
 * allocates and frees its whole working set (~4000 buffers). glibc returns large blocks to the
 * kernel (munmap / arena trim), so every request re-faulted that memory: ~8300 page faults
 * (~33 MB of kernel-zeroed pages) per request, measured. The pool keeps freed buffers on per-context
 * free lists (a context is only ever used by one thread: no locking) in size classes of four steps
 * per octave (<= 19% slack), so the working set is faulted once and reused. Small buffers bypass
 * it. Each buffer carries a 64-byte header with its class. PT_V21_POOL_MB caps the bytes kept per
 * context (default 32: the measured per-worker working set fits; 0 disables pooling). */
#define PT_POOL_MIN 65536
#define PT_POOL_HDR 64
/* Buffers are reference counted (header word 1) so a zero-copy slice can keep its source alive:
 * the owner tensor and every view hold one reference; the buffer is recycled at the last release. */
static void buf_ref_init(void *p) { ((int *) ((char *) p - PT_POOL_HDR))[1] = 1; }
static void buf_ref_add(void *p) { ((int *) ((char *) p - PT_POOL_HDR))[1]++; }
static size_t pool_cls_size(int k) { return (size_t) (PT_POOL_MIN / 4) * (size_t) (4 + (k & 3)) << (k >> 2); }
static int pool_cls(size_t bytes) {
  int k = 0;
  while (k < 63 && pool_cls_size(k) < bytes) k++;
  return k;
}
static long pool_cap_bytes(void) {
  static long cap = -1;
  if (cap < 0) cap = (getenv("PT_V21_POOL_MB") ? atol(getenv("PT_V21_POOL_MB")) : 32) << 20;
  return cap;
}
static void *buf_get(pt_ctx *c, size_t bytes) {
  int k = -1;
  size_t alloc = bytes;
  if (bytes >= PT_POOL_MIN && pool_cap_bytes() > 0) {
    k = pool_cls(bytes);
    alloc = pool_cls_size(k);
    void *h = c->pool[k];
    if (h) {                                   /* reuse: pop the class free list */
      c->pool[k] = *(void **) ((char *) h + 8);
      c->pool_bytes -= (long) alloc;
      buf_ref_init((char *) h + PT_POOL_HDR);
      return (char *) h + PT_POOL_HDR;
    }
  }
  char *h = (char *) aligned_alloc(64, ((alloc + 63) & ~(size_t) 63) + PT_POOL_HDR);
  if (!h) return NULL;
  *(int *) h = k;                              /* class, -1 = not pooled */
  ((int *) h)[1] = 1;
  return h + PT_POOL_HDR;
}

static void buf_put(pt_ctx *c, void *p) {
  char *h = (char *) p - PT_POOL_HDR;
  if (--((int *) h)[1] > 0)                    /* still referenced by a view (or its owner) */
    return;
  int k = *(int *) h;
  if (k >= 0 && c->pool_bytes + (long) pool_cls_size(k) <= pool_cap_bytes()) {
    *(void **) (h + 8) = c->pool[k];           /* push onto the class free list */
    c->pool[k] = h;
    c->pool_bytes += (long) pool_cls_size(k);
    return;
  }
  free(h);
}
static void pool_drain(pt_ctx *c) {
  for (int k = 0; k < 64; k++)
    while (c->pool[k]) {
      char *h = (char *) c->pool[k];
      c->pool[k] = *(void **) (h + 8);
      free(h);
    }
  c->pool_bytes = 0;
}

static void tv_free(pt_ctx *c, pt_tv *t) {
  if (t->alive && t->p) {
    if (t->base)
      buf_put(c, t->base);                     /* drops one reference: owner or view */
    if (!t->view)
      c->host_bytes_live -= (long) t->n * dt_sz(t->dt);
  }
  t->base = NULL;
  t->view = 0;
  if (t->dev_res) {
    /* the interpreter frees a tensor exactly at its last use, which is exactly when its device
     * buffer may be recycled (the residency sub-arena ignores pointers it does not own) */
    pt_v21_dev_free(c, t->dev_p);
    if (c->dev_live > 0)
      c->dev_live--;
    t->dev_res = 0;
    t->dev_p = NULL;
  }
  t->p = 0;
  t->alive = 0;
  t->ndim = 0;
  t->n = 0;
}

static int host_rows_on(void) {
  static int v = -1;
  if (v < 0) v = getenv("PT_V21_HOSTROWS") ? atoi(getenv("PT_V21_HOSTROWS")) : 1;
  return v;
}

/* tv_alloc_zero: 1 = zero-fill a fresh buffer (shape ops rely on it), 0 = the caller writes every
 * element, so the memset is skipped. Called with 0 by the conv/convT nodes, whose kernels fill the
 * whole padded output buffer (the D2H copy or the cout trim covers every element). */
static int tv_alloc_nz(pt_ctx *c, pt_tv *t, int dt, const int64_t *d, int ndim, int zero);

static int tv_alloc(pt_ctx *c, pt_tv *t, int dt, const int64_t *d, int ndim) {
  return tv_alloc_nz(c, t, dt, d, ndim, 1);
}

static int tv_alloc_nz(pt_ctx *c, pt_tv *t, int dt, const int64_t *d, int ndim, int zero) {
  int64_t n = 1;
  for (int i = 0; i < ndim; i++) n *= d[i];
  t->dt = dt;
  t->ndim = ndim;
  t->n = n;
  t->dev_res = 0;   /* a fresh tensor never refers to a chained device buffer */
  t->dev_p = NULL;
  for (int i = 0; i < PT_V21_MAX_RANK; i++) t->d[i] = i < ndim ? d[i] : 1;
  size_t bytes = (size_t) (n * dt_sz(dt));
  if (bytes == 0) bytes = dt_sz(dt);
  if (n > 400000000 || n < 0) {
    fprintf(stderr, "pt_v21: node %d (%s) op %d asked for %lld elements [", g_node, 
            g_node >= 0 ? pt_graph_tnames[pt_ndefs[g_node].out[0]] : "?", g_op, (long long) n);
    for (int i = 0; i < ndim; i++) fprintf(stderr, " %lld", (long long) d[i]);
    fprintf(stderr, " ]\n");
    return -1;
  }
  t->p = buf_get(c, bytes);
  t->base = t->p;
  t->view = 0;
  if (!t->p) {
    c->n_alloc_fail++;
    return -1;
  }
  c->host_bytes_live += (long) bytes;
  c->host_alloc_count++;
  if (c->host_bytes_live > c->host_bytes_peak)
    c->host_bytes_peak = c->host_bytes_live;
  if (zero)
    memset(t->p, 0, bytes);
  t->alive = 1;
  t->dev = 0;
  return 0;
}

/* shape of a tensor flattened to a scalar (0-d): returns its single value */
static double tv_scalar_f(const pt_tv *t) {
  if (!t->p) return 0.0;
  if (t->dt == DT_F32) return (double) ((const float *) t->p)[0];
  if (t->dt == DT_I64) return (double) ((const int64_t *) t->p)[0];
  if (t->dt == DT_I32) return (double) ((const int32_t *) t->p)[0];
  if (t->dt == DT_BOOL) return (double) ((const uint8_t *) t->p)[0];
  return 0.0;
}

/* ---------------- broadcast helpers ---------------- */
typedef struct {
  int ndim;
  int64_t d[PT_V21_MAX_RANK];
  int64_t st[PT_V21_MAX_RANK];   /* stride in elements; 0 = broadcast dim */
} bcast;

static int bcast_dims(const pt_tv *a, const pt_tv *b, const pt_tv *e, int64_t *d, int *ndim) {
  int n = a->ndim;
  if (b && b->ndim > n) n = b->ndim;
  if (e && e->ndim > n) n = e->ndim;
  if (n > PT_V21_MAX_RANK) return -1;
  for (int i = 0; i < n; i++) {
    int64_t da = i - n + a->ndim >= 0 && i - n + a->ndim < a->ndim ? a->d[i - n + a->ndim] : 1;
    int64_t db = b ? (i - n + b->ndim >= 0 && i - n + b->ndim < b->ndim ? b->d[i - n + b->ndim] : 1) : 1;
    int64_t de = e ? (i - n + e->ndim >= 0 && i - n + e->ndim < e->ndim ? e->d[i - n + e->ndim] : 1) : 1;
    int64_t mx = da > db ? da : db;
    if (de > mx) mx = de;
    if (mx != 1 && ((da != 1 && da != mx) || (db != 1 && db != mx) || (de != 1 && de != mx))) return -1;
    d[i] = mx;
  }
  *ndim = n;
  return 0;
}

static void bcast_prep(bcast *b, const pt_tv *t, const int64_t *d, int ndim) {
  b->ndim = ndim;
  for (int i = 0; i < ndim; i++) b->d[i] = d[i];
  int64_t st = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    int64_t td = i - ndim + t->ndim >= 0 && i - ndim + t->ndim < t->ndim ? t->d[i - ndim + t->ndim] : 1;
    b->st[i] = td == 1 ? 0 : st;
    st *= td;
  }
}

/* decompose a flat output index into element offsets for the three broadcast operands */
#define BC_LOOP(BODY)                                                        \
  for (int64_t i = 0; i < O->n; i++) {                                       \
    int64_t r = i, oa = 0, ob = 0, oe = 0;                                   \
    for (int q = O->ndim - 1; q >= 0; q--) {                                 \
      int64_t idx = r % O->d[q];                                             \
      r /= O->d[q];                                                          \
      oa += idx * ba.st[q];                                                  \
      ob += idx * bb.st[q];                                                  \
      oe += idx * be.st[q];                                                  \
    }                                                                        \
    BODY;                                                                    \
  }

static int bin_op(pt_ctx *c, int op, const pt_tv *A, const pt_tv *B, pt_tv *O) {
  int ndim;
  int64_t d[PT_V21_MAX_RANK];
  if (bcast_dims(A, B, 0, d, &ndim) < 0) return -1;
  int isbool = (op == PT_EQUAL || op == PT_LESS || op == PT_GREATER || op == PT_GEQ || op == PT_LEQ ||
               op == PT_AND || op == PT_OR || op == PT_XOR);
  int dt = isbool ? DT_BOOL : (A->dt == DT_I64 && B->dt == DT_I64 ? DT_I64 : DT_F32);
  if (tv_alloc_nz(c, O, dt, d, ndim, 0) < 0) return -1;
  bcast ba, bb, be;
  bcast_prep(&ba, A, d, ndim);
  bcast_prep(&bb, B, d, ndim);
  pt_tv dummy = *A;
  dummy.ndim = 0;
  dummy.n = 1;
  dummy.p = 0;
  be = ba;
  be.st[0] = 0;
  for (int i = 0; i < PT_V21_MAX_RANK; i++) be.st[i] = 0;
  int adt = A->dt, bdt = B->dt;
  if (adt == DT_I64 && bdt == DT_I64) {
    const int64_t *a = (const int64_t *) A->p, *b = (const int64_t *) B->p;
    if (isbool) {
      uint8_t *o = (uint8_t *) O->p;
      BC_LOOP(o[i] = (uint8_t) ((op == PT_EQUAL) ? (a[oa] == b[ob])
                                    : op == PT_LESS ? (a[oa] < b[ob])
                                    : op == PT_GREATER ? (a[oa] > b[ob])
                                    : op == PT_GEQ ? (a[oa] >= b[ob])
                                    : op == PT_LEQ ? (a[oa] <= b[ob])
                                    : op == PT_AND ? ((a[oa] != 0) && (b[ob] != 0))
                                    : op == PT_OR ? ((a[oa] != 0) || (b[ob] != 0)) : (a[oa] != 0) ^ (b[ob] != 0)));
    } else {
      int64_t *o = (int64_t *) O->p;
      BC_LOOP(o[i] = (op == PT_ADD) ? a[oa] + b[ob]
                                    : op == PT_SUB ? a[oa] - b[ob]
                                    : op == PT_MUL ? a[oa] * b[ob]
                                    : op == PT_DIV ? (int64_t) trunc((double) a[oa] / (double) b[ob])
                                    : op == PT_MAX ? (a[oa] > b[ob] ? a[oa] : b[ob])
                                    : op == PT_MIN ? (a[oa] < b[ob] ? a[oa] : b[ob])
                                    : op == PT_POW ? (int64_t) pow((double) a[oa], (double) b[ob]) : 0);
    }
  } else {
    double av, bv;
#define AV (adt == DT_F32 ? (double) ((const float *) A->p)[oa]                                    \
          : adt == DT_I64 ? (double) ((const int64_t *) A->p)[oa]                                 \
          : adt == DT_I32 ? (double) ((const int32_t *) A->p)[oa] : (double) ((const uint8_t *) A->p)[oa])
#define BV (bdt == DT_F32 ? (double) ((const float *) B->p)[ob]                                    \
          : bdt == DT_I64 ? (double) ((const int64_t *) B->p)[ob]                                 \
          : bdt == DT_I32 ? (double) ((const int32_t *) B->p)[ob] : (double) ((const uint8_t *) B->p)[ob])
    if (isbool) {
      uint8_t *o = (uint8_t *) O->p;
      BC_LOOP(o[i] = (uint8_t) ((op == PT_EQUAL) ? (AV == BV)
                                    : op == PT_LESS ? (AV < BV)
                                    : op == PT_GREATER ? (AV > BV)
                                    : op == PT_GEQ ? (AV >= BV)
                                    : op == PT_LEQ ? (AV <= BV)
                                    : op == PT_AND ? ((AV != 0) && (BV != 0))
                                    : op == PT_OR ? ((AV != 0) || (BV != 0)) : (AV != 0) ^ (BV != 0)));
    } else if (!isbool && dt == DT_F32 && adt == DT_F32 && bdt == DT_F32 && host_rows_on() &&
               (op == PT_ADD || op == PT_SUB || op == PT_MUL || op == PT_DIV || op == PT_MAX ||
                op == PT_MIN) && ndim >= 1 && O->n > 0 &&
               !(A->n == O->n && (B->n == O->n || B->n == 1))) {
      /* general fp32 broadcast: walk the output row by row with an odometer over the outer axes;
       * each operand's innermost stride is 0 (broadcast) or 1, so the inner loops are plain
       * vectorizable loops instead of a per-element div/mod decode through double. fp32 +,-,*,/,
       * max, min are correctly rounded, so results are bit-identical to the double path. */
      const float *a = (const float *) A->p, *b = (const float *) B->p;
      float *o = (float *) O->p;
      int64_t inner = d[ndim - 1], sa = ba.st[ndim - 1], sb = bb.st[ndim - 1];
      int64_t ix[PT_V21_MAX_RANK] = {0}, pa = 0, pb = 0;
      int64_t nrows = O->n / inner;
      for (int64_t ro = 0; ro < nrows; ro++) {
        const float *ar = a + pa, *br = b + pb;
        float *orow = o + ro * inner;
#define PT_ROWOP(EXPR)                                                                           \
  if (sa && sb) for (int64_t k = 0; k < inner; k++) { float x = ar[k], y = br[k]; orow[k] = EXPR; } \
  else if (sa) { float y = br[0]; for (int64_t k = 0; k < inner; k++) { float x = ar[k]; orow[k] = EXPR; } } \
  else if (sb) { float x = ar[0]; for (int64_t k = 0; k < inner; k++) { float y = br[k]; orow[k] = EXPR; } } \
  else { float x = ar[0], y = br[0]; float v = EXPR; for (int64_t k = 0; k < inner; k++) orow[k] = v; }
        if (op == PT_ADD) { PT_ROWOP(x + y) }
        else if (op == PT_SUB) { PT_ROWOP(x - y) }
        else if (op == PT_MUL) { PT_ROWOP(x * y) }
        else if (op == PT_DIV) { PT_ROWOP(x / y) }
        else if (op == PT_MAX) { PT_ROWOP(x > y ? x : y) }
        else { PT_ROWOP(x < y ? x : y) }
#undef PT_ROWOP
        for (int j = ndim - 2; j >= 0; j--) {
          ix[j]++;
          pa += ba.st[j];
          pb += bb.st[j];
          if (ix[j] < d[j])
            break;
          pa -= ba.st[j] * d[j];
          pb -= bb.st[j] * d[j];
          ix[j] = 0;
        }
      }
    } else if (!isbool && dt == DT_F32 && adt == DT_F32 && bdt == DT_F32 &&
               (op == PT_ADD || op == PT_SUB || op == PT_MUL || op == PT_DIV || op == PT_MAX ||
                op == PT_MIN) &&
               A->n == O->n && (B->n == O->n || B->n == 1)) {
    /* elementwise fast path: same-shape (or scalar-broadcast) fp32 on contiguous buffers. A plain
     * loop over 4-byte lanes is what gcc -O3 turns into NEON on the A57; the strided BC_LOOP below
     * cannot be vectorized and was 349 ms of a 4.4 s stream on ADD alone.
     * Bit-exactness: for +,-,*,max,min an fp32 op is correctly rounded, which is exactly what the
     * double-then-round path computes, so the outputs are unchanged bit for bit. */
      const float *a = (const float *) A->p, *b = (const float *) B->p;
      float *o = (float *) O->p;
      int64_t n = O->n;
      int bcast_scalar = (B->n == 1);
      float bs = bcast_scalar ? b[0] : 0.0f;
      if (op == PT_ADD) {
        if (bcast_scalar)
          for (int64_t i = 0; i < n; i++) o[i] = a[i] + bs;
        else
          for (int64_t i = 0; i < n; i++) o[i] = a[i] + b[i];
      } else if (op == PT_SUB) {
        if (bcast_scalar)
          for (int64_t i = 0; i < n; i++) o[i] = a[i] - bs;
        else
          for (int64_t i = 0; i < n; i++) o[i] = a[i] - b[i];
      } else if (op == PT_MUL) {
        if (bcast_scalar)
          for (int64_t i = 0; i < n; i++) o[i] = a[i] * bs;
        else
          for (int64_t i = 0; i < n; i++) o[i] = a[i] * b[i];
      } else if (op == PT_DIV) {
        if (bcast_scalar)
          for (int64_t i = 0; i < n; i++) o[i] = a[i] / bs;
        else
          for (int64_t i = 0; i < n; i++) o[i] = a[i] / b[i];
      } else if (op == PT_MAX) {
        if (bcast_scalar)
          for (int64_t i = 0; i < n; i++) o[i] = a[i] > bs ? a[i] : bs;
        else
          for (int64_t i = 0; i < n; i++) o[i] = a[i] > b[i] ? a[i] : b[i];
      } else {
        if (bcast_scalar)
          for (int64_t i = 0; i < n; i++) o[i] = a[i] < bs ? a[i] : bs;
        else
          for (int64_t i = 0; i < n; i++) o[i] = a[i] < b[i] ? a[i] : b[i];
      }
    } else {
      float *o = (float *) O->p;
      BC_LOOP(av = AV; bv = BV;
              o[i] = (float) ((op == PT_ADD) ? av + bv
                                    : op == PT_SUB ? av - bv
                                    : op == PT_MUL ? av * bv
                                    : op == PT_DIV ? av / bv
                                    : op == PT_POW ? pow(av, bv)
                                    : op == PT_MAX ? (av > bv ? av : bv)
                                    : op == PT_MIN ? (av < bv ? av : bv) : 0.0));
    }
#undef AV
#undef BV
  }
  (void) dummy;
  return 0;
}

static int un_op(pt_ctx *c, int op, const pt_tv *A, const pt_tv *G, pt_tv *O, float alpha) {
  int64_t d[PT_V21_MAX_RANK];
  memcpy(d, A->d, sizeof(d));
  int dt = (A->dt == DT_I64) ? DT_I64 : (A->dt == DT_I32 ? DT_I32 : DT_F32);
  if (tv_alloc_nz(c, O, dt, d, A->ndim, 0) < 0) return -1;
  int64_t n = A->n;
  if (A->dt == DT_I64 || A->dt == DT_I32) {
    const int64_t *a = (const int64_t *) A->p;
    int64_t *o = (int64_t *) O->p;
    for (int64_t i = 0; i < n; i++) {
      double v = (double) a[i];
      o[i] = (int64_t) ((op == PT_NEG) ? -v
                                   : op == PT_ABS ? fabs(v)
                                   : op == PT_FLOOR ? floor(v)
                                   : op == PT_CEIL ? ceil(v)
                                   : op == PT_ROUND ? rint(v)
                                   : op == PT_SQRT ? sqrt(v)
                                   : op == PT_SIGN ? (v > 0 ? 1 : v < 0 ? -1 : 0) : v);
    }
    return 0;
  }
  const float *a = (const float *) A->p;
  const float *g = G && G->p ? (const float *) G->p : 0;
  float *o = (float *) O->p;
  int64_t gn = G ? G->n : 1;
  /* Bulk fp32 transcendentals through glibc's single-precision expf/tanhf (<1 ulp) instead of the
   * double-precision switch. Small tensors keep the double path: they feed Floor/Cast shape
   * arithmetic, where a last-ulp difference could flip an integer length. */
  static int ff = -1;
  if (ff < 0) ff = getenv("PT_V21_F32MATH") ? atoi(getenv("PT_V21_F32MATH")) : 1;
  static int ft = -1;
  if (ft < 0) ft = getenv("PT_V21_FASTTRIG") ? atoi(getenv("PT_V21_FASTTRIG")) : 1;
  if (!g && ff && n >= 4096 && (op == PT_EXP || op == PT_TANH || op == PT_SIGMOID || op == PT_SIN || op == PT_COS)) {
    if (op == PT_EXP)
      for (int64_t i = 0; i < n; i++) o[i] = expf(a[i]);
    else if (op == PT_SIN)
      for (int64_t i = 0; i < n; i++) o[i] = sinf(a[i]);
    else if (op == PT_COS)
      for (int64_t i = 0; i < n; i++) o[i] = cosf(a[i]);
    else if (ft && op == PT_TANH) {
      tanh_tab65536_init();
      tanh_tab65536_eval_n(a, o, n, 0);
    }
    else if (op == PT_TANH)
      for (int64_t i = 0; i < n; i++) o[i] = tanhf(a[i]);
    else if (ft && op == PT_SIGMOID) {
      tanh_tab65536_init();
      tanh_tab65536_eval_n(a, o, n, 1);
    }
    else if (op == PT_SIGMOID)
      for (int64_t i = 0; i < n; i++) o[i] = 1.0f / (1.0f + expf(-a[i]));
    else if (op == PT_TANH)
      for (int64_t i = 0; i < n; i++) o[i] = tanhf(a[i]);
    else
      for (int64_t i = 0; i < n; i++) o[i] = 1.0f / (1.0f + expf(-a[i]));
    return 0;
  }
  if (!g && (op == PT_LEAKYRELU || op == PT_RELU || op == PT_NEG || op == PT_ABS)) {
    /* Pure arithmetic elementwise. A plain fp32 loop does not auto-vectorize on gcc 8 (control
     * flow), so use NEON intrinsics on aarch64 for a 4-wide branchless select; bit-exact: the
     * same fp32 operations and sign tests, just four at a time. */
    int64_t n2 = A->n;
#if defined(__aarch64__)
    int64_t i = 0;
    if (op == PT_LEAKYRELU) {
      float32x4_t va = vdupq_n_f32(alpha);
      for (; i + 4 <= n2; i += 4) {
        float32x4_t v = vld1q_f32(a + i);
        uint32x4_t m = vcgeq_f32(v, vdupq_n_f32(0.0f));
        float32x4_t r = vmulq_f32(v, va);
        vst1q_f32(o + i, vbslq_f32(m, v, r));
      }
    } else if (op == PT_RELU) {
      for (; i + 4 <= n2; i += 4) {
        float32x4_t v = vld1q_f32(a + i);
        uint32x4_t m = vcgtq_f32(v, vdupq_n_f32(0.0f));
        vst1q_f32(o + i, vbslq_f32(m, v, vdupq_n_f32(0.0f)));
      }
    } else if (op == PT_NEG) {
      for (; i + 4 <= n2; i += 4) {
        float32x4_t v = vld1q_f32(a + i);
        vst1q_f32(o + i, vnegq_f32(v));
      }
    } else {
      for (; i + 4 <= n2; i += 4) {
        float32x4_t v = vld1q_f32(a + i);
        vst1q_f32(o + i, vabsq_f32(v));
      }
    }
    for (; i < n2; i++) {
      if (op == PT_LEAKYRELU) o[i] = a[i] >= 0.0f ? a[i] : alpha * a[i];
      else if (op == PT_RELU) o[i] = a[i] > 0.0f ? a[i] : 0.0f;
      else if (op == PT_NEG) o[i] = -a[i];
      else o[i] = a[i] < 0.0f ? -a[i] : a[i];
    }
#else
    if (op == PT_LEAKYRELU) {
      for (int64_t i = 0; i < n2; i++) o[i] = a[i] >= 0.0f ? a[i] : alpha * a[i];
    } else if (op == PT_RELU) {
      for (int64_t i = 0; i < n2; i++) o[i] = a[i] > 0.0f ? a[i] : 0.0f;
    } else if (op == PT_NEG) {
      for (int64_t i = 0; i < n2; i++) o[i] = -a[i];
    } else {
      for (int64_t i = 0; i < n2; i++) o[i] = a[i] < 0.0f ? -a[i] : a[i];
    }
#endif
    return 0;
  }
  for (int64_t i = 0; i < n; i++) {
    double v = (double) a[i];
    double gv = g ? (double) g[gn == 1 ? 0 : i % gn] : 0.0;
    double r = v;
    switch (op) {
      case PT_NEG: r = -v; break;
      case PT_EXP: r = exp(v); break;
      case PT_SIN: r = sin(v); break;
      case PT_COS: r = cos(v); break;
      case PT_TANH: r = tanh(v); break;
      case PT_SIGMOID: r = 1.0 / (1.0 + exp(-v)); break;
      case PT_RELU: r = v > 0 ? v : 0.0; break;
      case PT_LEAKYRELU: r = v >= 0 ? v : (double) alpha * v; break;
      case PT_ABS: r = fabs(v); break;
      case PT_FLOOR: r = floor(v); break;
      case PT_CEIL: r = ceil(v); break;
      case PT_ROUND: r = rint(v); break;
      case PT_SQRT: r = sqrt(v); break;
      case PT_SIGN: r = v > 0 ? 1 : v < 0 ? -1 : 0; break;
      case PT_RECIPROCAL: r = 1.0 / v; break;
      case PT_LOG: r = log(v); break;
      case PT_SOFTPLUS: r = log(1.0 + exp(v)); break;
      case PT_ERF: r = erf(v); break;
      case PT_CLIP: r = v; if (G) r = v > gv ? v : gv; break;
      default: r = v; break;
    }
    o[i] = (float) r;
  }
  return 0;
}

/* ---------------- container ---------------- */
static int half2float_bits(uint16_t h) {
  uint32_t s = (uint32_t) (h >> 15) & 1u, e = (uint32_t) (h >> 10) & 0x1fu, m = (uint32_t) h & 0x3ffu;
  uint32_t r;
  if (e == 0) {
    if (m == 0) {
      r = s << 31;
    } else {
      /* fp16 subnormal: value = m/1024 * 2^-14.  Normalise it through fp32 so the tiny weights
       * (v2.1's fp16 decoder has them) keep their magnitude instead of flushing to zero. */
      float v = (float) m / 1024.0f * 6.103515625e-05f;   /* 2^-14 */
      uint32_t u;
      memcpy(&u, &v, 4);
      r = (u & 0x7fffffffu) | (s << 31);
    }
  } else if (e == 31) {
    r = (s << 31) | 0x7f800000u | (m << 13);
  } else {
    r = (s << 31) | ((e + 112u) << 23) | (m << 13);
  }
  return (int) r;
}



/* ------------------------------------------- shared host-side container payloads
 * Every stream needs the same container weights converted to fp32 (the decoder's 39 fp16 records
 * are 13.6M elements = 54.5 MB each). Four streams on one box do not need four copies: the
 * conversion is deterministic from the container bytes, so the first context builds it, the rest
 * alias it, and the last one to close frees it. Guarded by a mutex because the multi-stream driver
 * opens and closes contexts from different threads, and keyed by the container path so two
 * different containers in one process never share a payload. */
typedef struct {
  float *p;
  int ref;
} pt_hrec;
static pt_hrec *g_hcache;
static char *g_harena;
static size_t g_harena_bytes, g_harena_used;
static int g_harena_failed;
static size_t g_hcache_n;
static char g_hcache_path[512];
static pthread_mutex_t g_hlock = PTHREAD_MUTEX_INITIALIZER;

static float *host_cont_shared(int k, const char *path, int64_t nel) {
  float *out = NULL;
  pthread_mutex_lock(&g_hlock);
  if (!g_hcache_path[0])
    snprintf(g_hcache_path, sizeof(g_hcache_path), "%s", path);
  if (strcmp(g_hcache_path, path) != 0) {
    pthread_mutex_unlock(&g_hlock);   /* a different container: never share */
    return NULL;
  }
  if ((size_t) k >= g_hcache_n) {
    size_t nn = (size_t) k + 64;
    pt_hrec *nb = (pt_hrec *) realloc(g_hcache, nn * sizeof(pt_hrec));
    if (nb) {
      memset(nb + g_hcache_n, 0, (nn - g_hcache_n) * sizeof(pt_hrec));
      g_hcache = nb;
      g_hcache_n = nn;
    }
  }
  if ((size_t) k < g_hcache_n) {
    if (g_hcache[k].p && g_hcache[k].ref > 0) {
      g_hcache[k].ref++;
      out = g_hcache[k].p;
    }
  }
  (void) nel;
  pthread_mutex_unlock(&g_hlock);
  return out;
}

static void host_cont_shared_claim(int k, float *p) {
  pthread_mutex_lock(&g_hlock);
  if ((size_t) k < g_hcache_n) {
    g_hcache[k].p = p;
    g_hcache[k].ref = 1;
  }
  pthread_mutex_unlock(&g_hlock);
}

static void host_cont_shared_put(int k, float *p, int owned) {
  pthread_mutex_lock(&g_hlock);
  if (owned && (size_t) k < g_hcache_n && g_hcache[k].p == p) {
    if (--g_hcache[k].ref <= 0) {
      /* arena-backed payloads stay mapped (the arena is per process, freed at exit); only a
       * fallback malloc'd record is released here */
      if (!g_harena || (char *) p < g_harena || (char *) p >= g_harena + g_harena_bytes)
        free(p);
      g_hcache[k].p = NULL;
      g_hcache[k].ref = 0;
    }
  }
  pthread_mutex_unlock(&g_hlock);
}

static uint32_t h2f_table[65536];
static int h2f_table_ready = 0;
static void h2f_table_init(void) {
  if (h2f_table_ready)
    return;
  for (int h = 0; h < 65536; h++)
    h2f_table[h] = (uint32_t) half2float_bits((uint16_t) h);
  h2f_table_ready = 1;
}

/* One arena for the fp32 container payloads instead of 336 separate mallocs. The container index
 * gives every record's size up front, so the block can be mapped with MAP_POPULATE: the kernel then
 * fills 84 MB with a handful of page-table operations instead of ~21k demand faults during a
 * copy loop (measured: the record pass is 159 ms of the ~500 ms a cold process spends before it
 * starts inference). Records are 64-byte aligned so the NEON paths keep their alignment. */
static int harena_reserve(size_t bytes) {
  if (g_harena || g_harena_failed || bytes == 0)
    return g_harena ? 0 : -1;
  void *p = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  if (p == MAP_FAILED) {
    g_harena_failed = 1;
    return -1;
  }
  madvise(p, bytes, MADV_HUGEPAGE);   /* best effort: fewer TLB entries on the A57 */
  g_harena = (char *) p;
  g_harena_bytes = bytes;
  g_harena_used = 0;
  return 0;
}

static float *harena_alloc(size_t bytes) {
  size_t off = (g_harena_used + 63) & ~(size_t) 63;
  if (!g_harena || off + bytes > g_harena_bytes)
    return NULL;
  g_harena_used = off + bytes;
  return (float *) (g_harena + off);
}

/* One arena for all fp32 payloads: its size is the sum of the container records that the graph
 * actually references, computed from the record index (no allocation per record). */
static int reserve_payload_arena(pt_ctx *c, uint32_t n_tensor) {
  size_t need = 0;
  for (uint32_t i = 0; i < n_tensor; i++) {
    const uint8_t *r = (const uint8_t *) c->map + V21_HDR_SIZE + (size_t) i * V21_REC_SIZE;
    char nm[65];
    memcpy(nm, r, 64);
    nm[64] = 0;
    if (!nm[0])
      continue;
    int64_t nel;
    memcpy(&nel, r + 64 + 4 + 4 + 48, 8);
    if (nel <= 0)
      continue;
    need += (((size_t) nel * 4 + 64) + 63) & ~(size_t) 63;
  }
  return harena_reserve(need);
}

/* ---- parallel record pass (PT_V21_OPEN_PAR=1) -----------------------------------------------
 * The serialized open pass is ONE pass of work: the first context materializes every fp32 payload
 * (84 MB memcpy + 27 MB of fp16 LUT decode, ~100 ms measured) while the other three stream
 * threads wait on the lock, and the whole thing sits in the ~390 ms serialized startup that runs
 * before any inference. The payloads are shared, so the work can be split instead: prepare the
 * record -> ctab map, the payload arena, the shared cache and the fp16 LUT single-threaded under
 * open_lock; then every thread claims record indices with one atomic counter and materializes
 * them (the only lock is the arena bump, held for microseconds); a completion barrier, and each
 * context aliases the finished payloads itself. Bit-identical by construction: same conversion
 * code over the same bytes, only the thread that runs each record changes. */
typedef struct {
  int k;   /* ctab index, -1 when no node references this record */
  int64_t off, nel;
  int isz, nd;
  int64_t dims[PT_V21_MAX_RANK];
} pt_recinfo;

static pt_recinfo *g_recs;
static uint32_t g_recs_n;
static volatile int g_rec_next, g_rec_done;
static int g_par_state;   /* 0 prepared-not, 1 prepared, -1 prepare failed (fall back) */
static pthread_mutex_t g_arena_lock = PTHREAD_MUTEX_INITIALIZER;
static float *v21_load_cont(pt_ctx *c, uint32_t i, const char *name, int64_t off, int64_t n,
                            int itemsize);

/* The shared cache must already be sized when the parallel loaders publish: host_cont_shared grows
 * it lazily, and a claim for an unsized index is dropped -- which silently turns every alias into
 * a private copy (measured: peak RSS 468 MB instead of 272 MB, four payloads instead of one). */
static void hcache_prealloc(int n) {
  if ((size_t) n > g_hcache_n) {
    pt_hrec *nb = (pt_hrec *) realloc(g_hcache, (size_t) n * sizeof(pt_hrec));
    if (nb) {
      memset(nb + g_hcache_n, 0, ((size_t) n - g_hcache_n) * sizeof(pt_hrec));
      g_hcache = nb;
      g_hcache_n = (size_t) n;
    }
  }
}

static int par_claim(int k, float *p) {
  int ok = 0;
  pthread_mutex_lock(&g_hlock);
  if (k >= 0 && (size_t) k < g_hcache_n) {
    g_hcache[k].p = p;
    g_hcache[k].ref = 1;
    ok = 1;
  }
  pthread_mutex_unlock(&g_hlock);
  return ok;
}

static void fill_payload(const char *base, const pt_recinfo *ri, float *f) {
  const uint8_t *src = (const uint8_t *) base + ri->off;
  if (ri->isz == 4) {
    memcpy(f, src, (size_t) (ri->nel * 4));
  } else {
    const uint16_t *h = (const uint16_t *) src;
    for (int64_t k = 0; k < ri->nel; k++) {
      uint32_t u = h2f_table[h[k]];
      memcpy(&f[k], &u, 4);
    }
  }
}

static int par_prepare(pt_ctx *c, uint32_t n_tensor, const float *itemsize_tab) {
  g_recs = (pt_recinfo *) calloc((size_t) n_tensor, sizeof(pt_recinfo));
  if (!g_recs) {
    g_par_state = -1;
    return -1;
  }
  g_recs_n = n_tensor;
  for (uint32_t i = 0; i < n_tensor; i++) {
    const uint8_t *r = (const uint8_t *) c->map + V21_HDR_SIZE + (size_t) i * V21_REC_SIZE;
    pt_recinfo *ri = &g_recs[i];
    char nm[65];
    memcpy(nm, r, 64);
    nm[64] = 0;
    ri->k = -1;
    if (!nm[0])
      continue;
    for (int k = 0; k < c->n_cont; k++) {
      if (pt_graph_ctab[k] && !strcmp(pt_graph_ctab[k], nm)) {
        ri->k = k;
        break;
      }
    }
    if (ri->k < 0)
      continue;
    memcpy(&ri->nd, r + 64, 4);
    for (int z = 0; z < PT_V21_MAX_RANK; z++)
      memcpy(&ri->dims[z], r + 64 + 4 + 4 + 8 * z, 8);
    memcpy(&ri->nel, r + 64 + 4 + 4 + 48, 8);
    memcpy(&ri->off, r + 64 + 4 + 4 + 48 + 8, 8);
    ri->isz = itemsize_tab ? (int) itemsize_tab[i] : 4;
    if (ri->nel <= 0)
      ri->k = -1;
  }
  g_rec_next = 0;
  g_rec_done = 0;
  g_par_state = 1;
  return 0;
}

/* Claim loop: each record index is taken by exactly one thread (atomic counter), so the
 * materialization of a record happens once, on whichever stream thread gets there first. */
static void par_load(pt_ctx *c) {
  for (;;) {
    int i = __sync_fetch_and_add(&g_rec_next, 1);
    if ((uint32_t) i >= g_recs_n)
      break;
    pt_recinfo *ri = &g_recs[i];
    if (ri->k >= 0 && ri->nel > 0) {
      pthread_mutex_lock(&g_arena_lock);
      float *f = harena_alloc((size_t) (ri->nel * 4 + 64));
      pthread_mutex_unlock(&g_arena_lock);
      if (!f)
        f = (float *) malloc((size_t) (ri->nel * 4 + 64));
      if (f) {
        fill_payload((const char *) c->base, ri, f);
        par_claim(ri->k, f);
      } else {
        c->n_alloc_fail++;
      }
    }
    __sync_fetch_and_add(&g_rec_done, 1);
  }
}

static void par_barrier(void) {
  while (__sync_add_and_fetch(&g_rec_done, 0) < (int) g_recs_n)
    sched_yield();
}

static void par_alias(pt_ctx *c, const char *container) {
  for (uint32_t i = 0; i < g_recs_n; i++) {
    pt_recinfo *ri = &g_recs[i];
    if (ri->k < 0 || ri->nel <= 0)
      continue;
    int k = ri->k;
    c->cont_isz[k] = ri->isz;
    c->cont_n[k] = ri->nel;
    c->cont_ndim[k] = ri->nd;
    for (int z = 0; z < PT_V21_MAX_RANK; z++)
      c->cont_dims[k * PT_V21_MAX_RANK + z] = ri->dims[z];
    c->cont_shared[k] = 0;
    c->cont_f32[k] = host_cont_shared(k, container, ri->nel);
    if (!c->cont_f32[k]) {
      /* the load of this record failed (arena exhausted and malloc failed): materialize it here */
      pthread_mutex_lock(&g_arena_lock);
      c->cont_f32[k] = v21_load_cont(c, i, pt_graph_ctab[k], ri->off, ri->nel, ri->isz);
      pthread_mutex_unlock(&g_arena_lock);
      if (c->cont_f32[k]) {
        c->cont_shared[k] = 1;
        host_cont_shared_claim(k, c->cont_f32[k]);
      }
    }
  }
}

static float *v21_load_cont(pt_ctx *c, uint32_t i, const char *name, int64_t off, int64_t n,
                            int itemsize) {
  float *f = harena_alloc((size_t) (n * 4 + 64));
  if (!f) {
    f = (float *) malloc((size_t) (n * 4 + 64));   /* arena exhausted: fall back */
    if (!f) {
      c->n_alloc_fail++;
      return 0;
    }
  }
  const uint8_t *src = (const uint8_t *) c->base + off;
  if (itemsize == 4) {
    memcpy(f, src, (size_t) (n * 4));
  } else {
    h2f_table_init();
    const uint16_t *h = (const uint16_t *) src;
    for (int64_t k = 0; k < n; k++) {
      uint32_t u = h2f_table[h[k]];
      memcpy(&f[k], &u, 4);
    }
  }
  (void) name;
  return f;
}

static const char *ctab_lookup(pt_ctx *c, int idx) {
  if (idx < 0 || idx >= c->n_cont) return 0;
  return pt_graph_ctab[idx];
}

int pt_v21_n_cont(pt_ctx *c) { return c->n_cont; }
const float *pt_v21_cont(pt_ctx *c, int idx) {
  if (idx < 0 || idx >= c->n_cont) return 0;
  return c->cont_f32[idx];
}
size_t pt_v21_cont_bytes(pt_ctx *c, int idx) {
  if (idx < 0 || idx >= c->n_cont) return 0;
  return (size_t) c->cont_n[idx] * 4u;   /* every container weight is read as fp32 */
}

int pt_v21_cont_itemsize(pt_ctx *c, int idx) {
  if (idx < 0 || idx >= c->n_cont) return 0;
  return c->cont_isz[idx];
}

/* "prefix|d0,d1,..." -> rank + dims; the shape is separated from the comma fields by '|' */
static int parse_shape_dims(const char *s, int64_t *d, int max) {
  int nd = 0;
  const char *q = strchr(s, '|');
  if (!q) return 0;
  q++;
  while (*q) {
    long v = strtol(q, (char **) &q, 10);
    if (nd < max) d[nd++] = v;
    if (*q == ',') q++;
    else break;
  }
  return nd;
}

/* resolve a tensor defined by origin (graph input / container constant / int literal) */
static int resolve(pt_ctx *c, int tid) {
  pt_tv *t = &c->V[tid];
  if (t->p) return 0;
  const char *o = pt_tdefs[tid].origin;
  t->dt = pt_tdefs[tid].dt;
  if (strcmp(o, "IN") == 0) {
    /* graph inputs are supplied by pt_v21_run feeds */
    return -1;
  }
  if (strncmp(o, "CONT:", 5) == 0) {
    int idx = atoi(o + 5);
    const float *p = pt_v21_cont(c, idx);
    if (!p) return -1;
    t->p = (void *) p;
    t->dev = 2;                 /* container-aliased: never free */
    t->n = c->cont_n[idx];
    t->ndim = parse_shape_dims(o, t->d, PT_V21_MAX_RANK);
    for (int i = t->ndim; i < PT_V21_MAX_RANK; i++) t->d[i] = 1;
    if (t->ndim == 0) t->d[0] = t->n == 1 ? 1 : t->n;
    if (t->ndim == 0 && t->n != 1) t->ndim = 1;
    return 0;
  }
  if (strncmp(o, "LIT:", 4) == 0) {
    int off, n;
    char dt[8] = {0};
    sscanf(o + 4, "%d,%d,%7[a-zA-Z0-9]", &off, &n, dt);
    t->p = (void *) (pt_ipool + off);
    t->dev = 2;
    t->n = n;
    t->dt = dt[0] == 'I' ? DT_I64 : DT_F32;
    t->ndim = parse_shape_dims(o, t->d, PT_V21_MAX_RANK);
    for (int i = t->ndim; i < PT_V21_MAX_RANK; i++) t->d[i] = 1;
    if (t->ndim == 0) t->d[0] = n == 1 ? 1 : n;
    if (t->ndim == 0 && n != 1) t->ndim = 1;
    return 0;
  }
  return -1;
}

/* container record lookup for the resolved-CONT tensors: we need element counts, so keep the
 * per-record n from the container index in cont_n[]. */

static void dump_inputs(int idx) {
  const pt_node_def *nd = &pt_ndefs[idx];
  fprintf(stderr, "trace node %d op %d -> %s\n", idx, nd->op, pt_graph_tnames[nd->out[0]]);
  for (int i = 0; i < PT_V21_MAX_IN; i++) {
    if (nd->in[i] < 0) continue;
    pt_tv *t = &g_ctx->V[nd->in[i]];
    fprintf(stderr, "  in[%d] %s dt=%d ndim=%d n=%lld dev=%d dims=[", i, pt_graph_tnames[nd->in[i]],
            t->dt, t->ndim, (long long) t->n, t->dev);
    for (int q = 0; q < t->ndim; q++) fprintf(stderr, " %lld", (long long) t->d[q]);
    fprintf(stderr, " ] vals=[");
    for (int64_t q = 0; q < t->n && q < 8; q++) {
      if (t->dt == DT_F32) fprintf(stderr, " %g", (double) ((const float *) t->p)[q]);
      else fprintf(stderr, " %lld", (long long) ((const int64_t *) t->p)[q]);
    }
    fprintf(stderr, "]\n");
  }
}
pt_ctx *pt_v21_open(const char *container) {
  FILE *f = fopen(container, "rb");
  if (!f) {
    fprintf(stderr, "pt_v21: cannot open %s\n", container);
    return 0;
  }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fclose(f);
  pt_ctx *c = (pt_ctx *) calloc(1, sizeof(pt_ctx));
  c->maplen = (size_t) len;
  int fd = open(container, O_RDONLY);
  c->map = mmap(0, c->maplen, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (c->map == MAP_FAILED) {
    fprintf(stderr, "pt_v21: mmap %s failed\n", container);
    free(c);
    return 0;
  }
  v21_hdr h;
  memcpy(&h, c->map, sizeof(h));
  if (h.magic != PT_MAGIC_V21) {
    fprintf(stderr, "pt_v21: bad magic 0x%x (expected PTK1)\n", h.magic);
    free(c);
    return 0;
  }
  c->n_tensor = h.n_tensor;
  size_t data_off = ((size_t) (V21_HDR_SIZE + V21_REC_SIZE * (size_t) h.n_tensor) + 63) & ~(size_t) 63;
  c->base = (char *) c->map + data_off;
  c->n_cont = PT_N_CONT;
  c->cont_f32 = (float **) calloc((size_t) c->n_cont, sizeof(float *));
  c->cont_isz = (int *) calloc((size_t) c->n_cont, sizeof(int));
  c->cont_n = (int64_t *) calloc((size_t) c->n_cont, sizeof(int64_t));
  c->cont_ndim = (int *) calloc((size_t) c->n_cont, sizeof(int));
  c->cont_shared = (int *) calloc((size_t) c->n_cont, sizeof(int));
  c->cont_path = (char *) calloc(strlen(container) + 1, 1);
  if (c->cont_path) strcpy(c->cont_path, container);
  c->cont_dims = (int64_t *) calloc((size_t) c->n_cont * PT_V21_MAX_RANK, sizeof(int64_t));
  /* per-record payload width comes from the container's own dtype.table tensor (record order),
   * which is how the container tells a reader about the 39 fp16 decoder weights. */
  float *itemsize_tab = 0;
  for (uint32_t i = 0; i < h.n_tensor; i++) {
    char nm[65];
    memcpy(nm, (const char *) c->map + V21_HDR_SIZE + (size_t) i * V21_REC_SIZE, 64);
    nm[64] = 0;
    if (!strcmp(nm, "dtype.table")) {
      int32_t nd;
      int64_t off, nel;
      const uint8_t *r = (const uint8_t *) c->map + V21_HDR_SIZE + (size_t) i * V21_REC_SIZE;
      memcpy(&nd, r + 64, 4);
      memcpy(&nel, r + 64 + 4 + 4 + 48, 8);
      memcpy(&off, r + 64 + 4 + 4 + 48 + 8, 8);
      (void) nd;
      itemsize_tab = (float *) malloc((size_t) h.n_tensor * 4);
      memcpy(itemsize_tab, (const char *) c->map + data_off + off, (size_t) h.n_tensor * 4);
      break;
    }
  }
  if (!itemsize_tab) {
    fprintf(stderr, "pt_v21: container has no dtype.table record\n");
    free(c);
    return 0;
  }
  /* Contexts are opened concurrently (one per stream thread). Without this lock all of them find
   * the shared fp32 cache empty at the same moment and each converts its own copy (measured: peak
   * RSS ~195 MB per stream). Serializing the record pass lets the first context publish the
   * conversions and the rest alias them. */
  static pthread_mutex_t open_lock = PTHREAD_MUTEX_INITIALIZER;
  static int open_ser = -1;
  if (open_ser < 0) open_ser = getenv("PT_V21_OPEN_SERIAL") ? atoi(getenv("PT_V21_OPEN_SERIAL")) : 1;
  static int open_par = -1;
  if (open_par < 0) open_par = getenv("PT_V21_OPEN_PAR") ? atoi(getenv("PT_V21_OPEN_PAR")) : 1;
  int par_done = 0;
  if (open_par) {
    pthread_mutex_lock(&open_lock);
    reserve_payload_arena(c, h.n_tensor);
    h2f_table_init();   /* LUT first: par_load must not race its construction */
    hcache_prealloc(c->n_cont);
    if (g_par_state == 0)
      par_prepare(c, h.n_tensor, itemsize_tab);
    if (g_par_state > 0)
      par_done = 1;
    pthread_mutex_unlock(&open_lock);
    if (par_done) {
      par_load(c);
      par_barrier();
      par_alias(c, container);
    }
  }
  if (open_ser && !par_done) pthread_mutex_lock(&open_lock);
  if (!par_done) {
  {
    /* size the fp32 payload arena: every record that will be materialized on the host */
    size_t need = 0;
    for (uint32_t i = 0; i < h.n_tensor; i++) {
      const uint8_t *r = (const uint8_t *) c->map + V21_HDR_SIZE + (size_t) i * V21_REC_SIZE;
      char nm[65];
      memcpy(nm, r, 64);
      nm[64] = 0;
      if (!nm[0])
        continue;
      int64_t nel;
      memcpy(&nel, r + 64 + 4 + 4 + 48, 8);
      if (nel <= 0)
        continue;
      need += (((size_t) nel * 4 + 64) + 63) & ~(size_t) 63;
    }
    harena_reserve(need);
  }
  for (uint32_t i = 0; i < h.n_tensor; i++) {
    const uint8_t *r = (const uint8_t *) c->map + V21_HDR_SIZE + (size_t) i * V21_REC_SIZE;
    char nm[65];
    memcpy(nm, r, 64);
    nm[64] = 0;
    int32_t nd;
    int64_t rdims[6], nel, off;
    memcpy(&nd, r + 64, 4);
    for (int z = 0; z < 6; z++) memcpy(&rdims[z], r + 64 + 4 + 4 + 8 * z, 8);
    memcpy(&nel, r + 64 + 4 + 4 + 48, 8);
    memcpy(&off, r + 64 + 4 + 4 + 48 + 8, 8);
    int isz = itemsize_tab ? (int) itemsize_tab[i] : 4;
    if (nm[0] && nm[0] != '\0') {
      for (int k = 0; k < c->n_cont; k++) {
        if (c->cont_f32[k] == 0 && strcmp(pt_graph_ctab[k], nm) == 0) {
          c->cont_isz[k] = isz;
          c->cont_n[k] = nel;
          c->cont_ndim[k] = nd;
          for (int z = 0; z < 6; z++) c->cont_dims[k * PT_V21_MAX_RANK + z] = rdims[z];
          c->cont_shared[k] = 0;
          c->cont_f32[k] = host_cont_shared(k, container, nel);
          if (!c->cont_f32[k]) {
            c->cont_f32[k] = v21_load_cont(c, i, nm, off, nel, isz);
            if (c->cont_f32[k]) {
              c->cont_shared[k] = 1;
              host_cont_shared_claim(k, c->cont_f32[k]);
            }
          }
          break;
        }
      }
    }
  }
  }
  if (open_ser && !par_done) pthread_mutex_unlock(&open_lock);
  free(itemsize_tab);
  c->V = (pt_tv *) calloc(PT_N_TENSOR, sizeof(pt_tv));
  g_ctx = c;
  g_dump_tensor = getenv("PT_V21_DUMP_T") ? atoi(getenv("PT_V21_DUMP_T")) : -1;
  /* the consumer tables drive fusion and chaining decisions; contexts open concurrently, so the
   * one-time build must not run twice at once (a concurrent rebuild zeroes counts mid-use) */
  static pthread_mutex_t cons_lock = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_lock(&cons_lock);
  if (!g_cons_built) {
    for (int i = 0; i < PT_N_TENSOR; i++) {
      g_n_cons[i] = 0;
      g_cons_node[i] = -1;
      g_prod[i] = -1;
    }
    for (int i = 0; i < PT_N_NODE; i++)
      for (int j = 0; j < PT_V21_MAX_OUT; j++)
        if (pt_ndefs[i].out[j] >= 0 && pt_ndefs[i].out[j] < PT_N_TENSOR)
          g_prod[pt_ndefs[i].out[j]] = i;
    for (int i = 0; i < PT_N_NODE; i++)
      for (int j = 0; j < PT_V21_MAX_IN; j++)
        if (pt_ndefs[i].in[j] >= 0) {
          int t = pt_ndefs[i].in[j];
          if (g_n_cons[t] < PT_V21_MAX_CONS)
            g_cons_list[t][g_n_cons[t]] = i;
          g_n_cons[t]++;
          g_cons_node[t] = i;
        }
    for (int t = 0; t < PT_N_TENSOR; t++) {
      g_cons_dev[t] = 1;
      if (g_n_cons[t] < 1 || g_n_cons[t] > PT_V21_MAX_CONS) {
        g_cons_dev[t] = 0;
        continue;
      }
      for (int k = 0; k < g_n_cons[t]; k++) {
        const pt_node_def *cd = &pt_ndefs[g_cons_list[t][k]];
        int dev_ok = (cd->op == PT_CONV && cd->in[0] == t) ||
                     (cd->op == PT_ADD && (cd->in[0] == t || cd->in[1] == t)) ||
                     (cd->op == PT_LEAKYRELU && cd->in[0] == t);
        if (!dev_ok) {
          g_cons_dev[t] = 0;
          break;
        }
      }
    }
    g_cons_built = 1;
  }
  pthread_mutex_unlock(&cons_lock);
  c->fuse = getenv("PT_V21_FUSE") ? atoi(getenv("PT_V21_FUSE")) : 1;
  c->skip = (char *) calloc(PT_N_NODE, 1);
  c->prof = getenv("PT_V21_EKIND") ? 1 : 0;
  c->prof_node_on = c->prof && getenv("PT_V21_EKIND")[0] == '2';
  return c;
}

long pt_v21_stats_conv(pt_ctx *c) { return c->n_conv; }
long pt_v21_stats_convT(pt_ctx *c) { return c->n_convT; }
long pt_v21_stats_matmul(pt_ctx *c) { return c->n_matmul; }
double pt_v21_stats_ms(pt_ctx *c, int which) {
  double v = which == 0 ? c->t_conv : which == 1 ? c->t_convT : which == 2 ? c->t_matmul : c->t_total;
  return 1000.0 * v;
}

void pt_v21_close(pt_ctx *c) {
  if (getenv("PT_V21_MEM")) {
    long cont = 0;
    for (int i = 0; i < c->n_cont; i++)
      if (c->cont_f32[i])
        cont += (long) c->cont_n[i] * (long) c->cont_isz[i];
    fprintf(stderr,
            "MEM host: container_fp32=%ld MiB activation_peak=%ld MiB allocs=%ld  "
            "dev: weights=%ld MiB arena=%ld MiB stray=%ld MiB\n",
            cont >> 20, c->host_bytes_peak >> 20, c->host_alloc_count,
            (long) pt_v21_cuda_weights_mib(c), (long) c->arena_alloc_bytes >> 20,
            (long) c->dev_bytes_stray >> 20);
  }
  if (!c) return;
  if (c->map && c->map != MAP_FAILED) munmap(c->map, c->maplen);
  for (int i = 0; i < c->n_cont; i++) host_cont_shared_put(i, c->cont_f32[i], c->cont_shared[i]);
  free(c->cont_f32);
  free(c->cont_isz);
  free(c->cont_n);
  free(c->cont_ndim);
  free(c->cont_shared);
  free(c->cont_path);
  free(c->cont_dims);
  if (c->V)
    for (int i = 0; i < PT_N_TENSOR; i++)
      if (c->V[i].dev != 2) tv_free(c, &c->V[i]);
  pool_drain(c);
  free(c->V);
  free(c->wav);
  free(c);
}

/* ---------------- shape ops ---------------- */
static int op_reshape(pt_ctx *c, const pt_tv *A, const pt_tv *S, pt_tv *O) {
  int64_t d[PT_V21_MAX_RANK];
  int nd = 0;
  int64_t tgt = 1;
  int neg = -1;
  /* allowzero=0: a 0 in the shape vector copies the input's dimension at that position, with the
   * input dims left-padded with 1s up to the target rank (ONNXRuntime's reshape.cc behaviour).
   * The v2.1 encoder reshapes a rank-3 [1,192,T] tensor to rank 4 with a 0 in position 3, so the
   * left padding *is* the semantics, not an edge case. */
  int64_t ad[PT_V21_MAX_RANK];
  int lead = (int) S->n - A->ndim;
  for (int64_t i = 0; i < S->n; i++) ad[i] = i < lead ? 1 : A->d[(int) i - lead];
  for (int64_t i = 0; i < S->n; i++) {
    int64_t v = ((const int64_t *) S->p)[i];
    if (v == 0) v = ad[i];
    if (v == -1) {
      neg = nd;
      d[nd++] = 1;
    } else {
      d[nd++] = v;
      tgt *= v;
    }
  }
  if (neg >= 0) {
    d[neg] = tgt ? A->n / tgt : 0;
  }
  if (nd > PT_V21_MAX_RANK) return -1;
  if (tv_alloc_nz(c, O, A->dt, d, nd, 0) < 0) return -1;   /* the copy below fills it */
  if (O->n != A->n) {
    /* element-count mismatch: the target shape vector does not describe the data (a shape-vector
     * bug upstream).  Copy what fits and keep going - the parity dump will show where. */
    fprintf(stderr, "pt_v21: reshape element mismatch at node %d: %lld -> %lld\n", g_node,
            (long long) A->n, (long long) O->n);
  }
  int64_t cnt = A->n < O->n ? A->n : O->n;
  memcpy(O->p, A->p, (size_t) (cnt * dt_sz(A->dt)));
  if (O->n > cnt)   /* mismatch only: keep the tail zero, as the zero-filled buffer did */
    memset((char *) O->p + cnt * dt_sz(A->dt), 0, (size_t) ((O->n - cnt) * dt_sz(A->dt)));
  return 0;
}

static int op_transpose(pt_ctx *c, const pt_tv *A, const int *perm, pt_tv *O) {
  int nd = A->ndim;
  int64_t d[PT_V21_MAX_RANK];
  for (int i = 0; i < nd; i++) d[i] = A->d[perm[i]];
  if (tv_alloc_nz(c, O, A->dt, d, nd, 0) < 0) return -1;
  int64_t si[PT_V21_MAX_RANK], so[PT_V21_MAX_RANK];
  si[nd - 1] = 1;
  for (int i = nd - 2; i >= 0; i--) si[i] = si[i + 1] * A->d[i + 1];
  so[nd - 1] = 1;
  for (int i = nd - 2; i >= 0; i--) so[i] = so[i + 1] * d[i + 1];
  int es = dt_sz(A->dt);
  if (host_rows_on() && nd >= 1 && O->n > 0) {
    /* odometer over the output in order: output axis j advances the source offset by the source
     * stride of the input axis it came from, so the innermost loop is a strided copy with no
     * per-element div/mod decode */
    int64_t ss[PT_V21_MAX_RANK], ix[PT_V21_MAX_RANK];
    for (int j = 0; j < nd; j++) {
      ss[j] = si[perm[j]];
      ix[j] = 0;
    }
    int64_t inner = d[nd - 1], sl = ss[nd - 1], src = 0;
    int64_t nrows = O->n / inner;
    for (int64_t ro = 0; ro < nrows; ro++) {
      if (es == 4) {
        const uint32_t *a = (const uint32_t *) A->p + src;
        uint32_t *o = (uint32_t *) O->p + ro * inner;
        for (int64_t k = 0; k < inner; k++) o[k] = a[k * sl];
      } else {
        for (int64_t k = 0; k < inner; k++)
          memcpy((char *) O->p + (ro * inner + k) * es, (const char *) A->p + (src + k * sl) * es,
                 (size_t) es);
      }
      for (int j = nd - 2; j >= 0; j--) {
        ix[j]++;
        src += ss[j];
        if (ix[j] < d[j])
          break;
        src -= ss[j] * d[j];
        ix[j] = 0;
      }
    }
    return 0;
  }
  for (int64_t i = 0; i < A->n; i++) {
    int64_t r = i, oa = 0;
    for (int q = 0; q < nd; q++) {
      int64_t idx = r / si[q];
      r %= si[q];
      oa += idx * so[perm[q]];
    }
    memcpy((char *) O->p + oa * es, (const char *) A->p + i * es, (size_t) es);
  }
  return 0;
}

static int op_concat(pt_ctx *c, int axis, pt_tv **ins, int n, pt_tv *O) {
  int nd = ins[0]->ndim;
  if (axis < 0) axis += nd;
  int64_t d[PT_V21_MAX_RANK];
  memcpy(d, ins[0]->d, sizeof(d));
  int64_t tot = 0;
  for (int i = 0; i < n; i++) tot += ins[i]->d[axis];
  d[axis] = tot;
  if (tv_alloc_nz(c, O, ins[0]->dt, d, nd, 0) < 0) return -1;
  int es = dt_sz(ins[0]->dt);
  int64_t outer = 1, inner = 1;
  for (int i = 0; i < axis; i++) outer *= d[i];
  for (int i = axis + 1; i < nd; i++) inner *= d[i];
  for (int64_t o = 0; o < outer; o++) {
    int64_t dst = o * tot * inner;
    for (int i = 0; i < n; i++) {
      int64_t cnt = ins[i]->d[axis] * inner;
      memcpy((char *) O->p + dst * es, (const char *) ins[i]->p + o * ins[i]->d[axis] * inner * es,
             (size_t) (cnt * es));
      dst += cnt;
    }
  }
  return 0;
}

static int op_gather(pt_ctx *c, int axis, const void *table, const int64_t *tdims, int tnd,
                     int64_t tn, int tdt, const pt_tv *IDX, pt_tv *O) {
  if (axis < 0) axis += tnd;
  int64_t d[PT_V21_MAX_RANK];
  int nd = 0;
  for (int i = 0; i < axis; i++) d[nd++] = tdims[i];
  for (int i = 0; i < IDX->ndim; i++) d[nd++] = IDX->d[i];
  for (int i = axis + 1; i < tnd; i++) d[nd++] = tdims[i];
  for (int i = nd; i < PT_V21_MAX_RANK; i++) d[i] = 1;
  if (nd > PT_V21_MAX_RANK) return -1;
  if (tv_alloc_nz(c, O, tdt, d, nd, 0) < 0) return -1;
  int es = dt_sz(tdt);
  int64_t pre = 1, post = 1;
  for (int i = 0; i < axis; i++) pre *= tdims[i];
  for (int i = axis + 1; i < tnd; i++) post *= tdims[i];
  int64_t stride = tdims[axis] * post;
  for (int64_t i = 0; i < IDX->n; i++) {
    int64_t ix = IDX->p ? ((const int64_t *) IDX->p)[i] : 0;
    if (ix < 0) ix += tdims[axis];      /* ONNX Gather: negative indices count from the end */
    if (ix < 0 || ix >= tdims[axis]) ix = 0;
    int64_t row = ix * post;
    for (int64_t p = 0; p < pre; p++) {
      memcpy((char *) O->p + ((p * IDX->n + i) * post) * es,
             (const char *) table + ((p * stride + row)) * es, (size_t) (post * es));
    }
  }
  (void) tn;
  return 0;
}

static int op_slice(pt_ctx *c, const pt_tv *A, const int64_t *starts, const int64_t *ends,
                    const int64_t *axes, int naxes, int64_t step, pt_tv *O) {
  int nd = A->ndim;
  int64_t d[PT_V21_MAX_RANK], st[PT_V21_MAX_RANK], en[PT_V21_MAX_RANK], sp[PT_V21_MAX_RANK];
  if (naxes > PT_V21_MAX_RANK) naxes = PT_V21_MAX_RANK;   /* rank is capped, so are the axis lists */
  for (int i = 0; i < nd; i++) {
    st[i] = 0;
    en[i] = A->d[i];
    sp[i] = 1;
  }
  for (int i = 0; i < naxes; i++) {
    int ax = (int) axes[i];
    if (ax < 0) ax += nd;
    if (ax < 0 || ax >= nd) continue;
    int64_t dim = A->d[ax];
    /* ONNX Slice index normalisation is step-sign dependent (numpy semantics):
     *   step > 0: start in [0, dim], end in [0, dim], count = ceil((end-start)/step)
     *   step < 0: start in [-1, dim-1], end in [-1, dim-1], count = (end-start)/step
     * so ends = dim with step 1 keeps every element, and ends = INT64_MIN+1 with step -1 runs
     * down past index 0 (the rel-pos windows use both forms). */
    int64_t sv = starts[i], ev = ends[i];
    if (step > 0) {
      if (sv < 0) sv += dim;
      if (ev < 0) ev += dim;
      if (sv < 0) sv = 0;
      if (sv > dim) sv = dim;
      if (ev < 0) ev = 0;
      if (ev > dim) ev = dim;
    } else {
      if (sv < 0) sv += dim;
      if (ev < 0) ev += dim;
      if (sv < -1) sv = -1;
      if (sv > dim - 1) sv = dim - 1;
      if (ev < -1) ev = -1;
      if (ev > dim - 1) ev = dim - 1;
    }
    st[ax] = sv;
    en[ax] = ev;
    sp[ax] = step;
  }
  for (int i = 0; i < nd; i++)
    d[i] = sp[i] > 0 ? (en[i] - st[i] + sp[i] - 1) / sp[i] : (en[i] - st[i]) / sp[i];
  for (int i = 0; i < nd; i++)
    if (d[i] < 0) d[i] = 0;
  /* Zero-copy view. The selected region is one contiguous block of A exactly when, for some axis
   * k, every axis after k is taken whole with step 1, axis k is a step-1 range, and every axis
   * before k contributes one index. The output then points into A's buffer (holding a reference
   * on it) instead of copying it: the flow's gate slices are of this form, ~3 MB per request. */
  static int sview = -1;
  if (sview < 0) sview = getenv("PT_V21_SLICE_VIEW") ? atoi(getenv("PT_V21_SLICE_VIEW")) : 1;
  if (sview && nd >= 1 && A->p) {
    int k = nd - 1;
    while (k > 0 && st[k] == 0 && d[k] == A->d[k] && sp[k] == 1) k--;
    int ok = sp[k] == 1 || d[k] <= 1;
    for (int q = 0; q < k && ok; q++)
      if (d[q] != 1) ok = 0;
    int64_t n = 1;
    for (int q = 0; q < nd; q++) n *= d[q];
    if (ok && n > 0) {
      int64_t stride = 1, off = 0;
      for (int q = nd - 1; q >= 0; q--) {
        off += st[q] * stride;
        stride *= A->d[q];
      }
      O->dt = A->dt;
      O->ndim = nd;
      for (int q = 0; q < PT_V21_MAX_RANK; q++) O->d[q] = q < nd ? d[q] : 1;
      O->n = n;
      O->p = (char *) A->p + off * dt_sz(A->dt);
      O->dev_res = 0;
      O->dev_p = NULL;
      O->alive = 1;
      O->view = 1;
      if (A->base) {            /* pooled source: share it, released at this view's last use */
        buf_ref_add(A->base);
        O->base = A->base;
        O->dev = 0;
      } else {                  /* container / feed / seed storage outlives the run: never freed */
        O->base = NULL;
        O->dev = 2;
      }
      return 0;
    }
  }
  if (tv_alloc_nz(c, O, A->dt, d, nd, 0) < 0) return -1;
  int es = dt_sz(A->dt);
  int64_t si[PT_V21_MAX_RANK], so[PT_V21_MAX_RANK];
  si[nd - 1] = 1;
  for (int i = nd - 2; i >= 0; i--) si[i] = si[i + 1] * A->d[i + 1];
  so[nd - 1] = 1;
  for (int i = nd - 2; i >= 0; i--) so[i] = so[i + 1] * d[i + 1];
  static int sfast = -1;
  if (sfast < 0) sfast = getenv("PT_V21_SLICE_ROWS") ? atoi(getenv("PT_V21_SLICE_ROWS")) : 1;
  if (sfast && nd >= 1 && sp[nd - 1] == 1 && O->n > 0) {
    /* innermost axis is a unit-step range: every output row is one contiguous source run, so
     * decode the row's offset once and copy the row with a single memcpy instead of a div/mod
     * decode plus a 4-byte memcpy per element */
    int64_t row = d[nd - 1];
    int64_t nrows = O->n / row;
    size_t rb = (size_t) (row * es);
    for (int64_t ri = 0; ri < nrows; ri++) {
      int64_t r = ri * row, oa = st[nd - 1];
      for (int q = 0; q < nd - 1; q++) {
        int64_t idx = r / so[q];
        r %= so[q];
        oa += (st[q] + idx * sp[q]) * si[q];
      }
      memcpy((char *) O->p + ri * rb, (const char *) A->p + oa * es, rb);
    }
    return 0;
  }
  for (int64_t i = 0; i < O->n; i++) {
    int64_t r = i, oa = 0;
    for (int q = 0; q < nd; q++) {
      int64_t idx = r / so[q];
      r %= so[q];
      oa += (st[q] + idx * sp[q]) * si[q];
    }
    memcpy((char *) O->p + i * es, (const char *) A->p + oa * es, (size_t) es);
  }
  return 0;
}

static int op_pad(pt_ctx *c, const pt_tv *A, const int64_t *pads, double cv, int mode, pt_tv *O) {
  int nd = A->ndim;
  if (nd < 1 || nd > PT_V21_MAX_RANK) return -1;
  int64_t d[PT_V21_MAX_RANK], lo[PT_V21_MAX_RANK];
  for (int i = 0; i < nd; i++) {
    lo[i] = pads[i];
    d[i] = A->d[i] + pads[i] + pads[nd + i];
    if (d[i] < 0) return -1;
  }
  /* every path below writes every output element (the constant path zero-fills itself) */
  if (tv_alloc_nz(c, O, A->dt, d, nd, 0) < 0) return -1;
  int es = dt_sz(A->dt);
  int64_t si[PT_V21_MAX_RANK], so[PT_V21_MAX_RANK];
  si[nd - 1] = 1;
  for (int i = nd - 2; i >= 0; i--) si[i] = si[i + 1] * A->d[i + 1];
  so[nd - 1] = 1;
  for (int i = nd - 2; i >= 0; i--) so[i] = so[i + 1] * d[i + 1];
  (void) cv;   /* v2.1 pads with zeros only (the graph's constant_value inputs are all 0) */
  int nonneg = 1;
  for (int i = 0; i < 2 * nd; i++)
    if (pads[i] < 0) nonneg = 0;
  if (host_rows_on() && mode == 0 && nonneg) {
    /* constant (zero) padding without cropping: zero the output once, then place every source row
     * with one memcpy at its padded offset - instead of decoding each output element */
    memset(O->p, 0, (size_t) (O->n * es));
    int64_t row = A->d[nd - 1];
    if (row > 0 && A->n > 0) {
      int64_t nrows = A->n / row;
      size_t rb = (size_t) (row * es);
      for (int64_t ri = 0; ri < nrows; ri++) {
        int64_t r = ri * row, oa = lo[nd - 1];
        for (int q = 0; q < nd - 1; q++) {
          int64_t idx = r / si[q];
          r %= si[q];
          oa += (idx + lo[q]) * so[q];
        }
        memcpy((char *) O->p + oa * es, (const char *) A->p + ri * rb, rb);
      }
    }
    return 0;
  }
  if (host_rows_on() && mode == 1 && nonneg) {
    /* reflect padding without cropping: every output row is a whole source row with mirrored
     * segments on both edges, so the middle is one memcpy and the two edge strips are short
     * reversed copies. The reflection index is the same one the generic loop below computes
     * (src < 0 ? -src : 2L - src - 2), only hoisted out of the per-element decode. Requires
     * pad < L on every axis, otherwise a reflection could land outside the source and the generic
     * path's zero fill applies. */
    int fits = 1;
    for (int q = 0; q < nd; q++)
      if (lo[q] >= A->d[q] || pads[nd + q] >= A->d[q])
        fits = 0;
    if (fits) {
      int64_t L = A->d[nd - 1];
      int64_t nrows = O->n / d[nd - 1];
      /* row strides of the OUTER axes: so[] counts the last axis too, so divide it out */
      int64_t rso[PT_V21_MAX_RANK];
      for (int q = 0; q < nd - 1; q++) rso[q] = so[q] / d[nd - 1];
      char *op = (char *) O->p;
      for (int64_t ro = 0; ro < nrows; ro++) {
        int64_t r = ro, oa = 0;
        for (int q = 0; q < nd - 1; q++) {
          int64_t idx = r / rso[q];
          r %= rso[q];
          int64_t src = idx - lo[q];
          if (src < 0) src = -src;
          else if (src >= A->d[q]) src = 2 * A->d[q] - src - 2;
          oa += src * si[q];
        }
        const char *row = (const char *) A->p + oa * es;
        char *orow = op + ro * d[nd - 1] * es;
        for (int64_t j = 0; j < lo[nd - 1]; j++) {          /* left mirror */
          int64_t sj = lo[nd - 1] - j;
          memcpy(orow + j * es, row + sj * es, (size_t) es);
        }
        memcpy(orow + lo[nd - 1] * es, row, (size_t) (L * es));
        for (int64_t j = 0; j < pads[nd + nd - 1]; j++) {    /* right mirror */
          int64_t sj = L - 2 - j;
          memcpy(orow + (lo[nd - 1] + L + j) * es, row + sj * es, (size_t) es);
        }
      }
      return 0;
    }
  }
  for (int64_t i = 0; i < O->n; i++) {
    int64_t r = i, oa = 0;
    int inside = 1;
    for (int q = 0; q < nd; q++) {
      int64_t idx = r / so[q];
      r %= so[q];
      int64_t src = idx - lo[q];
      if (src < 0 || src >= A->d[q]) {
        if (mode == 2) {                        /* edge: clamp to the border element */
          src = src < 0 ? 0 : A->d[q] - 1;
        } else if (mode == 1) {                 /* reflect */
          src = src < 0 ? -src : 2 * A->d[q] - src - 2;
          if (src < 0 || src >= A->d[q]) inside = 0;
        } else if (mode == 3) {                 /* wrap */
          src = ((src % A->d[q]) + A->d[q]) % A->d[q];
        } else {                                /* constant: this element is padding */
          inside = 0;
          break;
        }
      }
      oa += src * si[q];
    }
    if (inside)
      memcpy((char *) O->p + i * es, (const char *) A->p + oa * es, (size_t) es);
    else
      memset((char *) O->p + i * es, 0, (size_t) es);
  }
  return 0;
}

static int op_expand(pt_ctx *c, const pt_tv *A, const int64_t *shape, int nshape, pt_tv *O) {
  int nd = nshape;
  if (nd == 0) {
    /* an empty shape vector means the result is a rank-0 scalar (the dec Expand on a Constant) */
    int64_t d0[PT_V21_MAX_RANK] = {1, 1, 1, 1, 1, 1};
    if (tv_alloc(c, O, A->dt, d0, 0) < 0) return -1;
    memcpy(O->p, A->p, (size_t) dt_sz(A->dt));
    O->ndim = 0;
    O->n = 1;
    return 0;
  }
  if (nd > PT_V21_MAX_RANK || nd < A->ndim) return -1;
  int64_t d[PT_V21_MAX_RANK];
  for (int i = 0; i < nd; i++) d[i] = shape[i];
  if (tv_alloc_nz(c, O, A->dt, d, nd, 0) < 0) return -1;
  int es = dt_sz(A->dt);
  /* A's dims line up with the output's LAST A->ndim dims (numpy broadcasting) */
  int off = nd - A->ndim;
  int64_t sa[PT_V21_MAX_RANK], so[PT_V21_MAX_RANK];
  if (A->ndim > 0) {
    sa[A->ndim - 1] = 1;
    for (int i = A->ndim - 2; i >= 0; i--) sa[i] = sa[i + 1] * A->d[i + 1];
  }
  so[nd - 1] = 1;
  for (int i = nd - 2; i >= 0; i--) so[i] = so[i + 1] * d[i + 1];
  for (int64_t i = 0; i < O->n; i++) {
    int64_t r = i, oa = 0;
    for (int q = 0; q < nd; q++) {
      int64_t idx = r / so[q];
      r %= so[q];
      int j = q - off;
      if (j < 0 || A->d[j] == 1) continue;      /* broadcast: that axis contributes no offset */
      oa += idx * sa[j];
    }
    memcpy((char *) O->p + i * es, (const char *) A->p + oa * es, (size_t) es);
  }
  return 0;
}

static int op_reduce(pt_ctx *c, int op, const pt_tv *A, const int64_t *axes, int naxes,
                     int keepdims, pt_tv *O) {
  int nd = A->ndim;
  int drop[PT_V21_MAX_RANK] = {0};
  if (naxes > PT_V21_MAX_RANK) naxes = PT_V21_MAX_RANK;
  for (int i = 0; i < naxes; i++) {
    int ax = (int) axes[i];
    if (ax < 0) ax += nd;
    if (ax >= 0 && ax < nd) drop[ax] = 1;
  }
  int64_t d[PT_V21_MAX_RANK];
  int nd2 = 0;
  for (int i = 0; i < nd; i++) d[nd2++] = drop[i] ? (keepdims ? 1 : 0) : A->d[i];
  if (!keepdims) {
    int k = 0;
    for (int i = 0; i < nd; i++) if (!drop[i]) d[k++] = A->d[i];
    nd2 = k;
  }
  if (nd2 == 0) nd2 = 0;   /* all axes dropped with keepdims=0 -> a rank-0 scalar (ONNX) */
  for (int i = nd2; i < PT_V21_MAX_RANK; i++) d[i] = 1;
  if (tv_alloc(c, O, A->dt, d, nd2) < 0) return -1;
  int es = dt_sz(A->dt);
  int64_t cnt = 1;
  int64_t nout = O->n;
  if (nout < 1) nout = 1;
  if (A->dt == DT_F32) {
    const float *a = (const float *) A->p;
    float *o = (float *) O->p;
    for (int64_t k = 0; k < nout; k++) o[k] = 0.0f;
    for (int64_t i = 0; i < A->n; i++) {
      int64_t r = i, oa = 0, ds = 1;
      for (int q = nd - 1; q >= 0; q--) {
        int64_t idx = r % A->d[q];
        r /= A->d[q];
        if (drop[q]) continue;
        oa += idx * ds;
        ds *= d[q];
      }
      if (oa >= nout) continue;
      if (op == PT_REDUCESUM || op == PT_REDUCEMEAN) o[oa] += a[i];
      else if (op == PT_REDUCEMAX) {
        if (i == 0 || a[i] > o[oa]) o[oa] = a[i];
      } else if (op == PT_REDUCEMIN) {
        if (i == 0 || a[i] < o[oa]) o[oa] = a[i];
      }
    }
    if (op == PT_REDUCEMEAN && cnt)
      for (int64_t k = 0; k < nout; k++) o[k] = (float) ((double) o[k] / (double) cnt);
  } else {
    const int64_t *a = (const int64_t *) A->p;
    int64_t *o = (int64_t *) O->p;
    for (int64_t k = 0; k < nout; k++) o[k] = 0;
    for (int64_t i = 0; i < A->n; i++) {
      int64_t r = i, oa = 0, ds = 1;
      for (int q = nd - 1; q >= 0; q--) {
        int64_t idx = r % A->d[q];
        r /= A->d[q];
        if (drop[q]) continue;
        oa += idx * ds;
        ds *= d[q];
      }
      if (oa >= nout) continue;
      if (op == PT_REDUCESUM) o[oa] += a[i];
      else if (op == PT_REDUCEMAX) {
        if (i == 0 || a[i] > o[oa]) o[oa] = a[i];
      } else if (op == PT_REDUCEMIN) {
        if (i == 0 || a[i] < o[oa]) o[oa] = a[i];
      }
    }
  }
  return 0;
}

static int op_cumsum(pt_ctx *c, const pt_tv *A, int axis, int exclusive, int reverse, pt_tv *O) {
  int nd = A->ndim;
  if (axis < 0) axis += nd;
  int64_t d[PT_V21_MAX_RANK];
  memcpy(d, A->d, sizeof(d));
  if (tv_alloc_nz(c, O, A->dt, d, nd, 0) < 0) return -1;
  int64_t outer = 1, inner = 1;
  for (int i = 0; i < axis; i++) outer *= d[i];
  for (int i = axis + 1; i < nd; i++) inner *= d[i];
  int64_t L = d[axis];
  for (int64_t o = 0; o < outer; o++) {
    for (int64_t q = 0; q < inner; q++) {
      double acc = 0.0;
      for (int64_t t = 0; t < L; t++) {
        int64_t ti = reverse ? L - 1 - t : t;
        int64_t idx = (o * L + ti) * inner + q;
        double v = A->dt == DT_F32 ? (double) ((const float *) A->p)[idx]
                                   : (double) ((const int64_t *) A->p)[idx];
        double out;
        if (exclusive) {
          out = acc;
          acc += v;
        } else {
          acc += v;
          out = acc;
        }
        if (O->dt == DT_F32) ((float *) O->p)[idx] = (float) out;
        else ((int64_t *) O->p)[idx] = (int64_t) out;
      }
    }
  }
  return 0;
}

static int op_softmax(pt_ctx *c, const pt_tv *A, int axis, pt_tv *O) {
  int nd = A->ndim;
  if (axis < 0) axis += nd;
  int64_t d[PT_V21_MAX_RANK];
  memcpy(d, A->d, sizeof(d));
  if (tv_alloc_nz(c, O, DT_F32, d, nd, 0) < 0) return -1;
  int64_t outer = 1, inner = 1, L = d[axis];
  for (int i = 0; i < axis; i++) outer *= d[i];
  for (int i = axis + 1; i < nd; i++) inner *= d[i];
  const float *a = (const float *) A->p;
  float *o = (float *) O->p;
  for (int64_t blk = 0; blk < outer; blk++) {
    for (int64_t q = 0; q < inner; q++) {
      const float *row = a + blk * L * inner + q;
      float *orow = o + blk * L * inner + q;
      double mx = -1e300;
      for (int64_t i = 0; i < L; i++) {
        double v = row[i * inner];
        if (v > mx) mx = v;
      }
      double sum = 0.0;
      for (int64_t i = 0; i < L; i++) {
        double e = exp((double) row[i * inner] - mx);
        orow[i * inner] = (float) e;
        sum += e;
      }
      double inv = sum > 0 ? 1.0 / sum : 0.0;
      for (int64_t i = 0; i < L; i++) orow[i * inner] = (float) ((double) orow[i * inner] * inv);
    }
  }
  return 0;
}

static int op_layernorm(pt_ctx *c, const pt_tv *A, int axis, float eps, const float *g,
                        const float *beta, pt_tv *O) {
  int nd = A->ndim;
  if (axis < 0) axis += nd;
  int64_t d[PT_V21_MAX_RANK];
  memcpy(d, A->d, sizeof(d));
  if (tv_alloc_nz(c, O, DT_F32, d, nd, 0) < 0) return -1;   /* every row is written */
  int64_t outer = 1, K = 1;
  for (int i = 0; i < axis; i++) outer *= d[i];
  for (int i = axis; i < nd; i++) K *= d[i];
  const float *a = (const float *) A->p;
  float *o = (float *) O->p;
  for (int64_t r = 0; r < outer; r++) {
    const float *row = a + r * K;
    float *orow = o + r * K;
    double mean = 0.0;
    for (int64_t i = 0; i < K; i++) mean += row[i];
    mean /= (double) K;
    double var = 0.0;
    for (int64_t i = 0; i < K; i++) {
      double dv = (double) row[i] - mean;
      var += dv * dv;
    }
    var /= (double) K;
    double inv = 1.0 / sqrt(var + (double) eps);
    for (int64_t i = 0; i < K; i++)
      orow[i] = (float) (((double) row[i] - mean) * inv * (double) g[i] + (double) beta[i]);
  }
  return 0;
}

/* 2-operand einsum, e.g. "bkt,kn->bnt" (the only form v2.1 uses) */
static int op_einsum(pt_ctx *c, const char *eq, const pt_tv *A, const pt_tv *B, pt_tv *O) {
  const char *arrow = strstr(eq, "->");
  if (!arrow) return -1;
  char lhs[32] = {0}, rhs[32] = {0};
  int nl = (int) (arrow - eq);
  memcpy(lhs, eq, (size_t) nl);
  strcpy(rhs, arrow + 2);
  int nA = 0, nB = 0, nO = 0;
  char sA[16], sB[16], sO[16];
  for (int i = 0; i < nl; i++) {
    if (lhs[i] == ',') {
      nB = (int) strlen(lhs) - i - 1;
      memcpy(sB, lhs + i + 1, (size_t) nB);
      nA = i;
      memcpy(sA, lhs, (size_t) nA);
      break;
    }
  }
  if (!nA) {
    nA = nl;
    memcpy(sA, lhs, (size_t) nA);
    return -1;
  }
  for (int i = 0; rhs[i]; i++) if (rhs[i] != ' ') sO[nO++] = rhs[i];
  if (A->ndim != nA || B->ndim != nB || O->ndim) {
    /* output rank = rhs letters */
  }
  int64_t dA[16], dB[16], dO[16];
  for (int i = 0; i < nA; i++) dA[i] = i < A->ndim ? A->d[i] : 1;
  for (int i = 0; i < nB; i++) dB[i] = i < B->ndim ? B->d[i] : 1;
  /* build output dims from rhs letters, resolving each letter's size from A/B */
  for (int i = 0; i < nO; i++) {
    int64_t v = 1;
    for (int k = 0; k < nA; k++) if (sA[k] == sO[i]) v = dA[k];
    for (int k = 0; k < nB; k++) if (sB[k] == sO[i]) v = dB[k];
    dO[i] = v;
  }
  if (nO > PT_V21_MAX_RANK) return -1;
  if (tv_alloc(c, O, DT_F32, dO, nO) < 0) return -1;
  /* Batched-matmul einsum: A = a0 a1 a2, B = a1 a3, O = a0 a3 a2 (v2.1's iSTFT basis contraction
   * is exactly `bkt,kn->bnt`). The generic path below walks the letter product space with a
   * modulo-decode per element, which cost 377 ms for 231k FMAs - two calls, the single biggest
   * non-GPU item in a stream. This path is the plain b-k-n-t loop with the innermost index
   * contiguous in both A and the output, and it keeps the same summation order (the summed index
   * ascends, in double, into a scratch row) so the result is bit-identical. */
  if (nA == 3 && nB == 2 && nO == 3 && sB[0] == sA[1] && sO[0] == sA[0] && sO[1] == sB[1] &&
      sO[2] == sA[2] && sA[0] != sA[1] && sA[1] != sA[2] && sA[0] != sA[2] && B->dt == DT_F32) {
    int64_t Bb = dA[0], K = dA[1], T = dA[2], N = dB[1];
    const float *pa = (const float *) A->p, *pb = (const float *) B->p;
    float *po = (float *) O->p;
    double *acc = (double *) malloc((size_t) N * T * sizeof(double) + 64);
    if (acc) {
      for (int64_t b = 0; b < Bb; b++) {
        for (int64_t i = 0; i < N * T; i++) acc[i] = 0.0;
        for (int64_t k = 0; k < K; k++) {
          const float *arow = pa + (b * K + k) * T;
          for (int64_t n = 0; n < N; n++) {
            double bv = (double) pb[k * N + n];
            if (bv == 0.0) continue;
            double *an = acc + n * T;   /* each (b,n) output row has its own k-sum */
            for (int64_t t = 0; t < T; t++) an[t] += bv * (double) arow[t];
          }
        }
        for (int64_t n = 0; n < N; n++) {
          float *orow = po + (b * N + n) * T;
          for (int64_t t = 0; t < T; t++) orow[t] = (float) acc[n * T + t];
        }
      }
      free(acc);
      return 0;   /* the generic letter-space loop below would redo the same work */
    }
  }
  /* strides for A/B per letter index */
  int64_t stA[16] = {0}, stB[16] = {0}, stO[16] = {0};
  stA[nA - 1] = 1;
  for (int i = nA - 2; i >= 0; i--) stA[i] = stA[i + 1] * dA[i + 1];
  stB[nB - 1] = 1;
  for (int i = nB - 2; i >= 0; i--) stB[i] = stB[i + 1] * dB[i + 1];
  stO[nO - 1] = 1;
  for (int i = nO - 2; i >= 0; i--) stO[i] = stO[i + 1] * dO[i + 1];
  /* sum over the letters not in the output */
  char suml[16];
  int ns = 0;
  for (int i = 0; i < nA; i++) {
    int in = 0;
    for (int k = 0; k < nO; k++) if (sO[k] == sA[i]) in = 1;
    if (!in) {
      int dup = 0;
      for (int k = 0; k < ns; k++) if (suml[k] == sA[i]) dup = 1;
      if (!dup) suml[ns++] = sA[i];
    }
  }
  int64_t cntS = 1;
  int64_t szS[16];
  for (int i = 0; i < ns; i++) {
    int64_t v = 1;
    for (int k = 0; k < nA; k++) if (sA[k] == suml[i]) v = dA[k];
    for (int k = 0; k < nB; k++) if (sB[k] == suml[i]) v = dB[k];
    szS[i] = v;
    cntS *= v;
  }
  const float *a = (const float *) A->p, *b = (const float *) B->p;
  float *o = (float *) O->p;
  for (int64_t oi = 0; oi < O->n; oi++) {
    /* decode output indices */
    int64_t idxO[16], r = oi;
    for (int i = 0; i < nO; i++) idxO[i] = 0;
    for (int i = nO - 1; i >= 0; i--) {
      idxO[i] = r % dO[i];
      r /= dO[i];
    }
    double acc = 0.0;
    for (int64_t s = 0; s < cntS; s++) {
      int64_t idxS[16], rr = s;
      for (int i = ns - 1; i >= 0; i--) {
        idxS[i] = rr % szS[i];
        rr /= szS[i];
      }
      int64_t oa = 0, ob = 0;
      for (int i = 0; i < nA; i++) {
        int64_t v = -1;
        for (int k = 0; k < nO; k++) if (sO[k] == sA[i]) v = idxO[k];
        if (v < 0) for (int k = 0; k < ns; k++) if (suml[k] == sA[i]) v = idxS[k];
        if (v < 0) v = 0;
        oa += v * stA[i];
      }
      for (int i = 0; i < nB; i++) {
        int64_t v = -1;
        for (int k = 0; k < nO; k++) if (sO[k] == sB[i]) v = idxO[k];
        if (v < 0) for (int k = 0; k < ns; k++) if (suml[k] == sB[i]) v = idxS[k];
        if (v < 0) v = 0;
        ob += v * stB[i];
      }
      acc += (double) a[oa] * (double) b[ob];
    }
    o[oi] = (float) acc;
  }
  return 0;
}

static int op_where(pt_ctx *c, const pt_tv *C, const pt_tv *X, const pt_tv *Y, pt_tv *O) {
  int64_t d[PT_V21_MAX_RANK];
  int nd;
  if (bcast_dims(C, X, Y, d, &nd) < 0) return -1;
  int dt = X->dt == DT_NONE ? Y->dt : X->dt;
  if (tv_alloc_nz(c, O, dt, d, nd, 0) < 0) return -1;
  bcast ba, bb, be;
  bcast_prep(&ba, X, d, nd);
  bcast_prep(&bb, Y, d, nd);
  bcast_prep(&be, C, d, nd);
  const uint8_t *cv = (const uint8_t *) C->p;
  int xs = dt_sz(X->dt), ys = dt_sz(Y->dt), os = dt_sz(dt);
  for (int64_t i = 0; i < O->n; i++) {
    int64_t r = i, ox = 0, oy = 0, oc = 0;
    for (int q = nd - 1; q >= 0; q--) {
      int64_t idx = r % d[q];
      r /= d[q];
      ox += idx * ba.st[q];
      oy += idx * bb.st[q];
      oc += idx * be.st[q];
    }
    memcpy((char *) O->p + i * os, cv[oc] ? (const char *) X->p + ox * xs : (const char *) Y->p + oy * ys,
           (size_t) os);
  }
  return 0;
}

static int op_cast(pt_ctx *c, const pt_tv *A, int to, pt_tv *O) {
  int64_t d[PT_V21_MAX_RANK];
  memcpy(d, A->d, sizeof(d));
  int dt = to == 1 ? DT_F32 : to == 7 ? DT_I64 : to == 6 ? DT_I64 : to == 9 ? DT_BOOL : to == 10 ? DT_F16 : DT_F32;
  if (tv_alloc_nz(c, O, dt, d, A->ndim, 0) < 0) return -1;
  for (int64_t i = 0; i < A->n; i++) {
    double v = 0.0;
    if (A->dt == DT_F32) v = (double) ((const float *) A->p)[i];
    else if (A->dt == DT_I64) v = (double) ((const int64_t *) A->p)[i];
    else if (A->dt == DT_I32) v = (double) ((const int32_t *) A->p)[i];
    else if (A->dt == DT_BOOL) v = (double) ((const uint8_t *) A->p)[i];
    if (dt == DT_F32) ((float *) O->p)[i] = (float) v;
    else if (dt == DT_I64) ((int64_t *) O->p)[i] = (int64_t) (v >= 0 ? floor(v) : ceil(v));
    else if (dt == DT_BOOL) ((uint8_t *) O->p)[i] = (uint8_t) (v != 0.0);
    else if (dt == DT_F16) {
      /* round-to-nearest-even fp32 -> fp16 (v2.1 casts to fp16 only in the decoder path) */
      uint32_t u;
      float f = (float) v;
      memcpy(&u, &f, 4);
      uint32_t sign = (u >> 16) & 0x8000u;
      int32_t e = (int32_t) ((u >> 23) & 0xff) - 127 + 15;
      uint32_t m = u & 0x800000u;
      uint16_t h;
      if (e <= 0) {
        h = (uint16_t) sign;                         /* subnormal fp16: flush to signed zero */
      } else if (e >= 31) {
        h = (uint16_t) (sign | 0x7c00u);             /* inf */
      } else {
        h = (uint16_t) (sign | ((uint32_t) e << 10) | (m >> 13));
      }
      ((uint16_t *) O->p)[i] = h;
    }
  }
  return 0;
}

static int op_split(pt_ctx *c, int axis, const pt_tv *A, const int64_t *sizes, int ns, pt_tv **outs) {
  int nd = A->ndim;
  if (axis < 0) axis += nd;
  int64_t d[PT_V21_MAX_RANK];
  int64_t off = 0;
  int es = dt_sz(A->dt);
  int64_t outer = 1, inner = 1;
  for (int i = 0; i < axis; i++) outer *= A->d[i];
  for (int i = axis + 1; i < nd; i++) inner *= A->d[i];
  for (int k = 0; k < ns; k++) {
    memcpy(d, A->d, sizeof(d));
    d[axis] = sizes[k];
    if (tv_alloc(c, outs[k], A->dt, d, nd) < 0) return -1;
    for (int64_t o = 0; o < outer; o++)
      memcpy((char *) outs[k]->p + o * sizes[k] * inner * es,
             (const char *) A->p + (o * A->d[axis] + off) * inner * es, (size_t) (sizes[k] * inner * es));
    off += sizes[k];
  }
  return 0;
}

static int op_unsqueeze(pt_ctx *c, const pt_tv *A, const int64_t *axes, int n, pt_tv *O) {
  int nd = A->ndim + n;
  if (nd > PT_V21_MAX_RANK) return -1;
  if (n > PT_V21_MAX_RANK) n = PT_V21_MAX_RANK;
  int64_t d[PT_V21_MAX_RANK];
  int ax[PT_V21_MAX_RANK];
  for (int i = 0; i < n; i++) {
    ax[i] = (int) axes[i];
    if (ax[i] < 0) ax[i] += nd;
  }
  for (int i = 0; i < nd; i++) d[i] = -1;
  for (int i = 0; i < n; i++) d[ax[i]] = 1;
  int k = 0;
  for (int i = 0; i < nd; i++) if (d[i] < 0) d[i] = A->d[k++];
  if (tv_alloc_nz(c, O, A->dt, d, nd, 0) < 0) return -1;   /* fully copied below */
  memcpy(O->p, A->p, (size_t) (A->n * dt_sz(A->dt)));
  return 0;
}

static int op_squeeze(pt_ctx *c, const pt_tv *A, const int64_t *axes, int n, pt_tv *O) {
  int64_t d[PT_V21_MAX_RANK];
  int nd = 0;
  int drop[PT_V21_MAX_RANK] = {0};
  if (n > PT_V21_MAX_RANK) n = PT_V21_MAX_RANK;
  if (n == 0) {
    for (int i = 0; i < A->ndim; i++) drop[i] = A->d[i] == 1;
  } else {
    for (int i = 0; i < n; i++) {
      int ax = (int) axes[i];
      if (ax < 0) ax += A->ndim;
      drop[ax] = 1;
    }
  }
  for (int i = 0; i < A->ndim; i++) if (!drop[i]) d[nd++] = A->d[i];
  if (nd == 0) {
    d[0] = 1;
    nd = 1;
  }
  if (tv_alloc_nz(c, O, A->dt, d, nd, 0) < 0) return -1;   /* fully copied below */
  memcpy(O->p, A->p, (size_t) (A->n * dt_sz(A->dt)));
  return 0;
}

static int op_range(pt_ctx *c, const pt_tv *a, const pt_tv *l, const pt_tv *d, pt_tv *O) {
  double lo = tv_scalar_f(a), hi = tv_scalar_f(l), st = tv_scalar_f(d);
  int64_t n = st > 0 ? (int64_t) ceil((hi - lo) / st) : 0;
  if (n < 0) n = 0;
  int64_t dd[1] = {n};
  int dt = a->dt == DT_I64 ? DT_I64 : DT_F32;
  if (tv_alloc(c, O, dt, dd, 1) < 0) return -1;
  for (int64_t i = 0; i < n; i++) {
    double v = lo + st * (double) i;
    if (dt == DT_I64) ((int64_t *) O->p)[i] = (int64_t) v;
    else ((float *) O->p)[i] = (float) v;
  }
  return 0;
}

static int op_const_of_shape(pt_ctx *c, const pt_tv *S, const void *fill, int fdt,
                             const int64_t *fv,
                             int fn, pt_tv *O) {
  int nd = (int) S->n;
  if (!S->p || nd < 1 || nd > PT_V21_MAX_RANK) return -1;
  int64_t d[PT_V21_MAX_RANK];
  for (int i = 0; i < nd; i++) d[i] = ((const int64_t *) S->p)[i];
  if (tv_alloc(c, O, fdt, d, nd) < 0) return -1;
  if (fdt == DT_F32) {
    const float *f = (const float *) fill;
    float *o = (float *) O->p;
    for (int64_t i = 0; i < O->n; i++) o[i] = f[fn == 1 ? 0 : i % fn];
  } else {
    const int64_t *f = (const int64_t *) fv;
    int64_t *o = (int64_t *) O->p;
    for (int64_t i = 0; i < O->n; i++) o[i] = f[fn == 1 ? 0 : i % fn];
  }
  return 0;
}

static int op_scatternd(pt_ctx *c, const pt_tv *D, const pt_tv *I, const pt_tv *U, pt_tv *O) {
  int64_t d[PT_V21_MAX_RANK];
  memcpy(d, D->d, sizeof(d));
  if (tv_alloc(c, O, D->dt, d, D->ndim) < 0) return -1;
  memcpy(O->p, D->p, (size_t) (D->n * dt_sz(D->dt)));
  int k = I->ndim > 0 ? (int) I->d[I->ndim - 1] : 0;
  int64_t nupd = k > 0 ? I->n / k : 0;
  int64_t strides[PT_V21_MAX_RANK];
  strides[D->ndim - 1] = 1;
  for (int i = D->ndim - 2; i >= 0; i--) strides[i] = strides[i + 1] * D->d[i + 1];
  int es = dt_sz(D->dt);
  /* each update writes the trailing block of the updates tensor: prod(updates.shape[k:]) (rank-
   * equal updates write a single scalar each) */
  int64_t usz = 1;
  for (int i = k; i < U->ndim; i++) usz *= U->d[i];
  for (int64_t u = 0; u < nupd; u++) {
    int64_t off = 0;
    for (int j = 0; j < k && j < D->ndim; j++) {
      int64_t ix = ((const int64_t *) I->p)[u * k + j];
      if (ix < 0) ix += D->d[j];
      if (ix < 0 || ix >= D->d[j]) continue;
      off += ix * strides[j];
    }
    if (off < 0 || off + usz > D->n) continue;
    memcpy((char *) O->p + off * es, (const char *) U->p + u * usz * es, (size_t) (usz * es));
  }
  return 0;
}

static int op_nonzero(pt_ctx *c, const pt_tv *A, pt_tv *O) {
  int64_t cnt = 0;
  for (int64_t i = 0; i < A->n; i++) {
    int nz = 0;
    if (A->dt == DT_F32) nz = ((const float *) A->p)[i] != 0.0f;
    else if (A->dt == DT_BOOL) nz = ((const uint8_t *) A->p)[i] != 0;
    else nz = ((const int64_t *) A->p)[i] != 0;
    if (nz) cnt++;
  }
  int64_t d[2] = {A->ndim, cnt};
  if (tv_alloc(c, O, DT_I64, d, 2) < 0) return -1;
  int64_t strides[PT_V21_MAX_RANK];
  strides[A->ndim - 1] = 1;
  for (int i = A->ndim - 2; i >= 0; i--) strides[i] = strides[i + 1] * A->d[i + 1];
  int64_t k = 0;
  for (int64_t i = 0; i < A->n; i++) {
    int nz = A->dt == DT_F32 ? ((const float *) A->p)[i] != 0.0f
           : A->dt == DT_BOOL ? ((const uint8_t *) A->p)[i] != 0 : ((const int64_t *) A->p)[i] != 0;
    if (!nz) continue;
    int64_t r = i;
    for (int q = 0; q < A->ndim; q++) {
      int64_t idx = r / strides[q];
      r %= strides[q];
      ((int64_t *) O->p)[q * cnt + k] = idx;
    }
    k++;
  }
  return 0;
}

/* ---------------- weight-bearing ops over the container's packed layouts ---------------- */
/* The fused epilogue, in fp32 exactly as the interpreter's elementwise ops compute it. */
static float pt_epi_cpu(float v, int op, float a) {
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

int pt_v21_conv_cpu(pt_ctx *c, const float *X, int64_t ldx, int64_t cin, float *Y, int64_t ldy,
                    int64_t n, int cout, int K, int stride, int pad, int dil, const float *wp,
                    const float *bias, int64_t batch, int wtab, int btab, int epi0, int epi1,
                    float a0, float a1) {
  (void) wtab;
  (void) btab;
  c->n_conv++;
  /* n is the OUTPUT length (the caller computed it from the graph shapes); both paths use that
   * convention so a CUDA swap cannot change the graph's shapes */
  int64_t Tout = n;
  int cout4 = (cout + 3) / 4 * 4;
  for (int64_t b = 0; b < batch; b++) {
    const float *x = X + b * cin * ldx;
    float *y = Y + b * cout4 * ldy;
    for (int co = 0; co < cout4; co++) {
      float bv = (bias && co < cout) ? bias[co] : 0.0f;   /* bias has cout entries; padded lanes 0 */
      for (int64_t t = 0; t < Tout; t++)
        y[(int64_t) co * ldy + t] = bv;
    }
    for (int k = 0; k < K; k++) {
      /* x is the zero-padded scratch whose column j is the graph's input column (j - pad);
       * tap k of output t reads column t*stride + k*dil, so indices stay in [0, T+2*pad). */
      const float *xs = x + (int64_t) (k * dil);
      const float *ws = wp + (int64_t) k * cin * 4;
      for (int blk = 0; blk < cout4 / 4; blk++) {
        const float *w = ws + (int64_t) blk * K * cin * 4;
        for (int ci = 0; ci < cin; ci++) {
          const float *xr = xs + (int64_t) ci * ldx;
          const float *wv = w + (int64_t) ci * 4;
          float w0 = wv[0], w1 = wv[1], w2 = wv[2], w3 = wv[3];
          float *yb = y + (int64_t) blk * 4 * ldy;
          for (int64_t t = 0; t < Tout; t++) {
            float *yr = yb + t;   /* the four rows this thread-block owns, at output column t */
            float xv = xr[t * stride];
            if (xv == 0.0f) continue;
            yr[0] += w0 * xv;
            yr[ldy] += w1 * xv;
            yr[2 * ldy] += w2 * xv;
            yr[3 * ldy] += w3 * xv;
          }
        }
      }
    }
  }
  /* fused epilogue: the same fp32 arithmetic the interpreter's elementwise nodes would have run */
  if (epi0 || epi1) {
    for (int64_t b = 0; b < batch; b++) {
      float *y = Y + b * cout4 * ldy;
      for (int co = 0; co < cout4; co++) {
        float *yr = y + (int64_t) co * ldy;
        for (int64_t t = 0; t < Tout; t++) {
          float v = yr[t];
          if (epi0)
            v = pt_epi_cpu(v, epi0, a0);
          if (epi1)
            v = pt_epi_cpu(v, epi1, a1);
          yr[t] = v;
        }
      }
    }
  }
  return 0;
}


static int devadd_on(void) {
  static int on = -1;
  if (on < 0)
    on = getenv("PT_V21_DEVADD") ? atoi(getenv("PT_V21_DEVADD")) : 0;
  return on;
}

/* Bisection cap: PT_V21_DEVADD_LIMIT stops the residency path after k device links per stream, so a
 * numerics failure can be attributed to the k-th link. Unlimited when unset. */
static int devadd_budget(pt_ctx *c) {
  static int cap = -1;
  if (cap < 0)
    cap = getenv("PT_V21_DEVADD_LIMIT") ? atoi(getenv("PT_V21_DEVADD_LIMIT")) : 1000000;
  if (c->devadd_used >= cap)
    return 0;
  c->devadd_used++;
  return 1;
}

/* Local shape of a device-Add link: the result flows out through at most one LeakyReLU, which the
 * Add kernel fuses (the same fp32 math the host un_op does). */
static void dev_add_shape(pt_ctx *c, const pt_node_def *nd, int *ft_out, int *epi_out, float *al_out) {
  int t = nd->out[0];
  *ft_out = t;
  *epi_out = 0;
  *al_out = 0.0f;
  if (g_n_cons[t] != 1)
    return;
  int cn = g_cons_node[t];
  if (cn < 0 || c->skip[cn] || pt_ndefs[cn].op != PT_LEAKYRELU || pt_ndefs[cn].in[0] != t)
    return;
  *ft_out = pt_ndefs[cn].out[0];
  *epi_out = 1;
  *al_out = pt_ndefs[cn].f0;
}

/* Is this Add link worth running on the device? Its result must reach a conv without ever being
 * read by a host op: the reader chain is [LeakyReLU] -> conv, or [LeakyReLU] -> another eligible Add
 * (the residual trees nest, so the chain recurses; depth is bounded because the graph is a DAG). */
/* ---------------- device residency eligibility (PT_V21_DEVADD)
 * A tensor may stay on the device when (a) every reader is an op the residency code runs on the
 * device (a conv consuming it as in[0], an Add, a LeakyReLU) and (b) its producer can publish it
 * there: a conv keeps its output for free, an Add or LeakyReLU runs on the device when one of its
 * operands is already resident (a host operand is uploaded, which is what the host path pays today).
 * The check walks backward through producers and forward over readers one level, so it terminates on
 * the graph's acyclicity. A result read by any other op is published on the host after the device
 * kernel: one download replaces the two operand downloads of the host path. */
static int dev_resident(pt_ctx *c, int t, int depth);

static int dev_runnable(pt_ctx *c, int node, int depth) {
  if (depth > 16 || node < 0 || node >= PT_N_NODE)
    return 0;
  const pt_node_def *nd = &pt_ndefs[node];
  if (nd->op != PT_ADD && nd->op != PT_LEAKYRELU)
    return 0;
  for (int j = 0; j < PT_V21_MAX_IN; j++)
    if (nd->in[j] >= 0 && dev_resident(c, nd->in[j], depth + 1))
      return 1;
  return 0;
}

static int dev_resident(pt_ctx *c, int t, int depth) {
  if (depth > 16 || t < 0 || t >= PT_N_TENSOR)
    return 0;
  int cnt = g_n_cons[t];
  if (cnt < 1 || cnt > PT_V21_MAX_CONS)
    return 0;
  for (int k = 0; k < cnt; k++) {
    int cn = g_cons_list[t][k];
    if (cn < 0 || cn >= PT_N_NODE || c->skip[cn])
      return 0;
    const pt_node_def *cd = &pt_ndefs[cn];
    if (cd->op == PT_CONV && cd->in[0] == t)
      continue;
    if (cd->op != PT_ADD && cd->op != PT_LEAKYRELU)
      return 0;
  }
  int pn = g_prod[t];
  if (pn < 0 || c->skip[pn])
    return 0;
  if (pt_ndefs[pn].op == PT_CONV)
    return 1;
  if ((pt_ndefs[pn].op == PT_ADD || pt_ndefs[pn].op == PT_LEAKYRELU) &&
      dev_runnable(c, pn, depth + 1))
    return 1;
  return 0;
}

static void dev_res_sync_host(pt_ctx *c, pt_tv *t) {
  if (!c->use_cuda || !t->dev_res || !t->p || t->n <= 0)
    return;
  pt_v21_dev_sync_host(c, t->p, t->dev_p, t->n);
  t->dev_res = 0;
  t->dev_p = NULL;
}

static int dev_add_link(pt_ctx *c, const pt_node_def *nd, pt_tv *A, pt_tv *B, pt_tv *O) {
  int dbg = getenv("PT_V21_DEVDBG") ? 1 : 0;
  int ni = (int) (nd - pt_ndefs);
  if (!devadd_on() || !c->use_cuda)
    return 0;
  if (A->dt != DT_F32 || B->dt != DT_F32 || A->n <= 0 || A->n != B->n) {
    if (dbg) fprintf(stderr, "DEVDBG %d dt/n A=%d n=%lld B=%d n=%lld\n", ni, A->dt, (long long) A->n, B->dt, (long long) B->n);
    dev_res_sync_host(c, A);
    dev_res_sync_host(c, B);
    return 0;
  }
  if (!A->alive || !B->alive || (!A->dev_res && !B->dev_res)) {
    if (dbg) fprintf(stderr, "DEVDBG %d alive a=%d b=%d devres a=%d b=%d\n", ni, A->alive, B->alive, A->dev_res, B->dev_res);
    return 0;
  }
  int ft, epi;
  float alpha;
  dev_add_shape(c, nd, &ft, &epi, &alpha);
  if (!devadd_budget(c) || !dev_runnable(c, ni, 0)) {
    dev_res_sync_host(c, A);
    dev_res_sync_host(c, B);
    if (dbg) fprintf(stderr, "DEVDBG %d chain=0 cons(out)=%d next_op=%d\n", ni, g_n_cons[ft],
                     g_cons_node[ft] >= 0 ? (int) pt_ndefs[g_cons_node[ft]].op : -1);
    return 0;
  }
  int64_t n = A->n;
  const float *ad = A->dev_res ? (const float *) A->dev_p : NULL;
  const float *bd = B->dev_res ? (const float *) B->dev_p : NULL;
  if (tv_alloc_nz(c, O, DT_F32, A->d, A->ndim, 0) < 0)   /* the operands share the graph shape */
    return 0;
  /* each operand that is not resident is uploaded from its own host copy */
  if (!ad) {
    ad = pt_v21_dev_upload(c, A->p, n);
    if (!ad) {
      dev_res_sync_host(c, A);
      dev_res_sync_host(c, B);
      return 0;
    }
  }
  if (!bd) {
    bd = pt_v21_dev_upload(c, B->p, n);
    if (!bd) {
      dev_res_sync_host(c, A);
      dev_res_sync_host(c, B);
      return 0;
    }
  }
  float *y = (float *) pt_v21_dev_alloc(c, (size_t) n * 4);
  if (!y || pt_v21_devadd_run(c, ad, bd, y, n, epi, alpha)) {
    if (y)
      pt_v21_dev_free(c, y);
    dev_res_sync_host(c, A);
    dev_res_sync_host(c, B);
    return 0;   /* sub-arena full or launch failure: recover the operands, run on the host */
  }
  if (!dev_resident(c, ft, 0)) {
    /* chain terminator: run the Add on the device, publish the result on the host, release the block */
    pt_v21_dev_sync_host(c, O->p, y, n);   /* the result's readers are host ops */
    pt_v21_dev_free(c, y);
    O->dev_res = 0;
    O->dev_p = NULL;
    dev_res_sync_host(c, A);
    dev_res_sync_host(c, B);
    if (getenv("PT_V21_CHAINLOG"))
      fprintf(stderr, "devadd-term: out %d elems=%lld\n", nd->out[0], (long long) n);
    return 1;
  }
  O->dev_res = 1;
  c->dev_live++;
  O->dev_p = y;   /* its host buffer is stale on purpose: its only reader is the next conv */
  if (ft != nd->out[0]) {
    /* the fused LeakyReLU is skipped and its tensor aliases this one, exactly as conv epilogue
       fusion does: the host buffer is freed once, by the last tensor that owns it */
    int cn = g_cons_node[nd->out[0]];
    c->skip[cn] = 1;
    c->V[ft] = c->V[nd->out[0]];
    c->V[ft].dev = 2;
    c->V[ft].alive = 1;
    c->V[nd->out[0]].dev = 2;
  }
  if (getenv("PT_V21_CHAINLOG"))
    fprintf(stderr, "devadd: out %d fused_epi=%d elems=%lld\n", nd->out[0], epi, (long long) n);
  return 1;
}


/* phase/tap conv_transpose1d over the packed [s*C4/4][ntap][Cin][4] layout, with the tail flush */
int pt_v21_convT_cpu(pt_ctx *c, const float *X, int64_t ldx, int64_t cin, int64_t tin, float *Y,
                 int64_t ldy, int cout, int s, int pad, int K, const float *wp, const float *bias,
                 int64_t batch, int wtab, int btab) {
  (void) wtab;
  (void) btab;
  c->n_convT++;
  int ntap = (K + s - 1) / s;
  int cout4 = (cout + 3) / 4 * 4;
  int64_t Q = tin + ntap - 1;
  int64_t Lout = (tin - 1) * s - 2 * pad + K;
  for (int64_t b = 0; b < batch; b++) {
  const float *xb = X + b * cin * ldx;
  float *yb = Y + b * cout4 * ldy;
  for (int co = 0; co < cout4; co++)
    for (int64_t t = 0; t < Lout; t++)
      yb[(int64_t) co * ldy + t] = (bias && co < cout) ? bias[co] : 0.0f;
  for (int64_t q = 0; q < Q; q++) {
    for (int r = 0; r < s; r++) {
      const float *wr = wp + ((int64_t) r * cout4 / 4) * ntap * cin * 4;
      for (int blk = 0; blk < cout4 / 4; blk++) {
        const float *w = wr + (int64_t) blk * ntap * cin * 4;
        for (int t = 0; t < ntap; t++) {
          /* the delay line at step q feeds tap t the input column (q - t): taps ahead of q read
           * the zero head of the line, and columns >= tin are the flushed tail */
          if (t > q) continue;
          int64_t tq = q - t;
          if (tq >= tin) continue;
          const float *xr = xb + (int64_t) tq * 1;      /* column tq per channel below */
          for (int ci = 0; ci < cin; ci++) {
            float xv = xr[(int64_t) ci * ldx];
            if (xv == 0.0f) continue;
            const float *wv = w + ((int64_t) t * cin + ci) * 4;
            for (int p = 0; p < 4; p++) {
              int co = blk * 4 + p;
              if (co >= cout4) continue;
              int64_t oi = q * s + r - pad;
              if (oi < 0 || oi >= Lout) continue;   /* the head/tail trim the python oracle masks */
              yb[(int64_t) co * ldy + oi] += wv[p] * xv;
            }
          }
        }
      }
    }
  }
  }   /* batch */
  return 0;
}

const char *pt_v21_ekind_report(pt_ctx *c, char *buf, int bufsz) {
  /* top op kinds by accumulated host time */
  int order[96];
  for (int i = 0; i < 96; i++) order[i] = i;
  for (int i = 0; i < 96; i++)
    for (int j = i + 1; j < 96; j++)
      if (c->t_ekind[order[j]] > c->t_ekind[order[i]]) {
        int t = order[i];
        order[i] = order[j];
        order[j] = t;
      }
  buf[0] = 0;
  int off = 0;
  static const char *names[96] = {
#define PT_X(n) #n,
    PT_OPS(PT_X)
#undef PT_X
  };
  for (int k = 0; k < 96 && off < bufsz - 40; k++) {
    int o = order[k];
    if (c->t_ekind[o] < 5e-5)
      break;
    if (k >= 30)
      break;
    off += snprintf(buf + off, bufsz - off, " %s=%.1f(%ld)", names[o], c->t_ekind[o] * 1000.0,
                    c->n_ekind[o]);
  }
  return buf;
}

double pt_v21_stat_ms(pt_ctx *c, int which) {
  switch (which) {
    case 0: return c->g_alloc_ms;
    case 1: return c->g_copy_ms;
    case 2: return c->g_kernel_ms;
    case 3: return c->g_free_ms;
    default: return 0.0;
  }
}

int pt_v21_matmul_cpu(pt_ctx *c, const float *A, int64_t m, int64_t k, int64_t lda, const float *B,
                  int64_t ldb, int64_t nn, float *C, int64_t ldc) {
  c->n_matmul++;
  for (int64_t i = 0; i < m; i++) {
    const float *ar = A + i * lda;
    float *cr = C + i * ldc;
    for (int64_t j = 0; j < nn; j++) cr[j] = 0.0f;
    for (int64_t p = 0; p < k; p++) {
      float a = ar[p];
      if (a == 0.0f) continue;
      const float *bp = B + (int64_t) p * ldb;
      for (int64_t j = 0; j < nn; j++) cr[j] += a * bp[j];
    }
  }
  return 0;
}
/* Dispatchers used by the graph. The CUDA build (pt_v21_cuda.cu) provides the same names and
 * forwards to the CPU kernels when ctx->use_cuda is 0, so it defines PT_V21_CUDA and skips these. */
#ifndef PT_V21_CUDA
int pt_v21_cuda_init(pt_ctx *c, int verbose) {
  (void) c;
  (void) verbose;
  return -1;   /* this binary was built without CUDA */
}
void pt_v21_cuda_shutdown(pt_ctx *c) { (void) c; }
void pt_v21_cuda_active(int delta) { (void) delta; }
int pt_v21_cuda_keepwarm(int iters) { (void) iters; return -1; }
long pt_v21_gpu_cur_freq(void) { return -1; }
double pt_v21_cuda_weights_mib(pt_ctx *c) { (void) c; return 0.0; }
int pt_v21_cuda_prewarm(void) { return 0; }
int pt_v21_conv_pads_in_kernel(pt_ctx *c) { (void) c; return 0; }
/* device-residency entry points: a CPU-only build never uses them (PT_V21_DEVADD is off by
 * default and there is no device), but the interpreter references them, so they must link. */
void pt_v21_dev_reclaim(pt_ctx *c) { (void) c; }
float *pt_v21_dev_upload(pt_ctx *c, const void *h, int64_t n) { (void) c; (void) h; (void) n; return 0; }
void pt_v21_dev_sync_host(pt_ctx *c, void *host, const void *dev, int64_t n) {
  (void) c; (void) host; (void) dev; (void) n;
}
void *pt_v21_dev_alloc(pt_ctx *c, size_t bytes) { (void) c; (void) bytes; return 0; }
void pt_v21_dev_free(pt_ctx *c, void *p) { (void) c; (void) p; }
int pt_v21_devadd_run(pt_ctx *c, const float *a, const float *b, float *y, int64_t n, int epi,
                      float alpha) { (void) c; (void) a; (void) b; (void) y; (void) n; (void) epi; (void) alpha; return 0; }
int pt_v21_devleaky_run(pt_ctx *c, const float *a, float *y, int64_t n, float alpha) {
  (void) c; (void) a; (void) y; (void) n; (void) alpha; return 0;
}

int pt_v21_conv(pt_ctx *c, const float *X, int64_t ldx, int64_t cin, float *Y, int64_t ldy,
                int64_t n, int cout, int K, int stride, int pad, int dil, const float *wp,
                const float *bias, int64_t batch, int wtab, int btab, int epi0, int epi1, float a0,
                float a1) {
  return pt_v21_conv_cpu(c, X, ldx, cin, Y, ldy, n, cout, K, stride, pad, dil, wp, bias, batch,
                         wtab, btab, epi0, epi1, a0, a1);
}
int pt_v21_convT(pt_ctx *c, const float *X, int64_t ldx, int64_t cin, int64_t tin, float *Y,
                 int64_t ldy, int cout, int s, int pad, int K, const float *wp, const float *bias,
                 int64_t batch, int wtab, int btab) {
  return pt_v21_convT_cpu(c, X, ldx, cin, tin, Y, ldy, cout, s, pad, K, wp, bias, batch, wtab, btab);
}
int pt_v21_matmul(pt_ctx *c, const float *A, int64_t m, int64_t k, int64_t lda, const float *B,
                  int64_t ldb, int64_t nn, float *C, int64_t ldc) {
  return pt_v21_matmul_cpu(c, A, m, k, lda, B, ldb, nn, C, ldc);
}
#endif


/* ---------------- run ---------------- */
static const int64_t *tvals(const pt_tv *t) { return (const int64_t *) t->p; }

static int nin(const pt_node_def *nd) {
  int k = 0;
  for (int i = 0; i < PT_V21_MAX_IN; i++) if (nd->in[i] >= 0) k++;
  return k;
}

extern const char *const pt_eq_tab[];
#define pt_op_eq(i) pt_eq_tab[i]

static int is_input_tensor(int tid) { return strncmp(pt_tdefs[tid].origin, "IN", 2) == 0; }

int pt_v21_checkpoints(pt_ctx *c, const char *dir, const int *ids, int n) {
  c->ck_n = n;
  c->ck_ids = ids;
  snprintf(c->ck_dir, sizeof(c->ck_dir), "%s", dir);
  return 0;
}

/* checkpoints must carry the tensor's own dtype, or the parity diff reads int64 payloads as
 * float bits and reports nonsense */
static int ck_save_typed(const char *path, const pt_tv *t) {
  if (t->dt == DT_F32) return pt_npy_save_f32_v21(path, (const float *) t->p, t->ndim, t->d);
  if (t->dt == DT_I64) return pt_npy_save_i64_v21(path, (const int64_t *) t->p, t->ndim, t->d);
  if (t->dt == DT_BOOL) return pt_npy_save_b1_v21(path, (const uint8_t *) t->p, t->ndim, t->d);
  return -1;
}

static void ck_save(pt_ctx *c, int tid) {
  if (c->ck_n <= 0) return;
  for (int i = 0; i < c->ck_n; i++) {
    if (c->ck_ids[i] == tid) {
      pt_tv *t = &c->V[tid];
      char path[640];
      char flat[512];
      const char *nm = pt_graph_tnames[tid];
      int q = 0;
      for (; nm[q] && q < 500; q++) flat[q] = nm[q] == '/' ? '_' : nm[q];
      flat[q] = 0;
      snprintf(path, sizeof(path), "%s/%s.npy", c->ck_dir, flat);
      ck_save_typed(path, t);
      break;
    }
  }
}

/* ---- streaming decoder (overlap-save) -------------------------------------------------------
 * The decoder tail is everything from /dec/conv_pre to the waveform: 245 nodes of conv, convT,
 * leakyrelu, mul, add, slice, reshape and the iSTFT basis einsums. Checked against the generated
 * tables: no PT_SCATTERND, no PT_RANGE and no global reduction in that range, and its slices are
 * relative (1, -1) -- so it is local in time and can be executed on sliding windows. The encoder
 * (text -> frames) runs once, which is what keeps a single decoder start per utterance.
 */
static int g_stream_tail = -1, g_stream_src = -1;   /* node split: read-only once located */
typedef struct { int id, dt, ndim; int64_t d[PT_V21_MAX_RANK], n, bytes; float *buf; } pt_stream_seed;
/* All mutable streaming state is PER CONTEXT: the server runs one context per worker thread and
 * several workers stream at once (keeping it in globals crashed the server under 4 concurrent
 * requests). */
typedef struct {
  float *frames;                  /* private copy of the frames tensor */
  int64_t n, d[PT_V21_MAX_RANK];
  int ndim, dt, hop;
  pt_stream_seed *seeds;          /* tensors the tail consumes from the encoder part */
  int nseeds;
} pt_stream_state;

static void stream_locate(void) {
  if (g_stream_tail >= 0) return;
  for (int i = 0; i < PT_N_NODE; i++) {
    const pt_node_def *nd = &pt_ndefs[i];
    if (nd->op != PT_CONV || nd->out[0] < 0) continue;
    const char *nm = pt_graph_tnames[nd->out[0]];
    if (nm && !strncmp(nm, "/dec/conv_pre/", 14)) {
      g_stream_tail = i;
      g_stream_src = nd->in[0];
      return;
    }
  }
  g_stream_tail = PT_N_NODE;   /* not found: streaming unavailable */
}

static pt_stream_state *st(pt_ctx *c) {
  if (!c->stream_state) c->stream_state = calloc(1, sizeof(pt_stream_state));
  return (pt_stream_state *) c->stream_state;
}

int pt_v21_stream_open(pt_ctx *c, const pt_tv *feeds, int64_t *frames, int *hop) {
  stream_locate();
  if (g_stream_tail >= PT_N_NODE || g_stream_src < 0) return -1;
  pt_stream_state *S = st(c);
  if (!S) return -1;
  /* drop any previous state for this context */
  for (int q = 0; q < S->nseeds; q++) free(S->seeds[q].buf);
  free(S->seeds);
  S->seeds = NULL;
  S->nseeds = 0;
  free(S->frames);
  S->frames = NULL;
  c->run_from = 0;
  c->run_to = g_stream_tail;
  float *w = 0;
  int64_t wn = 0;
  int rc = pt_v21_run(c, feeds, &w, &wn);
  c->run_from = c->run_to = 0;
  if (rc) return -1;
  pt_tv *f = &c->V[g_stream_src];
  if (!f->p || f->n <= 0) return -1;
  S->ndim = f->ndim;
  S->dt = f->dt;
  S->n = f->n;
  for (int q = 0; q < PT_V21_MAX_RANK; q++) S->d[q] = f->d[q];
  S->frames = (float *) malloc((size_t) S->n * 4);
  if (!S->frames) return -1;
  memcpy(S->frames, f->p, (size_t) S->n * 4);
  int64_t T = f->d[f->ndim - 1];
  /* capture every tensor the tail consumes from outside itself */
  int cap = 0;
  for (int idx = g_stream_tail; idx < PT_N_NODE; idx++) {
    const pt_node_def *nd = &pt_ndefs[idx];
    for (int k = 0; k < PT_V21_MAX_IN; k++) {
      int t = nd->in[k];
      if (t < 0 || t == g_stream_src) continue;
      if (g_prod[t] >= g_stream_tail) continue;   /* produced inside the tail: recomputed */
      if (!c->V[t].p || c->V[t].n <= 0) continue;
      int dup = 0;
      for (int q = 0; q < S->nseeds; q++)
        if (S->seeds[q].id == t) { dup = 1; break; }
      if (dup) continue;
      if (S->nseeds == cap) {
        cap = cap ? cap * 2 : 32;
        pt_stream_seed *nb = (pt_stream_seed *) realloc(S->seeds, (size_t) cap * sizeof(*nb));
        if (!nb) break;
        S->seeds = nb;
      }
      pt_stream_seed *sd = &S->seeds[S->nseeds];
      sd->id = t;
      sd->dt = c->V[t].dt;
      sd->ndim = c->V[t].ndim;
      sd->n = c->V[t].n;
      for (int q = 0; q < PT_V21_MAX_RANK; q++) sd->d[q] = c->V[t].d[q];
      sd->bytes = sd->n * dt_sz(c->V[t].dt);
      sd->buf = (float *) malloc((size_t) sd->bytes);
      if (!sd->buf) continue;
      memcpy(sd->buf, c->V[t].p, (size_t) sd->bytes);
      S->nseeds++;
    }
  }
  if (frames) *frames = T;
  /* hop (samples per frame) is a property of the model, not of the request: learn it from one
   * short window the first time and reuse it, instead of an extra decoder pass per request */
  static volatile int model_hop;
  if (model_hop <= 0) {
    int64_t probe = T < 16 ? T : 16;
    float *pp = 0;
    int64_t pn = 0;
    if (pt_v21_stream_chunk(c, 0, probe, &pp, &pn) == 0 && pn > 0 && pn % probe == 0)
      model_hop = (int) (pn / probe);
  }
  S->hop = model_hop;
  if (hop) *hop = S->hop;
  return S->hop > 0 ? 0 : -1;
}

int pt_v21_stream_chunk(pt_ctx *c, int64_t a, int64_t win, float **pcm, int64_t *n) {
  pt_stream_state *S = (pt_stream_state *) c->stream_state;
  if (g_stream_tail >= PT_N_NODE || !S || !S->frames) return -1;
  int64_t T = S->d[S->ndim - 1];
  if (a < 0) a = 0;
  if (win < 1) win = 1;
  if (a + win > T) win = T - a;
  if (win < 1) return -1;
  int64_t per = S->n / T;                    /* elements per frame (channels) */
  pt_tv inj;
  memset(&inj, 0, sizeof inj);
  inj.dt = S->dt;
  inj.ndim = S->ndim;
  for (int q = 0; q < PT_V21_MAX_RANK; q++) inj.d[q] = S->d[q];
  inj.d[S->ndim - 1] = win;
  inj.n = per * win;
  inj.p = (float *) malloc((size_t) inj.n * 4);
  if (!inj.p) return -1;
  int64_t lead = 1;
  for (int q = 0; q < S->ndim - 1; q++) lead *= S->d[q];
  for (int64_t ch = 0; ch < lead; ch++)
    memcpy((float *) inj.p + ch * win, S->frames + ch * T + a, (size_t) win * 4);
  c->stream_inject = &inj;
  c->stream_inject_id = g_stream_src;
  c->stream_seeds = S->seeds;
  c->stream_nseeds = S->nseeds;
  c->run_from = g_stream_tail;
  c->run_to = PT_N_NODE;
  int rc = pt_v21_run(c, NULL, pcm, n);
  c->run_from = c->run_to = 0;
  c->stream_inject = NULL;
  c->stream_inject_id = -1;
  c->stream_seeds = NULL;
  c->stream_nseeds = 0;
  free(inj.p);
  return rc;
}

void pt_v21_stream_close(pt_ctx *c) {
  pt_stream_state *S = (pt_stream_state *) c->stream_state;
  if (!S) return;
  for (int q = 0; q < S->nseeds; q++) free(S->seeds[q].buf);
  free(S->seeds);
  free(S->frames);
  free(S);
  c->stream_state = NULL;
}

static int pt_v21_run_graph(pt_ctx *c, const pt_tv *feeds, float **wav, int64_t *wav_n);
int pt_v21_run(pt_ctx *c, const pt_tv *feeds, float **wav, int64_t *wav_n) {
  if (c->use_cuda) pt_v21_cuda_active(1);
  int rc = pt_v21_run_graph(c, feeds, wav, wav_n);
  if (c->use_cuda) pt_v21_cuda_active(-1);
  return rc;
}
static int pt_v21_run_graph(pt_ctx *c, const pt_tv *feeds, float **wav, int64_t *wav_n) {
  double t0 = pt_v21_now();
  c->n_conv = c->n_convT = c->n_matmul = 0;
  c->t_conv = c->t_convT = c->t_matmul = 0.0;
  memset(c->t_ekind, 0, sizeof(c->t_ekind));
  memset(c->n_ekind, 0, sizeof(c->n_ekind));
  for (int i = 0; i < PT_N_TENSOR; i++) {
    pt_tv *t = &c->V[i];
    if (t->dev != 2) tv_free(c, t);
  }
  /* A context can run many requests (server mode). Fusion state is per run: the previous run's
   * skip flags would stop the producer from re-fusing while its consumers still skip (their
   * outputs never computed -> NULL operands), and its dev==2 aliases point at buffers their owners
   * just freed. Every node re-executes, so a clean tensor table is the correct starting state. */
  memset(c->V, 0, sizeof(pt_tv) * PT_N_TENSOR);
  if (c->skip) memset(c->skip, 0, PT_N_NODE);
  c->conv_in_dev = NULL;
  c->conv_keep = c->conv_keep_res = 0;
  c->conv_out_dev = NULL;
  int idx_from = c->run_from > 0 ? c->run_from : 0;
  int idx_to = c->run_to > 0 && c->run_to <= PT_N_NODE ? c->run_to : PT_N_NODE;
  /* graph inputs (ids in table order, all origin "IN") alias the caller's feeds */
  int ni = 0;
  for (int i = 0; i < PT_N_TENSOR && idx_from == 0 && strncmp(pt_tdefs[i].origin, "IN", 2) == 0; i++) {
    pt_tv *t = &c->V[i];
    t->dt = feeds[ni].dt;
    t->ndim = feeds[ni].ndim;
    for (int q = 0; q < PT_V21_MAX_RANK; q++) t->d[q] = q < feeds[ni].ndim ? feeds[ni].d[q] : 1;
    t->n = tv_n(t);
    t->p = feeds[ni].p;
    t->dev = 2;
    t->alive = 1;
    ni++;
  }
  /* streaming decoder: re-seed the tensors the tail consumes from the encoder part */
  if (c->stream_seeds && c->stream_nseeds > 0) {
    const pt_stream_seed *sd = (const pt_stream_seed *) c->stream_seeds;
    for (int q = 0; q < c->stream_nseeds; q++) {
      int t = sd[q].id;
      if (t < 0 || t >= PT_N_TENSOR || !sd[q].buf) continue;
      pt_tv *tv = &c->V[t];
      memset(tv, 0, sizeof *tv);
      tv->dt = (int8_t) sd[q].dt;
      tv->ndim = sd[q].ndim;
      for (int r = 0; r < PT_V21_MAX_RANK; r++) tv->d[r] = sd[q].d[r];
      tv->n = sd[q].n;
      tv->p = sd[q].buf;
      tv->dev = 2;      /* caller-owned: never freed by the interpreter */
      tv->alive = 1;
    }
  }
  /* streaming decoder: seed the frames tensor with the caller's window (dev 2 = never freed) */
  if (c->stream_inject && c->stream_inject_id >= 0 && c->stream_inject_id < PT_N_TENSOR) {
    c->V[c->stream_inject_id] = *c->stream_inject;
    c->V[c->stream_inject_id].dev = 2;
    c->V[c->stream_inject_id].alive = 1;
  }
  for (int idx = idx_from; idx < idx_to; idx++) {
    const pt_node_def *nd = &pt_ndefs[idx];
    pt_tv *in[PT_V21_MAX_IN], *out[PT_V21_MAX_OUT];
    for (int i = 0; i < PT_V21_MAX_IN; i++) {
      in[i] = nd->in[i] >= 0 ? &c->V[nd->in[i]] : 0;
      if (in[i]) resolve(c, nd->in[i]);
    }
    for (int i = 0; i < PT_V21_MAX_OUT; i++) out[i] = nd->out[i] >= 0 ? &c->V[nd->out[i]] : 0;
    int r = 0;
    g_node = idx;
    g_op = nd->op;
    if (g_trace == idx) dump_inputs(idx);
    if (c->skip && c->skip[idx]) {
      /* folded into a producer kernel; only its tensor bookkeeping is left to do */
      for (int i = 0; i < PT_V21_MAX_OUT; i++) {
        if (nd->out[i] >= 0) ck_save(c, nd->out[i]);
      }
      if (c->prof) {
        c->t_ekind[nd->op] += 0.0;
        c->n_ekind[nd->op]++;
      }
      for (int t = 0; t < PT_N_TENSOR; t++)
        ;   /* the shared per-node free below is skipped: this node owns nothing it allocated */
      continue;
    }
    /* Host readers of device-resident tensors must see a valid host copy. The conv keeps its chained
     * tensor's host copy current (the wrapper always downloads it), and the Add/LeakyReLU device paths
     * sync their own operands on fallback, so the residency tensors that can be stale here are exactly
     * the ones this sweep catches. */
    if (c->dev_live && c->use_cuda)
      for (int i = 0; i < PT_V21_MAX_IN; i++)
        if (in[i] && in[i]->dev_res && nd->op != PT_CONV && nd->op != PT_ADD && nd->op != PT_LEAKYRELU)
          dev_res_sync_host(c, in[i]);
    double tA = c->prof ? pt_v21_now() : 0.0;
    switch (nd->op) {
      case PT_CONST: {
        /* constants live in the container (floats) or the int pool: alias, never copy */
        int t = nd->out[0];
        if (resolve(c, t) < 0) {
          fprintf(stderr, "pt_v21: constant %s has no payload\n", pt_graph_tnames[t]);
          return -1;
        }
        c->V[t].alive = 1;
        if (g_trace == idx) {
          fprintf(stderr, "const %s ctab-origin=%s dt=%d ndim=%d n=%lld val=%g\n",
                  pt_graph_tnames[t], pt_tdefs[t].origin, c->V[t].dt, c->V[t].ndim,
                  (long long) c->V[t].n, c->V[t].dt == DT_F32 ? (double) ((const float *) c->V[t].p)[0] : -999.0);
        }
        break;
      }
      case PT_ADD: {
        if (dev_add_link(c, nd, in[0], in[1], out[0]))
          r = 0;   /* done on the device */
        else
          r = bin_op(c, PT_ADD, in[0], in[1], out[0]);
        break;
      }
      case PT_SUB: r = bin_op(c, PT_SUB, in[0], in[1], out[0]); break;
      case PT_MUL: r = bin_op(c, PT_MUL, in[0], in[1], out[0]); break;
      case PT_DIV: r = bin_op(c, PT_DIV, in[0], in[1], out[0]); break;
      case PT_POW: r = bin_op(c, PT_POW, in[0], in[1], out[0]); break;
      case PT_MAX: r = bin_op(c, PT_MAX, in[0], in[1], out[0]); break;
      case PT_MIN: r = bin_op(c, PT_MIN, in[0], in[1], out[0]); break;
      case PT_EQUAL: r = bin_op(c, PT_EQUAL, in[0], in[1], out[0]); break;
      case PT_LESS: r = bin_op(c, PT_LESS, in[0], in[1], out[0]); break;
      case PT_GREATER: r = bin_op(c, PT_GREATER, in[0], in[1], out[0]); break;
      case PT_GEQ: r = bin_op(c, PT_GEQ, in[0], in[1], out[0]); break;
      case PT_LEQ: r = bin_op(c, PT_LEQ, in[0], in[1], out[0]); break;
      case PT_AND: r = bin_op(c, PT_AND, in[0], in[1], out[0]); break;
      case PT_OR: r = bin_op(c, PT_OR, in[0], in[1], out[0]); break;
      case PT_XOR: r = bin_op(c, PT_XOR, in[0], in[1], out[0]); break;
      case PT_SUM: {
        int k = nin(nd);
        pt_tv acc = *in[0];
        int have_alloc = 0;
        for (int i = 1; i < k; i++) {
          pt_tv tmp;
          bin_op(c, PT_ADD, &acc, in[i], &tmp);
          if (have_alloc) tv_free(c, &acc);
          acc = tmp;
          have_alloc = 1;
        }
        *out[0] = acc;
        break;
      }
      case PT_NEG: r = un_op(c, PT_NEG, in[0], 0, out[0], 0); break;
      case PT_EXP: r = un_op(c, PT_EXP, in[0], 0, out[0], 0); break;
      case PT_SIN: r = un_op(c, PT_SIN, in[0], 0, out[0], 0); break;
      case PT_COS: r = un_op(c, PT_COS, in[0], 0, out[0], 0); break;
      case PT_TANH: r = un_op(c, PT_TANH, in[0], 0, out[0], 0); break;
      case PT_SIGMOID: r = un_op(c, PT_SIGMOID, in[0], 0, out[0], 0); break;
      case PT_RELU: r = un_op(c, PT_RELU, in[0], 0, out[0], 0); break;
      case PT_LEAKYRELU: {
        /* Device-resident LeakyReLU (PT_V21_DEVADD): the input is a resident conv/Add output and all
         * readers of the result are chainable, so the node is one kernel pass with no copies. */
        if (devadd_on() && c->use_cuda && in[0]->dev_res && in[0]->dt == DT_F32 && in[0]->n > 0 &&
            devadd_budget(c) && dev_runnable(c, idx, 0)) {
          int64_t n = in[0]->n;
          float *dst = (float *) pt_v21_dev_alloc(c, (size_t) n * 4);
          if (dst && tv_alloc_nz(c, out[0], DT_F32, in[0]->d, in[0]->ndim, 0) >= 0 &&
              pt_v21_devleaky_run(c, (const float *) in[0]->dev_p, dst, n, nd->f0) == 0) {
            if (dev_resident(c, nd->out[0], 0)) {
              out[0]->dev_res = 1;
              c->dev_live++;
              out[0]->dev_p = dst;   /* host buffer stale by design: only device readers remain */
            } else {
              pt_v21_dev_sync_host(c, out[0]->p, dst, n);   /* a host reader ends the chain */
              pt_v21_dev_free(c, dst);
            }
            dev_res_sync_host(c, in[0]);   /* the consumed operand's host buffer is now stale */
            r = 0;
            if (getenv("PT_V21_CHAINLOG"))
              fprintf(stderr, "devleaky: out %d elems=%lld\n", nd->out[0], (long long) n);
            break;
          }
          if (dst)
            pt_v21_dev_free(c, dst);
          dev_res_sync_host(c, in[0]);   /* the host path becomes a reader: make it valid */
        }
        r = un_op(c, PT_LEAKYRELU, in[0], 0, out[0], nd->f0);
        break;
      }
      case PT_SOFTPLUS: r = un_op(c, PT_SOFTPLUS, in[0], 0, out[0], 0); break;
      case PT_ERF: r = un_op(c, PT_ERF, in[0], 0, out[0], 0); break;
      case PT_ABS: r = un_op(c, PT_ABS, in[0], 0, out[0], 0); break;
      case PT_FLOOR: r = un_op(c, PT_FLOOR, in[0], 0, out[0], 0); break;
      case PT_CEIL: r = un_op(c, PT_CEIL, in[0], 0, out[0], 0); break;
      case PT_ROUND: r = un_op(c, PT_ROUND, in[0], 0, out[0], 0); break;
      case PT_SQRT: r = un_op(c, PT_SQRT, in[0], 0, out[0], 0); break;
      case PT_SIGN: r = un_op(c, PT_SIGN, in[0], 0, out[0], 0); break;
      case PT_RECIPROCAL: r = un_op(c, PT_RECIPROCAL, in[0], 0, out[0], 0); break;
      case PT_LOG: r = un_op(c, PT_LOG, in[0], 0, out[0], 0); break;
      case PT_CLIP: r = un_op(c, PT_CLIP, in[0], in[1], out[0], 0); break;
      case PT_CAST: r = op_cast(c, in[0], nd->i0, out[0]); break;
      case PT_NOT: {
        int64_t d[PT_V21_MAX_RANK];
        memcpy(d, in[0]->d, sizeof(d));
        tv_alloc(c, out[0], DT_BOOL, d, in[0]->ndim);
        for (int64_t i = 0; i < in[0]->n; i++)
          ((uint8_t *) out[0]->p)[i] = (uint8_t) (((const uint8_t *) in[0]->p)[i] == 0);
        break;
      }
      case PT_SHAPE: {
        int64_t d[1] = {in[0]->ndim};
        tv_alloc(c, out[0], DT_I64, d, 1);
        for (int i = 0; i < in[0]->ndim; i++) ((int64_t *) out[0]->p)[i] = in[0]->d[i];
        break;
      }
      case PT_RESHAPE: r = op_reshape(c, in[0], in[1], out[0]); break;
      case PT_TRANSPOSE: {
        int perm[PT_V21_MAX_RANK];
        for (int i = 0; i < nd->pn; i++) perm[i] = (int) pt_ipool[nd->pl + i];
        r = op_transpose(c, in[0], perm, out[0]);
        break;
      }
      case PT_CONCAT: r = op_concat(c, nd->i0, in, nin(nd), out[0]); break;
      case PT_SPLIT: {
        int64_t sz[PT_V21_MAX_OUT];
        int ns = nd->n_out;
        if (nd->pn == ns) {
          for (int i = 0; i < ns; i++) sz[i] = pt_ipool[nd->pl + i];
        } else {
          int64_t per = in[0]->d[nd->i0 < 0 ? in[0]->ndim + nd->i0 : nd->i0] / ns;
          for (int i = 0; i < ns; i++) sz[i] = per;
        }
        r = op_split(c, nd->i0, in[0], sz, ns, out);
        break;
      }
      case PT_UNSQUEEZE: r = op_unsqueeze(c, in[0], tvals(in[1]), (int) in[1]->n, out[0]); break;
      case PT_SQUEEZE: r = op_squeeze(c, in[0], in[1] ? tvals(in[1]) : 0, in[1] ? (int) in[1]->n : 0, out[0]); break;
      case PT_GATHER: {
        if (nd->w0 >= 0) {
          int ci = nd->w0;
          r = op_gather(c, nd->i0, c->cont_f32[ci], &c->cont_dims[ci * PT_V21_MAX_RANK],
                        c->cont_ndim[ci], c->cont_n[ci], DT_F32, in[1], out[0]);
        } else {
          r = op_gather(c, nd->i0, in[0]->p, in[0]->d, in[0]->ndim, in[0]->n, in[0]->dt, in[1],
                        out[0]);
        }
        break;
      }
      case PT_SLICE: {
        /* opset-10 Slice input order is (data, starts, ends, axes, steps) */
        int na = in[3] ? (int) in[3]->n : (int) in[0]->ndim;
        if (na > PT_V21_MAX_RANK) na = PT_V21_MAX_RANK;
        int64_t axes[PT_V21_MAX_RANK];
        for (int i = 0; i < na; i++) axes[i] = in[3] ? tvals(in[3])[i] : i;
        int64_t step = in[4] ? (int64_t) tv_scalar_f(in[4]) : 1;
        r = op_slice(c, in[0], tvals(in[1]), tvals(in[2]), axes, na, step, out[0]);
        break;
      }
      case PT_PAD: {
        int64_t pads[PT_V21_MAX_RANK * 2];
        const int64_t *src = nd->w0 >= 0 ? tvals(in[1]) : tvals(in[1]);
        for (int i = 0; i < PT_V21_MAX_RANK * 2; i++) pads[i] = i < in[1]->n ? src[i] : 0;
        double cv = in[2] ? tv_scalar_f(in[2]) : 0.0;
        if (nd->w0 >= 0) {
          int ci = nd->w0;
          pt_tv tab;
          tab.dt = DT_F32;
          tab.p = (void *) c->cont_f32[ci];
          tab.ndim = c->cont_ndim[ci];
          for (int z = 0; z < PT_V21_MAX_RANK; z++) tab.d[z] = c->cont_dims[ci * PT_V21_MAX_RANK + z];
          tab.n = c->cont_n[ci];
          r = op_pad(c, &tab, pads, cv, nd->i0, out[0]);
        } else {
          r = op_pad(c, in[0], pads, cv, nd->i0, out[0]);
        }
        break;
      }
      case PT_EXPAND: r = op_expand(c, in[0], tvals(in[1]), (int) in[1]->n, out[0]); break;
      case PT_SCATTERND: r = op_scatternd(c, in[0], in[1], in[2], out[0]); break;
      case PT_CUMSUM: {
        int ax = (int) tv_scalar_f(in[1]);
        r = op_cumsum(c, in[0], ax, nd->i0, nd->i1, out[0]);
        break;
      }
      case PT_REDUCESUM:
      case PT_REDUCEMAX:
      case PT_REDUCEMIN:
      case PT_REDUCEMEAN: {
        int na = in[1] ? (int) in[1]->n : 0;
        if (na > PT_V21_MAX_RANK) na = PT_V21_MAX_RANK;
        int64_t axes[PT_V21_MAX_RANK];
        for (int i = 0; i < na; i++) axes[i] = tvals(in[1])[i];
        if (na == 0 && nd->i1 == 0) {
          na = in[0]->ndim;
          for (int i = 0; i < na; i++) axes[i] = i;
        }
        r = op_reduce(c, nd->op, in[0], axes, na, nd->i0, out[0]);
        break;
      }
      case PT_SOFTMAX: r = op_softmax(c, in[0], nd->i0, out[0]); break;
      case PT_LAYERNORM:
        r = op_layernorm(c, in[0], nd->i0, nd->f0, c->cont_f32[nd->w0], c->cont_f32[nd->w1], out[0]);
        break;
      case PT_EINSUM: r = op_einsum(c, pt_op_eq(idx), in[0], in[1], out[0]); break;
      case PT_RANGE: r = op_range(c, in[0], in[1], in[2], out[0]); break;
      case PT_RANDNORM: {
        int64_t d[PT_V21_MAX_RANK];
        memcpy(d, in[0]->d, sizeof(d));
        r = tv_alloc(c, out[0], DT_F32, d, in[0]->ndim);   /* noise_scale=0: the sampler is 0 */
        break;
      }
      case PT_IDENTITY: {
        int64_t d[PT_V21_MAX_RANK];
        memcpy(d, in[0]->d, sizeof(d));
        tv_alloc(c, out[0], in[0]->dt, d, in[0]->ndim);
        memcpy(out[0]->p, in[0]->p, (size_t) (in[0]->n * dt_sz(in[0]->dt)));
        break;
      }
      case PT_NONZERO: r = op_nonzero(c, in[0], out[0]); break;
      case PT_WHERE: r = op_where(c, in[0], in[1], in[2], out[0]); break;
      case PT_CONSOFT: {
        /* fill source: float pool (fl) for fp32 fills, int pool (pl) for int64 fills */
        if (nd->fdt == DT_F32) {
          r = op_const_of_shape(c, in[0], pt_fpool + (nd->fl >= 0 ? nd->fl : 0), DT_F32, 0,
                                nd->fn >= 0 ? nd->fn : 1, out[0]);
        } else {
          const int64_t *fv = pt_ipool + (nd->pl >= 0 ? nd->pl : 0);
          r = op_const_of_shape(c, in[0], 0, nd->fdt, (int64_t *) fv, nd->fn >= 1 ? nd->fn : 1,
                                out[0]);
        }
        break;
      }
      case PT_CONV: {
        pt_tv *X = in[0];
        int K = nd->i0, stride = nd->i1, pad = nd->i2, dil = nd->i3;
        int cout = (int) nd->f0, cin = (int) nd->f1;
        int cout4 = (cout + 3) / 4 * 4;
        int64_t T = X->d[X->ndim - 1];
        int64_t N = X->ndim >= 3 ? X->d[0] : 1;   /* the iSTFT dec convs run batch 4 */
        int64_t Tout = (T + 2 * pad - (int64_t) (K - 1) * dil - 1) / stride + 1;
        int64_t d[3] = {N, cout, Tout};     /* the graph sees Cout rows; the packed kernel fills 4 */
        if (tv_alloc_nz(c, out[0], DT_F32, d, 3, 0) < 0) return -1;
        double t0 = pt_v21_now();
        /* The conv reads each input row from -pad to T+pad-1, so it consumes a *padded* row: the
         * rows are copied into a [Cin][T+2*pad] scratch, zero-filled in the pad margins.  The CUDA
         * conv kernel reads the same padded layout. */
        int64_t ldxp = T + 2 * (int64_t) pad;
        /* Padded input row: the kernel reads row ci at X + ci*ldx + (k*dil + t), and the graph
         * column that output t needs for tap k is (t + k*dil - pad), so the row stores the input
         * at offset +pad and the kernel is handed the row start (no negative indexing). */
        size_t xpad_bytes =
            (size_t) ((((size_t) N * cin * ldxp) * 4 + 63) & ~(size_t) 63);
        /* The GEMM conv kernel applies the zero padding itself (out-of-row taps read as 0), so on
         * the CUDA path the unpadded tensor is handed over as is: no padded host copy per conv. */
        static int nopad = -1;
        if (nopad < 0) nopad = getenv("PT_V21_CONV_NOPAD") ? atoi(getenv("PT_V21_CONV_NOPAD")) : 1;
        int unpadded = nopad && c->use_cuda && pt_v21_conv_pads_in_kernel(c);
        float *xp;
        if (unpadded) {
          xp = (float *) X->p;
          ldxp = T;
          c->conv_pad_in = pad;
        } else {
          c->conv_pad_in = -1;
          xp = (float *) aligned_alloc(64, xpad_bytes);
          memset(xp, 0, xpad_bytes);
          for (int64_t b = 0; b < N; b++)
            for (int ci = 0; ci < cin; ci++)
              memcpy(xp + ((int64_t) b * cin + ci) * ldxp + pad,
                     (const float *) X->p + ((int64_t) b * cin + ci) * T, (size_t) (T * 4));
        }
        int64_t ybytes = (size_t) ((((size_t) N * cout4 * Tout) * 4 + 63) & ~(size_t) 63);
        float *yp = cout4 == cout ? (float *) out[0]->p
                                  : (float *) aligned_alloc(64, ybytes);   /* Cout=1 layers pad to 4 */
        if (getenv("PT_V21_CONVLOG"))
          fprintf(stderr, "conv node %d %s T=%lld N=%lld cin=%d cout=%d K=%d s=%d pad=%d dil=%d"
                          " Tout=%lld ldxp=%lld\n", idx, pt_tdefs[nd->out[0]].origin, (long long) T,
                  (long long) N, cin, cout, K, stride, pad, dil, (long long) Tout,
                  (long long) ldxp);
        /* Fuse the elementwise nodes that immediately follow this conv into its epilogue.  A node
         * is fusable when its output feeds nothing else and it is LeakyReLU, ReLU, or a scalar
         * Mul/Add, and the chain runs in graph order.  The arithmetic is the same fp32 arithmetic
         * the interpreter would have done, so results are unchanged; what goes away is a host pass
         * over the activation tensor (LEAKYRELU was 144 ms per stream, MUL 60, ADD 106). */
        int epi[2] = {0, 0};
        float epa[2] = {0.0f, 0.0f};
        int fused_nodes[3] = {0, 0, 0};
        int n_fused = 0;
        if (c->fuse && c->skip && cout4 == cout) {
          int t = nd->out[0];
          for (int step = 0; step < 2; step++) {
            if (t < 0 || g_n_cons[t] != 1)
              break;
            int cn = g_cons_node[t];
            if (cn < 0 || c->skip[cn])
              break;
            const pt_node_def *cd = &pt_ndefs[cn];
            int op = 0;
            float sc = 0.0f;
            if (cd->op == PT_LEAKYRELU && cd->in[0] == t) {
              op = 1;
              sc = cd->f0;   /* the node's own alpha, the same value un_op() receives */
            }
            else if (cd->op == PT_RELU && cd->in[0] == t)
              op = 2;
            else if (cd->op == PT_MUL || cd->op == PT_ADD) {
              int other = cd->in[0] == t ? cd->in[1] : (cd->in[1] == t ? cd->in[0] : -1);
              if (other < 0)
                break;
              pt_tv *ov = &c->V[other];
              if (!ov->alive || ov->n != 1 || ov->dt != DT_F32)
                break;   /* only scalar constants, whose value is already materialized */
              op = cd->op == PT_MUL ? 3 : 4;
              sc = *((const float *) ov->p);
            } else {
              break;
            }
            epi[step] = op;
            epa[step] = sc;
            fused_nodes[n_fused++] = cn;
            t = cd->out[0];
          }
        }
        /* Device-resident conv chain: when the (post-epilogue) result's only consumer is another
         * conv, leave it in the device arena; the next conv reads it there. Needs the unpadded GEMM
         * path (in-kernel padding) and unpadded channel rows (cout % 4 == 0). */
        int fin_t = nd->out[0];
        for (int i = 0; i < n_fused; i++)
          fin_t = pt_ndefs[fused_nodes[i]].out[0];
        static int chain_on = -1;
        if (chain_on < 0) chain_on = getenv("PT_V21_CHAIN") ? atoi(getenv("PT_V21_CHAIN")) : 1;
        int keep = 0;
        if (chain_on && unpadded && cout4 == cout && g_n_cons[fin_t] == 1) {
          int cn = g_cons_node[fin_t];
          keep = cn >= 0 && !c->skip[cn] && pt_ndefs[cn].op == PT_CONV && pt_ndefs[cn].in[0] == fin_t;
          /* A device-Add consumer also qualifies, but ONLY when the link will really run on the
           * device: the same checks dev_add_link makes, so a kept (host-stale) buffer is never read
           * by the host path. */
          if (getenv("PT_V21_DEVDBG") && cn >= 0 && (pt_ndefs[cn].op == PT_ADD || pt_ndefs[cn].op == PT_LEAKYRELU))
            fprintf(stderr, "CS node %d fin_t=%d keep=%d unpadded=%d cout4=%d cout=%d cons(fin)=%d cn=%d/op%d chain=%d\n",
                    idx, fin_t, keep, unpadded, cout4, cout, g_n_cons[fin_t], cn, (int) pt_ndefs[cn].op,
                    dev_resident(c, fin_t, 0));
          if (!keep && devadd_on() && c->use_cuda && cn >= 0 && !c->skip[cn]
              && ((pt_ndefs[cn].op == PT_ADD && (pt_ndefs[cn].in[0] == fin_t || pt_ndefs[cn].in[1] == fin_t))
                  || (pt_ndefs[cn].op == PT_LEAKYRELU && pt_ndefs[cn].in[0] == fin_t))
              && dev_resident(c, fin_t, 0))
            keep = 1;
        }
        c->conv_in_dev = (unpadded && X->dev_res) ? (const float *) X->dev_p : NULL;
        c->conv_keep = keep;
        /* A tensor consumed by the residual Add tree outlives a single chain, so it must come from
         * the residency sub-arena (released at its last use), not the bump arena. */
        c->conv_keep_res = keep && devadd_on() && g_n_cons[fin_t] > 1;
        pt_v21_conv(c, xp, ldxp, cin, yp, Tout, Tout /*graph shape*/, cout, K, stride, pad, dil,
                    c->cont_f32[nd->w0], c->cont_f32[nd->w1], N, nd->w0, nd->w1, epi[0], epi[1],
                    epa[0], epa[1]);
        c->conv_in_dev = NULL;
        c->conv_keep = 0;
        if (n_fused) {
          /* The conv's buffer becomes the last fused node's tensor: the conv's own tensor id stops
           * owning it (dev 2 is never freed) so the buffer is freed exactly once, at the last use of
           * the final tensor, which is what the un-fused graph would have done. */
          int last_out = nd->out[0];
          for (int i = 0; i < n_fused; i++) {
            int cn = fused_nodes[i];
            int ao = pt_ndefs[cn].out[0];
            c->skip[cn] = 1;
            c->V[ao] = c->V[last_out];
            c->V[ao].dev = (i == n_fused - 1) ? 1 : 2;
            c->V[ao].alive = 1;
            last_out = ao;
            c->V[nd->out[0]].dev = 2;
          }
        }
        c->V[fin_t].dev_res = 0;
        c->V[fin_t].dev_p = NULL;
        if (c->conv_out_dev) {
          /* the host buffer of fin_t is left stale on purpose: its only reader is the next conv,
           * which takes the device copy */
          c->V[fin_t].dev_res = 1;
          c->V[fin_t].dev_p = c->conv_out_dev;
          if (getenv("PT_V21_CHAINLOG")) fprintf(stderr, "chain: node %d keeps its output\n", ni);
          c->conv_out_dev = NULL;
        }
        if (cout4 != cout) {
          for (int64_t b = 0; b < N; b++)
            memcpy((float *) out[0]->p + b * cout * Tout, yp + b * cout4 * Tout,
                   (size_t) (cout * Tout * 4));
          free(yp);
        }
        if (!unpadded)
          free(xp);
        c->t_conv += pt_v21_now() - t0;
        break;
      }
      case PT_CONVT: {
        pt_tv *X = in[0];
        int K = nd->i0, s = nd->i1, pad = nd->i2;
        int cout = (int) nd->f0, cin = (int) nd->f1;
        int cout4 = (cout + 3) / 4 * 4;
        int64_t Tin = X->d[X->ndim - 1];
        int64_t N = X->ndim >= 3 ? X->d[0] : 1;
        int64_t Lout = (Tin - 1) * s - 2 * pad + K;
        int64_t d[3] = {N, cout, Lout};
        if (tv_alloc_nz(c, out[0], DT_F32, d, 3, 0) < 0) return -1;
        if (getenv("PT_V21_CONVLOG"))
          fprintf(stderr, "convT node %d T=%lld N=%lld cin=%d cout=%d K=%d s=%d pad=%d Lout=%lld\n",
                  idx, (long long) Tin, (long long) N, cin, cout, K, s, pad, (long long) Lout);
        double t0 = pt_v21_now();
        if (cout4 == cout) {
          pt_v21_convT(c, (const float *) X->p, Tin, cin, Tin, (float *) out[0]->p, Lout, cout, s,
                       pad, K, c->cont_f32[nd->w0], c->cont_f32[nd->w1], N, nd->w0, nd->w1);
        } else {
          size_t sb = (size_t) ((((size_t) N * cout4 * Lout) * 4 + 63) & ~(size_t) 63);
          float *sc = (float *) aligned_alloc(64, sb);
          pt_v21_convT(c, (const float *) X->p, Tin, cin, Tin, sc, Lout, cout, s, pad, K,
                       c->cont_f32[nd->w0], c->cont_f32[nd->w1], N, nd->w0, nd->w1);
          for (int64_t b = 0; b < N; b++)
            memcpy((float *) out[0]->p + b * cout * Lout, sc + b * cout4 * Lout,
                   (size_t) (cout * Lout * 4));
          free(sc);
        }
        c->t_convT += pt_v21_now() - t0;
        break;
      }
      case PT_MATMUL: {
        pt_tv *A = in[0], *B = in[1];
        if (nd->w0 >= 0) {
          int ci = nd->w0;
          pt_tv tb;
          tb.dt = DT_F32;
          tb.p = (void *) c->cont_f32[ci];
          tb.ndim = c->cont_ndim[ci];
          for (int z = 0; z < PT_V21_MAX_RANK; z++) tb.d[z] = c->cont_dims[ci * PT_V21_MAX_RANK + z];
          tb.n = c->cont_n[ci];
          B = &tb;
        }
        int ar = A->ndim, br = B->ndim;
        int64_t M = ar >= 2 ? A->d[ar - 2] : 1;
        int64_t Kk = ar >= 2 ? A->d[ar - 1] : A->n;
        int64_t N = br >= 2 ? B->d[br - 1] : B->n;
        /* broadcast the batch dims */
        int br2 = ar - 2 > 0 ? ar - 2 : 0, br3 = br - 2 > 0 ? br - 2 : 0;
        int nbatch = br2 > br3 ? br2 : br3;
        int64_t bd[PT_V21_MAX_RANK] = {1, 1, 1, 1, 1, 1};
        int64_t nb = 1;
        for (int i = 0; i < nbatch; i++) {
          int64_t da = i - nbatch + br2 >= 0 && i - nbatch + br2 < br2 ? A->d[i - nbatch + br2] : 1;
          int64_t db = i - nbatch + br3 >= 0 && i - nbatch + br3 < br3 ? B->d[i - nbatch + br3] : 1;
          bd[i] = da > db ? da : db;
          nb *= bd[i];
        }
        int64_t d[PT_V21_MAX_RANK];
        for (int i = 0; i < nbatch; i++) d[i] = bd[i];
        d[nbatch] = M;
        d[nbatch + 1] = N;
        tv_alloc(c, out[0], DT_F32, d, nbatch + 2);
        double t0 = pt_v21_now();
        if (g_trace == idx || g_trace == -2) {
          fprintf(stderr, "matmul node %d %s: A dt=%d ndim=%d [", idx, pt_graph_tnames[nd->out[0]],
                  A->dt, A->ndim);
          for (int q = 0; q < A->ndim; q++) fprintf(stderr, " %lld", (long long) A->d[q]);
          fprintf(stderr, "] B src=%s ndim=%d [", nd->w0 >= 0 ? pt_graph_ctab[nd->w0] : pt_graph_tnames[nd->in[1]],
                  B->ndim);
          for (int q = 0; q < B->ndim; q++) fprintf(stderr, " %lld", (long long) B->d[q]);
          fprintf(stderr, "] M=%lld K=%lld N=%lld nb=%lld nbatch=%d\n", (long long) M, (long long) Kk,
                  (long long) N, (long long) nb, nbatch);
        }
        int64_t asd[PT_V21_MAX_RANK], bsd[PT_V21_MAX_RANK];
        asd[ar - 1] = 1;
        for (int i = ar - 2; i >= 0; i--) asd[i] = asd[i + 1] * A->d[i + 1];
        bsd[br - 1] = 1;
        for (int i = br - 2; i >= 0; i--) bsd[i] = bsd[i + 1] * B->d[i + 1];
        for (int64_t bi = 0; bi < nb; bi++) {
          int64_t r = bi, ao = 0, bo = 0, ds = 1;
          for (int q = nbatch - 1; q >= 0; q--) {
            int64_t ix = r % bd[q];
            r /= bd[q];
            int64_t qa = q - nbatch + br2, qb = q - nbatch + br3;
            if (qa >= 0 && qa < br2) ao += (A->d[qa] == 1 ? 0 : ix) * asd[qa];
            if (qb >= 0 && qb < br3) bo += (B->d[qb] == 1 ? 0 : ix) * bsd[qb];
            ds *= bd[q];
          }
          pt_v21_matmul(c, (const float *) A->p + ao, M, Kk, Kk, (const float *) B->p + bo, N, N,
                        (float *) out[0]->p + bi * M * N, N);
        }
        c->t_matmul += pt_v21_now() - t0;
        break;
      }
      default:
        fprintf(stderr, "pt_v21: node %d op %d (%s) unhandled\n", idx, nd->op,
                pt_graph_tnames[nd->out[0] >= 0 ? nd->out[0] : 0]);
        return -1;
    }
    if (c->prof) {
      c->t_ekind[nd->op] += pt_v21_now() - tA;
      c->n_ekind[nd->op]++;
      if (c->prof_node_on) {
        if (!c->prof_node) c->prof_node = (double *) calloc(PT_N_NODE, sizeof(double));
        if (c->prof_node) c->prof_node[idx] += pt_v21_now() - tA;
      }
    }
    if (r) {
      fprintf(stderr, "pt_v21: node %d op %d failed (%s)\n", idx, nd->op,
              pt_graph_tnames[nd->out[0] >= 0 ? nd->out[0] : 0]);
      return -1;
    }
    for (int i = 0; i < PT_V21_MAX_OUT; i++) {
      if (nd->out[i] >= 0) ck_save(c, nd->out[i]);
    }
    if (g_dump_tensor >= 0 && idx == g_dump_tensor) {
      /* raw fp32 dump of one node's output, for A/B-ing kernel variants against the reference path */
      pt_tv *t = &c->V[nd->out[0]];
      char path[128];
      snprintf(path, sizeof(path), "/tmp/pt_dump_%d.bin", idx);
      FILE *fp = fopen(path, "wb");
      if (fp) {
        fwrite(t->p, 4, (size_t) t->n, fp);
        fclose(fp);
      }
    }
    if (g_shapes) {
      for (int i = 0; i < PT_V21_MAX_OUT; i++) {
        if (nd->out[i] < 0) continue;
        pt_tv *t = &c->V[nd->out[i]];
        fprintf(g_shapes, "%d %s", nd->out[i], pt_graph_tnames[nd->out[i]]);
        for (int q = 0; q < t->ndim; q++) fprintf(g_shapes, " %lld", (long long) t->d[q]);
        fprintf(g_shapes, "\n");
      }
    }
    /* free inputs whose last consumer is this node */
    for (int i = 0; i < PT_V21_MAX_IN; i++) {
      int t = nd->in[i];
      if (t >= 0 && c->V[t].dev != 2 && pt_graph_last_use[t] == idx) tv_free(c, &c->V[t]);
    }
  }
  if (idx_to < PT_N_NODE) {   /* partial run (streaming encoder pass): no waveform expected */
    if (wav) *wav = NULL;
    if (wav_n) *wav_n = 0;
    c->t_total = pt_v21_now() - t0;
    return 0;
  }
  pt_tv *w = &c->V[pt_graph_wav_tensor];
  if (!w->p) {
    fprintf(stderr, "pt_v21: graph output 'wav' has no payload\n");
    return -1;
  }
  free(c->wav);
  c->wav_n = w->n;
  c->wav = (float *) malloc((size_t) (w->n * 4));
  memcpy(c->wav, w->p, (size_t) (w->n * 4));
  *wav = c->wav;
  *wav_n = c->wav_n;
  c->t_total = pt_v21_now() - t0;
  return 0;
}

/* ---------------- tiny .npy I/O (v1.8 conventions, big-endian header per numpy spec) ---------- */
/* header length on the wire: strlen(dict) + 1 for the newline, rounded so 10+len % 64 == 0 */
static size_t npy_header_len(size_t dict_len) {
  size_t len = dict_len + 1;
  while ((10 + len) % 64) len++;
  return len;
}

static void npy_write_header(FILE *f, const char *hdr, size_t hdrsz) {
  fwrite("\x93" "NUMPY", 1, 6, f);
  fputc(1, f);
  fputc(0, f);
  fputc((int) (hdrsz & 0xff), f);
  fputc((int) ((hdrsz >> 8) & 0xff), f);
  size_t l = strlen(hdr);
  fwrite(hdr, 1, l, f);
  for (size_t i = l; i + 1 < hdrsz; i++) fputc(' ', f);
  fputc('\n', f);
}

int pt_npy_save_f32_v21(const char *path, const float *d, int ndim, const int64_t *dim) {
  char hdr[1024];
  int n = snprintf(hdr, sizeof(hdr), "{'descr': '<f4', 'fortran_order': False, 'shape': (");
  for (int i = 0; i < ndim; i++)
    n += snprintf(hdr + n, sizeof(hdr) - n, "%lld%s", (long long) dim[i], ndim == 1 ? "," : ",");
  n += snprintf(hdr + n, sizeof(hdr) - n, "), }");
  size_t body = 1;
  for (int i = 0; i < ndim; i++) body *= (size_t) dim[i];
  body *= 4;
  /* numpy v1: data starts at 10 + header_len, and the header includes the terminating newline;
   * pad so that 10 + header_len is a multiple of 64. */
  size_t hdrsz = npy_header_len(strlen(hdr));
  FILE *f = fopen(path, "wb");
  if (!f) return -1;
  npy_write_header(f, hdr, hdrsz);
  fwrite(d, 4, body / 4, f);
  fclose(f);
  return 0;
}

int pt_npy_save_generic_v21(const char *path, const void *d, int ndim, const int64_t *dim,
                            const char *descr) {
  char hdr[1024];
  int n = snprintf(hdr, sizeof(hdr), "{'descr': '%s', 'fortran_order': False, 'shape': (", descr);
  for (int i = 0; i < ndim; i++)
    n += snprintf(hdr + n, sizeof(hdr) - n, "%lld%s", (long long) dim[i], ndim == 1 ? "," : ",");
  n += snprintf(hdr + n, sizeof(hdr) - n, "), }");
  size_t cnt = 1;
  for (int i = 0; i < ndim; i++) cnt *= (size_t) dim[i];
  size_t esz = strstr(descr, "f8") ? 8 : (descr[1] == 'i' && descr[2] == '8') ? 8
                                : descr[1] == 'b' ? 1 : 4;
  size_t hdrsz = npy_header_len(strlen(hdr));
  FILE *f = fopen(path, "wb");
  if (!f) return -1;
  npy_write_header(f, hdr, hdrsz);
  fwrite(d, esz, cnt, f);
  fclose(f);
  return 0;
}

int pt_npy_save_i64_v21(const char *path, const int64_t *d, int ndim, const int64_t *dim) {
  return pt_npy_save_generic_v21(path, d, ndim, dim, "<i8");
}
int pt_npy_save_b1_v21(const char *path, const uint8_t *d, int ndim, const int64_t *dim) {
  return pt_npy_save_generic_v21(path, d, ndim, dim, "|b1");
}

int pt_npy_load_v21(const char *path, void **data, int *dtype, int *ndim, int64_t *dim, int64_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  char magic[6];
  if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6)) {
    fclose(f);
    return -2;
  }
  int major = fgetc(f), minor = fgetc(f);
  (void) minor;
  int hdrsz;
  if (major == 1) {
    int c1 = fgetc(f), c2 = fgetc(f);
    hdrsz = c1 | (c2 << 8);
  } else {
    int c1 = fgetc(f), c2 = fgetc(f), c3 = fgetc(f), c4 = fgetc(f);
    hdrsz = c1 | (c2 << 8) | (c3 << 16) | (c4 << 24);
  }
  char *hdr = (char *) malloc((size_t) hdrsz + 1);
  if (fread(hdr, 1, (size_t) hdrsz, f) != (size_t) hdrsz) {
    free(hdr);
    fclose(f);
    return -3;
  }
  hdr[hdrsz] = 0;
  int dt = DT_F32, esz = 4;
  if (strstr(hdr, "'<f4'")) {
    dt = DT_F32;
    esz = 4;
  } else if (strstr(hdr, "'<f8'")) {
    dt = DT_F64;
    esz = 8;
  } else if (strstr(hdr, "'<i8'")) {
    dt = DT_I64;
    esz = 8;
  } else if (strstr(hdr, "'<i4'")) {
    dt = DT_I32;
    esz = 4;
  } else if (strstr(hdr, "'|b1'") || strstr(hdr, "'<b1'")) {
    dt = DT_BOOL;
    esz = 1;
  } else {
    free(hdr);
    fclose(f);
    return -4;
  }
  const char *sh = strstr(hdr, "'shape': (");
  int nd = 0;
  int64_t d[PT_V21_MAX_RANK] = {1, 1, 1, 1, 1, 1};
  if (sh) {
    sh += 9;
    while (*sh && *sh != ')') {
      if (*sh >= '0' && *sh <= '9') {
        long long v = strtoll(sh, (char **) &sh, 10);
        if (nd < PT_V21_MAX_RANK) d[nd++] = v;
      } else {
        sh++;
      }
    }
  }
  int64_t cnt = 1;
  for (int i = 0; i < nd; i++) cnt *= d[i];
  void *p = malloc((size_t) (cnt * esz + 16));
  if (fread(p, esz, (size_t) cnt, f) != (size_t) cnt) {
    free(p);
    free(hdr);
    fclose(f);
    return -5;
  }
  free(hdr);
  fclose(f);
  *data = p;
  *dtype = dt;
  *ndim = nd;
  for (int i = 0; i < PT_V21_MAX_RANK; i++) dim[i] = i < nd ? d[i] : 1;
  *n = cnt;
  return 0;
}
