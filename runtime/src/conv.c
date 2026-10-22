/* conv.c - conv1d / linear driver on top of the 4x16 micro-kernel, with fused epilogues
 * (bias in-kernel; activation / scale / residual applied per 4-row x 128-col strip while it is
 * L1-resident), plus a small fork-join pool for intra-op row/column splitting.
 */
#define _GNU_SOURCE
#include "pt.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STRIP 144 /* columns per strip: 9 tiles of 16 or 12 tiles of 12 */
#define MAXNK 4096

int pt_conv_bind(pt_conv_w *c, const pt_weights *w, const char *wp_name, const char *b_name, int cout, int dil) {
  const pt_tensor *t = pt_find(w, wp_name);
  if (!t || t->ndim != 4 || t->dim[3] != 4) { fprintf(stderr, "pt_conv_bind: missing/bad %s\n", wp_name); return -1; }
  c->wp = pt_f32(w, wp_name);
  c->b = b_name ? pt_f32(w, b_name) : 0;
  if (b_name && !c->b) { fprintf(stderr, "pt_conv_bind: missing %s\n", b_name); return -1; }
  c->cout4 = (int)t->dim[0] * 4;
  c->K = (int)t->dim[1];
  c->cin = (int)t->dim[2];
  c->cout = cout > 0 ? cout : c->cout4;   /* 0 = every packed row is a real output channel */
  c->dil = dil;
  if (c->cout > c->cout4 || c->cout <= c->cout4 - 4) { fprintf(stderr, "pt_conv_bind: %s cout %d vs %d\n", wp_name, c->cout, c->cout4); return -1; }
  if ((long)c->K * c->cin > MAXNK) { fprintf(stderr, "pt_conv_bind: %s nk too big\n", wp_name); return -1; }
  return 0;
}

/* ------------------------------------------------------------------ pool */
typedef void (*pt_job_fn)(void *ctx, long item);
static struct {
  pthread_mutex_t mu;
  pthread_cond_t cv, done_cv;
  pthread_t th[16];
  int n;          /* worker threads (excluding caller) */
  int busy;       /* a job is running */
  long gen;
  pt_job_fn fn;
  void *ctx;
  long nitems, next, active;
  int stop;
} P = { PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER };
static __thread int tl_threads = 1;

static void *pool_main(void *arg) {
  (void)arg;
  long seen = 0;
  pthread_mutex_lock(&P.mu);
  for (;;) {
    while (!P.stop && P.gen == seen) pthread_cond_wait(&P.cv, &P.mu);
    if (P.stop) break;
    seen = P.gen;
    P.active++;
    while (P.next < P.nitems) {
      long it = P.next++;
      pt_job_fn fn = P.fn;
      void *ctx = P.ctx;
      pthread_mutex_unlock(&P.mu);
      fn(ctx, it);
      pthread_mutex_lock(&P.mu);
    }
    if (--P.active == 0) pthread_cond_broadcast(&P.done_cv);
  }
  pthread_mutex_unlock(&P.mu);
  return 0;
}

void pt_pool_init(int nthreads) {
  if (P.n) return;
  if (nthreads > 16) nthreads = 16;
  P.n = nthreads > 1 ? nthreads - 1 : 0;
  for (int i = 0; i < P.n; i++) pthread_create(&P.th[i], 0, pool_main, 0);
}
void pt_pool_shutdown(void) {
  pthread_mutex_lock(&P.mu);
  P.stop = 1;
  pthread_cond_broadcast(&P.cv);
  pthread_mutex_unlock(&P.mu);
  for (int i = 0; i < P.n; i++) pthread_join(P.th[i], 0);
  P.n = 0;
  P.stop = 0;
}
int pt_pool_threads(void) { return P.n + 1; }
void pt_pool_set_local(int n) { tl_threads = n < 1 ? 1 : n; }
int pt_pool_get_local(void) { return tl_threads; }

/* Run fn(ctx, 0..nitems-1). Uses the pool only if this thread asked for >1 threads and the pool is free;
 * otherwise runs inline (so concurrent single-threaded streams never contend). */
static void parallel_for(long nitems, pt_job_fn fn, void *ctx) {
  if (tl_threads <= 1 || P.n == 0 || nitems <= 1) {
    for (long i = 0; i < nitems; i++) fn(ctx, i);
    return;
  }
  pthread_mutex_lock(&P.mu);
  if (P.busy) { /* someone else owns the pool: do it ourselves */
    pthread_mutex_unlock(&P.mu);
    for (long i = 0; i < nitems; i++) fn(ctx, i);
    return;
  }
  P.busy = 1;
  P.fn = fn; P.ctx = ctx; P.nitems = nitems; P.next = 0;
  P.gen++;
  pthread_cond_broadcast(&P.cv);
  while (P.next < P.nitems) {
    long it = P.next++;
    pthread_mutex_unlock(&P.mu);
    fn(ctx, it);
    pthread_mutex_lock(&P.mu);
  }
  while (P.active > 0) pthread_cond_wait(&P.done_cv, &P.mu);
  P.busy = 0;
  pthread_mutex_unlock(&P.mu);
}

/* ------------------------------------------------------------------ conv */
typedef struct {
  const pt_conv_w *c;
  const float *X;
  float *Y;
  long ldx, ldy, n, nk;
  const pt_epi *e;
  long off[MAXNK];
  long nstrips, ngroups, blk_per_group;
} conv_job;

static void epilogue_rows(const pt_epi *e, int co0, int cout, float *src, long lds, float *dst, long ldd, long ncols) {
  for (int i = 0; i < 4 && co0 + i < cout; i++) { /* zero-padded rows (co >= cout) are never read: skip */
    float *sr = src + i * lds, *d = dst + i * ldd;
    switch (e->act) { /* activation in place on the (L1-resident) strip */
      case PT_ACT_SILU: pt_silu_row(sr, ncols); break;
      case PT_ACT_TANH: pt_tanh_row(sr, ncols); break;
      case PT_ACT_SNAKE: pt_snake_row(sr, ncols, e->a[co0 + i], e->ia[co0 + i]); break;
      default: break;
    }
    if (e->scale != 1.0f && e->scale != 0.0f) {
      float sc = e->scale;
      for (long t = 0; t < ncols; t++) sr[t] *= sc;
    }
    if (e->res) { /* res may alias dst (in-place residual): read r[t] before writing d[t] */
      const float *r = e->res + (long)(co0 + i) * e->ldres;
      for (long t = 0; t < ncols; t++) d[t] = r[t] + sr[t];
    } else if (d != sr) {
      memcpy(d, sr, ncols * sizeof(float));
    }
  }
}

static int g_kernel8 =
#if defined(__aarch64__)
    1;
#else
    1;
#endif
void pt_conv_set_kernel(int use8x12) { g_kernel8 = use8x12; }

/* B-panel packing for wide-K linears (K == 1, Cin >= 128): rows of a 168..504-channel activation are ~4 KB
 * apart, so every 12-column tile touches Cin pages and thrashes the 32-entry L1 DTLB. Packing the strip
 * tile-major ([tile][j][12], contiguous) turns that into a sequential stream, amortised over all row blocks. */
static int g_pack = 2;
void pt_conv_set_pack(int on) { g_pack = on; }
static __thread float *tl_pack;
static __thread long tl_pack_cap;
static __thread long tl_off12[MAXNK];
static __thread long tl_off12_n;

static void conv_item_packed(conv_job *J, long t0, long ncols, long b0, long b1) {
  const pt_conv_w *c = J->c;
  long nk = J->nk, nt = (ncols + 11) / 12;
  long need = nt * nk * 12;
  if (need > tl_pack_cap) {
    free(tl_pack);
    tl_pack = aligned_alloc(64, (need * 4 + 63) & ~63L);
    tl_pack_cap = need;
  }
  if (tl_off12_n != nk) {
    for (long j = 0; j < nk; j++) tl_off12[j] = j * 48;
    tl_off12_n = nk;
  }
  for (long t = 0; t < nt; t++) {
    float *P = tl_pack + t * nk * 12;
    const float *x = J->X + t0 + t * 12;
    for (long j = 0; j < nk; j++) memcpy(P + j * 12, x + J->off[j] / 4, 48);
  }
  const pt_epi *e = J->e;
  int direct = !e || !e->res;
  float tmp[8 * STRIP] __attribute__((aligned(64)));
  long cw = (ncols + 3) & ~3L;
  for (long b = b0; b < b1; b += 2) {
    const float *A = c->wp + b * nk * 4;
    const float *bias = c->b ? c->b + b * 4 : 0;
    float *Yb = J->Y + b * 4 * J->ldy + t0;
    float *dst = direct ? Yb : tmp;
    long ldd = direct ? J->ldy : STRIP;
    for (long t = 0; t < nt; t++)
      pt_k8x12(dst + t * 12, ldd * 4, A, tl_pack + t * nk * 12, tl_off12, nk, 1, bias);
    for (int h = 0; h < 2; h++) {
      if (!e) continue;
      if (direct) epilogue_rows(e, (int)((b + h) * 4), c->cout, Yb + h * 4 * J->ldy, J->ldy, Yb + h * 4 * J->ldy, J->ldy, cw);
      else {
        pt_epi e2 = *e;
        e2.res = e->res + t0;
        epilogue_rows(&e2, (int)((b + h) * 4), c->cout, tmp + h * 4 * STRIP, STRIP, Yb + h * 4 * J->ldy, J->ldy, cw);
      }
    }
  }
}

static void conv_item(void *ctx, long item) {
  conv_job *J = (conv_job *)ctx;
  const pt_conv_w *c = J->c;
  long s = item % J->nstrips, g = item / J->nstrips;
  long t0 = s * STRIP;
  long ncols = J->n - t0 < STRIP ? J->n - t0 : STRIP;
  long cw = (ncols + 3) & ~3L;
  long b0 = g * J->blk_per_group, b1 = b0 + J->blk_per_group;
  if (b1 > c->cout4 / 4) b1 = c->cout4 / 4;
  if (g_pack && g_kernel8 && (c->K == 1 ? c->cin >= 128 : g_pack > 1) && (b1 - b0) >= 4) {
    long b1e = b0 + ((b1 - b0) & ~1L);
    conv_item_packed(J, t0, ncols, b0, b1e);
    b0 = b1e;   /* odd leftover block (if any) falls through to the unpacked path */
  }
  const pt_epi *e = J->e;
  int direct = !e || !e->res;
  float tmp[8 * STRIP] __attribute__((aligned(64)));
  for (long b = b0; b < b1;) {
    int rows8 = g_kernel8 && b + 1 < b1;
    const float *A = c->wp + b * J->nk * 4;
    const float *bias = c->b ? c->b + b * 4 : 0;
    float *Yb = J->Y + b * 4 * J->ldy + t0;
    float *dst = direct ? Yb : tmp;
    long ldd = direct ? J->ldy : STRIP;
    if (rows8) pt_k8x12(dst, ldd * 4, A, J->X + t0, J->off, J->nk, (ncols + 11) / 12, bias);
    else pt_k4x16(dst, ldd * 4, A, J->X + t0, J->off, J->nk, (ncols + 15) / 16, bias);
    for (int h = 0; h < (rows8 ? 2 : 1); h++) {
      if (!e) continue;
      if (direct) epilogue_rows(e, (int)((b + h) * 4), c->cout, Yb + h * 4 * J->ldy, J->ldy, Yb + h * 4 * J->ldy, J->ldy, cw);
      else {
        pt_epi e2 = *e;
        e2.res = e->res + t0;
        epilogue_rows(&e2, (int)((b + h) * 4), c->cout, tmp + h * 4 * STRIP, STRIP, Yb + h * 4 * J->ldy, J->ldy, cw);
      }
    }
    b += rows8 ? 2 : 1;
  }
}

void pt_conv(const pt_conv_w *c, const float *X, long ldx, float *Y, long ldy, long n, const pt_epi *e) {
  if (n <= 0) return;
  static __thread conv_job J; /* off[] table is large: keep it off the stack */
  J.c = c; J.X = X; J.Y = Y; J.ldx = ldx; J.ldy = ldy; J.n = n; J.e = e;
  J.nk = (long)c->K * c->cin;
  for (long k = 0, j = 0; k < c->K; k++)
    for (long ci = 0; ci < c->cin; ci++, j++) J.off[j] = (ci * ldx + k * c->dil) * 4;
  J.nstrips = (n + STRIP - 1) / STRIP;
  long nblk = c->cout4 / 4;
  long want = tl_threads > 1 ? tl_threads * 3 : 1;
  J.ngroups = 1;
  if (J.nstrips < want) {
    J.ngroups = (want + J.nstrips - 1) / J.nstrips;
    if (J.ngroups > nblk) J.ngroups = nblk;
  }
  J.blk_per_group = (nblk + J.ngroups - 1) / J.ngroups;
  if (J.blk_per_group > 1) J.blk_per_group = (J.blk_per_group + 1) & ~1L; /* keep 8-row pairs together */
  J.ngroups = (nblk + J.blk_per_group - 1) / J.blk_per_group;
  parallel_for(J.nstrips * J.ngroups, conv_item, &J);
}
