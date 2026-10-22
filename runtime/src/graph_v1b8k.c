/* graph_v1b8k.c - PrimeTTS v1b_8k forward pass (encoder -> host_regulate (C) -> decoder -> vocoder).
 *
 * Layout: every activation is channel-major [C][ld] (time contiguous), C padded to a multiple of 4.
 * Encoder/decoder buffers have PADL zero columns on the left and >= SLACK columns on the right.
 * The vocoder runs in chunks of mel frames: for a chunk of output samples [S0,S1) every layer computes
 * only the columns that the chunk needs ("valid" convolutions on absolute column windows), columns
 * outside the true signal [0,N) are forced to zero after every layer = ONNX zero padding semantics.
 * So chunked output is exactly the full-utterance output (no overlap-add, no crossfade).
 */
#define _GNU_SOURCE
#include "pt_v1b8k.h"
#include <math.h>
#include <stdarg.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define H 168
#define PADL 4
#define SLACK 64
#define LN_EPS 9.999999747378752e-06f
#define MELPAD 64     /* zero columns around the mel for the vocoder window */

static int ST_COUT[4], ST_STRIDE[4], ST_PAD[4], ST_KERNEL[4], ST_RATE[4]; /* from the .ptk "voc.geom" */
static int V_HOP;     /* samples per mel frame = product of the stage strides (128 at 8 kHz, 256 at 16 kHz) */
static int V_SLACK;   /* conv window overshoot: 16-column tile rounding + widest tap span in the graph */
static const int RB_K[3] = {3, 7, 11};
static const int RB_D[3] = {1, 3, 5};
#define RB_REACH 60   /* max over resblocks: sum_m (d_m + 1) * (K - 1) / 2 for K = 11 */

static inline long r4(long n) { return (n + 3) & ~3L; }
static inline long r16(long n) { return (n + 15) & ~15L; }
static inline long fdiv(long a, long b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

typedef struct { const float *g, *b; } ln_w;
typedef struct {
  ln_w ln1, ln2;
  const float *dw, *dwb;
  pt_conv_w point, ff0, ff3;
} blk_w;
typedef struct { ln_w ln; pt_conv_w l1, l3; } head_w;

struct pt_v1b8k_model {
  pt_weights w;
  int sample_rate, hop, abs_bins, max_frames;   /* from the container's model.meta */
  /* encoder */
  const float *emb_ph, *emb_tone, *emb_lang, *emb_spk;
  pt_conv_w spk_proj;
  blk_w eb[5];
  head_w hdur, hen, hbr, hpi;
  const float *eproj_w, *eproj_b, *bproj_w, *bproj_b;
  /* decoder */
  pt_conv_w fp0, fp2, lc0, lc2, ref0, ref2[4], ref4, pp0, pp2, gru_ih, mh1, mh3, pn0, pn2, pn4;
  const float *abs_emb, *gru_r, *gru_rb;
  ln_w mh;
  blk_w db[6];
  /* vocoder */
  pt_conv_w pre, ups[4], c1[12][3], c2[12][3], post;
  const float *up_a[4], *up_ia[4], *a1[12][3], *ia1[12][3], *a2[12][3], *ia2[12][3], *post_a, *post_ia;
};

/* vocoder window buffer: element (c, t) at base[c*ld + t - lo] */
typedef struct { float *base; long ld, lo, hi, cap; int C; } vwin;
/* per-tensor history: the last H columns ending at absolute column hi */
typedef struct { float *h; int C, len; long hi; } vhist;

/* encoder + decoder workspace: only needed until the mel is ready, so a server can pool these
 * (one per core) while every stream keeps just its mel + vocoder state. */
typedef struct {
  /* encoder [rows][ldT], column 0 at +PADL;  lct/lca/lcb alias ef/ed/ha (dead by then) */
  float *ex, *eh, *ed, *ep, *ef, *ha, *cond, *pit, *lct, *lca, *lcb, *logd;
  int64_t *dur;
  int32_t *tok, *ap;
  /* decoder [rows][ldF];  dg aliases R (free after the refiner), mp aliases df (free after the GRU) */
  float *R, *dx, *dh, *dd, *dp, *df, *dg, *rf, *mp;
} encdec_bufs;

struct pt_v1b8k_ws {
  pt_arena A;
  int maxT, maxF;
  long ldT, ldF;
  encdec_bufs b;
};

struct pt_v1b8k_stream {
  const pt_v1b8k_model *m;
  pt_arena A;
  int maxT, maxF, chunkF, firstF;
  long ldT, ldF;
  pt_v1b8k_ws *own_ws;                      /* private workspace (pt_v1b8k_stream_new) or NULL */
  pt_v1b8k_ws *(*ws_acquire)(void *);       /* shared workspace pool (server) */
  void (*ws_release)(void *, pt_v1b8k_ws *);
  void *ws_user;
  encdec_bufs w;                            /* bound for the duration of encoder+decoder */
  float *mel;
  float gh[2][84], grh[2][256];
  /* vocoder */
  vwin pre, ph[4], u[4], t1[4], t2[4], xa[4], xb[4], sum[4], wav;
  /* left-context history carried between vocoder chunks (see voc_chunk) */
  vhist hV[5], hU[4], hT1[4][3][3], hT2[4][3][3], hX[4][3][3];
  long vcols_comp, vcols_use;
  pt_tap_cb tap;
  void *tap_user;
};

/* ------------------------------------------------------------------ binding */
static const float *W(const pt_v1b8k_model *m, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static const float *W(const pt_v1b8k_model *m, const char *fmt, ...) {
  char nm[96];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(nm, sizeof nm, fmt, ap);
  va_end(ap);
  const float *p = pt_f32(&m->w, nm);
  if (!p) fprintf(stderr, "v1b8k: missing tensor %s\n", nm);
  return p;
}
static int bindc(pt_conv_w *c, const pt_v1b8k_model *m, int cout, int dil, const char *fmt, ...) {
  char pre[96], wp[110], b[110];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(pre, sizeof pre, fmt, ap);
  va_end(ap);
  snprintf(wp, sizeof wp, "%s.wp", pre);
  snprintf(b, sizeof b, "%s.b", pre);
  return pt_conv_bind(c, &m->w, wp, b, cout, dil);
}

static int bind_block(pt_v1b8k_model *m, blk_w *bw, const char *g, const char *blk, int i, int ff) {
  int e = 0;
  bw->ln1.g = W(m, "%s.%s.%d.norm1.g", g, blk, i);
  bw->ln1.b = W(m, "%s.%s.%d.norm1.beta", g, blk, i);
  bw->ln2.g = W(m, "%s.%s.%d.norm2.g", g, blk, i);
  bw->ln2.b = W(m, "%s.%s.%d.norm2.beta", g, blk, i);
  bw->dw = W(m, "%s.%s.%d.depth.dw", g, blk, i);
  bw->dwb = W(m, "%s.%s.%d.depth.b", g, blk, i);
  e |= bindc(&bw->point, m, H, 1, "%s.%s.%d.point", g, blk, i);
  e |= bindc(&bw->ff0, m, ff, 1, "%s.%s.%d.ff.0", g, blk, i);
  e |= bindc(&bw->ff3, m, H, 1, "%s.%s.%d.ff.3", g, blk, i);
  return e || !bw->ln1.g || !bw->ln1.b || !bw->ln2.g || !bw->ln2.b || !bw->dw || !bw->dwb;
}

static int bind_head(pt_v1b8k_model *m, head_w *h, const char *nm, int mid, int out) {
  h->ln.g = W(m, "enc.%s.0.g", nm);
  h->ln.b = W(m, "enc.%s.0.beta", nm);
  int e = bindc(&h->l1, m, mid, 1, "enc.%s.1", nm);
  e |= bindc(&h->l3, m, out, 1, "enc.%s.3", nm);
  return e || !h->ln.g || !h->ln.b;
}

/* Vocoder stage geometry comes from the container, so one binary serves every v1 variant
 * (v1b_8k: channels 48/24/12/6, strides 8/4/2/2, hop 128; v1b_16k: 80/40/20/10, strides 8/8/2/2, hop 256).
 * ST_RATE[s] = output rate of stage s in samples per mel frame = product of strides 0..s; the sample
 * rate, hop, frame limits and window slack all follow from these numbers - nothing below may assume
 * a particular variant. */
static int voc_geom(pt_v1b8k_model *m) {
  const float *g = W(m, "voc.geom");
  const float *meta = W(m, "model.meta");
  if (!g || !meta) return 1;
  int rate = 1, kmax = 6;   /* conv_post K7 -> span 6 */
  for (int s = 0; s < 4; s++) {
    int c = (int)g[s * 4], st = (int)g[s * 4 + 1], pd = (int)g[s * 4 + 2], k = (int)g[s * 4 + 3];
    if (c <= 0 || st <= 0 || pd < 0 || pd * 2 != st || k != 2 * st) return 1;   /* 2-tap-per-phase packing */
    ST_COUT[s] = c; ST_STRIDE[s] = st; ST_PAD[s] = pd; ST_KERNEL[s] = k;
    rate *= st;
    if (k - 1 > kmax) kmax = k - 1;
    ST_RATE[s] = rate;
  }
  int resblock_span = (RB_K[2] - 1) * RB_D[2];   /* K11 d5 convs: the widest read span in the graph */
  V_HOP = rate;
  V_SLACK = 16 + (resblock_span > kmax ? resblock_span : kmax);
  m->sample_rate = (int)meta[0];
  m->hop = (int)meta[1];
  m->abs_bins = (int)meta[2];
  m->max_frames = (int)meta[3];
  if (m->sample_rate <= 0 || m->hop != V_HOP || m->abs_bins <= 0 || m->max_frames <= 0) return 1;
  return 0;
}

pt_v1b8k_model *pt_v1b8k_load(const char *path) {
  pt_v1b8k_model *m = calloc(1, sizeof *m);
  if (!m || pt_weights_open(&m->w, path)) { free(m); return 0; }
  if (voc_geom(m)) { fprintf(stderr, "v1b8k: missing/invalid voc.geom or model.meta in %s\n", path); pt_weights_close(&m->w); free(m); return 0; }
  int e = 0;
  m->emb_ph = W(m, "enc.phone.emb");
  m->emb_tone = W(m, "enc.tone.emb");
  m->emb_lang = W(m, "enc.lang.emb");
  m->emb_spk = W(m, "enc.speaker.emb");
  e |= bindc(&m->spk_proj, m, H, 1, "enc.speaker_proj");
  for (int i = 0; i < 5; i++) e |= bind_block(m, &m->eb[i], "enc", "encoder", i, 672);
  e |= bind_head(m, &m->hdur, "duration_head", 168, 1);
  e |= bind_head(m, &m->hen, "energy_head", 84, 1);
  e |= bind_head(m, &m->hbr, "bright_head", 84, 1);
  e |= bind_head(m, &m->hpi, "pitch_head", 168, 2);
  m->eproj_w = W(m, "enc.energy_proj.w");
  m->eproj_b = W(m, "enc.energy_proj.b");
  m->bproj_w = W(m, "enc.bright_proj.w");
  m->bproj_b = W(m, "enc.bright_proj.b");
  e |= bindc(&m->fp0, m, H, 1, "dec.frame_proj.0");
  e |= bindc(&m->fp2, m, H, 1, "dec.frame_proj.2");
  e |= bindc(&m->lc0, m, 336, 1, "dec.local_ctx.0");
  e |= bindc(&m->lc2, m, H, 1, "dec.local_ctx.2");
  m->abs_emb = W(m, "dec.abs_frame.emb");
  e |= bindc(&m->ref0, m, 96, 1, "dec.frame_pitch_refiner.0");
  for (int g = 0; g < 4; g++) e |= bindc(&m->ref2[g], m, 24, 1, "dec.frame_pitch_refiner.2.g%d", g);
  e |= bindc(&m->ref4, m, 2, 1, "dec.frame_pitch_refiner.4");
  e |= bindc(&m->pp0, m, H, 1, "dec.pitch_proj.0");
  e |= bindc(&m->pp2, m, H, 1, "dec.pitch_proj.2");
  for (int i = 0; i < 6; i++) e |= bind_block(m, &m->db[i], "dec", "decoder", i, 504);
  e |= bindc(&m->gru_ih, m, 504, 1, "dec.frame_gru.wih");
  m->gru_r = W(m, "dec.frame_gru.r");
  m->gru_rb = W(m, "dec.frame_gru.rb");
  m->mh.g = W(m, "dec.mel_head.0.g");
  m->mh.b = W(m, "dec.mel_head.0.beta");
  e |= bindc(&m->mh1, m, H, 1, "dec.mel_head.1");
  e |= bindc(&m->mh3, m, 80, 1, "dec.mel_head.3");
  e |= bindc(&m->pn0, m, H, 1, "dec.postnet.0");
  e |= bindc(&m->pn2, m, H, 1, "dec.postnet.2");
  e |= bindc(&m->pn4, m, 80, 1, "dec.postnet.4");
  e |= bindc(&m->pre, m, 0, 1, "voc.conv_pre");   /* cout from the packed tensor = conv_pre Cout */
  for (int s = 0; s < 4; s++) {
    /* phase-stacked ConvTranspose: cout = stride * Cout4 rows, all real (padding inside each phase) */
    e |= bindc(&m->ups[s], m, ST_STRIDE[s] * (int)r4(ST_COUT[s]), 1, "voc.ups.%d", s);
    m->up_a[s] = W(m, "voc.up_acts.%d.a", s);
    m->up_ia[s] = W(m, "voc.up_acts.%d.ia", s);
    for (int j = 0; j < 3; j++) {
      int r = s * 3 + j;
      for (int k = 0; k < 3; k++) {
        e |= bindc(&m->c1[r][k], m, ST_COUT[s], RB_D[k], "voc.resblocks.%d.convs1.%d", r, k);
        e |= bindc(&m->c2[r][k], m, ST_COUT[s], 1, "voc.resblocks.%d.convs2.%d", r, k);
        m->a1[r][k] = W(m, "voc.resblocks.%d.acts1.%d.a", r, k);
        m->ia1[r][k] = W(m, "voc.resblocks.%d.acts1.%d.ia", r, k);
        m->a2[r][k] = W(m, "voc.resblocks.%d.acts2.%d.a", r, k);
        m->ia2[r][k] = W(m, "voc.resblocks.%d.acts2.%d.ia", r, k);
        if (!m->a1[r][k] || !m->a2[r][k] || !m->ia1[r][k] || !m->ia2[r][k]) e = 1;
        if (m->c1[r][k].K != RB_K[j]) e = 1;
      }
    }
  }
  m->post_a = W(m, "voc.post_act.a");
  m->post_ia = W(m, "voc.post_act.ia");
  e |= bindc(&m->post, m, 1, 1, "voc.conv_post");
  if (e || !m->emb_ph || !m->emb_tone || !m->emb_lang || !m->emb_spk || !m->eproj_w || !m->bproj_w || !m->abs_emb ||
      !m->gru_r || !m->gru_rb || !m->mh.g || !m->post_a) {
    fprintf(stderr, "v1b8k: weight binding failed\n");
    pt_v1b8k_free(m);
    return 0;
  }
  return m;
}

void pt_v1b8k_free(pt_v1b8k_model *m) {
  if (!m) return;
  pt_weights_close(&m->w);
  free(m);
}

int pt_v1b8k_sample_rate(const pt_v1b8k_model *m) { return m->sample_rate; }   /* 8000 | 16000 */
int pt_v1b8k_hop(const pt_v1b8k_model *m) { return m->hop; }                   /* samples per mel frame */
int pt_v1b8k_pre_rows(const pt_v1b8k_model *m) { return m->pre.cout4; }        /* voc conv_pre output rows */

/* Vocoder stage geometry (diagnostics / benchmarks): stage s has cout channels, expands by stride with a
 * kernel 2*stride ConvTranspose (packed as stride phases x 2 taps) and outputs at rate samples per frame. */
int pt_v1b8k_voc_stage(const pt_v1b8k_model *m, int s, int *cout, int *stride, int *kernel, int *rate) {
  if (s < 0 || s > 3 || !m) return -1;
  if (cout) *cout = ST_COUT[s];
  if (stride) *stride = ST_STRIDE[s];
  if (kernel) *kernel = ST_KERNEL[s];
  if (rate) *rate = ST_RATE[s];
  return 0;
}

/* ------------------------------------------------------------------ vocoder chunk plan */
typedef struct {
  long S0, S1;
  long vlo[5], vhi[5];   /* V_s = stage-s input (after snake), V_4 = post_act output; V_0 = conv_pre output */
  long qlo[4], qhi[4];   /* ConvT phase columns */
  long ulo[4], uhi[4];   /* ConvT output (stage input before resblocks) */
} vplan;

static void plan_chunk(vplan *p, long a, long b) {
  p->S0 = a * V_HOP;
  p->S1 = b * V_HOP;
  p->vlo[4] = p->S0 - 3;   /* conv_post K7 pad3 */
  p->vhi[4] = p->S1 + 3;
  for (int s = 3; s >= 0; s--) {
    long nlo = p->vlo[s + 1] - RB_REACH, nhi = p->vhi[s + 1] + RB_REACH;
    long st = ST_STRIDE[s], pd = ST_PAD[s];
    p->qlo[s] = fdiv(nlo + pd, st);
    p->qhi[s] = fdiv(nhi - 1 + pd, st) + 1;
    p->ulo[s] = p->qlo[s] * st - pd;
    p->uhi[s] = p->qhi[s] * st - pd;
    p->vlo[s] = p->qlo[s] - 1;
    p->vhi[s] = p->qhi[s];
  }
}

/* ------------------------------------------------------------------ stream */
static float *zalloc_rows(pt_arena *A, long rows, long ld) {
  float *p = pt_arena_alloc(A, (size_t)rows * ld * sizeof(float));
  if (p) memset(p, 0, (size_t)rows * ld * sizeof(float));
  return p;
}

/* alloc == 0: only count bytes (dry run used to size the arena) */
static float *ralloc(pt_arena *A, long rows, long ld, int alloc, int *err) {
  if (!alloc) {
    A->used += ((size_t)rows * ld * sizeof(float) + 63) & ~(size_t)63;
    return 0;
  }
  float *p = zalloc_rows(A, rows, ld);
  if (!p) *err = 1;
  return p;
}

static void vwin_alloc(pt_arena *A, vwin *v, int C, long cols, int alloc, int *err) {
  vwin t;
  t.C = (int)r4(C);
  t.ld = r16(cols + V_SLACK);
  t.cap = t.ld - V_SLACK;
  t.lo = t.hi = 0;
  t.base = ralloc(A, t.C, t.ld, alloc, err);
  if (alloc) *v = t;
}

static void hist_alloc(pt_arena *A, vhist *v, int C, int len, int alloc, int *err) {
  float *p = len > 0 ? ralloc(A, r4(C), len, alloc, err) : 0;
  if (alloc) { v->h = p; v->C = (int)r4(C); v->len = len; v->hi = 0; }
}

static size_t ws_plan(pt_v1b8k_ws *ws, int alloc) {
  pt_arena *A = &ws->A;
  size_t before = A->used;
  long ldT = ws->ldT, ldF = ws->ldF;
  int e = 0;
  encdec_bufs *w = &ws->b;
  float **encb[] = {&w->ex, &w->eh, &w->ed, &w->ep, &w->ef, &w->ha, &w->cond, &w->pit, &w->logd};
  int encr[] = {H, H, 336, H, 672, H, H, 4, 12};
  for (int i = 0; i < 9; i++) {
    float *p = ralloc(A, encr[i], ldT, alloc, &e);
    if (alloc && p) *encb[i] = p + PADL;
  }
  float **decb[] = {&w->R, &w->dx, &w->dh, &w->dd, &w->dp, &w->df, &w->rf};
  int decr[] = {180, H, H, 336, H, 504, 4};
  for (int i = 0; i < 7; i++) {
    float *p = ralloc(A, decr[i], ldF, alloc, &e);
    if (alloc && p) *decb[i] = p + PADL;
  }
  float *ints = ralloc(A, 1, 2 * (ws->maxT + 16) + 2 * (ws->maxF + 16), alloc, &e);
  if (alloc && ints) {
    w->dur = (int64_t *)ints;
    w->tok = (int32_t *)(w->dur + ws->maxT + 16);
    w->ap = w->tok + ws->maxF + 16;
    w->lct = w->ef;   /* 504 <= 672 rows */
    w->lca = w->ed;   /* 336 rows */
    w->lcb = w->ha;   /* 168 rows */
    w->dg = w->R;     /* 168 <= 180 rows */
    w->mp = w->df;    /* 80 <= 504 rows */
  }
  if (e) return 0;
  return A->used - before;
}

pt_v1b8k_ws *pt_v1b8k_ws_new(const pt_v1b8k_model *m, int max_phones, int max_frames) {
  (void)m;
  pt_v1b8k_ws *ws = calloc(1, sizeof *ws);
  if (!ws) return 0;
  ws->maxT = max_phones;
  ws->maxF = max_frames;
  ws->ldT = PADL + r16(max_phones) + SLACK;
  ws->ldF = PADL + r16(max_frames) + SLACK;
  size_t need = ws_plan(ws, 0) + (1 << 12);
  if (pt_arena_init(&ws->A, need)) { free(ws); return 0; }
  if (!ws_plan(ws, 1)) { pt_arena_free(&ws->A); free(ws); return 0; }
  return ws;
}
void pt_v1b8k_ws_free(pt_v1b8k_ws *ws) {
  if (!ws) return;
  pt_arena_free(&ws->A);
  free(ws);
}
size_t pt_v1b8k_ws_bytes(const pt_v1b8k_ws *ws) { return ws->A.cap; }

static size_t stream_plan(pt_v1b8k_stream *s, int alloc) {
  const pt_v1b8k_model *m = s->m;
  pt_arena *A = &s->A;
  size_t before = A->used;
  int e = 0;
  long ldM = r16(s->maxF + 2 * MELPAD + 16);
  float *mel = ralloc(A, 80, ldM, alloc, &e);
  if (alloc && mel) s->mel = mel + MELPAD;
  /* vocoder windows sized for the largest chunk (widths do not depend on the chunk position).
   * Stages run one after another inside a chunk and nothing in these windows outlives its stage (cross-chunk
   * state lives in the vhist histories), so U/T1/T2/XA/XB/PH share one block per role across the 4 stages;
   * SUM alternates between two blocks (stage s output is read by stage s+1), conv_pre output uses block 1. */
  vplan p;
  plan_chunk(&p, 1000, 1000 + s->chunkF);
  long need[6] = {0}, needS = 0;   /* floats per role: U, T1, T2, XA, XB, PH */
  long wcols[4], phcols[4], scols[5];
  scols[0] = p.vhi[0] - p.vlo[0] + 8;
  needS = m->pre.cout4 * r16(scols[0] + V_SLACK);   /* conv_pre output rows = conv_pre Cout */
  for (int st = 0; st < 4; st++) {
    long C4 = r4(ST_COUT[st]);
    wcols[st] = p.uhi[st] - p.ulo[st] + 8;
    phcols[st] = p.qhi[st] - p.qlo[st] + 8;
    scols[st + 1] = p.vhi[st + 1] - p.vlo[st + 1] + 8;
    long w = C4 * r16(wcols[st] + V_SLACK);
    for (int r = 0; r < 5; r++) if (w > need[r]) need[r] = w;
    long ph = ST_STRIDE[st] * C4 * r16(phcols[st] + V_SLACK);
    if (ph > need[5]) need[5] = ph;
    long sm = C4 * r16(scols[st + 1] + V_SLACK);
    if (sm > needS) needS = sm;
  }
  float *blk[6], *sblk[2];
  for (int r = 0; r < 6; r++) blk[r] = ralloc(A, 1, need[r], alloc, &e);
  for (int r = 0; r < 2; r++) sblk[r] = ralloc(A, 1, needS, alloc, &e);
  if (alloc && !e) {
    vwin *roles[5] = {s->u, s->t1, s->t2, s->xa, s->xb};
    for (int st = 0; st < 4; st++) {
      int C4 = (int)r4(ST_COUT[st]);
      for (int r = 0; r < 5; r++) {
        vwin *v = &roles[r][st];
        v->C = C4; v->ld = r16(wcols[st] + V_SLACK); v->cap = v->ld - V_SLACK; v->lo = v->hi = 0; v->base = blk[r];
      }
      vwin *v = &s->ph[st];
      v->C = ST_STRIDE[st] * C4; v->ld = r16(phcols[st] + V_SLACK); v->cap = v->ld - V_SLACK; v->lo = v->hi = 0; v->base = blk[5];
      v = &s->sum[st];
      v->C = C4; v->ld = r16(scols[st + 1] + V_SLACK); v->cap = v->ld - V_SLACK; v->lo = v->hi = 0; v->base = sblk[st & 1];
    }
    vwin *v = &s->pre;
    v->C = m->pre.cout4; v->ld = r16(scols[0] + V_SLACK); v->cap = v->ld - V_SLACK; v->lo = v->hi = 0; v->base = sblk[1];
  }
  vwin_alloc(A, &s->wav, 1, p.S1 - p.S0 + 8, alloc, &e);
  /* histories (sizes = how far left the consumers reach into the previous chunk's columns) */
  for (int st = 0; st <= 4; st++) {
    int C = st == 0 ? m->pre.cout4 : ST_COUT[st - 1];
    hist_alloc(A, &s->hV[st], C, st == 4 ? 6 : 1, alloc, &e);
  }
  for (int st = 0; st < 4; st++) {
    int C = ST_COUT[st];
    long HU = (p.uhi[st] - p.vhi[st + 1]) - 10 + ST_STRIDE[st];
    hist_alloc(A, &s->hU[st], C, (int)HU, alloc, &e);
    for (int j = 0; j < 3; j++) {
      int h = (RB_K[j] - 1) / 2;
      for (int k = 0; k < 3; k++) {
        hist_alloc(A, &s->hT1[st][j][k], C, (RB_K[j] - 1) * RB_D[k], alloc, &e);
        hist_alloc(A, &s->hT2[st][j][k], C, RB_K[j] - 1, alloc, &e);
        hist_alloc(A, &s->hX[st][j][k], C, k ? (RB_D[k] + 1) * h : 0, alloc, &e);
      }
    }
  }
  if (e) return 0;
  return A->used - before;
}

pt_v1b8k_stream *pt_v1b8k_stream_new_shared(const pt_v1b8k_model *m, int max_phones, int max_frames, int chunk_frames) {
  pt_v1b8k_stream *s = calloc(1, sizeof *s);
  if (!s) return 0;
  s->m = m;
  s->maxT = max_phones;
  s->maxF = max_frames;
  s->chunkF = chunk_frames < 4 ? 4 : chunk_frames;
  s->firstF = 3;
  s->ldT = PADL + r16(max_phones) + SLACK;
  s->ldF = PADL + r16(max_frames) + SLACK;
  memset(&s->A, 0, sizeof s->A);
  size_t need = stream_plan(s, 0) + (1 << 16);
  if (pt_arena_init(&s->A, need)) { free(s); return 0; }
  if (!stream_plan(s, 1)) { pt_arena_free(&s->A); free(s); return 0; }
  return s;
}

pt_v1b8k_stream *pt_v1b8k_stream_new(const pt_v1b8k_model *m, int max_phones, int max_frames, int chunk_frames) {
  pt_v1b8k_stream *s = pt_v1b8k_stream_new_shared(m, max_phones, max_frames, chunk_frames);
  if (!s) return 0;
  s->own_ws = pt_v1b8k_ws_new(m, max_phones, max_frames);
  if (!s->own_ws) { pt_v1b8k_stream_free(s); return 0; }
  return s;
}

void pt_v1b8k_set_ws_pool(pt_v1b8k_stream *s, pt_v1b8k_ws *(*acquire)(void *), void (*release)(void *, pt_v1b8k_ws *),
                          void *user) {
  s->ws_acquire = acquire;
  s->ws_release = release;
  s->ws_user = user;
}

void pt_v1b8k_stream_free(pt_v1b8k_stream *s) {
  if (!s) return;
  pt_v1b8k_ws_free(s->own_ws);
  pt_arena_free(&s->A);
  free(s);
}
size_t pt_v1b8k_stream_bytes(const pt_v1b8k_stream *s) { return s->A.cap + (s->own_ws ? s->own_ws->A.cap : 0); }
void pt_v1b8k_set_tap(pt_v1b8k_stream *s, pt_tap_cb cb, void *user) { s->tap = cb; s->tap_user = user; }
void pt_v1b8k_set_chunking(pt_v1b8k_stream *s, int first, int chunk) {
  s->firstF = first < 1 ? 1 : first;
  if (chunk >= 4 && chunk <= s->chunkF) s->chunkF = chunk;
}

/* optional phase profile (env PT_PROF=1): single-stream diagnostics only (not thread-safe) */
enum { PR_ENC, PR_REG, PR_DPRE, PR_DBLK, PR_GRU, PR_MEL, PR_VPRE, PR_VUPS, PR_VSN, PR_VC1, PR_VC2, PR_VSUM, PR_VPOST, PR_N };
static const char *PR_NAME[PR_N] = {"encoder", "regulator", "dec pre (proj/lc/refiner)", "dec 6 blocks", "dec GRU",
                                    "mel head + postnet", "voc conv_pre", "voc ConvT+interleave", "voc snake1 copies",
                                    "voc conv1 (+snake2)", "voc conv2 (+residual)", "voc stage sum/snake", "voc conv_post"};
static double g_prof[PR_N];
static int g_prof_on = -1;
static inline double prof_t(void) {
  if (g_prof_on < 0) { const char *e = getenv("PT_PROF"); g_prof_on = e && *e == '1'; }
  return g_prof_on ? pt_now() : 0;
}
#define PROF(id, t0) do { if (g_prof_on > 0) { double _t = pt_now(); g_prof[id] += _t - (t0); (t0) = _t; } } while (0)
void pt_v1b8k_prof_dump(void) {
  double tot = 0;
  for (int i = 0; i < PR_N; i++) tot += g_prof[i];
  if (tot <= 0) return;
  fprintf(stderr, "phase profile (PT_PROF=1):\n");
  for (int i = 0; i < PR_N; i++) fprintf(stderr, "  %-28s %8.1f ms  %5.1f%%\n", PR_NAME[i], g_prof[i] * 1e3, 100 * g_prof[i] / tot);
  memset(g_prof, 0, sizeof g_prof);
}

#define TAP(nm, x, C, ld, n) do { if (s->tap) s->tap(s->tap_user, nm, x, C, ld, n); } while (0)

/* ------------------------------------------------------------------ shared blocks */
static void zero_cols(float *x, int C, long ld, long from, long to) {
  if (to - from <= 0) return;
  for (int c = 0; c < C; c++) memset(x + c * ld + from, 0, (to - from) * sizeof(float));
}

static void linear(const pt_conv_w *c, const float *X, long ldx, float *Y, long ldy, long n, int act,
                   const float *res) {
  pt_epi e = {act, 0, 0, 1.0f, res, ldy};
  pt_conv(c, X, ldx, Y, ldy, n, (act || res) ? &e : 0);
}

/* gated depthwise-conv block shared by encoder (ff 672) and decoder (ff 504):
 *  h = LN1(x); d = dwconv7(h) (336 out, o reads o//2); g = d[:168]*sigmoid(d[168:]);
 *  p = point(g); x += p; h = LN2(x); x += ff3(silu(ff0(h))) */
static void dconv_block(const blk_w *b, float *x, float *h, float *d, float *p, float *f, long ld, long n) {
  long n4 = r4(n);
  pt_ln_res(x, 0, h, H, ld, n4, b->ln1.g, b->ln1.b, LN_EPS);
  zero_cols(h, H, ld, n, n + 4);  /* right zero padding for the K7 pad3 depthwise conv */
  for (int o = 0; o < 336; o++) pt_dw_row(d + o * ld, h + (o >> 1) * ld - 3, n4, b->dw + o * 7, 7, b->dwb[o]);
  for (int o = 0; o < H; o++) pt_sigmul_row(d + o * ld, d + (o + H) * ld, n4);
  linear(&b->point, d, ld, p, ld, n, 0, 0);
  pt_ln_res(x, p, h, H, ld, n4, b->ln2.g, b->ln2.b, LN_EPS);
  linear(&b->ff0, h, ld, f, ld, n, PT_ACT_SILU, 0);
  linear(&b->ff3, f, ld, x, ld, n, 0, x);
}

/* ------------------------------------------------------------------ encoder */
static int run_encoder(pt_v1b8k_stream *s, const int32_t *ph, const int32_t *tn, const int32_t *lg, int T, int sid) {
  const pt_v1b8k_model *m = s->m;
  long ld = s->ldT, n4 = r4(T);
  float spk[H];
  int sd = sid < 0 ? 0 : sid > 1 ? 1 : sid;
  /* speaker_proj: Gemm(emb[64], W^T) + b : tiny matvec */
  for (int c = 0; c < H; c++) {
    const float *wp = m->spk_proj.wp + (c / 4) * 64 * 4 + (c % 4);
    float acc = m->spk_proj.b[c];
    for (int k = 0; k < 64; k++) acc += wp[k * 4] * m->emb_spk[sd * 64 + k];
    spk[c] = acc;
  }
  for (int t = 0; t < T; t++) {
    int p = ph[t] < 0 ? 0 : ph[t] > 255 ? 255 : ph[t];
    int to = tn[t] < 0 ? 0 : tn[t] > 15 ? 15 : tn[t];   /* Clip(tone, max=15) */
    int la = lg[t] < 0 ? 0 : lg[t] > 3 ? 3 : lg[t];     /* Clip(lang, max=3)  */
    for (int c = 0; c < H; c++)
      s->w.ex[c * ld + t] = ((m->emb_ph[p * H + c] + m->emb_tone[to * H + c]) + m->emb_lang[la * H + c]) + spk[c];
  }
  TAP("enc_emb", s->w.ex, H, ld, T);
  for (int i = 0; i < 5; i++) {
    dconv_block(&m->eb[i], s->w.ex, s->w.eh, s->w.ed, s->w.ep, s->w.ef, ld, T);
    char nm[32];
    snprintf(nm, sizeof nm, "enc_blk%d", i);
    TAP(nm, s->w.ex, H, ld, T);
  }
  /* heads: LN -> linear -> SiLU -> linear */
  const head_w *hs[4] = {&m->hdur, &m->hen, &m->hbr, &m->hpi};
  for (int k = 0; k < 4; k++) {
    pt_ln_res(s->w.ex, 0, s->w.eh, H, ld, n4, hs[k]->ln.g, hs[k]->ln.b, LN_EPS);
    linear(&hs[k]->l1, s->w.eh, ld, s->w.ha, ld, T, PT_ACT_SILU, 0);
    linear(&hs[k]->l3, s->w.ha, ld, k < 3 ? s->w.logd + k * 4 * ld : s->w.pit, ld, T, 0, 0);
  }
  const float *le = s->w.logd + 4 * ld, *lb = s->w.logd + 8 * ld;
  for (int t = 0; t < T; t++) {
    /* durations = clamp_min(round(clip(exp(logd) - 1, 0, 80)), 1)  (Round = half to even) */
    float v = expf(s->w.logd[t]) - 1.0f;
    v = v < 0.f ? 0.f : v > 80.f ? 80.f : v;
    v = nearbyintf(v);
    s->w.dur[t] = v < 1.f ? 1 : (int64_t)v;
    float e = le[t], br = lb[t];
    for (int c = 0; c < H; c++)
      s->w.cond[c * ld + t] = (s->w.ex[c * ld + t] + (e * m->eproj_w[c] + m->eproj_b[c])) + (br * m->bproj_w[c] + m->bproj_b[c]);
    float p1 = s->w.pit[ld + t];
    s->w.pit[ld + t] = p1 < 0.f ? 0.f : p1 > 1.f ? 1.f : p1;
  }
  TAP("logdur", s->w.logd, 1, ld, T);
  TAP("conditioned", s->w.cond, H, ld, T);
  TAP("pitch", s->w.pit, 2, ld, T);
  return 0;
}

/* ------------------------------------------------------------------ regulator (exact host_regulate port) */
int pt_v1b8k_regulate(const float *cond, long ldc, const int64_t *dur, const float *pitch, long ldp, int T,
                      int abs_bins, int max_frames, float *frames, float *fm, float *lc, int32_t *abs_pos,
                      float *pf, long ld, int32_t *tok_out) {
  /* c = cond[0]; d = dur[0].astype(np.int64); d[d < 0] = 0 */
  int F = 0, tc = 0;
  for (int t = 0; t < T; t++) {
    int64_t d = dur[t] < 0 ? 0 : dur[t];
    F += (int)d;
    tc += d > 0;
  }
  /* tc = max(1, int((d > 0).sum())) */
  if (tc < 1) tc = 1;
  const float PI_F = (float)M_PI;
  long f = 0;
  for (int t = 0; t < T; t++) {
    int64_t d = dur[t] < 0 ? 0 : dur[t];
    int tp = t > 0 ? t - 1 : 0, tn = t < T - 1 ? t + 1 : T - 1;
    for (int64_t w = 0; w < d; w++, f++) {
      /* frames = np.repeat(c, d, axis=0) */
      for (int c = 0; c < H; c++) frames[c * ld + f] = cond[c * ldc + t];
      /* rel = (within / np.maximum(dpf - 1, 1)).astype(np.float32)   int64 / float32 -> float64 */
      float dpf = (float)d;
      double den = (double)(dpf - 1.0f > 1.0f ? dpf - 1.0f : 1.0f);
      float rel = (float)((double)w / den);
      /* token_pos = (tok / max(1, tc - 1)).astype(np.float32)   int64 / int -> float64 */
      float tpos = (float)((double)t / (double)(tc - 1 > 1 ? tc - 1 : 1));
      /* ld = (np.log1p(dpf) / 6.0).astype(np.float32)  (float32 ops) */
      float ldv = log1pf(dpf) / 6.0f;
      /* center = 1.0 - np.abs(rel * 2 - 1) */
      float center = 1.0f - fabsf(rel * 2.0f - 1.0f);
      /* np.sin(rel*np.pi), np.cos(rel*np.pi): float32 * python float -> float32 */
      float ang = rel * PI_F;
      float v[8] = {rel, 1.0f - rel, center, sinf(ang), cosf(ang), tpos, ldv, dpf / 40.0f};
      for (int k = 0; k < 8; k++) fm[k * ld + f] = v[k];
      /* lc = np.repeat(np.concatenate([prev, c, nxt], -1), d, axis=0)  (prev/nxt clamp at the ends) */
      if (lc)
        for (int c = 0; c < H; c++) {
          lc[c * ld + f] = cond[c * ldc + tp];
          lc[(H + c) * ld + f] = cond[c * ldc + t];
          lc[(2 * H + c) * ld + f] = cond[c * ldc + tn];
        }
      /* ap = np.minimum(pos*abs_bins//max(1, max_frames), abs_bins-1) */
      long mf = max_frames > 1 ? max_frames : 1;
      long apv = f * abs_bins / mf;
      abs_pos[f] = (int32_t)(apv < abs_bins - 1 ? apv : abs_bins - 1);
      /* pf = np.repeat(pitch[0], d, axis=0) */
      pf[f] = pitch[t];
      pf[ld + f] = pitch[ldp + t];
      if (tok_out) tok_out[f] = t;
    }
  }
  return F;
}

/* ------------------------------------------------------------------ GRU (ONNX semantics, linear_before_reset=1) */
static inline float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

static void gru_dir(const pt_v1b8k_model *m, int d, const float *gx, long ld, int F, float *out, float *h, float *rh) {
  const float *R = m->gru_r + (long)d * 252 * 84, *rb = m->gru_rb + d * 252;
  const float *gz = gx + (long)(d * 252) * ld, *gr = gz + 84 * ld, *gh = gz + 168 * ld;
  memset(h, 0, 84 * sizeof(float));
  for (int s = 0; s < F; s++) {
    int t = d ? F - 1 - s : s;
    for (int j = 0; j < 252; j++) {
      const float *r = R + j * 84;
      float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      for (int k = 0; k < 84; k += 4) {
        a0 += r[k] * h[k];
        a1 += r[k + 1] * h[k + 1];
        a2 += r[k + 2] * h[k + 2];
        a3 += r[k + 3] * h[k + 3];
      }
      rh[j] = ((a0 + a1) + (a2 + a3)) + rb[j];
    }
    for (int j = 0; j < 84; j++) {
      float z = sigm(gz[j * ld + t] + rh[j]);
      float r = sigm(gr[j * ld + t] + rh[84 + j]);
      float nn = tanhf(gh[j * ld + t] + r * rh[168 + j]);
      h[j] = (1.0f - z) * nn + z * h[j];
      out[(d * 84 + j) * ld + t] = h[j];
    }
  }
}

typedef struct { const pt_v1b8k_model *m; pt_v1b8k_stream *s; const float *gx; long ld; int F; float *out; } gru_job;
static void *gru_thread(void *p) {
  gru_job *g = p;
  gru_dir(g->m, 1, g->gx, g->ld, g->F, g->out, g->s->gh[1], g->s->grh[1]);
  return 0;
}

/* ------------------------------------------------------------------ decoder */
static int run_decoder(pt_v1b8k_stream *s, int T, int F, int threads) {
  const pt_v1b8k_model *m = s->m;
  long ld = s->ldF, ldT = s->ldT, n4 = r4(F);
  float *frames = s->w.R, *fm = s->w.R + H * ld, *pf = s->w.R + 176 * ld;
  /* regulator: frames / frame_meta / pitch_frame straight into the refiner's concat buffer [178][F] */
  double tp = prof_t();
  int F2 = pt_v1b8k_regulate(s->w.cond, ldT, s->w.dur, s->w.pit, ldT, T, m->abs_bins, m->max_frames, frames, fm, 0,
                             s->w.ap, pf, ld, s->w.tok);
  if (F2 != F) return -1;
  PROF(PR_REG, tp);
  TAP("reg_frame_meta", fm, 8, ld, F);
  /* x = frames + frame_proj(fm) */
  linear(&m->fp0, fm, ld, s->w.dd, ld, F, PT_ACT_SILU, 0);
  linear(&m->fp2, s->w.dd, ld, s->w.dx, ld, F, 0, frames);
  /* local_ctx is a per-frame MLP of repeat([prev,c,nxt]) == repeat(MLP(token-level [prev,c,nxt])):
   * compute it on T token columns (same per-column arithmetic -> bit-identical), then expand. */
  for (int t = 0; t < T; t++) {
    int tp = t > 0 ? t - 1 : 0, tn = t < T - 1 ? t + 1 : T - 1;
    for (int c = 0; c < H; c++) {
      s->w.lct[c * ldT + t] = s->w.cond[c * ldT + tp];
      s->w.lct[(H + c) * ldT + t] = s->w.cond[c * ldT + t];
      s->w.lct[(2 * H + c) * ldT + t] = s->w.cond[c * ldT + tn];
    }
  }
  linear(&m->lc0, s->w.lct, ldT, s->w.lca, ldT, T, PT_ACT_SILU, 0);
  linear(&m->lc2, s->w.lca, ldT, s->w.lcb, ldT, T, 0, 0);
  for (int c = 0; c < H; c++) {
    float *x = s->w.dx + c * ld;
    const float *l = s->w.lcb + c * ldT;
    for (int f = 0; f < F; f++) x[f] = (x[f] + l[s->w.tok[f]]) + m->abs_emb[(long)s->w.ap[f] * H + c];
  }
  /* frame pitch refiner on concat(frames, fm, pf) [178][F], K5 pad2 */
  zero_cols(s->w.R, 178, ld, F, F + 4);
  pt_epi silu = {PT_ACT_SILU, 0, 0, 1.0f, 0, 0};
  pt_conv(&m->ref0, s->w.R - 2, ld, s->w.dd, ld, F, &silu);
  zero_cols(s->w.dd, 96, ld, F, F + 4);
  for (int g = 0; g < 4; g++)
    pt_conv(&m->ref2[g], s->w.dd + g * 24 * ld - 2, ld, s->w.dp + g * 24 * ld, ld, F, &silu);
  linear(&m->ref4, s->w.dp, ld, s->w.rf, ld, F, 0, 0);
  for (int f = 0; f < F; f++) {
    s->w.rf[f] = pf[f] + s->w.rf[f];
    float v = pf[ld + f] + s->w.rf[ld + f];
    s->w.rf[ld + f] = v < 0.f ? 0.f : v > 1.f ? 1.f : v;
  }
  linear(&m->pp0, s->w.rf, ld, s->w.dd, ld, F, PT_ACT_SILU, 0);
  linear(&m->pp2, s->w.dd, ld, s->w.dx, ld, F, 0, s->w.dx);
  PROF(PR_DPRE, tp);
  TAP("dec_in", s->w.dx, H, ld, F);
  for (int i = 0; i < 6; i++) {
    dconv_block(&m->db[i], s->w.dx, s->w.dh, s->w.dd, s->w.dp, s->w.df, ld, F);
    char nm[32];
    snprintf(nm, sizeof nm, "dec_blk%d", i);
    TAP(nm, s->w.dx, H, ld, F);
  }
  /* bidirectional GRU: input projections as one GEMM, recurrences per direction */
  PROF(PR_DBLK, tp);
  linear(&m->gru_ih, s->w.dx, ld, s->w.df, ld, F, 0, 0);
  if (threads > 1) {
    gru_job g = {m, s, s->w.df, ld, F, s->w.dg};
    pthread_t th;
    pthread_create(&th, 0, gru_thread, &g);
    gru_dir(m, 0, s->w.df, ld, F, s->w.dg, s->gh[0], s->grh[0]);
    pthread_join(th, 0);
  } else {
    gru_dir(m, 0, s->w.df, ld, F, s->w.dg, s->gh[0], s->grh[0]);
    gru_dir(m, 1, s->w.df, ld, F, s->w.dg, s->gh[1], s->grh[1]);
  }
  PROF(PR_GRU, tp);
  TAP("dec_gru", s->w.dg, H, ld, F);
  /* x += gru ; mel_head LN fused */
  pt_ln_res(s->w.dx, s->w.dg, s->w.dh, H, ld, n4, m->mh.g, m->mh.b, LN_EPS);
  linear(&m->mh1, s->w.dh, ld, s->w.dd, ld, F, PT_ACT_SILU, 0);
  linear(&m->mh3, s->w.dd, ld, s->w.mp, ld, F, 0, 0);
  TAP("mel_pre", s->w.mp, 80, ld, F);
  /* postnet: mel = mel_pre + 0.1 * conv(tanh(conv(tanh(conv(mel_pre))))) */
  zero_cols(s->w.mp, 80, ld, F, F + 4);
  pt_epi th = {PT_ACT_TANH, 0, 0, 1.0f, 0, 0};
  pt_conv(&m->pn0, s->w.mp - 2, ld, s->w.dd, ld, F, &th);
  zero_cols(s->w.dd, H, ld, F, F + 4);
  pt_conv(&m->pn2, s->w.dd - 2, ld, s->w.dp, ld, F, &th);
  zero_cols(s->w.dp, H, ld, F, F + 4);
  long ldM = r16(s->maxF + 2 * MELPAD + 16);
  pt_epi pe = {PT_ACT_NONE, 0, 0, 0.1f, s->w.mp, ld};
  pt_conv(&m->pn4, s->w.dp - 2, ld, s->mel, ldM, F, &pe);
  zero_cols(s->mel, 80, ldM, F, F + MELPAD);
  PROF(PR_MEL, tp);
  TAP("mel", s->mel, 80, ldM, F);
  return 0;
}

/* ------------------------------------------------------------------ vocoder */
static inline float *vat(const vwin *v, int c, long t) { return v->base + c * v->ld + (t - v->lo); }

static int vset(vwin *v, long lo, long hi) {
  if (hi - lo > v->cap) { fprintf(stderr, "v1b8k: vocoder window %ld > cap %ld\n", hi - lo, v->cap); return -1; }
  v->lo = lo;
  v->hi = hi;
  return 0;
}

/* zero columns of v outside the true signal [0, N) */
static void vmask(vwin *v, long N) {
  if (v->lo < 0) zero_cols(v->base, v->C, v->ld, 0, (v->hi < 0 ? v->hi : 0) - v->lo);
  if (v->hi > N) {
    long a = (N > v->lo ? N : v->lo) - v->lo;
    zero_cols(v->base, v->C, v->ld, a, v->hi - v->lo);
  }
}

/* valid conv into an already-placed window: out columns [lo, hi) (needs in->lo <= lo - pad) */
static int vconv(const pt_conv_w *c, int pad, const vwin *in, vwin *out, long lo, long hi, const pt_epi *e) {
  if (lo < out->lo || hi > out->hi) { fprintf(stderr, "v1b8k: out window\n"); return -1; }
  if (lo - pad < in->lo || hi - pad + (long)(c->K - 1) * c->dil > in->hi) {
    fprintf(stderr, "v1b8k: conv window [%ld,%ld) pad %d outside input [%ld,%ld)\n", lo, hi, pad, in->lo, in->hi);
    return -1;
  }
  pt_conv(c, vat(in, 0, lo - pad), in->ld, vat(out, 0, lo), out->ld, hi - lo, e);
  return 0;
}

/* Place tensor window for a chunk. first chunk: [plan_lo, hi) computed from scratch; later chunks compute only
 * the new columns [prev_hi, hi) and the H history columns [prev_hi - H, prev_hi) are restored in front.
 * Returns the first column that must be computed. */
static long vbegin(vwin *w, vhist *h, int first, long plan_lo, long hi) {
  long lo = first ? plan_lo : h->hi;
  long blo = first ? lo : lo - h->len;
  if (vset(w, blo, hi)) return LONG_MIN;
  if (!first && h->len > 0)
    for (int c = 0; c < h->C && c < w->C; c++) memcpy(vat(w, c, blo), h->h + (long)c * h->len, h->len * sizeof(float));
  return lo;
}
static int vend(const vwin *w, vhist *h) {
  if (h->len > 0) {
    if (w->hi - h->len < w->lo) { fprintf(stderr, "v1b8k: window narrower than history\n"); return -1; }
    for (int c = 0; c < h->C && c < w->C; c++) memcpy(h->h + (long)c * h->len, vat(w, c, w->hi - h->len), h->len * sizeof(float));
  }
  h->hi = w->hi;
  return 0;
}

/* one vocoder chunk: output samples [p->S0, p->S1) into s->wav.
 * Exactness: every tensor value at an absolute column is a pure function of the mel (valid convolutions,
 * columns outside [0, N) forced to 0 = ONNX zero padding), so values computed in a previous chunk and carried
 * as history are identical to what a full-utterance pass computes. Zero recompute after the first chunk. */
static int voc_chunk(pt_v1b8k_stream *s, const vplan *p, int F, int first, int full_tap) {
  const pt_v1b8k_model *m = s->m;
  long ldM = r16(s->maxF + 2 * MELPAD + 16);
  vwin mel = {s->mel - MELPAD, ldM, -MELPAD, F + MELPAD, ldM, 80};
  if (p->vlo[0] - 3 < -MELPAD || p->vhi[0] + 3 > F + MELPAD) { fprintf(stderr, "v1b8k: mel window\n"); return -1; }
  long ncomp = 0;
  double tp = prof_t();
  /* conv_pre (K7 pad3) + fused snake(up_acts.0), then zero outside [0, F) */
  pt_epi e0 = {PT_ACT_SNAKE, m->up_a[0], m->up_ia[0], 1.0f, 0, 0};
  long lo = vbegin(&s->pre, &s->hV[0], first, p->vlo[0], p->vhi[0]);
  if (lo == LONG_MIN || vconv(&m->pre, 3, &mel, &s->pre, lo, p->vhi[0], &e0)) return -1;
  vmask(&s->pre, F);
  if (vend(&s->pre, &s->hV[0])) return -1;
  PROF(PR_VPRE, tp);
  vwin *V = &s->pre;
  for (int st = 0; st < 4; st++) {
    int C = ST_COUT[st], C4 = (int)r4(C), S = ST_STRIDE[st], pd = ST_PAD[st];
    long N = (long)F * ST_RATE[st];
    /* ConvTranspose as S phases of a 2-tap conv: P_r[q] = W_{r+S} x[q-1] + W_r x[q], q in [qlo, qhi) */
    vwin *PH = &s->ph[st], *U = &s->u[st];
    long qlo = first ? p->qlo[st] : (s->hU[st].hi + pd) / S; /* new q range: [prev qhi, qhi) */
    if (vset(PH, qlo, p->qhi[st]) || vconv(&m->ups[st], 1, V, PH, qlo, p->qhi[st], 0)) return -1;
    lo = vbegin(U, &s->hU[st], first, p->ulo[st], p->uhi[st]);
    if (lo == LONG_MIN || lo != qlo * S - pd) { fprintf(stderr, "v1b8k: U alignment %ld vs %ld\n", lo, qlo * S - pd); return -1; }
    long nq = p->qhi[st] - qlo;
    for (int c = 0; c < C; c++) {
      float *u = vat(U, c, lo);
      for (int r = 0; r < S; r++) {
        const float *ph = PH->base + (long)(r * C4 + c) * PH->ld;
        for (long q = 0; q < nq; q++) u[q * S + r] = ph[q];
      }
    }
    vmask(U, N);
    if (vend(U, &s->hU[st])) return -1;
    PROF(PR_VUPS, tp);
    if (full_tap) {
      char nm[32];
      snprintf(nm, sizeof nm, "voc_ups%d", st);
      TAP(nm, vat(U, 0, 0), C, U->ld, N);
    }
    long Rlo = p->vlo[st + 1], Rhi = p->vhi[st + 1];
    vwin *SUM = &s->sum[st];
    long rlo = vbegin(SUM, &s->hV[st + 1], first, Rlo, Rhi);
    if (rlo == LONG_MIN) return -1;
    if (st == 0) ncomp += Rhi - rlo;
    for (int j = 0; j < 3; j++) {
      int rb = st * 3 + j, h = (RB_K[j] - 1) / 2;
      long reach = 0;
      for (int k = 0; k < 3; k++) reach += (long)(RB_D[k] + 1) * h;
      long xlo = Rlo - reach, xhi = Rhi + reach;   /* plan range of X(m) (first chunk) */
      const vwin *x = U;
      vwin *xbuf[2] = {&s->xa[st], &s->xb[st]};
      for (int k = 0; k < 3; k++) {
        int d = RB_D[k];
        /* T1 = snake1(X) */
        vwin *T1 = &s->t1[st], *T2 = &s->t2[st];
        long l1 = vbegin(T1, &s->hT1[st][j][k], first, xlo, xhi);
        if (l1 == LONG_MIN || l1 < x->lo) { fprintf(stderr, "v1b8k: T1 needs X history\n"); return -1; }
        long n4 = r4(xhi - l1);
        for (int c = 0; c < C; c++) {
          float *dd = vat(T1, c, l1);
          memcpy(dd, vat(x, c, l1), (xhi - l1) * sizeof(float));
          pt_snake_row(dd, n4, m->a1[rb][k][c], m->ia1[rb][k][c]);
        }
        if (vend(T1, &s->hT1[st][j][k])) return -1;
        PROF(PR_VSN, tp);
        /* T2 = snake2(conv1(T1)) */
        long l2 = vbegin(T2, &s->hT2[st][j][k], first, xlo + d * h, xhi - d * h);
        pt_epi e1 = {PT_ACT_SNAKE, m->a2[rb][k], m->ia2[rb][k], 1.0f, 0, 0};
        if (l2 == LONG_MIN || vconv(&m->c1[rb][k], d * h, T1, T2, l2, xhi - d * h, &e1)) return -1;
        vmask(T2, N);
        if (vend(T2, &s->hT2[st][j][k])) return -1;
        PROF(PR_VC1, tp);
        if (st == 0) ncomp += (xhi - l1) + (xhi - d * h - l2);
        /* X' = X + conv2(T2) */
        long olo = xlo + (long)(d + 1) * h, ohi = xhi - (long)(d + 1) * h;
        vwin *xo = k < 2 ? xbuf[k & 1] : &s->t1[st];  /* final X(3) only feeds the stage sum */
        vhist *hx = k < 2 ? &s->hX[st][j][k + 1] : 0;
        vhist none = {0, 0, 0, 0};
        if (!hx) { none.hi = first ? olo : rlo; hx = &none; }
        long l3 = vbegin(xo, hx, first, olo, ohi);
        if (l3 == LONG_MIN || l3 < x->lo) { fprintf(stderr, "v1b8k: residual needs X history\n"); return -1; }
        pt_epi e2 = {PT_ACT_NONE, 0, 0, 1.0f, vat(x, 0, l3), x->ld};
        if (vconv(&m->c2[rb][k], h, T2, xo, l3, ohi, &e2)) return -1;
        vmask(xo, N);
        if (k < 2 && vend(xo, hx)) return -1;
        PROF(PR_VC2, tp);
        if (st == 0) ncomp += ohi - l3;
        x = xo;
        xlo = olo;
        xhi = ohi;
      }
      /* stage sum over the new columns: ((0 + r0) + r1) + r2 */
      for (int c = 0; c < C; c++) {
        float *sm = vat(SUM, c, rlo);
        const float *xv = vat(x, c, rlo);
        if (j == 0) memcpy(sm, xv, (Rhi - rlo) * sizeof(float));
        else for (long t = 0; t < Rhi - rlo; t++) sm[t] = sm[t] + xv[t];
      }
    }
    const float *na = st < 3 ? m->up_a[st + 1] : m->post_a, *nia = st < 3 ? m->up_ia[st + 1] : m->post_ia;
    long n4 = r4(Rhi - rlo);
    for (int c = 0; c < C; c++) {
      float *sm = vat(SUM, c, rlo);
      for (long t = 0; t < Rhi - rlo; t++) sm[t] = sm[t] / 3.0f;
    }
    if (full_tap) {
      char nm[32];
      snprintf(nm, sizeof nm, "voc_stage%d", st);
      TAP(nm, vat(SUM, 0, 0), C, SUM->ld, N);
    }
    for (int c = 0; c < C; c++) pt_snake_row(vat(SUM, c, rlo), n4, na[c], nia[c]);
    if (vend(SUM, &s->hV[st + 1])) return -1;
    PROF(PR_VSUM, tp);
    V = SUM;
  }
  /* conv_post (K7 pad3, 6 -> 1) + tanh */
  pt_epi et = {PT_ACT_TANH, 0, 0, 1.0f, 0, 0};
  if (vset(&s->wav, p->S0, p->S1) || vconv(&m->post, 3, V, &s->wav, p->S0, p->S1, &et)) return -1;
  PROF(PR_VPOST, tp);
  s->vcols_comp += ncomp;
  s->vcols_use += (p->S1 - p->S0) / V_HOP;
  return 0;
}

/* ------------------------------------------------------------------ synth */
int pt_v1b8k_synth(pt_v1b8k_stream *s, const int32_t *phone, const int32_t *tone, const int32_t *lang, int T,
                   int sid, pt_pcm_cb cb, void *user, pt_synth_info *info) {
  double t0 = pt_now();
  if (T <= 0 || T > s->maxT) return -2;
  int threads = pt_pool_get_local();
  pt_v1b8k_ws *ws = s->own_ws ? s->own_ws : s->ws_acquire ? s->ws_acquire(s->ws_user) : 0;
  if (!ws || ws->maxT < s->maxT || ws->maxF < s->maxF) return -7;
  s->w = ws->b;
  long ldT0 = s->ldT, ldF0 = s->ldF;
  s->ldT = ws->ldT;
  s->ldF = ws->ldF;
  double tp = prof_t();
  int erc = run_encoder(s, phone, tone, lang, T, sid);
  PROF(PR_ENC, tp);
  long F = 0;
  for (int t = 0; t < T; t++) F += s->w.dur[t] < 0 ? 0 : s->w.dur[t];
  if (!erc && F > 0 && F <= s->maxF) erc = run_decoder(s, T, (int)F, threads) ? -5 : 0;
  else if (!erc) erc = -4;
  s->ldT = ldT0;
  s->ldF = ldF0;
  memset(&s->w, 0, sizeof s->w);
  if (!s->own_ws && s->ws_release) s->ws_release(s->ws_user, ws);
  if (erc) return erc;
  double t_mel = pt_now();
  s->vcols_comp = s->vcols_use = 0;
  int full_tap = s->tap && F <= s->chunkF;
  long a = 0, cf = s->firstF;
  double t_first = 0;
  int rc = 0;
  if (full_tap) cf = F;
  while (a < F) {
    long b = a + cf < F ? a + cf : F;
    vplan p;
    plan_chunk(&p, a, b);
    if (voc_chunk(s, &p, (int)F, a == 0, full_tap)) return -6;
    if (!t_first) t_first = pt_now();
    if (cb && cb(user, s->wav.base, (int)(p.S1 - p.S0))) { rc = 1; break; }
    a = b;
    cf = cf * 2 < s->chunkF ? cf * 2 : s->chunkF;
  }
  if (info) {
    info->T = T;
    info->F = (int)F;
    info->samples = F * V_HOP;
    info->t_frontend_s = t_mel - t0;
    info->t_first_chunk_s = t_first - t0;
    info->t_total_s = pt_now() - t0;
    info->vocoder_cols_computed = s->vcols_comp;
    info->vocoder_cols_useful = s->vcols_use;
  }
  return rc;
}
