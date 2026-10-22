/* test_e2e.c - end-to-end parity of the C runtime vs onnxruntime golden vectors (tests/golden/<case>/).
 * Layer-by-layer (tap) comparison in single-chunk mode, then streamed (chunked) waveform comparison.
 * Usage: test_e2e <model.ptk> <golden_dir> <case>... */
#include "pt_v1b8k.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *dir; int fails; double worst_rel; } tapctx;

static int load(pt_npy *a, const char *dir, const char *name) {
  char p[512];
  snprintf(p, sizeof p, "%s/%s.npy", dir, name);
  if (pt_npy_load(a, p)) { fprintf(stderr, "cannot load %s\n", p); return -1; }
  return 0;
}

/* golden name + whether golden is time-major [.., n, C] */
static int map_tap(const char *nm, char *g, int *tmajor) {
  int i;
  *tmajor = 1;
  if (!strcmp(nm, "enc_emb")) strcpy(g, "enc_Mul");
  else if (sscanf(nm, "enc_blk%d", &i) == 1) sprintf(g, "enc_encoder.%d_Mul_3", i);
  else if (!strcmp(nm, "conditioned")) strcpy(g, "conditioned");
  else if (!strcmp(nm, "pitch")) strcpy(g, "pitch");
  else if (!strcmp(nm, "logdur")) strcpy(g, "enc_Squeeze");
  else if (!strcmp(nm, "reg_frame_meta")) strcpy(g, "reg_frame_meta");
  else if (!strcmp(nm, "dec_in")) strcpy(g, "dec_Add_4");
  else if (sscanf(nm, "dec_blk%d", &i) == 1) sprintf(g, "dec_decoder.%d_Mul_3", i);
  else if (!strcmp(nm, "dec_gru")) strcpy(g, "dec_frame_gru_Transpose_2");
  else if (!strcmp(nm, "mel_pre")) { strcpy(g, "dec_Transpose_2"); *tmajor = 0; }
  else if (!strcmp(nm, "mel")) { strcpy(g, "mel"); *tmajor = 0; }
  else if (sscanf(nm, "voc_ups%d", &i) == 1) { sprintf(g, "voc_ups.%d_ConvTranspose", i); *tmajor = 0; }
  else if (sscanf(nm, "voc_stage%d", &i) == 1) {
    if (i == 0) strcpy(g, "voc_Div"); else sprintf(g, "voc_Div_%d", i);
    *tmajor = 0;
  } else return -1;
  return 0;
}

static void tap(void *u, const char *nm, const float *x, int C, long ld, long n) {
  tapctx *t = u;
  char g[128];
  int tm;
  if (map_tap(nm, g, &tm)) return;
  pt_npy a;
  if (load(&a, t->dir, g)) { t->fails++; return; }
  const float *r = a.data;
  long nn = tm ? a.dim[a.ndim - 2] : a.dim[a.ndim - 1];
  long CC = tm ? a.dim[a.ndim - 1] : a.dim[a.ndim - 2];
  if (!strcmp(g, "enc_Squeeze")) { nn = a.dim[1]; CC = 1; tm = 1; }
  if (nn != n || CC != C) { printf("  %-16s SHAPE MISMATCH golden C=%ld n=%ld vs C=%d n=%ld\n", nm, CC, nn, C, n); t->fails++; pt_npy_free(&a); return; }
  double mx = 0, ma = 0, sum = 0;
  for (int c = 0; c < C; c++)
    for (long i = 0; i < n; i++) {
      float ref = tm ? r[i * C + c] : r[(long)c * n + i];
      double d = fabs((double)x[c * ld + i] - ref);
      if (!(d <= mx)) mx = d;   /* catches NaN */
      if (fabs(ref) > ma) ma = fabs(ref);
      sum += d;
    }
  double rel = mx / (ma > 0 ? ma : 1);
  if (rel > t->worst_rel || rel != rel) t->worst_rel = rel;
  printf("  %-16s C=%3d n=%6ld  max|ref|=%9.4f  maxabs=%.3e  meanabs=%.3e  rel=%.3e\n", nm, C, n, ma, mx,
         sum / ((double)C * n), rel);
  pt_npy_free(&a);
}

typedef struct { float *buf; long n, cap; } pcm_acc;
static int on_pcm(void *u, const float *pcm, int n) {
  pcm_acc *a = u;
  if (a->n + n > a->cap) return 1;
  memcpy(a->buf + a->n, pcm, n * sizeof(float));
  a->n += n;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s model.ptk golden_dir case...\n", argv[0]); return 2; }
  pt_v1b8k_model *m = pt_v1b8k_load(argv[1]);
  if (!m) return 1;
  int bad = 0;
  printf("%-16s %6s %6s %10s %10s %10s %10s %s\n", "case", "F", "samp", "maxabs", "meanabs", "rel_max", "rel_l2", "status");
  for (int ci = 3; ci < argc; ci++) {
    char dir[512];
    snprintf(dir, sizeof dir, "%s/%s", argv[2], argv[ci]);
    pt_npy ph, tn, lg, sid, wav;
    if (load(&ph, dir, "phone") || load(&tn, dir, "tone") || load(&lg, dir, "lang") || load(&sid, dir, "sid") ||
        load(&wav, dir, "wav")) return 1;
    int T = (int)ph.dim[1];
    int32_t *p = malloc(T * 4), *t = malloc(T * 4), *l = malloc(T * 4);
    for (int i = 0; i < T; i++) {
      p[i] = (int32_t)((int64_t *)ph.data)[i];
      t[i] = (int32_t)((int64_t *)tn.data)[i];
      l[i] = (int32_t)((int64_t *)lg.data)[i];
    }
    int s_id = (int)((int64_t *)sid.data)[0];
    long nref = wav.n;
    const float *ref = wav.data;
    /* (1) layer-by-layer: one chunk covering the whole utterance */
    printf("[%s] layer taps (single chunk):\n", argv[ci]);
    pt_v1b8k_stream *s1 = pt_v1b8k_stream_new(m, 512, 1200, 1200);
    tapctx tc = {dir, 0, 0};
    pt_v1b8k_set_tap(s1, tap, &tc);
    pcm_acc A1 = {malloc(sizeof(float) * (nref + 4096)), 0, nref + 4096};
    pt_synth_info inf;
    int rc = pt_v1b8k_synth(s1, p, t, l, T, s_id, on_pcm, &A1, &inf);
    pt_v1b8k_stream_free(s1);
    /* (2) streaming: 3-frame first chunk, doubling to 32-frame chunks */
    pt_v1b8k_stream *s2 = pt_v1b8k_stream_new(m, 512, 1200, 32);
    pcm_acc A2 = {malloc(sizeof(float) * (nref + 4096)), 0, nref + 4096};
    int rc2 = pt_v1b8k_synth(s2, p, t, l, T, s_id, on_pcm, &A2, &inf);
    pt_v1b8k_stream_free(s2);
    for (int k = 0; k < 2; k++) {
      pcm_acc *A = k ? &A2 : &A1;
      int r = k ? rc2 : rc;
      double mx = 0, ma = 0, se = 0, sr = 0, sa = 0;
      long n = A->n < nref ? A->n : nref;
      for (long i = 0; i < n; i++) {
        double d = fabs((double)A->buf[i] - ref[i]);
        if (!(d <= mx)) mx = d;
        if (fabs(ref[i]) > ma) ma = fabs(ref[i]);
        se += d * d; sr += (double)ref[i] * ref[i]; sa += d;
      }
      double rel = mx / ma, rl2 = sqrt(se / sr);
      int ok = r == 0 && A->n == nref && rel < 1e-3 && rl2 < 1e-3;
      bad |= !ok || (k == 0 && tc.fails);
      printf("%-16s %6d %6ld %10.3e %10.3e %10.3e %10.3e %s (%s)\n", argv[ci], inf.F, A->n, mx, sa / n, rel, rl2,
             ok ? "PASS" : "FAIL", k ? "streamed 3->32 frame chunks" : "single chunk");
    }
    {
      double dd = 0;
      long nn = A1.n < A2.n ? A1.n : A2.n;
      for (long i = 0; i < nn; i++) { double d = fabs((double)A1.buf[i] - A2.buf[i]); if (!(d <= dd)) dd = d; }
      printf("%-16s streamed vs single-chunk: max |diff| = %.3e over %ld samples%s\n", argv[ci], dd, nn,
             dd == 0 ? " (bit-identical)" : "");
    }
    free(A1.buf); free(A2.buf); free(p); free(t); free(l);
    pt_npy_free(&ph); pt_npy_free(&tn); pt_npy_free(&lg); pt_npy_free(&sid); pt_npy_free(&wav);
  }
  pt_v1b8k_free(m);
  printf(bad ? "E2E: FAIL\n" : "E2E: PASS\n");
  return bad;
}
