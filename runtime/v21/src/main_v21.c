/* main_v21.c - single-stream v2.1 driver: run one golden case through the graph and report parity.
 *
 *   ./pt_v21 --container build/v21.ptk --case tests/golden_v21/T81 [--checkpoints DIR] [--quiet]
 *
 * Reads the case's graph-input .npy files, runs the graph, compares the waveform against the
 * case's wav.npy (SNDR in dB, the M2 gate is >= 60 dB), and reports per-op timings.
 */
#include "pt_v21.h"
#include "pt_graph_tab.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int load_feed(const char *dir, const char *name, pt_tv *t) {
  char path[512];
  snprintf(path, sizeof(path), "%s/%s.npy", dir, name);
  void *p;
  int dt, nd;
  int64_t d[PT_V21_MAX_RANK], n;
  if (pt_npy_load_v21(path, &p, &dt, &nd, d, &n)) {
    fprintf(stderr, "main_v21: cannot load %s\n", path);
    return -1;
  }
  t->dt = dt;
  t->ndim = nd;
  for (int i = 0; i < PT_V21_MAX_RANK; i++) t->d[i] = d[i];
  t->n = n;
  t->p = p;
  t->alive = 1;
  t->dev = 0;
  return 0;
}

int main(int argc, char **argv) {
  const char *container = "build/v21.ptk";
  const char *dir = "tests/golden_v21/T81";
  const char *ck = 0;
  const char *ckdir = 0;
  const char *shapes = 0;
  int trace_idx = -1;
  const char *out = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--container") && i + 1 < argc) container = argv[++i];
    else if (!strcmp(argv[i], "--case") && i + 1 < argc) dir = argv[++i];
    else if (!strcmp(argv[i], "--checkpoints") && i + 1 < argc) ck = argv[++i];
    else if (!strcmp(argv[i], "--ckdir") && i + 1 < argc) ckdir = argv[++i];
    else if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace_idx = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--trace-matmul")) trace_idx = -2;
    else if (!strcmp(argv[i], "--shapes") && i + 1 < argc) shapes = argv[++i];
    else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
  }
  pt_ctx *c = pt_v21_open(container);
  if (!c) return 2;
  if (getenv("PT_V21_CHECK_CONST")) {
    /* container spot-check: print the first payload value of the named container tensors */
    for (int i = 0; i < PT_N_CONT; i++) {
      if (!strcmp(pt_graph_ctab[i], "const.0007") || !strcmp(pt_graph_ctab[i], "const.0076") ||
          !strcmp(pt_graph_ctab[i], "const.0255") || !strcmp(pt_graph_ctab[i], "dtype.table")) {
        const float *p = pt_v21_cont(c, i);
        printf("ctab[%d] %s -> %p first=%g\n", i, pt_graph_ctab[i], (void *) p,
               p ? (double) p[0] : 0.0);
      }
    }
  }
  if (trace_idx >= 0 || trace_idx == -2) pt_v21_trace(trace_idx);
  if (shapes) {
    FILE *f = fopen(shapes, "w");
    if (!f) {
      fprintf(stderr, "main_v21: cannot write %s\n", shapes);
      return 6;
    }
    pt_v21_shape_log(f);
  }
  if (ck && ckdir) {
    /* checkpoints are opt-in via a names file: one graph tensor name per line */
    static int ids[4096];
    int n = 0;
    FILE *f = fopen(ck, "r");
    if (f) {
      char line[512];
      while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\r\n")] = 0;
        for (int q = 0; q < PT_N_TENSOR && n < 4096; q++)
          if (!strcmp(pt_graph_tnames[q], line)) ids[n++] = q;
      }
      fclose(f);
    }
    pt_v21_checkpoints(c, ckdir, ids, n);
    printf("checkpoints: %d tensors -> %s\n", n, ckdir);
  }
  static const char *feeds_names[] = {"x", "tone", "lang", "x_lengths", "sid", "noise_scale",
                                      "length_scale"};
  pt_tv feeds[8];
  memset(feeds, 0, sizeof(feeds));
  for (int i = 0; i < 7; i++)
    if (load_feed(dir, feeds_names[i], &feeds[i])) return 3;
  float *wav = 0;
  int64_t wn = 0;
  if (pt_v21_run(c, feeds, &wav, &wn)) {
    fprintf(stderr, "main_v21: run failed\n");
    return 4;
  }
  /* parity vs the golden waveform */
  void *gp;
  int dt, nd;
  int64_t gd[PT_V21_MAX_RANK], gn;
  char path[512];
  snprintf(path, sizeof(path), "%s/wav.npy", dir);
  if (pt_npy_load_v21(path, &gp, &dt, &nd, gd, &gn)) {
    fprintf(stderr, "main_v21: cannot load %s\n", path);
    return 5;
  }
  const float *g = (const float *) gp;
  int64_t n = wn < gn ? wn : gn;
  double se = 0.0, sp = 0.0, mx = 0.0;
  for (int64_t i = 0; i < n; i++) {
    double d = (double) wav[i] - (double) g[i];
    se += d * d;
    sp += (double) g[i] * (double) g[i];
    if (fabs(d) > mx) mx = fabs(d);
  }
  double sndr = se > 0 ? 10.0 * log10(sp / se) : 1e9;
  printf("case %s: L=%lld golden=%lld  max|d|=%.3e  SNDR=%.2f dB\n", dir, (long long) wn,
         (long long) gn, mx, sndr);
  printf("  conv=%ld convT=%ld matmul=%ld  conv %.1f ms  convT %.1f ms  matmul %.1f ms  total %.1f ms\n",
         pt_v21_stats_conv(c), pt_v21_stats_convT(c), pt_v21_stats_matmul(c),
         pt_v21_stats_ms(c, 0), pt_v21_stats_ms(c, 1), pt_v21_stats_ms(c, 2), pt_v21_stats_ms(c, 3));
  if (out) {
    int64_t d3[3] = {1, 1, wn};
    pt_npy_save_f32_v21(out, wav, 3, d3);
    printf("  wrote %s\n", out);
  }
  pt_v21_close(c);
  return sndr >= 60.0 ? 0 : 1;
}
