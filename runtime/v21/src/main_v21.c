/* main_v21.c - single-stream v2.1 driver: run one golden case through the graph and report parity.
 *
 *   ./pt_v21 --container build/v21.ptk --case tests/golden_v21/T81 [--checkpoints DIR] [--quiet]
 *
 * Reads the case's graph-input .npy files, runs the graph, compares the waveform against the
 * case's wav.npy (SNDR in dB, the M2 gate is >= 60 dB), and reports per-op timings.
 */
#include "pt_v21.h"
#include "pt_graph_tab.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
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

/* One request's work, extracted verbatim from main so --serve-fifo can repeat it on an
 * already-warm context: the container record pass, CUDA context, weight upload and clock ramp
 * belong to main, not to this. Returns the SNDR in dB, or -1 when the case could not be loaded
 * or run.
 *
 * Feed lifetime: load_feed mallocs each payload and the engine copies them into the graph during
 * the run, so the previous request's set is freed at the START of the next one -- every pointer is
 * live for the whole run, and nothing accumulates across requests. The waveform is owned by the
 * engine (pt_v21_run_graph frees c->wav on the next run and at close), so it is never freed here. */
static double run_case(pt_ctx *c, const char *dir, const char *out, void **feedp) {
  static const char *feeds_names[] = {"x", "tone", "lang", "x_lengths", "sid", "noise_scale",
                                      "length_scale"};
  for (int i = 0; i < 7; i++) { free(feedp[i]); feedp[i] = 0; }
  pt_tv feeds[8];
  memset(feeds, 0, sizeof(feeds));
  for (int i = 0; i < 7; i++)
    if (load_feed(dir, feeds_names[i], &feeds[i])) return -1;
  for (int i = 0; i < 7; i++) feedp[i] = feeds[i].p;
  float *wav = 0;
  int64_t wn = 0;
  if (pt_v21_run(c, feeds, &wav, &wn)) return -1;
  /* parity vs the golden waveform */
  void *gp;
  int dt, nd;
  int64_t gd[PT_V21_MAX_RANK], gn;
  char path[512];
  snprintf(path, sizeof(path), "%s/wav.npy", dir);
  if (pt_npy_load_v21(path, &gp, &dt, &nd, gd, &gn)) {
    fprintf(stderr, "main_v21: cannot load %s\n", path);
    free(gp);
    return -1;
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
  if (out) {
    int64_t d3[3] = {1, 1, wn};
    pt_npy_save_f32_v21(out, wav, 3, d3);
    printf("  wrote %s\n", out);
  }
  free(gp);
  return sndr;
}

/* One request line from the serve fifo, read the way a per-request client requires: never block
   on a writer, reopen when the last one closes (a non-blocking fifo read gives EAGAIN while a
   writer is idle and 0 when none is open). Returns 0 only when the fifo cannot be opened. */
static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Clock ramp on arrival (conversation latency, invisible to a back-to-back benchmark).
 * nvhost_podgov parks the GPU within ~2 s of idle: on this board a T81 request costs 0.48 s at
 * the 921.6 MHz max and 1.0-1.7 s at the parked 76.8 MHz (measured, PULSAR-ASM .auto/combined
 * run #277). In a conversation the gap between the ASR finishing and the reply text being ready
 * is exactly that idle window -- so the stage the user actually waits for is the one that runs
 * on a parked clock. When a request arrives parked, hold the SM busy on the same ALU kernel the
 * startup ramp uses until the governor reaches max (or a budget expires), then serve.
 * When the clock is already at max -- back-to-back traffic, and what the harness measures -- this
 * returns without launching anything, so it cannot perturb a measured turn. The burst is a
 * private stream over a private buffer (pt_v21_cuda_keepwarm); no graph data is touched and the
 * waveform is bit-identical either way.
 * PT_V21_RAMP_MS=0 disables it. The startup ramp uses 64000 iters and takes ~287 ms from parked,
 * so 16000 per burst polls the clock four times per that ramp's worth of work. */
static void serve_ramp(void) {
  const char *e = getenv("PT_V21_RAMP_MS");
  double budget = e ? atof(e) : 500.0;
  if (budget <= 0.0) return;
  int iters = getenv("PT_V21_RAMP_ITERS") ? atoi(getenv("PT_V21_RAMP_ITERS")) : 16000;
  if (iters <= 0) return;
  long mx = pt_v21_gpu_max_freq(), cur = pt_v21_gpu_cur_freq();
  if (mx <= 0 || cur <= 0 || cur >= mx) return;   /* at max (or unreadable): nothing to do */
  long c0 = cur;
  double t0 = now_ms();
  int nb = 0;
  while (cur < mx && now_ms() - t0 < budget) {
    if (pt_v21_cuda_keepwarm(iters)) break;
    nb++;
    cur = pt_v21_gpu_cur_freq();
  }
  fprintf(stderr, "pt_v21 serve: clock parked at request, ramped %ld -> %ld MHz in %d bursts (%.0f ms)\n",
          c0 / 1000000, cur / 1000000, nb, now_ms() - t0);
}

static int serve_read(const char *path, char *out, size_t cap) {
  int fd = open(path, O_RDONLY | O_NONBLOCK);
  if (fd < 0) { fprintf(stderr, "main_v21: cannot open %s (%s)\n", path, strerror(errno)); return 0; }
  size_t n = 0; int idle = 0;
  for (;;) {
    char ch;
    ssize_t r = read(fd, &ch, 1);
    if (r == 1) {
      idle = 0;
      if (ch == '\n') break;
      if (n + 1 < cap) out[n++] = ch;
      continue;
    }
    if (r < 0 && errno != EAGAIN && errno != EINTR) { close(fd); return 0; }
    if (n && ++idle > 5) break;                      /* writer done without a trailing newline */
    if (!n) {                                          /* no writer: reopen and poll */
      close(fd);
      fd = open(path, O_RDONLY | O_NONBLOCK);
      if (fd < 0) return 0;
    }
    struct timespec ts = { 0, 20000000 };
    nanosleep(&ts, 0);
  }
  out[n] = 0;
  close(fd);
  return 1;
}

int main(int argc, char **argv) {
  const char *container = "build/v21.ptk";
  const char *dir = "tests/golden_v21/T81";
  const char *ck = 0;
  const char *ckdir = 0;
  const char *shapes = 0;
  int trace_idx = -1;
  int use_cuda = 1;   /* CUDA is the default; --no-cuda / --cpu selects the portable CPU path */
  int s_win = 0, s_ctx = 0;   /* --stream WIN CTX: overlap-save decoder verification */
  const char *out = 0;
  const char *serve_fifo = 0;   /* --serve-fifo PATH: stay up, one case per request line */
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--container") && i + 1 < argc) container = argv[++i];
    else if (!strcmp(argv[i], "--case") && i + 1 < argc) dir = argv[++i];
    else if (!strcmp(argv[i], "--serve-fifo") && i + 1 < argc) serve_fifo = argv[++i];
    else if (!strcmp(argv[i], "--checkpoints") && i + 1 < argc) ck = argv[++i];
    else if (!strcmp(argv[i], "--ckdir") && i + 1 < argc) ckdir = argv[++i];
    else if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace_idx = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--trace-matmul")) trace_idx = -2;
    else if (!strcmp(argv[i], "--shapes") && i + 1 < argc) shapes = argv[++i];
    else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "--cuda")) use_cuda = 1;   /* accepted for compatibility */
    else if (!strcmp(argv[i], "--stream") && i + 2 < argc) { s_win = atoi(argv[++i]); s_ctx = atoi(argv[++i]); }
    else if (!strcmp(argv[i], "--no-cuda") || !strcmp(argv[i], "--cpu")) use_cuda = 0;
  }
  if (use_cuda)
    pt_v21_cuda_prewarm();   /* context + clock ramp overlap the container record pass */
  pt_ctx *c = pt_v21_open(container);
  if (!c) return 2;
  if (use_cuda && pt_v21_cuda_init(c, 1)) {
    fprintf(stderr, "pt_v21: cuda init failed\n");
    return 2;
  }
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
  if (serve_fifo) {
    /* Resident TTS. A one-shot process pays ~0.4 s before it computes anything (container record
       pass, CUDA context, weight upload, clock ramp -- the same startup the HTTP server documents);
       in a conversation that tax is per turn. Here it is paid once at boot and each turn is one
       request line: "<case dir>" or "<case dir> <out.npy>", answered by "PT_DONE <n> rc=<rc>
       sndr=<dB>" on stdout, which is what the client waits for. The computation is run_case(),
       the same code the one-shot path runs, and the harness gates the two arms' waveforms against
       each other byte for byte. */
    void *feedp[7] = { 0, 0, 0, 0, 0, 0, 0 };
    for (int req = 0; ; req++) {
      char rl[1024];
      printf("PT_AWAIT %d\n", req);
      fflush(stdout);
      if (!serve_read(serve_fifo, rl, sizeof rl)) break;
      if (!rl[0]) { req--; continue; }
      if (!strcmp(rl, "Q")) break;
      if (use_cuda) serve_ramp();   /* no-op unless the governor parked the clock */
      char *sp = strchr(rl, ' ');
      if (sp) *sp++ = 0;
      double sndr = run_case(c, rl, (sp && *sp) ? sp : 0, feedp);
      printf("PT_DONE %d rc=%d sndr=%.2f\n", req, (sndr >= 0.0 && sndr >= 60.0) ? 0 : 1, sndr);
      fflush(stdout);
      if (sndr < 0.0) break;      /* a case that cannot be loaded or run is not a transient fault */
    }
    for (int i = 0; i < 7; i++) free(feedp[i]);
    pt_v21_close(c);
    return 0;
  }
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
  if (s_win > 0) {
    /* overlap-save decoder: encoder once, then sliding windows; emit each window's interior */
    int64_t T = 0;
    int hop = 0;
    if (pt_v21_stream_open(c, feeds, &T, &hop)) {
      fprintf(stderr, "main_v21: stream open failed\n");
      return 4;
    }
    size_t cap = (size_t) (T + 2 * s_win) * (size_t) (hop > 0 ? hop : 256);
    float *out = (float *) malloc(cap * 4);
    int64_t olen = 0;
    int64_t step = s_win - 2 * s_ctx;
    if (step < 1) step = 1;
    for (int64_t a = 0;; a += step) {
      int64_t win_eff = s_win;
      if (a + win_eff > T) win_eff = T - a;
      int first = (a == 0), last = (a + win_eff >= T);
      float *pcm = 0;
      int64_t n = 0;
      if (pt_v21_stream_chunk(c, a, win_eff, &pcm, &n)) { fprintf(stderr, "chunk failed\n"); return 4; }
      int64_t lo = first ? 0 : (int64_t) s_ctx * hop;
      int64_t hi = last ? n : (int64_t) (win_eff - s_ctx) * hop;
      if (hi > n) hi = n;
      if (getenv("PT_V21_STREAMLOG"))
        fprintf(stderr, "win a=%lld win=%lld n=%lld lo=%lld hi=%lld olen=%lld\n",
                (long long) a, (long long) win_eff, (long long) n, (long long) lo, (long long) hi,
                (long long) olen);
      if (hi > lo && olen + (hi - lo) <= (int64_t) cap) {
        memcpy(out + olen, pcm + lo, (size_t) (hi - lo) * 4);
        olen += hi - lo;
      }
      if (last) break;
    }
    pt_v21_stream_close(c);
    wav = out;
    wn = olen;
    fprintf(stderr, "stream: T=%lld hop=%d win=%d ctx=%d -> %lld samples\n", (long long) T, hop,
            s_win, s_ctx, (long long) olen);
  } else if (pt_v21_run(c, feeds, &wav, &wn)) {
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
    if (c->prof_node_on && c->prof_node) {
    for (int rep = 0; rep < 20; rep++) {
      int best = -1;
      double bv = 0.0;
      for (int n2 = 0; n2 < PT_N_NODE; n2++)
        if (c->prof_node[n2] > bv) {
          bv = c->prof_node[n2];
          best = n2;
        }
      if (best < 0) break;
      c->prof_node[best] = -1.0;
      printf("  node %d %s: %.1f ms\n", best, pt_graph_tnames[pt_ndefs[best].out[0]], 1000.0 * bv);
    }
  }
printf("  conv=%ld convT=%ld matmul=%ld  conv %.1f ms  convT %.1f ms  matmul %.1f ms  total %.1f ms\n",
         pt_v21_stats_conv(c), pt_v21_stats_convT(c), pt_v21_stats_matmul(c),
         pt_v21_stats_ms(c, 0), pt_v21_stats_ms(c, 1), pt_v21_stats_ms(c, 2), pt_v21_stats_ms(c, 3));
  if (getenv("PT_V21_EKIND")) {
    char rep[512];
    fprintf(stderr, "  host op kinds (ms):%s\n", pt_v21_ekind_report(c, rep, sizeof(rep)));
  }
  if (use_cuda)
    fprintf(stderr, "  gpu split: alloc %.1f ms  copy %.1f ms  kernel+sync %.1f ms  free %.1f ms\n",
            pt_v21_stat_ms(c, 0), pt_v21_stat_ms(c, 1), pt_v21_stat_ms(c, 2), pt_v21_stat_ms(c, 3));
  if (out) {
    int64_t d3[3] = {1, 1, wn};
    pt_npy_save_f32_v21(out, wav, 3, d3);
    printf("  wrote %s\n", out);
  }
  pt_v21_close(c);
  return sndr >= 60.0 ? 0 : 1;
}
