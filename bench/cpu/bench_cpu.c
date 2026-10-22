/* bench_cpu.c - PrimeTTS v1b_8k CPU runtime benchmark (run on the Jetson Nano).
 *
 *   bench_cpu <v1b8k.ptk> <requests.txt> [sections...]   sections: kernels single mt conc concN admit (default all)
 *
 * kernels : GFLOP/s of the conv path for the model's real layer shapes (4x16 vs 8x12 micro-kernel)
 * single  : one stream, one thread: RTF, first-chunk latency, mel-ready time, chunk recompute overhead
 * mt      : one stream, 4 threads (intra-op split): latency mode
 * conc    : N = 1..8 concurrent streams (pinned round-robin to the 4 cores, 3 requests per stream slot,
 *           all arriving at t=0, no admission throttling): per-request RTF, aggregate audio-s per wall-s,
 *           first-chunk latency, worst playback lateness (= pre-roll a client needs to never starve)
 * admit   : 16 requests arriving at once through the admission policy (capacity from `single`)
 */
#define _GNU_SOURCE
#include "pt_v1b8k.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

typedef struct { char name[32]; int sid, T; int32_t *p, *t, *l; } req_in;
static req_in R[64];
static int NR = 0;
static double g_single_rtf = 0;

static long peak_rss_kb(void) {
  FILE *f = fopen("/proc/self/status", "r");
  char line[256];
  long v = -1;
  while (f && fgets(line, sizeof line, f))
    if (!strncmp(line, "VmHWM:", 6)) v = atol(line + 6);
  if (f) fclose(f);
  return v;
}

static int load_requests(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) return -1;
  while (NR < 64 && fscanf(f, "%31s %d %d", R[NR].name, &R[NR].sid, &R[NR].T) == 3) {
    int T = R[NR].T;
    R[NR].p = malloc(T * 4); R[NR].t = malloc(T * 4); R[NR].l = malloc(T * 4);
    for (int i = 0; i < T; i++) if (fscanf(f, "%d", &R[NR].p[i]) != 1) return -1;
    for (int i = 0; i < T; i++) if (fscanf(f, "%d", &R[NR].t[i]) != 1) return -1;
    for (int i = 0; i < T; i++) if (fscanf(f, "%d", &R[NR].l[i]) != 1) return -1;
    NR++;
  }
  fclose(f);
  return NR;
}

/* ---------------------------------------------------------------- kernels */
static float *rnd(long n) {
  float *p = aligned_alloc(64, ((n * 4 + 63) / 64) * 64 + 1024);
  unsigned s = 1;
  for (long i = 0; i < n + 256; i++) { s = s * 1103515245u + 12345u; p[i] = ((s >> 9) & 0xffff) / 65536.0f - 0.5f; }
  return p;
}

static double time_conv(int Cout, int Cin, int K, int dil, long n, int act, int threads) {
  int C4 = (Cout + 3) / 4 * 4;
  long ldx = n + (long)(K - 1) * dil + 64, ldy = n + 64;
  float *X = rnd((long)Cin * ldx), *Wp = rnd((long)C4 * Cin * K), *B = rnd(C4), *Y = rnd((long)C4 * ldy);
  float *a = rnd(C4), *ia = rnd(C4);
  for (int i = 0; i < C4; i++) { a[i] = 1.0f; ia[i] = 1.0f; }
  pt_conv_w c = {Wp, B, Cout, C4, Cin, K, dil};
  pt_epi e = {act, a, ia, 1.0f, 0, 0};
  pt_pool_set_local(threads);
  pt_conv(&c, X, ldx, Y, ldy, n, act ? &e : 0);
  int reps = 0;
  double t0 = pt_now(), t1;
  do { pt_conv(&c, X, ldx, Y, ldy, n, act ? &e : 0); reps++; t1 = pt_now(); } while (t1 - t0 < 0.3);
  pt_pool_set_local(1);
  free(X); free(Wp); free(B); free(Y); free(a); free(ia);
  return 2.0 * Cout * Cin * K * n * reps / (t1 - t0) / 1e9;
}

static void bench_kernels(void) {
  struct { const char *what; int Cout, Cin, K, dil; long n; int act; } L[] = {
    {"voc stage0 resblock conv 48x48 K11 d5 +snake", 48, 48, 11, 5, 1200, PT_ACT_SNAKE},
    {"voc stage0 resblock conv 48x48 K3 d1", 48, 48, 3, 1, 1200, PT_ACT_NONE},
    {"voc stage1 resblock conv 24x24 K7 d3 +snake", 24, 24, 7, 3, 4400, PT_ACT_SNAKE},
    {"voc stage2 resblock conv 12x12 K11 d1", 12, 12, 11, 1, 8400, PT_ACT_NONE},
    {"voc stage3 resblock conv 6x6 K7 d3 +snake", 6, 6, 7, 3, 16500, PT_ACT_SNAKE},
    {"voc ups0 (8 phases) 384x96 K2", 384, 96, 2, 1, 150, PT_ACT_NONE},
    {"voc conv_pre 96x80 K7 +snake", 96, 80, 7, 1, 150, PT_ACT_SNAKE},
    {"dec ff0 504x168 (linear) +silu", 504, 168, 1, 1, 500, PT_ACT_SILU},
    {"dec ff3 168x504 (linear)", 168, 504, 1, 1, 500, PT_ACT_NONE},
    {"dec gru input proj 504x168", 504, 168, 1, 1, 500, PT_ACT_NONE},
    {"dec postnet 168x168 K5 +tanh", 168, 168, 5, 1, 500, PT_ACT_TANH},
    {"enc ff0 672x168 (linear, T=40)", 672, 168, 1, 1, 40, PT_ACT_SILU},
  };
  int n = sizeof L / sizeof L[0];
  printf("\n## kernels: conv path GFLOP/s (fp32, epilogue included)\n\n");
  printf("_shape list is the v1b_8k vocoder/decoder layers (fixed table); for other variants use `single`/`mt`/`concN`_\n\n");
  printf("| layer shape | n cols | 4x16, 1 thr | 8x12, 1 thr, no packing | 8x12 + packing K=1 only, 1 thr | 8x12 + packing incl. K>1 (default), 4 thr | **8x12 + packing incl. K>1 (default), 1 thr** |\n|---|---|---|---|---|---|---|\n");
  for (int i = 0; i < n; i++) {
    pt_conv_set_kernel(0);
    pt_conv_set_pack(0);
    double g4 = time_conv(L[i].Cout, L[i].Cin, L[i].K, L[i].dil, L[i].n, L[i].act, 1);
    pt_conv_set_kernel(1);
    double g8 = time_conv(L[i].Cout, L[i].Cin, L[i].K, L[i].dil, L[i].n, L[i].act, 1);
    pt_conv_set_pack(1);
    double g8p = time_conv(L[i].Cout, L[i].Cin, L[i].K, L[i].dil, L[i].n, L[i].act, 1);
    pt_conv_set_pack(2);
    double g8t = time_conv(L[i].Cout, L[i].Cin, L[i].K, L[i].dil, L[i].n, L[i].act, 4);
    pt_conv_set_pack(2);
    double g8p2 = time_conv(L[i].Cout, L[i].Cin, L[i].K, L[i].dil, L[i].n, L[i].act, 1);
    printf("| %s | %ld | %.2f | %.2f | %.2f | %.2f | **%.2f** |\n", L[i].what, L[i].n, g4, g8, g8p, g8t, g8p2);
  }
  /* raw micro-kernel peak: 168 x 168 K1 equivalent, all in L1 */
  long nk = 256, nt = 8;
  float *A = rnd(nk * 8), *X = rnd(nk * 16 + 4096), *Y = rnd(8 * 256), *bias = rnd(8);
  long *off = malloc(nk * 8);
  for (long j = 0; j < nk; j++) off[j] = (j % 8) * 16 * 4;  /* L1-resident operands */
  double t0 = pt_now(), t1;
  long reps = 0;
  do { pt_k4x16(Y, 256 * 4, A, X, off, nk, nt, bias); reps++; t1 = pt_now(); } while (t1 - t0 < 0.3);
  double g4 = 2.0 * 4 * 16 * nt * nk * reps / (t1 - t0) / 1e9;
  reps = 0; t0 = pt_now();
  do { pt_k8x12(Y, 256 * 4, A, X, off, nk, nt, bias); reps++; t1 = pt_now(); } while (t1 - t0 < 0.3);
  double g8 = 2.0 * 8 * 12 * nt * nk * reps / (t1 - t0) / 1e9;
  printf("\nmicro-kernel, operands L1-resident (nk=256): k4x16 %.2f GFLOP/s, k8x12 %.2f GFLOP/s "
         "(A57 peak 8 flop/cycle = 11.8 GFLOP/s at 1.479 GHz)\n", g4, g8);
  /* activations */
  long m = 1 << 14;
  float *v = rnd(m);
  for (long i = 0; i < m; i++) v[i] *= 4;
  struct { const char *nm; int k; } acts[] = {{"snake", 0}, {"silu", 1}, {"tanh", 2}};
  printf("activations (16K elements, in L1): ");
  for (int k = 0; k < 3; k++) {
    reps = 0; t0 = pt_now();
    do {
      if (k == 0) pt_snake_row(v, m, 0.9f, 1.1f);
      else if (k == 1) pt_silu_row(v, m);
      else pt_tanh_row(v, m);
      reps++; t1 = pt_now();
      if (k == 1) for (long i = 0; i < 64; i++) v[i] = 1.0f;
    } while (t1 - t0 < 0.2);
    printf("%s %.0f Melem/s  ", acts[k].nm, (double)m * reps / (t1 - t0) / 1e6);
  }
  printf("\n");
  free(A); free(X); free(Y); free(bias); free(off); free(v);
}

/* ---------------------------------------------------------------- single stream */
static int null_cb(void *u, const float *p, int n) { (void)u; (void)p; (void)n; return 0; }

static void bench_single(const pt_v1b8k_model *m, int threads) {
  pt_pool_set_local(threads);
  pt_v1b8k_stream *s = pt_v1b8k_stream_new(m, 400, 1000, 32);
  pt_v1b8k_stream *s1 = pt_v1b8k_stream_new(m, 400, 1000, 1000); /* whole utterance in one vocoder chunk */
  pt_v1b8k_set_chunking(s1, 1000, 1000);
  printf("\n## single stream, %d thread%s (chunks: 3 frames first, doubling to 32 frames)\n\n", threads,
         threads > 1 ? "s (intra-op split)" : "");
  printf("| request | phones | audio s | synth s (best of 3) | RTF | mel ready ms | first chunk ms | non-streamed synth s | streaming overhead |\n");
  printf("|---|---|---|---|---|---|---|---|---|\n");
  double sa = 0, st = 0;
  for (int i = 0; i < NR; i++) {
    pt_synth_info best = {0}, inf;
    for (int rep = 0; rep < 4; rep++) {
      pt_v1b8k_synth(s, R[i].p, R[i].t, R[i].l, R[i].T, R[i].sid, null_cb, 0, &inf);
      if (rep == 1 || (rep > 1 && inf.t_total_s < best.t_total_s)) best = inf; /* rep 0 = warm-up */
    }
    double au = best.samples / (double)pt_v1b8k_sample_rate(m), bf = 1e9;
    for (int rep = 0; rep < 3; rep++) {
      pt_v1b8k_synth(s1, R[i].p, R[i].t, R[i].l, R[i].T, R[i].sid, null_cb, 0, &inf);
      if (inf.t_total_s < bf) bf = inf.t_total_s;
    }
    sa += au; st += best.t_total_s;
    printf("| %s | %d | %.2f | %.3f | %.3f | %.1f | %.1f | %.3f | %+.1f%% |\n", R[i].name, best.T, au, best.t_total_s,
           best.t_total_s / au, best.t_frontend_s * 1e3, best.t_first_chunk_s * 1e3, bf, (best.t_total_s / bf - 1) * 100);
  }
  printf("| **all** | | %.2f | %.3f | **%.3f** | | | | |\n", sa, st, st / sa);
  pt_v1b8k_prof_dump();
  if (threads == 1) g_single_rtf = st / sa;
  printf("\nstream arena: %.1f MB (limits 400 phones / 1000 frames = 16 s)\n", pt_v1b8k_stream_bytes(s) / 1e6);
  pt_v1b8k_stream_free(s);
  pt_v1b8k_stream_free(s1);
  pt_pool_set_local(1);
}

/* ---------------------------------------------------------------- concurrency */
static void run_conc(const pt_v1b8k_model *m, int N, int per_slot, double capacity, double safety, int maxs,
                     int narr, double spread) {
  int n = narr > 0 ? narr : N * per_slot;
  int sr = pt_v1b8k_sample_rate(m);
  pt_v1b8k_request *q = calloc(n, sizeof *q);
  int use[] = {0, 1, 2, 3, 4};  /* zh, en, mix, zh sid1, long (8.5 s) */
  double t0 = pt_now() + 0.05;
  for (int i = 0; i < n; i++) {
    req_in *r = &R[use[i % 5] % NR];
    q[i].phone = r->p; q[i].tone = r->t; q[i].lang = r->l; q[i].T = r->T; q[i].sid = r->sid;
    q[i].arrival_s = t0 + spread * i;
  }
  pt_v1b8k_sched_cfg cfg = {4, maxs, capacity, safety, 32, 1, 4, 256, 1000};
  pt_v1b8k_run_batch(m, &cfg, q, n);
  double audio = 0, tend = 0, rtf_sum = 0, rtf_max = 0, fc_sum = 0, fc_max = 0, late_max = -1e9, wait_max = 0;
  int ok = 0;
  for (int i = 0; i < n; i++) {
    double au = q[i].info.samples / (double)sr;
    double rtf = (q[i].t_done_s - q[i].t_admit_s) / au;
    double fc = q[i].t_first_audio_s - q[i].t_admit_s;
    audio += au;
    if (q[i].t_done_s > tend) tend = q[i].t_done_s;
    rtf_sum += rtf; if (rtf > rtf_max) rtf_max = rtf;
    fc_sum += fc; if (fc > fc_max) fc_max = fc;
    if (q[i].max_lateness_s > late_max) late_max = q[i].max_lateness_s;
    if (q[i].t_admit_s - q[i].arrival_s > wait_max) wait_max = q[i].t_admit_s - q[i].arrival_s;
    ok += q[i].status == 0;
  }
  double wall = tend - t0;
  printf("| %d | %d | %.1f | %.2f | **%.2f** | %.3f | %.3f | %.0f | %.0f | %+.0f | %.0f | %d/%d |\n", N, n, audio, wall,
         audio / wall, rtf_sum / n, rtf_max, fc_sum / n * 1e3, fc_max * 1e3, late_max * 1e3, wait_max * 1e3, ok, n);
  free(q);
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s model.ptk requests.txt [kernels single mt conc admit]\n", argv[0]); return 2; }
  int all = argc == 3;
  int want[5] = {all, all, all, all, all};
  const char *secs[5] = {"kernels", "single", "mt", "conc", "admit"};
  int only_n = 0;
  for (int i = 3; i < argc; i++) {
    for (int k = 0; k < 5; k++) if (!strcmp(argv[i], secs[k])) want[k] = 1;
    if (!strncmp(argv[i], "conc", 4) && argv[i][4]) { want[3] = 1; only_n = atoi(argv[i] + 4); }
  }
  pt_pool_init(4);
  double tl = pt_now();
  pt_v1b8k_model *m = pt_v1b8k_load(argv[1]);
  if (!m || load_requests(argv[2]) <= 0) { fprintf(stderr, "load failed\n"); return 1; }
  printf("# bench_cpu: PrimeTTS v1 (%d Hz, hop %d) on %ld cores, model load %.1f ms, %d requests\n", pt_v1b8k_sample_rate(m),
         pt_v1b8k_hop(m), sysconf(_SC_NPROCESSORS_ONLN), (pt_now() - tl) * 1e3, NR);
  if (want[0]) bench_kernels();
  if (want[1] || want[4]) bench_single(m, 1);
  if (want[2]) bench_single(m, 4);
  if (want[3]) {
    printf("\n## concurrent streams (N in flight, pinned round-robin on 4 cores, 3 requests per slot, all arrive at t=0, "
           "no admission throttle)\n\n");
    printf("| N | requests | audio s | wall s | audio-s per wall-s | mean RTF/request | max RTF | mean first-chunk ms "
           "| max first-chunk ms | worst lateness ms | max queue wait ms | ok |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for (int N = 1; N <= 8; N++)
      if (!only_n || N == only_n) run_conc(m, N, 3, 1e9, 1e9, N, 0, 0);
  }
  if (want[4] && g_single_rtf > 0) {
    double cap = 1.0 / g_single_rtf;
    printf("\n## admission policy: capacity %.2f audio-s/s per core (1 / single-stream RTF %.3f), safety 0.85, "
           "max 8 streams; 16 requests arriving %s\n\n", cap, g_single_rtf, "every 100 ms");
    printf("| N cap | requests | audio s | wall s | audio-s per wall-s | mean RTF/request | max RTF | mean first-chunk ms "
           "| max first-chunk ms | worst lateness ms | max queue wait ms | ok |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    run_conc(m, 8, 0, cap, 0.85, 8, 16, 0.1);
  }
  printf("\npeak RSS (VmHWM): %.1f MB\n", peak_rss_kb() / 1024.0);
  pt_v1b8k_free(m);
  pt_pool_shutdown();
  return 0;
}
