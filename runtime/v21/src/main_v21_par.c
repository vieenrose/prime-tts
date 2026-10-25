/* main_v21_par.c - M3 multi-stream driver: run N requests concurrently on one Nano GPU.
 *
 *   ./pt_v21_par --container build/v21.ptk --case tests/golden_v21/T81 --cuda [--parallel N]
 *
 * Each stream opens its own context (its own mmap of the container and host activation buffers) and
 * gets its own CUDA stream and device arena; the GPU weights are uploaded once and shared
 * read-only. The point of the run is the number the operator cares about: N concurrent real-time
 * requests, each producing audio faster than realtime, and the memory that costs.
 */
#define _GNU_SOURCE            /* CPU_SETSIZE / CPU_SET etc. are GNU extensions on this glibc */
#include "pt_v21.h"
#include "pt_graph_tab.h"
#include "pt_cpucap.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct {
  const char *container;
  const char *dir;
  int use_cuda, id;
  double ms, sndr, maxd, audio_s;
  long L;
  int rc;
} pt_job;

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int load_feed(const char *dir, const char *name, pt_tv *t) {
  char path[512];
  snprintf(path, sizeof(path), "%s/%s.npy", dir, name);
  void *p;
  int dt, nd;
  int64_t d[PT_V21_MAX_RANK], n;
  if (pt_npy_load_v21(path, &p, &dt, &nd, d, &n)) {
    fprintf(stderr, "par: cannot load %s\n", path);
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

static void *job_run(void *arg) {
  pt_job *j = (pt_job *) arg;
  static const char *fn[] = {"x", "tone", "lang", "x_lengths", "sid", "noise_scale", "length_scale"};
  pt_v21_trace(-1);
  if (j->use_cuda)
    pt_v21_cuda_prewarm();   /* context + clock ramp in flight while this thread opens the container */
  pt_ctx *c = pt_v21_open(j->container);
  if (!c) {
    j->rc = 2;
    return 0;
  }
  if (j->use_cuda && pt_v21_cuda_init(c, 0)) {
    j->rc = 3;
    return 0;
  }
  pt_tv feeds[8];
  memset(feeds, 0, sizeof(feeds));
  for (int i = 0; i < 7; i++)
    if (load_feed(j->dir, fn[i], &feeds[i])) {
      j->rc = 4;
      return 0;
    }
  double t0 = now_ms();
  float *wav = 0;
  int64_t wn = 0;
  if (pt_v21_run(c, feeds, &wav, &wn)) {
    j->rc = 5;
    return 0;
  }
  j->ms = now_ms() - t0;
  j->L = (long) wn;
  char path[512];
  snprintf(path, sizeof(path), "%s/wav.npy", j->dir);
  void *gp;
  int dt, nd;
  int64_t gd[PT_V21_MAX_RANK], gn;
  if (pt_npy_load_v21(path, &gp, &dt, &nd, gd, &gn)) {
    j->rc = 6;
    return 0;
  }
  const float *g = (const float *) gp;
  int64_t n = wn < gn ? wn : gn;
  double se = 0, sp = 0, mx = 0;
  for (int64_t i = 0; i < n; i++) {
    double d = (double) wav[i] - (double) g[i];
    se += d * d;
    sp += (double) g[i] * (double) g[i];
    if (fabs(d) > mx) mx = fabs(d);
  }
  j->maxd = mx;
  j->sndr = se > 0 && sp > 0 ? 10.0 * log10(sp / se) : 0.0;
  /* 16 kHz: wall time per audio second is the real-time factor */
  j->audio_s = (double) n / 16000.0;
  pt_v21_close(c);
  j->rc = 0;
  return 0;
}

int main(int argc, char **argv) {
  apply_cpu_cap();
  const char *container = "build/v21.ptk";
  const char *dir = "tests/golden_v21/T81";
  int use_cuda = 1;   /* CUDA is the default; --no-cuda / --cpu selects the portable CPU path */
  int N = 8;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--container") && i + 1 < argc) container = argv[++i];
    else if (!strcmp(argv[i], "--case") && i + 1 < argc) dir = argv[++i];
    else if (!strcmp(argv[i], "--parallel") && i + 1 < argc) N = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--cuda")) use_cuda = 1;   /* accepted for compatibility */
    else if (!strcmp(argv[i], "--no-cuda") || !strcmp(argv[i], "--cpu")) use_cuda = 0;
  }
  if (N < 1) N = 1;
  if (N > 8) N = 8;
  static pt_job jobs[8];
  static pthread_t th[8];
  for (int i = 0; i < N; i++) {
    jobs[i] = (pt_job){container, dir, use_cuda, i, 0, 0, 0, 0, 0, 0};
  }
  double t0 = now_ms();
  for (int i = 1; i < N; i++) pthread_create(&th[i], 0, job_run, &jobs[i]);
  job_run(&jobs[0]);   /* the calling thread is stream 0 */
  for (int i = 1; i < N; i++) pthread_join(th[i], 0);
  double wall = now_ms() - t0;
  double max_ms = 0, sum_audio = 0, min_sndr = 1e9;
  for (int i = 0; i < N; i++) {
    if (jobs[i].rc) {
      fprintf(stderr, "stream %d failed rc=%d\n", i, jobs[i].rc);
      return 7;
    }
    printf("stream %d: L=%ld  %.0f ms  SNDR=%.2f dB  max|d|=%.3e  RTF=%.3f\n", i, jobs[i].L,
           jobs[i].ms, jobs[i].sndr, jobs[i].maxd, jobs[i].ms / (jobs[i].audio_s * 1000.0));
    if (jobs[i].ms > max_ms) max_ms = jobs[i].ms;
    sum_audio += jobs[i].audio_s;
    if (jobs[i].sndr < min_sndr) min_sndr = jobs[i].sndr;
  }
  printf("N=%d  wall %.0f ms  slowest stream %.0f ms  audio %.2f s  throughput %.2fx realtime  "
         "min SNDR %.2f dB\n",
         N, wall, max_ms, sum_audio, (sum_audio * 1000.0) / wall, min_sndr);
  return min_sndr >= 60.0 ? 0 : 1;
}
