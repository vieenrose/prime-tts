/* sched.c - multi-stream server loop for v1b_8k: up to 8 concurrent requests on 4 Cortex-A57 cores.
 *
 * Model: one worker thread per admitted request, pinned to the least-loaded core, each with its own
 * pre-allocated stream (arena) -> no malloc and no locks on the synthesis path; weights are shared.
 * Admission (real-time guarantee): with k streams time-sharing one core, each stream runs at
 * k * RTF_single.  A request is admitted only if, after placing it on the least-loaded core,
 *     (k_core + 1) / capacity_rt <= safety          (capacity_rt = 1 / RTF_single, measured)
 * and the total in flight stays <= max_streams.  Otherwise it waits (FIFO) for a completion.
 * Every PCM chunk is time-stamped: playback is assumed to start when the first chunk is ready, chunk k
 * is due at t_first + (samples before k) / 8000 and lateness = ready - due (<= 0: never starved).
 */
#define _GNU_SOURCE
#include "pt_v1b8k.h"
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct server server;
typedef struct {
  server *sv;
  pt_v1b8k_request *r;
  pt_v1b8k_stream *st;
  int core, slot;
  double t_first;
  long samples;
  pthread_t th;
} worker;

struct server {
  const pt_v1b8k_sched_cfg *cfg;
  pthread_mutex_t mu;
  pthread_cond_t cv;
  int per_core[64];
  int active;
  int slot_busy[PT_MAX_STREAM];
  pt_v1b8k_stream *streams[PT_MAX_STREAM];
  pt_v1b8k_ws *ws[PT_MAX_STREAM];
  int ws_free[PT_MAX_STREAM], n_ws, n_ws_free;
  double ws_wait_max;
  worker w[PT_MAX_STREAM];
};

static int wrap_cb(void *u, const float *pcm, int n) {
  worker *w = u;
  double now = pt_now();
  if (w->samples == 0) {
    w->t_first = now;
    w->r->t_first_audio_s = now;
  }
  double due = w->t_first + (double)w->samples / PT_V1B8K_SR;
  double late = now - due;
  if (late > w->r->max_lateness_s) w->r->max_lateness_s = late;
  w->samples += n;
  return w->r->cb ? w->r->cb(w->r->user, pcm, n) : 0;
}

/* encoder+decoder workspace pool: blocks while all workspaces are in use */
static pt_v1b8k_ws *ws_acquire(void *u) {
  server *sv = u;
  double t0 = pt_now();
  pthread_mutex_lock(&sv->mu);
  while (sv->n_ws_free == 0) pthread_cond_wait(&sv->cv, &sv->mu);
  pt_v1b8k_ws *w = sv->ws[sv->ws_free[--sv->n_ws_free]];
  double wt = pt_now() - t0;
  if (wt > sv->ws_wait_max) sv->ws_wait_max = wt;
  pthread_mutex_unlock(&sv->mu);
  return w;
}
static void ws_release(void *u, pt_v1b8k_ws *w) {
  server *sv = u;
  pthread_mutex_lock(&sv->mu);
  for (int i = 0; i < sv->n_ws; i++)
    if (sv->ws[i] == w) sv->ws_free[sv->n_ws_free++] = i;
  pthread_cond_broadcast(&sv->cv);
  pthread_mutex_unlock(&sv->mu);
}

static void *worker_main(void *p) {
  worker *w = p;
  server *sv = w->sv;
  if (sv->cfg->pin) {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(w->core, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
  }
  pt_pool_set_local(1);
  w->samples = 0;
  w->r->max_lateness_s = -1e9;
  w->r->status = pt_v1b8k_synth(w->st, w->r->phone, w->r->tone, w->r->lang, w->r->T, w->r->sid, wrap_cb, w,
                                &w->r->info);
  w->r->t_done_s = pt_now();
  pthread_mutex_lock(&sv->mu);
  sv->per_core[w->core]--;
  sv->active--;
  sv->slot_busy[w->slot] = 0;
  pthread_cond_broadcast(&sv->cv);
  pthread_mutex_unlock(&sv->mu);
  return 0;
}

static void sleep_until(double t) {
  double now = pt_now();
  if (t <= now) return;
  struct timespec ts;
  double d = t - now;
  ts.tv_sec = (time_t)d;
  ts.tv_nsec = (long)((d - ts.tv_sec) * 1e9);
  nanosleep(&ts, 0);
}

int pt_v1b8k_run_batch(const pt_v1b8k_model *m, const pt_v1b8k_sched_cfg *cfg, pt_v1b8k_request *req, int n) {
  server sv;
  memset(&sv, 0, sizeof sv);
  sv.cfg = cfg;
  pthread_mutex_init(&sv.mu, 0);
  pthread_cond_init(&sv.cv, 0);
  int maxs = cfg->max_streams < 1 ? 1 : cfg->max_streams > PT_MAX_STREAM ? PT_MAX_STREAM : cfg->max_streams;
  int ncores = cfg->ncores < 1 ? 1 : cfg->ncores > 64 ? 64 : cfg->ncores;
  int maxT = cfg->max_phones > 0 ? cfg->max_phones : 256, maxF = cfg->max_frames > 0 ? cfg->max_frames : 1000;
  sv.n_ws = cfg->n_workspaces > 0 ? cfg->n_workspaces : ncores;
  if (sv.n_ws > maxs) sv.n_ws = maxs;
  for (int i = 0; i < sv.n_ws; i++) {
    sv.ws[i] = pt_v1b8k_ws_new(m, maxT, maxF);
    if (!sv.ws[i]) return -1;
    sv.ws_free[sv.n_ws_free++] = i;
  }
  for (int i = 0; i < maxs; i++) {
    sv.streams[i] = pt_v1b8k_stream_new_shared(m, maxT, maxF, cfg->chunk_frames > 0 ? cfg->chunk_frames : 32);
    if (!sv.streams[i]) return -1;
    pt_v1b8k_set_ws_pool(sv.streams[i], ws_acquire, ws_release, &sv);
  }
  for (int i = 0; i < n; i++) {
    pt_v1b8k_request *r = &req[i];
    r->status = 2;
    sleep_until(r->arrival_s);
    pthread_mutex_lock(&sv.mu);
    int core = -1;
    for (;;) {
      core = 0;
      for (int c = 1; c < ncores; c++)
        if (sv.per_core[c] < sv.per_core[core]) core = c;
      double rtf_after = (sv.per_core[core] + 1) / (cfg->capacity_rt > 0 ? cfg->capacity_rt : 1e9);
      if (sv.active < maxs && rtf_after <= cfg->safety) break;
      pthread_cond_wait(&sv.cv, &sv.mu);
    }
    int slot = 0;
    while (sv.slot_busy[slot]) slot++;
    sv.slot_busy[slot] = 1;
    sv.per_core[core]++;
    sv.active++;
    worker *w = &sv.w[slot];
    if (w->th) pthread_join(w->th, 0); /* reap the previous user of this slot */
    memset(w, 0, sizeof *w);
    w->sv = &sv;
    w->r = r;
    w->st = sv.streams[slot];
    w->core = core;
    w->slot = slot;
    r->t_admit_s = pt_now();
    pthread_create(&w->th, 0, worker_main, w);
    pthread_mutex_unlock(&sv.mu);
  }
  for (int i = 0; i < maxs; i++)
    if (sv.w[i].th) pthread_join(sv.w[i].th, 0);
  for (int i = 0; i < maxs; i++) pt_v1b8k_stream_free(sv.streams[i]);
  for (int i = 0; i < sv.n_ws; i++) pt_v1b8k_ws_free(sv.ws[i]);
  return 0;
}
