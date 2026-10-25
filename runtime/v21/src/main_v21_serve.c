/* main_v21_serve.c - persistent PrimeTTS v2.1 server: OpenAI-shaped POST /v1/audio/speech.
 *
 *   ./pt_v21_serve --container build/v21.ptk [--port 8090] [--bind 0.0.0.0] [--workers 8]
 *                  [--noise 0] [--default-sid 0]
 *
 * Server mode is the latency lever the benchmark driver cannot show: the container record pass,
 * CUDA context, weight upload and clock ramp (~287 ms of serialized startup per process) are paid
 * ONCE here, and every request runs on an already-warm context. N worker threads each own one
 * context (shared resident weights, own CUDA stream and arena -- exactly the multi-stream driver's
 * layout), so up to N requests synthesize concurrently and the rest wait in a bounded queue.
 *
 * Protocol (HTTP/1.1, one request per connection, no dependencies beyond libc + pthreads):
 *   POST /v1/audio/speech   OpenAI request shape: {"model", "voice", "response_format", "speed",
 *                           ...}. The engine has no text frontend (g2pw/g2p_en run on the gateway,
 *                           server/tts_gateway.py), so the text "input" must come with the model's
 *                           input sequences as an extension field:
 *                             "input_ids": {"phone": [...], "tone": [...], "lang": [...]}
 *                           (or [[phone], [tone], [lang]]), already blank-interleaved (add_blank).
 *                           "voice": "0" | "1" | "2" (sid); other names -> --default-sid.
 *                           "speed": length_scale = 1/speed.  "noise_scale": optional extension.
 *                           "response_format": "pcm" (s16le mono 16 kHz) | "wav".
 *                           "priority": optional deadline (unix ms); the queue is earliest-deadline-
 *                           first, default = arrival time (FIFO for clients that do not set it).
 *   GET  /v1/audio/voices   the three speaker ids.
 *   GET  /health            worker / queue state.
 */
#define _GNU_SOURCE            /* CPU_SETSIZE / CPU_SET (pt_cpucap.h) are GNU extensions */
#include "pt_v21.h"
#include "pt_graph_tab.h"
#include "pt_cpucap.h"
#include <arpa/inet.h>
#include <ctype.h>
#include <math.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

/* v2.1 embedding sizes (net.enc_p.emb_phone/tone/lang, net.emb_g): ids outside them are a 400 */
#define N_PHONE 88
#define N_TONE 6
#define N_LANG 2
#define N_SID 3
#define SR 16000
#define MAX_T 4096          /* input tokens per request (blank-interleaved) */
#define MAX_BODY (1 << 20)
#define MAX_HDR 16384
#define MAX_QUEUE 64
#define MAX_CONN 64

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int sink_chunk(int fd, const void *data, size_t n);   /* HTTP/1.1 chunk writer (below) */

/* ---------------------------------------------------------------- job queue */
typedef struct job {
  int64_t T;
  int64_t *ids;            /* 3*T: phone | tone | lang */
  int64_t sid;
  float noise, length_scale;
  double prio;             /* deadline, unix ms: smaller runs first (default = arrival time) */
  int sink_fd;             /* >= 0: stream PCM to this socket as it is synthesized */
  int64_t sent;            /* bytes already streamed */
  /* result */
  int16_t *pcm;
  int64_t n;
  int rc;
  double t_enq, t_start, t_end;
  int done;
  pthread_mutex_t mu;
  pthread_cond_t cv;
  struct job *next;
} job;

static pthread_mutex_t g_qmu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_qcv = PTHREAD_COND_INITIALIZER;
static job *g_qhead, *g_qtail;
static int g_qlen, g_busy, g_ready, g_nworkers;
static double g_t0;
static volatile double g_last_req;   /* ms, end of the latest request (keep-warm window anchor) */
static double g_kw_window = 30000;   /* --keepwarm SEC: hold the clock this long after traffic */
static int g_kw_iters = 16000, g_kw_period = 100;  /* burst size / period (PT_V21_KW_ITERS/_MS) */
static int g_kw_adapt = 1;
static long g_gpu_max = -1;   /* devfreq max_freq, Hz */

/* Keep the GPU clock up between requests: the governor drops it within seconds of idle and the
 * next request then runs at a fraction of full clock (+350-750 ms on T81). Bursts run only inside
 * the window after real traffic and only while every worker is idle, so they never compete with a
 * synthesis and the GPU goes back to idle (power, co-tenants) when traffic stops. */
static void *keepwarm_main(void *arg) {
  (void) arg;
  for (;;) {
    usleep((useconds_t) g_kw_period * 1000);
    if (g_kw_window <= 0 || g_ready < g_nworkers) continue;
    double since = now_ms() - g_last_req;
    if (since > g_kw_window) continue;
    pthread_mutex_lock(&g_qmu);
    int idle = g_busy == 0 && g_qlen == 0;
    pthread_mutex_unlock(&g_qmu);
    if (!idle) continue;
    /* feedback on the governor's own clock: the smallest burst that holds max clock is box- and
     * load-dependent (picard needs ~64k iters per 100 ms over a 20 s gap; pesi idles at max). Grow
     * while below max, decay while at max. A fixed PT_V21_KW_ITERS disables the adaptation. */
    if (g_kw_adapt) {
      long cur = pt_v21_gpu_cur_freq();
      if (cur > 0 && g_gpu_max > 0) {
        if (cur < g_gpu_max) g_kw_iters = g_kw_iters * 2 > 64000 ? 64000 : g_kw_iters * 2;
        else g_kw_iters = g_kw_iters * 4 / 5 < 2000 ? 2000 : g_kw_iters * 4 / 5;
      }
    }
    pt_v21_cuda_keepwarm(g_kw_iters);
  }
  return 0;
}
static volatile int g_nconn;

static int queue_push(job *j) {
  pthread_mutex_lock(&g_qmu);
  if (g_qlen >= MAX_QUEUE) {
    pthread_mutex_unlock(&g_qmu);
    return -1;
  }
  /* earliest-deadline-first, stable among equal deadlines: a client that streams a reply as
   * sentences tags each with the time its audio is needed, so every request's first sentence runs
   * before anyone's later sentences (those have the earlier sentences' playback time as slack) */
  job **pp = &g_qhead;
  while (*pp && (*pp)->prio <= j->prio) pp = &(*pp)->next;
  j->next = *pp;
  *pp = j;
  if (!j->next) g_qtail = j;
  g_qlen++;
  pthread_cond_signal(&g_qcv);
  pthread_mutex_unlock(&g_qmu);
  return 0;
}

static job *queue_pop(void) {
  pthread_mutex_lock(&g_qmu);
  while (!g_qhead) pthread_cond_wait(&g_qcv, &g_qmu);
  job *j = g_qhead;
  g_qhead = j->next;
  if (!g_qhead) g_qtail = 0;
  g_qlen--;
  g_busy++;
  pthread_mutex_unlock(&g_qmu);
  return j;
}

/* ---------------------------------------------------------------- workers */
typedef struct {
  const char *container;
  int use_cuda, id;
} worker_arg;

typedef struct { int64_t xl, sid; float ns, ls; } feed_scalars;

/* The scalar feeds must outlive the run: the tensors alias these fields. */
static void build_feeds(job *j, pt_tv f[8], feed_scalars *sc) {
  int64_t T = j->T;
  sc->xl = T;
  sc->sid = j->sid;
  sc->ns = j->noise;
  sc->ls = j->length_scale;
  memset(f, 0, sizeof(pt_tv) * 8);
  /* feed order = graph input order: x, tone, lang, x_lengths, sid, noise_scale, length_scale */
  for (int i = 0; i < 3; i++) {
    f[i].dt = DT_I64;
    f[i].ndim = 2;
    f[i].d[0] = 1;
    f[i].d[1] = T;
    f[i].n = T;
    f[i].p = j->ids + (size_t) i * T;
  }
  void *sp[4] = {&sc->xl, &sc->sid, &sc->ns, &sc->ls};
  int dts[4] = {DT_I64, DT_I64, DT_F32, DT_F32};
  for (int i = 0; i < 4; i++) {
    f[3 + i].dt = dts[i];
    f[3 + i].ndim = 1;
    f[3 + i].d[0] = 1;
    f[3 + i].n = 1;
    f[3 + i].p = sp[i];
  }
  for (int i = 0; i < 7; i++) {
    for (int q = f[i].ndim; q < PT_V21_MAX_RANK; q++) f[i].d[q] = 1;
    f[i].alive = 1;
  }
}

static int16_t to_s16(float v) {
  float x = v * 32767.0f;
  return (int16_t) (x > 32767.0f ? 32767 : x < -32768.0f ? -32768 : lrintf(x));
}

/* Streaming synthesis: one encoder pass, then the decoder tail on sliding windows (overlap-save),
 * so the response carries ONE utterance -- one decoder start, continuous prosody -- while audio is
 * still delivered as it is produced. Context of PT_V21_STREAM_CTX frames (default 16) makes
 * the result numerically identical to a single full-graph run (measured: 133 dB vs full). */
static int synth_stream(pt_ctx *c, job *j) {
  pt_tv f[8];
  feed_scalars sc;
  build_feeds(j, f, &sc);
  int64_t T = 0;
  int hop = 0;
  if (pt_v21_stream_open(c, f, &T, &hop)) return -1;
  /* Overlap-save windows: the FIRST window is short (fast first audio), later windows are long
   * (less overlap recomputed). Cost is (W / (W - 2*CTX)) on the decoder part. */
  int ctx = 16, win_first = 48, win = 256;
  if (getenv("PT_V21_STREAM_CTX")) ctx = atoi(getenv("PT_V21_STREAM_CTX"));
  if (getenv("PT_V21_STREAM_WIN")) win_first = win = atoi(getenv("PT_V21_STREAM_WIN"));
  if (getenv("PT_V21_STREAM_WIN_FIRST")) win_first = atoi(getenv("PT_V21_STREAM_WIN_FIRST"));
  if (getenv("PT_V21_STREAM_WIN_MAX")) win = atoi(getenv("PT_V21_STREAM_WIN_MAX"));
  if (win_first < 2 * ctx + 8) win_first = 2 * ctx + 8;
  if (win < 2 * ctx + 8) win = 2 * ctx + 8;
  int16_t *buf = NULL;
  int64_t bufcap = 0;
  int rc = 0;
  int64_t a = 0, emitted = 0;
  while (emitted < T) {
    int64_t w = (a == 0) ? win_first : win;
    int64_t win_eff = w;
    if (a + win_eff > T) win_eff = T - a;
    int last = (a + win_eff >= T);
    float *pcm = 0;
    int64_t n = 0;
    if (pt_v21_stream_chunk(c, a, win_eff, &pcm, &n)) { rc = -1; break; }
    int64_t lo = (a == 0) ? 0 : ctx;
    int64_t hi = last ? win_eff : (win_eff - ctx);
    int64_t e0 = a + lo, e1 = a + hi;
    if (e0 < emitted) e0 = emitted;
    if (e1 > T) e1 = T;
    int64_t cnt = e1 > e0 ? (e1 - e0) * hop : 0;
    int64_t skip = (e0 - a) * hop;
    if (cnt > 0 && skip >= 0 && skip + cnt <= n) {
      if (cnt > bufcap) {
        bufcap = cnt;
        buf = (int16_t *) realloc(buf, (size_t) bufcap * 2);
        if (!buf) { rc = -1; break; }
      }
      for (int64_t i = 0; i < cnt; i++) buf[i] = to_s16(pcm[skip + i]);
      if (sink_chunk(j->sink_fd, buf, (size_t) cnt * 2)) { rc = -1; break; }   /* client gone */
      j->sent += cnt * 2;
      j->n += cnt;
      emitted = e1;
    }
    if (last) break;
    a = e1 - ctx;
    if (a <= 0) a = 1;
  }
  free(buf);
  pt_v21_stream_close(c);
  return rc;
}

static int synth(pt_ctx *c, job *j) {
  pt_tv f[8];
  feed_scalars sc;
  build_feeds(j, f, &sc);
  if (j->sink_fd >= 0) return synth_stream(c, j);
  float *wav = 0;
  int64_t wn = 0;
  if (pt_v21_run(c, f, &wav, &wn)) return -1;
  j->pcm = (int16_t *) malloc((size_t) (wn > 0 ? wn : 1) * 2);
  if (!j->pcm) return -1;
  for (int64_t i = 0; i < wn; i++) j->pcm[i] = to_s16(wav[i]);
  j->n = wn;
  return 0;
}

static void *worker_main(void *arg) {
  worker_arg *w = (worker_arg *) arg;
  pt_v21_trace(-1);
  if (w->use_cuda) pt_v21_cuda_prewarm();
  pt_ctx *c = pt_v21_open(w->container);
  if (!c) {
    fprintf(stderr, "serve: worker %d: cannot open %s\n", w->id, w->container);
    exit(2);
  }
  if (w->use_cuda && pt_v21_cuda_init(c, 0)) {
    fprintf(stderr, "serve: worker %d: CUDA init failed\n", w->id);
    exit(3);
  }
  /* one warm-up synthesis: first-touch of the activation buffers and the arena, so the first
   * real request pays none of it */
  {
    enum { WT = 41 };
    int64_t ids[3 * WT];
    for (int i = 0; i < WT; i++) {
      ids[i] = (i & 1) ? 10 : 0;   /* blank-interleaved filler phone */
      ids[WT + i] = (i & 1) ? 1 : 0;
      ids[2 * WT + i] = 0;
    }
    job wj;
    memset(&wj, 0, sizeof(wj));
    wj.T = WT;
    wj.ids = ids;
    wj.length_scale = 1.0f;
    wj.sink_fd = -1;   /* buffered warm-up: no socket */
    if (synth(c, &wj)) {
      fprintf(stderr, "serve: worker %d: warm-up synthesis failed\n", w->id);
      exit(4);
    }
    free(wj.pcm);
  }
  pthread_mutex_lock(&g_qmu);
  int all = ++g_ready == g_nworkers;
  pthread_mutex_unlock(&g_qmu);
  if (all) fprintf(stderr, "serve: ready in %.0f ms (%d warm worker(s))\n", now_ms() - g_t0, g_nworkers);
  for (;;) {
    job *j = queue_pop();
    j->t_start = now_ms();
    j->rc = synth(c, j);
    j->t_end = now_ms();
    pthread_mutex_lock(&g_qmu);
    g_busy--;
    pthread_mutex_unlock(&g_qmu);
    g_last_req = now_ms();
    pthread_mutex_lock(&j->mu);
    j->done = 1;
    pthread_cond_signal(&j->cv);
    pthread_mutex_unlock(&j->mu);
  }
  return 0;
}

/* ---------------------------------------------------------------- minimal JSON reader
 * Enough for the request object: top-level member lookup, strings, numbers, int arrays. Values
 * are skipped structurally (strings respected), so a key inside "input" text never matches. */
static const char *ws(const char *p, const char *e) {
  while (p < e && isspace((unsigned char) *p)) p++;
  return p;
}

static const char *skip_str(const char *p, const char *e) {   /* p at '"' */
  for (p++; p < e; p++) {
    if (*p == '\\') p++;
    else if (*p == '"') return p + 1;
  }
  return 0;
}

static const char *skip_val(const char *p, const char *e) {
  p = ws(p, e);
  if (p >= e) return 0;
  if (*p == '"') return skip_str(p, e);
  if (*p == '{' || *p == '[') {
    int depth = 0;
    while (p < e) {
      if (*p == '"') {
        p = skip_str(p, e);
        if (!p) return 0;
        continue;
      }
      if (*p == '{' || *p == '[') depth++;
      else if (*p == '}' || *p == ']') {
        if (--depth == 0) return p + 1;
      }
      p++;
    }
    return 0;
  }
  while (p < e && *p != ',' && *p != '}' && *p != ']' && !isspace((unsigned char) *p)) p++;
  return p;
}

/* value of member `key` in the object starting at p ('{'), or NULL */
static const char *obj_get(const char *p, const char *e, const char *key) {
  p = ws(p, e);
  if (p >= e || *p != '{') return 0;
  p++;
  size_t kl = strlen(key);
  for (;;) {
    p = ws(p, e);
    if (p >= e || *p == '}') return 0;
    if (*p != '"') return 0;
    const char *ks = p + 1, *ke = skip_str(p, e);
    if (!ke) return 0;
    int match = (size_t) (ke - 1 - ks) == kl && !memcmp(ks, key, kl);
    p = ws(ke, e);
    if (p >= e || *p != ':') return 0;
    p = ws(p + 1, e);
    if (match) return p;
    p = skip_val(p, e);
    if (!p) return 0;
    p = ws(p, e);
    if (p < e && *p == ',') p++;
  }
}

static int get_str(const char *v, const char *e, char *out, size_t n) {
  if (!v || v >= e || *v != '"') return -1;
  size_t k = 0;
  for (v++; v < e && *v != '"'; v++) {
    char ch = *v;
    if (ch == '\\' && v + 1 < e) ch = *++v;
    if (k + 1 < n) out[k++] = ch;
  }
  out[k] = 0;
  return 0;
}

static int get_num(const char *v, const char *e, double *out) {
  if (!v || v >= e) return -1;
  char buf[64];
  size_t k = 0;
  while (v < e && k + 1 < sizeof(buf) && (isdigit((unsigned char) *v) || strchr("+-.eE", *v))) buf[k++] = *v++;
  buf[k] = 0;
  if (!k) return -1;
  char *end;
  *out = strtod(buf, &end);
  return *end ? -1 : 0;
}

/* int array at v -> out (max n), returns count or -1 */
static int64_t get_iarr(const char *v, const char *e, int64_t *out, int64_t n) {
  v = ws(v, e);
  if (!v || v >= e || *v != '[') return -1;
  v++;
  int64_t k = 0;
  for (;;) {
    v = ws(v, e);
    if (v >= e) return -1;
    if (*v == ']') return k;
    char *end;
    long long x = strtoll(v, &end, 10);
    if (end == v || k >= n) return -1;
    out[k++] = x;
    v = ws(end, e);
    if (v < e && *v == ',') v++;
  }
}

/* ---------------------------------------------------------------- HTTP */
static int send_all(int fd, const void *buf, size_t n) {
  const char *p = (const char *) buf;
  while (n) {
    ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
    if (k <= 0) return -1;
    p += k;
    n -= (size_t) k;
  }
  return 0;
}

/* one HTTP/1.1 chunk (Transfer-Encoding: chunked) */
static int sink_chunk(int fd, const void *data, size_t n) {
  char hdr[32];
  int hn = snprintf(hdr, sizeof hdr, "%zx\r\n", n);
  if (send_all(fd, hdr, (size_t) hn)) return -1;
  if (send_all(fd, data, n)) return -1;
  return send_all(fd, "\r\n", 2);
}

static void reply(int fd, int code, const char *ctype, const void *body, size_t n) {
  const char *msg = code == 200 ? "OK" : code == 400 ? "Bad Request" : code == 404 ? "Not Found" :
                    code == 405 ? "Method Not Allowed" : code == 413 ? "Payload Too Large" :
                    code == 503 ? "Service Unavailable" : "Internal Server Error";
  char h[256];
  int hn = snprintf(h, sizeof(h),
                    "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                    "Connection: close\r\n\r\n", code, msg, ctype, n);
  if (send_all(fd, h, (size_t) hn) == 0 && n) send_all(fd, body, n);
}

static void reply_err(int fd, int code, const char *message) {
  char b[512];
  int n = snprintf(b, sizeof(b), "{\"error\":{\"message\":\"%s\",\"type\":\"invalid_request_error\"}}",
                   message);
  reply(fd, code, "application/json", b, (size_t) n);
}

static int g_default_sid;
static float g_default_noise;

static void handle_speech(int fd, const char *body, size_t blen) {
  const char *e = body + blen;
  const char *b = ws(body, e);
  if (b >= e || *b != '{') {
    reply_err(fd, 400, "body must be a JSON object");
    return;
  }
  char fmt[16] = "wav";
  const char *v = obj_get(b, e, "response_format");
  if (v && get_str(v, e, fmt, sizeof(fmt))) {
    reply_err(fd, 400, "response_format must be a string");
    return;
  }
  int is_pcm = !strcmp(fmt, "pcm");
  if (!is_pcm && strcmp(fmt, "wav")) {
    reply_err(fd, 400, "response_format must be pcm or wav");
    return;
  }
  const char *ii = obj_get(b, e, "input_ids");
  if (!ii) {
    reply_err(fd, 400, "this engine has no text frontend: send input_ids (use the tts_gateway for text input)");
    return;
  }
  int64_t *ids = (int64_t *) malloc(sizeof(int64_t) * 3 * MAX_T);
  if (!ids) {
    reply_err(fd, 500, "out of memory");
    return;
  }
  int64_t n[3] = {-1, -1, -1};
  if (*ii == '{') {
    const char *keys[3] = {"phone", "tone", "lang"};
    for (int i = 0; i < 3; i++) {
      const char *a = obj_get(ii, e, keys[i]);
      n[i] = a ? get_iarr(a, e, ids + (size_t) i * MAX_T, MAX_T) : -1;
    }
  } else if (*ii == '[') {
    const char *p = ws(ii + 1, e);
    for (int i = 0; i < 3 && p && p < e; i++) {
      n[i] = get_iarr(p, e, ids + (size_t) i * MAX_T, MAX_T);
      p = skip_val(p, e);
      if (p) p = ws(p, e);
      if (p && p < e && *p == ',') p = ws(p + 1, e);
    }
  }
  if (n[0] < 1 || n[0] != n[1] || n[0] != n[2]) {
    free(ids);
    reply_err(fd, 400, "input_ids needs phone/tone/lang int arrays of equal length 1..4096");
    return;
  }
  int64_t T = n[0];
  const int64_t lim[3] = {N_PHONE, N_TONE, N_LANG};
  for (int i = 0; i < 3; i++)
    for (int64_t t = 0; t < T; t++)
      if (ids[(size_t) i * MAX_T + t] < 0 || ids[(size_t) i * MAX_T + t] >= lim[i]) {
        free(ids);
        reply_err(fd, 400, "input_ids value out of range (phone<88, tone<6, lang<2)");
        return;
      }
  /* compact phone|tone|lang to stride T */
  memmove(ids + T, ids + MAX_T, sizeof(int64_t) * (size_t) T);
  memmove(ids + 2 * T, ids + 2 * MAX_T, sizeof(int64_t) * (size_t) T);

  int64_t sid = g_default_sid;
  v = obj_get(b, e, "voice");
  if (v) {
    char vs[64];
    double dv;
    if (*v == '"' && !get_str(v, e, vs, sizeof(vs))) {
      char *end;
      long x = strtol(vs, &end, 10);
      if (*vs && !*end && x >= 0 && x < N_SID) sid = x;
    } else if (!get_num(v, e, &dv) && dv >= 0 && dv < N_SID) {
      sid = (int64_t) dv;
    }
  }
  double speed = 1.0, noise = g_default_noise;
  v = obj_get(b, e, "speed");
  if (v && (get_num(v, e, &speed) || speed < 0.25 || speed > 4.0)) {
    free(ids);
    reply_err(fd, 400, "speed must be a number in [0.25, 4]");
    return;
  }
  v = obj_get(b, e, "noise_scale");
  if (v && (get_num(v, e, &noise) || noise < 0 || noise > 2)) {
    free(ids);
    reply_err(fd, 400, "noise_scale must be a number in [0, 2]");
    return;
  }

  struct timespec rt;
  clock_gettime(CLOCK_REALTIME, &rt);
  double prio = rt.tv_sec * 1000.0 + rt.tv_nsec / 1e6;
  v = obj_get(b, e, "priority");
  if (v && get_num(v, e, &prio)) {
    free(ids);
    reply_err(fd, 400, "priority must be a number (deadline, unix ms)");
    return;
  }
  job j;
  memset(&j, 0, sizeof(j));
  j.prio = prio;
  pthread_mutex_init(&j.mu, 0);
  pthread_cond_init(&j.cv, 0);
  j.T = T;
  j.ids = ids;
  j.sid = sid;
  j.noise = (float) noise;
  j.length_scale = (float) (1.0 / speed);
  j.t_enq = now_ms();
  /* pcm streams: headers go out now, then the worker writes body chunks as it synthesizes
   * (one encoder pass, windowed decoder -> one utterance, continuous prosody, early audio) */
  j.sink_fd = is_pcm ? fd : -1;
  if (is_pcm) {
    static const char hdr[] = "HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\n"
                              "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    if (send_all(fd, hdr, sizeof(hdr) - 1)) {
      free(ids);
      return;
    }
  }
  if (queue_push(&j)) {
    free(ids);
    if (!is_pcm) reply_err(fd, 503, "queue full");
    return;
  }
  pthread_mutex_lock(&j.mu);
  while (!j.done) pthread_cond_wait(&j.cv, &j.mu);
  pthread_mutex_unlock(&j.mu);
  free(ids);
  if (j.rc) {
    free(j.pcm);
    reply_err(fd, 500, "synthesis failed");
    return;
  }
  size_t pb = (size_t) j.n * 2;   /* pcm: bytes already streamed (j.sent) */
  fprintf(stderr, "serve: T=%lld sid=%lld audio=%.2fs queue=%.0fms synth=%.0fms rtf=%.3f fmt=%s\n",
          (long long) T, (long long) sid, j.n / (double) SR, j.t_start - j.t_enq,
          j.t_end - j.t_start, (j.t_end - j.t_start) / 1000.0 / (j.n / (double) SR + 1e-9), fmt);
  if (is_pcm) {
    send_all(fd, "0\r\n\r\n", 5);   /* end of the chunked body */
  } else {
    unsigned char *w = (unsigned char *) malloc(44 + pb);
    if (!w) {
      free(j.pcm);
      reply_err(fd, 500, "out of memory");
      return;
    }
    uint32_t u;
    uint16_t s;
    memcpy(w, "RIFF", 4);
    u = (uint32_t) (36 + pb); memcpy(w + 4, &u, 4);
    memcpy(w + 8, "WAVEfmt ", 8);
    u = 16; memcpy(w + 16, &u, 4);
    s = 1; memcpy(w + 20, &s, 2);              /* PCM */
    s = 1; memcpy(w + 22, &s, 2);              /* mono */
    u = SR; memcpy(w + 24, &u, 4);
    u = SR * 2; memcpy(w + 28, &u, 4);
    s = 2; memcpy(w + 32, &s, 2);
    s = 16; memcpy(w + 34, &s, 2);
    memcpy(w + 36, "data", 4);
    u = (uint32_t) pb; memcpy(w + 40, &u, 4);
    memcpy(w + 44, j.pcm, pb);
    reply(fd, 200, "audio/wav", w, 44 + pb);
    free(w);
  }
  free(j.pcm);
  pthread_mutex_destroy(&j.mu);
  pthread_cond_destroy(&j.cv);
}

static void *conn_main(void *arg) {
  int fd = (int) (intptr_t) arg;
  char *buf = (char *) malloc(MAX_HDR + MAX_BODY + 1);
  size_t have = 0;
  char *hend = 0;
  if (!buf) goto out;
  while (!hend) {
    if (have >= MAX_HDR) {
      reply_err(fd, 413, "headers too large");
      goto out;
    }
    ssize_t k = recv(fd, buf + have, MAX_HDR - have, 0);
    if (k <= 0) goto out;
    have += (size_t) k;
    buf[have] = 0;
    hend = strstr(buf, "\r\n\r\n");
  }
  {
    char method[8] = {0}, path[256] = {0};
    if (sscanf(buf, "%7s %255s", method, path) != 2) {
      reply_err(fd, 400, "bad request line");
      goto out;
    }
    char *q = strchr(path, '?');
    if (q) *q = 0;
    size_t clen = 0;
    int expect100 = 0;
    for (char *l = strstr(buf, "\r\n"); l && l < hend; l = strstr(l + 2, "\r\n")) {
      if (!strncasecmp(l + 2, "Content-Length:", 15)) clen = (size_t) strtoull(l + 17, 0, 10);
      if (!strncasecmp(l + 2, "Expect:", 7) && strstr(l + 9, "100-continue") &&
          strstr(l + 9, "100-continue") < strstr(l + 2, "\r\n"))
        expect100 = 1;
    }
    if (!strcmp(method, "GET")) {
      if (!strcmp(path, "/health")) {
        char b[192];
        pthread_mutex_lock(&g_qmu);
        int n = snprintf(b, sizeof(b), "{\"status\":\"%s\",\"workers\":%d,\"ready\":%d,\"busy\":%d,\"queued\":%d,\"gpu_mhz\":%ld}",
                         g_ready == g_nworkers ? "ok" : "starting", g_nworkers, g_ready, g_busy, g_qlen,
                         pt_v21_gpu_cur_freq() / 1000000);
        pthread_mutex_unlock(&g_qmu);
        reply(fd, 200, "application/json", b, (size_t) n);
      } else if (!strcmp(path, "/v1/audio/voices") || !strcmp(path, "/audio/voices")) {
        static const char v[] = "{\"voices\":[\"0\",\"1\",\"2\"]}";
        reply(fd, 200, "application/json", v, sizeof(v) - 1);
      } else {
        reply_err(fd, 404, "not found");
      }
      goto out;
    }
    if (strcmp(method, "POST")) {
      reply_err(fd, 405, "method not allowed");
      goto out;
    }
    if (strcmp(path, "/v1/audio/speech") && strcmp(path, "/audio/speech")) {
      reply_err(fd, 404, "not found");
      goto out;
    }
    if (clen == 0 || clen > MAX_BODY) {
      reply_err(fd, clen ? 413 : 400, clen ? "body too large" : "Content-Length required");
      goto out;
    }
    if (expect100) send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n", 25);
    size_t hlen = (size_t) (hend + 4 - buf), got = have - hlen;
    char *body = buf + hlen;
    while (got < clen) {
      ssize_t k = recv(fd, body + got, clen - got, 0);
      if (k <= 0) goto out;
      got += (size_t) k;
    }
    body[clen] = 0;
    if (g_ready < g_nworkers) {
      reply_err(fd, 503, "warming up");
      goto out;
    }
    handle_speech(fd, body, clen);
  }
out:
  free(buf);
  close(fd);
  __sync_sub_and_fetch(&g_nconn, 1);
  return 0;
}

int main(int argc, char **argv) {
  apply_cpu_cap();
  const char *container = "build/v21.ptk", *bind_ip = "0.0.0.0";
  int use_cuda = 1;   /* CUDA is the default; --no-cuda selects the portable CPU path */
  int port = 8090, nw = 8;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--container") && i + 1 < argc) container = argv[++i];
    else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--bind") && i + 1 < argc) bind_ip = argv[++i];
    else if (!strcmp(argv[i], "--workers") && i + 1 < argc) nw = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--noise") && i + 1 < argc) g_default_noise = (float) atof(argv[++i]);
    else if (!strcmp(argv[i], "--default-sid") && i + 1 < argc) g_default_sid = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--keepwarm") && i + 1 < argc) g_kw_window = atof(argv[++i]) * 1000.0;
    else if (!strcmp(argv[i], "--cuda")) use_cuda = 1;   /* accepted for compatibility */
    else if (!strcmp(argv[i], "--no-cuda") || !strcmp(argv[i], "--cpu")) use_cuda = 0;
    else {
      fprintf(stderr, "usage: %s --container F [--no-cuda] [--port P] [--bind IP] [--workers N] "
                      "[--noise X] [--default-sid S] [--keepwarm SEC]\n", argv[0]);
      return 1;
    }
  }
  if (nw < 1) nw = 1;
  if (nw > 8) nw = 8;
  if (g_default_sid < 0 || g_default_sid >= N_SID) g_default_sid = 0;
  g_nworkers = nw;
  signal(SIGPIPE, SIG_IGN);
  if (getenv("PT_V21_KW_ITERS")) {
    g_kw_iters = atoi(getenv("PT_V21_KW_ITERS"));
    g_kw_adapt = 0;
  }
  {
    FILE *f = fopen("/sys/devices/platform/host1x/57000000.gpu/devfreq/57000000.gpu/max_freq", "r");
    char b[32];
    if (f && fgets(b, sizeof b, f)) g_gpu_max = atol(b);
    if (f) fclose(f);
  }
  if (getenv("PT_V21_KW_MS")) g_kw_period = atoi(getenv("PT_V21_KW_MS"));
  if (g_kw_period < 10) g_kw_period = 10;
  if (!use_cuda) g_kw_window = 0;

  int ls = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  struct sockaddr_in a;
  memset(&a, 0, sizeof(a));
  a.sin_family = AF_INET;
  a.sin_port = htons((uint16_t) port);
  if (inet_pton(AF_INET, bind_ip, &a.sin_addr) != 1 || bind(ls, (struct sockaddr *) &a, sizeof(a)) ||
      listen(ls, 64)) {
    perror("serve: bind/listen");
    return 1;
  }

  g_t0 = now_ms();
  static worker_arg wa[8];
  for (int i = 0; i < nw; i++) {
    wa[i] = (worker_arg){container, use_cuda, i};
    pthread_t th;
    pthread_create(&th, 0, worker_main, &wa[i]);
    pthread_detach(th);
  }
  g_last_req = now_ms();   /* the start-up itself counts as traffic: stay warm for the first window */
  {
    pthread_t kt;
    pthread_create(&kt, 0, keepwarm_main, 0);
    pthread_detach(kt);
  }
  fprintf(stderr, "serve: listening on %s:%d, starting %d worker(s)%s\n", bind_ip, port, nw,
          use_cuda ? " (CUDA)" : "");
  for (;;) {
    int fd = accept(ls, 0, 0);
    if (fd < 0) continue;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    if (__sync_add_and_fetch(&g_nconn, 1) > MAX_CONN) {
      reply_err(fd, 503, "too many connections");
      close(fd);
      __sync_sub_and_fetch(&g_nconn, 1);
      continue;
    }
    pthread_t th;
    if (pthread_create(&th, 0, conn_main, (void *) (intptr_t) fd)) {
      close(fd);
      __sync_sub_and_fetch(&g_nconn, 1);
      continue;
    }
    pthread_detach(th);
  }
}
