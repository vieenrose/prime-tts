/* pt_v1b8k.h - PrimeTTS v1 CPU runtime API (v1b_8k and v1b_16k: same encoder/decoder/vocoder topology,
 * the stage geometry, hop size and sample rate come from the .ptk container). */
#ifndef PT_V1B8K_H_
#define PT_V1B8K_H_
#include "pt.h"

typedef struct pt_v1b8k_model pt_v1b8k_model;
typedef struct pt_v1b8k_stream pt_v1b8k_stream;
typedef struct pt_v1b8k_ws pt_v1b8k_ws;   /* encoder+decoder workspace (needed only until the mel is ready) */

/* Called for every finished PCM chunk (float, [-1,1], mono, at the model's sample rate).
 * Return non-zero to cancel (barge-in). */
typedef int (*pt_pcm_cb)(void *user, const float *pcm, int n);

/* Optional debug tap: called with intermediate tensors (channel-major [C][ld], n valid columns). */
typedef void (*pt_tap_cb)(void *user, const char *name, const float *x, int C, long ld, long n);

typedef struct {
  int T, F;                 /* phones, frames */
  long samples;
  double t_frontend_s;      /* encoder + regulator + decoder (mel ready) */
  double t_first_chunk_s;   /* from synth start to first PCM chunk delivered */
  double t_total_s;
  long vocoder_cols_computed, vocoder_cols_useful; /* stage-0 resblock columns computed / output frames */
} pt_synth_info;

pt_v1b8k_model *pt_v1b8k_load(const char *ptk_path);
void pt_v1b8k_free(pt_v1b8k_model *m);
int pt_v1b8k_sample_rate(const pt_v1b8k_model *m);   /* container model.meta: 8000 | 16000 */
int pt_v1b8k_hop(const pt_v1b8k_model *m);           /* samples per mel frame = product of stage strides */

/* A stream owns all activation memory (one arena, sized from the limits). One synth at a time per stream.
 * chunk_frames: vocoder chunk size in mel frames (1 frame = 1/hop of a second, 16 ms in v1); first chunk is 3 frames. */
pt_v1b8k_stream *pt_v1b8k_stream_new(const pt_v1b8k_model *m, int max_phones, int max_frames, int chunk_frames);
void pt_v1b8k_stream_free(pt_v1b8k_stream *s);
/* Memory-lean variant for servers: the stream owns only mel + vocoder state (~1-3 MB); the encoder/decoder
 * workspace (~9 MB at 256 phones / 1000 frames) is borrowed from a pool for the mel phase of each synth. */
pt_v1b8k_stream *pt_v1b8k_stream_new_shared(const pt_v1b8k_model *m, int max_phones, int max_frames, int chunk_frames);
void pt_v1b8k_set_ws_pool(pt_v1b8k_stream *s, pt_v1b8k_ws *(*acquire)(void *), void (*release)(void *, pt_v1b8k_ws *),
                          void *user);
pt_v1b8k_ws *pt_v1b8k_ws_new(const pt_v1b8k_model *m, int max_phones, int max_frames);
void pt_v1b8k_ws_free(pt_v1b8k_ws *ws);
size_t pt_v1b8k_ws_bytes(const pt_v1b8k_ws *ws);
size_t pt_v1b8k_stream_bytes(const pt_v1b8k_stream *s);
void pt_v1b8k_set_tap(pt_v1b8k_stream *s, pt_tap_cb cb, void *user);
/* first_chunk_frames (default 3), chunk growth doubles until chunk_frames */
void pt_v1b8k_set_chunking(pt_v1b8k_stream *s, int first_chunk_frames, int chunk_frames);

/* phone/tone/lang ids (frontend output, no blanks), sid in {0,1}. Streams PCM through cb.
 * Returns 0 on success, 1 if cancelled by cb, <0 on error (e.g. limits exceeded). */
int pt_v1b8k_synth(pt_v1b8k_stream *s, const int32_t *phone, const int32_t *tone, const int32_t *lang, int T,
                   int sid, pt_pcm_cb cb, void *user, pt_synth_info *info);

/* Exact C port of model/scripts/synth_from_text.py::host_regulate (exposed for tests).
 * cond: [168][ldc] channel-major (T columns); dur[T]; pitch: [2][ldp]. Outputs channel-major with
 * row stride ld (columns 0..F-1): frames[168], frame_meta[8], local_ctx_raw[504] (may be NULL),
 * abs_pos[F], pitch_frame[2]. Returns F. */
int pt_v1b8k_regulate(const float *cond, long ldc, const int64_t *dur, const float *pitch, long ldp, int T,
                      int abs_bins, int max_frames, float *frames, float *frame_meta, float *local_ctx_raw,
                      int32_t *abs_pos, float *pitch_frame, long ld, int32_t *tok_out);

/* env PT_PROF=1: accumulate a per-phase time profile (single stream); print + reset with: */
void pt_v1b8k_prof_dump(void);

/* ---- multi-stream server (runtime/src/sched.c) ---- */
typedef struct {
  const int32_t *phone, *tone, *lang;
  int T, sid;
  pt_pcm_cb cb;
  void *user;
  double arrival_s;         /* pt_now() timestamp when the request arrived */
  /* outputs */
  pt_synth_info info;
  int status;               /* 0 ok, 1 cancelled, <0 error, 2 rejected by admission */
  double t_admit_s, t_first_audio_s, t_done_s;  /* absolute pt_now() timestamps */
  double max_lateness_s;    /* worst (chunk ready time - playback deadline), <= 0 means always in time */
} pt_v1b8k_request;

typedef struct {
  int ncores;               /* worker threads == streams in flight; 1 synth per worker, pinned 1 per core */
  int max_streams;          /* admission cap (<= 8) */
  double capacity_rt;       /* measured audio-seconds per wall-second per core (1 / single-stream RTF) */
  double safety;            /* admit while sum(demand) <= ncores * capacity_rt * safety  */
  int chunk_frames;
  int pin;                  /* pin worker i to core i % ncores */
  int n_workspaces;         /* encoder+decoder workspaces shared by all streams (default ncores) */
  int max_phones, max_frames;
} pt_v1b8k_sched_cfg;

/* Serve n requests (in arrival order) with ncores workers; each admitted request streams to its cb in
 * real time order. Requests that would overload capacity wait in FIFO until load drops. Returns 0. */
int pt_v1b8k_run_batch(const pt_v1b8k_model *m, const pt_v1b8k_sched_cfg *cfg, pt_v1b8k_request *req, int n);

#endif
