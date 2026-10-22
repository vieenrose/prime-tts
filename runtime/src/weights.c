/* weights.c - mmap a .ptk container; arena scratch allocator.
 * mmap matters on this box: weights stay in page cache, shared by all 8 streams,
 * so RAM cost per stream is activations only (we have ~185 MB free).
 */
#define _GNU_SOURCE
#include "pt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#pragma pack(push, 1)
typedef struct {
  uint32_t magic, version, n_tensor;
  int32_t reserved;
  int64_t data_bytes;
} pt_hdr;
#pragma pack(pop)

_Static_assert(sizeof(pt_tensor) == 136, "pt_tensor record must match tools/export_ptk.py (<64si4x6qqq)");

int pt_weights_open(pt_weights *w, const char *path) {
  memset(w, 0, sizeof(*w));
  w->fd = -1;
  int fd = open(path, O_RDONLY);
  if (fd < 0) { perror(path); return -1; }
  struct stat st;
  pt_hdr h;
  if (fstat(fd, &st) || st.st_size < (off_t)sizeof(pt_hdr) || read(fd, &h, sizeof(h)) != (ssize_t)sizeof(h)) {
    close(fd); return -1;
  }
  if (h.magic != PT_MAGIC) { fprintf(stderr, "pt: bad magic %08x in %s\n", h.magic, path); close(fd); return -1; }
  if (h.n_tensor > PT_MAX_TENSOR) { fprintf(stderr, "pt: %u tensors > cap\n", h.n_tensor); close(fd); return -1; }
  size_t recb = (size_t)h.n_tensor * sizeof(pt_tensor);
  if (read(fd, w->t, recb) != (ssize_t)recb) { close(fd); return -1; }
  size_t hd = sizeof(pt_hdr) + recb;
  size_t hdp = (hd + 63) & ~(size_t)63;          /* data starts 64-byte aligned */
  w->maplen = hdp + (size_t)h.data_bytes;
  if ((size_t)st.st_size < w->maplen) { fprintf(stderr, "pt: truncated container %s\n", path); close(fd); return -1; }
  w->map = mmap(0, w->maplen, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
  if (w->map == MAP_FAILED) { perror("mmap"); close(fd); w->map = 0; return -1; }
  w->fd = fd;
  w->magic = h.magic;
  w->version = h.version;
  w->n_tensor = h.n_tensor;
  w->data_bytes = h.data_bytes;
  w->base = (char *)w->map + hdp;
  return 0;
}

void pt_weights_close(pt_weights *w) {
  if (w->map) munmap(w->map, w->maplen);
  if (w->fd >= 0) close(w->fd);
  memset(w, 0, sizeof(*w));
  w->fd = -1;
}

const pt_tensor *pt_find(const pt_weights *w, const char *name) {
  for (uint32_t i = 0; i < w->n_tensor; i++)
    if (!strcmp(w->t[i].name, name)) return &w->t[i];
  return 0;
}

const float *pt_f32(const pt_weights *w, const char *name) {
  const pt_tensor *t = pt_find(w, name);
  return t ? (const float *)((const char *)w->base + t->off) : 0;
}

int pt_arena_init(pt_arena *a, size_t bytes) {
  a->cap = (bytes + 63) & ~(size_t)63;
  a->p = aligned_alloc(64, a->cap);
  a->used = a->peak = 0;
  return a->p ? 0 : -1;
}

void pt_arena_free(pt_arena *a) { free(a->p); a->p = 0; a->cap = a->used = 0; }

void *pt_arena_alloc(pt_arena *a, size_t bytes) {
  size_t b = (bytes + 63) & ~(size_t)63;
  if (a->used + b > a->cap) {
    fprintf(stderr, "pt: arena exhausted (need %zu, used %zu of %zu)\n", b, a->used, a->cap);
    return 0;
  }
  void *p = (char *)a->p + a->used;
  a->used += b;
  if (a->used > a->peak) a->peak = a->used;
  return p;
}
