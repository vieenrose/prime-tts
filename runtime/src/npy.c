/* npy.c - minimal .npy reader/writer (little-endian, C order; float32 / int64 / bool). Tests & tools only. */
#include "pt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int pt_npy_load(pt_npy *a, const char *path) {
  memset(a, 0, sizeof(*a));
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  unsigned char pre[10];
  if (fread(pre, 1, 10, f) != 10 || memcmp(pre, "\x93NUMPY", 6)) { fclose(f); return -1; }
  unsigned hl;
  if (pre[6] == 1) hl = pre[8] | (pre[9] << 8);
  else {
    unsigned char b[2];
    if (fread(b, 1, 2, f) != 2) { fclose(f); return -1; }
    hl = pre[8] | (pre[9] << 8) | (b[0] << 16) | (b[1] << 24);
  }
  char *h = malloc(hl + 1);
  if (fread(h, 1, hl, f) != hl) { free(h); fclose(f); return -1; }
  h[hl] = 0;
  const char *d = strstr(h, "'descr':");
  if (!d || strstr(h, "'fortran_order': True")) { free(h); fclose(f); return -1; }
  size_t esz;
  if (strstr(d, "<f4")) { a->dtype = 'f'; esz = 4; }
  else if (strstr(d, "<i8")) { a->dtype = 'i'; esz = 8; }
  else if (strstr(d, "|b1")) { a->dtype = 'b'; esz = 1; }
  else { fprintf(stderr, "npy: unsupported dtype in %s\n", path); free(h); fclose(f); return -1; }
  const char *s = strstr(h, "'shape':");
  s = strchr(s, '(') + 1;
  a->n = 1;
  while (*s && *s != ')') {
    while (*s == ' ' || *s == ',') s++;
    if (*s == ')') break;
    long v = strtol(s, (char **)&s, 10);
    a->dim[a->ndim++] = v;
    a->n *= v;
  }
  free(h);
  a->data = malloc(a->n * esz + 64);
  if (fread(a->data, esz, a->n, f) != (size_t)a->n) { fclose(f); pt_npy_free(a); return -1; }
  fclose(f);
  return 0;
}

void pt_npy_free(pt_npy *a) { free(a->data); a->data = 0; }

int pt_npy_save_f32(const char *path, const float *d, int ndim, const long *dim) {
  FILE *f = fopen(path, "wb");
  if (!f) return -1;
  char hdr[256], shp[128] = "";
  long n = 1;
  for (int i = 0; i < ndim; i++) {
    char b[32];
    snprintf(b, sizeof b, "%ld,%s", dim[i], (ndim == 1 || i < ndim - 1) ? (i < ndim - 1 ? " " : "") : "");
    strcat(shp, b);
    n *= dim[i];
  }
  int hl = snprintf(hdr, sizeof hdr, "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }", shp);
  int tot = 10 + hl + 1;
  int pad = (64 - tot % 64) % 64;
  fwrite("\x93NUMPY\x01\x00", 1, 8, f);
  unsigned short h2 = (unsigned short)(hl + pad + 1);
  fwrite(&h2, 2, 1, f);
  fwrite(hdr, 1, hl, f);
  for (int i = 0; i < pad; i++) fputc(' ', f);
  fputc('\n', f);
  fwrite(d, 4, n, f);
  fclose(f);
  return 0;
}
