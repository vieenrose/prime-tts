/* test_kernels.c - every micro-kernel and the conv driver vs naive references.
 * (a) asm (or host) kernels vs pt_ref_* on random data,  (b) pt_conv vs a direct triple loop,
 * (c) numpy-generated vectors (tests/kern/*.npy from tests/gen_kernel_vectors.py) when present.
 * Prints max abs / relative error per kernel and GFLOP/s of the conv path. */
#include "pt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned rs = 12345;
static float frand(void) { rs = rs * 1103515245u + 12345u; return ((rs >> 8) & 0xffff) / 32768.0f - 1.0f; }
static float *rnd(long n, float sc) { float *p = aligned_alloc(64, ((n * 4 + 63) / 64) * 64 + 256); for (long i = 0; i < n; i++) p[i] = frand() * sc; return p; }
static int fails = 0;
static void report(const char *nm, const float *a, const float *b, long n, double tol) {
  double mx = 0, ma = 0;
  for (long i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (!(d <= mx)) mx = d; if (fabs(b[i]) > ma) ma = fabs(b[i]); }
  double rel = mx / (ma > 0 ? ma : 1);
  int ok = rel <= tol;
  fails += !ok;
  printf("  %-34s n=%7ld maxabs=%.3e rel=%.3e %s\n", nm, n, mx, rel, ok ? "ok" : "FAIL");
}

/* pack W[co][ci][k] like tools/export_ptk.py::pack_conv */
static float *pack(const float *W, int Cout, int Cin, int K) {
  int C4 = (Cout + 3) / 4 * 4;
  float *p = calloc((size_t)C4 * Cin * K, 4);
  for (int co = 0; co < Cout; co++)
    for (int ci = 0; ci < Cin; ci++)
      for (int k = 0; k < K; k++) p[(((size_t)(co / 4) * K + k) * Cin + ci) * 4 + co % 4] = W[((size_t)co * Cin + ci) * K + k];
  return p;
}

static void test_conv(int Cout, int Cin, int K, int dil, long n, int act, int resid, int threads) {
  long pad = (long)dil * (K - 1) / 2, ldx = n + 2 * pad + 64, ldy = n + 64;
  float *X = rnd((long)Cin * ldx, 1.0f), *Wt = rnd((long)Cout * Cin * K, 0.3f), *B = rnd((Cout + 3) / 4 * 4, 0.5f);
  for (int c = Cout; c < (Cout + 3) / 4 * 4; c++) B[c] = 0;
  float *Y = calloc((size_t)((Cout + 3) / 4 * 4) * ldy + 64, 4), *R = calloc((size_t)Cout * n, 4);
  float *res = rnd((long)((Cout + 3) / 4 * 4) * ldy, 1.0f);
  float *a = rnd(Cout + 4, 1.0f), *ia = malloc(4 * (Cout + 4));
  for (int c = 0; c < Cout + 4; c++) { a[c] = 0.2f + fabsf(a[c]); ia[c] = 1.0f / a[c]; }
  for (int co = 0; co < Cout; co++)
    for (long t = 0; t < n; t++) {
      double s = B[co];
      for (int ci = 0; ci < Cin; ci++)
        for (int k = 0; k < K; k++) s += (double)Wt[((size_t)co * Cin + ci) * K + k] * X[ci * ldx + t + k * dil];
      float y = (float)s;
      if (act == PT_ACT_SILU) y = y / (1 + expf(-y));
      if (act == PT_ACT_TANH) y = tanhf(y);
      if (act == PT_ACT_SNAKE) { float q = sinf(a[co] * y); y = y + q * q * ia[co]; }
      if (resid) y = res[co * ldy + t] + y;
      R[co * n + t] = y;
    }
  pt_conv_w cw = {pack(Wt, Cout, Cin, K), B, Cout, (Cout + 3) / 4 * 4, Cin, K, dil};
  pt_epi e = {act, a, ia, 1.0f, resid ? res : 0, ldy};
  if (resid) memcpy(Y, res, sizeof(float) * ((Cout + 3) / 4 * 4) * ldy), e.res = Y; /* in-place residual */
  pt_pool_set_local(threads);
  pt_conv(&cw, X, ldx, Y, ldy, n, (act || resid) ? &e : 0);
  /* timing */
  int reps = 0;
  double t0 = pt_now(), t1;
  float *Y2 = calloc((size_t)((Cout + 3) / 4 * 4) * ldy + 64, 4);
  do { pt_conv(&cw, X, ldx, Y2, ldy, n, 0); reps++; t1 = pt_now(); } while (t1 - t0 < 0.2);
  pt_pool_set_local(1);
  double gf = 2.0 * Cout * Cin * K * n * reps / (t1 - t0) / 1e9;
  float *Yc = malloc(sizeof(float) * Cout * n);
  for (int co = 0; co < Cout; co++) memcpy(Yc + co * n, Y + co * ldy, n * 4);
  char nm[96];
  snprintf(nm, sizeof nm, "conv %dx%dxK%d d%d n%ld a%d r%d t%d", Cout, Cin, K, dil, n, act, resid, threads);
  report(nm, Yc, R, (long)Cout * n, act == PT_ACT_NONE ? 2e-5 : 2e-4);
  printf("      -> %.2f GFLOP/s\n", gf);
  free(X); free(Wt); free(B); free(Y); free(Y2); free(R); free(res); free(a); free(ia); free(Yc); free((void *)cw.wp);
}

/* ---------------- numpy vectors (tests/gen_kernel_vectors.py) ---------------- */
static int ld_npy(pt_npy *a, const char *dir, const char *case_, const char *nm) {
  char p[512];
  snprintf(p, sizeof p, "%s/%s/%s.npy", dir, case_, nm);
  return pt_npy_load(a, p);
}

static void numpy_conv(const char *dir, const char *cs) {
  pt_npy X, Wn, B, A, R, Y, P;
  if (ld_npy(&X, dir, cs, "X") || ld_npy(&Wn, dir, cs, "W") || ld_npy(&B, dir, cs, "b") || ld_npy(&A, dir, cs, "a") ||
      ld_npy(&R, dir, cs, "R") || ld_npy(&Y, dir, cs, "Y") || ld_npy(&P, dir, cs, "p")) { printf("  %s: missing\n", cs); fails++; return; }
  long *p = P.data;
  int Co = (int)p[0], Ci = (int)p[1], K = (int)p[2], d = (int)p[3], act = (int)p[5], res = (int)p[6];
  long n = p[4], span = (long)(K - 1) * d, C4 = (Co + 3) / 4 * 4;
  long ldx = n + span + 64, ldy = n + 64;
  float *x = calloc(Ci * ldx, 4), *y = calloc(C4 * ldy, 4), *bias = calloc(C4, 4), *a = calloc(C4 + 4, 4), *ia = calloc(C4 + 4, 4);
  for (int c = 0; c < Ci; c++) memcpy(x + c * ldx, (float *)X.data + c * (n + span), (n + span) * 4);
  memcpy(bias, B.data, Co * 4);
  for (int c = 0; c < Co; c++) { a[c] = ((float *)A.data)[c]; ia[c] = 1.0f / a[c]; }
  if (res) for (int c = 0; c < Co; c++) memcpy(y + c * ldy, (float *)R.data + c * n, n * 4);
  pt_conv_w cw = {pack(Wn.data, Co, Ci, K), bias, Co, (int)C4, Ci, K, d};
  pt_epi e = {act, a, ia, 1.0f, res ? y : 0, ldy};
  pt_conv(&cw, x, ldx, y, ldy, n, (act || res) ? &e : 0);
  float *yc = malloc(Co * n * 4);
  for (int c = 0; c < Co; c++) memcpy(yc + c * n, y + c * ldy, n * 4);
  char nm[96];
  snprintf(nm, sizeof nm, "npy %s", cs);
  report(nm, yc, Y.data, Co * n, 3e-5);
  free(x); free(y); free(bias); free(a); free(ia); free(yc); free((void *)cw.wp);
  pt_npy_free(&X); pt_npy_free(&Wn); pt_npy_free(&B); pt_npy_free(&A); pt_npy_free(&R); pt_npy_free(&Y); pt_npy_free(&P);
}

static void numpy_vectors(const char *dir) {
  char mf[512];
  snprintf(mf, sizeof mf, "%s/MANIFEST.json", dir);
  FILE *f = fopen(mf, "r");
  if (!f) { printf("numpy vectors: %s not found (run tests/gen_kernel_vectors.py) - skipped\n", dir); return; }
  char buf[4096];
  size_t nb = fread(buf, 1, sizeof buf - 1, f);
  buf[nb] = 0;
  fclose(f);
  printf("numpy reference vectors (%s):\n", dir);
  char *lst = strchr(buf, '[');
  for (char *q = lst ? strstr(lst, "\"conv_") : 0; q; q = strstr(q + 1, "\"conv_")) {
    char cs[64];
    if (sscanf(q + 1, "%63[^\"]", cs) == 1) numpy_conv(dir, cs);
  }
  pt_npy x, g, ys, yl, yt, yg, dx, dwv, dy, lx, lr, lg, lb, lxs, ly;
  if (!ld_npy(&x, dir, "act", "x") && !ld_npy(&g, dir, "act", "g") && !ld_npy(&ys, dir, "act", "snake") &&
      !ld_npy(&yl, dir, "act", "silu") && !ld_npy(&yt, dir, "act", "tanh") && !ld_npy(&yg, dir, "act", "sigmul")) {
    long n = x.n;
    float *v = malloc(n * 4);
    memcpy(v, x.data, n * 4); pt_snake_row(v, n, 0.83f, 1.0f / 0.83f); report("npy snake_row", v, ys.data, n, 3e-6);
    memcpy(v, x.data, n * 4); pt_silu_row(v, n); report("npy silu_row", v, yl.data, n, 3e-6);
    for (long i = 0; i < n; i++) v[i] = ((float *)x.data)[i] / 4;
    pt_tanh_row(v, n); report("npy tanh_row", v, yt.data, n, 3e-6);
    memcpy(v, x.data, n * 4); pt_sigmul_row(v, g.data, n); report("npy sigmul_row (GLU)", v, yg.data, n, 3e-6);
    free(v);
  } else fails++;
  if (!ld_npy(&dx, dir, "dw", "x") && !ld_npy(&dwv, dir, "dw", "w") && !ld_npy(&dy, dir, "dw", "y")) {
    float *v = malloc(dy.n * 4 + 64);
    pt_dw_row(v, dx.data, dy.n, dwv.data, 7, 0.25f);
    report("npy dw_row K7", v, dy.data, dy.n, 3e-6);
    free(v);
  } else fails++;
  if (!ld_npy(&lx, dir, "ln", "x") && !ld_npy(&lr, dir, "ln", "r") && !ld_npy(&lg, dir, "ln", "g") &&
      !ld_npy(&lb, dir, "ln", "b") && !ld_npy(&lxs, dir, "ln", "xsum") && !ld_npy(&ly, dir, "ln", "y")) {
    long C = lx.dim[0], n = lx.dim[1];
    float *xx = malloc(C * n * 4), *yy = malloc(C * n * 4);
    memcpy(xx, lx.data, C * n * 4);
    pt_ln_res(xx, lr.data, yy, C, n, n, lg.data, lb.data, 1e-5f);
    report("npy ln_res (x += r)", xx, lxs.data, C * n, 1e-6);
    report("npy ln_res (y)", yy, ly.data, C * n, 3e-5);
    free(xx); free(yy);
  } else fails++;
}

int main(int argc, char **argv) {
  pt_pool_init(4);
  numpy_vectors(argc > 1 ? argv[1] : "tests/kern");
  printf("micro-kernels vs reference:\n");
  {
    long nk = 300, nt = 5, ld = 16 * nt + 32;
    float *A = rnd(nk * 8, 1), *X = rnd(nk * 4 + 16 * nt * 2 + 64, 1), *bias = rnd(8, 1);
    long *off = malloc(nk * 8);
    for (long j = 0; j < nk; j++) off[j] = ((j * 7) % (nk * 4)) * 4;
    float *Y1 = calloc(8 * ld, 4), *Y2 = calloc(8 * ld, 4);
    pt_k4x16(Y1, ld * 4, A, X, off, nk, nt, bias);
    pt_ref_k4x16(Y2, ld * 4, A, X, off, nk, nt, bias);
    report("k4x16", Y1, Y2, 4 * ld, 1e-5);
    memset(Y1, 0, 32 * ld); memset(Y2, 0, 32 * ld);
    long nt12 = 6;
    pt_k8x12(Y1, ld * 4, A, X, off, nk, nt12, bias);
    pt_ref_k8x12(Y2, ld * 4, A, X, off, nk, nt12, bias);
    report("k8x12", Y1, Y2, 8 * ld, 1e-5);
  }
  {
    long n = 4096;
    float *x = rnd(n, 30.f), *y1 = malloc(n * 4), *y2 = malloc(n * 4), *g = rnd(n, 12.f);
    memcpy(y1, x, n * 4); memcpy(y2, x, n * 4);
    pt_snake_row(y1, n, 0.73f, 1 / 0.73f); pt_ref_snake_row(y2, n, 0.73f, 1 / 0.73f); report("snake_row |x|<30", y1, y2, n, 2e-6);
    memcpy(y1, x, n * 4); memcpy(y2, x, n * 4);
    pt_snake_row(y1, n, 37.f, 1 / 37.f); pt_ref_snake_row(y2, n, 37.f, 1 / 37.f); report("snake_row a=37", y1, y2, n, 2e-6);
    memcpy(y1, x, n * 4); memcpy(y2, x, n * 4);
    pt_silu_row(y1, n); pt_ref_silu_row(y2, n); report("silu_row", y1, y2, n, 2e-6);
    for (long i = 0; i < n; i++) x[i] *= 0.2f;
    memcpy(y1, x, n * 4); memcpy(y2, x, n * 4);
    pt_tanh_row(y1, n); pt_ref_tanh_row(y2, n); report("tanh_row", y1, y2, n, 2e-6);
    memcpy(y1, x, n * 4); memcpy(y2, x, n * 4);
    pt_sigmul_row(y1, g, n); pt_ref_sigmul_row(y2, g, n); report("sigmul_row", y1, y2, n, 2e-6);
    float w[7] = {0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.05f, 0.9f};
    pt_dw_row(y1, x, n - 16, w, 7, 0.3f); pt_ref_dw_row(y2, x, n - 16, w, 7, 0.3f); report("dw_row K7", y1, y2, n - 16, 2e-6);
    long C = 168, ld = 128 + 16, nn = 128;
    float *xa = rnd(C * ld, 2), *xb = malloc(C * ld * 4), *r = rnd(C * ld, 1), *ya = calloc(C * ld, 4), *yb = calloc(C * ld, 4); /* unwritten tail columns compare equal */
    float *gg = rnd(C, 1), *bb = rnd(C, 1);
    memcpy(xb, xa, C * ld * 4);
    pt_ln_res(xa, r, ya, C, ld, nn, gg, bb, 1e-5f); pt_ref_ln_res(xb, r, yb, C, ld, nn, gg, bb, 1e-5f);
    report("ln_res (x += r)", xa, xb, C * ld, 1e-6);
    report("ln_res (y)", ya, yb, C * ld, 2e-5);
    pt_ln_res(xa, 0, ya, C, ld, nn, gg, bb, 1e-5f); pt_ref_ln_res(xb, 0, yb, C, ld, nn, gg, bb, 1e-5f);
    report("ln (no residual)", ya, yb, C * ld, 2e-5);
  }
  printf("conv driver vs direct loop:\n");
  test_conv(168, 168, 1, 1, 81, PT_ACT_NONE, 0, 1);
  test_conv(672, 168, 1, 1, 81, PT_ACT_SILU, 0, 1);
  test_conv(168, 672, 1, 1, 530, PT_ACT_NONE, 1, 1);
  test_conv(96, 178, 5, 1, 300, PT_ACT_SILU, 0, 1);
  test_conv(80, 168, 5, 1, 300, PT_ACT_TANH, 0, 1);
  test_conv(48, 48, 11, 5, 2000, PT_ACT_SNAKE, 0, 1);
  test_conv(48, 48, 11, 1, 2000, PT_ACT_NONE, 1, 1);
  test_conv(24, 24, 7, 3, 4000, PT_ACT_SNAKE, 0, 1);
  test_conv(12, 12, 11, 5, 8000, PT_ACT_NONE, 1, 1);
  test_conv(6, 6, 3, 1, 8000, PT_ACT_SNAKE, 0, 1);
  test_conv(384, 96, 2, 1, 300, PT_ACT_NONE, 0, 1);
  test_conv(1, 6, 7, 1, 4000, PT_ACT_TANH, 0, 1);
  test_conv(48, 48, 11, 5, 2000, PT_ACT_SNAKE, 0, 4);
  test_conv(504, 168, 1, 1, 530, PT_ACT_NONE, 0, 4);
  pt_conv_set_pack(0);
  test_conv(168, 672, 1, 1, 530, PT_ACT_NONE, 1, 1);   /* unpacked path for the same shapes */
  test_conv(672, 168, 1, 1, 81, PT_ACT_SILU, 0, 1);
  pt_conv_set_pack(1);
  test_conv(84, 168, 1, 1, 77, PT_ACT_SILU, 0, 1);      /* odd number of 4-row blocks: packed + leftover */
  pt_pool_shutdown();
  printf(fails ? "KERNELS: FAIL (%d)\n" : "KERNELS: PASS\n", fails);
  return fails != 0;
}
