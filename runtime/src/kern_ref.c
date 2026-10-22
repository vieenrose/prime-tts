/* kern_ref.c - portable reference implementations of the micro-kernels (contract: pt.h).
 * Used (1) as the oracle in tests/test_kernels.c and (2) as the implementation on non-aarch64 hosts.
 * Plain scalar C on purpose: exact libm activations, no reassociation tricks.
 */
#include "pt.h"
#include <math.h>

void pt_ref_k4x16(float *Y, long ldy, const float *A, const float *X, const long *off, long nk, long ntile,
                  const float *bias) {
  long ldyf = ldy / 4;
  for (long tt = 0; tt < ntile; tt++) {
    float acc[4][16];
    for (int i = 0; i < 4; i++)
      for (int t = 0; t < 16; t++) acc[i][t] = bias ? bias[i] : 0.f;
    for (long j = 0; j < nk; j++) {
      const float *x = (const float *)((const char *)X + off[j]) + tt * 16;
      for (int i = 0; i < 4; i++) {
        float a = A[j * 4 + i];
        for (int t = 0; t < 16; t++) acc[i][t] += a * x[t];
      }
    }
    for (int i = 0; i < 4; i++)
      for (int t = 0; t < 16; t++) Y[i * ldyf + tt * 16 + t] = acc[i][t];
  }
}

void pt_ref_k8x12(float *Y, long ldy, const float *A, const float *X, const long *off, long nk, long ntile,
                  const float *bias) {
  long ldyf = ldy / 4;
  for (long tt = 0; tt < ntile; tt++) {
    float acc[8][12];
    for (int i = 0; i < 8; i++)
      for (int t = 0; t < 12; t++) acc[i][t] = bias ? bias[i] : 0.f;
    for (long j = 0; j < nk; j++) {
      const float *x = (const float *)((const char *)X + off[j]) + tt * 12;
      for (int i = 0; i < 8; i++) {
        float a = (i < 4) ? A[j * 4 + i] : A[nk * 4 + j * 4 + (i - 4)];
        for (int t = 0; t < 12; t++) acc[i][t] += a * x[t];
      }
    }
    for (int i = 0; i < 8; i++)
      for (int t = 0; t < 12; t++) Y[i * ldyf + tt * 12 + t] = acc[i][t];
  }
}

void pt_ref_snake_row(float *y, long n, float a, float ia) {
  for (long i = 0; i < n; i++) {
    float s = sinf(a * y[i]);
    y[i] = y[i] + s * s * ia;
  }
}

void pt_ref_silu_row(float *y, long n) {
  for (long i = 0; i < n; i++) y[i] = y[i] / (1.f + expf(-y[i]));
}

void pt_ref_tanh_row(float *y, long n) {
  for (long i = 0; i < n; i++) y[i] = tanhf(y[i]);
}

void pt_ref_sigmul_row(float *y, const float *g, long n) {
  for (long i = 0; i < n; i++) y[i] = y[i] * (1.f / (1.f + expf(-g[i])));
}

void pt_ref_dw_row(float *y, const float *x, long n, const float *w, long K, float b) {
  for (long t = 0; t < n; t++) {
    float s = b;
    for (long k = 0; k < K; k++) s += w[k] * x[t + k];
    y[t] = s;
  }
}

void pt_ref_ln_res(float *x, const float *r, float *y, long C, long ld, long n, const float *g, const float *b,
                   float eps) {
  for (long t = 0; t < n; t++) {
    float s = 0.f;
    for (long c = 0; c < C; c++) {
      if (r) x[c * ld + t] += r[c * ld + t];
      s += x[c * ld + t];
    }
    float mean = s / (float)C, v = 0.f;
    for (long c = 0; c < C; c++) {
      float d = x[c * ld + t] - mean;
      v += d * d;
    }
    float inv = 1.f / sqrtf(v / (float)C + eps);
    for (long c = 0; c < C; c++) y[c * ld + t] = (x[c * ld + t] - mean) * inv * g[c] + b[c];
  }
}

#if !defined(__aarch64__)
/* Non-aarch64 hosts: the public kernels are the reference ones. */
void pt_k4x16(float *Y, long ldy, const float *A, const float *X, const long *off, long nk, long ntile,
              const float *bias) { pt_ref_k4x16(Y, ldy, A, X, off, nk, ntile, bias); }
void pt_k8x12(float *Y, long ldy, const float *A, const float *X, const long *off, long nk, long ntile,
              const float *bias) { pt_ref_k8x12(Y, ldy, A, X, off, nk, ntile, bias); }
void pt_snake_row(float *y, long n, float a, float ia) { pt_ref_snake_row(y, n, a, ia); }
void pt_silu_row(float *y, long n) { pt_ref_silu_row(y, n); }
void pt_tanh_row(float *y, long n) { pt_ref_tanh_row(y, n); }
void pt_sigmul_row(float *y, const float *g, long n) { pt_ref_sigmul_row(y, g, n); }
void pt_dw_row(float *y, const float *x, long n, const float *w, long K, float b) { pt_ref_dw_row(y, x, n, w, K, b); }
void pt_ln_res(float *x, const float *r, float *y, long C, long ld, long n, const float *g, const float *b,
               float eps) { pt_ref_ln_res(x, r, y, C, ld, n, g, b, eps); }
#endif
