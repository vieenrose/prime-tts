#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Unit test of the window kernel's cooperative weight staging:
 * weight layout = float4[cout4/4 groups][K*cin], the 4 lanes are the group's 4 channels.
 * Staging (from pt_conv_kernel_win): i in [0, 16*8*11), k = i%11, c = (i/11)&7, ci = i/(11*8),
 * value = ((c&4) ? w4 + kc : w4)[k*cin + ci] lane (c&3). */
__global__ void stage(const float *__restrict__ wp, float *__restrict__ out, int cin, int K) {
  __shared__ float Ws[16][8][11];
  const int tid = threadIdx.x;
  const int kc = K * cin;
  const float4 *w4 = (const float4 *)wp;
  for (int i = tid; i < 16 * 8 * 11; i += 128) {
    int k = i % 11, c = (i / 11) & 7, ci = i / (11 * 8);
    float v = 0.f;
    if (k < K) {
      const float4 *w = ((c & 4) ? w4 + kc : w4) + k * cin + ci;
      v = (&w->x)[c & 3];
    }
    Ws[ci][c][k] = v;
  }
  __syncthreads();
  for (int i = tid; i < 16 * 8 * 11; i += 128) {
    int k = i % 11, c = (i / 11) & 7, ci = i / (11 * 8);
    out[i] = Ws[ci][c][k];
  }
}

int main(void) {
  int cin = 8, K = 3, kc = K * cin;
  int ng = 2;   /* 8 channels in groups of 4 */
  float *hw = new float[ng * kc * 4];
  for (int g = 0; g < ng; g++)
    for (int r = 0; r < kc; r++)
      for (int j = 0; j < 4; j++)
        hw[(g * kc + r) * 4 + j] = (float)(g * 1000 + r * 10 + j);
  float *dw, *do_;
  cudaMalloc(&dw, sizeof(float) * ng * kc * 4);
  cudaMalloc(&do_, sizeof(float) * 16 * 8 * 11);
  cudaMemcpy(dw, hw, sizeof(float) * ng * kc * 4, cudaMemcpyHostToDevice);
  stage<<<1, 128>>>(dw, do_, cin, K);
  float *ho = new float[16 * 8 * 11];
  cudaMemcpy(ho, do_, sizeof(float) * 16 * 8 * 11, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int ci = 0; ci < 16; ci++)
    for (int c = 0; c < 8; c++)
      for (int k = 0; k < 11; k++) {
        float exp = 0.f;
        if (k < K && ci < cin) {
          int g = c >> 2, r = k * cin + ci, j = c & 3;
          exp = hw[(g * kc + r) * 4 + j];
        }
        float got = ho[ci * 88 + c * 11 + k];
        if (got != exp) {
          if (bad < 6)
            printf("MISMATCH ci=%d c=%d k=%d got %.0f exp %.0f\n", ci, c, k, got, exp);
          bad++;
        }
      }
  printf("staging mismatches: %d of %d\n", bad, 16 * 8 * 11);
  return 0;
}
