#include <cstdio>
#include <cuda_runtime.h>

/* Minimal window test: cin=8, cout=8, K=3, pad_in=1, x = 1 everywhere, weights = 1.
 * Expected y[t] = number of taps k with s = t + k - pad_in inside [0, ldx). */
__global__ void win(const float *__restrict__ x, int64_t ldx, int cin, int K, int pad_in,
                    float *__restrict__ y, int64_t Tout, const float *__restrict__ wp) {
  __shared__ float Ws[16][8][11];
  const int tid = threadIdx.x;
  const int co0 = 0;
  const int64_t t0 = (int64_t) blockIdx.x * 128 + (tid >> 3) * 8;
  const float4 *w4 = (const float4 *)wp;
  const int kc = K * cin;
  const int cl = tid & 7;
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; j++)
    acc[j] = 0.f;
  const int64_t base = t0 - pad_in;
  for (int cb = 0; cb < cin; cb += 16) {
    for (int i = tid; i < 16 * 8 * 11; i += 128) {
      int k = i % 11, c = (i / 11) & 7, ci = i / (11 * 8);
      float v = 0.f;
      if (k < K && cb + ci < cin && co0 + c < 8) {
        const float4 *w = ((c & 4) ? w4 + kc : w4) + k * cin + (cb + ci);
        v = (&w->x)[c & 3];
      }
      Ws[ci][c][k] = v;
    }
    __syncthreads();
#pragma unroll
    for (int ci = 0; ci < 16; ci++) {
      if (cb + ci >= cin)
        break;
      float xv[18];
      const float *row = x + (int64_t) (cb + ci) * ldx;
#pragma unroll
      for (int j = 0; j < 18; j++) {
        int64_t sq = base + j;
        xv[j] = (sq >= 0 && sq < ldx) ? row[sq] : 0.f;
      }
      for (int k = 0; k < K; k++) {
        float wv = Ws[ci][cl][k];
#pragma unroll
        for (int i = 0; i < 8; i++)
          acc[i] += wv * xv[i + k];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int64_t t = t0 + j;
    if (t >= Tout)
      continue;
    y[(int64_t) co0 * Tout + t] = acc[j];   /* channel co0 only: acc[j] -> t0+j */
  }
}

int main(void) {
  int cin = 8, cout = 8, K = 3, pad = 1;
  long long T = 32;
  float *hx = new float[cin * T];
  float *hw = new float[(cout / 4) * K * cin * 4];
  for (int i = 0; i < cin * T; i++)
    hx[i] = 1.f;
  for (int i = 0; i < (cout / 4) * K * cin * 4; i++)
    hw[i] = 1.f;
  float *dx, *dw, *dy;
  cudaMalloc(&dx, sizeof(float) * cin * T);
  cudaMalloc(&dw, sizeof(float) * (cout / 4) * K * cin * 4);
  cudaMalloc(&dy, sizeof(float) * T);
  cudaMemcpy(dx, hx, sizeof(float) * cin * T, cudaMemcpyHostToDevice);
  cudaMemcpy(dw, hw, sizeof(float) * (cout / 4) * K * cin * 4, cudaMemcpyHostToDevice);
  win<<<1, 128>>>(dx, T, cin, K, pad, dy, T, dw);
  float hy[64];
  cudaMemcpy(hy, dy, sizeof(float) * T, cudaMemcpyDeviceToHost);
  printf("got : ");
  for (int i = 0; i < 16; i++)
    printf("%.1f ", hy[i]);
  printf("\nexp : ");
  for (int i = 0; i < 16; i++) {
    int cnt = 0;
    for (int k = 0; k < K; k++) {
      long long s = (long long) i + k - pad;
      if (s >= 0 && s < T)
        cnt++;
    }
    printf("%.1f ", (float)(cnt * cin));
  }
  printf("\n");
  return 0;
}
