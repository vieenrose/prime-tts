// CPU microbenchmarks for Jetson Nano (Cortex-A57, 4 cores, ARMv8.0 fp32 NEON).
// Measures: scalar fp32 GEMM, NEON GEMM variants, thread scaling, memcpy BW, L2/L3 behaviour.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <arm_neon.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

#define M 256
#define N 256
#define K 256
static float A[M*K], B[K*N], C[M*N];

static void gemm_scalar(void){
  for(int i=0;i<M;i++) for(int j=0;j<N;j++){
    float s=0; const float*__restrict a=&A[i*K]; const float*__restrict b=&B[j];
    for(int k=0;k<K;k++) s += a[k]*b[k*N];
    C[i*N+j]=s;
  }
}
// NEON: 4-wide dot product along K, column-strided B (non-coalesced) -> baseline NEON
static void gemm_neon_strided(void){
  for(int i=0;i<M;i++) for(int j=0;j<N;j+=4){
    float32x4_t s0=vdupq_n_f32(0),s1=s0,s2=s0,s3=s0;
    for(int k=0;k<K;k++){
      float av=A[i*K+k];
      float32x4_t bv=vld1q_f32(&B[k*N+j]);
      s0=vfmaq_n_f32(s0,bv,av); // emulate per-lane
    }
    vst1q_f32(&C[i*N+j],s0);
  }
}
// Classic 4x4 tile: load 4 rows of A (contig in K), 4 cols of B -> register blocking
static void gemm_neon_4x4(void){
  for(int i=0;i<M;i+=4) for(int j=0;j<N;j+=4){
    float32x4_t c0=vdupq_n_f32(0),c1=c0,c2=c0,c3=c0;
    for(int k=0;k<K;k+=4){
      float32x4_t a0=vld1q_f32(&A[i*K+k]),a1=vld1q_f32(&A[(i+1)*K+k]),a2=vld1q_f32(&A[(i+2)*K+k]),a3=vld1q_f32(&A[(i+3)*K+k]);
      float32x4_t b0=vld1q_f32(&B[k*N+j]),b1=vld1q_f32(&B[(k+1)*N+j]),b2=vld1q_f32(&B[(k+2)*N+j]),b3=vld1q_f32(&B[(k+3)*N+j]);
      // transpose-free broadcast approach: c += a_k * b_k per row (rank-1 update)
      c0=vfmaq_n_f32(c0,b0,a0[0]); c1=vfmaq_n_f32(c1,b0,a1[0]); c2=vfmaq_n_f32(c2,b0,a2[0]); c3=vfmaq_n_f32(c3,b0,a3[0]);
      c0=vfmaq_n_f32(c0,b1,a0[1]); c1=vfmaq_n_f32(c1,b1,a1[1]); c2=vfmaq_n_f32(c2,b1,a2[1]); c3=vfmaq_n_f32(c3,b1,a3[1]);
      c0=vfmaq_n_f32(c0,b2,a0[2]); c1=vfmaq_n_f32(c1,b2,a1[2]); c2=vfmaq_n_f32(c2,b2,a2[2]); c3=vfmaq_n_f32(c3,b2,a3[2]);
      c0=vfmaq_n_f32(c0,b3,a0[3]); c1=vfmaq_n_f32(c1,b3,a1[3]); c2=vfmaq_n_f32(c2,b3,a2[3]); c3=vfmaq_n_f32(c3,b3,a3[3]);
    }
    vst1q_f32(&C[i*N+j],c0); vst1q_f32(&C[(i+1)*N+j],c1); vst1q_f32(&C[(i+2)*N+j],c2); vst1q_f32(&C[(i+3)*N+j],c3);
  }
}

typedef struct{ int n; void(*fn)(void); } Job;
static void* thr(void* p){ Job* j=(Job*)p; for(int i=0;i<j->n;i++) j->fn(); return 0; }

static double run(void(*fn)(void), int reps, int threads){
  double t0=now();
  if(threads==1){ fn(); for(int i=1;i<reps;i++) fn(); }
  else { pthread_t id[8]; Job jobs[8];
    for(int t=0;t<threads;t++){ jobs[t].n=reps/threads; jobs[t].fn=fn; pthread_create(&id[t],0,thr,&jobs[t]); }
    for(int t=0;t<threads;t++) pthread_join(id[t],0);
  }
  return (now()-t0)/reps;
}

int main(int argc,char**argv){
  double fl=2.0*M*N*K;
  for(int i=0;i<M*K;i++)A[i]=(i%7)*0.125f;
  for(int i=0;i<K*N;i++)B[i]=(i%5)*0.125f;
  int reps=6;
  double s=run(gemm_scalar,reps,1);  printf("scalar  fp32 GEMM  %6.2f GFLOP/s (%.2f ms)\n", fl/s/1e9, s*1e3);
  double n1=run(gemm_neon_strided,reps,1); printf("neon-strided     %6.2f GFLOP/s (%.2f ms)\n", fl/n1/1e9, n1*1e3);
  double n2=run(gemm_neon_4x4,reps,1);  printf("neon-4x4  fp32   %6.2f GFLOP/s (%.2f ms)\n", fl/n2/1e9, n2*1e3);
  double n4=run(gemm_neon_4x4,reps,4);  printf("neon-4x4 x4thr   %6.2f GFLOP/s (%.2f ms)\n", fl/n4/1e9, n4*1e3);
  double n2t=run(gemm_neon_4x4,reps,2); printf("neon-4x4 x2thr   %6.2f GFLOP/s (%.2f ms)\n", fl/n2t/1e9, n2t*1e3);
  // memcpy bandwidth (cache effects): 8MB, 64MB
  size_t sz[3]={1<<20, 8<<20, 48<<20};
  for(int i=0;i<3;i++){
    char *x=malloc(sz[i]),*y=malloc(sz[i]); memset(x,1,sz[i]); memset(y,2,sz[i]);
    double t=now(); int r=8; for(int k=0;k<r;k++) memcpy(y,x,sz[i]); t=(now()-t)/r;
    printf("memcpy %4zu MB  %6.2f GB/s (r+w)\n", sz[i]>>20, 2.0*sz[i]/t/1e9);
    free(x);free(y);
  }
  (void)argv;
  return 0;
}
