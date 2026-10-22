/* A57 NEON fp32 pure-issue peak probe (dependency-free, asm, no memory traffic in the loop).
 *
 * Why: two of our recorded ceilings differ by 2x - 11.8 GFLOP/s/core (8 FLOP/cycle at 1.479 GHz)
 * vs 23.5 GFLOP/s/core (16 FLOP/cycle, 2 ASIMD pipes x 4 lanes x 2 for FMA). Cortex-A57 issues 2
 * 128-bit ASIMD ops per cycle, so the peak depends on the op mix: vmul.4s = 4 FLOPs/op (8 per
 * cycle), vmla.4s = 8 FLOPs/op (16 per cycle). This probe measures each op mix in isolation with
 * 16 independent accumulator chains so the loop is throughput-bound, not latency-bound, and prints
 * FLOP/cycle/core using the measured CPU frequency.
 *
 * Build (Nano): gcc -O2 -o issue_probe issue_probe.c issue_kern.S
 * Run:  taskset -c 0 ./issue_probe 20000000 ; taskset -c 0,1,2,3 ./issue_probe 20000000
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern long long vmla_bench(long long iters);   /* 16 independent vmla.4s  = 128 FLOPs/iteration */
extern long long vmul_bench(long long iters);   /* 16 independent vmul.4s  =  64 FLOPs/iteration */
extern long long vadd_bench(long long iters);   /* 16 independent vadd.4s  =  64 FLOPs/iteration */

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* CPU frequency from cpufreq sysfs (no sudo needed to read). */
static double cpu_mhz(void) {
  FILE *f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r");
  if (!f) return 0.0;
  long k = 0;
  if (fscanf(f, "%ld", &k) != 1) k = 0;
  fclose(f);
  return k / 1000.0;
}

int main(int argc, char **argv) {
  long long iters = argc > 1 ? atoll(argv[1]) : 20000000LL;
  double mhz = cpu_mhz();
  printf("iters=%lld  cpu0=%0.1f MHz\n", iters, mhz);

  struct { const char *name; long long (*fn)(long long); long long flop_per_iter; } k[3] = {
    { "vmla.4s x16", vmla_bench, 128 },   /* 16 ops x 4 lanes x 2 (mul+add) */
    { "vmul.4s x16", vmul_bench, 64 },    /* 16 ops x 4 lanes x 1 */
    { "vadd.4s x16", vadd_bench, 64 },
  };
  for (int i = 0; i < 3; i++) {
    k[i].fn(100000);                                   /* warm up, fill caches */
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
      double t0 = now();
      long long s = k[i].fn(iters);
      double dt = now() - t0;
      if (dt < best) best = dt;
      if (s == 123456789) printf("");                 /* keep results live */
    }
    double gflops = (double)iters * k[i].flop_per_iter / best / 1e9;
    double flop_cycle = mhz > 0 ? gflops * 1000.0 / mhz : 0.0;
    printf("%-13s  %6.2f GFLOP/s/core   %5.2f FLOP/cycle/core  (peak is 16 for FMA, 8 for mul/add)\n",
           k[i].name, gflops, flop_cycle);
  }
  return 0;
}
