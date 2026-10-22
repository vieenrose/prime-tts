/* A57 NEON fp32 pure-issue peak probe (dependency-free asm driver, no memory traffic in the loop).
 *
 * Why: two recorded ceilings differ by 2x - 11.8 GFLOP/s/core (8 FLOP/cycle at 1.479 GHz) vs
 * 23.5 GFLOP/s/core (16 FLOP/cycle: Cortex-A57 issues 2 128-bit ASIMD ops/cycle). The peak depends
 * on the op mix: fmul/fadd on 4 lanes = 4 FLOPs/op (8/cycle), fmla on 4 lanes = 8 FLOPs/op
 * (16/cycle). This measures each op mix with 16 independent accumulator chains so the loop is
 * throughput-bound, converting to FLOP/cycle/core with the measured cpufreq.
 *
 * Build (Nano): gcc -O2 -std=gnu11 -D_GNU_SOURCE -o issue_probe issue_probe.c issue_kern.S
 * Run:  taskset -c 0 ./issue_probe 8000000
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern long long fmla_bench(long long iters);   /* 16 fmla.4s = 128 FLOPs/iteration */
extern long long fmul_bench(long long iters);   /* 16 fmul.4s =  64 FLOPs/iteration */
extern long long fadd_bench(long long iters);   /* 16 fadd.4s =  64 FLOPs/iteration */

/* Timing source on this L4T box: clock_gettime(CLOCK_MONOTONIC[_COARSE]) returned 0 for the process
 * (measured: dt came out equal to the full uptime, then 0 deltas after priming) and clock() returned
 * nan, so we time with getrusage(RUSAGE_SELF).ru_utime - CPU time, which is what a bound loop needs. */
#include <sys/resource.h>
static double now_s(void) {
  struct rusage u;
  if (getrusage(RUSAGE_SELF, &u)) return 0.0;
  return (double)u.ru_utime.tv_sec + 1e-6 * (double)u.ru_utime.tv_usec;
}

static double cpu0_mhz(void) {
  FILE *f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r");
  if (!f) return 0.0;
  long k = 0;
  if (fscanf(f, "%ld", &k) != 1) k = 0;
  fclose(f);
  return k / 1000.0;
}

int main(int argc, char **argv) {
  long long iters = argc > 1 ? atoll(argv[1]) : 8000000LL;
  double mhz = cpu0_mhz();
  printf("iters=%lld cpu0=%.1f MHz (timing: getrusage CPU time)\n", iters, mhz);

  struct { const char *name; long long (*fn)(long long); long long flop_iter; } k[3] = {
    { "fmla.4s x16", fmla_bench, 128 },   /* 16 ops x 4 lanes x 2 (mul+add) */
    { "fmul.4s x16", fmul_bench, 64 },    /* 16 ops x 4 lanes x 1 */
    { "fadd.4s x16", fadd_bench, 64 },
  };
  for (int i = 0; i < 3; i++) {
    volatile long long keep = k[i].fn(200000);      /* warm up */
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
      double t0 = now_s();
      long long s = k[i].fn(iters);
      double dt = now_s() - t0;
      keep += s;
      printf("  rep: dt=%.6f s  checksum=%lld\n", dt, keep);
      if (dt > 0.0 && dt < best) best = dt;
    }
    if (best <= 0.0 || best > 1e29) { printf("%-13s  TIMING FAILED\n", k[i].name); continue; }
    double gflops = (double)iters * (double)k[i].flop_iter / best / 1e9;
    double per_cycle = mhz > 0.0 ? gflops * 1000.0 / mhz : 0.0;
    printf("%-13s  %6.2f GFLOP/s/core  %5.2f FLOP/cycle/core   (issue peak: 16 for FMA, 8 for mul/add)\n",
           k[i].name, gflops, per_cycle);
  }
  return 0;
}
