/* pt_cpucap.h - process CPU cap shared by the PrimeTTS v2.1 drivers (par benchmark, serve).
 * Call apply_cpu_cap() first in main(), before any thread exists: threads inherit the mask. */
#ifndef PT_CPUCAP_H
#define PT_CPUCAP_H
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Operator default: cap PrimeTTS to ONE core so co-tenants keep maximal CPU headroom. The N=4
 * wall is GPU-serialized on the single SM, so T81-class speech still runs ~6.9x realtime
 * (RTF ~0.58, measured); SHORT utterances (T20) sit at RTF ~0.96-1.02 — at/over the real-time
 * line. 2 cores is the floor for the full case range (2-core A/B, .auto/ideas.md).
 * The cap keeps the upper-most allowed cpu (cpu0 stays with IRQs/system).
 * PT_V21_CPUS=all disables the cap; PT_V21_CPUS="2,3" or "2-3" pins an explicit set. */
static int parse_cpu_list(const char *s, cpu_set_t *set) {
  CPU_ZERO(set);
  const char *p = s;
  while (*p) {
    char *end;
    long a = strtol(p, &end, 10);
    if (end == p || a < 0 || a >= CPU_SETSIZE) return -1;
    long b = a;
    if (*end == '-') {
      p = end + 1;
      b = strtol(p, &end, 10);
      if (end == p || b < a || b >= CPU_SETSIZE) return -1;
    }
    for (long c = a; c <= b; c++) CPU_SET(c, set);
    p = end;
    if (*p == ',') p++;
    else if (*p) return -1;
  }
  return CPU_COUNT(set) > 0 ? 0 : -1;
}

static void apply_cpu_cap(void) {
  const char *env = getenv("PT_V21_CPUS");
  if (env && !strcmp(env, "all")) return;
  cpu_set_t allowed, want, both;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed)) return; /* unknown: leave scheduling alone */
  int navail = CPU_COUNT(&allowed);
  int cap = 1;                               /* operator default: one core */
  CPU_ZERO(&want);
  if (env && *env) {
    if (parse_cpu_list(env, &want)) {
      fprintf(stderr, "par: bad PT_V21_CPUS='%s', using the default cap\n", env);
      CPU_ZERO(&want);
    } else {                               /* intersect the request with the allowed set */
      CPU_AND(&both, &want, &allowed);
      if (CPU_COUNT(&both) == 0)
        fprintf(stderr, "par: PT_V21_CPUS='%s' outside the allowed set, using the default cap\n", env);
      else
        want = both;
    }
  }
  if (CPU_COUNT(&want) == 0) {             /* default: the upper half of the allowed cpus */
    int keep = cap < navail ? cap : navail;
    for (int c = CPU_SETSIZE - 1; c >= 0 && CPU_COUNT(&want) < keep; c--)
      if (CPU_ISSET(c, &allowed)) CPU_SET(c, &want);
  }
  int taken = CPU_COUNT(&want);
  if (taken == 0 || taken >= navail) return; /* nothing to give up */
  if (sched_setaffinity(0, sizeof(want), &want)) {
    fprintf(stderr, "par: cpu cap failed (errno=%s), running unrestricted\n", strerror(errno));
    return;
  }
  char buf[128];
  int off = 0;
  for (int c = 0; c < CPU_SETSIZE && off < (int) sizeof(buf) - 8; c++)
    if (CPU_ISSET(c, &want)) off += snprintf(buf + off, sizeof(buf) - off, "%d,", c);
  if (off) buf[off - 1] = 0;
  fprintf(stderr, "par: cpu cap: %d of %d cpus [%s] (PT_V21_CPUS=all to disable)\n", taken, navail, buf);
}

#endif
