#!/bin/bash
# .auto/measure.sh - v2.1 latency benchmark, measured on the Jetson Nano.
#
# Primary metric: n4_wall_ms  = wall time of 4 concurrent T81 requests (the operator target is 4
# real-time streams, so the aggregate wall of four requests is the latency that matters).
# Secondaries: per-stream time at N=1 and N=4, min SNDR across streams (quality guard), free RAM.
#
# Nothing is measured on the x86 box: the target device is the Nano (4x A57 + sm_53 Maxwell).
set -uo pipefail

NANO="${NANO:-picard@100.84.151.116}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ARENA="${ARENA_MB:-8}"
N4="${N4:-4}"
CASE="${CASE:-tests/golden_v21/T81}"

fail() { echo "measure: $1" >&2; exit 2; }

# ---- Nano memory guard (hard rule: abort below 60 MB free, the GPU counts against system RAM)
free_mb=$(ssh -o ConnectTimeout=8 "$NANO" 'free -m | sed -n 2p | tr -s " " | cut -d" " -f4' 2>/dev/null)
[ -n "$free_mb" ] || fail "cannot read free RAM on $NANO"
[ "$free_mb" -ge 60 ] || fail "Nano free RAM ${free_mb} MB < 60 MB guard"

# ---- sync + build (build time is deliberately NOT part of the metric)
rsync -a -q "$REPO/runtime/" "$NANO:primetts/runtime/" 2>/dev/null
rsync -a -q "$REPO/build/v21.ptk" "$NANO:primetts/build/" 2>/dev/null
# build both drivers: checks.sh gates quality with pt_v21_nano, the metric uses pt_v21_par
bld=$(ssh "$NANO" 'cd ~/primetts && make -f runtime/v21/Makefile par nano 2>&1 | grep -E "error:" | head -6')
[ -z "$bld" ] || { echo "$bld"; fail "build failed on the Nano"; }
ssh "$NANO" 'test -x ~/primetts/build/pt_v21_par' || fail "pt_v21_par missing on the Nano"

# ---- N=4 (primary). Sample free RAM *and* the driver's peak RSS during the run: on the Nano the
# GPU's allocations count against system RAM, so the process high-water mark is the number that
# says whether four streams actually fit on the box.
run4=$(ssh "$NANO" "NANO_ARENA=$ARENA NANO_CASE=$CASE NANO_N4=$N4 bash -s" <<'REMOTE'
cd ~/primetts
rm -f /tmp/ar_mem.txt /tmp/ar_freq.txt
GF=/sys/devices/platform/host1x/57000000.gpu/devfreq/57000000.gpu/cur_freq
CF=/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
PT_V21_ARENA_MB="$NANO_ARENA" ./build/pt_v21_par --container build/v21.ptk   --case "$NANO_CASE" --cuda --parallel "$NANO_N4" > /tmp/ar_out.txt 2>&1 &
pid=$!
hwm=0
nsamp=0
while kill -0 "$pid" 2>/dev/null; do
  h=$(awk '/VmHWM/{print $2}' /proc/"$pid"/status 2>/dev/null)
  f=$(free -m | sed -n 2p | tr -s " " | cut -d" " -f4)
  q=$(cat "$GF" 2>/dev/null)
  cf=$(cat "$CF" 2>/dev/null)
  nsamp=$((nsamp + 1))
  [ -n "$h" ] && [ "$h" -gt "$hwm" ] && hwm=$h
  [ -n "$f" ] && echo "$f" >> /tmp/ar_mem.txt
  # skip the first second (startup is always at the idle clock): what matters is whether the
  # governor holds max clocks through the inference phase
  if [ "$nsamp" -gt 4 ] && [ -n "$q" ]; then echo "$q" >> /tmp/ar_freq.txt; fi
  if [ "$nsamp" -gt 4 ] && [ -n "$cf" ]; then echo "$cf" >> /tmp/ar_cpufreq.txt; fi
  sleep 0.25
done
wait "$pid"
cat /tmp/ar_out.txt
echo "MINFREE $(sort -n /tmp/ar_mem.txt | head -1)"
echo "PEAKRSS $((hwm / 1024))"
echo "MINFREQ $(sort -n /tmp/ar_freq.txt 2>/dev/null | head -1)"
echo "MINCPUFREQ $(sort -n /tmp/ar_cpufreq.txt 2>/dev/null | head -1)"
REMOTE
)
echo "$run4" | sed -n '1,4p'
wall=$(echo "$run4" | sed -n 's/^N=[0-9][0-9]*  wall \([0-9][0-9]*\) ms.*/\1/p' | head -1)
sndr=$(echo "$run4" | sed -n 's/.*min SNDR \([0-9][0-9.]*\) dB$/\1/p' | head -1)
minfree=$(echo "$run4" | sed -n 's/^MINFREE \([0-9][0-9]*\)$/\1/p' | head -1)
peakrss=$(echo "$run4" | sed -n 's/^PEAKRSS \([0-9][0-9]*\)$/\1/p' | head -1)
minfreq=$(echo "$run4" | sed -n 's/^MINFREQ \([0-9][0-9]*\)$/\1/p' | head -1)
[ -n "${minfreq:-}" ] && echo "min GPU freq during N=4 run: $minfreq Hz (921600000 = full clock; lower means the governor throttled or idled)"
mincpufreq=$(echo "$run4" | sed -n 's/^MINCPUFREQ \([0-9][0-9]*\)$/\1/p' | head -1)
[ -n "${mincpufreq:-}" ] && echo "min CPU freq during N=4 run: $mincpufreq Hz (1479000 = max; low + slow run = thermal throttle, max + slow run = co-tenant contention)"
[ -n "$wall" ] || fail "no wall time in N=4 output"
[ -n "$sndr" ] || fail "no SNDR in N=4 output"

# ---- N=1 single-stream latency
run1=$(ssh "$NANO" "cd ~/primetts && PT_V21_ARENA_MB=$ARENA ./build/pt_v21_par --container build/v21.ptk --case $CASE --cuda --parallel 1 2>&1")
s1=$(echo "$run1" | sed -n 's/.*slowest stream \([0-9][0-9]*\) ms.*/\1/p' | head -1)

# ---- GPU phase split (where the time actually goes; the driver prints this under PT_V21_CUDAPROF)
split=$(ssh "$NANO" "cd ~/primetts && PT_V21_CUDAPROF=1 PT_V21_EKIND=1 PT_V21_ARENA_MB=$ARENA ./build/pt_v21_nano --container build/v21.ptk --case $CASE --cuda 2>&1 | grep -E 'gpu split|host op kinds' | head -2")
echo "$split"

echo "METRIC n4_wall_ms=$wall"
echo "METRIC n1_stream_ms=${s1:-0}"
echo "METRIC min_sndr_db=$sndr"
echo "METRIC min_free_mb=${minfree:-$free_mb}"
echo "METRIC peak_rss_mb=${peakrss:-0}"
