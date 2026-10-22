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
ARENA="${ARENA_MB:-32}"
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

# ---- N=4 (primary). Also sample free RAM during the run: memory is the Nano's real constraint.
run4=$(ssh "$NANO" "cd ~/primetts && (for i in \$(seq 1 40); do free -m | sed -n 2p | tr -s ' ' | cut -d' ' -f4 >> /tmp/ar_mem.txt; sleep 0.3; done > /dev/null 2>&1 &) ; PT_V21_ARENA_MB=$ARENA ./build/pt_v21_par --container build/v21.ptk --case $CASE --cuda --parallel $N4 2>&1; echo \"MINFREE \$(sort -n /tmp/ar_mem.txt | head -1)\"; rm -f /tmp/ar_mem.txt")
echo "$run4" | sed -n '1,4p'
wall=$(echo "$run4" | sed -n 's/^N=[0-9][0-9]*  wall \([0-9][0-9]*\) ms.*/\1/p' | head -1)
sndr=$(echo "$run4" | sed -n 's/.*min SNDR \([0-9][0-9.]*\) dB$/\1/p' | head -1)
minfree=$(echo "$run4" | sed -n 's/^MINFREE \([0-9][0-9]*\)$/\1/p' | head -1)
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
