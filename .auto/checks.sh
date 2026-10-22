#!/bin/bash
# .auto/checks.sh - quality gate for the latency loop. Two jobs:
#  (1) prove the benchmark is honest: the container, goldens, feeds, spec and the parity gate are
#      the exact bytes they were when the session started (md5 manifest), and the gate's own
#      tolerances were not loosened;
#  (2) prove quality survived: SNDR >= 60 dB per case on the Nano, GPU path and CPU path.
# Output stays minimal; only failures print.
set -uo pipefail

NANO="${NANO:-picard@100.84.151.116}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

die() { echo "CHECK FAILED: $1"; exit 1; }

# ---- 1. integrity manifest (anti-cheat)
MAN=".auto/quality_manifest.txt"
[ -f "$MAN" ] || die "missing $MAN - regenerate only from a known-good tree, never to make a run pass"
md5sum -c "$MAN" --status >/dev/null || die "protected files changed (container / goldens / feeds / spec / gate)"

grep -q -- '--rel-fp16", type=float, default=2e-3' tests/v21_parity_c.py || die "parity gate fp16 tolerance edited"
grep -q -- '--sndr-min", type=float, default=60.0' tests/v21_parity_c.py || die "parity gate SNDR floor edited"
grep -q '"noise_scale"' runtime/v21/src/main_v21_par.c || die "driver no longer feeds noise_scale"

# ---- 2. audio quality on the Nano (the driver exits 0 only when min SNDR >= 60 dB)
for c in T20 T81; do
  out=$(ssh -o ConnectTimeout=8 "$NANO" "cd ~/primetts && ./build/pt_v21_nano --container build/v21.ptk --case tests/golden_v21/$c --cuda 2>&1 | grep -oE 'SNDR=[-0-9.]+' | tail -1")
  v=$(echo "$out" | sed 's/SNDR=//')
  [ -n "$v" ] || die "no SNDR line for $c (GPU)"
  ok=$(awk -v x="$v" 'BEGIN { print (x >= 60.0) ? "y" : "n" }')
  [ "$ok" = "y" ] || die "SNDR $v dB below 60 for $c on the GPU path"
done
cv=$(ssh "$NANO" 'cd ~/primetts && ./build/pt_v21_nano --container build/v21.ptk --case tests/golden_v21/T81 2>&1 | grep -oE "SNDR=[-0-9.]+" | tail -1' | sed 's/SNDR=//')
[ -n "$cv" ] || die "no SNDR line for the CPU path"
ok=$(awk -v x="$cv" 'BEGIN { print (x >= 60.0) ? "y" : "n" }')
[ "$ok" = "y" ] || die "CPU path SNDR $cv dB below 60"
exit 0
