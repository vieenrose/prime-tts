#!/bin/bash
# ab1.sh "<envA>" "<envB>" NPAIRS [CASE] -- order-alternating N=4 pairs on picard, prints walls + means
A="$1"; B="$2"; NP="${3:-8}"; CASE="${4:-T81}"; BIN="${BIN:-./build/pt_v21_par}"
# an arm of the form "@<binary> [ENV=..]" runs that binary instead of $BIN
run1 () { local b="$BIN" e="$1"; if [ "${e#@}" != "$e" ]; then b="${e%% *}"; b="${b#@}"; e="${e#* }"; [ "$e" = "$1" ] && e=""; fi
  ssh picard@100.84.151.116 "cd ~/primetts && env $e $b --container build/v21.ptk --case tests/golden_v21/$CASE --cuda --parallel 4 2>&1" 2>/dev/null | grep -oE 'wall [0-9]+' | grep -oE '[0-9]+'; }
run1 "$A" >/dev/null   # discarded warmup: the first draw of a batch runs cold
sa=0; sb=0; wins=0
for i in $(seq 1 $NP); do
  if [ $((i % 2)) -eq 1 ]; then a=$(run1 "$A"); b=$(run1 "$B"); else b=$(run1 "$B"); a=$(run1 "$A"); fi
  sa=$((sa+a)); sb=$((sb+b)); [ "$b" -lt "$a" ] && wins=$((wins+1))
  echo "pair$i A=$a B=$b d=$((b-a))"
done
echo "MEAN A=$((sa/NP)) B=$((sb/NP)) d=$(((sb-sa)/NP)) Bwins=$wins/$NP  $(ssh picard@100.84.151.116 uptime 2>/dev/null | sed 's/.*average: /load /')"
