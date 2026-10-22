#!/bin/sh
# Build + run every runtime test. Native on the Nano (asm kernels) or x86 (portable C kernels).
#   tests/run_tests.sh                      (from the repo root)
# Golden/numpy vectors are generated on the dev host (needs .venv with onnxruntime):
#   .venv/bin/python tools/export_ptk.py && .venv/bin/python tests/gen_golden.py && .venv/bin/python tests/gen_kernel_vectors.py
set -e
cd "$(dirname "$0")/.."
ARCH=$(uname -m)
make -C runtime -j4 >/dev/null
B=build/$ARCH
PTK=${PTK:-build/v1b8k.ptk}
GOLD=${GOLD:-tests/golden}
echo "== kernels ($ARCH)"
$B/test_kernels tests/kern
echo "== end-to-end vs onnxruntime golden"
$B/test_e2e $PTK $GOLD zh_greeting en_thanks mix_booking zh_phone_sid1 random_T90 random_T23
