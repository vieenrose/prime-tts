#!/usr/bin/env python3
"""M2 parity gate for the v2.1 C runtime: run build/pt_v21_host on a golden case and compare
every graph output against the onnxruntime node dump (tests/golden_v21/_probe_T81).

The runtime writes its intermediates as .npy (--checkpoints/--ckdir); this script walks the graph
in node order and applies the same tolerances tests/v21_reference.py uses:
  rel < --rel (1e-3) for the fp32-weight groups, rel < --rel-fp16 (2e-3) for the decoder, whose
  weights are fp16 in the container; SNDR >= --sndr-min (60 dB) on the waveform.
The sampler node and the path feeding the noise_scale multiply are exempt: parity runs use
noise_scale = 0 and the runtime evaluates the sampler as zeros (docs/v21_layers.md).

  .venv/bin/python tests/v21_parity_c.py --case T81 [--binary build/pt_v21_host]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import onnx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/pt_v21_host")
    ap.add_argument("--container", default="build/v21.ptk")
    ap.add_argument("--case", default="T81")
    ap.add_argument("--probe", default="tests/golden_v21/_probe_T81")
    ap.add_argument("--probe-case", default="T81")
    ap.add_argument("--golden", default="tests/golden_v21")
    ap.add_argument("--rel", type=float, default=1e-3)
    ap.add_argument("--rel-fp16", type=float, default=2e-3)
    ap.add_argument("--sndr-min", type=float, default=60.0)
    ap.add_argument("--keep", default="", help="directory to keep the runtime's dumps in")
    args = ap.parse_args()

    case_dir = os.path.join(ROOT, args.golden, args.case)
    g = onnx.load(os.path.join(ROOT, "model/v21_mbistft_16k/primetts_v21_3voice.onnx")).graph
    order = [o for n in g.node for o in n.output if o]
    tmp = args.keep or tempfile.mkdtemp(prefix="v21par_")
    os.makedirs(tmp, exist_ok=True)
    namefile = os.path.join(tmp, "_names.txt")
    with open(namefile, "w") as f:
        f.write("\n".join(nm for nm in order if nm) + "\n")
    cmd = [os.path.join(ROOT, args.binary), "--container", os.path.join(ROOT, args.container),
           "--case", case_dir, "--checkpoints", namefile, "--ckdir", tmp, "--out",
           os.path.join(tmp, "wav.npy")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode not in (0, 1):
        print(f"runtime exited {r.returncode}")
        return 2
    out = r.stdout.strip().splitlines()
    line = [l for l in out if l.startswith("case ")]
    if not line:
        print("no parity line in runtime output")
        return 2
    parts = dict(kv.split("=", 1) for kv in line[0].split()[1:] if "=" in kv)
    sndr = float(parts.get("SNDR", "-1"))

    NOISE = {"/RandomNormalLike_output_0", "/Mul_5_output_0"}
    fails, checked, exempt = [], 0, 0
    worst = {}
    for nm in order:
        if not nm or nm in NOISE:
            exempt += 1
            continue
        flat = nm.replace("/", "_")
        f = os.path.join(tmp, flat + ".npy")
        q = os.path.join(ROOT, args.probe, flat + ".npy")
        if not (os.path.exists(f) and os.path.exists(q)):
            continue
        a = np.load(f)
        b = np.load(q)
        if a.shape != b.shape:
            fails.append((nm, f"shape {a.shape} vs {b.shape}", 0.0))
            continue
        if a.size == 0:
            continue
        tol = args.rel_fp16 if nm.startswith("/dec/") else args.rel
        group = "dec" if nm.startswith("/dec/") else ("dp" if nm.startswith("/dp") else
                "flow" if nm.startswith("/flow") else "enc" if nm.startswith("/enc") else "root")
        if a.dtype == np.float32 and b.dtype == np.float32:
            # same metric tests/v21_reference.py uses: max|a-b| over the tensor's own scale
            scale = max(float(np.max(np.abs(b.astype(np.float64)))), 1e-30)
            rel = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) / scale
        elif a.dtype == np.int64 and b.dtype == np.int64:
            rel = 0.0 if np.array_equal(a, b) else 1.0
        elif a.dtype == np.bool_ and b.dtype == np.bool_:
            rel = 0.0 if np.array_equal(a, b) else 1.0
        else:
            rel = 0.0 if np.array_equal(a, b) else 1.0
        checked += 1
        if rel > worst.get(group, (0.0, ""))[0]:
            worst[group] = (rel, nm)
        if rel > tol:
            fails.append((nm, f"rel {rel:.3e} > {tol:.0e}", rel))
    print(f"case {args.case}: SNDR={sndr:.2f} dB  nodes compared={checked}  exempt={exempt}"
          f"  over gate={len(fails)}")
    for grp, (rel, nm) in sorted(worst.items()):
        tol = args.rel_fp16 if grp == "dec" else args.rel
        print(f"    {grp:12s} worst rel={rel:.3e}  ({nm})  tol={tol:.0e}")
    for nm, why, _ in fails[:12]:
        print(f"    FAIL {nm}: {why}")
    ok = sndr >= args.sndr_min and not fails
    print("M2 parity:", "PASS" if ok else "FAIL")
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
