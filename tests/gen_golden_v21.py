#!/usr/bin/env python3
"""Golden vectors for the v2.1 (`v21_mbistft_16k`, 16 kHz) runtime, generated with onnxruntime.

One whole-graph model, one case per T bucket:
  inputs  x / tone / lang [1,T] int64, x_lengths [1], sid [1], noise_scale [1] float, length_scale
  output  wav [1,1,L] float32
  plus a curated set of intermediates (the FIRST node output of every (group, op_type) pair in
  graph order) so the C/CUDA runtime can be checked layer by layer, op kind by op kind.

Front-end caveat (measured, not assumed): the zh-TW text front-end (g2pw / g2p_en / entity
normalisation) is NOT ported, so these goldens do not start from text.  The phone/tone/lang ids
are taken verbatim from bench/cpu/requests.txt (the same id vectors the CPU bench runs),
concatenated in file order and cut to the target T:
    T=20   zh_greeting(21)          -> first 20 ids
    T=40   zh_greeting(21) + en_thanks(19)
    T=81   + mix_booking(24) + first 17 of zh_phone_sid1(29)
    T=120  + zh_phone_sid1(29) + first 27 of random_T90(90)
    T=29   zh_phone_sid1 verbatim, sid=1 (speaker-id coverage)
v2.1 is stochastic at noise_scale>0, so every case runs with noise_scale=0.0.

Usage: .venv/bin/python tests/gen_golden_v21.py [--out tests/golden_v21] [--T 20 40 81 120]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort

MODEL = "model/v21_mbistft_16k/primetts_v21_3voice.onnx"
REQUESTS = "bench/cpu/requests.txt"
GROUPS = [("enc_p", "enc"), ("enc", "enc"), ("dp", "dp"), ("flow", "flow"), ("dec", "dec")]


def load_requests(path: str = REQUESTS) -> list[tuple[str, int, np.ndarray]]:
    """rows of bench/cpu/requests.txt: name sid T phone[T] tone[T] lang[T]."""
    out = []
    for ln in open(path):
        t = ln.split()
        if len(t) < 3:
            continue
        sid, T = int(t[1]), int(t[2])
        flat = np.array([int(v) for v in t[3:]], np.int64)
        assert flat.shape[0] == 3 * T, (t[0], T, flat.shape)
        out.append((t[0], sid, flat))
    return out


def ids_for_T(T: int, rows: list[tuple[str, int, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Concatenate request rows in file order, cut to T; report which rows were used."""
    ph, tn, lg, used = [], [], [], []
    for nm, _sid, flat in rows:
        ph += list(flat[:flat.size // 3])
        tn += list(flat[flat.size // 3:2 * (flat.size // 3)])
        lg += list(flat[2 * (flat.size // 3):])
        used.append(nm)
        if len(ph) >= T:
            break
    if len(ph) < T:
        raise SystemExit(f"requests.txt gives only {len(ph)} ids, cannot build T={T}")
    return (np.array([ph[:T]], np.int64), np.array([tn[:T]], np.int64),
            np.array([lg[:T]], np.int64), "+".join(used))


def case_list(Ts: list[int], rows: list[tuple[str, int, np.ndarray]]) -> list[dict]:
    cases = []
    for T in Ts:
        ph, tn, lg, src = ids_for_T(T, rows)
        cases.append(dict(name=f"T{T}", T=T, sid=0, ph=ph, tn=tn, lg=lg, src=src))
    # speaker-id coverage: zh_phone_sid1 verbatim (its own row, sid=1)
    nm, sid, flat = next(r for r in rows if r[0] == "zh_phone_sid1")
    T = flat.size // 3
    cases.append(dict(name="sid1_T29", T=T, sid=sid,
                      ph=np.array([flat[:T]], np.int64), tn=np.array([flat[T:2 * T]], np.int64),
                      lg=np.array([flat[2 * T:]], np.int64), src=nm))
    return cases


def probe_session(path: str, names: list[str]) -> ort.InferenceSession:
    """Un-optimised session with the chosen intermediates exposed as extra graph outputs.

    Serialise the whole ModelProto (not just the graph) so opset_import/ir_version survive -
    onnxruntime rejects a bare graph ("No graph was found in the protobuf").
    """
    m = onnx.load(path)
    si = onnx.shape_inference.infer_shapes(m, strict_mode=False)
    et = {v.name: v.type.tensor_type.elem_type for v in list(si.graph.value_info) + list(si.graph.output)}
    existing = {o.name for o in m.graph.output}
    for nm in names:
        if nm not in existing:
            m.graph.output.append(helper.make_tensor_value_info(nm, et.get(nm, onnx.TensorProto.FLOAT), None))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])


def intermediates(path: str) -> list[str]:
    """First node output of every (group, op_type) pair, in graph order: every op kind the
    runtime implements, and every group, at least once."""
    g = onnx.load(path).graph
    seen, names = set(), []
    for n in g.node:
        head = (n.name or n.output[0]).strip("/").split("/")[0]
        grp = dict(GROUPS).get(head, "root")
        for o in n.output:
            key = (grp, n.op_type)
            if o and key not in seen:
                seen.add(key)
                names.append(o)
            break
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/golden_v21")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--T", type=int, nargs="+", default=[20, 40, 81, 120])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    rows = load_requests()
    cases = case_list(args.T, rows)
    names = intermediates(args.model)

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    fast = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    probe = probe_session(args.model, names)
    graph_outputs = [o.name for o in onnx.load(args.model).graph.output]

    manifest = []
    for c in cases:
        T = c["T"]
        feeds = {"x": c["ph"], "tone": c["tn"], "lang": c["lg"],
                 "x_lengths": np.array([T], np.int64), "sid": np.array([c["sid"]], np.int64),
                 "noise_scale": np.array([0.0], np.float32), "length_scale": np.array([1.0], np.float32)}
        t0 = time.time()
        wav = fast.run(graph_outputs, feeds)[0]
        t_fast = time.time() - t0
        t0 = time.time()
        vals = probe.run(names, feeds)
        t_probe = time.time() - t0
        d = os.path.join(args.out, c["name"])
        os.makedirs(d, exist_ok=True)
        save = {"x": c["ph"], "tone": c["tn"], "lang": c["lg"], "x_lengths": np.array([T], np.int64),
                "sid": np.array([c["sid"]], np.int64), "noise_scale": np.array([0.0], np.float32),
                "length_scale": np.array([1.0], np.float32), "wav": wav}
        for nm, v in zip(names, vals):
            save["probe_" + nm.strip("/").replace("/", "_")] = v
        for k, v in save.items():
            np.save(os.path.join(d, k + ".npy"), np.ascontiguousarray(v))
        manifest.append(dict(case=c["name"], T=T, sid=c["sid"], ids_from=c["src"],
                             samples=int(wav.shape[-1]), audio_s=round(wav.shape[-1] / 16000, 3),
                             wav_rms=round(float(np.sqrt((wav.astype(np.float64) ** 2).mean())), 6),
                             intermediates=len(names), ort_s=round(t_fast, 3), probe_s=round(t_probe, 3)))
        print(f"{c['name']:10s} T={T:3d} sid={c['sid']} ids from {c['src']:52s} "
              f"samples={wav.shape[-1]:6d} ({wav.shape[-1]/16000:.2f}s audio) "
              f"ort {t_fast:.2f}s probe {t_probe:.2f}s")
    json.dump(dict(model=os.path.relpath(args.model), noise_scale=0.0, length_scale=1.0,
                   sample_rate=16000,
                   front_end="zh-TW g2pw/g2p_en front-end NOT ported: ids come from "
                             "bench/cpu/requests.txt, not from text",
                   intermediate_policy="first node output per (group, op_type) in graph order",
                   cases=manifest, files_per_case=sorted(save)),
              open(os.path.join(args.out, "MANIFEST.json"), "w"), indent=1)
    print(f"wrote {args.out}: {len(manifest)} cases x {len(names)} intermediates "
          f"(+ x/tone/lang/x_lengths/sid/noise_scale/length_scale/wav) + MANIFEST.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
