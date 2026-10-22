#!/usr/bin/env python3
"""fp16 sensitivity probe for PrimeTTS v2.1 (measured on host with onnxruntime).

Why: the Nano's fp32 issue peak is 88 GFLOP/s and v2.1 needs 95 GFLOP/s for eight real-time
streams, while a K-paired half2 GEMM measured 121-171 GFLOP/s on this board
(bench/gpu/hgemm.cu). half2 compute means fp16 weights and fp16 activations, so the open question
is not "can we go fast enough" but "does fp16 survive the flow model + iSTFT head".

This script answers it by running the real ONNX graph with its weights rounded to fp16
(storage stays fp32, values rounded), which is exactly what an fp16-storage runtime does at load
time, and comparing the waveform against the fp32 reference. Variants:
  w      every initializer rounded to fp16 (what fp16 weight storage does)
  wconv  only Conv/ConvTranspose weights rounded (the bulk of a half2 GEMM runtime)
  a      fp32 weights, fp16 rounding injected on the output of every node of a chosen op set
         (what fp16 activation storage does, approximated per op)
Reported: relative L2 of the waveform, max abs deviation, and SNDR = 20*log10(rms(ref)/rms(err)).

Usage: .venv/bin/python tools/fp16_probe.py [--model model/v21_mbistft_16k/primetts_v21_3voice.onnx]
           [--variant w] [--T 60] [--frames]
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

TARGETS = ("Conv", "ConvTranspose", "MatMul", "Gemm")


def build_case(T: int, seed: int = 3, noise_scale: float = 0.0):
    """A synthetic phoneme sequence: v2.1 takes (x, tone, lang, x_lengths, sid, noise_scale, length_scale)."""
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 60, size=(1, T)).astype(np.int64)
    tone = rng.integers(0, 5, size=(1, T)).astype(np.int64)
    lang = np.ones((1, T), np.int64)
    return {
        "x": x,
        "tone": tone,
        "lang": lang,
        "x_lengths": np.array([T], np.int64),
        "sid": np.array([0], np.int64),
        "noise_scale": np.array([noise_scale], np.float32),   # 0 => deterministic (mean path)
        "length_scale": np.array([1.0], np.float32),
    }


def run(model, feed: dict, optimize: bool = True) -> np.ndarray:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    if not optimize:
        # ORT's graph optimizer folds Cast(FLOAT16)->Cast(FLOAT) pairs away, which silently
        # measures nothing; the activation-rounding variant must run with optimizations off.
        from onnxruntime import GraphOptimizationLevel
        so.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    out = s.run(None, feed)
    return np.asarray(out[0], np.float32).reshape(-1)


def round_to_fp16(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.float32:
        return arr.astype(np.float16).astype(np.float32)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr


GROUPS = {
    "enc": ("/enc", "enc"),
    "dec": ("/dec", "dec"),
    "istft": ("/istft", "istft", "/head", "head", "/mrf", "mel_refiner"),
}


def variant_weights(model: onnx.ModelProto, which: str) -> onnx.ModelProto:
    """Round initializers to fp16 values.

    which: 'w' = every initializer; 'wconv' = only weights feeding Conv/ConvTranspose/MatMul/Gemm;
    'g:<group>' = only conv weights whose node name matches a module group in GROUPS.
    """
    m = copy.deepcopy(model)
    used = set()
    if which.startswith("g:"):
        pats = GROUPS[which[2:]]
        for n in m.graph.node:
            if n.op_type in TARGETS and len(n.input) > 1 and any(p in n.name for p in pats):
                used.add(n.input[1])
    elif which == "wconv":
        for n in m.graph.node:
            if n.op_type in TARGETS and len(n.input) > 1:
                used.add(n.input[1])
    for init in m.graph.initializer:
        if which != "w" and init.name not in used:
            continue
        a = numpy_helper.to_array(init)
        new = numpy_helper.from_array(round_to_fp16(a), init.name)
        init.CopyFrom(new)
    return m


def insert_fp16_activation_casts(model: onnx.ModelProto, ops: tuple) -> onnx.ModelProto:
    """Cast every output of `ops` through fp16 and back: what fp16 activation storage does."""
    m = copy.deepcopy(model)
    graph_outputs = {o.name for o in m.graph.output}
    out_nodes = []
    for n in m.graph.node:
        if n.op_type in ops and n.output[0] not in graph_outputs:
            old = n.output[0]
            h = old + "_h"
            n.output[0] = h
            out_nodes.append(n)
            out_nodes.append(onnx.helper.make_node("Cast", [h], [old], to=onnx.TensorProto.FLOAT,
                                                   name=old + "_castback"))
        else:
            out_nodes.append(n)
    g = onnx.helper.make_graph(out_nodes, m.graph.name, list(m.graph.input), list(m.graph.output),
                              list(m.graph.initializer))
    mm = onnx.helper.make_model(g, opset_imports=list(m.opset_import), producer_name="fp16_probe")
    mm.ir_version = model.ir_version
    return mm


def report(name: str, ref: np.ndarray, got: np.ndarray) -> dict:
    n = min(ref.size, got.size)
    a, b = ref[:n], got[:n]
    err = b - a
    rel = float(np.linalg.norm(err) / max(np.linalg.norm(a), 1e-12))
    snr = 20 * np.log10(max(np.sqrt(np.mean(a * a)), 1e-12) / max(np.sqrt(np.mean(err * err)), 1e-12))
    d = dict(variant=name, samples=int(n), rel_l2=rel, max_abs=float(np.max(np.abs(err))), sndr_db=float(snr))
    print(f"{name:12s} samples={n:7d} relL2={rel:.3e} max|err|={d['max_abs']:.3e} SNDR={snr:5.1f} dB")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/v21_mbistft_16k/primetts_v21_3voice.onnx")
    ap.add_argument("--variant", default="all", choices=["all", "w", "wconv", "act", "wact", "g:enc", "g:dec", "g:istft"])
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--noise-scale", type=float, default=0.0,
                    help="0.0 makes the flow sampler deterministic; any >0 adds sampling noise to the comparison")
    ap.add_argument("--out", default="spec/v21_fp16_probe.json")
    args = ap.parse_args()

    m = onnx.load(args.model)
    feed = build_case(args.T, args.seed, args.noise_scale)
    res = []
    ref = run(m, feed)
    res.append(report("fp32 ref", ref, ref))
    variants = ["w", "wconv", "g:enc", "g:dec", "g:istft"] if args.variant == "all" else [args.variant]
    for v in variants:
        got = run(variant_weights(m, v), feed)
        res.append(report(v, ref, got))
    if args.variant in ("all", "act"):
        got = run(insert_fp16_activation_casts(m, ("Conv", "ConvTranspose")), feed, optimize=False)
        res.append(report("act(fp16)", ref, got))
    if args.variant in ("all", "wact"):
        mm = insert_fp16_activation_casts(variant_weights(m, "wconv"), ("Conv", "ConvTranspose"))
        got = run(mm, feed, optimize=False)
        res.append(report("wconv+act", ref, got))
    with open(args.out, "w") as f:
        json.dump(dict(model=args.model, T=args.T, seed=args.seed, results=res), f, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
