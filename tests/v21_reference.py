#!/usr/bin/env python3
"""Pure-numpy oracle for v2.1, executed over build/v21.ptk + spec/v21_layers.json.

The C/CUDA runtime is written against THIS file: it is a readable, dependency-free
(numpy + onnxruntime for the check mode) implementation of the exact graph v2.1 is, reading its
weights out of the container in the packed layouts the runtime will read (conv [Cout4/4][K][Cin][4],
ConvTranspose as s phases x ceil(K/s) taps with the tap-j delay line), so a layout the oracle can
evaluate is a layout the C runtime can evaluate.

Gates (docs/v21_plan.md M1):
  per-node relative error < 1e-3 vs onnxruntime (default: --check tests/golden_v21/_probe_T81,
  a full dump of every node output for the T=81 golden case);
  end-to-end SNDR >= 60 dB vs the golden waveform (tests/golden_v21/<case>/wav.npy).

Measured facts the oracle relies on (do not re-derive them here):
  * noise: v2.1 is stochastic, so RandomNormalLike is evaluated as ZEROS and every case runs with
    noise_scale=0.0 (docs/NEXT.md); with noise_scale=0 the graph multiplies that noise away.
  * fp16: the container's '#h' payloads (the 39 decoder conv weights) are fp16-rounded values; the
    oracle does fp32 math on them, i.e. it models "fp16 weights + fp32 accumulate", which is the
    measured 67.4 dB configuration (docs/v21_plan.md), not fp16 math.
  * the decoder's iSTFT head needs the conv-transpose tail flush (nt-1 extra zero-input delay-line
    steps); gate (a) in tools/export_v21.py proves that mapping against an ONNX reference.

Usage:
  .venv/bin/python tests/v21_reference.py                      # T=81 golden, full node check
  .venv/bin/python tests/v21_reference.py --case T20           # waveform parity only
  .venv/bin/python tests/v21_reference.py --case all           # every golden case, SNDR per case
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import collections
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from export_v21 import (align64, conv_transpose1d_packed, group_of, pack_conv, pack_convtranspose,  # noqa: E402
                        r4, read_ptk, unpack_conv, unpack_convtranspose)

GOLDEN = "tests/golden_v21"
MODEL = "model/v21_mbistft_16k/primetts_v21_3voice.onnx"
CAST = {1: np.float32, 2: np.uint8, 3: np.int8, 5: np.int16, 6: np.int32, 7: np.int64,
        9: np.bool_, 10: np.float16, 11: np.float64}


def conv1d(X, W, B, s, p, d):
    """ONNX conv1d by kernel tap: tap k of a stride-s dilation-d conv reads xp[:, t*s + k*d]."""
    Cout, Cin, K = W.shape
    T = X.shape[-1]
    xp = np.zeros(X.shape[:-1] + (T + 2 * p,), X.dtype)
    xp[..., p:p + T] = X
    Tout = (T + 2 * p - d * (K - 1) - 1) // s + 1
    cols = np.arange(Tout) * s + np.arange(K)[None, :, None] * d  # (1,K,Tout)
    cols = cols.reshape(-1)
    out = np.zeros(X.shape[:-2] + (Cout, Tout), np.float32)
    taps = np.ascontiguousarray(W[:, :, :, None].reshape(Cout, Cin, K).transpose(2, 0, 1))  # [K,Cout,Cin]
    xp32 = np.ascontiguousarray(xp, np.float32)
    for k in range(K):
        out += taps[k] @ xp32[..., cols[k * Tout:(k + 1) * Tout]]
    return out + np.asarray(B, np.float32)[:Cout][None, :, None]


def convtranspose(X, W, B, s, p):
    """ONNX ConvTranspose via the packed phase layout (the runtime's own evaluation)."""
    Cin, Cout, K = W.shape
    xs = X.reshape(-1, Cin, X.shape[-1])
    Wp = pack_convtranspose(W, s, p)
    out = np.stack([conv_transpose1d_packed(xs[i], Wp, B, s, p, Cout, K) for i in range(xs.shape[0])])
    return out.reshape(X.shape[:-2] + (Cout, -1))


class Oracle:
    """Executes spec["nodes"] in graph order over container weights."""

    def __init__(self, container="build/v21.ptk", layers_json="spec/v21_layers.json"):
        self.back = read_ptk(container)
        self.spec = json.load(open(layers_json))
        self.layers = self.spec["layers"]
        self.by_idx = {l["idx"]: l for l in self.layers}
        self.weights = {}     # ONNX tensor name -> array, rebuilt from the container
        self.constants = {}   # Constant-node output name -> array
        self._weights()
        self._constants()

    def arr(self, name):
        # np.ascontiguousarray() would turn a 0-d (scalar) payload into shape (1,); keep the
        # recorded rank exactly as the container's dims say.
        return np.asarray(self.back[name][0], np.float32)

    def _weights(self):
        for l in self.layers:
            op, packed, p = l.get("op"), l.get("packed") or [], l.get("params", {})
            if not packed:
                continue
            onnx_names = l.get("onnx") or []
            if op == "conv1d":
                W = unpack_conv(self.arr(packed[0]))[: int(p["cout"])]
                if l.get("dtype") == "fp16_hilo" and len(packed) > 1:
                    W = W + unpack_conv(self.arr(packed[1]))[: int(p["cout"])]
                self.weights[onnx_names[0]] = np.ascontiguousarray(W)
                if len(onnx_names) > 1:
                    self.weights[onnx_names[1]] = self.arr(packed[1])[: int(p["cout"])]
            elif op == "conv_transpose1d":
                W = unpack_convtranspose(self.arr(packed[0]), int(p["stride"]), int(p["cout"]),
                                         int(p["cin"]), int(p["kernel"]))
                self.weights[onnx_names[0]] = np.ascontiguousarray(W)
                if len(onnx_names) > 1:
                    self.weights[onnx_names[1]] = self.arr(packed[1])[: int(p["cout"])]
            elif op in ("embedding", "pad_relative_position", "layernorm"):
                self.weights[onnx_names[0]] = self.arr(packed[0])
                if op == "layernorm" and len(onnx_names) > 1:
                    self.weights[onnx_names[1]] = self.arr(packed[1])
            elif op == "matmul" and onnx_names:
                W = unpack_conv(self.arr(packed[0]))[:, :, 0]
                self.weights[onnx_names[0]] = np.ascontiguousarray(W.T)

    def _constants(self):
        for c in self.spec["constants"]:
            if c.get("container"):
                self.constants[c["output"]] = self.arr(c["container"])
            elif c.get("values") is not None:
                dt = {"int64": np.int64, "int32": np.int32, "bool": np.bool_,
                      "float32": np.float32}.get(c["dtype"], np.float32)
                self.constants[c["output"]] = np.array(c["values"], dt).reshape(c["shape"])

    # ---- op implementations (ONNX opset 17 semantics, the set v2.1 actually uses) -------
    def _w(self, node, i):
        """Weight tensor for input i: container-derived when the exporter packed this node."""
        nm = node["inputs"][i]
        return self.weights.get(nm, None)

    def _conv(self, node, vals, transpose):
        X = np.ascontiguousarray(vals[node["inputs"][0]], np.float32)
        W = self._w(node, 1)
        if W is None:                                   # weight is a graph tensor (folded)
            W = np.ascontiguousarray(vals[node["inputs"][1]], np.float32)
        a = node["attrs"]
        s, d = int(a["strides"][0]), int(a.get("dilations", [1])[0])
        p = int(a["pads"][0])
        assert int(a.get("group", 1)) == 1, node["name"]
        bn = node["inputs"][2] if len(node["inputs"]) > 2 else ""
        B = np.zeros(W.shape[0], np.float32)
        if bn:
            v = vals.get(bn)
            if v is not None:
                v = np.asarray(v, np.float32).reshape(-1)
                B[: min(v.shape[0], W.shape[0])] = v[: min(v.shape[0], W.shape[0])]
        return convtranspose(X, W, B, s, p) if transpose else conv1d(X, W, B, s, p, d)

    def _slice(self, x, starts, ends, axes=None, steps=None):
        rank = x.ndim
        starts = [int(v) for v in np.asarray(starts).reshape(-1)]
        ends = [int(v) for v in np.asarray(ends).reshape(-1)]
        axes = list(range(len(starts))) if axes is None else [int(v) for v in np.asarray(axes).reshape(-1)]
        steps = [1] * len(starts) if steps is None else [int(v) for v in np.asarray(steps).reshape(-1)]
        for st, en, ax, sp in zip(starts, ends, axes, steps):
            ax = ax + rank if ax < 0 else ax
            n = x.shape[ax]
            if sp >= 0:
                st, en = min(max(st, -n), n), min(max(en, -n), n)
            else:
                st, en = min(max(st, -n - 1), n - 1), min(max(en, -n - 1), n - 1)
            sl = [slice(None)] * rank
            sl[ax] = slice(st, en, sp)
            x = x[tuple(sl)]
        return x

    def _pad(self, node, vals):
        x = vals[node["inputs"][0]]
        pads = [int(v) for v in np.asarray(vals[node["inputs"][1]]).reshape(-1)]
        r = x.ndim
        assert len(pads) == 2 * r, (node["name"], pads, x.shape)
        pads = [(pads[i], pads[i + r]) for i in range(r)]
        mode = str(node["attrs"].get("mode", "constant"))
        vn = node["inputs"][2] if len(node["inputs"]) > 2 else ""
        val = 0.0
        if vn and vn in vals:
            val = float(np.asarray(vals[vn]).reshape(-1)[0])
        if mode == "constant":
            return np.pad(x, pads, mode="constant", constant_values=val)
        return np.pad(x, pads, mode={"reflect": "reflect", "edge": "edge", "wrap": "wrap"}[mode])

    def _scatternd(self, data, indices, updates):
        out = np.array(data, copy=True)
        idx = np.asarray(indices, np.int64)
        k = int(idx.shape[-1])
        upd_shape = idx.shape[:-1] + tuple(data.shape[k:])
        upd = np.asarray(updates, data.dtype).reshape(upd_shape)
        for i in np.ndindex(idx.shape[:-1]):
            key = tuple(int(v) % out.shape[j] for j, v in enumerate(idx[i]))
            out[key] = upd[i]
        return out

    def _cumsum(self, x, axis, exclusive=False, reverse=False):
        ax = int(np.asarray(axis).reshape(-1)[0])
        ax = ax + x.ndim if ax < 0 else ax
        y = x[..., ::-1] if reverse else x
        y = np.cumsum(y, axis=ax, dtype=np.float32 if x.dtype == np.float32 else None)
        if exclusive:
            z = np.zeros_like(y)
            sl_a, sl_b = [slice(None)] * x.ndim, [slice(None)] * x.ndim
            sl_a[ax] = slice(1, None)
            sl_b[ax] = slice(None, -1)
            z[tuple(sl_a)] = y[tuple(sl_b)]
            y = z
        return y[..., ::-1] if reverse else y

    def _reduce(self, x, axes, keepdims, fn, noop_empty=False):
        if axes is None or np.asarray(axes).size == 0:
            ax = None if not noop_empty else tuple(range(x.ndim))
            ax = tuple(range(x.ndim)) if ax is None else ax
        else:
            ax = tuple(int(v) + x.ndim if int(v) < 0 else int(v) for v in np.asarray(axes).reshape(-1))
        return fn(x, axis=ax, keepdims=bool(keepdims))

    def _ln(self, x, scale, bias, axis, eps):
        ax = int(axis)
        ax = ax + x.ndim if ax < 0 else ax
        axes = tuple(range(ax, x.ndim))
        x = np.asarray(x, np.float32)
        mean = x.mean(axis=axes, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * np.asarray(scale, np.float32) + np.asarray(bias, np.float32)

    def _softmax(self, x, axis):
        x = np.asarray(x, np.float32)
        ax = int(axis)
        m = x.max(axis=ax, keepdims=True)
        e = np.exp(x - m)
        return e / e.sum(axis=ax, keepdims=True)

    def _range(self, start, limit, delta):
        start, limit, delta = (float(np.asarray(v).reshape(-1)[0]) for v in (start, limit, delta))
        n = int(np.ceil((limit - start) / delta)) if delta > 0 else 0
        n = max(n, 0)
        return start + delta * np.arange(n, dtype=np.float32 if isinstance(start, float) else np.int64)

    def _const_of_shape(self, shape, fill):
        shape = [int(v) for v in np.asarray(shape).reshape(-1)]
        v = np.array(fill["value"], {"int64": np.int64, "int32": np.int32, "bool": np.bool_,
                                     "float32": np.float32}.get(fill["dtype"], np.float32))
        val = v.reshape(-1)[0] if v.size else np.array(0, np.float32)
        return np.full(shape, val, val.dtype if hasattr(val, "dtype") else np.float32)

    def _expand(self, x, shape):
        sh = tuple(int(v) for v in np.asarray(shape).reshape(-1))
        if not sh:                        # shape tensor collapsed to a scalar/empty: identity
            return np.asarray(x)
        if np.ndim(x) == 0 and len(sh) == 1:   # scalar broadcast, e.g. x=1.0 -> [n]
            return np.broadcast_to(np.asarray(x).item(), sh).copy()
        return np.broadcast_to(np.asarray(x), sh)

    def _split(self, x, axis, splits):
        axis = int(axis)
        if splits is None:
            n = x.shape[axis] // len(splits) if splits else x.shape[axis]
            return [x]
        cuts = np.cumsum([int(v) for v in np.asarray(splits).reshape(-1)])[:-1]
        return list(np.split(x, cuts, axis=axis))

    def _div(self, x, y):
        if np.issubdtype(np.asarray(x).dtype, np.integer) and np.issubdtype(np.asarray(y).dtype, np.integer):
            q = np.trunc(np.asarray(x, np.float64) / np.asarray(y, np.float64))
            return q.astype(np.asarray(x).dtype)
        return np.asarray(x, np.float32) / np.asarray(y, np.float32)

    def _reshape(self, x, shape):
        sh = [int(v) for v in np.asarray(shape).reshape(-1)]
        out = [x.shape[i] if (v == 0 and i < x.ndim) else v for i, v in enumerate(sh)]
        return x.reshape(out)

    def _unsqueeze(self, x, axes):
        for ax in sorted(int(v) for v in np.asarray(axes).reshape(-1)):
            x = np.expand_dims(x, ax)
        return x

    def _squeeze(self, x, axes):
        for ax in sorted((a + x.ndim if a < 0 else a) for a in
                         [int(v) for v in np.asarray(axes).reshape(-1)]):
            x = np.squeeze(x, axis=ax)
        return x

    def exec(self, node, vals, entry):
        op = node["op"]
        ins = [i for i in node["inputs"]]
        a = node["attrs"]
        g = lambda i: vals[ins[i]] if i < len(ins) and ins[i] else None  # noqa: E731

        if op in ("Constant", "ConstantOfShape") and op == "Constant":
            return []                                     # pre-seeded from the constants table
        if op == "ConstantOfShape":
            return [self._const_of_shape(g(0), entry["fill"])]
        if op == "Conv":
            return [self._conv(node, vals, False)]
        if op == "ConvTranspose":
            return [self._conv(node, vals, True)]
        if op == "MatMul":
            return [np.matmul(np.asarray(g(0), np.float32), np.asarray(g(1), np.float32))]
        if op == "Add":
            return [np.asarray(g(0)) + np.asarray(g(1))]
        if op == "Sub":
            return [np.asarray(g(0)) - np.asarray(g(1))]
        if op == "Mul":
            return [np.asarray(g(0)) * np.asarray(g(1))]
        if op == "Div":
            return [self._div(g(0), g(1))]
        if op == "Pow":
            return [np.power(np.asarray(g(0), np.float32), np.asarray(g(1), np.float32))]
        if op == "Neg":
            return [-np.asarray(g(0))]
        if op in ("Exp", "Sin", "Cos", "Tanh", "Sigmoid", "Relu", "Ceil", "Floor", "Abs", "Sqrt",
                  "Round", "Sign"):
            f = {"Exp": np.exp, "Sin": np.sin, "Cos": np.cos, "Tanh": np.tanh,
                 "Sigmoid": lambda v: 1.0 / (1.0 + np.exp(-v)),
                 "Relu": lambda v: np.maximum(v, 0), "Ceil": np.ceil, "Floor": np.floor,
                 "Abs": np.abs, "Sqrt": np.sqrt, "Round": np.round, "Sign": np.sign}[op]
            return [f(np.asarray(g(0), np.float32))]
        if op == "LeakyRelu":
            al = float(a.get("alpha", 0.01))
            v = np.asarray(g(0), np.float32)
            return [np.where(v >= 0, v, al * v)]
        if op == "Clip":
            v = np.asarray(g(0), np.float32)
            lo, hi = g(1), g(2)
            if lo is not None and np.asarray(lo).size:
                v = np.maximum(v, np.asarray(lo, np.float32))
            if hi is not None and np.asarray(hi).size:
                v = np.minimum(v, np.asarray(hi, np.float32))
            return [v]
        if op == "Cast":
            return [np.asarray(g(0)).astype(CAST[int(a["to"])])]
        if op == "Shape":
            return [np.array(np.asarray(g(0)).shape, np.int64)]
        if op == "Reshape":
            return [self._reshape(np.asarray(g(0)), np.asarray(g(1)))]
        if op == "Transpose":
            perm = [int(v) for v in a["perm"]]
            return [np.transpose(np.asarray(g(0)), perm)]
        if op == "Concat":
            xs = [np.asarray(vals[n]) for n in ins if n]
            return [np.concatenate(xs, axis=int(a["axis"]))]
        if op == "Split":
            axis = int(a["axis"])
            sp = g(1) if len(ins) > 1 else a.get("split")
            x = np.asarray(g(0))
            if sp is None:
                k = x.shape[axis]
                return [x]
            cuts = np.cumsum([int(v) for v in np.asarray(sp).reshape(-1)])[:-1]
            return list(np.split(x, cuts, axis=axis))
        if op == "Unsqueeze":
            return [self._unsqueeze(np.asarray(g(0)), g(1))]
        if op == "Squeeze":
            return [self._squeeze(np.asarray(g(0)), g(1))]
        if op == "Gather":
            return [np.take(np.asarray(g(0)), np.asarray(g(1), np.int64), axis=int(a.get("axis", 0)))]
        if op == "Slice":
            return [self._slice(np.asarray(g(0)), g(1), g(2), g(3), g(4))]
        if op == "Pad":
            return [self._pad(node, vals)]
        if op == "Where":
            return [np.where(np.asarray(g(0)).astype(bool), np.asarray(g(1)), np.asarray(g(2)))]
        if op in ("Equal", "Less", "Greater", "GreaterOrEqual", "LessOrEqual", "And", "Or", "Not",
                  "Xor"):
            if op == "Not":
                return [np.logical_not(np.asarray(g(0)))]
            f = {"Equal": np.equal, "Less": np.less, "Greater": np.greater,
                 "GreaterOrEqual": np.greater_equal, "LessOrEqual": np.less_equal,
                 "And": np.logical_and, "Or": np.logical_or, "Xor": np.logical_xor}[op]
            return [f(np.asarray(g(0)), np.asarray(g(1)))]
        if op == "Expand":
            return [self._expand(np.asarray(g(0)), np.asarray(g(1)))]
        if op == "ScatterND":
            return [self._scatternd(np.asarray(g(0)), np.asarray(g(1)), np.asarray(g(2)))]
        if op == "CumSum":
            return [self._cumsum(np.asarray(g(0)), g(1), bool(int(a.get("exclusive", 0))),
                                 bool(int(a.get("reverse", 0))))]
        if op in ("ReduceSum", "ReduceMax", "ReduceMin", "ReduceMean"):
            fn = {"ReduceSum": np.sum, "ReduceMax": np.max, "ReduceMin": np.min,
                  "ReduceMean": np.mean}[op]
            x = np.asarray(g(0), np.float32) if fn in (np.sum, np.mean) else np.asarray(g(0))
            return [self._reduce(x, g(1) if len(ins) > 1 else None, a.get("keepdims", 1), fn)]
        if op == "Softmax":
            return [self._softmax(g(0), int(a.get("axis", -1)))]
        if op == "LayerNormalization":
            return [self._ln(g(0), g(1), g(2), int(a.get("axis", -1)), float(a.get("epsilon", 1e-5)))]
        if op == "Einsum":
            eq = str(a["equation"]).replace("\u2192", "->")
            return [np.einsum(eq, np.asarray(g(0), np.float32), np.asarray(g(1), np.float32))]
        if op == "Range":
            return [self._range(g(0), g(1), g(2))]
        if op == "RandomNormalLike":
            # v2.1 parity runs at noise_scale=0 (docs/NEXT.md): the graph multiplies this noise by
            # noise_scale, so the parity-exact value is zero.  Documented, not assumed.
            dt = CAST.get(int(a.get("dtype", 1)), np.float32)
            return [np.zeros(np.asarray(g(0)).shape, dt)]
        if op == "Identity":
            return [np.asarray(g(0))]
        if op == "NonZero":
            return [np.stack(np.nonzero(np.asarray(g(0))), 0).astype(np.int64)]
        raise NotImplementedError(f"op {op} at {node['name']} is not implemented in the oracle")

    def run(self, feeds):
        vals = dict(self.constants)
        vals.update(self.weights)
        vals.update({k: np.asarray(v) for k, v in feeds.items()})
        for node in self.spec["nodes"]:
            if node["op"] == "Constant":
                continue
            entry = self.by_idx.get(node["idx"], {})
            outs = self.exec(node, vals, entry)
            names = [o for o in node["outputs"]]
            for nm, v in zip(names, outs):
                if nm:
                    vals[nm] = v
        return vals


def load_case(case_dir):
    feeds = {}
    for k in ("x", "tone", "lang", "x_lengths", "sid", "noise_scale", "length_scale"):
        feeds[k] = np.load(os.path.join(case_dir, k + ".npy"))
    return feeds


def rel_err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:                       # empty == empty is exact; shape mismatch is caught above
        return 0.0
    scale = max(float(np.max(np.abs(b))), 1e-30)
    return float(np.max(np.abs(a - b))) / scale


def sndr(est, ref):
    est, ref = np.asarray(est, np.float64).reshape(-1), np.asarray(ref, np.float64).reshape(-1)
    n = min(est.shape[0], ref.shape[0])
    est, ref = est[:n], ref[:n]
    d = est - ref
    p = float(np.sum(ref ** 2))
    q = float(np.sum(d ** 2))
    return 10.0 * np.log10(p / max(q, 1e-30))


def noise_exemption(o):
    """Nodes whose value carries the v2.1 sampler (RandomNormalLike) and dies at the noise_scale
    multiply - measured: /RandomNormalLike -> /Mul_5(noise, exp(logs)) -> /Mul_6(x, noise_scale).

    Parity runs at noise_scale=0.0, so from the noise_scale multiply onward the branch is exactly
    zero in onnxruntime too; only the sampler and the products that feed that multiply can differ,
    and they differ by construction.  The exempt set is RandomNormalLike plus every node in its
    forward closure whose outputs feed (directly or transitively) a node consuming noise_scale.
    They are reported, not gated - everything downstream of /Mul_6 IS gated (measured rel 0).
    """
    nodes = o.spec["nodes"]
    prod = {n["outputs"][0]: n for n in nodes if n["outputs"]}
    consumers = collections.defaultdict(list)
    for n in nodes:
        for i in n["inputs"]:
            if i:
                consumers[i].append(n)
    anc = set()                                        # tensors that reach a noise_scale consumer
    stack = [i for n in nodes if "noise_scale" in n["inputs"] for i in n["inputs"] if i]
    while stack:
        t = stack.pop()
        if t in anc:
            continue
        anc.add(t)
        p = prod.get(t)
        if p:
            stack += [i for i in p["inputs"] if i]
    ex, stack = set(), []
    for n in nodes:
        if n["op"] == "RandomNormalLike":
            ex.add(n["idx"])
            stack += [o for o in n["outputs"] if o]
    while stack:
        t = stack.pop()
        for c in consumers.get(t, []):
            outs = [o for o in c["outputs"] if o]
            if outs and c["idx"] not in ex and all(o in anc for o in outs):
                ex.add(c["idx"])
                stack += outs
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="build/v21.ptk")
    ap.add_argument("--layers-json", default="spec/v21_layers.json")
    ap.add_argument("--golden", default=GOLDEN)
    ap.add_argument("--case", default="T81", help="golden case dir name, or 'all'")
    ap.add_argument("--probe", default=os.path.join(GOLDEN, "_probe_T81"),
                    help="full onnxruntime node dump for --probe-case; 'none' to skip the node gate")
    ap.add_argument("--probe-case", default="T81",
                    help="the case the node dump was generated for; other cases get the audio gate only")
    ap.add_argument("--rel", type=float, default=1e-3, help="per-node rel gate, fp32-weight groups")
    ap.add_argument("--rel-fp16", type=float, default=2e-3,
                    help="per-node rel gate for the decoder (its weights are fp16 in the container)")
    ap.add_argument("--sndr-min", type=float, default=60.0)
    args = ap.parse_args()

    o = Oracle(args.container, args.layers_json)
    cases = sorted(d for d in os.listdir(args.golden) if os.path.isdir(os.path.join(args.golden, d))
                   and not d.startswith("_")) if args.case == "all" else [args.case]
    fail, all_worst = 0, []
    for c in cases:
        d = os.path.join(args.golden, c)
        feeds = load_case(d)
        noise = float(np.asarray(feeds["noise_scale"]).reshape(-1)[0])
        t0 = time.time()
        vals = o.run(feeds)
        dt = time.time() - t0
        wav = vals.get("wav")
        golden_wav = np.load(os.path.join(d, "wav.npy"))
        s = sndr(wav, golden_wav)
        line = (f"{c:10s} T={int(feeds['x_lengths'][0]):3d} sid={int(feeds['sid'][0])} "
                f"samples={golden_wav.shape[-1]:6d} oracle {dt:5.2f}s  SNDR={s:6.2f} dB")
        if s < args.sndr_min:
            fail += 1
        probe_dir = None if args.probe in ("none", "") else args.probe
        if probe_dir and c != args.probe_case:
            probe_dir = None                       # the node dump exists for one case only
        if probe_dir and os.path.isdir(probe_dir):
            exempt = noise_exemption(o) if noise == 0.0 else set()
            per = {}
            over, worst = 0, []
            noise_rows = []
            for node in o.spec["nodes"]:
                nm = node["outputs"][0] if node["outputs"] else ""
                p = os.path.join(probe_dir, nm.replace("/", "_") + ".npy")
                if not nm or not os.path.exists(p) or nm not in vals:
                    continue
                r = rel_err(vals[nm], np.load(p))
                grp = group_of(node["name"] or nm)
                key = "dec (fp16 w)" if grp == "dec" else grp
                st = per.setdefault(key, dict(n=0, worst=0.0, worst_node="", over=0))
                st["n"] += 1
                if r > st["worst"]:
                    st["worst"], st["worst_node"] = r, f"{node['op']} {nm}"
                if node["idx"] in exempt:
                    noise_rows.append((r, node["op"], nm))
                    continue
                thr = args.rel_fp16 if grp == "dec" else args.rel
                if r > thr:
                    over += 1
                    st["over"] += 1
                    worst.append((r, node["op"], nm))
            line += f"  nodes={sum(v['n'] for v in per.values())} over gate: {over}"
            line += f"  noise-branch exempt: {len(noise_rows)}"
            fail += 1 if over else 0
            print(line)
            for k in sorted(per):
                st = per[k]
                print(f"    {k:14s} n={st['n']:5d} worst rel={st['worst']:.3e}  "
                      f"({st['worst_node']})  over={st['over']}")
            if noise_rows:
                print(f"    noise branch (not gated, noise_scale=0): "
                      + ", ".join(f"{op}" for _r, op, _n in noise_rows[:6]))
            worst.sort(reverse=True)
            for r, op, nm in worst[:10]:
                all_worst.append((r, c, op, nm))
        else:
            print(line)
    if all_worst:
        all_worst.sort(reverse=True)
        print("worst gated nodes (rel, case, op, tensor):")
        for r, c, op, nm in all_worst[:12]:
            print(f"  {r:.3e}  {c:8s} {op:16s} {nm}")
    if fail:
        print(f"GATE FAILED: {fail} failure(s) (fp32 rel {args.rel:g}, dec fp16 rel {args.rel_fp16:g}, "
              f"SNDR >= {args.sndr_min} dB)")
        return 2
    print(f"gates pass: per-node rel < {args.rel:g} (fp32 groups) / {args.rel_fp16:g} (dec, fp16 "
          f"weights), SNDR >= {args.sndr_min} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
