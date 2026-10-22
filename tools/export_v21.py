#!/usr/bin/env python3
"""Export PrimeTTS v2.1 (model/v21_mbistft_16k/primetts_v21_3voice.onnx) into ONE .ptk
container plus an execution-ordered layer spec the CUDA runtime can be written from.

Container = the v1 layout (runtime/py/pt.py, runtime/src/weights.c) unchanged:
  header  <IIIiq   magic=0x50544B31 'PTK1', version, n_tensor, reserved, data_bytes
  records n_tensor x <64si4x6qqq  name, ndim, dim[6], nelem, off   (136 bytes each)
  pad     zeros up to a 64-byte boundary
  data    payloads, each 64-byte aligned; `off` is relative to the data start

v2.1 adds two things, both carried inside that layout (no format break):
  * payload width per tensor: fp32 (4 B) by default, fp16 (2 B) for tensors whose name ends in
    '#h'.  A tensor named `dtype.table` (fp32, one entry per record, in record order) stores the
    same fact numerically (4.0 / 2.0) for readers that prefer data over name conventions.
    The v1 reader (pt.py) assumes 4 B payloads, so it reads '#h' payloads as garbage until the
    runtime lane teaches it the width; every fp16 tensor is named and recorded explicitly in
    spec/v21_layers.json, so the conversion is mechanical.
  * Constant nodes are exported too, not just initializers (v2.1 has 1924 of them; three of the
    graph's conv weights come from Constant nodes, the rest are shape/mask plumbing).  Float
    constants go into the container as `const.<nodeidx>`; int64/int32/bool constants (index and
    slice vectors) are recorded verbatim as literals in the JSON `constants` table instead,
    because the fp32 payload format cannot carry int64.

Packed layouts are the v1 conventions (runtime/include/pt.h), generalised for arbitrary kernel
and stride:
  "<grp>.<mod>.wp"   conv weight [Cout,Cin,K] -> [Cout4/4][K][Cin][4] (4 consecutive output
                     channels contiguous, Cout zero-padded to a multiple of 4).  A linear
                     y = x @ M is the K=1 case with W = M^T.
  "<grp>.<mod>.b"    conv bias, zero-padded to Cout4, added once per output channel.
  "<grp>.<mod>.wp#h" same packing, fp16 payload (decoder conv weights only).
  ConvTranspose (stride s, kernel K, pad p) -> s phases of an ntap=ceil(K/s)-tap conv, phase
                     major: row (r*C4 + c) is phase r / output channel c, tap j is ONNX kernel
                     tap k = r + j*s.  Packed with pack_conv, so one 4-row block = 4 channels of
                     one phase.  The runtime writes out[q*s + r - pad] = P_r[q]; output length is
                     (Tin-1)*s - 2*pad + K.  check_convT() proves the packing reproduces the ONNX
                     conv_transpose1d output bit-for-bit, for every stride in this model.
  HiLo split (--hilo enc) is the 8-stream upgrade path: W = Whi + Wlo, both halves fp16, packed
  as `<name>.whi#h` / `<name>.wlo#h`.  Gate (a) checks the sum against the ONNX tensor.

HARD GATES (all three must pass, otherwise exit non-zero naming the offending tensor):
  (a) every exported weight, reconstructed from the packed tensors (HiLo summed), matches the
      ONNX tensor elementwise within 1e-6 (fp16 tensors are compared against the fp16-rounded
      ONNX tensor, which is what fp16 storage can represent);
  (b) every initializer and every Constant tensor that feeds a compute node has a packed entry;
  (c) container payload bytes match the predicted dtype budget (fp32/fp16 itemsize policy,
      padding included) within 2%.

Usage: .venv/bin/python tools/export_v21.py [--model model/v21_mbistft_16k/primetts_v21_3voice.onnx]
           [--out build/v21.ptk] [--layers-json spec/v21_layers.json]
           [--probe-dir tests/golden_v21/_probe] [--hilo none|enc|dec]

--probe-dir supplies the ONE weight v2.1 computes at run time (/dec/ConvTranspose_2's 4x4x4
kernel, built by ScatterND in the graph).  It is read from a tools/probe_shapes.py dump
(naming convention: tensor name with '/' replaced by '_', .npy).
"""
from __future__ import annotations

import argparse
import math
import collections
import json
import os
import struct

import numpy as np
import onnx
from onnx import helper, numpy_helper

MAGIC = 0x50544B31
VERSION = 2
HDR = struct.Struct("<IIIiq")
REC = struct.Struct("<64si4x6qqq")
assert HDR.size == 24 and REC.size == 136

FP32, FP16 = 4, 2
H16 = "#h"  # name suffix meaning "fp16 payload"
ITEMSIZE = {FP32: 4, FP16: 2}


def align64(n: int) -> int:
    return (n + 63) // 64 * 64


def r4(n: int) -> int:
    return (n + 3) // 4 * 4


def pack_conv(W: np.ndarray) -> np.ndarray:
    """[Cout,Cin,K] -> [Cout4/4][K][Cin][4] (v1 micro-kernel layout)."""
    Cout, Cin, K = W.shape
    C4 = r4(Cout)
    Wp = np.zeros((C4, Cin, K), np.float32)
    Wp[:Cout] = W
    return np.ascontiguousarray(Wp.reshape(C4 // 4, 4, Cin, K).transpose(0, 3, 2, 1))


def unpack_conv(Wp: np.ndarray) -> np.ndarray:
    """inverse of pack_conv: [Cout4/4][K][Cin][4] -> [Cout4][Cin][K]."""
    b, K, Cin, _ = Wp.shape
    return np.ascontiguousarray(Wp.transpose(0, 3, 2, 1)).reshape(b * 4, Cin, K)


def convT_ntap(K: int, s: int) -> int:
    return (K + s - 1) // s


def pack_convtranspose(W: np.ndarray, s: int, pad: int) -> np.ndarray:
    """ONNX ConvTranspose W [Cin,Cout,K] -> s phases x ntap taps, then pack_conv.

    phase r / output channel c / tap j reads ONNX kernel tap k = r + j*s, so the phase-r
    conv output P_r[q] lands at output index t = q*s + r - pad (the ONNX index relation).
    """
    Cin, Cout, K = W.shape
    C4, nt = r4(Cout), convT_ntap(K, s)
    Wph = np.zeros((s * C4, Cin, nt), np.float32)
    for r in range(s):
        for j in range(nt):
            k = r + j * s
            if k < K:
                Wph[r * C4:r * C4 + Cout, :, j] = W[:, :, k].T
    return pack_conv(Wph)


def unpack_convtranspose(Wp: np.ndarray, s: int, Cout: int, Cin: int, K: int) -> np.ndarray:
    Wph = unpack_conv(Wp)
    nt = Wph.shape[2]
    C4 = r4(Cout)
    W = np.zeros((Cin, Cout, K), np.float32)
    for r in range(s):
        for j in range(nt):
            k = r + j * s
            if k < K:
                W[:, :, k] = Wph[r * C4:r * C4 + Cout, :, j].T
    return W


def split_hilo(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Markidis 2-term split: W = hi + lo with both terms fp16-representable (fp32 values here)."""
    hi = W.astype(np.float16).astype(np.float32)
    lo = (W - hi).astype(np.float16).astype(np.float32)
    return hi, lo


def as_fp16_values(W: np.ndarray) -> np.ndarray:
    return W.astype(np.float16).astype(np.float32)


def pad_vec(b: np.ndarray, n: int, fill: float = 0.0) -> np.ndarray:
    out = np.full(n, fill, np.float32)
    out[: b.shape[0]] = b
    return out


class GateFailure(Exception):
    """Raised by a hard gate; main() prints the offending tensor and exits non-zero."""


def conv1d_ref(X: np.ndarray, W: np.ndarray, B: np.ndarray, stride: int, pad: int, dilation: int) -> np.ndarray:
    """ONNX conv1d reference: X [Cin,T] -> [Cout,Tout].  Used only by the packing checks.

    patches[c,t,k] = xp[c, t*stride + k*dilation]: the flattened index is t-major, so the reshape
    is (Cin, Tout, K) and the contraction emits the Cout axis ("ock,ctk->ot").
    """
    Cout, Cin, K = W.shape
    T = X.shape[1]
    xp = np.zeros((Cin, T + 2 * pad), np.float32)
    xp[:, pad:pad + T] = X
    Tout = (T + 2 * pad - dilation * (K - 1) - 1) // stride + 1
    idx = (np.arange(Tout)[:, None] * stride + np.arange(K)[None, :] * dilation).reshape(-1)
    patches = xp[:, idx].reshape(Cin, Tout, K)
    return np.einsum("ock,ctk->ot", W, patches) + np.asarray(B, np.float32)[:Cout, None]


def conv_transpose1d_ref(X: np.ndarray, W: np.ndarray, B: np.ndarray, stride: int, pad: int) -> np.ndarray:
    """ONNX conv_transpose1d reference: X [Cin,Tin] -> [Cout,Tout], Tout=(Tin-1)*s-2*pad+K.

    Bias is one value per output channel, broadcast over the Tout columns (bias[:, None]).
    """
    Cin, Cout, K = W.shape
    Tin = X.shape[1]
    Tout = (Tin - 1) * stride - 2 * pad + K
    out = np.tile(np.asarray(B, np.float32)[:Cout][:, None], (1, Tout))
    for k in range(K):
        t = np.arange(Tin) * stride + k - pad
        ok = (t >= 0) & (t < Tout)
        if not ok.any():
            continue
        contrib = W[:, :, k].T @ X[:, ok]  # [Cout, n_ok]
        out[:, t[ok]] += contrib
    return out


def conv_transpose1d_packed(X: np.ndarray, Wp: np.ndarray, B: np.ndarray, s: int, pad: int,
                            Cout: int, K: int) -> np.ndarray:
    """What the runtime does: s phases x ntap taps over q, then out[q*s + r - pad] = P_r[q].

    Phase r / tap j is ONNX kernel tap k = r + j*s, and for output index t = q*s + r - pad that
    tap reads input q - j (because t = (q-j)*s + k - pad), so tap j is a j-step delay of x.
    With K > s (nt > 1) the last nt-1 steps consume zero input and produce the tail outputs, so
    the delay line runs Tin + nt - 1 steps (the runtime flushes them at stream end).
    """
    Cin, Tin = X.shape
    nt = convT_ntap(K, s)
    Wph = unpack_conv(Wp)  # [s*C4][Cin][nt]
    C4 = r4(Cout)
    # taps j>0 read x[q-j], so the last nt-1 phase steps consume zero input and produce the
    # tail outputs (t up to (Tin-1)*s + K - 1 - pad): the delay line runs Tin + nt - 1 steps.
    Q = Tin + nt - 1
    xp = np.zeros((Cin, Q), np.float32)
    xp[:, :Tin] = X
    P = np.zeros((s, C4, Q), np.float32)
    for r in range(s):
        for j in range(nt):
            wr = Wph[r * C4:(r + 1) * C4, :, j]  # [C4,Cin]
            if j == 0:
                P[r] += wr @ xp
            else:
                P[r, :, j:] += wr @ xp[:, :-j]
        P[r] += B[:C4][:, None]
    Tout = (Tin - 1) * s - 2 * pad + K
    out = np.zeros((Cout, Tout), np.float32)
    for q in range(Q):
        for r in range(s):
            t = q * s + r - pad
            if 0 <= t < Tout:
                out[:, t] = P[r, :Cout, q]
    return out


# --------------------------------------------------------------------------- container

def write_ptk(path: str, tensors: list[tuple[str, np.ndarray, int]]) -> tuple[int, int]:
    """tensors: ordered [(name, array, itemsize)].  Payload width comes from itemsize; values are
    rounded to fp16 when itemsize==2.  Returns (n_tensor, data_bytes)."""
    recs, off = [], 0
    for name, a, isz in tensors:
        assert len(name) < 64, name
        dims = list(a.shape) + [0] * (6 - a.ndim)
        recs.append(REC.pack(name.encode(), a.ndim, *dims, a.size, off))
        off += align64(a.size * isz)
    hd = HDR.size + REC.size * len(tensors)
    hdp = align64(hd)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(HDR.pack(MAGIC, VERSION, len(tensors), 0, off))
        for r in recs:
            f.write(r)
        f.write(b"\0" * (hdp - hd))
        for name, a, isz in tensors:
            b = (a.astype(np.float16) if isz == 2 else np.asarray(a, np.float32)).tobytes()
            f.write(b)
            f.write(b"\0" * (align64(len(b)) - len(b)))
    return len(tensors), off


def read_ptk(path: str) -> dict[str, tuple[np.ndarray, int]]:
    """Reader that knows the fp16 payloads: name'#h' -> 2-byte values.  Mirrors the C side."""
    raw = open(path, "rb").read()
    magic, ver, n, _res, nbytes = HDR.unpack_from(raw, 0)
    if magic != MAGIC:
        raise GateFailure(f"bad magic 0x{magic:x} in {path}")
    data_off = align64(HDR.size + n * REC.size)
    out: dict[str, tuple[np.ndarray, int]] = {}
    for i in range(n):
        name, ndim, d0, d1, d2, d3, d4, d5, nelem, off = REC.unpack_from(raw, HDR.size + i * REC.size)
        name = name.split(b"\0")[0].decode()
        isz = 2 if name.endswith(H16) else 4
        dt = np.float16 if isz == 2 else np.float32
        dims = [d for d in (d0, d1, d2, d3, d4, d5)][: max(ndim, 0)]
        arr = np.frombuffer(raw, dtype=dt, count=nelem, offset=data_off + off)
        out[name] = (arr.reshape(dims) if ndim else arr.reshape(())).copy(), isz
    got = align64(HDR.size + n * REC.size)
    assert data_off == got
    return out


# --------------------------------------------------------------------------- graph model

GROUP_OF = {"/enc_p": "enc", "/dp": "dp", "/flow": "flow", "/dec": "dec", "/emb_g": "tok", "/": "root"}


def group_of(node_name: str) -> str:
    for pat, g in GROUP_OF.items():
        if node_name.startswith(pat):
            return g
    return "root"


def module_of(node_name: str) -> str:
    """'/enc_p/encoder/attn_layers.0/conv_q/Conv' -> 'encoder.attn_layers.0.conv_q'.

    Collapses the torch Sequential scopes and drops the trailing op name.  Nodes that sit
    directly under a group ('/dec/ConvTranspose_2') keep their own name, so packed tensor names
    stay unique ('dec.ConvTranspose_2.wp')."""
    segs = [s for s in node_name.split("/") if s]
    if len(segs) <= 2:  # node sits directly under a group root: keep its own name
        return segs[-1]
    out: list[str] = []
    for s in segs[:-1]:
        if out and s.startswith(out[-1] + "."):
            out[-1] = s
        else:
            out.append(s)
    return ".".join(out) or (segs[-1] if segs else "root")


def attrs_of(node) -> dict:
    a = {}
    for x in node.attribute:
        v = helper.get_attribute_value(x)
        if isinstance(v, bytes):
            v = v.decode()
        a[x.name] = v
    return a


# ops the CUDA runtime must implement as fused elementwise / reduction kernels (the plan's list:
# LayerNorm, Snake/SiLU, Softmax, Exp, mean, Slice-masking, ScatterND/Where) + the shape algebra
# v2.1 needs to move data around.  kind tags tell the runtime which entries are arithmetic
# (fusable) and which are index bookkeeping (compile-time for a fixed T bucket).
ACT_OPS = {"Add", "Sub", "Mul", "Div", "Neg", "Tanh", "Sigmoid", "Relu", "LeakyRelu", "Clip",
           "Exp", "Sin", "Cos", "Pow", "Sqrt", "Floor", "Ceil", "Round", "Abs", "Sqrt"}
REDUCE_OPS = {"Softmax", "LayerNormalization", "ReduceSum", "ReduceMax", "ReduceMean", "ReduceMin",
              "CumSum", "Mean"}
MASK_OPS = {"Where", "Equal", "Less", "Greater", "GreaterOrEqual", "Not", "And", "Or", "ScatterND",
            "NonZero", "Cast", "Sign"}
SHAPE_OPS = {"Shape", "Reshape", "Transpose", "Concat", "Slice", "Split", "Unsqueeze", "Squeeze",
             "Expand", "Pad", "Gather", "Range", "ConstantOfShape", "Identity", "Einsum",
             "RandomNormalLike", "Tile", "Max", "Min", "Sub", "Mul", "Add", "Div", "Equal", "Cast",
             "Gemm", "Flatten", "NonZero", "EyeLike", "Neg", "Where", "Greater"}


class Exporter:
    def __init__(self, model_path: str, probe_dir: str | None, hilo: str = "none"):
        self.model_path = model_path
        self.probe_dir = probe_dir
        self.hilo = hilo
        self.tensors: list[tuple[str, np.ndarray, int]] = []   # (name, array, itemsize)
        self.names: set[str] = set()
        self.layers: list[dict] = []
        self.constants: list[dict] = []      # Constant-node table (JSON side)
        self.nodes: list[dict] = []          # every node, execution order
        self.covered: set[str] = set()       # initializer names with a packed entry
        self.covered_const: set[str] = set()  # Constant outputs exported
        self.convT: list[dict] = []          # for check_convT
        self.conv: list[dict] = []           # for check_conv
        self.bytes_by_group: dict[str, int] = collections.Counter()
        self.params_by_group: dict[str, int] = collections.Counter()
        self.m = onnx.load(model_path)
        self.g = self.m.graph
        self.init = {i.name: numpy_helper.to_array(i) for i in self.g.initializer}
        self.producer = {o: n for n in self.g.node for o in n.output}
        self.folded: list[str] = []      # graph-computed weights folded to constants
        self._fold_cache: dict[str, np.ndarray] = {}
        self.gate_e: list[str] = []      # fold-vs-ONNXRuntime evidence lines
        self.const: dict[str, np.ndarray] = {}
        self.const_node: dict[str, int] = {}
        for idx, n in enumerate(self.g.node):
            if n.op_type == "Constant":
                self.const[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
                self.const_node[n.output[0]] = idx
        self.consumers: dict[str, list] = {}
        for n in self.g.node:
            for i in n.input:
                self.consumers.setdefault(i, []).append(n)

    # ---- container helpers -------------------------------------------------------------
    def add(self, name: str, arr: np.ndarray, group: str, itemsize: int = FP32, src: str = "") -> str:
        assert name not in self.names, f"duplicate tensor name {name}"
        assert len(name) < 64, f"tensor name too long ({len(name)}): {name}"
        a = np.asarray(arr, np.float32)
        a = np.asarray(a, order="C")   # C-contiguous, rank-preserving (ascontiguousarray turns a
        # 0-d scalar payload into shape (1,), which would mis-shape the 33 scalar constants)
        self.tensors.append((name, a, itemsize))
        self.names.add(name)
        self.bytes_by_group[group] += align64(a.size * itemsize)
        self.params_by_group[group] += a.size
        return name

    # ---- constant folding of graph-computed weights -------------------------------------------
    # 30 of v2.1's 155 Conv/ConvTranspose/MatMul/Gemm nodes take their weight from a graph VALUE
    # whose producers are Reshape/Transpose/Unsqueeze/Constant/Mul - all constant-foldable, because
    # none of them depend on the audio/text inputs.  Folding them at export time is what makes the
    # runtime dependency-free: the weight becomes a container tensor.  The probe dump is used as
    # GROUND TRUTH for the fold (gate e), never as the source - a dump value is one execution's
    # activation, and only matches a weight when the weight really is input-independent.
    FOLD_OPS = ("Constant", "Reshape", "Transpose", "Unsqueeze", "Squeeze", "Mul", "Add", "Sub",
                "Div", "Cast", "Concat", "Slice", "Gather", "Neg", "Exp", "Sqrt", "Abs", "MatMul",
                "Softmax", "ScatterND", "Tile", "Expand", "Where", "Clip", "Pad", "Relu", "Sigmoid",
                "Tanh", "Softplus", "Erf", "ReduceSum", "Mean", "Gemm", "Shape", "Identity",
                "Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual", "Not", "And", "Or",
                "Xor", "Flatten", "Range", "CumSum", "Max", "Min", "Sum", "Reciprocal", "Pow",
                "Log", "Floor", "Ceil", "Round", "Sign", "ConstantOfShape", "EyeLike")

    def fold(self, name: str, _seen: frozenset = frozenset()) -> np.ndarray:
        """Evaluate a value of the graph as a constant, or raise GateFailure.  Memoized: shared
        sub-DAGs of constants are evaluated once."""
        if name in self._fold_cache:
            return self._fold_cache[name]
        v = self._fold(name, _seen)
        self._fold_cache[name] = v
        return v

    def _fold(self, name: str, _seen: frozenset) -> np.ndarray:
        if name in self.init:
            return np.asarray(self.init[name], np.float32)
        if name in _seen:
            raise GateFailure(f"cycle folding weight {name}")
        node = self.producer.get(name)
        if node is None:
            raise GateFailure(f"weight {name} is a graph INPUT - not constant-foldable")
        a = attrs_of(node)
        op = node.op_type
        if op not in self.FOLD_OPS:
            raise GateFailure(f"weight {name}: unsupported fold op {op} ({node.name})")
        ins = [self.fold(i, _seen | {name}) for i in node.input if i and (i in self.init or i in self.producer)]
        if op == "Constant":
            return np.asarray(numpy_helper.to_array(node.attribute[0].t), np.float32)
        if op == "Reshape":
            shape = [int(v) for v in np.asarray(ins[1]).reshape(-1)]
            return np.reshape(ins[0], shape)
        if op == "Transpose":
            return np.transpose(ins[0], a.get("perm", None))
        if op in ("Unsqueeze", "Squeeze"):
            axes = tuple(int(v) for v in (np.asarray(ins[1]).reshape(-1) if len(ins) > 1
                                          else a.get("axes", [])))
            return (np.expand_dims(ins[0], axes) if op == "Unsqueeze" else np.squeeze(ins[0], axes))
        if op in ("Mul", "Add", "Sub", "Div"):
            f = {"Mul": np.multiply, "Add": np.add, "Sub": np.subtract, "Div": np.divide}[op]
            return f(ins[0], ins[1])
        if op == "Cast":
            return ins[0].astype(onnx.helper.tensor_dtype_to_np_dtype(int(a["to"])))
        if op == "Concat":
            return np.concatenate(ins, int(a["axis"]))
        if op == "Slice":
            st, en = list(np.asarray(ins[1]).astype(int)), list(np.asarray(ins[2]).astype(int))
            ax = list(np.asarray(ins[3]).astype(int)) if len(ins) > 3 else list(range(len(st)))
            sp = list(np.asarray(ins[4]).astype(int)) if len(ins) > 4 else [1] * len(st)
            sl = [slice(0, None, 1)] * ins[0].ndim
            for a_, s_, e_, p_ in zip(ax, st, en, sp):
                sl[a_] = slice(s_, e_, p_)
            return ins[0][tuple(sl)]
        if op == "Gather":
            return np.take(ins[0], np.asarray(ins[1]).astype(int), axis=int(a["axis"]))
        if op == "MatMul":
            return np.matmul(ins[0], ins[1])
        if op == "Gemm":
            A, B = ins[0], ins[1]
            if int(a.get("transB", 0)):
                B = B.T
            if int(a.get("transA", 0)):
                A = A.T
            out = float(a.get("alpha", 1.0)) * (A @ B)
            if len(ins) > 2:
                out = out + float(a.get("beta", 1.0)) * ins[2]
            return out
        if op == "ScatterND":
            out = np.array(ins[0], np.float32, copy=True)
            ix = np.asarray(ins[1]).astype(np.int64)
            up = np.asarray(ins[2], np.float32)
            if ix.ndim == 1:                        # one update, index path of length r
                out[tuple(int(v) for v in ix)] = up
                return out
            n = int(np.prod(ix.shape[:-1]))          # [N..., r] -> N updates, each an r-path
            paths = ix.reshape(n, ix.shape[-1])
            vals = up.reshape(n, *up.shape[ix.ndim - 1:])
            for i in range(n):
                out[tuple(int(v) for v in paths[i])] = vals[i]
            return out
        if op == "Tile":
            return np.tile(ins[0], [int(v) for v in np.asarray(ins[1]).astype(int)])
        if op == "Expand":
            return np.broadcast_to(ins[0], [int(v) for v in np.asarray(ins[1]).astype(int)]).copy()
        if op == "Where":
            return np.where(ins[0] != 0, ins[1], ins[2])
        if op == "Clip":
            lo = float(ins[1].reshape(-1)[0]) if len(ins) > 1 and ins[1].size else None
            hi = float(ins[2].reshape(-1)[0]) if len(ins) > 2 and ins[2].size else None
            return np.clip(ins[0], lo, hi)
        if op == "Pad":
            pads = [int(v) for v in np.asarray(ins[1]).astype(int).reshape(-1)]
            k = len(pads) // 2
            return np.pad(ins[0], [(pads[i], pads[i + k]) for i in range(k)], mode="constant")
        if op in ("Relu", "Sigmoid", "Tanh", "Softplus", "Erf"):
            f = {"Relu": lambda x: np.maximum(x, 0), "Sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
                 "Tanh": np.tanh, "Softplus": lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0),
                 "Erf": lambda x: np.vectorize(math.erf)(x).astype(np.float32)}[op]
            return f(ins[0])
        if op == "ReduceSum":
            ax = tuple(int(v) for v in np.asarray(ins[1]).astype(int)) if len(ins) > 1 \
                 else tuple(int(v) for v in a.get("axes", []))
            return np.sum(ins[0], axis=ax, keepdims=bool(int(a.get("keepdims", 1))))
        if op == "Mean":
            ax = tuple(int(v) for v in a.get("axes", []))
            return np.mean(ins[0], axis=ax, keepdims=bool(int(a.get("keepdims", 1))))
        if op == "Shape":
            return np.array(ins[0].shape, np.float32)
        if op == "ConstantOfShape":
            shape = tuple(int(v) for v in np.asarray(ins[0]).astype(int).reshape(-1))
            v = numpy_helper.to_array(a["value"]) if "value" in a else np.array(0.0, np.float32)
            return np.full(shape, v, np.float32)
        if op == "EyeLike":
            m = int(ins[0].shape[0])
            k = int(a.get("k", 0))
            dt = onnx.helper.tensor_dtype_to_np_dtype(int(a["dtype"])) if "dtype" in a else np.float32
            return np.eye(m, dtype=dt)
        if op == "Identity":
            return ins[0]
        if op in ("Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual"):
            f = {"Equal": np.equal, "Less": np.less, "Greater": np.greater,
                 "LessOrEqual": np.less_equal, "GreaterOrEqual": np.greater_equal}[op]
            return f(ins[0], ins[1]).astype(np.float32)
        if op in ("Not", "And", "Or", "Xor"):
            f = {"Not": np.logical_not, "And": np.logical_and, "Or": np.logical_or,
                 "Xor": np.logical_xor}[op]
            args = [x.astype(bool) for x in ins]
            return (args[0] if op == "Not" else f(args[0], args[1])).astype(np.float32)
        if op in ("Max", "Min", "Sum"):
            f = {"Max": np.maximum, "Min": np.minimum, "Sum": np.add}[op]
            out = ins[0]
            for x in ins[1:]:
                out = f(out, x)
            return out
        if op == "Flatten":
            ax = int(a.get("axis", 1))
            return ins[0].reshape(int(np.prod(ins[0].shape[:ax])), -1)
        if op == "Range":
            return np.arange(float(ins[0]), float(ins[1]), float(ins[2]), dtype=np.float32)
        if op == "CumSum":
            return np.cumsum(ins[0], axis=int(np.asarray(ins[1]).reshape(-1)[0]) if len(ins) > 1 else 0,
                             dtype=np.float32)
        if op in ("Reciprocal", "Pow", "Log", "Floor", "Ceil", "Round", "Sign"):
            f = {"Reciprocal": lambda x: 1.0 / x, "Pow": np.power, "Log": np.log,
                 "Floor": np.floor, "Ceil": np.ceil, "Round": np.round, "Sign": np.sign}[op]
            return f(*ins)
        if op == "Softmax":
            ax = int(a.get("axis", -1))
            m = ins[0].max(axis=ax, keepdims=True)
            e = np.exp(ins[0] - m)
            return e / e.sum(axis=ax, keepdims=True)
        return {"Neg": np.negative, "Exp": np.exp, "Sqrt": np.sqrt, "Abs": np.abs,
                "Softmax": lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()}[op](ins[0])

    def dynamic_weight(self, onnx_name: str) -> np.ndarray:
        """Weight v2.1 computes at run time: fold it from constants, and (when a probe dump exists)
        check the fold against ONNXRuntime's own value for that tensor."""
        w = np.ascontiguousarray(self.fold(onnx_name), np.float32)
        self.folded.append(onnx_name)
        if self.probe_dir:
            f = os.path.join(self.probe_dir, onnx_name.replace("/", "_") + ".npy")
            if os.path.exists(f):
                d = np.ascontiguousarray(np.load(f), np.float32)
                if d.shape != w.shape:
                    raise GateFailure(f"folded weight {onnx_name} shape {w.shape} != probe dump "
                                      f"{d.shape} (the value is NOT input-independent)")
                dev = float(np.max(np.abs(d - w)))
                if dev > 1e-5:
                    raise GateFailure(f"GATE FAILED (e) fold vs ONNXRuntime: {onnx_name} "
                                      f"max|folded-dumped|={dev:.3e}")
                self.gate_e.append(f"  fold {onnx_name:34s} shape={str(w.shape):22s} "
                                   f"max|fold-ORT|={dev:.2e}  OK")
            else:
                self.gate_e.append(f"  fold {onnx_name:34s} shape={str(w.shape):22s} (no dump to compare)")
        return w

    def epilogue(self, node) -> str:
        """Consumer ops immediately after this node's output (what the runtime should fuse)."""
        cs = self.consumers.get(node.output[0], [])
        ops = sorted({c.op_type for c in cs})
        if not ops:
            return "none"
        if ops == ["LeakyRelu"]:
            return f"leaky_relu({attrs_of(cs[0]).get('alpha', 0.01):g})"
        if ops == ["Mul", "Sigmoid"]:
            return "glu(sigmoid)"
        if ops == ["Add"]:
            return "residual_add"
        if ops == ["Softmax"]:
            return "softmax"
        return ",".join(ops)

    def kind_of(self, op: str, node) -> str:
        if op in ("Conv", "ConvTranspose", "MatMul", "Gemm", "LayerNormalization"):
            return "weight"
        if op in REDUCE_OPS:
            return "reduce"
        if op in ACT_OPS:
            return "act"
        if op in MASK_OPS:
            return "mask"
        if op in SHAPE_OPS:
            return "shape"
        return "shape"

    # ---- source-tensor lookup (for the round-trip gate) ---------------------------------
    def weight_source(self, l: dict) -> tuple[str | None, np.ndarray | None]:
        """The ONNX-shaped tensor a packed entry must reproduce, plus its ONNX name."""
        if l["op"] == "constant":
            return l["src"], np.ascontiguousarray(self.const[l["src"]], np.float32)
        if not l.get("onnx"):
            return None, None
        nm = l["onnx"][0]
        if nm in self.init:
            W = np.ascontiguousarray(self.init[nm], np.float32)
        elif nm in self.const:
            W = np.ascontiguousarray(self.const[nm], np.float32)
        else:
            W = np.ascontiguousarray(self.dynamic_weight(nm), np.float32)
        if l["op"] == "matmul" and W.ndim == 2:
            W = W.T
        return nm, W

    def bias_source(self, l: dict) -> np.ndarray:
        """The ONNX bias vector a packed bias entry must reproduce (zeros when the op has none)."""
        op = l["op"]
        n = self.g.node[l["idx"]]
        if op in ("conv1d", "conv_transpose1d"):
            bn = n.input[2] if len(n.input) > 2 and n.input[2] in self.init else None
            b = self.init[bn] if bn else np.zeros(l["params"]["cout"], np.float32)
        elif op == "layernorm":
            b = self.init[n.input[2]]
        else:
            b = np.zeros(l["params"]["cout4"], np.float32)
        b = np.ascontiguousarray(np.asarray(b, np.float32))
        return pad_vec(b, max(len(b), l["params"].get("cout", len(b))))

    # ---- pass over the graph -----------------------------------------------------------
    def run(self) -> None:
        graph_outputs = {o.name for o in self.g.output}
        for idx, n in enumerate(self.g.node):
            a = attrs_of(n)
            grp = group_of(n.name) if n.name else group_of(n.output[0])
            mod = strip_group_prefix(module_of(n.name)) if n.name else strip_group_prefix(module_of(n.output[0]))
            entry: dict = dict(idx=idx, graph=grp, module=mod, onnx_op=n.op_type,
                               op=n.op_type.lower(), name=n.name, inputs=list(n.input),
                               outputs=list(n.output), attrs=a,
                               kind=self.kind_of(n.op_type, n), epilogue="none",
                               packed=[], dtype="fp32", params={}, shape="")
            if n.op_type == "Constant":
                entry["kind"] = "constant"
                self.export_constant(idx, n, entry)
            elif n.op_type == "ConstantOfShape":
                # the fill value is an attribute tensor: record it numerically (the JSON writer
                # stringifies a TensorProto, which the numpy oracle and the C runtime cannot use)
                v = numpy_helper.to_array(a["value"])
                entry["fill"] = dict(value=v.reshape(-1).tolist(), dtype=str(v.dtype),
                                     shape=list(v.shape))
            elif n.op_type == "Conv":
                self.conv_entry(idx, n, grp, mod, a, entry)
            elif n.op_type == "ConvTranspose":
                self.convT_entry(idx, n, grp, mod, a, entry)
            elif n.op_type in ("MatMul", "Gemm"):
                self.matmul_entry(idx, n, grp, mod, a, entry)
            elif n.op_type == "LayerNormalization":
                self.layernorm_entry(idx, n, grp, mod, a, entry)
            elif n.op_type == "Gather" and n.input[0] in self.init:
                self.embedding_entry(idx, n, grp, mod, a, entry)
            elif n.op_type == "Pad" and len(n.input) > 1 and n.input[1] in self.init:
                self.pad_table_entry(idx, n, grp, mod, a, entry)
            elif n.op_type == "Pad" and n.input[0] in self.init:
                self.pad_weight_entry(idx, n, grp, mod, a, entry)
            elif n.op_type == "Pad":
                self.pad_const_entry(idx, n, grp, mod, a, entry)
            self.layers.append(entry)
            self.nodes.append(dict(idx=idx, op=n.op_type, name=n.name, inputs=list(n.input),
                                   outputs=list(n.output), attrs=a,
                                   kind=entry["kind"], packed=entry["packed"]))
        self.model_meta(graph_outputs)

    # ---- per-op exporters --------------------------------------------------------------
    def export_constant(self, idx: int, n, entry: dict | None = None) -> None:
        """Constant node -> container (float) or JSON literal (int/bool), per the docstring."""
        a = numpy_helper.to_array(n.attribute[0].t)
        out = n.output[0]
        grp = group_of(out)
        if entry is not None:
            entry["src"] = out
        self.covered_const.add(out)
        rec: dict = dict(idx=idx, output=out, group=grp, dtype=str(a.dtype), shape=list(a.shape),
                         container=None, values=None)
        if a.dtype == np.float32:
            name = self.add(f"const.{idx:04d}", a, grp, FP32)
            rec["container"] = name
        rec["values"] = a.reshape(-1).tolist() if a.size <= 64 else None
        self.constants.append(rec)

    def conv_entry(self, idx, n, grp, mod, a, entry) -> None:
        wn = n.input[1]
        W = self.init.get(wn)
        wsrc = wn
        if W is None:
            W = self.dynamic_weight(wn)
        W = np.ascontiguousarray(W, np.float32)
        assert W.ndim == 3, (wn, W.shape)
        Cout, Cin, K = W.shape
        grpn = int(a.get("group", 1))
        stride, dil = int(a["strides"][0]), int(a["dilations"][0])
        pad = int(a["pads"][0])
        assert grpn == 1, f"v2.1 export handles group==1 convs only, {n.name} has group={grpn}"
        is_head = mod.startswith("subband") or mod.startswith("conv_post") or n.name == "/dec/Conv"
        h16 = (grp == "dec" and not is_head)
        split = (self.hilo == grp)
        item = FP16 if (h16 and not split) else FP32
        suffix = H16 if (h16 and not split) else ""
        b = self.init.get(n.input[2]) if len(n.input) > 2 and n.input[2] in self.init else np.zeros(Cout, np.float32)
        if split:
            hi, lo = split_hilo(W)
            wp = self.add(f"{grp}.{mod}.whi{H16}", pack_conv(hi), grp, FP16)
            wl = self.add(f"{grp}.{mod}.wlo{H16}", pack_conv(lo), grp, FP16)
            packed = [wp, wl]
            dtype = "fp16_hilo"
        else:
            wp = self.add(f"{grp}.{mod}.wp{suffix}", pack_conv(W), grp, item)
            packed = [wp]
            dtype = "fp16" if h16 else "fp32"
        bp = self.add(f"{grp}.{mod}.b", pad_vec(np.asarray(b, np.float32), r4(Cout)), grp, FP32)
        packed.append(bp)
        entry.update(op="conv1d", onnx=[wn] + ([n.input[2]] if len(n.input) > 2 else []),
                     packed=packed, dtype=dtype,
                     packed_shape=list(self.tensors[-2][1].shape),
                     params=dict(kernel=K, stride=stride, pad=pad, dilation=dil, group=1,
                                 cin=Cin, cout=Cout, cout4=r4(Cout)),
                     W_shape=list(W.shape), B_shape=list(np.asarray(b).shape),
                     shape=f"[{Cin}][T] -> [{Cout}][(T + 2*{pad} - {dil}*({K}-1) - 1)/{stride} + 1]",
                     epilogue=self.epilogue(n))
        src = wsrc if wn in self.init else None
        self.conv.append(dict(name=wp, W=W, B=np.asarray(b, np.float32), stride=stride, pad=pad,
                              dilation=dil, Cin=Cin, Cout=Cout, onnx=wn,
                              static=(wn in self.init)))
        if wn in self.init:
            self.covered.add(wn)
        if len(n.input) > 2 and n.input[2] in self.init:
            self.covered.add(n.input[2])

    def convT_entry(self, idx, n, grp, mod, a, entry) -> None:
        wn = n.input[1]
        static = wn in self.init
        W = self.init.get(wn)
        if W is None:
            W = self.dynamic_weight(wn)
        W = np.ascontiguousarray(W, np.float32)
        assert W.ndim == 3, (wn, W.shape)
        Cin, Cout, K = W.shape
        s = int(a["strides"][0])
        pad = int(a["pads"][0])
        assert int(a.get("group", 1)) == 1 and int(a["dilations"][0]) == 1, n.name
        b = self.init.get(n.input[2]) if len(n.input) > 2 and n.input[2] in self.init else np.zeros(Cout, np.float32)
        h16 = grp == "dec" and not (mod.startswith("ConvTranspose") and Cout <= 16)
        item = FP16 if h16 else FP32
        suffix = H16 if h16 else ""
        wp = self.add(f"{grp}.{mod}.wp{suffix}", pack_convtranspose(W, s, pad), grp, item)
        bp = self.add(f"{grp}.{mod}.b", pad_vec(np.asarray(b, np.float32), r4(Cout)), grp, FP32)
        nt = convT_ntap(K, s)
        entry.update(op="conv_transpose1d", onnx=[wn] + ([n.input[2]] if len(n.input) > 2 else []),
                     packed=[wp, bp], dtype="fp16" if h16 else "fp32",
                     packed_shape=list(self.tensors[-2][1].shape),
                     params=dict(kernel=K, stride=s, pad=pad, dilation=1, group=1, cin=Cin,
                                 cout=Cout, cout4=r4(Cout), phases=s, taps_per_phase=nt,
                                 phase_shift=-pad),
                     W_shape=list(W.shape), B_shape=list(np.asarray(b).shape),
                     impl=f"{s} phases x {nt} taps; out[q*{s} + r - {pad}] = P_r[q] "
                          f"(tap j = ONNX kernel tap r + j*{s}); bias added once per channel",
                     shape=f"[{Cin}][T] -> [{Cout}][(T-1)*{s} - {2*pad} + {K}]",
                     epilogue=self.epilogue(n))
        self.convT.append(dict(name=wp, W=W, B=np.asarray(b, np.float32), s=s, pad=pad, K=K,
                               Cin=Cin, Cout=Cout, static=static))
        if wn in self.init:
            self.covered.add(wn)
        if len(n.input) > 2 and n.input[2] in self.init:
            self.covered.add(n.input[2])

    def matmul_entry(self, idx, n, grp, mod, a, entry) -> None:
        static = [i for i in n.input if i in self.init]
        params: dict = {}
        packed: list[str] = []
        if static:
            M = np.ascontiguousarray(self.init[static[0]], np.float32)
            if n.op_type == "Gemm" and int(a.get("transB", 0)) == 1:
                W = M
            else:
                W = M.T if M.ndim == 2 else M
            wp = self.add(f"{grp}.{mod}.wp", pack_conv(W[:, :, None]), grp, FP32)
            bp = self.add(f"{grp}.{mod}.b", pad_vec(np.zeros(W.shape[0], np.float32), r4(W.shape[0])), grp, FP32)
            packed = [wp, bp]
            params = dict(kernel=1, cin=int(W.shape[1]), cout=int(W.shape[0]), cout4=r4(int(W.shape[0])))
            self.covered.add(static[0])
        entry.update(op="matmul", onnx=static, packed=packed, params=params,
                     impl="y = A @ B (both dynamic in v2.1: attention scores/values, iSTFT basis)",
                     shape="[M,K]x[K,N] -> [M,N]", epilogue=self.epilogue(n))
        for i in static:
            self.covered.add(i)

    def layernorm_entry(self, idx, n, grp, mod, a, entry) -> None:
        g = np.ascontiguousarray(self.init[n.input[1]], np.float32)
        be = np.ascontiguousarray(self.init[n.input[2]], np.float32)
        gp = self.add(f"{grp}.{mod}.g", g, grp, FP32)
        bp = self.add(f"{grp}.{mod}.beta", be, grp, FP32)
        entry.update(op="layernorm", onnx=[n.input[1], n.input[2]], packed=[gp, bp], dtype="fp32",
                     params=dict(C=int(g.shape[0]), axis=int(a.get("axis", -1)),
                                 eps=float(a["epsilon"])),
                     formula="y = (x-mean)/sqrt(var+eps)*g + beta over the axis, fp32 accumulator",
                     shape="per row over channels", epilogue=self.epilogue(n))
        self.covered.add(n.input[1])
        self.covered.add(n.input[2])

    def embedding_entry(self, idx, n, grp, mod, a, entry) -> None:
        E = np.ascontiguousarray(self.init[n.input[0]], np.float32)
        name = self.add(f"{grp}.{mod}.emb", E, grp, FP32)
        entry.update(op="embedding", onnx=[n.input[0]], packed=[name], dtype="fp32",
                     shape_table=list(E.shape), params=dict(rows=int(E.shape[0]), cols=int(E.shape[1])),
                     shape="row gather on axis %d" % int(a.get("axis", 0)))
        self.covered.add(n.input[0])

    def pad_table_entry(self, idx, n, grp, mod, a, entry) -> None:
        """Attention relative-position embedding tables arrive as a Pad(input, pads, value) pair."""
        E = np.ascontiguousarray(self.init[n.input[1]], np.float32)
        name = self.add(f"{grp}.{mod}.padtab", E, grp, FP32)
        entry.update(op="pad_relative_position", onnx=[n.input[1]], packed=[name], dtype="fp32",
                     params=dict(pads=[int(v) for v in self.const_pads(n)]),
                     impl="append the learned relative-position rows (the ONNX Pad's value tensor)",
                     shape=str(list(E.shape)))
        self.covered.add(n.input[1])

    def pad_weight_entry(self, idx, n, grp, mod, a, entry) -> None:
        """Weight Pad: the DATA input is an initializer and the pads are graph-computed (every one
        of v2.1's 51 Pad nodes has dynamic pads).  The table goes in the container; the runtime
        pads it to the query length with the pads the graph computes from the sequence length.

        Measured (probe): attn emb_rel_k/v (1,9,96) + pads [0,76,0,0,76,0] -> (1,161,96);
        the 76 = (len-1) - (rel-1)/2 grows with the utterance, so the pads are runtime data.
        """
        T = np.ascontiguousarray(self.init[n.input[0]], np.float32)
        tail = (n.name or n.output[0]).split("/")[-1].lower()  # 'pad' vs 'pad_5' (k vs v table)
        name = self.add(f"{grp}.{mod}.{tail}.relpos", T, grp, FP32)
        pads = self.const_pads(n)
        entry.update(op="pad_relative_position", onnx=[n.input[0]], packed=[name], dtype="fp32",
                     params=dict(table=list(T.shape), rows=int(T.shape[1]), cols=int(T.shape[2]),
                                 pads=None if pads is None else [int(v) for v in pads],
                                 pads_source=(n.input[1] if len(n.input) > 1 else None),
                                 mode=str(a.get("mode", "constant"))),
                     impl="constant Pad of the learned relative-position table to the attention "
                          "width; pads are computed by the graph from the sequence length "
                          "(pads_source), the container holds the unpadded table",
                     shape=f"{list(T.shape)} -> padded on axis {1} at runtime",
                     epilogue=self.epilogue(n))
        self.covered.add(n.input[0])

    def pad_const_entry(self, idx, n, grp, mod, a, entry) -> None:
        pads = self.const_pads(n)
        entry.update(op="pad", params=dict(pads=[int(v) for v in pads] if pads is not None else None,
                                           mode=a.get("mode", "constant")),
                     impl="constant pad (edge padding for the decoder / waveform lowpass tail)")

    def const_pads(self, n):
        if len(n.input) < 2:
            return None
        p = n.input[1]
        if p in self.const:
            return self.const[p].reshape(-1)
        if p in self.init:
            return self.init[p].reshape(-1)
        return None

    def model_meta(self, graph_outputs) -> None:
        cfg = {}
        cfgpath = os.path.join(os.path.dirname(os.path.dirname(self.model_path)),
                               "configs", "zhtw_mb_istft_16k_v21b.json")
        if os.path.exists(cfgpath):
            cfg = json.load(open(cfgpath))
        d, mm = cfg.get("data", {}), cfg.get("model", {})
        self.meta = dict(
            sample_rate=d.get("sampling_rate", 16000),
            n_mels=d.get("n_mel_channels", 80),
            hop_length_mel=d.get("hop_length", 256),
            subbands=mm.get("subbands", 4),
            gen_istft_n_fft=mm.get("gen_istft_n_fft", 16),
            gen_istft_hop_size=mm.get("gen_istft_hop_size", 4),
            n_speakers=mm.get("n_speakers", 3),
            num_tones=mm.get("num_tones", 6),
            num_langs=mm.get("num_langs", 2),
            inter_channels=mm.get("inter_channels", 192),
            hidden_channels=mm.get("hidden_channels", 192),
            filter_channels=mm.get("filter_channels", 768),
            n_heads=mm.get("n_heads", 2),
            n_layers=mm.get("n_layers", 6),
            min_phone_len=d.get("min_phone_len"),
            max_phone_len=d.get("max_phone_len"),
            max_wav_sec=d.get("max_wav_sec"),
            add_blank=d.get("add_blank"),
        )
        vals = [self.meta["sample_rate"], self.meta["n_mels"], self.meta["subbands"],
                self.meta["gen_istft_n_fft"], self.meta["gen_istft_hop_size"], self.meta["n_speakers"],
                self.meta["num_tones"], self.meta["num_langs"], self.meta["inter_channels"],
                self.meta["filter_channels"], self.meta["n_heads"], self.meta["n_layers"]]
        self.add("model.meta", np.array(vals, np.float32), "meta", FP32)
        # decoder geometry as data: per ConvTranspose stage [Cin, Cout, stride, pad, kernel]
        stages = [l for l in self.layers if l["op"] == "conv_transpose1d" and l["graph"] == "dec"
                  and l["params"].get("cout", 0) > 16]
        stages.sort(key=lambda l: l["idx"])
        geom = [[l["params"]["cin"], l["params"]["cout"], l["params"]["stride"], l["params"]["pad"],
                 l["params"]["kernel"]] for l in stages]
        flat = [v for row in geom for v in row] + [0.0] * (24 - len(geom) * 5)
        self.add("dec.geom", np.array(flat, np.float32), "meta", FP32)
        self.meta["dec_stages"] = geom
        self.meta["samples_per_frame"] = (self.meta["gen_istft_hop_size"] *
                                          np.prod([g[2] for g in geom], dtype=np.float64) *
                                          self.meta["gen_istft_hop_size"])
        # per-record payload width, in record order (self-describing container: a reader that
        # parses the record table can get the width from data instead of the name suffix)
        widths = [float(isz) for _n, _a, isz in self.tensors] + [float(FP32)]
        self.add("dtype.table", np.array(widths, np.float32), "meta", FP32)


# --------------------------------------------------------------------------- gates

def gate_round_trip(ex: "Exporter", path: str, tol: float = 1e-6) -> list[dict]:
    """(a) Rebuild every weight from the packed container tensors and compare with ONNX.

    fp16 tensors are compared against the fp16-rounded ONNX tensor: that is the closest value
    fp16 storage can represent, so a clean export must match it EXACTLY (measured diff 0).
    """
    back = read_ptk(path)
    rows = []
    for l in ex.layers:
        if not l["packed"]:
            continue
        refname, ref = ex.weight_source(l)
        if ref is None:
            continue
        rebuilt = rebuild(l, back)
        isz = 2 if l["packed"][0].endswith(H16) else FP32
        # fp16 storage can only represent fp16-rounded values, so the round-trip target is
        # round_fp16(ONNX tensor); fp32/HiLo targets the ONNX tensor itself.
        want = as_fp16_values(ref) if (isz == 2 and l["dtype"] == "fp16") else ref
        dW = float(np.max(np.abs(rebuilt - want)))
        dB = 0.0
        # biases are always stored fp32; compare them with the ONNX bias directly
        if l["op"] in ("conv1d", "conv_transpose1d", "layernorm", "matmul") and len(l["packed"]) > 1:
            n = int(l["params"].get("cout", l["params"].get("C", rebuilt.shape[-1])))
            got_B = np.asarray(back[l["packed"][1]][0].reshape(-1)[:n], np.float32)
            want_B = np.asarray(ex.bias_source(l), np.float32)[:n]
            dB = float(np.max(np.abs(got_B - want_B)))
        if dW > tol or dB > tol:
            raise GateFailure(f"gate(a) round-trip: {refname} via {l['packed'][0]} "
                              f"max|W-rebuilt|={dW:.3e} max|bias-b|={dB:.3e} (tol {tol:g}) "
                              f"[op={l['op']} dtype={l['dtype']} shape={list(rebuilt.shape)}]")
        rows.append(dict(op=l["op"], src=refname, packed=l["packed"][0], itemsize=isz,
                         elems=int(rebuilt.size), max_dW=dW, max_dB=dB))
    return rows


def rebuild(l: dict, back: dict) -> np.ndarray:
    """Rebuild the ONNX-shaped weight from packed container tensors (HiLo entries sum)."""
    p = l["params"]
    if l["op"] == "conv1d":
        W = unpack_conv(back[l["packed"][0]][0])[: p["cout"]]
        if l["dtype"] == "fp16_hilo" and len(l["packed"]) > 1:
            W = W + unpack_conv(back[l["packed"][1]][0])[: p["cout"]]
        return W
    if l["op"] == "conv_transpose1d":
        return unpack_convtranspose(back[l["packed"][0]][0], p["stride"], p["cout"], p["cin"], p["kernel"])
    if l["op"] == "matmul":
        return unpack_conv(back[l["packed"][0]][0])[:, :, 0]
    return back[l["packed"][0]][0]


def conv1d_brute(X: np.ndarray, W: np.ndarray, B: np.ndarray, stride: int, pad: int, dilation: int) -> np.ndarray:
    """ONNX conv1d, independent of the packed layout and of conv1d_ref's patch-indexing.

    One matmul per kernel tap with its own explicit output-index math: tap k of a stride-s,
    dilation-d conv reads xp[:, t*s + k*d] (xp is the zero-padded input).  The gate compares the
    packed evaluation against THIS, so a stride/pad/dilation mapping error cannot cancel out.
    """
    Cout, Cin, K = W.shape
    T = X.shape[1]
    xp = np.zeros((Cin, T + 2 * pad), np.float32)
    xp[:, pad:pad + T] = X
    Tout = (T + 2 * pad - dilation * (K - 1) - 1) // stride + 1
    out = np.zeros((Cout, Tout), np.float32)
    for k in range(K):
        cols = np.arange(Tout) * stride + k * dilation
        out += W[:, :, k] @ xp[:, cols]
    return out + np.asarray(B, np.float32)[:Cout][:, None]


def conv_transpose1d_brute(X: np.ndarray, W: np.ndarray, B: np.ndarray, stride: int, pad: int) -> np.ndarray:
    """ONNX ConvTranspose, independent of the phase packing: scatter each (input q, tap k) product
    to output index t = q*stride + k - pad.  Bias is per output channel, broadcast over time."""
    Cin, Cout, K = W.shape
    Tin = X.shape[1]
    Tout = (Tin - 1) * stride - 2 * pad + K
    out = np.zeros((Cout, Tout), np.float32)
    for q in range(Tin):
        for k in range(K):
            t = q * stride + k - pad
            if 0 <= t < Tout:
                out[:, t] += W[:, :, k].T @ X[:, q]
    return out + np.asarray(B, np.float32)[:Cout][:, None]


def _gate_layout(kind: str, name: str, onnx: str, W: np.ndarray, Cin: int, Cout: int,
                 K: int, packed: np.ndarray, s: int = 1, nt: int | None = None) -> None:
    """Hard layout assertion for one gate entry.  Never skips: on a mismatch it names the entry,
    the ex.conv/ex.convT list it came from, its ONNX input tensor and every shape involved, so a
    convT-layout tensor reaching the conv loop is visible in the message, not silently passed."""
    lst = f"ex.{kind}"
    if W.ndim != 3:
        raise GateFailure(f"gate(a) {kind} layout: entry {name} (from {lst}, ONNX input {onnx}) has "
                          f"W.ndim={W.ndim} W.shape={tuple(W.shape)}; the packing gate needs the raw "
                          f"3-D ONNX weight (a packed 4-D [Cout4/4][K][Cin][4] tensor belongs in the "
                          f"container, not in {lst})")
    rows, cols, kk = W.shape
    if kind == "conv":
        want = (Cout, Cin, K)
        hint = ""
        if rows == Cin and cols == Cout and Cout != Cin:
            hint = (" -> looks like a ConvTranspose-layout tensor [Cin,Cout,K]: it was appended by "
                    "convT_entry (self.convT.append), not by conv_entry (self.conv.append)")
    else:
        want = (Cin, Cout, K)
        hint = (" -> looks like a Conv-layout tensor [Cout,Cin,K]: it was appended by conv_entry "
                "(self.conv.append), not by convT_entry (self.convT.append)"
                ) if rows == Cout and cols == Cin and Cin != Cout else ""
    if tuple(W.shape) != want:
        raise GateFailure(f"gate(a) {kind} layout: entry {name} (from {lst}, ONNX input {onnx}) has "
                          f"W.shape={tuple(W.shape)}, expected {want} (Cout={Cout} Cin={Cin} K={K}) "
                          f"for this {kind} entry{hint}")
    if kk != K:
        raise GateFailure(f"gate(a) {kind} layout: entry {name} (from {lst}, ONNX input {onnx}) has "
                          f"kernel axis {kk}, entry says K={K} (W.shape={tuple(W.shape)})")
    ph, Kp, Cinp, four = packed.shape
    taps = nt if nt is not None else K
    exp = (r4(Cout) // 4 * max(s, 1), taps, Cin, 4)
    if tuple(packed.shape) != exp:
        raise GateFailure(f"gate(a) {kind} packed shape: entry {name} (from {lst}, ONNX input "
                          f"{onnx}) packed {tuple(packed.shape)}, expected {exp} "
                          f"[r4(Cout)/4 * phases {s}][taps {taps}][Cin {Cin}][4]")
    if four != 4:
        raise GateFailure(f"gate(a) {kind} packed shape: entry {name} last axis {four} != 4")


def gate_packing(ex: "Exporter", tol: float = 1e-5) -> list[str]:
    """Prove the packed layouts reproduce the ONNX conv / conv-transpose OUTPUTS (not just the
    weights) - this is what makes the runtime's phase interleave trustworthy.

    Strict: every entry is layout-checked first (_gate_layout), and a packed output that disagrees
    with the ONNX reference by more than tol raises GateFailure (main exits non-zero).
    """
    out = []
    rng = np.random.default_rng(7)
    for c in ex.conv:
        W = np.asarray(c["W"], np.float32)
        _gate_layout("conv", c["name"], c["onnx"], W, c["Cin"], c["Cout"], int(W.shape[2]) if W.ndim == 3 else -1,
                     np.asarray(pack_conv(W) if W.ndim == 3 else W, np.float32))
        K = int(W.shape[2])
        Wp = np.asarray(pack_conv(W), np.float32)
        # T chosen so Tout >= 8 for any pad/dilation, and > the convT tap delay (gate sees a real signal)
        T = c["dilation"] * (K - 1) + 2 * c["pad"] + 8
        # A Conv whose ONNX bias input is a longer tensor than its Cout is a FUSED ADD bias (v2.1 has
        # 32 such convs); the export path for those is the Add epilogue, not the conv bias, so the
        # packing gate compares against a zero bias and reports the shape mismatch in the line.
        nb = int(np.asarray(c["B"]).reshape(-1).shape[0])
        bias = c["B"] if nb == c["Cout"] else np.zeros(c["Cout"], np.float32)
        note = "" if nb == c["Cout"] else f" [bias input is {nb} long -> fused-Add, gate uses 0 bias]"
        X = rng.standard_normal((c["Cin"], T)).astype(np.float32)
        want = conv1d_brute(X, W, bias, c["stride"], c["pad"], c["dilation"])
        got = conv1d_ref(X, unpack_conv(Wp)[: c["Cout"]], pad_vec(bias, r4(c["Cout"]))[: c["Cout"]],
                         c["stride"], c["pad"], c["dilation"])
        d = float(np.max(np.abs(want - got)))
        scale = max(1.0, float(np.max(np.abs(want))))
        rel = d / scale
        if d > tol * scale:                     # fp32 summation-order noise is scale-relative
            raise GateFailure(f"gate(a) conv packing: {c['name']} (from ex.conv, ONNX input "
                              f"{c['onnx']}, W={tuple(W.shape)} Cin={c['Cin']} Cout={c['Cout']} "
                              f"K={K} s={c['stride']} pad={c['pad']} dil={c['dilation']} "
                              f"packed={tuple(Wp.shape)}) max|packed-ONNX_brute|={d:.3e} rel={rel:.1e} "
                              f"(tol {tol:g} x out-scale {scale:.3g})")
        out.append(f"  conv  {c['name']:44s} Cin={c['Cin']:4d} Cout={c['Cout']:4d} K={K:2d} "
                   f"s={c['stride']} pad={c['pad']:2d} dil={c['dilation']} T={T:3d} "
                   f"out={want.shape[1]:4d} max|packed-ONNX_brute|={d:.2e}{note}  OK")
    for e in ex.convT:
        K, s, pad = int(e["K"]), int(e["s"]), int(e["pad"])
        nt = convT_ntap(K, s)
        X = rng.standard_normal((e["Cin"], K + 2 * pad + 4)).astype(np.float32)
        Wp = np.asarray(pack_convtranspose(e["W"], s, pad), np.float32)
        _gate_layout("convT", e["name"], e.get("onnx", ""), e["W"], e["Cin"], e["Cout"], K, Wp,
                     s=s, nt=nt)
        Bp = pad_vec(e["B"], r4(e["Cout"]))
        want = conv_transpose1d_brute(X, e["W"], e["B"], s, pad)
        got = conv_transpose1d_packed(X, Wp, Bp, s, pad, e["Cout"], K)
        d = float(np.max(np.abs(want - got))) if want.shape == got.shape else float("inf")
        scale = max(1.0, float(np.max(np.abs(want))))
        if d > tol * scale:
            raise GateFailure(f"gate(a) convT packing: {e['name']} (from ex.convT, "
                              f"W={tuple(e['W'].shape)} Cin={e['Cin']} Cout={e['Cout']} K={K} s={s} "
                              f"pad={pad} taps={nt} packed={tuple(Wp.shape)}) "
                              f"want={want.shape} packed={got.shape} max|packed-ONNX_brute|={d:.3e} "
                              f"rel={d/scale:.1e} (tol {tol:g} x out-scale {scale:.3g})")
        out.append(f"  convT {e['name']:44s} Cin={e['Cin']:4d} Cout={e['Cout']:4d} K={K:2d} s={s:2d} "
                   f"pad={pad:2d} phases={s:2d} taps/phase={nt} T={X.shape[1]:3d} "
                   f"out={got.shape[1]:4d} max|packed-ONNX_brute|={d:.2e}  OK")
    return out


def gate_coverage(ex: "Exporter") -> dict:
    """(b) every weight tensor (initializer or Constant feeding a compute node) has an entry."""
    missing_init = sorted(set(ex.init) - ex.covered)
    if missing_init:
        raise GateFailure(f"gate(b) coverage: {len(missing_init)} initializers with no packed entry, "
                          f"first: {missing_init[0]}")
    compute_ops = ("Conv", "ConvTranspose", "MatMul", "Gemm", "LayerNormalization", "Gather")
    miss_const = [n.name for n in ex.g.node
                  if n.op_type in compute_ops and any(i in ex.const for i in n.input)
                  and not all(i in ex.covered_const for i in n.input if i in ex.const)]
    if miss_const:
        raise GateFailure(f"gate(b) coverage: Constant-fed compute node with no exported weight: "
                          f"{miss_const[0]}")
    static_consumers = sum(1 for n in ex.g.node if n.op_type != "Constant"
                           and any(i in ex.init or i in ex.const for i in n.input))
    return dict(initializers=len(ex.init), covered=len(ex.covered),
                constant_tensors=len(ex.const), constant_exported=len(ex.covered_const),
                static_tensor_total=len(ex.init) + len(ex.const),
                nodes_consuming_static_tensors=static_consumers,
                weight_entries=sum(1 for l in ex.layers if l["packed"]))


def predict_budget(ex: "Exporter") -> dict:
    """Predicted container bytes from the dtype policy + shapes, computed independently of the
    writer (so gate (c) compares container vs policy, not writer vs writer)."""
    pred = collections.Counter()
    for l in ex.layers:
        p, op = l.get("params", {}), l["op"]
        if op == "conv1d":
            isz = 2 if l["dtype"] == "fp16" else 4
            nwp = 2 if l["dtype"] == "fp16_hilo" else 1
            pred[l["graph"]] += nwp * align64(p["cout4"] * p["cin"] * p["kernel"] * isz)
            pred[l["graph"]] += align64(p["cout4"] * 4)
        elif op == "conv_transpose1d":
            isz = 2 if l["dtype"] == "fp16" else 4
            pred[l["graph"]] += align64(p["phases"] * p["cout4"] * p["cin"] * p["taps_per_phase"] * isz)
            pred[l["graph"]] += align64(p["cout4"] * 4)
        elif op == "matmul":
            if l["packed"]:
                pred[l["graph"]] += align64(p["cout4"] * p["cin"] * 4) + align64(p["cout4"] * 4)
        elif op == "layernorm":
            pred[l["graph"]] += align64(p["C"] * 4) * 2
        elif op == "embedding":
            pred[l["graph"]] += align64(p["rows"] * p["cols"] * 4)
        elif op == "pad_relative_position":
            pred[l["graph"]] += align64(int(ex.init[l["onnx"][0]].size) * 4)
        elif op == "constant":
            a = ex.const[l["src"]]
            pred[l["graph"]] += align64(a.size * 4)
    return dict(pred)


# --------------------------------------------------------------------------- main

def budget_table(ex: "Exporter", predicted: dict, container_bytes: int, container_path: str) -> str:
    rows = []
    groups = sorted(set(list(ex.bytes_by_group) + list(predicted)))
    tot_pred = tot_act = 0
    for g in groups:
        act = ex.bytes_by_group[g]
        pred = predicted.get(g, 0)
        tot_pred += pred
        tot_act += act
        fp16 = sum(align64(a.size * 2) for nm, a, isz in ex.tensors if isz == 2 and group_of_name(nm) == g)
        fp32 = sum(align64(a.size * 4) for nm, a, isz in ex.tensors if isz == 4 and group_of_name(nm) == g)
        rows.append((g, ex.params_by_group[g], fp32, fp16, pred, act))
    lines = [f"{'group':7s} {'params':>11s} {'fp32 B':>11s} {'fp16 B':>11s} {'predicted':>11s} {'container':>11s}"]
    for g, p, f32, f16, pred, act in rows:
        lines.append(f"{g:7s} {p:11,d} {f32:11,d} {f16:11,d} {pred:11,d} {act:11,d}")
    lines.append(f"{'TOTAL':7s} {sum(r[1] for r in rows):11,d} "
                 f"{sum(r[2] for r in rows):11,d} {sum(r[3] for r in rows):11,d} "
                 f"{tot_pred:11,d} {tot_act:11,d}")
    dev = (tot_act - tot_pred) / tot_pred * 100.0
    lines.append(f"container payload {container_bytes:,d} B; predicted dtype budget {tot_pred:,d} B; "
                 f"deviation {dev:+.2f}% (gate 2%)")
    if abs(dev) > 2.0:
        raise GateFailure(f"gate(c) budget: container {container_bytes:,d} vs predicted {tot_pred:,d} "
                          f"({dev:+.2f}%)")
    return "\n".join(lines)


STRIP_FIRST = {"enc_p", "dp", "flow", "dec", "emb_g"}


def strip_group_prefix(module: str) -> str:
    """Node names already carry the group as their first segment; drop it so packed names read
    'dec.ups.0.wp', not 'dec.dec.ups.0.wp'."""
    s = module.split(".")
    return ".".join(s[1:]) if len(s) > 1 and s[0] in STRIP_FIRST else module


def group_of_name(name: str) -> str:
    return name.split(".")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/v21_mbistft_16k/primetts_v21_3voice.onnx")
    ap.add_argument("--out", default="build/v21.ptk")
    ap.add_argument("--layers-json", default="spec/v21_layers.json")
    ap.add_argument("--probe-dir", default="tests/golden_v21/_probe")
    ap.add_argument("--hilo", default="none", choices=["none", "enc", "dec"],
                    help="split the chosen group's conv weights into fp16 hi+lo pairs (8-stream tier)")
    args = ap.parse_args()

    ex = Exporter(args.model, args.probe_dir, args.hilo)
    ex.run()

    print(f"model {args.model}: {len(ex.g.node)} nodes, {len(ex.init)} initializers "
          f"({sum(int(a.size) for a in ex.init.values()):,d} params), {len(ex.const)} Constant nodes")

    # gate (e): graph-computed weights folded to constants, each checked against ONNXRuntime's own
    # value for that tensor from the probe dump (the fold must be input-independent to match).
    print(f"gate (e) constant folding: {len(ex.folded)} graph-computed weight(s) folded")
    print("\n".join(ex.gate_e))

    # gate (a) part 1: packed layouts reproduce ONNX conv / conv-transpose outputs
    pack_lines = gate_packing(ex)
    print(f"packing gate: {len(ex.conv)} conv1d + {len(ex.convT)} conv_transpose1d layouts checked "
          f"against an ONNX reference (random [Cin][8] input):")
    print("\n".join(pack_lines))

    n_t, data_bytes = write_ptk(args.out, ex.tensors)
    ex.container_bytes = data_bytes

    # gate (a) part 2: rebuild every weight from the written container and compare with ONNX
    rows = gate_round_trip(ex, args.out)
    dws = [r["max_dW"] for r in rows]
    dbs = [r["max_dB"] for r in rows]
    print(f"round-trip gate: {len(rows)} weight entries rebuilt from {args.out}; "
          f"max|W-rebuilt|={max(dws):.3e}  max|bias-b|={max(dbs):.3e}  (tol 1e-6)  OK")
    per_op = collections.Counter(r["op"] for r in rows)
    print("  entries per op:", dict(per_op))

    cov = gate_coverage(ex)
    print("coverage gate:", json.dumps(cov))

    pred = predict_budget(ex)
    print("dtype budget table (bytes per group):")
    print(budget_table(ex, pred, data_bytes, args.out))

    # spec json
    op_counts = collections.Counter(l["onnx_op"] for l in ex.layers)
    dt_counts = collections.Counter(l["dtype"] for l in ex.layers if l["packed"])
    doc = {
        "model": os.path.relpath(args.model),
        "container": os.path.relpath(args.out),
        "container_format": {
            "magic": "0x50544B31", "version": VERSION, "record": "<64si4x6qqq",
            "alignment": 64,
            "payload_width": "fp32 (4 B) default; tensors named '#h'-suffixed carry fp16 (2 B); "
                             "'dtype.table' (fp32, one itemsize per record, record order)",
            "note": "the v1 reader (runtime/py/pt.py, runtime/src/weights.c) assumes 4 B payloads; "
                    "the runtime lane must honour the per-record width (dtype.table) for v2.1",
        },
        "meta": ex.meta,
        "graph_inputs": [dict(name=i.name, elem_type=i.type.tensor_type.elem_type,
                              shape=[d.dim_value if d.HasField("dim_value") else d.dim_param
                                     for d in i.type.tensor_type.shape.dim])
                         for i in ex.g.input],
        "graph_outputs": [dict(name=o.name, elem_type=o.type.tensor_type.elem_type,
                               shape=[d.dim_value if d.HasField("dim_value") else d.dim_param
                                      for d in o.type.tensor_type.shape.dim])
                          for o in ex.g.output],
        "counts": dict(nodes=len(ex.g.node), initializers=len(ex.init),
                       initializer_params=int(sum(int(a.size) for a in ex.init.values())),
                       constant_nodes=len(ex.const), constant_params=int(sum(a.size for a in ex.const.values())),
                       tensors=n_t, payload_bytes=data_bytes, layers=len(ex.layers),
                       nodes_listed=len(ex.nodes), weight_entries=len(rows)),
        "op_counts": dict(op_counts.most_common()),
        "dtype_counts": dict(dt_counts),
        "bytes_by_group": dict(ex.bytes_by_group),
        "params_by_group": dict(ex.params_by_group),
        "predicted_bytes_by_group": dict(pred),
        "notes": [
            "dtype policy (measured, docs/v21_plan.md): fp32 encoder/flow/dp conv weights, fp16 decoder "
            "conv weights (measured 67.4 dB for that set), fp16 activations are free (87.9 dB); "
            "embeddings, LayerNorm, Softmax, flow scales and the iSTFT head stay fp32.",
            "fp16 payload tensors are the decoder conv_pre/ups./resblocks. weights; their values are "
            "exactly round_fp16(ONNX weight) - gate (a) checks against that reference.",
            "v2.1 activations are [N,C,T] for convs and [N,T,C] around the encoder attention/LayerNorm; "
            "the Transpose entries in `layers` are the boundaries the runtime folds into its own layout.",
            "26 MatMul nodes have no weights (attention scores/values, iSTFT cos-sin basis): the runtime "
            "must run them as real GEMMs, they are not K=1 convs like v1's linears.",
            "The iSTFT head is: subband_conv_post -> [4 subbands, 16 bins] magnitude/phase -> Exp/Sin/Cos "
            "-> Einsum bkt,kn->bnt with a 16x16 constant basis -> three stride-4 ConvTransposes "
            "(two of them constant delta kernels = zero-insert upsamplers, one with a ScatterND-built "
            "4x4x4 kernel) -> Pad -> the 63-tap [1,4,63] lowpass to `wav`.",
            "/dec/ConvTranspose_2's kernel is computed by the graph (ScatterND of 1/4 into a 4x4x4 "
            "zero tensor, x4); it is exported from a probe dump, see dynamic_weight().",
            "zh-TW front-end coverage: known gap, not measured (g2pw is not ported); this export covers "
            "the phoneme-id graph only.",
        ],
        "constants": ex.constants,
        "layers": ex.layers,
        "nodes": ex.nodes,
    }
    os.makedirs(os.path.dirname(args.layers_json) or ".", exist_ok=True)
    with open(args.layers_json, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    print(f"wrote {args.out}: {n_t} tensors, {data_bytes/1e6:.2f} MB payload "
          f"({sum(a.size for _n, a, _i in ex.tensors):,d} elements); "
          f"{len(ex.layers)} layer entries + {len(ex.nodes)} node records -> {args.layers_json} "
          f"({os.path.getsize(args.layers_json)/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateFailure as e:
        print(f"GATE FAILED: {e}")
        raise SystemExit(2)
