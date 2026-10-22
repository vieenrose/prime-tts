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
    """ONNX conv1d reference: X [Cin,T] -> [Cout,Tout].  Used only by the packing checks."""
    Cout, Cin, K = W.shape
    T = X.shape[1]
    xp = np.zeros((Cin, T + 2 * pad), np.float32)
    xp[:, pad:pad + T] = X
    Tout = (T + 2 * pad - dilation * (K - 1) - 1) // stride + 1
    out = np.zeros((Cout, Tout), np.float32)
    for co in range(Cout):
        acc = np.full(Tout, B[co], np.float32)
        for k in range(K):
            for ci in range(Cin):
                acc += W[co, ci, k] * xp[ci, k * dilation::stride]
        out[co] = acc
    return out


def conv_transpose1d_ref(X: np.ndarray, W: np.ndarray, B: np.ndarray, stride: int, pad: int) -> np.ndarray:
    """ONNX conv_transpose1d reference: X [Cin,Tin] -> [Cout,Tout], Tout=(Tin-1)*s-2*pad+K."""
    Cin, Cout, K = W.shape
    Tin = X.shape[1]
    Tout = (Tin - 1) * stride - 2 * pad + K
    out = np.tile(B.astype(np.float32)[None, :], (Cout, 1))
    for q in range(Tin):
        for k in range(K):
            t = q * stride + k - pad
            if 0 <= t < Tout:
                out[:, t] += W[:, :, k].T @ X[:, q]
    return out


def conv_transpose1d_packed(X: np.ndarray, Wp: np.ndarray, B: np.ndarray, s: int, pad: int,
                            Cout: int, K: int) -> np.ndarray:
    """What the runtime does: s phases x ntap taps over q, then out[q*s + r - pad] = P_r[q]."""
    Cin, Tin = X.shape
    nt = convT_ntap(K, s)
    Wph = unpack_conv(Wp)  # [s*C4][Cin][nt]
    C4 = r4(Cout)
    P = np.zeros((s, C4, Tin), np.float32)
    for r in range(s):
        for j in range(nt):
            wr = Wph[r * C4:(r + 1) * C4, :, j]  # [C4,Cin]
            P[r] += wr @ X
        P[r] += B[:C4][None, :]
    Tout = (Tin - 1) * s - 2 * pad + K
    out = np.zeros((Cout, Tout), np.float32)
    for q in range(Tin):
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
    """'/enc_p/encoder/attn_layers.0/conv_q/Conv' -> 'encoder.attn_layers.0.conv_q'."""
    segs = [s for s in node_name.split("/") if s][:-1]
    out: list[str] = []
    for s in segs:
        if out and s.startswith(out[-1] + "."):
            out[-1] = s
        else:
            out.append(s)
    return ".".join(out) or "root"


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
        a = np.ascontiguousarray(np.asarray(arr, np.float32))
        self.tensors.append((name, a, itemsize))
        self.names.add(name)
        self.bytes_by_group[group] += align64(a.size * itemsize)
        self.params_by_group[group] += a.size
        return name

    def dynamic_weight(self, onnx_name: str) -> np.ndarray:
        """Weight v2.1 computes at run time: read it from a probe dump (probe_shapes naming)."""
        if not self.probe_dir:
            raise GateFailure(
                f"weight {onnx_name} is computed by the graph, not a constant: pass --probe-dir "
                f"(a tools/probe_shapes.py --dump-dir) so it can be exported")
        f = os.path.join(self.probe_dir, onnx_name.replace("/", "_") + ".npy")
        if not os.path.exists(f):
            raise GateFailure(f"dynamic weight {onnx_name} not found in probe dump {f}")
        return np.ascontiguousarray(np.load(f), np.float32)

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

    # ---- pass over the graph -----------------------------------------------------------
    def run(self) -> None:
        graph_outputs = {o.name for o in self.g.output}
        for idx, n in enumerate(self.g.node):
            a = attrs_of(n)
            grp = group_of(n.name) if n.name else group_of(n.output[0])
            mod = module_of(n.name) if n.name else module_of(n.output[0])
            entry: dict = dict(idx=idx, graph=grp, module=mod, onnx_op=n.op_type,
                               op=n.op_type.lower(), name=n.name, inputs=list(n.input),
                               outputs=list(n.output), attrs=a,
                               kind=self.kind_of(n.op_type, n), epilogue="none",
                               packed=[], dtype="fp32", params={}, shape="")
            if n.op_type == "Constant":
                entry["kind"] = "constant"
                self.export_constant(idx, n)
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
            elif n.op_type == "Pad":
                self.pad_const_entry(idx, n, grp, mod, a, entry)
            self.layers.append(entry)
            self.nodes.append(dict(idx=idx, op=n.op_type, name=n.name, inputs=list(n.input),
                                   outputs=list(n.output), attrs=a,
                                   kind=entry["kind"], packed=entry["packed"]))
        self.model_meta(graph_outputs)

    # ---- per-op exporters --------------------------------------------------------------
    def export_constant(self, idx: int, n) -> None:
        """Constant node -> container (float) or JSON literal (int/bool), per the docstring."""
        a = numpy_helper.to_array(n.attribute[0].t)
        out = n.output[0]
        grp = group_of(out)
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
        item = FP16 if h16 else FP32
        suffix = H16 if h16 else ""
        b = self.init.get(n.input[2]) if len(n.input) > 2 and n.input[2] in self.init else np.zeros(Cout, np.float32)
        wp = self.add(f"{grp}.{mod}.wp{suffix}", pack_conv(W), grp, item)
        bp = self.add(f"{grp}.{mod}.b", pad_vec(np.asarray(b, np.float32), r4(Cout)), grp, FP32)
        entry.update(op="conv1d", onnx=[wn] + ([n.input[2]] if len(n.input) > 2 else []),
                     packed=[wp, bp], dtype="fp16" if h16 else "fp32",
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
