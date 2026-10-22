#!/usr/bin/env python3
"""Gate (d): read build/v21.ptk back from disk and prove it unpacks to the ONNX weights.

Everything here is derived independently of the exporter's in-memory state: the container is
re-opened, its records are re-parsed, the expected packed shapes are RE-DERIVED from the layer
spec's params, and every packed conv / conv-transpose / embedding / relpos / LayerNorm weight is
unpacked and compared elementwise with the ONNX tensor (initializers, Constant-node tensors, and -
for the four graph-computed weights - the tensor value ONNXRuntime produced in the probe dump).

Checks (any failure exits non-zero and names the tensor):
  (1) container structure: magic/version, 64-byte aligned non-overlapping payloads,
      record nelem x itemsize == payload span, header data_bytes == last payload end;
  (2) per-entry shape: container record shape == shape re-derived from params
      conv  -> [r4(cout)/4][kernel][cin][4], bias [cout4]
      convT -> [phases * r4(cout)/4][taps_per_phase][cin][4], bias [cout4]
      embed -> [rows][cols], relpos -> table dims, layernorm -> [C] x2;
  (3) gate (d) value: unpacked conv/convT weight == ONNX value within 1e-6 elementwise max,
      bias within 1e-6 (fp16 entries are compared against round_fp16(ONNX), which fp16 storage
      represents exactly, so their expected diff is 0);
  (4) fp16 payload hygiene: every '#h' payload is exactly fp16-representable, every other
      payload is fp32, and 'dtype.table' (one itemsize per record, record order) agrees with
      the name convention;
  (5) no container tensor is unreferenced by the layer spec / constants table, and no referenced
      tensor is missing from the container.

Usage: .venv/bin/python tests/v21_container_smoke.py
       [--container build/v21.ptk] [--layers-json spec/v21_layers.json]
       [--model model/v21_mbistft_16k/primetts_v21_3voice.onnx]
       [--probe-dir tests/golden_v21/_probe]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from export_v21 import (HDR, MAGIC, REC, VERSION, align64, as_fp16_values, r4, unpack_conv,  # noqa: E402
                        unpack_convtranspose)

TOL = 1e-6


class SmokeFailure(Exception):
    pass


def probe_tensor(probe_dir: str, name: str) -> np.ndarray:
    """ONNXRuntime's own value for a graph-computed tensor (probe dump, '/' -> '_')."""
    p = os.path.join(probe_dir, name.replace("/", "_") + ".npy")
    if not os.path.exists(p):
        raise SmokeFailure(f"no probe dump for graph-computed weight {name} (looked for {p})")
    return np.ascontiguousarray(np.load(p), np.float32)


def check_structure(path: str) -> tuple[dict[str, tuple[np.ndarray, int]], int]:
    """(1) re-parse the container: header, record alignment, payload spans."""
    raw = open(path, "rb").read()
    magic, ver, n, _res, data_bytes = HDR.unpack_from(raw, 0)
    if magic != MAGIC:
        raise SmokeFailure(f"bad magic 0x{magic:x} in {path}")
    if ver != VERSION:
        raise SmokeFailure(f"container version {ver} != {VERSION} in {path}")
    if HDR.size + n * REC.size > len(raw):
        raise SmokeFailure(f"header claims {n} records ({HDR.size + n * REC.size} B) but the file "
                           f"is only {len(raw)} B")
    data_off = align64(HDR.size + n * REC.size)
    if data_off + data_bytes > len(raw):
        raise SmokeFailure(f"header data section [{data_off}, {data_off + data_bytes}) exceeds "
                           f"file size {len(raw)} B")
    prev_end, out = 0, {}
    for i in range(n):
        nm, ndim, d0, d1, d2, d3, d4, d5, nelem, off = REC.unpack_from(raw, HDR.size + i * REC.size)
        nm = nm.split(b"\0")[0].decode()
        isz = 2 if nm.endswith("#h") else 4
        if off < prev_end:
            raise SmokeFailure(f"record {i} ({nm}) offset {off} overlaps the previous payload "
                               f"(ends {prev_end})")
        if off % 64:
            raise SmokeFailure(f"record {i} ({nm}) offset {off} is not 64-byte aligned")
        if nelem * isz > data_bytes - off:
            raise SmokeFailure(f"record {i} ({nm}) payload {nelem * isz} B runs past the data "
                               f"section end ({data_bytes} B at offset {off})")
        dt = np.float16 if isz == 2 else np.float32
        dims = [d for d in (d0, d1, d2, d3, d4, d5)][: max(ndim, 0)]
        arr = np.frombuffer(raw, dtype=dt, count=nelem, offset=data_off + off)
        out[nm] = (arr.reshape(dims) if ndim else arr.reshape(())).copy(), isz
        prev_end = off + align64(nelem * isz)
    if prev_end > data_bytes:
        raise SmokeFailure(f"last payload ends at {prev_end}, header says data_bytes={data_bytes}")
    return out, data_bytes


def expected_shapes(op: str, p: dict) -> list[tuple[str, tuple]]:
    """(2) the packed shapes re-derived from the layer entry's params."""
    c4 = int(p.get("cout4", 0))
    if op == "conv1d":
        w = (c4 // 4 * int(p.get("group", 1)), int(p["kernel"]), int(p["cin"]), 4)
        return [("wp", w), ("b", (c4,))]
    if op == "conv_transpose1d":
        w = (int(p["phases"]) * (c4 // 4), int(p["taps_per_phase"]), int(p["cin"]), 4)
        return [("wp", w), ("b", (c4,))]
    if op == "matmul":
        return [("wp", (c4 // 4, 1, int(p["cin"]), 4)), ("b", (c4,))]
    if op == "embedding":
        return [("emb", (int(p["rows"]), int(p["cols"])))]
    if op == "pad_relative_position":
        return [("relpos", tuple(int(v) for v in p["table"]))]
    if op == "layernorm":
        return [("g", (int(p["C"]),)), ("beta", (int(p["C"]),))]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="build/v21.ptk")
    ap.add_argument("--layers-json", default="spec/v21_layers.json")
    ap.add_argument("--model", default="model/v21_mbistft_16k/primetts_v21_3voice.onnx")
    ap.add_argument("--probe-dir", default="tests/golden_v21/_probe")
    args = ap.parse_args()

    spec = json.load(open(args.layers_json))
    back, data_bytes = check_structure(args.container)
    g = onnx.load(args.model).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    const = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in g.node
             if n.op_type == "Constant"}

    def reference(nm: str) -> tuple[str, np.ndarray]:
        """The value the container must reproduce, plus where that value came from."""
        if nm in init:
            return "initializer", np.ascontiguousarray(init[nm], np.float32)
        if nm in const:
            return "constant", np.ascontiguousarray(const[nm], np.float32)
        return "probe", probe_tensor(args.probe_dir, nm)

    lines: list[str] = []
    per_op: dict[str, dict[str, float]] = {}
    checked = 0
    for l in spec["layers"]:
        op, packed = l.get("op"), l.get("packed") or []
        if not packed:
            continue
        p = l.get("params", {})
        # (2) shapes re-derived from params must equal the container's recorded shapes
        for (role, want) in expected_shapes(op, p):
            cand = [nm for nm in packed if nm.endswith("." + role) or nm.endswith("." + role + "#h")]
            if not cand:
                raise SmokeFailure(f"{l['name']} ({op}): no packed tensor for role {role} "
                                   f"in {packed}")
            nm = cand[0]
            if nm not in back:
                raise SmokeFailure(f"{l['name']} ({op}): packed tensor {nm} missing from "
                                   f"{args.container}")
            got = tuple(back[nm][0].shape)
            if got != want:
                raise SmokeFailure(f"{l['name']} ({op}) {nm}: container shape {got} != shape "
                                   f"re-derived from params {want}")
        nm_w = packed[0]
        arr_w, isz = back[nm_w]
        src_nm = l["onnx"][0]
        src, ref = reference(src_nm)

        # (3) unpack and compare with the ONNX value
        if op == "conv1d":
            W = unpack_conv(np.ascontiguousarray(arr_w, np.float32))[: int(p["cout"])]
        elif op == "conv_transpose1d":
            W = unpack_convtranspose(np.ascontiguousarray(arr_w, np.float32), int(p["stride"]),
                                     int(p["cout"]), int(p["cin"]), int(p["kernel"]))
        else:
            W = np.ascontiguousarray(arr_w, np.float32)
        want_ref = as_fp16_values(ref) if isz == 2 else ref
        if W.shape != want_ref.shape:
            raise SmokeFailure(f"{l['name']} ({op}) {nm_w}: unpacked shape {W.shape} != ONNX "
                               f"shape {want_ref.shape} (source {src} {src_nm})")
        dW = float(np.max(np.abs(W - want_ref)))
        dB = 0.0
        if len(packed) > 1:
            n = int(p.get("cout", p.get("C", 0)))
            gotB = np.asarray(back[packed[1]][0].reshape(-1)[:n], np.float32)
            if op in ("conv1d", "conv_transpose1d"):
                bn = l["onnx"][1] if len(l["onnx"]) > 1 else None
                wantB = np.zeros(n, np.float32)
                if bn is not None:
                    _, v = reference(bn)
                    wantB = np.asarray(v, np.float32).reshape(-1)[:n]
            elif op == "layernorm":
                _, v = reference(l["onnx"][1])
                wantB = np.asarray(v, np.float32).reshape(-1)[:n]
            else:
                wantB = np.zeros(n, np.float32)
            dB = float(np.max(np.abs(gotB - wantB)))
        if dW > TOL or dB > TOL:
            raise SmokeFailure(f"gate(d) {l['name']} ({op}) via {nm_w}: max|unpacked-ONNX|={dW:.3e} "
                               f"max|bias-b|={dB:.3e} (tol {TOL:g}, source {src} {src_nm}, "
                               f"shape {W.shape}, itemsize {isz})")
        st = per_op.setdefault(op, dict(n=0, dW=0.0, dB=0.0))
        st["n"] += 1
        st["dW"] = max(st["dW"], dW)
        st["dB"] = max(st["dB"], dB)
        checked += 1
        if op in ("conv1d", "conv_transpose1d"):
            lines.append(f"  {op:16s} {nm_w:44s} W={str(tuple(W.shape)):22s} isz={isz} "
                         f"src={src:11s} max|unpacked-ONNX|={dW:.2e} max|bias|={dB:.2e}  OK")

    # (4) fp16 payload hygiene + dtype.table agreement
    for nm, (arr, isz) in back.items():
        if nm == "dtype.table":
            continue
        if isz == 2:
            if np.max(np.abs(arr.astype(np.float32) - as_fp16_values(arr.astype(np.float32)))) > 0.0:
                raise SmokeFailure(f"{nm}: '#h' payload is not exactly fp16-representable")
        elif arr.dtype != np.float32:
            raise SmokeFailure(f"{nm}: non-'#h' payload has dtype {arr.dtype}, expected float32")
    if "dtype.table" not in back:
        raise SmokeFailure("container has no dtype.table tensor")
    table, tis = back["dtype.table"]
    if tis != 4:
        raise SmokeFailure(f"dtype.table payload is {tis} B/item, expected 4")
    order = list(back)  # read_ptk preserves record order
    flat = table.reshape(-1)
    if flat.shape[0] != len(order):
        raise SmokeFailure(f"dtype.table has {flat.shape[0]} entries for {len(order)} records")
    bad = [nm for nm, v in zip(order, flat) if int(v) != back[nm][1]]
    if bad:
        raise SmokeFailure(f"dtype.table disagrees with the '#h' name convention for "
                           f"{len(bad)} tensors, first: {bad[0]}")

    # (5) reference bookkeeping
    refs = set()
    for l in spec["layers"]:
        refs.update(l.get("packed") or [])
    for c in spec["constants"]:
        if c.get("container"):
            refs.add(c["container"])
    refs.update({"model.meta", "dec.geom", "dtype.table"})
    unref = sorted(set(back) - refs)
    if unref:
        raise SmokeFailure(f"{len(unref)} container tensors referenced by nothing, first: {unref[0]}")
    missing = sorted(refs - set(back))
    if missing:
        raise SmokeFailure(f"{len(missing)} tensors the spec references are not in the container, "
                           f"first: {missing[0]}")

    print("\n".join(lines))
    print(f"gate (d): {checked} packed weight entries from {args.container} "
          f"({data_bytes:,d} B payload, {len(back)} tensors) unpacked and compared with ONNX "
          f"(tol {TOL:g}):")
    for op, st in sorted(per_op.items()):
        print(f"  {op:22s} {st['n']:4d} entries  max|unpacked-ONNX|={st['dW']:.3e}  "
              f"max|bias-b|={st['dB']:.3e}")
    n16 = sum(1 for v in back.values() if v[1] == 2)
    print(f"  structure: {len(back)} records, 64-B aligned, non-overlapping; dtype.table matches "
          f"the '#h' convention for all {len(back)} records; {len(back) - n16} fp32 / {n16} fp16 "
          f"payloads")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as e:
        print(f"GATE FAILED: {e}")
        raise SystemExit(2)
