#!/usr/bin/env python3
"""Generate the C graph tables the v2.1 runtime executes from (spec/v21_layers.json + the ONNX model).

Why generated: the runtime is dependency-free C + CUDA (docs/v21_plan.md), so it must not parse JSON
at start-up.  This script turns the 4066-node graph into flat C arrays: one tensor table (dtype +
origin: graph input / container tensor / int-constant literal / node output), one node table (op,
inputs, outputs, params, container indices), and int/float pools for the small per-node lists
(slice steps, transpose perms, split sizes, int64 constants, ConstantOfShape fills).

Dtypes come from onnx.shape_inference where available, and from producer rules otherwise (the v2.1
graph has 1380 shape-algebra nodes whose value_info shape inference does not always type).

Usage: .venv/bin/python tools/gen_v21_graph.py [--spec spec/v21_layers.json]
               [--model model/v21_mbistft_16k/primetts_v21_3voice.onnx]
               [--out runtime/v21/gen/pt_graph.c] [--header runtime/v21/gen/pt_graph_tab.h]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import shape_inference

OP_ENUM = ["NOP", "CONST", "CONSOFT", "CONSTMAT", "ADD", "SUB", "MUL", "DIV", "POW", "NEG", "EXP",
           "SIN", "COS", "TANH", "SIGMOID", "RELU", "LEAKYRELU", "CLIP", "CAST", "SHAPE", "RESHAPE",
           "TRANSPOSE", "CONCAT", "SPLIT", "UNSQUEEZE", "SQUEEZE", "GATHER", "SLICE", "PAD", "WHERE",
           "EQUAL", "LESS", "GREATER", "GEQ", "LEQ", "AND", "OR", "NOT", "XOR", "EXPAND", "SCATTERND",
           "CUMSUM", "REDUCESUM", "REDUCEMAX", "REDUCEMIN", "REDUCEMEAN", "SOFTMAX", "LAYERNORM",
           "EINSUM", "RANGE", "RANDNORM", "IDENTITY", "NONZERO", "FLOOR", "CEIL", "ABS", "SIGN",
           "ROUND", "SQRT", "MAX", "MIN", "SUM", "RECIPROCAL", "LOG", "CONV", "CONVT", "MATMUL",
           "FLATTEN", "EYELIKE", "TILE", "SOFTPLUS", "ERF", "GELU", "MEAN"]
OP_OF = {"Constant": "CONST", "ConstantOfShape": "CONSOFT", "Add": "ADD", "Sub": "SUB", "Mul": "MUL",
         "Div": "DIV", "Pow": "POW", "Neg": "NEG", "Exp": "EXP", "Sin": "SIN", "Cos": "COS",
         "Tanh": "TANH", "Sigmoid": "SIGMOID", "Relu": "RELU", "LeakyRelu": "LEAKYRELU",
         "Clip": "CLIP", "Cast": "CAST", "Shape": "SHAPE", "Reshape": "RESHAPE",
         "Transpose": "TRANSPOSE", "Concat": "CONCAT", "Split": "SPLIT", "Unsqueeze": "UNSQUEEZE",
         "Squeeze": "SQUEEZE", "Gather": "GATHER", "Slice": "SLICE", "Pad": "PAD", "Where": "WHERE",
         "Equal": "EQUAL", "Less": "LESS", "Greater": "GREATER", "GreaterOrEqual": "GEQ",
         "LessOrEqual": "LEQ", "And": "AND", "Or": "OR", "Not": "NOT", "Xor": "XOR",
         "Expand": "EXPAND", "ScatterND": "SCATTERND", "CumSum": "CUMSUM", "ReduceSum": "REDUCESUM",
         "ReduceMax": "REDUCEMAX", "ReduceMin": "REDUCEMIN", "ReduceMean": "REDUCEMEAN",
         "Mean": "REDUCEMEAN", "Softmax": "SOFTMAX", "LayerNormalization": "LAYERNORM",
         "Einsum": "EINSUM", "Range": "RANGE", "RandomNormalLike": "RANDNORM", "Identity": "IDENTITY",
         "NonZero": "NONZERO", "Floor": "FLOOR", "Ceil": "CEIL", "Abs": "ABS", "Sign": "SIGN",
         "Round": "ROUND", "Sqrt": "SQRT", "Max": "MAX", "Min": "MIN", "Sum": "SUM",
         "Reciprocal": "RECIPROCAL", "Log": "LOG", "Conv": "CONV", "ConvTranspose": "CONVT",
         "MatMul": "MATMUL", "Gemm": "CONSTMAT", "Flatten": "FLATTEN", "EyeLike": "EYELIKE",
         "Tile": "TILE", "Softplus": "SOFTPLUS", "Erf": "ERF", "Gemm": "CONSTMAT"}
# v2.1 carries every integer as int64; the C side has one integer element type (int64), so int32
# elem_types map onto it rather than introducing a 4-byte tensor the op code cannot read.
ELEM = {1: "F32", 2: "I64", 3: "I64", 4: "I64", 5: "I64", 6: "I64", 7: "I64", 8: "STR", 9: "BOOL",
        10: "F16", 11: "F64", 12: "I64", 13: "I64", 14: "I64", 15: "I64", 16: "I64", 17: "I64",
        100: "F16"}
DT_C = {"F32": "DT_F32", "I64": "DT_I64", "I32": "DT_I32", "BOOL": "DT_BOOL", "F16": "DT_F16",
        "F64": "DT_F64", "NONE": "DT_NONE"}
# producer-rule dtypes for shape-algebra nodes shape inference does not type
RULE_DT = {"Shape": "I64", "Cast": "attr:to", "Equal": "BOOL", "Less": "BOOL", "Greater": "BOOL",
           "GreaterOrEqual": "BOOL", "LessOrEqual": "BOOL", "Not": "BOOL", "And": "BOOL", "Or": "BOOL",
           "Xor": "BOOL", "Range": "in:0", "RandomNormalLike": "attr:dtype",
           "ConstantOfShape": "attr:value_dtype", "NonZero": "I64", "CumSum": "in:0",
           "Gather": "in:0", "Slice": "in:0", "Pad": "in:0", "Concat": "in:0", "Reshape": "in:0",
           "Transpose": "in:0", "Squeeze": "in:0", "Unsqueeze": "in:0", "Expand": "in:0",
           "Split": "in:0", "ScatterND": "in:0", "Identity": "in:0", "ReduceSum": "in:0",
           "ReduceMax": "in:0", "ReduceMin": "in:0", "ReduceMean": "in:0", "Mean": "in:0",
           "Sum": "in:0", "Max": "in:0", "Min": "in:0", "Where": "in:1"}


def pool():
    return [], []


def emit_array(out, name, ctype, vals, per_line=12):
    out.append(f"const {ctype} {name}[] = {{")
    line = []
    for i, v in enumerate(vals):
        line.append(v if isinstance(v, str) else repr(v))
        if len(line) == per_line:
            out.append("  " + ", ".join(line) + ",")
            line = []
    if line:
        out.append("  " + ", ".join(line))
    out.append("};")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="spec/v21_layers.json")
    ap.add_argument("--model", default="model/v21_mbistft_16k/primetts_v21_3voice.onnx")
    ap.add_argument("--out", default="runtime/v21/gen/pt_graph.c")
    ap.add_argument("--header", default="runtime/v21/gen/pt_graph_tab.h")
    args = ap.parse_args()

    spec = json.load(open(args.spec))
    m = onnx.load(args.model)
    g = m.graph
    si = shape_inference.infer_shapes(m, strict_mode=False)
    et = {}
    for vi in list(si.graph.value_info) + list(si.graph.input) + list(si.graph.output):
        if vi.type.tensor_type.elem_type:
            et[vi.name] = ELEM.get(vi.type.tensor_type.elem_type, "F32")
    init = {i.name: numpy_helper_arr(i) for i in g.initializer}
    const_tensor.update({n.output[0]: n for n in g.node if n.op_type == "Constant"})
    prod = {o: n for n in g.node for o in n.output}
    layers = {l["idx"]: l for l in spec["layers"]}

    # ---- tensor table -------------------------------------------------------------
    names: list[str] = []
    ids: dict[str, int] = {}

    def tid(nm: str) -> int:
        if nm not in ids:
            ids[nm] = len(names)
            names.append(nm)
        return ids[nm]

    origin = {}       # tensor id -> "IN" | "CONT:<cname>" | "LIT:<pool_off>,<n>,<dt>" | "OUT"
    dtype = {}        # tensor id -> F32/I64/BOOL/...

    for i in g.input:
        t = tid(i.name)
        origin[t] = "IN"
        dtype[t] = ELEM.get(i.type.tensor_type.elem_type, "F32")
    for nm, a in init.items():
        t = tid(nm)
        origin[t] = "INIT"          # resolved to a container tensor below
        # dtype from the tensor's own storage: an int64 initializer typed as fp32 turns a shape
        # vector into float bits and the whole shape algebra reads garbage
        dtype[t] = ("I64" if a.dtype in (np.int64, np.int32, np.uint64, np.uint32)
                    else "BOOL" if a.dtype == np.bool_ else "F32")
    for nm, n in const_tensor.items():
        t = tid(nm)
        a = numpy_helper_arr(n)
        dtype[t] = "F32" if a.dtype == np.float32 else ("I64" if a.dtype in (np.int64, np.int32)
                                                        else "BOOL")
    for n in g.node:
        for o in n.output:
            if o:
                tid(o)
    for n in g.node:
        for o in n.output:
            if not o:
                continue
            t = ids[o]
            origin.setdefault(t, "OUT")
            if t in dtype:
                continue
            if o in et:
                dtype[t] = et[o]
                continue
            d = derive_dtype(n)
            dtype[t] = d

    def derive_dtype(node) -> str:
        if node.op_type in ("Constant",):
            a = numpy_helper_arr(const_tensor[node.output[0]])
            return "F32" if a.dtype == np.float32 else ("I64" if a.dtype in (np.int64, np.int32)
                                                        else "BOOL")
        rule = RULE_DT.get(node.op_type)
        if rule is None:
            return "F32"
        if rule == "BOOL":
            return "BOOL"
        if rule == "I64":
            return "I64"
        if rule.startswith("attr:"):
            key = rule[5:]
            a = attrs(node)
            if key == "value_dtype":
                v = a.get("value")
                return "F32" if getattr(v, "data_type", 1) == 1 else "I64"
            return ELEM.get(int(a.get(key, 1)), "F32")
        if rule.startswith("in:"):
            i = int(rule[3:])
            nm = node.input[i] if len(node.input) > i else ""
            if nm in ids and dtype.get(ids[nm]):
                return dtype[ids[nm]]
            p = prod.get(nm)
            if p is not None:
                return derive_dtype(p)
            return "F32"
        return "F32"

    # constants table: float -> container, int/bool -> literal pool
    ipool: list[int] = []
    # container-name table: every tensor the exporter put in build/v21.ptk, in first-appearance
    # order (weights/biases first, then constants).  The C runtime resolves these names to device
    # pointers once at load; node w0/w1 and tensor origins index into it.
    cnames: list[str] = []
    ctab: dict[str, int] = {}

    def cidx(nm: str | None) -> int:
        if not nm:
            return -1
        if nm not in ctab:
            ctab[nm] = len(cnames)
            cnames.append(nm)
        return ctab[nm]
    for c in spec["constants"]:
        t = tid(c["output"])
        if c.get("container"):
            origin[t] = f"CONT:{cidx(c['container'])}|{proto_shape(c['output'])}"
        else:
            off = len(ipool)
            vals = [int(v) for v in (c["values"] or [])]
            ipool += vals
            # the literal's rank matters (a rank-0 constant is a scalar; a rank-1 one is a vector),
            # so the shape travels with the pool offset
            origin[t] = (f"LIT:{off},{len(vals)},{'I64' if c['dtype'] == 'int64' else 'I32'}|"
                         f"{proto_shape(c['output'])}")

    # initializers -> container tensor via the layer entries that export them
    init_cont: dict[str, str] = {}
    for l in spec["layers"]:
        p = l.get("params", {})
        op, packed, onx = l.get("op"), l.get("packed") or [], l.get("onnx") or []
        if not packed:
            continue
        if op in ("conv1d", "conv_transpose1d"):
            if onx:
                init_cont[onx[0]] = packed[0]
            if len(onx) > 1 and len(packed) > 1:
                init_cont[onx[1]] = packed[1]
        elif op in ("embedding", "pad_relative_position"):
            init_cont[onx[0]] = packed[0]
        elif op == "layernorm":
            init_cont[onx[0]] = packed[0]
            if len(onx) > 1:
                init_cont[onx[1]] = packed[1]
        elif op == "matmul" and onx:
            init_cont[onx[0]] = packed[0]
    init_dims = {i.name: list(i.dims) for i in g.initializer}
    for nm, cn in init_cont.items():
        t = ids[nm]
        shp = ",".join(str(v) for v in init_dims.get(nm, []))
        if not shp and str(origin.get(t, "")).startswith("CONT:"):
            continue      # a Constant output folded as a weight: keep the Constant's own rank
        origin[t] = f"CONT:{cidx(cn)}|{shp}"

    # ---- node table ---------------------------------------------------------------
    fpool: list[float] = []
    nodes = []
    for idx, n in enumerate(g.node):
        l = layers.get(idx, {})
        a = attrs(n)
        op = OP_OF.get(n.op_type, "NOP")
        ins = [tid(i) if i else -1 for i in n.input][:12]
        outs = [tid(o) for o in n.output if o]
        i0 = i1 = i2 = i3 = -1
        f0 = f1 = 0.0
        w0 = w1 = -1
        pl, pn = -1, 0
        fl, fn, fdt = -1, 0, "F32"
        eq = ""
        if op == "CONV" or op == "CONVT":
            p = l.get("params", {})
            i0, i1, i2, i3 = int(p["kernel"]), int(p["stride"]), int(p["pad"]), int(p["dilation"])
            w0 = cidx(l["packed"][0])
            w1 = cidx(l["packed"][1])
            f0 = float(p["cout"])
            f1 = float(p["cin"])
        elif op == "LAYERNORM":
            i0, i1 = int(a.get("axis", -1)), -1
            f0 = float(a.get("epsilon", 1e-5))
            w0, w1 = cidx(l["packed"][0]), cidx(l["packed"][1])
        elif op == "GATHER":
            i0 = int(a.get("axis", 0))   # every Gather carries its axis, not just embeddings
            if l.get("packed"):
                w0 = cidx(l["packed"][0])
        elif op == "PAD":
            mode = str(a.get("mode", "constant"))
            i0 = {"constant": 0, "reflect": 1, "edge": 2, "wrap": 3}[mode]
            if l.get("op") == "pad_relative_position":
                w0 = cidx(l["packed"][0])
        elif op == "CAST":
            i0 = int(a.get("to", 1))
        elif op == "TRANSPOSE":
            perm = [int(v) for v in a.get("perm", [])]
            pl, pn = len(ipool), len(perm)
            ipool += perm
        elif op == "CONCAT":
            i0 = int(a.get("axis", 0))
        elif op == "SPLIT":
            i0 = int(a.get("axis", 0))
            sp = n.input[1] if len(n.input) > 1 else None
            vals = []
            if sp and sp in ids and origin[ids[sp]].startswith("LIT"):
                off, ln = parse_lit(origin[ids[sp]])
                vals = ipool[off:off + ln]
            elif "split" in a:
                vals = [int(v) for v in a["split"]]
            if vals:
                pl, pn = len(ipool), len(vals)
                ipool += vals
        elif op == "SLICE":
            pl = len(ipool)
            pn = 1
            ipool.append(int(a.get("steps", 1)) if "steps" in a else 1)
        elif op in ("REDUCESUM", "REDUCEMAX", "REDUCEMIN", "REDUCEMEAN"):
            i0 = int(a.get("keepdims", 1))
            i1 = int(a.get("noop_with_empty_axes", 0))
        elif op == "CUMSUM":
            i0, i1 = int(a.get("exclusive", 0)), int(a.get("reverse", 0))
        elif op == "SOFTMAX":
            i0 = int(a.get("axis", -1))
        elif op == "LEAKYRELU":
            f0 = float(a.get("alpha", 0.01))
        elif op == "EINSUM":
            eq = str(a.get("equation", "")).replace("\u2192", "->")
        elif op == "CONSOFT":
            v = numpy_helper_arr_attr(a.get("value"))
            fl, fn = len(fpool), int(v.size)
            fdt = "F32" if v.dtype == np.float32 else "I64"
            if v.dtype == np.float32:
                fpool += [float(x) for x in v.reshape(-1)]
            else:
                fl = -1
                pl = len(ipool)
                pn = 1
                ipool.append(int(v.reshape(-1)[0]))
        elif op == "RANDNORM":
            i0 = int(a.get("dtype", 1))
        nodes.append(dict(idx=idx, op=op, ins=ins, outs=outs, i0=i0, i1=i1, i2=i2, i3=i3,
                          f0=f0, f1=f1, w0=w0, w1=w1, pl=pl, pn=pn, fl=fl, fn=fn, fdt=fdt,
                          eq=eq, name=n.name or (outs[0] if outs else str(idx))))

    # liveness: last node index that reads each tensor (for allocator reuse in the C runtime)
    last_use = [-1] * len(names)
    for k, nd in enumerate(nodes):
        for t in nd["ins"]:
            if t >= 0:
                last_use[t] = k
    graph_outs = {o.name for o in g.output}
    for t, nm in enumerate(names):
        if nm in graph_outs:
            last_use[t] = len(nodes) - 1

    # ---- emit -------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = [f"/* generated by tools/gen_v21_graph.py from {os.path.basename(args.spec)} + "
           f"{os.path.basename(args.model)} - do not edit */",
           '#include "pt_v21.h"',
           '#include "pt_graph_tab.h"',
           '#include <string.h>', ""]
    emit_array(out, "pt_ipool", "int64_t", ipool)
    emit_array(out, "pt_fpool", "float", [repr(f) for f in fpool], 8)
    hdr = [
        "/* generated by tools/gen_v21_graph.py - tensor/node tables for the v2.1 runtime */",
        "#ifndef PT_GRAPH_TAB_H_", "#define PT_GRAPH_TAB_H_", '#include "pt_v21.h"',
        f"#define PT_N_TENSOR {len(names)}", f"#define PT_N_NODE {len(nodes)}",
        f"#define PT_N_CONT {len(cnames)}",
        "extern const int64_t pt_ipool[];", "extern const float pt_fpool[];",
        "extern const pt_tensor_def pt_tdefs[];", "extern const pt_node_def pt_ndefs[];",
        "extern const char *const pt_eq_tab[];",
        f"extern const int pt_graph_last_use[{len(names)}];",
        f"extern const char *const pt_graph_tnames[{len(names)}];",
        f"extern const char *const pt_graph_ctab[{len(cnames)}];",
        "extern const int pt_graph_wav_tensor;", "", "#endif /* PT_GRAPH_TAB_H_ */", ""]
    out.append("const int pt_graph_last_use[PT_N_TENSOR] = {")
    for i in range(0, len(last_use), 12):
        out.append("  " + ", ".join(str(v) for v in last_use[i:i + 12]) + ",")
    out.append("};")
    out.append("const char *const pt_graph_tnames[PT_N_TENSOR] = {")
    for nm in names:
        out.append(f"  \"{nm}\",")
    out.append("};")
    out.append("const char *const pt_graph_ctab[PT_N_CONT] = {")
    for cn in cnames:
        out.append(f"  \"{cn}\",")
    out.append("};")
    out.append(f"const int pt_graph_wav_tensor = {ids.get('wav', -1)};")
    out.append("")
    out.append("const pt_tensor_def pt_tdefs[PT_N_TENSOR] = {")
    for t, nm in enumerate(names):
        o = origin.get(t, "NONE")
        dt = dtype.get(t, "F32")
        out.append(f"  {{ {DT_C[dt]}, \"{o}\" }},")
    out.append("};")
    out.append("")
    out.append("const pt_node_def pt_ndefs[PT_N_NODE] = {")
    for nd in nodes:
        ins = ",".join(str(v) for v in (nd["ins"] + [-1] * 12)[:12])
        outs = ",".join(str(v) for v in (nd["outs"] + [-1] * 4)[:4])
        out.append(f"  {{ PT_{nd['op']}, {nd['i0']}, {nd['i1']}, {nd['i2']}, {nd['i3']}, "
                   f"{nd['f0']!r}f, {nd['f1']!r}f, {nd['w0']}, {nd['w1']}, {nd['pl']}, {nd['pn']}, "
                   f"{nd['fl']}, {nd['fn']}, {DT_C[nd['fdt']]}, "
                   f"{{ {ins} }}, {{ {outs} }}, {len(nd['outs'])} }},")
    out.append("};")
    out.append("")
    out.append("const char *const pt_eq_tab[PT_N_NODE] = {")
    for nd in nodes:
        out.append(f"  \"{nd['eq']}\",")
    out.append("};")
    out.append("")
    out.append("/* op enum, mirrors PT_OPS in pt_v21.h */")
    out.append("const int pt_op_ids[] = {")
    out.append("  " + ", ".join(f"PT_{o}" for o in OP_ENUM) + ",")
    out.append("};")
    open(args.out, "w").write("\n".join(out) + "\n")
    open(args.header, "w").write("\n".join(hdr) + "\n")
    print(f"wrote {args.out}: {len(names)} tensors, {len(nodes)} nodes, "
          f"{len(ipool)} int pool, {len(fpool)} float pool, {len(ctab)} container constants")
    dts = {}
    for t in range(len(names)):
        dts[dtype.get(t, "F32")] = dts.get(dtype.get(t, "F32"), 0) + 1
    print("tensor dtypes:", dts)
    ops = {}
    for nd in nodes:
        ops[nd["op"]] = ops.get(nd["op"], 0) + 1
    print("node ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    return 0


const_tensor: dict = {}


def proto_shape(tname: str) -> str:
    """rank/dims straight from the Constant node's TensorProto (the exporter's own `shape` field is
    the flattened storage shape, which loses the rank of the constants v2.1 reshapes with)"""
    n = const_tensor.get(tname)
    if n is None:
        return ""
    for a in n.attribute:
        if a.name == "value":
            return ",".join(str(v) for v in list(a.t.dims))
    return ""


def attrs(node) -> dict:
    from onnx import helper
    a = {}
    for x in node.attribute:
        v = helper.get_attribute_value(x)
        if isinstance(v, bytes):
            v = v.decode()
        a[x.name] = v
    return a


def numpy_helper_arr(x):
    """array for an initializer TensorProto OR a Constant node (its 'value' attribute)"""
    from onnx import numpy_helper
    if hasattr(x, "op_type") and x.op_type == "Constant":
        for a in x.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
        raise KeyError("Constant without a value attribute")
    return numpy_helper.to_array(x)


def numpy_helper_arr_attr(t):
    from onnx import numpy_helper
    return numpy_helper.to_array(t)


def parse_lit(o: str) -> tuple[int, int]:
    body = o.split(":", 1)[1].split(",")
    return int(body[0]), int(body[1])


if __name__ == "__main__":
    sys.exit(main())
