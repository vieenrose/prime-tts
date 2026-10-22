#!/usr/bin/env python3
"""Probe every intermediate shape/value of an ONNX graph by exposing all node outputs.

Usage: probe_shapes.py <model.onnx> <out.json> [--case case.json] [--dump-tensors]

Gives exact tensor shapes and per-node MAC counts for a real input, which is what the
from-scratch runtime needs (kernels are shape-parameterised, not ONNX-generic).
"""
import argparse, json, sys, collections
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort


def build_probe(model):
    """Expose every node output as a graph output, with its real element type."""
    si = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    et = {}
    for vi in list(si.graph.value_info) + list(si.graph.input) + list(si.graph.output):
        t = vi.type.tensor_type.elem_type
        if t:
            et[vi.name] = t
    g = model.graph
    existing = {o.name for o in g.output}
    names = []
    for n in g.node:
        for o in n.output:
            if o and o not in existing and o not in names:
                names.append(o)
    out = list(g.output)
    for nm in names:
        out.append(helper.make_tensor_value_info(nm, et.get(nm, onnx.TensorProto.FLOAT), None))
    g.ClearField("output")
    g.output.extend(out)
    return model, names


def macs(node, shapes, weights):
    op = node.op_type
    def s(n):
        return shapes.get(n)
    attrs = {a.name: (helper.get_attribute_value(a)) for a in node.attribute}
    ins = [s(i) for i in node.input if s(i) is not None]
    if op in ("Conv", "ConvTranspose") and len(ins) < 2 and weights:
        w0 = weights.get(node.input[1])
        if w0 is not None:
            ins.append(list(w0.shape))
    if op == "Conv":
        if len(ins) < 2:
            return 0
        w = ins[1]
        K = np.prod(w[2:]) if len(w) > 2 else 1
        out = s(node.output[0])
        if out is None:
            return 0
        # output [N, Cout, T] ; weight [Cout, Cin/group, K]
        cout, cin = w[0], w[1]
        T = np.prod(out[2:])
        return int(cout * cin * K * T)
    if op == "ConvTranspose":
        if len(ins) < 2:
            return 0
        w = ins[1]
        K = np.prod(w[2:]) if len(w) > 2 else 1
        out = s(node.output[0])
        if out is None:
            return 0
        cout, cin = w[1], w[0]
        T = np.prod(out[2:])
        return int(cout * cin * K * T)
    if op == "MatMul":
        if len(ins) < 2:
            return 0
        a, b = ins
        n = np.prod(a[:-1]) if len(a) > 1 else 1
        k, m = a[-1], b[-1]
        return int(n * k * m)
    if op in ("Add", "Mul", "Sub", "Div", "Tanh", "Sigmoid", "Exp", "Relu", "LeakyRelu", "Pow", "Sin", "Cos",
              "Neg", "Clip", "Cast", "Softmax", "LayerNormalization"):
        out = s(node.output[0])
        return int(np.prod(out)) if out is not None else 0
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--case", help="json with input lists; else random")
    ap.add_argument("--dump-dir", help="save all intermediate tensors as .npy")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sid", type=int, default=0)
    args = ap.parse_args()

    m = onnx.load(args.model)
    m2, extra = build_probe(m)
    onnx.save(m2, "/tmp/_probe.onnx")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession("/tmp/_probe.onnx", sess_options=so, providers=["CPUExecutionProvider"])

    feed = json.load(open(args.case)) if args.case else {}
    rng = np.random.default_rng(args.seed)
    T = args.frames
    for i in sess.get_inputs():
        if i.name in feed:
            v = np.asarray(feed[i.name])
            feed[i.name] = v.astype(np.int64) if "int64" in i.type else v.astype(np.float32)
            continue
        shape = [d if isinstance(d, int) and d > 0 else T for d in i.shape]
        isint = "int64" in i.type
        if "bool" in i.type:
            v = np.zeros(shape, bool)
        elif i.name in ("noise_scale",):
            v = np.array([0.0], np.float32)
        elif i.name in ("length_scale",):
            v = np.array([1.0], np.float32)
        elif i.name in ("sid", "speaker"):
            v = np.array([args.sid], np.int64)
        elif "length" in i.name and isint:
            v = np.full(shape, shape[-1] if shape[-1] > 1 else T, np.int64)
        elif isint:
            v = np.zeros(shape, np.int64)
            emb = {}
            for it in m.graph.initializer:
                if it.name.endswith(".weight") and "emb" in it.name:
                    emb[it.name.split(".")[-2]] = int(numpy_helper.to_array(it).shape[0])
            alias = {"x": "emb_phone", "phone": "emb_phone", "tone": "emb_tone", "lang": "emb_lang"}
            lim = emb.get(alias.get(i.name, ""), 80)
            if shape and len(shape) >= 2 and i.name in ("x", "phone", "lang", "tone"):
                v[..., 1::2] = rng.integers(1, max(2, lim), size=v[..., 1::2].shape)
            elif len(shape) == 1:
                v[:] = rng.integers(1, 4, size=shape)
        else:
            v = (rng.random(shape) * 0.2).astype(np.float32)
        feed[i.name] = v
    out = sess.run(None, feed)
    names = [o.name for o in sess.get_outputs()]
    shapes = {n: list(a.shape) for n, a in zip(names, out) if isinstance(a, np.ndarray)}

    per_node = []
    by_group = collections.Counter()
    total = 0
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    for n in m.graph.node:
        c = macs(n, shapes, inits)
        per_node.append((n.op_type, n.name, c, shapes.get(n.output[0])))
        total += c
        by_group[n.name.split("/")[1] if "/" in n.name else n.name.split("_")[0]] += c
    per_node.sort(key=lambda x: -x[2])
    rep = {
        "model": args.model,
        "feed_shapes": {k: list(np.shape(v)) for k, v in feed.items()},
        "total_macs": total,
        "top_nodes": [{"op": o, "name": nm, "macs": c, "out_shape": sh} for o, nm, c, sh in per_node[:args.top]],
        "by_group": dict(by_group.most_common(40)),
        "op_macs": dict(collections.Counter({o: sum(c for oo, _, c, _ in per_node if oo == o) for o in set(p[0] for p in per_node)})),
        "shapes": {k: v for k, v in shapes.items() if len(v) >= 2},
    }
    json.dump(rep, open(args.out, "w"), indent=1)
    print("total MACs:", total, "( %.3f GMAC )" % (total / 1e9))
    print("output shapes:", [(n, shapes.get(n)) for n in names if n in ("wav",)][:1])
    print("top ops by MAC:")
    agg = collections.Counter()
    for o, nm, c, sh in per_node:
        agg[o] += c
    for o, c in agg.most_common(12):
        print(f"   {o:18s} {c/1e6:12.1f} M")
    print("top nodes:")
    for o, nm, c, sh in per_node[:18]:
        print(f"   {o:14s} {c/1e6:9.1f}M {str(sh):28s} {nm[:52]}")
    if args.dump_dir:
        import os
        os.makedirs(args.dump_dir, exist_ok=True)
        for n, a in zip(names, out):
            if isinstance(a, np.ndarray):
                np.save(f"{args.dump_dir}/{n.replace('/','_')}.npy", a.astype(np.float32))
        print("dumped", len(names), "tensors to", args.dump_dir)


main()
