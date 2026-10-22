#!/usr/bin/env python3
"""Export PrimeTTS v1b_8k (encoder + decoder + vocoder ONNX) into ONE .ptk container.

Container layout (read by runtime/src/weights.c):
  header  <IIIiq   magic=0x50544B31 'PTK1', version, n_tensor, reserved, data_bytes
  records n_tensor x <64si4x6qqq  name, ndim, dim[6], nelem, off   (136 bytes each)
  pad     zeros up to a 64-byte boundary
  data    fp32 payloads, each 64-byte aligned; `off` is relative to the data start

Kernel-friendly layouts (see runtime/include/pt.h):
  "<mod>.wp"  conv / linear weights packed for the 4-row micro-kernel:
              [Cout4/4][K][Cin][4]  (4 consecutive output channels contiguous),
              Cout zero-padded to a multiple of 4.  Linear y = x @ M is a K=1 conv with W = M^T.
  "<mod>.b"   bias, zero-padded to Cout4.
  "<mod>.dw"  depthwise weights [Cout][K] (output channel o reads input o // (Cout/Cin)).
  "<mod>.g"/"<mod>.b"  LayerNorm gamma / beta.
  "<mod>.a"/"<mod>.ia" Snake alpha = clip(exp(log_alpha), 1e-4, 100) and 1/alpha, padded with 1.
  ConvTranspose (stride s, K = 2s) -> s phases of a 2-tap conv, stacked as Cout4*s rows
              (phase r rows first); tap0 = W[:, :, r+s] (reads x[q-1]), tap1 = W[:, :, r] (reads x[q]).
  GRU         "frame_gru.wih.wp/.b": input projection of both directions (504 rows),
              "frame_gru.r" [2][252][84] recurrent, "frame_gru.rb" [2][252].

Also writes spec/v1b8k_layers.json: execution-ordered list of weight-bearing layers with the
ONNX initializer names, shapes, op params, packed names and shape formulas.

Usage: .venv/bin/python tools/export_ptk.py [--model-dir model/v1b_8k] [--out build/v1b8k.ptk]
"""
import argparse, json, os, struct
import numpy as np
import onnx
from onnx import numpy_helper, helper

MAGIC = 0x50544B31
VERSION = 1
REC = struct.Struct("<64si4x6qqq")
HDR = struct.Struct("<IIIiq")
assert REC.size == 136 and HDR.size == 24


def modpath(node_name):
    """'/encoder.0/ff/ff.0/MatMul' -> 'encoder.0.ff.0' (collapse torch Sequential scopes)."""
    segs = [s for s in node_name.split("/") if s][:-1]
    out = []
    for s in segs:
        if out and s.startswith(out[-1] + "."):
            out[-1] = s
        else:
            out.append(s)
    return ".".join(out)


def r4(n):
    return (n + 3) // 4 * 4


def pack_conv(W):
    """W [Cout, Cin, K] -> [Cout4/4][K][Cin][4]"""
    Cout, Cin, K = W.shape
    C4 = r4(Cout)
    Wp = np.zeros((C4, Cin, K), np.float32)
    Wp[:Cout] = W
    return np.ascontiguousarray(Wp.reshape(C4 // 4, 4, Cin, K).transpose(0, 3, 2, 1))


def pad_vec(b, n, fill=0.0):
    out = np.full(n, fill, np.float32)
    out[: b.shape[0]] = b
    return out


class Exporter:
    def __init__(self):
        self.tensors = {}  # name -> np.float32 array
        self.layers = []

    def add(self, name, arr):
        assert name not in self.tensors, name
        assert len(name) < 64, name
        self.tensors[name] = np.ascontiguousarray(arr, np.float32)

    def graph(self, tag, path):
        m = onnx.load(path)
        g = m.graph
        inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
        consumers = {}
        for n in g.node:
            for i in n.input:
                consumers.setdefault(i, []).append(n)
        consts = {}
        for n in g.node:
            if n.op_type == "Constant":
                consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)

        def bias_after(n):
            """MatMul followed by Add(bias_initializer, out) -> bias array + init name"""
            for c in consumers.get(n.output[0], []):
                if c.op_type == "Add":
                    for i in c.input:
                        if i in inits:
                            return inits[i], i
            return None, None

        def fused_after(n):
            """describe the elementwise epilogue pattern following a node (documentation)"""
            out = n.output[0]
            cs = consumers.get(out, [])
            if n.op_type == "MatMul":
                b = [c for c in cs if c.op_type == "Add"]
                if b:
                    out = b[0].output[0]
                    cs = consumers.get(out, [])
            ops = sorted(c.op_type for c in cs)
            if ops == ["Mul", "Sigmoid"]:
                return "silu"
            if ops == ["Tanh"]:
                return "tanh"
            if "Mul" in ops and any(c.op_type == "Mul" and any("acts" in x or "act" in x for x in c.input) for c in cs):
                return "snake(next)"
            if any(c.op_type == "Mul" and "Clip" in "".join(c.input) for c in cs):
                return "snake(next)"
            return "none"

        for n in g.node:
            a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
            mod = modpath(n.name)
            if n.op_type == "Conv":
                W = inits[n.input[1]]
                B = inits[n.input[2]] if len(n.input) > 2 and n.input[2] in inits else np.zeros(W.shape[0], np.float32)
                grp = a.get("group", 1)
                K = W.shape[2]
                pad = a["pads"][0]
                dil = a["dilations"][0]
                entry = dict(graph=tag, module=mod, op="conv1d", onnx=[n.input[1]] + ([n.input[2]] if len(n.input) > 2 else []),
                             W_shape=list(W.shape), group=grp, kernel=K, pad=pad, dilation=dil, stride=a["strides"][0],
                             epilogue=fused_after(n))
                if grp == 1:
                    self.add(f"{tag}.{mod}.wp", pack_conv(W))
                    self.add(f"{tag}.{mod}.b", pad_vec(B, r4(W.shape[0])))
                    entry.update(packed=[f"{tag}.{mod}.wp", f"{tag}.{mod}.b"], packed_shape=list(self.tensors[f"{tag}.{mod}.wp"].shape),
                                 shape="[Cin][T] -> [Cout][T + 2*pad - dil*(K-1)]")
                elif W.shape[1] == 1:  # depthwise (with channel multiplier)
                    self.add(f"{tag}.{mod}.dw", W[:, 0, :])
                    self.add(f"{tag}.{mod}.b", B)
                    entry.update(op="dwconv1d", multiplier=W.shape[0] // grp, packed=[f"{tag}.{mod}.dw", f"{tag}.{mod}.b"],
                                 shape="[Cin][T] -> [Cin*mult][T]; out o reads in o//mult")
                else:  # grouped conv: one packed conv per group
                    og = W.shape[0] // grp
                    names = []
                    for gi in range(grp):
                        self.add(f"{tag}.{mod}.g{gi}.wp", pack_conv(W[gi * og:(gi + 1) * og]))
                        self.add(f"{tag}.{mod}.g{gi}.b", pad_vec(B[gi * og:(gi + 1) * og], r4(og)))
                        names += [f"{tag}.{mod}.g{gi}.wp", f"{tag}.{mod}.g{gi}.b"]
                    entry.update(op="grouped_conv1d", packed=names, shape=f"{grp} groups of [{W.shape[1]}][T] -> [{og}][T]")
                self.layers.append(entry)
            elif n.op_type == "ConvTranspose":
                W = inits[n.input[1]]  # [Cin, Cout, K]
                B = inits[n.input[2]]
                s = a["strides"][0]
                K = W.shape[2]
                pad = a["pads"][0]
                assert K == 2 * s and pad * 2 == s, (K, s, pad)
                Cin, Cout, _ = W.shape
                C4 = r4(Cout)
                Wph = np.zeros((s * C4, Cin, 2), np.float32)
                Bph = np.zeros(s * C4, np.float32)
                for r in range(s):
                    Wph[r * C4:r * C4 + Cout, :, 0] = W[:, :, r + s].T  # reads x[q-1]
                    Wph[r * C4:r * C4 + Cout, :, 1] = W[:, :, r].T      # reads x[q]
                    Bph[r * C4:r * C4 + Cout] = B
                self.add(f"{tag}.{mod}.wp", pack_conv(Wph))
                self.add(f"{tag}.{mod}.b", Bph)
                self.layers.append(dict(graph=tag, module=mod, op="conv_transpose1d", onnx=[n.input[1], n.input[2]],
                                        W_shape=list(W.shape), stride=s, kernel=K, pad=pad,
                                        packed=[f"{tag}.{mod}.wp", f"{tag}.{mod}.b"], packed_shape=list(self.tensors[f"{tag}.{mod}.wp"].shape),
                                        impl="s phases x 2-tap conv (pad 1) over q in [0,Tin], then out[q*s+r-pad] = P_r[q]",
                                        shape="[Cin][T] -> [Cout][T*stride]"))
            elif n.op_type in ("MatMul", "Gemm"):
                wn = [i for i in n.input if i in inits]
                if not wn:
                    continue
                M = inits[wn[0]]
                if n.op_type == "Gemm":
                    assert a.get("transB", 0) == 1
                    W = M  # [Cout, Cin]
                    B = inits[n.input[2]]
                    bn = n.input[2]
                else:
                    W = M.T  # [Cout, Cin]
                    B, bn = bias_after(n)
                    if B is None:
                        B = np.zeros(W.shape[0], np.float32)
                if W.shape[1] == 1:  # 1 -> C projection (energy/bright): plain vectors
                    self.add(f"{tag}.{mod}.w", W[:, 0])
                    self.add(f"{tag}.{mod}.b", B)
                    packed = [f"{tag}.{mod}.w", f"{tag}.{mod}.b"]
                else:
                    self.add(f"{tag}.{mod}.wp", pack_conv(W[:, :, None]))
                    self.add(f"{tag}.{mod}.b", pad_vec(B, r4(W.shape[0])))
                    packed = [f"{tag}.{mod}.wp", f"{tag}.{mod}.b"]
                self.layers.append(dict(graph=tag, module=mod, op="linear", onnx=[wn[0]] + ([bn] if bn else []),
                                        W_shape=[int(W.shape[1]), int(W.shape[0])], packed=packed, epilogue=fused_after(n),
                                        shape="[Cin][T] -> [Cout][T] (K=1 conv)"))
            elif n.op_type == "LayerNormalization":
                self.add(f"{tag}.{mod}.g", inits[n.input[1]])
                self.add(f"{tag}.{mod}.beta", inits[n.input[2]])
                self.layers.append(dict(graph=tag, module=mod, op="layernorm", onnx=[n.input[1], n.input[2]],
                                        C=int(inits[n.input[1]].shape[0]), eps=float(a["epsilon"]),
                                        packed=[f"{tag}.{mod}.g", f"{tag}.{mod}.beta"], shape="over channels, per time column"))
            elif n.op_type == "Gather" and n.input[0] in inits:
                E = inits[n.input[0]]
                self.add(f"{tag}.{mod}.emb", E)
                self.layers.append(dict(graph=tag, module=mod, op="embedding", onnx=[n.input[0]], shape_table=list(E.shape),
                                        packed=[f"{tag}.{mod}.emb"]))
            elif n.op_type == "Exp" and n.input[0] in inits and n.input[0].endswith("log_alpha"):
                la = inits[n.input[0]].reshape(-1)
                alpha = np.clip(np.exp(la.astype(np.float32)), np.float32(1e-4), np.float32(100.0)).astype(np.float32)
                C4 = r4(la.shape[0])
                mn = n.input[0][: -len(".log_alpha")]
                self.add(f"{tag}.{mn}.a", pad_vec(alpha, C4, 1.0))
                self.add(f"{tag}.{mn}.ia", pad_vec((np.float32(1.0) / alpha).astype(np.float32), C4, 1.0))
                self.layers.append(dict(graph=tag, module=mn, op="snake", onnx=[n.input[0]], C=int(la.shape[0]),
                                        formula="x + sin(a*x)^2 / a, a = clip(exp(log_alpha), 1e-4, 100)",
                                        packed=[f"{tag}.{mn}.a", f"{tag}.{mn}.ia"]))
            elif n.op_type == "GRU":
                W = inits[n.input[1]]  # [2, 3H, In]
                R = inits[n.input[2]]  # [2, 3H, H]
                B = inits[n.input[3]]  # [2, 6H]
                H = a["hidden_size"]
                assert a.get("linear_before_reset", 0) == 1
                Wcat = np.concatenate([W[0], W[1]], 0)  # [6H, In]
                self.add(f"{tag}.frame_gru.wih.wp", pack_conv(Wcat[:, :, None]))
                self.add(f"{tag}.frame_gru.wih.b", np.concatenate([B[0, :3 * H], B[1, :3 * H]]))
                self.add(f"{tag}.frame_gru.r", R)
                self.add(f"{tag}.frame_gru.rb", np.stack([B[0, 3 * H:], B[1, 3 * H:]]))
                self.layers.append(dict(graph=tag, module="frame_gru", op="bigru", onnx=list(n.input[1:4]), hidden=H,
                                        gate_order="z,r,h (ONNX)", linear_before_reset=1,
                                        packed=[f"{tag}.frame_gru.wih.wp", f"{tag}.frame_gru.wih.b", f"{tag}.frame_gru.r", f"{tag}.frame_gru.rb"],
                                        shape="[168][F] -> [2*84][F] (fwd | bwd)"))
            elif n.op_type == "Add" and n.output[0] == "mel" or (n.op_type == "Mul" and any(i in consts for i in n.input) and "postnet" in "".join(n.input)):
                pass

    def write(self, path):
        names = list(self.tensors)
        recs, off = [], 0
        for nm in names:
            a = self.tensors[nm]
            dims = list(a.shape) + [0] * (6 - a.ndim)
            recs.append(REC.pack(nm.encode(), a.ndim, *dims, a.size, off))
            off += (a.nbytes + 63) // 64 * 64
        hd = HDR.size + REC.size * len(names)
        hdp = (hd + 63) // 64 * 64
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(HDR.pack(MAGIC, VERSION, len(names), 0, off))
            for r in recs:
                f.write(r)
            f.write(b"\0" * (hdp - hd))
            for nm in names:
                b = self.tensors[nm].tobytes()
                f.write(b)
                f.write(b"\0" * ((len(b) + 63) // 64 * 64 - len(b)))
        return len(names), off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model/v1b_8k")
    ap.add_argument("--out", default="build/v1b8k.ptk")
    ap.add_argument("--layers-json", default="spec/v1b8k_layers.json")
    args = ap.parse_args()
    ex = Exporter()
    ex.graph("enc", f"{args.model_dir}/acoustic_encoder.onnx")
    ex.graph("dec", f"{args.model_dir}/acoustic_decoder.onnx")
    ex.graph("voc", f"{args.model_dir}/vocoder.onnx")
    # Vocoder geometry as data so the runtime is not locked to one variant: per stage
    # [stage_channels (ConvT Cin, = resblock channels), stride, pad, kernel].
    stages = sorted((l for l in ex.layers if l["graph"] == "voc" and l["module"].startswith("ups.")),
                    key=lambda l: int(l["module"].split(".")[1]))
    assert len(stages) == 4, stages
    ex.add("voc.geom", np.array([[l["W_shape"][1], l["stride"], l["pad"], l["kernel"]] for l in stages],
                                np.float32))
    n, nb = ex.write(args.out)
    meta = json.load(open(f"{args.model_dir}/meta.json"))
    nparam = sum(t.size for t in ex.tensors.values())
    doc = {
        "model": f"PrimeTTS {os.path.basename(args.model_dir)}",
        "container": os.path.basename(args.out),
        "meta": meta,
        "notes": [
            "All activations are channel-major [C][T] (time contiguous); ONNX MatMuls over [T][C] become K=1 convs.",
            "Cout is zero-padded to a multiple of 4 everywhere; padded channels stay exactly 0 through every op.",
            "frame_mask / token mask are all-ones in this pipeline (host_regulate: np.ones), so the mask multiplies are identities.",
            "Encoder/decoder GLU dwconv: groups=168, 336 outputs: out o reads in o//2; glu = out[:168] * sigmoid(out[168:]).",
            "Vocoder stage output = (rb0 + rb1 + rb2) / 3, each rb = 3 x [x += conv2(snake(conv1(snake(x))))].",
        ],
        "layers": ex.layers,
    }
    os.makedirs(os.path.dirname(args.layers_json) or ".", exist_ok=True)
    json.dump(doc, open(args.layers_json, "w"), indent=1)
    print(f"wrote {args.out}: {n} tensors, {nb/1e6:.2f} MB payload ({nparam/1e6:.3f} M floats incl. padding); "
          f"{len(ex.layers)} layers -> {args.layers_json}")


if __name__ == "__main__":
    main()
