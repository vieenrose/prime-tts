#!/usr/bin/env python3
"""Generate golden vectors for the v1b_8k C runtime from onnxruntime (x86, fp32).

For each case: inputs (phone/tone/lang ids, sid), encoder outputs, host_regulate outputs,
decoder mel, vocoder wav (from default-optimised ORT sessions, exactly like
model/scripts/synth_from_text.py), plus intermediate tensors taken from an un-optimised
probe session (graph outputs added) for layer-by-layer checks.

Phone ids are hand-built from model/scripts/symbol_table.json + the frontend's rules
(frontend_bopomofo.py: zh syllable -> zhuyin units + tone, empty-rime sibilant -> +ㄭ,
English word -> arpabet + stress digit as tone, then 'SP' with lang EN; punctuation lang ZH).

Usage: .venv/bin/python tests/gen_golden.py [--out tests/golden]
"""
import argparse, json, os
import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort

MD = "model/v1b_8k"
SYM = json.load(open("model/scripts/symbol_table.json"))["symbols"] + ["ㄭ"]  # ㄭ appended (frontend SYLLABIC)
S2I = {s: i for i, s in enumerate(SYM)}
ZH, EN = 0, 1


def zh(*sylls):
    """sylls: ('ㄋㄧ',3) ...  or punctuation string"""
    out = []
    for s in sylls:
        if isinstance(s, str):
            out.append((s, 0, ZH))
            continue
        units, tone = s
        u = list(units)
        if len(u) == 1 and u[0] in "ㄓㄔㄕㄖㄗㄘㄙ":
            u = [u[0], "ㄭ"]
        out += [(x, tone, ZH) for x in u]
    return out


def en(*words):
    out = []
    for w in words:
        for p in w.split():
            st = 0
            if p[-1].isdigit():
                st = int(p[-1]); p = p[:-1]
            out.append((p, st, EN))
        out.append(("SP", 0, EN))
    return out


def ids(seq):
    return (np.array([[S2I[p] for p, _, _ in seq]], np.int64), np.array([[t for _, t, _ in seq]], np.int64),
            np.array([[l for _, _, l in seq]], np.int64))


def cases():
    c = {}
    c["zh_greeting"] = (ids(zh(("ㄋㄧ", 3), ("ㄏㄠ", 3), ",", ("ㄏㄨㄢ", 1), ("ㄧㄥ", 2), ("ㄕ", 3), ("ㄩㄥ", 4),
                               ("ㄩ", 3), ("ㄧㄣ", 1), ("ㄈㄨ", 2), ("ㄨ", 4), ".")), 0)
    c["en_thanks"] = (ids(en("TH AE1 NG K", "Y UW1", "F AO1 R", "K AO1 L IH0 NG") + zh(".")), 0)
    c["mix_booking"] = (ids(zh(("ㄨㄛ", 3), ("ㄧㄠ", 4), ("ㄩ", 4), ("ㄩㄝ", 1)) + [("SP", 0, ZH)] + en("M AH1 N D EY2")
                            + zh(("ㄒㄧㄚ", 4), ("ㄨ", 3), ("ㄙㄢ", 1), ("ㄉㄧㄢ", 3), ".")), 0)
    c["zh_phone_sid1"] = (ids(zh(("ㄑㄧㄥ", 3), ("ㄅㄛ", 1), ("ㄉㄚ", 3), ("ㄌㄧㄥ", 2), ("ㄐㄧㄡ", 3), ("ㄧ", 1), ("ㄦ", 4), ",",
                                 ("ㄙㄢ", 1), ("ㄙ", 4), ("ㄨ", 3), ("ㄌㄧㄡ", 4), ("ㄑㄧ", 1), ("ㄅㄚ", 1), ".")), 1)
    rng = np.random.default_rng(4)
    T = 90
    ph = rng.integers(4, 87, size=(1, T)).astype(np.int64)
    tn = rng.integers(0, 6, size=(1, T)).astype(np.int64)
    lg = (ph >= 41).astype(np.int64) * (ph <= 79)
    c["random_T90"] = ((ph, tn, lg.astype(np.int64)), 0)
    rng = np.random.default_rng(7)
    T = 23
    ph = rng.integers(4, 41, size=(1, T)).astype(np.int64)
    tn = rng.integers(1, 6, size=(1, T)).astype(np.int64)
    c["random_T23"] = ((ph, tn, np.zeros((1, T), np.int64)), 0)
    return c


def host_regulate(cond, dur, pitch, abs_bins, max_frames):
    # verbatim copy of model/scripts/synth_from_text.py::host_regulate
    c = cond[0]; d = dur[0].astype(np.int64); d[d < 0] = 0
    T, H = c.shape
    frames = np.repeat(c, d, axis=0); Fn = frames.shape[0]
    tok = np.repeat(np.arange(T), d); starts = np.cumsum(d) - d
    within = np.arange(Fn) - starts[tok]; dpf = d[tok].astype(np.float32)
    rel = (within / np.maximum(dpf - 1, 1)).astype(np.float32)
    tc = max(1, int((d > 0).sum())); token_pos = (tok / max(1, tc - 1)).astype(np.float32)
    ld = (np.log1p(dpf) / 6.0).astype(np.float32); center = 1.0 - np.abs(rel * 2 - 1)
    fm = np.stack([rel, 1 - rel, center, np.sin(rel*np.pi), np.cos(rel*np.pi), token_pos, ld, dpf/40.0], -1).astype(np.float32)
    prev = np.concatenate([c[:1], c[:-1]], 0); nxt = np.concatenate([c[1:], c[-1:]], 0)
    lc = np.repeat(np.concatenate([prev, c, nxt], -1), d, axis=0).astype(np.float32)
    pos = np.arange(Fn); ap = np.minimum(pos*abs_bins//max(1, max_frames), abs_bins-1).astype(np.int64)
    pf = np.repeat(pitch[0], d, axis=0).astype(np.float32)
    return {"frames": frames[None].astype(np.float32), "frame_meta": fm[None], "local_ctx_raw": lc[None],
            "abs_pos": ap[None], "pitch_frame": pf[None], "frame_mask": np.ones((1, Fn), bool)}


PROBE = {
    "enc": ["/Mul_output_0"] + [f"/encoder.{i}/Mul_3_output_0" for i in range(5)] + ["/Squeeze_output_0"],
    "dec": ["/Add_4_output_0"] + [f"/decoder.{i}/Mul_3_output_0" for i in range(6)]
           + ["/frame_gru/Transpose_2_output_0", "/Transpose_2_output_0", "/Concat_1_output_0"],
    "voc": ["/conv_pre/Conv_output_0", "/ups.0/ConvTranspose_output_0", "/Div_output_0", "/ups.1/ConvTranspose_output_0",
            "/Div_1_output_0", "/ups.2/ConvTranspose_output_0", "/Div_2_output_0", "/ups.3/ConvTranspose_output_0",
            "/Div_3_output_0"],
}


def probe_session(path, names):
    m = onnx.load(path)
    si = onnx.shape_inference.infer_shapes(m)
    types = {v.name: v.type.tensor_type.elem_type for v in si.graph.value_info}
    for n in names:
        m.graph.output.append(helper.make_tensor_value_info(n, types.get(n, onnx.TensorProto.FLOAT), None))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])


def tag(name):
    return name.strip("/").replace("/", "_").replace("_output_0", "")


def main():
    global MD
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/golden")
    ap.add_argument("--model-dir", default=MD)
    args = ap.parse_args()
    MD = args.model_dir
    meta = json.load(open(f"{MD}/meta.json"))
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sA = ort.InferenceSession(f"{MD}/acoustic_encoder.onnx", so, providers=["CPUExecutionProvider"])
    sB = ort.InferenceSession(f"{MD}/acoustic_decoder.onnx", so, providers=["CPUExecutionProvider"])
    sV = ort.InferenceSession(f"{MD}/vocoder.onnx", so, providers=["CPUExecutionProvider"])
    pA = probe_session(f"{MD}/acoustic_encoder.onnx", PROBE["enc"])
    pB = probe_session(f"{MD}/acoustic_decoder.onnx", PROBE["dec"])
    pV = probe_session(f"{MD}/vocoder.onnx", PROBE["voc"])
    bn = ["frames", "frame_meta", "local_ctx_raw", "abs_pos", "pitch_frame", "frame_mask"]
    manifest = []
    for name, ((ph, tn, lg), sid) in cases().items():
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        spk = np.array([sid], np.int64)
        feeds = {"phone": ph, "tone": tn, "lang": lg, "speaker": spk}
        cond, dur, pitch = sA.run(None, feeds)
        reg = host_regulate(cond, dur, pitch, meta["abs_frame_bins"], meta["max_frames"])
        df = {n: (reg[n].astype(np.float32) if reg[n].dtype != bool else reg[n]) for n in bn}
        df["abs_pos"] = reg["abs_pos"].astype(np.int64)
        mel = sB.run(None, df)[0]
        wav = sV.run(None, {"mel": mel.astype(np.float32)})[0]
        save = {"phone": ph, "tone": tn, "lang": lg, "sid": spk, "conditioned": cond, "durations": dur, "pitch": pitch,
                "mel": mel, "wav": wav}
        for k in bn:
            if k != "frame_mask":
                save["reg_" + k] = reg[k]
        # intermediates (un-optimised session) + margin of the duration rounding
        oa = pA.run(None, feeds)
        for n, v in zip(PROBE["enc"], oa[3:]):
            save["enc_" + tag(n)] = v
        ob = pB.run(None, df)
        for n, v in zip(PROBE["dec"], ob[1:]):
            save["dec_" + tag(n)] = v
        ov = pV.run(None, {"mel": mel.astype(np.float32)})
        for n, v in zip(PROBE["voc"], ov[1:]):
            save["voc_" + tag(n)] = v
        x = np.clip(np.exp(save["enc_Squeeze"].astype(np.float32)) - 1, 0, 80)
        margin = float(np.min(np.abs(x - np.floor(x) - 0.5)))
        for k, v in save.items():
            np.save(os.path.join(d, k + ".npy"), np.ascontiguousarray(v))
        F = int(mel.shape[2])
        manifest.append(dict(case=name, T=int(ph.shape[1]), sid=int(sid), frames=F, samples=int(wav.shape[-1]),
                             audio_s=round(wav.shape[-1] / meta["sample_rate"], 3), dur_round_margin=round(margin, 4),
                             probe_vs_opt_wav_maxabs=None))
        print(f"{name:16s} T={ph.shape[1]:3d} F={F:4d} samples={wav.shape[-1]:6d} "
              f"({wav.shape[-1]/8000:.2f}s)  dur-rounding margin={margin:.4f}")
    json.dump(dict(model_dir=MD, cases=manifest, files_per_case=sorted(save.keys())),
              open(os.path.join(args.out, "MANIFEST.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
