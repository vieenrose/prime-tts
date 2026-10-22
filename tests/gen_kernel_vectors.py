#!/usr/bin/env python3
"""numpy reference vectors for every runtime kernel -> tests/kern/<case>/*.npy (float64 math, cast to fp32).
Checked by tests/test_kernels.c (section "numpy vectors"). Shapes are the model's real layer shapes.
Usage: .venv/bin/python tests/gen_kernel_vectors.py"""
import os, json
import numpy as np

OUT = "tests/kern"
rng = np.random.default_rng(2024)


def save(d, **kw):
    os.makedirs(d, exist_ok=True)
    for k, v in kw.items():
        np.save(f"{d}/{k}.npy", np.ascontiguousarray(v))


def silu(x): return x / (1 + np.exp(-x))
def snake(x, a): return x + np.sin(a[:, None] * x) ** 2 / a[:, None]


cases = []
# conv1d: (name, Cout, Cin, K, dil, n, act)  act: 0 none 1 silu 2 tanh 3 snake ; resid flag
for name, Co, Ci, K, d, n, act, res in [
    ("conv_dec_point", 168, 168, 1, 1, 97, 0, 0), ("conv_dec_ff0", 504, 168, 1, 1, 163, 1, 0),
    ("conv_dec_ff3_res", 168, 504, 1, 1, 163, 0, 1), ("conv_refiner0", 96, 178, 5, 1, 141, 1, 0),
    ("conv_postnet0", 168, 80, 5, 1, 97, 2, 0), ("conv_voc_s0_k11d5", 48, 48, 11, 5, 517, 3, 0),
    ("conv_voc_s0_k7d1_res", 48, 48, 7, 1, 300, 0, 1), ("conv_voc_s1_k7d3", 24, 24, 7, 3, 1100, 3, 0),
    ("conv_voc_s2_k11d1", 12, 12, 11, 1, 2050, 3, 0), ("conv_voc_s3_k3d5_res", 6, 6, 3, 5, 4099, 0, 1),
    ("conv_post", 1, 6, 7, 1, 4096, 2, 0)]:
    span = (K - 1) * d
    X = rng.standard_normal((Ci, n + span)).astype(np.float32)
    W = (rng.standard_normal((Co, Ci, K)) / np.sqrt(Ci * K)).astype(np.float32)
    b = (0.1 * rng.standard_normal(Co)).astype(np.float32)
    a = (0.3 + rng.random(Co)).astype(np.float32)
    R = rng.standard_normal((Co, n)).astype(np.float32)
    Xd, Wd = X.astype(np.float64), W.astype(np.float64)
    Y = np.zeros((Co, n)) + b[:, None]
    for k in range(K):
        Y += Wd[:, :, k] @ Xd[:, k * d:k * d + n]
    if act == 1: Y = silu(Y)
    if act == 2: Y = np.tanh(Y)
    if act == 3: Y = snake(Y, a.astype(np.float64))
    if res: Y = R + Y
    save(f"{OUT}/{name}", X=X, W=W, b=b, a=a, R=R, Y=Y.astype(np.float32),
         p=np.array([Co, Ci, K, d, n, act, res], np.int64))
    cases.append(name)
# elementwise + dw + layernorm
x = (6 * rng.standard_normal(4096)).astype(np.float32)
g = (4 * rng.standard_normal(4096)).astype(np.float32)
xd = x.astype(np.float64)
save(f"{OUT}/act", x=x, g=g, snake=(xd + np.sin(0.83 * xd) ** 2 / 0.83).astype(np.float32),
     silu=silu(xd).astype(np.float32), tanh=np.tanh(xd / 4).astype(np.float32),
     sigmul=(xd / (1 + np.exp(-g.astype(np.float64)))).astype(np.float32))
w = rng.standard_normal(7).astype(np.float32)
xs = rng.standard_normal(4096 + 6).astype(np.float32)
dw = 0.25 + np.array([np.dot(w.astype(np.float64), xs[t:t + 7]) for t in range(4096)])
save(f"{OUT}/dw", x=xs, w=w, y=dw.astype(np.float32))
C, n = 168, 160
xl = (3 * rng.standard_normal((C, n))).astype(np.float32)
rl = rng.standard_normal((C, n)).astype(np.float32)
gl = rng.standard_normal(C).astype(np.float32)
bl = rng.standard_normal(C).astype(np.float32)
xs2 = xl.astype(np.float64) + rl
mu = xs2.mean(0)
var = ((xs2 - mu) ** 2).mean(0)
yl = (xs2 - mu) / np.sqrt(var + 1e-5) * gl[:, None] + bl[:, None]
save(f"{OUT}/ln", x=xl, r=rl, g=gl, b=bl, xsum=xs2.astype(np.float32), y=yl.astype(np.float32))
json.dump({"conv_cases": cases}, open(f"{OUT}/MANIFEST.json", "w"), indent=1)
print("wrote", OUT, len(cases), "conv cases + act/dw/ln")
