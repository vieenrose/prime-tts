#!/usr/bin/env python3
"""Write bench/cpu/requests.txt from the golden test inputs (hand-built zh-TW / en / code-mix ids).
Line format: name sid T  p_1..p_T  t_1..t_T  l_1..l_T"""
import json, os, numpy as np
G = "tests/golden"
names = [c["case"] for c in json.load(open(f"{G}/MANIFEST.json"))["cases"]]
with open("bench/cpu/requests.txt", "w") as f:
    for n in names:
        d = f"{G}/{n}"
        p, t, l = (np.load(f"{d}/{k}.npy")[0] for k in ("phone", "tone", "lang"))
        sid = int(np.load(f"{d}/sid.npy")[0])
        f.write(" ".join([n, str(sid), str(len(p))] + [str(int(x)) for x in np.concatenate([p, t, l])]) + "\n")
print("wrote bench/cpu/requests.txt", names)
