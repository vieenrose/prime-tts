"""Reader for the .ptk weight container produced by tools/export_ptk.py.

Layout (see runtime/src/weights.c for the C side):
  header  <IIIiq   magic=0x50544B31, version, n_tensor, reserved, data_bytes
  records n_tensor x <64si4x6qqq  name, ndim, dim[6], nelem, off   (136 bytes each)
  pad to a 64-byte boundary
  data    fp32 payloads, each 64-byte aligned; `off` is relative to the data start

Used by tools/verify_parity.py and tests/gen_golden.py to make sure the exported container and
the runtime's view of it match, without needing the native build.
"""
from __future__ import annotations

import mmap
import os
import struct

import numpy as np

MAGIC = 0x50544B31
HDR = struct.Struct("<IIIiq")
REC = struct.Struct("<64si4x6qqq")


class PtkContainer:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        self.m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        magic, ver, n, _res, nbytes = HDR.unpack_from(self.m, 0)
        if magic != MAGIC:
            raise ValueError(f"bad magic 0x{magic:x} in {path}")
        self.version, self.n_tensor, self.data_bytes = ver, n, nbytes
        self.data_off = (HDR.size + n * REC.size + 63) // 64 * 64
        self.tensors: dict[str, np.ndarray] = {}
        for i in range(n):
            name, ndim, d0, d1, d2, d3, d4, d5, nelem, off = REC.unpack_from(self.m, HDR.size + i * REC.size)
            name = name.split(b"\0")[0].decode()
            dims = [d for d in (d0, d1, d2, d3, d4, d5)][: max(ndim, 0)]
            arr = np.frombuffer(self.m, dtype="<f4", count=nelem, offset=self.data_off + off)
            self.tensors[name] = arr.reshape(dims) if ndim else arr.reshape(())

    @classmethod
    def open(cls, path: str) -> "PtkContainer":
        return cls(path)

    def get(self, name: str) -> np.ndarray:
        return self.tensors[name]

    def names(self, prefix: str = "") -> list[str]:
        return sorted(n for n in self.tensors if n.startswith(prefix))

    def close(self) -> None:
        self.m.close()
        self.f.close()

    def __enter__(self) -> "PtkContainer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def selfcheck(path: str) -> dict:
    c = PtkContainer.open(path)
    tot = sum(int(t.size) for t in c.tensors.values())
    res = dict(tensors=c.n_tensor, floats=int(tot), bytes=c.data_bytes,
               aligned=all((c.data_off + o) % 64 == 0 for o in
                           [REC.unpack_from(c.m, HDR.size + i * REC.size)[9] for i in range(c.n_tensor)]))
    c.close()
    return res


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(selfcheck(sys.argv[1]), indent=1))
