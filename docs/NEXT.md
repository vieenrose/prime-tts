# Next-session handoff (2026-10-22)

Goal (operator's words): **PrimeTTS v2.1 (`v21_mbistft_16k`, 16 kHz) on Jetson Nano in a
dependency-free C + aarch64 asm + CUDA C runtime, lowest latency, streaming input AND output,
4 parallel real-time requests.** "Forget gemini" — the Gemini cost work is finished in `docs/cost.md`.

## Kickoff prompt to paste

> Continue per docs/NEXT.md. Verify the listed artifacts on disk before believing any prior report.
> Work inline (not in subagent lanes) on task 1, the gate_packing fix in tools/export_v21.py.

## Rules that must survive into the next session

- `.venv/bin/python` for all host Python (system `python3` is broken: protobuf). onnx 1.23
  (`node.op_type`, `numpy_helper.to_array`); onnxruntime needs `mm.ir_version = model.ir_version`.
- Nano: `ssh picard@100.84.151.116` (key auth, Tailscale IP). Repo copy `~/primetts`
  (`rsync -a --delete --exclude .git ./ picard@100.84.151.116:~/primetts/`). No sudo, no apt,
  CUDA at `/usr/local/cuda/bin/nvcc` (not on PATH), sm_53, gcc 8.4, python 3.6.
- Nano memory guard: check `free -m` + `tegrastats` (`lfb`) before every sweep; abort below 60 MB
  free RAM. GPU memory lives in **NvMap** and counts as system memory, invisible in RSS —
  check `grep NvMapMemFree /proc/meminfo` and `/dev/nvmap` fd holders, not RSS.
- Do **not** kill/restart `fwd35_serve` (operator's `voicechat-jetson` LLM server) without asking.
  It was the only `/dev/nvmap` holder; it is currently stopped, GPU free.
- Never print, echo, log or copy `gemini-live-credential.txt`.
- v2.1 is stochastic: **every parity/comparison uses `noise_scale=0`.**
- Never run `git` inside a subagent lane; lanes must not spawn lanes.
- **Believe no lane report.** Three lanes died this session (`saving a checkpoint part failed`,
  `engine stopped unexpectedly (exit code 1)`); one claimed M1 complete and its artifacts did not
  exist. Only commit things verified with `ls`/`wc`/own runs. Prefer inline work in short steps;
  if you must delegate, use one short lane (<20 min, host-only, explicit stop condition).

## Current state on disk (verified by me)

| artifact | state |
|---|---|
| `tools/export_v21.py` | 1175 lines, runs; gate (e) folding green; crashes in `gate_packing` |
| `tests/golden_v21/_probe/` | 4071 tensors (probe dump) — exists |
| `build/v21.ptk`, `spec/v21_layers.json` | **do not exist** (M1 not finished) |
| `tests/gen_golden_v21.py`, `tests/v21_reference.py`, `docs/v21_layers.md` | **do not exist** |
| `tools/export_ptk.py` + `build/v1b8k.ptk`, `build/v1b16k.ptk` | v1b tier, works, parity green |
| CPU runtime + `bench/cpu/RESULTS16.md`, `bench/cpu/RESULTS_ISSUE.md` | works, measured |
| `bench/gpu/{sgemm,hgemm}.cu` + `bench/gpu/RESULTS.md` | measured rates (fp32 62, half2 121–171 GF/s) |

Commits (newest first): `eca8285` folding, `7c5c2bc` Nano NvMap facts, `172c738` A57 issue probe,
`3452f5d` A57 probe + partial exporter, `6a01941` M1 (message contains a **false** lane report),
`2ed75a3` 16 kHz CPU results + graph trace, `68e6d30` target=4 streams, `eb44991` v2.1 feasibility.

## Task 1 — fix `gate_packing` (start here)

Run: `.venv/bin/python tools/export_v21.py` (defaults: `--model model/v21_mbistft_16k/primetts_v21_3voice.onnx
--out build/v21.ptk --layers-json spec/v21_layers.json --probe-dir tests/golden_v21/_probe --hilo none`).

Current crash: `ValueError: operands could not be broadcast together with shapes (192,6) (768,1)` in
`conv1d_ref` (line ~167), called from `gate_packing`'s conv loop (line ~934).

Diagnosis (measured): `ex.conv` entries individually have `W.shape[0] == Cout` and `len(B) == Cout`
(checked with an instrumented `Exporter.run()`: 0 mismatches). The failing shape pair (W rows 192 vs
Cout/bias 768) is exactly a **ConvTranspose-layout tensor `[Cin, Cout, K]`** — so the conv gate loop
is iterating an entry whose `W` is convT-layout, or a 4-D **packed** tensor fed to the 3-D reference.
Fix by checking, in the conv loop, `W.ndim == 3 and W.shape[0] == c["Cout"]`, printing the offending
entry name + shapes when it fails (do not silently skip), and confirming which list the entry came
from (`conv_entry` appends at the `self.conv.append(...)` site, `convT_entry` at `self.convT.append`).
Keep the gate strict: it must exit non-zero on real packing errors.

Then, in order:
1. `gate_coverage` (every weight tensor feeding Conv/ConvTranspose/MatMul/Gemm has a packed entry) and
   gate (c) budget (container payload within 2% of predicted dtype bytes) → must print and pass.
2. Produce `build/v21.ptk` + `spec/v21_layers.json`. Expected payload ~116 MB fp32
   (29,080,457 params, parent-verified; Constant-node floats add only 0.003M params).
3. `tests/v21_container_smoke.py`: read the container back, re-derive shapes, assert every packed
   conv/convT weight **unpacks to the ONNX value within 1e-6 elementwise max** (gate d).
4. Regenerate the probe dump if needed:
   `.venv/bin/python tools/probe_shapes.py model/v21_mbistft_16k/primetts_v21_3voice.onnx \
      tests/golden_v21/_probe.json --frames 81 --dump-dir tests/golden_v21/_probe`
5. `tests/gen_golden_v21.py` → `tests/golden_v21/` (T=20/40/81/120, `noise_scale=0`, phone/tone/lang
   ids from `bench/cpu/requests.txt`; the zh-TW g2p/g2pw front-end is NOT ported — say so in docs).
6. `tests/v21_reference.py`: pure-numpy oracle over the container (the C runtime is written against it).
   Gate: per-layer rel err < 1e-3 vs onnxruntime, end-to-end SNDR ≥ 60 dB.
7. CUDA single stream (M2), then 4-stream batching + CUDA Graphs (M3), streaming pipeline (M4),
   admission/latency policy (M5). Gates and dtype policy are in `docs/v21_plan.md`.

## Facts to keep using (all measured; canonical lists live elsewhere — do not re-derive)

Hardware + GEMM/issue rates: `bench/gpu/RESULTS.md`, `bench/cpu/RESULTS_ISSUE.md`. CPU concurrency
and RSS: `bench/cpu/RESULTS16.md`, `bench/cpu/RESULTS.md`. fp16 sensitivity: `spec/v21_fp16_probe.json`
via `tools/fp16_probe.py`. Nano memory/GPU-region state and dtype policy: `docs/v21_plan.md` §6.
Cost/modality facts: `docs/cost.md`. The four numbers that drive every decision: half2 GEMM
121-171 GF/s, fp32 GEMM 62 GF/s, A57 9-10 GFLOP/s/core, v2.1 11.9 GFLOP per audio-second.

## Open items (not blockers)

- Sustained (60–120 s) half2 rate at real clocks — a 921 MHz burst number is not a promise.
- Chunked-K promotion rate (accumulate 16–32 K in half2, promote to fp32): target ≥130 GF/s.
- v1b_8k single-stream regression 0.186 → 0.225 (stride-8 recovery parked, non-blocking).
- zh-TW front-end (g2pw / g2p_en / entity norm) is unported — the latency/cost story is complete,
  the text front-end is not.
