# Next-session handoff (2026-10-22)

Goal (operator's words): **PrimeTTS v2.1 (`v21_mbistft_16k`, 16 kHz) on Jetson Nano in a
dependency-free C + aarch64 asm + CUDA C runtime, lowest latency, streaming input AND output,
4 parallel real-time requests.** "Forget gemini" — the Gemini cost work is finished in `docs/cost.md`.

## Kickoff prompt to paste

> Continue per docs/NEXT.md. Verify the listed artifacts on disk before believing any prior report.
> M1 is done and gated (see the measured table). Work inline (not in subagent lanes) on task 2:
> the M2 CUDA C single-stream runtime, written against tests/v21_reference.py.

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

## Current state on disk (re-verified 2026-10-22 with ls/wc and own runs)

| artifact | state |
|---|---|
| `tools/export_v21.py` | 1317 lines, runs in 2.6 s, exit 0; gates (a)(b)(c)(e) print and pass |
| `build/v21.ptk` | **exists**: 339 tensors, 89,077,824 B payload, 29,084,706 elements |
| `spec/v21_layers.json` | **exists**: 4066 layers + 4066 nodes + constants + graph_inputs/outputs (4.01 MB) |
| `tests/golden_v21/_probe/` | 4071 tensors (random-id probe, T=81) - exists |
| `tests/golden_v21/_probe_T81/` | 4071 tensors for the **golden** T=81 ids (248 MB, gitignored) - exists |
| `tests/gen_golden_v21.py` + `tests/golden_v21/{T20,T40,T81,T120,sid1_T29}/` | **exists**: inputs + wav + 99 intermediates per case, MANIFEST.json |
| `tests/v21_container_smoke.py` | **exists**: gate (d); 159 entries, max|unpacked-ONNX| = 0 |
| `tests/v21_reference.py` | **exists**: pure-numpy oracle over the container; all gates green |
| `docs/v21_layers.md` | **exists**: layouts, group/byte table, gates + measured numbers, front-end gap |
| `tools/export_ptk.py` + `build/v1b8k.ptk`, `build/v1b16k.ptk` | v1b tier, works, parity green |
| CPU runtime + `bench/cpu/RESULTS16.md`, `bench/cpu/RESULTS_ISSUE.md` | works, measured |
| `bench/gpu/{sgemm,hgemm}.cu` + `bench/gpu/RESULTS.md` | measured rates (fp32 62, half2 121-171 GF/s) |

Commits (newest first): `987fc66` numpy oracle (M1 step 6), `4b4ef5f` v2.1 goldens, `045ac89` gate (d)
smoke test, `b75277a` gate_packing fix + container/spec on disk, `214b9a8` this handoff, `eca8285`
folding, `7c5c2bc` Nano NvMap facts, `172c738` A57 issue probe, `3452f5d` A57 probe, `6a01941` M1
(message contains a **false** lane report - that work did not exist).

## Task 1 (gate_packing) - DONE. The diagnosis written in the previous version of this file was wrong

The crash was never a packing bug; it was in the gate's own reference helpers (each fix verified
against nested-loop ONNX implementations):
  * `conv1d_ref` emitted the **Cin** axis (`"ock,ckt->ct"` sums Cout away) and reshaped the t-major
    patch index as `(Cin,K,Tout)`. For every Cin==Cout conv (most of the encoder) the gate compared
    two identical broken values, so it validated nothing; it only crashed at the first Cin!=Cout conv
    (ffn conv_1, W=(768,192,3), B=(768)): einsum (192,8) + bias (768,1). Now `patches[c,t,k]`
    (reshape `(Cin,Tout,K)`) with `"ock,ctk->ot"`.
  * `conv_transpose1d_ref` built its bias with `.repeat()`, giving shape (1, Cout*Tout); only Cout==1
    entries survived. Now `tile(bias[:, None])`.
  * `conv_transpose1d_packed` added the bias as a **row** and ran the delay line only `Tin` steps, so
    it dropped the tail outputs every K>s stride-4 convT produces (tap j needs q up to Tin-1+j).
    Now bias as a column and `Q = Tin + nt - 1` (the tail flush).
  * gate (b): the 12 attention `emb_rel_k/v` tables are Pad **data** inputs with graph-computed pads
    (all 51 v2.1 Pads have dynamic pads; measured `[0,76,0,0,76,0]` -> (1,161,96) at T=81), so
    `pad_weight_entry` now exports them (op `pad_relative_position`, `pads_source` recorded).
  * the gate compared the packed path against the SAME function, so a stride/pad/tap mapping error
    cancelled out. It now compares against an independent reference (`conv1d_brute`: one matmul per
    kernel tap with its own index math; `conv_transpose1d_brute`: scatter (q,k) products to
    t=q*s+k-pad) at tol 1e-5 x out-scale, and `_gate_layout` fails loudly naming the entry, the list
    (ex.conv / ex.convT) and every shape involved.

Proven strict by 8 negative trials, all exit non-zero: convT-layout W in ex.conv; packed 4-D W in
ex.conv; conv-layout W in ex.convT; convT map `k=r*s+j`; flipped `pack_conv` transpose; flipped tap
delay sign; delay line not flushed; conv bias dropped.

## M1 is complete and gated (measured on this tree)

```
gate (a) packing    124 conv + 5 convT: max|packed-ONNX_brute| = 0 (conv), 3.4e-07 (convT)
gate (a) round-trip 159 entries: max|W-rebuilt| = 0, max|bias-b| = 0
gate (b) coverage   294/294 initializers, 1924/1924 Constant tensors
gate (c) budget     89,077,824 B vs 89,197,056 predicted = -0.13 % (tol 2 %)
gate (d) smoke      tests/v21_container_smoke.py: 159 entries, max|unpacked-ONNX| = 0
gate (e) folding    4 graph-computed weights, max|fold-ORT| = 0
oracle node gate    tests/v21_reference.py --case T81: 4066 nodes compared, 0 over gate
                    worst rel: enc 3.3e-06  flow 8.5e-06  dp 1.3e-05  dec(fp16 w) 1.9e-03
oracle audio gate   SNDR 69.66 / 69.91 / 70.09 / 67.92 / 67.40 dB (T20/T40/T81/T120/sid1_T29)
```

The oracle runs the full graph in 0.19-0.76 s per case (numpy, fp32 math on container values; the 39
decoder conv tensors carry fp16 weight *values*). The dec per-node gate is 2e-3, not 1e-3, because
those weights are fp16 in the container (the measured 67.4 dB configuration). Goldens start from
phoneme ids in `bench/cpu/requests.txt`: the **zh-TW front-end (g2pw/g2p_en/entity norm) is not
ported** - say so in any doc that cites these goldens.

## Task 2 - M2: CUDA C single-stream runtime (next)

Write it against `tests/v21_reference.py` (it is the behavioural spec): same packed layouts, same
convT phase/tap mapping **with the tail flush**, dynamic rel-pos pads (`pads_source`), noise branch
zero at `noise_scale=0`, the 26 dynamic MatMuls as real GEMMs, payload widths from `dtype.table`.
Gate: SNDR >= 60 dB vs `tests/golden_v21/*/wav.npy`, then per-node vs `tests/golden_v21/_probe_T81`.
Nano: `ssh picard@100.84.151.116`, nvcc `/usr/local/cuda/bin/nvcc` (sm_53, gcc 8.4, python 3.6);
`free -m` + `grep NvMapMemFree /proc/meminfo` + `tegrastats` (`lfb`) before every sweep, abort below
60 MB free RAM. Then M3 (4-stream batching + CUDA Graphs), M4 (streaming pipeline), M5 (admission).
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
