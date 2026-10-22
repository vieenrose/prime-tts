// v2.1 wave 4: finish the exporter (it is 624 lines and ends mid-class) + measure the fp16
// sensitivity/footprint Pareto that the real 139 MB fp32 footprint forces.
const ctx = `
PROJECT: dependency-free PrimeTTS v2.1 runtime on Jetson Nano, **4 parallel real-time requests**
at 16 kHz, lowest latency, streaming in/out. No onnxruntime/ggml/cuBLAS at runtime - our own C +
aarch64 asm + CUDA C.
cwd: /home/louis/gemini-live-with-local-tts-jetson-nano. Host python: ALWAYS .venv/bin/python
(3.11, onnx 1.23, onnxruntime 1.28, numpy 2.4); system python3 is broken. onnx 1.23: node.op_type,
weights via onnx.numpy_helper.
Nano (only if needed, short): ssh picard@100.84.151.116, no sudo, CUDA 10.2 at
/usr/local/cuda/bin/nvcc, sm_53, ~165 MB free RAM, 1.2 GB swap in use - NEVER allocate >120 MB there.
HARD RULES: never run git. Never touch model/, .venv/, ref/, workflows/, server/, .gitignore, bench/,
runtime/. Report measured numbers only, and never report an artifact you did not verify exists on disk.
Time-box: a verified, gated deliverable beats exhaustive exploration.
`

const finish = {
  key: 'v21-export', agent: 'worker', label: 'Finish tools/export_v21.py and produce the container',
  task: `${ctx}
CONTEXT: tools/export_v21.py exists (624 lines) and **ends mid-class** - the previous lane died
while writing it. It contains align64/r4/pack_conv/unpack_conv/convT_ntap/pack_convtranspose/
unpack_convtranspose/split_hilo/as_fp16_values, conv1d_ref, conv_transpose1d_ref,
conv_transpose1d_packed, write_ptk, read_ptk, group_of, module_of, attrs_of, and class Exporter with
__init__/add/dynamic_weight/epilogue/kind_of/run/export_constant/conv_entry/convT_entry/matmul_entry/
layernorm_entry/embedding_entry/pad_table_entry/pad_const_entry/const_pads/model_meta. It has NO
main(), no CLI, no gate invocation, no budget printing. Do not rewrite this code - complete it.

MODEL: model/v21_mbistft_16k/primetts_v21_3voice.onnx (4066 nodes, 1908 weight nodes, 1924 Constant
nodes; **parent-verified**: 29.080M initializer params = 116.32 MB fp32, Constant-node floats add
0.003M, so the fp32 container is 116 MB + packing padding). ref/v21_graph_trace.txt is the
execution-ordered trace; docs/v21_plan.md and bench/gpu/RESULTS.md hold the measured rates.
**Parent-verified** fp16 SNDR (host graph surgery, noise_scale=0, tools/fp16_probe.py, T=40):
decoder conv weights fp16 = 67.4 dB; encoder conv weights fp16 = 34.3 dB; all conv/matmul weights =
32.2 dB; every initializer = 22.4 dB; fp16 conv-output activations = 87.9 dB. A previous lane
reported 42.8/38.2/56.8 dB from a numpy reimplementation whose files were lost - treat those as
unverified; the Pareto lane is re-measuring them. Nano has ~165 MB free RAM, so footprint binds.

DO, in this order:
1. Complete the file: main() with argparse (--model, --out build/v21.ptk, --layers-json
   spec/v21_layers.json, --hilo, --probe-dir, --budget-out), invoking the existing gates and
   printing the dtype budget table (bytes per group, fp32 vs fp16 variants, total) and coverage
   (weight nodes covered / 1908).
2. Gates must FAIL the run (non-zero exit, print the offending tensor/layer) if: (a) unpacking every
   packed weight (HiLo sum where applicable) differs from the ONNX tensor by >1e-6 elementwise max;
   (b) any weight node feeding Conv/ConvTranspose/MatMul/Gemm has no packed entry; (c) container
   bytes deviate from the predicted dtype budget by >2%.
3. Run it: produce build/v21.ptk + spec/v21_layers.json for the default policy (fp32 encoder, fp16
   decoder conv weights, fp32 elsewhere) and print gate output verbatim + total MB.
4. Sanity: write tests/v21_container_smoke.py that opens the container with runtime/py/pt.py-style
   reading extended for the fp16 ('#h') width convention the exporter uses, checks tensor count,
   64-byte alignment, and recomputes two conv layers (one fp32, one fp16) with conv1d_ref /
   conv_transpose1d_packed against the ONNX weights, printing rel err.
5. Verify the container + smoke test exist on disk (ls) before reporting; report the numbers you
   measured. Do not spawn lanes.` }

const pareto = {
  key: 'v21-dtype', agent: 'worker', label: 'fp16 sensitivity/footprint Pareto for v2.1',
  task: `${ctx}
PROBLEM: v2.1 in fp32 needs 116 MB of weights (parent-verified: 29.080M params x 4 B; Nano has
~165 MB free RAM, 1.2 GB swap already in use) and 4 streams need ~50 GFLOP/s (measured Nano fp32
GEMM 62 GF/s, half2 121-171 GF/s). fp16 halves both, but quality is uneven - parent-verified
(host graph surgery, noise_scale=0, T=40, tools/fp16_probe.py): decoder conv weights fp16 = 67.4 dB,
encoder conv weights fp16 = 34.3 dB, all conv/matmul = 32.2 dB, every initializer = 22.4 dB, fp16
conv-output activations = 87.9 dB. A lost numpy reimplementation reported 42.8/38.2/56.8 dB, so the
absolute numbers are method-dependent: **re-measure everything fresh in one consistent harness**.
We need the **Pareto curve**: which module groups may go fp16, at what quality, and
how many MB each choice saves.

BUILD tools/fp16_groups.py (new file) using the techniques already in tools/fp16_probe.py (read it;
reuse its run()/report() conventions; keep noise_scale=0; when you insert Cast chains, run with
graph optimization disabled or the optimizer folds Cast pairs away and measures nothing; set
mm.ir_version = model.ir_version when rebuilding a graph, onnxruntime here rejects IR 14).

MEASURE, one group at a time (only that group's conv/matmul weights rounded to fp16 values,
everything else fp32), reporting relL2 + SNDR + MB saved:
 - each decoder resblock group: dec/resblocks.0..2 (per index, not lumped)
 - decoder upsample ConvTranspose stages (per stage)
 - decoder pre/post convs
 - flow layers (flows.0/1/2 separately)
 - encoder self-attention blocks (per block if time allows, else first/last block + all-blocks)
 - encoder feed-forward/conv blocks
 - iSTFT head (gen_istft_* / mrf matrices)
 - speaker/conditioning paths (gin, speaker_proj)
Then measure the two combined policies you predict to be Pareto-optimal, e.g.
 (A) dec resblocks+upsample fp16, enc/flow/head fp32;
 (B) dec+flow fp16, enc+head fp32;
and report their SNDR and total MB.

TARGET to recommend: SNDR >= 45 dB with total weight bytes <= 100 MB. Report the achieved
best (SNDR, MB) pairs and state explicitly if the target is not reachable with per-group fp16
alone - and if not, what it costs (which groups must stay fp32, and what the split-fp16 HiLo
alternative would do to bytes and time: HiLo pairs cost the same bytes as fp32 and half the
half2 rate, i.e. ~80 GF/s effective).

Write spec/v21_fp16_groups.json (raw measurements) and docs/v21_dtype_policy.md (the table, the
recommended policy, the arithmetic: bytes and GFLOP/s implications at 4 streams using the measured
rates in bench/gpu/RESULTS.md). Do NOT modify tools/export_v21.py, docs/v21_plan.md, tools/
fp16_probe.py, tests/, build/, runtime/. Do not spawn lanes. Verify files exist on disk before
reporting.` }

const res = await runs.all([finish, pareto])
return res.map(r => ({ runId: r.runId, ok: r.ok, out: (r.output || '').slice(0, 4000) }))
