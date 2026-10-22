// v2.1 M1 lane, wave 3. Sole writer in the main tree (the parent's CPU lane finished; the
// A57 issue probe is run by the parent itself to keep one writer per cwd).
const ctx = `
PROJECT: dependency-free PrimeTTS runtime on Jetson Nano, now targeting **4 parallel real-time
requests** at 16 kHz with the lowest latency, streaming input and output. No onnxruntime/ggml/
cuBLAS in the runtime - our own C + aarch64 asm + CUDA C.
cwd: /home/louis/gemini-live-with-local-tts-jetson-nano
Local python: ALWAYS .venv/bin/python (3.11, onnx 1.23, onnxruntime 1.28, numpy 2.4). System python3
is broken (protobuf). onnx 1.23: node.op_type, weights via onnx.numpy_helper.
Jetson Nano: ssh picard@100.84.151.116 - 4x Cortex-A57 (NEON fp32 only, no int8 dot-product, no
fp16 arithmetic), MAXN, CUDA 10.2 at /usr/local/cuda/bin/nvcc, sm_53 Tegra X1 921 MHz, gcc 8.4,
Python 3.6, NO sudo, NO apt, ~165 MB free RAM. M1 is host-side; Nano use is optional and must be
short (<=60 s) with \`free -m\` checked first.
HARD RULES: never run git. Never touch model/, .venv/, ref/, workflows/, server/, .gitignore, or
another lane's files. Report measured numbers only. Work in small verified steps: write, run, read.
Time-box: a working, gated Stage 1 beats exhaustive exploration.
`

const v21m1 = {
  key: 'v21-m1', agent: 'worker', label: 'v2.1 export container + goldens + numpy oracle',
  task: `${ctx}
READ FIRST (facts exist; do not re-derive them): docs/v21_plan.md (measured rates, dtype policy,
milestones), bench/gpu/RESULTS.md, ref/v21_graph_trace.txt (4315 lines, execution-ordered v2.1
graph: index, op, name, inputs, outputs, weight, attrs - dumped by a previous run),
tools/export_ptk.py (v1 exporter = your template: packed layouts, hard round-trip gate style),
runtime/py/pt.py (container format), tools/probe_shapes.py, tools/fp16_probe.py.

MODEL: model/v21_mbistft_16k/primetts_v21_3voice.onnx - 4066 nodes, 1908 weight nodes, 1924
Constant nodes, 29.08M initializer params. Inputs x[1,T] int64, tone[1,T], lang[1,T], x_lengths[1],
sid[1], noise_scale[1], length_scale[1]. Output wav @16 kHz. Output is STOCHASTIC when
noise_scale>0: every comparison uses noise_scale=0.

MEASURED DTYPE POLICY (4-stream target): fp32 encoder conv weights (fp16 encoder = 34.3 dB SNDR,
unacceptable), fp16 decoder conv weights (67.4 dB), fp16 activations (87.9 dB - free), fp32 for
embeddings/LayerNorm/softmax/flow scales/iSTFT head. split-fp16 HiLo is an OPTIONAL flag for a
future 8-stream tier, not needed now. Budget: ~59 MB fp32 encoder + ~27 MB fp16 decoder = 86 MB.

STAGE 1 (the deliverable if you run out of time) - tools/export_v21.py -> build/v21.ptk +
spec/v21_layers.json:
- export initializers AND Constant nodes; keep the v1 container format, 64-byte alignment.
- per-layer entries in execution order: graph/module/op/packed tensor names/shapes/params (stride,
  pad, dilation, group, kernel, epilogue, dtype: fp32|fp16|fp16_hilo) for Conv/ConvTranspose/
  MatMul/Gemm, PLUS the elementwise/reduction ops the runtime must fuse (LayerNorm, Snake, Softmax,
  Exp, mean, Slice-masking, ScatterND/Where if present in the trace) listed as entries with their
  inputs/outputs, so the CUDA runtime can be written straight from the json.
- HARD GATE - exit non-zero and print the offending tensor if: (a) reconstructing each weight from
  the packed tensors (HiLo sum where applicable) differs from the ONNX tensor by >1e-6 elementwise
  max; (b) any of the 1908 weight nodes has no packed entry; (c) total container bytes deviates from
  the predicted dtype budget by >2%.
- print a dtype budget table (bytes per group enc/dec/istft/other) reconciled to 86 MB.

STAGE 2 - tests/gen_golden_v21.py -> tests/golden_v21/<case>/ with onnxruntime at noise_scale=0:
cases T=20/40/81/120, states the model's legal T range from its meta and records gaps, and includes
the exact phoneme/tone/lang sequences from bench/cpu/requests.txt so v2.1 goldens align with the
v1b vectors. Dump per-layer intermediates for a probe subset (encoder output, durations, mel, flow x
at each flow step, iSTFT magnitude/phase) using tools/probe_shapes.py naming conventions.

STAGE 3 (only if 1+2 are green) - tests/v21_reference.py: pure-numpy execution of the container
graph (conv1d, conv_transpose1d, layernorm, snake/silu, softmax attention, flow coupling steps,
iSTFT head) reproducing the golden. Report per-layer rel err, end-to-end SNDR, and oracle wall time
per utterance.

Also write docs/v21_layers.md: op inventory + chosen dtype per layer + MACs reconciled to the
measured 19.605 GMAC at T=81 (show the arithmetic and any gap), and state plainly
"zh-TW front-end coverage: known gap, not measured" (g2pw is not ported).

SCOPE - create/edit ONLY: tools/export_v21.py, tests/gen_golden_v21.py, tests/v21_reference.py,
build/v21.ptk, spec/v21_layers.json, docs/v21_layers.md, tests/golden_v21/.
Do NOT touch runtime/, tools/export_ptk.py, tools/fp16_probe.py, bench/, server/, docs/cost.md,
docs/v21_plan.md, ref/, model/, .gitignore. Do not spawn lanes.

REPORT (even if stages are incomplete): what completed, gate output verbatim, dtype budget table,
file list, exact repro commands, top 3 risks for the CUDA runtime.` }

const res = await runs.all([v21m1])
return res.map(r => ({ runId: r.runId, ok: r.ok, out: (r.output || '').slice(0, 4000) }))
