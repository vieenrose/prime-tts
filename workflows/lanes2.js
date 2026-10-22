// PrimeTTS Nano lanes, wave 2. Workers on a capable tier (parent stays on its own default model).
const MODEL = "openrouter/deepseek/deepseek-v4.1-flash"

const ctx = `
PROJECT: optimize HuggingFace PrimeTTS for Jetson Nano, paired with Gemini Live (speech in / text out), to serve
8 parallel real-time TTS streams with the lowest possible cost, zh-TW + English.
cwd: /home/louis/gemini-live-with-local-tts-jetson-nano  (model/ holds the downloaded ONNX weights)
Local python: ALWAYS .venv/bin/python (python3.11, onnx 1.23, onnxruntime 1.28, numpy 2.4). System python3 is broken
(protobuf conflict). onnx 1.23 uses node.op_type, weights via onnx.numpy_helper.to_array.
Jetson Nano (operator-confirmed): ssh picard@100.84.151.116 - L4T R32.5.1, 4x Cortex-A57 ARMv8.0 (NEON fp32 only,
no int8 dot-product, no fp16 arithmetic), MAXN power mode, CUDA 10.2 at /usr/local/cuda/bin/nvcc (not on PATH),
sm_53 Tegra X1 921 MHz 256 KB L2, gcc 8.4, Python 3.6, NO sudo, NO apt, ~185 MB free RAM, 14 GB free disk.
Measured there already: NEON fp32 GEMM 3.74 GFLOP/s per core and 9.78 GFLOP/s at 4 threads; memcpy 5.0-5.9 GB/s;
CUDA axpy 5.5 GB/s; tiny-kernel launch 26 us.
HARD RULES: write ONLY inside your own output paths. Never run git. Never touch model/, .venv/, runtime/, tools/,
tests/, workflows/, or other lanes' directories. Never print or copy the API key. No sudo. Nano scratch dir is
/tmp/<lane-name>; kill your processes; sustained load under 4 minutes; never disturb the box's running services.
Work in small verified steps: compile, run, read the numbers. Report a concise summary plus the exact paths you wrote.
Time-box: a measured, working deliverable beats exhaustive exploration.
`

const v21 = {
  key: 'v21-spec', agent: 'worker', label: 'Reverse-engineer PrimeTTS v2.1 graph', model: MODEL,
  task: `${ctx}
YOUR LANE: fully reverse-engineer model/v21_mbistft_16k/primetts_v21_3voice.onnx (MB-iSTFT-VITS, 37.9M params,
16 kHz, 3 voices, 4066 nodes; inputs x/tone/lang/x_lengths/sid/noise_scale/length_scale; output wav).
Your output paths: spec/v21_graph.md, spec/v21_macs.json, ref/v21_numpy.py, ref/golden/, tools/ort_profile_v21.py
Useful starting point (parent's tool, already working): .venv/bin/python tools/probe_shapes.py <onnx> <out.json>
--frames 81 --dump-dir <dir> prints total MACs, per-op MAC totals, per-node shapes, and dumps every intermediate.
Deliverables:
1. spec/v21_graph.md - execution-ordered semantic layer list collapsed from the ONNX node soup: phoneme/tone/lang
   embedding sums, length mask, encoder (transformer n_layers=6, heads=2, hidden=192, filter=768, kernel 3),
   speaker conditioning (sid, gin_channels=256), normalizing flow (n_layers_q=3 x 3 affine coupling, exact conv
   configs, reverse order), decoder (upsample_rates [4,4], kernel 16, initial channel 512, resblock kernels [3,7,11]
   dilations [1,3,5]), multi-band iSTFT head (n_fft=16, hop=4, subbands=4, exported as irFFT + overlap-add matrices),
   final padding/shape. Per layer: exact ONNX initializer names + shapes, op params (pad/stride/dilation/group),
   tensor shapes as functions of T (=2*phones+1) and frame count, constants (eps, scales). Flag every construct that
   is awkward in a hand-written runtime (Slice masking, Shape/Gather/Range, ScatterND, Where, transposes,
   random_normal).
2. spec/v21_macs.json - per-layer MACs and activation bytes for T=81, plus per-audio-second and totals; name the
   dominant layers and give the ratio dec/flow/enc.
3. ref/v21_numpy.py - faithful from-scratch numpy runtime (expect 800-1200 lines) that loads weights directly from
   ONNX initializers and reproduces onnxruntime output. No silent gaps: any sub-op you cannot reproduce must raise
   an explicit exception and be listed in the spec with the reason. One function per semantic layer, each
   taking/returning float32 arrays with documented shapes, mirroring spec/v21_graph.md 1:1. It is the parity oracle
   for the parent's aarch64-asm and CUDA kernels, so exactness matters more than brevity.
4. Parity: validate vs onnxruntime on >=5 cases (random valid ids + a realistic zh-TW/en code-mix id sequence,
   sids 0/1/2), printing max and mean abs error; target relative error < 1e-3. Use noise_scale=0 for determinism and
   document determinism behaviour.
5. ref/golden/ - per case: inputs .npy, expected wav, and float32 checkpoints of ~12 key intermediates (encoder out,
   flow out, decoder upsampled samples, iSTFT spectra, final wav), plus ref/golden/MANIFEST.md (file, shape, one
   regeneration command). Add a --check mode to ref/v21_numpy.py that loads golden checkpoints and prints per-layer
   max/mean abs error.
6. tools/ort_profile_v21.py - onnxruntime enable_profiling run printing the top-20 nodes by kernel time (ranking only).
Success bar: parity < 1e-3 printed, MAC table present, golden regenerable by one command.
If ONNX semantics are ambiguous, state the ambiguity in the spec instead of guessing.` }

const v1 = {
  key: 'v1b8k-spec', agent: 'worker', label: 'Reverse-engineer PrimeTTS v1b_8k graph', model: MODEL,
  task: `${ctx}
YOUR LANE: fully reverse-engineer the 8 kHz model: model/v1b_8k/{acoustic_encoder.onnx, acoustic_decoder.onnx,
vocoder.onnx, meta.json, inflect_combined_v1b.gguf} plus model/scripts/synth_from_text.py (numpy host_regulate length
regulator) and model/scripts/symbol_table.json.
Your output paths: spec/v1b8k_graph.md, spec/v1b8k_macs.json, spec/frontend.md, ref/v1b8k_numpy.py,
ref/golden_v1b8k/, ref/texts.txt
Useful starting point: .venv/bin/python tools/probe_shapes.py <onnx> <out.json> --frames 81 --dump-dir <dir>.
Deliverables:
1. spec/v1b8k_graph.md - exact layer-by-layer spec of all three graphs (encoder: BiGRU + depthwise gated Conv-FFN +
   postnet + duration/pitch heads; the numpy length regulator: exact algorithm, abs_frame_bins, max_frames,
   frame_meta semantics; decoder; Snake-HiFiGAN vocoder incl. the Snake activation formula) with initializer
   names/shapes, op params, and shapes as functions of phone count and frame count. Also document the .gguf container
   layout (header, sections, mapping to the three ONNX graphs) so a C loader can read weights straight from it.
2. spec/v1b8k_macs.json - MACs and activation bytes per layer, per audio-second at 8 kHz, totals, dominant layers.
3. ref/v1b8k_numpy.py - faithful from-scratch numpy runtime (encoder -> length regulator -> decoder -> vocoder)
   reproducing onnxruntime end to end; no silent gaps (raise explicitly for anything unimplemented). One function
   per layer, mirroring the spec 1:1. Validate >=5 cases; print max/mean abs error, target < 1e-3. Add --check mode
   against golden checkpoints.
4. ref/golden_v1b8k/ - inputs, expected 8 kHz waveform, intermediates (encoder out, durations, regulated frames, mel,
   wav) + MANIFEST.md.
5. spec/frontend.md - text -> phone/tone/lang ids (scripts/frontend_bopomofo.py: g2pw bopomofo with polyphone
   disambiguation, g2p_en, 88-symbol table, blank insertion) and scripts/text_norm.py entity normalization.
   CRITICAL: quantify frontend compute cost - does polyphone disambiguation need a BERT forward pass, how many ms on
   x86, and would it be viable on a Nano CPU? Time-boxed optional (<=15 min): create a SEPARATE venv
   (/home/louis/.local/bin/uv venv --python 3.11 .venv-fe, then uv pip install -p .venv-fe/bin/python g2pw g2p_en
   cn2an) and synthesize real sentences; if it works, write ref/texts.txt with 12 benchmark sentences (zh-TW, en,
   code-mix, entity-heavy phone numbers/addresses/prices) and record exact install steps.
Success bar: parity < 1e-3 printed, frontend cost quantified, gguf layout documented.
Read the python source for regulator semantics; do not invent them.` }

const nano = {
  key: 'nano-roofline', agent: 'worker', label: 'Jetson Nano roofline microbenchmarks', model: MODEL,
  task: `${ctx}
YOUR LANE: measure the real Nano roofline for a from-scratch TTS runtime - CPU (hand-written aarch64) and GPU
(CUDA C, sm_53, CUDA 10.2). Your output paths: bench/nano/ (sources, RESULTS.md, raw logs/). Nano scratch:
/tmp/roofline. Build with gcc -O3 -march=armv8-a+fp+simd and /usr/local/cuda/bin/nvcc -O3 -arch=sm_53.
Deliverables:
1. CPU aarch64 kernels: 8x8 tile fp32 GEMM; conv1d as GEMM and as direct sliding window with pad/dilation;
   depthwise conv1d; fused LayerNorm+LeakyRelu; softmax; Snake activation (x + sin^2 x). Measure GFLOP/s at 1/2/4
   threads, best tile/block sizes for the 2 MB L2, and the per-call overhead of small kernels (1e3..1e5 elements) so
   the fusion payoff is quantified.
2. CUDA C: (a) tiled fp32 SGEMM GFLOPS at 256^3 and at realistic TTS shapes (192x192x81, 512x512xK conv-as-GEMM);
   (b) conv1d kernels; (c) per-kernel launch overhead; (d) KEY MEASUREMENT: CUDA Graph capture + instantiate + replay
   of N-kernel sequences (N=10,100,500) vs N individual launches - report us/kernel for both; (e) cooperative-launch
   persistent kernel: grid.sync() cost on this 1-SM-group device, and feasibility of a single-kernel whole-network
   design; (f) __half storage: bandwidth and compute effect; (g) zero-copy host buffers (cudaHostAlloc mapped) and
   pinned transfer times; (h) 8 concurrent streams of small kernels vs 1 batched larger kernel (amortization
   evidence); (i) cudaMalloc/free cost and per-stream memory footprint.
3. Sustained load: 3-minute 4-core GEMM + GPU GEMM while sampling /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq,
   GPU frequency nodes and thermal zones readable without sudo; report throttling.
4. bench/nano/RESULTS.md: roofline table (GFLOP/s by kernel and thread count, bytes/s, launch/graph overheads,
   throttling) plus explicit design implications: target kernel count per inference, minimum useful GEMM size,
   batching payoff, fp32-only rationale, where fusion wins. Give the one-line build/run command for each program.
Success bar: numbers measured on the device, sources compile and run, CUDA-graph vs per-launch quantified.
If CUDA 10.2 lacks an API (graphs, cooperative launch), report the exact compile/link error and the workaround.` }

const res = await runs.all([v21, v1, nano])
return res.map(r => ({ runId: r.runId, ok: r.ok, out: (r.output || '').slice(0, 3000) }))
