// 4 independent lanes, disjoint output directories, no git, no shared files.
const ctx = `
PROJECT: optimize HuggingFace PrimeTTS for Jetson Nano, paired with Gemini Live (speech-in / text-out),
goal = 8 parallel real-time TTS streams at max efficiency, near speech-to-speech experience at lower cost, zh-TW/en.
REPO/cwd: /home/louis/gemini-live-with-local-tts-jetson-nano  (git repo; model/ has downloaded ONNX weights)
Nano SSH (operator-confirmed): ssh picard@100.84.151.116   (Jetson Nano, L4T R32.5.1, 4x Cortex-A57 ARMv8.0,
 fp32-only NEON, MAXN power mode, CUDA 10.2 at /usr/local/cuda/bin/nvcc (NOT on PATH), sm_53 Tegra X1, gcc 8.4,
 Python 3.6, NO sudo, NO apt, ~185 MB free RAM, 14 GB free disk, internet reachable)
Already measured on that Nano: fp32 NEON GEMM 3.74 GFLOP/s per core, 9.78 GFLOP/s at 4 threads, memcpy 5.0-5.9 GB/s,
 CUDA axpy bandwidth 5.5 GB/s, tiny-kernel launch 26 us.
Local host: venv .venv (python3.11, onnx 1.23, onnxruntime 1.28, numpy 2.4) -> use .venv/bin/python. uv at /home/louis/.local/bin/uv.
HARD RULES: only create/modify files inside YOUR output paths below. Never run git. Never touch model/, .venv/,
or other lanes' directories (runtime/, tools/, tests/ belong to the parent). Never print or copy any API key.
No sudo. Nano-side work goes in /tmp/nano-bench; kill your processes; sustained load under ~4 minutes.
Time-box: a working, MEASURED deliverable beats exhaustive exploration. End with a concise findings summary and the
exact paths you produced.
`

const v21 = { key: 'v21-spec', label: 'Reverse-engineer PrimeTTS v2.1 graph', agent: 'worker', task: `${ctx}
YOUR LANE: fully reverse-engineer model/v21_mbistft_16k/primetts_v21_3voice.onnx (MB-iSTFT-VITS, 37.9M params,
16 kHz, 3 voices, 4066 nodes; inputs x/tone/lang/x_lengths/sid/noise_scale/length_scale; output wav).
Output paths (create): spec/v21_graph.md, spec/v21_macs.json, ref/v21_numpy.py, ref/golden/, tools/ort_profile.py
1. spec/v21_graph.md - semantic layer list in execution order, collapsed from the ONNX node soup: phoneme/tone/lang
   embedding sums, length mask, encoder (transformer n_layers=6, heads=2, hidden=192, filter=768, kernel_size=3),
   speaker conditioning (sid, gin_channels=256), normalizing flow (n_layers_q=3 x 3 affine coupling, exact conv
   configs, reverse order), decoder (upsample_rates [4,4], kernel 16, initial channel 512, resblock kernels [3,7,11]
   dilations [1,3,5]), multi-band iSTFT head (gen_istft_n_fft=16, hop=4, subbands=4, exported as irFFT + overlap-add
   matrices), final padding/shape. For every layer: exact ONNX initializer names + shapes, op params
   (pad/stride/dilation/group), tensor shapes as functions of T (=2*phones+1) and frame count, constants (eps,
   scales). Flag everything awkward for a hand-written runtime (Slice masking, Shape/Gather/Range, ScatterND, Where,
   random_normal use, transposes).
2. spec/v21_macs.json - per-layer MACs + activation bytes for a representative T (e.g. 40 phones -> T=81) and per
   audio-second, plus totals; name the dominant layers.
3. ref/v21_numpy.py - from-scratch numpy implementation loading weights directly from ONNX initializers, reproducing
   the onnxruntime output for arbitrary inputs. Validate against onnxruntime on >=5 cases (random valid ids plus a
   realistic zh-TW/en code-mix id sequence, sids 0/1/2): report max and mean absolute error, target relative < 1e-3.
   Use noise_scale=0 for determinism where possible and document determinism behaviour.
4. ref/golden/ - per case: input .npy arrays, expected wav, plus float32 checkpoints of ~12 key intermediates
   (encoder out, flow out, decoder upsampled samples, iSTFT spectra, final wav) so aarch64-asm and CUDA kernels can
   be validated layer by layer. Write ref/golden/MANIFEST.md listing every file, shape, and the single command to
   regenerate them.
5. tools/ort_profile.py - onnxruntime with enable_profiling, print top-20 nodes by kernel time (ranking only).
Success bar: parity < 1e-3 verified by a printed report; MAC numbers present; every golden tensor regenerable by one
command. Stop/ask: if graph semantics are ambiguous, state the ambiguity in the spec instead of guessing.` }

const v1 = { key: 'v1b8k-spec', label: 'Reverse-engineer PrimeTTS v1b_8k graph', agent: 'worker', task: `${ctx}
YOUR LANE: fully reverse-engineer the 8 kHz model: model/v1b_8k/{acoustic_encoder.onnx, acoustic_decoder.onnx,
vocoder.onnx, meta.json, inflect_combined_v1b.gguf} plus model/scripts/synth_from_text.py (numpy host_regulate length
regulator) and model/scripts/symbol_table.json.
Output paths (create): spec/v1b8k_graph.md, spec/v1b8k_macs.json, spec/frontend.md, ref/v1b8k_numpy.py,
ref/golden_v1b8k/, ref/texts.txt
1. spec/v1b8k_graph.md - exact layer-by-layer spec of all three graphs (encoder: BiGRU + depthwise gated Conv-FFN +
   postnet + duration/pitch heads; the numpy length regulator with exact algorithm, abs_frame_bins, max_frames,
   frame_meta semantics; decoder; Snake-HiFiGAN vocoder incl. the Snake activation formula), with initializer
   names/shapes, op params, shapes as functions of phone count and frame count. Document the .gguf container layout
   (header, sections, mapping to the three ONNX graphs) so a C loader can read weights from it.
2. spec/v1b8k_macs.json - MACs + activation bytes per layer, per audio-second at 8 kHz, totals, dominant layers.
3. ref/v1b8k_numpy.py - from-scratch numpy implementation (encoder -> length regulator -> decoder -> vocoder)
   reproducing onnxruntime end to end; validate >=5 cases with max/mean abs error, target < 1e-3.
4. ref/golden_v1b8k/ - inputs, expected 8 kHz waveform, intermediates (encoder out, durations, regulated frames, mel,
   wav) + MANIFEST.md.
5. spec/frontend.md - text -> phone/tone/lang ids (scripts/frontend_bopomofo.py: g2pw bopomofo + polyphone
   disambiguation, g2p_en, 88-symbol table, blank insertion) and scripts/text_norm.py entity normalization.
   CRITICAL: quantify frontend compute cost (does polyphone disambiguation require a BERT forward pass? ms on x86?
   would it run on a Nano CPU?). Time-boxed optional (<=15 min): install into a SEPARATE venv with
   "uv venv --python 3.11 .venv-fe && uv pip install -p .venv-fe/bin/python g2pw g2p_en cn2an" and synthesize real
   sentences; if it works, write ref/texts.txt (12 benchmark sentences: zh-TW, en, code-mix, entity-heavy phone
   numbers/addresses/prices) and record the exact install steps.
Success bar: parity < 1e-3 verified; frontend cost quantified; gguf layout documented.
Stop/ask: report ambiguities explicitly; read the python source for regulator semantics, do not invent them.` }

const nano = { key: 'nano-roofline', label: 'Jetson Nano roofline microbenchmarks', agent: 'worker', task: `${ctx}
YOUR LANE: measure the real roofline of the Jetson Nano for a from-scratch TTS runtime, CPU (hand-written aarch64
NEON) and GPU (CUDA C, sm_53, CUDA 10.2). Output paths (create): bench/nano/ (sources, RESULTS.md, raw logs).
1. CPU aarch64 kernels (asm/intrinsics): 8x8 tile fp32 GEMM; conv1d as GEMM and as direct sliding window with
   pad/dilation; depthwise conv1d; fused LayerNorm+LeakyRelu; softmax; Snake activation (x + sin^2(x)). Measure
   GFLOP/s at 1/2/4 threads, best tile/block sizes for L2=2MB, and the per-call overhead of small kernels (1e3..1e5
   elements) so the fusion payoff is quantified.
2. CUDA C with /usr/local/cuda/bin/nvcc -O3 -arch=sm_53: (a) tiled fp32 SGEMM GFLOPS at 256^3 and at realistic TTS
   shapes (192x192x81, 512x512xK conv-as-GEMM); (b) conv1d kernels; (c) measured per-kernel launch overhead;
   (d) THE KEY MEASUREMENT: CUDA Graph capture+instantiate+replay of N-kernel sequences (N=10,100,500) versus N
   individual launches - report us/kernel for both; (e) cooperative launch persistent kernel: cost of grid.sync() on
   this 1-SM-group device and feasibility of a single-kernel whole-network design; (f) __half storage: bandwidth and
   compute effect; (g) zero-copy host buffers (cudaHostAlloc HostMapped) and pinned transfer times; (h) 8 concurrent
   streams of small kernels vs 1 batched larger kernel (amortization evidence); (i) cudaMalloc/free cost and per-stream
   memory footprint.
3. Sustained load reality: 3-minute 4-core GEMM plus GPU GEMM, sampling CPU freq
   (/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq), GPU freq nodes and thermal zones readable without sudo;
   report throttling.
4. bench/nano/RESULTS.md: roofline table (GFLOP/s by kernel type and thread count, bytes/s, launch/graph overheads,
   throttling) plus explicit design implications (target kernel count per inference, minimum useful GEMM size,
   batching payoff, fp32-only rationale, where fusion wins). Include the one-line build/run command for each program.
Success bar: numbers measured on the device; sources compile and run; CUDA-graph vs per-launch comparison quantified.
Stop/ask: if CUDA 10.2 lacks an API (graphs, cooperative launch), report the exact compile/link error and workaround.` }

const gem = { key: 'gemini-bridge', label: 'Gemini Live speech-in bridge + cost model', agent: 'worker', task: `${ctx}
YOUR LANE: build and MEASURE the Gemini side: user speech -> Gemini Live (speech-in, transcript out) -> Gemini text
model reply (text-out) -> local TTS request. Output paths (create): server/, docs/proto.md, docs/cost.md
Constraints: the key is in gemini-live-credential.txt (source it into env). NEVER echo/log/print the key. Add
server/.gitignore covering *credential*. Keep total API spend trivial (a handful of short calls, < 1 minute of audio
total) and report exact token usage consumed.
1. server/live_asr.py: Live API model gemini-3.8-live with response_modalities=["AUDIO"] and
   input_audio_transcription enabled; stream 16 kHz mono PCM from a file in realistic chunks; collect
   interim_input_transcription + final input_transcription. MEASURE: (i) time from final audio chunk to final
   input_transcription; (ii) usageMetadata: audio input tokens AND audio OUTPUT tokens billed for the model's
   discarded spoken reply (this number is the crux of the cost argument - report it exactly); (iii) whether
   output_transcription is available and what it costs.
2. server/reply_gen.py: streaming reply with gemini-3.8-flash and gemini-3.8-flash-lite (streamGenerateContent),
   zh-TW contact-centre system prompt, fed the transcript. MEASURE time-to-first-token, tokens/sec, output token count
   for typical 1-2 sentence replies.
3. server/tts_client.py + server/mock_tts.py: Nano TTS protocol: TCP, JSON request
   {"text":...,"sid":0,"sr":8000,"noise_scale":0.667,"req_id":...} -> framed 16-bit LE mono PCM stream, chunk-level
   streaming so playback starts before synthesis finishes. Document docs/proto.md (framing, chunking,
   cancellation/barge-in, keepalive, 8-connection multiplexing).
4. server/pipeline.py: wire 1+2+3 into an offline demo: zh-TW wav in -> assistant text -> TTS PCM out. MEASURE
   end-to-end from end-of-user-speech to first TTS audio byte, with per-stage breakdown.
5. docs/cost.md: measured token usage -> $/conversation-minute for (a) native 3.8 Live speech-to-speech, (b) this
   hybrid (Live ASR + Flash text + local TTS), (c) text-out + Gemini TTS audio (3.8 Flash-Lite TTS $6/1M audio).
   Show arithmetic; note zh-TW (BCP-47 zh-TW) support for each leg.
6. server/README.md: exact run commands, deps via "uv pip install -p .venv/bin/python google-genai websockets
   aiohttp numpy", and what is known about zh-TW support in Live vs Flash.
Success bar: real measured numbers (with token usage) for all three legs, protocol doc, working demo command.
Stop/ask: if the key is invalid/quota-limited, report the exact HTTP error (key redacted) and stop - do not retry
aggressively.` }

const res = await runs.all([v21, v1, nano, gem])
return res.map(r => ({ runId: r.runId, ok: r.ok, out: (r.output || '').slice(0, 4000) }))
