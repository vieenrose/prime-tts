# PrimeTTS v2.1 on the Jetson Nano

A C/CUDA inference runtime for **PrimeTTS v2.1** (`v21_mbistft_16k`, 37.9M parameters, zh-TW +
English, 16 kHz) on the original **Jetson Nano** (Tegra X1: 4× Cortex-A57, 128-core Maxwell `sm_53`,
4 GB shared RAM).

**Live demo:** [huggingface.co/spaces/Luigi/PrimeTTS-Nano](https://huggingface.co/spaces/Luigi/PrimeTTS-Nano)
— type text and hear it synthesized on a Jetson Nano, streamed back as it is produced.

- Pure **C (gnu11) + CUDA + pthreads** at runtime — no PyTorch, ONNX Runtime or TensorRT.
- **Bit-faithful:** every waveform stays inside the reference parity envelope (SNDR ≥ 60 dB; 70.09 dB
  on the reference utterance).
- **Real-time at 8 concurrent requests on one CPU core.**
- **OpenAI TTS API compatible** (`POST /v1/audio/speech`, streaming) — works as the TTS backend of
  [Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech) voice bots
  ([details](#openai-tts-api-compatibility)).

## Performance

Measured on a Jetson Nano, one CPU core for TTS, reference utterance T81 (3.62 s of audio):

| scenario | wall time | realtime factor |
|---|---|---|
| 1 request | 0.50 s | 7.2× |
| 4 concurrent requests (4 CPU cores) | 1.78 s | 8.1× aggregate |
| 8 concurrent requests, server mode | 3.2 s | 9.1× aggregate |

**Time to first audio** with 8 concurrent calls on one core (server mode, streaming):

| calls | 3.6 s sentence each | short clause each (0.9 s) |
|---|---|---|
| all 8 start at the same instant | 1.37 s median | 0.85 s median |
| arriving at random, ~0.5 s apart | 0.52 s median, 0.80 s max | 0.17 s median, 0.41 s max |

Single request through the text gateway: first audio 0.44 s for a short sentence, 1.45 s for a
58-character paragraph.

Against the starting point of this runtime, 4-request latency improved **5.8×** (10.3 s → 1.78 s) and
single-request latency **7.2×** (3.86 s → 0.54 s).

![Wall time of 4 concurrent requests across the optimization work](docs/latency-session.svg)

## How it works

- **Graph interpreter** (`pt_v21_host.c`): executes the model's 4066 nodes (41 op kinds) from tables
  generated from the ONNX export, with NEON elementwise ops, table-based activations and fused
  epilogues.
- **GPU kernels** (`pt_v21_cuda.cu`): implicit-GEMM convolution and transposed convolution with
  register-tiled, coalesced staging; weights uploaded once and shared by all streams.
- **Concurrency:** each request has its own context, CUDA stream and device arena; waiting threads
  sleep only when another stream needs the CPU.
- **Low CPU overhead per request:** tensor buffers are recycled from a per-worker pool (no page
  faults after warm-up), contiguous slices are zero-copy views, and table activations are vectorized
  — the host side is the bottleneck when TTS runs on a single core.
- **Streaming decoder:** the text encoder runs once per request; the decoder runs on overlapping
  windows (overlap-save), so audio starts early while remaining identical to a single full
  synthesis (max deviation 9e-8).

## Repository layout

```
runtime/v21/src/      runtime: interpreter, CUDA kernels, drivers, server
runtime/v21/gen/      generated graph tables (do not edit; `make ... gen`)
server/tts_gateway.py text gateway: zh-TW/en frontend (g2pw + g2p_en) -> Nano server
demo/                 web demo (Hugging Face Space)
docs/server.md        server protocol, setup, latency
tools/                ONNX export and graph-table generation
tests/                golden cases and the per-node parity gate
bench/, .auto/        benchmarks and the measurement harness
```

## Quick start

**1. Prepare the model** (on a workstation). Place the model at
`model/v21_mbistft_16k/primetts_v21_3voice.onnx`, then:

```bash
.venv/bin/python tools/export_v21.py        # -> build/v21.ptk
make -f runtime/v21/Makefile gen            # graph tables
```

**2. Build on the Nano** (gcc 8, CUDA 10, no root needed):

```bash
make -f runtime/v21/Makefile nano par serve
```

**3. Run:**

```bash
# one request, checked against the golden waveform
./build/pt_v21_nano --container build/v21.ptk --case tests/golden_v21/T81
# 8 concurrent requests
./build/pt_v21_par  --container build/v21.ptk --case tests/golden_v21/T81
# server: 8 warm workers on port 8090
./build/pt_v21_serve --container build/v21.ptk --port 8090
```

**4. Text API** (on any host with Python; setup in [`docs/server.md`](docs/server.md)):

```bash
server/.venv-gateway/bin/python server/tts_gateway.py --nano http://<nano>:8090 --port 8091

curl -X POST http://localhost:8091/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "您好，歡迎使用語音助理。", "voice": "0", "response_format": "wav"}' -o out.wav
```

See [OpenAI TTS API compatibility](#openai-tts-api-compatibility) to use it from a voice bot.

## OpenAI TTS API compatibility

The text gateway (`server/tts_gateway.py`) implements the **OpenAI text-to-speech API**, so any
OpenAI-compatible TTS client can use it unchanged — including the
[Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech) voice-bot
framework (`--tts openai` backend). Point clients at the gateway (port 8091), not at the Nano
server directly.

| endpoint | supported |
|---|---|
| `POST /v1/audio/speech` | `model` (any string), `input` (zh-TW / English text), `voice` (`"0"`, `"1"`, `"2"`), `response_format` (`pcm`, `wav`), `speed` (0.25–4), `stream_format: "audio"` |
| `GET /v1/audio/voices` | `{"voices": ["0", "1", "2"]}` |
| `GET /health` | gateway and Nano status |

Output is 16-bit mono PCM at **16 kHz**. With `pcm`, audio streams as it is synthesized (chunked
HTTP). Not supported: `mp3`/`opus`/`aac`/`flac`, `instructions`, and the vLLM `stream: true`
extension. No API key is required (any `Authorization` header is ignored).

**Using it in a Hugging Face speech-to-speech voice bot:**

```bash
speech-to-speech local \
  --tts openai \
  --openai_tts_base_url http://<gateway-host>:8091/v1 \
  --openai_tts_model primetts-v21 \
  --openai_tts_voice 0 \
  --openai_tts_response_format pcm \
  --openai_tts_sample_rate 16000
```

Keep `--openai_tts_stream` at its default (`false`). The startup "Warmup" request of
speech-to-speech is answered normally. Each LLM text chunk becomes one request and is streamed back
as it is synthesized.

## Configuration

| option | default | effect |
|---|---|---|
| `--parallel N` / `--workers N` | 8 | concurrent requests (driver / server) |
| `--no-cuda` | off | CPU-only path |
| `--keepwarm SEC` | 30 | keep the GPU clock up between requests (server) |
| `PT_V21_CPUS` | one core | `all`, or a list such as `2,3` |
| `PT_V21_WAIT_US` | 200 | GPU wait poll interval; `0` = driver wait |
| `PT_V21_STREAM_WIN`, `PT_V21_STREAM_CTX` | 48, 16 | streaming window / context, in frames |
| `PT_V21_POOL_MB` | 32 | per-worker tensor buffer pool; `0` disables |

Two CPU cores are recommended when short utterances must stay real-time under full load.

## Quality and testing

- `.auto/checks.sh` — integrity manifest of model, goldens and gate, plus SNDR ≥ 60 dB on the Nano
  (GPU and CPU paths).
- `tests/v21_parity_c.py --from-dumps <dir>` — per-node comparison against the reference
  (4010 nodes, 0 over tolerance).
- `demo/test/playback_e2e.mjs` — checks that the demo player renders exactly the delivered samples.

Parity runs use `noise_scale = 0`; sampling noise is not implemented yet.

## Known limitations

- `noise_scale` has no effect (output is deterministic).
- Output formats: `pcm` and `wav` only.
- The Nano exposes one SM; scaling beyond ~8 concurrent requests would need fp16 (half2) compute.

## Further reading

- [`docs/server.md`](docs/server.md) — server and gateway protocol
- `.auto/ideas.md` — optimization notes and measurements
- `bench/gpu/RESULTS.md` — Nano GPU roofline
