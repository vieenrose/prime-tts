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
- **OpenAI-compatible server** (`POST /v1/audio/speech`), streaming, drop-in for
  [Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech).

## Performance

Measured on a Jetson Nano, reference utterance T81 (3.62 s of audio):

| scenario | wall time | realtime factor |
|---|---|---|
| 1 request | 0.54 s | 6.8× |
| 4 concurrent requests (4 CPU cores) | 1.78 s | 8.1× aggregate |
| 4 concurrent requests, server mode (1 core) | 1.61 s | 9.0× aggregate |
| 8 concurrent requests, server mode (1 core) | 3.17 s | 9.1× aggregate |

Streaming through the server (1 core, text in, first audio → end of synthesis):

| input | first audio | complete | audio length |
|---|---|---|---|
| short sentence (12 chars) | 0.44 s | 0.64 s | 2.0 s |
| paragraph (58 chars) | 1.45 s | 2.61 s | 10.9 s |

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

With Hugging Face speech-to-speech:

```bash
speech-to-speech local --tts openai --openai_tts_base_url http://<gateway>:8091/v1 \
  --openai_tts_voice 0 --openai_tts_response_format pcm --openai_tts_sample_rate 16000
```

## Configuration

| option | default | effect |
|---|---|---|
| `--parallel N` / `--workers N` | 8 | concurrent requests (driver / server) |
| `--no-cuda` | off | CPU-only path |
| `--keepwarm SEC` | 30 | keep the GPU clock up between requests (server) |
| `PT_V21_CPUS` | one core | `all`, or a list such as `2,3` |
| `PT_V21_WAIT_US` | 200 | GPU wait poll interval; `0` = driver wait |
| `PT_V21_STREAM_WIN`, `PT_V21_STREAM_CTX` | 48, 16 | streaming window / context, in frames |

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
