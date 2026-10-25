# PrimeTTS v2.1 server mode (OpenAI-compatible `/v1/audio/speech`)

Two processes:

```
HF speech-to-speech / any OpenAI TTS client
   │  POST /v1/audio/speech  {"input": "<text>", "voice": "0", "response_format": "pcm", ...}
   ▼
server/tts_gateway.py   (s2s host: zh-TW/en frontend — g2pw + g2p_en + text_norm — needs torch)
   │  text → sentences → phone/tone/lang ids (blank-interleaved) → one Nano request per sentence
   ▼  POST /v1/audio/speech  {"input_ids": {...}, "priority": <deadline>, ...}
build/pt_v21_serve      (Jetson Nano: C + CUDA, warm contexts, resident weights)
```

The frontend cannot run in the C runtime (no new dependencies; g2pw is a BERT model), so the
gateway owns text and the Nano owns audio. Both speak the same OpenAI request shape.

## Why server mode

The per-process benchmark driver pays ~287 ms of serialized startup per request (container
record pass, CUDA context, 111 MB weight upload, clock ramp). The server pays it once at boot
(`ready in ~2.1 s` with 8 warm workers). Measured on picard (1-core cap, T81 golden ids):

| | per-process driver | server |
|---|---|---|
| 1 request, T81 (3.62 s audio) | ~0.53 s stream + startup | 0.54-0.58 s (HTTP incl.) |
| 4 concurrent T81 | ~1.95-2.1 s | **1.61-1.63 s** |

End-to-end through the gateway (text in, HF s2s request shape, this host → picard over Tailscale):

| | first audio byte | total | audio |
|---|---|---|---|
| s2s "Warmup" | 0.22-0.48 s | same | 0.61 s |
| one zh sentence | 0.44-0.61 s | same | 2.0 s |
| 4-sentence paragraph | **0.34 s** | 1.47 s | 11.0 s |
| 4 concurrent paragraphs | **0.42-0.98 s** | 4.8-5.4 s | 4 × 11.0 s, no underrun |

## Nano server: `pt_v21_serve`

```bash
make -f runtime/v21/Makefile serve        # on the Nano (CUDA)
./build/pt_v21_serve --container build/v21.ptk --port 8090 [--workers 8] [--keepwarm 30]
```

- `--workers N` (default 8, max 8): warm contexts = concurrent syntheses; more requests queue
  (max 64, then 503). The process keeps the 1-core CPU cap of the other drivers (`PT_V21_CPUS`).
  For 8 parallel T81 on one core: 8 workers serve them in **3.17 s wall** (streams 3.09-3.17 s,
  per-stream RTF ~0.87, 9.14x realtime aggregate) vs ~3.7 s per-process -- server mode is the N=8
  optimization. 4 vs 8 workers for 8 concurrent requests is a tie (the SM serializes either way).
- Streaming knobs (rarely needed): `PT_V21_STREAM_WIN` (frames per decoded window, default 48) and
  `PT_V21_STREAM_CTX` (context frames kept on each side, default 16). The first chunk of audio is
  `(WIN - CTX)` frames ≈ 0.5 s; smaller WIN cuts first-audio latency, at a little more overlap work.
- `--keepwarm SEC` (default 30, 0 = off): the GPU governor drops to 76.8 MHz within seconds of
  idle, and the next request then runs ~2.5x slower (T81: 1316-1665 ms after 5 s idle vs 542-577
  ms warm). For SEC seconds after the last request, a background thread issues short ALU bursts
  while no worker is busy, sized by feedback on `cur_freq` (grow below max, decay at max), which
  holds full clock across 20 s gaps (559-616 ms post-idle). After the window the GPU idles
  normally (power, GPU co-tenants). `PT_V21_KW_ITERS` fixes the burst size, `PT_V21_KW_MS` the
  period (100 ms).
- `GET /health` → `{"status","workers","ready","busy","queued","gpu_mhz"}`;
  `GET /v1/audio/voices` → `{"voices":["0","1","2"]}`.

### `POST /v1/audio/speech` (Nano)

| field | |
|---|---|
| `input_ids` | **required**: `{"phone":[...],"tone":[...],"lang":[...]}` or `[[phone],[tone],[lang]]`, equal length 1..4096, blank-interleaved already; phone < 88, tone < 6, lang < 2 |
| `voice` | `"0"`/`"1"`/`"2"` (speaker id); other names → `--default-sid` |
| `response_format` | `"pcm"` (s16le mono 16 kHz, `audio/pcm`) or `"wav"` (default) |
| `speed` | 0.25-4, `length_scale = 1/speed` |
| `priority` | optional deadline in unix ms; the worker queue is earliest-deadline-first (default = arrival = FIFO) |
| `noise_scale` | accepted; **currently no effect**: the runtime evaluates the sampler as zeros (the parity setting), so output is deterministic |
| `input`, `model`, `stream_format` | ignored (text needs the gateway) |

A request with text but no `input_ids` gets a 400 that names the gateway. Errors use the OpenAI
shape `{"error":{"message","type"}}`.

Audio quality: streaming is **exact**. One request = one utterance: the encoder runs once and the
decoder tail (245 local nodes) executes on sliding windows with 16 frames (256 ms) of context —
measured against a single full-graph run this converges to max|d| 9e-8 (133 dB) at 20 frames and
83-88 dB at 14-16 frames. Delivery as PCM16 adds the usual ~2 dB quantization cost (SNDR 68.2 dB on
T81 vs the float golden).

There is deliberately **no edge fading**: the server streams inside one synthesis, so prosody is
continuous and the model starts once per request — the workaround a per-sentence split needed.
delivering it as PCM16 adds int16 quantization, which brings SNDR vs the float golden to 68.12 dB
(T81) / 68.08 (T20) / 64.62 (sid1_T29) — exactly the predicted quantization+model combination.
Any PCM16 endpoint has this cost.

## Gateway: `server/tts_gateway.py`

```bash
uv venv --python 3.11 server/.venv-gateway
uv pip install --python server/.venv-gateway/bin/python g2pw g2p_en cn2an "nltk==3.8.1" numpy onnxruntime transformers
uv pip install --python server/.venv-gateway/bin/python torch --index-url https://download.pytorch.org/whl/cpu
NLTK_DATA=<dir with cmudict + averaged_perceptron_tagger> \
  server/.venv-gateway/bin/python server/tts_gateway.py --nano http://<nano>:8090 --port 8091
```

The first g2pw call downloads its model (607 MB, ~7 min once) into `--g2pw-home` (default
`model/scripts/`, gitignored; g2pw resolves `./G2PWModel` from the cwd, so the gateway pins it);
startup then loads the frontend (~30 s)
before listening. The nltk data shipped with the `Luigi/PrimeTTS-vs-Inflect-Nano-v1` Space works.

- `POST /v1/audio/speech`: standard OpenAI fields (`input`, `voice`, `response_format` pcm|wav,
  `speed`; `noise_scale` forwarded). `pcm` streams with chunked transfer, one chunk per sentence
  as soon as it is ready, in order; `wav` returns one file.
- Sentence splitting on `。！？!?；;…` / newline / `. `, long runs cut at commas (≤120 chars), and the
  **first sentence is cut at its first comma** (>10 chars) so the opening chunk is short.
- Scheduling is earliest-deadline-first end to end: sentence *i* is due at request arrival +
  estimated playback of sentences 0..i-1 (~5 zh chars/s), on the gateway's single frontend
  thread and on the Nano's worker queue (`priority`). Under load every request's first sentence
  runs before anyone's later ones. (Measured: with a plain lock and fixed lookahead, 3 of 4
  concurrent paragraphs waited 6.6 s for first audio while the Nano idled; EDF + first-clause cut:
  0.42-0.98 s.)
- `--no-blank` disables the blank interleave (v2.1 was trained with `add_blank: true`).

## HuggingFace speech-to-speech

```bash
speech-to-speech local --tts openai \
  --openai_tts_base_url http://<gateway-host>:8091/v1 \
  --openai_tts_model primetts-v21 --openai_tts_voice 0 \
  --openai_tts_response_format pcm --openai_tts_sample_rate 16000
```

s2s sends `{"model","input","voice","response_format":"pcm","stream_format":"audio"}` and reads
the body incrementally; its startup "Warmup" request is answered (0.61 s of audio). Do not pass
`--openai_tts_stream true` (vLLM-Omni extension; ignored here, and it forbids `speed`). The
standard `/v1/audio/speech` has no streaming *input* — s2s sends one request per LLM text chunk,
which is exactly the short-request pattern server mode makes cheap.

Not implemented: true chunked vocoding inside one sentence (the runtime returns a sentence's
waveform when it is complete), mp3/opus/aac/flac formats, server-side cancel (closing the
connection stops delivery; the sentence in synthesis finishes), stochastic sampling.
