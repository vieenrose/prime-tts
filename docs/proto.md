# Nano TTS wire protocol (v1)

TCP, one connection per stream, max 8 concurrent (`MAX_STREAMS = 8`). Little-endian throughout.
Implemented by `server/mock_tts.py` (mock) and `server/tts_client.py` (client); the Nano runtime
serves the same wire format with real audio.

## 1. Request (client -> server, one JSON line)

```json
{"req_id": 1, "ids": [[phone ids], [tone ids], [lang ids]], "text": "原始文字(僅供記錄)",
 "sid": 0, "sr": 8000, "noise_scale": 0.667, "session_id": "optional", "start_seq": 0}
```
- `ids` are the model's phoneme inputs (the frontend runs on the server host, not on the Nano -
  see `spec/frontend.md`). `text` is logging only; synthesis uses `ids`.
- `sid`: 0 = young female (v1b_8k has one voice; v2.1 exposes 0/1/2).
- `sr`: 8000 (v1b_8k native) or 16000 (v2.1 native; 8k delivery = server-side downsample).
- `noise_scale`: sampler temperature; 0 = deterministic (used by the parity tests).

Cancel / barge-in: `{"req_id": 1, "cancel": true}` - the server stops emitting frames for that
request and answers with a `cancelled` control line. Keepalive: `{"ping": true}` -> `{"pong": true}`.

## 2. Audio frames (server -> client)

Frame header: `struct("<II")` = `u32 magic | u32 seq`.
- `magic = 0x50545359` ("PTSY"), so the first wire byte is always `0x59`.
- Payload: 16-bit signed little-endian mono PCM, exactly one chunk = 40 ms
  (`chunk_samples(sr) = sr*40/1000`; 320 samples = 640 bytes at 8 kHz).
- `seq` counts chunks from 0.
- Terminal frame: header only, `seq = 0xFFFFFFFF`, zero-length payload.

Control messages are JSON lines that only appear **between** frames. A reader therefore peeks one
byte at a frame boundary: `0x59` starts a frame header, `{` starts a control line.

## 3. Streaming contract (why this shape)

The first audio byte must arrive long before the sentence finishes. The Nano runtime synthesizes in
chunks of 32 frames (vocoder chunking) and pushes 40 ms frames as soon as they exist, so a client
can start playback after the first frame. Chunk size is fixed so the client can compute frame
duration from `sr` without a length field.

Measured with the real Nano runtime (Phase 1, v1b_8k, 1 stream, 1 thread): first 48 ms of audio
100-140 ms after request; 4 threads: 63-72 ms. Under 8-way load the first-chunk mean was 489 ms
and max 1283 ms when all 8 requests landed at the same instant (`bench/cpu/logs/`).

## 4. Backpressure and admission

The server admits a stream only if it can keep up: measured worst lateness over the whole run was
+42 ms at 8 concurrent streams, so a ~120 ms (3-frame) playback buffer covers everything measured.
Admission policy (`runtime/src/sched.c`): with 16 requests arriving every 100 ms all requests
completed in real time, worst queue wait 245 ms. A server busy signal is
`{"req_id":n,"queued":ms}` before frames start.

## 5. Timing budget per turn (measured, see docs/cost.md)

| stage | measured |
|---|---|
| TTS request -> first audio byte (mock, 40 ms frames) | 129 ms |
| TTS request -> first audio byte (real Nano, 1 stream) | 63-140 ms |
| reply text generation, first token (3.8 Flash) | 284 ms |
| reply text generation (3.5 Flash-Lite, whole turn) | 2.58 s |
