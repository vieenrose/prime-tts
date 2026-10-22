#!/usr/bin/env python3
"""Mock of the Jetson Nano PrimeTTS v1b_8k TCP service: emits sine-wave PCM frames.

Implements the wire protocol in docs/proto.md (see that file for framing, chunk size,
cancel/barge-in, keepalive, concurrency and backpressure), with the chunk pacing
measured on the real Nano in bench/cpu/RESULTS.md:
  - first audio chunk 126 ms after the request (mean, single stream, 1 thread);
  - later chunks every rtf * 40 ms with rtf = 0.191 (measured single-stream RTF);
  - 8 concurrent streams max, matching the Nano's measured real-time capacity.

Run: .venv/bin/python server/mock_tts.py --port 8765
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct
import time

MAGIC = 0x50545359  # "PTSY", little-endian on the wire, so frames start with byte 0x59
TERMINAL_SEQ = 0xFFFFFFFF  # terminal frame: header only, zero-length payload
CHUNK_MS = 40
SR_DEFAULT = 8000
NOISE_SCALE_DEFAULT = 0.667
MAX_STREAMS = 8
AMPLITUDE = 0.30  # full-scale amplitude of the mock sine
FREQ_HZ = 440.0
# Mock duration heuristic: the real engine takes durations from the duration head
# (zh_greeting: 21 tokens -> 2.608 s). 0.11 s/token ≈ zh-TW average speech rate.
SEC_PER_TOKEN = 0.11

HEADER = struct.Struct("<II")


def frame(seq: int, payload: bytes) -> bytes:
    return HEADER.pack(MAGIC, seq) + payload


def chunk_samples(sr: int) -> int:
    """One 40 ms chunk of mono s16le at `sr` (320 samples = 640 bytes at 8 kHz)."""
    return round(sr * CHUNK_MS / 1000)


def sine_pcm(sr: int, sample_offset: int, n_samples: int) -> bytes:
    """Deterministic 440 Hz sine, phase-continuous across chunks (mock audio payload)."""
    out = bytearray(2 * n_samples)
    for i in range(n_samples):
        v = int(AMPLITUDE * 32767.0 * math.sin(2 * math.pi * FREQ_HZ * (sample_offset + i) / sr))
        out[2 * i:2 * i + 2] = struct.pack("<h", max(-32768, min(32767, v)))
    return bytes(out)


class MockTtsServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        rtf: float = 0.191,
        first_chunk_ms: float = 126.0,
        sec_per_token: float = SEC_PER_TOKEN,
        max_streams: int = MAX_STREAMS,
        log: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.rtf = rtf
        self.first_chunk_ms = first_chunk_ms
        self.sec_per_token = sec_per_token
        self.max_streams = max_streams
        self.log = log
        self._slots = asyncio.Semaphore(max_streams)
        self._server: asyncio.Server | None = None

    def _log(self, peer: str, msg: str) -> None:
        if self.log:
            print(f"[mock_tts {peer}] {msg}", flush=True)

    async def start(self) -> None:
        self._server = await asyncio.start_server(self.handle, self.host, self.port)
        self._log(self.host, f"listening on {self.host}:{self.port}")

    async def serve_forever(self) -> None:
        await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Connection loop: JSON-line control plane, one active stream per connection."""
        peer = "peer"
        info = writer.get_extra_info("peername")
        if info:
            peer = f"{info[0]}:{info[1]}"
        active: dict = {"task": None, "req_id": None}

        async def stop_active(reason: str) -> None:
            task = active["task"]
            if task is not None and not task.done():
                task.cancel()  # barge-in: the stream stops at the next chunk boundary
                self._log(peer, f"{reason}: cancelling req_id={active['req_id']}")

        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    self._log(peer, "frame truncated at EOF")
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as exc:
                    await self._control(writer, {"error": f"bad json: {exc}"})
                    continue
                if msg.get("ping"):
                    await self._control(writer, {"pong": True, "t_ns": time.monotonic_ns()})
                    continue
                if msg.get("cancel"):
                    req_id = int(msg.get("req_id", -1))
                    if active["task"] is not None and not active["task"].done() and active["req_id"] == req_id:
                        await stop_active("cancel")
                    else:
                        await self._control(writer, {"req_id": req_id, "ack": "no_stream"})
                    continue
                if "ids" not in msg:
                    await self._control(writer, {"error": "request needs ids"})
                    continue
                await stop_active("barge-in")
                active["task"] = asyncio.create_task(self._stream(peer, msg, writer))
                active["req_id"] = int(msg.get("req_id", 0))
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            await stop_active("disconnect")
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _control(self, writer: asyncio.StreamWriter, obj: dict) -> None:
        """Control-plane JSON line. Written only between frames (docs/proto.md §3)."""
        try:
            writer.write((json.dumps(obj) + "\n").encode())
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass

    async def _stream(self, peer: str, req: dict, writer: asyncio.StreamWriter) -> None:
        req_id = int(req.get("req_id", 0))
        sr = int(req.get("sr", SR_DEFAULT))
        ids = req.get("ids") or []
        n_tokens = len(ids[0]) if ids and isinstance(ids[0], list) else 0
        audio_s = float(req.get("duration_s") or n_tokens * self.sec_per_token)
        n_samples = max(1, int(round(audio_s * sr)))
        chunk_n = chunk_samples(sr)
        n_chunks = math.ceil(n_samples / chunk_n)
        t_start = time.monotonic()
        t_first: float | None = None
        seq = 0
        cancelled = False
        async with self._slots:  # admission: 8 streams at a time, extra requests queue
            await self._control(
                writer,
                {
                    "req_id": req_id,
                    "ack": "started",
                    "audio_s": round(audio_s, 3),
                    "chunks": n_chunks,
                    "chunk_samples": chunk_n,
                    "sr": sr,
                    "sid": int(req.get("sid", 0)),
                    "noise_scale": float(req.get("noise_scale", NOISE_SCALE_DEFAULT)),
                },
            )
            try:
                await asyncio.sleep(self.first_chunk_ms / 1000.0)  # encoder + first mel + vocoder
                for k in range(n_chunks):
                    if k > 0:
                        await asyncio.sleep(self.rtf * CHUNK_MS / 1000.0)  # compute pacing
                    off = k * chunk_n
                    m = min(chunk_n, n_samples - off)
                    # Every data frame is exactly one 40 ms chunk; the tail is zero-padded
                    # so payload length is implied by sr (docs/proto.md §3).
                    payload = sine_pcm(sr, off, m) + bytes(2 * (chunk_n - m))
                    writer.write(frame(seq, payload))
                    await writer.drain()  # backpressure: block until the socket buffer drains
                    if t_first is None:
                        t_first = time.monotonic()
                    seq += 1
            except asyncio.CancelledError:
                cancelled = True
            except (ConnectionResetError, BrokenPipeError):
                pass
            writer.write(frame(TERMINAL_SEQ, b""))
            await self._control(
                writer,
                {
                    "req_id": req_id,
                    "ack": "cancelled" if cancelled else "done",
                    "chunks_sent": seq,
                    "audio_ms_sent": seq * CHUNK_MS,
                    "ms_to_first_chunk": None if t_first is None else (t_first - t_start) * 1000.0,
                },
            )
            self._log(peer, f"req_id={req_id} -> {seq} chunks ({seq * CHUNK_MS} ms audio), cancelled={cancelled}")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--rtf", type=float, default=0.191, help="measured Nano single-stream RTF, sets chunk pacing")
    ap.add_argument("--first-chunk-ms", type=float, default=126.0, help="measured Nano mean first-chunk latency")
    ap.add_argument("--sec-per-token", type=float, default=SEC_PER_TOKEN)
    ap.add_argument("--max-streams", type=int, default=MAX_STREAMS)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    srv = MockTtsServer(
        args.host,
        args.port,
        rtf=args.rtf,
        first_chunk_ms=args.first_chunk_ms,
        sec_per_token=args.sec_per_token,
        max_streams=args.max_streams,
        log=not args.quiet,
    )
    await srv.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
