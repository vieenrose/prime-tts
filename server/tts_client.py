#!/usr/bin/env python3
"""Client for the Nano TTS TCP protocol (docs/proto.md): request, frames, cancel, ping.

The server pushes binary audio frames; control messages (ack, pong, done) are JSON
lines that only appear between frames, so the reader peeks one byte at a frame
boundary: 0x59 'Y' starts a frame header, '{' starts a control line.

Run (start server/mock_tts.py first, in another shell):
    .venv/bin/python server/mock_tts.py --port 8765
    .venv/bin/python server/tts_client.py --text "你好，歡迎来电" --out server/tts_out_8k.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time
import wave

from mock_tts import HEADER, MAGIC, TERMINAL_SEQ, chunk_samples

FIRST_BYTE_FRAME = 0x59  # low byte of MAGIC on the wire
FIRST_BYTE_CONTROL = 0x7B  # '{'


class TtsFrame:
    __slots__ = ("seq", "payload", "t")

    def __init__(self, seq: int, payload: bytes, t: float) -> None:
        self.seq = seq
        self.payload = payload
        self.t = t  # time.monotonic when the frame was fully read


class TtsSession:
    """One TCP connection = one session with one active stream (docs/proto.md §2)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, *, sr: int = 8000) -> None:
        self.host = host
        self.port = port
        self.sr = sr
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.events: list[dict] = []  # control-plane messages, in arrival order

    async def __aenter__(self) -> "TtsSession":
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        return self

    async def __aexit__(self, *exc) -> None:
        if self.writer is not None:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass

    async def request(
        self,
        req_id: int,
        text: str,
        ids: list[list[int]],
        *,
        sid: int = 0,
        sr: int | None = None,
        noise_scale: float = 0.667,
        duration_s: float | None = None,
    ) -> float:
        """Send one JSON request line; returns the wall time it left the client."""
        sr = sr or self.sr
        msg: dict = {
            "req_id": int(req_id),
            "text": text,
            "ids": [[int(v) for v in ids[0]], [int(v) for v in ids[1]], [int(v) for v in ids[2]]],
            "sid": int(sid),
            "sr": int(sr),
            "noise_scale": float(noise_scale),
        }
        if duration_s is not None:
            msg["duration_s"] = float(duration_s)
        assert self.writer is not None
        self.writer.write((json.dumps(msg) + "\n").encode())
        await self.writer.drain()
        return time.monotonic()

    async def cancel(self, req_id: int) -> float:
        assert self.writer is not None
        self.writer.write((json.dumps({"req_id": int(req_id), "cancel": True}) + "\n").encode())
        await self.writer.drain()
        return time.monotonic()

    async def ping(self) -> float:
        assert self.writer is not None
        self.writer.write((json.dumps({"ping": True}) + "\n").encode())
        await self.writer.drain()
        return time.monotonic()

    async def _read_event(self, head: bytes) -> dict:
        """Parse one control line, given its first byte (already consumed by the caller)."""
        assert self.reader is not None
        line = await self.reader.readline()
        obj = json.loads(head + line)
        obj["t_client"] = time.monotonic()  # client-side arrival time, for ack latency
        self.events.append(obj)
        return obj

    async def stream(self, *, sr: int | None = None, timeout_s: float = 30.0):
        """Yield TtsFrame objects for the current stream, then stop at the terminal frame.

        Control lines are collected into `events` as they appear. Payload lengths are
        implied by the chunk size for `sr` (docs/proto.md §3), the terminal frame
        (seq = 0xFFFFFFFF) carries no payload and ends the generator.
        """
        sr = sr or self.sr
        payload_n = chunk_samples(sr) * 2
        assert self.reader is not None
        async with asyncio.timeout(timeout_s):
            while True:
                b = await self.reader.read(1)
                if not b:
                    return
                if b[0] == FIRST_BYTE_CONTROL:
                    await self._read_event(b)
                    continue
                if b[0] != FIRST_BYTE_FRAME:
                    raise ValueError(f"bad frame magic byte {b!r}, stream is desynced")
                header = b + await self.reader.readexactly(7)
                magic, seq = HEADER.unpack(header)
                if magic != MAGIC:
                    raise ValueError(f"bad magic 0x{magic:08X}, stream is desynced")
                if seq == TERMINAL_SEQ:
                    return
                payload = await self.reader.readexactly(payload_n)
                yield TtsFrame(seq, payload, time.monotonic())


    async def drain_events(self, timeout_s: float = 2.0) -> list[dict]:
        """Read the control lines that follow the terminal frame (the done/cancelled ack)."""
        got: list[dict] = []
        assert self.reader is not None
        while True:
            try:
                b = await asyncio.wait_for(self.reader.read(1), timeout_s)
            except (TimeoutError, asyncio.TimeoutError):
                return got
            if not b:
                return got
            if b[0] != FIRST_BYTE_CONTROL:
                raise ValueError(f"expected a control line, got {b!r}")
            got.append(await self._read_event(b))


def ids_placeholder(text: str) -> list[list[int]]:
    """Stand-in ids for the mock: one token per character, sid 0, tone/lang 0.

    The production path builds these with model/scripts/frontend_bopomofo.py
    (bopomofo + arpabet -> phone/tone/lang ids). The mock only uses the token count.
    """
    n = max(1, len(text))
    return [[10] * n, [3] * n, [0] * n]


def write_wav(path: str, pcm: bytes, sr: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--text", default="你好，歡迎来电")
    ap.add_argument("--sr", type=int, default=8000)
    ap.add_argument("--req-id", type=int, default=1)
    ap.add_argument("--out", default="server/tts_out_8k.wav")
    ap.add_argument("--cancel-after-chunks", type=int, default=0, help="N>0: cancel after N chunks (barge-in test)")
    ap.add_argument("--verbose", action="store_true", help="print every chunk arrival")
    args = ap.parse_args()

    async with TtsSession(args.host, args.port, sr=args.sr) as s:
        t_req = await s.request(args.req_id, args.text, ids_placeholder(args.text), sr=args.sr)
        pcm = bytearray()
        t_first = None
        n_chunks = 0
        async for fr in s.stream(sr=args.sr):
            if t_first is None:
                t_first = fr.t
            n_chunks += 1
            pcm += fr.payload
            if args.verbose or n_chunks == 1 or n_chunks % 10 == 0:
                print(f"chunk {fr.seq}: {len(fr.payload)} bytes at +{(fr.t - t_req) * 1000:.1f} ms")
            if args.cancel_after_chunks and n_chunks >= args.cancel_after_chunks:
                t_c = await s.cancel(args.req_id)
                print(f"cancel sent at +{(t_c - t_req) * 1000:.1f} ms")
        t_end = time.monotonic()
        await s.drain_events()
        printable = [{k: v for k, v in e.items() if k != "t_client"} for e in s.events]
        print(f"ack/events: {printable}")
        print(
            f"request -> first audio byte: {(t_first - t_req) * 1000:.1f} ms"
            if t_first
            else "no audio received"
        )
        print(f"chunks={n_chunks} audio_ms={n_chunks * 40} wall={((t_end - t_req) * 1000):.1f} ms")
        write_wav(args.out, bytes(pcm), args.sr)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
