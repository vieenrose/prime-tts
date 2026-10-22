#!/usr/bin/env python3
"""Offline end-to-end demo of the hybrid voice pipeline, with per-stage timings.

    16 kHz zh-TW wav -> gemini-3.8-live transcript -> gemini-3.8-flash reply text
    -> TTS request to the Nano TTS service (mock by default)

Headline number: wall ms from the end of the user's speech (last mic chunk) to the
first TTS audio byte on the socket, plus the per-stage breakdown.

Run (repo root, after `source ./gemini-live-credential.txt`):
    .venv/bin/python server/make_input_wav.py
    .venv/bin/python server/pipeline.py --wav server/input_zh_greeting_16k.wav
The mock Nano server starts inside this process; use --no-mock to point at a
server/mock_tts.py started separately.
"""
from __future__ import annotations

import argparse
import asyncio
import time

import live_asr
import mock_tts
import reply_gen
import tts_client


async def run(
    wav_path: str,
    *,
    reply_model: str = "gemini-3.8-flash",
    language: str = "zh-TW",
    port: int = 8765,
    out_wav: str = "server/pipeline_tts_8k.wav",
    mock: bool = True,
    mock_first_chunk_ms: float = 126.0,
    mock_rtf: float = 0.191,
) -> dict:
    """Run the three legs once and return the measured timings (ms) and token usage."""
    srv = None
    if mock:
        srv = mock_tts.MockTtsServer(
            "127.0.0.1", port, rtf=mock_rtf, first_chunk_ms=mock_first_chunk_ms, log=False
        )
        await srv.start()
    try:
        # (1) ASR leg
        asr = await live_asr.transcribe(wav_path, language_codes=[language] if language else None)
        if asr.t_final is None:
            raise RuntimeError("ASR leg produced no final transcript (check the Live session log)")
        t_speech_end = asr.t_last_chunk  # end of user speech, in wall time

        # (2) reply leg
        reply = await reply_gen.generate_reply(asr.text, reply_model)
        if reply.error:
            raise RuntimeError(f"reply leg failed: {reply.error}")

        # (3) TTS leg (mock Nano, real protocol)
        async with tts_client.TtsSession("127.0.0.1", port, sr=8000) as tts:
            t_req = await tts.request(
                1, reply.text, tts_client.ids_placeholder(reply.text), sr=8000, noise_scale=0.667
            )
            pcm = bytearray()
            t_first = t_last = None
            n_chunks = 0
            async for fr in tts.stream(sr=8000):
                n_chunks += 1
                t_first = fr.t if t_first is None else t_first
                t_last = fr.t
                pcm += fr.payload
            t_done = time.monotonic()
            await tts.drain_events()
        tts_client.write_wav(out_wav, bytes(pcm), 8000)

        ack = next((e for e in tts.events if e.get("req_id") == 1 and e.get("ack") == "started"), {})
        fin = next((e for e in tts.events if e.get("req_id") == 1 and e.get("ack") in ("done", "cancelled")), {})
        summary = live_asr.usage_summary(asr.usage)
        return {
            "asr": asr,
            "reply": reply,
            "tts_events": [{k: v for k, v in e.items() if k != "t_client"} for e in tts.events],
            "timings_ms": {
                "asr_last_chunk_to_final_transcript": asr.ms(asr.t_final),
                "asr_last_chunk_to_turn_complete": asr.ms(asr.t_turn_complete),
                "transcript_to_reply_call": (reply.t_start - asr.t_final) * 1000.0,
                "reply_time_to_first_token": reply.ttft_ms,
                "reply_generation_window": reply.gen_s * 1000.0,
                "reply_end_to_tts_request": (t_req - reply.t_end) * 1000.0,
                "tts_request_to_ack": None if not ack else (ack["t_client"] - t_req) * 1000.0,
                "tts_request_to_first_audio_byte": None if t_first is None else (t_first - t_req) * 1000.0,
                "tts_request_to_last_audio_byte": None if t_last is None else (t_last - t_req) * 1000.0,
                "e2e_end_of_speech_to_first_tts_audio_byte": None
                if t_first is None
                else (t_first - t_speech_end) * 1000.0,
                "e2e_end_of_speech_to_last_tts_audio_byte": None
                if t_last is None
                else (t_last - t_speech_end) * 1000.0,
            },
            "tts": {
                "chunks": n_chunks,
                "audio_ms": n_chunks * mock_tts.CHUNK_MS,
                "wall_ms": (t_done - t_req) * 1000.0,
                "ack": ack,
                "final": fin,
                "out_wav": out_wav,
            },
            "tokens": {
                "live_audio_in": summary["prompt_modality_tokens"].get("AUDIO"),
                "live_audio_out_discarded": summary["response_modality_tokens"].get("AUDIO"),
                "live_prompt_total": summary["prompt_token_count"],
                "live_response_total": summary["response_token_count"],
                "reply_completion": reply.completion_tokens,
                "reply_usage": reply.usage,
            },
        }
    finally:
        if srv is not None:
            await srv.stop()


def report(r: dict) -> None:
    asr: live_asr.AsrResult = r["asr"]
    reply: reply_gen.ReplyResult = r["reply"]
    t = r["timings_ms"]
    live_asr.report(asr)
    print()
    reply_gen.report(reply)
    print()
    print("== TTS leg (mock Nano, protocol docs/proto.md)")
    print(f"   chunks={r['tts']['chunks']} audio_ms={r['tts']['audio_ms']} "
          f"wall_ms={r['tts']['wall_ms']:.1f} ack={r['tts']['ack']} final={r['tts']['final']}")
    print(f"   audio written to {r['tts']['out_wav']}")
    print()
    print("== per-stage breakdown (ms, wall clock)")
    for k, v in t.items():
        print(f"   {k:45s} {v:.1f}" if v is not None else f"   {k:45s} n/a")
    print()
    print("== tokens")
    for k, v in r["tokens"].items():
        print(f"   {k:28s} {v}")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", required=True, help="mono 16-bit PCM WAV, 16 kHz (see make_input_wav.py)")
    ap.add_argument("--reply-model", default="gemini-3.8-flash")
    ap.add_argument("--language", default="zh-TW", help="BCP-47 code for the ASR leg; empty = omit")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out-wav", default="server/pipeline_tts_8k.wav")
    ap.add_argument("--no-mock", action="store_true", help="use an external server/mock_tts.py")
    ap.add_argument("--mock-first-chunk-ms", type=float, default=126.0)
    ap.add_argument("--mock-rtf", type=float, default=0.191)
    args = ap.parse_args()
    r = await run(
        args.wav,
        reply_model=args.reply_model,
        language=args.language,
        port=args.port,
        out_wav=args.out_wav,
        mock=not args.no_mock,
        mock_first_chunk_ms=args.mock_first_chunk_ms,
        mock_rtf=args.mock_rtf,
    )
    report(r)


if __name__ == "__main__":
    asyncio.run(main())
