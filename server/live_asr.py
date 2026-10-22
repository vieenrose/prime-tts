#!/usr/bin/env python3
"""Stage 1 of the voice pipeline: user speech -> user transcript, on gemini-3.8-live.

Streams a 16 kHz mono s16le PCM WAV to the Live API with
`input_audio_transcription` enabled and response_modalities=["AUDIO"], which is the
only modality set the native-audio Live models accept. The model's own spoken reply
is generated and billed, then discarded here: this stage only wants the user
transcript, so the discarded audio-out token count is reported exactly.

Run (repo root, after `source ./gemini-live-credential.txt`):
    .venv/bin/python server/live_asr.py --wav server/input_zh_greeting_16k.wav
"""
from __future__ import annotations

import argparse
import asyncio
import time
import wave
from dataclasses import dataclass, field

import google.genai as genai
from google.genai import types

MODEL = "gemini-3.8-live"
CHUNK_MS = 100  # realtime pacing of the simulated microphone
AUDIO_TOKENS_PER_S = 25  # published rate for Live audio, used for cross-checks


def read_pcm16(path: str) -> tuple[bytes, int]:
    """Read a mono, 16-bit little-endian, uncompressed WAV as raw PCM bytes."""
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getcomptype() != "NONE":
            raise ValueError(
                f"{path}: need mono 16-bit PCM WAV, got ch={w.getnchannels()} "
                f"w={w.getsampwidth()} comp={w.getcomptype()}"
            )
        return w.readframes(w.getnframes()), w.getframerate()


@dataclass
class AsrResult:
    text: str = ""
    finished: bool | None = None
    language_code: str | None = None
    interims: list[dict] = field(default_factory=list)  # {t_ms, text}
    usage_events: list[dict] = field(default_factory=list)  # raw usageMetadata dicts
    usage: dict = field(default_factory=dict)  # last (cumulative) usageMetadata seen
    t_last_chunk: float = 0.0  # wall time the last audio chunk was sent
    t_first_interim: float | None = None
    t_final: float | None = None
    t_turn_complete: float | None = None
    audio_out_bytes: int = 0
    audio_out_mime: str | None = None
    reply_text_parts: list[str] = field(default_factory=list)
    audio_in_bytes: int = 0
    audio_s: float = 0.0
    connect_s: float = 0.0
    session_s: float = 0.0
    language_codes_sent: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def ms(self, t: float | None) -> float | None:
        """ms from the last audio chunk to time t."""
        return None if t is None else (t - self.t_last_chunk) * 1000.0


def _modality_tokens(details) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in details or []:
        modality = str(getattr(d, "modality", "")).replace("MODALITY_", "")
        out[modality] = int(getattr(d, "token_count", 0) or 0)
    return out


def usage_summary(usage: dict) -> dict:
    return {
        "prompt_token_count": usage.get("prompt_token_count"),
        "response_token_count": usage.get("response_token_count"),
        "total_token_count": usage.get("total_token_count"),
        "thoughts_token_count": usage.get("thoughts_token_count"),
        "prompt_modality_tokens": _modality_tokens(usage.get("prompt_tokens_details")),
        "response_modality_tokens": _modality_tokens(usage.get("response_tokens_details")),
    }


async def transcribe(
    wav_path: str,
    *,
    model: str = MODEL,
    language_codes: list[str] | None = None,
    staff_silent: bool = False,
    realtime: bool = True,
    tail_s: float = 8.0,
    timeout_s: float = 60.0,
) -> AsrResult:
    """Send the whole WAV over Live as realtime mic input; return transcript + usage.

    realtime=True paces chunks 100 ms apart in wall time, so the gaps measured here
    are what a live caller would see. tail_s is how long reception stays open after
    the final transcript, so the discarded spoken reply is fully generated and its
    audio-out tokens are final.
    """
    pcm, sr = read_pcm16(wav_path)
    res = AsrResult(audio_in_bytes=len(pcm), audio_s=len(pcm) / 2 / sr)
    chunk_bytes = int(sr * CHUNK_MS / 1000) * 2
    client = genai.Client()

    trans_cfg: dict = {}
    if language_codes:
        trans_cfg["language_codes"] = list(language_codes)
        res.language_codes_sent = list(language_codes)
    config = {"response_modalities": ["AUDIO"], "input_audio_transcription": trans_cfg}
    if staff_silent:
        # The staff-recommended workaround on discuss.ai.google.dev (post 7 of thread 113843):
        # native-audio models need the AUDIO output channel open, so keep it open and tell the
        # model to stay silent. Whether it actually stops generating (and paying for) audio is
        # measured by the token details below.
        config["speech_config"] = {"voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}}
        config["system_instruction"] = {"parts": [{"text": (
            "You are a transcription service for a contact centre. Listen to the caller and DO NOT "
            "SPEAK: produce no spoken reply of any kind. Your only output is the verbatim transcript "
            "of the caller's speech, produced by input_audio_transcription.")}]}
        res.notes.append("staff_silent config applied (speech_config Puck + silent system_instruction)")

    t_connect0 = time.monotonic()
    async with client.aio.live.connect(model=model, config=config) as session:
        res.connect_s = time.monotonic() - t_connect0
        res.notes.append(
            f"connected in {res.connect_s * 1000:.0f} ms, "
            f"input_audio_transcription={trans_cfg}, response_modalities=['AUDIO']"
        )

        async def sender() -> None:
            t0 = time.monotonic()
            for i in range(0, len(pcm), chunk_bytes):
                # Pace at stream rate: chunk k leaves at t0 + k*CHUNK_MS.
                if realtime:
                    due = t0 + (i // chunk_bytes) * CHUNK_MS / 1000
                    await asyncio.sleep(max(0.0, due - time.monotonic()))
                # send_realtime_input(media=...) is for text/image parts; realtime *audio*
                # must go through audio= (media= audio is accepted by the SDK and silently
                # ignored by the server: no interim_input_transcription, no usage).
                await session.send_realtime_input(
                    audio=types.Blob(data=pcm[i:i + chunk_bytes], mime_type=f"audio/pcm;rate={sr}")
                )
            res.t_last_chunk = time.monotonic()
            await session.send_realtime_input(audio_stream_end=True)

        async def receiver() -> None:
            # session.receive() blocks forever when the model goes silent (no turn events at all),
            # which is exactly the staff-recommended silent-transcriber config. Bound every message
            # wait by `quiet_s` after the mic stream ends so a turn can always be closed.
            gen = session.receive()
            quiet_s = max(8.0, tail_s)
            while True:
                try:
                    msg = await asyncio.wait_for(gen.__anext__(), timeout=quiet_s)
                except (asyncio.TimeoutError, StopAsyncIteration):
                    res.notes.append(f"no server message for {quiet_s:.1f} s -> closing turn")
                    return
                now = time.monotonic()
                sc = msg.server_content
                if sc is not None:
                    if sc.interim_input_transcription is not None:
                        txt = getattr(sc.interim_input_transcription, "text", "") or ""
                        res.interims.append({"t_ms": (now - res.t_last_chunk) * 1000.0, "text": txt})
                        if res.t_first_interim is None:
                            res.t_first_interim = now
                    if sc.input_transcription is not None:
                        tr = sc.input_transcription
                        res.text = tr.text or ""
                        res.finished = tr.finished
                        res.language_code = tr.language_code
                        if res.t_final is None:
                            res.t_final = now
                    if sc.model_turn is not None:
                        for part in sc.model_turn.parts or []:
                            inline = getattr(part, "inline_data", None)
                            if inline is not None and inline.data:
                                res.audio_out_bytes += len(inline.data)
                                res.audio_out_mime = inline.mime_type or res.audio_out_mime
                            elif getattr(part, "text", None):
                                res.reply_text_parts.append(part.text)
                    if sc.turn_complete:
                        res.t_turn_complete = now
                if msg.usage_metadata is not None:
                    usage = msg.usage_metadata.model_dump(mode="json", exclude_none=True)
                    res.usage_events.append(usage)
                    res.usage = usage  # Live usageMetadata is cumulative per session
                # Stay in the tail window so the discarded spoken reply finishes:
                # its audio-out token count is the number we have to report exactly.
                if res.t_final is not None and now - res.t_final > tail_s:
                    return

        async with asyncio.timeout(timeout_s):
            # Live is bidirectional: the spoken reply arrives while mic chunks go out.
            await asyncio.gather(sender(), receiver())

    res.session_s = time.monotonic() - t_connect0
    if res.t_last_chunk == 0.0:
        res.notes.append("sender did not finish; timings are relative to stream end only")
    return res


def report(res: AsrResult) -> None:
    print(
        f"model={MODEL} in={res.audio_s:.3f} s ({res.audio_in_bytes / 2:.0f} samples "
        f"at 16 kHz), connect {res.connect_s * 1000:.0f} ms, session wall {res.session_s:.2f} s"
    )
    print(f"transcript (finished={res.finished}, language_code={res.language_code}): {res.text!r}")
    print(f"interim transcripts: {len(res.interims)}")
    for it in res.interims:
        print(f"  {it['t_ms']:8.1f} ms vs last chunk  {it['text']!r}")
    print(f"last audio chunk -> first interim   : {res.ms(res.t_first_interim)} ms")
    if res.t_final is None:
        print("no final transcript received (t_final=None)")
    else:
        print(f"last audio chunk -> final transcript: {res.ms(res.t_final):.1f} ms")
    print(f"last audio chunk -> turn_complete   : {res.ms(res.t_turn_complete)} ms")
    print(
        f"discarded spoken reply: {res.audio_out_bytes} audio bytes "
        f"(mime {res.audio_out_mime}, ~{res.audio_out_bytes / 2 / 24000:.2f} s at 24 kHz), "
        f"reply text: {''.join(res.reply_text_parts)!r}"
    )
    for i, u in enumerate(res.usage_events):
        print(f"usage event {i}: {u}")
    print("final usageMetadata summary:", usage_summary(res.usage))
    summary = usage_summary(res.usage)
    pin = summary["prompt_modality_tokens"].get("AUDIO")
    pout = summary["response_modality_tokens"].get("AUDIO")
    if pin is not None:
        print(f"audio-in tokens {pin} vs 25/s x {res.audio_s:.2f} s = {25 * res.audio_s:.0f}")
    if pout is not None:
        print(
            f"AUDIO OUT TOKENS OF THE DISCARDED SPOKEN REPLY = {pout} "
            f"(at 25 tokens/s = {pout / 25:.2f} s of speech)"
        )
    for n in res.notes:
        print("note:", n)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", required=True, help="mono 16-bit PCM WAV, 16 kHz")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument(
        "--language",
        default="zh-TW",
        help="BCP-47 code passed to input_audio_transcription.language_codes; empty string omits it",
    )
    ap.add_argument("--no-realtime", action="store_true", help="send all chunks as fast as possible")
    ap.add_argument("--tail-s", type=float, default=8.0)
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument("--staff-silent", action="store_true",
                    help="apply the staff-recommended silent-transcriber config (docs/cost.md probe)")
    args = ap.parse_args()
    codes = [args.language] if args.language else None
    staff_silent = args.staff_silent
    res = await transcribe(
        args.wav,
        model=args.model,
        language_codes=codes,
        staff_silent=args.staff_silent,
        realtime=not args.no_realtime,
        tail_s=args.tail_s,
        timeout_s=args.timeout_s,
    )
    report(res)


if __name__ == "__main__":
    asyncio.run(main())
