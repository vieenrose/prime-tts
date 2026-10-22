#!/usr/bin/env python3
"""Perception leg B: audio-in straight to a text model (no Live, no audio-out billed).

Measures: time to first token, transcript+reply token counts, and $/conversation-minute,
so the Live-ASR leg (server/live_asr.py) can be compared against it with numbers.

  .venv/bin/python server/audio_in_flash.py --wav server/input_zh_greeting_16k.wav
"""
import argparse, base64, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402


PROMPT_TEXT = (
    "Caller audio is attached. Follow the instructions in the system message."
)
PROMPT = (
    "You are the ASR+reply stage of a zh-TW contact-centre agent. "
    "Step 1: transcribe the caller's speech verbatim. Step 2: on a new line, write the "
    "assistant's spoken reply to the caller in 自然的中繁體中文, 1-2 short sentences, no markdown, "
    "no emoji, plain text suitable for TTS. Format:\nTRANSCRIPT: <text>\nREPLY: <text>"
)


async def run(wav_path, model, stream=True):
    key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=key)
    data = open(wav_path, "rb").read()
    audio = types.Part.from_bytes(data=data, mime_type="audio/wav")
    t0 = time.perf_counter()
    ttft = None
    text = ""
    usage = None
    cfg = { "temperature": 0.2, "max_output_tokens": 400}
    # SDK 1.x GenerateContentConfig has no thinking_level; keep the config it accepts.
    if stream:
        it = await client.aio.models.generate_content_stream(model=model, contents=[audio, PROMPT_TEXT], config=cfg)
        async for chunk in it:
            if ttft is None and chunk.text:
                ttft = time.perf_counter()
            if chunk.text:
                text += chunk.text
            if getattr(chunk, "usage_metadata", None):
                usage = chunk.usage_metadata
    else:
        r = await client.aio.models.generate_content(model=model, contents=[audio, PROMPT_TEXT], config=cfg)
        ttft = time.perf_counter()
        text = r.text or ""
        usage = r.usage_metadata
    t1 = time.perf_counter()
    u = {}
    if usage is not None:
        d = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
        u = d
    return dict(total_s=t1 - t0, ttft_s=(ttft - t0) if ttft else None, text=text, usage=u)


def cost(u, text_in_rate, audio_in_rate, text_out_rate):
    """Token counts come in prompt_tokens_details (per modality), not named fields."""
    ti = ai = to = 0
    for d in (u.get("prompt_tokens_details") or []):
        mod = (d.get("modality") or "").upper()
        n = d.get("token_count") or 0
        if mod == "AUDIO":
            ai += n
        elif mod == "TEXT":
            ti += n
    to = u.get("candidates_token_count") or u.get("output_tokens") or 0
    return (ti * text_in_rate + ai * audio_in_rate + to * text_out_rate) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="server/input_zh_greeting_16k.wav")
    ap.add_argument("--models", default="gemini-3.8-flash,gemini-3.5-flash-lite")
    ap.add_argument("--no-stream", action="store_true")
    a = ap.parse_args()
    import wave
    with wave.open(a.wav, "rb") as w:
        dur = w.getnframes() / w.getframerate()
    print(f"input wav: {a.wav}  {dur:.3f} s of caller speech")
    # published paid $/1M tokens
    RATES = {"gemini-3.8-flash": (0.30, 1.00, 2.50), "gemini-3.5-flash-lite": (0.10, 1.00, 0.40)}
    for m in a.models.split(","):
        try:

            import asyncio
            r = asyncio.run(run(a.wav, m, stream=not a.no_stream))
        except Exception as e:  # noqa: BLE001
            print(f"{m}: ERROR {type(e).__name__}: {str(e)[:400]}")
            continue
        u = r["usage"]
        rates = RATES.get(m, (0.30, 1.00, 2.50))
        c = cost(u, *rates)
        print(f"\n== {m} ==")
        print(f"  total {r['total_s']*1000:.0f} ms   ttft {(r['ttft_s'] or 0)*1000:.0f} ms")
        print(f"  usage {json.dumps(u)}")
        print(f"  audio-in tokens {sum((d.get('token_count') or 0) for d in (u.get('prompt_tokens_details') or []) if (d.get('modality') or '').upper()=='AUDIO')} = {dur:.2f}s x 25/s = {dur*25:.0f}")
        print(f"  api cost for this turn: ${c:.6f}  -> ${c/dur*60:.5f} per caller-minute")
        print("  text:\n" + "\n".join("    " + l for l in (r["text"] or "").splitlines()))


main()
