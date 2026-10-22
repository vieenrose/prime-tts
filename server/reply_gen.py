#!/usr/bin/env python3
"""Stage 2 of the voice pipeline: user transcript -> contact-centre reply text.

Streaming generateContent (streamGenerateContent under the hood) with a zh-TW
contact-centre system prompt. Reports time-to-first-token, output tokens/sec and the
completion token count, which is what the cost model needs.

Run (repo root, after `source ./gemini-live-credential.txt`):
    .venv/bin/python server/reply_gen.py --transcript "你好，我想問這個月的話費為什麼多了五百元"
    .venv/bin/python server/reply_gen.py --models gemini-3.8-flash gemini-3.8-flash-lite
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass, field

import google.genai as genai
from google.genai import types

MODELS_DEFAULT = ["gemini-3.8-flash", "gemini-3.8-flash-lite"]
DEFAULT_TRANSCRIPT = "你好，我想問這個月的話費為什麼多了五百元？"

SYSTEM_PROMPT_ZH_TW = (
    "你是台灣電信客服中心的語音助理。用繁體中文（zh-TW）口語回答，語氣禮貌、簡潔、專業。"
    "回答要適合直接用語音播出：不超過兩句，不用項目符號、不用 markdown、不用英文標點。"
    "不出個人識別資料，不收敏感資訊。遇到需要查帳的請求，先確認用戶的帳號末五碼，"
    "再說明下一步。只用繁體中文回答。"
)


@dataclass
class ReplyResult:
    model: str
    text: str = ""
    chunks: int = 0
    t_start: float = 0.0
    t_first_token: float | None = None
    t_end: float = 0.0
    usage: dict = field(default_factory=dict)
    completion_tokens: int | None = None
    error: str | None = None

    @property
    def ttft_ms(self) -> float | None:
        return None if self.t_first_token is None else (self.t_first_token - self.t_start) * 1000.0

    @property
    def gen_s(self) -> float:
        """Wall time of the token-producing window (first token -> stream end)."""
        return 0.0 if self.t_first_token is None else self.t_end - self.t_first_token

    @property
    def tokens_per_s(self) -> float | None:
        if self.completion_tokens is None or self.gen_s <= 0:
            return None
        return self.completion_tokens / self.gen_s


def _redact(msg: str) -> str:
    """Strip the API key out of an error string (SDK errors echo the request)."""
    key = os.environ.get("GEMINI_API_KEY", "")
    return msg.replace(key, "<redacted>") if key else msg


async def generate_reply(
    transcript: str,
    model: str,
    *,
    system_instruction: str = SYSTEM_PROMPT_ZH_TW,
    max_output_tokens: int = 200,
    temperature: float = 0.4,
    timeout_s: float = 60.0,
) -> ReplyResult:
    """Stream the reply text and time it. Errors are captured, not raised."""
    res = ReplyResult(model=model)
    client = genai.Client()
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    res.t_start = time.monotonic()
    try:
        async with asyncio.timeout(timeout_s):
            stream = await client.aio.models.generate_content_stream(
                model=model, contents=[transcript], config=config
            )
            async for chunk in stream:
                res.t_end = time.monotonic()
                res.chunks += 1
                text = chunk.text or ""
                if text:
                    if res.t_first_token is None:
                        res.t_first_token = time.monotonic()
                    res.text += text
                if chunk.usage_metadata is not None:
                    res.usage = chunk.usage_metadata.model_dump(mode="json", exclude_none=True)
    except (TimeoutError, asyncio.TimeoutError):
        res.error = f"timeout after {timeout_s} s"
    except Exception as exc:  # API errors: report the exact message, key redacted
        res.error = _redact(f"{type(exc).__name__}: {' '.join(str(exc).split())}")
    if res.usage.get("candidates_token_count") is not None:
        res.completion_tokens = int(res.usage["candidates_token_count"])
    elif res.usage.get("response_token_count") is not None:
        res.completion_tokens = int(res.usage["response_token_count"])
    return res


def report(res: ReplyResult) -> None:
    print(f"== model={res.model}")
    if res.error:
        print(f"   error: {res.error}")
        return
    ttft = res.ttft_ms
    tps = res.tokens_per_s
    print(f"   reply: {res.text!r}")
    print(
        f"   chunks={res.chunks}  time-to-first-token={ttft:.1f} ms  "
        f"generation window={res.gen_s * 1000:.1f} ms  tokens/sec={tps}  "
        f"completion tokens={res.completion_tokens}"
    )
    print(f"   usageMetadata={res.usage}")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transcript", default=DEFAULT_TRANSCRIPT)
    ap.add_argument("--transcript-file", help="read the transcript from a file (overrides --transcript)")
    ap.add_argument("--models", nargs="+", default=MODELS_DEFAULT)
    ap.add_argument("--max-output-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.4)
    args = ap.parse_args()

    transcript = args.transcript
    if args.transcript_file:
        transcript = open(args.transcript_file, encoding="utf-8").read().strip()
    print(f"transcript: {transcript!r}")
    for model in args.models:
        report(await generate_reply(transcript, model,
                                   max_output_tokens=args.max_output_tokens,
                                   temperature=args.temperature))


if __name__ == "__main__":
    asyncio.run(main())
