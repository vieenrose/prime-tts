# Cost per caller-minute, measured on this pipeline

Prices are the published paid-tier rates (per 1M tokens, USD, Gemini Developer API):
3.8 Live audio-in $3.00 ($0.005/min), audio-out $12.00 ($0.018/min), text-out $4.50, text-in $0.75;
3.8 Flash text-in $0.30, text-out $2.50, audio-in $1.00; 3.5 Flash-Lite text-in $0.10, text-out $0.40;
3.8 Flash-Lite TTS audio-out $6.00. Audio = 25 tokens/second. Local TTS on the Nano costs no tokens.

## Measured token counts (one turn: 2.608 s of caller speech, 3.32 s of agent speech)

| leg | measured tokens | how |
|---|---|---|
| Live `gemini-3.8-live` audio-in | **65** (= 2.608 s x 25/s) | `server/live_asr.py` |
| Live discarded spoken reply (audio-out) | **561** (3.32 s reply -> the model spoke ~22 s of audio) | `usage_metadata` at `turn_complete` |
| Live per-session prompt overhead | **628 text + 65 audio** | same |
| Direct audio-in to `gemini-3.8-flash` | **66 audio + 14 text in, 56 thoughts, 17 out** | `server/audio_in_flash.py` |
| Direct audio-in to `gemini-3.5-flash-lite` | **66 audio + 14 text in, 15 out** | same |
| Local TTS leg | 0 tokens | Nano runtime |

Facts found while measuring (they change the design):
- `gemini-3.8-live` supports **only** the AUDIO response modality. There is no text-out from it;
  text of the user comes from `input_audio_transcription`, and text of the model from
  `output_transcription`, and both are in addition to the billed audio output.
- The Live model always **generates** its spoken reply, so a "speech-in / text-out" leg pays for
  audio output it throws away, and pays the time to generate it.
- `gemini-3.8-flash-lite` does not exist: `models.list` omits it and `models.get` returns
  HTTP 404 `Model is not found: models/gemini-3.8-flash-lite for api version v1beta`.
  `gemini-3.8-flash-lite-tts` does exist. 3.5 Flash-Lite is the working cheap text model.
- Live needs ~0.8 s of trailing silence in the mic stream before it closes the turn.

## Cost per caller-minute (measured tokens, published rates)

| architecture | $/caller-minute | vs native |
|---|---|---|
| native `gemini-3.8-live` speech-to-speech | **$0.0274** | 1.0x |
| hybrid via Live ASR (audio-in + discarded audio-out) + 3.8 Flash reply + local TTS | **$0.1727** | **6.3x MORE expensive** |
| audio-in -> 3.8 Flash (+56 thoughts) + local TTS | **$0.00259** | **10.6x cheaper** |
| audio-in -> 3.5 Flash-Lite + local TTS | **$0.00169** | **16.2x cheaper** |
| text-out + Gemini cloud TTS (3.8 Flash-Lite TTS audio-out) | $0.0141 | 1.9x cheaper |

Arithmetic for the winner (3.5 Flash-Lite row): audio-in 66 x $1.00/1M = $0.000066, text-in 14 x
$0.10/1M = $0.0000014, text-out 15 x $0.40/1M = $0.000006 -> $0.0000734 per 2.608 s turn
= $0.00169 per caller-minute. TTS is free locally, so the only Gemini cost is the audio you send.

The Live-ASR row is expensive because the discarded reply is 91% of that leg's cost: 561 audio-out
tokens x $12/1M = $0.0067 per turn, which is more than the entire native audio-to-audio turn
($0.0012). Use Live only when you need its turn-taking/barge-in behaviour, not as the ASR leg.

## Latency (measured, same turn)

| path | end of caller speech -> first agent audio byte |
|---|---|
| Live ASR -> 3.8 Flash -> TTS (mock) | 4721 ms (4267 ms of it is waiting for the Live final transcript) |
| audio-in -> 3.5 Flash-Lite -> TTS | ~2.6 s to first reply token, then TTS first byte 63-140 ms (real Nano, 1 stream) |
| audio-in -> 3.8 Flash -> TTS | 4.84 s to first token (56 thinking tokens); with `thinking` reduced this is the same order as Lite |

## zh-TW notes

- Live ASR transcribed the zh-TW input correctly (`input_transcription`), 3.8 Flash replied in
  natural zh-TW. 3.5 Flash-Lite returned character-spaced Chinese output
  (`你 好 ， 欢 迎 ...`) in one measured call - it needs a formatting guard before TTS.
- Live native audio: Google lists 24-70 languages for Live voice output but the response language
  cannot be pinned; zh-TW output quality is the thing the local PrimeTTS model replaces.
