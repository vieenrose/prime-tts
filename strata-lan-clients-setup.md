# Strata Swift on the LAN host — client setup guide

Connect OpenCode, pi, and Claude Code on a LAN PC to the shared model server.

## Server facts

| | |
|---|---|
| Address (LAN) | `http://192.168.40.8:1919` (Tailscale: `http://100.98.177.63:1919`) |
| API key | `pesi28825252` (direct `:1919`) · bridge `:30002` accepts any key |
| Model id | `swift-1.5-iq3_s` |
| Context | 262144 tokens |
| Speed | ~228 tok/s decode (measured 400-token answer), 4 concurrent slots |

## Prerequisites (PC side)

- Pick the route that resolves on this PC: LAN `192.168.40.8` and Tailscale `100.98.177.63` are the
  same host. Verified from the linux box that runs this build: **LAN is unreachable, Tailscale works**,
  so its `~/.pi/agent/models.json` uses `http://100.98.177.63:1919/v1`. Test both before editing a
  client config; the LAN address will simply time out if the PC is not on that subnet.
- Reach the host: `curl -s -m 10 http://192.168.40.8:1919/v1/models -H "Authorization: Bearer pesi28825252"` must list `swift-1.5-iq3_s` as loaded. Timeout = firewall (allow TCP 1919 from this PC). `authentication_error` = retype the key by hand (no whitespace).
- Per-client tools below (Node/npm for OpenCode; `pi` binary; `claude` binary).

---

## 1. OpenCode

1. Install: `npm i -g opencode-ai` (skip if `opencode --version` works).
2. Write `~/.config/opencode/opencode.json` (merge with existing content, back it up first):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "strata/strata",
  "small_model": "strata/strata",
  "autoupdate": false,
  "compaction": { "auto": true, "buffer": 13672 },
  "share": "disabled",
  "provider": {
    "strata": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Strata Swift (LAN)",
      "options": {
        "baseURL": "http://192.168.40.8:1919/v1",
        "apiKey": "pesi28825252"
      },
      "models": {
        "strata": {
          "name": "Swift 1.5 IQ3_S (local)",
          "tool_call": true,
          "reasoning": true,
          "limit": { "context": 262144, "input": 258048, "output": 8192 }
        }
      }
    }
  }
}
```

Notes: model alias `strata` is intentional (the server serves its loaded model regardless of name); baseURL keeps the trailing `/v1`; `limit.input` (ctx − 4096) is required or OpenCode never compacts.

3. Verify: `opencode run "create /tmp/t.txt containing 408" < /dev/null` (first run downloads the npm plugin; allow a minute). `/tmp/t.txt` containing 408 = routing + auth + model + tool calls all proven.

## 2. pi coding agent

1. Merge into `~/.pi/agent/models.json` (back it up first):

```json
"strata": {
  "baseUrl": "http://192.168.40.8:1919/v1",
  "api": "openai-completions",
  "apiKey": "pesi28825252",
  "compat": { "supportsDeveloperRole": false, "supportsReasoningEffort": false },
  "models": [{
    "id": "swift-1.5-iq3_s", "reasoning": true, "input": ["text", "image"],
    "contextWindow": 262144, "maxTokens": 8192
  }]
}
```

Notes: `baseUrl` keeps `/v1`; model id is the real served id (unlike OpenCode's alias). The
`iq3_xxs` id is retired — a config still naming it fails with a model-not-found before any request
is sent, so check this line first if pi reports an unknown model. `input` lists `image` because the
server advertises `input_modalities: ["text", "image"]`; drop it back to `["text"]` if a client
build rejects image content parts.

2. Verify: `pi --provider strata --model swift-1.5-iq3_s`, ask a trivial question, then a tool task (create a file).

## 3. Claude Code via the LiteLLM bridge (Anthropic dialect: base URL has NO `/v1`)

Claude Code does not talk to Strata directly. It talks to a LiteLLM proxy on the host
(`:30002`), which holds the real Strata key server-side, sanitizes request shapes Strata
would reject (cache-control pileups, stray thinking params), and aliases the model id.
The proxy accepts any client credential, so stale logins or substituted keys on the PC
cannot 401 — if the PC reaches the port, it is served.

```sh
ANTHROPIC_BASE_URL="http://192.168.40.8:30002" \
ANTHROPIC_API_KEY="local" \
ANTHROPIC_AUTH_TOKEN="local" \
ANTHROPIC_DEFAULT_MODEL="swift-1.5-iq3_s" \
ANTHROPIC_MODEL="swift-1.5-iq3_s" \
ANTHROPIC_SMALL_FAST_MODEL="swift-1.5-iq3_s" \
ANTHROPIC_DEFAULT_SONNET_MODEL="swift-1.5-iq3_s" \
ANTHROPIC_DEFAULT_OPUS_MODEL="swift-1.5-iq3_s" \
ANTHROPIC_DEFAULT_HAIKU_MODEL="swift-1.5-iq3_s" \
CLAUDE_CODE_SUBAGENT_MODEL="swift-1.5-iq3_s" \
CLAUDE_CODE_SUBAGENT_MODEL_FORCE="1" \
CLAUDE_CODE_MAX_CONTEXT_TOKENS="256000" \
CLAUDE_CODE_MAX_OUTPUT_TOKENS="64000" \
API_TIMEOUT_MS="900000" \
CLAUDE_CODE_DISABLE_1M_CONTEXT="1" \
CLAUDE_CODE_AUTO_COMPACT_WINDOW="200000" \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1" \
MAX_THINKING_TOKENS="0" \
claude --permission-mode bypassPermissions
```

Notes:
- Port `30002` = LiteLLM bridge (direct `:1919` also works but skips sanitization). Key `local` is accepted blindly; the real key never leaves the host.
- `SUBAGENT_MODEL` + `FORCE=1` (+ `SMALL_FAST`, `DEFAULT_*`) pin every role — main, background, compaction summaries, whole subagent tree — on this model. Verify once: trigger a subagent and a `/compact`, confirm the host proxy log shows all traffic.
- `bypassPermissions`: no classifier is consulted (server-side verdicts arrive with proxy v1.107+; until then this avoids the fail-closed classifier path).
- `MAX_THINKING_TOKENS=0` keeps transcripts to `text`+`tool_use` only, so sessions resume cleanly (server-emitted `thinking` blocks carry empty signatures, which wedge resumes).
- One model per session: `/clear` when switching models; never resume a Qwen-worked transcript on Anthropic or vice versa.

Verify: trivial prompt, then a tool task (create a file).

---

## Appendix — troubleshooting

- Auth failures are architecturally impossible through the bridge (it accepts any credential). Failures mean plumbing: `curl .../30002/v1/models` non-200 = bridge down (check `:1919` direct health for Strata itself); timeouts on long turns = raise `API_TIMEOUT_MS`.
- `invalid x-api-key` wording (not `missing or wrong`): request went to real Anthropic — base URL wrong.
- Session wedged with a thinking/signature error: strip `thinking`/`redacted_thinking` blocks from `~/.claude/projects/<slug>/<session>.jsonl` (keep a backup) and resume the fixed file — or `Esc ×2` → `/rewind`, or `/clear`.
- Requests queue instead of parallelizing: server admits 4 concurrent slots; beyond that they wait (`GET /v1/status` → `concurrency.serving`).
