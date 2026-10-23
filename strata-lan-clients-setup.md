# GLM-5.3-Flash (Maya-M) on the LAN host — client setup guide

Connect OpenCode, pi, and Claude Code on a LAN PC to the shared model server (Maya-M, GLM-5.3-Flash).

## Server facts

| | |
|---|---|
| Address (LAN) | `http://192.168.40.8:8090` (Tailscale: `http://100.98.177.63:8090`) |
| LiteLLM bridge (Claude Code) | `http://192.168.40.8:30002` (Tailscale: `http://100.98.177.63:30002`), any key (`local`) |
| API key | `pesi28825252` (direct `:8090`) · bridge `:30002` accepts any key |
| Model id | `glm-5.3-flash` (Maya-M) |
| Context | 131072 tokens |
| Speed | ~51 tok/s decode, 131072 ctx (measured 400-token answer), several requests at once (up to 8) |

## Prerequisites (PC side)

- Reach the host: `curl -s -m 10 http://192.168.40.8:8090/v1/models -H "Authorization: Bearer pesi28825252"` must list `glm-5.3-flash` as loaded. Timeout = firewall (allow TCP 8090 from this PC). `authentication_error` = retype the key by hand (no whitespace).
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
      "name": "PLACEHOLDER_KEEP",
      "options": {
        "baseURL": "http://192.168.40.8:8090/v1",
        "apiKey": "pesi28825252"
      },
      "models": {
        "strata": {
          "name": "GLM-5.3 Flash Maya-M (local)",
          "tool_call": true,
          "reasoning": true,
          "limit": { "context": 131072, "input": 126976, "output": 8192 }
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
  "baseUrl": "http://192.168.40.8:8090/v1",
  "api": "openai-completions",
  "apiKey": "pesi28825252",
  "compat": { "supportsDeveloperRole": false, "supportsReasoningEffort": false },
  "models": [{
    "id": "glm-5.3-flash", "reasoning": true, "input": ["text"],
    "contextWindow": 131072, "maxTokens": 8192
  }]
}
```

Notes: `baseUrl` keeps `/v1`; model id is the real served id (unlike OpenCode's alias).

2. Verify: `pi --provider strata --model glm-5.3-flash`, ask a trivial question, then a tool task (create a file).

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
ANTHROPIC_DEFAULT_MODEL="glm-5.3-flash" \
ANTHROPIC_MODEL="glm-5.3-flash" \
ANTHROPIC_SMALL_FAST_MODEL="glm-5.3-flash" \
ANTHROPIC_DEFAULT_SONNET_MODEL="glm-5.3-flash" \
ANTHROPIC_DEFAULT_OPUS_MODEL="glm-5.3-flash" \
ANTHROPIC_DEFAULT_HAIKU_MODEL="glm-5.3-flash" \
CLAUDE_CODE_SUBAGENT_MODEL="glm-5.3-flash" \
CLAUDE_CODE_SUBAGENT_MODEL_FORCE="1" \
CLAUDE_CODE_MAX_CONTEXT_TOKENS="131072" \
CLAUDE_CODE_MAX_OUTPUT_TOKENS="64000" \
API_TIMEOUT_MS="900000" \
CLAUDE_CODE_DISABLE_1M_CONTEXT="1" \
CLAUDE_CODE_AUTO_COMPACT_WINDOW="110000" \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1" \
MAX_THINKING_TOKENS="0" \
claude --permission-mode bypassPermissions
```

Notes:
- Port `30002` = LiteLLM bridge (direct `:8090` also works but skips sanitization). Key `local` is accepted blindly; the real key never leaves the host.
- `SUBAGENT_MODEL` + `FORCE=1` (+ `SMALL_FAST`, `DEFAULT_*`) pin every role — main, background, compaction summaries, whole subagent tree — on this model. Verify once: trigger a subagent and a `/compact`, confirm the host proxy log shows all traffic.
- `bypassPermissions`: no classifier is consulted (server-side verdicts arrive with proxy v1.107+; until then this avoids the fail-closed classifier path).
- `MAX_THINKING_TOKENS=0` keeps transcripts to `text`+`tool_use` only, so sessions resume cleanly (server-emitted `thinking` blocks carry empty signatures, which wedge resumes).
- One model per session: `/clear` when switching models; never resume a Qwen-worked transcript on Anthropic or vice versa.

Verify: trivial prompt, then a tool task (create a file).

---

## Appendix — troubleshooting

- Which path: OpenCode and pi go direct to `:8090` (proven, fastest). Only Claude Code goes through the `:30002` bridge (auth leniency + request sanitization its harness needs).

- Auth failures are architecturally impossible through the bridge (it accepts any credential). Failures mean plumbing: `curl .../30002/v1/models` non-200 = bridge down (check `:8090` direct health for Strata itself); timeouts on long turns = raise `API_TIMEOUT_MS`.
- `invalid x-api-key` wording (not `missing or wrong`): request went to real Anthropic — base URL wrong.
- Session wedged with a thinking/signature error: strip `thinking`/`redacted_thinking` blocks from `~/.claude/projects/<slug>/<session>.jsonl` (keep a backup) and resume the fixed file — or `Esc ×2` → `/rewind`, or `/clear`.
- Requests queue instead of parallelizing: server admits 4 concurrent slots; beyond that they wait (`GET /v1/status` → `concurrency.serving`).
