#!/usr/bin/env python3
"""OpenAI-compatible TTS gateway for the PrimeTTS v2.1 Nano server (text in, PCM16 out).

    POST /v1/audio/speech   {"model", "input", "voice", "response_format": "pcm"|"wav", "speed", ...}
    GET  /v1/audio/voices   proxied from the Nano
    GET  /health            gateway + Nano state

This is the half of server mode that cannot run in the Nano's C runtime: the zh-TW/en text frontend
(g2pw bopomofo + g2p_en arpabet + entity normalisation, model/scripts/frontend_bopomofo.py) needs
torch/transformers. The gateway turns text into the model's phone/tone/lang ids, interleaves the
blank token (v2.1 was trained with add_blank=true), and forwards them to `pt_v21_serve` as the
`input_ids` extension of the same request shape. Audio comes back as s16le mono 16 kHz.

Streaming: `input` is split into sentences; each sentence is synthesized as its own Nano request,
and with response_format=pcm its audio is written to the client as soon as it is ready, in order --
so the first audio of a long reply waits for the first sentence, not the whole paragraph.
Scheduling is earliest-deadline-first on both the frontend queue and the Nano's worker queue
(sentence deadline = arrival + estimated playback of the sentences before it), so under load every
request's first sentence runs before anyone's later ones. response_format=wav collects everything.

HuggingFace speech-to-speech:
    speech-to-speech local --tts openai --openai_tts_base_url http://<gateway>:8091/v1 \\
        --openai_tts_voice 0 --openai_tts_sample_rate 16000 --openai_tts_response_format pcm

Run (frontend deps live in server/.venv-gateway, see server/README-gateway.md):
    NLTK_DATA=<nltk_data dir> server/.venv-gateway/bin/python server/tts_gateway.py \\
        --nano http://100.84.151.116:8090 --port 8091
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import heapq
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SR = 16000
REPO = Path(__file__).resolve().parent.parent

def wav_header(n_bytes: int) -> bytes:
    return (b"RIFF" + struct.pack("<I", 36 + n_bytes) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16) + b"data" + struct.pack("<I", n_bytes))


class Frontend:
    """text -> blank-interleaved phone/tone/lang ids.

    g2pw is not re-entrant and contends badly across threads (GIL + torch intra-op threads), so all
    calls run on ONE worker thread fed by a small queue.
    """

    def __init__(self, scripts_dir: Path, add_blank: bool = True) -> None:
        sys.path.insert(0, str(scripts_dir))
        import frontend_bopomofo  # noqa: E402  (needs the gateway venv)
        self.F = frontend_bopomofo
        self.add_blank = add_blank
        self.cv = threading.Condition()
        self.heap: list = []
        self.seq = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _ids_now(self, text: str) -> dict | None:
        o = self.F.text_to_ids(text)
        ph, tn, lg = o["phone_ids"], o["tone_ids"], o["lang_ids"]
        if not ph:
            return None
        if self.add_blank:   # [0, x0, 0, x1, 0, ...]: what the v2.1 README feeds the ONNX model
            ph = [0] + [v for x in ph for v in (x, 0)]
            tn = [0] + [v for x in tn for v in (x, 0)]
            lg = [0] + [v for x in lg for v in (x, 0)]
        return {"phone": ph, "tone": tn, "lang": lg}

    def _run(self) -> None:
        while True:
            with self.cv:
                while not self.heap:
                    self.cv.wait()
                _, _, text, fut = heapq.heappop(self.heap)
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(self._ids_now(text))
            except Exception as e:  # noqa: BLE001
                fut.set_exception(e)

    def submit(self, deadline: float, text: str) -> cf.Future:
        fut: cf.Future = cf.Future()
        with self.cv:
            self.seq += 1
            heapq.heappush(self.heap, (deadline, self.seq, text, fut))
            self.cv.notify()
        return fut

    def ids(self, text: str) -> dict | None:
        return self.submit(0.0, text).result()


# speaking-rate estimate used only to order work (deadlines), never to shape audio
CHARS_PER_S_ZH = 5.0


def est_audio_s(text: str) -> float:
    zh = sum(1 for c in text if "\u3400" <= c <= "\u9fff")
    other_words = len(re.findall(r"[A-Za-z0-9]+", text))
    return zh / CHARS_PER_S_ZH + other_words * 0.35


class Gateway:
    """text -> ids (frontend) -> ONE Nano request -> streamed PCM back to the client.

    One request per utterance is what keeps prosody continuous and gives the model a single start:
    the Nano runs the encoder once and streams the decoder output in windows (overlap-save), so the
    audio is one continuous synthesis rather than per-sentence restarts.
    """

    def __init__(self, nano: str, frontend: Frontend, timeout: float) -> None:
        self.nano = nano.rstrip("/")
        self.fe = frontend
        self.timeout = timeout

    def stream(self, text: str, voice, speed: float, extra: dict):
        ids = self.fe.ids(text)
        if ids is None:
            return
        body = {"input_ids": ids, "voice": voice, "speed": speed, "response_format": "pcm", **extra}
        req = urllib.request.Request(self.nano + "/v1/audio/speech", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            while True:
                chunk = r.read(8192)     # returns as soon as bytes arrive: pass them straight on
                if not chunk:
                    return
                yield chunk


def make_handler(gw: Gateway):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "primetts-gateway/1"

        def log_message(self, fmt, *args):  # one line per request, to stderr
            sys.stderr.write("gateway: " + (fmt % args) + "\n")

        def end_headers(self):  # browsers fetch the public demo from another origin (CORS)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.end_headers()

        def _json(self, code: int, obj) -> None:
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        @staticmethod
        def _chain(first: bytes, rest):
            if first:
                yield first
            for c in rest:
                if c:
                    yield c

        def _err(self, code: int, msg: str) -> None:
            self._json(code, {"error": {"message": msg, "type": "invalid_request_error"}})

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/v1/audio/voices", "/audio/voices"):
                try:
                    with urllib.request.urlopen(gw.nano + "/v1/audio/voices", timeout=5) as r:
                        return self._json(200, json.loads(r.read()))
                except Exception as e:  # noqa: BLE001
                    return self._err(502, f"nano unreachable: {e}")
            if path == "/health":
                try:
                    with urllib.request.urlopen(gw.nano + "/health", timeout=5) as r:
                        nano = json.loads(r.read())
                except Exception as e:  # noqa: BLE001
                    nano = {"status": "unreachable", "error": str(e)}
                return self._json(200, {"status": "ok" if nano.get("status") == "ok" else "degraded",
                                        "nano": nano})
            return self._err(404, "not found")

        def do_POST(self):
            path = self.path.split("?")[0]
            if path not in ("/v1/audio/speech", "/audio/speech"):
                return self._err(404, "not found")
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._err(400, "body must be JSON")
            text = req.get("input")
            if not isinstance(text, str) or not text.strip():
                return self._err(400, "input must be a non-empty string")
            fmt = req.get("response_format", "pcm")
            if fmt not in ("pcm", "wav"):
                return self._err(400, "response_format must be pcm or wav")
            voice = req.get("voice", "0")
            try:
                speed = float(req.get("speed", 1.0))
            except (TypeError, ValueError):
                return self._err(400, "speed must be a number")
            extra = {k: req[k] for k in ("noise_scale",) if k in req}
            t0 = time.monotonic()
            chunks = gw.stream(text, voice, speed, extra)
            try:
                if fmt == "wav":
                    pcm = b"".join(chunks)
                    body = wav_header(len(pcm)) + pcm
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    first = time.monotonic()
                else:
                    # the first sentence is synthesized before the headers go out, so a Nano error
                    # on it is still a proper HTTP error; a later failure can only cut the stream
                    first_chunk = next(chunks, b"")
                    first = time.monotonic()
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/pcm")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    try:
                        for c in self._chain(first_chunk, chunks):
                            self.wfile.write(b"%x\r\n%s\r\n" % (len(c), c))
                            self.wfile.flush()
                        self.wfile.write(b"0\r\n\r\n")
                    except (urllib.error.URLError, TimeoutError) as e:
                        sys.stderr.write(f"gateway: stream cut after first sentence: {e}\n")
                        self.close_connection = True
                        return
            except urllib.error.HTTPError as e:
                return self._err(502, f"nano returned HTTP {e.code}: {e.read()[:200]!r}")
            except (urllib.error.URLError, TimeoutError) as e:
                return self._err(502, f"nano unreachable: {e}")
            except (BrokenPipeError, ConnectionResetError):
                return  # client went away (barge-in): stop writing
            sys.stderr.write(f"gateway: {len(text)} chars first-audio {1000 * (first - t0):.0f} ms "
                             f"total {1000 * (time.monotonic() - t0):.0f} ms fmt={fmt}\n")

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nano", default="http://127.0.0.1:8090", help="pt_v21_serve base URL")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--scripts", default=str(REPO / "model" / "scripts"), help="frontend_bopomofo.py dir")
    ap.add_argument("--no-blank", action="store_true", help="do not interleave the blank token")
    ap.add_argument("--g2pw-home", default=str(REPO / "model" / "scripts"),
                    help="dir holding (or receiving) g2pw's G2PWModel/: g2pw resolves it from the cwd")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    t = time.monotonic()
    os.chdir(args.g2pw_home)   # G2PWConverter() loads/downloads ./G2PWModel (607 MB): pin its home
    fe = Frontend(Path(args.scripts), add_blank=not args.no_blank)
    fe.ids("您好")   # load g2pw / g2p_en now, not on the first request
    fe.ids("Warmup")
    sys.stderr.write(f"gateway: frontend ready in {time.monotonic() - t:.1f}s; nano={args.nano}\n")
    gw = Gateway(args.nano, fe, args.timeout)
    srv = ThreadingHTTPServer((args.bind, args.port), make_handler(gw))
    srv.daemon_threads = True
    sys.stderr.write(f"gateway: listening on {args.bind}:{args.port}\n")
    srv.serve_forever()


if __name__ == "__main__":
    main()
