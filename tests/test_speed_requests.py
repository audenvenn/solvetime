"""The speed test's requests fit an engine like LM Studio: it returns max_tokens - 1 tokens, and it ignores
chat_template_kwargs.enable_thinking (only its own field, reasoning_effort "none", turns thinking off).

With a 1-token prefill request and no reasoning_effort, the 27B's speed test on 2026-11-19 got no token back."""

import json
import os
import signal
import socket
import sys
import tempfile
import unittest
from unittest import mock

from solvetime import speed
from solvetime.engines import Engine

SERVER = r"""
import http.server, json, sys

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"data": [{"id": "m"}]}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        n = max(0, body.get("max_tokens", 16) - 1)                   # LM Studio's: one fewer than asked
        thinking = body.get("reasoning_effort") != "none"            # chat_template_kwargs is ignored
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        field = "reasoning_content" if thinking else "content"
        for _ in range(n):
            chunk = {"choices": [{"delta": {field: "w "}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        usage = {"choices": [], "usage": {"prompt_tokens": 4100, "completion_tokens": n}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass

http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


class Words:
    def encode(self, text):
        return type("Encoding", (), {"ids": text.split()})()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LMStudioLike(unittest.TestCase):
    def test_prefill_and_decode_come_back(self):
        handler = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, handler)
        port = free_port()

        def fake(model, port):
            return Engine("fake", model, "m", [sys.executable, "-c", SERVER, str(port)], port, "m", start_timeout=60,
                          no_thinking={"reasoning_effort": "none"})

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(speed.ENGINES, {"fake": fake}), \
                mock.patch.object(speed, "missing", lambda engine: None), \
                mock.patch.object(speed, "tokenizer", lambda model: Words()), \
                mock.patch.object(speed, "calibrate", lambda: {}):
            speed.main(["--engine", "fake", "--model", "qwen3.5-9b", "--port", str(port), "--out", d, "--runs", "1"])
            path = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jsonl")][0]
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        tests = [r["test"] for r in rows]
        self.assertNotIn("failure", tests, rows[-1])
        self.assertEqual(tests.count("prefill"), 1)
        decodes = [r for r in rows if r["test"].startswith("decode")]
        self.assertEqual(len(decodes), 4)
        for r in decodes:   # the reply's text, not reasoning: thinking was off
            self.assertEqual(r["completion_tokens"], 511)


    def test_reasoning_tokens_count(self):
        # A model that keeps thinking with thinking off (Inkling 2 at reasoning_effort "none" writes a thinking block,
        # 2026-11-21): its reasoning was timed, so it is counted, and recorded as reasoning.
        handler = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, handler)
        port = free_port()

        def fake(model, port):   # no fields of its own for "off": the fake keeps thinking
            return Engine("fake", model, "m", [sys.executable, "-c", SERVER, str(port)], port, "m", start_timeout=60)

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(speed.ENGINES, {"fake": fake}), \
                mock.patch.object(speed, "missing", lambda engine: None), \
                mock.patch.object(speed, "tokenizer", lambda model: Words()), \
                mock.patch.object(speed, "calibrate", lambda: {}):
            speed.main(["--engine", "fake", "--model", "qwen3.5-9b", "--port", str(port), "--out", d, "--runs", "1"])
            path = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jsonl")][0]
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        decodes = [r for r in rows if r["test"].startswith("decode")]
        self.assertEqual(len(decodes), 4)
        for r in decodes:
            self.assertEqual((r["completion_tokens"], r["reasoning_tokens"]), (511, 511))
            self.assertGreater(r["decode_tok_s"], 0)


if __name__ == "__main__":
    unittest.main()
