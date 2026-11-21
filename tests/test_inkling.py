"""Inkling 2 Mini (2026-11-21): its template ignores enable_thinking and reads reasoning_effort ("high" when absent),
and its card gives only temperature 1.0. The run, its thinking-switch probe and the speed test send its own switch;
the Qwen models' requests do not change. SOLVETIME_LLAMA_SERVER picks the llama-server build."""

import http.server
import json
import os
import threading
import unittest
from unittest import mock

from solvetime import engines, models
from solvetime.run import sampling, thinking_switch


class Fields(unittest.TestCase):
    def test_inkling_requests(self):
        on, off = sampling("inkling-2-mini", "thinking"), sampling("inkling-2-mini", "no-thinking")
        body_on, body_off = on.apply({}), off.apply({})
        for body in (body_on, body_off):
            self.assertEqual((body["temperature"], body["top_p"], body["top_k"], body["min_p"]), (1.0, 1.0, 0, 0.0))
            self.assertNotIn("enable_thinking", json.dumps(body))
        self.assertNotIn("reasoning_effort", body_on)                       # the template's default: high
        # a template variable only: llama.cpp gives a top-level reasoning_effort precedence and renders the default
        self.assertNotIn("reasoning_effort", body_off)
        self.assertEqual(body_off["chat_template_kwargs"], {"reasoning_effort": "none"})
        self.assertEqual(models.thinking_fields("inkling-2-mini", "thinking"), {})
        self.assertEqual(models.thinking_fields("inkling-2-mini", "no-thinking", {"x": 1}), models.INKLING2_OFF)

    def test_qwen_requests_unchanged(self):
        self.assertEqual(models.thinking_fields("qwen3.8-27b", "thinking"), {"chat_template_kwargs": {"enable_thinking": True}})
        self.assertEqual(models.thinking_fields("qwen3.8-27b", "no-thinking", {"reasoning_effort": "none"}),
                         {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"})
        body = sampling("qwen3.8-27b", "no-thinking").apply({})
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_params_and_builds(self):
        m = models.model("inkling-2-mini")
        self.assertEqual(m.params, 26_070_390_312)
        self.assertEqual(models.build("inkling-2-mini", "omlx"), "mlx-community/Inkling-2-Mini-oQ3e-mtp")


class Probe(unittest.TestCase):
    def test_the_probe_sends_the_given_fields(self):
        seen = []

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': '391'}}]})}\n\ndata: [DONE]\n\n".encode())

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        result = thinking_switch(server.server_address[1], "m", models.thinking_fields("inkling-2-mini", "no-thinking"),
                                 models.thinking_fields("inkling-2-mini", "thinking"))
        self.assertEqual(seen[0]["chat_template_kwargs"], {"reasoning_effort": "none"})
        self.assertNotIn("reasoning_effort", json.dumps(seen[1]))
        self.assertNotIn("enable_thinking", json.dumps(seen))
        self.assertEqual(result, "ignored: never thinks")                # no reasoning either way, from this fake


class LlamaServer(unittest.TestCase):
    def test_the_build_to_run(self):
        env = {"SOLVETIME_GGUF": "/m/inkling.gguf", "SOLVETIME_LLAMA_SERVER": "/opt/llama-b12600/llama-server"}
        with mock.patch.dict(os.environ, env):
            e = engines.llama_cpp("inkling-2-mini", 8890)
        self.assertEqual(e.command[0], "/opt/llama-b12600/llama-server")
        self.assertEqual(e.version_command, ["/opt/llama-b12600/llama-server", "--version"])
        with mock.patch.dict(os.environ, {"SOLVETIME_GGUF": "/m/inkling.gguf"}):
            os.environ.pop("SOLVETIME_LLAMA_SERVER", None)
            self.assertEqual(engines.llama_cpp("inkling-2-mini", 8890).command[0], "llama-server")


if __name__ == "__main__":
    unittest.main()
