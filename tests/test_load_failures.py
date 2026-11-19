"""An engine that cannot start leaves a record that says so, and a half-downloaded model is not taken for one.

- `solvetime speed` records a failed first request as "does not start", as `solvetime run` does (oMLX's memory
  guard answers 507; the speed test used to stop with a traceback and an empty results file).
- LM Studio's model counts as downloaded only when its download has finished (a folder with a `.part` file left
  once passed the check, and the run then recorded LM Studio as not starting).
"""

import json
import os
import signal
import socket
import sys
import tempfile
import unittest
from unittest import mock

from solvetime import engines, speed
from solvetime.engines import Engine

SERVER = r"""
import http.server, json, sys

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data": [{"id": "m"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({"error": {"message": "Model 'm' (15.70GB) does not fit under the dynamic memory ceiling"}})
        self.send_response(507)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass

http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SpeedLoadFailure(unittest.TestCase):
    def test_a_refused_first_request_is_recorded(self):
        handler = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, handler)   # speed.main installs its own
        port = free_port()

        def fake(model, port):
            return Engine("fake", model, "m", [sys.executable, "-c", SERVER, str(port)], port, "m", start_timeout=60)

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(speed.ENGINES, {"fake": fake}), \
                mock.patch.object(speed, "missing", lambda engine: None), \
                mock.patch.object(speed, "tokenizer", lambda model: None), \
                mock.patch.object(speed, "calibrate", lambda: {}):
            speed.main(["--engine", "fake", "--model", "qwen3.5-9b", "--port", str(port), "--out", d])
            files = [f for f in os.listdir(d) if f.endswith(".jsonl")]
            self.assertEqual(len(files), 1)
            with open(os.path.join(d, files[0])) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual([r["test"] for r in rows], ["load"])
        self.assertRegex(rows[0]["load_failure"], r"^its first request failed: HTTP 507: .*does not fit")
        with self.assertRaises(OSError):   # the engine was stopped
            socket.create_connection(("127.0.0.1", port), timeout=2).close()


class LMStudioDownload(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.home = d.name
        patches = [mock.patch.object(engines, "HOME", self.home),
                   mock.patch.dict(os.environ, {"SOLVETIME_LMS": sys.executable})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.engine = engines.lmstudio("qwen3.8-27b", 8899)

    def folder(self, *files: str) -> str:
        path = os.path.join(self.home, ".lmstudio", "models", "lmstudio-community", "Qwen3.8-27B-MLX-4bit")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {"a.weight": "model-00001-of-00002.safetensors",
                                      "b.weight": "model-00002-of-00002.safetensors"}}, f)
        for name in files:
            open(os.path.join(path, name), "w").close()
        return path

    def test_not_downloaded(self):
        self.assertRegex(engines.missing(self.engine), r"is not downloaded\. ")

    def test_a_partial_download_is_not_one(self):
        path = self.folder("model-00001-of-00002.safetensors", "downloading_model-00002-of-00002.safetensors.part")
        self.assertRegex(engines.missing(self.engine), r"is not downloaded completely\. ")
        os.rename(os.path.join(path, "downloading_model-00002-of-00002.safetensors.part"),
                  os.path.join(path, "model-00002-of-00002.safetensors"))
        self.assertIsNone(engines.missing(self.engine))

    def test_a_missing_shard(self):
        self.folder("model-00001-of-00002.safetensors")
        self.assertRegex(engines.missing(self.engine), r"is not downloaded completely\. ")


class MTPLXDownload(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.home = d.name
        for p in (mock.patch.object(engines, "HOME", self.home), mock.patch.dict(os.environ, {"SOLVETIME_MTPLX": sys.executable})):
            p.start()
            self.addCleanup(p.stop)
        self.engine = engines.mtplx("qwen3.8-27b", 8896)

    def test_not_pulled(self):
        self.assertRegex(engines.missing(self.engine), r"is not downloaded\. `mtplx pull ")

    def test_half_pulled_then_complete(self):
        folder = os.path.join(self.home, ".mtplx", "models", "Youssofal--Qwen3.8-27B-MTPLX-Bare-Speed")
        os.makedirs(folder)
        with open(os.path.join(folder, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}, f)
        open(os.path.join(folder, "model-00001-of-00002.safetensors"), "w").close()
        self.assertRegex(engines.missing(self.engine), r"is not downloaded completely\. ")
        open(os.path.join(folder, "model-00002-of-00002.safetensors"), "w").close()
        self.assertIsNone(engines.missing(self.engine))


if __name__ == "__main__":
    unittest.main()
