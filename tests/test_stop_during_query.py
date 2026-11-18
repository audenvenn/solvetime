"""A SIGTERM that arrives while the agent waits for the engine ends the run at once: mini-swe-agent's retries must
not swallow it (they catch every exception they are not told to abort on)."""

import glob
import http.server
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAKE = r'''
import http.server, json, sys, time
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if body.get("stream"):        # the harness's own probes: answer at once
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            self.wfile.write(b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\ndata: [DONE]\n\n')
        else:                         # the agent's query: an engine that never answers
            time.sleep(600)
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


class StopDuringQuery(unittest.TestCase):
    def test_sigterm_during_a_query_ends_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "bin"))
            open(os.path.join(d, "fake.py"), "w").write(FAKE)
            server = os.path.join(d, "bin", "llama-server")
            open(server, "w").write(f"#!/bin/bash\nexec {sys.executable} {d}/fake.py \"$@\"\n")
            os.chmod(server, 0o755)
            open(os.path.join(d, "fake.gguf"), "w").write("x")
            code = (f"import sys; sys.path.insert(0, {HERE!r})\nimport solvetime.run as r\n"
                    "r.calibrate = lambda: {'calibration': 'skipped in this test'}\n"
                    f"r.main(['--engine', 'llama.cpp', '--model', 'qwen3.6-35b-a3b', '--tasks', 'inventory', "
                    f"'--attempts', '1', '--out', {d!r} + '/out', '--port', '8979'])\n")
            env = dict(os.environ, PATH=os.path.join(d, "bin") + ":" + os.environ["PATH"],
                       SOLVETIME_GGUF=os.path.join(d, "fake.gguf"))
            p = subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, start_new_session=True)
            try:
                deadline = time.time() + 120
                while time.time() < deadline and not glob.glob(os.path.join(d, "out", "*", "trajectories")):
                    time.sleep(0.5)
                time.sleep(8)                # the agent's first query is now waiting on the engine
                p.send_signal(signal.SIGTERM)
                p.wait(30)
            finally:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
                subprocess.run(["pkill", "-f", os.path.join(d, "fake.py")])
            self.assertEqual(p.returncode, 128 + signal.SIGTERM)
            meta = json.load(open(glob.glob(os.path.join(d, "out", "*", "run.json"))[0]))
            self.assertEqual(meta.get("stopped"), "SIGTERM")


if __name__ == "__main__":
    unittest.main()
