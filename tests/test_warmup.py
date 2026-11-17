"""An engine that answers its first request with an error does not start: the error is not an empty reply."""

import http.server
import json
import threading
import unittest

from solvetime.run import first_token, thinking_switch


class Engine(http.server.BaseHTTPRequestHandler):
    status = 200

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.status != 200:
            body = json.dumps({"error": {"message": "Model (15.70GB) does not fit under the dynamic memory ceiling"}})
            self.send_response(self.status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ("Hel", "lo"):
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


class Warmup(unittest.TestCase):
    def serve(self, status):
        handler = type("H", (Engine,), {"status": status})
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def test_a_reply(self):
        port = self.serve(200)
        _, content, _ = first_token(port, "m", [{"role": "user", "content": "Say hello."}])
        self.assertEqual(content, "Hello")

    def test_an_error_raises(self):
        port = self.serve(507)
        with self.assertRaisesRegex(RuntimeError, "HTTP 507: .*does not fit"):
            first_token(port, "m", [{"role": "user", "content": "Say hello."}])
        self.assertTrue(thinking_switch(port, "m").startswith("probe failed: HTTP 507"))


if __name__ == "__main__":
    unittest.main()
