"""An error inside a stream is an error (2026-11-22).

MTPLX answers a streamed request it refuses with HTTP 200 and an event {"error": {...}}, which is how its memory guard
sends a refusal. The probe and the speed test skipped events without choices. So the probe took a refusal for an empty
reply and recorded "ignored: never thinks", and a refusal after some tokens would have passed for a whole reply."""

import http.server
import json
import threading
import unittest

from solvetime import speed
from solvetime.run import first_token, stream_error, thinking_switch

REFUSAL = {"error": {"message": "insufficient memory: the other apps on this Mac leave 2.1 GiB free", "type": "server_error"}}


class Server(http.server.BaseHTTPRequestHandler):
    events: list = []

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for ev in self.events:
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


def token(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


class StreamErrors(unittest.TestCase):
    def serve(self, events: list) -> int:
        handler = type("H", (Server,), {"events": events})
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        return httpd.server_address[1]

    def test_the_message(self):
        self.assertEqual(stream_error(REFUSAL["error"]),
                         "error in the stream: insufficient memory: the other apps on this Mac leave 2.1 GiB free")
        self.assertEqual(stream_error("overloaded"), "error in the stream: overloaded")

    def test_a_refusal_is_not_an_empty_reply(self):
        port = self.serve([REFUSAL])
        with self.assertRaisesRegex(RuntimeError, "insufficient memory"):
            first_token(port, "m", [{"role": "user", "content": "hi"}])
        self.assertEqual(thinking_switch(port, "m")[:46], "probe failed: error in the stream: insufficien")

    def test_a_refusal_after_some_tokens_is_not_a_whole_reply(self):
        port = self.serve([token("17"), token(" times"), REFUSAL])
        with self.assertRaisesRegex(RuntimeError, "insufficient memory"):
            speed.stream(port, "m", [{"role": "user", "content": "hi"}], 64, 0.0)

    def test_a_whole_reply_still_counts(self):
        port = self.serve([token("391")])
        _, content, _ = first_token(port, "m", [{"role": "user", "content": "hi"}])
        self.assertEqual(content, "391")
        self.assertEqual(speed.stream(port, "m", [{"role": "user", "content": "hi"}], 64, 0.0)[2], "391")


if __name__ == "__main__":
    unittest.main()
