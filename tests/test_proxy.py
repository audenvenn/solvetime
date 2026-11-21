"""The metering proxy against a fake engine: python -m pytest tests (or python tests/test_proxy.py)."""

import http.client
import http.server
import json
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solvetime.proxy import Overrides, Proxy  # noqa: E402
from solvetime.run import repeats  # noqa: E402


class FakeEngine(http.server.BaseHTTPRequestHandler):
    """Echoes what it was sent: the request body comes back in the reply's content."""

    protocol_version = "HTTP/1.1"
    seen = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeEngine.seen.append(body)
        usage = {"prompt_tokens": 11, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 5}}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for ev in ({"choices": [{"delta": {"content": "he"}}]}, {"choices": [{"delta": {"content": "llo"}}]},
                       {"choices": [{"delta": {}, "finish_reason": "stop"}]}, {"choices": [], "usage": usage}):
                data = f"data: {json.dumps(ev)}\n\n".encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.write(b"%x\r\n%s\r\n" % (len(b"data: [DONE]\n\n"), b"data: [DONE]\n\n"))
            self.wfile.write(b"0\r\n\r\n")
            return
        reply = json.dumps({"choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
                            "usage": usage, "timings": {"prompt_ms": 7.0}}).encode()
        if body.get("chunked_json"):   # a JSON reply sent chunked, as oMLX 0.8 sends them
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for part in (reply[:20], reply[20:]):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
            self.wfile.write(b"0\r\n\r\n")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


class ProxyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        cls.proxy = Proxy(f"http://127.0.0.1:{cls.engine.server_address[1]}",
                          Overrides(temperature=0.6, top_k=20, enable_thinking=False)).start()

    @classmethod
    def tearDownClass(cls):
        cls.proxy.stop()
        cls.engine.shutdown()

    def post(self, body):
        port = int(self.proxy.url.rsplit(":", 1)[1])
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data

    def test_overrides_and_accounting(self):
        self.proxy.meter.take()
        status, data = self.post({"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 1.5,
                                  "chat_template_kwargs": {"keep": 1}})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["choices"][0]["message"]["content"], "hello")
        sent = FakeEngine.seen[-1]
        self.assertEqual(sent["temperature"], 0.6)
        self.assertEqual(sent["top_k"], 20)
        self.assertEqual(sent["chat_template_kwargs"], {"keep": 1, "enable_thinking": False})
        (row,) = self.proxy.meter.take()
        self.assertEqual((row["prompt_tokens"], row["completion_tokens"], row["cached_tokens"]), (11, 3, 5))
        self.assertEqual(row["finish"], "stop")
        self.assertEqual(row["timings"], {"prompt_ms": 7.0})

    def test_a_json_reply_sent_chunked_is_accounted(self):
        # oMLX sends its JSON replies chunked; the meter read only data: lines from them and lost their usage
        # (rounds 2 and 3, found 2026-11-21)
        self.proxy.meter.take()
        status, data = self.post({"model": "m", "messages": [{"role": "user", "content": "hi"}], "chunked_json": True})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["choices"][0]["message"]["content"], "hello")
        (row,) = self.proxy.meter.take()
        self.assertEqual((row["prompt_tokens"], row["completion_tokens"]), (11, 3))
        self.assertEqual(row["finish"], "stop")

    def test_streaming_passes_through(self):
        self.proxy.meter.take()
        status, data = self.post({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True})
        self.assertEqual(status, 200)
        text = "".join(json.loads(line[5:])["choices"][0]["delta"].get("content", "")
                       for line in data.decode().splitlines()
                       if line.startswith("data: {") and json.loads(line[5:])["choices"])
        self.assertEqual(text, "hello")
        (row,) = self.proxy.meter.take()
        self.assertTrue(row["stream"])
        self.assertIsNotNone(row["ttft"])
        self.assertEqual(row["completion_tokens"], 3)

    def test_repeats(self):
        rows = [{"reply": r} for r in "abacab"]
        self.assertEqual(repeats(rows, [])["max_reply_repeats"], 3)          # not consecutive, still counted

        def call(i, cmd):
            return {"role": "assistant", "tool_calls": [{"id": str(i), "function": {
                "arguments": json.dumps({"command": cmd})}}]}

        def result(i, out):
            return {"role": "tool", "tool_call_id": str(i), "content": out}

        msgs = [call(1, "ls"), result(1, "a.py"), call(2, "python3 t.py"), result(2, "FAIL"),
                call(3, "python3 t.py"), result(3, "ok"), call(4, "ls"), result(4, "a.py")]
        self.assertEqual(repeats([], msgs)["max_action_repeats"], 2)        # the test run changed its output


if __name__ == "__main__":
    unittest.main()
