"""A metering proxy between the agent and the engine.

Requests are passed through as the agent sent them (a non-streaming request
stays non-streaming), so the engine answers exactly as it would without the
proxy. Two things are changed, the same way for every engine: the sampling
options and the thinking switch (`Overrides`), so that no agent or engine
default makes runs differ.

Per request the proxy records the wall time, the engine's token counts, the
finish reason, any timing fields the engine adds to its reply (llama.cpp's
`timings`, for one), and fingerprints of the request's last message and of
the reply, from which a runner can tell an agent that repeats itself.
"""

from __future__ import annotations

import hashlib
import http.client
import http.server
import json
import threading
import time
import urllib.parse
from dataclasses import dataclass, field


@dataclass
class Overrides:
    """Options forced into every chat request. None leaves the agent's value alone."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    enable_thinking: bool | None = None
    max_tokens: int | None = None

    def apply(self, body: dict) -> dict:
        for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "max_tokens"):
            value = getattr(self, key)
            if value is not None:
                body[key] = value
        if self.enable_thinking is not None:
            kwargs = dict(body.get("chat_template_kwargs") or {})
            kwargs["enable_thinking"] = self.enable_thinking
            body["chat_template_kwargs"] = kwargs
        return body


def _fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


@dataclass
class Meter:
    rows: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, row: dict):
        with self.lock:
            self.rows.append(row)

    def take(self) -> list:
        with self.lock:
            rows, self.rows = self.rows, []
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


class Proxy:
    """Listens on 127.0.0.1:port (0: any free port) and forwards to `upstream` (http://host:port)."""

    def __init__(self, upstream: str, overrides: Overrides | None = None, port: int = 0, timeout: float = 1800):
        self.upstream = urllib.parse.urlparse(upstream)
        self.overrides = overrides or Overrides()
        self.meter = Meter()
        self.timeout = timeout
        proxy = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # quiet
                pass

            def _forward(self, method: str):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                row = None
                if method == "POST" and self.path.rstrip("/").endswith("/chat/completions") and raw:
                    body = proxy.overrides.apply(json.loads(raw))
                    raw = json.dumps(body).encode()
                    messages = body.get("messages") or []
                    row = {"t_start": time.time(), "stream": bool(body.get("stream")), "messages": len(messages),
                           "last_message": _fingerprint(messages[-1] if messages else None)}
                headers = {k: v for k, v in self.headers.items()
                           if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
                headers["Content-Length"] = str(len(raw))
                conn = http.client.HTTPConnection(proxy.upstream.hostname, proxy.upstream.port, timeout=proxy.timeout)
                t0 = time.perf_counter()
                try:
                    conn.request(method, self.path, raw if raw else None, headers)
                    resp = conn.getresponse()
                    self.send_response(resp.status)
                    chunked = resp.getheader("Transfer-Encoding", "").lower() == "chunked" or row and row["stream"]
                    for k, v in resp.getheaders():
                        if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                            self.send_header(k, v)
                    if chunked:
                        self._relay_stream(resp, row, t0)
                    else:
                        data = resp.read()
                        if row is not None:  # recorded before the client has the reply, so a reader never misses it
                            row["seconds"] = time.perf_counter() - t0
                            row["status"] = resp.status
                            self._account(row, data)
                            proxy.meter.add(row)
                            row = None
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                except (OSError, http.client.HTTPException) as e:
                    if row is not None and "seconds" not in row:
                        row.update(seconds=time.perf_counter() - t0, status=0, error=f"{type(e).__name__}: {e}")
                    try:
                        self.send_error(502, f"upstream: {e}")
                    except OSError:
                        pass
                finally:
                    conn.close()
                    if row is not None and not row.get("_added"):
                        proxy.meter.add(row)

            def _relay_stream(self, resp, row, t0):
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                first, text, usage, finish = None, "", None, None
                buf = b""
                while True:
                    piece = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                    if not piece:
                        break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))
                    self.wfile.flush()
                    if row is None:
                        continue
                    buf += piece
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line.startswith(b"data:") or line == b"data: [DONE]":
                            continue
                        try:
                            ev = json.loads(line[5:])
                        except ValueError:
                            continue
                        usage = ev.get("usage") or usage
                        for ch in ev.get("choices") or []:
                            d = ch.get("delta") or {}
                            if first is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                                first = time.perf_counter() - t0
                            text += d.get("content") or ""
                            finish = ch.get("finish_reason") or finish
                if row is not None:  # recorded before the stream's last chunk reaches the client
                    row.update(seconds=time.perf_counter() - t0, status=resp.status, ttft=first, finish=finish,
                               reply=_fingerprint(text), **_usage(usage))
                    proxy.meter.add(row)
                    row["_added"] = True
                self.wfile.write(b"0\r\n\r\n")

            def _account(self, row, data):
                try:
                    reply = json.loads(data)
                except ValueError:
                    row["reply"] = _fingerprint(data.decode("utf-8", "replace"))
                    return
                choice = (reply.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                row.update(finish=choice.get("finish_reason"),
                           reply=_fingerprint([message.get("content"), message.get("tool_calls")]),
                           **_usage(reply.get("usage")))
                for key in ("timings", "metrics", "x_timings"):  # engine-specific timing reports, kept as they are
                    if key in reply:
                        row[key] = reply[key]

            def do_POST(self):
                self._forward("POST")

            def do_GET(self):
                self._forward("GET")

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self) -> "Proxy":
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def _usage(usage: dict | None) -> dict:
    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    return {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": details.get("cached_tokens")}
