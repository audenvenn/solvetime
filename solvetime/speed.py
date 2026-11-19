"""Prefill and decode speed of one engine: solvetime speed --engine kapitsa --model qwen3.6-35b-a3b

The same requests for every engine, over its OpenAI-compatible API, thinking
off, after one uncounted warm-up:

- prefill: a 4K-token prompt (data/prefill_4k.txt, Python standard library
  source, and a one-line question), a unique first line per request so that
  no prefix cache helps, timed to the first token;
- decode, two prompts: writing a module from a request ("write") and
  continuing a module ("continue"), up to 512 tokens each, at temperature 0.6
  (top-p 0.95, top-k 20) and at temperature 0, timed from the first token to
  the last.

Tokens are counted with the Qwen tokenizer for every engine alike (the
engine's own counts are kept too). While each decode runs, the engine's
process tree is sampled: RSS (resident pages, memory-mapped weights
included) and phys_footprint (which leaves clean mapped pages out). Results:
results/speed-<run>.jsonl, one line per measurement.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid

from .engines import ENGINES, harness_version, missing
from .models import MODELS, model as model_entry
from .run import calibrate, first_token, memory_at_start, memory_counters, stop_on_term, wait_up

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = {
    "write": ("Write a complete Python module that implements a thread-safe LRU cache with per-entry time-to-live "
              "expiry. Include type hints, docstrings, and a unittest test suite with at least ten tests. Output only "
              "the code."),
    # A continuation that cannot end early: the docstring lists twenty functions and two are written, so
    # writing the rest takes well over 512 tokens (the first prompt of this kind, which ended a single
    # function, came back after 9 to 26 tokens with thinking off).
    "continue": ("Continue this Python module: implement, in order, every function its docstring lists that is not "
                 "written yet. Output only code.\n\n"
                 '"""Small helpers for lists of numbers.\n\nFunctions: mean, median, mode, variance, stdev, minimum, '
                 "maximum, value_range, clamp, normalize, moving_average, cumulative_sum, differences, percentile, "
                 'zscores, histogram, deduplicate, chunks, interleave, rotate.\n"""\n\n'
                 "from __future__ import annotations\n\nimport math\nfrom collections import Counter\n\n\n"
                 "def mean(xs: list[float]) -> float:\n    \"\"\"The arithmetic mean.\"\"\"\n    return sum(xs) / len(xs)\n\n\n"
                 "def median(xs: list[float]) -> float:\n    \"\"\"The middle value, or the mean of the two middle values.\"\"\"\n"
                 "    s = sorted(xs)\n    n = len(s)\n    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2\n\n\n"),
}
SAMPLING = {0.6: {"top_p": 0.95, "top_k": 20}, 0.0: {}}


def tokenizer(model: str):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    return Tokenizer.from_file(hf_hub_download(model_entry(model).tokenizer, "tokenizer.json"))


def prefill_body() -> str:
    lines = open(os.path.join(HERE, "data", "prefill_4k.txt"), encoding="utf-8").read().splitlines(keepends=True)
    return "".join(line for line in lines if not line.startswith("# Prefill text") and not line.startswith("# Python is"))


def system_available() -> int:
    """Memory the system could hand out now: free, inactive, speculative and purgeable pages (vm_stat)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    size = int(re.search(r"page size of (\d+) bytes", out).group(1))
    pages = 0
    for name in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        m = re.search(rf"{name}:\s+(\d+)", out)
        pages += int(m.group(1)) if m else 0
    return pages * size


class MemorySampler:
    """RSS and phys_footprint of a process tree, sampled every second; the peaks. Also the least memory the
    system had available meanwhile: on a quiet machine, available-before-start minus that is everything the
    engine takes (its own memory, GPU buffers, and the pages of memory-mapped weights alike)."""

    def __init__(self, pid: int):
        self.pid, self.rss, self.footprint, self.stop = pid, 0, 0, threading.Event()
        self.available_min = system_available()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _tree(self):
        pids, todo = [], [self.pid]
        while todo:
            p = todo.pop()
            pids.append(p)
            out = subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout.split()
            todo += [int(x) for x in out]
        return pids

    def _run(self):
        while not self.stop.is_set():
            pids = self._tree()
            out = subprocess.run(["ps", "-o", "rss=", "-p", ",".join(map(str, pids))], capture_output=True, text=True)
            self.rss = max(self.rss, sum(int(x) for x in out.stdout.split()) * 1024)
            fp = 0
            for p in pids:
                m = re.search(r"phys_footprint:\s*([0-9]+)", subprocess.run(
                    ["footprint", "-f", "bytes", "-p", str(p)], capture_output=True, text=True).stdout)
                fp += int(m.group(1)) if m else 0
            self.footprint = max(self.footprint, fp)
            self.available_min = min(self.available_min, system_available())
            self.stop.wait(1.0)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def stream(port: int, model: str, messages: list, max_tokens: int, temperature: float, timeout: float = 900):
    """-> (seconds to the first token, seconds from the first to the last, text, the engine's usage)."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": temperature,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            **SAMPLING.get(temperature, {})}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.monotonic()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:300]!r}")
    first = last = None
    text, usage = "", None
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ev = json.loads(line[5:])
        usage = ev.get("usage") or usage
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                now = time.monotonic()
                first = first or now
                last = now
                text += d.get("content") or ""
    conn.close()
    if first is None:
        raise RuntimeError("no tokens came back")
    return first - t0, last - first, text, usage or {}


def main(argv=None):
    stop_on_term()
    ap = argparse.ArgumentParser(prog="solvetime speed", description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--model", required=True, choices=sorted(MODELS), help="the checkpoint (models.py)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--port", type=int, default=8891)
    ap.add_argument("--out", default="results")
    ap.add_argument("--note", default="", help="the conditions of this measurement, shown in the table "
                    "(e.g. \"experts cached\")")
    args = ap.parse_args(argv)

    engine = ENGINES[args.engine](args.model, args.port)
    problem = missing(engine)
    if problem:
        sys.exit(problem)
    tok = tokenizer(args.model)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, time.strftime("speed-%Y%m%d-%H%M%S") + f"-{engine.name}-{args.model}.jsonl")
    meta = {"engine": engine.name, "ours": engine.ours, "version": engine.version(), "model": args.model,
            "pack": engine.pack, "settings": engine.settings, "command": engine.command, "harness": harness_version(),
            "note": args.note, **calibrate()}
    out = open(path, "w")
    log = open(path.replace(".jsonl", ".engine.log"), "w")
    meta["available_before_start"] = system_available()   # before the engine starts: the baseline
    meta["memory_at_start"] = memory_at_start()
    t_start = time.monotonic()
    proc = subprocess.Popen(engine.command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env=engine.server_env())
    try:
        problem = wait_up(f"http://127.0.0.1:{engine.port}{engine.health_path}", proc, engine.start_timeout)
        if not problem:
            try:   # as in `solvetime run`: an engine that cannot answer "Say hello." does not start
                first_token(engine.port, engine.api_model, [{"role": "user", "content": "Say hello."}])
            except (OSError, http.client.HTTPException, RuntimeError) as e:
                problem = f"its first request failed: {e}"[:600]   # e.g. oMLX's 507 from its memory guard
        if problem:
            out.write(json.dumps({"test": "load", "load_failure": problem, **meta}) + "\n")
            print(f"does not start: {problem}")
            return
        meta["cold_start_seconds"] = round(time.monotonic() - t_start, 2)
        meta.update(engine.pack_info())
        out.write(json.dumps({"test": "meta", **meta}) + "\n")
        body = prefill_body()

        def record(**row):
            row.update(engine=engine.name, model=args.model, **memory_counters())
            out.write(json.dumps(row) + "\n")
            out.flush()
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

        test = "warm-up"
        try:
            stream(engine.port, engine.api_model, [{"role": "user", "content": PROMPTS["write"]}], 64, 0.6)
            test = "prefill"
            for _ in range(args.runs):
                content = f"Request {uuid.uuid4().hex}.\n\n{body}\n\nIn one sentence: what does the code above do?"
                ttft, _, _, usage = stream(engine.port, engine.api_model, [{"role": "user", "content": content}], 1, 0.0)
                n = usage.get("prompt_tokens") or len(tok.encode(content).ids) + 10
                record(test="prefill", prompt_tokens=n, ttft_s=ttft, prefill_tok_s=n / ttft)
            for name, prompt in PROMPTS.items():
                for temperature in (0.6, 0.0):
                    test = f"decode {name} T={temperature}"
                    for _ in range(args.runs):
                        content = prompt if name == "continue" else f"Request {uuid.uuid4().hex}. {prompt}"
                        with MemorySampler(proc.pid) as mem:
                            ttft, span, text, usage = stream(engine.port, engine.api_model,
                                                             [{"role": "user", "content": content}], 512, temperature)
                        n = len(tok.encode(text).ids)
                        record(test=test, completion_tokens=n,
                               engine_completion_tokens=usage.get("completion_tokens"), ttft_s=ttft,
                               decode_tok_s=(n - 1) / span if span else None, rss_peak_bytes=mem.rss,
                               footprint_peak_bytes=mem.footprint,
                               system_taken_bytes=meta["available_before_start"] - mem.available_min)
        except (RuntimeError, OSError, http.client.HTTPException) as e:
            # The engine stopped answering (out of GPU memory, usually): say so, with the end of its log, and
            # measure nothing more of it.
            log.flush()
            tail = open(log.name, errors="replace").read().splitlines()[-12:]
            record(test="failure", during=test, error=f"{type(e).__name__}: {e}"[:300], engine_alive=proc.poll() is None,
                   engine_log_tail=[line[-300:] for line in tail])
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        out.close()
        log.close()
        print(f"results: {path}")


def table(paths: list[str]) -> str:
    """Markdown: one row per engine and model, medians with ranges."""
    lines = ["| Model | Engine | Bits per weight | Prefill 4K | Write T 0.6 | Write T 0 | Continue T 0.6 | Continue T 0 "
             "| Memory taken (system) | RSS | Footprint | Notes |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in paths:
        rows = [json.loads(line) for line in open(p)]
        meta = next((r for r in rows if r["test"] in ("meta", "load")), {})
        if meta.get("load_failure"):
            lines.append(f"| {meta['model']} | {meta['engine']} | | | | | | | | | | does not start: {meta['load_failure']} |")
            continue

        def cell(test, key, fmt):
            vals = [r[key] for r in rows if r["test"] == test and r.get(key)
                    and (key != "decode_tok_s" or (r.get("completion_tokens") or 0) >= 256)]
            if not vals and any(r["test"] == test for r in rows):
                return "too short to time"
            return f"{fmt.format(statistics.median(vals))} ({fmt.format(min(vals))}-{fmt.format(max(vals))})" if vals else ""

        failed = next((r for r in rows if r["test"] == "failure"), None)
        rss = max((r.get("rss_peak_bytes") or 0 for r in rows), default=0)
        fp = max((r.get("footprint_peak_bytes") or 0 for r in rows), default=0)
        taken = max((r.get("system_taken_bytes") or 0 for r in rows), default=0)
        from .report import engine_name
        lines.append(f"| {meta.get('model')} | {engine_name(meta)} | {meta.get('bits_per_weight') or ''} "
                     f"| {cell('prefill', 'prefill_tok_s', '{:.0f}')} | "
                     + " | ".join(cell(f"decode {n} T={t}", "decode_tok_s", "{:.1f}") for n in PROMPTS for t in (0.6, 0.0))
                     + "".join(f" | {n / 1e9:.1f} GB" if n else " | " for n in (taken, rss, fp)) + " | "
                     + "; ".join(x for x in (meta.get("note"), f"stopped during {failed['during']}: {failed['error']}"
                                             if failed else "") if x) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
