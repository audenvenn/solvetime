"""Run the tasks through one engine: solvetime run --engine kapitsa --model qwen3.6-35b-a3b --set core --attempts 2

Per run: the engine is started and warmed up with one request that is not
counted (cold start, process start to that request's first token, is its own
number), the swap and page-out counters are read before and after, and a
short calibration times the GPU so that drift between nights shows up.
Attempts go round-robin: every task's attempt 1, then every task's attempt 2.

Per attempt: a fresh sandboxed directory with the task's files (see
sandbox.py), the pinned agent (mini-swe-agent's default agent and prompts,
one bash tool) talking to the engine through the metering proxy, then the
hidden check, in the sandbox too. One JSON line per attempt goes to
results/<run>/attempts.jsonl, the agent's trajectory next to it, the engine's
log, versions, build and memory into run.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import collections
import http.client
import json
import os
import pathlib
import platform
import re
import signal
import subprocess
import sys
import time
import urllib.request

from .engines import ENGINES, Engine, harness_version, missing
from .models import MODELS, model as model_entry, thinking_fields
from .proxy import Overrides, Proxy
from .sandbox import Sandbox, SandboxedEnvironment
from .tasks import TASKS

# Thinking on or off: each model card's sampling for both is in models.py, the same for every engine.
THINKING = ("thinking", "no-thinking")


def sampling(model: str, thinking: str) -> Overrides:
    """The model card's sampling for thinking or non-thinking use (models.py), sent with every request."""
    return Overrides(**model_entry(model).sampling[thinking])


STEP_LIMIT = 50
WALL_SECONDS = 900
COMMAND_SECONDS = 120


ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def last_error(path: str | None) -> str:
    """The last line of an engine's log that reports an error, without terminal escapes; '' if none."""
    if not path:
        return ""
    try:
        with open(path, errors="replace") as f:
            lines = [ANSI.sub("", line).strip() for line in f.read().replace("\r", "\n").splitlines()]
    except OSError:
        return ""
    errors = [line for line in lines if re.search(r"\b(error|fatal)\b", line, re.I)]
    return errors[-1][:400] if errors else ""


def wait_up(url: str, proc: subprocess.Popen, timeout: int, log_path: str | None = None) -> str | None:
    """None when the server answers; else why not, with the last error its log reports (a launcher's exit code
    alone, e.g. LM Studio's -15 when `lms load` refuses, does not say why)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            why = last_error(log_path)
            return f"exited with code {proc.returncode}" + (f": {why}" if why else "")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status < 500:
                    return None
        except OSError:
            pass
        time.sleep(1)
    return f"not answering after {timeout} s"


def http_error(status: int, body: bytes) -> str:
    """An engine's error reply in one line: its own message when the body is JSON with one (oMLX, LM Studio and the
    OpenAI shape: {"error": {"message": ...}}), else the start of the body. Cutting the raw JSON at a fixed length
    left records ending mid-field ('"type":"server_e')."""
    text = body.decode("utf-8", "replace")
    try:
        data = json.loads(text)
    except ValueError:
        return f"HTTP {status}: {text[:400]}"
    error = data.get("error", data) if isinstance(data, dict) else data
    message = error.get("message") or error.get("detail") if isinstance(error, dict) else error
    return f"HTTP {status}: {str(message or text)[:400]}"


def first_token(port: int, model: str, messages: list, extra: dict | None = None, max_tokens: int = 16,
                timeout: float = 900) -> tuple[float, str, str]:
    """A streamed request straight to the engine: (seconds to its first token, content, reasoning)."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0}
    body.update(extra or {})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.monotonic()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:   # an error is not an empty reply (e.g. oMLX's 507: the model does not fit its memory guard)
        raise RuntimeError(http_error(resp.status, resp.read(65536)))
    first, content, reasoning = None, "", ""
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        try:
            ev = json.loads(line[5:])
        except ValueError:
            continue
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            piece, think = d.get("content") or "", d.get("reasoning_content") or d.get("reasoning") or ""
            if (piece or think) and first is None:
                first = time.monotonic() - t0
            content += piece
            reasoning += think
    conn.close()
    return (first if first is not None else time.monotonic() - t0), content, reasoning


def thinking_switch(port: int, model: str, off: dict | None = None, on: dict | None = None) -> str:
    """Does the engine honour the model's thinking switch (models.thinking_fields: by default the template's
    enable_thinking)? Probed both ways."""
    q = [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}]
    off = {"chat_template_kwargs": {"enable_thinking": False}} if off is None else off
    on = {"chat_template_kwargs": {"enable_thinking": True}} if on is None else on
    try:
        _, c_off, r_off = first_token(port, model, q, off, 64)
        _, c_on, r_on = first_token(port, model, q, on, 64)
    except (OSError, http.client.HTTPException, RuntimeError) as e:
        return f"probe failed: {e}"
    off = bool(r_off) or "<think>" in c_off
    on = bool(r_on) or "<think>" in c_on
    return {(False, True): "honoured", (True, True): "ignored: always thinks",
            (False, False): "ignored: never thinks", (True, False): "inverted"}[(off, on)]


def footprint_peak(pid: int) -> int | None:
    """Peak physical footprint in bytes of a process and its descendants (macOS `footprint`)."""
    pids, todo = [], [pid]
    while todo:
        p = todo.pop()
        pids.append(p)
        out = subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout.split()
        todo += [int(x) for x in out]
    total = 0
    for p in pids:
        out = subprocess.run(["footprint", "-f", "bytes", "-p", str(p)], capture_output=True, text=True).stdout
        m = re.search(r"phys_footprint_peak:\s*([0-9]+)", out)
        if m:
            total += int(m.group(1))
    return total or None


def memory_counters() -> dict:
    """Swap in use and pages paged out so far: a run that pages out did not really fit."""
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([0-9.]+)M", swap)
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    po = re.search(r"Pageouts:\s+([0-9]+)", vm)
    return {"swap_used_mb": float(m.group(1)) if m else None, "pageouts": int(po.group(1)) if po else None}


# A fixed GPU workload, the same for every run: a float16 matmul and a large copy. It runs in a process of its own:
# MLX keeps the ~3.5 GB it used cached afterwards (clear_cache() empties the cache but the process keeps its pages),
# and in this process that memory would stay taken from the engine for the whole run.
CALIBRATE = """
import json, time
try:
    import mlx.core as mx
except ImportError:
    print(json.dumps({"calibration": "mlx not installed"}))
    raise SystemExit
a = mx.random.normal((4096, 4096)).astype(mx.float16)
b = mx.random.normal((4096, 4096)).astype(mx.float16)
mx.eval(a @ b)
t0 = time.monotonic()
for _ in range(20):
    mx.eval(a @ b)
tflops = 20 * 2 * 4096 ** 3 / (time.monotonic() - t0) / 1e12
x = mx.zeros((256 * 2 ** 20,), dtype=mx.float32)
mx.eval(x + 1)
t0 = time.monotonic()
for _ in range(10):
    mx.eval(x + 1)
gbps = 10 * 2 * x.nbytes / (time.monotonic() - t0) / 1e9
print(json.dumps({"calibration_fp16_matmul_tflops": round(tflops, 2), "calibration_copy_gb_s": round(gbps, 1)}))
"""


def memory_at_start() -> dict:
    """What the Mac has in use before the engine starts (vm_stat, bytes). "available" is what the system could hand
    out (free, inactive, speculative and purgeable pages), as the speed test counts it; the rest of it is in use by
    macOS and the other open programs: their own memory, wired memory, the compressor, and cached files."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    size = int(re.search(r"page size of (\d+) bytes", out).group(1))
    v = {k.strip(): int(n) * size for k, n in re.findall(r"^(.+?):\s+(\d+)\.", out, re.M)}
    return {"available": sum(v.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")),
            "apps": v.get("Anonymous pages"), "wired": v.get("Pages wired down"),
            "compressed": v.get("Pages occupied by compressor"), "cached_files": v.get("File-backed pages")}


def calibrate() -> dict:
    """The GPU's speed on a fixed workload (MLX, when installed), measured in a process that ends before the engine
    starts, so that none of its memory is held during the run."""
    try:
        out = subprocess.run([sys.executable, "-c", CALIBRATE], capture_output=True, text=True, timeout=300)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as e:
        return {"calibration": f"failed: {e}"[:200]}


def repeats(rows: list[dict], messages: list[dict]) -> dict:
    """How often the agent repeated itself: the most frequent reply, and the most frequent (command, output)."""
    replies = collections.Counter(r["reply"] for r in rows if r.get("reply"))
    actions = collections.Counter()
    pending = {}
    for m in messages:
        for call in m.get("tool_calls") or []:
            try:
                pending[call.get("id")] = json.loads(call["function"]["arguments"]).get("command", "")
            except (KeyError, ValueError, TypeError, AttributeError):
                pass
        if m.get("role") == "tool" and m.get("tool_call_id") in pending:
            actions[(pending.pop(m["tool_call_id"]), str(m.get("content"))[:2000])] += 1
    return {"max_reply_repeats": max(replies.values(), default=0),
            "max_action_repeats": max(actions.values(), default=0)}


def check(task: dict, box: Sandbox) -> tuple[bool, str]:
    """The hidden check, in the attempt's sandbox (it imports what the agent wrote), then the files it must
    not change."""
    code = f"ORIGINAL_FILES = {task['files']!r}\n{task['check']}"
    try:
        r = box.run_python(code, COMMAND_SECONDS)
    except subprocess.TimeoutExpired:
        return False, "check timed out"
    for name in task["keep"]:
        path = os.path.join(box.work, name)
        if not os.path.isfile(path) or open(path, encoding="utf-8").read() != task["files"][name]:
            return False, f"{name} was changed"
    if r.returncode != 0:
        lines = (r.stderr or r.stdout).strip().splitlines()
        return False, lines[-1][:200] if lines else f"exit code {r.returncode}"
    return True, ""


def attempt(engine: Engine, proxy: Proxy, task: dict, number: int, out_dir: str, proc: subprocess.Popen) -> dict:
    import yaml
    from minisweagent import package_dir
    from minisweagent.agents.default import DefaultAgent
    import litellm
    from minisweagent.models.litellm_model import LitellmModel

    litellm.suppress_debug_info = True   # no "Provider List" / "Give Feedback" lines in the output

    config = yaml.safe_load(open(os.path.join(package_dir, "config", "mini.yaml")))
    box = Sandbox()
    for name, text in task["files"].items():
        path = os.path.join(box.work, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    model = LitellmModel(model_name=f"openai/{engine.api_model}", cost_tracking="ignore_errors",
                         model_kwargs={"api_base": proxy.url + "/v1", "api_key": "local", "timeout": WALL_SECONDS,
                                       "drop_params": True},
                         **{k: v for k, v in config.get("model", {}).items()
                            if k in ("observation_template", "format_error_template")})
    # mini-swe-agent retries a failed query (tenacity catches every exception) unless it is in this list; without
    # SystemExit there, the SIGTERM of stop_on_term() arriving during a query is retried for minutes instead of
    # ending the run (the memory watchdog's stop of 2026-11-18 19:01 kept retrying a server that was gone)
    model.abort_exceptions = [*model.abort_exceptions, SystemExit]
    env = SandboxedEnvironment(box, timeout=COMMAND_SECONDS, env=config.get("environment", {}).get("env", {}))
    agent_cfg = {k: v for k, v in config["agent"].items() if k in ("system_template", "instance_template")}
    traj = pathlib.Path(out_dir, "trajectories", f"{task['name']}.{number}.json")
    agent = DefaultAgent(model, env, step_limit=STEP_LIMIT, cost_limit=0, wall_time_limit_seconds=WALL_SECONDS,
                         output_path=traj, **agent_cfg)
    proxy.meter.take()
    before = memory_counters()
    t0 = time.monotonic()
    try:
        result = agent.run(task=task["prompt"])
        exit_status, error = result.get("exit_status", ""), ""
    except Exception as e:  # the agent gave up on an exception (an engine error, usually)
        exit_status, error = type(e).__name__, str(e)[:300]
    seconds = time.monotonic() - t0
    rows = proxy.meter.take()
    after = memory_counters()
    with open(os.path.join(out_dir, "trajectories", f"{task['name']}.{number}.replies.jsonl"), "w") as f:
        for r in rows:                       # every reply as the engine sent it, rejected ones included
            f.write(json.dumps({"seconds": r.get("seconds"), "finish": r.get("finish"), **(r.pop("_raw", None) or {})})
                    + "\n")
    solved, why = check(task, box)
    box.close()
    rep = repeats(rows, agent.messages)
    looped = rep["max_reply_repeats"] >= 3 or rep["max_action_repeats"] >= 3
    if solved:
        failure = ""
    elif proc.poll() is not None:
        failure = "engine crash"
    elif looped and exit_status in ("TimeExceeded", "LimitsExceeded", "RepeatedFormatError"):
        failure = "loop"
    elif exit_status == "TimeExceeded":
        failure = "timeout"
    elif exit_status == "LimitsExceeded":
        failure = "step limit"
    elif exit_status in ("RepeatedFormatError", "FormatError"):
        failure = "malformed output"
    elif error:
        failure = f"error: {error}"
    else:
        failure = "wrong"

    def total(key):
        values = [r.get(key) for r in rows if r.get(key) is not None]
        return sum(values) if values else None

    return {"task": task["name"], "set": task["group"], "attempt": number, "solved": solved, "failure": failure,
            "looped": looped, **rep, "check": why, "exit_status": exit_status, "seconds": round(seconds, 2),
            "requests": len(rows), "request_seconds": round(sum(r.get("seconds") or 0 for r in rows), 2),
            "prompt_tokens": total("prompt_tokens"), "completion_tokens": total("completion_tokens"),
            "cached_tokens": total("cached_tokens"),
            "http_errors": sum(1 for r in rows if (r.get("status") or 0) >= 400 or r.get("error")),
            "swap_delta_mb": (after["swap_used_mb"] or 0) - (before["swap_used_mb"] or 0),
            "pageouts_delta": (after["pageouts"] or 0) - (before["pageouts"] or 0)}


STOPPED = None   # set when the run was ended from outside


def stop_on_term():
    """SIGTERM (the memory watchdog's, a queue's timeout) ends the run through its `finally`: the engine is
    stopped and what was measured is written. Python's default would exit at once and leave no run.json."""
    def raise_exit(signum, frame):
        global STOPPED
        STOPPED = "SIGTERM"
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, raise_exit)


def main(argv=None):
    stop_on_term()
    ap = argparse.ArgumentParser(prog="solvetime run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--model", required=True, choices=sorted(MODELS), help="the checkpoint (models.py)")
    ap.add_argument("--set", default="core", choices=["core", "extended", "all"])
    ap.add_argument("--tasks", help="comma-separated names (default: the whole set)")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--thinking", default="thinking", choices=THINKING)
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    engine = ENGINES[args.engine](args.model, args.port)
    problem = missing(engine)
    if problem:
        sys.exit(problem)
    names = args.tasks.split(",") if args.tasks else [n for n, t in TASKS.items() if args.set in ("all", t["group"])]
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{engine.name}-{args.model}"
    out_dir = os.path.join(args.out, run_id)
    os.makedirs(os.path.join(out_dir, "trajectories"), exist_ok=True)
    meta = {"engine": engine.name, "ours": engine.ours, "version": engine.version(), "model": args.model,
            "pack": engine.pack, "settings": engine.settings, "command": engine.command, "harness": harness_version(), "thinking": args.thinking,
            "sampling": sampling(args.model, args.thinking).__dict__, "agent": "mini-swe-agent " + _mini_version(),
            "step_limit": STEP_LIMIT, "wall_seconds": WALL_SECONDS, "mac": _mac(), "started": time.strftime("%F %T"),
            "tasks": names, "attempts": args.attempts, **calibrate(), "memory_before": memory_counters()}
    meta["memory_at_start"] = memory_at_start()   # after the calibration's process has ended, before the engine starts
    with open(os.path.join(out_dir, "run.json"), "w") as f:   # written again at the end; a run killed outright
        json.dump(meta, f, indent=2)                           # (no "finished") still leaves what it was
    log = open(os.path.join(out_dir, "engine.log"), "w")
    t_start = time.monotonic()
    proc = subprocess.Popen(engine.command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env=engine.server_env())
    print(f"{engine.name} {meta['version']}: {engine.pack} | starting", flush=True)
    problem = wait_up(f"http://127.0.0.1:{engine.port}{engine.health_path}", proc, engine.start_timeout,
                      log.name)
    try:
        if problem:
            meta["load_failure"] = problem
            print(f"does not start: {problem}", flush=True)
            return
        try:   # warm-up, not counted: lazy page-in of mapped weights and first-time kernel builds land here
            ttft, _, _ = first_token(engine.port, engine.api_model, [{"role": "user", "content": "Say hello."}])
            meta["cold_start_seconds"] = round(time.monotonic() - t_start - 0.0, 2)
            meta["warmup_first_token_seconds"] = round(ttft, 2)
        except (OSError, http.client.HTTPException, RuntimeError) as e:
            # An engine that cannot answer "Say hello." cannot run the tasks either: its attempts would only fail
            # the same way, slowly (the agent retries each request for minutes). Recorded as not starting.
            meta["load_failure"] = f"its first request failed: {e}"[:600]
            print(f"does not start: {meta['load_failure']}", flush=True)
            return
        meta["thinking_switch"] = thinking_switch(engine.port, engine.api_model,
                                                  thinking_fields(args.model, "no-thinking", engine.no_thinking),
                                                  thinking_fields(args.model, "thinking"))
        overrides = sampling(args.model, args.thinking)
        meta["overrides_sent"] = {k: v for k, v in overrides.__dict__.items() if v is not None}
        meta.update(engine.pack_info())
        if overrides.enable_thinking is False and engine.no_thinking:
            overrides = dataclasses.replace(overrides, extra=engine.no_thinking)
            meta["overrides_sent"] = {**meta["overrides_sent"], **engine.no_thinking}
        proxy = Proxy(f"http://127.0.0.1:{engine.port}", overrides).start()
        with open(os.path.join(out_dir, "attempts.jsonl"), "a") as out:
            for k in range(1, args.attempts + 1):          # round-robin: every task once, then again
                for name in names:
                    row = attempt(engine, proxy, TASKS[name], k, out_dir, proc)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(f"  {name} #{k}: {'solved' if row['solved'] else row['failure']} in {row['seconds']:.0f} s, "
                          f"{row['requests']} requests, {row['completion_tokens']} tokens written", flush=True)
                    if proc.poll() is not None:
                        meta["engine_exit"] = proc.returncode
                        return
        proxy.stop()
    finally:
        meta["peak_footprint_bytes"] = footprint_peak(proc.pid) if proc.poll() is None else None
        meta["memory_after"] = memory_counters()
        meta["finished"] = time.strftime("%F %T")
        if STOPPED:
            meta["stopped"] = STOPPED   # e.g. by the memory watchdog: the attempts after the last one recorded did not run
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        engine.cleanup()
        with open(os.path.join(out_dir, "run.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"results: {out_dir}", flush=True)


def _mini_version() -> str:
    try:
        import minisweagent
        return minisweagent.__version__
    except ImportError:
        return "?"


def _mac() -> dict:
    def sysctl(name):
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()
    return {"chip": sysctl("machdep.cpu.brand_string"), "memory_gb": round(int(sysctl("hw.memsize") or 0) / 2**30),
            "macos": platform.mac_ver()[0]}


if __name__ == "__main__":
    main()
