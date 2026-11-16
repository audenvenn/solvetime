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
from .proxy import Overrides, Proxy
from .sandbox import Sandbox, SandboxedEnvironment
from .tasks import TASKS

# The model card's sampling (Qwen3.5 / 3.6 / 3.8), the same for every engine.
SAMPLING = {"thinking": Overrides(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,
                                  enable_thinking=True),
            "no-thinking": Overrides(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5,
                                     enable_thinking=False)}
STEP_LIMIT = 50
WALL_SECONDS = 900
COMMAND_SECONDS = 120


def wait_up(url: str, proc: subprocess.Popen, timeout: int) -> str | None:
    """None when the server answers; else why not."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            return f"exited with code {proc.returncode}"
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status < 500:
                    return None
        except OSError:
            pass
        time.sleep(1)
    return f"not answering after {timeout} s"


def first_token(port: int, model: str, messages: list, extra: dict | None = None, max_tokens: int = 16,
                timeout: float = 900) -> tuple[float, str, str]:
    """A streamed request straight to the engine: (seconds to its first token, content, reasoning)."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0}
    body.update(extra or {})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.monotonic()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
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


def thinking_switch(port: int, model: str) -> str:
    """Does the engine honour chat_template_kwargs.enable_thinking? Probed both ways."""
    q = [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}]
    try:
        _, c_off, r_off = first_token(port, model, q, {"chat_template_kwargs": {"enable_thinking": False}}, 64)
        _, c_on, r_on = first_token(port, model, q, {"chat_template_kwargs": {"enable_thinking": True}}, 64)
    except (OSError, http.client.HTTPException) as e:
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


def calibrate() -> dict:
    """A fixed GPU workload (MLX, when installed): a float16 matmul and a large copy."""
    try:
        import mlx.core as mx
    except ImportError:
        return {"calibration": "mlx not installed"}
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
    return {"calibration_fp16_matmul_tflops": round(tflops, 2), "calibration_copy_gb_s": round(gbps, 1)}


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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="solvetime run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--model", required=True, help="qwen3.6-35b-a3b, qwen3.8-27b or qwen3.5-9b")
    ap.add_argument("--set", default="core", choices=["core", "extended", "all"])
    ap.add_argument("--tasks", help="comma-separated names (default: the whole set)")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--thinking", default="thinking", choices=sorted(SAMPLING))
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
            "sampling": SAMPLING[args.thinking].__dict__, "agent": "mini-swe-agent " + _mini_version(),
            "step_limit": STEP_LIMIT, "wall_seconds": WALL_SECONDS, "mac": _mac(), "started": time.strftime("%F %T"),
            "tasks": names, "attempts": args.attempts, **calibrate(), "memory_before": memory_counters()}
    log = open(os.path.join(out_dir, "engine.log"), "w")
    t_start = time.monotonic()
    proc = subprocess.Popen(engine.command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env=dict(os.environ, **engine.env))
    print(f"{engine.name} {meta['version']}: {engine.pack} | starting", flush=True)
    problem = wait_up(f"http://127.0.0.1:{engine.port}{engine.health_path}", proc, engine.start_timeout)
    try:
        if problem:
            meta["load_failure"] = problem
            print(f"does not start: {problem}", flush=True)
            return
        try:   # warm-up, not counted: lazy page-in of mapped weights and first-time kernel builds land here
            ttft, _, _ = first_token(engine.port, engine.api_model, [{"role": "user", "content": "Say hello."}])
            meta["cold_start_seconds"] = round(time.monotonic() - t_start - 0.0, 2)
            meta["warmup_first_token_seconds"] = round(ttft, 2)
        except (OSError, http.client.HTTPException) as e:
            meta["warmup_error"] = str(e)
        meta["thinking_switch"] = thinking_switch(engine.port, engine.api_model)
        meta["overrides_sent"] = {k: v for k, v in SAMPLING[args.thinking].__dict__.items() if v is not None}
        meta.update(engine.pack_info())
        proxy = Proxy(f"http://127.0.0.1:{engine.port}", SAMPLING[args.thinking]).start()
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
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
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
