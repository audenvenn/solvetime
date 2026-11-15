"""Run the tasks through one engine: solvetime run --engine kapitsa --model qwen3.6-35b-a3b --set core --attempts 2

For each attempt: a fresh directory with the task's files, the pinned agent
(mini-swe-agent's default agent and prompts, one bash tool) talking to the
engine through the metering proxy, then the hidden check. One JSON line per
attempt goes to results/<run>/attempts.jsonl, the agent's trajectory next to
it, the engine's log and peak memory into run.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

from .engines import ENGINES, Engine
from .proxy import Overrides, Proxy
from .tasks import TASKS

# The model card's sampling for thinking mode (Qwen3.5 / 3.6 / 3.8, coding), the same for every engine.
SAMPLING = {"thinking": Overrides(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,
                                  enable_thinking=True),
            "no-thinking": Overrides(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5,
                                     enable_thinking=False)}
STEP_LIMIT = 50
WALL_SECONDS = 900
COMMAND_SECONDS = 120


def wait_up(url: str, proc: subprocess.Popen, timeout: int) -> str | None:
    """None when the server answers; else why not."""
    t0 = time.time()
    while time.time() - t0 < timeout:
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


def check(task: dict, work: str) -> tuple[bool, str]:
    """The hidden check: the task's snippet in its directory, then the files it must not change."""
    code = f"ORIGINAL_FILES = {task['files']!r}\n{task['check']}"
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=work, capture_output=True, text=True,
                           timeout=COMMAND_SECONDS, start_new_session=True)
    except subprocess.TimeoutExpired:
        return False, "check timed out"
    for name in task["keep"]:
        path = os.path.join(work, name)
        if not os.path.isfile(path) or open(path, encoding="utf-8").read() != task["files"][name]:
            return False, f"{name} was changed"
    if r.returncode != 0:
        lines = (r.stderr or r.stdout).strip().splitlines()
        return False, lines[-1][:200] if lines else f"exit code {r.returncode}"
    return True, ""


def looped(rows: list[dict], n: int = 3) -> bool:
    """n replies in a row with the same content: the agent repeats itself."""
    streak, last = 1, None
    for row in rows:
        reply = row.get("reply")
        streak = streak + 1 if reply is not None and reply == last else 1
        last = reply
        if streak >= n:
            return True
    return False


def attempt(engine: Engine, proxy: Proxy, task: dict, number: int, out_dir: str, proc: subprocess.Popen) -> dict:
    import yaml
    from minisweagent import package_dir
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.litellm_model import LitellmModel

    config = yaml.safe_load(open(os.path.join(package_dir, "config", "mini.yaml")))
    work = tempfile.mkdtemp(prefix=f"solvetime-{task['name']}-")
    for name, text in task["files"].items():
        path = os.path.join(work, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w", encoding="utf-8").write(text)
    model = LitellmModel(model_name=f"openai/{engine.api_model}", cost_tracking="ignore_errors",
                         model_kwargs={"api_base": proxy.url + "/v1", "api_key": "local", "timeout": WALL_SECONDS,
                                       "drop_params": True},
                         **{k: v for k, v in config.get("model", {}).items() if k in ("observation_template",
                                                                                        "format_error_template")})
    env = LocalEnvironment(cwd=work, timeout=COMMAND_SECONDS, env=config.get("environment", {}).get("env", {}))
    agent_cfg = {k: v for k, v in config["agent"].items() if k in ("system_template", "instance_template")}
    traj = os.path.join(out_dir, "trajectories", f"{task['name']}.{number}.json")
    agent = DefaultAgent(model, env, step_limit=STEP_LIMIT, cost_limit=0, wall_time_limit_seconds=WALL_SECONDS,
                         output_path=__import__("pathlib").Path(traj), **agent_cfg)
    proxy.meter.take()
    t0 = time.time()
    try:
        result = agent.run(task=task["prompt"])
        exit_status, error = result.get("exit_status", ""), ""
    except Exception as e:  # the agent gave up on an exception (an engine error, usually)
        exit_status, error = type(e).__name__, str(e)[:300]
    seconds = time.time() - t0
    rows = proxy.meter.take()
    solved, why = check(task, work)
    if solved:
        failure = ""
    elif proc.poll() is not None:
        failure = "engine crash"
    elif exit_status == "TimeExceeded":
        failure = "timeout"
    elif exit_status == "LimitsExceeded":
        failure = "step limit"
    elif exit_status in ("RepeatedFormatError", "FormatError"):
        failure = "malformed output"
    elif looped(rows):
        failure = "loop"
    elif error:
        failure = f"error: {error}"
    else:
        failure = "wrong"
    shutil.rmtree(work, ignore_errors=True)

    def total(key):
        values = [r.get(key) for r in rows if r.get(key) is not None]
        return sum(values) if values else None

    return {"task": task["name"], "set": task["group"], "attempt": number, "solved": solved, "failure": failure,
            "check": why, "exit_status": exit_status, "seconds": round(seconds, 2), "requests": len(rows),
            "request_seconds": round(sum(r.get("seconds") or 0 for r in rows), 2),
            "prompt_tokens": total("prompt_tokens"), "completion_tokens": total("completion_tokens"),
            "cached_tokens": total("cached_tokens"), "http_errors": sum(1 for r in rows if (r.get("status") or 0) >= 400 or r.get("error"))}


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
    names = args.tasks.split(",") if args.tasks else [n for n, t in TASKS.items() if args.set in ("all", t["group"])]
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{engine.name}-{args.model}"
    out_dir = os.path.join(args.out, run_id)
    os.makedirs(os.path.join(out_dir, "trajectories"), exist_ok=True)
    meta = {"engine": engine.name, "ours": engine.ours, "version": engine.version(), "model": args.model,
            "pack": engine.pack, "command": engine.command, "thinking": args.thinking,
            "sampling": SAMPLING[args.thinking].__dict__, "agent": "mini-swe-agent " + _mini_version(),
            "step_limit": STEP_LIMIT, "wall_seconds": WALL_SECONDS, "mac": _mac(), "started": time.strftime("%F %T")}
    log = open(os.path.join(out_dir, "engine.log"), "w")
    proc = subprocess.Popen(engine.command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env=dict(os.environ, **engine.env))
    print(f"{engine.name} {meta['version']}: {engine.pack} | starting", flush=True)
    problem = wait_up(f"http://127.0.0.1:{engine.port}{engine.health_path}", proc, engine.start_timeout)
    attempts_path = os.path.join(out_dir, "attempts.jsonl")
    try:
        if problem:
            meta["load_failure"] = problem
            print(f"does not start: {problem}", flush=True)
            return
        meta["pack_bytes"] = engine.pack_bytes()
        proxy = Proxy(f"http://127.0.0.1:{engine.port}", SAMPLING[args.thinking]).start()
        with open(attempts_path, "a") as out:
            for name in names:
                for k in range(1, args.attempts + 1):
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
        meta["finished"] = time.strftime("%F %T")
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        json.dump(meta, open(os.path.join(out_dir, "run.json"), "w"), indent=2)
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
