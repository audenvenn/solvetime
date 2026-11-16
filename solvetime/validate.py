"""solvetime validate BUNDLE_OR_RESULTS: whether submitted runs are what they say.

For every run, in a bundle from `solvetime submit` (a .tar.gz) or a results directory:

- settings: the agent and its version, the step and time limits, the sampling sent, whether the engine
  honoured the thinking switch, and that the tasks are solvetime's own;
- records: every attempt has its trajectory, and the two agree on how the attempt ended;
- re-scoring: each attempt's commands, taken from its trajectory, are replayed in order in a fresh sandbox
  holding the task's files, and the hidden check is run on the result. A replayed attempt that comes out
  differently from what was recorded is listed: it was edited, or a command did something that does not
  repeat (time, randomness, the network, which the sandbox does not allow).

The replay runs commands written by a model, from someone else's machine, under the same sandbox as the
runs themselves: writes only in a temporary directory, no network, the home directory unreadable.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import tarfile
import tempfile

from .models import MODELS
from .run import COMMAND_SECONDS, STEP_LIMIT, THINKING, WALL_SECONDS, check, sampling
from .sandbox import Sandbox
from .tasks import TASKS

AGENT = "mini-swe-agent 2.4.6"
WORK = re.compile(r"(?:/private)?/var/folders/[^\s'\"]+?/solvetime-[A-Za-z0-9_]+/work")
# A search of the whole disk (or of home) cannot change the task's files unless it acts on what it finds;
# replaying one costs up to the command limit, so such searches are left out of a replay.
SCAN = re.compile(r"^\s*find\s+(/|~|\$HOME)(\s|$)")
ACTS = re.compile(r"-(exec|execdir|ok|okdir|delete|fprint\w*|fls)\b|>|\bxargs\b")
READS = {"head", "tail", "grep", "sort", "uniq", "wc", "cut"}


def disk_scan(cmd: str) -> bool:
    """A lone search of the whole disk or of home, at most piped into programs that only read: it cannot
    change the task's files. Anything compound (;, &&, ||, a substitution) or acting on what it finds is not."""
    c = cmd.replace("2>/dev/null", "").replace("2>&1", "")
    if not SCAN.match(c) or ACTS.search(c) or re.search(r";|&&|\|\||`|\$\(", c):
        return False
    return all((s.split() or [""])[0] in READS for s in c.split("|")[1:])


def commands(trajectory: dict) -> list[str]:
    """The bash commands an attempt ran, in order."""
    out = []
    for m in trajectory.get("messages", []):
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name") != "bash":
                continue
            try:
                cmd = json.loads(fn.get("arguments") or "{}").get("command")
            except json.JSONDecodeError:
                cmd = None
            if cmd:
                out.append(cmd)
    return out


def original_work_dir(trajectory: dict) -> str | None:
    """Where the attempt's files were, as its commands and their outputs show it (pwd, ls -la, errors)."""
    for m in trajectory.get("messages", []):
        texts = [str(m.get("content") or "")] + [json.dumps(c) for c in m.get("tool_calls") or []]
        for text in texts:
            hit = WORK.search(text)
            if hit:
                return hit.group(0)
    return None


def replay(task: dict, trajectory: dict) -> tuple[bool, str]:
    """Re-run an attempt's commands on the task's files in a fresh sandbox, then the hidden check."""
    box = Sandbox()
    try:
        for name, text in task["files"].items():
            path = os.path.join(box.work, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        old = original_work_dir(trajectory)
        # The attempt's directory, however its commands spelled it (/var/folders/..., /private/var/folders/...,
        # or /System/Volumes/Data/private/var/folders/... as a search of the disk finds it), becomes this
        # replay's own.
        moved = None
        if old:
            base_id = re.search(r"/solvetime-([A-Za-z0-9_]+)/work", old).group(1)
            moved = re.compile(r"(?:/System/Volumes/Data)?(?:/private)?/var/folders/[^\s'\"]*?/solvetime-"
                               + re.escape(base_id) + r"(?![A-Za-z0-9_])")
        new_base = os.path.realpath(box.base)
        # Replayed commands follow each other within milliseconds, so an edit that keeps a file's size can
        # land in the same second as the bytecode Python cached for it, which would then pass for current
        # and run the old code. The attempt itself had the model's thinking between its commands. So the
        # replay keeps no bytecode cache.
        env = dict(box.env, PYTHONDONTWRITEBYTECODE="1")
        for cmd in commands(trajectory):
            if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd:
                break
            if disk_scan(cmd):
                continue
            if moved:
                cmd = moved.sub(new_base, cmd)
            try:
                subprocess.run(box.command(cmd), shell=True, cwd=box.work, env=env, capture_output=True,
                               timeout=COMMAND_SECONDS, start_new_session=True)
            except subprocess.TimeoutExpired:
                pass
        return check(task, box)
    finally:
        box.close()


def settings_problems(meta: dict) -> list[str]:
    problems = []
    if meta.get("agent") != AGENT:
        problems.append(f"agent is {meta.get('agent')!r}, not {AGENT!r}")
    if meta.get("step_limit") != STEP_LIMIT or meta.get("wall_seconds") != WALL_SECONDS:
        problems.append(f"limits are {meta.get('step_limit')} steps and {meta.get('wall_seconds')} s, "
                        f"not {STEP_LIMIT} and {WALL_SECONDS}")
    name, thinking = meta.get("model", ""), meta.get("thinking", "")
    want = sampling(name, thinking) if name in MODELS and thinking in THINKING else None
    if want is None:
        problems.append(f"unknown model or thinking setting: {name!r}, {thinking!r}")
    else:
        expected = {k: v for k, v in want.__dict__.items() if v is not None and k != "extra"}
        sent = {k: v for k, v in (meta.get("overrides_sent") or {}).items() if k in want.__dict__ and k != "extra"}
        if sent != expected:          # an engine's own fields for the same setting (reasoning_effort) may come on top
            problems.append(f"sampling sent {sent}, expected {expected}")
    if meta.get("thinking_switch") not in ("honoured", None):
        problems.append(f"the engine's thinking switch: {meta.get('thinking_switch')}")
    unknown = [t for t in meta.get("tasks") or [] if t not in TASKS]
    if unknown:
        problems.append(f"tasks solvetime does not have: {unknown}")
    return problems


def validate_run(run_dir: str, rescore: bool = True) -> dict:
    meta = json.load(open(os.path.join(run_dir, "run.json")))
    path = os.path.join(run_dir, "attempts.jsonl")
    attempts = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    problems = settings_problems(meta)
    mismatches, replayed, todo = [], 0, []
    for a in attempts:
        tpath = os.path.join(run_dir, "trajectories", f"{a['task']}.{a['attempt']}.json")
        if not os.path.exists(tpath):
            problems.append(f"{a['task']} #{a['attempt']}: no trajectory")
            continue
        traj = json.load(open(tpath))
        ended = (traj.get("info") or {}).get("exit_status")
        if ended and a.get("exit_status") and ended != a["exit_status"] and a["failure"] != "engine crash":
            problems.append(f"{a['task']} #{a['attempt']}: recorded {a['exit_status']}, trajectory says {ended}")
        if rescore and a["task"] in TASKS and a["failure"] != "engine crash":
            todo.append((a, traj))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda at: replay(TASKS[at[0]["task"]], at[1]), todo))
    for (a, _), (solved, why) in zip(todo, results):
        replayed += 1
        if solved != a["solved"]:
            mismatches.append(f"{a['task']} #{a['attempt']}: recorded {'solved' if a['solved'] else a['failure']}, "
                              f"replayed {'solved' if solved else 'not solved (' + why + ')'}")
    return {"run": os.path.basename(run_dir), "engine": meta.get("engine"), "version": meta.get("version"),
            "model": meta.get("model"), "attempts": len(attempts), "solved": sum(a["solved"] for a in attempts),
            "replayed": replayed, "mismatches": mismatches, "problems": problems}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="solvetime validate", description=__doc__.split("\n\n")[0])
    ap.add_argument("path", help="a bundle (.tar.gz) from solvetime submit, or a results directory")
    ap.add_argument("--no-rescore", action="store_true", help="check settings and records only")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        root = args.path
        if os.path.isfile(root):
            with tarfile.open(root) as tar:
                tar.extractall(tmp, filter="data")      # no absolute paths, links out or devices
            root = tmp
        run_dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "run.json"), recursive=True))
        if not run_dirs:
            raise SystemExit(f"no runs in {args.path}")
        bad = 0
        for d in run_dirs:
            r = validate_run(d, rescore=not args.no_rescore)
            ok = not r["problems"] and not r["mismatches"]
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} {r['run']}: {r['engine']} {r['version'] or ''}, {r['model']}: "
                  f"{r['solved']}/{r['attempts']} solved" + (f", {r['replayed']} replayed" if r["replayed"] else ""))
            for line in r["problems"] + r["mismatches"]:
                print(f"      {line}")
        print(f"{len(run_dirs)} runs, {bad} with problems")
        raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
