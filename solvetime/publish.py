"""solvetime publish RESULTS_DIR OUT_DIR: runs and speed tests, ready to commit, with their tables.

Copies every agent run (run.json, attempts.jsonl, engine.log, trajectories/) and every speed test
(speed-*.jsonl and its engine log) from RESULTS_DIR into OUT_DIR, with the home directory replaced by ~ in
every file, then writes OUT_DIR/README.md: the runs pooled per engine build (solved, with 95% intervals),
pairwise Fisher exact tests, one row per run, the speed table, and the engines that were not run.

--runs and --speed take shell patterns of names in RESULTS_DIR (default: all). --notes FILE adds the engines
that were not run: a JSON list of {model, engine, version, pack, result, why, how}.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess

from . import report, speed
from .engines import harness_version

HOME = os.path.expanduser("~")


def scrub(text: str) -> str:
    return text.replace(HOME, "~")


def copy_tree(src: str, dst: str) -> int:
    """Copy a run's files, text with the home directory scrubbed; -> bytes written."""
    total = 0
    for root, _, files in os.walk(src):
        for name in files:
            s = os.path.join(root, name)
            d = os.path.join(dst, os.path.relpath(s, src))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            with open(s, encoding="utf-8", errors="replace") as f:
                text = scrub(f.read())
            with open(d, "w", encoding="utf-8") as f:
                f.write(text)
            total += len(text.encode())
    return total


def describe_sampling(s: dict) -> str:
    names = [("temperature", "temperature"), ("top_p", "top-p"), ("top_k", "top-k"), ("min_p", "min-p"),
             ("presence_penalty", "presence penalty")]
    parts = [f"{label} {s[key]:g}" for key, label in names if s.get(key) is not None]
    if s.get("enable_thinking") is not None:
        parts.append("thinking " + ("on" if s["enable_thinking"] else "off"))
    return ", ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="solvetime publish", description=__doc__.split("\n\n")[0])
    ap.add_argument("results")
    ap.add_argument("out")
    ap.add_argument("--runs", nargs="*", default=["*"], help="patterns of run directory names")
    ap.add_argument("--speed", nargs="*", default=["speed-*.jsonl"], help="patterns of speed result names")
    ap.add_argument("--notes", help="JSON list of engines that were not run, and why")
    ap.add_argument("--title", default="Results")
    ap.add_argument("--about", help="a Markdown file about these runs, placed after the summary line")
    ap.add_argument("--names", help="a JSON object: the names the tables show for recorded build names")
    ap.add_argument("--exclude", help="a JSON object: speed runs left out of the speed table, each with why: "
                    '{"speed-….jsonl": {"prefill": [2, 3], "why": "…"}} (run numbers from 1; the file is published)')
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    if args.names:
        report.DISPLAY.update(json.load(open(args.names)))
    names = sorted(os.listdir(args.results))
    runs = [n for n in names if os.path.isfile(os.path.join(args.results, n, "run.json"))
            and any(fnmatch.fnmatch(n, p) for p in args.runs)]
    speeds = [n for n in names if n.startswith("speed-") and n.endswith(".jsonl")
              and any(fnmatch.fnmatch(n, p) for p in args.speed)]
    size = 0
    for n in runs:
        size += copy_tree(os.path.join(args.results, n), os.path.join(args.out, "runs", n))
    for n in speeds:
        for f in (n, n.replace(".jsonl", ".engine.log")):
            s = os.path.join(args.results, f)
            if os.path.exists(s):
                os.makedirs(os.path.join(args.out, "speed"), exist_ok=True)
                with open(s, encoding="utf-8", errors="replace") as fh:
                    text = scrub(fh.read())
                with open(os.path.join(args.out, "speed", f), "w", encoding="utf-8") as fh:
                    fh.write(text)
                size += len(text.encode())
    exclude = json.load(open(args.exclude)) if args.exclude else {}
    if exclude:   # published beside the data, so a reader sees what was left out and why
        with open(os.path.join(args.out, "speed-exclusions.json"), "w") as f:
            json.dump(exclude, f, indent=2)
    notes = json.load(open(args.notes)) if args.notes else []
    if notes:
        with open(os.path.join(args.out, "not-run.json"), "w") as f:
            json.dump(notes, f, indent=2)

    root = os.path.join(args.out, "runs")
    metas = [json.load(open(os.path.join(root, n, "run.json"))) for n in runs]
    macs = sorted({json.dumps(m.get("mac"), sort_keys=True) for m in metas})
    lines = [f"# {args.title}", ""]
    for mac in macs:
        m = json.loads(mac) or {}
        lines.append(f"Machine: {m.get('chip', '?')}, {m.get('memory_gb', '?')} GB, macOS {m.get('macos', '?')}.")
    if metas:
        first, last = min(m["started"] for m in metas), max(m.get("finished") or m["started"] for m in metas)
        agents = sorted({m.get("agent", "?") for m in metas})
        sampling = sorted({describe_sampling(m.get("sampling") or {}) for m in metas})
        lines += [f"Runs from {first} to {last}. Agent: {', '.join(agents)}; limits: {metas[0].get('step_limit')} "
                  f"steps and {metas[0].get('wall_seconds')} s per attempt. Sampling sent with every request: "
                  f"{'; '.join(sampling)}. Tables made with solvetime {harness_version()}.", ""]
    if args.about:
        lines += [open(args.about).read().strip(), ""]
    lines += ["## Agent tasks, each engine build pooled over its task sets", "",
              "Solved with a 95% Wilson interval. \"Ended by malformed output\": the attempt stopped after three "
              "replies in a row that the agent could not use (no tool call). \"Replies without a tool call\" counts "
              "every such reply, recovered or not, out of all replies. \"Available at start\": the memory macOS could "
              "hand out just before the engine started (vm_stat: free, inactive, speculative and purgeable pages); the rest "
              "was in use by macOS and the other open programs. On a 24 GB Mac it decides whether a 15-19 GB build fits. "
              "Runs made before 2026-11-18 did not record it.", "",
              report.pooled_table(root, notes), "",
              "## Pairwise differences", "",
              "Two-sided Fisher exact tests. With a few dozen attempts per engine, only large differences reach "
              "p < 0.05; a larger p means the counts cannot tell the two apart, not that they are equal.", "",
              report.pairwise_table(root), "",
              "## Time", "",
              "Total time: the attempts end to end (agent, sandbox and engine). Per solved task: total time over "
              "tasks solved. Time in the engine: the requests themselves. Decode and prefill during the run: from "
              "each engine's own log, medians over the requests that wrote at least 64 tokens and that read at "
              "least 256 new tokens. A run with settings beyond the defaults has times that compare with nothing "
              "else.", "",
              report.time_table(root), "",
              "## Each run", "",
              "Peak footprint is the engine's phys_footprint, which leaves out clean memory-mapped pages: it "
              "understates an engine that maps its weights from the file. To compare memory across engines, use "
              "the speed section's memory taken from the system.", "",
              report.table(root), ""]
    speed_paths = [os.path.join(args.out, "speed", n) for n in speeds]
    if speed_paths:
        lines += ["## Speed", "",
                  "`solvetime speed`: thinking off, after one uncounted warm-up; prefill on a 4K-token prompt, decode "
                  "of up to 512 tokens on two prompts, three runs each (median, range). Memory taken (system): what "
                  "the system had available before the engine started, minus the least it had while the engine ran "
                  "(free, inactive, speculative and purgeable pages), on an otherwise quiet machine.", "",
                  speed.table(speed_paths, exclude), ""]
        if exclude:
            lines[-2:-1] = ["Runs left out of a cell are named in its row's notes, and listed with the reason in "
                            "speed-exclusions.json.", "", lines[-2]]
    if notes:
        lines += ["## Not run", ""]
        for n in notes:
            lines.append(f"- {n['model']}, {n['engine']} {n.get('version', '')} ({n.get('pack', '')}): "
                         f"{n['result']}. {n['why']}" + (f" How: {n['how']}" if n.get("how") else ""))
        lines.append("")
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(scrub("\n".join(lines)))
    print(f"{len(runs)} runs, {len(speeds)} speed tests, {size / 1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
