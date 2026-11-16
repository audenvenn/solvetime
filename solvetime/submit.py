"""solvetime submit RESULTS_DIR: one bundle of runs and speed tests to send in, with nothing personal in it.

Copies the chosen runs (every run by default) and speed tests into a .tar.gz next to RESULTS_DIR, with the
home directory replaced by ~, the Mac's names by <host>, and anything that looks like an email address by
<email>, and adds machine.json (chip, memory, macOS, the engines found and their versions, solvetime's own
version). Attach the bundle to a "Results from your Mac" issue. `solvetime validate BUNDLE` checks one.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import platform
import re
import subprocess
import tarfile
import tempfile
import time

from .engines import ENGINES, harness_version, missing

HOME = os.path.expanduser("~")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _names() -> list[str]:
    """The Mac's own names (host name, local host name, computer name): they may carry the owner's name."""
    out = {platform.node(), platform.node().split(".")[0]}
    for key in ("ComputerName", "LocalHostName", "HostName"):
        r = subprocess.run(["scutil", "--get", key], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            out.add(r.stdout.strip())
    return sorted((n for n in out if len(n) >= 6), key=len, reverse=True)   # "Mac" alone would clobber text


def scrubber():
    names = _names()

    def scrub(text: str) -> str:
        text = text.replace(HOME, "~")
        for n in names:
            text = text.replace(n, "<host>")
        # email-shaped strings; the agent's tasks use example.com and example.co.uk addresses on purpose
        return EMAIL.sub(lambda m: m.group(0) if re.search(r"@example\.(com|co\.uk|org)$", m.group(0)) else "<email>",
                         text)
    return scrub


def machine() -> dict:
    run = lambda *a: subprocess.run(list(a), capture_output=True, text=True).stdout.strip()
    engines = {}
    for name, make in ENGINES.items():
        e = make("qwen3.6-35b-a3b", 0)
        engines[name] = None if missing(e) else e.version()
    return {"chip": run("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_gb": round(int(run("sysctl", "-n", "hw.memsize") or 0) / 2**30),
            "gpu_cores": run("sh", "-c", "system_profiler SPDisplaysDataType | sed -n 's/.*Total Number of Cores: //p' | head -1"),
            "macos": platform.mac_ver()[0], "solvetime": harness_version(), "engines": engines,
            "bundled": time.strftime("%F %T")}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="solvetime submit", description=__doc__.split("\n\n")[0])
    ap.add_argument("results")
    ap.add_argument("--runs", nargs="*", default=["*"], help="patterns of run directory names (default: all)")
    ap.add_argument("--speed", nargs="*", default=["speed-*.jsonl"], help="patterns of speed result names")
    ap.add_argument("--out", help="the bundle's path (default: solvetime-results-<date>.tar.gz beside RESULTS)")
    args = ap.parse_args(argv)

    scrub = scrubber()
    names = sorted(os.listdir(args.results))
    runs = [n for n in names if os.path.isfile(os.path.join(args.results, n, "run.json"))
            and any(fnmatch.fnmatch(n, p) for p in args.runs)]
    speeds = [n for n in names if n.startswith("speed-") and (n.endswith(".jsonl") or n.endswith(".engine.log"))
              and any(fnmatch.fnmatch(n.replace(".engine.log", ".jsonl"), p) for p in args.speed)]
    if not runs and not speeds:
        raise SystemExit(f"nothing to bundle in {args.results}")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.results)),
                                   time.strftime("solvetime-results-%Y%m%d-%H%M%S.tar.gz"))
    root = "solvetime-results"
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(out, "w:gz") as tar:
        def add(src: str, arc: str):
            dst = os.path.join(tmp, arc)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(src, encoding="utf-8", errors="replace") as f:
                text = scrub(f.read())
            with open(dst, "w", encoding="utf-8") as f:
                f.write(text)
            tar.add(dst, arcname=f"{root}/{arc}")
        for n in runs:
            for dirpath, _, files in os.walk(os.path.join(args.results, n)):
                for f in files:
                    src = os.path.join(dirpath, f)
                    add(src, os.path.join("runs", os.path.relpath(src, args.results)))
        for n in speeds:
            add(os.path.join(args.results, n), os.path.join("speed", n))
        m = os.path.join(tmp, "machine.json")
        with open(m, "w") as f:
            f.write(scrub(json.dumps(machine(), indent=2)))
        tar.add(m, arcname=f"{root}/machine.json")
    print(f"{len(runs)} runs and {len([s for s in speeds if s.endswith('.jsonl')])} speed tests -> {out} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")
    print("Check it with `solvetime validate " + out + "`, then attach it to a \"Results from your Mac\" issue.")


if __name__ == "__main__":
    main()
