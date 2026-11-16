"""solvetime submit RESULTS_DIR: one bundle of runs and speed tests to send in, with nothing personal in it.

Copies the chosen runs (every run by default) and speed tests into a .tar.gz next to RESULTS_DIR. What could
name you is replaced wherever it appears: your home directory by ~, your Mac's names by <host>, the email
addresses in your git configuration by <email>, your account's full name by <name>. JSON files are scrubbed
value by value, so they stay valid. It adds machine.json (chip, memory, macOS, the engines found and their
versions, solvetime's own version). Attach the bundle to a "Results from your Mac" issue;
`solvetime validate BUNDLE` checks one first.

Other email-shaped strings are left alone: they are the tasks' own test data (an email validator is one of
them), and changing them would change what the agent did.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time

from .engines import ENGINES, harness_version

HOME = os.path.expanduser("~")


def _run(*args) -> str:
    try:
        return subprocess.run(list(args), capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _names() -> list[str]:
    """The Mac's own names (host name, local host name, computer name): they may carry the owner's name."""
    out = {platform.node(), platform.node().split(".")[0]}
    for key in ("ComputerName", "LocalHostName", "HostName"):
        out.add(_run("scutil", "--get", key))
    return [n for n in out if len(n) >= 6]          # "Mac" alone would clobber ordinary text


def targets() -> list[tuple[str, str]]:
    """(what could name the user, what replaces it), longest first."""
    pairs = [(HOME, "~")] + [(n, "<host>") for n in _names()]
    for email in _run("git", "config", "--global", "--get-all", "user.email").splitlines():
        if "@" in email:
            pairs.append((email.strip(), "<email>"))
    full = _run("id", "-F")
    if len(full) >= 4:
        pairs.append((full, "<name>"))
    return sorted({p for p in pairs if p[0]}, key=lambda p: len(p[0]), reverse=True)


def scrubber():
    pairs = targets()

    def text(s: str) -> str:
        for old, new in pairs:
            s = s.replace(old, new)
        return s

    def value(v):
        if isinstance(v, str):
            return text(v)
        if isinstance(v, list):
            return [value(x) for x in v]
        if isinstance(v, dict):
            return {text(k): value(x) for k, x in v.items()}
        return v

    def scrub(content: str, name: str = "") -> str:
        """A file's content, scrubbed: JSON value by value (it stays valid), anything else as text."""
        try:
            if name.endswith(".json"):
                return json.dumps(value(json.loads(content)), indent=2, ensure_ascii=False)
            if name.endswith(".jsonl"):
                return "".join(json.dumps(value(json.loads(line)), ensure_ascii=False) + "\n"
                               for line in content.splitlines() if line.strip())
        except json.JSONDecodeError:
            pass
        return text(content)
    return scrub


def machine() -> dict:
    engines = {}
    for name, make in ENGINES.items():
        e = make("qwen3.6-35b-a3b", 0)
        installed = shutil.which(e.command[0]) or os.path.isfile(e.command[0])
        engines[name] = (e.version() or "installed") if installed else None
    return {"chip": _run("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_gb": round(int(_run("sysctl", "-n", "hw.memsize") or 0) / 2**30),
            "gpu_cores": _run("sh", "-c", "system_profiler SPDisplaysDataType | sed -n 's/.*Total Number of Cores: //p' | head -1"),
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
                text = scrub(f.read(), src)
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
            f.write(scrub(json.dumps(machine(), indent=2), "machine.json"))
        tar.add(m, arcname=f"{root}/machine.json")
    print(f"{len(runs)} runs and {len([s for s in speeds if s.endswith('.jsonl')])} speed tests -> {out} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")
    print("Check it with `solvetime validate " + out + "`, then attach it to a \"Results from your Mac\" issue.")


if __name__ == "__main__":
    main()
