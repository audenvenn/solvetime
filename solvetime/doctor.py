"""solvetime doctor: what this Mac can run, without starting any engine.

For every engine: whether it is installed (and how to install it if not) and its version. Then the sandbox
the agent's commands run in: the task Python (created on first use) and a sandboxed command, with no network
and no access to the home directory.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess

from .engines import ENGINES, missing
from .models import MODELS, build
from .sandbox import Sandbox, run_in_group


def main(argv=None):
    ap = argparse.ArgumentParser(prog="solvetime doctor", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen3.6-35b-a3b", choices=sorted(MODELS),
                    help="the model whose builds are checked (models.py)")
    args = ap.parse_args(argv)

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip() or 0)
    print(f"{chip or platform.machine()}, {mem / 2**30:.0f} GB, macOS {platform.mac_ver()[0]}")
    print()
    for name, make in ENGINES.items():
        engine = make(args.model, 0)
        problem = missing(engine)
        if problem:
            print(f"  {name:10s} not ready: {problem.split(': ', 1)[-1]}")
        else:
            print(f"  {name:10s} {engine.version() or '(version unknown)'}: {build(args.model, name)}")
    print()
    box = Sandbox()
    try:
        home = os.path.expanduser("~")
        probe = (f"echo sandboxed; ls {home} > /dev/null 2>&1 && echo 'home READABLE' || echo 'home not readable'; "
                 "curl -s -m 3 https://example.com > /dev/null 2>&1 && echo 'network REACHABLE' || echo 'no network'; "
                 "python3 -c 'import sys; print(\"task Python\", sys.version.split()[0])'")
        r = run_in_group(box.command(probe), shell=True, cwd=box.work, env=box.env, timeout=60)
        out = " / ".join(line.strip() for line in r.stdout.splitlines() if line.strip())
        ok = "home not readable" in out and "no network" in out
        print(f"  sandbox    {'ok' if ok else 'PROBLEM'}: {out}")
    finally:
        box.close()


if __name__ == "__main__":
    main()
