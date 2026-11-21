"""solvetime publish RESULTS_DIR OUT_DIR: runs and speed tests, ready to commit, with their tables.

Copies every agent run (run.json, attempts.jsonl, engine.log, trajectories/) and every speed test
(speed-*.jsonl and its engine log) from RESULTS_DIR into OUT_DIR, scrubbed of what identifies this Mac and its
user (scrub()), then writes OUT_DIR/README.md: the runs pooled per engine build (solved, with 95% intervals),
pairwise Fisher exact tests, one row per run, the speed table, and the engines that were not run.

--runs and --speed take shell patterns of names in RESULTS_DIR (default: all). --notes FILE adds the engines
that were not run: a JSON list of {model, engine, version, pack, result, why, how}.
"""

from __future__ import annotations

import argparse
import fnmatch
import getpass
import json
import os
import re
import socket
import subprocess
import tempfile

from . import report, speed
from .engines import harness_version

HOME = os.path.expanduser("~")
ACCOUNT = getpass.getuser()
GENERIC = {"mac", "macmini", "macbook", "macbookair", "macbookpro", "imac", "macstudio", "macpro", "localhost"}
# oMLX logs the names and addresses it answers to.
OMLX_ALIASES = re.compile(r"(server aliases: )\[[^\]]*\]")


def _out(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def temp_dir_parts() -> tuple[str, str] | None:
    """The two parts of this account's temporary directory, /var/folders/<2>/<id>/T."""
    m = re.search(r"/var/folders/([^/]+)/([^/]+)/T(?:/|$)", os.path.realpath(tempfile.gettempdir()))
    return (m.group(1), m.group(2)) if m else None


TEMP = temp_dir_parts()


def local_names() -> list[tuple[str, str]]:
    """What identifies this Mac and its user in the records, each with its replacement:
    - the per-user temporary directory /var/folders/<2>/<id>/T, which every attempt's workspace sits in;
    - the host names;
    - the account's full name;
    - the Mac's own network addresses (127.0.0.1 stays).
    Longest first, so that a name inside another is not replaced first."""
    pairs = []
    if TEMP:
        pairs += [(f"{TEMP[0]}/{TEMP[1]}", "xx/user"), (TEMP[1], "user")]
    hosts = {_out("scutil", "--get", k).strip() for k in ("LocalHostName", "ComputerName", "HostName")}
    hosts |= {socket.gethostname(), socket.gethostname().split(".")[0]}
    pairs += [(h, "mac") for h in hosts if len(h) > 2 and re.sub(r"[^a-z]", "", h.lower()) not in GENERIC]
    full = _out("id", "-F").strip()
    if len(full) > 3:
        pairs.append((full, "user"))
    for addr in re.findall(r"\binet6? ([0-9a-f.:]+)", _out("ifconfig")):
        if addr not in ("127.0.0.1", "::1") and not addr.startswith("fe80"):
            pairs.append((addr, "192.0.2.1" if "." in addr else "2001:db8::1"))
    return sorted(set(pairs), key=lambda p: -len(p[0]))


NAMES = local_names()


def scrub(text: str, names: list[tuple[str, str]] | None = None, home: str = HOME, account: str = ACCOUNT,
          temp: tuple[str, str] | None = TEMP) -> str:
    """The records without what identifies this Mac and its user: the home directory becomes ~, the
    temporary-directory id, host names, full name and network addresses are replaced, and so is the account name,
    for instance as the owner in `ls -l`. A name after a slash, as in a github.com or huggingface.co link, is
    left: it is the public handle. Workspace paths keep their shape (/var/folders/xx/user/T/solvetime-<id>/work),
    which a replay needs. An agent sometimes types the temporary directory from memory, cut short or
    misspelled, so anything under this account's own /var/folders/<2>/ goes too, and so does anything that starts
    with the id's first 8 characters. The system accounts' folders (/var/folders/zz/...) are the same on every Mac
    and stay."""
    text = OMLX_ALIASES.sub(r"\1[localhost, 127.0.0.1 and this Mac's own names: not published]", text.replace(home, "~"))
    for old, new in NAMES if names is None else names:
        if re.fullmatch(r"[0-9.]+", old) and old.count(".") == 3:   # IPv4, not part of a longer number
            text = re.sub(rf"(?<![0-9.]){re.escape(old)}(?![0-9]|\.[0-9])", new, text)
        elif re.fullmatch(r"[0-9a-f:]+", old) and ":" in old:   # IPv6, not part of a longer address
            text = re.sub(rf"(?<![0-9a-f:]){re.escape(old)}(?![0-9a-f:])", new, text)
        else:
            text = text.replace(old, new)
    if temp:
        text = re.sub(rf"(/var/folders/){re.escape(temp[0])}/", r"\1xx/", text)
        text = re.sub(rf"{re.escape(temp[1][:8])}[A-Za-z0-9_+-]*", "user", text)
    return re.sub(rf"(?<![/\w.@-]){re.escape(account)}(?![\w-])", "user", text) if account else text


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
    ap.add_argument("--names", help="a JSON object: the names the tables show for recorded build names, and for "
                    "recorded packs (e.g. a local model folder -> the Hugging Face repo and revision it equals)")
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
    # A pack recorded as a local folder is published under the name --names gives it (the repo and revision the
    # folder equals): the folder's path tells a reader nothing they can use (2026-11-21).
    for n in runs:
        path = os.path.join(args.out, "runs", n, "run.json")
        meta = json.load(open(path))
        if meta.get("pack") in report.DISPLAY:
            meta["pack"] = report.DISPLAY[meta["pack"]]
            with open(path, "w") as f:
                json.dump(meta, f, indent=2)
    for n in speeds:
        path = os.path.join(args.out, "speed", n)
        rows = [json.loads(line) for line in open(path) if line.strip()]
        if any(r.get("pack") in report.DISPLAY for r in rows):
            for r in rows:
                if r.get("pack") in report.DISPLAY:
                    r["pack"] = report.DISPLAY[r["pack"]]
            with open(path, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
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
                  f"steps, and a {metas[0].get('wall_seconds')} s budget the agent checks before each query (a query "
                  f"in flight and its retries can run past it). Sampling sent with every request: "
                  f"{'; '.join(sampling)}. Tables made with solvetime {harness_version()}.", ""]
    if args.about:
        lines += [open(args.about).read().strip(), ""]
    lines += ["## Agent tasks, each engine build pooled over its task sets", "",
              "Solved: the attempt's final workspace passed the task's check, work done before an error included, "
              "with a 95% Wilson interval. \"Failed checks after a repeated-format exit\": attempts the agent stopped "
              "after three replies in a row it could not use, whose check then failed. \"Requests\": every request "
              "the agent sent, the ones that failed included. \"Agent tool-call error notices\" and \"Agent "
              "output-limit notices\": the agent's own feedback messages, for a reply without a usable tool call and "
              "for a reply cut off at the output limit. \"Estimated available before launch\": the memory macOS could "
              "hand out just before the engine started (vm_stat: free, inactive, speculative and purgeable pages), "
              "measured after the harness's GPU calibration; runs made before 2026-11-18 did not record it. A run "
              "whose engine never served an attempt is listed with how it stopped, and no score.", "",
              report.pooled_table(root, notes), "",
              "## Pairwise differences", "",
              "Two-sided Fisher exact tests. With a few dozen attempts per engine, only large differences reach "
              "p < 0.05; a larger p means the counts cannot tell the two apart, not that they are equal.", "",
              report.pairwise_table(root), "",
              "## Time", "",
              "Total time: the attempts end to end (agent, sandbox and engine). Per solved task: total time over "
              "tasks solved. Summed proxy request duration: the requests' durations added up; with retries it can "
              "exceed the elapsed time. Decode and prefill during the run: from "
              "each engine's own log, medians over the requests that wrote at least 64 tokens and that read at "
              "least 256 new tokens; an engine whose log has no timing of its own for these phases says so. Elapsed totals describe each listed configuration, and include the time "
              "spent on failed attempts.", "",
              report.time_table(root), "",
              "## Each run", "",
              "Peak footprint is the engine's phys_footprint, which leaves out clean memory-mapped pages: it "
              "understates an engine that maps its weights from the file. To compare memory across engines, use "
              "the speed section's memory taken from the system.", "",
              report.table(root), ""]
    speed_paths = [os.path.join(args.out, "speed", n) for n in speeds]
    if speed_paths:
        lines += ["## Speed", "",
                  "`solvetime speed`: thinking off, by each model's own switch, after one uncounted warm-up; prefill on "
                  "a prompt of Python standard-library source (its size in the model's own tokens is in the cell), "
                  "decode of up to 512 tokens on two prompts, three runs each (median, range). Memory taken (system): "
                  "what the system had available before the engine started, minus the least it had while the "
                  "decode requests ran, which is when it is sampled (free, inactive, speculative and purgeable "
                  "pages); other processes can change it.", "",
                  speed.table(speed_paths, exclude), ""]
        if exclude:
            lines[-2:-1] = ["Runs left out of a cell are named in its row's notes, and listed with the reason in "
                            "speed-exclusions.json.", "", lines[-2]]
    speed_notes = [n for n in notes if report.speed_only(n)]   # engines with an agent score but no speed result
    if speed_notes:
        if not speed_paths:
            lines += ["## Speed", ""]
        for n in speed_notes:
            lines.append(f"- {n['model']}, {n['engine']} {n.get('version', '')} ({n.get('pack', '')}): "
                         f"no speed result. {n['why']}")
        lines.append("")
    notes = [n for n in notes if not report.speed_only(n)]
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
