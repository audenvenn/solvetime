"""solvetime report RESULTS_DIR: a markdown table, one row per run (an engine and a model)."""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics


def runs(root: str):
    for meta_path in sorted(glob.glob(os.path.join(root, "*", "run.json"))):
        meta = json.load(open(meta_path))
        path = os.path.join(os.path.dirname(meta_path), "attempts.jsonl")
        attempts = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
        yield meta, attempts


def gb(n):
    return f"{n / 1e9:.1f} GB" if n else ""


def table(root: str) -> str:
    lines = ["| Model | Engine | Build | Size | Peak memory | Solved | Median time | Total time | Tokens written "
             "(median) | Requests (median) | Failures |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for meta, attempts in runs(root):
        engine = meta["engine"] + (" (ours)" if meta.get("ours") else "") + (f" {meta['version']}" if meta.get("version") else "")
        if meta.get("load_failure"):
            lines.append(f"| {meta['model']} | {engine} | {meta['pack']} | | | does not start: {meta['load_failure']} "
                         "| | | | | |")
            continue
        n = len(attempts)
        solved = sum(a["solved"] for a in attempts)
        secs = [a["seconds"] for a in attempts]
        toks = [a["completion_tokens"] for a in attempts if a.get("completion_tokens") is not None]
        reqs = [a["requests"] for a in attempts]
        failures = collections.Counter(a["failure"].split(":")[0] for a in attempts if not a["solved"])
        lines.append("| {} | {} | {} | {} | {} | {}/{} | {} | {} | {} | {} | {} |".format(
            meta["model"], engine, meta["pack"], gb(meta.get("pack_bytes")), gb(meta.get("peak_footprint_bytes")),
            solved, n, f"{statistics.median(secs):.0f} s" if secs else "", f"{sum(secs) / 60:.1f} min" if secs else "",
            f"{statistics.median(toks):.0f}" if toks else "", f"{statistics.median(reqs):.0f}" if reqs else "",
            ", ".join(f"{k} {v}" for k, v in failures.most_common()) or "none"))
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="solvetime report", description=__doc__)
    ap.add_argument("results")
    args = ap.parse_args(argv)
    print(table(args.results), end="")


if __name__ == "__main__":
    main()
