"""solvetime report RESULTS_DIR: a markdown table, one row per run (an engine and a model)."""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import statistics


def runs(root: str):
    for meta_path in sorted(glob.glob(os.path.join(root, "*", "run.json"))):
        meta = json.load(open(meta_path))
        meta["_dir"] = os.path.dirname(meta_path)
        path = os.path.join(os.path.dirname(meta_path), "attempts.jsonl")
        attempts = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
        yield meta, attempts


def gb(n):
    return f"{n / 1e9:.1f} GB" if n else ""


def engine_name(meta: dict) -> str:
    """"kapitsa (ours) 0.2.36 (fe6a17a)": the engine, and its version without the engine's name repeated."""
    version = (meta.get("version") or "").strip()
    for prefix in (meta["engine"], "version:", "mtplx"):
        if version.lower().startswith(prefix.lower()):
            version = version[len(prefix):].strip()
    return meta["engine"] + (" (ours)" if meta.get("ours") else "") + (f" {version}" if version else "")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n."""
    if not n:
        return 0.0, 0.0
    p = k / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / (1 + z * z / n)), min(1.0, (centre + half) / (1 + z * z / n))


def fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test of [[a, b], [c, d]]: the probability, with the margins fixed, of a table as
    unlikely as this one or less."""
    n1, n2, k = a + b, c + d, a + c
    total = math.comb(n1 + n2, k)
    prob = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / total
    observed = prob(a)
    return min(1.0, sum(prob(x) for x in range(max(0, k - n2), min(k, n1) + 1) if prob(x) <= observed * (1 + 1e-9)))


def rejected_replies(run_dir: str, task: str, attempt: int) -> tuple[int, int]:
    """Replies the agent rejected in one attempt, from its trajectory: (no tool call, cut off at the token limit)."""
    path = os.path.join(run_dir, "trajectories", f"{task}.{attempt}.json")
    if not os.path.exists(path):
        return 0, 0
    no_call = cut_off = 0
    for m in json.load(open(path)).get("messages", []):
        if m.get("role") != "user":
            continue
        text = str(m.get("content"))
        no_call += text.startswith("Tool call error:")
        cut_off += text.startswith("Your previous response reached the output token limit")
    return no_call, cut_off


def pct(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, 95% CI {100 * lo:.0f}-{100 * hi:.0f}%)" if n else ""


def groups(root: str, notes: list | None = None):
    """Runs of the same model, engine, build and pack, whatever their task sets, pooled: name -> dict."""
    out = collections.OrderedDict()
    for meta, attempts in runs(root):
        if meta.get("load_failure"):
            continue
        key = (meta["model"], meta["engine"], meta.get("version"), meta["pack"])
        g = out.setdefault(key, {"meta": meta, "attempts": [], "sets": set(), "no_call": 0, "cut_off": 0,
                                 "replies": 0, "runs": [], "settings": []})
        if (meta.get("settings") or "defaults") not in g["settings"]:
            g["settings"].append(meta.get("settings") or "defaults")
        run_dir = meta["_dir"]
        g["runs"].append(os.path.basename(run_dir))
        for a in attempts:
            g["attempts"].append(a)
            g["sets"].add(a["set"])
            nc, co = rejected_replies(run_dir, a["task"], a["attempt"])
            g["no_call"] += nc
            g["cut_off"] += co
            g["replies"] += a.get("requests") or 0
    return out


def pooled_table(root: str, notes: list | None = None) -> str:
    """One row per model and engine build, every task set pooled; engines that were not run, and why."""
    lines = ["| Model | Engine | Build | Settings | Bits per weight | Tasks | Solved | Ended by malformed output | "
             "Replies without a tool call | Replies cut off at the token limit | Median time, solved attempts | "
             "Other failures |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (model, engine, version, pack), g in groups(root).items():
        att = g["attempts"]
        n, solved = len(att), sum(a["solved"] for a in att)
        malformed = sum(1 for a in att if a["failure"] in ("malformed output", "loop")
                        and a.get("exit_status") == "RepeatedFormatError")
        others = collections.Counter(a["failure"].split(":")[0] for a in att if not a["solved"]
                                     and not (a["failure"] in ("malformed output", "loop")
                                              and a.get("exit_status") == "RepeatedFormatError"))
        secs = [a["seconds"] for a in att if a["solved"]]
        tasks = len({a["task"] for a in att})
        lines.append(f"| {model} | {engine_name(g['meta'])} | {pack} | {'; '.join(g['settings'])} | "
                     f"{g['meta'].get('bits_per_weight') or ''} | "
                     f"{tasks} ({', '.join(sorted(g['sets']))}) | {pct(solved, n)} | {malformed}/{n} | "
                     f"{g['no_call']} of {g['replies']} | {g['cut_off']} | "
                     f"{f'{statistics.median(secs):.0f} s' if secs else ''} | "
                     f"{', '.join(f'{k} {v}' for k, v in others.most_common()) or 'none'} |")
    for note in notes or []:
        lines.append(f"| {note['model']} | {note['engine']} {note.get('version', '')} | {note.get('pack', '')} | | "
                     f"{note.get('bits_per_weight', '')} | | not run: {note['result']} | | | | | {note['why']} |")
    return "\n".join(lines) + "\n"


def pairwise_table(root: str) -> str:
    """For each model, every pair of engine builds: solved, and attempts ended by malformed output, with the
    two-sided Fisher exact p-value. With a few dozen attempts per engine, only large differences reach p < 0.05."""
    lines = ["| Model | A | B | Solved, A vs B | p | Ended by malformed output, A vs B | p |", "|---|---|---|---|---|---|---|"]
    by_model = collections.defaultdict(list)
    for key, g in groups(root).items():
        by_model[key[0]].append((key, g))
    for model, items in by_model.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (ka, ga), (kb, gb_) = items[i], items[j]

                def counts(g):
                    att = g["attempts"]
                    bad = sum(1 for a in att if a["failure"] in ("malformed output", "loop")
                              and a.get("exit_status") == "RepeatedFormatError")
                    return sum(a["solved"] for a in att), bad, len(att)
                sa, ba, na = counts(ga)
                sb, bb, nb = counts(gb_)
                lines.append(f"| {model} | {ka[1]} | {kb[1]} | {sa}/{na} vs {sb}/{nb} | "
                             f"{fisher(sa, na - sa, sb, nb - sb):.3f} | {ba}/{na} vs {bb}/{nb} | "
                             f"{fisher(ba, na - ba, bb, nb - bb):.3f} |")
    return "\n".join(lines) + "\n"


def table(root: str) -> str:
    lines = ["| Model | Engine | Build | Settings | Bits per weight | Size | Peak footprint | Cold start | Solved | Median time, "
             "solved attempts | Total time per solved task | Total time | Tokens written (median) | Requests (median) "
             "| Failures | Paged out |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for meta, attempts in runs(root):
        engine = engine_name(meta)
        if meta.get("load_failure"):
            lines.append(f"| {meta['model']} | {engine} | {meta['pack']} | {meta.get('settings') or 'defaults'} | | | | | "
                         f"does not start: {meta['load_failure']} | | | | | | | |")
            continue
        n = len(attempts)
        solved = sum(a["solved"] for a in attempts)
        secs = [a["seconds"] for a in attempts]
        solved_secs = [a["seconds"] for a in attempts if a["solved"]]
        toks = [a["completion_tokens"] for a in attempts if a.get("completion_tokens") is not None]
        reqs = [a["requests"] for a in attempts]
        failures = collections.Counter(a["failure"].split(":")[0] for a in attempts if not a["solved"])
        paged = sum(a.get("pageouts_delta") or 0 for a in attempts) * 16384
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} |".format(
            meta["model"], engine, meta["pack"], meta.get("settings") or "defaults", meta.get("bits_per_weight") or "",
            gb(meta.get("pack_bytes")),
            gb(meta.get("peak_footprint_bytes")),
            f"{meta['cold_start_seconds']:.0f} s" if meta.get("cold_start_seconds") else "",
            solved, n, f"{statistics.median(solved_secs):.0f} s" if solved_secs else "",
            f"{sum(secs) / solved:.0f} s" if solved else "nothing solved",
            f"{sum(secs) / 60:.1f} min" if secs else "",
            f"{statistics.median(toks):.0f}" if toks else "", f"{statistics.median(reqs):.0f}" if reqs else "",
            ", ".join(f"{k} {v}" for k, v in failures.most_common()) or "none", gb(paged) if paged else "none"))
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="solvetime report", description=__doc__)
    ap.add_argument("results")
    ap.add_argument("--speed", action="store_true", help="the speed results (speed-*.jsonl) instead of the agent runs")
    args = ap.parse_args(argv)
    if args.speed:
        from .speed import table as speed_table
        print(speed_table(sorted(glob.glob(os.path.join(args.results, "speed-*.jsonl")))), end="")
    else:
        print(table(args.results), end="")


if __name__ == "__main__":
    main()
