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


# Names the tables show instead of the recorded ones (publish --names), e.g. a release's name for the build it
# shipped: {"raceline (ours) 0.7.1 (v0.7.1-6-g3de235d)": "raceline 0.8.0 (ours, 3de235d)"}. run.json keeps the
# recorded version.
DISPLAY: dict[str, str] = {}


def engine_name(meta: dict) -> str:
    """"kapitsa 0.2.36 (fe6a17a)": the engine, and its version without the engine's name repeated (or the name
    DISPLAY gives it)."""
    version = (meta.get("version") or "").strip()
    for prefix in (meta["engine"], "version:", "mtplx"):
        if version.lower().startswith(prefix.lower()):
            version = version[len(prefix):].strip()
    name = meta["engine"] + (" (ours)" if meta.get("ours") else "") + (f" {version}" if version else "")
    return DISPLAY.get(name, name)


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


def pval(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


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
                                 "replies": 0, "runs": [], "settings": [], "available": []})
        if (meta.get("settings") or "defaults") not in g["settings"]:
            g["settings"].append(meta.get("settings") or "defaults")
        run_dir = meta["_dir"]
        g["runs"].append(os.path.basename(run_dir))
        available = (meta.get("memory_at_start") or {}).get("available")
        if available:
            g["available"].append(available)
        for a in attempts:
            g["attempts"].append(a)
            g["sets"].add(a["set"])
            nc, co = rejected_replies(run_dir, a["task"], a["attempt"])
            g["no_call"] += nc
            g["cut_off"] += co
            g["replies"] += a.get("requests") or 0
    return out


def engine_log_rates(run_dir: str, engine: str) -> tuple[list[float], list[float]]:
    """Per-request speeds the engine logged during the run, for the engines that log them: (decode tok/s of
    requests that wrote at least 64 tokens, prefill tok/s of requests that read at least 256 new tokens)."""
    import re
    path = os.path.join(run_dir, "engine.log")
    if not os.path.exists(path):
        return [], []
    text = open(path, errors="replace").read()
    decode, prefill = [], []
    if engine == "raceline":
        for m in re.finditer(r"prompt (\d+) tok \((\d+) cached, ([0-9.]+)s\)\s+out (\d+) tok at ([0-9.]+) tok/s", text):
            new, secs, out, rate = int(m[1]) - int(m[2]), float(m[3]), int(m[4]), float(m[5])
            if out >= 64:
                decode.append(rate)
            if new >= 256 and secs > 0:
                prefill.append(new / secs)
    elif engine == "kapitsa":
        for m in re.finditer(r"chat: prompt (\d+) \(cached (\d+)\) in ([0-9.]+) ms.*?, (\d+) tokens in ([0-9.]+) ms", text):
            new, ms, out, out_ms = int(m[1]) - int(m[2]), float(m[3]), int(m[4]), float(m[5])
            if out >= 64 and out_ms > 0:
                decode.append(out / out_ms * 1000)
            if new >= 256 and ms > 0:
                prefill.append(new / ms * 1000)
    elif engine == "llama.cpp":
        for m in re.finditer(r"prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens.*?\n.*?\beval time =\s*([0-9.]+) ms /\s*(\d+) tokens", text):
            ms, new, out_ms, out = float(m[1]), int(m[2]), float(m[3]), int(m[4])
            if out >= 64 and out_ms > 0:
                decode.append(out / out_ms * 1000)
            if new >= 256 and ms > 0:
                prefill.append(new / ms * 1000)
    return decode, prefill


def time_table(root: str) -> str:
    """One row per run: how long its attempts took end to end, per solved task, and in the engine, with the
    speeds the engine logged while the agent worked (medians). Runs with settings beyond the defaults are
    marked: their times compare with nothing else."""
    lines = ["| Model | Engine | Task set | Settings | Attempts | Solved | Total time | Per solved task | Requests | "
             "Tokens written | Time in the engine | Decode during the run (median) | Prefill during the run (median) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for meta, attempts in runs(root):
        if meta.get("load_failure") or not attempts:
            continue
        n, solved = len(attempts), sum(a["solved"] for a in attempts)
        wall = sum(a["seconds"] for a in attempts)
        req = sum(a.get("request_seconds") or 0 for a in attempts)
        toks = sum(a.get("completion_tokens") or 0 for a in attempts)
        decode, prefill = engine_log_rates(meta["_dir"], meta["engine"])
        med = lambda v: f"{statistics.median(v):.0f} tok/s ({len(v)} requests)" if v else "not logged"
        sets = ", ".join(sorted({a["set"] for a in attempts}))
        settings = meta.get("settings") or "defaults"
        lines.append(f"| {meta['model']} | {engine_name(meta)} | {sets} | {settings}"
                     f"{' (times not comparable)' if settings != 'defaults' else ''} | {n} | {solved} | "
                     f"{wall / 60:.1f} min | {f'{wall / solved:.0f} s' if solved else ''} | "
                     f"{sum(a['requests'] for a in attempts)} | {toks} | {req / 60:.1f} min | {med(decode)} | "
                     f"{med(prefill)} |")
    return "\n".join(lines) + "\n"


def gb_range(values: list) -> str:
    """"12.3 GB", or "11.0-12.3 GB" over several runs; "" when none was recorded."""
    if not values:
        return ""
    lo, hi = min(values) / 1e9, max(values) / 1e9
    return f"{lo:.1f} GB" if hi - lo < 0.05 else f"{lo:.1f}-{hi:.1f} GB"


def pooled_table(root: str, notes: list | None = None) -> str:
    """One row per model and engine build, every task set pooled; engines that were not run, and why."""
    lines = ["| Model | Engine | Build | Settings | Bits per weight | Available at start | Tasks | Solved | "
             "Ended by malformed output | Replies without a tool call | Replies cut off at the token limit | "
             "Median time, solved attempts | Other failures |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
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
                     f"{g['meta'].get('bits_per_weight') or ''} | {gb_range(g['available'])} | "
                     f"{tasks} ({', '.join(sorted(g['sets']))}) | {pct(solved, n)} | {malformed}/{n} | "
                     f"{g['no_call']} of {g['replies']} | {g['cut_off']} | "
                     f"{f'{statistics.median(secs):.0f} s' if secs else ''} | "
                     f"{', '.join(f'{k} {v}' for k, v in others.most_common()) or 'none'} |")
    for note in notes or []:
        lines.append(f"| {note['model']} | {note['engine']} {note.get('version', '')} | {note.get('pack', '')} | | "
                     f"{note.get('bits_per_weight', '')} | | | not run: {note['result']} | | | | | {note['why']} |")
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
                lines.append(f"| {model} | {engine_name(ga['meta'])} | {engine_name(gb_['meta'])} | {sa}/{na} vs {sb}/{nb} | "
                             f"{pval(fisher(sa, na - sa, sb, nb - sb))} | {ba}/{na} vs {bb}/{nb} | "
                             f"{pval(fisher(ba, na - ba, bb, nb - bb))} |")
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
