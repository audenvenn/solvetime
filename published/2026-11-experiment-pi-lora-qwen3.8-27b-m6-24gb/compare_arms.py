"""The pi-LoRA experiment's tables, from the runs' own records.

Usage: python3 compare_arms.py RUN_DIR...   (for this page: python3 compare_arms.py runs/*)

It expects exactly four runs: each arm (without the adapter, and with it: a run whose recorded launch command has
--lora) on each task set (core, extended), every task twice, with the same model file, engine, window and sampling.
Anything else is refused. In each engine log, the requests before the agent's first one are the harness's own (a
warm-up and two probes of the thinking switch); they are left out, and the remaining timings must match the
agent's records request for request (the count, and the tokens written in total). Numbers are rounded half up.
"""

import json
import math
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal

TIMING = re.compile(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens.*?\n.*?\|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens")


def fmt(x, places=1):
    """x rounded half up (26.65 -> 26.7), with thousands separators."""
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return f"{q:,}"


def load(d):
    meta = json.load(open(os.path.join(d, "run.json")))
    rows = [json.loads(line) for line in open(os.path.join(d, "attempts.jsonl")) if line.strip()]
    command = meta.get("command") or []
    lora = command[command.index("--lora") + 1] if "--lora" in command else None
    sets = {r.get("set") for r in rows}
    if len(sets) != 1:
        raise SystemExit(f"{d}: attempts from more than one task set: {sets}")
    timings = [(float(pm), int(pn), float(dm), int(dn)) for pm, pn, dm, dn in TIMING.findall(open(os.path.join(d, "engine.log"), errors="replace").read())]
    requests = sum(r["requests"] for r in rows)
    setup = len(timings) - requests
    if setup != 3:
        raise SystemExit(f"{d}: {len(timings)} timings for {requests} agent requests; expected 3 setup requests before them")
    agent = timings[setup:]
    written = sum(r["completion_tokens"] for r in rows)
    if sum(t[3] for t in agent) != written:
        raise SystemExit(f"{d}: the engine wrote {sum(t[3] for t in agent)} tokens for the agent's requests, the agent counted {written}")
    return {"dir": os.path.basename(d.rstrip("/")), "meta": meta, "rows": rows, "arm": "with" if lora else "without",
            "lora": os.path.basename(lora) if lora else None, "set": sets.pop(), "agent": agent}


def check(runs):
    keys = Counter((r["arm"], r["set"]) for r in runs)
    expected = {(a, s) for a in ("without", "with") for s in ("core", "extended")}
    if set(keys) != expected or any(n != 1 for n in keys.values()):
        raise SystemExit(f"expected one run per arm and task set, got {dict(keys)}")
    for r in runs:
        tasks = Counter(x["task"] for x in r["rows"])
        if len(tasks) != 8 or set(tasks.values()) != {2}:
            raise SystemExit(f"{r['dir']}: expected 8 tasks twice each, got {dict(tasks)}")
    same = lambda f: {json.dumps(f(r["meta"]), sort_keys=True) for r in runs}
    for name, f in (("model file", lambda m: m.get("pack")), ("engine", lambda m: m.get("version")),
                    ("sampling", lambda m: m.get("sampling")), ("thinking", lambda m: m.get("thinking")),
                    ("window and cache", lambda m: (m.get("settings") or "").split(", lora")[0])):
        if len(same(f)) != 1:
            raise SystemExit(f"the runs differ in {name}: {same(f)}")
    if len({r["lora"] for r in runs if r["lora"]}) != 1:
        raise SystemExit("the adapter runs used different adapter files")


def sign_test(lower, higher):
    """Two-sided exact sign test, ties dropped."""
    n, k = lower + higher, min(lower, higher)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(dirs):
    runs = [load(d) for d in dirs]
    check(runs)
    arms = {a: [r for r in runs if r["arm"] == a] for a in ("without", "with")}
    rows = {a: [x for r in rs for x in r["rows"]] for a, rs in arms.items()}
    print("## Runs, in the order they ran\n")
    print("| Run | Started | Solved | Total time | Estimated available before launch | System-wide anonymous memory |")
    print("|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda r: r["meta"]["started"]):
        m = r["meta"]["memory_at_start"]
        print(f"| {r['arm']} the adapter, {r['set']} | {r['meta']['started'][11:]} | {sum(x['solved'] for x in r['rows'])}/{len(r['rows'])} "
              f"| {fmt(sum(x['seconds'] for x in r['rows']) / 60)} min | {fmt(m['available'] / 1e9)} GB | {fmt(m['apps'] / 1e9)} GB |")
    print("\n## Each arm\n")
    print("| Arm | Solved | Steps per attempt, mean (median) | Tokens written per attempt, mean (median) | Time per attempt, mean (median) | Total time |")
    print("|---|---|---|---|---|---|")
    for a in ("without", "with"):
        x = rows[a]
        req, tok, sec = [r["requests"] for r in x], [r["completion_tokens"] for r in x], [r["seconds"] for r in x]
        print(f"| {a.capitalize()} the adapter | {sum(r['solved'] for r in x)}/{len(x)} | {fmt(st.mean(req), 2)} ({fmt(st.median(req), 0)}) "
              f"| {fmt(st.mean(tok), 0)} ({fmt(st.median(tok), 0)}) | {fmt(st.mean(sec), 0)} s ({fmt(st.median(sec), 0)} s) | {fmt(sum(sec) / 60)} min |")
    print("\nExits:", {a: dict(Counter(r["exit_status"] for r in x)) for a, x in rows.items()})
    print("\n## Each task (means of the two attempts, without / with the adapter)\n")
    print("| Task | Set | Solved | Steps | Tokens written | Seconds |")
    print("|---|---|---|---|---|---|")
    by = defaultdict(lambda: defaultdict(list))
    for a, x in rows.items():
        for r in x:
            by[r["task"]][a].append(r)
    tally = {k: [0, 0, 0] for k in ("requests", "completion_tokens", "seconds")}
    saved = {"failed in both arms": 0.0, "solved in both arms": 0.0, "other": 0.0}
    for task in sorted(by, key=lambda t: (by[t]["without"][0]["set"], t)):
        w, l = by[task]["without"], by[task]["with"]
        cells = []
        for k in tally:
            mw, ml = st.mean(r[k] for r in w), st.mean(r[k] for r in l)
            tally[k][0 if ml < mw else 1 if ml > mw else 2] += 1
            cells.append(f"{fmt(mw, 1)} / {fmt(ml, 1)}" if k == "requests" else f"{fmt(mw, 0)} / {fmt(ml, 0)}")
        sw, sl = sum(r["solved"] for r in w), sum(r["solved"] for r in l)
        kind = "failed in both arms" if sw == sl == 0 else "solved in both arms" if sw == len(w) and sl == len(l) else "other"
        saved[kind] += sum(r["seconds"] for r in w) - sum(r["seconds"] for r in l)
        print(f"| {task} | {w[0]['set']} | {sw}/{len(w)} / {sl}/{len(l)} | " + " | ".join(cells) + " |")
    print()
    for k, (lo, hi, tie) in tally.items():
        p = sign_test(lo, hi)
        print(f"- {k}: with the adapter lower on {lo} tasks, higher on {hi}, equal on {tie}; two-sided sign test "
              f"p = {p:.3f} unadjusted, {min(1.0, 3 * p):.3f} after a Bonferroni adjustment across the three")
    print("- time saved with the adapter: " + ", ".join(f"{k} {fmt(v / 60, 2)} min" for k, v in saved.items()))
    print("\n## The engine, on the agent's requests only (llama-server's own timings)\n")
    print("| Arm | Agent requests | Prompt processing | Writing tokens (decode) |")
    print("|---|---|---|---|")
    tot = {}
    for a, rs in arms.items():
        t = [x for r in rs for x in r["agent"]]
        pm, pn, dm, dn = (sum(x[i] for x in t) for i in range(4))
        tot[a] = (pm, pn, dm, dn)
        print(f"| {a.capitalize()} the adapter | {len(t)} | {fmt(pn, 0)} tokens in {fmt(pm / 60000, 2)} min ({fmt(pn / pm * 1000)} tok/s) "
              f"| {fmt(dn, 0)} tokens in {fmt(dm / 60000, 2)} min ({fmt(dn / dm * 1000, 2)} tok/s) |")
    (pw, nw, dw, _), (pl, nl, dl, _) = tot["without"], tot["with"]
    at_base = nl / (nw / pw)
    print(f"\n- prompt time per token with the adapter: {fmt((pl / nl) / (pw / nw) * 100 - 100)}% more")
    print(f"- the adapter arm's prompts at the other arm's rate: {fmt(at_base / 60000, 2)} min instead of {fmt(pl / 60000, 2)} "
          f"(an estimate: {fmt((pl - at_base) / 60000, 2)} min)")
    print(f"- decoding: {fmt((dw - dl) / 60000, 2)} min less with the adapter; prompts: {fmt((pl - pw) / 60000, 2)} min more")
    for a in ("without", "with"):
        x = rows[a]
        print(f"- decoding's share of the {a}-adapter arm's total time: {fmt(tot[a][2] / 1000 / sum(r['seconds'] for r in x) * 100)}%")
    print("- the first agent prompt of each run (the same prompt in both arms for a task set):")
    for s in ("core", "extended"):
        firsts = {r["arm"]: r["agent"][0] for r in runs if r["set"] == s}
        print(f"  - {s}: " + "; ".join(f"{a} the adapter {firsts[a][1]} tokens at {fmt(firsts[a][1] / firsts[a][0] * 1000)} tok/s" for a in ("without", "with")))


if __name__ == "__main__":
    main(sys.argv[1:])
