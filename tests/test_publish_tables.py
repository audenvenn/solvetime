"""The published tables, from records like the 27B round's (2026-11-19):
- a run with settings beyond the defaults never pools with one without them, and the pairs say which is which;
- an engine that returns no usage has its tokens "not reported", not 0;
- an engine whose server runs outside the launched process tree has its RSS and footprint "not measured";
- a speed record of an engine that did not start shows the engine's display name;
- declared speed runs are left out of a cell and named in the row's notes;
- validate does not hold a run whose engine never started to the sampling it would have sent."""

import json
import os
import tempfile
import unittest
from unittest import mock

from solvetime import report, speed, validate


def attempt(task, n, solved, tokens=900):
    return {"task": task, "attempt": n, "set": "core", "solved": solved, "seconds": 300.0, "request_seconds": 200.0,
            "requests": 9, "completion_tokens": tokens, "failure": "" if solved else "timeout",
            "exit_status": "Submitted" if solved else "TimeExceeded", "pageouts_delta": 0}


class Tables(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.root = d.name

    def run_dir(self, name, meta, attempts):
        path = os.path.join(self.root, name)
        os.makedirs(path)
        base = {"model": "qwen3.8-27b", "pack": "mlx-community/Qwen3.8-27B-4bit", "settings": "",
                "memory_at_start": {"available": 14e9}}
        with open(os.path.join(path, "run.json"), "w") as f:
            json.dump({**base, **meta}, f)
        with open(os.path.join(path, "attempts.jsonl"), "w") as f:
            f.writelines(json.dumps(a) + "\n" for a in attempts)
        open(os.path.join(path, "engine.log"), "w").close()

    def test_settings_never_pool(self):
        self.run_dir("a-mlx-lm", {"engine": "mlx-lm", "version": "0.32.2"}, [attempt("t", 1, True), attempt("t", 2, False)])
        self.run_dir("b-mlx-lm", {"engine": "mlx-lm", "version": "0.32.2", "settings": "max tokens 16384"},
                     [attempt("t", 1, True), attempt("t", 2, True)])
        self.assertEqual(len(report.groups(self.root)), 2)
        pooled = report.pooled_table(self.root)
        self.assertNotIn("4 tasks", pooled)
        self.assertEqual(pooled.count("| mlx-lm 0.32.2 |"), 2)
        pairs = report.pairwise_table(self.root)
        self.assertIn("| mlx-lm 0.32.2 | mlx-lm 0.32.2 (max tokens 16384) | 1/2 vs 2/2 |", pairs)

    def test_tokens_not_reported(self):
        self.run_dir("a-omlx", {"engine": "omlx", "version": "0.8.0"},
                     [attempt("t", 1, True, tokens=None), attempt("t", 2, True, tokens=None)])
        self.assertIn("| 18 | not reported |", report.time_table(self.root))
        self.assertIn("| not reported |", report.table(self.root))

    def test_tokens_reported_by_some_attempts(self):
        # oMLX reported usage for 1 of 16 attempts in round 3 (2026-11-21): the sum says how many
        self.run_dir("a-omlx", {"engine": "omlx", "version": "0.8.1"},
                     [attempt("t", 1, True, tokens=46), attempt("t", 2, True, tokens=None)])
        self.assertIn("| 18 | 46 (1/2 attempts) |", report.time_table(self.root))

    def test_a_pack_by_its_published_name(self):
        folder = "/Users/someone/models/Qwen3.8-27B-4bit"
        self.run_dir("a-omlx", {"engine": "omlx", "version": "0.8.0", "pack": folder}, [attempt("t", 1, True)])
        with mock.patch.dict(report.DISPLAY, {folder: "mlx-community/Qwen3.8-27B-4bit (b935172)"}):
            for table in (report.pooled_table(self.root), report.table(self.root)):
                self.assertIn("| mlx-community/Qwen3.8-27B-4bit (b935172) |", table)
                self.assertNotIn(folder, table)

    def test_detached_footprint(self):
        self.run_dir("a-lmstudio", {"engine": "lmstudio", "version": "lmstudio llmster 0.0.27-1",
                                    "peak_footprint_bytes": 2e8}, [attempt("t", 1, True)])
        row = report.table(self.root).splitlines()[-1]
        self.assertIn("| not measured |", row)
        self.assertNotIn("0.2 GB", row)


    def test_unscored_starts_and_cells_per_row(self):
        self.run_dir("a-omlx", {"engine": "omlx", "version": "0.8.0",
                                "load_failure": "its first request failed: HTTP 507: does not fit"}, [])
        self.run_dir("b-lmstudio", {"engine": "lmstudio", "version": "lmstudio llmster 0.0.27-1",
                                    "load_failure": "exited with code -15: Error: Model loading was stopped"}, [])
        self.run_dir("c-mtplx", {"engine": "mtplx", "version": "mtplx 2.12.3"},
                     [attempt("t", 1, False, tokens=138), attempt("t", 2, False, tokens=None)])
        notes = [{"model": "qwen3.8-27b", "engine": "MTPLX", "result": "speed not measured", "why": "no token"},
                 {"model": "qwen3.8-27b", "engine": "Ollama", "result": "does not fit", "why": "critical"}]
        pooled = report.pooled_table(self.root, notes)
        self.assertIn("0 attempts scored — server started, warm-up refused: its first request failed", pooled)
        self.assertIn("0 attempts scored — engine exited before answering: exited with code -15", pooled)
        self.assertNotIn("speed not measured", pooled)          # a speed-only note belongs under Speed
        self.assertIn("not run: does not fit", pooled)
        self.assertIn("Failed checks after a repeated-format exit", pooled)
        self.assertIn("Failed checks after a repeated-format exit, A vs B", report.pairwise_table(self.root))
        self.assertIn("Summed proxy request duration", report.time_table(self.root))
        each = report.table(self.root)
        self.assertIn("| 138 (1/2 attempts) |", each)
        for table in (pooled, each, report.time_table(self.root)):
            rows = [r for r in table.strip().splitlines() if r.startswith("|")]
            cells = lambda r: len(r.strip()[1:-1].split("|"))
            self.assertEqual({cells(r) for r in rows}, {cells(rows[0])}, table)


class SpeedTable(unittest.TestCase):
    def write(self, d, name, rows):
        path = os.path.join(d, name)
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        return path

    def test_does_not_start_shows_the_display_name(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(report.DISPLAY, {"omlx 0.8.0": "oMLX 0.8.0"}):
            p = self.write(d, "speed-x-omlx.jsonl", [{"test": "load", "load_failure": "HTTP 507", "engine": "omlx",
                                                      "version": "0.8.0", "model": "qwen3.8-27b"}])
            self.assertIn("| qwen3.8-27b | oMLX 0.8.0 |", speed.table([p]))

    def test_left_out_runs_and_detached_memory(self):
        meta = {"test": "meta", "engine": "omlx", "version": "0.8.0", "model": "qwen3.8-27b"}
        prefill = [{"test": "prefill", "prefill_tok_s": v} for v in (317.0, 302.0, 301.0)]
        decode = [{"test": "decode write T=0.6", "decode_tok_s": 9.9, "completion_tokens": 511,
                   "rss_peak_bytes": 15e9, "footprint_peak_bytes": 15e9, "system_taken_bytes": 16e9}]
        with tempfile.TemporaryDirectory() as d:
            p = self.write(d, "speed-y-omlx.jsonl", [meta, *prefill, *decode])
            whole = speed.table([p])
            self.assertIn("| 302 (301-317) |", whole)
            kept = speed.table([p], {"speed-y-omlx.jsonl": {"prefill": [2, 3], "why": "overlapped other work"}})
            self.assertIn("| 317 (1 retained of 3) |", kept)
            self.assertIn("prefill runs 2, 3 left out: overlapped other work", kept)
            self.assertIn("| 15.0 GB | 15.0 GB |", kept)
            q = self.write(d, "speed-z-lmstudio.jsonl", [{**meta, "engine": "lmstudio"}, *prefill, *decode])
            self.assertIn("| 16.0 GB | not measured | not measured |", speed.table([q]))


    def test_settings_column_and_cells_per_row(self):
        meta = {"test": "meta", "engine": "omlx", "version": "0.8.0", "model": "qwen3.8-27b", "settings": "memory guard aggressive"}
        rows = [meta, {"test": "prefill", "prefill_tok_s": 317.0},
                {"test": "decode write T=0.6", "decode_tok_s": 9.9, "completion_tokens": 511}]
        with tempfile.TemporaryDirectory() as d:
            a = self.write(d, "speed-a.jsonl", rows)
            b = self.write(d, "speed-b.jsonl", [{"test": "load", "engine": "omlx", "version": "0.8.0", "model": "qwen3.8-27b",
                                                 "load_failure": "its first request failed: HTTP 507"}])
            out = speed.table([a, b])
        self.assertIn("| memory guard aggressive |", out)
        self.assertIn("| defaults |", out)
        self.assertIn("no speed result — server started, warm-up refused", out)
        lines = [r for r in out.strip().splitlines() if "---" not in r]
        cells = lambda r: len(r.strip()[1:-1].split("|"))
        self.assertEqual({cells(r) for r in lines}, {cells(lines[0])}, out)


class Publish(unittest.TestCase):
    def test_speed_only_notes_go_under_speed(self):
        from solvetime import publish
        with tempfile.TemporaryDirectory() as res, tempfile.TemporaryDirectory() as out:
            run = os.path.join(res, "20261119-000000-mtplx-qwen3.8-27b")
            os.makedirs(run)
            meta = {"model": "qwen3.8-27b", "engine": "mtplx", "version": "mtplx 2.12.3", "pack": "p", "settings": "",
                    "started": "2026-11-19 21:16:21", "finished": "2026-11-19 22:28:00", "agent": "mini-swe-agent 2.4.6",
                    "step_limit": 50, "wall_seconds": 900, "sampling": {}, "mac": {"chip": "M6", "memory_gb": 24}}
            with open(os.path.join(run, "run.json"), "w") as f:
                json.dump(meta, f)
            with open(os.path.join(run, "attempts.jsonl"), "w") as f:
                f.write(json.dumps(attempt("t", 1, False)) + "\n")
            open(os.path.join(run, "engine.log"), "w").close()
            notes = os.path.join(res, "not-run.json")
            with open(notes, "w") as f:
                json.dump([{"model": "qwen3.8-27b", "engine": "MTPLX", "result": "speed not measured", "why": "no token back."},
                           {"model": "qwen3.8-27b", "engine": "Ollama", "result": "does not fit", "why": "critical."}], f)
            publish.main([res, out, "--speed", "none-*", "--notes", notes])
            page = open(os.path.join(out, "README.md")).read()
        speed_part = page[page.index("## Speed"):page.index("## Not run")]
        self.assertIn("MTPLX", speed_part)
        self.assertIn("no speed result. no token back.", speed_part)
        self.assertNotIn("MTPLX", page[page.index("## Not run"):])
        self.assertIn("a 900 s budget the agent checks before each query", page)


class Validate(unittest.TestCase):
    def test_a_record_of_an_engine_that_did_not_start(self):
        meta = {"agent": validate.AGENT, "step_limit": validate.STEP_LIMIT, "wall_seconds": validate.WALL_SECONDS,
                "model": "qwen3.8-27b", "thinking": "thinking", "tasks": [],
                "load_failure": "its first request failed: HTTP 507"}
        self.assertFalse([p for p in validate.settings_problems(meta) if "sampling" in p])
        started = {**meta, "load_failure": None, "overrides_sent": {}}
        self.assertTrue([p for p in validate.settings_problems(started) if "sampling" in p])


if __name__ == "__main__":
    unittest.main()
