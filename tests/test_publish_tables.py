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

    def test_detached_footprint(self):
        self.run_dir("a-lmstudio", {"engine": "lmstudio", "version": "lmstudio llmster 0.0.27-1",
                                    "peak_footprint_bytes": 2e8}, [attempt("t", 1, True)])
        row = report.table(self.root).splitlines()[-1]
        self.assertIn("| not measured |", row)
        self.assertNotIn("0.2 GB", row)


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
            self.assertIn("| 317 (317-317) |", kept)
            self.assertIn("prefill runs 2, 3 left out: overlapped other work", kept)
            self.assertIn("| 15.0 GB | 15.0 GB |", kept)
            q = self.write(d, "speed-z-lmstudio.jsonl", [{**meta, "engine": "lmstudio"}, *prefill, *decode])
            self.assertIn("| 16.0 GB | not measured | not measured |", speed.table([q]))


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
