"""Fixes from the independent review of round 3's page (2026-11-21):
- a separate drafter's file is left out of a build's counted bytes (oMLX's oQ builds of Inkling 2 keep their DSpark
  drafter in mtp.safetensors, which the model's parameter count does not include);
- a run whose engine never served an attempt shows the engine's message to its first sentence;
- a median request count keeps its half;
- the speed table's prefill cell gives the prompt's size in the model's own tokens;
- a pack recorded as a local folder is published under the name --names gives it."""

import json
import os
import struct
import tempfile
import unittest
from unittest import mock

from solvetime import engines, publish, report, speed


def safetensors(path, tensors):
    header = {name: {"dtype": "U8", "shape": [n], "data_offsets": [0, n]} for name, n in tensors.items()}
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + b"\0" * max(tensors.values()))


class Fixes(unittest.TestCase):
    def test_drafter_files_left_out(self):
        with tempfile.TemporaryDirectory() as d:
            safetensors(os.path.join(d, "model-00001-of-00001.safetensors"), {"layers.0.w": 1000, "vision_tower.w": 50})
            safetensors(os.path.join(d, "mtp.safetensors"), {"layers.0.w": 200})
            self.assertEqual(engines.text_weight_bytes(d), 1200)
            self.assertEqual(engines.text_weight_bytes(d, ("mtp.safetensors",)), 1000)

    def test_unscored_first_sentence(self):
        meta = {"load_failure": "its first request failed: HTTP 400: Prefill would require ~14.86 GB peak but the "
                                "prefill limit is 14.58 GB (margin 1.27 GB). Close other apps, or raise the tier"}
        out = report.unscored(meta)
        self.assertTrue(out.endswith("(margin 1.27 GB)."), out)
        self.assertNotIn("Close other apps", out)

    def test_prefill_prompt_size(self):
        rows = [{"test": "prefill", "prompt_tokens": n} for n in (5013, 5014, 5014)]
        self.assertEqual(speed.prompt_size(rows), ", 5.0K tokens")
        self.assertEqual(speed.prompt_size([{"test": "decode write T=0.6"}]), "")

    def test_published_pack_names(self):
        with tempfile.TemporaryDirectory() as res, tempfile.TemporaryDirectory() as out:
            folder = os.path.join(os.path.expanduser("~"), "models", "Some-Model-oQ3e")
            run = os.path.join(res, "20261121-000000-omlx-inkling-2-mini")
            os.makedirs(run)
            meta = {"model": "inkling-2-mini", "engine": "omlx", "version": "0.8.1", "pack": folder, "settings": "",
                    "started": "2026-11-21 13:47:00", "finished": "2026-11-21 15:10:00", "agent": "mini-swe-agent 2.4.6",
                    "step_limit": 50, "wall_seconds": 900, "sampling": {}, "mac": {"chip": "M6", "memory_gb": 24}}
            json.dump(meta, open(os.path.join(run, "run.json"), "w"), indent=2)
            attempt = {"task": "t", "attempt": 1, "set": "core", "solved": True, "seconds": 60.0, "request_seconds": 50.0,
                       "requests": 9, "completion_tokens": 10, "failure": "", "exit_status": "Submitted"}
            open(os.path.join(run, "attempts.jsonl"), "w").write(json.dumps(attempt) + "\n")
            open(os.path.join(run, "engine.log"), "w").close()
            with open(os.path.join(res, "speed-20261121-000000-omlx-inkling-2-mini.jsonl"), "w") as f:
                f.write(json.dumps({"test": "meta", "engine": "omlx", "version": "0.8.1", "model": "inkling-2-mini",
                                    "pack": folder}) + "\n")
            names = os.path.join(res, "names.json")
            json.dump({publish.scrub(folder): "example/Some-Model-oQ3e (abc1234)"}, open(names, "w"))
            with mock.patch.dict(report.DISPLAY, {}):
                publish.main([res, out, "--names", names])
            self.assertEqual(json.load(open(os.path.join(out, "runs", os.path.basename(run), "run.json")))["pack"],
                             "example/Some-Model-oQ3e (abc1234)")
            first = json.loads(open(os.path.join(out, "speed", "speed-20261121-000000-omlx-inkling-2-mini.jsonl")).readline())
            self.assertEqual(first["pack"], "example/Some-Model-oQ3e (abc1234)")
            self.assertNotIn("Some-Model-oQ3e |", open(os.path.join(out, "README.md")).read().replace("example/", ""))


if __name__ == "__main__":
    unittest.main()
