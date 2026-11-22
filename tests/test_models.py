"""The model table: every model is complete, an engine without a build says how to name one, and a model whose
parameters are not written down gets them from the official repo (here: from the cache)."""

import json
import os
import tempfile
import unittest
from unittest import mock

from solvetime import engines, models
from solvetime.proxy import Overrides
from solvetime.run import THINKING, sampling


class ModelTable(unittest.TestCase):
    def test_every_model_is_complete(self):
        for m in models.MODELS.values():
            self.assertTrue(set(m.builds) <= set(engines.ENGINES), m.name)
            self.assertEqual(set(m.sampling), set(THINKING), m.name)
            for thinking in THINKING:
                self.assertIsInstance(sampling(m.name, thinking), Overrides)
            # thinking off: the template's enable_thinking, or the model's own switch (Inkling 2: reasoning_effort)
            off = sampling(m.name, "no-thinking")
            self.assertTrue(off.enable_thinking is False or models.thinking_fields(m.name, "no-thinking"), m.name)
            self.assertNotEqual(models.thinking_fields(m.name, "no-thinking"), models.thinking_fields(m.name, "thinking"))
        self.assertEqual(set(models.BUILD_VARIABLE), set(engines.ENGINES))

    def test_each_qwen_runs_at_its_own_cards_temperature(self):
        # Qwen3.8's card: 1.0 in thinking mode (until 2026-11-22 solvetime sent it the 3.5 and 3.6 cards' 0.6).
        self.assertEqual(sampling("qwen3.8-27b", "thinking").temperature, 1.0)
        for name in ("qwen3.6-35b-a3b", "qwen3.5-9b"):   # their cards' setting for precise coding tasks
            self.assertEqual(sampling(name, "thinking").temperature, 0.6)
        for name in ("qwen3.8-27b", "qwen3.6-35b-a3b", "qwen3.5-9b"):
            self.assertEqual(sampling(name, "no-thinking").temperature, 0.7)

    def test_thinking_off_on_an_engine_that_reads_no_template_variable(self):
        own = {"chat_template_kwargs": {"reasoning_effort": "none"}}   # Inkling 2's thinking off (#56)
        # MTPLX reads enable_thinking and no other template variable, so it is sent beside Inkling's own switch
        mtplx = engines.mtplx("inkling-2-mini", 8890)
        self.assertEqual(mtplx.own_switch_off, {"enable_thinking": False})
        self.assertEqual(models.thinking_fields("inkling-2-mini", "no-thinking", mtplx.no_thinking, mtplx.own_switch_off),
                         {**own, "enable_thinking": False})
        self.assertEqual(models.thinking_fields("inkling-2-mini", "thinking", mtplx.no_thinking, mtplx.own_switch_off), {})
        # every other engine's Inkling requests are as in round 3
        for name in ("omlx", "llama.cpp"):
            e = engines.ENGINES[name]("inkling-2-mini", 8890)
            self.assertEqual(models.thinking_fields("inkling-2-mini", "no-thinking", e.no_thinking, e.own_switch_off), own)
        # and MTPLX's Qwen requests are unchanged
        self.assertEqual(mtplx.no_thinking, {})

    def test_unknown_model(self):
        with self.assertRaises(SystemExit):
            models.model("qwen9-1t")

    def test_no_build_names_the_variable(self):
        new = models.Model("new-27b", "Example/New-27B", 27 * 10**9, models.QWEN3_SAMPLING, "Qwen/Qwen3.5-9B", {})
        env = {"SOLVETIME_RACELINE": "/bin/echo"}
        with mock.patch.dict(models.MODELS, {"new-27b": new}), mock.patch.dict(os.environ, env):
            os.environ.pop("SOLVETIME_RACELINE_FILE", None)
            problem = engines.missing(engines.raceline("new-27b", 0))
            self.assertIn("no build of new-27b", problem)
            self.assertIn("SOLVETIME_RACELINE_FILE", problem)
            os.environ["SOLVETIME_OLLAMA_TAG"] = "new:27b"
            self.assertEqual(engines.ollama("new-27b", 0).api_model, "new:27b")
            os.environ.pop("SOLVETIME_OLLAMA_TAG")

    def test_parameters_from_the_cache(self):
        new = models.Model("new-27b", "Example/New-27B", None, models.QWEN3_SAMPLING, "Qwen/Qwen3.5-9B", {})
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "params.json")
            json.dump({"Example/New-27B": 2 * 10**10}, open(cache, "w"))
            with mock.patch.dict(models.MODELS, {"new-27b": new}), mock.patch.object(models, "CACHE", cache):
                self.assertEqual(models.text_params("new-27b"), 2 * 10**10)
                self.assertEqual(engines.bits(10**10, "new-27b"), 4.0)


if __name__ == "__main__":
    unittest.main()
