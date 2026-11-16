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
            self.assertEqual(sampling(m.name, "no-thinking").enable_thinking, False)
        self.assertEqual(set(models.BUILD_VARIABLE), set(engines.ENGINES))

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
