"""Memory guards other than the engines' defaults: oMLX's tier and LM Studio's model-loading guardrail, passed to
the engine and shown in the tables; LM Studio's is set for the run only and put back afterwards."""

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from solvetime import engines

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER = os.path.join(HERE, "solvetime", "launchers", "lmstudio.sh")

FAKE_LMS = """#!/bin/bash
case "$1" in
  load) python3 -c "import json, sys; print('mode during load:', json.load(open(sys.argv[1]))['localModels']['modelLoadingGuardrails']['mode'])" "$SOLVETIME_LMSTUDIO_SETTINGS"
        echo "Error: Model loading was stopped due to insufficient system resources."; exit 1;;
  log) sleep 30;;
  *) exit 0;;
esac
"""


class OMLXGuard(unittest.TestCase):
    def engine(self, **env):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"SOLVETIME_OMLX_MODEL": d, **env}):
            if "SOLVETIME_OMLX_MEMORY_GUARD" not in env:
                os.environ.pop("SOLVETIME_OMLX_MEMORY_GUARD", None)
            return engines.omlx("qwen3.8-27b", 8892)

    def test_the_default(self):
        e = self.engine()
        self.assertNotIn("--memory-guard", e.command)
        self.assertEqual(e.settings, "")

    def test_a_tier(self):
        e = self.engine(SOLVETIME_OMLX_MEMORY_GUARD="aggressive")
        self.assertEqual(e.command[-2:], ["--memory-guard", "aggressive"])
        self.assertEqual(e.settings, "memory guard aggressive")

    def test_its_own_settings_folder(self):
        e = self.engine()
        base = e.command[e.command.index("--base-path") + 1]
        self.assertEqual(os.listdir(base), [])          # fresh: nothing an earlier run saved
        open(os.path.join(base, "settings.json"), "w").close()
        e = self.engine()
        self.assertEqual(os.listdir(base), [])

    def test_not_a_tier(self):
        with self.assertRaises(ValueError):
            self.engine(SOLVETIME_OMLX_MEMORY_GUARD="loose")


class LMStudioGuardrails(unittest.TestCase):
    def test_passed_and_shown(self):
        with mock.patch.dict(os.environ, {"SOLVETIME_LMSTUDIO_GUARDRAILS": "low", "SOLVETIME_LMSTUDIO_CTX": "16384"}):
            e = engines.lmstudio("qwen3.8-27b", 8895)
        self.assertEqual(e.env["SOLVETIME_LMSTUDIO_GUARDRAILS"], "low")
        self.assertEqual(e.settings, "context 16384, guardrails low")

    def test_the_default(self):
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("SOLVETIME_LMSTUDIO_GUARDRAILS", None)
            e = engines.lmstudio("qwen3.8-27b", 8895)
        self.assertNotIn("SOLVETIME_LMSTUDIO_GUARDRAILS", e.env)
        self.assertNotIn("guardrails", e.settings)

    def test_the_launcher_sets_it_for_the_run_and_puts_it_back(self):
        with tempfile.TemporaryDirectory() as d:
            lms = os.path.join(d, "lms")
            with open(lms, "w") as f:
                f.write(FAKE_LMS)
            os.chmod(lms, os.stat(lms).st_mode | stat.S_IEXEC)
            settings = os.path.join(d, "settings.json")
            before = {"appLanguage": "en", "localModels": {"autoFitMinContextLength": 16384, "modelLoadingGuardrails": {
                "mode": "relaxed", "customThresholdBytes": 4294967296, "alwaysAllowLoadAnyway": False}}}
            with open(settings, "w") as f:
                json.dump(before, f)
            env = dict(os.environ, SOLVETIME_LMS=lms, SOLVETIME_LMSTUDIO_SETTINGS=settings,
                       SOLVETIME_LMSTUDIO_GUARDRAILS="off")
            p = subprocess.run(["/bin/bash", LAUNCHER, "8899", "qwen/qwen3.8-27b", "solvetime", "16384"], env=env,
                               capture_output=True, text=True, timeout=60, start_new_session=True)
            with open(settings) as f:
                after = json.load(f)
        self.assertIn("### guardrails off for this run (was relaxed)", p.stdout)
        self.assertIn("mode during load: off", p.stdout)
        self.assertIn("### guardrails back to relaxed", p.stdout)
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
