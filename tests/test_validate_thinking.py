"""validate checks that a run got the thinking it asked for: a model that keeps thinking with its switch off
(Inkling 2 writes a short thinking block at reasoning_effort "none", 2026-11-21) is no problem for a run with
thinking on, and is one for a run with thinking off."""

import unittest

from solvetime import validate


def problems(mode, switch):
    meta = {"agent": validate.AGENT, "step_limit": validate.STEP_LIMIT, "wall_seconds": validate.WALL_SECONDS,
            "model": "inkling-2-mini", "thinking": mode, "tasks": [], "load_failure": "not started",
            "thinking_switch": switch}
    return [p for p in validate.settings_problems(meta) if "thinking switch" in p]


class ThinkingSwitch(unittest.TestCase):
    def test_thinking_on(self):
        self.assertEqual(problems("thinking", "honoured"), [])
        self.assertEqual(problems("thinking", "ignored: always thinks"), [])
        self.assertTrue(problems("thinking", "ignored: never thinks"))
        self.assertTrue(problems("thinking", "inverted"))

    def test_thinking_off(self):
        self.assertEqual(problems("no-thinking", "honoured"), [])
        self.assertEqual(problems("no-thinking", "ignored: never thinks"), [])
        self.assertTrue(problems("no-thinking", "ignored: always thinks"))

    def test_not_probed_and_failed_probes(self):
        self.assertEqual(problems("thinking", None), [])
        self.assertTrue(problems("thinking", "probe failed: HTTP 507"))


if __name__ == "__main__":
    unittest.main()
