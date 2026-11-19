"""mlx-lm's reply limit: its server's default (512 tokens) unless SOLVETIME_MLX_MAX_TOKENS sets one, which the
tables then show."""

import os
import unittest
from unittest import mock

from solvetime import engines


class ReplyLimit(unittest.TestCase):
    def engine(self, **env):
        with mock.patch.dict(os.environ, env):
            if "SOLVETIME_MLX_MAX_TOKENS" not in env:
                os.environ.pop("SOLVETIME_MLX_MAX_TOKENS", None)
            return engines.mlx_lm("qwen3.8-27b", 8890)

    def test_the_default(self):
        e = self.engine()
        self.assertNotIn("--max-tokens", e.command)
        self.assertEqual(e.settings, "")

    def test_a_limit_is_passed_and_shown(self):
        e = self.engine(SOLVETIME_MLX_MAX_TOKENS="16384")
        self.assertEqual(e.command[-2:], ["--max-tokens", "16384"])
        self.assertEqual(e.settings, "max tokens 16384")

    def test_not_a_number(self):
        with self.assertRaises(ValueError):
            self.engine(SOLVETIME_MLX_MAX_TOKENS="lots")


if __name__ == "__main__":
    unittest.main()
