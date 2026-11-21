"""The memory reading before an engine starts waits for the calibration's GPU memory to be released: it stays wired
for about a second after the calibration's process ends (2026-11-21), and a reading taken at once read 2-3.5 GB
low. The wait ends when wired memory is back near its level before the calibration, or after the timeout."""

import unittest
from unittest import mock

from solvetime import run


class Release(unittest.TestCase):
    def readings(self, wired):
        values = iter(wired)
        return lambda: {"wired": next(values, wired[-1])}

    def test_waits_until_released(self):
        # before: 2.6 GB; at once: 4.7; still 4.7; then 2.7, within 0.3 GB of before
        with mock.patch.object(run, "memory_at_start", self.readings([2.6e9, 4.7e9, 4.7e9, 2.7e9])), \
                mock.patch.object(run.time, "sleep", lambda s: None):
            out = run.calibrated(lambda: {"calibration_fp16_matmul_tflops": 18.9})
        self.assertEqual(out["calibration_fp16_matmul_tflops"], 18.9)
        self.assertIn("calibration_release_seconds", out)

    def test_released_at_once(self):
        calls = []
        with mock.patch.object(run, "memory_at_start", lambda: calls.append(1) or {"wired": 2.6e9}):
            out = run.calibrated(lambda: {})
        self.assertEqual(len(calls), 2)             # before, and once after
        self.assertLess(out["calibration_release_seconds"], 0.1)

    def test_gives_up_after_the_timeout(self):
        with mock.patch.object(run, "memory_at_start", self.readings([2.6e9, 6.0e9])):
            out = run.calibrated(lambda: {}, timeout=0.3)
        self.assertGreaterEqual(out["calibration_release_seconds"], 0.3)


if __name__ == "__main__":
    unittest.main()
