"""The statistics of the report: the Wilson interval and the two-sided Fisher exact test."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solvetime.report import fisher, wilson  # noqa: E402


class Stats(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson(14, 16)
        self.assertAlmostEqual(lo, 0.640, places=3)
        self.assertAlmostEqual(hi, 0.965, places=3)
        self.assertEqual(wilson(0, 16)[0], 0.0)
        self.assertEqual(wilson(16, 16)[1], 1.0)
        self.assertEqual(wilson(0, 0), (0.0, 0.0))

    def test_fisher(self):
        from math import comb
        # Equal rows (16, 16) make the null distribution symmetric, so p is twice one tail:
        # [[14, 2], [7, 9]], 21 in the first column: 2 * (C(16,14)C(16,7) + C(16,15)C(16,6) + C(16,16)C(16,5)) / C(32,21)
        tail = comb(16, 14) * comb(16, 7) + comb(16, 15) * comb(16, 6) + comb(16, 16) * comb(16, 5)
        self.assertAlmostEqual(fisher(14, 2, 7, 9), 2 * tail / comb(32, 21), places=12)
        self.assertAlmostEqual(fisher(14, 2, 7, 9), 0.023333, places=6)
        # [[0, 16], [9, 7]]: both extremes of the 9 in the first column, 2 * C(16,0)C(16,9) / C(32,9)
        self.assertAlmostEqual(fisher(0, 16, 9, 7), 2 * comb(16, 9) / comb(32, 9), places=12)
        self.assertEqual(fisher(5, 5, 5, 5), 1.0)
        self.assertAlmostEqual(fisher(3, 1, 1, 3), 17 / 35, places=12)   # Fisher's tea-tasting table


if __name__ == "__main__":
    unittest.main()
