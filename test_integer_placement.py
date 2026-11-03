"""Verify both integer placement formulations on a small coverage matrix."""
import unittest
import numpy as np
from optimize_building_sites import solve


class PlacementMILPTest(unittest.TestCase):
    def test_minimum_and_maximum(self):
        matrix = np.array([[1,0],[0,1]], dtype=bool)
        minimum = solve(matrix, 'min_sites', target=2, time_limit=10)
        maximum = solve(matrix, 'max_coverage', budget=1, time_limit=10)
        self.assertTrue(minimum.success)
        self.assertAlmostEqual(minimum.fun, 2)
        self.assertTrue(maximum.success)
        self.assertAlmostEqual(maximum.fun, -1)


if __name__ == '__main__':
    unittest.main()
