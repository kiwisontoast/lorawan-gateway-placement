"""Verify location-specific coverage preservation."""
import unittest
import numpy as np
from preserve_baseline_bonn import solve_preserving


class PreserveTest(unittest.TestCase):
    def test_preserves_specific_cells_not_only_the_count(self):
        matrix=np.array([[1,0],[0,1],[0,1]],dtype=bool)
        baseline=np.array([1,0,0],dtype=bool)
        result=solve_preserving(matrix,baseline,budget=1,time_limit=10)
        self.assertTrue(result.success)
        self.assertGreater(result.x[0],.5)
        self.assertLess(result.x[1],.5)
        self.assertAlmostEqual(result.fun,-1)


if __name__=='__main__':
    unittest.main()
