"""Known-answer checks for the redundancy-aware placement integer programs."""
import unittest
import numpy as np
from optimize_v2 import greedy_cover, solve_cover

# Four cells, four sites. Site 0 covers cells 0-1, site 1 cells 1-2, site 2 cells 2-3, site 3 all.
A = np.array([[1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 1, 1], [0, 0, 1, 1]], dtype=bool)


class OptimizeV2Test(unittest.TestCase):
    def test_single_coverage_minimum(self):
        res, chosen, covered = solve_cover(A, 'min_sites', r=1, target=4)
        self.assertEqual(chosen.sum(), 1)
        self.assertTrue(chosen[3] and covered.all())

    def test_redundancy_needs_more_sites(self):
        res, chosen, covered = solve_cover(A, 'min_sites', r=2, target=4)
        self.assertEqual(res.status, 0)
        self.assertEqual(chosen.sum(), 3)  # site 3 plus sites 0 and 2
        self.assertTrue((A[:, chosen].sum(axis=1) >= 2).all())

    def test_allowed_sites_and_budget(self):
        res, chosen, covered = solve_cover(A, 'max_coverage', budget=1, allowed=[True, True, True, False])
        self.assertFalse(chosen[3])
        self.assertEqual(covered.sum(), 2)

    def test_preserve_specific_cells(self):
        res, chosen, covered = solve_cover(A, 'max_coverage', budget=1, allowed=[True, True, True, False],
                                           preserve=[False, False, False, True])
        self.assertTrue(chosen[2] and covered[3])

    def test_infeasible_preservation_detected(self):
        res, chosen, _ = solve_cover(A, 'min_sites', r=2, preserve=[True, False, False, False],
                                     allowed=[True, True, True, False])
        self.assertIsNone(chosen)

    def test_greedy_with_redundancy(self):
        chosen, covered = greedy_cover(A, 3, r=2)
        self.assertTrue(covered.all())


if __name__ == '__main__':
    unittest.main()
