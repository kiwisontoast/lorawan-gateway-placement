"""Tests for the greedy scenario-coverage selection."""
import unittest
import numpy as np
from placement_bonn import greedy


class GreedyTest(unittest.TestCase):
    def test_selects_best_incremental_coverage(self):
        matrix = np.array([[1,1,0], [1,0,0], [0,1,1], [0,0,1]], dtype=bool)
        chosen, coverage, covered = greedy(matrix, np.array(['a','b','c']), 2)
        self.assertEqual(chosen, ['a','c'])
        self.assertEqual(coverage, [.5, 1.0])
        self.assertTrue(covered.all())

    def test_stops_when_no_site_adds_coverage(self):
        matrix = np.array([[1,1], [0,0]], dtype=bool)
        chosen, coverage, _ = greedy(matrix, np.array(['a','b']), 2)
        self.assertEqual(chosen, ['a'])
        self.assertEqual(coverage, [.5])


if __name__ == '__main__':
    unittest.main()
