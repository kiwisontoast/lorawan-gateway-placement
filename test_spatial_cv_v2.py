import unittest

import numpy as np

import spatial_cv_v2 as cv


class SpatialFolds(unittest.TestCase):
    def test_buffer_covers_neighbours_only(self):
        keys = set(cv.buffered((np.array([10]), np.array([20]))))
        self.assertEqual(len(keys), (2 * cv.BUFFER_CELLS + 1) ** 2)
        self.assertIn(cv.key(11, 21), keys)
        self.assertNotIn(cv.key(10 + cv.BUFFER_CELLS + 1, 20), keys)

    def test_cells_in_one_block_share_a_fold(self):
        b = cv.BLOCK_CELLS
        block_fold = {int(cv.key(2, 5)): 3}
        gx = np.array([2 * b, 2 * b + b - 1, 2 * b + b])
        gy = np.array([5 * b, 5 * b + b - 1, 5 * b])
        folds = cv.fold_of(gx, gy, block_fold)
        self.assertEqual(list(folds[:2]), [3, 3])
        self.assertTrue(np.isnan(folds[2]))


if __name__ == '__main__':
    unittest.main()
