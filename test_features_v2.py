"""Synthetic checks of the v2 Fresnel, diffraction, building, and foliage features."""
import site
import unittest
from pathlib import Path

site.addsitedir(str(Path(__file__).resolve().parent / '.geo_deps'))
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import from_origin
from features_v2 import Grids, knife_edge_db, path_features

X0, YTOP, W, H = 360000, 5630000, 1200, 50


def grids(dom_strips=(), tree_strips=(), building_strips=()):
    g = Grids.__new__(Grids)
    g.transform, g.shape = from_origin(X0, YTOP, 2, 2), (H, W)
    g.dgm = np.full((H, W), 100, dtype='float32')
    g.dom = g.dgm.copy()
    g.landcover = np.full((H, W), 30, dtype='uint8')
    g.buildings = np.zeros((H, W), dtype='int32')
    for i, (c0, c1, height) in enumerate(building_strips, 1):
        g.dom[:, c0:c1] = 100 + height
        g.buildings[:, c0:c1] = i
        g.landcover[:, c0:c1] = 50
    for c0, c1, height in tree_strips:
        g.dom[:, c0:c1] = 100 + height
        g.landcover[:, c0:c1] = 10
    return g


def link(length_m=2000, height=30):
    to_ll = Transformer.from_crs(25832, 4326, always_xy=True)
    lon0, lat0 = to_ll.transform(X0 + 10, YTOP - 50)
    lon1, lat1 = to_ll.transform(X0 + 10 + length_m, YTOP - 50)
    return pd.DataFrame({'longitude_sensor': [lon0], 'latitude_sensor': [lat0],
                         'longitude_gateway': [lon1], 'latitude_gateway': [lat1], 'antenna_height': [height]})


class FeatureV2Test(unittest.TestCase):
    def test_knife_edge_reference_values(self):
        self.assertAlmostEqual(float(knife_edge_db(0)), 6.03, places=2)
        self.assertAlmostEqual(float(knife_edge_db(1)), 13.9, places=1)
        self.assertEqual(float(knife_edge_db(-1)), 0.0)

    def test_clear_flat_path(self):
        row = path_features(link(1000), grids()).iloc[0]
        self.assertAlmostEqual(row.distance_m, 1000, delta=1)
        self.assertEqual(row.terrain_diffraction_db, 0)
        self.assertEqual(row.surface_los_blocked_frac, 0)
        self.assertEqual(row.buildings_crossed, 0)
        self.assertEqual(row.foliage_in_path_m, 0)

    def test_building_wall_blocks_and_diffracts(self):
        row = path_features(link(), grids(building_strips=[(500, 510, 40)])).iloc[0]
        self.assertEqual(row.buildings_crossed, 1)
        self.assertEqual(row.buildings_blocking, 1)
        self.assertGreater(row.surface_diffraction_db, 6)
        self.assertAlmostEqual(row.building_path_m, 20, delta=8)
        self.assertEqual(row.terrain_diffraction_db, 0)
        self.assertEqual(row.foliage_in_path_m, 0)

    def test_tree_strip_is_foliage_not_building(self):
        row = path_features(link(), grids(tree_strips=[(300, 320, 25)])).iloc[0]
        self.assertEqual(row.buildings_crossed, 0)
        self.assertAlmostEqual(row.foliage_in_path_m, 40, delta=8)
        self.assertGreater(row.surface_max_excess_m, 0)

    def test_two_separate_buildings_counted(self):
        row = path_features(link(), grids(building_strips=[(300, 305, 5), (700, 705, 5)])).iloc[0]
        self.assertEqual(row.buildings_crossed, 2)
        self.assertEqual(row.buildings_blocking, 0)


if __name__ == '__main__':
    unittest.main()
