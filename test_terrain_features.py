"""Check terrain path features against synthetic flat and blocked rasters."""
import site
import tempfile
import unittest
from pathlib import Path

site.addsitedir(str(Path(__file__).resolve().parent / '.geo_deps'))
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
from terrain_bonn import path_features


class TerrainFeatureTest(unittest.TestCase):
    def test_flat_ground_and_surface_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            ground_path = Path(directory) / 'ground.tif'
            surface_path = Path(directory) / 'surface.tif'
            transform = from_origin(360000, 5630000, 20, 20)
            profile = dict(driver='GTiff', width=20, height=20, count=1,
                           dtype='float32', crs='EPSG:25832', transform=transform, nodata=-9999)
            ground = np.full((20, 20), 100, dtype='float32')
            surface = ground.copy()
            surface[:, 9:11] = 125
            for path, data in [(ground_path, ground), (surface_path, surface)]:
                with rasterio.open(path, 'w', **profile) as ds:
                    ds.write(data, 1)
            converter = Transformer.from_crs(25832, 4326, always_xy=True)
            lon0, lat0 = converter.transform(360030, 5629810)
            lon1, lat1 = converter.transform(360370, 5629810)
            frame = pd.DataFrame({'longitude_sensor': [lon0], 'latitude_sensor': [lat0],
                                  'longitude_gateway': [lon1], 'latitude_gateway': [lat1],
                                  'antenna_height': [10]})
            row = path_features(frame, ground_path, surface_path).iloc[0]
            self.assertAlmostEqual(row.ground_delta_m, 0)
            self.assertLess(row.max_ground_above_link_m, 0)
            self.assertEqual(row.ground_obstructed_fraction, 0)
            self.assertGreater(row.max_surface_above_link_m, 0)
            self.assertGreater(row.surface_obstructed_fraction, 0)


if __name__ == '__main__':
    unittest.main()
