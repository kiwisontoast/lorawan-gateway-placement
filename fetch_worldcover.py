"""Read ESA WorldCover 2020 (v100, 10 m) for the study area onto the 2 m grid.

The 2020 map matches the 2020-21 radio survey period. Only the study window is
read from the public cloud-optimized GeoTIFF. Nearest-neighbour resampling keeps
class codes. Licence: CC BY 4.0, (c) ESA WorldCover project 2020.
"""
import json
import site
from pathlib import Path

ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / '.geo_deps'))
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds

URL = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_N48E006_Map.tif'
OUT = ROOT / 'data/terrain'
X0, Y_TOP, RES, WIDTH, HEIGHT = 360000, 5632000, 2, 7000, 10000
CLASSES = {10: 'tree cover', 20: 'shrubland', 30: 'grassland', 40: 'cropland', 50: 'built-up',
           60: 'bare / sparse vegetation', 70: 'snow and ice', 80: 'permanent water', 90: 'herbaceous wetland',
           95: 'mangroves', 100: 'moss and lichen'}


def main():
    dst_transform = from_origin(X0, Y_TOP, RES, RES)
    bounds = (X0, Y_TOP - HEIGHT * RES, X0 + WIDTH * RES, Y_TOP)
    out = np.zeros((HEIGHT, WIDTH), dtype='uint8')
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
        with rasterio.open('/vsicurl/' + URL) as src:
            west, south, east, north = transform_bounds('EPSG:25832', src.crs, *bounds, densify_pts=21)
            window = from_bounds(west - .01, south - .01, east + .01, north + .01, src.transform).round_offsets().round_lengths()
            data = src.read(1, window=window)
            reproject(data, out, src_transform=src.window_transform(window), src_crs=src.crs,
                      dst_transform=dst_transform, dst_crs='EPSG:25832', resampling=Resampling.nearest)
    if (out == 0).any():
        raise ValueError(f'{int((out == 0).sum())} grid cells received no land-cover value')
    profile = dict(driver='GTiff', width=WIDTH, height=HEIGHT, count=1, dtype='uint8', crs='EPSG:25832',
                   transform=dst_transform, nodata=0, compress='deflate', tiled=True)
    with rasterio.open(OUT / 'bonn_worldcover2020_2m.tif', 'w', **profile) as ds:
        ds.write(out, 1)
    values, counts = np.unique(out, return_counts=True)
    share = {CLASSES.get(int(v), str(v)): round(float(c) / out.size, 4) for v, c in zip(values, counts)}
    (OUT / 'worldcover_metadata.json').write_text(json.dumps({
        'source': URL, 'product': 'ESA WorldCover 10 m 2020 v100', 'licence': 'CC BY 4.0',
        'attribution': '(c) ESA WorldCover project 2020 / Contains modified Copernicus Sentinel data (2020)',
        'resampling': 'nearest neighbour onto the 2 m EPSG:25832 study grid', 'class_share': share}, indent=2),
        encoding='utf-8')
    print('Land-cover shares:', share)


if __name__ == '__main__':
    main()
