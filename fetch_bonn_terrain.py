"""Fetch a coarse terrain raster covering all Bonn receptions and gateways.

Requires requests, pandas, pyproj and rasterio. NRW WCS access requires network.
Raster is explicitly downsampled to ~20 m for an initial terrain-only experiment.
"""
import json
import site
from pathlib import Path

ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / '.geo_deps'))
import pandas as pd
import requests
import rasterio
from pyproj import Transformer

SURVEY = ROOT / 'data/lora-bonn/survey'
OUT = ROOT / 'data/terrain'
URL = 'https://www.wcs.nrw.de/geobasis/wcs_nw_dgm'


def main():
    obs = pd.read_csv(SURVEY / 'samples.csv', usecols=['latitude', 'longitude'])
    gw = pd.read_csv(SURVEY / 'gateways.csv', usecols=['latitude', 'longitude'])
    points = pd.concat([obs, gw], ignore_index=True)
    transformer = Transformer.from_crs(4326, 25832, always_xy=True)
    x, y = transformer.transform(points.longitude.to_numpy(), points.latitude.to_numpy())
    bounds = [int(min(x))-100, int(min(y))-100, int(max(x))+100, int(max(y))+100]
    xmin, ymin, xmax, ymax = bounds
    params = [('service', 'WCS'), ('version', '2.0.1'), ('request', 'GetCoverage'),
              ('coverageId', 'nw_dgm'), ('format', 'image/tiff'),
              ('subset', f'x({xmin},{xmax})'), ('subset', f'y({ymin},{ymax})'),
              ('scaleFactor', '0.05')]
    print('Requesting EPSG:25832 bounds:', bounds, flush=True)
    response = requests.get(URL, params=params, timeout=180)
    response.raise_for_status()
    if not response.content[:4] in (b'II*\x00', b'MM\x00*'):
        raise ValueError(f'Expected GeoTIFF, got {response.headers.get("Content-Type")}: {response.text[:500]}')
    OUT.mkdir(parents=True, exist_ok=True)
    raster = OUT / 'bonn_dgm_20m.tif'
    raster.write_bytes(response.content)
    with rasterio.open(raster) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 25832:
            raise ValueError(f'Unexpected terrain CRS: {ds.crs}')
        info = {'source': URL, 'coverage': 'nw_dgm', 'request_url': response.url,
                'requested_bounds_epsg25832': bounds, 'width': ds.width, 'height': ds.height,
                'resolution_m': list(ds.res), 'crs': str(ds.crs), 'nodata': ds.nodata,
                'bytes': len(response.content), 'source_model': 'DGM1 downsampled by WCS factor 0.05',
                'licence': 'Data licence Germany zero 2.0; see NRW product metadata',
                'note': 'Current service data; acquisition dates may differ from 2020-2021 survey.'}
        print(json.dumps(info, indent=2))
        (OUT / 'terrain_metadata.json').write_text(json.dumps(info, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
