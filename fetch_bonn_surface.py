"""Fetch a downsampled NRW surface raster matching the terrain request."""
import json
import site
from pathlib import Path

ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / '.geo_deps'))
import requests
import rasterio


def main():
    out = ROOT / 'data/terrain'
    metadata = json.loads((out / 'terrain_metadata.json').read_text(encoding='utf-8'))
    xmin, ymin, xmax, ymax = metadata['requested_bounds_epsg25832']
    url = 'https://www.wcs.nrw.de/geobasis/wcs_nw_dom'
    params = [('service', 'WCS'), ('version', '2.0.1'), ('request', 'GetCoverage'),
              ('coverageId', 'nw_dom'), ('format', 'image/tiff'),
              ('subset', f'x({xmin},{xmax})'), ('subset', f'y({ymin},{ymax})'),
              ('scaleFactor', '0.05')]
    r = requests.get(url, params=params, timeout=180)
    r.raise_for_status()
    if r.content[:4] not in (b'II*\x00', b'MM\x00*'):
        raise ValueError(f'Expected GeoTIFF; got {r.text[:500]}')
    path = out / 'bonn_dom_20m.tif'
    path.write_bytes(r.content)
    with rasterio.open(path) as surface, rasterio.open(out / 'bonn_dgm_20m.tif') as ground:
        if surface.crs != ground.crs or surface.transform != ground.transform or surface.shape != ground.shape:
            raise ValueError('Surface raster is not aligned to ground raster')
        print('Surface and terrain rasters aligned:', surface.shape, surface.res)
        info = {'source': url, 'request_url': r.url, 'coverage': 'nw_dom', 'bytes': len(r.content),
                'resolution_m': surface.res, 'crs': str(surface.crs), 'nodata': surface.nodata,
                'licence': 'Data licence Germany zero 2.0',
                'note': 'Current surface data; acquisition dates not matched to 2020-21 survey.'}
        (out / 'surface_metadata.json').write_text(json.dumps(info, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
