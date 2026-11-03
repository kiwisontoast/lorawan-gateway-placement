"""Download NRW 1 m DGM1/DOM1 tiles for the Bonn study area; build 2 m mosaics.

The 1 m tiles are kept in data/nrw1m/ (about 1.2 GB, git-ignored). The 2 m
mosaics aggregate each 2 x 2 block: bare-earth DGM by mean, surface DOM by
maximum, so narrow tall obstacles are preserved rather than averaged away.
Licence: Data licence Germany zero 2.0 (Geobasis NRW).
"""
import json
import re
import site
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / '.geo_deps'))
import numpy as np
import rasterio
import requests
from rasterio.transform import from_origin

BASE = 'https://www.opengeodata.nrw.de/produkte/geobasis/hm'
PRODUCTS = {'dgm': f'{BASE}/dgm1_tiff/dgm1_tiff', 'dom': f'{BASE}/dom1_tiff/dom1_tiff'}
TILES_DIR = ROOT / 'data/nrw1m'
OUT = ROOT / 'data/terrain'
# Study area: every 1 km tile touched by the v1 terrain request bounds.
X_KM = range(360, 374)
Y_KM = range(5612, 5632)
X0, Y_TOP = 360000, 5632000
RES = 2
WIDTH, HEIGHT = 14000 // RES, 20000 // RES


def tile_index(product):
    data = requests.get(f'{PRODUCTS[product]}/index.json', timeout=120).json()
    found = {}
    for ds in data['datasets']:
        for f in ds['files']:
            m = re.search(r'_32_(\d{3})_(\d{4})_1_nw_(\d{4})\.tif$', f['name'])
            if m and int(m[1]) in X_KM and int(m[2]) in Y_KM:
                found[(int(m[1]), int(m[2]))] = {'name': f['name'], 'size': int(f['size']),
                                                 'year': int(m[3]), 'timestamp': f.get('timestamp')}
    missing = [(x, y) for x in X_KM for y in Y_KM if (x, y) not in found]
    if missing:
        raise ValueError(f'{product}: {len(missing)} study tiles missing from index, e.g. {missing[:3]}')
    return found


def download(product, info):
    path = TILES_DIR / product / info['name']
    if path.exists() and path.stat().st_size == info['size']:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"{PRODUCTS[product]}/{info['name']}", timeout=300)
    r.raise_for_status()
    if len(r.content) != info['size']:
        raise ValueError(f"Size mismatch for {info['name']}")
    tmp = path.with_suffix('.part')
    tmp.write_bytes(r.content)
    tmp.replace(path)
    return path


def mosaic(product, tiles):
    out = np.full((HEIGHT, WIDTH), np.nan, dtype='float32')
    for (x, y), info in sorted(tiles.items()):
        with rasterio.open(TILES_DIR / product / info['name']) as ds:
            if ds.crs.to_epsg() != 25832 or abs(ds.res[0] - 1) > 1e-9 or abs(ds.res[1] - 1) > 1e-9:
                raise ValueError(f"Unexpected tile grid: {info['name']} {ds.crs} {ds.res}")
            col0, row0 = (ds.transform.c - X0), (Y_TOP - ds.transform.f)
            if abs(col0 - round(col0)) > 1e-6 or abs(row0 - round(row0)) > 1e-6 or int(round(col0)) % RES or int(round(row0)) % RES:
                raise ValueError(f"Tile not aligned to the 2 m grid: {info['name']} {ds.transform}")
            band = ds.read(1, masked=True).filled(np.nan).astype('float32')
        h, w = band.shape
        if h % RES or w % RES:
            raise ValueError(f"Tile size not divisible by {RES}: {info['name']}")
        blocks = band.reshape(h // RES, RES, w // RES, RES)
        # A block with any missing 1 m value stays missing rather than being filled.
        agg = blocks.max(axis=(1, 3)) if product == 'dom' else blocks.mean(axis=(1, 3))
        r0, c0 = int(round(row0)) // RES, int(round(col0)) // RES
        r1, c1 = min(r0 + agg.shape[0], HEIGHT), min(c0 + agg.shape[1], WIDTH)
        if r0 < 0 or c0 < 0 or r0 >= HEIGHT or c0 >= WIDTH:
            continue
        out[r0:r1, c0:c1] = agg[:r1 - r0, :c1 - c0]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {'source': PRODUCTS, 'licence': 'Data licence Germany zero 2.0', 'grid_origin_epsg25832': [X0, Y_TOP],
            'resolution_m': RES, 'width': WIDTH, 'height': HEIGHT,
            'aggregation': {'dgm': 'mean of 2x2 1 m cells', 'dom': 'maximum of 2x2 1 m cells'}, 'tiles': {}}
    for product in PRODUCTS:
        tiles = tile_index(product)
        print(f'{product}: {len(tiles)} tiles, {sum(t["size"] for t in tiles.values())/1e6:.0f} MB', flush=True)
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda kv: download(product, kv[1]), tiles.items()))
        print(f'{product}: downloaded; building {RES} m mosaic', flush=True)
        grid = mosaic(product, tiles)
        if np.isnan(grid).any():
            print(f'{product}: {int(np.isnan(grid).sum())} missing 2 m cells', flush=True)
        profile = dict(driver='GTiff', width=WIDTH, height=HEIGHT, count=1, dtype='float32', crs='EPSG:25832',
                       transform=from_origin(X0, Y_TOP, RES, RES), nodata=np.nan, compress='deflate',
                       predictor=3, tiled=True)
        with rasterio.open(OUT / f'bonn_{product}_2m.tif', 'w', **profile) as ds:
            ds.write(grid, 1)
        years = {}
        for t in tiles.values():
            years[t['year']] = years.get(t['year'], 0) + 1
        meta['tiles'][product] = {'count': len(tiles), 'acquisition_years': years,
                                  'files': {f'{x}_{y}': t for (x, y), t in sorted(tiles.items())}}
        del grid
    (OUT / 'nrw_2m_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('Wrote', OUT / 'bonn_dgm_2m.tif', OUT / 'bonn_dom_2m.tif')


if __name__ == '__main__':
    main()
