"""Stream NRW LoD2 CityGML tiles for the study area into a compact building table.

About 5 GB of CityGML is downloaded one tile at a time, parsed, and discarded;
only data/lod2/bonn_lod2_parts.parquet and a tile manifest are kept. One row is
one Building or BuildingPart with its own ground surface. Heights are DHHN2016
(the same vertical datum as DGM1/DOM1). The model is current (2025-26), not a
2020-21 building inventory. Licence: Data licence Germany zero 2.0.
"""
import json
import site
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / '.geo_deps'))
import pandas as pd
import requests
from lxml import etree
from shapely.geometry import Polygon
from shapely.ops import unary_union

URL = 'https://www.opengeodata.nrw.de/produkte/geobasis/3dg/lod2_gml/lod2_gml'
OUT = ROOT / 'data/lod2'
X_KM = range(360, 374)
Y_KM = range(5612, 5632)
NS = {'core': 'http://www.opengis.net/citygml/1.0', 'bldg': 'http://www.opengis.net/citygml/building/1.0',
      'gml': 'http://www.opengis.net/gml'}


def coords(poslist):
    values = [float(v) for v in poslist.text.split()]
    return [(values[i], values[i + 1], values[i + 2]) for i in range(0, len(values) - 2, 3)]


def parse_part(part, building_id):
    ground, roof_z, ground_z = [], [], []
    for pl in part.iterfind('bldg:boundedBy/bldg:GroundSurface//gml:posList', NS):
        ring = coords(pl)
        ground_z += [p[2] for p in ring]
        if len(ring) >= 4:
            poly = Polygon([(p[0], p[1]) for p in ring])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty:
                ground.append(poly)
    for pl in part.iterfind('bldg:boundedBy/bldg:RoofSurface//gml:posList', NS):
        roof_z += [p[2] for p in coords(pl)]
    if not ground or not roof_z:
        return None
    footprint = unary_union(ground)
    height = part.findtext('bldg:measuredHeight', namespaces=NS)
    return {'building_id': building_id, 'part_id': part.get('{http://www.opengis.net/gml}id'),
            'function': part.findtext('bldg:function', namespaces=NS),
            'measured_height_m': float(height) if height else None,
            'ground_z_m': min(ground_z), 'roof_max_z_m': max(roof_z),
            'area_m2': footprint.area, 'centroid_x': footprint.centroid.x, 'centroid_y': footprint.centroid.y,
            'footprint_wkb': footprint.wkb}


def parse_tile(name, size):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        with requests.get(f'{URL}/{name}', timeout=600, stream=True) as r:
            r.raise_for_status()
            with path.open('wb') as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        if path.stat().st_size != size:
            raise ValueError(f'Size mismatch for {name}')
        rows = []
        member = f"{{{NS['core']}}}cityObjectMember"
        for _, elem in etree.iterparse(str(path), events=('end',), tag=member, huge_tree=True):
            for building in elem.iterfind('bldg:Building', NS):
                bid = building.get('{http://www.opengis.net/gml}id')
                for part in [building, *building.iterfind('bldg:consistsOfBuildingPart/bldg:BuildingPart', NS)]:
                    row = parse_part(part, bid)
                    if row:
                        row['tile'] = name
                        rows.append(row)
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
    return name, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = requests.get(f'{URL}/index.json', timeout=120).json()
    wanted = {f'LoD2_32_{x}_{y}_1_NW.gml' for x in X_KM for y in Y_KM}
    files = {f['name']: f for ds in index['datasets'] for f in ds['files'] if f['name'] in wanted}
    if len(files) != len(wanted):
        raise ValueError(f'{len(wanted) - len(files)} study tiles missing from the LoD2 index')
    print(f'{len(files)} LoD2 tiles, {sum(int(f["size"]) for f in files.values())/1e9:.2f} GB', flush=True)
    rows, done = [], 0
    with ProcessPoolExecutor(6) as pool:
        futures = [pool.submit(parse_tile, n, int(f['size'])) for n, f in files.items()]
        for fut in as_completed(futures):
            name, tile_rows = fut.result()
            rows += tile_rows
            done += 1
            if done % 20 == 0:
                print(f'parsed {done}/{len(files)} tiles, {len(rows)} parts', flush=True)
    table = pd.DataFrame(rows)
    # A building crossing a tile edge appears in both tiles; keep one copy.
    before = len(table)
    table = table.drop_duplicates('part_id').reset_index(drop=True)
    table.to_parquet(OUT / 'bonn_lod2_parts.parquet', index=False)
    manifest = {'source': URL, 'licence': 'Data licence Germany zero 2.0', 'tiles': files,
                'parts': len(table), 'duplicate_parts_removed': before - len(table),
                'buildings': int(table.building_id.nunique()),
                'vertical_datum': 'DHHN2016 (as DGM1/DOM1)',
                'note': 'Current LoD2 model; post-dates the 2020-21 radio survey.'}
    (OUT / 'lod2_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'{len(table)} building parts from {table.building_id.nunique()} buildings', flush=True)


if __name__ == '__main__':
    main()
