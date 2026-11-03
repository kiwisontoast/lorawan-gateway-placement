"""Download OSM building candidates within the Bonn demand bounding box.

Data: OpenStreetMap contributors, ODbL 1.0, https://www.openstreetmap.org/copyright.
The bounding-box centres returned by Overpass are not guaranteed rooftop points.
"""
import json
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/osm'
URL = 'https://overpass-api.de/api/interpreter'


def main():
    demand = pd.read_csv(ROOT / 'results/bonn_placement/demand_cells.csv')
    bounds = (demand.latitude.min()-0.002, demand.longitude.min()-0.002,
              demand.latitude.max()+0.002, demand.longitude.max()+0.002)
    s,w,n,e = bounds
    query = f'[out:json][timeout:180];(way["building"]({s},{w},{n},{e});relation["building"]["type"="multipolygon"]({s},{w},{n},{e}););out center tags;'
    print('Requesting OSM buildings in bbox:', bounds, flush=True)
    r = requests.post(URL, data={'data': query}, headers={'User-Agent':'lorawan-gateway-placement/1.0 (Bonn building research)'}, timeout=240)
    r.raise_for_status()
    data = r.json()
    if 'elements' not in data or 'remark' in data:
        raise ValueError(f'Incomplete Overpass response: {data.get("remark")}')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'bonn_buildings.json').write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    print('Buildings returned:', len(data['elements']), 'OSM data timestamp:', data.get('osm3s',{}).get('timestamp_osm_base'))
    (OUT/'metadata.json').write_text(json.dumps({'source':URL,'bbox_south_west_north_east':bounds,
        'query':query,'osm_base_time':data.get('osm3s',{}).get('timestamp_osm_base'),
        'licence':'ODbL 1.0', 'attribution':'© OpenStreetMap contributors',
        'copyright_url':'https://www.openstreetmap.org/copyright'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
