"""Fetch OSM geometries for mapped candidate ways; not evidence of roof access.

© OpenStreetMap contributors, ODbL 1.0: https://www.openstreetmap.org/copyright
"""
import json
from pathlib import Path
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/osm'


def main():
    sites=pd.read_csv(ROOT/'results/bonn_building_placement/candidate_sites.csv')
    ids=sites.loc[(sites.kind=='building') & sites.site_id.str.startswith('osm_way_'),'site_id']
    ids=sorted(int(s.rsplit('_',1)[1]) for s in ids)
    query=f'[out:json][timeout:120];way(id:{",".join(map(str,ids))});out geom;'
    r=requests.post('https://overpass-api.de/api/interpreter',data={'data':query},timeout=180,
                    headers={'User-Agent':'lorawan-gateway-placement/1.0 (Bonn gateway study)'})
    r.raise_for_status(); data=r.json()
    if 'remark' in data or 'elements' not in data:
        raise ValueError(f'Incomplete Overpass output: {data.get("remark")}')
    if set(ids)!={e['id'] for e in data['elements']}:
        raise ValueError('Missing building ways: do not silently treat results as complete')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'candidate_footprints.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    (OUT/'candidate_footprints_metadata.json').write_text(json.dumps({'source':'https://overpass-api.de/api/interpreter',
        'query':query,'osm_base_time':data.get('osm3s',{}).get('timestamp_osm_base'),
        'attribution':'© OpenStreetMap contributors','licence':'ODbL 1.0',
        'url':'https://www.openstreetmap.org/copyright'},indent=2),encoding='utf-8')
    print('Footprints downloaded:',len(data['elements']), 'OSM base:',data.get('osm3s',{}).get('timestamp_osm_base'))


if __name__=='__main__':
    main()
