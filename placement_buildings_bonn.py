"""Explore Bonn gateway candidates associated with OSM building records.

Building points are Overpass bounding-box centres, not verified roof points.
Copyright OpenStreetMap contributors; ODbL 1.0, https://www.openstreetmap.org/copyright.
"""
import json
import site
from pathlib import Path
ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / '.geo_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from placement_bonn import make_points, score_pairs, greedy
from reception_bonn import build_pairs, model, MODELS

OUT = ROOT / 'results/bonn_building_placement'
EXCLUDED = {'garage','garages','shed','hut','greenhouse','carport','roof','construction',
            'house','detached','semidetached_house','apartments','residential','terrace',
            'bungalow','cabin','dormitory','static_caravan'}


def building_candidates(demand, existing):
    raw = json.loads((ROOT/'data/osm/bonn_buildings.json').read_text(encoding='utf-8'))
    rows = []
    for item in raw['elements']:
        tags = item.get('tags', {})
        center = item.get('center')
        if not center or tags.get('building') in EXCLUDED:
            continue
        rows.append({'site_id':f"osm_{item['type']}_{item['id']}", 'kind':'building',
                     'longitude':center['lon'], 'latitude':center['lat'],
                     'building_type':tags.get('building','unknown'),
                     'osm_height':tags.get('height'), 'osm_levels':tags.get('building:levels')})
    sites = pd.DataFrame(rows)
    project = Transformer.from_crs(4326,25832,always_xy=True)
    sites['x'],sites['y'] = project.transform(sites.longitude.to_numpy(),sites.latitude.to_numpy())
    sites['gx'] = np.floor(sites.x/2000).astype(int)
    sites['gy'] = np.floor(sites.y/2000).astype(int)
    demand_tiles = demand.assign(gx=np.floor(demand.x/2000).astype(int), gy=np.floor(demand.y/2000).astype(int))
    targets = demand_tiles.groupby(['gx','gy'],as_index=False).agg(target_x=('x','mean'), target_y=('y','mean'))
    sites = sites.merge(targets,on=['gx','gy'],how='inner',validate='many_to_one')
    with rasterio.open(ROOT/'data/terrain/bonn_dgm_20m.tif') as dgm, rasterio.open(ROOT/'data/terrain/bonn_dom_20m.tif') as dom:
        if dgm.transform != dom.transform or dgm.crs != dom.crs or dgm.shape != dom.shape:
            raise ValueError('Terrain/surface rasters must be aligned')
        col = np.floor((sites.x.to_numpy()-dgm.transform.c)/dgm.transform.a).astype(int)
        row = np.floor((sites.y.to_numpy()-dgm.transform.f)/dgm.transform.e).astype(int)
        valid = (col>=0)&(col<dgm.width)&(row>=0)&(row<dgm.height)
        ground = dgm.read(1,masked=True).filled(np.nan)
        surface = dom.read(1,masked=True).filled(np.nan)
        z_ground = np.full(len(sites),np.nan); z_surface = np.full(len(sites),np.nan)
        z_ground[valid]=ground[row[valid],col[valid]]
        z_surface[valid]=surface[row[valid],col[valid]]
    sites['surface_height_proxy_m'] = z_surface-z_ground
    # The 20 m surface pixel may represent a nearby object or vegetation.
    # Keep nominal antenna heights within the training sites' range.
    sites = sites[sites.surface_height_proxy_m.between(10,28)].copy()
    sites['antenna_height'] = sites.surface_height_proxy_m+2
    sites['distance_to_tile_center'] = np.hypot(sites.x-sites.target_x, sites.y-sites.target_y)
    sites = sites.sort_values(['gx','gy','distance_to_tile_center','site_id']).groupby(['gx','gy']).head(2)
    tiles_with_candidates = len(sites[['gx','gy']].drop_duplicates())
    columns = ['site_id','kind','longitude','latitude','x','y','antenna_height',
               'surface_height_proxy_m','building_type','osm_height','osm_levels']
    sites = sites[columns].sort_values('site_id').reset_index(drop=True)
    all_sites = pd.concat([existing[existing.kind=='existing'],sites],ignore_index=True)
    return all_sites, len(targets), tiles_with_candidates


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    demand, original_sites = make_points()
    sites, target_tiles, covered_tiles = building_candidates(demand,original_sites)
    if len(sites)<=9:
        raise ValueError('No usable building candidates were found')
    sites.to_csv(OUT/'candidate_sites.csv',index=False)
    demand.to_csv(OUT/'demand_cells.csv',index=False)
    print(f'{len(demand)} demand cells; {len(sites)-9} OSM building candidates in {covered_tiles}/{target_tiles} occupied 2-km tiles; 9 existing gateways',flush=True)
    pairs = build_pairs()
    fits = {}
    for scenario in ['in_range','same_day']:
        training = pairs.loc[pairs[scenario] & ((pairs.n_heard-pairs.heard)>0)]
        fits[scenario] = model().fit(training[MODELS['terrain_surface']],training.heard)
        print(f'Trained {scenario} on {len(training)} links',flush=True)
    del pairs
    site_ids=sites.site_id.to_numpy()
    is_existing=sites.kind.to_numpy()=='existing'
    records=[]; selections=[]; example={}
    for offset in [-2,0,2]:
        adjusted=sites.copy()
        adjusted.loc[adjusted.kind=='building','antenna_height'] += offset
        paths=score_pairs(demand,adjusted,16)
        for activity,fit in fits.items():
            scores=fit.predict_proba(paths[MODELS['terrain_surface']])[:,1].reshape(len(demand),len(sites))
            for threshold in [0.3,0.5,0.7]:
                matrix=scores>=threshold
                baseline=matrix[:,is_existing].any(axis=1)
                chosen, curve, coverage=greedy(matrix,site_ids,9)
                for k in range(1,10):
                    actual=min(k,len(chosen))
                    records.append({'activity_scenario':activity,'height_offset_m':offset,'threshold':threshold,
                                    'budget_k':k,'selected_k':actual,'greedy_fraction':curve[actual-1] if actual else 0,
                                    'existing_fraction':baseline.mean(),'clipped_link_count':int(paths.distance_was_clipped.sum())})
                    for rank,site in enumerate(chosen[:actual],1):
                        selections.append({'activity_scenario':activity,'height_offset_m':offset,'threshold':threshold,
                                           'budget_k':k,'rank':rank,'site_id':site})
                if (activity,offset,threshold)==('same_day',0,0.5):
                    example={'baseline':baseline,'coverage':coverage,'chosen':chosen}
                    np.savez_compressed(OUT/'illustrative_coverage_matrix.npz', matrix=matrix,
                                        site_ids=site_ids.astype(str), cell_ids=demand.cell_id.to_numpy(dtype=str),
                                        baseline=baseline)
        print(f'Completed building antenna offset {offset} m',flush=True)
    result=pd.DataFrame(records); result.to_csv(OUT/'scenario_results.csv',index=False,float_format='%.6f')
    pd.DataFrame(selections).to_csv(OUT/'selected_sites.csv',index=False)
    base=result[(result.activity_scenario=='same_day')&(result.height_offset_m==0)&(result.threshold==0.5)]
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    ax.plot(base.budget_k,base.greedy_fraction*100,marker='o',label='Greedy selection')
    ax.axhline(base.existing_fraction.iloc[0]*100,color='red',linestyle='--',label='Nine existing sites')
    ax.set(xlabel='Gateway budget',ylabel='Demand cells meeting score threshold (%)',ylim=(0,105),
           title='OSM-building candidate simulation: threshold 0.5')
    ax.legend(); fig.savefig(OUT/'coverage_curve.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,name in zip(axes,['baseline','coverage']):
        ax.scatter(demand.x,demand.y,c=np.where(example[name],'#137b5b','#dc8051'),s=9)
        chosen=is_existing if name=='baseline' else sites.site_id.isin(example['chosen']).to_numpy()
        ax.scatter(sites.loc[chosen,'x'],sites.loc[chosen,'y'],marker='^',s=65,c='#294ea1',edgecolors='white')
        ax.set(title='Nine existing sites' if name=='baseline' else 'Greedy selected sites, budget 9',xlabel='Easting (m)',ylabel='Northing (m)')
        ax.tick_params(axis='x',labelrotation=40); ax.set_aspect('equal')
    fig.suptitle('Illustrative modeled scores; © OpenStreetMap contributors')
    fig.savefig(OUT/'placement_map.png',dpi=160); plt.close(fig)
    print(base[['budget_k','selected_k','greedy_fraction','existing_fraction']].to_string(index=False))


if __name__=='__main__':
    main()
