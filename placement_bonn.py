"""Scenario-only Bonn gateway placement using inferred-reception scores.

Run after terrain data preparation: python placement_bonn.py
Predictions are conditional scores, not real-world reception probabilities.
"""
import site
from pathlib import Path

ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / '.geo_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from reception_bonn import build_pairs, model, MODELS
from terrain_bonn import path_features, TERRAIN_FEATURES, SURFACE_FEATURES

OUT = ROOT / 'results/bonn_placement'
CELL_METRES = 500
SITE_METRES = 2000
HEIGHTS = [12, 16, 24]
THRESHOLDS = [0.3, 0.5, 0.7]
K_MAX = 9


def make_points(start_date=None, end_date=None):
    data = pd.read_csv(ROOT / 'data/lora-bonn/survey/samples.csv', usecols=['latitude', 'longitude', 'transceived_at'])
    if start_date is not None:
        data = data.loc[data.transceived_at >= start_date]
    if end_date is not None:
        data = data.loc[data.transceived_at <= end_date]
    data = data[['latitude', 'longitude']]
    if data.empty:
        raise ValueError('No demand locations for requested period')
    # Unique locations prevent repeated receptions from implicitly weighting roads.
    data = data.drop_duplicates()
    to_utm = Transformer.from_crs(4326, 25832, always_xy=True)
    to_ll = Transformer.from_crs(25832, 4326, always_xy=True)
    data['x'], data['y'] = to_utm.transform(data.longitude.to_numpy(), data.latitude.to_numpy())
    data['gx'] = np.floor(data.x / CELL_METRES).astype(int)
    data['gy'] = np.floor(data.y / CELL_METRES).astype(int)
    # Representative is an actual observed coordinate nearest each cell centroid.
    means = data.groupby(['gx', 'gy'])[['x', 'y']].transform('mean')
    data['dist'] = (data.x - means.x)**2 + (data.y - means.y)**2
    demand = data.sort_values('dist').drop_duplicates(['gx', 'gy']).sort_values(['gx', 'gy']).reset_index(drop=True)
    demand['cell_id'] = ['cell_%04d' % i for i in range(len(demand))]
    demand['weight'] = 1.0
    demand = demand[['cell_id', 'longitude', 'latitude', 'x', 'y', 'weight']]
    # One candidate at an observed coordinate per 2 km tile; not verified rooftops.
    c = demand.copy()
    c['gx'] = np.floor(c.x / SITE_METRES).astype(int)
    c['gy'] = np.floor(c.y / SITE_METRES).astype(int)
    means = c.groupby(['gx', 'gy'])[['x', 'y']].transform('mean')
    c['dist'] = (c.x-means.x)**2 + (c.y-means.y)**2
    c = c.sort_values('dist').drop_duplicates(['gx', 'gy']).sort_values(['gx', 'gy']).reset_index(drop=True)
    c['site_id'] = ['candidate_%03d' % i for i in range(len(c))]
    c['kind'] = 'hypothetical'
    existing = pd.read_csv(ROOT / 'data/lora-bonn/survey/gateways.csv', dtype={'id': str})
    existing['x'], existing['y'] = to_utm.transform(existing.longitude.to_numpy(), existing.latitude.to_numpy())
    existing['site_id'] = 'existing_' + existing.id
    existing['kind'] = 'existing'
    sites = pd.concat([existing[['site_id','kind','longitude','latitude','x','y','antenna_height']],
                       c[['site_id','kind','longitude','latitude','x','y']]], ignore_index=True)
    return demand, sites


def score_pairs(demand, sites, height):
    sites = sites.copy()
    sites.loc[sites.kind == 'hypothetical', 'antenna_height'] = height
    frame = demand[['cell_id', 'longitude', 'latitude']].rename(columns={
        'longitude': 'longitude_sensor', 'latitude': 'latitude_sensor'}).merge(
        sites[['site_id','longitude','latitude','antenna_height']].rename(columns={
            'longitude': 'longitude_gateway','latitude': 'latitude_gateway'}), how='cross')
    lat1 = np.radians(frame.latitude_sensor.to_numpy()); lat2 = np.radians(frame.latitude_gateway.to_numpy())
    dlat = lat2 - lat1
    dlon = np.radians(frame.longitude_gateway.to_numpy()-frame.longitude_sensor.to_numpy())
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    distance = 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))
    # Same-position candidate/demand pairs are outside Bonn's 218 m minimum.
    # Use a conservative minimum distance of 218 m rather than log10(0).
    frame['log_distance_km'] = np.log10(np.maximum(distance, 0.218))
    frame['distance_was_clipped'] = distance < 0.218
    f = path_features(frame)
    if f.isna().any().any():
        raise ValueError('Some candidate links are outside the raster coverage')
    frame[TERRAIN_FEATURES + SURFACE_FEATURES] = f
    return frame


def greedy(matrix, site_ids, k):
    covered = np.zeros(matrix.shape[0], dtype=bool)
    chosen = []
    coverage = []
    available = np.ones(matrix.shape[1], dtype=bool)
    for _ in range(min(k, matrix.shape[1])):
        gains = (matrix & ~covered[:,None]).sum(axis=0)
        gains[~available] = -1
        j = int(np.argmax(gains))
        if gains[j] <= 0:
            break
        available[j] = False
        covered |= matrix[:,j]
        chosen.append(site_ids[j])
        coverage.append(float(covered.mean()))
    return chosen, coverage, covered


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    demand, sites = make_points()
    demand.to_csv(OUT / 'demand_cells.csv', index=False)
    sites.to_csv(OUT / 'candidate_sites.csv', index=False)
    print(f'Demand cells: {len(demand)}; candidates: {len(sites)} ({sum(sites.kind == "existing")} existing)', flush=True)
    pairs = build_pairs()
    fits = {}
    for scenario in ['in_range', 'same_day']:
        train = pairs.loc[pairs[scenario] & ((pairs.n_heard - pairs.heard) > 0)]
        fits[scenario] = model().fit(train[MODELS['terrain_surface']], train.heard)
        print(f'Trained {scenario} on {len(train):,} inferred-label links', flush=True)
    del pairs
    records, selections, maps = [], [], {}
    site_ids = sites.site_id.to_numpy()
    existing_mask = sites.kind.to_numpy() == 'existing'
    for height in HEIGHTS:
        paths = score_pairs(demand, sites, height)
        n_clipped = int(paths.distance_was_clipped.sum())
        for scenario, fit in fits.items():
            scores = fit.predict_proba(paths[MODELS['terrain_surface']])[:, 1].reshape(len(demand), len(sites))
            for threshold in THRESHOLDS:
                matrix = scores >= threshold
                existing_covered = matrix[:,existing_mask].any(axis=1)
                chosen, coverage, covered = greedy(matrix, site_ids, K_MAX)
                for k in range(1, K_MAX+1):
                    actual_k = min(k, len(chosen))
                    rate = coverage[actual_k-1] if actual_k else 0.0
                    records.append({'activity_scenario': scenario, 'candidate_height_m': height,
                                    'score_threshold': threshold, 'budget_k': k,
                                    'selected_k': actual_k, 'optimized_fraction': rate,
                                    'existing_nine_fraction': existing_covered.mean(),
                                    'clipped_link_count': n_clipped})
                    for rank, site_id in enumerate(chosen[:actual_k], 1):
                        selections.append({'activity_scenario': scenario, 'candidate_height_m': height,
                                           'score_threshold': threshold, 'budget_k': k,
                                           'rank': rank, 'site_id': site_id})
                if (height, scenario, threshold) == (16, 'same_day', 0.5):
                    maps = {'existing': existing_covered, 'optimized': covered, 'chosen': chosen}
        print(f'Scored candidate height {height} m; {n_clipped} short links clipped', flush=True)
    result = pd.DataFrame(records)
    result.to_csv(OUT / 'scenario_results.csv', index=False, float_format='%.6f')
    pd.DataFrame(selections).to_csv(OUT / 'selected_sites.csv', index=False)
    pd.DataFrame({'cell_id': demand.cell_id, 'existing_covered': maps['existing'],
                  'greedy_covered': maps['optimized']}).to_csv(OUT / 'base_coverage_flags.csv', index=False)
    base = result[(result.activity_scenario=='same_day') & (result.candidate_height_m==16) & (result.score_threshold==0.5)]
    fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
    ax.plot(base.budget_k, base.optimized_fraction*100, marker='o', label='Greedy selected sites')
    ax.axhline(base.existing_nine_fraction.iloc[0]*100, color='red', linestyle='--', label='All nine existing sites')
    ax.set(xlabel='Gateway budget (k)', ylabel='Road-cell score coverage (%)',
           title='Illustrative simulation: threshold 0.5, candidate height 16 m', ylim=(0,105))
    ax.legend()
    fig.savefig(OUT / 'coverage_curve.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12,6), constrained_layout=True)
    for ax, name in zip(axes, ['existing','optimized']):
        ax.scatter(demand.x, demand.y, c=np.where(maps[name], '#137b5b','#dc8051'), s=9)
        pick = sites.kind == 'existing' if name == 'existing' else sites.site_id.isin(maps['chosen'])
        ax.scatter(sites.loc[pick,'x'], sites.loc[pick,'y'], marker='^', s=65, c='#294ea1', edgecolors='white')
        ax.set(title='All existing sites' if name == 'existing' else f'Greedy sites (up to {K_MAX})',
               xlabel='Easting (m)', ylabel='Northing (m)')
        ax.tick_params(axis='x', labelrotation=40)
        ax.set_aspect('equal')
    fig.suptitle('Illustrative score coverage: green covered; orange uncovered; blue gateways')
    fig.savefig(OUT / 'placement_map.png', dpi=160); plt.close(fig)
    print(base[['budget_k','selected_k','optimized_fraction','existing_nine_fraction']].to_string(index=False))
    print('Results:', OUT)


if __name__ == '__main__':
    main()
