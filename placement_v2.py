"""Version-2 period placement: LoD2 rooftops, terrain high points and a grid,
scored by the physics-constrained LightGBM model, optimized with redundancy.

Run after models_v2.py: python placement_v2.py, then python robustness_v2.py
Scores are inferred, conditional reception scores, not delivery probabilities.
Sensitivity runs: --mast-m 25, --no-forest-hills, each with its own --out.
"""
import argparse
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
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import wkb

import features_v2 as fv
from models_v2 import FINAL, PAIR_KEY, PERIOD_GATEWAYS, SETS, START, END, demand_cells, make_model, pair_table
from optimize_v2 import greedy_cover, solve_cover

OUT = ROOT / 'results/v2_placement'
SURVEY = ROOT / 'data/lora-bonn/survey'
THRESHOLDS = [0.3, 0.5, 0.7]
MAIN_THRESHOLD = 0.5
REDUNDANCY = [1, 2]
MAST_M = 15.0          # hypothetical pole height for grid and high-point sites (inside 12-30 m training range)
TREE_COVER = 10        # WorldCover class
ROOF_MAST_M = 2.0      # antenna above the highest LoD2 roof point
ROOF_RANGE = (10, 28)  # roof height above ground: antenna stays within 12-30 m
NEAR_DEMAND_M = 1000


def candidates(demand, grids, mast_m=MAST_M, forest_hills=True):
    to_ll = Transformer.from_crs(25832, 4326, always_xy=True)
    gw = pd.read_csv(SURVEY / 'gateways.csv', dtype={'id': str})
    gw = gw[gw.id.isin(PERIOD_GATEWAYS)]
    to_utm = Transformer.from_crs(4326, 25832, always_xy=True)
    ex = pd.DataFrame({'site_id': 'existing_' + gw.id, 'kind': 'existing', 'longitude': gw.longitude, 'latitude': gw.latitude,
                       'antenna_height': gw.antenna_height.astype(float)})
    ex['x'], ex['y'] = to_utm.transform(ex.longitude.to_numpy(), ex.latitude.to_numpy())
    tree = cKDTree(demand[['x', 'y']].to_numpy())

    def near(x, y):
        return tree.query(np.column_stack([x, y]))[0] <= NEAR_DEMAND_M

    def ground(x, y):
        row, col = grids.index(np.asarray(x), np.asarray(y))
        return grids.dgm[row, col]

    # LoD2 rooftops: tallest eligible building part in each 1 km tile near demand.
    parts = pd.read_parquet(fv.LOD2)
    parts = parts[parts.area_m2 >= 100].copy()
    pts = [wkb.loads(b).representative_point() for b in parts.footprint_wkb]
    parts['x'], parts['y'] = [p.x for p in pts], [p.y for p in pts]
    inside = (parts.x > grids.transform.c + 50) & (parts.x < grids.transform.c + 2 * grids.shape[1] - 50) & \
             (parts.y < grids.transform.f - 50) & (parts.y > grids.transform.f - 2 * grids.shape[0] + 50)
    parts = parts[inside]
    parts['antenna_height'] = parts.roof_max_z_m + ROOF_MAST_M - ground(parts.x, parts.y)
    roof_h = parts.antenna_height - ROOF_MAST_M
    parts = parts[roof_h.between(*ROOF_RANGE) & near(parts.x, parts.y)]
    parts['tile'] = (parts.x // 1000).astype(int).astype(str) + '_' + (parts.y // 1000).astype(int).astype(str)
    roofs = parts.sort_values(['tile', 'antenna_height', 'part_id'], ascending=[True, False, True]).groupby('tile').head(1)
    roofs = pd.DataFrame({'site_id': 'roof_' + roofs.part_id, 'kind': 'rooftop', 'x': roofs.x, 'y': roofs.y,
                          'antenna_height': roofs.antenna_height, 'lod2_function': roofs.function,
                          'roof_area_m2': roofs.area_m2})
    # Terrain high points: highest bare-earth 2 m cell per 2 km tile, off buildings and water.
    hills = []
    step = 1000  # 2 km in 2 m cells
    for r0 in range(0, grids.shape[0], step):
        for c0 in range(0, grids.shape[1], step):
            z = grids.dgm[r0:r0 + step, c0:c0 + step].copy()
            z[(grids.buildings[r0:r0 + step, c0:c0 + step] != 0) | (grids.landcover[r0:r0 + step, c0:c0 + step] == 80)] = np.nan
            z[:25, :] = z[-25:, :] = np.nan
            z[:, :25] = z[:, -25:] = np.nan
            if np.isnan(z).all():
                continue
            r, c = np.unravel_index(np.nanargmax(z), z.shape)
            hills.append((grids.transform.c + (c0 + c + .5) * 2, grids.transform.f - (r0 + r + .5) * 2))
    hills = pd.DataFrame(hills, columns=['x', 'y'])
    hills = hills[near(hills.x, hills.y)].reset_index(drop=True)
    row, col = grids.index(hills.x.to_numpy(), hills.y.to_numpy())
    hills['landcover'] = grids.landcover[row, col]
    # Ids are assigned before any drop so a hill keeps its id across scenarios.
    hills = hills.assign(site_id=['hill_%03d' % i for i in range(len(hills))], kind='high_point', antenna_height=mast_m)
    if not forest_hills:
        hills = hills[hills.landcover != TREE_COVER]
    # Regular 1 km grid of masts.
    gx, gy = np.meshgrid(np.arange(360500, 374000, 1000), np.arange(5612500, 5632000, 1000))
    grid = pd.DataFrame({'x': gx.ravel().astype(float), 'y': gy.ravel().astype(float)})
    grid = grid[near(grid.x, grid.y)].reset_index(drop=True)
    grid = grid.assign(site_id=['grid_%03d' % i for i in range(len(grid))], kind='grid', antenna_height=mast_m)
    new = pd.concat([roofs, hills, grid], ignore_index=True)
    new['longitude'], new['latitude'] = to_ll.transform(new.x.to_numpy(), new.y.to_numpy())
    return pd.concat([ex, new], ignore_index=True)


def score(model, cols, demand, sites, grids):
    links = demand[['cell_id', 'longitude', 'latitude']].rename(
        columns={'longitude': 'longitude_sensor', 'latitude': 'latitude_sensor'}).merge(
        sites[['site_id', 'longitude', 'latitude', 'antenna_height']].rename(
            columns={'longitude': 'longitude_gateway', 'latitude': 'latitude_gateway'}), how='cross')
    feats = fv.path_features(links, grids)
    p = model.predict_proba(feats[cols])[:, 1].reshape(len(demand), len(sites))
    return p, int(feats.distance_was_clipped.sum())


def period_model():
    pairs = pair_table()
    period = pairs[pairs.transceived_at.between(START, END) & pairs.gw.isin(PERIOD_GATEWAYS)].copy()
    period['n_heard'] = period.groupby(PAIR_KEY).heard.transform('sum')
    train = period[period.same_day & ((period.n_heard - period.heard) > 0)]
    kind, fset = FINAL
    cols = SETS[fset]
    return make_model(kind, cols).fit(train[cols], train.heard), cols, len(train)


def run_experiments(a, kinds, tag):
    """All optimization questions for one binary score matrix."""
    existing, new = kinds == 'existing', kinds != 'existing'
    m = a.shape[0]
    rows, designs = [], {}

    def record(name, r, res, chosen, covered, **extra):
        row = {'matrix': tag, 'experiment': name, 'redundancy': r, **extra,
               'status': None if res is None else int(res.status),
               'optimal': bool(res is not None and res.status == 0),
               'mip_gap': None if res is None or getattr(res, 'mip_gap', None) is None else float(res.mip_gap)}
        if chosen is not None:
            row.update(sites=int(chosen.sum()), existing_used=int((chosen & existing).sum()),
                       new_used=int((chosen & new).sum()), covered_cells=int(covered.sum()),
                       covered_fraction=float(covered.mean()))
            designs[(tag, name, r)] = chosen
        else:
            row.update(sites=None, covered_cells=None, note='infeasible with these sites')
        rows.append(row)

    for r in REDUNDANCY:
        base = a[:, existing].sum(axis=1) >= r
        rows.append({'matrix': tag, 'experiment': 'existing_all', 'redundancy': r, 'sites': int(existing.sum()),
                     'existing_used': int(existing.sum()), 'new_used': 0, 'covered_cells': int(base.sum()),
                     'covered_fraction': float(base.mean()), 'optimal': True})
        designs[(tag, 'existing_all', r)] = existing.copy()
        record('existing_subset_preserve', r, *solve_cover(a, 'min_sites', r=r, allowed=existing, preserve=base))
        record('greenfield_match_count', r, *solve_cover(a, 'min_sites', r=r, target=int(base.sum()), allowed=new),
               target_cells=int(base.sum()))
        record('greenfield_preserve', r, *solve_cover(a, 'min_sites', r=r, allowed=new, preserve=base))
        record('full_pool_preserve', r, *solve_cover(a, 'min_sites', r=r, preserve=base))
        for frac in (0.90, 0.95):
            record(f'greenfield_reach_{int(frac*100)}pct', r,
                   *solve_cover(a, 'min_sites', r=r, target=int(np.ceil(frac * m)), allowed=new),
                   target_cells=int(np.ceil(frac * m)))
    return rows, designs


def curves(a, kinds, tag, r, k_max=10):
    rows = []
    for pool, allowed in [('existing_only', kinds == 'existing'), ('greenfield', kinds != 'existing'), ('full_pool', None)]:
        top = int(allowed.sum()) if allowed is not None else k_max
        for k in range(1, min(k_max, top) + 1):
            res, chosen, covered = solve_cover(a, 'max_coverage', r=r, budget=k, allowed=allowed)
            rows.append({'matrix': tag, 'redundancy': r, 'pool': pool, 'budget': k, 'method': 'integer',
                         'covered_cells': int(covered.sum()), 'optimal': bool(res.status == 0),
                         'mip_gap': float(getattr(res, 'mip_gap', 0) or 0)})
        chosen, covered = greedy_cover(a, min(k_max, top), r=r, allowed=allowed)
        for k in range(1, len(chosen) + 1):
            rows.append({'matrix': tag, 'redundancy': r, 'pool': pool, 'budget': k, 'method': 'greedy',
                         'covered_cells': int((a[:, chosen[:k]].sum(axis=1) >= r).sum()), 'optimal': None})
    return rows


def plot(demand, sites, a, designs, curve_rows):
    c = pd.DataFrame(curve_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, r in zip(axes, REDUNDANCY):
        sub = c[(c.matrix == f'tau_{MAIN_THRESHOLD}') & (c.redundancy == r)]
        for pool, color in [('existing_only', '#b5562b'), ('greenfield', '#2b6cb5'), ('full_pool', '#3b8a4a')]:
            s = sub[(sub.pool == pool) & (sub.method == 'integer')]
            g = sub[(sub.pool == pool) & (sub.method == 'greedy')]
            ax.plot(s.budget, 100 * s.covered_cells / len(demand), marker='o', color=color, label=f'{pool} (integer)')
            ax.plot(g.budget, 100 * g.covered_cells / len(demand), linestyle=':', color=color, label=f'{pool} (greedy)')
        base = a[:, sites.kind.to_numpy() == 'existing'].sum(axis=1) >= r
        ax.axhline(100 * base.mean(), color='black', linestyle='--', linewidth=.8, label='all 8 existing')
        ax.set(xlabel='Gateways', ylabel='Cells meeting score rule (%)', ylim=(0, 102),
               title=f'Redundancy r = {r} (score >= {MAIN_THRESHOLD})')
    axes[0].legend(fontsize=7, loc='lower right')
    fig.savefig(OUT / 'coverage_curves.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), constrained_layout=True)
    tag = f'tau_{MAIN_THRESHOLD}'
    panels = [('existing_all', 1, 'All 8 existing gateways'), ('greenfield_match_count', 1, 'Greenfield minimum, r = 1'),
              ('greenfield_match_count', 2, 'Greenfield minimum, r = 2')]
    marker = {'existing': '^', 'rooftop': 's', 'high_point': 'D', 'grid': 'o'}
    for ax, (name, r, title) in zip(axes, panels):
        chosen = designs.get((tag, name, r))
        if chosen is None:
            ax.set_title(title + ' (infeasible)'); continue
        cov = a[:, chosen].sum(axis=1) >= r
        ax.scatter(demand.x, demand.y, s=9, c=np.where(cov, '#16815e', '#d36342'))
        ax.scatter(sites.x, sites.y, s=4, c='#cccccc')
        for kind, mk in marker.items():
            s = sites[chosen & (sites.kind == kind).to_numpy()]
            ax.scatter(s.x, s.y, marker=mk, s=80, c='#312b80', edgecolors='white', label=kind if len(s) else None)
        ax.set(title=f'{title}: {int(chosen.sum())} sites, {int(cov.sum())}/{len(demand)} cells', xlabel='Easting (m)', ylabel='Northing (m)')
        ax.set_aspect('equal'); ax.tick_params(axis='x', labelrotation=40); ax.legend(fontsize=7, loc='lower left')
    fig.suptitle('Modelled score coverage; green covered, orange not; grey dots are unselected candidates')
    fig.savefig(OUT / 'design_maps.png', dpi=160)
    plt.close(fig)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--mast-m', type=float, default=MAST_M, help='pole height on grid and high-point sites')
    ap.add_argument('--no-forest-hills', action='store_true', help='drop high points in WorldCover tree cover')
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    OUT = args.out if args.out.is_absolute() else ROOT / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    grids = fv.Grids()
    demand = demand_cells()
    sites = candidates(demand, grids, args.mast_m, not args.no_forest_hills)
    sites.to_csv(OUT / 'candidate_sites.csv', index=False)
    demand.to_csv(OUT / 'demand_cells.csv', index=False)
    print('candidates:', sites.kind.value_counts().to_dict(), flush=True)
    model, cols, n_train = period_model()
    p, clipped = score(model, cols, demand, sites, grids)
    np.savez_compressed(OUT / 'score_matrix.npz', scores=p.astype('float32'), site_ids=sites.site_id.to_numpy(str),
                        cell_ids=demand.cell_id.to_numpy(str))
    kinds = sites.kind.to_numpy()
    rows, designs, curve_rows = [], {}, []
    for tau in THRESHOLDS:
        a = p >= tau
        r_, d_ = run_experiments(a, kinds, f'tau_{tau}')
        rows += r_; designs |= d_
        if tau == MAIN_THRESHOLD:
            for r in REDUNDANCY:
                curve_rows += curves(a, kinds, f'tau_{tau}', r)
    # Scenario: 06df (labelled "Old"; last reception on the period's final day) is unavailable.
    keep = sites.site_id.to_numpy() != 'existing_06df'
    a = p[:, keep] >= MAIN_THRESHOLD
    r_, d_ = run_experiments(a, kinds[keep], 'tau_0.5_without_06df')
    rows += r_
    exp = pd.DataFrame(rows)
    exp.to_csv(OUT / 'experiments.csv', index=False, float_format='%.4f')
    pd.DataFrame(curve_rows).to_csv(OUT / 'coverage_curves.csv', index=False)
    chosen_rows = [{'matrix': t, 'experiment': n, 'redundancy': r, 'site_id': s, 'kind': k}
                   for (t, n, r), c in designs.items() for s, k in zip(sites.site_id[c], sites.kind[c])]
    pd.DataFrame(chosen_rows).to_csv(OUT / 'designs.csv', index=False)
    a = p >= MAIN_THRESHOLD
    plot(demand, sites, a, designs, curve_rows)
    (OUT / 'run_metadata.json').write_text(json.dumps({
        'period': [START, END], 'gateways': PERIOD_GATEWAYS, 'model': FINAL, 'training_pairs': n_train,
        'demand_cells': len(demand), 'candidates': sites.kind.value_counts().to_dict(),
        'clipped_links': clipped, 'thresholds': THRESHOLDS, 'redundancy': REDUNDANCY,
        'mast_height_m': args.mast_m, 'forest_high_points': not args.no_forest_hills, 'roof_mast_m': ROOF_MAST_M, 'roof_height_range_m': ROOF_RANGE,
        'candidate_max_distance_to_demand_m': NEAR_DEMAND_M}, indent=2), encoding='utf-8')
    show = exp[exp.matrix == f'tau_{MAIN_THRESHOLD}'][['experiment', 'redundancy', 'sites', 'existing_used', 'new_used', 'covered_cells', 'optimal']]
    print(show.to_string(index=False), flush=True)
    print(exp[exp.matrix == 'tau_0.5_without_06df'][['experiment', 'redundancy', 'sites', 'covered_cells', 'optimal']].to_string(index=False))


if __name__ == '__main__':
    main()
