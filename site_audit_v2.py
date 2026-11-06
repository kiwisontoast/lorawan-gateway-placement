"""Aerial check of the sites chosen in the double-coverage (r = 2) designs.

For each scenario folder, re-solve the r = 2 designs with k = 6, 7 and 8 sites
(new-only and mixed pools), measure the surroundings of every chosen site on the
2 m surface model, and fetch a 300 m NRW orthophoto (DOP, Data licence Germany
zero 2.0) of each. The photos are judged by eye; the judgements live in
results/site_audit/verdicts.csv and are merged into site_audit.csv.

Run after placement_v2.py for each scenario:
    python site_audit_v2.py baseline_check mast25 mast15_clear_canopy mast25_clear_canopy
"""
import site
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / '.geo_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import features_v2 as fv
from optimize_v2 import solve_cover
from placement_v2 import MAIN_THRESHOLD, clutter_p90

SCENARIOS = ROOT / 'results/v2_sensitivity'
OUT = ROOT / 'results/site_audit'
PHOTOS = OUT / 'photos'
BUDGETS = (6, 7, 8)
HALF = 150  # photo half-width, m
WMS = ('https://www.wms.nrw.de/geobasis/wms_nw_dop?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=nw_dop_rgb'
       '&STYLES=&CRS=EPSG:25832&BBOX={:.0f},{:.0f},{:.0f},{:.0f}&WIDTH=600&HEIGHT=600&FORMAT=image/jpeg')
LANDCOVER = {10: 'tree', 20: 'shrub', 30: 'grass', 40: 'crop', 50: 'built', 60: 'bare', 80: 'water', 90: 'wetland'}


def designs(scenarios):
    rows = []
    for scen in scenarios:
        d = SCENARIOS / scen
        z = np.load(d / 'score_matrix.npz')
        sites = pd.read_csv(d / 'candidate_sites.csv').set_index('site_id').loc[z['site_ids']].reset_index()
        a = z['scores'] >= MAIN_THRESHOLD
        existing = (sites.kind == 'existing').to_numpy()
        for pool, allowed in (('greenfield', ~existing), ('full_pool', None)):
            for k in BUDGETS:
                _, chosen, covered = solve_cover(a, 'max_coverage', r=2, budget=k, allowed=allowed)
                rows += [{'scenario': scen, 'pool': pool, 'budget': k, 'modelled_cells': int(covered.sum()),
                          'site_id': s.site_id, 'kind': s.kind, 'x': s.x, 'y': s.y, 'antenna_height': s.antenna_height}
                         for s in sites[chosen].itertuples()]
        print(f'designs: {scen}', flush=True)
    return pd.DataFrame(rows)


def surroundings(sites, grids):
    row, col = grids.index(sites.x.to_numpy(), sites.y.to_numpy())
    return sites.assign(ground_m=grids.dgm[row, col].astype(float),
                        landcover=[LANDCOVER.get(int(v), int(v)) for v in grids.landcover[row, col]],
                        clutter_p90_m=clutter_p90(grids, sites.x, sites.y))


def photo(s):
    f = PHOTOS / f'{s.site_id}.jpg'
    if not f.exists():
        with urllib.request.urlopen(WMS.format(s.x - HALF, s.y - HALF, s.x + HALF, s.y + HALF), timeout=60) as r:
            f.write_bytes(r.read())
    return f


def sheets(sites):
    for old in OUT.glob('sheet_*.*'):
        old.unlink()
    sites = sites.sort_values(['kind', 'site_id']).reset_index(drop=True)
    for i in range(0, len(sites), 6):
        chunk = sites.iloc[i:i + 6]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10.4), constrained_layout=True)
        for ax, s in zip(axes.ravel(), chunk.itertuples()):
            ax.imshow(plt.imread(photo(s)), extent=(-HALF, HALF, -HALF, HALF))
            ax.add_patch(plt.Circle((0, 0), 25, fill=False, color='red', lw=1.2))
            ax.plot([-8, 8], [0, 0], 'r-', [0, 0], [-8, 8], 'r-', lw=1.2)
            ax.set_title(f'{s.site_id[:30]} ({s.kind})\nground {s.ground_m:.0f} m, antenna {s.antenna_height:.0f} m, '
                         f'clutter p90 {s.clutter_p90_m:.0f} m, {s.landcover}', fontsize=9)
            ax.set_xticks([-HALF, 0, HALF]); ax.set_yticks([-HALF, 0, HALF]); ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis('off')
        fig.suptitle('NRW DOP orthophotos (dl-de/zero-2.0), 300 m square; red circle = 25 m around the site', fontsize=10)
        fig.savefig(OUT / f'sheet_{i // 6 + 1}.jpg', dpi=110, pil_kwargs={'quality': 85})
        plt.close(fig)


def main(scenarios):
    PHOTOS.mkdir(parents=True, exist_ok=True)
    d = designs(scenarios)
    d.drop(columns=['x', 'y']).to_csv(OUT / 'designs_r2.csv', index=False, float_format='%.2f')
    # A site's antenna height depends on the scenario; keep the tallest seen.
    sites = d.sort_values('antenna_height', ascending=False).drop_duplicates('site_id')[
        ['site_id', 'kind', 'x', 'y', 'antenna_height']]
    grids = fv.Grids()
    sites = surroundings(sites, grids)
    del grids
    sheets(sites)
    verdicts = OUT / 'verdicts.csv'
    if verdicts.exists():
        sites = sites.merge(pd.read_csv(verdicts), on='site_id', how='left')
    sites['used_in'] = sites.site_id.map(d.groupby('site_id').apply(
        lambda g: ' '.join(sorted({f'{s}/{p}/{k}' for s, p, k in zip(g.scenario, g.pool, g.budget)})), include_groups=False))
    sites.to_csv(OUT / 'site_audit.csv', index=False, float_format='%.1f')
    missing = sites[sites.get('verdict', pd.Series(index=sites.index, dtype=object)).isna()]
    print(f'{len(sites)} sites; {len(missing)} without a verdict', *missing.site_id, sep='\n')


if __name__ == '__main__':
    main(sys.argv[1:] or ['baseline_check', 'mast25', 'mast15_clear_canopy', 'mast25_clear_canopy'])
