"""Additional geographic and chronological tests of received-power models."""
import site
from pathlib import Path
ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / '.geo_deps'))
import numpy as np
import pandas as pd
from pyproj import Transformer
from baseline_bonn import load_data, fit_log_distance, summarize
from geoai_bonn import new_model
from terrain_bonn import TERRAIN_FEATURES, SURFACE_FEATURES

OUT = ROOT / 'results/bonn_validation'
FEATURES = ['log_distance_km', 'antenna_height', *TERRAIN_FEATURES, *SURFACE_FEATURES]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_data().reset_index(drop=True)
    f = pd.read_csv(ROOT / 'results/bonn_terrain/terrain_features.csv', dtype={'gw':str})
    identifiers = ['gw','transceived_at','pkt_number','latitude_sensor','longitude_sensor']
    if len(d) != len(f):
        raise ValueError('Feature row count does not match original observations')
    for col in identifiers:
        if col in ['latitude_sensor','longitude_sensor']:
            ok = np.allclose(d[col], f[col], atol=1e-6, rtol=0)
        else:
            ok = (d[col].astype(str) == f[col].astype(str)).all()
        if not ok:
            raise ValueError(f'Feature row alignment failed for {col}')
    d[TERRAIN_FEATURES + SURFACE_FEATURES] = f[TERRAIN_FEATURES + SURFACE_FEATURES]
    d['log_distance_km'] = np.log10(d.distance_km)
    transform = Transformer.from_crs(4326, 25832, always_xy=True)
    x, y = transform.transform(d.longitude_sensor.to_numpy(), d.latitude_sensor.to_numpy())
    # Define geographic divisions by distinct sensor positions, avoiding repeated
    # receptions of the same coordinate being placed in different folds.
    unique = d[['longitude_sensor','latitude_sensor']].drop_duplicates()
    ux, uy = transform.transform(unique.longitude_sensor.to_numpy(), unique.latitude_sensor.to_numpy())
    x_cut, y_cut = np.median(ux), np.median(uy)
    quadrant = (x >= x_cut).astype(int) + 2*(y >= y_cut).astype(int)
    days = sorted(d.transceived_at.unique())
    split_date = days[int(len(days)*0.75)]
    folds = [(f'spatial_quadrant_{i}', quadrant == i) for i in range(4)]
    folds.append((f'time_since_{split_date}', d.transceived_at.to_numpy() >= split_date))
    rows = []
    for fold, test in folds:
        train = ~test
        a, b = fit_log_distance(d.loc[train])
        preds = {'log_distance': a + b*d.loc[test, 'log_distance_km']}
        fit = new_model()
        fit.fit(d.loc[train, FEATURES], d.loc[train, 'observed_rpp_dbm'])
        preds['boosted_terrain_surface'] = fit.predict(d.loc[test, FEATURES])
        for name, pred in preds.items():
            rows.append({'split': fold, 'model': name, 'train_receptions': int(train.sum()),
                         'test_gateways': d.loc[test, 'gw'].nunique(),
                         'train_gateways': d.loc[train, 'gw'].nunique(),
                         **summarize(d.loc[test, 'observed_rpp_dbm'], pred)})
        print(f'Completed {fold}', flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / 'split_metrics.csv', index=False, float_format='%.6f')
    (OUT / 'split_definition.txt').write_text(
        f'Four quadrants in EPSG:25832 at unique-position median easting {x_cut:.2f} m and northing {y_cut:.2f} m.\n'
        f'Temporal test: date >= {split_date}; earlier dates train.\n', encoding='utf-8')
    print(results.to_string(index=False, float_format=lambda x: f'{x:.2f}'))


if __name__ == '__main__':
    main()
