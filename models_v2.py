"""Version-2 coverage models: LightGBM, feature ablation, and new-site footprint tests.

Run after fetch_nrw_1m.py, fetch_lod2_buildings.py and fetch_worldcover.py:
    python models_v2.py
Labels are the same inferred, conditional reception labels as reception_bonn.py
(a packet observed by another gateway; same-day activity proxy). Model settings
are fixed in advance and are never tuned on a held-out gateway.
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
from lightgbm import LGBMClassifier, LGBMRegressor
from pyproj import Transformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import features_v2 as fv
import terrain_bonn
from reception_bonn import build_pairs

OUT = ROOT / 'results/v2_models'
CACHE = ROOT / 'data/derived/pair_features_v2.parquet'
SURVEY = ROOT / 'data/lora-bonn/survey'
START, END = '2020-11-10', '2021-01-26'
PERIOD_GATEWAYS = ['015c', '0172', '01c6', '06df', '08d5', '08fc', '090e', '0917']
# Packet key with the pair table's column names (reception_bonn.KEY uses the survey names).
PAIR_KEY = ['transceived_at', 'pkt_number', 'latitude_sensor', 'longitude_sensor']
V1 = [f'v1_{c}' for c in terrain_bonn.TERRAIN_FEATURES + terrain_bonn.SURFACE_FEATURES]
SETS = {
    'distance_height': fv.BASE_FEATURES,
    'v1_20m': fv.BASE_FEATURES + V1,
    'v2_terrain': fv.BASE_FEATURES + fv.TERRAIN_FEATURES,
    'v2_surface': fv.BASE_FEATURES + fv.TERRAIN_FEATURES + fv.SURFACE_FEATURES,
    'v2_buildings': fv.BASE_FEATURES + fv.TERRAIN_FEATURES + fv.SURFACE_FEATURES + fv.BUILDING_FEATURES,
    'v2_full': fv.ALL_FEATURES,
    # Absolute ground elevations can act as gateway identifiers; test without them.
    'v2_site_agnostic': [f for f in fv.ALL_FEATURES if f not in ('ground_sensor_m', 'ground_gateway_m')],
}
# Physically expected direction of each feature's effect on reception / power.
MONOTONE = {'log_distance_km': -1, 'antenna_height': 1, 'terrain_diffraction_db': -1,
            'surface_diffraction_db': -1, 'foliage_in_path_m': -1, 'buildings_blocking': -1}
CONFIGS = [('hgb', 'distance_height'), ('hgb', 'v1_20m'), ('lgbm', 'v1_20m'), ('lgbm', 'v2_terrain'),
           ('lgbm', 'v2_surface'), ('lgbm', 'v2_buildings'), ('lgbm', 'v2_full'), ('hgb', 'v2_full'),
           ('lgbm_mono', 'v2_full'), ('lgbm_mono', 'v2_site_agnostic')]
FINAL = ('lgbm_mono', 'v2_full')  # chosen before running, as the physics-constrained full model


def make_model(kind, features, regression=False):
    if kind == 'hgb':  # v1 configuration
        cls = HistGradientBoostingRegressor if regression else HistGradientBoostingClassifier
        return cls(max_iter=120 if regression else 80, max_leaf_nodes=15, min_samples_leaf=100,
                   learning_rate=.07, l2_regularization=1.0, early_stopping=False, random_state=42)
    params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=42, n_jobs=8, deterministic=True, force_row_wise=True, verbose=-1)
    if kind == 'lgbm_mono':
        params.update(monotone_constraints=[MONOTONE.get(f, 0) for f in features],
                      monotone_constraints_method='advanced')
    return (LGBMRegressor if regression else LGBMClassifier)(**params)


def pair_table():
    """All 9-gateway packet-gateway pairs with v1 (20 m) and v2 (2 m) features; cached."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    pairs = build_pairs(with_features=False)
    print(f'v2 features for {len(pairs):,} links', flush=True)
    grids = fv.Grids()
    v2 = fv.path_features(pairs, grids).drop(columns=['antenna_height'])
    del grids
    print('v1 features', flush=True)
    v1 = terrain_bonn.path_features(pairs).add_prefix('v1_')
    pairs = pairs.drop(columns=['log_distance_km', 'first', 'last'])
    table = pd.concat([pairs, v2, v1], axis=1)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(CACHE, index=False)
    return table


def eligible(pairs, held_out, scenario='same_day'):
    """Masks from reception_bonn.py: conditional on another (training) gateway hearing the packet."""
    test = (pairs.gw == held_out).to_numpy()
    held = pairs.loc[test, ['packet_id', 'heard']].set_index('packet_id').heard
    known = pairs.n_heard.to_numpy() - pairs.packet_id.map(held).fillna(0).to_numpy()
    allowed = pairs[scenario].to_numpy()
    train = ~test & allowed & ((known - pairs.heard.to_numpy()) > 0)
    hold = test & allowed & (known > 0)
    return train, hold


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def classification(pairs, gateways):
    rows, preds = [], {}
    for g in gateways:
        train, hold = eligible(pairs, g)
        tr, te = pairs.loc[train], pairs.loc[hold]
        for kind, fset in CONFIGS:
            cols = SETS[fset]
            p = make_model(kind, cols).fit(tr[cols], tr.heard).predict_proba(te[cols])[:, 1]
            preds[(kind, fset, g)] = p
            y = te.heard.to_numpy()
            rows.append({'held_out_gateway': g, 'model': kind, 'features': fset, 'test_pairs': len(te),
                         'auc': roc_auc_score(y, p), 'brier': brier_score_loss(y, p),
                         'log_loss': log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), 'ece': ece(y, p)})
        print(f'classification: held out {g}', flush=True)
    return pd.DataFrame(rows), preds


def regression(pairs, gateways):
    obs = pairs[pairs.heard == 1].merge(
        pd.read_csv(SURVEY / 'samples.csv', dtype={'gw': str}).rename(columns={'latitude': 'latitude_sensor', 'longitude': 'longitude_sensor'}),
        on=['transceived_at', 'pkt_number', 'latitude_sensor', 'longitude_sensor', 'gw'], validate='one_to_one')
    obs['rpp'] = obs.rssi + np.minimum(obs.snr, 0)
    rows = []
    for g in gateways:
        test = (obs.gw == g).to_numpy()
        tr, te = obs[~test], obs[test]
        x = np.column_stack([np.ones((~test).sum()), tr.log_distance_km])
        a, b = np.linalg.lstsq(x, tr.rpp, rcond=None)[0]
        preds = {('log_distance', 'distance'): a + b * te.log_distance_km}
        for kind, fset in [('hgb', 'v1_20m'), ('lgbm', 'v2_full'), ('lgbm_mono', 'v2_full')]:
            cols = SETS[fset]
            preds[(kind, fset)] = make_model(kind, cols, regression=True).fit(tr[cols], tr.rpp).predict(te[cols])
        for (kind, fset), p in preds.items():
            e = np.asarray(p) - te.rpp.to_numpy()
            rows.append({'held_out_gateway': g, 'model': kind, 'features': fset, 'n': len(te),
                         'mae_db': np.abs(e).mean(), 'rmse_db': np.sqrt((e**2).mean()), 'bias_db': e.mean()})
        print(f'regression: held out {g}', flush=True)
    return pd.DataFrame(rows)


def demand_cells():
    from placement_bonn import CELL_METRES, make_points
    demand, _ = make_points(START, END)
    demand['gx'] = np.floor(demand.x / CELL_METRES).astype(int)
    demand['gy'] = np.floor(demand.y / CELL_METRES).astype(int)
    return demand


def period_pairs(pairs):
    """Pairs of the eight-gateway period, with eligibility rebuilt and 500 m cell indices."""
    from placement_bonn import CELL_METRES
    period = pairs[pairs.transceived_at.between(START, END) & pairs.gw.isin(PERIOD_GATEWAYS)].copy()
    # Rebuild eligibility within the period-only, eight-gateway network.
    period = period.drop(columns=['packet_id', 'n_heard'])
    keys = period[PAIR_KEY].drop_duplicates().reset_index(drop=True)
    keys['packet_id'] = np.arange(len(keys))
    period = period.merge(keys, on=PAIR_KEY)
    period['n_heard'] = period.groupby('packet_id').heard.transform('sum')
    period = period.reset_index(drop=True)
    project = Transformer.from_crs(4326, 25832, always_xy=True)
    x, y = project.transform(period.longitude_sensor.to_numpy(), period.latitude_sensor.to_numpy())
    period['gx'], period['gy'] = np.floor(x / CELL_METRES).astype(int), np.floor(y / CELL_METRES).astype(int)
    return period


def footprint_test(pairs, demand, grids, configs, threshold=0.5, min_packets=5):
    """Leave one gateway out of training, predict its cell footprint, compare to its receptions."""
    period = period_pairs(pairs)
    gws = pd.read_csv(SURVEY / 'gateways.csv', dtype={'id': str}).set_index('id')
    rep = demand[['cell_id', 'gx', 'gy', 'longitude', 'latitude']]
    rows, cells_out = [], []
    for g in PERIOD_GATEWAYS:
        train, hold = eligible(period, g)
        tr, te = period.loc[train], period.loc[hold].copy()
        links = rep.rename(columns={'longitude': 'longitude_sensor', 'latitude': 'latitude_sensor'}).assign(
            longitude_gateway=gws.loc[g, 'longitude'], latitude_gateway=gws.loc[g, 'latitude'],
            antenna_height=float(gws.loc[g, 'antenna_height']))
        links = pd.concat([links, fv.path_features(links, grids).drop(columns=['antenna_height'])], axis=1)
        # The v1 feature set needs the 20 m features at the representative points too.
        links[V1] = terrain_bonn.path_features(links).to_numpy()
        for kind, fset in configs:
            cols = SETS[fset]
            model = make_model(kind, cols).fit(tr[cols], tr.heard)
            te['p'] = model.predict_proba(te[cols])[:, 1]
            cell = te.groupby(['gx', 'gy']).agg(n=('heard', 'size'), observed=('heard', 'mean'), mean_p=('p', 'mean')).reset_index()
            cell = rep.merge(cell, on=['gx', 'gy'], how='inner')
            cell = cell.merge(links[['cell_id']].assign(rep_p=model.predict_proba(links[cols])[:, 1]), on='cell_id')
            cell = cell[cell.n >= min_packets]
            truth = cell.observed >= threshold
            for basis in ['mean_p', 'rep_p']:
                pred = cell[basis] >= threshold
                tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
                rows.append({'held_out_gateway': g, 'model': kind, 'features': fset, 'prediction_basis': basis,
                             'cells': len(cell), 'observed_covered': int(truth.sum()), 'predicted_covered': int(pred.sum()),
                             'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
                             'f1': 2 * tp / max(2 * tp + fp + fn, 1), 'iou': tp / max(tp + fp + fn, 1),
                             'accuracy': float((pred == truth).mean()),
                             'cell_rate_mae': float((cell[basis] - cell.observed).abs().mean())})
            cells_out.append(cell.assign(held_out_gateway=g, model=kind, features=fset))
        print(f'footprint: held out {g}', flush=True)
    return pd.DataFrame(rows), pd.concat(cells_out, ignore_index=True)


def plot_ablation(summary):
    s = summary.copy()
    s['label'] = s.model + ' / ' + s.features
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.barh(s.label, s.mean_auc, xerr=[s.mean_auc - s.min_auc, s.max_auc - s.mean_auc], color='#4b6f8a', ecolor='#999')
    ax.set(xlabel='Leave-one-gateway-out AUC (mean, bars = gateway range)', xlim=(0.6, 1.0),
           title='Feature and model ablation (inferred conditional labels)')
    ax.invert_yaxis()
    fig.savefig(OUT / 'ablation_auc.png', dpi=160)
    plt.close(fig)


def plot_calibration(pairs, preds, gateways):
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    for kind, fset in [('hgb', 'v1_20m'), FINAL]:
        ys, ps = [], []
        for g in gateways:
            _, hold = eligible(pairs, g)
            ys.append(pairs.loc[hold, 'heard'].to_numpy()); ps.append(preds[(kind, fset, g)])
        y, p = np.concatenate(ys), np.concatenate(ps)
        bins = np.clip((p * 10).astype(int), 0, 9)
        frame = pd.DataFrame({'b': bins, 'y': y, 'p': p}).groupby('b').mean()
        ax.plot(frame.p, frame.y, marker='o', label=f'{kind} / {fset}')
    ax.plot([0, 1], [0, 1], color='black', linewidth=.8)
    ax.set(xlabel='Predicted score (held-out gateway)', ylabel='Observed reception fraction', title='Calibration, pooled')
    ax.legend()
    fig.savefig(OUT / 'calibration.png', dpi=160)
    plt.close(fig)


def main(stages=('classification', 'regression', 'footprint')):
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = pair_table()
    missing = pairs[fv.ALL_FEATURES + V1].isna().mean()
    print('feature missing fractions (>0):', missing[missing > 0].round(4).to_dict(), flush=True)
    gateways = sorted(pairs.gw.unique())
    if 'classification' in stages:
        run_classification(pairs, gateways)
    if 'regression' in stages:
        run_regression(pairs, gateways)
    if 'footprint' in stages:
        run_footprint(pairs)
    (OUT / 'run_metadata.json').write_text(json.dumps({
        'labels': 'same-day activity proxy; reception conditional on another gateway observing the packet',
        'configs': CONFIGS, 'final_model': FINAL, 'monotone_constraints': MONOTONE,
        'footprint_threshold': 0.5, 'footprint_min_packets_per_cell': 5,
        'lightgbm_version': __import__('lightgbm').__version__}, indent=2), encoding='utf-8')


def run_classification(pairs, gateways):
    cls, preds = classification(pairs, gateways)
    cls.to_csv(OUT / 'classification_metrics.csv', index=False, float_format='%.6f')
    summary = cls.groupby(['model', 'features'], sort=False).agg(
        mean_auc=('auc', 'mean'), min_auc=('auc', 'min'), max_auc=('auc', 'max'),
        mean_brier=('brier', 'mean'), mean_log_loss=('log_loss', 'mean'), mean_ece=('ece', 'mean')).reset_index()
    ref = cls[(cls.model == 'hgb') & (cls.features == 'v1_20m')].set_index('held_out_gateway').auc
    summary['gateways_better_than_v1'] = [int((cls[(cls.model == m) & (cls.features == f)].set_index('held_out_gateway').auc > ref).sum())
                                          for m, f in zip(summary.model, summary.features)]
    summary.to_csv(OUT / 'classification_summary.csv', index=False, float_format='%.4f')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.3f}'), flush=True)
    plot_ablation(summary)
    plot_calibration(pairs, preds, gateways)


def run_regression(pairs, gateways):
    reg = regression(pairs, gateways)
    reg.to_csv(OUT / 'regression_metrics.csv', index=False, float_format='%.6f')
    reg_sum = reg.groupby(['model', 'features'], sort=False).apply(
        lambda d: pd.Series({'pooled_mae_db': np.average(d.mae_db, weights=d.n),
                             'pooled_bias_db': np.average(d.bias_db, weights=d.n),
                             'mean_gateway_mae_db': d.mae_db.mean()}), include_groups=False).reset_index()
    reg_sum.to_csv(OUT / 'regression_summary.csv', index=False, float_format='%.4f')
    print(reg_sum.to_string(index=False, float_format=lambda v: f'{v:.2f}'), flush=True)


def run_footprint(pairs):
    grids = fv.Grids()
    fp, cells = footprint_test(pairs, demand_cells(), grids, [('hgb', 'distance_height'), ('hgb', 'v1_20m'), ('lgbm', 'v2_full'), FINAL])
    fp.to_csv(OUT / 'footprint_metrics.csv', index=False, float_format='%.6f')
    cells.to_csv(OUT / 'footprint_cells.csv', index=False, float_format='%.6f')
    fsum = fp.groupby(['model', 'features', 'prediction_basis'], sort=False)[
        ['precision', 'recall', 'f1', 'iou', 'accuracy', 'cell_rate_mae']].mean().reset_index()
    fsum.to_csv(OUT / 'footprint_summary.csv', index=False, float_format='%.4f')
    print(fsum.to_string(index=False, float_format=lambda v: f'{v:.3f}'), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', nargs='+', default=['classification', 'regression', 'footprint'],
                        choices=['classification', 'regression', 'footprint'])
    main(parser.parse_args().stages)
