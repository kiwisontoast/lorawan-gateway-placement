"""Spatially blocked validation of the final model, for the eight period gateways.

Leave-one-gateway-out alone lets the held-out gateway's test packets share truck
positions with training packets heard by other gateways, so sensor-side features
(ground height, local clutter) can carry over. Here each fold also holds out a
set of 2 km blocks: the model is trained without gateway g and without any packet
within one 500 m cell of the test blocks, then tested on g inside the blocks.
Predictions from all folds are pooled per gateway; plain leave-one-gateway-out on
the same period table is run alongside as the reference.

Run after models_v2.py: python spatial_cv_v2.py, then redo the error adjustment
with the blocked error profiles:
    python robustness_v2.py --profiles results/v2_spatial_cv/footprint_metrics_blocked.csv --write-to results/v2_spatial_cv
"""
import site
from pathlib import Path

ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / '.geo_deps'))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import features_v2 as fv
from models_v2 import FINAL, PERIOD_GATEWAYS, SETS, SURVEY, demand_cells, eligible, make_model, pair_table, period_pairs

OUT = ROOT / 'results/v2_spatial_cv'
BLOCK_CELLS = 4    # 4 x 500 m = 2 km blocks
BUFFER_CELLS = 1   # training packets within one cell of a test block are dropped
FOLDS = 5
THRESHOLD, MIN_PACKETS = 0.5, 5


def key(gx, gy):
    return np.asarray(gx, dtype=np.int64) * 100_000 + np.asarray(gy, dtype=np.int64)


def fold_of(gx, gy, block_fold):
    b = key(np.floor_divide(gx, BLOCK_CELLS), np.floor_divide(gy, BLOCK_CELLS))
    return pd.Series(b).map(block_fold).to_numpy()


def buffered(cells_in_fold):
    """Keys of the fold's cells dilated by BUFFER_CELLS."""
    gx, gy = cells_in_fold
    d = np.arange(-BUFFER_CELLS, BUFFER_CELLS + 1)
    dx, dy = np.meshgrid(d, d)
    return np.unique(key(gx[:, None] + dx.ravel(), gy[:, None] + dy.ravel()))


def footprint_rows(g, validation, cell):
    cell = cell[cell.n >= MIN_PACKETS]
    truth, pred = cell.observed >= THRESHOLD, cell.rep_p >= THRESHOLD
    tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
    return {'held_out_gateway': g, 'validation': validation, 'model': FINAL[0], 'features': FINAL[1],
            'prediction_basis': 'rep_p', 'cells': len(cell), 'observed_covered': int(truth.sum()),
            'predicted_covered': int(pred.sum()), 'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
            'f1': 2 * tp / max(2 * tp + fp + fn, 1), 'iou': tp / max(tp + fp + fn, 1),
            'accuracy': float((pred == truth).mean())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    period = period_pairs(pair_table())
    demand = demand_cells()
    grids = fv.Grids()
    gws = pd.read_csv(SURVEY / 'gateways.csv', dtype={'id': str}).set_index('id')
    cols = SETS[FINAL[1]]
    rng = np.random.default_rng(7)
    blocks = np.unique(key(period.gx // BLOCK_CELLS, period.gy // BLOCK_CELLS))
    block_fold = dict(zip(rng.permutation(blocks), np.arange(len(blocks)) % FOLDS))
    period['fold'] = fold_of(period.gx, period.gy, block_fold)
    period['cell_key'] = key(period.gx, period.gy)
    rep = demand[['cell_id', 'gx', 'gy', 'longitude', 'latitude']].copy()
    rep['fold'] = fold_of(rep.gx, rep.gy, block_fold)
    rows, auc_rows = [], []
    for g in PERIOD_GATEWAYS:
        links = rep.rename(columns={'longitude': 'longitude_sensor', 'latitude': 'latitude_sensor'}).assign(
            longitude_gateway=gws.loc[g, 'longitude'], latitude_gateway=gws.loc[g, 'latitude'],
            antenna_height=float(gws.loc[g, 'antenna_height']))
        links = pd.concat([links, fv.path_features(links, grids).drop(columns=['antenna_height'])], axis=1)
        train, hold = eligible(period, g)
        te = period.loc[hold, ['gx', 'gy', 'fold', 'heard'] + cols].copy()
        # Reference: plain leave-one-gateway-out on the same table.
        model = make_model(FINAL[0], cols).fit(period.loc[train, cols], period.loc[train, 'heard'])
        te['p_logo'] = model.predict_proba(te[cols])[:, 1]
        links['p_logo'] = model.predict_proba(links[cols])[:, 1]
        te['p_blocked'], links['p_blocked'] = np.nan, np.nan
        dropped = []
        for k in range(FOLDS):
            test_cells = period.loc[period.fold == k, ['gx', 'gy']].drop_duplicates()
            near = period.cell_key.isin(buffered((test_cells.gx.to_numpy(), test_cells.gy.to_numpy()))).to_numpy()
            tr = train & ~near
            dropped.append(1 - tr.sum() / train.sum())
            model = make_model(FINAL[0], cols).fit(period.loc[tr, cols], period.loc[tr, 'heard'])
            m = (te.fold == k).to_numpy()
            te.loc[m, 'p_blocked'] = model.predict_proba(te.loc[m, cols])[:, 1]
            m = (links.fold == k).to_numpy()
            if m.any():
                links.loc[m, 'p_blocked'] = model.predict_proba(links.loc[m, cols])[:, 1]
        for v in ('logo', 'blocked'):
            p = te[f'p_{v}']
            auc_rows.append({'held_out_gateway': g, 'validation': v, 'test_pairs': int(p.notna().sum()),
                             'auc': roc_auc_score(te.heard[p.notna()], p[p.notna()]),
                             'mean_train_dropped': float(np.mean(dropped)) if v == 'blocked' else 0.0})
            cell = te.groupby(['gx', 'gy']).agg(n=('heard', 'size'), observed=('heard', 'mean')).reset_index()
            cell = links[['cell_id', 'gx', 'gy', f'p_{v}']].rename(columns={f'p_{v}': 'rep_p'}).merge(cell, on=['gx', 'gy'])
            rows.append(footprint_rows(g, v, cell))
        print(f'held out {g}: AUC logo {auc_rows[-2]["auc"]:.3f}, blocked {auc_rows[-1]["auc"]:.3f}; '
              f'training dropped per fold {np.mean(dropped):.0%}', flush=True)
    fp = pd.DataFrame(rows)
    auc = pd.DataFrame(auc_rows)
    fp.to_csv(OUT / 'footprint_metrics_all.csv', index=False, float_format='%.6f')
    fp[fp.validation == 'blocked'].drop(columns='validation').to_csv(OUT / 'footprint_metrics_blocked.csv', index=False, float_format='%.6f')
    auc.to_csv(OUT / 'auc_by_gateway.csv', index=False, float_format='%.6f')
    summary = fp.groupby('validation')[['precision', 'recall', 'f1']].mean().join(
        auc.groupby('validation').auc.agg(mean_auc='mean', min_auc='min'))
    summary.to_csv(OUT / 'summary.csv', float_format='%.4f')
    print(summary.to_string(float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    main()
