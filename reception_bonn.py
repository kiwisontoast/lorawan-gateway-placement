"""Sensitivity experiment for inferred Bonn packet receptions.

The target is reception by a gateway, conditional on the packet being observed
by another gateway. Inferred misses are not verified transmission failures.
Run with python reception_bonn.py after downloading both elevation rasters.
"""
import site
from pathlib import Path

ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / '.geo_deps'))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from terrain_bonn import path_features, TERRAIN_FEATURES, SURFACE_FEATURES

SURVEY = ROOT / 'data/lora-bonn/survey'
OUT = ROOT / 'results/bonn_reception'
KEY = ['transceived_at', 'pkt_number', 'latitude', 'longitude']
MODELS = {'distance_height': ['log_distance_km', 'antenna_height'],
          'terrain_surface': ['log_distance_km', 'antenna_height', *TERRAIN_FEATURES, *SURFACE_FEATURES]}


def model():
    return HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
                                          learning_rate=.07, l2_regularization=1,
                                          early_stopping=False, random_state=42)


def build_pairs(start_date=None, end_date=None, gateway_ids=None, with_features=True):
    s = pd.read_csv(SURVEY / 'samples.csv', dtype={'gw': str})
    if start_date is not None:
        s = s.loc[s.transceived_at >= start_date].copy()
    if end_date is not None:
        s = s.loc[s.transceived_at <= end_date].copy()
    if gateway_ids is not None:
        s = s.loc[s.gw.isin(gateway_ids)].copy()
    if s.empty:
        raise ValueError('No receptions available for requested period and gateways')
    gateways = pd.read_csv(SURVEY / 'gateways.csv', dtype={'id': str}).rename(
        columns={'id': 'gw', 'latitude': 'latitude_gateway', 'longitude': 'longitude_gateway'})
    if gateway_ids is not None:
        gateways = gateways.loc[gateways.gw.isin(gateway_ids)].copy()
        if set(gateways.gw) != set(gateway_ids):
            raise ValueError('Requested gateway missing from gateways.csv')
    packets = s[KEY].drop_duplicates().reset_index(drop=True)
    packets['packet_id'] = np.arange(len(packets))
    seen = s[KEY + ['gw']].merge(packets, on=KEY, validate='many_to_one')
    if seen.duplicated(['packet_id', 'gw']).any():
        raise ValueError('Provisional packet key has a repeated gateway')
    counts = seen.groupby('packet_id').size().rename('n_heard').reset_index()
    packets = packets.merge(counts, on='packet_id', validate='one_to_one')
    pairs = packets.merge(gateways, how='cross')
    heard = seen[['packet_id', 'gw']].assign(heard=1)
    pairs = pairs.merge(heard, how='left', on=['packet_id', 'gw'], validate='one_to_one')
    pairs['heard'] = pairs.heard.fillna(0).astype('int8')
    activity = s.groupby('gw').agg(first=('transceived_at', 'min'), last=('transceived_at', 'max')).reset_index()
    pairs = pairs.merge(activity, on='gw', validate='many_to_one')
    pairs['in_range'] = pairs.transceived_at.between(pairs['first'], pairs['last'])
    days = s[['gw', 'transceived_at']].drop_duplicates().assign(same_day=True)
    pairs = pairs.merge(days, on=['gw', 'transceived_at'], how='left', validate='many_to_one')
    pairs['same_day'] = pairs.same_day.fillna(False).astype(bool)
    pairs = pairs.rename(columns={'latitude': 'latitude_sensor', 'longitude': 'longitude_sensor'})
    lat1 = np.radians(pairs.latitude_sensor.to_numpy()); lat2 = np.radians(pairs.latitude_gateway.to_numpy())
    dlat = lat2 - lat1
    dlon = np.radians(pairs.longitude_gateway.to_numpy() - pairs.longitude_sensor.to_numpy())
    a = np.sin(dlat / 2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon / 2)**2
    dist = 2*6371*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    if (dist <= 0).any():
        raise ValueError('Non-positive link distance')
    pairs['log_distance_km'] = np.log10(dist)
    if not with_features:
        return pairs
    print(f'Computing geographic features for {len(pairs):,} candidate links', flush=True)
    features = path_features(pairs)
    if features.isna().any().any():
        raise ValueError('Candidate links extend outside valid raster coverage')
    pairs[TERRAIN_FEATURES + SURFACE_FEATURES] = features
    return pairs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs()
    rows = []
    for held_out in sorted(pairs.gw.unique()):
        # The set of packets eligible for testing is created solely from other
        # gateways. During training the held-out gateway is also excluded.
        test = pairs.gw == held_out
        held_heard = pairs.loc[test, ['packet_id', 'heard']].set_index('packet_id').heard
        known = pairs.n_heard.to_numpy() - pairs.packet_id.map(held_heard).to_numpy()
        # A training target is eligible only if some *other training* gateway
        # observed that packet, mirroring the conditional test definition.
        other_train = known - pairs.heard.to_numpy()
        for scenario in ['in_range', 'same_day']:
            allowed = pairs[scenario].to_numpy()
            train_mask = (~test.to_numpy()) & allowed & (other_train > 0)
            test_mask = test.to_numpy() & allowed & (known > 0)
            train = pairs.loc[train_mask]
            hold = pairs.loc[test_mask]
            if train.heard.nunique() != 2 or hold.heard.nunique() != 2:
                raise ValueError(f'Both classes required for {held_out}: {scenario}')
            for name, columns in MODELS.items():
                fitted = model()
                fitted.fit(train[columns], train.heard)
                score = fitted.predict_proba(hold[columns])[:, 1]
                rows.append({'held_out_gateway': held_out, 'activity_scenario': scenario, 'model': name,
                             'train_pairs': len(train), 'test_pairs': len(hold),
                             'test_received': int(hold.heard.sum()),
                             'test_inferred_misses': int((1-hold.heard).sum()),
                             'test_reception_fraction': float(hold.heard.mean()),
                             'roc_auc': roc_auc_score(hold.heard, score),
                             'brier': brier_score_loss(hold.heard, score),
                             'log_loss': log_loss(hold.heard, score)})
        print(f'Evaluated held-out gateway {held_out}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'gateway_metrics.csv', index=False, float_format='%.6f')
    summary = df.groupby(['activity_scenario', 'model']).agg(
        gateways=('held_out_gateway','size'), test_pairs=('test_pairs','sum'),
        test_received=('test_received','sum'), test_inferred_misses=('test_inferred_misses','sum'),
        mean_gateway_auc=('roc_auc','mean'), mean_gateway_brier=('brier','mean'),
        min_gateway_auc=('roc_auc','min'), max_gateway_auc=('roc_auc','max')).reset_index()
    summary.to_csv(OUT / 'summary.csv', index=False, float_format='%.6f')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('Results:', OUT)


if __name__ == '__main__':
    main()
