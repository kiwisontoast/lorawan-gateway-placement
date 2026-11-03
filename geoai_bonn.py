"""Compare held-out-gateway received-power regressors using Bonn receptions.

Run: python geoai_bonn.py. Requires NumPy, pandas, scikit-learn and Matplotlib.
Models predict received packet power for observed receptions, not coverage.
"""
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from baseline_bonn import load_data, fit_log_distance, summarize

OUT = Path(__file__).resolve().parent / 'results' / 'bonn_geoai'
FEATURE_SETS = {
    'boosted_distance': ['log_distance_km'],
    'boosted_distance_height': ['log_distance_km', 'antenna_height'],
}


def new_model():
    # Fixed in advance: no tuning on held-out gateway observations.
    return HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15,
                                         min_samples_leaf=100, learning_rate=0.07,
                                         l2_regularization=1.0, random_state=42,
                                         early_stopping=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    data['log_distance_km'] = np.log10(data.distance_km)
    if data.antenna_height.isna().any():
        raise ValueError('Gateway antenna heights are required')
    predictions = data[['gw', 'transceived_at', 'pkt_number', 'latitude_sensor',
                        'longitude_sensor', 'distance_km', 'antenna_height',
                        'observed_rpp_dbm']].copy()
    names = ['log_distance'] + list(FEATURE_SETS)
    for name in names:
        predictions[name] = np.nan
    fold_rows = []
    for gateway in sorted(data.gw.unique()):
        test = data.gw == gateway
        train = ~test
        intercept, slope = fit_log_distance(data.loc[train])
        predictions.loc[test, 'log_distance'] = intercept + slope * data.loc[test, 'log_distance_km']
        for name, features in FEATURE_SETS.items():
            model = new_model()
            model.fit(data.loc[train, features], data.loc[train, 'observed_rpp_dbm'])
            predictions.loc[test, name] = model.predict(data.loc[test, features])
        for name in names:
            fold_rows.append({'held_out_gateway': gateway, 'model': name,
                              'held_out_height_m': float(data.loc[test, 'antenna_height'].iloc[0]),
                              **summarize(predictions.loc[test, 'observed_rpp_dbm'],
                                          predictions.loc[test, name])})
        print(f'Evaluated held-out gateway {gateway}', flush=True)
    folds = pd.DataFrame(fold_rows)
    overall = pd.DataFrame([{'model': name, **summarize(predictions.observed_rpp_dbm, predictions[name])}
                            for name in names])
    predictions.to_csv(OUT / 'held_out_predictions.csv', index=False, float_format='%.6f')
    folds.to_csv(OUT / 'gateway_metrics.csv', index=False, float_format='%.6f')
    overall.to_csv(OUT / 'overall_metrics.csv', index=False, float_format='%.6f')
    full_model = new_model()
    full_model.fit(data[FEATURE_SETS['boosted_distance_height']], data.observed_rpp_dbm)
    joblib.dump({'model': full_model, 'features': FEATURE_SETS['boosted_distance_height'],
                 'target': 'rssi + min(snr, 0), dBm',
                 'scikit_learn_version': __import__('sklearn').__version__}, OUT / 'model.joblib')
    pivot = folds.pivot(index='held_out_gateway', columns='model', values='mae_db')
    ax = pivot[names].plot.bar(figsize=(10, 5), rot=0)
    ax.set(xlabel='Held-out gateway', ylabel='Mean absolute error (dB)',
           title='Received-power prediction error by held-out gateway')
    ax.legend(['Log-distance', 'Boosted: distance', 'Boosted: distance + height'])
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / 'gateway_mae.png', dpi=160)
    plt.close(ax.figure)
    print('\nPooled held-out results:')
    print(overall.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    print('\nGateway metrics:')
    print(folds.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    print(f'\nOutputs: {OUT}')


if __name__ == '__main__':
    main()
