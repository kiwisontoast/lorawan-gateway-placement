"""Evaluate physics-based received packet power baselines on Bonn receptions.

Run from the project root: python baseline_bonn.py
Requires numpy, pandas and matplotlib. Writes outputs to results/bonn_baseline/.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SURVEY = ROOT / 'data' / 'lora-bonn' / 'survey'
OUT = ROOT / 'results' / 'bonn_baseline'
FREQUENCY_MHZ = 868.0
LINK_BUDGET_DBM = 15.0  # Authors' notebook: 14 dBm + 3 dBi - 2 dB.


def load_data():
    samples = pd.read_csv(SURVEY / 'samples.csv', dtype={'gw': str})
    gateways = pd.read_csv(SURVEY / 'gateways.csv', dtype={'id': str})
    data = samples.merge(gateways, left_on='gw', right_on='id',
                         suffixes=('_sensor', '_gateway'), validate='many_to_one', how='left')
    needed = ['latitude_sensor', 'longitude_sensor', 'latitude_gateway',
              'longitude_gateway', 'rssi', 'snr']
    if data[needed].isna().any().any():
        raise ValueError('Missing measurements or unmatched gateway coordinates')
    lat1 = np.radians(data.latitude_sensor.to_numpy())
    lat2 = np.radians(data.latitude_gateway.to_numpy())
    dlat = lat2 - lat1
    dlon = np.radians(data.longitude_gateway.to_numpy() - data.longitude_sensor.to_numpy())
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    data['distance_km'] = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    if (data.distance_km <= 0).any():
        raise ValueError('A non-positive distance cannot be used with log-distance models')
    data['observed_rpp_dbm'] = data.rssi + np.minimum(data.snr, 0)
    data['free_space_dbm'] = LINK_BUDGET_DBM - (
        32.44 + 20 * np.log10(data.distance_km) + 20 * np.log10(FREQUENCY_MHZ))
    return data


def fit_log_distance(train):
    """Fit RPP = intercept + slope * log10(distance in km) by least squares."""
    x = np.log10(train.distance_km.to_numpy())
    matrix = np.column_stack([np.ones(len(x)), x])
    coeff, _, rank, _ = np.linalg.lstsq(matrix, train.observed_rpp_dbm.to_numpy(), rcond=None)
    if rank != 2:
        raise ValueError('Training distances do not support fitting a log-distance model')
    return coeff


def summarize(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    return {'n': len(error), 'mae_db': np.mean(np.abs(error)),
            'rmse_db': np.sqrt(np.mean(error**2)), 'bias_db': np.mean(error)}


def evaluate(data):
    predictions = data[['gw', 'transceived_at', 'pkt_number', 'latitude_sensor',
                        'longitude_sensor', 'distance_km', 'rssi', 'snr',
                        'observed_rpp_dbm', 'free_space_dbm']].copy()
    predictions['log_distance_dbm'] = np.nan
    folds = []
    for gateway in sorted(data.gw.unique()):
        held_out = data.gw == gateway
        intercept, slope = fit_log_distance(data.loc[~held_out])
        predictions.loc[held_out, 'log_distance_dbm'] = (
            intercept + slope * np.log10(data.loc[held_out, 'distance_km']))
        for model, column in [('free_space', 'free_space_dbm'),
                              ('log_distance', 'log_distance_dbm')]:
            metrics = summarize(predictions.loc[held_out, 'observed_rpp_dbm'],
                                predictions.loc[held_out, column])
            folds.append({'held_out_gateway': gateway, 'model': model, **metrics,
                          'training_intercept_dbm_at_1km': intercept if model == 'log_distance' else np.nan,
                          'training_path_loss_exponent': -slope / 10 if model == 'log_distance' else np.nan})
    overall = []
    for model, column in [('free_space', 'free_space_dbm'), ('log_distance', 'log_distance_dbm')]:
        overall.append({'model': model, **summarize(predictions.observed_rpp_dbm, predictions[column])})
    return predictions, pd.DataFrame(folds), pd.DataFrame(overall)


def make_plot(predictions):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, (label, col) in zip(axes, [('Free space', 'free_space_dbm'),
                                      ('Log-distance (held-out gateway)', 'log_distance_dbm')]):
        ax.hexbin(predictions.observed_rpp_dbm, predictions[col], gridsize=60,
                  mincnt=1, bins='log', cmap='viridis')
        lo = min(predictions.observed_rpp_dbm.min(), predictions[col].min())
        hi = max(predictions.observed_rpp_dbm.max(), predictions[col].max())
        ax.plot([lo, hi], [lo, hi], color='red', linewidth=1, label='Perfect prediction')
        ax.set(xlabel='Observed received packet power (dBm)',
               ylabel='Predicted received packet power (dBm)', title=label)
        ax.legend(loc='upper left')
    fig.savefig(OUT / 'predicted_vs_observed.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    bins = np.linspace(0, predictions.distance_km.max(), 27)
    for label, col in [('Free space', 'free_space_dbm'), ('Log-distance', 'log_distance_dbm')]:
        work = predictions[['distance_km', 'observed_rpp_dbm', col]].copy()
        work['bin'] = pd.cut(work.distance_km, bins, include_lowest=True)
        work['error'] = work[col] - work.observed_rpp_dbm
        summary = work.groupby('bin', observed=True).agg(distance=('distance_km', 'median'), error=('error', 'median'))
        ax.plot(summary.distance, summary.error, marker='o', markersize=3, label=label)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set(xlabel='Distance (km)', ylabel='Median prediction error (dB)', title='Prediction bias by distance')
    ax.legend()
    fig.savefig(OUT / 'error_by_distance.png', dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    predictions, folds, overall = evaluate(data)
    predictions.to_csv(OUT / 'held_out_predictions.csv', index=False, float_format='%.6f')
    folds.to_csv(OUT / 'gateway_metrics.csv', index=False, float_format='%.6f')
    overall.to_csv(OUT / 'overall_metrics.csv', index=False, float_format='%.6f')
    intercept, slope = fit_log_distance(data)
    pd.DataFrame([{'intercept_dbm_at_1km': intercept, 'slope_db_per_log10_km': slope,
                   'path_loss_exponent': -slope / 10, 'n_receptions': len(data)}]).to_csv(
                       OUT / 'full_data_parameters.csv', index=False, float_format='%.6f')
    make_plot(predictions)
    print('Overall metrics for held-out gateways (positive bias = predicted stronger than observed):')
    print(overall.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    print('\nMetrics by gateway:')
    print(folds[['held_out_gateway', 'model', 'n', 'mae_db', 'rmse_db', 'bias_db']].to_string(
        index=False, float_format=lambda v: f'{v:.2f}'))
    print(f'\nFull-data model: RPP = {intercept:.3f} + {slope:.3f} log10(distance_km) dBm')
    print(f'Outputs: {OUT}')


if __name__ == '__main__':
    main()
