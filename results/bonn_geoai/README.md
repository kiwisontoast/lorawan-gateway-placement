# First machine-learning model

`python geoai_bonn.py` (NumPy, pandas, Matplotlib, scikit-learn, joblib) loads the survey through `baseline_bonn.py` and writes the outputs here.

## Question and method

Can a gradient-boosted tree predict received packet power at an unseen gateway better than the fitted log-distance line? The target is the authors' RSSI + min(SNR, 0) in dBm, and every observation is a successful reception.

Each of nine folds holds out one gateway and refits the log-distance baseline on the other eight. Two `HistGradientBoostingRegressor` models with fixed settings use (1) log-distance and (2) log-distance plus antenna height. Nothing is tuned on the held-out gateway. Pooled metrics weight gateways by record count, and bias is prediction minus observation.

| Model | Pooled held-out MAE |
|---|---:|
| Fitted log-distance | 8.44 dB |
| Boosted trees, distance | 8.39 dB |
| Boosted trees, distance and height | 7.57 dB |

Adding height helps overall, but the height model is worse than log-distance at four of the nine gateways. There are only nine antenna heights to learn from, and the 55 m site is taller than every other one, which trees can't extrapolate to. Nearby positions and copies of the same packet can also land on both sides of a split, so holding out a gateway tests an unseen site, not independent samples.

## Outputs

- `overall_metrics.csv`, `gateway_metrics.csv`: MAE, RMSE and signed bias in dB.
- `gateway_mae.png`: per-gateway error.
- `held_out_predictions.csv` and `model.joblib` are written by the script but not committed. The saved model is the distance-and-height version refitted on all nine gateways after evaluation; it plays no part in the held-out metrics.

## Limits

This regression only sees received packets, so it can't estimate delivery probability or coverage, and it uses no terrain, building or land-cover data. `../bonn_terrain/README.md` adds terrain and surface features. Small differences in pooled error shouldn't be read as real improvements without checking all nine gateways.
