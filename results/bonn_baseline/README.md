# Received-power baseline

`python baseline_bonn.py` (NumPy, pandas, Matplotlib) reads the Bonn survey CSVs and writes the tables and figures in this folder.

## Models and evaluation

The target is the Bonn authors' received packet power (RPP): RSSI + min(SNR, 0), in dBm. Other radios may report signal strength differently.

- Free space: `15 - [32.44 + 20 log10(distance_km) + 20 log10(868)]` dBm, using the authors' 15 dB combined link budget, 868 MHz and great-circle distance. It assumes a clear path and has nothing fitted.
- Log-distance: `a + b log10(distance_km)` dBm, fitted by ordinary least squares to individual received packets from the training gateways. It ignores terrain, buildings and antenna height.

Each of nine folds holds out every reading from one gateway and fits the log-distance line on the other eight. The overall metrics pool the held-out predictions, so busier gateways count for more; the per-gateway table shows the spread. MAE and RMSE are in dB. Bias is prediction minus observation, so a positive bias means the predictions are too strong.

| Model | Pooled held-out MAE | RMSE |
|---|---:|---:|
| Free space | 39.71 dB | 40.91 dB |
| Log-distance | 8.44 dB | 10.62 dB |

`full_data_parameters.csv` is a separate fit on all nine gateways, for plotting. It was not used for the held-out numbers, and because it fits individual receptions rather than distance-bin medians it need not match the authors' published coefficients.

## Files

- `overall_metrics.csv`: pooled metrics over all nine held-out gateways.
- `gateway_metrics.csv`: each gateway's metrics and the line fitted without it.
- `full_data_parameters.csv`: the all-data fit.
- `predicted_vs_observed.png`: density plots; the red line is perfect agreement.
- `error_by_distance.png`: median signed error by distance bin.
- `held_out_predictions.csv` (written by the script, not committed): every reception with its distance and predictions. Gateway IDs are strings with leading zeros.

## Limits

The data only contains packets that arrived, so a model scored on them can be optimistic about weak links, and nothing here measures delivery probability or coverage. Readings are spatially correlated and come from just nine gateway locations. The link budget and frequency are the authors' assumed constants; comparing with other hardware needs that hardware's own measured settings.
