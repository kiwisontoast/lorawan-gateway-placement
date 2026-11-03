# Spatial and temporal holdouts

`python validation_bonn.py` compares two received-power regressions, log-distance and boosted trees with terrain and surface features, on held-out regions and a held-out time period. The target is the Bonn authors' received-power convention, every record is a successful reception, and the terrain-feature rows are checked against the survey before fitting.

The spatial tests hold out one quadrant at a time and train on the other three. Quadrant boundaries are the median projected easting and northing of the unique sensor positions. The temporal test trains on dates before 2021-01-08 and tests on the rest. Boundaries, gateway counts and full results are in `split_definition.txt` and `split_metrics.csv`.

| Test | Log-distance MAE | Terrain/surface MAE |
|---|---:|---:|
| Spatial quadrant 0 | 7.24 dB | 5.59 dB |
| Spatial quadrant 1 | 8.07 dB | 6.16 dB |
| Spatial quadrant 2 | 8.29 dB | 6.18 dB |
| Spatial quadrant 3 | 7.33 dB | 5.78 dB |
| From 2021-01-08 | 7.79 dB | 5.27 dB |

Models are refitted for each split, and metrics weight receptions equally within the split.

These tests answer a different question from the held-out-gateway experiments: every gateway appears in both training and test, so they check new places and later dates, not new sites. Identical positions stay on one side, but there is no buffer, so close neighbours can sit just across a quadrant boundary. The time test is a single split, and the set of active gateways changes over time. None of this measures missed packets or delivery probability. If these numbers are used to choose features, a final comparison would need a test set that wasn't used for the choice.
