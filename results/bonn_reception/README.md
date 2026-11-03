# Reception classifier with inferred labels

`python reception_bonn.py` needs the terrain and surface rasters from `fetch_bonn_terrain.py` and `fetch_bonn_surface.py`, and the same Python environment as `terrain_bonn.py`. It builds 508,860 packet-gateway combinations, takes several minutes, and writes the metrics in this folder.

## Question

Given that some other gateway heard a packet, can the model tell which of the remaining gateways heard it too? The data can't answer the broader question of how likely an arbitrary transmission is to arrive, because packets nobody heard are missing.

A packet is a group of readings with the same date, packet number, latitude and longitude, as in the authors' notebook. A label of 1 is an observed reception and 0 is an inferred miss: the gateway didn't log a packet it could have heard. The grouping may merge transmissions and gateway uptime is unknown, so the 0s are provisional.

Two assumptions about when a gateway was active are compared:

- `in_range`: every day between its first and last reception, inclusive.
- `same_day`: any date on which it received at least one packet.

Both can include outages and backhaul problems, and `same_day` relies on the target gateway itself, so neither is independent evidence of uptime.

## Validation design

Each of nine folds holds out one gateway. Its test set contains only packets heard by at least one other gateway, chosen without using the held-out gateway's own receptions, so a positive label is never what puts a packet in the test set. Training uses the other eight gateways, and a training link counts only if a different training gateway heard the packet. Both sides use the same activity assumption.

Training and test can still share packets and nearby road positions through other gateways, so this tests unseen gateway sites, not unseen places or times. The terrain and surface features come from the 20 m rasters described in `../bonn_terrain/README.md`.

Two histogram gradient-boosting classifiers with fixed settings are compared: distance and antenna height alone, and the same plus terrain and surface features. `gateway_metrics.csv` has AUC, Brier score, log loss, class balance and counts for each gateway, scenario and model; `summary.csv` averages gateways equally rather than by packet count. AUC measures how well the model ranks heard above unheard links (0.5 is chance, 1.0 is perfect). Brier and log loss are scored against the same uncertain labels, so they don't show real-world calibration.

## Results

| Activity assumption | Model | Mean gateway AUC | Mean gateway Brier |
|---|---|---:|---:|
| First to last observed date | Distance + height | 0.822 | 0.205 |
| First to last observed date | Terrain + surface | 0.878 | 0.145 |
| Same-day activity | Distance + height | 0.820 | 0.205 |
| Same-day activity | Terrain + surface | 0.879 | 0.145 |

Under the same-day assumption, the terrain-and-surface model's AUC ranged from 0.714 to 0.956 across gateways. The two assumptions agree closely, but neither sees real outages, so that agreement doesn't show the results are safe from them. Counts in the summary are packet-gateway pairs; one packet contributes to several gateways.

No classifier is saved from this experiment, and the scores here are not coverage probabilities.
