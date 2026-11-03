# How few gateways does a city need?

Dev Shroff and Enoch Zhu.

## Research question

Can a gradient-boosting coverage model, trained on measured LoRaWAN signal strength with terrain, building and land-cover features, plan gateway placements that reach a target coverage with fewer gateways than a city's existing deployment?

## Two problems, one model

- A, minimum cost: the fewest gateways covering at least X% of city cells (partial set cover).
- B, maximum coverage: the best coverage from k gateways (budgeted maximum coverage).
- Adjustable inputs: coverage target X, redundancy r (each cell heard by ≥ r gateways), budget k. Every gateway costs the same, so cost = gateway count.

Integer programs, with `a_ij = 1` when the model says a device in cell i reaches a gateway at candidate site j:

```
A:  min Σ x_j          s.t.  r·y_i ≤ Σ_j a_ij x_j,  Σ w_i y_i ≥ X·Σ w_i
B:  max Σ w_i y_i      s.t.  r·y_i ≤ Σ_j a_ij x_j,  Σ x_j ≤ k
    x_j, y_i ∈ {0,1}
```

`scipy.optimize.milp` (HiGHS) solves these exactly at city scale. A greedy pass (most new coverage per gateway) is the explainable baseline.

## Pipeline (v2)

1. Features per device-gateway pair (`features_v2.py`): distance, antenna height, terrain and surface obstruction on a 2 m grid built from 1 m NRW DGM1/DOM1, first-Fresnel-zone clearance and knife-edge diffraction, LoD2 buildings crossed and blocking, WorldCover 2020 foliage depth and land-cover fractions.
2. Coverage model (`models_v2.py`): LightGBM with monotone physics constraints predicts reception (and received power), trained on observed receptions plus inferred misses, and tested by leaving one gateway out, including a new-site footprint test.
3. Candidate sites (`placement_v2.py`): the eight existing gateways, the tallest LoD2 roof per 1 km tile, the highest bare-ground point per 2 km tile, and a 1 km grid of masts.
4. Coverage matrix: every candidate scored against 392 measured road cells (2020-11-10 to 2021-01-26).
5. Optimizer (`optimize_v2.py`): integer programs A and B with redundancy r = 1 or 2, compared with the existing network, its best subset and a greedy baseline. `robustness_v2.py` applies the model's measured new-site error.

## Data

| Dataset | Role | Notes |
|---|---|---|
| [mclab-hbrs/lora-bonn](https://github.com/mclab-hbrs/lora-bonn) | Training labels | Garbage-truck GPS sensors in Bonn, Germany: 131,057 receptions of 56,540 packets at 9 gateways (coordinates and antenna heights included), July 2020 to March 2021, plus 6 classical model predictions per reading. Details in [data/README.md](data/README.md). |
| [OpenGeodata.NRW](https://www.opengeodata.nrw.de/) | Features | 1 m DSM (DOM1) and DTM (DGM1), flown 2021-22, modelled at 2 m; LoD2 3D buildings (2025-26) |
| ESA WorldCover 2020 | Features | 10 m land cover, matched to the survey year |
| [oliveiraleo/LoRa-RSSI-dataset-outdoor](https://github.com/oliveiraleo/LoRa-RSSI-dataset-outdoor) | Transfer test (deferred) | Campus surveys, Juiz de Fora, Brazil. CC BY-NC-SA 4.0 (non-commercial). |

## Known limits

- Placements are scored in simulation by the validated model; no gateways get installed.
- Trucks measured streets only, so city cells follow the road network.
- New candidate sites, especially high points, are outside the training distribution. Leave-one-gateway-out testing is the check for that.
- Only packets received by at least one gateway appear, so misses are inferred and reception estimates are conditional on what was observed. Gateway uptime and packet identity are unavailable; see [the dataset inspection](data/bonn-inspection.md).

## Motivation

The idea started with planning LoRa sensor networks at remote mine sites for tailings-dam monitoring. A planner that uses geometry alone (line of sight and Fresnel clearance) tends to be optimistic, so this project learns coverage from real measurements instead, in a city where open data exists.

## Files

| Path | What |
|---|---|
| `features_v2.py`, `models_v2.py`, `optimize_v2.py`, `placement_v2.py`, `robustness_v2.py` | v2 pipeline; results in `results/v2_models/` and `results/v2_placement/` |
| `fetch_nrw_1m.py`, `fetch_lod2_buildings.py`, `fetch_worldcover.py` | v2 data downloads (about 6 GB transferred, 1.6 GB kept; git-ignored) |
| `test_features_v2.py`, `test_optimize_v2.py` | Synthetic physics checks and known-answer optimizer checks |
| `requirements-bonn.txt` | Package versions used for the Bonn pipeline |
| `*_bonn.py` and `results/bonn_*` | First prototype on 20 m rasters, kept as a baseline |
| `data/` | Dataset notes and clone commands in [data/README.md](data/README.md) |
