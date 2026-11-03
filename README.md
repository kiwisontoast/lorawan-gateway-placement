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

## Planned pipeline

1. Features per device-gateway pair: distance, free-space path loss, DSM line-of-sight and Fresnel obstruction, buildings crossed, land cover, heights.
2. Coverage model: LightGBM predicts link success (and RSSI) from measured links.
3. Candidate sites: a grid over the city plus rooftops and local high points from the DSM.
4. Coverage matrix: score every candidate against every city cell.
5. Optimizer: solve A and B, and sweep X and k for the trade-off curve.

## Data

| Dataset | Role | Notes |
|---|---|---|
| [mclab-hbrs/lora-bonn](https://github.com/mclab-hbrs/lora-bonn) | Training labels | Garbage-truck GPS sensors in Bonn, Germany: 131,057 receptions of 56,540 packets at 9 gateways (coordinates and antenna heights included), July 2020 to March 2021, plus 6 classical model predictions per reading. Details in [data/README.md](data/README.md). |
| [OpenGeodata.NRW](https://www.opengeodata.nrw.de/) | Features | 1 m DSM (DOM1) and DTM (DGM1), LoD2 3D buildings |
| ESA WorldCover | Features | 10 m land cover |
| [oliveiraleo/LoRa-RSSI-dataset-outdoor](https://github.com/oliveiraleo/LoRa-RSSI-dataset-outdoor) | Transfer test | Campus surveys, Juiz de Fora, Brazil. CC BY-NC-SA 4.0 (non-commercial). |

## Known limits

- Placements are scored in simulation by the validated model; no gateways get installed.
- Trucks measured streets only, so city cells follow the road network.
- New candidate sites, especially high points, are outside the training distribution. Leave-one-gateway-out testing is the check for that.
- Only packets received by at least one gateway appear, so misses are inferred and reception estimates are conditional on what was observed. See [the dataset inspection](data/bonn-inspection.md).

## Motivation

The idea started with planning LoRa sensor networks at remote mine sites for tailings-dam monitoring. A planner that uses geometry alone (line of sight and Fresnel clearance) tends to be optimistic, so this project learns coverage from real measurements instead, in a city where open data exists.
