# Greedy placement simulation

A first placement experiment: score hypothetical sites with the inferred-label classifier, then pick sites greedily. The classifier's score is conditional on another gateway having heard the packet, and it is applied here to sites and cells it never saw, so the results show how the method behaves rather than how a real network would perform. The same model was trained on the existing gateways it is compared against.

## Inputs and method

`python placement_bonn.py` rebuilds the outputs from the Bonn survey and the terrain and surface rasters (setup in `../bonn_terrain/README.md`).

Demand is 412 cells of 500 m with equal weight. Each is represented by the real truck position nearest the centroid of the cell's sampled positions, so the cells follow the trucks' routes rather than the full road network, population or area. Repeated readings don't add weight.

There are 52 candidate sites: the nine existing gateways at their recorded antenna heights and 43 hypothetical sites, one per occupied 2 km tile, at an observed road position near the tile centroid. The hypothetical sites are just coordinates, not checked rooftops or mounting points. Some of the nine existing gateways were never active at the same time, so the "existing network" is a combined baseline.

The classifier is refitted on all inferred packet-gateway labels under each activity assumption (`same_day` and `in_range`), using only packets another gateway heard, and the same geographic features are computed for every cell-site pair. Scores are thresholded at 0.3, 0.5 or 0.7 to make a binary coverage matrix; the thresholds are cutoffs, not 30%, 50% or 70% delivery rates. A greedy pass then adds whichever site covers the most uncovered cells, up to nine sites, with existing gateways eligible and ties broken by site order. There is no exact optimizer in this experiment.

Hypothetical sites are tried at 12, 16 and 24 m above ground, with existing heights fixed. 49 of the 21,424 cell-site pairs are closer than 218 m, about the shortest measured reception; their distance input is clipped to 218 m while the terrain features use the real geometry, so those predictions are extrapolations. Every path was checked to lie inside the rasters.

## Results

With the `same_day` assumption, 16 m sites and a 0.5 threshold, the combined existing sites cover 93.7% of cells under the model's score rule. Greedy selection covers 92.5% with three sites and 95.1% with four, so four sites match the modelled existing network in this scenario, and nine reach 99.5%.

Across all 18 combinations of activity assumption, height and threshold, greedy selection matched or beat the corresponding existing baseline with three or four sites. Which four sites it picks changes between scenarios. The sweep doesn't touch the bigger uncertainties: the labels, how well the scores are calibrated, whether the sites are usable, how the cells were sampled, other model choices, roof heights and raster resolution.

## Outputs

- `demand_cells.csv`, `candidate_sites.csv`: input locations in EPSG:25832 and WGS84. Hypothetical heights vary by scenario, so they are blank in the site file.
- `scenario_results.csv`: modelled coverage for all 18 scenarios at budgets 1-9. `selected_k` can be below the budget when no site adds coverage. The existing-network fraction always uses all nine sites.
- `selected_sites.csv`: ranked sites for each scenario and budget.
- `coverage_curve.png`, `placement_map.png`: the same-day, 16 m, 0.5 scenario. The map shows the nine-site greedy selection, not the four-site match.
- `base_coverage_flags.csv`: cell-level results behind the map.

Tests: `python -m unittest test_placement test_terrain_features` checks the greedy step and the obstruction features on synthetic inputs.
