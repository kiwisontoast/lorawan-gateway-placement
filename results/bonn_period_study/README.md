# Placement over a period when eight gateways ran together

The earlier placement experiments compared against all nine gateways, which were never active at the same time. This study restricts everything to one period with a consistent set of eight. Scores still predict an inferred reception label conditional on another gateway hearing the packet, and the solver's guarantees apply to the thresholded score matrix, not to a physical network.

## Period and existing network

The period is 2020-11-10 to 2021-01-26 inclusive. In it, eight gateways (`015c`, `0172`, `01c6`, `06df`, `08d5`, `08fc`, `090e`, `0917`) each logged at least one reception on every one of the 56 survey dates, for 54,395 reception rows. `244d` has no receptions in the period and is left out. Daily receptions make this a fairer eight-gateway baseline than the earlier combined nine, though they don't prove the gateways were up for every transmission.

Demand is 392 equal-weight 500 m cells with measured truck positions in the period, each represented by one measured position. These follow the trucks' routes, not all roads, people or land area, and some fall outside Bonn's city limits. The eight existing gateways, at their positions and heights from `survey/gateways.csv`, are also candidates.

## Candidate buildings

We fetched OSM footprints for the 82 building ways chosen in the earlier building experiment (that selection was made over the whole survey area). The three OSM relations in that selection are left out rather than treated as simple polygons. A way is kept if it has a valid polygon, a point confirmed inside the footprint, at least 100 m² of area, and a 20 m surface-minus-ground height between 10 and 28 m; 51 remain. Each gets an antenna 2 m above that height, giving 59 candidates with the eight existing sites. The size and height filters are our assumptions, and other buildings weren't searched.

The footprints are current OSM data (timestamp 2026-10-31), not a 2020-21 inventory, and a point inside a footprint says nothing about roof access, power, ownership or structure. `building=yes` can include homes. Only three candidates have an OSM height tag; the model uses the raster height, and a 20 m pixel can catch a neighbouring tree or building. The -2 m height offset can drop below the training gateways' range. 36 cell-site distances are clipped to 218 m and remain extrapolations.

Data © OpenStreetMap contributors, [ODbL](https://www.openstreetmap.org/copyright). NRW DGM/DOM sources and dates are in `../bonn_terrain/README.md` and `data/terrain/`.

## Classifier for the period

The reception classifier is retrained on period observations from the eight gateways, with the same packet keys and conditional labels as `../bonn_reception/README.md`, using distance, antenna height, terrain and surface features. The two earlier activity assumptions select exactly the same rows here, because every gateway logged something on every survey day, so they no longer give two scenarios.

Leave-one-gateway-out AUC ranges from 0.723 to 0.958, mean 0.893 over gateways. That is a ranking score on uncertain labels, not 89.3% accuracy. Copies of the same packet at other gateways, and nearby positions, can appear on both sides of a split. The classifier used for placement is then refitted on all eight gateways, so the baseline comparison is model-based.

## Exact optimization

The main scenario uses a score cutoff of 0.5 and the raster building heights; a cell counts as covered by a site when the score reaches the cutoff. Three questions get three different answers:

| Objective | Sites | Modelled covered cells | Change from existing |
|---|---:|---:|---|
| Existing eight gateways | 8 | 369 / 392 (94.1%) | reference |
| Fewest sites keeping every covered cell | 4 | 370 / 392 (94.4%) | none lost, one gained |
| Most coverage from four sites, keeping every covered cell | 4 | 371 / 392 (94.6%) | none lost, two gained |
| Most coverage from four sites, no preservation | 4 | 380 / 392 (96.9%) | 14 gained, three lost |

The unconstrained four-site solution covers more cells in total but drops three that the existing network covers, for a net gain of 11. The minimum-site solver isn't asked to add cells, so the coverage-maximizing version with preservation can do better. All of these solves reported HiGHS optimal status with zero MIP gap. The optimum is for this score matrix, candidate set and period; it doesn't show four real gateways could cover the city.

For the main scenario, maximum coverage was solved for every budget from 1 to 8. Minimum-site programs were also solved for nine combinations of threshold (0.3, 0.5, 0.7) and new-building antenna offset (-2, 0, +2 m), each against its own existing baseline, and all nine needed four sites. Those nine compare cell counts; keeping every specific cell was solved for the main scenario only.

## Files and reproduction

- `study_metadata.json`, `candidate_sites.csv`, `demand_cells.csv`: period, sites, assumptions and the footprint checks.
- `gateway_validation.csv`: held-out-gateway classifier results for the period.
- `scenario_results.csv`, `integer_results.csv`, `selected_sites.csv`: greedy sweep and solver results; check `status`, `optimal` and `mip_gap` for each run.
- `preserve_results.json`, `preserve_sites.csv`: the cell-preserving solves, rechecked against the binary matrix.
- `coverage_curve.png`, `placement_map.png`, `coverage_changes.png`: the budget curve, existing vs. best-four map, and cells gained and lost under both four-site objectives. `cell_coverage_changes.csv` and `coverage_change_summary.csv` hold the mapped values.

Run `python fetch_candidate_footprints.py` (current OSM, so it changes over time), then `python study_period_bonn.py`, `python preserve_baseline_bonn.py` and `python render_period_comparison.py`. Keep the downloaded OSM JSON and rasters for exact reruns. The scripts use pandas, NumPy, SciPy, Matplotlib, scikit-learn, pyproj, rasterio and Shapely, and also look in a local `.geo_deps` folder. Tests: `python -m unittest test_integer_placement test_preserve_baseline`; the second checks that the same cells are kept, not just the same number.

## Future work

We didn't contact the dataset's authors, so the original transmission logs, device IDs and continuous uptime records stay as stated limitations. Beyond that: check roof access and heights at candidate sites, use building and elevation data from the survey years, widen the candidate set, add uptime data or controlled transmissions, model capacity and interference, and test predictions in the field.
