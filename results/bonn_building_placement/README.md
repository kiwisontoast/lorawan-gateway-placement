# Placement on mapped buildings

The second placement experiment moves candidate sites onto OpenStreetMap buildings and adds an exact integer program. As in the greedy simulation, scores come from a classifier trained on inferred labels and the existing Bonn gateways, so the comparison with the existing network is model-based, not independent.

## Building data and candidates

`fetch_bonn_buildings.py` queries OpenStreetMap for building ways and multipolygon relations inside the measurement bounding box; the download and query metadata are in `data/osm/`. It returned 185,670 building records with an OSM timestamp of 2026-10-31, five years after the radio campaign. © OpenStreetMap contributors, [Open Database License](https://www.openstreetmap.org/copyright).

`placement_buildings_bonn.py` drops buildings tagged as residential, houses, apartments, garages, sheds, and other small outbuildings or construction. Generic `building=yes` stays eligible, though that tag says nothing about use, ownership or roof access. Overpass `out center` gives the centre of each object's bounding box, which can fall outside the footprint. OSM height and level tags are recorded but not used.

Antenna height comes from the 20 m NRW rasters: surface minus ground at the mapped centre gives a rough building height, candidates between 10 and 28 m get an antenna 2 m above that (12-30 m), and the two centres nearest the demand centroid in each occupied 2 km tile are kept. A 20 m pixel can pick up a neighbouring building or a tree. Only three of the 85 selected buildings have an OSM `height` tag to compare with.

That gives 85 building candidates across all 43 occupied 2 km tiles, plus the nine existing gateways at their recorded heights. Demand is the same 412 equal-weight 500 m cells as the greedy simulation. 64 of the 38,728 cell-site links are shorter than the 218 m minimum seen in training; their distance input is clipped to 218 m and they remain extrapolations.

## Result

Same-day activity assumption, unadjusted building heights, score threshold 0.5:

| Layout | Cells above the score threshold |
|---|---:|
| All nine existing sites (combined baseline) | 386 / 412 (93.7%) |
| Greedy selection, four sites | 394 / 412 (95.6%) |
| Greedy selection, nine sites | 410 / 412 (99.5%) |

The four-site solution keeps three existing gateways and adds one `building=yes` centre. One of the three is labelled "old" in the source metadata, and the nine existing gateways were never all active together.

`optimize_building_sites.py` solves two binary integer programs with SciPy/HiGHS on this one scenario matrix. Four is the fewest of the 94 candidates that matches the nine-site baseline's 386 cells, and 394 is the most cells any four sites can cover. Both solves reported optimal status with zero MIP gap, and coverage was recomputed from the chosen sites as a check. The optimum holds for this score matrix only; it doesn't show four real gateways would be enough.

## Sensitivity and limits

The greedy sweep varies the activity assumption (`in_range`, `same_day`), the threshold (0.3, 0.5, 0.7) and the building antenna height (-2, 0, +2 m), 18 combinations in all. Greedy matched or beat the corresponding nine-site baseline with three or four sites in every one, but only the main scenario was solved exactly.

The rasters are coarse, mix buildings with vegetation, and have no Fresnel or diffraction term, so a path crossing the surface is only a proxy for obstruction. The -2 m offset puts some antennas below the 12 m minimum of the training gateways. The optimizer may also favour existing sites the model was trained on, or buildings whose height is overestimated. A real plan would need roof access and mounting data, measured misses, data from matching dates and an independent check.

## Reproduction and files

1. `python fetch_bonn_buildings.py` downloads current OSM data, which changes over time; keep the saved JSON and metadata for exact reruns.
2. `python placement_buildings_bonn.py` writes the candidates, scenario metrics, greedy picks, map, curve and `illustrative_coverage_matrix.npz`. It needs the terrain and surface rasters (see `../bonn_terrain/README.md`).
3. `python optimize_building_sites.py` solves the minimum-site and maximum-coverage programs and writes `integer_program_results.json` and `integer_program_sites.csv`. Each solve has a 60-second limit, so check the status and MIP gap on new runs.
4. `python -m unittest test_integer_placement` checks both formulations on a tiny matrix with a known answer; `test_placement` and `test_terrain_features` still apply.

Coordinates in the CSVs are EPSG:25832 and WGS84. The map compares the nine existing sites with the nine-site greedy selection, not the four-site optimum.
