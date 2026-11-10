# Results

## Current pipeline (v2)

| Folder | Written by | Contents |
|---|---|---|
| `v2_models/` | `models_v2.py` | Feature ablation, reception and power metrics per held-out gateway, the new-site footprint test, calibration plot |
| `v2_placement/` | `placement_v2.py`, `robustness_v2.py` | Candidate sites, road cells, the score matrix, optimal designs for r = 1 and 2, and the error-adjusted coverage curves |
| `v2_sensitivity/` | `placement_v2.py --out`, `robustness_v2.py --out` | One subfolder per scenario (25 m masts, forested high points removed, masts that clear the canopy); `summary.csv` collects them. `baseline_check/` reruns the defaults to confirm they reproduce `v2_placement/`. |
| `v2_spatial_cv/` | `spatial_cv_v2.py` | Validation with 2 km spatial blocks, and the error adjustment rerun with those error profiles |
| `site_audit/` | `site_audit_v2.py` | The chosen sites' surroundings on the surface model, orthophoto contact sheets, and a verdict per site in `verdicts.csv` |
| `figures/` | `figures_v2.py` | Maps of the existing and proposed networks and summary charts |

## First prototype

These came first and are kept for comparison. They use 20 m rasters, a scikit-learn model and OpenStreetMap buildings as candidates; each folder's README explains its experiment.

| Folder | Experiment |
|---|---|
| `bonn_baseline/` | Free-space and log-distance baselines for received power |
| `bonn_geoai/` | First boosted-tree power model |
| `bonn_terrain/` | Terrain and surface path features |
| `bonn_reception/` | Reception classifier with inferred misses |
| `bonn_validation/` | Spatial quadrant and time holdouts |
| `bonn_placement/` | Greedy placement on hypothetical road-side sites |
| `bonn_building_placement/` | Placement on OSM buildings with an exact integer program |
| `bonn_period_study/` | The same over the period when eight gateways ran together |
