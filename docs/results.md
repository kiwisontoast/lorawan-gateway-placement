# Methods and results

Technical notes for the v2 pipeline: data and labels, the coverage model, placement, the error adjustment, robustness checks and limitations. The README has the short version and the run instructions.

A gateway is the receiver, usually on a roof or mast, that collects messages from low-power sensors; LoRaWAN is the long-range radio standard they use.

## 1. Summary

The question was whether a machine-learned coverage model could plan a LoRaWAN network that covers a city with fewer gateways than the one the city built. The model is LightGBM, trained on 131,057 receptions in Bonn, Germany, with features from 1 m terrain and surface models, 3D buildings, 2020 land cover, and Fresnel-zone and diffraction physics. An integer program chooses sites for single or double coverage.

- Single coverage: taken at face value, the model says three new sites on Bonn's high ground cover as many measured road cells as the existing eight gateways. With the error the model shows at sites it has never seen applied, matching the existing network takes six or seven sites on average, and even eight beat it only 86-88% of the time. There is no robust saving.
- Double coverage: only 71% of cells are heard by two or more of the existing eight. Counting only masts that clear the local tree canopy, seven gateways (three of today's eight plus four rooftop antennas) double-cover more cells than the existing eight with 95% probability after the error adjustment, 348 cells on average against 307.

The conclusion: a terrain-, building- and vegetation-aware coverage model indicates that Bonn's gateways were sited for basic reach, not resilience. Moving a similar number of gateways to high ground would substantially improve redundant coverage, while cutting the gateway count for basic coverage is within the model's error.

## 2. Data and design choices

| Component | What is used | Why |
|---|---|---|
| Labels | Observed receptions plus inferred misses: a gateway that was active that day but did not log a packet that another gateway heard | The Bonn data only contains received packets. There are no device IDs, times of day, uptime logs, or packets that no gateway heard, so a miss is an inference. The model's score means "heard by this gateway, given that some other gateway heard it", not an absolute delivery probability. |
| Existing network | The eight gateways active together from 2020-11-10 to 2021-01-26 | The nine gateways were never all active at once (244d stopped in October 2020; 08fc started in November). In this window each of the eight logged packets on every survey day. 06df, labelled "(Old)" in the source, stops on the window's last day, so a scenario without it is also tested. |
| Demand | 392 road cells of 500 m, each represented by a real truck position | The sensors rode on garbage trucks, so only cells along their routes can be checked. Results describe the measured road network, not population or area. |
| Model | LightGBM 4.7 with monotone constraints: reception falls with distance, diffraction loss, foliage and blocking buildings, and rises with antenna height. Compared with scikit-learn gradient boosting. | Unconstrained trees can extrapolate erratically at sites unlike the training gateways. The constraints keep predictions physically sensible at new sites at almost no cost in accuracy. |
| Elevation | All 280 1 m DGM1/DOM1 tiles (1.2 GB, flown 2021-22), modelled on a 2 m grid with the surface aggregated by maximum so narrow obstacles survive | A 1 m mosaic of the study area is 250 million cells per layer (about 1 GB each), more than the working machine's free memory alongside the models. Paths are sampled every 4 m, which makes 2 m the useful resolution. The first prototype used a 20 m resample. |
| Buildings | All 280 LoD2 CityGML tiles (5 GB) reduced to 283,891 buildings with footprints and measured roof heights | Used for buildings crossed and blocking, and for real roof heights at candidate sites. LoD2 is current (2025-26), so buildings built or demolished since 2020-21 are wrong for the survey period. |
| Land cover | ESA WorldCover 2020 (10 m), the survey year | Separates trees from buildings, which the surface model alone cannot. The study area is 49% tree cover and 20% built-up. |
| Propagation physics | First-Fresnel-zone clearance plus single knife-edge diffraction (ITU-R P.526) on terrain and surface, with 4/3-Earth curvature | The Fresnel zone is the ellipsoid around the direct path that must be mostly clear for a strong signal. Multi-edge diffraction or ray tracing was beyond the compute budget; the strongest edge is a standard approximation, and the model learns corrections from data. |
| Candidates | Tallest LoD2 roof per 1 km tile (148), highest bare-ground point per 2 km tile (31), a 1 km grid (147), plus the 8 existing gateways: 334 sites | Heights are limited to what the model saw in training (antennas 12-30 m above ground; 15 m masts on grid and high-point sites). The 160 m Post Tower and other tall roofs are excluded as extrapolation. Roof access, ownership and power are unknown. |
| Redundancy | r = 1 and r = 2 | This turned out to matter most (Section 3.4). |
| Baselines | Free space and a fitted log-distance model | The dataset also ships six classical model predictions, but its ITM and Egli columns are identical in every row, so they can't serve as independent baselines. |
| Comparison | The best subset of existing gateways, a greenfield design from new sites only, and a mixed pool, each with and without the error adjustment | The first prototype found four sites could replace eight, but four existing gateways already did that under its model, and its one new building was 6.8 m tall in LoD2, not the 16.5 m its 20 m pixel suggested. The comparison has to separate redundancy already in the network from genuinely better placement. |

The Juiz de Fora campus dataset (one gateway) was set aside as a possible transfer test; it can't test placement.

## 3. Results

Unless stated, tests use leave-one-gateway-out validation: the model is trained without one gateway and tested on it, so the score reflects a site the model has never seen, which is the situation of every candidate site.

### 3.1 Coverage model

Reception classifier (does gateway g hear the packet?), mean over nine held-out gateways. AUC measures how well the model ranks heard above unheard links: 0.5 is chance, 1.0 is perfect.

| Model | Features | Mean AUC | Worst gateway |
|---|---|---:|---:|
| Gradient boosting | distance and antenna height | 0.820 | 0.716 |
| Gradient boosting | + 20 m terrain and surface (first prototype) | 0.878 | 0.718 |
| LightGBM | + 2 m terrain, Fresnel, diffraction | 0.887 | 0.766 |
| LightGBM | + surface obstruction | 0.891 | 0.781 |
| LightGBM | + LoD2 buildings | 0.891 | 0.780 |
| LightGBM | + WorldCover foliage and land cover (full) | 0.895 | 0.807 |
| LightGBM, monotone | full | 0.893 | 0.792 |
| LightGBM, monotone | full minus absolute elevations | 0.894 | 0.807 |

The new layers mainly help the hardest site: 06df, a 55 m tower, rises from 0.72 to 0.81. Removing absolute ground elevations, which could let the model recognise individual gateways, does not hurt, so the gain comes from path geometry rather than memorising sites. Held-out predictions are close to calibrated: links scored 0.5 are heard about half the time (`results/v2_models/calibration.png`).

Received power (the dataset authors' received-packet-power measure), pooled error over held-out gateways: fitted log-distance 8.44 dB, first prototype 6.84 dB, v2 LightGBM 5.91 dB, better than the prototype at 8 of 9 gateways.

### 3.2 Predicting a site the model has never seen

For each of the eight period gateways, the model was trained without it, predicted which road cells it would cover, and was compared with the cells it actually heard at least half the time (cells with at least five packets).

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Distance and height | 0.734 | 0.478 | 0.450 |
| First prototype (20 m) | 0.807 | 0.661 | 0.665 |
| v2 monotone LightGBM | 0.802 | 0.706 | 0.714 |

Precision is the share of cells predicted for a new site that it really covers; recall is the share of the cells it really covers that were predicted. About one predicted cell in five is wrong, and the errors vary a lot by site: the model under-predicts the tall 06df (recall 0.47) and over-predicts 01c6 (precision 0.39). These per-site error profiles feed the error adjustment in Section 3.4.

### 3.3 Placement

Score cutoff 0.5; 392 road cells; all integer programs solved to proven optimality with SciPy/HiGHS.

| Question | Single coverage (r = 1) | Double coverage (r = 2) |
|---|---|---|
| Existing eight gateways | 368 cells (94%) | 278 cells (71%) |
| Fewest existing gateways keeping all of those cells | 5 | 8 (none can be removed) |
| Fewest new sites matching the existing cell count | 3 | 4 (335 cells) |
| Fewest new sites for 95% of cells | 3 | 7 |
| Fewest sites from the mixed pool keeping every existing cell | 4 (2 existing, 2 new) | 6 (2 existing, 4 new) |

The selected new sites are mostly on Bonn's high ground (Venusberg and the Ennert/Siebengebirge edge, 150-190 m) rather than in the Rhine valley (55-60 m), where six of the eight existing gateways sit. In the mixed pool the optimizer keeps the 55 m tower 06df plus one of the two existing high-ground gateways (08fc or 0917). Three new sites match single coverage at cutoffs 0.3, 0.5 and 0.7 alike. Without 06df, the existing seven cover 358 cells and two new sites match them. The greedy "best next site" heuristic comes within 0-12 cells of the integer optimum (`results/v2_placement/coverage_curves.png`, `design_maps.png`).

### 3.4 Accounting for model error

The placement numbers above trust the model completely. To see how much that matters, each design was perturbed 2,000 times, giving every site the held-out error profile of a randomly chosen real gateway, with the same treatment for existing and new sites.

| | Existing 8 (error-adjusted mean) | Sites to match on average | Chance that 6 / 8 sites beat the existing 8 |
|---|---:|---:|---:|
| Single coverage | 375 cells | 6 (mixed) / 7 (new only) | 63-78% / 86-88% |
| Double coverage | 308 cells | 5 | 91-94% / 99% |

The saving in gateway count for single coverage disappears into the error; the redundancy gain does not (`results/v2_placement/error_adjusted_curves.png`).

### 3.5 Robustness

Each scenario reruns placement and the error adjustment. The table gives the chance that k sites double-cover more cells than the existing eight (new-only / mixed pool where they differ).

| Scenario | 6 sites | 7 sites | 8 sites |
|---|---:|---:|---:|
| Baseline: 15 m masts, all 31 high points | 91 / 94% | 96 / 97% | 99% |
| 25 m masts on grid and high-point sites | 93 / 94% | 97 / 99% | 98 / 99% |
| 15 m masts, high points in tree cover removed (26 of 31) | 76 / 87% | 92% | 93% |
| 25 m masts, high points in tree cover removed | 88 / 90% | 88 / 96% | 96 / 98% |
| 15 m masts that clear the local canopy (Section 3.6) | 70 / 74% | 88 / 95% | 94 / 96% |
| 25 m masts that clear the local canopy | 90 / 91% | 96% | 96 / 98% |
| Baseline, with spatially blocked error profiles | 90 / 94% | 96 / 97% | 99% |

- Mast height: at 25 m, which clears the 23-28 m canopy and is still inside the 12-30 m training range, the result is unchanged or slightly stronger.
- Forested high points: most bare-ground high points are in WorldCover tree cover, where a 15 m mast would sit below the canopy. Without them the optimizer switches to grid masts and rooftops; the redundancy advantage holds but needs about one more site. With 25 m masts most of the gap is recovered, and no high point is selected at all.
- Spatially blocked validation (`spatial_cv_v2.py`): leave-one-gateway-out lets the held-out gateway's test packets share truck positions with training packets heard by other gateways. Also holding out 2 km blocks, with a 500 m buffer dropped from training, lowers mean AUC on the period table from 0.912 to 0.902. The footprint test barely moves (precision 0.802 to 0.794, recall 0.706 to 0.710), so the error adjustment is unchanged. The exception is 06df, the 55 m tower, whose AUC falls from 0.82 to 0.76: for sites unlike the training gateways, part of the model's skill comes from having seen nearby positions.

Single coverage stays within the error in every scenario: even eight sites beat the existing network only 74-93% of the time. Removing forested high points alone turned out not to be enough (Section 3.6). All figures are in `results/v2_sensitivity/summary.csv` and `results/v2_spatial_cv/`.

### 3.6 Aerial check and the canopy rule

`site_audit_v2.py` re-solves the double-coverage designs with six to eight sites in each main scenario, measures the trees and buildings around every chosen site on the 2 m surface model, and pulls a 300 m NRW orthophoto of each. Contact sheets and per-site judgements are in `results/site_audit/`.

All rooftop picks are real, sizeable buildings: apartment and office blocks, two churches, a hillside high-rise, hospital buildings and institutional complexes. Roof access would need each owner's agreement.

Most 15 m masts are not usable. In the baseline designs, three or four of every six to eight sites are masts inside forest with 20-31 m canopy, well above a 15 m pole. Removing only forested high points did not fix this, because the optimizer swapped in 1 km grid masts that were also in forest.

`placement_v2.py --clear-canopy` adds a canopy rule: a new grid or high-point mast counts only if it is at least as tall as the 90th-percentile tree or building height within 50 m. With it, 87 of 147 grid sites and 16 of 31 high points remain at 15 m, and every site chosen in the six- to eight-site designs looks buildable from the air.

With 15 m masts that must clear the canopy, seven gateways (keeping three of today's eight, 06df, 08fc and 0917, and adding four rooftop antennas) double-cover more of the measured roads than the existing eight with 95% probability: an error-adjusted mean of 348 cells against 307, about 13% more. Using new sites only, seven reach 88% and eight 94%. With 25 m masts, six sites are enough (90-91%). For single coverage there is no robust saving.

## 4. Limitations

- Labels are inferred misses conditional on another gateway hearing the packet; true outages and packets no one heard are invisible.
- The error adjustment uses errors measured at eight existing gateways. New hilltop and rooftop sites may differ more, and an optimizer naturally favours sites with optimistic predictions; calibration at high scores limits but does not remove this.
- The canopy rule and aerial check keep only sites that look buildable, but ownership, roof access and power are not checked.
- LoD2 building data (2025-26) post-dates the survey; the 1 m surface models are from 2021-22.
- Demand covers the trucks' road network only, with equal weight per 500 m cell.
- Capacity, interference and duty-cycle limits are not modelled.

## Data licences

Bonn LoRaWAN survey (Rademacher et al., Data licence Germany attribution 2.0); NRW DGM1, DOM1, LoD2 and orthophotos (Data licence Germany zero 2.0); ESA WorldCover 2020 (CC BY 4.0).
