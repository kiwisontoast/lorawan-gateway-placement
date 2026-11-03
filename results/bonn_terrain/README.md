# Terrain and surface features (20 m)

## Data

NRW terrain (DGM, bare earth) and surface (DOM, the tops of buildings and vegetation) rasters, 657 x 959 pixels at about 20 m, downsampled from the 1 m products in the WCS request. The rasters, requests and CRS are in `data/terrain/`. NRW publishes the [DGM](https://www.opengeodata.nrw.de/produkte/geobasis/hm/dgm1_tiff/dgm1_tiff/) and [DOM](https://www.opengeodata.nrw.de/produkte/geobasis/hm/dom1_tiff/dom1_tiff/) under Data licence Germany zero 2.0. They came from the current services, and their flight dates haven't been matched to the 2020-21 radio campaign.

## Features

`terrain_bonn.py` computes ground elevation at the sensor and the gateway, their difference, and how far the ground and the surface rise above the straight path between them. Each path is sampled at 201 evenly spaced points, with a 2 m sensor height (as in the Bonn authors' notebook) and the reported antenna height taken as height above local ground. The longest path is about 13 km, so samples can be up to about 65 m apart. The obstruction features skip the endpoints and record the largest rise above the line and the fraction of samples above it.

All 131,057 paths fell inside the rasters. 16,344 had at least one bare-ground sample above the line, and 128,448 had at least one surface sample above it. These are geometry counts, not failed links.

## Results

Each gateway is held out in turn, as in the earlier experiments. The log-distance and distance-plus-height models are the same as before; the third model adds five terrain features and the fourth adds two surface-obstruction features. All four use the same received-packet rows.

| Model | Pooled held-out MAE | RMSE |
|---|---:|---:|
| Log-distance | 8.44 dB | 10.62 dB |
| Boosted distance + height | 7.57 dB | 9.82 dB |
| Boosted distance + height + terrain | 7.53 dB | 9.75 dB |
| Boosted distance + height + terrain + surface | 6.82 dB | 8.90 dB |

Terrain alone lowered MAE at five of nine gateways and raised it at four. Terrain plus surface lowered it at six and raised it at three; at 06df, for example, it scored 9.46 dB against 8.74 dB for distance and height. Pooled results weight gateways by observation count, and nine gateways with correlated readings can't support a strong general claim.

## Limits

The targets are received packets only, so these are power predictions, not delivery probabilities, and gateway uptime and packet identity are still unknown. The DOM doesn't separate buildings from trees, endpoint heights rest on DGM ground rather than the real mounting, and there is no Fresnel-zone or diffraction term. At 20 m and 201 samples, narrow obstacles between samples can be missed, and GPS error adds to that. The 2 m sensor height and above-ground antenna heights are assumptions.

With only nine gateways, a model that sees per-site height and elevation may partly learn the sites themselves. Other cities, radios and frequencies would need their own test before using a Bonn-trained model. This experiment does not compute coverage or placement.

## Reproduction

`python fetch_bonn_terrain.py` and then `python fetch_bonn_surface.py` download the rasters; NRW can update the service, so keep the copies in `data/terrain/` for exact reruns. `python terrain_bonn.py` computes the features and evaluates the models, and `python -m unittest test_terrain_features` checks the features against a synthetic obstacle.

The scripts need pandas, NumPy, requests, pyproj, rasterio, Matplotlib and scikit-learn, and also look for packages in a local `.geo_deps` folder. This run used Python 3.12, rasterio 1.5.2, pyproj 3.8.0 and scikit-learn 1.5.2.

The script also writes row-level `terrain_features.csv` and held-out predictions (not committed) next to the committed metrics. Its `path_features` function takes any endpoint coordinates and gateway heights in the same format, so it also works for candidate links inside the raster extent. Paths that leave the rasters raise an error instead of being dropped.
