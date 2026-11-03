"""Extract approximate terrain and surface path features; evaluate gateways.

Run after fetch_bonn_terrain.py. Uses 201 equally spaced samples per link, a
20 m downsampled DGM/DOM, and a 2 m sensor height. This is an approximate
surface obstruction proxy, not a validated line-of-sight determination.
"""
import site
from pathlib import Path

ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / '.geo_deps'))
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from baseline_bonn import load_data, fit_log_distance, summarize
from geoai_bonn import new_model

RASTER = ROOT / 'data/terrain/bonn_dgm_20m.tif'
SURFACE = ROOT / 'data/terrain/bonn_dom_20m.tif'
OUT = ROOT / 'results/bonn_terrain'
N_SAMPLES = 201
SENSOR_HEIGHT_M = 2.0
TERRAIN_FEATURES = ['ground_at_sensor_m', 'ground_at_gateway_m',
                    'ground_delta_m', 'max_ground_above_link_m',
                    'ground_obstructed_fraction']
SURFACE_FEATURES = ['max_surface_above_link_m', 'surface_obstructed_fraction']


def path_features(data, raster_path=RASTER, surface_path=SURFACE):
    """Calculate features using two endpoints and surface; reusable for candidates."""
    transform = Transformer.from_crs(4326, 25832, always_xy=True)
    x0, y0 = transform.transform(data.longitude_sensor.to_numpy(), data.latitude_sensor.to_numpy())
    x1, y1 = transform.transform(data.longitude_gateway.to_numpy(), data.latitude_gateway.to_numpy())
    heights = data.antenna_height.to_numpy(dtype=float)
    out = np.full((len(data), len(TERRAIN_FEATURES) + len(SURFACE_FEATURES)), np.nan)
    with rasterio.open(raster_path) as ds, rasterio.open(surface_path) as surface_ds:
        if ds.crs is None or ds.crs.to_epsg() != 25832:
            raise ValueError('Expected terrain raster in EPSG:25832')
        if ds.transform.b != 0 or ds.transform.d != 0:
            raise ValueError('Expected north-up terrain raster')
        if surface_ds.crs != ds.crs or surface_ds.transform != ds.transform or surface_ds.shape != ds.shape:
            raise ValueError('Surface and terrain rasters must be aligned')
        raster = ds.read(1, masked=True).filled(np.nan)
        surface_raster = surface_ds.read(1, masked=True).filled(np.nan)
        t = np.linspace(0, 1, N_SAMPLES)[None, :]
        for start in range(0, len(data), 2000):
            stop = min(start + 2000, len(data))
            xx = x0[start:stop, None] + (x1[start:stop] - x0[start:stop])[:, None] * t
            yy = y0[start:stop, None] + (y1[start:stop] - y0[start:stop])[:, None] * t
            col = np.floor((xx - ds.transform.c) / ds.transform.a).astype(int)
            row = np.floor((yy - ds.transform.f) / ds.transform.e).astype(int)
            valid = (row >= 0) & (row < ds.height) & (col >= 0) & (col < ds.width)
            ground = np.full(xx.shape, np.nan)
            surface = np.full(xx.shape, np.nan)
            ground[valid] = raster[row[valid], col[valid]]
            surface[valid] = surface_raster[row[valid], col[valid]]
            complete = np.isfinite(ground).all(axis=1) & np.isfinite(surface).all(axis=1)
            if not complete.any():
                continue
            z = ground[complete]
            line = (z[:, :1] + SENSOR_HEIGHT_M) * (1-t) + (z[:, -1:] + heights[start:stop][complete, None]) * t
            obstruction = z[:, 1:-1] - line[:, 1:-1]
            idx = np.flatnonzero(complete) + start
            out[idx, 0] = z[:, 0]
            out[idx, 1] = z[:, -1]
            out[idx, 2] = z[:, -1] - z[:, 0]
            out[idx, 3] = obstruction.max(axis=1)
            out[idx, 4] = (obstruction > 0).mean(axis=1)
            surface_obstruction = surface[complete, 1:-1] - line[:, 1:-1]
            out[idx, 5] = surface_obstruction.max(axis=1)
            out[idx, 6] = (surface_obstruction > 0).mean(axis=1)
    return pd.DataFrame(out, columns=TERRAIN_FEATURES + SURFACE_FEATURES, index=data.index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    data['log_distance_km'] = np.log10(data.distance_km)
    features = path_features(data)
    data[TERRAIN_FEATURES + SURFACE_FEATURES] = features
    missing = features.isna().any(axis=1)
    print(f'Complete paths: {(~missing).sum()} / {len(data)}', flush=True)
    if missing.any():
        raise ValueError(f'{missing.sum()} paths have missing terrain; no rows silently excluded')
    data[['gw', 'transceived_at', 'pkt_number', 'latitude_sensor', 'longitude_sensor',
          *TERRAIN_FEATURES, *SURFACE_FEATURES]].to_csv(OUT / 'terrain_features.csv', index=False, float_format='%.6f')
    model_features = {'boosted_distance_height': ['log_distance_km', 'antenna_height'],
                      'boosted_with_terrain': ['log_distance_km', 'antenna_height', *TERRAIN_FEATURES],
                      'boosted_with_surface': ['log_distance_km', 'antenna_height', *TERRAIN_FEATURES, *SURFACE_FEATURES]}
    data['log_distance'] = np.nan
    for name in model_features:
        data[name] = np.nan
    rows = []
    for gateway in sorted(data.gw.unique()):
        test = data.gw == gateway
        train = ~test
        a, b = fit_log_distance(data.loc[train])
        data.loc[test, 'log_distance'] = a + b * data.loc[test, 'log_distance_km']
        for name, columns in model_features.items():
            model = new_model()
            model.fit(data.loc[train, columns], data.loc[train, 'observed_rpp_dbm'])
            data.loc[test, name] = model.predict(data.loc[test, columns])
        for name in ['log_distance', *model_features]:
            rows.append({'held_out_gateway': gateway, 'model': name,
                         **summarize(data.loc[test, 'observed_rpp_dbm'], data.loc[test, name])})
        print(f'Evaluated gateway {gateway}', flush=True)
    names = ['log_distance', *model_features]
    overall = pd.DataFrame([{'model': name, **summarize(data.observed_rpp_dbm, data[name])} for name in names])
    overall.to_csv(OUT / 'overall_metrics.csv', index=False, float_format='%.6f')
    pd.DataFrame(rows).to_csv(OUT / 'gateway_metrics.csv', index=False, float_format='%.6f')
    data[['gw', 'transceived_at', 'pkt_number', 'latitude_sensor', 'longitude_sensor',
          'observed_rpp_dbm', *names]].to_csv(OUT / 'held_out_predictions.csv', index=False, float_format='%.6f')
    print(overall.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print('Obstructed terrain paths:', int((data.max_ground_above_link_m > 0).sum()))
    print('Obstructed surface paths:', int((data.max_surface_above_link_m > 0).sum()))
    print('Results:', OUT)


if __name__ == '__main__':
    main()
