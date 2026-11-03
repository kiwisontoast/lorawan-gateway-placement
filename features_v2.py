"""Version-2 radio-path features on the 2 m NRW / LoD2 / WorldCover grid.

Each link is sampled about every 4 m. Heights include a 4/3-Earth curvature
bulge. Features cover terrain and surface obstruction, first-Fresnel-zone
clearance, single-edge knife-edge diffraction (ITU-R P.526, strongest edge
only), LoD2 buildings crossed, foliage depth (WorldCover tree/shrub pixels
outside LoD2 footprints whose surface rises above the path), and land-cover
fractions. These are approximations for a learned model, not a propagation
prediction in their own right.
"""
import site
from pathlib import Path

ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / '.geo_deps'))
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

TERRAIN = ROOT / 'data/terrain'
DGM, DOM = TERRAIN / 'bonn_dgm_2m.tif', TERRAIN / 'bonn_dom_2m.tif'
LANDCOVER = TERRAIN / 'bonn_worldcover2020_2m.tif'
BUILDINGS = ROOT / 'data/lod2/bonn_lod2_buildings_2m.tif'
LOD2 = ROOT / 'data/lod2/bonn_lod2_parts.parquet'
FREQUENCY_HZ = 868e6
WAVELENGTH_M = 299_792_458 / FREQUENCY_HZ
EARTH_RADIUS_M = 6_371_000 * 4 / 3
SENSOR_HEIGHT_M = 2.0
STEP_M = 4.0
GATEWAY_EXCLUSION_M = 10.0  # ignore the gateway's own roof or mast structure
SIZES = np.array([51, 101, 201, 401, 801, 1601, 2401, 3201, 4001])
TREE_CLASSES = (10, 20)
OPEN_CLASSES = (30, 40, 60, 90)

BASE_FEATURES = ['log_distance_km', 'antenna_height']
TERRAIN_FEATURES = ['ground_sensor_m', 'ground_gateway_m', 'elevation_angle_deg', 'terrain_max_excess_m',
                    'terrain_nu_max', 'terrain_diffraction_db', 'terrain_fresnel_blocked_frac']
SURFACE_FEATURES = ['surface_max_excess_m', 'surface_nu_max', 'surface_diffraction_db',
                    'surface_fresnel_blocked_frac', 'surface_los_blocked_frac', 'near_sensor_excess_m']
BUILDING_FEATURES = ['buildings_crossed', 'buildings_blocking', 'building_path_m', 'building_excess_max_m']
LANDCOVER_FEATURES = ['foliage_in_path_m', 'foliage_in_fresnel_m', 'frac_tree', 'frac_built', 'frac_open',
                      'frac_water', 'sensor_built', 'sensor_tree']
ALL_FEATURES = BASE_FEATURES + TERRAIN_FEATURES + SURFACE_FEATURES + BUILDING_FEATURES + LANDCOVER_FEATURES


def knife_edge_db(nu):
    """ITU-R P.526 single knife-edge approximation; zero below nu = -0.78."""
    nu = np.asarray(nu, dtype=float)
    loss = 6.9 + 20 * np.log10(np.sqrt((nu - 0.1)**2 + 1) + nu - 0.1)
    return np.where(nu > -0.78, loss, 0.0)


def build_building_raster():
    """Rasterize LoD2 footprints to the 2 m grid; values are 1-based building indices."""
    from rasterio.features import rasterize
    from shapely import wkb
    parts = pd.read_parquet(LOD2)
    codes, _ = pd.factorize(parts.building_id)
    with rasterio.open(DGM) as ref:
        shapes = ((wkb.loads(g), int(c) + 1) for g, c in zip(parts.footprint_wkb, codes))
        grid = rasterize(shapes, out_shape=ref.shape, transform=ref.transform, fill=0, dtype='int32')
        profile = ref.profile | dict(dtype='int32', nodata=0, predictor=2)
    with rasterio.open(BUILDINGS, 'w', **profile) as ds:
        ds.write(grid, 1)
    return grid


class Grids:
    """Holds the aligned 2 m rasters in memory (about 1 GB)."""
    def __init__(self, dgm=DGM, dom=DOM, landcover=LANDCOVER, buildings=BUILDINGS):
        with rasterio.open(dgm) as a, rasterio.open(dom) as b, rasterio.open(landcover) as c:
            if not (a.transform == b.transform == c.transform and a.shape == b.shape == c.shape):
                raise ValueError('2 m rasters are not aligned')
            self.transform, self.shape = a.transform, a.shape
            self.dgm = a.read(1).astype('float32')
            self.dom = b.read(1).astype('float32')
            self.landcover = c.read(1)
        if Path(buildings).exists():
            with rasterio.open(buildings) as d:
                if d.transform != self.transform or d.shape != self.shape:
                    raise ValueError('Building raster is not aligned')
                self.buildings = d.read(1)
        else:
            self.buildings = build_building_raster()
        if self.transform.b or self.transform.d:
            raise ValueError('Expected north-up rasters')

    def index(self, x, y):
        col = np.floor((x - self.transform.c) / self.transform.a).astype(np.int64)
        row = np.floor((y - self.transform.f) / self.transform.e).astype(np.int64)
        if (col < 0).any() or (row < 0).any() or (col >= self.shape[1]).any() or (row >= self.shape[0]).any():
            raise ValueError('Path sample outside the 2 m raster extent')
        return row, col


def _runs(ids):
    """Count runs of non-zero, changing IDs along each row."""
    prev = np.concatenate([np.zeros((ids.shape[0], 1), ids.dtype), ids[:, :-1]], axis=1)
    return (ids != 0) & (ids != prev)


def _chunk(g, x0, y0, x1, y1, height, n):
    m = len(x0)
    t = np.linspace(0, 1, n, dtype='float64')[None, :]
    xx = x0[:, None] + (x1 - x0)[:, None] * t
    yy = y0[:, None] + (y1 - y0)[:, None] * t
    row, col = g.index(xx, yy)
    ground = g.dgm[row, col]
    surface = g.dom[row, col]
    cover = g.landcover[row, col]
    bid = g.buildings[row, col]
    if np.isnan(ground).any() or np.isnan(surface).any():
        raise ValueError('Missing DGM/DOM value on a path')
    dist = np.hypot(x1 - x0, y1 - y0)
    d1 = dist[:, None] * t
    d2 = dist[:, None] - d1
    bulge = d1 * d2 / (2 * EARTH_RADIUS_M)
    z0 = ground[:, :1] + SENSOR_HEIGHT_M
    z1 = ground[:, -1:] + height[:, None]
    line = z0 + (z1 - z0) * t
    with np.errstate(invalid='ignore', divide='ignore'):
        r1 = np.sqrt(WAVELENGTH_M * d1 * d2 / np.maximum(dist[:, None], 1e-9))
    interior = (d1 >= 2.0) & (d2 >= GATEWAY_EXCLUSION_M)
    out = {}
    for name, height_map in [('terrain', ground + bulge), ('surface', surface + bulge)]:
        excess = np.where(interior, height_map - line, -np.inf)
        nu = np.where(interior & (r1 > 0), np.sqrt(2) * (height_map - line) / np.where(r1 > 0, r1, 1), -np.inf)
        frac_den = np.maximum(interior.sum(axis=1), 1)
        has = interior.any(axis=1)
        out[f'{name}_max_excess_m'] = np.where(has, excess.max(axis=1), np.nan)
        nu_max = np.where(has, nu.max(axis=1), np.nan)
        out[f'{name}_nu_max'] = nu_max
        out[f'{name}_diffraction_db'] = np.where(has, knife_edge_db(np.nan_to_num(nu_max, nan=-10)), 0.0)
        out[f'{name}_fresnel_blocked_frac'] = ((height_map - line > -0.6 * r1) & interior).sum(axis=1) / frac_den
        if name == 'surface':
            above = (height_map - line > 0) & interior
            out['surface_los_blocked_frac'] = above.sum(axis=1) / frac_den
            near = interior & (d1 <= 100)
            out['near_sensor_excess_m'] = np.where(near.any(axis=1),
                                                   np.where(near, height_map - line, -np.inf).max(axis=1), np.nan)
    in_building = (bid != 0) & interior
    starts = _runs(np.where(interior, bid, 0))
    step = dist / (n - 1)
    out['buildings_crossed'] = starts.sum(axis=1)
    above_line = (surface + bulge - line) > 0
    # A crossed building blocks when any of its sampled cells is above the line.
    blocking = np.zeros(m)
    for i in np.flatnonzero(in_building.any(axis=1)):
        ids = bid[i][in_building[i] & above_line[i]]
        blocking[i] = len(np.unique(ids))
    out['buildings_blocking'] = blocking
    out['building_path_m'] = in_building.sum(axis=1) * step
    out['building_excess_max_m'] = np.where(in_building.any(axis=1),
                                            np.where(in_building, surface + bulge - line, -np.inf).max(axis=1), np.nan)
    tree = np.isin(cover, TREE_CLASSES) & (bid == 0) & interior & (surface - ground > 2)
    out['foliage_in_path_m'] = (tree & above_line).sum(axis=1) * step
    out['foliage_in_fresnel_m'] = (tree & ((surface + bulge - line) > -0.6 * r1)).sum(axis=1) * step
    out['frac_tree'] = np.isin(cover, TREE_CLASSES).mean(axis=1)
    out['frac_built'] = (cover == 50).mean(axis=1)
    out['frac_open'] = np.isin(cover, OPEN_CLASSES).mean(axis=1)
    out['frac_water'] = (cover == 80).mean(axis=1)
    out['sensor_built'] = (cover[:, 0] == 50).astype(float)
    out['sensor_tree'] = np.isin(cover[:, 0], TREE_CLASSES).astype(float)
    out['ground_sensor_m'] = ground[:, 0]
    out['ground_gateway_m'] = ground[:, -1]
    out['elevation_angle_deg'] = np.degrees(np.arctan2((z1 - z0)[:, 0], np.maximum(dist, 1.0)))
    out['distance_m'] = dist
    return out


def path_features(data, grids=None, chunk_samples=1_500_000):
    """Compute v2 features for columns longitude/latitude _sensor/_gateway and antenna_height."""
    g = grids or Grids()
    project = Transformer.from_crs(4326, 25832, always_xy=True)
    x0, y0 = project.transform(data.longitude_sensor.to_numpy(), data.latitude_sensor.to_numpy())
    x1, y1 = project.transform(data.longitude_gateway.to_numpy(), data.latitude_gateway.to_numpy())
    height = data.antenna_height.to_numpy(dtype=float)
    dist = np.hypot(x1 - x0, y1 - y0)
    n_needed = np.ceil(dist / STEP_M).astype(int) + 1
    size = SIZES[np.minimum(np.searchsorted(SIZES, n_needed), len(SIZES) - 1)]
    names = [f for f in ALL_FEATURES if f not in BASE_FEATURES] + ['distance_m']
    result = {k: np.full(len(data), np.nan) for k in names}
    for n in np.unique(size):
        idx = np.flatnonzero(size == n)
        per = max(1, chunk_samples // n)
        for s in range(0, len(idx), per):
            j = idx[s:s + per]
            part = _chunk(g, x0[j], y0[j], x1[j], y1[j], height[j], int(n))
            for k in names:
                result[k][j] = part[k]
    frame = pd.DataFrame(result, index=data.index)
    # Same convention as v1: distances under the 218 m training minimum are clipped.
    frame['log_distance_km'] = np.log10(np.maximum(frame.distance_m, 218.0) / 1000)
    frame['distance_was_clipped'] = frame.distance_m < 218.0
    frame['antenna_height'] = height
    return frame
