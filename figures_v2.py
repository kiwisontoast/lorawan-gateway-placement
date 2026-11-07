"""Figures for the main results: maps and charts in one palette (cream / ink / terracotta).

Run after the placement, robustness and site-audit scripts:
    python figures_v2.py
Writes PNGs (JPEG for the photo sheet) to results/figures/. Charts use Inter if its static TTFs are in .fonts/
(git-ignored; from github.com/rsms/inter releases), otherwise the default sans.
Terracotta marks proposed sites and blue existing ones; coverage counts use a
one-hue teal ramp. Both stay distinguishable under common colour-vision
deficiencies on the cream background.
"""
import site
from pathlib import Path

ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / '.geo_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib import font_manager
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = ROOT / 'results/figures'
SCEN = ROOT / 'results/v2_sensitivity/mast15_clear_canopy'
PHOTOS = ROOT / 'results/site_audit/photos'
HEADLINE = ['existing_06df', 'existing_08fc', 'existing_0917', 'roof_UUID_0a7652f9-9e1a-4b3d-b576-94ad7391dee2',
            'roof_DENW36AL10001v8m', 'roof_UUID_60b93780-b98b-412c-a3ab-b5b61449ffb6',
            'roof_UUID_4072d493-245f-44b4-8072-60cc3d859413']
# Earlier picks that the aerial check rejected (15 m masts in forest; grid_057 from the forest-hill rerun).
REJECTED = ['hill_017', 'hill_024', 'hill_015', 'grid_057']

CREAM, SAND, INK, MUTED, GRID = '#F6F3EC', '#ECE6D9', '#14262B', '#5C6B6E', '#DDD6C8'
PROPOSED, EXISTING = '#9C4719', '#1C6FA0'
RAMP = ['#8FB4BE', '#4A8392', '#1D4F5C']  # 0 / 1 / 2+ gateways
WATER = '#C9D6DA'
DPI = 200


def style():
    fonts = ROOT / '.fonts'
    for f in fonts.glob('Inter-*.ttf'):
        font_manager.fontManager.addfont(str(f))
    family = 'Inter' if any(fonts.glob('Inter-*.ttf')) else 'sans-serif'
    plt.rcParams.update({
        'font.family': family, 'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK,
        'axes.edgecolor': GRID, 'axes.linewidth': 0.8, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'axes.facecolor': CREAM, 'figure.facecolor': CREAM,
        'savefig.facecolor': CREAM, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.8, 'legend.frameon': False,
        'legend.fontsize': 10})


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, pil_kwargs={'quality': 90} if name.endswith('.jpg') else None)
    plt.close(fig)
    print('wrote', name)


# ---------------------------------------------------------------- maps

def basemap(ax, extent):
    """Grey hillshade of bare-earth terrain with the Rhine and other water."""
    with rasterio.open(ROOT / 'data/terrain/bonn_dgm_20m.tif') as src:
        z = src.read(1).astype(float)
        b = src.bounds
    shade = LightSource(azdeg=315, altdeg=40).hillshade(z, vert_exag=4, dx=20, dy=20)
    ax.imshow(shade, cmap=ListedColormap(plt.cm.Greys(np.linspace(0.05, 0.45, 256))[::-1]),
              extent=(b.left, b.right, b.bottom, b.top), alpha=0.55, zorder=0)
    with rasterio.open(ROOT / 'data/terrain/bonn_worldcover2020_2m.tif') as src:
        lc = src.read(1)[::10, ::10]
        b = src.bounds
    water = np.ma.masked_where(lc != 80, lc)
    ax.imshow(water, cmap=ListedColormap([WATER]), extent=(b.left, b.right, b.bottom, b.top), zorder=1)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # 2 km scale bar and north arrow, bottom left.
    x0, y0 = extent[0] + 600, extent[2] + 600
    ax.plot([x0, x0 + 2000], [y0, y0], color=INK, lw=2, solid_capstyle='butt', zorder=6)
    ax.text(x0 + 1000, y0 + 220, '2 km', ha='center', va='bottom', fontsize=9, color=INK, zorder=6)
    ax.annotate('N', xy=(x0 + 2700, y0 + 900), xytext=(x0 + 2700, y0), ha='center', va='bottom',
                fontsize=9, color=INK, arrowprops=dict(arrowstyle='-|>', color=INK, lw=1), zorder=6)
    # Label the Rhine at the water pixel nearest the map centre.
    wy, wx = np.nonzero(lc == 80)
    px, py = b.left + (wx + .5) * 20, b.top - (wy + .5) * 20
    cx, cy = (extent[0] + extent[1]) / 2, (extent[2] + extent[3]) / 2
    i = np.argmin((px - cx) ** 2 + (py - cy) ** 2)
    ax.text(px[i] + 350, py[i], 'Rhine', fontsize=9, color=MUTED, style='italic', rotation=60, zorder=2)


def map_extent(demand, pad=1200):
    return (demand.x.min() - pad, demand.x.max() + pad, demand.y.min() - pad, demand.y.max() + pad)


def fig_network(sites, demand):
    ext = map_extent(demand)
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * (ext[3] - ext[2]) / (ext[1] - ext[0])), constrained_layout=True)
    basemap(ax, ext)
    ax.scatter(demand.x, demand.y, s=4, c=MUTED, alpha=0.35, linewidths=0, zorder=2)
    ex = sites[sites.kind == 'existing']
    kept = ex.site_id.isin(HEADLINE)
    new = sites[sites.site_id.isin(HEADLINE) & (sites.kind != 'existing')]
    ring = dict(edgecolors=CREAM, linewidths=2, zorder=5)
    ax.scatter(ex.x[~kept], ex.y[~kept], s=90, marker='o', facecolors=CREAM, edgecolors=EXISTING, linewidths=2, zorder=5)
    ax.scatter(ex.x[kept], ex.y[kept], s=110, marker='o', c=EXISTING, **ring)
    ax.scatter(new.x, new.y, s=120, marker='s', c=PROPOSED, **ring)
    handles = [Line2D([], [], ls='', marker='o', ms=9, mfc=EXISTING, mec=CREAM, mew=1.5, label='Existing gateway, kept'),
               Line2D([], [], ls='', marker='o', ms=8, mfc=CREAM, mec=EXISTING, mew=2, label='Existing gateway, removed'),
               Line2D([], [], ls='', marker='s', ms=9, mfc=PROPOSED, mec=CREAM, mew=1.5, label='New rooftop antenna'),
               Line2D([], [], ls='', marker='o', ms=3, mfc=MUTED, mec='none', alpha=.6, label='Measured road cell (500 m)')]
    ax.legend(handles=handles, loc='upper left', fontsize=9, borderaxespad=0.6, facecolor=CREAM, framealpha=0.85, frameon=True,
              edgecolor='none')
    save(fig, 'map_existing_vs_proposed.png')


def fig_coverage(sites, demand, scores):
    ext = map_extent(demand)
    a = scores >= 0.5
    nets = [('Today: 8 gateways', sites.kind.to_numpy() == 'existing'),
            ('Proposed: 7 gateways', sites.site_id.isin(HEADLINE).to_numpy())]
    h = 6.0
    fig, axes = plt.subplots(1, 2, figsize=(2 * h * (ext[1] - ext[0]) / (ext[3] - ext[2]) + 0.4, h + 0.9),
                             constrained_layout=True)
    for ax, (title, on) in zip(axes, nets):
        basemap(ax, ext)
        n = a[:, on].sum(axis=1)
        cls = np.minimum(n, 2)
        ax.scatter(demand.x, demand.y, s=26, marker='s', c=[RAMP[c] for c in cls], linewidths=0, zorder=3)
        g = sites[on]
        ax.scatter(g.x, g.y, s=70, marker='^', c=INK, edgecolors=CREAM, linewidths=1.5, zorder=5)
        double = int((n >= 2).sum())
        ax.text(0.02, 0.985, f'{title}\n{double} of {len(demand)} cells heard by 2 or more ({double / len(demand):.0%})',
                transform=ax.transAxes, ha='left', va='top', fontsize=11, color=INK, zorder=7,
                bbox=dict(facecolor=CREAM, edgecolor='none', alpha=0.85, pad=4))
    fig.text(0.5, 0.005, 'Model prediction at the 0.5 score cutoff, before the error adjustment',
             ha='center', va='bottom', fontsize=8.5, color=MUTED)
    handles = [Patch(color=RAMP[0], label='No gateway'), Patch(color=RAMP[1], label='1 gateway'),
               Patch(color=RAMP[2], label='2 or more (redundant)'),
               Line2D([], [], ls='', marker='^', ms=8, mfc=INK, mec=CREAM, label='Gateway')]
    fig.legend(handles=handles, loc='outside lower center', ncol=4, fontsize=9.5, borderaxespad=1.6)
    save(fig, 'map_double_coverage_today_vs_proposed.png')


# ---------------------------------------------------------------- charts

def fig_error_adjusted():
    c = pd.read_csv(SCEN / 'error_adjusted_curves.csv')
    c = c[(c.redundancy == 2) & (c.error_model == 'site')]
    base = c[c.pool == 'existing_all'].iloc[0]
    g = c[c.pool == 'full_pool'].sort_values('budget')
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.fill_between(g.budget, g.p05, g.p95, color=PROPOSED, alpha=0.10, linewidth=0)
    ax.plot(g.budget, g['mean'], color=PROPOSED, lw=2, solid_capstyle='round', zorder=3)
    ax.scatter(g.budget, g['mean'], s=36, color=PROPOSED, edgecolors=CREAM, linewidths=2, zorder=4)
    ax.axhline(base['mean'], color=EXISTING, lw=2, zorder=2)
    ax.text(1.05, base['mean'] + 4, f"Today's 8 gateways: {base['mean']:.0f} cells", color=INK, fontsize=10, va='bottom')
    k7 = g[g.budget == 7].iloc[0]
    ax.annotate(f"7 gateways: {k7['mean']:.0f} cells\nbeats today {k7.prob_beats_existing_mean:.0%} of the time",
                xy=(7, k7['mean']), xytext=(7.4, k7['mean'] - 75), fontsize=10, color=INK,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    ax.set(xlabel='Number of gateways (best mix of existing sites and new rooftops or masts)',
           ylabel='Road cells heard by 2+ gateways', xticks=range(1, 11), xlim=(0.6, 10.4), ylim=(0, 392))
    ax.text(10.4, 8, 'Line: average over 2,000 simulations of model error; band: 5th to 95th percentile',
            ha='right', fontsize=8.5, color=MUTED)
    save(fig, 'chart_error_adjusted_redundancy.png')


def fig_robustness():
    s = pd.read_csv(ROOT / 'results/v2_sensitivity/summary.csv')
    s = s[(s.redundancy == 2) & (s.pool == 'full_pool')]
    rows = [('15 m masts, trees ignored (first run)', (s.mast == '15 m') & (s.high_points == 'all') & (s.masts_clear_canopy == 'no') & (s.error_profiles == 'leave_one_gateway_out')),
            ('Same, stricter validation (spatial blocks)', (s.mast == '15 m') & (s.high_points == 'all') & (s.error_profiles == 'blocked')),
            ('25 m masts, trees ignored', (s.mast == '25 m') & (s.high_points == 'all') & (s.masts_clear_canopy == 'no')),
            ('25 m masts that clear the trees', (s.mast == '25 m') & (s.masts_clear_canopy == 'yes')),
            ('15 m masts that clear the trees (headline)', (s.mast == '15 m') & (s.masts_clear_canopy == 'yes'))]
    vals = [(lab, s[m].iloc[0]) for lab, m in rows]
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    for i, (lab, r) in enumerate(vals):
        head = 'headline' in lab
        for k, mk in ((6, 'o'), (7, 's')):
            v = r[f'p_beats_existing_k{k}']
            ax.scatter(v, i, s=70 if k == 7 else 50, marker=mk, color=PROPOSED if k == 7 else MUTED,
                       edgecolors=CREAM, linewidths=2, zorder=3)
        ax.plot([r.p_beats_existing_k6, r.p_beats_existing_k7], [i, i], color=GRID, lw=2, zorder=2)
        ax.text(r.p_beats_existing_k7 + 0.012, i, f"{r.p_beats_existing_k7:.0%}", va='center', fontsize=9.5,
                fontweight='bold' if head else 'normal')
    ax.set_yticks(range(len(vals)), [v[0] for v in vals])
    for t, (lab, _) in zip(ax.get_yticklabels(), vals):
        t.set_color(INK if 'headline' in lab else MUTED)
    ax.set(xlim=(0.5, 1.04), xlabel='Chance of double-covering more road cells than today\'s 8 gateways')
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(handles=[Line2D([], [], ls='', marker='s', ms=8, mfc=PROPOSED, mec=CREAM, label='7 gateways'),
                       Line2D([], [], ls='', marker='o', ms=7, mfc=MUTED, mec=CREAM, label='6 gateways')],
              loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, borderaxespad=0.2)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    save(fig, 'chart_robustness.png')


def fig_model_layers():
    s = pd.read_csv(ROOT / 'results/v2_models/classification_summary.csv')
    steps = [('hgb', 'distance_height', 'Distance and antenna height only'),
             ('hgb', 'v1_20m', '+ 20 m terrain (first prototype)'),
             ('lgbm', 'v2_terrain', '+ 2 m terrain, Fresnel zone, diffraction'),
             ('lgbm', 'v2_surface', '+ surface heights (trees, roofs)'),
             ('lgbm', 'v2_buildings', '+ 3D buildings'),
             ('lgbm', 'v2_full', '+ land cover (foliage)'),
             ('lgbm_mono', 'v2_full', 'Final: physics-constrained')]
    vals = [(lab, s[(s.model == m) & (s.features == f)].iloc[0]) for m, f, lab in steps]
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    for i, (lab, r) in enumerate(vals):
        final = lab.startswith('Final')
        ax.plot([r.min_auc, r.mean_auc], [i, i], color=GRID, lw=2, zorder=2)
        ax.scatter(r.min_auc, i, s=40, marker='o', facecolors=CREAM, edgecolors=MUTED, linewidths=1.5, zorder=3)
        ax.scatter(r.mean_auc, i, s=70, color=PROPOSED if final else MUTED, edgecolors=CREAM, linewidths=2, zorder=4)
        ax.text(r.mean_auc + 0.004, i, f'{r.mean_auc:.3f}', va='center', fontsize=9.5,
                fontweight='bold' if final else 'normal')
    ax.set_yticks(range(len(vals)), [v[0] for v in vals]); ax.invert_yaxis()
    ax.set(xlim=(0.68, 0.93), xlabel='AUC at a gateway the model never saw (0.5 = guessing, 1 = perfect)')
    ax.legend(handles=[Line2D([], [], ls='', marker='o', ms=8, mfc=MUTED, mec=CREAM, label='Average over 9 gateways'),
                       Line2D([], [], ls='', marker='o', ms=6, mfc=CREAM, mec=MUTED, label='Hardest gateway')],
              loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, borderaxespad=0.2)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    save(fig, 'chart_model_layers.png')


def fig_new_site_test():
    s = pd.read_csv(ROOT / 'results/v2_models/footprint_summary.csv')
    s = s[s.prediction_basis == 'rep_p']
    models = [('hgb', 'distance_height', 'Distance and height only'), ('hgb', 'v1_20m', 'First prototype (20 m)'),
              ('lgbm_mono', 'v2_full', 'Final model')]
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, constrained_layout=True)
    for ax, (col, title) in zip(axes, [('precision', 'Precision: predicted coverage that was real'),
                                       ('recall', 'Recall: real coverage that was found')]):
        for i, (m, f, lab) in enumerate(models):
            v = s[(s.model == m) & (s.features == f)][col].iloc[0]
            final = lab == 'Final model'
            ax.barh(i, v, height=0.36, color=PROPOSED if final else GRID, zorder=2)
            ax.text(v + 0.015, i, f'{v:.2f}', va='center', fontsize=10, fontweight='bold' if final else 'normal')
        ax.set_title(title, fontsize=10, loc='left', color=INK)
        ax.set_xlim(0, 1.0); ax.xaxis.grid(True); ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(range(3), [m[2] for m in models]); axes[0].invert_yaxis()
    save(fig, 'chart_new_site_test.png')


def fig_aerial(audit):
    a = audit.set_index('site_id')
    rows = [('Earlier designs: 15 m masts under the forest canopy', REJECTED),
            ('Final design: the four new rooftop antennas', [s for s in HEADLINE if s.startswith('roof')])]
    fig = plt.figure(figsize=(12, 7.2), layout='constrained')
    half = 100
    for r, (sub, (title, ids)) in enumerate(zip(fig.subfigures(2, 1, hspace=0.04), rows)):
        sub.set_facecolor(CREAM)
        sub.suptitle(title, x=0.005, ha='left', fontsize=11.5, color=INK, fontweight='bold')
        for ax, sid in zip(sub.subplots(1, 4), ids):
            img = plt.imread(PHOTOS / f'{sid}.jpg')
            c = img.shape[0] // 2
            k = int(img.shape[0] * half / 300)
            ax.imshow(img[c - k:c + k, c - k:c + k])
            ax.add_patch(plt.Circle((k, k), k * 15 / half, fill=False, color=CREAM, lw=2))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            s = a.loc[sid]
            if r == 0:
                cap = f'{s.clutter_p90_m:.0f} m trees around a 15 m mast'
            else:
                cap = f'{s.setting[0].upper()}{s.setting[1:]}'
            ax.set_title(cap, fontsize=9.5, color=INK)
    fig.text(0.995, 0.003, 'Orthophotos: Geobasis NRW, dl-de/zero-2.0. Each tile is 200 m across.',
             ha='right', va='bottom', fontsize=8, color=MUTED)
    save(fig, 'photos_aerial_check.jpg')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    sites = pd.read_csv(SCEN / 'candidate_sites.csv')
    demand = pd.read_csv(SCEN / 'demand_cells.csv')
    z = np.load(SCEN / 'score_matrix.npz')
    sites = sites.set_index('site_id').loc[z['site_ids']].reset_index()
    fig_network(sites, demand)
    fig_coverage(sites, demand, z['scores'])
    fig_error_adjusted()
    fig_robustness()
    fig_model_layers()
    fig_new_site_test()
    fig_aerial(pd.read_csv(ROOT / 'results/site_audit/site_audit.csv'))


if __name__ == '__main__':
    main()
