"""Error-adjusted coverage: how many sites are needed once new-site prediction
error, measured by leave-one-gateway-out footprint tests, is taken into account?

Run after placement_v2.py. Two error models perturb each design's binary matrix:
  pooled: every predicted-covered cell is real with the pooled held-out precision,
          every predicted-uncovered cell is secretly covered with the pooled
          false-omission rate; cells are independent.
  site:   each selected site draws the (precision, false-omission) profile of one
          of the eight held-out gateways, so whole footprints can be worse
          (like 06df) or better together.
The same perturbation is applied to existing gateways, i.e. every site is treated
as if the model had never seen it, which is the fair basis for comparison.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models_v2 import FINAL
from optimize_v2 import solve_cover

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/v2_placement'
PROFILES = ROOT / 'results/v2_models/footprint_metrics.csv'
THRESHOLD = 0.5
SIMS = 2000
K_MAX = 10


def gateway_profiles(path=PROFILES):
    f = pd.read_csv(path, dtype={'held_out_gateway': str})
    f = f[(f.model == FINAL[0]) & (f.features == FINAL[1]) & (f.prediction_basis == 'rep_p')].copy()
    tp = f.precision * f.predicted_covered
    fn = f.observed_covered - tp
    tn = f.cells - f.predicted_covered - fn
    f['false_omission'] = fn / (fn + tn)
    pooled = (tp.sum() / f.predicted_covered.sum(), fn.sum() / (fn + tn).sum())
    return f[['held_out_gateway', 'precision', 'false_omission']], pooled


def simulate(sub, r, profiles, pooled, rng, mode):
    n = sub.shape[1]
    draws = np.empty(SIMS)
    for s in range(SIMS):
        if mode == 'pooled':
            prec = np.full(n, pooled[0]); fo = np.full(n, pooled[1])
        else:
            pick = rng.integers(0, len(profiles), n)
            prec = profiles.precision.to_numpy()[pick]; fo = profiles.false_omission.to_numpy()[pick]
        u = rng.random(sub.shape)
        real = np.where(sub, u < prec[None, :], u < fo[None, :])
        draws[s] = (real.sum(axis=1) >= r).sum()
    return draws


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT, help='placement_v2.py output folder to read (and write)')
    ap.add_argument('--profiles', type=Path, default=PROFILES, help='footprint metrics giving the error profiles')
    ap.add_argument('--write-to', type=Path, help='folder for the outputs, if not --out')
    args = ap.parse_args()
    absolute = lambda p: p if p.is_absolute() else ROOT / p
    z = np.load(absolute(args.out) / 'score_matrix.npz')
    OUT = absolute(args.write_to or args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    a = z['scores'] >= THRESHOLD
    kinds = np.array([s.split('_')[0] for s in z['site_ids']])
    existing = kinds == 'existing'
    profiles, pooled = gateway_profiles(absolute(args.profiles))
    rng = np.random.default_rng(11)
    rows = []
    for r in (1, 2):
        for mode in ('pooled', 'site'):
            base = simulate(a[:, existing], r, profiles, pooled, rng, mode)
            rows.append({'redundancy': r, 'error_model': mode, 'pool': 'existing_all', 'budget': int(existing.sum()),
                         'modelled_cells': int((a[:, existing].sum(axis=1) >= r).sum()),
                         'mean': base.mean(), 'p05': np.percentile(base, 5), 'p95': np.percentile(base, 95),
                         'prob_beats_existing_mean': np.nan})
            for pool, allowed in (('greenfield', ~existing), ('full_pool', None)):
                for k in range(1, K_MAX + 1):
                    res, chosen, covered = solve_cover(a, 'max_coverage', r=r, budget=k, allowed=allowed)
                    sim = simulate(a[:, chosen], r, profiles, pooled, rng, mode)
                    rows.append({'redundancy': r, 'error_model': mode, 'pool': pool, 'budget': k,
                                 'modelled_cells': int(covered.sum()), 'mean': sim.mean(),
                                 'p05': np.percentile(sim, 5), 'p95': np.percentile(sim, 95),
                                 'prob_beats_existing_mean': float((sim >= base.mean()).mean()),
                                 'optimal': bool(res.status == 0)})
            print(f'r={r} {mode} done', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'error_adjusted_curves.csv', index=False, float_format='%.4f')
    summary = []
    for (r, mode, pool), g in out[out.pool != 'existing_all'].groupby(['redundancy', 'error_model', 'pool']):
        base = out[(out.pool == 'existing_all') & (out.redundancy == r) & (out.error_model == mode)].iloc[0]
        modelled_k = g[g.modelled_cells >= base.modelled_cells].budget.min()
        mean_k = g[g['mean'] >= base['mean']].budget.min()
        sure_k = g[g.p05 >= base['mean']].budget.min()
        summary.append({'redundancy': r, 'error_model': mode, 'pool': pool,
                        'existing_modelled_cells': int(base.modelled_cells), 'existing_error_adjusted_mean': base['mean'],
                        'sites_to_match_modelled': modelled_k, 'sites_to_match_error_adjusted_mean': mean_k,
                        'sites_for_95pct_confidence': sure_k})
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / 'error_adjusted_summary.csv', index=False, float_format='%.1f')
    print(summary.to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, r in zip(axes, (1, 2)):
        for pool, color in (('greenfield', '#2b6cb5'), ('full_pool', '#3b8a4a')):
            g = out[(out.redundancy == r) & (out.error_model == 'site') & (out.pool == pool)]
            ax.plot(g.budget, g.modelled_cells, color=color, linestyle=':', label=f'{pool}, modelled')
            ax.plot(g.budget, g['mean'], color=color, marker='o', label=f'{pool}, error-adjusted mean')
            ax.fill_between(g.budget, g.p05, g.p95, color=color, alpha=.15)
        b = out[(out.redundancy == r) & (out.error_model == 'site') & (out.pool == 'existing_all')].iloc[0]
        ax.axhline(b['mean'], color='black', linestyle='--', label='8 existing, error-adjusted mean')
        ax.axhline(b.modelled_cells, color='black', linestyle=':', linewidth=.8, label='8 existing, modelled')
        ax.set(xlabel='Gateways', ylabel=f'Cells covered by >= {r} gateway(s), of 392', title=f'r = {r}; site-level error model, 5-95% band')
    axes[0].legend(fontsize=7, loc='lower right')
    fig.savefig(OUT / 'error_adjusted_curves.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
