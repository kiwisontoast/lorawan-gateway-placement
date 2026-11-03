"""Binary integer programs for gateway placement with redundancy (SciPy / HiGHS).

x_j = 1 selects site j; y_i = 1 marks cell i as covered, which requires at least
r selected sites whose matrix entry a_ij is true:  r * y_i <= sum_j a_ij x_j.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, eye, hstack, vstack


def solve_cover(matrix, mode, r=1, target=None, budget=None, allowed=None, preserve=None, time_limit=120):
    """mode 'min_sites': fewest sites with at least `target` covered cells (and every `preserve` cell).
    mode 'max_coverage': most covered cells with at most `budget` sites.
    `allowed` restricts usable sites; returns (scipy result, selected mask, covered mask)."""
    a = np.asarray(matrix, dtype=bool)
    m, n = a.shape
    allowed = np.ones(n, bool) if allowed is None else np.asarray(allowed, bool)
    preserve = np.zeros(m, bool) if preserve is None else np.asarray(preserve, bool)
    if preserve.any() and (a[preserve][:, allowed].sum(axis=1) < r).any():
        return None, None, None  # some preserved cell cannot reach r allowed sites: infeasible
    link = hstack([-csr_matrix(a.astype(float)), r * eye(m, format='csr')], format='csr')
    rows, upper = [link], [np.zeros(m)]
    if mode == 'min_sites':
        objective = np.r_[np.ones(n), np.zeros(m)]
        if target is not None:
            rows.append(csr_matrix(np.r_[np.zeros(n), -np.ones(m)].reshape(1, -1)))
            upper.append([-float(target)])
    elif mode == 'max_coverage':
        if budget is None:
            raise ValueError('budget required')
        objective = np.r_[np.zeros(n), -np.ones(m)]
        rows.append(csr_matrix(np.r_[np.ones(n), np.zeros(m)].reshape(1, -1)))
        upper.append([float(budget)])
    else:
        raise ValueError(mode)
    lower = np.zeros(n + m)
    lower[n:] = preserve.astype(float)
    ub = np.r_[allowed.astype(float), np.ones(m)]
    res = milp(objective, integrality=np.ones(n + m), bounds=Bounds(lower, ub),
               constraints=LinearConstraint(vstack(rows, format='csr'), -np.inf, np.concatenate(upper)),
               options={'time_limit': time_limit, 'mip_rel_gap': 0.0})
    if res.x is None:
        return res, None, None
    chosen = res.x[:n] > 0.5
    covered = a[:, chosen].sum(axis=1) >= r
    # Verify the solver's answer independently of its own y variables.
    if mode == 'min_sites' and target is not None and res.status == 0 and covered.sum() < target:
        raise ValueError('Selected sites do not reach the target')
    if mode == 'max_coverage' and chosen.sum() > budget:
        raise ValueError('Selected sites exceed the budget')
    if not np.all(covered[preserve]):
        raise ValueError('Selected sites do not preserve required cells')
    if not np.all(allowed[chosen]):
        raise ValueError('Selected a disallowed site')
    return res, chosen, covered


def greedy_cover(matrix, k, r=1, allowed=None):
    """Explainable baseline: repeatedly add the site that completes the most cells (ties by order)."""
    a = np.asarray(matrix, dtype=bool)
    allowed = np.ones(a.shape[1], bool) if allowed is None else np.asarray(allowed, bool).copy()
    counts = np.zeros(a.shape[0], int)
    chosen = []
    for _ in range(k):
        covered = counts >= r
        # Gain = newly r-covered cells; break ties by progress toward r on uncovered cells.
        gain = (a & ((counts + 1) >= r)[:, None] & ~covered[:, None]).sum(axis=0) * 1000 \
            + (a & ~covered[:, None]).sum(axis=0)
        gain[~allowed] = -1
        j = int(np.argmax(gain))
        if gain[j] <= 0:
            break
        allowed[j] = False
        counts += a[:, j]
        chosen.append(j)
    return chosen, counts >= r
