"""Solve illustrative building-site placement as a binary integer program.

The input matrix is a model-score scenario, not observed radio coverage.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix, eye, hstack, vstack

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/bonn_building_placement'


def solve(matrix, mode, target=None, budget=None, time_limit=60):
    m, n = matrix.shape
    # x_j selects sites; y_i marks cells as score-covered.
    link = hstack([-csr_matrix(matrix.astype(float)), eye(m, format='csr')], format='csr')
    if mode == 'max_coverage':
        assert budget is not None
        last = csr_matrix(np.r_[np.ones(n), np.zeros(m)].reshape(1,-1))
        objective = np.r_[np.zeros(n), -np.ones(m)]
        upper = np.r_[np.zeros(m), budget]
    elif mode == 'min_sites':
        assert target is not None
        last = csr_matrix(np.r_[np.zeros(n), -np.ones(m)].reshape(1,-1))
        objective = np.r_[np.ones(n), np.zeros(m)]
        upper = np.r_[np.zeros(m), -target]
    else:
        raise ValueError(mode)
    constraint = LinearConstraint(vstack([link,last],format='csr'), -np.inf, upper)
    return milp(objective, integrality=np.ones(n+m), bounds=Bounds(0,1),
                constraints=constraint, options={'time_limit':time_limit, 'mip_rel_gap':0.0})


def main():
    data = np.load(OUT/'illustrative_coverage_matrix.npz')
    matrix=data['matrix']; site_ids=data['site_ids']; baseline=data['baseline']
    target=int(baseline.sum())
    records=[]; selections=[]
    for mode, kwargs in [('min_sites',{'target':target}),('max_coverage',{'budget':4})]:
        result=solve(matrix,mode,**kwargs)
        row={'problem':mode,'status':int(result.status),'message':str(result.message),
             'success':bool(result.success),'baseline_covered_cells':target,
             'mip_gap':float(result.mip_gap) if getattr(result,'mip_gap',None) is not None else None}
        if result.x is not None:
            chosen=result.x[:matrix.shape[1]] > 0.5
            covered=matrix[:,chosen].any(axis=1) if chosen.any() else np.zeros(matrix.shape[0],dtype=bool)
            row.update(selected_sites=int(chosen.sum()),covered_cells=int(covered.sum()),
                       covered_fraction=float(covered.mean()))
            selections += [{'problem':mode,'site_id':name} for name in site_ids[chosen]]
            if mode=='min_sites' and covered.sum()<target:
                raise ValueError('Solver result does not reach target')
            if mode=='max_coverage' and chosen.sum()>4:
                raise ValueError('Solver result exceeds budget')
        records.append(row)
        print(json.dumps(row),flush=True)
    (OUT/'integer_program_results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    pd.DataFrame(selections,columns=['problem','site_id']).to_csv(OUT/'integer_program_sites.csv',index=False)


if __name__=='__main__':
    main()
