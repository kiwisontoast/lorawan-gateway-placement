"""Optimize while preserving every baseline-covered cell in the scenario matrix."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix, eye, hstack, vstack

OUT=Path(__file__).resolve().parent/'results/bonn_period_study'


def solve_preserving(matrix, baseline, budget=None, time_limit=60):
    m,n=matrix.shape
    link=hstack([-csr_matrix(matrix.astype(float)),eye(m,format='csr')],format='csr')
    lo=np.zeros(n+m); lo[n:]=baseline.astype(float)
    bounds=Bounds(lo,np.ones(n+m))
    if budget is None:
        objective=np.r_[np.ones(n),np.zeros(m)]
        constraint=LinearConstraint(link,-np.inf,np.zeros(m))
    else:
        budget_row=csr_matrix(np.r_[np.ones(n),np.zeros(m)].reshape(1,-1))
        constraint=LinearConstraint(vstack([link,budget_row],format='csr'),-np.inf,np.r_[np.zeros(m),budget])
        objective=np.r_[np.zeros(n),-np.ones(m)]
    return milp(objective,integrality=np.ones(n+m),bounds=bounds,constraints=constraint,
                options={'time_limit':time_limit,'mip_rel_gap':0.0})


def main():
    z=np.load(OUT/'main_coverage_matrix.npz')
    matrix=z['matrix']; baseline=z['baseline']; ids=z['site_ids']; n=len(ids)
    records=[]; selections=[]
    for problem,budget in [('min_sites_preserve',None),('max_coverage_preserve_budget4',4)]:
        res=solve_preserving(matrix,baseline,budget)
        row={'problem':problem,'status':int(res.status),'message':str(res.message),
             'optimal':bool(res.status==0),
             'mip_gap':float(res.mip_gap) if getattr(res,'mip_gap',None) is not None else None}
        if res.x is not None:
            selected=res.x[:n]>0.5
            covered=matrix[:,selected].any(axis=1) if selected.any() else np.zeros(len(baseline),dtype=bool)
            if not np.all(covered[baseline]): raise ValueError('Selected sites do not preserve baseline cells')
            if budget is not None and selected.sum()>budget: raise ValueError('Selected sites exceed budget')
            row.update(selected_sites=int(selected.sum()),covered_cells=int(covered.sum()),
                       additional_cells=int((covered & ~baseline).sum()),lost_cells=int((~covered & baseline).sum()))
            selections += [{'problem':problem,'site_id':s} for s in ids[selected]]
        records.append(row); print(json.dumps(row),flush=True)
    (OUT/'preserve_results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    pd.DataFrame(selections,columns=['problem','site_id']).to_csv(OUT/'preserve_sites.csv',index=False)


if __name__=='__main__':
    main()
