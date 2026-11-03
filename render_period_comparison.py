"""Render where four-site scenarios gain and lose modeled score coverage."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent/'results/bonn_period_study'


def main():
    z=np.load(OUT/'main_coverage_matrix.npz')
    matrix=z['matrix']; ids=z['site_ids']; baseline=z['baseline']
    demand=pd.read_csv(OUT/'demand_cells.csv'); sites=pd.read_csv(OUT/'candidate_sites.csv')
    if not np.array_equal(demand.cell_id.to_numpy(dtype=str),z['cell_ids']):
        raise ValueError('Demand cell order differs from saved matrix')
    selected=pd.read_csv(OUT/'selected_sites.csv')
    max4=selected[(selected.height_offset_m==0)&(selected.score_threshold==0.5)&(selected.budget_k=='exact_4')].site_id
    preservation=pd.read_csv(OUT/'preserve_sites.csv')
    keep4=preservation[preservation.problem=='max_coverage_preserve_budget4'].site_id
    results=[]; summary=[]
    fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    palette={'retained':'#16815e','gained':'#3366a3','lost':'#d36342','uncovered':'#aaaaaa'}
    for ax,(name,names,title) in zip(axes,[('max_coverage',max4,'Maximum total coverage'),
                                            ('preserve_all',keep4,'Preserve all baseline cells')]):
        chosen=np.isin(ids,names)
        if chosen.sum()!=4: raise ValueError('Expected four sites for map')
        new=matrix[:,chosen].any(axis=1)
        status=np.select([baseline & new,~baseline & new,baseline & ~new],['retained','gained','lost'],default='uncovered')
        for label,color in palette.items():
            mask=status==label
            ax.scatter(demand.loc[mask,'x'],demand.loc[mask,'y'],s=11,c=color)
        chosen_ids=ids[chosen]
        loc=sites[sites.site_id.isin(chosen_ids)]
        ax.scatter(loc.x,loc.y,marker='^',s=85,c='#312b80',edgecolors='white',linewidth=.5)
        ax.set(title=title,xlabel='Easting (m)',ylabel='Northing (m)'); ax.set_aspect('equal'); ax.tick_params(axis='x',labelrotation=40)
        for i,cell in enumerate(z['cell_ids']): results.append({'cell_id':cell,'scenario':name,'status':status[i]})
        summary.append({'scenario':name,**{label:int((status==label).sum()) for label in palette},
                        'baseline_covered':int(baseline.sum()),'new_covered':int(new.sum())})
    legend=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,label=label.title(),markersize=7) for label,c in palette.items()]
    legend.append(Line2D([0],[0],marker='^',color='w',markerfacecolor='#312b80',label='Selected site',markersize=8))
    fig.legend(handles=legend,loc='outside lower center',ncol=5)
    fig.suptitle('Four-site scenarios compared with existing eight; © OpenStreetMap contributors')
    fig.savefig(OUT/'coverage_changes.png',dpi=160); plt.close(fig)
    pd.DataFrame(results).to_csv(OUT/'cell_coverage_changes.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'coverage_change_summary.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':
    main()
