"""Time-specific Bonn simulation with interior OSM building points.

This is an exploratory conditional-score scenario, not delivery probability.
OSM data: © OpenStreetMap contributors, ODbL: openstreetmap.org/copyright.
"""
import json
import site
from pathlib import Path
ROOT=Path(__file__).resolve().parent
site.addsitedir(str(ROOT/'.geo_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform as transform_geometry
from sklearn.metrics import roc_auc_score, brier_score_loss
from reception_bonn import build_pairs, model, MODELS
from placement_bonn import make_points, score_pairs, greedy
from optimize_building_sites import solve

OUT=ROOT/'results/bonn_period_study'
START='2020-11-10'; END='2021-01-26'
GATEWAYS=['015c','0172','01c6','06df','08d5','08fc','090e','0917']
THRESHOLDS=[0.3,0.5,0.7]; OFFSETS=[-2,0,2]; K_MAX=8


def build_sites():
    sites=pd.read_csv(ROOT/'results/bonn_building_placement/candidate_sites.csv')
    existing=sites[(sites.kind=='existing') & sites.site_id.str.replace('existing_','',regex=False).isin(GATEWAYS)].copy()
    if len(existing)!=len(GATEWAYS):
        raise ValueError('Period gateway list does not match candidate sites')
    old=sites[(sites.kind=='building') & sites.site_id.str.startswith('osm_way_')].copy()
    shapes=json.loads((ROOT/'data/osm/candidate_footprints.json').read_text(encoding='utf-8'))
    ways={f"osm_way_{item['id']}": item for item in shapes['elements']}
    projector=Transformer.from_crs(4326,25832,always_xy=True)
    rows=[]; invalid=[]
    for _, item in old.iterrows():
        feature=ways.get(item.site_id)
        if not feature or not feature.get('geometry'):
            invalid.append(item.site_id); continue
        polygon=Polygon([(v['lon'],v['lat']) for v in feature['geometry']])
        if not polygon.is_valid or polygon.is_empty or polygon.area==0:
            invalid.append(item.site_id); continue
        interior=polygon.representative_point()
        if not polygon.contains(interior):
            raise ValueError('Candidate not inside its footprint')
        area=transform_geometry(projector.transform,polygon).area
        if area < 100:
            continue
        x,y=projector.transform(interior.x,interior.y)
        row=item.to_dict(); row.update(longitude=interior.x,latitude=interior.y,x=x,y=y,footprint_area_m2=area)
        rows.append(row)
    buildings=pd.DataFrame(rows)
    if buildings.empty:
        raise ValueError('No valid building footprints available')
    with rasterio.open(ROOT/'data/terrain/bonn_dgm_20m.tif') as ground, rasterio.open(ROOT/'data/terrain/bonn_dom_20m.tif') as surface:
        if ground.transform != surface.transform or ground.crs != surface.crs or ground.shape != surface.shape:
            raise ValueError('Raster mismatch')
        x=buildings.x.to_numpy(); y=buildings.y.to_numpy()
        col=np.floor((x-ground.transform.c)/ground.transform.a).astype(int)
        row=np.floor((y-ground.transform.f)/ground.transform.e).astype(int)
        inside=(col>=0)&(col<ground.width)&(row>=0)&(row<ground.height)
        g=ground.read(1,masked=True).filled(np.nan); s=surface.read(1,masked=True).filled(np.nan)
        h=np.full(len(buildings),np.nan); h[inside]=s[row[inside],col[inside]]-g[row[inside],col[inside]]
    buildings['surface_height_proxy_m']=h
    buildings=buildings[buildings.surface_height_proxy_m.between(10,28)].copy()
    buildings['antenna_height']=buildings.surface_height_proxy_m+2
    buildings['position_method']='point inside OSM footprint'
    existing['position_method']='original Bonn gateway coordinate'
    sites=pd.concat([existing,buildings],ignore_index=True)
    return sites,{'initial_building_ways':len(old),'invalid_or_missing_ways':len(invalid),
                  'building_ways_after_area_and_height_filters':len(buildings),
                  'relations_excluded':int((pd.read_csv(ROOT/'results/bonn_building_placement/candidate_sites.csv').site_id.str.startswith('osm_relation_')).sum())}


def validate_period(pairs):
    rows=[]
    for gateway in GATEWAYS:
        test=pairs.gw==gateway
        held=pairs.loc[test,['packet_id','heard']].set_index('packet_id').heard
        other=pairs.n_heard.to_numpy()-pairs.packet_id.map(held).to_numpy()
        train_mask=(~test.to_numpy()) & pairs.same_day.to_numpy() & ((other-pairs.heard.to_numpy())>0)
        test_mask=test.to_numpy() & pairs.same_day.to_numpy() & (other>0)
        train=pairs.loc[train_mask]; hold=pairs.loc[test_mask]
        if hold.heard.nunique()!=2 or train.heard.nunique()!=2:
            raise ValueError('Validation requires both labels')
        fit=model().fit(train[MODELS['terrain_surface']],train.heard)
        pred=fit.predict_proba(hold[MODELS['terrain_surface']])[:,1]
        rows.append({'held_out_gateway':gateway,'train_pairs':len(train),'test_pairs':len(hold),
                     'received':int(hold.heard.sum()),'inferred_misses':int((1-hold.heard).sum()),
                     'auc':roc_auc_score(hold.heard,pred),'brier':brier_score_loss(hold.heard,pred)})
        print('Period validation gateway',gateway,flush=True)
    return pd.DataFrame(rows)


def result_row(res, matrix, n, mode):
    out={'status':int(res.status),'message':str(res.message),'optimal':bool(res.status==0),
         'mip_gap':float(res.mip_gap) if getattr(res,'mip_gap',None) is not None else None}
    if res.x is not None:
        pick=res.x[:n]>0.5
        covered=matrix[:,pick].any(axis=1) if pick.any() else np.zeros(matrix.shape[0],dtype=bool)
        out.update(selected_k=int(pick.sum()),covered_cells=int(covered.sum()),covered_fraction=float(covered.mean()))
        if mode=='max' and pick.sum()>8:
            raise ValueError('Invalid budget result')
        return out,pick,covered
    return out,None,None


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=pd.read_csv(ROOT/'data/lora-bonn/survey/samples.csv',dtype={'gw':str})
    within=source[source.transceived_at.between(START,END)]
    if set(within.gw)!=set(GATEWAYS):
        raise ValueError('Unexpected gateways observed in study period; inspect before continuing')
    day_counts=within.groupby('gw').transceived_at.nunique()
    demand,_=make_points(START,END); sites,quality=build_sites()
    demand.to_csv(OUT/'demand_cells.csv',index=False); sites.to_csv(OUT/'candidate_sites.csv',index=False)
    meta={'period_start':START,'period_end':END,'gateway_ids':GATEWAYS,'receptions':len(within),
          'measurement_days':int(within.transceived_at.nunique()),'days_with_receptions_by_gateway':day_counts.to_dict(),
          'demand_cells':len(demand),'candidate_sites':len(sites),**quality}
    (OUT/'study_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Study metadata:',json.dumps(meta),flush=True)
    pairs=build_pairs(START,END,GATEWAYS)
    same_masks=bool((pairs.in_range==pairs.same_day).all())
    print('Activity proxies identical for period:',same_masks,flush=True)
    if not same_masks:
        raise ValueError('Activity proxy assumptions differ; analyze both before continuing')
    cv=validate_period(pairs); cv.to_csv(OUT/'gateway_validation.csv',index=False,float_format='%.6f')
    train=pairs[pairs.same_day & ((pairs.n_heard-pairs.heard)>0)]
    fitted=model().fit(train[MODELS['terrain_surface']],train.heard)
    print('Fitted period classifier on pairs:',len(train),flush=True)
    del pairs,train
    ids=sites.site_id.to_numpy(); is_existing=sites.kind.to_numpy()=='existing'
    rows=[]; opt_rows=[]; selections=[]; main_results={}
    for offset in OFFSETS:
        temp=sites.copy(); temp.loc[temp.kind=='building','antenna_height']+=offset
        paths=score_pairs(demand,temp,16)
        scores=fitted.predict_proba(paths[MODELS['terrain_surface']])[:,1].reshape(len(demand),len(sites))
        for threshold in THRESHOLDS:
            matrix=scores>=threshold; baseline=matrix[:,is_existing].any(axis=1)
            chosen,curve,cover=greedy(matrix,ids,K_MAX)
            for k in range(1,K_MAX+1):
                n=min(k,len(chosen))
                rows.append({'height_offset_m':offset,'score_threshold':threshold,'budget_k':k,'selected_k':n,
                             'greedy_fraction':curve[n-1] if n else 0,'existing_fraction':baseline.mean(),
                             'clipped_link_count':int(paths.distance_was_clipped.sum())})
                for rank,name in enumerate(chosen[:n],1):
                    selections.append({'height_offset_m':offset,'score_threshold':threshold,'budget_k':k,'rank':rank,'site_id':name})
            exact=solve(matrix,'min_sites',target=int(baseline.sum()),time_limit=30)
            outcome,pick,_=result_row(exact,matrix,len(sites),'min')
            outcome.update(height_offset_m=offset,score_threshold=threshold,problem='min_sites',budget=None,
                           baseline_cells=int(baseline.sum()))
            if pick is not None:
                if outcome['covered_cells'] < baseline.sum():
                    raise ValueError('Minimum-site solution does not match baseline')
                selections += [{'height_offset_m':offset,'score_threshold':threshold,'budget_k':'exact_min','rank':rank,'site_id':name}
                               for rank,name in enumerate(ids[pick],1)]
            opt_rows.append(outcome)
            if offset==0 and threshold==0.5:
                main_results={'matrix':matrix,'baseline':baseline,'score_threshold':threshold,'greedy_curve':curve}
                np.savez_compressed(OUT/'main_coverage_matrix.npz',matrix=matrix,site_ids=ids.astype(str),
                                    cell_ids=demand.cell_id.to_numpy(dtype=str),baseline=baseline)
                for k in range(1,K_MAX+1):
                    exact_k=solve(matrix,'max_coverage',budget=k,time_limit=30)
                    r,picked,covered=result_row(exact_k,matrix,len(sites),'max')
                    r.update(height_offset_m=offset,score_threshold=threshold,problem='max_coverage',budget=k,
                             baseline_cells=int(baseline.sum()))
                    if picked is not None:
                        if picked.sum()>k: raise ValueError('Max-coverage solution exceeds budget')
                        selections += [{'height_offset_m':offset,'score_threshold':threshold,'budget_k':f'exact_{k}',
                                        'rank':rank,'site_id':name} for rank,name in enumerate(ids[picked],1)]
                        if k==4: main_results['selected_4']=picked; main_results['coverage_4']=covered
                    opt_rows.append(r)
        print('Evaluated height offset',offset,flush=True)
    result=pd.DataFrame(rows); opt=pd.DataFrame(opt_rows)
    result.to_csv(OUT/'scenario_results.csv',index=False,float_format='%.6f')
    opt.to_csv(OUT/'integer_results.csv',index=False,float_format='%.6f')
    pd.DataFrame(selections).to_csv(OUT/'selected_sites.csv',index=False)
    main=result[(result.height_offset_m==0)&(result.score_threshold==.5)]
    maxes=opt[(opt.problem=='max_coverage')&(opt.height_offset_m==0)&(opt.score_threshold==.5)].sort_values('budget')
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    ax.plot(main.budget_k,100*main.greedy_fraction,marker='o',label='Greedy')
    ax.plot(maxes.budget,100*maxes.covered_fraction,marker='x',label='Integer solver (check status)')
    ax.axhline(100*main.existing_fraction.iloc[0],color='red',linestyle='--',label='Eight existing sites')
    ax.set(xlabel='Gateway budget',ylabel='Cells meeting score rule (%)',ylim=(0,105),title='Period-specific simulated score coverage')
    ax.legend(); fig.savefig(OUT/'coverage_curve.png',dpi=160); plt.close(fig)
    if 'selected_4' in main_results:
        fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
        for ax,name,marker in [(axes[0],'baseline',is_existing),(axes[1],'coverage_4',main_results['selected_4'])]:
            ax.scatter(demand.x,demand.y,c=np.where(main_results[name],'#137b5b','#dc8051'),s=9)
            ax.scatter(sites.loc[marker,'x'],sites.loc[marker,'y'],marker='^',s=65,c='#294ea1',edgecolors='white')
            ax.set(title='Eight observed gateways' if name=='baseline' else 'Integer model: four sites',xlabel='Easting (m)',ylabel='Northing (m)')
            ax.tick_params(axis='x',labelrotation=40); ax.set_aspect('equal')
        fig.suptitle('Simulated scores; © OpenStreetMap contributors')
        fig.savefig(OUT/'placement_map.png',dpi=160); plt.close(fig)
    print('Period held-out gateway AUC range/mean:',cv.auc.min(),cv.auc.max(),cv.auc.mean())
    print('Main scenario:',main[['budget_k','greedy_fraction','existing_fraction']].to_string(index=False))
    print('Integer solver statuses:',opt.status.value_counts().to_dict())


if __name__=='__main__':
    main()
