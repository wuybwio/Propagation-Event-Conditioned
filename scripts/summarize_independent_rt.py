"""Scene/group summaries of the locked independent propagation experiment."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

def score(error,mask):
    e=np.asarray(error)[mask]
    return dict(mae_db=float(np.abs(e).mean()),rmse_db=float(np.sqrt(np.mean(e*e))),bias_db=float(e.mean()),n_queries=len(e)) if len(e) else dict(mae_db=np.nan,rmse_db=np.nan,bias_db=np.nan,n_queries=0)

def summarize(frame,keys,out,prefix):
    metrics=['mae_db','rmse_db','bias_db']
    # Training seeds and station cases never count as geographical replicates.
    cases=frame.groupby(keys+['fold','scene_id','case_index'],as_index=False)[metrics].mean()
    scenes=cases.groupby(keys+['fold','scene_id'],as_index=False)[metrics].mean()
    groups=scenes.groupby(keys+['fold'],as_index=False)[metrics].mean()
    cases.to_csv(out/f'{prefix}_case_metrics.csv',index=False)
    scenes.to_csv(out/f'{prefix}_scene_metrics.csv',index=False)
    groups.to_csv(out/f'{prefix}_group_metrics.csv',index=False)
    overall=scenes.groupby(keys)[metrics].agg(['mean','std','count'])
    overall.columns=['_'.join(c) for c in overall.columns];overall.reset_index().to_csv(out/f'{prefix}_overall.csv',index=False)
    return scenes,groups,overall.reset_index()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--partial',action='store_true');args=ap.parse_args();root=args.output_dir
    resultfiles=sorted((root/'rt_results/primary').glob('*.npz'))
    if not args.partial:assert len(resultfiles)==20,len(resultfiles)
    out=root/'analysis';out.mkdir(exist_ok=True)
    comparison={p.stem:dict(np.load(p,allow_pickle=False)) for p in (root/'comparison_inputs').glob('*.npz')}
    rows=[];diagnostics=[];sensitivity=[]
    exclusions=json.loads((root/'geometry_exclusions.json').read_text())['excluded_case_indices']
    for p in resultfiles:
        a=dict(np.load(p,allow_pickle=False));fold='_'.join(p.stem.split('_')[:2]);c=comparison[fold]
        geometry=dict(np.load(root/'solver_inputs'/p.name,allow_pickle=False));metadata=json.loads(str(a['metadata_json']))
        for j,case in enumerate(a['case_indices']):
            k=int(np.flatnonzero(c['case_indices']==case)[0]);assert np.array_equal(a['query_flat_indices'][j],c['query_flat_indices'][k])
            gain=a['path_gain_linear'][j];resolved=(gain>0)&(int(case) not in exclusions);unfloored=31+10*np.log10(np.maximum(gain,1e-30));target=np.maximum(unfloored,-140)
            agl=a['query_xyz_m'][j,:,2]-geometry['terrain_m'].reshape(-1)[a['query_flat_indices'][j]%(24*24)]
            masks=dict(all=np.ones(192,bool),resolved=resolved,LOS=resolved&a['los'][j],NLOS=resolved&~a['los'][j],low=resolved&(agl<40),middle=resolved&(agl>=40)&(agl<80),high=resolved&(agl>=80))
            diagnostics.append(dict(fold=fold,scene_id=p.stem,case_index=int(case),queries=192,resolved=int(resolved.sum()),los=int(a['los'][j].sum()),
                excluded_source_geometry=int(case) in exclusions,raw_solver_resolved=int((gain>0).sum()),below_floor_resolved=int((resolved&(unfloored < -140)).sum()),solver_seconds=float(a['solver_seconds'][j]),**metadata['geometry']))
            methods=[key for key in c if key not in ['case_indices','scene_id','query_flat_indices']]
            for method in methods:
                name,sep,seed=method.partition('__');prediction=c[method][k]
                for stratum,mask in masks.items():
                    rows.append(dict(fold=fold,scene_id=p.stem,case_index=int(case),method=name,seed=int(seed) if sep else -1,stratum=stratum,floor_dbm=-140,**score(prediction-target,mask)))
                for floor in [-130,-150]:
                    rows.append(dict(fold=fold,scene_id=p.stem,case_index=int(case),method=name,seed=int(seed) if sep else -1,stratum='all',floor_dbm=floor,**score(prediction-np.maximum(unfloored,floor),masks['all'])))
            if j==0:
                for mode in ['convergence','materials']:
                    f=root/'rt_results'/mode/p.name
                    if not f.exists():continue
                    b=dict(np.load(f,allow_pickle=False));assert b['case_indices'][0]==case
                    second=b['path_gain_linear'][0];common=resolved&(second>0)
                    second_db=np.maximum(31+10*np.log10(np.maximum(second,1e-30)),-140)
                    sensitivity.append(dict(fold=fold,scene_id=p.stem,case_index=int(case),mode=mode,n_common=int(common.sum()),
                        primary_resolved=int(resolved.sum()),alternative_resolved=int((second>0).sum()),changed_resolution=int(((second>0)!=resolved).sum()),
                        **score(second_db-target,common),all_query_mae_db=float(np.abs(second_db-target).mean()),
                        max_abs_difference_db=float(np.max(np.abs(second_db[common]-target[common]))) if common.any() else np.nan))
    frame=pd.DataFrame(rows);frame.to_csv(out/'frozen_query_set_metrics.csv',index=False)
    _,_,overall=summarize(frame,['method','stratum','floor_dbm'],out,'frozen')
    pd.DataFrame(diagnostics).to_csv(out/'solver_case_diagnostics.csv',index=False)
    if sensitivity:pd.DataFrame(sensitivity).to_csv(out/'solver_sensitivity.csv',index=False)
    adapfile=root/'sparse_adaptation/case_metrics.csv'
    if adapfile.exists():
        adapted=pd.read_csv(adapfile);scenes,groups,adapt_overall=summarize(adapted,['method'],out,'adaptation')
        paired=[];rng=np.random.default_rng(20261107)
        # Prespecified comparisons; intervals exploratory, no winner-based claim.
        pivot=groups.pivot(index='fold',columns='method',values='mae_db')
        for base in ['internal_nominal_frozen','internal_nominal_affine','FSPL_affine','ridge_adapted','hist_gradient_boosting_adapted','cnn_residual_adapted','fno_residual_adapted','full_v3_frozen']:
            if base not in pivot or 'full_v3_adapted' not in pivot:continue
            difference=(pivot[base]-pivot['full_v3_adapted']).to_numpy()
            bootstrap=difference[rng.integers(0,len(difference),(10000,len(difference)))].mean(axis=1)
            signs=np.array([[1 if code&(1<<j) else -1 for j in range(len(difference))] for code in range(2**len(difference))])
            exact_p=float(np.mean(np.abs((signs*difference).mean(axis=1))>=abs(difference.mean())-1e-12))
            paired.append(dict(comparator=base,contrast='comparator minus RT adapted CMF-PENO-RF',n_groups=len(difference),
                mean_mae_improvement_db=float(difference.mean()),ci_low=float(np.quantile(bootstrap,.025)),ci_high=float(np.quantile(bootstrap,.975)),
                improved_groups=int((difference>0).sum()),two_sided_exact_sign_flip_p=exact_p,group_differences=difference.tolist()))
        (out/'adaptation_paired_contrasts.json').write_text(json.dumps(paired,indent=2))
        print(adapt_overall[['method','mae_db_mean','rmse_db_mean']].to_string(index=False),flush=True)
    record=dict(completed_scenes=len(resultfiles),queries=len(diagnostics)*192,station_cases=len(diagnostics),
        resolved_queries=sum(r['resolved'] for r in diagnostics),LOS_queries=sum(r['los'] for r in diagnostics),
        no_measurements_collected=True,independent_solver_computation=True,source_targets_or_events_read_by_solver=False,
        descriptive_unit='20 scene averages nested in five groups; seeds and three station cases averaged first',
        protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),result_file_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in resultfiles})
    (out/'analysis_manifest.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)
    print(overall.query('stratum=="all" and floor_dbm==-140')[['method','mae_db_mean','rmse_db_mean']].to_string(index=False),flush=True)

if __name__=='__main__':main()
