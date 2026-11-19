"""Prespecified sparse-RT adaptation versus equally observed analytical controls."""
from pathlib import Path
import argparse, copy, hashlib, json, random, time
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from benchmark_reference_vs_network import build_model
from jiangxi_radio.volume_simulation import input_volume

def make_cache(root,project):
    file=root/'adaptation_inputs.npz'
    if file.exists():return
    manifest=json.loads((root/'sampling_manifest.json').read_text());ids=np.array(sorted(r['case_index'] for r in manifest))
    source=project/'outputs/jiangxi_province_validation/folds/region_01/cmf_peno_rf_s32.npz'
    with np.load(source,allow_pickle=False) as archive:
        keys=['x_m','y_m','z_m','terrain_m','surface_m','air_mask','station_xyz_m','coarse_dbm','event_features','event_type','event_mask','scene_id','geo_group']
        data={k:archive[k][ids] for k in keys}
    volume=np.stack([input_volume(data,i) for i in range(len(ids))])
    np.savez_compressed(file,case_indices=ids,volume=volume,**data)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--project',type=Path,default=Path('/home/wuybwio/projects/guihua'))
    ap.add_argument('--fold',default=None);args=ap.parse_args();root=args.output_dir.resolve()
    protocol=json.loads((root/'protocol.json').read_text());settings=protocol['sparse_adaptation']
    lock=json.loads((root/'protocol_lock.json').read_text())
    assert hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()==lock['protocol_sha256']
    results=sorted((root/'rt_results/primary').glob('*.npz'));assert len(results)==20
    make_cache(root,args.project)
    data=dict(np.load(root/'adaptation_inputs.npz',allow_pickle=False));ids=data['case_indices'];lookup={int(v):i for i,v in enumerate(ids)}
    amendment=json.loads((root/'adaptation_mask_amendment.json').read_text())
    assert amendment['fit_only_resolved_paths']
    target=np.zeros((60,192),np.float32);query=np.zeros((60,192),np.int64);valid=np.zeros((60,192),bool)
    for p in results:
        with np.load(p,allow_pickle=False) as a:
            for j,case in enumerate(a['case_indices']):
                i=lookup[int(case)];target[i]=np.maximum(31+10*np.log10(np.maximum(a['path_gain_linear'][j],1e-30)),-140.)
                valid[i]=a['path_gain_linear'][j]>0
                query[i]=a['query_flat_indices'][j]
    exclusions=json.loads((root/'geometry_exclusions.json').read_text())
    for case in exclusions['excluded_case_indices']:valid[lookup[case]]=False
    controls={k:np.zeros_like(target) for k in ['FSPL','internal_nominal']}
    for p in sorted((root/'comparison_inputs').glob('*.npz')):
        with np.load(p,allow_pickle=False) as a:
            for j,case in enumerate(a['case_indices']):
                for k in controls:controls[k][lookup[int(case)]]=a[k][j]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    # Full scene geometry and event descriptors are inputs; only the 192 RT query values provide loss.
    tensors=[torch.from_numpy(data['volume']),torch.from_numpy(data['coarse_dbm'][:,None]),
             torch.from_numpy(data['event_features']),torch.from_numpy(data['event_type']).long(),torch.from_numpy(data['event_mask']).bool()]
    q=torch.from_numpy(query).to(device);truth=torch.from_numpy(target).to(device);valid_tensor=torch.from_numpy(valid).to(device)
    out=root/'sparse_adaptation';out.mkdir(exist_ok=True)
    # Secondary controls were registered before any adaptation fit; retain the
    # original neural/affine protocol rather than rewriting it retrospectively.
    control_lock=json.loads((root/'non_neural_control_protocol.json').read_text())
    features=[]
    for i in range(60):
        channels=np.concatenate([data['volume'][i],data['event_features'][i].reshape(56,16,24,24),
            data['event_type'][i].astype(np.float32),data['event_mask'][i].astype(np.float32)],axis=0)
        features.append(channels.reshape(len(channels),-1)[:,query[i]].T)
    features=np.asarray(features)
    all_rows=[]
    for fold_index in range(5):
        fold=f'region_{fold_index+1:02d}'
        if args.fold and fold!=args.fold:continue
        if (out/f'{fold}_complete.json').exists():continue
        groups=[f'region_{j+1:02d}' for j in range(5)]
        testgroup=groups[fold_index];valgroup=groups[(fold_index+1)%5];calgroup=groups[(fold_index+2)%5]
        test=np.flatnonzero(data['geo_group']==testgroup);val=np.flatnonzero(data['geo_group']==valgroup)
        fit=np.flatnonzero(~np.isin(data['geo_group'],[testgroup,valgroup,calgroup]))
        assert (len(fit),len(val),len(test))==(24,12,12)
        usable=valid.any(axis=1)
        fit=fit[usable[fit]];val=val[usable[val]];test=test[usable[test]]
        def equal_scene_score(predictions,indices):
            case_errors=np.array([np.abs(predictions[j]-target[i])[valid[i]].mean() for j,i in enumerate(indices)])
            return float(np.mean([case_errors[data['scene_id'][indices]==scene].mean() for scene in np.unique(data['scene_id'][indices])]))
        def rows_for(pred,name):
            return [dict(fold=fold,case_index=int(ids[i]),scene_id=str(data['scene_id'][i]),method=name,
                         mae_db=float(np.abs(pred[j]-target[i])[valid[i]].mean()),rmse_db=float(np.sqrt(np.mean((pred[j]-target[i])[valid[i]]**2))),
                         bias_db=float((pred[j]-target[i])[valid[i]].mean()),n_queries=int(valid[i].sum())) for j,i in enumerate(test)]
        for name,field in controls.items():
            xf=field[fit][valid[fit]];yf=target[fit][valid[fit]]
            bias=float(np.mean(yf-xf));affine=np.linalg.lstsq(np.stack([xf,np.ones_like(xf)],axis=1),yf,rcond=None)[0]
            for kind,pred in [('frozen',field[test]),('bias',field[test]+bias),('affine',affine[0]*field[test]+affine[1])]:
                all_rows.extend(rows_for(pred,f'{name}_{kind}'))
            (out/f'{fold}_{name}_calibration.json').write_text(json.dumps(dict(bias_db=bias,slope=float(affine[0]),intercept=float(affine[1]),fitting_cases=ids[fit].tolist()),indent=2))
        # All pointwise controls receive the local voxel channels and the full
        # four-slot event descriptors, types and validity mask, unlike CNN/FNO.
        residual=target-controls['FSPL']
        xf=features[fit][valid[fit]];yf=residual[fit][valid[fit]]
        xv=features[val].reshape(-1,features.shape[-1]);xt=features[test].reshape(-1,features.shape[-1])
        for family,grid in [('ridge',control_lock['ridge_alpha']),('hist_gradient_boosting',control_lock['boost_iterations'])]:
            candidates=[]
            with threadpool_limits(limits=1):
                for value in grid:
                    estimator=make_pipeline(StandardScaler(),Ridge(alpha=value)) if family=='ridge' else HistGradientBoostingRegressor(
                        max_iter=value,max_leaf_nodes=15,max_depth=None,learning_rate=.05,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=20261107)
                    estimator.fit(xf,yf)
                    vp=estimator.predict(xv).reshape(len(val),192)+controls['FSPL'][val]
                    candidates.append((equal_scene_score(vp,val),value,estimator))
                best_control=min(candidates,key=lambda t:t[0])
                pred=best_control[2].predict(xt).reshape(len(test),192)+controls['FSPL'][test]
            all_rows.extend(rows_for(pred,family+'_adapted'))
            (out/f'{fold}_{family}_selection.json').write_text(json.dumps(dict(
                parameter_grid=grid,validation_mae=[v[0] for v in candidates],selected_parameter=best_control[1],
                feature_count=int(features.shape[-1]),event_input=True,fitting_cases=ids[fit].tolist(),validation_cases=ids[val].tolist(),test_cases=ids[test].tolist()),indent=2))
        for variant in settings['model_variants']:
            seed=settings['training_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            ckpt=args.project/f'outputs/cmf_peno_rf_v3_province/{fold}_seed2026/{variant}/best.pt'
            checkpoint=torch.load(ckpt,map_location='cpu',weights_only=False);model=build_model(checkpoint,8).to(device)
            model.variance_head.requires_grad_(False);model.log_variance_bias.requires_grad_(False)
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=settings['lr'],weight_decay=settings['weight_decay'])
            def output(i):return model(*[t[i:i+1].to(device) for t in tensors])['mean_dbm'].flatten()[q[i]]
            @torch.inference_mode()
            def evaluate(indices):
                model.eval();return torch.stack([output(int(i)) for i in indices]).cpu().numpy()
            def val_score():
                predictions=evaluate(val)
                # Exactly four scenes, equal weighting after station-case averaging.
                return equal_scene_score(predictions,val)
            best=val_score();best_epoch=0;state=copy.deepcopy(model.state_dict());history=[dict(epoch=0,val_mae=best)]
            all_rows.extend(rows_for(evaluate(test),variant+'_frozen'))
            start=time.perf_counter();rng=np.random.default_rng(seed)
            for epoch in range(1,settings['epochs']+1):
                model.train();losses=[]
                for i in rng.permutation(fit):
                    optimizer.zero_grad(set_to_none=True);pred=output(int(i))
                    residual_valid=(pred-truth[i])[valid_tensor[i]]/20
                    loss=torch.nn.functional.smooth_l1_loss(residual_valid,torch.zeros_like(residual_valid))
                    assert torch.isfinite(loss), (fold,variant,epoch,int(ids[i]))
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip']);optimizer.step();losses.append(float(loss.detach()))
                score=val_score();history.append(dict(epoch=epoch,val_mae=score,train_loss=float(np.mean(losses))))
                assert np.isfinite(score), (fold,variant,epoch)
                if score<best:best=score;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
                print(json.dumps(dict(fold=fold,variant=variant,epoch=epoch,val_mae=score,best_epoch=best_epoch)),flush=True)
            model.load_state_dict(state);pred=evaluate(test);all_rows.extend(rows_for(pred,variant+'_adapted'))
            torch.save(dict(state_dict={k:v.cpu() for k,v in state.items()},initial_checkpoint=str(ckpt),
                            initial_checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),protocol_lock=lock,
                            selected_epoch=best_epoch,validation_mae=best,history=history,resolved_mask_amendment=amendment),out/f'{fold}_{variant}.pt')
            np.savez_compressed(out/f'{fold}_{variant}_heldout.npz',case_indices=ids[test],mean_dbm=pred,target_dbm=target[test],query_flat_indices=query[test],valid_rt=valid[test])
            (out/f'{fold}_{variant}_training.json').write_text(json.dumps(dict(history=history,selected_epoch=best_epoch,validation_mae=best,
                training_seconds=time.perf_counter()-start,fitting_cases=ids[fit].tolist(),validation_cases=ids[val].tolist(),test_cases=ids[test].tolist(),
                excluded_calibration_group=calgroup,seed=seed),indent=2))
            del model,optimizer;torch.cuda.empty_cache()
        frame=pd.DataFrame([r for r in all_rows if r['fold']==fold]);frame.to_csv(out/f'{fold}_case_metrics.csv',index=False)
        (out/f'{fold}_complete.json').write_text(json.dumps(dict(fold=fold,rows=len(frame),protocol_sha256=lock['protocol_sha256']),indent=2))
    frames=[pd.read_csv(p) for p in out.glob('region_*_case_metrics.csv')]
    if frames:pd.concat(frames,ignore_index=True).to_csv(out/'case_metrics.csv',index=False)

if __name__=='__main__':main()
