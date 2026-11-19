"""Audit archived random seeds, Nanchang roles, events and Fig. 7 bootstrap."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from jiangxi_radio.propagation_events import build_propagation_events

ROOT=Path('outputs/reproducibility_R2_6_20261107')
SOURCE=Path('/home/wuybwio/projects/guihua/outputs')

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ROOT.mkdir(exist_ok=True,parents=True)
    rows=[]
    for cohort in ['cmf_peno_rf_v3_province','cmf_peno_rf_v3_loso']:
        for p in sorted((SOURCE/cohort).glob('*_seed*/*/training.json')):
            m=json.loads(p.read_text());args=m['arguments'];seed=int(args['seed'])
            assert seed==int(p.parent.parent.name.rsplit('seed',1)[1])
            rows.append(dict(cohort=cohort,run=p.parent.parent.name,variant=p.parent.name,
                             seed=seed,dataset_sha256=m['dataset_sha256'],training_metadata_sha256=digest(p)))
    seeds=pd.DataFrame(rows)
    assert len(seeds[seeds.cohort=='cmf_peno_rf_v3_province'])==135
    assert len(seeds[seeds.cohort=='cmf_peno_rf_v3_loso'])==12
    assert set(seeds.seed)=={2026,2027,2028}
    joint=[]
    for p in sorted(Path('outputs/joint_ablation_20261106/runs').glob('*_seed*/no_multilevel_no_spectral/training.json')):
        m=json.loads(p.read_text());seed=int(m['arguments']['seed'])
        assert seed==int(p.parent.parent.name.rsplit('seed',1)[1])
        joint.append(dict(cohort='revision_joint_removal',run=p.parent.parent.name,variant=p.parent.name,
                          seed=seed,dataset_sha256=m['dataset_sha256'],training_metadata_sha256=digest(p)))
    assert len(joint)==15 and {j['seed'] for j in joint}=={2026,2027,2028}
    seeds=pd.concat([seeds,pd.DataFrame(joint)],ignore_index=True)
    seeds.to_csv(ROOT/'training_seed_audit.csv',index=False)
    roles=[]; generation=[]
    for scene in ['west','center','east2','north']:
        with np.load(SOURCE/'nc_cross_scene'/scene/'dataset_fabdem.npz') as n:
            source_split=n['split'];m=json.loads(str(n['metadata_json'].item()))
            assert len(source_split)==30
            generation.append(dict(scene=scene,seed=m['arguments']['seed'],
                                   original_train=int((source_split==0).sum()),
                                   original_val=int((source_split==1).sum()),
                                   original_test=int((source_split==2).sum())))
    for held in ['west','center','east2','north']:
        with np.load(SOURCE/'cmf_peno_rf_loso/datasets'/(held+'_rf_v2_s32.npz')) as n:
            sceneids=n['scene_id'].astype(str);split=n['split'];src=n['source_station_index']
            for scene in sorted(set(sceneids)):
                ix=np.flatnonzero(sceneids==scene)
                with np.load(SOURCE/'nc_cross_scene'/scene/'dataset_fabdem.npz') as original:
                    expected=original['split'][src[ix]].copy()
                if scene==held:expected[:]=2
                else:expected[expected==2]=1
                assert np.array_equal(split[ix],expected)
                counts=[int((split[ix]==v).sum()) for v in [0,1,2]]
                assert counts==([0,0,30] if scene==held else [18,12,0])
                roles.append(dict(held_out_scene=held,source_scene=scene,train=counts[0],
                                  validation=counts[1],test=counts[2]))
            for seed in [2026,2027,2028]:
                for part,value in [('val',1),('test',2)]:
                    p=SOURCE/'cmf_peno_rf_v3_loso'/f'{held}_seed{seed}'/'full_v3'/('eval_'+part)/'predictions.npz'
                    with np.load(p) as pred:assert np.array_equal(pred['case_indices'],np.flatnonzero(split==value))
    pd.DataFrame(roles).to_csv(ROOT/'nanchang_fold_roles_verified.csv',index=False)
    pd.DataFrame(generation).to_csv(ROOT/'nanchang_source_sampling_seeds.csv',index=False)

    stats=SOURCE/'cmf_peno_rf_v3_province/province_statistics'
    scene=pd.read_csv(stats/'province_scene_metrics.csv')
    paired=pd.read_csv(stats/'province_paired_ablations.csv')
    ref=scene[scene.variant=='full_v3'].set_index(['geo_group','scene_id'])
    higher={'coverage_iou','blind_iou','balanced_accuracy','coverage_precision','coverage_recall','blind_recall'}
    checks=[];group_rows=[]
    for j,r in paired.iterrows():
        other=scene[scene.variant==r.variant].set_index(['geo_group','scene_id'])
        delta=other[r.metric]-ref[r.metric]
        improvement=(other[r.metric].abs()-ref[r.metric].abs()) if r.metric=='bias_db' else (-delta if r.metric in higher else delta)
        groups=improvement.groupby(level='geo_group').mean().sort_index()
        assert len(groups)==5 and len(improvement)==20
        seed=2026+j
        rng=np.random.default_rng(seed)
        means=groups.to_numpy()[rng.integers(0,5,size=(10000,5))].mean(axis=1)
        lo,hi=np.quantile(means,[.025,.975])
        assert np.allclose([lo,hi],[r.paired_ci95_low,r.paired_ci95_high],atol=1e-13,rtol=1e-12),(j,r.variant,r.metric,seed)
        assert np.isclose(improvement.mean(),r.mean_improvement_reference_over_variant,atol=1e-13)
        checks.append(dict(archive_row_zero_based=j,variant=r.variant,metric=r.metric,
                           bootstrap_seed=seed,resamples=10000,groups_per_resample=5,
                           low=float(lo),high=float(hi),archive_reproduced=True))
        for group,value in groups.items():group_rows.append(dict(variant=r.variant,metric=r.metric,geo_group=group,paired_improvement=value))
    pd.DataFrame(checks).to_csv(ROOT/'fig7_bootstrap_seed_manifest.csv',index=False)
    pd.DataFrame(group_rows).to_csv(ROOT/'fig7_paired_group_source.csv',index=False)
    pd.DataFrame(checks).query("metric in ['mae_db','rmse_db','blind_iou','balanced_accuracy']").to_csv(ROOT/'fig7_displayed_interval_verification.csv',index=False)
    event_checks=[]
    for cohort,path in [('Jiangxi',SOURCE/'jiangxi_province_validation/folds/region_01/cmf_peno_rf_s32.npz'),
                        ('Nanchang',SOURCE/'cmf_peno_rf_loso/datasets/center_rf_v2_s32.npz')]:
        with np.load(path) as n:
            k=0
            # One archived case per cohort is sufficient to test the deterministic
            # extraction contract, while all Nanchang splits are verified above.
            ef=n['event_features'][k];et=n['event_type'][k];em=n['event_mask'][k]
            data={v:n[v][k] for v in ['x_m','y_m','z_m','terrain_m','surface_m','station_xyz_m']}
        fresh=build_propagation_events(data['x_m'],data['y_m'],data['z_m'],data['terrain_m'],
            data['surface_m'],data['station_xyz_m'],samples=32,max_events=4,frequency_mhz=2500)
        assert np.allclose(ef[:,:12],fresh['event_features'],rtol=1e-6,atol=1e-6),cohort
        assert np.array_equal(et,fresh['event_type']) and np.array_equal(em,fresh['event_mask'])
        assert np.array_equal(ef[:,13],em.astype(np.float32))
        assert all(np.all(ef[k,12]==k/4) for k in range(4))
        assert np.min(ef[:,3])>=-10 and np.max(ef[:,3])<=10
        assert np.min(ef[:,11])>=0 and np.max(ef[:,11])<=1
        event_checks.append(dict(cohort=cohort,case_index=0,base_features_reconstructed=True,
                                 event_rank_verified=True,event_validity_verified=True))
    result=dict(training_runs_verified=len(seeds),original_training_runs=147,revision_training_runs=15,
                training_seeds=[2026,2027,2028],
                nanchang_fold_scene_roles_verified=len(roles),nanchang_validation_test_archives_verified=24,
                nanchang_source_sampling=generation,
                paired_bootstrap_rows_reproduced=len(checks),bootstrap_base_seed=2026,
                bootstrap_replicates=10000,seed_schedule='2026 + zero-based row in the archived paired-results CSV',
                event_extraction_checks=event_checks,paired_source_sha256=digest(stats/'province_paired_ablations.csv'))
    (ROOT/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
