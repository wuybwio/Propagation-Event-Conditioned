"""Lock geometry-only Sionna RT queries and a sparse adaptation comparison."""
from pathlib import Path
import argparse, datetime, hashlib, json
import numpy as np

VARIANTS=['full_v3','cnn_residual','fno_residual','no_multilevel','no_spectral','deterministic','no_events','no_local','no_event_consistency']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--project',type=Path,default=Path('/home/wuybwio/projects/guihua'))
    args=ap.parse_args();root=args.output_dir
    if (root/'protocol_lock.json').exists():raise FileExistsError('Protocol already locked; use existing artifacts.')
    root.mkdir(parents=True,exist_ok=True);(root/'solver_inputs').mkdir(exist_ok=True)
    (root/'comparison_inputs').mkdir(exist_ok=True)
    protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='independently implemented electromagnetic solver comparison, not measurement validation',
        solver='Sionna RT 2.2.0 / Mitsuba 3.9.1 / Dr.Jit 1.5.0',
        frequency_hz=2.5e9,transmitter_power_dbm=23.,receiver_gain_dbi=8.,
        antennas='single isotropic element, vertical polarization, reciprocal station-to-query links',
        scene_selection='all 20 original geographical test scenes, no outcome-based scene selection',
        sampling_seed=20261107,stations_per_scene=3,query_points_per_station=192,
        query_sampling='64 valid-air queries from each AGL band [2,40), [40,80), [80,inf); fill shortages without replacement from remaining valid points; exclude horizontal boundary centres and distance <5m',
        geometry='archived 24x24 DEM triangulation and raster surface building prisms; common metric coordinate datum; independent electromagnetic computation',
        materials={'primary':{'terrain':'medium_dry_ground','buildings':'concrete'},
                   'sensitivity':{'terrain':'wet_ground','buildings':'brick'}},
        max_depth=3,samples_per_src=262144,max_num_paths_per_src=1000000,
        los=True,specular_reflection=True,diffuse_reflection=False,refraction=False,
        diffraction=True,edge_diffraction=True,diffraction_lit_region=False,
        path_power='incoherent sum of squared path amplitudes; no imposed lognormal shadowing',
        unresolved_path_policy='zero resolved path reported separately; primary dBm floor -140, sensitivity -130 and -150',
        convergence='first selected station in each geographical group: 4x ray samples and max_depth 5; same queries',
        material_sensitivity='first selected station in each geographical group, wet ground and brick',
        frozen_comparison=dict(models=VARIANTS,seeds=[2026,2027,2028],
            controls=['FSPL','dominant_geometry','internal_nominal'],no_fit_to_rt=True),
        sparse_adaptation=dict(prespecified=True,initial_seed=2026,training_seed=20261107,
            model_variants=['full_v3','cnn_residual','fno_residual'],epochs=20,lr=0.0001,
            optimizer='AdamW',weight_decay=0.0001,batch_size=1,gradient_clip=5.,
            objective='SmoothL1((predicted_mean - RT_power)/20) averaged only over identical sampled points',
            architecture='unchanged pretrained mean model; variance frozen; no auxiliary synthetic-label loss',
            selection='lowest equal-scene validation MAE including epoch zero; no test selection',
            roles='original fold split: two groups fitting, next cyclic group validation, separate calibration group unused, one group held-out test',
            controls=['FSPL and nominal reference: uncalibrated, fitted bias, fitted affine slope/intercept'],
            interpretation='new sparse RT adaptation experiment; not an advantage of the original frozen synthetic-only model'),
        units='192 points nested in a station; 3 station cases averaged within scene; 4 scenes nested in each of five groups; seeds averaged before spatial summaries',
        primary_metrics=['MAE_dB','RMSE_dB','bias_dB'],
        secondary_metrics='LOS/NLOS and AGL bands; mean-only exceedance under original threshold/shadowing, explicitly imposed not calibrated',
        model_benefit_claim='conditional on actual held-out results against equally calibrated analytical controls; no assumed benefit or timing equivalence')
    rows=[];source_hashes={};rng=np.random.default_rng(protocol['sampling_seed'])
    for group in range(1,6):
        fold=f'region_{group:02d}';file=args.project/f'outputs/jiangxi_province_validation/folds/{fold}/cmf_peno_rf_s32.npz'
        with np.load(file,allow_pickle=False) as ds:
            scene_ids=ds['scene_id'];splits=ds['split'];test=np.flatnonzero(splits==2)
            xall,yall,zall,terrain,surface,air,stations=[ds[k] for k in ['x_m','y_m','z_m','terrain_m','surface_m','air_mask','station_xyz_m']]
            coarse,dominant,nominal=[ds[k] for k in ['coarse_dbm','teacher_dbm','fidelity_dbm']]
            source_hashes[str(file)]=sha(file)
            chosen=[];flat_queries=[]
            for scene in sorted(set(scene_ids[test])):
                candidates=test[scene_ids[test]==scene]
                cases=np.sort(rng.choice(candidates,3,replace=False))
                queries=[];xyz=[]
                for i in cases:
                    z,y,x=np.meshgrid(zall[i],yall[i],xall[i],indexing='ij')
                    positions=np.stack([x,y,z],axis=-1)
                    agl=z-terrain[i][None]
                    valid=air[i].copy() & (agl>=2) & (z-surface[i][None]>=1)
                    valid[:,:,0]=False;valid[:,:,-1]=False;valid[:,0,:]=False;valid[:,-1,:]=False
                    valid &= np.linalg.norm(positions-stations[i],axis=-1)>=5
                    picked=[]
                    for lo,hi in [(2,40),(40,80),(80,np.inf)]:
                        ids=np.flatnonzero(valid & (agl>=lo)&(agl<hi))
                        picked.extend(rng.choice(ids,min(64,len(ids)),replace=False).tolist())
                    rest=np.setdiff1d(np.flatnonzero(valid),picked)
                    if len(picked)<192:picked.extend(rng.choice(rest,192-len(picked),replace=False).tolist())
                    picked=np.sort(picked).astype(np.int64);assert len(np.unique(picked))==192
                    queries.append(picked);xyz.append(positions.reshape(-1,3)[picked])
                    rows.append(dict(fold=fold,scene_id=str(scene),case_index=int(i),query_count=192))
                inputfile=root/'solver_inputs'/f'{scene}.npz'
                np.savez_compressed(inputfile,scene_id=str(scene),case_indices=cases,
                    x_m=xall[cases[0]],y_m=yall[cases[0]],z_m=zall[cases[0]],
                    terrain_m=terrain[cases[0]],surface_m=surface[cases[0]],
                    station_xyz_m=stations[cases],query_xyz_m=np.asarray(xyz),query_flat_indices=np.asarray(queries))
                source_hashes[str(inputfile)]=sha(inputfile)
                chosen.extend(cases.tolist());flat_queries.extend(queries)
            chosen=np.asarray(chosen);flat_queries=np.asarray(flat_queries)
            arrays=dict(case_indices=chosen,scene_id=scene_ids[chosen],query_flat_indices=flat_queries,
                FSPL=np.stack([coarse[i].reshape(-1)[q] for i,q in zip(chosen,flat_queries)]),
                dominant_geometry=np.stack([dominant[i].reshape(-1)[q] for i,q in zip(chosen,flat_queries)]),
                internal_nominal=np.stack([nominal[i,-1].reshape(-1)[q] for i,q in zip(chosen,flat_queries)]))
            for variant in VARIANTS:
                for seed in [2026,2027,2028]:
                    path=args.project/f'outputs/cmf_peno_rf_v3_province/{fold}_seed{seed}/{variant}/eval_test/predictions.npz'
                    with np.load(path,allow_pickle=False) as pred:
                        indices=pred['case_indices'];positions=np.searchsorted(indices,chosen)
                        assert np.array_equal(indices[positions],chosen)
                        field=pred['mean_dbm']
                        arrays[f'{variant}__{seed}']=np.stack([field[j].reshape(-1)[q] for j,q in zip(positions,flat_queries)])
                    source_hashes[str(path)]=sha(path)
            np.savez_compressed(root/'comparison_inputs'/f'{fold}.npz',**arrays)
        print(f'Locked {fold}: four scenes, 12 station cases, 2304 queries',flush=True)
    assert len(rows)==60
    (root/'sampling_manifest.json').write_text(json.dumps(rows,indent=2))
    (root/'input_hashes.json').write_text(json.dumps(source_hashes,indent=2))
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    lock=dict(protocol_sha256=sha(root/'protocol.json'),sampling_sha256=sha(root/'sampling_manifest.json'),
              input_hashes_sha256=sha(root/'input_hashes.json'),created_before_rt_results=True,
              output_results_present=any((root/'rt_results').glob('*.npz')) if (root/'rt_results').exists() else False)
    assert not lock['output_results_present']
    (root/'protocol_lock.json').write_text(json.dumps(lock,indent=2))
    print('Protocol locked before any independent RT outputs.',flush=True)

if __name__=='__main__':main()
