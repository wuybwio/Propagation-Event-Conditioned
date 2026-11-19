"""Check numerical provenance, geometrical exclusions, and held-out adaptation."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args();r=args.output_dir
    inside=[];resolved_total=0;sample_count=0;cases=[];allzero=[]
    for p in sorted((r/'solver_inputs').glob('*.npz')):
        a=dict(np.load(p,allow_pickle=False));b=dict(np.load(r/'rt_results/primary'/p.name,allow_pickle=False))
        assert np.array_equal(a['case_indices'],b['case_indices'])
        assert np.array_equal(a['query_flat_indices'],b['query_flat_indices'])
        assert np.allclose(a['query_xyz_m'],b['query_xyz_m'])
        meta=json.loads(str(b['metadata_json']));assert not meta['target_or_event_arrays_read']
        assert meta['solver']['sionna-rt']=='2.2.0' and meta['settings']['max_depth']==3
        assert meta['settings']['samples_per_src']==262144
        assert meta['deterministic'] and meta['power_sum']=='incoherent'
        for j,c in enumerate(a['case_indices']):
            s=a['station_xyz_m'][j];ix=np.argmin(abs(a['x_m']-s[0]));iy=np.argmin(abs(a['y_m']-s[1]))
            t=float(a['terrain_m'][iy,ix]);h=float(a['surface_m'][iy,ix])
            if t<s[2]<h and h-t>1:inside.append(dict(scene=p.stem,case_index=int(c),station_xyz_m=s.tolist(),raster_ground_m=t,raster_roof_m=h))
            valid=b['path_gain_linear'][j]>0
            if not valid.any():allzero.append(int(c))
            cases.append(int(c));resolved_total+=int(valid.sum());sample_count+=len(valid)
            assert len(np.unique(a['query_flat_indices'][j]))==192
    assert len(cases)==len(set(cases))==60 and resolved_total==10761 and sample_count==11520
    assert allzero==[269] and [row['case_index'] for row in inside]==[185,269], (allzero,inside)
    for mode in ['convergence','materials']:
        files=list((r/'rt_results'/mode).glob('*.npz'));assert len(files)==5
        assert all(p.stem.endswith('_scene_01') for p in files)
    sanity=json.loads((r/'free_space_sanity.json').read_text());assert sanity['max_abs_error_db']<.01
    out=r/'sparse_adaptation';metrics=pd.read_csv(out/'case_metrics.csv')
    assert metrics.case_index.nunique()==58 and metrics.scene_id.nunique()==20 and metrics.method.nunique()==14
    assert np.isfinite(metrics[['mae_db','rmse_db','bias_db']].to_numpy()).all() and (metrics.n_queries>0).all()
    for p in out.glob('*_training.json'):
        d=json.loads(p.read_text());assert len(d['history'])==21 and d['seed']==20261107
        assert d['selected_epoch']==min(d['history'],key=lambda h:h['val_mae'])['epoch']
        assert not set(d['test_cases'])&set(d['fitting_cases']) and not set(d['test_cases'])&set(d['validation_cases'])
        assert not set(d['fitting_cases'])&set(d['validation_cases'])
        assert all(np.isfinite(h['val_mae']) for h in d['history'])
        assert not {185,269}&set(d['fitting_cases']+d['validation_cases']+d['test_cases'])
    assert len(list(out.glob('*_training.json')))==15
    record=dict(passed=True,scene_count=20,attempted_station_cases=60,usable_station_cases=58,
        query_count=11520,resolved_queries=resolved_total,unresolved_queries=sample_count-resolved_total,
        excluded_all_unresolved_case=269,inside_building_cases=inside,
        primary_exclusion='Two source positions inside opaque raster buildings; common geometry-based exclusion for every fitting, validation and test method; no replacement selected',
        no_measurements_collected=True,independent_geometry_only_solver=True,
        initial_invalid_adaptation_run='Archived separately and not reported; empty-mask case handling corrected and all 15 fits rerun',
        free_space_max_abs_error_db=sanity['max_abs_error_db'],adaptation_fits=15,folds=5,adaptation_methods=14,
        protocol_sha256=hashlib.sha256((r/'protocol.json').read_bytes()).hexdigest(),
        caveat='Cross-solver evidence on shared coarse geometry and assumed materials; no empirical physical validation or end-to-end RT speedup established')
    (r/'analysis/experiment_audit.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
