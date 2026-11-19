"""Lock validation-only choices, then evaluate the joint-removal checkpoints.
Run after run_joint_ablation_revision.py has finished.
"""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, os, subprocess
import numpy as np
import pandas as pd

ROOT=Path("/mnt/e/project/guihua/outputs/joint_ablation_20261106")
OLD=Path("/home/wuybwio/projects/guihua/outputs/cmf_peno_rf_v3_province")
CODE=ROOT/"code"
PY="/home/wuybwio/miniconda3/envs/wio311/bin/python"
JOINT="no_multilevel_no_spectral"
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
def runpath(fold,seed,variant):
    return (ROOT/"runs" if variant==JOINT else OLD)/f"{fold}_seed{seed}"/variant

def select():
    assert (ROOT/"training_complete.json").exists(), "Wait for all 15 fits and validation evaluations"
    protocol=json.loads((ROOT/"protocol.json").read_text())
    assert digest(ROOT/"protocol.json")== (ROOT/"protocol.sha256").read_text().strip()
    records=[]; provenance=[]
    for fold in protocol["folds"]:
        for seed in protocol["seeds"]:
            alignment=None
            for variant in protocol["candidate_variants"]:
                p=runpath(fold,seed,variant)
                df=pd.read_csv(p/"eval_val/case_metrics.csv")
                meta=json.loads((p/"eval_val/metrics.json").read_text())
                assert meta["split"]=="val"
                expected=next(j["dataset_sha256"] for j in protocol["provenance"] if j["fold"]==fold)
                assert meta["dataset_sha256"]==expected
                assert len(df)==120 and df.scene_id.nunique()==4 and not df.case_index.duplicated().any()
                assert np.isfinite(df.mae_db).all()
                keys=df.sort_values("case_index")[["case_index","scene_id","geo_group"]].reset_index(drop=True)
                if alignment is None: alignment=keys
                else: pd.testing.assert_frame_equal(keys,alignment)
                train=json.loads((p/"training.json").read_text())
                assert train["dataset_sha256"]==expected
                active=train["active_parameter_count"]
                for scene,g in df.groupby("scene_id"):
                    records.append(dict(fold=fold,seed=seed,variant=variant,scene_id=scene,
                                        mae_db=float(g.mae_db.mean()),active_parameters=active))
                provenance.append(dict(fold=fold,seed=seed,variant=variant,
                     validation_csv=str(p/"eval_val/case_metrics.csv"),
                     validation_sha256=digest(p/"eval_val/case_metrics.csv"),
                     checkpoint=str(p/"best.pt"),checkpoint_sha256=digest(p/"best.pt")))
    frame=pd.DataFrame(records)
    frame.to_csv(ROOT/"validation_scene_seed_scores.csv",index=False)
    scores=frame.groupby(["fold","variant"],as_index=False).agg(
        validation_mae_db=("mae_db","mean"),active_parameters=("active_parameters","first"))
    scores.to_csv(ROOT/"validation_candidate_scores.csv",index=False)
    decisions=[]
    for fold,g in scores.groupby("fold"):
        best=float(g.validation_mae_db.min())
        tied=g[g.validation_mae_db<=best+1e-6].sort_values(["active_parameters","variant"])
        row=tied.iloc[0]
        decisions.append(dict(fold=fold,selected_variant=row.variant,validation_mae_db=float(row.validation_mae_db)))
    payload=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(ROOT/"protocol.json"),decisions=decisions,provenance=provenance,
        scope="Retrospective revision on previously inspected outer test groups; not fresh independent validation.")
    lock=ROOT/"selection_lock.json"
    if lock.exists():
        old=json.loads(lock.read_text())
        assert old["decisions"]==decisions and old["provenance"]==provenance
        assert digest(lock)==(ROOT/"selection_lock.sha256").read_text().strip()
    else:
        lock.write_text(json.dumps(payload,indent=2))
        (ROOT/"selection_lock.sha256").write_text(digest(lock)+"\n")
    print("SELECTION LOCKED: "+json.dumps(decisions),flush=True)
    return protocol

def evaluate(protocol):
    env={**os.environ,"PYTHONPATH":str(CODE/"src"),"PYTHONUNBUFFERED":"1","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"}
    for j in protocol["provenance"]:
        p=runpath(j["fold"],j["seed"],JOINT)
        for split in ("calibration","test"):
            output=p/f"eval_{split}"
            if split=="test":
                calibration=p/"variance_calibration.json"
                if not calibration.exists():
                    cmd=[PY,str(CODE/"scripts/fit_cmf_peno_variance_temperature.py"),
                         "--validation-predictions",str(p/"eval_calibration/predictions.npz"),"--output",str(calibration)]
                    with (p.parent/"calibration_fit.log").open("w") as f:
                        subprocess.run(cmd,env=env,cwd=CODE,stdout=f,stderr=subprocess.STDOUT,check=True)
            if (output/"metrics.json").exists(): continue
            assert not output.exists(), f"Incomplete evaluation: {output}"
            cmd=[PY,str(CODE/"scripts/evaluate_cmf_peno_rf_v3.py"),"--dataset",j["dataset"],
                 "--checkpoint",str(p/"best.pt"),"--split",split,"--output-dir",str(output),"--device","cuda"]
            if split=="test": cmd+=["--calibration-json",str(calibration)]
            with (p.parent/f"eval_{split}.log").open("w") as f:
                subprocess.run(cmd,env=env,cwd=CODE,stdout=f,stderr=subprocess.STDOUT,check=True)
        print(f"TEST DONE {j['fold']} seed {j['seed']}",flush=True)
    (ROOT/"evaluation_complete.json").write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).isoformat(),runs=15,selection_sha256=digest(ROOT/"selection_lock.json")),indent=2))

if __name__=="__main__":
    evaluate(select())
