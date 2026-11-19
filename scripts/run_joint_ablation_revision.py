"""Run the 15 combined-removal fits using the frozen experiment code snapshot."""
from pathlib import Path
import subprocess, os, json, time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT=Path("/mnt/e/project/guihua/outputs/joint_ablation_20261106")
CODE=ROOT/"code"
PY="/home/wuybwio/miniconda3/envs/wio311/bin/python"
protocol=json.loads((ROOT/"protocol.json").read_text())
env={**os.environ,"PYTHONPATH":str(CODE/"src"),"PYTHONUNBUFFERED":"1"}
def run(job):
    fold,seed=job["fold"],job["seed"]
    target=ROOT/"runs"/f"{fold}_seed{seed}"/"no_multilevel_no_spectral"
    target.parent.mkdir(parents=True,exist_ok=True)
    log=target.parent/"train.log"
    if not (target/"training.json").exists():
        if target.exists():
            raise RuntimeError(f"Incomplete run requires recovery: {target}")
        cmd=[PY,str(CODE/"scripts/train_cmf_peno_rf_v3.py"),"--dataset",job["dataset"],
             "--output-dir",str(target),"--variant","no_multilevel_no_spectral",
             "--epochs","60","--variance-epochs","20","--seed",str(seed),"--device","cuda","--threads","4"]
        print(f"START {fold} seed {seed}",flush=True)
        with log.open("w") as stream:
            subprocess.run(cmd,env=env,cwd=CODE,stdout=stream,stderr=subprocess.STDOUT,check=True)
    if not (target/"eval_val/metrics.json").exists():
        cmd=[PY,str(CODE/"scripts/evaluate_cmf_peno_rf_v3.py"),"--dataset",job["dataset"],
             "--checkpoint",str(target/"best.pt"),"--split","val","--output-dir",str(target/"eval_val"),"--device","cuda"]
        with (target.parent/"eval_val.log").open("w") as stream:
            subprocess.run(cmd,env=env,cwd=CODE,stdout=stream,stderr=subprocess.STDOUT,check=True)
    meta=json.loads((target/"training.json").read_text())
    result={"fold":fold,"seed":seed,"best_val_mae_db":meta["best_val_mae_db"],"training_seconds":meta["training_seconds"]}
    print("DONE "+json.dumps(result),flush=True)
    return result
if __name__=="__main__":
    workers=int(sys.argv[1]) if len(sys.argv)>1 else 2
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run,j) for j in protocol["provenance"]]
        for f in as_completed(futures):
            results.append(f.result())
            (ROOT/"training_progress.json").write_text(json.dumps(results,indent=2))
    (ROOT/"training_complete.json").write_text(json.dumps(results,indent=2))
    print("All 15 combined-removal training/validation runs finished.",flush=True)
