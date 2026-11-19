"""Audit and summarize the revision's four-candidate architecture comparison."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from evaluate_joint_ablation_revision import ROOT, OLD, JOINT, runpath, digest

def main():
    assert (ROOT/"evaluation_complete.json").exists()
    lock=json.loads((ROOT/"selection_lock.json").read_text())
    assert digest(ROOT/"selection_lock.json")==(ROOT/"selection_lock.sha256").read_text().strip()
    protocol=json.loads((ROOT/"protocol.json").read_text())
    out=ROOT/"analysis"; out.mkdir(exist_ok=True)
    metrics=["mae_db","rmse_db","bias_db","probability_mse","coverage_iou","blind_iou","balanced_accuracy","false_coverage_rate"]
    rows=[]; freeze=[]; eval_provenance=[]
    for fold in protocol["folds"]:
        for seed in protocol["seeds"]:
            alignment=None
            for variant in protocol["candidate_variants"]:
                p=runpath(fold,seed,variant)
                meta=json.loads((p/"eval_test/metrics.json").read_text())
                assert meta["split"]=="test" and not meta["cross_dataset_evaluation"]
                expected=next(j["dataset_sha256"] for j in protocol["provenance"] if j["fold"]==fold)
                assert meta["dataset_sha256"]==expected
                df=pd.read_csv(p/"eval_test/case_metrics.csv")
                assert len(df)==120 and df.scene_id.nunique()==4
                assert set(df.geo_group)=={fold}
                assert not df.case_index.duplicated().any()
                keys=df.sort_values("case_index")[["case_index","scene_id","geo_group","voxels"]].reset_index(drop=True)
                if alignment is None: alignment=keys
                else: pd.testing.assert_frame_equal(keys,alignment)
                assert np.isfinite(df[["mae_db","rmse_db","bias_db","probability_mse"]]).all().all()
                df=df.assign(fold=fold,seed=seed,variant=variant)
                rows.append(df)
                eval_provenance.append(dict(fold=fold,seed=seed,variant=variant,
                    path=str(p/"eval_test/case_metrics.csv"),sha256=digest(p/"eval_test/case_metrics.csv")))
                if variant==JOINT:
                    a=torch.load(p/"best_mean.pt",map_location="cpu",weights_only=False)
                    b=torch.load(p/"best.pt",map_location="cpu",weights_only=False)
                    changed=[k for k in a["state_dict"] if not k.startswith("variance_head") and k!="log_variance_bias"
                             and not torch.equal(a["state_dict"][k],b["state_dict"][k])]
                    assert not changed,changed
                    assert not b["variant_spec"]["use_spectral"] and not b["variant_spec"]["use_multilevel_supervision"]
                    assert [b["arguments"][k] for k in ("stage_weight","frequency_weight","orthogonal_weight")]==[0.,0.,0.]
                    freeze.append(dict(fold=fold,seed=seed,mean_state_unchanged=True))
                    del a,b
    raw=pd.concat(rows,ignore_index=True)
    raw.to_csv(out/"case_metrics.csv",index=False)
    ss=raw.groupby(["variant","fold","seed","geo_group","scene_id"],as_index=False)[metrics].mean()
    scene=ss.groupby(["variant","geo_group","scene_id"],as_index=False)[metrics].mean()
    # Verify new aggregation exactly reproduces the archived original 60 scene rows.
    archive=pd.read_csv(OLD/"province_statistics/province_scene_metrics.csv")
    keys=["variant","geo_group","scene_id"]
    a=archive[archive.variant.isin(protocol["candidate_variants"][:-1])].set_index(keys)[metrics].sort_index()
    b=scene[scene.variant!=JOINT].set_index(keys)[metrics].sort_index()
    assert np.allclose(a,b,equal_nan=True,rtol=1e-10,atol=1e-12)
    choices={d["fold"]:d["selected_variant"] for d in lock["decisions"]}
    selected=pd.concat([scene[(scene.geo_group==f)&(scene.variant==v)] for f,v in choices.items()])
    selected=selected.assign(source_variant=selected.variant,variant="validation_selected")
    scene=pd.concat([scene,selected],ignore_index=True)
    assert len(scene)==100 and scene.groupby("variant").size().eq(20).all()
    assert not scene[metrics].isna().any().any()
    ss.to_csv(out/"scene_seed_metrics.csv",index=False)
    scene.to_csv(out/"scene_metrics.csv",index=False)
    group=scene.groupby(["variant","geo_group"],as_index=False)[metrics].mean()
    group.to_csv(out/"group_metrics.csv",index=False)
    overall=[]
    for v,g in scene.groupby("variant"):
        row=dict(variant=v,n_scenes=len(g),n_groups=g.geo_group.nunique())
        for m in metrics:
            row[m+"_mean"]=float(g[m].mean()); row[m+"_sd"]=float(g[m].std(ddof=1))
        overall.append(row)
    overall=pd.DataFrame(overall)
    overall.to_csv(out/"overall.csv",index=False)
    base=group[group.variant=="full_v3"].set_index("geo_group")[metrics]
    deltas=[]
    for v in list(protocol["candidate_variants"][1:])+["validation_selected"]:
        df=group[group.variant==v].set_index("geo_group")[metrics]-base
        for geo,r in df.iterrows(): deltas.append(dict(variant=v,geo_group=geo,**r.to_dict()))
    delta=pd.DataFrame(deltas)
    delta.to_csv(out/"group_differences.csv",index=False)
    interaction=(group[group.variant==JOINT].set_index("geo_group")[metrics]
                 -group[group.variant=="no_multilevel"].set_index("geo_group")[metrics]
                 -group[group.variant=="no_spectral"].set_index("geo_group")[metrics]+base)
    interaction.to_csv(out/"nonadditivity_contrast.csv")
    pairwise=[]
    for left,right in [(JOINT,"no_multilevel"),(JOINT,"no_spectral"),("no_multilevel","no_spectral")]:
        difference=(group[group.variant==left].set_index("geo_group")[metrics]
                    -group[group.variant==right].set_index("geo_group")[metrics])
        for geo,r in difference.iterrows():
            pairwise.append(dict(left=left,right=right,geo_group=geo,**r.to_dict()))
    pd.DataFrame(pairwise).to_csv(out/"reduced_pairwise_differences.csv",index=False)
    # Descriptive paired directions; five overlapping outer-training fits are not independent studies.
    directions=[]
    for v,g in delta.groupby("variant"):
        for m in metrics:
            if m=="bias_db": continue
            higher=m in ("coverage_iou","blind_iou","balanced_accuracy")
            directions.append(dict(variant=v,metric=m,mean_delta=g[m].mean(),
                min_delta=g[m].min(),max_delta=g[m].max(),
                n_groups_improved=int(((g[m]>0) if higher else (g[m]<0)).sum())))
    pd.DataFrame(directions).to_csv(out/"direction_counts.csv",index=False)
    (out/"mean_freeze_audit.json").write_text(json.dumps(freeze,indent=2))
    (out/"provenance.json").write_text(json.dumps(dict(
        protocol_sha256=digest(ROOT/"protocol.json"),selection_sha256=digest(ROOT/"selection_lock.json"),
        test_files=eval_provenance,original_scene_scores_reproduced=True),indent=2))
    (out/"results.json").write_text(json.dumps(dict(overall=overall.to_dict("records"),
        decisions=lock["decisions"],directions=directions,
        mean_interaction_mae=float(interaction.mae_db.mean()),
        mean_interaction_rmse=float(interaction.rmse_db.mean())),indent=2))
    for font in Path("/mnt/c/Windows/Fonts").glob("times*.ttf"): font_manager.fontManager.addfont(str(font))
    font_manager.findfont("Times New Roman",fallback_to_default=False)
    plt.rcParams.update({"font.family":"Times New Roman","font.size":9,"axes.labelsize":9,
        "axes.titlesize":10,"xtick.labelsize":8,"ytick.labelsize":8,"pdf.fonttype":42,"ps.fonttype":42})
    names={"full_v3":"Complete configuration","no_multilevel":"No multi-level supervision",
           "no_spectral":"No spectral operator",JOINT:"Joint removal","validation_selected":"Validation-selected"}
    colors=["#666666","#3d80b2","#d38335","#29936f"]
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),layout="constrained")
    for ax,m,letter in zip(axs,["mae_db","rmse_db"],"ab"):
        for v,c,marker in zip(protocol["candidate_variants"],colors,["o","s","^","D"]):
            g=group[group.variant==v].sort_values("geo_group")
            ax.plot(range(1,6),g[m],color=c,marker=marker,ms=4,lw=1,label=names[v])
        chosen=group[group.variant=="validation_selected"].sort_values("geo_group")
        ax.scatter(range(1,6),chosen[m],s=90,facecolors="none",edgecolors="black",lw=.8,label="Validation-selected",zorder=5)
        ax.set_xticks(range(1,6),[f"G{i}" for i in range(1,6)])
        ax.set_xlabel("Held-out geographical group")
        ax.set_ylabel("MAE (dB)" if m=="mae_db" else "RMSE (dB)")
        ax.set_title(f"({letter}) Held-out "+("MAE" if m=="mae_db" else "RMSE"),loc="left")
        ax.grid(axis="y",alpha=.15)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False,fontsize=8)
    for fmt in ("pdf","png"): fig.savefig(out/f"joint_ablation_group_comparison.{fmt}",dpi=400)
    plt.close(fig)
    print(overall[["variant","mae_db_mean","mae_db_sd","rmse_db_mean","rmse_db_sd"]].to_string(index=False))
    print("Choices: "+json.dumps(choices))
    print("Analysis verified and saved: "+str(out))

if __name__=="__main__": main()
