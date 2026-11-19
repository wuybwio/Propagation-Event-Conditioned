"""Publication figure from complete independent RT and adaptation source tables."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

NAMES={'FSPL':'FSPL','dominant_geometry':'Geometry comparator','internal_nominal':'Nominal internal rule',
 'full_v3':'CMF-PENO-RF frozen','cnn_residual':'CNN frozen','fno_residual':'FNO frozen',
 'FSPL_affine':'Affine FSPL','internal_nominal_affine':'Affine nominal rule',
 'internal_nominal_frozen':'Unadapted nominal rule',
 'ridge_adapted':'Ridge residual','hist_gradient_boosting_adapted':'Boosted-tree residual',
 'cnn_residual_adapted':'CNN adapted','fno_residual_adapted':'FNO adapted','full_v3_adapted':'CMF-PENO-RF adapted'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args();root=args.output_dir
    for font in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
        font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/'+font)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,'axes.titlesize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'savefig.dpi':600})
    a=root/'analysis';frozen=pd.read_csv(a/'frozen_scene_metrics.csv');frozen=frozen[(frozen.stratum=='resolved')&(frozen.floor_dbm==-140)]
    adapted=pd.read_csv(a/'adaptation_scene_metrics.csv');groups=pd.read_csv(a/'adaptation_group_metrics.csv');sensitivity=pd.read_csv(a/'solver_sensitivity.csv')
    assert frozen.scene_id.nunique()==20 and adapted.scene_id.nunique()==20
    fig,axes=plt.subplots(2,2,figsize=(7.16,5.8),layout='constrained',gridspec_kw={'height_ratios':[1.25,1]})
    def panel(ax,data,methods,label,title):
        for j,m in enumerate(methods):
            values=data.loc[data.method==m,'mae_db'].to_numpy();assert len(values)==20,(m,len(values))
            color='#2166ac' if 'full_v3' in m else '#6e6e6e'
            ax.errorbar(values.mean(),j,xerr=values.std(ddof=1),fmt='o',color=color,capsize=2,markersize=4,lw=1)
        ax.set_yticks(range(len(methods)),[NAMES[m] for m in methods]);ax.invert_yaxis();ax.set_xlim(left=0);ax.grid(axis='x',alpha=.15)
        ax.set_xlabel('Held-out scene MAE (dB)');ax.set_title(title,loc='left',pad=8)
        ax.text(-.09,1.06,label,transform=ax.transAxes,fontweight='bold',fontsize=10)
    panel(axes[0,0],frozen,['FSPL','dominant_geometry','internal_nominal','cnn_residual','fno_residual','full_v3'],'a','Frozen synthetic-only models')
    panel(axes[0,1],adapted,['internal_nominal_frozen','FSPL_affine','internal_nominal_affine','ridge_adapted','hist_gradient_boosting_adapted','cnn_residual_adapted','fno_residual_adapted','full_v3_adapted'],'b','Sparse RT adaptation and unadapted rule')
    ax=axes[1,0]
    for m,color,marker in [('internal_nominal_frozen','#252525','x'),('internal_nominal_affine','#999999','s'),('hist_gradient_boosting_adapted','#b97b43','^'),('full_v3_adapted','#2166ac','o')]:
        values=groups[groups.method==m].sort_values('fold');ax.plot(np.arange(1,6),values.mae_db,marker=marker,label=NAMES[m],color=color,lw=1,markersize=4)
    ax.set(xticks=np.arange(1,6),xlabel='Held-out geographic group',ylabel='Group MAE (dB)');ax.grid(alpha=.15);ax.legend(loc='best',frameon=False)
    ax.text(-.13,1.06,'c',transform=ax.transAxes,fontweight='bold',fontsize=10)
    ax.set_title('Geographic heterogeneity after adaptation',loc='left',pad=8)
    ax=axes[1,1]
    for j,mode in enumerate(['convergence','materials']):
        data=sensitivity[sensitivity['mode']==mode].sort_values('fold');assert len(data)==5
        ax.scatter(j+np.linspace(-.10,.10,5),data.mae_db,s=16,c='#6e6e6e',zorder=3)
        ax.plot([j-.18,j+.18],[data.mae_db.mean()]*2,color='#2166ac',lw=1.6)
    ax.set_xticks([0,1],['4× rays + depth 5','Alternative materials']);ax.set_ylabel('Power-change MAE (dB)');ax.set_xlim(-.5,1.5);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15)
    ax.set_title('Solver sensitivity on shared resolved links',loc='left',pad=8);ax.text(-.13,1.06,'d',transform=ax.transAxes,fontweight='bold',fontsize=10)
    folder=root/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'figure10_independent_rt_and_adaptation.pdf')
    fig.savefig(folder/'figure10_independent_rt_and_adaptation.svg')
    fig.savefig(folder/'figure10_independent_rt_and_adaptation.png',dpi=600)
    fig.savefig(folder/'figure10_independent_rt_and_adaptation.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(folder/'figure10_preview.png',dpi=300)
    plt.close(fig)
    (folder/'figure10_contract.json').write_text(json.dumps(dict(
        conclusion='Independent computational discrepancy and adaptation benefit must be assessed separately from internal emulation and runtime.',
        archetype='quantitative grid',backend='python matplotlib',font='Times New Roman',width_mm=181.864,minimum_font_pt=7,dpi=600,
        inputs=['frozen_scene_metrics.csv','adaptation_scene_metrics.csv','adaptation_group_metrics.csv','solver_sensitivity.csv'],
        error_bars='sample standard deviation over 20 scene averages, nested in five geographical groups; descriptive, not 20 independent studies',
        all_query_policy='Panels a-c use shared resolved links; unresolved link counts and all-query -140 dBm floor diagnostics are separate',
        panel_d='paired changes on shared resolved paths, five cases, not a guarantee of RT convergence',no_data_exclusion_for_visual_convenience=True),indent=2))
    print(folder/'figure10_independent_rt_and_adaptation.pdf')

if __name__=='__main__':main()
