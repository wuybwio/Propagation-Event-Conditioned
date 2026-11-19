"""Reproduce the R2-4 planning-domain audit, source tables and figures.

Uses archived evaluations without changing fields, selections or scores.
Figure contract: distinguish common-volume siting from disjoint-window allocation;
quantitative grid plus a representative four-window map; all archived methods,
budgets and statistical units retained. Times New Roman, 183 mm, vector PDF/SVG,
600 dpi PNG. No hypothesis tests or claims of operational coverage are added.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch
from pyproj import Transformer

PAPER = Path('/mnt/e/paper(wio)/Propagation-Event-Conditioned Multi-Fidelity Neural Operator for Probabilistic 3D Radio Monitoring and Station Planning')
LABELS = {'full_v3':'CMF-PENO-RF', 'FSPL':'FSPL', 'Random':'Random',
          'Uniform':'Uniform', 'ReferenceGreedy':'Reference greedy',
          'ExactReference':'Exact reference', 'deterministic':'Deterministic',
          'cnn_residual':'CNN residual', 'fno_residual':'FNO residual',
          'no_multilevel':'No auxiliary supervision', 'no_events':'No events',
          'no_local':'No local operator', 'no_spectral':'No spectral operator',
          'no_event_consistency':'No event consistency'}
METRICS = ['mean_detection_probability','coverage_fraction','blind_fraction',
           'minimum_detection_probability']
FIG_WIDTH_MM = 183


def save(fig, out, name):
    fig.savefig(out/(name+'.pdf'), facecolor='white')
    fig.savefig(out/(name+'.svg'), facecolor='white')
    fig.savefig(out/(name+'.png'), dpi=600, facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('/home/wuybwio/projects/guihua/outputs'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/planning_domain_revision_20261107'))
    a=p.parse_args(); out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    source=a.data_root/'jiangxi_planning_internal'
    overall=pd.read_csv(source/'summary/planning_overall.csv')
    raw=pd.read_csv(source/'summary/planning_raw.csv')
    # Repeat averages precede seed averages; neither constitutes extra geography.
    run=raw.groupby(['fold','seed','planning_scope','planning_unit','geo_group','scene_id','method','budget'],as_index=False)[METRICS].mean()
    units=run.groupby(['planning_scope','planning_unit','geo_group','scene_id','method','budget'],as_index=False)[METRICS].mean()
    checks=[]
    for row in overall.itertuples():
        d=units[(units.planning_scope==row.planning_scope)&(units.method==row.method)&(units.budget==row.budget)]
        for metric in METRICS:
            assert np.isclose(d[metric].mean(),getattr(row,metric+'_mean'),atol=1e-12)
            assert np.isclose(d[metric].std(ddof=1),getattr(row,metric+'_std'),atol=1e-12)
        assert len(d)==(20 if row.planning_scope=='local_scene' else 5)
    overall.to_csv(out/'planning_all_scopes_source.csv',index=False)
    units.to_csv(out/'planning_statistical_units.csv',index=False)
    ds=a.data_root/'jiangxi_province_validation/folds/region_01/cmf_peno_rf_s32.npz'
    with np.load(ds,allow_pickle=False) as n:
        data={k:n[k] for k in ['x_m','y_m','z_m','terrain_m','surface_m','air_mask',
                              'station_xyz_m','scene_id','geo_group','split','source_station_index']}
        meta=json.loads(str(n['metadata_json'].item()))['base_metadata']['source_scenes']
        nominal=n['fidelity_dbm'][:,-1]
    scenes=np.asarray(data['scene_id']).astype(str)
    for scene in sorted(set(scenes)):
        ix=np.flatnonzero(scenes==scene); first=ix[0]
        assert len(ix)==30
        for key in ['x_m','y_m','z_m','terrain_m','surface_m','air_mask']:
            assert all(np.array_equal(data[key][first],data[key][j]) for j in ix), (scene,key)
        mask=data['z_m'][first,:,None,None]>data['surface_m'][first,None]
        assert np.array_equal(mask,data['air_mask'][first]),scene
        args=meta[scene]['arguments']
        assert args['antenna_agl_m']==20 and args['stations']==30
        assert len(np.unique(data['station_xyz_m'][ix],axis=0))==30
        checks.append(dict(scene_id=scene,candidates=30,valid_air_voxels=int(mask.sum()),
                           shared_grid_verified=True,mask_equals_z_above_surface=True,
                           source_antenna_agl_m=20,source_seed=args['seed'],
                           terrain_min_m=float(data['terrain_m'][first].min()),
                           terrain_max_m=float(data['terrain_m'][first].max()),
                           origin_utm_x_m=meta[scene]['origin_utm_m'][0],
                           origin_utm_y_m=meta[scene]['origin_utm_m'][1]))
    pd.DataFrame(checks).to_csv(out/'scene_grid_and_mask_audit.csv',index=False)
    alignment=[]
    variants=['full_v3','deterministic','cnn_residual','fno_residual','no_multilevel',
              'no_events','no_local','no_spectral','no_event_consistency']
    for fold in ['region_01','region_02','region_03','region_04','region_05']:
        with np.load(a.data_root/'jiangxi_province_validation/folds'/fold/'cmf_peno_rf_s32.npz') as n:
            expected=np.flatnonzero(n['split']==2)
            masks=n['air_mask'][expected]
        for seed in [2026,2027,2028]:
            for variant in variants:
                path=a.data_root/'cmf_peno_rf_v3_province'/f'{fold}_seed{seed}'/variant/'eval_test/predictions.npz'
                with np.load(path) as n:
                    assert np.array_equal(n['case_indices'],expected),str(path)
                    assert np.array_equal(n['mask'],masks),str(path)
                alignment.append(dict(fold=fold,seed=seed,variant=variant,cases=len(expected),
                                      indices_aligned=True,masks_aligned=True))
    pd.DataFrame(alignment).to_csv(out/'prediction_alignment_audit.csv',index=False)

    selections=json.loads((source/'region_01_seed2026/selected_indices.json').read_text())
    test=np.flatnonzero(data['split']==2)
    group_choice=selections['group:region_01']['full_v3:4:0']
    chosen_global=set(test[group_choice]); map_rows=[]; rescored=[]
    chosen_scenes=sorted(set(scenes[test]))
    for scene in chosen_scenes:
        ix=test[scenes[test]==scene]
        local=selections['scene:'+scene]['full_v3:4:0']
        for j,case in enumerate(ix):
            station=data['station_xyz_m'][case]
            map_rows.append(dict(scene_id=scene,candidate_id=j+1,case_index=int(case),
                                 source_station_index=int(data['source_station_index'][case]),
                                 x_m=float(station[0]),y_m=float(station[1]),z_m=float(station[2]),
                                 local_selected=j in local,group_selected=case in chosen_global))
        prob=ndtr((nominal[ix[local]].astype(float)+100)/6)
        union=1-np.prod(1-prob,axis=0); mask=data['air_mask'][ix[0]]
        values={'mean_detection_probability':union[mask].mean(),
                'coverage_fraction':(union[mask]>=.9).mean()}
        stored=raw[(raw.fold=='region_01')&(raw.seed==2026)&(raw.scene_id==scene)&
                   (raw.method=='full_v3')&(raw.budget==4)].iloc[0]
        for k,v in values.items(): assert np.isclose(v,stored[k],atol=2e-7),(scene,k,v,stored[k])
        rescored.append(dict(scene_id=scene,**values))
    block=[]; blockmask=[]
    for scene in chosen_scenes:
        ix=test[scenes[test]==scene]; selected=[j for j in ix if j in chosen_global]
        union=1-np.prod(1-ndtr((nominal[selected].astype(float)+100)/6),axis=0)
        # Empty selection has zero union on this block, not a propagated field.
        block.append(union.ravel());blockmask.append(data['air_mask'][ix[0]].ravel())
    union=np.concatenate(block);mask=np.concatenate(blockmask)
    stored=raw[(raw.fold=='region_01')&(raw.seed==2026)&(raw.planning_scope=='geographical_group')&
               (raw.method=='full_v3')&(raw.budget==4)].iloc[0]
    assert np.isclose(union[mask].mean(),stored.mean_detection_probability,atol=2e-7)
    assert np.isclose((union[mask]>=.9).mean(),stored.coverage_fraction,atol=2e-7)
    pd.DataFrame(map_rows).to_csv(out/'figure9_candidate_selected_source.csv',index=False)
    (out/'audit.json').write_text(json.dumps(dict(scenes_audited=len(checks),
        candidates_audited=sum(r['candidates'] for r in checks),grid_and_mask_checks='passed',
        planning_summary_reproduced_rows=len(overall),source_raw_records=len(raw),
        example_selected_sets_rescored=5,geographic_groups=5,local_scene_summaries=20,
        prediction_archives_index_and_mask_verified=len(alignment),
        repeats_averaged_before_seeds=True,group_cross_scene_fields='zero outside own block',
        example_selection='lexicographically first held-out group and first seed; no performance filtering',
        source_summary_sha256=hashlib.sha256((source/'summary/planning_overall.csv').read_bytes()).hexdigest()),indent=2))

    for font in ('times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf'):
        font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/'+font)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':7,'axes.titlesize':7.5,
        'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.6})
    baseline=['full_v3','FSPL','Random','Uniform','ReferenceGreedy','ExactReference']
    ablate=['full_v3','deterministic','cnn_residual','fno_residual','no_multilevel',
            'no_events','no_local','no_spectral','no_event_consistency']
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#555555','#000000',
            '#56B4E9','#A6761D','#6A51A3']
    fig,axes=plt.subplots(2,2,figsize=(FIG_WIDTH_MM/25.4,169/25.4))
    handles=[]
    for k,(ax,scope,methods) in enumerate(zip(axes.ravel(),
                ['local_scene']*2+['geographical_group']*2,[baseline,ablate]*2)):
        here=[]
        for i,method in enumerate(methods):
            d=overall[(overall.planning_scope==scope)&(overall.method==method)].sort_values('budget')
            if d.empty: continue
            line,=ax.plot(d.budget,d.mean_detection_probability_mean,color=colors[i],
                marker=['o','s','^','D','v','x','P','+','*'][i],markersize=3,
                linewidth=1.2 if method=='full_v3' else .8,
                linestyle='-' if i<5 else '--',label=LABELS[method],zorder=4 if method=='full_v3' else 2)
            here.append(line)
        if k<2: handles.append(here)
        scope_label='Single scene (20 scene summaries)' if k<2 else 'Four-window allocation (5 group summaries)'
        ax.set_title(scope_label,pad=5)
        ax.text(-.16,1.04,'abcd'[k],transform=ax.transAxes,fontweight='bold',fontsize=9)
        ax.set(xticks=[1,2,3,4,6,8],xlim=(.8,8.2),ylim=(.94,1.002) if k<2 else (0,1.03),
               xlabel='Candidate-site budget',ylabel='Reference-rescored mean union probability')
        if k<2: ax.set_yticks([.94,.96,.98,1.00])
        ax.grid(alpha=.16,linewidth=.5)
    fig.subplots_adjust(left=.085,right=.985,top=.96,bottom=.25,hspace=.50,wspace=.31)
    fig.legend(handles[0],[h.get_label() for h in handles[0]],loc='lower center',
               bbox_to_anchor=(.50,.12),ncol=3,frameon=False,handlelength=1.8,columnspacing=1.3)
    fig.legend(handles[1],[h.get_label() for h in handles[1]],loc='lower center',
               bbox_to_anchor=(.5,.008),ncol=3,frameon=False,handlelength=1.8,columnspacing=1.3)
    save(fig,out,'figure8_planning_domains')

    fig,axes=plt.subplots(2,2,figsize=(FIG_WIDTH_MM/25.4,170/25.4))
    terrain_range=max(float(np.ptp(data['terrain_m'][test[scenes[test]==s][0]])) for s in chosen_scenes)
    transform=Transformer.from_crs(4326,32650,always_xy=True)
    maps=pd.DataFrame(map_rows)
    for k,(ax,scene) in enumerate(zip(axes.ravel(),chosen_scenes)):
        case=test[scenes[test]==scene][0]; x=data['x_m'][case];y=data['y_m'][case]
        terrain=data['terrain_m'][case];rel=terrain-terrain.min()
        mesh=ax.pcolormesh(x,y,rel,cmap='YlGn',vmin=0,vmax=terrain_range,shading='nearest',rasterized=True)
        if np.ptp(terrain)>.1: ax.contour(x,y,rel,levels=5,colors='#708569',linewidths=.35,alpha=.7)
        info=meta[scene];origin=np.array(info['origin_utm_m'])
        geo=json.loads((a.data_root/'jiangxi_province_validation/scenes'/scene/'buildings/buildings.geojson').read_text())
        for f in geo['features']:
            g=f['geometry']; polygons=[g['coordinates']] if g['type']=='Polygon' else g['coordinates']
            for poly in polygons:
                c=np.array(poly[0]);xx,yy=transform.transform(c[:,0],c[:,1]);xy=np.column_stack([xx,yy])-origin
                ax.add_patch(Polygon(xy,closed=True,facecolor='#e5e5e5',edgecolor='#686868',linewidth=.3))
        d=maps[maps.scene_id==scene]
        ax.scatter(d.x_m,d.y_m,s=10,c='#505050',edgecolors='white',linewidths=.3,zorder=5)
        local=d[d.local_selected];group=d[d.group_selected]
        ax.scatter(local.x_m,local.y_m,s=58,facecolors='none',edgecolors='#0072B2',linewidths=1.1,zorder=6)
        ax.scatter(group.x_m,group.y_m,s=67,marker='*',c='#E69F00',edgecolors='#3d2e0b',linewidths=.45,zorder=7)
        for r in d[d.local_selected|d.group_selected].itertuples():
            ax.annotate(str(r.candidate_id),(r.x_m,r.y_m),xytext=(4,5),textcoords='offset points',
                        fontsize=6.5,bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.3),zorder=8)
        ax.set(xlim=(-12,512),ylim=(-12,512),xticks=[0,100,200,300,400,500],yticks=[0,100,200,300,400,500],
               xlabel='Local east (m)',ylabel='Local north (m)')
        ax.set_aspect('equal'); ax.set_title(scene.replace('_',' ')+' | local 4, group '+str(len(group)))
        ax.text(-.17,1.04,'abcd'[k],transform=ax.transAxes,fontweight='bold',fontsize=9)
    fig.subplots_adjust(left=.075,right=.88,top=.955,bottom=.13,wspace=.29,hspace=.33)
    cax=fig.add_axes([.91,.23,.017,.60]);cb=fig.colorbar(mesh,cax=cax)
    cb.set_label('Terrain elevation above each window minimum (m)',fontsize=7)
    legend=[Line2D([],[],marker='o',linestyle='none',color='#505050',markersize=3,label='Candidate (30 per window)'),
            Line2D([],[],marker='o',linestyle='none',markerfacecolor='none',markeredgecolor='#0072B2',markersize=6,label='Selected locally (4 per window)'),
            Line2D([],[],marker='*',linestyle='none',color='#E69F00',markeredgecolor='#3d2e0b',markersize=7,label='Selected jointly (4 across windows)'),
            Patch(facecolor='#e5e5e5',edgecolor='#686868',label='Building footprint')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.49,.015),ncol=2,frameon=False)
    save(fig,out,'figure9_candidate_selected_map')
    for name in ['figure8_planning_domains.pdf','figure9_candidate_selected_map.pdf']:
        import shutil
        shutil.copy2(out/name,PAPER/name)
    policies=['FSPL','Random','Uniform','ReferenceGreedy','full_v3']
    tex=[]
    for scope in ['local_scene','geographical_group']:
        heading='Single scene: 20 scene summaries nested in 5 groups' if scope=='local_scene' else 'Four-window allocation: 5 group summaries'
        tex.append(r'\multicolumn{4}{@{}l}{\textit{'+heading+r'}}\\'+'\n'+r'\addlinespace[2pt]')
        for method in policies:
            r=overall[(overall.planning_scope==scope)&(overall.method==method)&(overall.budget==4)].iloc[0]
            cells=['$'+format(r[m+'_mean'],'.4f')+r'\pm'+format(r[m+'_std'],'.4f')+'$' for m in METRICS[:3]]
            tex.append(LABELS[method]+' & '+' & '.join(cells)+r'\\')
        tex.append(r'\addlinespace[4pt]')
    (out/'planning_table_rows.tex').write_text('\n'.join(tex))
    (out/'figure_QA.md').write_text('''# R2-4 figure audit
Backend: Python matplotlib only. Font: Times New Roman (user requirement).
Final sizes: 183 x 169 mm and 183 x 170 mm. PDF/SVG text editable; PNG 600 dpi.
Figure 8 retains every archived method and budget within its domain; exact reference
is available at B=1,2,3 only in the single-scene domain. Lines show means, not CIs.
Single-scene y axes are expanded to 0.94--1.002; allocation axes use 0--1.03.
Random's 100 repeats are averaged first, then 3 seeds, before scene/group summaries.
Twenty scene summaries remain nested in five geographic groups. No significance tests.
Figure 9 is the first held-out group and first archived seed, with no outcome filtering.
Four windows use separate local coordinates: adjacency does not denote a continuous map.
Terrain colours subtract each window's minimum; same relative-height scale across panels.
Buildings are projected source footprints, clipped by axes to the original 500-m windows.
All 120 candidates retained. Only selected IDs labelled; complete IDs exported in CSV.
Original data, selections, budgets, field thresholds and scores are unchanged.
''')
    print((out/'audit.json').read_text())
    print((out/'planning_table_rows.tex').read_text())


if __name__=='__main__': main()
