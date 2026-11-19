from pathlib import Path
import datetime,json,numpy as np
r=Path('outputs/independent_rt_20261107');inside=[]
for p in sorted((r/'solver_inputs').glob('*.npz')):
    a=np.load(p,allow_pickle=False)
    for j,c in enumerate(a['case_indices']):
        s=a['station_xyz_m'][j];ix=np.argmin(abs(a['x_m']-s[0]));iy=np.argmin(abs(a['y_m']-s[1]))
        t=float(a['terrain_m'][iy,ix]);h=float(a['surface_m'][iy,ix])
        if t<s[2]<h and h-t>1:inside.append(dict(scene=p.stem,case_index=int(c),station_xyz_m=s.tolist(),raster_ground_m=t,raster_roof_m=h))
assert [row['case_index'] for row in inside]==[185,269]
(r/'geometry_exclusions.json').write_text(json.dumps(dict(registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    excluded_case_indices=[185,269],geometry_diagnostics=inside,rule='Station lies inside an opaque raster building prism; outside scope of outdoor nonpenetrating RT configuration',
    all_methods_identical_exclusion=True,replacement_selected=False,all_scenes_retained=True,
    raw_results_retained=True,initial_invalid_case_adaptation_runs='diagnostic only, not used in manuscript; all 15 fits rerun with geometric source validity'),indent=2))
print(inside)
