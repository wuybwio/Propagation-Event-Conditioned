"""Build distinct coarse-input / geometric-teacher simulation volumes."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from jiangxi_radio.volume_simulation import demo_scene, load_building_scene, make_dataset, sample_stations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo',action='store_true')
    source.add_argument('--gba-geojson')
    source.add_argument('--nc-build-geojson')
    source.add_argument('--overture-geojson')
    p.add_argument('--dem-dir')
    p.add_argument('--dem-source', choices=['srtm','fabdem'], default='srtm')
    p.add_argument('--longitude',type=float,default=115.85)
    p.add_argument('--latitude',type=float,default=28.68)
    p.add_argument('--size-m',type=float,default=500)
    p.add_argument('--terrain-step-m',type=float,default=10)
    p.add_argument('--shape',nargs=3,type=int,default=[16,24,24],metavar=('Z','Y','X'))
    p.add_argument('--top-agl-m',type=float,default=160)
    p.add_argument('--stations',type=int,default=30)
    p.add_argument('--antenna-agl-m',type=float,default=20)
    p.add_argument('--frequency-mhz',type=float,default=2500)
    p.add_argument('--emitter-dbm',type=float,default=23)
    p.add_argument('--receiver-gain-dbi',type=float,default=8)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--output',required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.suffix != '.npz':
        p.error('--output must end with .npz')
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(f'refusing overwrite: {output}')
    if args.demo:
        scene, meta = demo_scene(), {'geometry':'explicit_synthetic_demo','buildings':4}
    else:
        if not args.dem_dir:
            p.error('--dem-dir required with building geometry')
        geometry_path = args.gba_geojson or args.nc_build_geojson or args.overture_geojson
        building_source = ('gba' if args.gba_geojson else
                           'nc_build' if args.nc_build_geojson else 'overture')
        scene,meta = load_building_scene(geometry_path,args.dem_dir,
                                         longitude=args.longitude,latitude=args.latitude,size_m=args.size_m,
                                         terrain_step_m=args.terrain_step_m,dem_source=args.dem_source,
                                         building_source=building_source)
    stations = sample_stations(scene,args.stations,args.seed,args.antenna_agl_m)
    print(json.dumps({'scene':meta,'shape':args.shape,'station_cases':len(stations)},ensure_ascii=False),flush=True)
    start = time.perf_counter()
    data = make_dataset(scene,stations=stations,shape=tuple(args.shape),top_agl_m=args.top_agl_m,
                        frequency_mhz=args.frequency_mhz,emitter_dbm=args.emitter_dbm,
                        receiver_gain_dbi=args.receiver_gain_dbi,terrain_step_m=args.terrain_step_m,
                        progress=True)
    meta.update({'schema':'station_volume_v1','evidence':'simulation_only',
                 'measurement_validation':False,'teacher':'FSPL_plus_dominant_geometry_knife_edge',
                 'coarse':'FSPL_only','station_cases':'sampled_hypothetical_monitoring_receivers',
                 'teacher_parameters':{'max_loss_db':60.,'effective_earth_radius_factor':4/3,
                                       'terrain_step_m':args.terrain_step_m},
                 'input_channels':['solid_occupancy','ground_minus_z','surface_minus_z',
                                   'relative_x','relative_y','relative_z','fspl_power','agl'],
                 'building_model':'vertical_prisms_flat_roof_on_representative_point_DEM',
                 'not_modelled':['reflections','multi_edge_combination','diffuse_scattering','materials','vegetation'],
                 'split':'station_x_blocks_60_20_20_with_sampling_guard_bands_same_scene',
                 'vertical_axis':'uniform_absolute_elevation_same_as_DEM',
                 'arguments':vars(args),'build_seconds':time.perf_counter()-start,
                 'split_counts':{name:int((data['split']==i).sum()) for i,name in enumerate(['train','val','test'])}})
    data['metadata_json'] = np.asarray(json.dumps(meta,ensure_ascii=False))
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**data)
    output.with_suffix('.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'output':str(output),'split_counts':meta['split_counts'],'build_seconds':meta['build_seconds']}))


if __name__ == '__main__':
    main()
