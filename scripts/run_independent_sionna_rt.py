"""Independent EM calculation: this script cannot read targets or event tensors."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, os, struct, time
import numpy as np

ALLOWED={'scene_id','case_indices','x_m','y_m','z_m','terrain_m','surface_m',
         'station_xyz_m','query_xyz_m','query_flat_indices'}

def write_ply(path,vertices,faces):
    vertices=np.asarray(vertices,dtype='<f4');faces=np.asarray(faces,dtype='<i4')
    header=('ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n'
            'element face %d\nproperty list uchar int vertex_indices\nend_header\n')%(len(vertices),len(faces))
    with Path(path).open('wb') as out:
        out.write(header.encode());out.write(vertices.tobytes())
        records=np.empty(len(faces),dtype=[('count','u1'),('indices','<i4',(3,))]);records['count']=3;records['indices']=faces
        out.write(records.tobytes())

def make_meshes(data,folder):
    folder.mkdir(parents=True,exist_ok=True)
    x,y=np.asarray(data['x_m']),np.asarray(data['y_m']);t=np.asarray(data['terrain_m']);s=np.asarray(data['surface_m'])
    offset=float(np.median(t));ny,nx=t.shape
    yy,xx=np.meshgrid(y,x,indexing='ij');vertices=np.stack([xx,yy,t-offset],axis=-1).reshape(-1,3)
    faces=[]
    for j in range(ny-1):
        for i in range(nx-1):
            a=j*nx+i;b=a+1;c=a+nx;d=c+1;faces.extend([[a,b,d],[a,d,c]])
    write_ply(folder/'terrain.ply',vertices,faces)
    edges=lambda a:np.r_[a[0]-(a[1]-a[0])/2,(a[:-1]+a[1:])/2,a[-1]+(a[-1]-a[-2])/2]
    xe,ye=edges(x),edges(y);occupied=s>t+1.
    verts=[];tri=[];lookup={}
    def quad(points):
        indices=[]
        for point in points:
            key=tuple(np.asarray(point,dtype=np.float32))
            if key not in lookup:lookup[key]=len(verts);verts.append(key)
            indices.append(lookup[key])
        a,b,c,d=indices;tri.extend([[a,b,c],[a,c,d]])
    base=float(t.min()-offset-5.)
    for j,i in np.argwhere(occupied):
        x0,x1=xe[i:i+2];y0,y1=ye[j:j+2];h=float(s[j,i]-offset)
        quad([(x0,y0,h),(x1,y0,h),(x1,y1,h),(x0,y1,h)])
        for dj,di,endpoints in [(-1,0,[(x0,y0),(x1,y0)]),(0,1,[(x1,y0),(x1,y1)]),
                                (1,0,[(x1,y1),(x0,y1)]),(0,-1,[(x0,y1),(x0,y0)])]:
            nj,ni=j+dj,i+di
            low=float(s[nj,ni]-offset) if 0<=nj<ny and 0<=ni<nx and occupied[nj,ni] else base
            if low>=h-1e-5:continue
            (a,b),(c,d)=endpoints;quad([(a,b,low),(c,d,low),(c,d,h),(a,b,h)])
    if len(tri):write_ply(folder/'buildings.ply',verts,tri)
    return offset,dict(terrain_vertices=len(vertices),terrain_triangles=len(faces),building_cells=int(occupied.sum()),
                       building_vertices=len(verts),building_triangles=len(tri),terrain_relief_m=float(np.ptp(t)))

def configure_cache(root):
    for var,name in [('XDG_CACHE_HOME','xdg'),('DRJIT_CACHE_DIR','drjit'),('CUDA_CACHE_PATH','cuda')]:
        directory=root/'cache'/name;directory.mkdir(parents=True,exist_ok=True);os.environ[var]=str(directory.resolve())

def configure_scene(data,folder,materials):
    from sionna.rt import load_scene,SceneObject,ITURadioMaterial,PlanarArray
    offset,geometry=make_meshes(data,folder)
    scene=load_scene();scene.frequency=2.5e9
    array=PlanarArray(num_rows=1,num_cols=1,pattern='iso',polarization='V')
    scene.tx_array=array;scene.rx_array=array
    ground=ITURadioMaterial('ground_material',materials['terrain'],thickness=10.)
    buildings=ITURadioMaterial('building_material',materials['buildings'],thickness=.3)
    objects=[SceneObject(fname=str((folder/'terrain.ply').resolve()),name='terrain',radio_material=ground)]
    if (folder/'buildings.ply').exists():objects.append(SceneObject(fname=str((folder/'buildings.ply').resolve()),name='buildings',radio_material=buildings))
    scene.edit(add=objects)
    return scene,offset,geometry

def coefficients(paths):
    re,im=paths.a
    power=np.asarray(re,dtype=np.float64)**2+np.asarray(im,dtype=np.float64)**2
    return power.reshape(power.shape[0],-1).sum(axis=1)

def verify_free_space(root):
    from sionna.rt import load_scene,PlanarArray,Transmitter,Receiver,PathSolver
    scene=load_scene();scene.frequency=2.5e9
    array=PlanarArray(num_rows=1,num_cols=1,pattern='iso',polarization='V');scene.tx_array=array;scene.rx_array=array
    scene.add(Transmitter('tx',position=[0,0,20],power_dbm=23))
    positions=np.array([[100,0,20],[0,100,50],[100,100,100]],dtype=float)
    for i,p in enumerate(positions):scene.add(Receiver(f'rx{i}',position=p.tolist()))
    paths=PathSolver(deterministic=True)(scene,max_depth=0,los=True,refraction=False,seed=20261107)
    gain=coefficients(paths);received=31+10*np.log10(gain)
    distance=np.linalg.norm(positions-[0,0,20],axis=1)
    exact=31-20*np.log10(4*np.pi*distance*2.5e9/299792458.)
    error=float(np.max(np.abs(received-exact)));assert error<.01,(received,exact)
    (root/'free_space_sanity.json').write_text(json.dumps(dict(rt_dbm=received.tolist(),friis_dbm=exact.tolist(),max_abs_error_db=error),indent=2))
    print(f'Free-space normalization check passed: {error:.6g} dB',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--scene',default=None);ap.add_argument('--mode',choices=['primary','convergence','materials'],default='primary')
    ap.add_argument('--sanity-only',action='store_true')
    ap.add_argument('--backend',choices=['llvm','cuda'],default='llvm')
    args=ap.parse_args();root=args.output_dir.resolve()
    configure_cache(root)
    import mitsuba as mi
    mi.set_variant(args.backend+'_ad_mono_polarized')
    import sionna.rt
    import drjit as dr
    from sionna.rt import Transmitter,Receiver,PathSolver
    versions={p:importlib.metadata.version(p) for p in ['sionna-rt','mitsuba','drjit','numpy']}
    versions['mitsuba_variant']=mi.variant();(root/'solver_environment.json').write_text(json.dumps(versions,indent=2))
    verify_free_space(root)
    if args.sanity_only:return
    lock=json.loads((root/'protocol_lock.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    assert hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()==lock['protocol_sha256']
    hashes=json.loads((root/'input_hashes.json').read_text())
    inputs=sorted((root/'solver_inputs').glob('*.npz'))
    if args.scene:inputs=[p for p in inputs if p.stem==args.scene];assert inputs
    elif args.mode!='primary':inputs=[p for p in inputs if p.stem.endswith('_scene_01')]
    results=root/'rt_results'/args.mode;results.mkdir(parents=True,exist_ok=True)
    for file in inputs:
        output=results/file.name
        if output.exists():print(f'Resume: {file.stem} already complete',flush=True);continue
        assert hashlib.sha256(file.read_bytes()).hexdigest()==hashes[str(file.relative_to(Path.cwd()))] if str(file.relative_to(Path.cwd())) in hashes else hashlib.sha256(file.read_bytes()).hexdigest() in hashes.values()
        with np.load(file,allow_pickle=False) as archive:
            assert set(archive.files)==ALLOWED,archive.files
            data={k:archive[k] for k in archive.files}
        mat=protocol['materials']['sensitivity' if args.mode=='materials' else 'primary']
        start=time.perf_counter();scene,offset,geometry=configure_scene(data,root/'meshes'/args.mode/file.stem,mat)
        mesh_seconds=time.perf_counter()-start
        count=len(data['case_indices']) if args.mode=='primary' else 1
        gains=[];los_flags=[];times=[];path_counts=[]
        solver=PathSolver(deterministic=True)
        for position in range(count):
            case=int(data['case_indices'][position]);station=np.asarray(data['station_xyz_m'][position],float).copy();station[2]-=offset
            scene.add(Transmitter('tx',position=station.tolist(),power_dbm=23.))
            queries=np.asarray(data['query_xyz_m'][position],float).copy();queries[:,2]-=offset
            settings=dict(max_depth=5 if args.mode=='convergence' else protocol['max_depth'],
                samples_per_src=protocol['samples_per_src']*(4 if args.mode=='convergence' else 1),
                max_num_paths_per_src=protocol['max_num_paths_per_src'],
                los=True,specular_reflection=True,diffuse_reflection=False,refraction=False,
                diffraction=True,edge_diffraction=True,diffraction_lit_region=False,seed=20261107+case)
            start=time.perf_counter();gain_parts=[];los_parts=[];count_parts=[]
            for begin in range(0,len(queries),protocol.get('receiver_batch_size',8)):
                chunk=queries[begin:begin+protocol.get('receiver_batch_size',8)]
                for k,q in enumerate(chunk):scene.add(Receiver(f'rx{k}',position=q.tolist()))
                paths=solver(scene,**settings);gain_parts.append(coefficients(paths));dr.sync_thread()
                re,im=paths.a
                count_parts.append(((np.asarray(re)**2+np.asarray(im)**2)>0).reshape(len(chunk),-1).sum(axis=1))
                direct=solver(scene,max_depth=0,los=True,refraction=False,seed=20261107+case)
                los_parts.append(coefficients(direct)>0)
                for k in range(len(chunk)):scene.remove(f'rx{k}')
                del paths,direct
                dr.flush_malloc_cache()
            gain=np.concatenate(gain_parts);los=np.concatenate(los_parts);counts=np.concatenate(count_parts)
            elapsed=time.perf_counter()-start
            assert gain.shape==(192,) and np.isfinite(gain).all() and np.all(gain>=0)
            gains.append(gain);los_flags.append(los);times.append(elapsed);path_counts.append(counts)
            scene.remove('tx')
            print(json.dumps(dict(scene=file.stem,case_index=case,mode=args.mode,seconds=round(elapsed,3),
                                  resolved=int((gain>0).sum()),los=int(los.sum()),queries=len(gain))),flush=True)
        metadata=dict(schema='independent_sionna_rt_sampled_links_v1',solver=versions,
            solver_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            protocol_lock=lock,settings=settings,deterministic=True,materials=mat,
            geometry=geometry,geometry_build_seconds=mesh_seconds,input_file_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            target_or_event_arrays_read=False,power_sum='incoherent',source_power_dbm=23.,additional_receiver_gain_db=8.,
            note='Independent EM implementation on shared raster geometry; assumed materials; no measurements; no calibration to reference targets')
        np.savez_compressed(output,case_indices=data['case_indices'][:count],query_flat_indices=data['query_flat_indices'][:count],
            query_xyz_m=data['query_xyz_m'][:count],path_gain_linear=np.asarray(gains),los=np.asarray(los_flags),
            resolved_path_count=np.asarray(path_counts),solver_seconds=np.asarray(times),metadata_json=json.dumps(metadata))
        output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
