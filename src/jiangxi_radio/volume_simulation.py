"""Local, simulation-only 3-D station fields. No measured accuracy claim.

Teacher: FSPL plus a dominant knife edge over terrain and exact footprint
intersections. This is not multipath ray tracing. Inputs never include the
teacher loss; the inexpensive FSPL field is the residual baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from shapely.geometry import LineString, Point, box, shape as geometry_shape
from shapely.ops import transform
from shapely.strtree import STRtree

from .environment_features import knife_edge_diffraction_loss_db


@dataclass
class Building:
    footprint: object
    height_m: float


class Scene:
    def __init__(self, x: np.ndarray, y: np.ndarray, terrain: np.ndarray,
                 buildings: list[Building]):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.terrain = np.asarray(terrain, float)
        if (self.x.ndim != 1 or self.y.ndim != 1 or len(self.x) < 2 or len(self.y) < 2
                or not np.all(np.diff(self.x) > 0) or not np.all(np.diff(self.y) > 0)
                or not np.isfinite(self.x).all() or not np.isfinite(self.y).all()
                or self.terrain.shape != (len(self.y), len(self.x))
                or not np.isfinite(self.terrain).all()):
            raise ValueError('finite increasing axes and finite aligned terrain required')
        self.buildings = buildings
        domain = box(self.x[0], self.y[0], self.x[-1], self.y[-1])
        for b in buildings:
            if (not np.isfinite(b.height_m) or b.height_m <= 0
                    or b.footprint.geom_type not in ('Polygon', 'MultiPolygon')
                    or not b.footprint.is_valid or b.footprint.is_empty
                    or not domain.covers(b.footprint)):
                raise ValueError('positive building height and valid in-domain footprint required')
        self.tree = STRtree([b.footprint for b in buildings])
        self.roofs = np.array([
            self.ground(np.array([[b.footprint.representative_point().x,
                                   b.footprint.representative_point().y]]))[0] + b.height_m
            for b in buildings
        ])

    def ground(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float)
        if (xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all()
                or np.any(xy < [self.x[0], self.y[0]])
                or np.any(xy > [self.x[-1], self.y[-1]])):
            raise ValueError('query outside the finite local terrain domain')
        ix = np.clip(np.searchsorted(self.x, xy[:, 0], side='right')-1, 0, len(self.x)-2)
        iy = np.clip(np.searchsorted(self.y, xy[:, 1], side='right')-1, 0, len(self.y)-2)
        a = (xy[:,0]-self.x[ix])/(self.x[ix+1]-self.x[ix])
        b = (xy[:,1]-self.y[iy])/(self.y[iy+1]-self.y[iy])
        return ((1-a)*(1-b)*self.terrain[iy,ix] + a*(1-b)*self.terrain[iy,ix+1]
                + (1-a)*b*self.terrain[iy+1,ix] + a*b*self.terrain[iy+1,ix+1])

    def surface(self, xy: np.ndarray) -> np.ndarray:
        result = self.ground(xy)
        for i, coords in enumerate(xy):
            point = Point(coords)
            for index in self.tree.query(point, predicate='intersects'):
                result[i] = max(result[i], self.roofs[index])
        return result

    def air_mask(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError('finite xyz coordinates required')
        return xyz[:,2] > self.surface(xyz[:,:2])


def simulate_station(scene: Scene, station: np.ndarray, queries: np.ndarray, *,
                     frequency_mhz: float = 2500., emitter_dbm: float = 23.,
                     receiver_gain_dbi: float = 8., terrain_step_m: float = 10.,
                     max_loss_db: float = 60., earth_factor: float = 4/3) -> dict:
    station, queries = np.asarray(station, float), np.asarray(queries, float)
    if (station.shape != (3,) or not np.isfinite(station).all()
            or queries.ndim != 2 or queries.shape[1] != 3 or not np.isfinite(queries).all()
            or not all(np.isfinite(v) for v in (frequency_mhz, emitter_dbm, receiver_gain_dbi,
                                                 terrain_step_m, max_loss_db, earth_factor))
            or min(frequency_mhz, terrain_step_m, max_loss_db, earth_factor) <= 0):
        raise ValueError('invalid coordinates or propagation parameters')
    scene.ground(queries[:,:2])
    if not scene.air_mask(station[None])[0]:
        raise ValueError('station must be outside solid terrain/buildings')
    distance = np.maximum(np.linalg.norm(queries-station, axis=1), 1.)
    coarse = emitter_dbm + receiver_gain_dbi - (32.45 + 20*np.log10(frequency_mhz)
                                               + 20*np.log10(distance/1000))
    losses = np.zeros(len(queries))
    wavelength = 299792458. / (frequency_mhz * 1e6)
    for i, q in enumerate(queries):
        horizontal = np.linalg.norm(q[:2]-station[:2])
        if horizontal < 1e-6:
            continue
        fractions = np.linspace(0, 1, max(3, int(np.ceil(horizontal/terrain_step_m))+1))[1:-1]
        positions = station[:2] + fractions[:,None]*(q[:2]-station[:2])
        tops = list(scene.ground(positions))
        fs = list(fractions)
        line = LineString([station[:2], q[:2]])
        # Include footprint entry/exit and midpoint, so even a thin wall is
        # examined independently of DEM sampling distance. Polygon holes survive.
        for index in scene.tree.query(line, predicate='intersects'):
            intersection = scene.buildings[index].footprint.intersection(line)
            parts = list(intersection.geoms) if hasattr(intersection, 'geoms') else [intersection]
            for part in parts:
                if part.is_empty:
                    continue
                if part.geom_type == 'LineString':
                    coordinates = [part.coords[0], part.interpolate(.5, normalized=True).coords[0],
                                   part.coords[-1]]
                elif part.geom_type == 'Point':
                    coordinates = [part.coords[0]]
                else:
                    continue
                for coords in coordinates:
                    f = line.project(Point(coords), normalized=True)
                    if 1e-6 < f < 1-1e-6:
                        fs.append(f)
                        tops.append(scene.roofs[index])
        f = np.asarray(fs)
        d1, d2 = np.maximum(f*horizontal, .01), np.maximum((1-f)*horizontal, .01)
        los = station[2] + f*(q[2]-station[2]) - d1*d2/(2*6371000.*earth_factor)
        v = (np.asarray(tops)-los)*np.sqrt(2*(d1+d2)/(wavelength*d1*d2))
        losses[i] = min(float(knife_edge_diffraction_loss_db(v).max()), max_loss_db)
    return {'coarse_dbm': coarse.astype('float32'), 'teacher_dbm': (coarse-losses).astype('float32'),
            'loss_db': losses.astype('float32')}


def split_stations(stations: np.ndarray, x_bounds: tuple[float, float]) -> np.ndarray:
    stations = np.asarray(stations, float)
    width = x_bounds[1]-x_bounds[0]
    if width <= 0 or stations.ndim != 2 or stations.shape[1] != 3:
        raise ValueError('station xyz and increasing bounds required')
    fraction = (stations[:,0]-x_bounds[0])/width
    if not np.isfinite(stations).all() or np.any((fraction < 0)|(fraction > 1)):
        raise ValueError('stations outside split domain')
    return np.where(fraction < .6, 0, np.where(fraction < .8, 1, 2)).astype('int64')


def sample_stations(scene: Scene, count: int, seed: int, antenna_agl_m: float = 20.) -> np.ndarray:
    if count < 6 or not np.isfinite(antenna_agl_m) or antenna_agl_m <= 0:
        raise ValueError('at least six stations and positive antenna height required')
    rng = np.random.default_rng(seed)
    train_count, val_count = int(count*.6), max(1,int(count*.2))
    result = []
    # Guard bands between station regions. These are scenario stations, not
    # fabricated measurements or replacements for the real inventory.
    for lo, hi, n in [(0.03,.55,train_count),(.63,.75,val_count),(.83,.97,count-train_count-val_count)]:
        for _ in range(n):
            for attempt in range(10000):
                x = scene.x[0]+rng.uniform(lo,hi)*(scene.x[-1]-scene.x[0])
                y = rng.uniform(scene.y[0],scene.y[-1])
                q = np.array([x,y,scene.ground(np.array([[x,y]]))[0]+antenna_agl_m])
                # Avoid placing a mast through a building, even if its top clears it.
                if scene.surface(q[None,:2])[0] <= q[2]-antenna_agl_m+1e-5:
                    result.append(q)
                    break
            else:
                raise ValueError('cannot sample unoccupied station footprints in each split')
    return np.asarray(result)


def make_dataset(scene: Scene, *, stations: np.ndarray, shape=(16,24,24),
                 top_agl_m: float = 160., progress: bool = False, **radio) -> dict:
    if len(shape) != 3 or any(int(v) != v or v < 4 for v in shape):
        raise ValueError('volume dimensions must be integers >= 4')
    if not np.isfinite(top_agl_m) or top_agl_m <= 0:
        raise ValueError('positive vertical extent required')
    nz, ny, nx = map(int, shape)
    x = np.linspace(scene.x[0], scene.x[-1], nx)
    y = np.linspace(scene.y[0], scene.y[-1], ny)
    xx, yy = np.meshgrid(x,y)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    ground = scene.ground(xy).reshape(ny,nx)
    roof = scene.surface(xy).reshape(ny,nx)
    z0 = float(ground.min())
    # Uniform absolute-z layers, not warped AGL layers labelled as Cartesian.
    z = z0 + (np.arange(nz)+.5)*(float(ground.max())+top_agl_m-z0)/nz
    zz, yyy, xxx = np.meshgrid(z,y,x,indexing='ij')
    queries = np.column_stack((xxx.ravel(),yyy.ravel(),zz.ravel()))
    air = zz > roof[None]
    if not air.any():
        raise ValueError('volume contains no outdoor air voxels')
    stations = np.asarray(stations,float)
    if stations.ndim != 2 or stations.shape[1] != 3 or len(np.unique(stations,axis=0)) != len(stations):
        raise ValueError('unique station xyz cases required')
    split = split_stations(stations,(x[0],x[-1]))
    if set(split) != {0,1,2}:
        raise ValueError('all three spatial station splits must be nonempty')
    coarse, teacher = [], []
    start = time.perf_counter()
    for index, station in enumerate(stations):
        output = simulate_station(scene,station,queries,**radio)
        coarse.append(output['coarse_dbm'].reshape(shape))
        teacher.append(output['teacher_dbm'].reshape(shape))
        if progress:
            print(f'case={index+1}/{len(stations)} elapsed_s={time.perf_counter()-start:.1f}',flush=True)
    return {'x_m': x.astype('float32'),'y_m': y.astype('float32'),'z_m': z.astype('float32'),
            'terrain_m': ground.astype('float32'),'surface_m': roof.astype('float32'),
            'air_mask': air,'station_xyz_m': stations.astype('float32'),'split': split,
            'coarse_dbm': np.array(coarse),'teacher_dbm': np.array(teacher)}


def input_volume(data: dict, index: int, path_samples: int = 0) -> np.ndarray:
    if not isinstance(path_samples, (int, np.integer)) or not 0 <= path_samples <= 1024:
        raise ValueError('path_samples must be an integer in [0,1024]')
    # Legacy datasets store one common grid/environment.  Multi-scene packs
    # store axes and environment fields per station case so that a case never
    # accidentally receives another scene's terrain or buildings.
    def case_axis(name: str) -> np.ndarray:
        value = np.asarray(data[name])
        return value[index] if value.ndim == 2 else value

    def case_field(name: str) -> np.ndarray:
        value = np.asarray(data[name])
        return value[index] if value.ndim == 3 else value

    x_axis, y_axis, z_axis = (case_axis(name) for name in ('x_m', 'y_m', 'z_m'))
    terrain = case_field('terrain_m')
    surface = case_field('surface_m')
    air_mask = np.asarray(data['air_mask'])
    air_mask = air_mask[index] if air_mask.ndim == 4 else air_mask
    z,y,x = np.meshgrid(z_axis,y_axis,x_axis,indexing='ij')
    scale = max(float(np.ptp(x_axis)),float(np.ptp(y_axis)),float(np.ptp(z_axis)),1.)
    station = data['station_xyz_m'][index]
    ground = np.broadcast_to(terrain,z.shape)
    roof = np.broadcast_to(surface,z.shape)
    inputs = np.stack((~air_mask.astype(bool), (ground-z)/scale, (roof-z)/scale,
                     (x-station[0])/scale,(y-station[1])/scale,(z-station[2])/scale,
                     (data['coarse_dbm'][index]+100.)/50.,(z-ground)/scale)).astype('float32')
    if path_samples:
        from .volume_path_features import path_geometry_features
        path = path_geometry_features(x_axis,y_axis,z_axis,
                                      surface,station,samples=path_samples)
        inputs = np.concatenate((inputs,path),axis=0)
    return inputs


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def load_gba_scene(path: str | Path, dem_dir: str | Path, *, longitude: float,
                   latitude: float, size_m: float = 500., terrain_step_m: float = 10.) -> tuple[Scene, dict]:
    return load_building_scene(path,dem_dir,longitude=longitude,latitude=latitude,
                              size_m=size_m,terrain_step_m=terrain_step_m)


def load_building_scene(path: str | Path, dem_dir: str | Path, *, longitude: float,
                       latitude: float, size_m: float = 500., terrain_step_m: float = 10.,
                       dem_source: str = 'srtm', building_source: str = 'gba') -> tuple[Scene, dict]:
    import ijson
    from pyproj import Transformer
    from .terrain import sample_srtm_elevation
    if not (113 <= longitude <= 119 and 24 <= latitude <= 31):
        raise ValueError('this local UTM50N loader is restricted to the Jiangxi study region')
    if not np.isfinite(size_m) or not 50 <= size_m <= 2000 or not 1 <= terrain_step_m <= 100:
        raise ValueError('use a 50-2000 m local tile and 1-100 m terrain sampling step')
    if dem_source not in ('srtm','fabdem') or building_source not in ('gba','nc_build','overture'):
        raise ValueError('explicit supported terrain and building sources required')
    if building_source in ('nc_build','overture'):
        with Path(path).open('rb') as stream:
            audit = next(ijson.items(stream,'audit',use_float=True),None)
        expected_schema = ('nc_build_local_v1' if building_source == 'nc_build'
                           else 'overture_local_v1')
        if not audit or audit.get('schema') != expected_schema:
            raise ValueError(f'{building_source} input has no matching audited local schema')
        if not np.allclose([longitude,latitude,size_m],
                           [audit['longitude'],audit['latitude'],audit['size_m']],rtol=0,atol=1e-8):
            raise ValueError('simulation ROI must match the audited building extraction ROI')
    project = Transformer.from_crs(4326,32650,always_xy=True)
    inverse = Transformer.from_crs(32650,4326,always_xy=True)
    cx,cy = project.transform(longitude,latitude)
    x0,y0 = cx-size_m/2,cy-size_m/2
    axis = np.linspace(0,size_m,int(np.ceil(size_m/terrain_step_m))+1)
    xx,yy = np.meshgrid(axis,axis)
    lon,lat = inverse.transform(xx.ravel()+x0,yy.ravel()+y0)
    if dem_source == 'fabdem':
        from .geotiff_terrain import sample_geotiff_elevation
        elevation, terrain_metadata = sample_geotiff_elevation(np.column_stack((lon,lat)),dem_dir)
    else:
        elevation = sample_srtm_elevation(np.column_stack((lon,lat)),dem_dir,nodata_m=np.nan)
        terrain_metadata = {'source':'SRTM','directory':str(dem_dir)}
    if not np.isfinite(elevation).all():
        raise ValueError('missing/invalid DEM in local tile; refusing silent flat ground')
    corner_lon,corner_lat = inverse.transform([x0,x0,x0+size_m,x0+size_m],
                                               [y0,y0+size_m,y0,y0+size_m])
    bounds = box(min(corner_lon),min(corner_lat),max(corner_lon),max(corner_lat))
    buildings, skipped_invalid, low_heights = [],0,0
    def local_project(x,y,z=None):
        a,b = project.transform(x,y)
        return np.asarray(a)-x0,np.asarray(b)-y0
    with Path(path).open('rb') as stream:
        for feature in ijson.items(stream,'features.item',use_float=True):
            geom = geometry_shape(feature['geometry'])
            if not geom.intersects(bounds):
                continue
            properties = feature.get('properties') or {}
            h = properties.get('height', properties.get('height_m'))
            if not geom.is_valid or h is None or not np.isfinite(h) or h <= 0:
                skipped_invalid += 1
                continue
            clipped = transform(local_project,geom).intersection(box(0,0,size_m,size_m))
            if clipped.is_empty or clipped.geom_type not in ('Polygon','MultiPolygon'):
                continue
            low_heights += int(h < 2)
            buildings.append(Building(clipped,float(h)))
    if skipped_invalid:
        raise ValueError(f'{skipped_invalid} invalid geometry/height records in ROI; audit before simulation')
    if not buildings and building_source != 'overture':
        raise ValueError('no buildings found; check WGS84 geometry, ROI and path')
    scene = Scene(axis,axis,elevation.reshape(len(axis),len(axis)),buildings)
    metadata = {'source_geometry': str(path),'source_sha256':sha256_file(path),
                'dem_dir':str(dem_dir),'coordinate_crs':'EPSG:32650',
                'origin_utm_m':[x0,y0],'longitude':longitude,'latitude':latitude,
                'size_m':size_m,'buildings':len(buildings),'height_below_2m_count':low_heights,
                'terrain':terrain_metadata,'building_source':building_source,
                'height_evidence':(audit.get('height_evidence')
                                   if building_source in ('nc_build','overture')
                                   else 'GBA_remote_sensing_estimate_used_as_simulation_parameter'),
                'building_base':'representative_point_DEM; source_altitude_ignored'}
    if building_source in ('nc_build','overture'):
        metadata['building_audit'] = audit
    return scene,metadata


def demo_scene() -> Scene:
    axis = np.linspace(0,240,25)
    xx,yy = np.meshgrid(axis,axis)
    terrain = 8*np.exp(-((xx-180)**2+(yy-150)**2)/2500)
    buildings = [Building(box(60,30,85,100),55), Building(box(120,100,145,200),80),
                 Building(box(30,165,80,190),35), Building(box(175,25,205,65),45)]
    return Scene(axis,axis,terrain,buildings)
