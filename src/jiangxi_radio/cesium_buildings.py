"""Read the uncompressed, Z-up CesiumLab b3dm subset used by nc_build.

Unknown encodings fail explicitly. Heights are unverified source attributes;
mesh consistency does not establish survey accuracy.
"""
import itertools
import json
from pathlib import Path
import struct

import numpy as np
from shapely import set_precision
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform as geom_transform, unary_union


def _matrix(node):
    if 'matrix' in node:
        return np.asarray(node['matrix'], float).reshape(4,4,order='F')
    t = np.eye(4)
    x,y,z,w = node.get('rotation', [0,0,0,1])
    if not np.isclose(x*x+y*y+z*z+w*w, 1):
        raise ValueError('non-unit node quaternion')
    t[:3,:3] = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]) @ np.diag(node.get('scale',[1,1,1]))
    t[:3,3] = node.get('translation',[0,0,0])
    return t


def iter_tile_contents(tileset):
    """Yield all content, including ADD parents; accumulate external transforms."""
    def document(path, parent, ancestors):
        path = path.resolve()
        if path in ancestors:
            raise ValueError('cyclic external tileset')
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if data.get('asset',{}).get('gltfUpAxis','Y') != 'Z':
            raise ValueError(f'only audited Z-up tilesets supported: {path}')
        yield from node(data['root'], parent, path.parent, ancestors | {path})

    def node(item, parent, folder, ancestors):
        matrix = parent @ np.asarray(item.get('transform', np.eye(4).flatten(order='F')), float).reshape(4,4,order='F')
        contents = item.get('contents', [item['content']] if 'content' in item else [])
        for content in contents:
            uri = content.get('uri', content.get('url'))
            path = (folder / uri).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.suffix.lower() == '.json':
                yield from document(path, matrix, ancestors)
            elif path.suffix.lower() == '.b3dm':
                yield path, matrix, content.get('boundingVolume', item.get('boundingVolume'))
            else:
                raise ValueError(f'unsupported tile content: {path}')
        for child in item.get('children') or []:
            yield from node(child, matrix, folder, ancestors)
    yield from document(Path(tileset), np.eye(4), set())


def _apply(points, matrix):
    return np.asarray(points) @ matrix[:3,:3].T + matrix[:3,3]


def decode_b3dm(path):
    raw = Path(path).read_bytes()
    magic,version,length,fj,fb,bj,bb = struct.unpack_from('<4s6I',raw)
    if magic != b'b3dm' or version != 1 or length != len(raw):
        raise ValueError(f'invalid b3dm: {path}')
    feature = json.loads(raw[28:28+fj])
    batch_start = 28+fj+fb
    batch = json.loads(raw[batch_start:batch_start+bj])
    batch_binary = raw[batch_start+bj:batch_start+bj+bb]
    count = feature['BATCH_LENGTH']
    properties = [{} for _ in range(count)]
    types = {'BYTE':'i1','UNSIGNED_BYTE':'u1','SHORT':'<i2','UNSIGNED_SHORT':'<u2',
             'INT':'<i4','UNSIGNED_INT':'<u4','FLOAT':'<f4','DOUBLE':'<f8'}
    for key,value in batch.items():
        if isinstance(value, dict):
            if value.get('type') != 'SCALAR' or value.get('componentType') not in types:
                raise ValueError(f'unsupported batch attribute {key}')
            value = np.frombuffer(batch_binary, dtype=types[value['componentType']],
                                  count=count, offset=value['byteOffset']).tolist()
        if not isinstance(value,list) or len(value) != count:
            raise ValueError(f'invalid batch attribute {key}')
        for p,v in zip(properties,value):
            p[key] = v
    glb = raw[batch_start+bj+bb:]
    magic,version,length = struct.unpack_from('<4sII',glb)
    if magic != b'glTF' or version != 2 or length != len(glb):
        raise ValueError('uncompressed GLB 2 required')
    chunks, offset = {},12
    while offset < len(glb):
        size, kind = struct.unpack_from('<II', glb, offset)
        chunks[kind] = glb[offset+8:offset+8+size]
        offset += 8+size
    data = json.loads(chunks[0x4E4F534A])
    binary = chunks[0x004E4942]
    if data.get('extensionsRequired') or any('uri' in b for b in data['buffers']):
        raise ValueError('compressed/extensions/external buffers are unsupported')
    def accessor(index):
        a = data['accessors'][index]
        if 'sparse' in a or a.get('normalized'):
            raise ValueError('sparse/normalized accessor is unsupported')
        view = data['bufferViews'][a['bufferView']]
        if view.get('buffer',0) != 0 or view.get('extensions'):
            raise ValueError('nonlocal/compressed bufferView unsupported')
        dtype = np.dtype({5120:'i1',5121:'u1',5122:'<i2',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']])
        components = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]
        stride = view.get('byteStride', dtype.itemsize*components)
        start = view.get('byteOffset',0)+a.get('byteOffset',0)
        result = np.ndarray((a['count'],components), dtype=dtype, buffer=binary,
                            offset=start, strides=(stride,dtype.itemsize)).copy()
        return result[:,0] if components == 1 else result
    by_batch = [[] for _ in properties]
    rtc = feature.get('RTC_CENTER',[0,0,0])
    if not isinstance(rtc,list):
        raise ValueError('binary RTC_CENTER unsupported')
    def visit(index, parent, seen):
        if index in seen:
            raise ValueError('cyclic glTF nodes')
        node = data['nodes'][index]
        if node.get('skin') is not None:
            raise ValueError('skinned geometry unsupported')
        matrix = parent @ _matrix(node)
        if 'mesh' in node:
            for primitive in data['meshes'][node['mesh']]['primitives']:
                if primitive.get('mode',4) != 4 or primitive.get('extensions') or primitive.get('targets'):
                    raise ValueError('only static uncompressed triangles supported')
                attrs = primitive['attributes']
                vertices = _apply(accessor(attrs['POSITION']),matrix)+rtc
                batches = accessor(attrs['_BATCHID'])
                if not np.isfinite(vertices).all() or not np.equal(batches,np.round(batches)).all():
                    raise ValueError('invalid vertex or batch id')
                indices = accessor(primitive['indices']) if 'indices' in primitive else np.arange(len(vertices))
                indices = indices.astype(int).reshape(-1,3)
                ids = batches[indices].astype(int)
                if np.any(ids < 0) or np.any(ids >= count) or np.any(ids != ids[:,:1]):
                    raise ValueError('triangle crosses building batch IDs')
                for bid in np.unique(ids[:,0]):
                    by_batch[bid].append(vertices[indices[ids[:,0] == bid]])
        for child in node.get('children',[]):
            visit(child,matrix,seen | {index})
    for index in data['scenes'][data.get('scene',0)]['nodes']:
        visit(index,np.eye(4),set())
    return [{'properties':p, 'triangles':np.concatenate(t)} for p,t in zip(properties,by_batch) if t]


def footprint_from_triangles(triangles):
    # A centimetre grid removes floating-point wall slivers without filling holes.
    polygons = [Polygon(np.round(t[:,:2],2)) for t in triangles]
    polygons = [p for p in polygons if p.area > 1e-5 and p.is_valid]
    if not polygons:
        raise ValueError('no nondegenerate horizontal geometry')
    result = unary_union([set_precision(p,.01) for p in polygons])
    if result.geom_type not in ('Polygon','MultiPolygon') or not result.is_valid:
        raise ValueError('invalid projected footprint')
    return result


def merge_records(records, same_id_policy='reject'):
    if same_id_policy not in ('reject','preserve-mesh-parts'):
        raise ValueError('unknown same-ID policy')
    grouped = {}
    for record in records:
        grouped.setdefault(record['id'],[]).append(record)
    result = []
    for identity, group in grouped.items():
        heights = [r['height'] for r in group]
        if max(heights)-min(heights) > .05:
            if same_id_policy == 'reject':
                raise ValueError(f'conflicting heights for building {identity}: {sorted(set(heights))}; '
                                 'review geometry before opting into preserve-mesh-parts')
            for height in sorted(set(heights)):
                part = [r for r in group if r['height'] == height]
                for record in part:
                    if abs(record['mesh_vertical_span_m']-height) > .5:
                        raise ValueError(f'mesh/attribute height conflict for {identity}')
                for item in merge_records(part):
                    item['source_building_id'] = identity
                    item['id'] = f'{identity}:h{height:g}'
                    item['same_id_height_variants'] = sorted(set(heights))
                    result.append(item)
            continue
        geometries = [r['geometry'] for r in group]
        merged = unary_union(geometries)
        # Disjoint components sharing an ID may be parts, or a reused identity.
        if len(group) > 1 and merged.geom_type == 'MultiPolygon':
            for g in geometries[1:]:
                if g.distance(geometries[0]) > 2:
                    raise ValueError(f'conflicting distant footprints for building {identity}')
        result.append({**group[0], 'geometry':merged, 'source_records':len(group),
                       'geometry_variants':len({g.wkb for g in geometries})})
    return result


def extract_local_buildings(tileset, *, longitude, latitude, size_m=500, same_id_policy='reject'):
    from pyproj import Transformer
    from .volume_simulation import sha256_file
    if not (113 <= longitude <= 119 and 24 <= latitude <= 31 and 50 <= size_m <= 2000):
        raise ValueError('Jiangxi local 50-2000 m ROI required')
    project = Transformer.from_crs(4326,32650,always_xy=True)
    ecef = Transformer.from_crs(4978,32650,always_xy=True)
    inverse = Transformer.from_crs(32650,4326,always_xy=True)
    cx,cy = project.transform(longitude,latitude)
    origin = np.array([cx-size_m/2,cy-size_m/2])
    roi = box(0,0,size_m,size_m)
    def to_local(points):
        x,y,z = ecef.transform(*points.T)
        return np.column_stack((x-origin[0],y-origin[1],z))
    records, sources, visited = [], [], 0
    for path,matrix,bounds in iter_tile_contents(tileset):
        visited += 1
        if bounds and 'box' in bounds:
            v = np.asarray(bounds['box'],float)
            corners = v[:3]+np.array(list(itertools.product([-1,1],repeat=3))) @ v[3:].reshape(3,3)
            local = to_local(_apply(corners,matrix))
            if not box(*local[:,:2].min(axis=0),*local[:,:2].max(axis=0)).buffer(2).intersects(roi):
                continue
        candidates = decode_b3dm(path)
        kept = 0
        for candidate in candidates:
            triangles = to_local(_apply(candidate['triangles'].reshape(-1,3),matrix)).reshape(-1,3,3)
            xy = triangles[:,:,:2].reshape(-1,2)
            if not box(*xy.min(axis=0),*xy.max(axis=0)).intersects(roi):
                continue
            geom = footprint_from_triangles(triangles).intersection(roi)
            if geom.is_empty or geom.area < .01:
                continue
            props = candidate['properties']
            height = props.get('height')
            if height is None or not np.isfinite(height) or height <= 0:
                raise ValueError(f'nonpositive/missing height in {path}')
            records.append({'id':str(props['id']), 'height':float(height), 'geometry':geom,
                            'source_altitude_not_used':props.get('altitude'),
                            'mesh_vertical_span_m':float(np.ptp(triangles[:,:,2]))})
            kept += 1
        if kept:
            sources.append({'path':str(path),'sha256':sha256_file(path),'kept_records':kept})
    merged = merge_records(records,same_id_policy=same_id_policy)
    if not merged:
        raise ValueError('no building geometry intersects ROI')
    def geographic(x,y,z=None):
        return inverse.transform(np.asarray(x)+origin[0],np.asarray(y)+origin[1])
    features = []
    for record in merged:
        properties = {k:v for k,v in record.items() if k != 'geometry'}
        properties['height_evidence'] = 'cesium_model_attribute_unverified'
        features.append({'type':'Feature','properties':properties,
                         'geometry':mapping(geom_transform(geographic,record['geometry']))})
    audit = {'schema':'nc_build_local_v1','height_evidence':'cesium_model_attribute_unverified',
             'measured_height_validation':False,'rsrp_attributes_used':False,
             'tileset':str(tileset),'tileset_sha256':sha256_file(tileset),
             'longitude':longitude,'latitude':latitude,'size_m':size_m,
             'origin_utm_m':origin.tolist(),'coordinate_crs':'EPSG:32650',
             'tile_contents_visited':visited,'source_records':len(records),'building_components':len(merged),
             'unique_source_building_ids':len({r['id'] for r in records}),
             'multi_geometry_ids':sum(r['geometry_variants'] > 1 for r in merged),
             'footprint_method':'union_projected_mesh_triangles_0.01m_snap_clip_ROI',
             'deduplication':'same_ID_same_height_union_all_ADD_LODs',
             'same_id_policy':same_id_policy,
             'multiheight_ids_for_review':{r['source_building_id']:r['same_id_height_variants']
                                           for r in merged if 'source_building_id' in r},
             'height_range_m':[min(r['height'] for r in merged),max(r['height'] for r in merged)],
             'source_tiles':sources}
    return {'type':'FeatureCollection','features':features,'audit':audit}, audit
