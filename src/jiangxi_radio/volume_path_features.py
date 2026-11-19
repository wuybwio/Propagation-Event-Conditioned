"""Geometry-only path descriptors from a coarse surface raster, never labels."""
from __future__ import annotations

import numpy as np

PATH_FEATURE_VERSION = 'coarse_surface_path_v1'
PATH_CHANNELS = ['minimum_clearance_div_50m', 'blocked_sample_fraction',
                 'minimum_clearance_path_fraction']


def path_geometry_features(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                           surface: np.ndarray, station: np.ndarray, *,
                           samples: int = 32, chunk_size: int = 4096) -> np.ndarray:
    """Sample interior straight-line points; bilinear tops may smear roof edges.

    Clearance is line altitude minus surface altitude. Position is measured
    from the station (0) toward the query (1); ties use the first minimum.
    No Fresnel correction, Earth curvature, fine polygons or radio labels.
    """
    if (not isinstance(samples, (int, np.integer)) or not 1 <= samples <= 1024
            or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1):
        raise ValueError('samples must be an integer in [1,1024]; positive chunk size required')
    x,y,z = (np.asarray(v, dtype=float) for v in (x,y,z))
    surface,station = np.asarray(surface,float),np.asarray(station,float)
    if any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all()
           or not np.all(np.diff(a)>0) for a in (x,y,z)):
        raise ValueError('finite increasing grid axes required')
    if surface.shape != (len(y),len(x)) or not np.isfinite(surface).all():
        raise ValueError('finite aligned surface raster required')
    if (station.shape != (3,) or not np.isfinite(station).all()
            or not x[0] <= station[0] <= x[-1] or not y[0] <= station[1] <= y[-1]):
        raise ValueError('finite station within horizontal raster bounds required')
    # Local coordinates avoid large georeferencing offsets affecting ties.
    origin=np.array([x[0],y[0],station[2]])
    x,y,z,surface,station=x-origin[0],y-origin[1],z-origin[2],surface-origin[2],station-origin
    zz,yy,xx = np.meshgrid(z,y,x,indexing='ij')
    queries = np.column_stack((xx.ravel(),yy.ravel(),zz.ravel()))
    fractions = np.arange(1,samples+1,dtype=float)/(samples+1)
    result = np.empty((3,len(queries)),dtype='float32')
    for start in range(0,len(queries),chunk_size):
        q=queries[start:start+chunk_size]
        points=station+(q[:,None,:]-station)*fractions[None,:,None]
        sx,sy=points[...,0],points[...,1]
        ix=np.clip(np.searchsorted(x,sx,side='right')-1,0,len(x)-2)
        iy=np.clip(np.searchsorted(y,sy,side='right')-1,0,len(y)-2)
        a=(sx-x[ix])/(x[ix+1]-x[ix]); b=(sy-y[iy])/(y[iy+1]-y[iy])
        tops=((1-a)*(1-b)*surface[iy,ix]+a*(1-b)*surface[iy,ix+1]
              +(1-a)*b*surface[iy+1,ix]+a*b*surface[iy+1,ix+1])
        clearance=points[...,2]-tops
        minimum=clearance.argmin(axis=1)
        result[0,start:start+len(q)]=clearance[np.arange(len(q)),minimum]/50.
        result[1,start:start+len(q)]=(clearance<0).mean(axis=1)
        result[2,start:start+len(q)]=fractions[minimum]
    return result.reshape(3,len(z),len(y),len(x))
