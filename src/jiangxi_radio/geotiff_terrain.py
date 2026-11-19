"""Strict WGS84, north-up GeoTIFF mosaic sampling using actual raster bounds."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np


def sample_geotiff_elevation(points, directory):
    """Bilinear elevation; fail on uncovered/NoData neighbours, including seams.

    Rasterio already accounts for PixelIsPoint georeferencing. Do not apply a
    second half-pixel correction to its transform or infer bounds from filenames.
    """
    import rasterio
    from rasterio.windows import Window
    from .volume_simulation import sha256_file
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('finite (N,2) longitude/latitude points required')
    files = sorted(Path(directory).glob('*.tif'))
    if not files:
        raise ValueError(f'no GeoTIFF coverage in {directory}')
    with ExitStack() as stack:
        datasets = [stack.enter_context(rasterio.open(p)) for p in files]
        base = datasets[0].transform
        for ds in datasets:
            t = ds.transform
            if (ds.crs != rasterio.crs.CRS.from_epsg(4326) or ds.count != 1
                    or t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0
                    or not np.allclose([t.a,t.e], [base.a,base.e], rtol=1e-9, atol=0)):
                raise ValueError('aligned single-band north-up EPSG:4326 tiles required')
            offsets = np.array([(t.c-base.c)/base.a, (t.f-base.f)/base.e])
            if not np.allclose(offsets, np.round(offsets), atol=1e-5, rtol=0):
                raise ValueError('GeoTIFF mosaic pixel centres are not aligned')
        cx = (points[:,0]-base.c)/base.a - .5
        cy = (points[:,1]-base.f)/base.e - .5
        ix, iy = np.floor(cx), np.floor(cy)
        wx, wy = cx-ix, cy-iy
        result = np.zeros(len(points))
        used = set()
        for dx,dy,weight in [(0,0,(1-wx)*(1-wy)), (1,0,wx*(1-wy)),
                              (0,1,(1-wx)*wy), (1,1,wx*wy)]:
            active = weight > 1e-8
            lon = base.c + (ix+dx+.5)*base.a
            lat = base.f + (iy+dy+.5)*base.e
            found = np.zeros(len(points), bool)
            for index, ds in enumerate(datasets):
                col = np.floor((lon-ds.transform.c)/ds.transform.a).astype(int)
                row = np.floor((lat-ds.transform.f)/ds.transform.e).astype(int)
                selected = active & ~found & (col >= 0) & (col < ds.width) & (row >= 0) & (row < ds.height)
                if not selected.any():
                    continue
                cc, rr = col[selected], row[selected]
                window = Window(int(cc.min()), int(rr.min()), int(np.ptp(cc)+1), int(np.ptp(rr)+1))
                raster = ds.read(1, window=window, masked=True)
                values = raster[rr-rr.min(), cc-cc.min()]
                if np.ma.getmaskarray(values).any() or not np.isfinite(values.data).all():
                    raise ValueError(f'NoData in required interpolation neighbours: {ds.name}')
                result[selected] += values.data * weight[selected]
                found[selected] = True
                used.add(index)
            if (active & ~found).any():
                bad = points[np.flatnonzero(active & ~found)[0]]
                raise ValueError(f'missing GeoTIFF coverage near {bad.tolist()}; no zero filling')
        metadata = {'source':'FABDEM', 'sampling':'bilinear_across_actual_tile_bounds',
                    'height_evidence':'public_ground_elevation_estimate_not_survey',
                    'tiles':[{'path':str(files[i]), 'sha256':sha256_file(files[i]),
                              'resolution_deg':list(datasets[i].res),
                              'bounds':list(datasets[i].bounds), 'nodata':datasets[i].nodata}
                             for i in sorted(used)]}
    return result, metadata
