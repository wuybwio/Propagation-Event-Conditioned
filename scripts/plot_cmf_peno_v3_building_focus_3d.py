#!/usr/bin/env python3
"""Terrain/building-focused 3-D coverage visualization with orthogonal cuts.

Compared with the enclosing-box plot, this renderer draws the terrain surface
and geometry-derived building volumes explicitly.  It then shows the local
propagation field from the station, across building edges and into nearby
unobstructed air voxels, making obstruction boundaries and spatial diffusion
visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jiangxi_radio.frozen_v3_release import normal_detection_probability
from jiangxi_radio.paper_visualization import activate_publication_style, save_publication_figure
from plot_cmf_peno_v3_all_methods_3d import (
    METHOD_LABELS,
    METHOD_ORDER,
    _case_axis,
    _case_field,
    _case_mask,
    _load_npz,
    _ordered_methods,
    _prediction_arg,
    _select_case,
)


def _edges(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=float)
    if len(centres) < 2 or np.any(np.diff(centres) <= 0):
        raise ValueError("voxel axes must be strictly increasing")
    mid = 0.5 * (centres[:-1] + centres[1:])
    return np.concatenate(([centres[0] - (mid[0] - centres[0])], mid,
                           [centres[-1] + (centres[-1] - mid[-1])]))


def _building_occupancy(terrain: np.ndarray, surface: np.ndarray,
                        z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    footprint = (surface - terrain) > 0.5
    occupied = (
        (z[:, None, None] > terrain[None, :, :] + 0.5)
        & (z[:, None, None] <= surface[None, :, :] + 0.5)
    )
    return footprint, occupied


def _dilate_xy(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    result = np.zeros_like(mask, dtype=bool)
    ny, nx = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.zeros_like(mask, dtype=bool)
            ys = max(0, dy), min(ny, ny + dy)
            xs = max(0, dx), min(nx, nx + dx)
            src_y = max(0, -dy), min(ny, ny - dy)
            src_x = max(0, -dx), min(nx, nx - dx)
            shifted[ys[0]:ys[1], xs[0]:xs[1]] = mask[src_y[0]:src_y[1], src_x[0]:src_x[1]]
            result |= shifted
    return result


def _geometry_crop_indices(x, y, footprint, size_m, station, mode):
    if size_m <= 0:
        return np.arange(len(x)), np.arange(len(y)), (float(np.mean(x)), float(np.mean(y)))
    if mode == "station":
        centres = [(float(station[0]), float(station[1]))]
    else:
        centres = [(float(cx), float(cy)) for cx in x for cy in y]
    best = None
    for cx, cy in centres:
        xi = np.flatnonzero(np.abs(x - cx) <= 0.5 * size_m)
        yi = np.flatnonzero(np.abs(y - cy) <= 0.5 * size_m)
        if len(xi) < 3 or len(yi) < 3:
            continue
        count = int(footprint[np.ix_(yi, xi)].sum())
        candidate = (count, -abs(cx - np.mean(x)) - abs(cy - np.mean(y)), xi, yi, cx, cy)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise ValueError("crop-size-m leaves fewer than three X/Y samples")
    return best[2], best[3], (best[4], best[5])


def _building_faces(axis, terrain, surface, footprint, field, z, x, y,
                    z_reference, cmap, norm, terrain_exaggeration=1.0,
                    building_height_exaggeration=1.0):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    xe, ye = _edges(x), _edges(y)
    faces = []
    face_colors = []
    outlines = []
    for iy, ix in zip(*np.nonzero(footprint)):
        x0, x1 = xe[ix], xe[ix + 1]
        y0, y1 = ye[iy], ye[iy + 1]
        z0 = float((terrain[iy, ix] - z_reference) * terrain_exaggeration)
        z1 = z0 + float(surface[iy, ix] - terrain[iy, ix]) * building_height_exaggeration
        if z1 <= z0 + 0.5:
            continue
        roof_power = float(np.interp(surface[iy, ix] + 2.0, z,
                                     field[:, iy, ix]))
        roof_color = tuple(cmap(norm(roof_power))[:3]) + (0.74,)
        faces.append([(x0, y0, z1), (x1, y0, z1),
                      (x1, y1, z1), (x0, y1, z1)])
        face_colors.append(roof_color)

        # Shared voxel edges are omitted; only the exposed silhouette and
        # meaningful roof steps remain visible above the field sheets.
        sides = (
            (-1, 0, (x0, y0), (x1, y0)),
            (0, 1, (x1, y0), (x1, y1)),
            (1, 0, (x1, y1), (x0, y1)),
            (0, -1, (x0, y1), (x0, y0)),
        )
        for dy, dx, first, second in sides:
            ny, nx = iy + dy, ix + dx
            adjacent = (0 <= ny < len(y) and 0 <= nx < len(x)
                        and footprint[ny, nx])
            neighbor_top = (
                (terrain[ny, nx] - z_reference) * terrain_exaggeration
                + (surface[ny, nx] - terrain[ny, nx]) * building_height_exaggeration
                if adjacent else z0
            )
            if z1 <= neighbor_top + 1.0:
                continue
            wall_base = max(z0, float(neighbor_top))
            faces.append([
                (first[0], first[1], wall_base),
                (second[0], second[1], wall_base),
                (second[0], second[1], z1),
                (first[0], first[1], z1),
            ])
            face_colors.append((0.28, 0.42, 0.52, 0.18))
            outlines.append(((first[0], first[1], z1),
                             (second[0], second[1], z1)))
    if not faces:
        return
    collection = Poly3DCollection(
        faces,
        facecolors=face_colors,
        edgecolor="none",
        alpha=None,
    )
    collection.set_zorder(35)
    axis.add_collection3d(collection)
    outline_collection = Line3DCollection(
        outlines, colors=(0.12, 0.20, 0.25, 0.82), linewidths=0.65,
    )
    outline_collection.set_zorder(40)
    axis.add_collection3d(outline_collection)


def _probability_colors(field, valid, cmap, norm, alpha=0.84):
    colors = cmap(norm(field))
    colors[..., 3] = np.where(valid, alpha, 0.0)
    # ``plot_surface`` consumes one colour per quadrilateral, whereas the
    # coordinate grids contain one extra row and column of vertices.
    return colors[:-1, :-1]


def _draw_coverage_layers(axis, field, air, occupied, terrain, x, y, z,
                          z_reference, cmap, norm, max_voxels,
                          terrain_exaggeration=1.0, height_limit=None):
    """Draw transparent terrain-following coverage layers.

    A solid voxel cloud hides the scene when most values are high.  We instead
    render a few terrain-following layers, which retain the spatial diffusion
    while leaving the terrain and building geometry visible.
    """
    xg, yg = np.meshgrid(x, y)
    z_display = z - z_reference
    # The first layer is deliberately translucent.  A nearly opaque field
    # hides both the terrain relief and the building walls in a 3-D renderer.
    for offset, alpha in ((2.0, 0.16), (16.0, 0.09), (32.0, 0.045)):
        indices = np.argmin(np.abs(z[None, None, :] -
                                   (terrain[:, :, None] + offset)), axis=2)
        yy, xx = np.indices(terrain.shape)
        # Use take_along_axis explicitly: all three arrays are z-y-x, while
        # the terrain-following layer is y-x.  This avoids accidental
        # broadcasting when NumPy advanced-indexing rules differ by shape.
        z_indices = indices[None, ...]
        selected = np.take_along_axis(field, z_indices, axis=0)[0]
        air_layer = np.take_along_axis(air, z_indices, axis=0)[0]
        occupied_layer = np.take_along_axis(
            occupied, z_indices, axis=0,
        )[0]
        # Draw the full terrain-following coverage layer.  Occupied cells are
        # retained and assigned the global minimum colour rather than left
        # blank, so building footprints remain part of the field.
        valid = air_layer | occupied_layer
        selected = np.where(air_layer, selected, norm.vmin)
        valid &= (indices >= 0) & (indices < len(z)) & np.isfinite(selected)
        if not valid.any():
            continue
        colors = cmap(norm(selected))
        colors[..., 3] = np.where(valid, alpha, 0.0)
        z_layer = (terrain - z_reference) * terrain_exaggeration + offset
        if height_limit is not None:
            valid &= z_layer <= height_limit
        axis.plot_surface(
            xg, yg, z_layer,
            facecolors=colors[:-1, :-1],
            shade=False,
            linewidth=0.12,
            edgecolor=(0.16, 0.18, 0.20, 0.10),
            antialiased=True,
            zorder=10,
        )


def _draw_continuous_volume(axis, field, air, occupied, terrain, x, y, z,
                            z_reference, cmap, norm, terrain_exaggeration=1.0,
                            height_limit=None, coverage_alpha=0.18):
    """Draw transparent field sheets through and above the building volume."""
    xg, yg = np.meshgrid(x, y)
    for offset, opacity in ((3.0, coverage_alpha * 0.90),
                            (18.0, coverage_alpha * 0.48),
                            (32.0, coverage_alpha * 0.36),
                            (44.0, coverage_alpha * 0.28)):
        z_layer = (terrain - z_reference) * terrain_exaggeration + offset
        if height_limit is not None and float(np.nanmin(z_layer)) > height_limit:
            continue
        indices = np.argmin(np.abs(z[:, None, None] -
                                   (terrain[None, :, :] + offset)), axis=0)
        sampled = np.take_along_axis(field, indices[None], axis=0)[0]
        air_layer = np.take_along_axis(air, indices[None], axis=0)[0]
        occupied_layer = np.take_along_axis(occupied, indices[None], axis=0)[0]
        valid = air_layer | occupied_layer
        if not valid.any():
            continue
        selected = np.where(air_layer, sampled, norm.vmin)
        selected = np.where(np.isfinite(selected), selected, norm.vmin)
        if height_limit is not None:
            valid &= z_layer <= height_limit
        colors = cmap(norm(selected))
        colors[..., 3] = np.where(valid, opacity, 0.0)
        axis.plot_surface(
            xg, yg, z_layer,
            facecolors=colors[:-1, :-1],
            shade=False,
            linewidth=0.0,
            edgecolor="none",
            antialiased=True,
            zorder=30,
        )


def _draw_top_field(axis, field, z, x, y, z_reference, cmap, norm,
                    height_limit, alpha):
    """A faint upper field closes the building-height display volume."""
    sample_height = height_limit - 4.0
    absolute_height = z_reference + sample_height
    values = np.stack([
        np.interp(absolute_height, z, field[:, iy, ix])
        for iy in range(len(y)) for ix in range(len(x))
    ]).reshape(len(y), len(x))
    colors = cmap(norm(values))
    colors[..., 3] = alpha
    xg, yg = np.meshgrid(x, y)
    axis.plot_surface(
        xg, yg, np.full_like(xg, sample_height),
        facecolors=colors[:-1, :-1], shade=False, linewidth=0,
        edgecolor="none", antialiased=False, zorder=32,
    )


def _draw_back_curtains(axis, field, air, occupied, x, y, z, z_reference,
                        cmap, norm, height_limit, alpha=0.52):
    """Show the local-height field on two rear scene boundaries."""
    z_display = z - z_reference
    heights = np.linspace(float(z_display.min()), height_limit, 24)
    xg, zg = np.meshgrid(x, heights)
    plane = np.stack([np.interp(z_reference + heights, z, field[:, -1, ix])
                      for ix in range(len(x))], axis=1)
    colors = cmap(norm(plane))
    colors[..., 3] = alpha
    axis.plot_surface(
        xg, np.full_like(xg, y[-1]), zg,
        facecolors=colors[:-1, :-1], shade=False, linewidth=0,
        edgecolor="none", antialiased=False, zorder=8,
    )

    yg, zg = np.meshgrid(y, heights)
    plane = np.stack([np.interp(z_reference + heights, z, field[:, iy, -1])
                      for iy in range(len(y))], axis=1)
    colors = cmap(norm(plane))
    colors[..., 3] = alpha * 0.78
    axis.plot_surface(
        np.full_like(yg, x[-1]), yg, zg,
        facecolors=colors[:-1, :-1], shade=False, linewidth=0,
        edgecolor="none", antialiased=False, zorder=8,
    )


def _draw_front_curtains(axis, field, terrain, surface, x, y, z,
                         z_reference, cmap, norm, height_limit, alpha):
    heights = np.linspace(float(max(0.0, z[0] - z_reference)),
                          height_limit, 30)
    xg, zg = np.meshgrid(x, heights)
    plane = np.stack([np.interp(z_reference + heights, z, field[:, 0, ix])
                      for ix in range(len(x))], axis=1)
    solid = z_reference + heights[:, None] <= surface[0, :][None, :]
    plane = np.where(solid, norm.vmin, plane)
    colors = cmap(norm(plane))
    colors[..., 3] = np.where(
        z_reference + heights[:, None] >= terrain[0, :][None, :],
        alpha, 0.0,
    )
    axis.plot_surface(
        xg, np.full_like(xg, y[0]), zg,
        facecolors=colors[:-1, :-1], shade=False, linewidth=0,
        edgecolor="none", antialiased=False, zorder=9,
    )

    yg, zg = np.meshgrid(y, heights)
    plane = np.stack([np.interp(z_reference + heights, z, field[:, iy, 0])
                      for iy in range(len(y))], axis=1)
    solid = z_reference + heights[:, None] <= surface[:, 0][None, :]
    plane = np.where(solid, norm.vmin, plane)
    colors = cmap(norm(plane))
    colors[..., 3] = np.where(
        z_reference + heights[:, None] >= terrain[:, 0][None, :],
        alpha * 0.75, 0.0,
    )
    axis.plot_surface(
        np.full_like(yg, x[0]), yg, zg,
        facecolors=colors[:-1, :-1], shade=False, linewidth=0,
        edgecolor="none", antialiased=False, zorder=9,
    )


def _draw_terrain(axis, terrain, field, air, occupied, x, y, z, z_reference,
                  cmap, norm, terrain_exaggeration=1.0):
    """Draw a softly shaded terrain surface with an extruded outer edge."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    xg, yg = np.meshgrid(x, y)
    terrain_display = (terrain - z_reference) * terrain_exaggeration
    terrain_cmap = LinearSegmentedColormap.from_list(
        "terrain_relief", ["#dce4d0", "#abc4ac", "#759b90"], N=256,
    )
    lo, hi = np.percentile(terrain, [2, 98])
    relief = Normalize(float(lo), float(max(hi, lo + 0.25)))(terrain)
    colors = terrain_cmap(relief)
    slope_y, slope_x = np.gradient(terrain_display, y, x)
    light = np.clip(0.89 - 0.30 * slope_x - 0.22 * slope_y, 0.64, 1.04)
    colors[..., :3] *= light[..., None]
    colors[..., 3] = 1.0
    axis.plot_surface(
        xg, yg, terrain_display,
        facecolors=colors[:-1, :-1],
        shade=False,
        linewidth=0.0,
        edgecolor="none",
        antialiased=True,
        zorder=5,
    )
    levels = np.linspace(float(np.min(terrain_display)),
                         float(np.max(terrain_display)), 7)[1:-1]
    if len(levels) and np.ptp(terrain_display) > 1.0:
        axis.contour(xg, yg, terrain_display, levels=levels,
                     colors="#536e62", linewidths=0.44, alpha=0.48,
                     zorder=6)


def _draw_cut_2d(axis, field, air, occupied, footprint, terrain, surface,
                 x, y, z, z_reference, view, indices, cmap, norm,
                 contour_level, vertical_top):
    zi, xi, yi = indices
    z_display = z - z_reference
    if view == "z":
        values = np.where(air[zi], field[zi], norm.vmin)
        values = np.ma.masked_where(~(air[zi] | occupied[zi]), values)
        axis.pcolormesh(_edges(x), _edges(y), values, cmap=cmap, norm=norm,
                        shading="flat", rasterized=True)
        axis.contour(x, y, occupied[zi].astype(float), levels=[0.5],
                     colors="#24323a", linewidths=0.8)
        finite = values.compressed()
        if finite.size and finite.min() < contour_level < finite.max():
            axis.contour(x, y, values, levels=[contour_level],
                         colors="#1e293b", linewidths=0.9,
                         linestyles="dashed")
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_xlim(float(_edges(x)[0]), float(_edges(x)[-1]))
        axis.set_ylim(float(_edges(y)[0]), float(_edges(y)[-1]))
        axis.set_aspect("equal", adjustable="box")
    else:
        horizontal = y if view == "x" else x
        field_cut = field[:, :, xi] if view == "x" else field[:, yi, :]
        terrain_cut = terrain[:, xi] if view == "x" else terrain[yi, :]
        surface_cut = surface[:, xi] if view == "x" else surface[yi, :]
        building_cut = footprint[:, xi] if view == "x" else footprint[yi, :]
        ground = terrain_cut - z_reference
        roofs = surface_cut - z_reference
        # Interpolate the archived absolute-elevation samples onto an AGL
        # grid, then drape that grid over each terrain column. This removes
        # half-cell gaps between the power raster and the ground surface.
        agl_edges = np.linspace(0.0, max(2.0, vertical_top - ground.min()), 40)
        agl_centres = 0.5 * (agl_edges[:-1] + agl_edges[1:])
        values = np.empty((len(agl_centres), len(horizontal)), dtype=float)
        for column, elevation in enumerate(terrain_cut):
            values[:, column] = np.interp(
                elevation + agl_centres, z, field_cut[:, column],
            )
            values[agl_centres <= surface_cut[column] - elevation, column] = norm.vmin
        ground_edges = np.interp(_edges(horizontal), horizontal, ground)
        horizontal_grid = np.broadcast_to(
            _edges(horizontal)[None, :],
            (len(agl_edges), len(horizontal) + 1),
        )
        height_grid = ground_edges[None, :] + agl_edges[:, None]
        axis.pcolormesh(horizontal_grid, height_grid, values,
                        cmap=cmap, norm=norm, shading="flat",
                        rasterized=True)
        axis.fill_between(horizontal, float(ground.min()) - 3.0, ground,
                          color="#b4c4ac", alpha=1.0, zorder=3)
        axis.plot(horizontal, ground, color="#425d50", linewidth=1.1, zorder=4)
        axis.step(horizontal, np.where(building_cut, roofs, np.nan),
                  where="mid", color="#24323a", linewidth=0.85, zorder=4)
        axis.set_xlabel("Y (m)" if view == "x" else "X (m)")
        axis.set_ylabel("Height (m)")
        axis.set_xlim(float(_edges(horizontal)[0]),
                      float(_edges(horizontal)[-1]))
        axis.set_ylim(float(ground.min() - 3.0),
                      float(vertical_top))
    axis.tick_params(labelsize=7)
    axis.grid(False)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _draw_cut(axis, field, air, near, occupied, footprint, terrain,
              surface, x, y, z, z_reference, view, indices, cmap, norm,
              contour_level, height_limit=None, terrain_exaggeration=1.0):

    zi, xi, yi = indices
    xe, ye = _edges(x), _edges(y)
    z_display = z - z_reference
    if view == "z":
        xg, yg = np.meshgrid(x, y)
        valid = air[zi] | occupied[zi]
        slice_field = np.where(air[zi], field[zi], norm.vmin)
        slice_field = np.where(valid, slice_field, np.nan)
        axis.plot_surface(
            xg, yg, np.full_like(xg, z_display[zi]),
            facecolors=_probability_colors(slice_field, np.isfinite(slice_field), cmap, norm),
            shade=False, linewidth=0.0, antialiased=False,
        )
        axis.contour(xg, yg, occupied[zi].astype(float), levels=[0.5],
                     zdir="z", offset=float(z_display[zi]),
                     colors="#20262b", linewidths=0.75)
        finite = slice_field[np.isfinite(slice_field)]
        if finite.size and finite.min() < contour_level < finite.max():
            axis.contour(xg, yg, slice_field, levels=[contour_level],
                         zdir="z", offset=float(z_display[zi]),
                         colors="#111827", linewidths=0.9)
    elif view == "x":
        yg, zg = np.meshgrid(y, z_display)
        zg = zg + (terrain[:, xi] - z_reference)[None, :] * (terrain_exaggeration - 1.0)
        valid = air[:, :, xi] | occupied[:, :, xi]
        slice_field = np.where(air[:, :, xi], field[:, :, xi], norm.vmin)
        slice_field = np.where(valid, slice_field, np.nan)
        axis.plot_surface(
            np.full_like(yg, x[xi]), yg, zg,
            facecolors=_probability_colors(slice_field, np.isfinite(slice_field), cmap, norm),
            shade=False, linewidth=0.0, antialiased=False,
        )
        axis.contour(yg, zg, occupied[:, :, xi].astype(float), levels=[0.5],
                     zdir="x", offset=float(x[xi]),
                     colors="#20262b", linewidths=0.75)
        axis.plot(np.full_like(y, x[xi]), y,
                  (terrain[:, xi] - z_reference) * terrain_exaggeration,
                  color="#3f3f46", linewidth=1.15, zorder=6)
        axis.plot(np.full_like(y, x[xi]), y,
                  (terrain[:, xi] - z_reference) * terrain_exaggeration +
                  (surface[:, xi] - terrain[:, xi]),
                  color="#263544", linewidth=1.05, zorder=7)
        finite = slice_field[np.isfinite(slice_field)]
        if finite.size and finite.min() < contour_level < finite.max():
            axis.contour(yg, zg, slice_field, levels=[contour_level],
                         zdir="x", offset=float(x[xi]),
                         colors="#111827", linewidths=0.9)
    else:
        xg, zg = np.meshgrid(x, z_display)
        zg = zg + (terrain[yi, :] - z_reference)[None, :] * (terrain_exaggeration - 1.0)
        valid = air[:, yi, :] | occupied[:, yi, :]
        slice_field = np.where(air[:, yi, :], field[:, yi, :], norm.vmin)
        slice_field = np.where(valid, slice_field, np.nan)
        axis.plot_surface(
            xg, np.full_like(xg, y[yi]), zg,
            facecolors=_probability_colors(slice_field, np.isfinite(slice_field), cmap, norm),
            shade=False, linewidth=0.0, antialiased=False,
        )
        axis.contour(xg, zg, occupied[:, yi, :].astype(float), levels=[0.5],
                     zdir="y", offset=float(y[yi]),
                     colors="#20262b", linewidths=0.75)
        axis.plot(x, np.full_like(x, y[yi]),
                  (terrain[yi, :] - z_reference) * terrain_exaggeration,
                  color="#3f3f46", linewidth=1.15, zorder=6)
        axis.plot(x, np.full_like(x, y[yi]),
                  (terrain[yi, :] - z_reference) * terrain_exaggeration +
                  (surface[yi, :] - terrain[yi, :]),
                  color="#263544", linewidth=1.05, zorder=7)
        finite = slice_field[np.isfinite(slice_field)]
        if finite.size and finite.min() < contour_level < finite.max():
            axis.contour(xg, zg, slice_field, levels=[contour_level],
                         zdir="y", offset=float(y[yi]),
                         colors="#111827", linewidths=0.9)

    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(y.min()), float(y.max()))
    if view == "z":
        axis.set_zlim(float(z_display[zi]) - 2.0, float(z_display[zi]) + 2.0)
    else:
        axis.set_zlim(float(min(0.0, (terrain - z_reference).min() * terrain_exaggeration)),
                      float(height_limit if height_limit is not None else z_display.max()))
    axis.set_xlabel("X (m)", labelpad=1, fontsize=8)
    axis.set_ylabel("Y (m)", labelpad=1, fontsize=8)
    if view == "z":
        axis.set_zticks([])
        axis.set_zlabel("")
    else:
        axis.set_zlabel("")
    axis.view_init(elev=27, azim=-55)
    axis.set_box_aspect((1.0, 1.0, 0.08 if view == "z" else 0.72))
    axis.tick_params(labelsize=7, pad=0)
    axis.grid(True, alpha=0.15)


def _draw_3d(axis, probability, air, near, footprint, occupied, terrain,
              surface, x, y, z, z_reference, cmap, norm, max_points,
              terrain_exaggeration=1.0, building_height_exaggeration=1.0,
              height_limit=None, coverage_alpha=0.18):
    _draw_terrain(axis, terrain, probability, air, occupied, x, y, z,
                  z_reference, cmap, norm, terrain_exaggeration)
    _draw_back_curtains(axis, probability, air, occupied, x, y, z,
                        z_reference, cmap, norm, height_limit,
                        alpha=min(0.55, coverage_alpha * 0.60))
    _draw_front_curtains(axis, probability, terrain, surface, x, y, z,
                         z_reference, cmap, norm, height_limit,
                         alpha=min(0.33, coverage_alpha * 0.40))
    _draw_continuous_volume(axis, probability, air, occupied, terrain, x, y, z,
                            z_reference, cmap, norm, terrain_exaggeration,
                            height_limit, coverage_alpha)
    _building_faces(axis, terrain, surface, footprint, probability, z, x, y,
                    z_reference, cmap, norm, terrain_exaggeration,
                    building_height_exaggeration)
    _draw_top_field(axis, probability, z, x, y, z_reference, cmap, norm,
                    height_limit, alpha=min(0.38, coverage_alpha * 0.42))
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(y.min()), float(y.max()))
    axis.set_zlim(float(min(0.0, (terrain - z_reference).min() * terrain_exaggeration)),
                  float(height_limit))
    axis.set_xlabel("X (m)", labelpad=1, fontsize=8)
    axis.set_ylabel("Y (m)", labelpad=1, fontsize=8)
    axis.set_zlabel("")
    axis.view_init(elev=27, azim=-55)
    axis.set_box_aspect((1.0, 1.0, 0.78))
    axis.tick_params(labelsize=7, pad=0)
    axis.grid(True, alpha=0.15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prediction", type=_prediction_arg, action="append", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--threshold-dbm", type=float, default=-100.0)
    parser.add_argument("--shadow-std-db", type=float, default=6.0)
    parser.add_argument("--field-mode", choices=("power", "probability"), default="power",
                        help="plot received power in dBm (recommended) or detection probability")
    parser.add_argument("--coverage-probability", type=float, default=0.9)
    parser.add_argument("--slice-height-m", type=float, default=80.0)
    parser.add_argument("--crop-size-m", type=float, default=300.0,
                        help="local X/Y window centred on the selected station; <=0 keeps the full scene")
    parser.add_argument("--crop-center", choices=("buildings", "station"),
                        default="buildings", help="geometry-only crop selection shared by every method")
    parser.add_argument("--building-radius-cells", type=int, default=2)
    parser.add_argument("--propagation-radius-m", type=float, default=150.0,
                        help="radius around the station and buildings in which the coverage field is shown")
    parser.add_argument("--max-probability-points", type=int, default=1400)
    parser.add_argument("--terrain-exaggeration", type=float, default=4.0,
                        help="visual exaggeration applied to terrain relief in the 3-D panel")
    parser.add_argument("--building-height-exaggeration", type=float, default=2.5,
                        help="visual exaggeration applied to building height differences")
    parser.add_argument("--height-cap-m", type=float, default=-1.0,
                        help="upper display height relative to median terrain; <=0 uses highest building plus margin")
    parser.add_argument("--coverage-alpha", type=float, default=0.18,
                        help="alpha of the 3-D coverage cloud in the terrain/building panel")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"use a new output directory: {args.output_dir}")
    if not 0.0 < args.coverage_probability < 1.0:
        raise ValueError("coverage probability must be strictly between zero and one")
    if args.terrain_exaggeration <= 0:
        raise ValueError("terrain-exaggeration must be positive")
    if args.building_height_exaggeration <= 0:
        raise ValueError("building-height-exaggeration must be positive")
    if args.height_cap_m == 0:
        raise ValueError("height-cap-m must be negative or positive")
    if not 0.0 < args.coverage_alpha <= 1.0:
        raise ValueError("coverage-alpha must be in (0, 1]")
    if args.dpi < 300:
        raise ValueError("dpi must be at least 300")

    data = _load_npz(args.dataset)
    predictions = {}
    case_indices = None
    for name, path in args.prediction:
        archive = _load_npz(path)
        required = {"case_indices", "mean_dbm", "mask"}
        missing = sorted(required - set(archive))
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        current = np.asarray(archive["case_indices"], dtype=int)
        if case_indices is None:
            case_indices = current
        elif not np.array_equal(current, case_indices):
            raise ValueError(f"prediction case indices differ for variant {name}")
        predictions[name] = archive
    if "full_v3" not in predictions:
        raise ValueError("include --prediction full_v3=... to select a representative case")

    case_index, prediction_position = _select_case(
        data, predictions, args.scene_id, args.case_index
    )
    x = _case_axis(data, "x_m", case_index)
    y = _case_axis(data, "y_m", case_index)
    z = _case_axis(data, "z_m", case_index)
    terrain = _case_field(data, "terrain_m", case_index)
    surface = _case_field(data, "surface_m", case_index)
    air = _case_mask(data, case_index)
    station = np.asarray(data["station_xyz_m"][case_index], dtype=float)
    complete_footprint = (surface - terrain) > 0.5
    x_index, y_index, crop_centre = _geometry_crop_indices(
        x, y, complete_footprint, args.crop_size_m, station, args.crop_center
    )
    if args.crop_size_m > 0:
        x = x[x_index]
        y = y[y_index]
        terrain = terrain[np.ix_(y_index, x_index)]
        surface = surface[np.ix_(y_index, x_index)]
        air = air[:, y_index, :][:, :, x_index]
    footprint, occupied = _building_occupancy(terrain, surface, z)
    near_xy = _dilate_xy(footprint, args.building_radius_cells)
    xg_local, yg_local = np.meshgrid(x, y)
    station_radius = (
        (xg_local - station[0]) ** 2 + (yg_local - station[1]) ** 2
        <= args.propagation_radius_m ** 2
    )
    # The union makes the field continuous from the station through building
    # edges and into nearby unobstructed terrain rather than stopping at the
    # first occupied footprint.
    near_xy |= station_radius
    near = np.broadcast_to(near_xy, occupied.shape).copy()
    z_reference = float(np.median(terrain))
    print(
        "Geometry diagnostics: "
        f"terrain=[{float(np.nanmin(terrain)):.2f}, {float(np.nanmax(terrain)):.2f}] m, "
        f"surface=[{float(np.nanmin(surface)):.2f}, {float(np.nanmax(surface)):.2f}] m, "
        f"building_footprints={int(footprint.sum())}, "
        f"building_voxels={int(occupied.sum())}, "
        f"crop_shape={terrain.shape}"
    )
    if not footprint.any():
        raise ValueError(
            "No building geometry remains in the selected case/crop. "
            "Increase --crop-size-m, or choose a case with nonzero "
            "surface_m - terrain_m; changing plotting parameters cannot "
            "recover geometry outside the selected crop."
        )

    def crop_field(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        return value[:, y_index, :][:, :, x_index]

    power_fields = {
        "InternalReference": crop_field(data["fidelity_dbm"][case_index, -1]),
        "FSPL": crop_field(data["coarse_dbm"][case_index]),
        "GeometricTeacher": crop_field(data["fidelity_dbm"][case_index, 1]),
    }
    for name in METHOD_ORDER:
        if name in predictions:
            power_fields[name] = crop_field(predictions[name]["mean_dbm"][prediction_position])
    for name in predictions:
        power_fields.setdefault(name, crop_field(predictions[name]["mean_dbm"][prediction_position]))
    fields = (power_fields if args.field_mode == "power" else {
        name: normal_detection_probability(value, 0.0,
                                          args.threshold_dbm, args.shadow_std_db)
        for name, value in power_fields.items()
    })
    expected = (len(z), len(y), len(x))
    for name, field in fields.items():
        if field.shape != expected or not np.isfinite(field).all():
            raise ValueError(f"{name} {args.field_mode} field shape {field.shape} is invalid; expected {expected}")

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    activate_publication_style()
    mpl.rcParams["font.family"] = "Times New Roman"
    # Times New Roman is explicitly required; the static validator only lists Arial/Helvetica.
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    names = ["InternalReference", "FSPL", "GeometricTeacher"] + [
        name for name in METHOD_ORDER if name in fields
    ]
    # Deep blue -> cyan/green -> orange-red, matching the reference 3-D
    # coverage visualisation rather than the pale yellow-dominant palette.
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "blue_to_orange_red",
        ["#173f8a", "#2b7ab8", "#3ca5b1", "#91c5a0",
         "#e9d477", "#f4a24d", "#de5c32"],
        N=256,
    )
    if args.field_mode == "power":
        norm = Normalize(-125.0, -35.0)
        from scipy.special import ndtri
        contour_level = args.threshold_dbm + args.shadow_std_db * ndtri(args.coverage_probability)
    else:
        norm = Normalize(0.0, 1.0)
        contour_level = args.coverage_probability
    z_display = z - z_reference
    building_top_display = float(np.nanmax(
        (terrain - z_reference) * args.terrain_exaggeration
        + (surface - terrain) * args.building_height_exaggeration
    ))
    auto_height_cap = building_top_display + 18.0
    height_limit = float(args.height_cap_m if args.height_cap_m > 0
                         else auto_height_cap)
    vertical_cut_top = float(np.max(surface - z_reference) + 8.0)
    zi = int(np.argmin(np.abs(z_display - args.slice_height_m)))
    inner_x = np.arange(max(0, len(x) // 4), min(len(x), 3 * len(x) // 4 + 1))
    inner_y = np.arange(max(0, len(y) // 4), min(len(y), 3 * len(y) // 4 + 1))
    xi = int(inner_x[np.argmax(footprint.sum(axis=0)[inner_x])])
    yi = int(inner_y[np.argmax(footprint.sum(axis=1)[inner_y])])
    if footprint[:, xi].sum() == 0 or footprint[yi].sum() == 0:
        raise ValueError("selected vertical cuts must intersect building geometry")
    pages = [names[start:start + 3] for start in range(0, len(names), 3)]
    views = (
        ("3d", "3-D terrain, buildings and field"),
        ("z", f"Z = {z_display[zi]:.0f} m above median terrain"),
        ("x", f"X = {x[xi]:.0f} m"),
        ("y", f"Y = {y[yi]:.0f} m"),
    )
    args.output_dir.mkdir(parents=True)
    files = []
    for page_index, page_names in enumerate(pages, start=1):
        show_error = args.field_mode == "power"
        fig_width = 9.0 if show_error else 7.2
        fig = plt.figure(figsize=(fig_width, 2.18 * len(page_names) + 0.55),
                         layout="constrained")
        grid = fig.add_gridspec(
            len(page_names), 7 if show_error else 5,
            width_ratios=([1.45, 1, 1, 1, 0.055, 1, 0.055]
                          if show_error else [1.45, 1, 1, 1, 0.055]),
        )
        error_norm = Normalize(0.0, 15.0) if show_error else None
        error_cmap = colormaps["magma"] if show_error else None
        for row, name in enumerate(page_names):
            for column, (view, view_title) in enumerate(views):
                if view == "3d":
                    axis = fig.add_subplot(grid[row, column], projection="3d")
                    axis.computed_zorder = False
                    axis.set_proj_type("ortho")
                    _draw_3d(axis, fields[name], air, near, footprint, occupied,
                              terrain, surface, x, y, z, z_reference,
                              cmap, norm, args.max_probability_points,
                              args.terrain_exaggeration,
                              args.building_height_exaggeration,
                              height_limit,
                              args.coverage_alpha)
                else:
                    axis = fig.add_subplot(grid[row, column])
                    _draw_cut_2d(axis, fields[name], air, occupied, footprint,
                                 terrain, surface, x, y, z, z_reference, view,
                                 (zi, xi, yi), cmap, norm, contour_level,
                                 vertical_cut_top)
                label = ("Internal reference" if name == "InternalReference" else
                         "No multi-level" if name == "no_multilevel" else
                         "CMF-PENO-RF" if name == "full_v3" else
                         METHOD_LABELS.get(name, name.replace("_", " ")))
                panel = chr(ord("a") + 4 * row + column)
                axis.set_title(
                    f"({panel}) {label if column == 0 else view_title}", fontsize=7.3,
                    fontweight="bold" if name == "full_v3" else "normal",
                    pad=22 if view == "z" else 2,
                )
            if show_error:
                error_axis = fig.add_subplot(grid[row, 5])
                error = np.ma.masked_where(
                    ~air[zi],
                    np.abs(fields[name][zi] - fields["InternalReference"][zi]),
                )
                error_axis.pcolormesh(_edges(x), _edges(y), error,
                                      cmap=error_cmap, norm=error_norm,
                                      shading="flat", rasterized=True)
                error_axis.contour(x, y, footprint.astype(float), levels=[0.5],
                                   colors="#25343c", linewidths=0.55)
                error_axis.set_xlim(float(_edges(x)[0]), float(_edges(x)[-1]))
                error_axis.set_ylim(float(_edges(y)[0]), float(_edges(y)[-1]))
                error_axis.set_aspect("equal", adjustable="box")
                error_axis.set_xlabel("X (m)")
                error_axis.set_ylabel("Y (m)")
                error_axis.tick_params(labelsize=7)
                error_axis.grid(False)
                error_axis.set_title(
                    f"({chr(ord('m') + row)}) |Error|; MAE {error.mean():.2f} dB",
                    fontsize=7.3, pad=22,
                )
        color_axis = fig.add_subplot(grid[:, 4])
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), cax=color_axis,
            label=("Mean received power (dBm)" if args.field_mode == "power"
                   else "Received-power exceedance probability"),
        )
        color_axis.tick_params(labelsize=7)
        color_axis.yaxis.label.set_size(7.5)
        if show_error:
            error_color_axis = fig.add_subplot(grid[:, 6])
            fig.colorbar(
                ScalarMappable(norm=error_norm, cmap=error_cmap),
                cax=error_color_axis, extend="max",
                label="Absolute error (dB)",
            )
            error_color_axis.tick_params(labelsize=7)
            error_color_axis.yaxis.label.set_size(7.5)
        stem = args.output_dir / f"fig_building_focus_3d_coverage_p{page_index:02d}"
        from jiangxi_radio.paper_visualization import validate_publication_layout
        validate_publication_layout(fig)
        png, pdf, svg, tif = (stem.with_suffix(ext) for ext in (".png", ".pdf", ".svg", ".tif"))
        fig.savefig(png, dpi=300, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        fig.savefig(svg, facecolor="white")
        fig.savefig(tif, dpi=300, facecolor="white")
        files.extend([str(png), str(pdf), str(svg), str(tif)])
        plt.close(fig)

    metadata = {
        "schema": "cmf_peno_rf_building_focus_3d_v1",
        "evidence": ("terrain surface, building volumes and model/reference received-power fields"
                     if args.field_mode == "power" else
                     "terrain surface, building volumes and model/reference detection-probability fields"),
        "dataset": str(args.dataset.resolve()),
        "scene_id": args.scene_id,
        "dataset_case_index": int(case_index),
        "prediction_position": int(prediction_position),
        "methods": names,
        "field_mode": args.field_mode,
        "crop_size_m": float(args.crop_size_m),
        "crop_center_method": args.crop_center,
        "crop_center_xy_m": list(crop_centre),
        "building_footprint_count": int(footprint.sum()),
        "building_voxel_count": int(occupied.sum()),
        "building_radius_cells": int(args.building_radius_cells),
        "propagation_radius_m": float(args.propagation_radius_m),
        "terrain_exaggeration": float(args.terrain_exaggeration),
        "building_height_exaggeration": float(args.building_height_exaggeration),
        "height_cap_m_relative_to_median_terrain": float(height_limit),
        "main_view_height_m_relative_to_median_terrain": float(height_limit),
        "vertical_cut_top_m_relative_to_median_terrain": vertical_cut_top,
        "main_view_rendering": (
            "transparent full-width field sheets and four boundary curtains; "
            "roof colour samples the received-power field 2 m above each roof"
        ),
        "vertical_cut_rendering": (
            "archived absolute-height field samples linearly interpolated "
            "onto a terrain-following AGL display grid"
        ),
        "solid_colour_policy": (
            "building interiors shown at the colour-scale minimum for display only; "
            "excluded from field metrics"
        ),
        "coverage_alpha": float(args.coverage_alpha),
        "max_probability_points": int(args.max_probability_points),
        "coverage_probability_threshold": float(args.coverage_probability),
        "slice_height_m_above_median_terrain": float(z_display[zi]),
        "slice_x_m": float(x[xi]),
        "slice_y_m": float(y[yi]),
        "color_scale": ("fixed -125 to -35 dBm; orange/red is high mean received power"
                        if args.field_mode == "power" else
                        "fixed 0 to 1 mean-only exceedance probability"),
        "error_panels": (
            "absolute difference from the internal reference at the same "
            "horizontal slice; 0-15 dB display scale, with values above 15 dB "
            "saturated; building and invalid voxels excluded"
        ),
        "probability_cutoff_power_dbm": float(args.threshold_dbm +
                                            args.shadow_std_db * ndtri(args.coverage_probability))
                                            if args.field_mode == "power" else None,
        "reference_is_internal_rule_based": True,
        "dpi": args.dpi,
        "files": files,
    }
    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
