#!/usr/bin/env python3
"""Plot province-wide station distribution and a Nanchang spatial zoom.

These maps show station and validation-scene locations only. They do not
represent a continuous province-wide radio-coverage prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jiangxi_radio.paper_visualization import (
    activate_publication_style,
    save_publication_figure,
)
from jiangxi_radio.routes import sample_geodesic_route

CITY_NAMES_EN = {
    "共青城": "Gongqingcheng",
    "景德镇": "Jingdezhen",
    "南昌": "Nanchang",
    "吉安": "Jian",
    "井冈山": "Jinggangshan",
    "兴国": "Xingguo",
    "于都": "Yudu",
    "瑞金": "Ruijin",
    "赣州": "Ganzhou",
}


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame


def _plot_stations(axis, stations: pd.DataFrame, *, legend: bool) -> None:
    grades = sorted(stations["grade"].dropna().unique().tolist()) if "grade" in stations else []
    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
    if grades:
        for index, grade in enumerate(grades):
            subset = stations[stations.grade == grade]
            axis.scatter(
                subset.longitude,
                subset.latitude,
                s=25 if legend else 18,
                color=palette[index % len(palette)],
                edgecolors="white",
                linewidths=0.35,
                label=f"Grade {grade} (n={len(subset)})" if legend else None,
                zorder=3,
            )
    else:
        axis.scatter(
            stations.longitude,
            stations.latitude,
            s=20,
            color="#0072B2",
            edgecolors="white",
            linewidths=0.35,
            label=f"Monitoring stations (n={len(stations)})" if legend else None,
            zorder=3,
        )


def _plot_scene_centres(axis, scenes: pd.DataFrame) -> None:
    groups = sorted(scenes.geo_group.astype(str).unique()) if "geo_group" in scenes else []
    palette = ["#D55E00", "#56B4E9", "#009E73", "#CC79A7", "#E69F00"]
    for index, group in enumerate(groups):
        subset = scenes[scenes.geo_group.astype(str) == group]
        axis.scatter(
            subset.center_longitude,
            subset.center_latitude,
            s=39,
            marker="x",
            linewidths=1.0,
            color=palette[index % len(palette)],
            label=f"Validation scenes: {group}",
            zorder=4,
        )


def _route_lines(routes: pd.DataFrame, cities: pd.DataFrame) -> list[tuple[str, object, str]]:
    city_xy = {
        str(row.city).strip(): (float(row.longitude), float(row.latitude))
        for row in cities.itertuples(index=False)
    }
    lines = []
    for route_index, row in enumerate(routes.itertuples(index=False), start=1):
        include = str(getattr(row, "include_in_study", "true")).strip().lower()
        if include in {"false", "0", "no", "否"}:
            continue
        names = [name.strip() for name in str(getattr(row, "waypoints", "")).split("|") if name.strip()]
        if len(names) < 2:
            names = [str(getattr(row, "origin", "")).strip(), str(getattr(row, "destination", "")).strip()]
        names = [name for name in names if name]
        missing = [name for name in names if name not in city_xy]
        if len(names) < 2 or missing:
            continue
        sampled = sample_geodesic_route([city_xy[name] for name in names], spacing_km=2.0)
        route_name = "-".join(CITY_NAMES_EN.get(name, f"Waypoint{index + 1}") for index, name in enumerate(names))
        lines.append((f"Route {route_index}: {route_name}", sampled, str(getattr(row, "status", ""))))
    return lines


def _plot_routes(axis, lines: list[tuple[str, object, str]]) -> None:
    palette = ["#6A3D9A", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for index, (route_name, points, status) in enumerate(lines):
        active = any(token in status for token in ("已开通", "运营中", "运行"))
        color = palette[index % len(palette)] if active else "#666666"
        axis.plot(
            points[:, 0], points[:, 1],
            color=color,
            linewidth=2.0,
            linestyle="-" if active else "--",
            alpha=0.92,
            label=route_name,
            zorder=7,
        )
        middle = len(points) // 2
        if len(points) > 1:
            xy, xytext = _route_arrow_points(points)
            axis.annotate(
                "", xy=xy, xytext=xytext,
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.35},
                zorder=8,
            )


def _route_arrow_points(points):
    middle = len(points) // 2
    next_index = middle + (middle < len(points) - 1)
    return points[next_index, :2], points[middle, :2]


def _plot_actions(axis, actions: pd.DataFrame) -> None:
    styles = {
        "new": ("#D55E00", "New station", "o"),
        "upgrade": ("#E69F00", "Upgrade", "^"),
        "relocate": ("#009E73", "Relocation", "s"),
    }
    required = {"action_type", "new_longitude", "new_latitude"}
    missing = sorted(required - set(actions.columns))
    if missing:
        raise ValueError(f"action table is missing columns: {missing}")
    for kind, (color, label, marker) in styles.items():
        subset = actions[actions.action_type.astype(str).str.lower() == kind].dropna(
            subset=["new_longitude", "new_latitude"]
        )
        if not subset.empty:
            axis.scatter(
                subset.new_longitude, subset.new_latitude, s=38, color=color,
                marker=marker, edgecolors="white", linewidths=0.45,
                label=f"{label} (n={len(subset)})", zorder=6,
            )


def _decorate(axis) -> None:
    axis.set_xlabel("Longitude (degrees)")
    axis.set_ylabel("Latitude (degrees)")
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.set_aspect("equal", adjustable="box")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", type=Path, default=Path("data/stations.csv"))
    parser.add_argument("--cities", type=Path, default=Path("data/city_coordinates.csv"))
    parser.add_argument("--routes", type=Path, default=Path("data/routes.csv"))
    parser.add_argument("--actions", type=Path, help="optional action CSV with action_type and new site coordinates")
    parser.add_argument("--action-height-m", type=float, help="optional height_m filter for the action table")
    parser.add_argument("--scene-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--nanchang-bounds",
        type=float,
        nargs=4,
        default=(115.45, 116.35, 28.25, 29.10),
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="longitude/latitude bounds for the Nanchang regional zoom",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"choose a new or empty output directory: {args.output_dir}")

    stations = _read_csv(args.stations, {"station_id", "longitude", "latitude"})
    for column in ("longitude", "latitude"):
        stations[column] = pd.to_numeric(stations[column], errors="coerce")
    stations = stations.dropna(subset=["longitude", "latitude"]).copy()
    if stations.empty:
        raise ValueError("no stations have valid longitude/latitude coordinates")

    cities = _read_csv(args.cities, {"city", "longitude", "latitude"})
    routes = _read_csv(args.routes, {"route_name", "origin", "destination", "waypoints"})
    route_lines = _route_lines(routes, cities)
    actions = _read_csv(
        args.actions,
        {"action_type", "new_longitude", "new_latitude"},
    ) if args.actions else None
    if actions is not None and args.action_height_m is not None:
        if "height_m" not in actions:
            raise ValueError("--action-height-m requires height_m in the action table")
        actions = actions[
            pd.to_numeric(actions.height_m, errors="coerce").eq(args.action_height_m)
        ].copy()
    scenes = None
    if args.scene_manifest:
        scenes = _read_csv(
            args.scene_manifest,
            {"center_longitude", "center_latitude"},
        )

    import matplotlib.pyplot as plt

    activate_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7.0, 6.0), layout="constrained")
    _plot_stations(axis, stations, legend=True)
    if scenes is not None:
        _plot_scene_centres(axis, scenes)
    if actions is not None:
        _plot_actions(axis, actions)
    _plot_routes(axis, route_lines)
    for city_index, city in enumerate(cities.itertuples(index=False)):
        axis.scatter(
            float(city.longitude), float(city.latitude), marker="*", s=45,
            color="#333333", edgecolors="white", linewidths=0.4,
            label="Approximate city waypoint" if city_index == 0 else None,
            zorder=9,
        )
    _decorate(axis)
    axis.set_title("Jiangxi monitoring stations, low-altitude routes and validation sites")
    axis.text(
        0.01, 0.01,
        "Routes connect approximate public city nodes; they are not recorded flight tracks.",
        transform=axis.transAxes, fontsize=6.5, va="bottom",
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), fontsize=6.5, ncol=3)
    province_files = save_publication_figure(
        fig, args.output_dir / "fig_jiangxi_stations_routes_validation_sites"
    )
    plt.close(fig)

    lon_min, lon_max, lat_min, lat_max = args.nanchang_bounds
    local = stations[
        stations.longitude.between(lon_min, lon_max)
        & stations.latitude.between(lat_min, lat_max)
    ]
    if local.empty:
        raise ValueError(
            "Nanchang zoom contains no stations; check --nanchang-bounds "
            "against the station coordinate extent"
        )
    fig, axis = plt.subplots(figsize=(6.2, 5.4), layout="constrained")
    _plot_stations(axis, local, legend=True)
    if actions is not None:
        local_actions = actions[
            pd.to_numeric(actions.new_longitude, errors="coerce").between(lon_min, lon_max)
            & pd.to_numeric(actions.new_latitude, errors="coerce").between(lat_min, lat_max)
        ]
        _plot_actions(axis, local_actions)
    _plot_routes(axis, route_lines)
    nanchang = cities[cities.city.astype(str).str.contains("南昌|Nanchang", case=False, regex=True)]
    if not nanchang.empty:
        axis.scatter(
            nanchang.longitude, nanchang.latitude, marker="*", s=95,
            color="#D55E00", edgecolors="white", linewidths=0.5,
            label="Nanchang city centre (approx.)", zorder=5,
        )
    axis.set_xlim(lon_min, lon_max)
    axis.set_ylim(lat_min, lat_max)
    _decorate(axis)
    axis.set_title("Monitoring stations around Nanchang")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), fontsize=6.5, ncol=2)
    local_files = save_publication_figure(
        fig, args.output_dir / "fig_nanchang_stations_and_routes"
    )
    plt.close(fig)

    print("Province distribution:")
    print("\n".join(province_files))
    print("Nanchang zoom:")
    print("\n".join(local_files))
    print(f"Stations plotted: province={len(stations)}, Nanchang window={len(local)}")
    print("Note: these are location maps, not continuous radio-coverage maps.")


if __name__ == "__main__":
    main()
