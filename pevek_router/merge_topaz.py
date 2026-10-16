"""Подмена течений (и, по желанию, толщины/сплочённости льда) в env.nc полями Copernicus TOPAZ5.

Файл TOPAZ5 скачивается пользователем (нужна бесплатная учётка Copernicus Marine), например:
    copernicusmarine subset --dataset-id cmems_mod_arc_phy_anfc_6km_detided_PT1H-i \\
        --variable vxo --variable vyo --variable siconc --variable sithick \\
        --minimum-longitude 148 --maximum-longitude 180 --minimum-latitude 68 --maximum-latitude 78.5 \\
        --start-datetime 2026-10-16T00:00:00 --end-datetime 2026-10-21T00:00:00 \\
        --output-filename topaz_pevek.nc

Затем:
    python -m pevek_router.merge_topaz data_open/env.nc topaz_pevek.nc data_open/env_topaz.nc [--ice]
    python -m pevek_router.run_pilot --env data_open/env_topaz.nc --out out_topaz --departures 0 24 48
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from .grid import make_grid
from .loaders import _find_var, _latlon, regrid


def _series(ds, var, grid, times):
    lat2, lon2 = _latlon(ds)
    tname = "time" if "time" in ds[var].dims else ds[var].dims[0]
    src_t = pd.to_datetime(ds[tname].values)
    arr = ds[var]
    if "depth" in arr.dims:
        arr = arr.isel(depth=0)
    out, cache = [], {}
    for t in times:
        k = int(np.argmin(np.abs(src_t - t)))
        if k not in cache:
            v = arr.isel({tname: k}).values.squeeze().astype(float)
            cache[k] = regrid(lat2, lon2, v, grid)
        out.append(cache[k])
    gap_h = float(np.max(np.min(np.abs(src_t.values[None, :] - times.values[:, None]), axis=1))
                  / np.timedelta64(1, "h"))
    return np.array(out), gap_h


def _rotate_xy_currents(ds):
    """Если в файле течения заданы по осям сетки (vxo, vyo), пересчитываем их в восток/север (_ue, _vn).
    Угол оси x сетки относительно востока находим численно по 2D-полям широты и долготы."""
    names = {v: ds[v].attrs.get("standard_name", "") for v in ds.data_vars}
    vx = next((v for v, sn in names.items() if v == "vxo" or sn == "sea_water_x_velocity"), None)
    vy = next((v for v, sn in names.items() if v == "vyo" or sn == "sea_water_y_velocity"), None)
    if vx is None or vy is None:
        return
    lat, lon = _latlon(ds)
    if lat.ndim == 1:  # регулярная сетка — оси уже восток/север
        ds["_ue"], ds["_vn"] = ds[vx], ds[vy]
        return
    xdim = ds[vx].dims[-1]
    ax = list(ds["latitude" if "latitude" in ds else "lat"].dims).index(xdim)
    dlon = np.gradient(np.unwrap(np.radians(lon), axis=ax), axis=ax)
    dlat = np.gradient(np.radians(lat), axis=ax)
    theta = np.arctan2(dlat, dlon * np.cos(np.radians(lat)))  # угол оси x от востока против часовой
    c = xr.DataArray(np.cos(theta), dims=ds[vx].dims[-2:])
    s_ = xr.DataArray(np.sin(theta), dims=ds[vx].dims[-2:])
    ds["_ue"] = ds[vx] * c - ds[vy] * s_
    ds["_vn"] = ds[vx] * s_ + ds[vy] * c
    print(f"  течения {vx}/{vy} повернуты из осей сетки в восток/север")


def merge(env_path, topaz_path, out_path, ice=False):
    env = xr.open_dataset(env_path).load()
    grid = make_grid()
    times = pd.DatetimeIndex(env.time.values)
    tz = xr.open_dataset(topaz_path)
    water = ~grid.land
    used = []
    _rotate_xy_currents(tz)  # TOPAZ5 хранит vxo/vyo вдоль осей полярной стереографической сетки
    pairs = [("uc", ("eastward_sea_water_velocity",), ("uo", "_ue")),
             ("vc", ("northward_sea_water_velocity",), ("vo", "_vn"))]
    if ice:
        pairs += [("sit", ("sea_ice_thickness",), ("sithick",)),
                  ("sic", ("sea_ice_area_fraction",), ("siconc",))]
    for key, std, names in pairs:
        v = _find_var(tz, std, names)
        arr, gap = _series(tz, v, grid, times)
        if key == "sic" and np.nanmax(arr) > 1.5:
            arr = arr / 100.0
        if gap > 6:
            print(f"  внимание: {v} — ближайший срок TOPAZ отстоит до {gap:.0f} ч от сроков окружения")
        fill = 0.0
        env[key] = (("time", "lat", "lon"),
                    np.where(water[None], np.nan_to_num(np.clip(arr, 0, None) if key in ("sit", "sic") else arr,
                                                        nan=fill), np.nan if key == "sic" else 0.0))
        used.append(key)
    src = env.attrs.get("source", "")
    if ice:
        src = src.split("; течения и толщина льда RTOFS")[0]
        add = "течения, сплочённость и толщина льда TOPAZ5 (Copernicus Marine)"
    else:
        src = src.replace("течения и толщина льда RTOFS", "толщина льда RTOFS")
        add = "течения TOPAZ5 (Copernicus Marine)"
    env.attrs["source"] = src.rstrip(".") + "; " + add
    env.to_netcdf(out_path)
    print("Заменено:", used, "→", out_path)
    return env


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("topaz")
    ap.add_argument("out")
    ap.add_argument("--ice", action="store_true", help="также взять толщину и сплочённость льда из TOPAZ5")
    a = ap.parse_args()
    merge(a.env, a.topaz, a.out, a.ice)
