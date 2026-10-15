"""Загрузка реальных данных и приведение их к сетке района.

Итоговый формат «окружения» (xarray.Dataset, оси time, lat, lon) — тот же, что у synthetic.make_scenario:
    sic            сплочённость льда, 0..1
    sit            толщина льда, м
    u10, v10       ветер на 10 м (на восток / на север), м/с
    hs             значительная высота волн, м
    wave_dir_from  направление, ОТКУДА идут волны, град (метеорологическое)
    uc, vc         поверхностное течение (на восток / на север), м/с

Источники (каждый необязателен — чего нет, заполняется нулями/климатом с предупреждением):
    ERA5 (Copernicus CDS, пакет `cdsapi`)                 — ветер, волны (архив/реанализ, задержка ~5 сут)
    Copernicus Marine ARC-MFC (пакет `copernicusmarine`)  — лёд, течения (TOPAZ), волны (прогноз 10 сут)
    OSI SAF OSI-401 (THREDDS met.no, OPeNDAP)             — сплочённость льда по пассивному СВЧ, ~10 км
    CS2SMOS / ESA Sea Ice CCI (Sentinel-3, CryoSat-2)     — толщина льда (зимний сезон), из локальных файлов

Учётные данные: ~/.cdsapirc для CDS; `copernicusmarine login` для Copernicus Marine.
"""
from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

from .grid import Grid

# ----------------------------------------------------------------------------- общие утилиты

def regrid(src_lat, src_lon, values, grid: Grid, method="linear"):
    """Интерполяция поля с произвольной (в т. ч. криволинейной) сетки на сетку района."""
    src_lat = np.asarray(src_lat, float)
    src_lon = np.asarray(src_lon, float) % 360.0
    if src_lat.ndim == 1 and src_lon.ndim == 1 and np.asarray(values).ndim == 2:
        src_lon, src_lat = np.meshgrid(src_lon, src_lat)
    v = np.asarray(values, float).ravel()
    pts = np.column_stack([src_lat.ravel(), src_lon.ravel()])
    ok = np.isfinite(v) & np.isfinite(pts).all(axis=1)
    LON, LAT = np.meshgrid(grid.lon % 360.0, grid.lat)
    out = griddata(pts[ok], v[ok], (LAT, LON), method=method)
    if method != "nearest":  # дыры у берега заполняем ближайшим значением
        near = griddata(pts[ok], v[ok], (LAT, LON), method="nearest")
        out = np.where(np.isfinite(out), out, near)
    return out


def _bbox(grid: Grid, pad=0.6):
    return (float(grid.lat.min() - pad), float(grid.lat.max() + pad),
            float(grid.lon.min() - 1.5 * pad), float(min(grid.lon.max() + 1.5 * pad, 180.0)))


def _find_var(ds: xr.Dataset, standard_names=(), names=()):
    for v in ds.data_vars:
        if ds[v].attrs.get("standard_name") in standard_names or v in names:
            return v
    raise KeyError(f"Не найдена переменная {standard_names or names}; есть: {list(ds.data_vars)}")


def _latlon(ds):
    for la, lo in (("latitude", "longitude"), ("lat", "lon"), ("nav_lat", "nav_lon")):
        if la in ds.variables and lo in ds.variables:
            return ds[la].values, ds[lo].values
    raise KeyError("В наборе нет координат широты/долготы")


def _to_grid_series(ds, var, grid, times, tname="time", scale=1.0):
    """Поле var(t, y, x) -> массив (len(times), lat, lon) с ближайшим по времени шагом."""
    lat2, lon2 = _latlon(ds)
    src_t = pd.to_datetime(ds[tname].values)
    out = np.empty((len(times),) + grid.shape)
    cache = {}
    for k, t in enumerate(times):
        idx = int(np.argmin(np.abs(src_t - t)))
        if idx not in cache:
            vals = ds[var].isel({tname: idx}).values.squeeze() * scale
            cache[idx] = regrid(lat2, lon2, vals, grid)
        out[k] = cache[idx]
    return out


# ----------------------------------------------------------------------------- ERA5 (CDS)

def fetch_era5(grid: Grid, start: str, end: str, outdir="data", step_h=3):
    """Скачивает ERA5: ветер 10 м и волны (отдельными запросами — у них разные сетки)."""
    import cdsapi

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    days = pd.date_range(s.normalize(), e.normalize(), freq="D")
    lat_min, lat_max, lon_min, lon_max = _bbox(grid)
    base = {
        "product_type": ["reanalysis"],
        "year": sorted({f"{d.year}" for d in days}),
        "month": sorted({f"{d.month:02d}" for d in days}),
        "day": sorted({f"{d.day:02d}" for d in days}),
        "time": [f"{h:02d}:00" for h in range(0, 24, step_h)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [lat_max, lon_min, lat_min, lon_max],  # N, W, S, E
    }
    c = cdsapi.Client()
    files = {}
    for tag, variables in (
        ("wind", ["10m_u_component_of_wind", "10m_v_component_of_wind"]),
        ("waves", ["significant_height_of_combined_wind_waves_and_swell", "mean_wave_direction",
                   "mean_wave_period"]),
    ):
        f = outdir / f"era5_{tag}_{s:%Y%m%d}_{e:%Y%m%d}.nc"
        if not f.exists():
            c.retrieve("reanalysis-era5-single-levels", dict(base, variable=variables), str(f))
        files[tag] = f
    return files


def read_era5(files: dict, grid: Grid, times):
    out = {}
    if "wind" in files:
        ds = xr.open_dataset(files["wind"])
        t = "valid_time" if "valid_time" in ds.coords else "time"
        out["u10"] = _to_grid_series(ds, "u10", grid, times, t)
        out["v10"] = _to_grid_series(ds, "v10", grid, times, t)
    if "waves" in files:
        ds = xr.open_dataset(files["waves"])
        t = "valid_time" if "valid_time" in ds.coords else "time"
        out["hs"] = _to_grid_series(ds, "swh", grid, times, t)
        # направление интерполируем через компоненты, чтобы не ломаться на 0/360
        rad = np.radians(ds["mwd"])
        ds = ds.assign(_ws=np.sin(rad), _wc=np.cos(rad))
        s_ = _to_grid_series(ds, "_ws", grid, times, t)
        c_ = _to_grid_series(ds, "_wc", grid, times, t)
        out["wave_dir_from"] = (np.degrees(np.arctan2(s_, c_)) + 360) % 360
    return out


# ----------------------------------------------------------------------------- Copernicus Marine (ARC-MFC)

ARC_PHY_PRODUCT = "ARCTIC_ANALYSISFORECAST_PHY_002_001"   # TOPAZ5: лёд, течения
ARC_WAV_PRODUCT = "ARCTIC_ANALYSIS_FORECAST_WAV_002_014"  # волновой прогноз (WAM)


def list_cmems_datasets(product_id: str):
    """Список dataset_id внутри продукта (идентификаторы меняются между версиями — не хардкодим)."""
    import copernicusmarine as cm

    cat = cm.describe(product_id=product_id)
    ids = []
    for p in getattr(cat, "products", []):
        for d in getattr(p, "datasets", []):
            ids.append(d.dataset_id)
    return ids


def open_cmems(dataset_id: str, grid: Grid, start: str, end: str):
    """Открывает набор ARCO (лениво) и вырезает район по 2D-широте/долготе (сетка полярно-стереографическая)."""
    import copernicusmarine as cm

    ds = cm.open_dataset(dataset_id=dataset_id, start_datetime=start, end_datetime=end)
    lat2, lon2 = _latlon(ds)
    if lat2.ndim == 1:  # регулярная сетка — режем по координатам
        lat_min, lat_max, lon_min, lon_max = _bbox(grid)
        latn = "latitude" if "latitude" in ds.coords else "lat"
        lonn = "longitude" if "longitude" in ds.coords else "lon"
        return ds.sel({latn: slice(lat_min, lat_max), lonn: slice(lon_min, lon_max)})
    lat_min, lat_max, lon_min, lon_max = _bbox(grid)
    lon360 = lon2 % 360
    m = (lat2 >= lat_min) & (lat2 <= lat_max) & (lon360 >= lon_min) & (lon360 <= lon_max)
    yy, xx = np.where(m)
    ydim, xdim = ds[list(ds.data_vars)[0]].dims[-2:]
    return ds.isel({ydim: slice(yy.min(), yy.max() + 1), xdim: slice(xx.min(), xx.max() + 1)})


def read_cmems_phy(dataset_id: str, grid: Grid, times, start, end):
    ds = open_cmems(dataset_id, grid, start, end)
    if "depth" in ds.dims:
        ds = ds.isel(depth=0)
    out = {}
    for key, std, names in (
        ("sic", ("sea_ice_area_fraction",), ("siconc", "fice")),
        ("sit", ("sea_ice_thickness",), ("sithick", "hice")),
        ("uc", ("eastward_sea_water_velocity",), ("uo", "vxo")),
        ("vc", ("northward_sea_water_velocity",), ("vo", "vyo")),
    ):
        try:
            v = _find_var(ds, std, names)
        except KeyError as exc:
            warnings.warn(str(exc))
            continue
        tname = "time" if "time" in ds.coords else list(ds[v].dims)[0]
        out[key] = _to_grid_series(ds.load() if ds.nbytes < 2e9 else ds, v, grid, times, tname)
    if "sic" in out and np.nanmax(out["sic"]) > 1.5:  # в процентах
        out["sic"] = out["sic"] / 100.0
    return out


def read_cmems_waves(dataset_id: str, grid: Grid, times, start, end):
    ds = open_cmems(dataset_id, grid, start, end)
    v_hs = _find_var(ds, ("sea_surface_wave_significant_height",), ("VHM0",))
    v_dir = _find_var(ds, ("sea_surface_wave_from_direction",), ("VMDR",))
    rad = np.radians(ds[v_dir])
    ds = ds.assign(_ws=np.sin(rad), _wc=np.cos(rad))
    hs = _to_grid_series(ds, v_hs, grid, times)
    s_ = _to_grid_series(ds, "_ws", grid, times)
    c_ = _to_grid_series(ds, "_wc", grid, times)
    return {"hs": hs, "wave_dir_from": (np.degrees(np.arctan2(s_, c_)) + 360) % 360}


# ----------------------------------------------------------------------------- OSI SAF (сплочённость)

OSISAF_URL = ("https://thredds.met.no/thredds/dodsC/osisaf/met.no/ice/conc/{d:%Y}/{d:%m}/"
              "ice_conc_nh_polstere-100_multi_{d:%Y%m%d}1200.nc")


def read_osisaf_sic(grid: Grid, times, url_template=OSISAF_URL):
    """Суточная сплочённость OSI-401 (~10 км). Шаблон URL проверьте в каталоге THREDDS met.no."""
    out = np.empty((len(times),) + grid.shape)
    cache = {}
    for k, t in enumerate(times):
        d = pd.Timestamp(t).normalize()
        if d not in cache:
            with xr.open_dataset(url_template.format(d=d)) as ds:
                v = ds["ice_conc"].isel(time=0).values / 100.0
                cache[d] = regrid(ds["lat"].values, ds["lon"].values, v, grid)
        out[k] = cache[d]
    return out


# ----------------------------------------------------------------------------- толщина: CS2SMOS / CCI (Sentinel-3, CryoSat-2)

def read_thickness_file(path, grid: Grid):
    """Толщина льда из файла CS2SMOS (AWI) или ESA Sea Ice CCI L3C (Sentinel-3A/B, CryoSat-2).

    Альтиметрическая толщина надёжна только в холодный сезон (≈ октябрь–апрель) и на сетке 25 км,
    поэтому у берега и в Чаунской губе её стоит заменять моделью (TOPAZ) или климатом молодого льда.
    """
    with xr.open_dataset(path) as ds:
        v = _find_var(ds, ("sea_ice_thickness",), ("analysis_sea_ice_thickness", "sea_ice_thickness"))
        arr = ds[v]
        if "time" in arr.dims:
            arr = arr.isel(time=0)
        lat2, lon2 = _latlon(ds)
        return regrid(lat2, lon2, arr.values, grid)


# ----------------------------------------------------------------------------- сборка окружения

def build_environment(grid: Grid, start: str, end: str, step_h=3, era5_files=None,
                      cmems_phy_id=None, cmems_wav_id=None, use_osisaf=False, thickness_file=None):
    """Собирает окружение из доступных источников с приоритетами:
       лёд:      OSI SAF (наблюдения) > TOPAZ;   толщина: CS2SMOS/CCI (наблюдения, зимой) > TOPAZ
       ветер:    ERA5;                           волны:   CMEMS ARC WAV > ERA5;   течения: TOPAZ
    """
    times = pd.date_range(start, end, freq=f"{step_h}h")
    fields = {}
    if era5_files:
        fields.update(read_era5(era5_files, grid, times))
    if cmems_phy_id:
        phy = read_cmems_phy(cmems_phy_id, grid, times, start, end)
        for k, v in phy.items():
            fields.setdefault(k, v)
    if cmems_wav_id:
        fields.update(read_cmems_waves(cmems_wav_id, grid, times, start, end))
    if use_osisaf:
        fields["sic"] = read_osisaf_sic(grid, times)
    if thickness_file:
        h = read_thickness_file(thickness_file, grid)
        if "sit" in fields:  # наблюдения там, где они есть; модель — у берега и в пробелах
            fields["sit"] = np.where(np.isfinite(h)[None], h[None], fields["sit"])
        else:
            fields["sit"] = np.repeat(h[None], len(times), axis=0)

    defaults = dict(sic=0.0, sit=0.0, u10=0.0, v10=0.0, hs=0.0, wave_dir_from=0.0, uc=0.0, vc=0.0)
    missing = [k for k in defaults if k not in fields]
    if missing:
        warnings.warn(f"Нет данных для {missing} — заполнено нулями. Результат будет неполным.")
    water = ~grid.land
    data = {}
    for k, d in defaults.items():
        arr = fields.get(k, np.full((len(times),) + grid.shape, d))
        arr = np.where(np.isfinite(arr), arr, d)
        if k in ("sic", "sit", "hs"):
            arr = np.clip(arr, 0, None)
        if k == "sic":
            arr = np.clip(arr, 0, 1)
        data[k] = (("time", "lat", "lon"), np.where(water[None], arr, np.nan if k in ("sic", "hs") else 0.0))
    return xr.Dataset(data, coords=dict(time=times, lat=grid.lat, lon=grid.lon),
                      attrs=dict(source="real data: " + ", ".join(sorted(fields)),
                                 created=dt.datetime.now(dt.timezone.utc).isoformat()))
