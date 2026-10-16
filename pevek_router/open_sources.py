"""Открытые оперативные источники без регистрации (NOAA NOMADS + OSI SAF на THREDDS met.no).

Собирает окружение на ближайшие ~5 суток от последнего доступного срока:
    ветер 10 м, T2м        — NOAA GFS 0.25°           (nomads.ncep.noaa.gov, GRIB2, байтовые диапазоны по .idx)
    волны Hs и направление  — NOAA GFS-Wave global 0.25°
    течения, лёд (прогноз)  — NOAA RTOFS-Global (HYCOM + CICE, 1/12°), NetCDF
    сплочённость льда       — NCEP Sea Ice Analysis (наблюдение по пассивному СВЧ, 1/12°) на NOMADS;
                              запасной вариант — EUMETSAT OSI SAF OSI-401 (THREDDS met.no)

Сплочённость на срок прогноза = наблюдение OSI SAF + изменение по прогнозу RTOFS относительно его анализа
(«прогноз аномалии»): наблюдения задают положение льда, модель — его эволюцию.
Толщина — из RTOFS (CICE); где модель льда не видит, а спутник видит — принимается молодой лёд 0.10 м.

Запуск:
    python -m pevek_router.open_sources --out data_open
    python -m pevek_router.run_pilot --env data_open/env.nc --out out_real --departures 0 24 48 72
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .grid import Grid, make_grid
from .loaders import regrid

NOMADS = "https://nomads.ncep.noaa.gov/pub/data/nccf/com"
THREDDS = "https://thredds.met.no/thredds"
UA = {"User-Agent": "pevek-ice-router/0.1 (research pilot)"}
S = requests.Session()
S.headers.update(UA)
_PACE_S = 0.6  # NOMADS ограничивает частоту запросов (~120/мин с одного адреса)


def _get(url, retries=6, **kw):
    """GET с паузой между запросами и повторами при сбоях/ограничении частоты."""
    import time
    for k in range(retries):
        time.sleep(_PACE_S)
        try:
            r = S.get(url, timeout=kw.pop("timeout", 120), **kw)
            if r.status_code == 200 or r.status_code == 206:
                return r
            err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            err = str(e)
        time.sleep(min(60, 5 * 2 ** k))
    raise RuntimeError(f"{url}: {err}")


# ----------------------------------------------------------------------------- GRIB по .idx

def _idx_ranges(url, patterns):
    """Из .idx GRIB2 возвращает байтовые диапазоны записей, подходящих под regex-шаблоны."""
    for attempt in range(4):
        r = _get(url + ".idx")
        lines = [ln for ln in r.text.strip().splitlines() if ln]
        if lines and all(ln.count(":") >= 3 for ln in lines):
            break
    else:
        raise RuntimeError(f"Некорректный .idx: {url}")
    recs = []
    for k, ln in enumerate(lines):
        parts = ln.split(":")
        start = int(parts[1])
        end = int(lines[k + 1].split(":")[1]) - 1 if k + 1 < len(lines) else None
        recs.append((start, end, ":".join(parts[3:])))
    out = [(s, e) for s, e, d in recs if any(re.search(p, d) for p in patterns)]
    if len(out) < len(patterns):
        raise RuntimeError(f"В {url} не найдены все поля {patterns}")
    return out


def _grib_fields(url, patterns, workdir):
    """Скачивает только нужные записи GRIB2 и открывает их как xarray (через cfgrib)."""
    f = Path(workdir) / (Path(url).name + ".sub.grib2")
    if not f.exists():  # кэш: при повторном запуске уже скачанное не качаем
        rng = sorted(_idx_ranges(url, patterns))
        merged = [list(rng[0])]
        for s_, e_ in rng[1:]:  # соседние записи — одним запросом
            if merged[-1][1] is not None and s_ == merged[-1][1] + 1:
                merged[-1][1] = e_
            else:
                merged.append([s_, e_])
        blobs = [_get(url, headers={"Range": f"bytes={s_}-{'' if e_ is None else e_}"}).content
                 for s_, e_ in merged]
        tmp = f.with_suffix(".part")
        tmp.write_bytes(b"".join(blobs))
        tmp.rename(f)
    dss = __import__("cfgrib").open_datasets(str(f), backend_kwargs=dict(indexpath=""))
    return xr.merge([d.drop_vars([c for c in ("heightAboveGround", "surface", "step", "valid_time")
                                   if c in d.coords], errors="ignore") for d in dss], compat="override")


def latest_gfs_cycle(max_back_h=36):
    now = pd.Timestamp.utcnow().tz_localize(None).floor("6h")
    for k in range(0, max_back_h + 1, 6):
        c = now - pd.Timedelta(hours=k)
        url = f"{NOMADS}/gfs/prod/gfs.{c:%Y%m%d}/{c:%H}/wave/gridded/gfswave.t{c:%H}z.global.0p25.f120.grib2.idx"
        if S.head(url, timeout=30).status_code == 200:
            return c
    raise RuntimeError("Не найден полный цикл GFS за последние 36 ч")


def fetch_gfs(grid: Grid, cycle, hours, workdir):
    """Ветер 10 м, T2м (GFS) и волны (GFS-Wave) на сроки hours от цикла."""
    out = {k: [] for k in ("u10", "v10", "t2m", "hs", "wave_dir_from")}
    for h in hours:
        base = f"{NOMADS}/gfs/prod/gfs.{cycle:%Y%m%d}/{cycle:%H}"
        atm = _grib_fields(f"{base}/atmos/gfs.t{cycle:%H}z.pgrb2.0p25.f{h:03d}",
                           [r"^UGRD:10 m above ground", r"^VGRD:10 m above ground", r"^TMP:2 m above ground"],
                           workdir)
        wav = _grib_fields(f"{base}/wave/gridded/gfswave.t{cycle:%H}z.global.0p25.f{h:03d}.grib2",
                           [r"^HTSGW:surface", r"^DIRPW:surface"], workdir)
        a = atm.sel(latitude=slice(grid.lat.max() + 1, grid.lat.min() - 1),
                    longitude=slice(grid.lon.min() - 1, grid.lon.max() + 1))
        w = wav.sel(latitude=slice(grid.lat.max() + 1, grid.lat.min() - 1),
                    longitude=slice(grid.lon.min() - 1, grid.lon.max() + 1))
        la, lo = a.latitude.values, a.longitude.values
        out["u10"].append(regrid(la, lo, a["u10"].values, grid))
        out["v10"].append(regrid(la, lo, a["v10"].values, grid))
        out["t2m"].append(regrid(la, lo, a["t2m"].values - 273.15, grid))
        wl, wo = w.latitude.values, w.longitude.values
        hs = w["swh"].values
        out["hs"].append(regrid(wl, wo, np.where(np.isfinite(hs), hs, 0.0), grid))
        dname = next(v for v in w.data_vars if v != "swh")  # DIRPW (cfgrib: dirpw)
        d = np.radians(w[dname].values)
        s_ = regrid(wl, wo, np.nan_to_num(np.sin(d)), grid)
        c_ = regrid(wl, wo, np.nan_to_num(np.cos(d)), grid)
        out["wave_dir_from"].append((np.degrees(np.arctan2(s_, c_)) + 360) % 360)
        print(f"  GFS +{h:03d} ч ok")
    return {k: np.array(v) for k, v in out.items()}


# ----------------------------------------------------------------------------- RTOFS

def latest_rtofs_day(max_back_d=3):
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    for k in range(max_back_d + 1):
        d = today - pd.Timedelta(days=k)
        url = f"{NOMADS}/rtofs/prod/rtofs.{d:%Y%m%d}/rtofs_glo_2ds_f120_prog.nc"
        if S.head(url, timeout=30).status_code == 200:
            return d
    raise RuntimeError("Не найден полный прогноз RTOFS за последние 3 суток")


def _download(url, dest):
    dest = Path(dest)
    if dest.exists():
        return dest
    with S.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest



def _rtofs_box(ds, grid, pad=1.0):
    lat = ds["Latitude"].values
    lon = ds["Longitude"].values % 360
    m = ((lat >= grid.lat.min() - pad) & (lat <= grid.lat.max() + pad)
         & (lon >= grid.lon.min() - 2 * pad) & (lon <= grid.lon.max() + 2 * pad))
    yy, xx = np.where(m)
    ydim, xdim = ds["Latitude"].dims
    return ds.isel({ydim: slice(yy.min(), yy.max() + 1), xdim: slice(xx.min(), xx.max() + 1)})


def fetch_rtofs(grid: Grid, day, hours_from_day, workdir, keep_files=False):
    """Течения (prog) и лёд (ice) RTOFS на заданные часы от 00 UTC дня выпуска.
    Файлы глобальные (сотни МБ) — вырезаем район и удаляем исходник."""
    out = {k: [] for k in ("uc", "vc", "sic_m", "sit_m")}
    for h in hours_from_day:
        tag = f"n{-h:03d}" if h < 0 else f"f{h:03d}"
        cache = Path(workdir) / f"rtofs_{day:%Y%m%d}_{tag}_{grid.shape[0]}x{grid.shape[1]}.npz"
        if cache.exists():  # уже вырезанный и перенесённый на сетку район
            z = np.load(cache)
            for k in out:
                out[k].append(z[k])
            print(f"  RTOFS {tag} (кэш)")
            continue
        n0 = {k: len(v) for k, v in out.items()}
        for kind in ("prog", "ice"):
            url = f"{NOMADS}/rtofs/prod/rtofs.{day:%Y%m%d}/rtofs_glo_2ds_{tag}_{kind}.nc"
            f = Path(workdir) / Path(url).name
            if not f.exists():
                _download(url, f)
            with xr.open_dataset(f) as ds:
                b = _rtofs_box(ds, grid).isel(MT=0).load()
            if kind == "prog":
                b = b.isel(Layer=0) if "Layer" in b.dims else b
                la, lo = b["Latitude"].values, b["Longitude"].values
                out["uc"].append(regrid(la, lo, b["u_velocity"].values, grid))
                out["vc"].append(regrid(la, lo, b["v_velocity"].values, grid))
            else:
                la, lo = b["Latitude"].values, b["Longitude"].values
                c = b["ice_coverage"].values
                out["sic_m"].append(regrid(la, lo, np.where(np.isfinite(c), c, 0.0), grid))
                t = b["ice_thickness"].values
                out["sit_m"].append(regrid(la, lo, np.where(np.isfinite(t), t, 0.0), grid))
            if not keep_files:
                f.unlink(missing_ok=True)
        np.savez_compressed(cache, **{k: out[k][n0[k]] for k in out})
        print(f"  RTOFS {tag} ok")
    return {k: np.array(v) for k, v in out.items()}


# ----------------------------------------------------------------------------- NCEP Sea Ice Analysis

def fetch_ncep_ice(grid: Grid, workdir, max_back_d=4):
    """Суточный анализ сплочённости NCEP (AMSR2/SSMIS, сетка 5' ≈ 9 км по широте)."""
    import cfgrib  # noqa: F401
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    for k in range(0, max_back_d + 1):
        d = today - pd.Timedelta(days=k)
        url = f"{NOMADS}/seaice_analysis/prod/seaice_analysis.{d:%Y%m%d}/seaice.t00z.5min.grb.grib2"
        if S.head(url, timeout=30).status_code != 200:
            continue
        f = _download(url, Path(workdir) / f"seaice_{d:%Y%m%d}.grib2")
        ds = xr.open_dataset(f, engine="cfgrib", backend_kwargs=dict(indexpath=""))
        v = list(ds.data_vars)[0]
        a = ds[v].sel(latitude=slice(grid.lat.max() + 1, grid.lat.min() - 1),
                      longitude=slice(grid.lon.min() - 1, grid.lon.max() + 1)).load()
        ds.close()
        c = a.values
        c = np.where((c >= 0) & (c <= 1.0), c, np.nan)  # >1 — служебные коды (суша, побережье)
        return d, np.clip(np.nan_to_num(regrid(a.latitude.values, a.longitude.values, c, grid,
                                               method="nearest")), 0, 1)
    raise RuntimeError("NCEP Sea Ice Analysis недоступен")


# ----------------------------------------------------------------------------- OSI SAF

def fetch_osisaf(grid: Grid, max_back_d=4):
    """Последнее суточное поле сплочённости OSI-401 (OPeNDAP, только нужный фрагмент)."""
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    last_err = None
    for k in range(1, max_back_d + 1):
        d = today - pd.Timedelta(days=k)
        url = (f"{THREDDS}/dodsC/osisaf/met.no/ice/conc/{d:%Y}/{d:%m}/"
               f"ice_conc_nh_polstere-100_multi_{d:%Y%m%d}1200.nc")
        try:
            with xr.open_dataset(url) as ds:
                lat, lon = ds["lat"].values, ds["lon"].values % 360
                m = ((lat >= grid.lat.min() - 1) & (lat <= grid.lat.max() + 1)
                     & (lon >= grid.lon.min() - 3) & (lon <= grid.lon.max() + 3))
                yy, xx = np.where(m)
                sub = ds.isel(yc=slice(yy.min(), yy.max() + 1), xc=slice(xx.min(), xx.max() + 1))
                c = sub["ice_conc"].isel(time=0).values / 100.0
                st = sub["status_flag"].isel(time=0).values if "status_flag" in sub else None
                if st is not None:
                    c = np.where(st == 1, np.nan, c)  # суша
                field = regrid(sub["lat"].values, sub["lon"].values, c, grid)
                return d, np.clip(np.nan_to_num(field), 0, 1)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"OSI SAF недоступен: {last_err}")


# ----------------------------------------------------------------------------- сборка

def build(out_dir="data_open", horizon_h=120, step_h=3, keep_cache=False):
    grid = make_grid()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "cache"
    work.mkdir(exist_ok=True)

    cyc = latest_gfs_cycle()
    hours = list(range(0, horizon_h + 1, step_h))
    times = pd.DatetimeIndex([cyc + pd.Timedelta(hours=h) for h in hours])
    print("GFS цикл:", cyc)
    gfs = fetch_gfs(grid, cyc, hours, work)

    rday = latest_rtofs_day()
    print("RTOFS выпуск:", rday.date())
    # RTOFS 2ds выдаётся ежечасно до f192; лёд и течения меняются медленно — берём каждые 12 ч и интерполируем
    rs = 12
    r_hours = sorted({int(rs * np.floor(((t - rday) / pd.Timedelta(hours=1)) / rs)) for t in times}
                     | {int(rs * np.ceil(((t - rday) / pd.Timedelta(hours=1)) / rs)) for t in times})
    r_hours = sorted({0} | {h for h in r_hours if h >= 0})  # 0 — анализ, опорный для наблюдений льда
    rt = fetch_rtofs(grid, rday, r_hours, work)
    rt_times = pd.DatetimeIndex([rday + pd.Timedelta(hours=h) for h in r_hours])

    def interp_t(arr):
        x = np.asarray((rt_times - rt_times[0]) / pd.Timedelta(hours=1), float)
        xi = np.clip(np.asarray((times - rt_times[0]) / pd.Timedelta(hours=1), float), x.min(), x.max())
        k = np.clip(np.searchsorted(x, xi) - 1, 0, len(x) - 2) if len(x) > 1 else np.zeros(len(xi), int)
        if len(x) == 1:
            return np.repeat(arr[:1], len(times), axis=0)
        f = ((xi - x[k]) / (x[k + 1] - x[k]))[:, None, None]
        return arr[k] * (1 - f) + arr[k + 1] * f

    uc, vc = interp_t(rt["uc"]), interp_t(rt["vc"])
    sic_m, sit_m = interp_t(rt["sic_m"]), interp_t(rt["sit_m"])

    obs = None
    for name, fn in (("NCEP Sea Ice Analysis", lambda: fetch_ncep_ice(grid, work)),
                     ("OSI SAF", lambda: fetch_osisaf(grid))):
        try:
            oday, sic_obs = fn()
            obs = name
            break
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"{name} недоступен: {e}")
    if obs:
        print(obs, "дата:", oday.date())
        # модельная сплочённость RTOFS на дату наблюдения — опорная для «прогноза аномалии»
        r0 = rt["sic_m"][0] if rt_times[0] >= oday else rt["sic_m"][int(np.argmin(np.abs(rt_times - oday)))]
        sic = np.clip(sic_obs[None] + (sic_m - r0[None]), 0, 1)
        sic_src = f"{obs} {oday:%d.%m.%Y} + эволюция по RTOFS"
    else:
        sic, sic_src = sic_m, "RTOFS"
    sit = np.where(sit_m > 0.02, sit_m, np.where(sic > 0.05, 0.10, 0.0))

    water = ~grid.land
    def W(a, fill=0.0):
        return (("time", "lat", "lon"), np.where(water[None], np.nan_to_num(a, nan=fill), np.nan))
    env = xr.Dataset(
        dict(sic=W(sic), sit=W(sit), u10=W(gfs["u10"]), v10=W(gfs["v10"]), t2m=W(gfs["t2m"]),
             hs=W(np.clip(gfs["hs"] * (1 - np.clip(sic, 0, 1)) ** 1.5, 0, None)),
             wave_dir_from=W(gfs["wave_dir_from"]), uc=W(uc), vc=W(vc)),
        coords=dict(time=times, lat=grid.lat, lon=grid.lon),
        attrs=dict(source=(f"REAL: ветер GFS и волны GFS-Wave (выпуск {cyc:%d.%m.%Y %H} UTC); "
                           f"течения и толщина льда RTOFS (выпуск {rday:%d.%m.%Y}); сплочённость: {sic_src}"),
                   created=dt.datetime.now(dt.timezone.utc).isoformat()),
    )
    for v in ("u10", "v10", "uc", "vc", "wave_dir_from", "sit", "t2m"):
        env[v] = env[v].fillna(0.0)
    env.to_netcdf(out / "env.nc")
    if not keep_cache:
        for f in work.glob("*"):
            f.unlink()
        work.rmdir()
    print("Сохранено:", out / "env.nc")
    return env


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_open")
    ap.add_argument("--hours", type=int, default=120)
    ap.add_argument("--step", type=int, default=3)
    a = ap.parse_args()
    build(a.out, a.hours, a.step)
