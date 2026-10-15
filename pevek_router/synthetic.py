"""Синтетический, но физически правдоподобный сценарий для проверки конвейера.

Сценарий «20–25 октября»: начало ледообразования в Восточно-Сибирском море.
  * Северная кромка дрейфующего льда (~72.3–72.8° с. ш.), медленно смещается к югу.
  * Остаточный Айонский ледяной массив — язык однолетнего льда к северо-западу от Чаунской губы.
  * Полоса молодого льда (нилас, серый лёд) вдоль побережья, нарастает со временем; в Чаунской губе — быстрее.
  * Циклон проходит с запада на восток на 2–4 сутки: ветер до ~17–18 м/с с учётом фона, разгон волны в открытой воде.
  * Восточное прибрежное течение ~0.15 м/с + ветровой дрейф.

Это НЕ реальные данные: сценарий нужен только для отладки маршрутизации.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from . import config
from .grid import Grid


def _local_km(grid: Grid):
    LON, LAT = np.meshgrid(grid.lon, grid.lat)
    lat0 = 70.5
    x = (LON - 166.0) * 111.32 * np.cos(np.radians(lat0))
    y = (LAT - lat0) * 110.57
    return LAT, LON, x, y


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_scenario(grid: Grid, start="2026-10-20T00:00", hours=120, step_h=config.TIME_STEP_H,
                  seed=7) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    LAT, LON, X, Y = _local_km(grid)
    water = ~grid.land
    times = pd.date_range(start, periods=hours // step_h + 1, freq=f"{step_h}h")
    nt = times.size
    km_lat = 110.57 * (grid.lat[1] - grid.lat[0])
    km_lon = 111.32 * np.cos(np.radians(70.5)) * (grid.lon[1] - grid.lon[0])
    dist_coast = ndimage.distance_transform_edt(water, sampling=(km_lat, km_lon))  # км

    # Мелкомасштабная «текстура» льда (неоднородность сплочённости)
    noise = ndimage.gaussian_filter(rng.standard_normal(grid.shape), sigma=3)
    noise = noise / noise.std()

    # Чаунская губа (грубо): южнее 70.05 и 168.8–171.2 в. д.
    chaun = (LAT < 70.05) & (LON > 168.8) & (LON < 171.2)

    sic = np.zeros((nt,) + grid.shape)
    sit = np.zeros_like(sic)
    u10 = np.zeros_like(sic)
    v10 = np.zeros_like(sic)
    uc = np.zeros_like(sic)
    vc = np.zeros_like(sic)

    # --- ветер: фон + циклон
    track_t = np.array([0, 24, 60, 96, 120])
    track_x = np.array([-900, -700, -150, 450, 650])  # км
    track_y = np.array([150, 130, 200, 260, 300])
    drift_x = drift_y = 0.0
    for k, t in enumerate((times - times[0]).total_seconds() / 3600):
        cx, cy = np.interp(t, track_t, track_x), np.interp(t, track_t, track_y)
        strength = np.interp(t, [0, 24, 48, 84, 110, 120], [0.3, 0.6, 1.0, 1.0, 0.6, 0.4])
        dx, dy = X - cx, Y - cy
        r = np.hypot(dx, dy) + 1e-6
        R = 380.0
        prof = (r / R) * np.exp(1 - r / R)  # максимум на радиусе R
        vmax = 15.0 * strength
        # циркуляция против часовой стрелки + приток 20° к центру
        alpha = np.radians(20)
        tx, ty = -dy / r, dx / r
        rx, ry = -dx / r, -dy / r
        ux = vmax * prof * (np.cos(alpha) * tx + np.sin(alpha) * rx)
        uy = vmax * prof * (np.cos(alpha) * ty + np.sin(alpha) * ry)
        bg_u, bg_v = -4.0, -2.5  # фоновый северо-восточный ветер ~5 м/с
        u10[k] = bg_u + ux
        v10[k] = bg_v + uy

        # --- дрейф льда ~2 % ветра (в точке массива), накапливаем
        i0, j0 = grid.index_of(71.2, 165.5)
        drift_x += 0.02 * u10[k, i0, j0] * step_h * 3.6  # км
        drift_y += 0.02 * v10[k, i0, j0] * step_h * 3.6

        day = t / 24.0
        # --- северная кромка дрейфующего льда
        edge = 72.55 + 0.25 * np.sin((LON - 152) / 9.0) - 0.06 * day
        pack = 0.95 * _sigmoid((LAT - edge) / 0.12)
        # --- Айонский массив: язык льда, смещаемый дрейфом
        ax = X - (-10 + drift_x)
        ay = Y - (95 + drift_y)
        tongue_core = np.exp(-(ax / 95.0) ** 2) * _sigmoid((ay + 75) / 18.0)
        ayon = np.clip(0.88 * tongue_core + 0.08 * noise * tongue_core, 0, 0.95)
        # --- молодой лёд у побережья (полоса растёт со временем)
        band_w = 12 + 6 * day
        coastal = np.clip((0.15 + 0.09 * day) * np.exp(-dist_coast / band_w), 0, 0.7)
        coastal = np.where(chaun, np.clip(0.25 + 0.13 * day, 0, 0.85), coastal)
        c = np.maximum.reduce([pack, ayon, coastal])
        c = np.clip(c + 0.04 * noise * (c > 0.15), 0, 1)
        c[c < 0.05] = 0.0
        sic[k] = np.where(water, c, np.nan)

        h_pack = 1.0 + 0.1 * noise
        h_ayon = 0.65 + 0.08 * noise
        h_young = 0.06 + 0.045 * day
        h = np.where(pack >= np.maximum(ayon, coastal), h_pack,
                     np.where(ayon >= coastal, h_ayon, h_young))
        sit[k] = np.where(water & (c > 0), np.clip(h, 0.02, 2.5), 0.0)

        # --- течения: восточное прибрежное + ветровой дрейф 1.5 %
        coast_cur = 0.15 * np.exp(-dist_coast / 60.0)
        uc[k] = coast_cur + 0.015 * u10[k]
        vc[k] = -0.03 * np.exp(-dist_coast / 60.0) + 0.015 * v10[k]

    # --- волнение: разгон по ветру до суши или льда > 50 %, затухание во льду
    hs = np.zeros_like(sic)
    wdir_from = np.zeros_like(sic)
    for k in range(nt):
        U = np.hypot(u10[k], v10[k])
        hs[k] = _fetch_limited_hs(grid, u10[k], v10[k], np.nan_to_num(sic[k], nan=1.0), km_lat, km_lon)
        hs[k] *= (1 - np.nan_to_num(sic[k])) ** 1.5
        wdir_from[k] = (np.degrees(np.arctan2(-u10[k], -v10[k])) + 360) % 360
        hs[k] = np.where(water, hs[k], np.nan)
        _ = U

    ds = xr.Dataset(
        dict(
            sic=(("time", "lat", "lon"), sic),
            sit=(("time", "lat", "lon"), sit),
            u10=(("time", "lat", "lon"), u10),
            v10=(("time", "lat", "lon"), v10),
            hs=(("time", "lat", "lon"), hs),
            wave_dir_from=(("time", "lat", "lon"), wdir_from),
            uc=(("time", "lat", "lon"), uc),
            vc=(("time", "lat", "lon"), vc),
        ),
        coords=dict(time=times, lat=grid.lat, lon=grid.lon),
        attrs=dict(source="SYNTHETIC demo scenario (not real data)"),
    )
    return ds


def _fetch_limited_hs(grid, u, v, sic, km_lat, km_lon, max_fetch_km=450.0, step_km=8.0):
    """Hs по JONSWAP с ограничением разгона: g*Hs/U^2 = 0.0016*sqrt(gF/U^2), не больше 0.21 U^2/g."""
    g = 9.81
    U = np.hypot(u, v) + 1e-6
    ex, ey = -u / U, -v / U  # навстречу ветру (откуда дует)
    ni, nj = grid.shape
    I, J = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    fetch = np.zeros(grid.shape)
    alive = np.ones(grid.shape, bool)
    n_steps = int(max_fetch_km / step_km)
    for s in range(1, n_steps + 1):
        ii = np.round(I + ey * s * step_km / km_lat).astype(int)
        jj = np.round(J + ex * s * step_km / km_lon).astype(int)
        inside = (ii >= 0) & (ii < ni) & (jj >= 0) & (jj < nj)
        iic, jjc = np.clip(ii, 0, ni - 1), np.clip(jj, 0, nj - 1)
        blocked = grid.land[iic, jjc] | (sic[iic, jjc] > 0.5)
        # за пределами района считаем открытую воду только на севере/востоке (дальше — море)
        blocked = np.where(inside, blocked, False)
        alive &= ~blocked
        fetch += np.where(alive, step_km, 0.0)
    F = fetch * 1000.0
    hs = 0.0016 * U * np.sqrt(np.maximum(F, 1.0) / g)
    return np.minimum(hs, 0.21 * U ** 2 / g)
