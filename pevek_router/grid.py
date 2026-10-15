"""Регулярная широтно-долготная сетка района и маска суши."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config

R_EARTH = 6371000.0


@dataclass
class Grid:
    lat: np.ndarray  # 1D, по возрастанию
    lon: np.ndarray  # 1D, по возрастанию
    land: np.ndarray  # 2D bool [lat, lon]

    @property
    def shape(self):
        return (self.lat.size, self.lon.size)

    def index_of(self, lat, lon):
        i = int(round((lat - self.lat[0]) / (self.lat[1] - self.lat[0])))
        j = int(round((lon - self.lon[0]) / (self.lon[1] - self.lon[0])))
        return np.clip(i, 0, self.lat.size - 1), np.clip(j, 0, self.lon.size - 1)

    def nearest_water(self, lat, lon):
        i0, j0 = self.index_of(lat, lon)
        if not self.land[i0, j0]:
            return i0, j0
        ii, jj = np.where(~self.land)
        d = (ii - i0) ** 2 + ((jj - j0) * 0.6) ** 2
        k = int(np.argmin(d))
        return int(ii[k]), int(jj[k])


def make_grid(lat_min=config.LAT_MIN, lat_max=config.LAT_MAX, lon_min=config.LON_MIN,
              lon_max=config.LON_MAX, dlat=config.DLAT, dlon=config.DLON) -> Grid:
    from global_land_mask import globe

    lat = np.round(np.arange(lat_min, lat_max + 1e-9, dlat), 4)
    lon = np.round(np.arange(lon_min, lon_max + 1e-9, dlon), 4)
    LON, LAT = np.meshgrid(lon, lat)
    # ячейка считается сушей, если суша в центре и хотя бы в 2 из 4 подточек (сглаживаем шум маски)
    sub = []
    for di in (-0.25, 0.25):
        for dj in (-0.25, 0.25):
            sub.append(globe.is_land(LAT + di * dlat, np.where(LON + dj * dlon > 180, LON + dj * dlon - 360,
                                                               LON + dj * dlon)))
    centre = globe.is_land(LAT, LON)
    land = centre & (np.sum(sub, axis=0) >= 2)
    return Grid(lat=lat, lon=lon, land=land)


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    """Начальный курс (рад, от севера по часовой)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.arctan2(x, y)
