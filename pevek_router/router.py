"""Оптимизация маршрута: время-зависимый поиск пути на сетке с выбором скорости на каждом шаге.

Стоимость шага ($) = цена_топлива * топливо(т) + стоимость_часа * время(ч).
  * «Быстрый»          — стоимость часа очень велика → минимум времени;
  * «Сбалансированный» — типичная ставка тайм-чартера;
  * «Экономичный»      — дешёвое время → оптимизатор снижает скорость и обходит лёд/волну.
На каждом шаге выбирается скорость по воде из набора (или максимально возможная),
с учётом ограничения мощности, ледовых ограничений класса и добровольного снижения на волне.
Путь по сетке затем «распрямляется» (string pulling) и пересчитывается непрерывно.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from . import config
from .grid import Grid, bearing, haversine
from .ship import KN, ShipParams, fuel_rate_t_per_h, max_speed, navigable, shaft_power

# 16 направлений: 8 соседей + «ходы коня» (сглаживают зигзаги сетки)
MOVES = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
         (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
MIN_SOG = 0.5  # м/с


@dataclass
class Strategy:
    key: str
    label: str
    hour_cost: float  # $/ч
    fuel_price: float = config.FUEL_PRICE_USD_T


@dataclass
class RouteResult:
    strategy: str
    label: str
    lat: list
    lon: list
    t_hours: list           # время прихода в каждую точку от отхода
    ts: list                # время (ч) в конце каждого 2-км шага — для профилей
    speed_kn: list          # скорость по воде на участке
    sog_kn: list
    sic: list
    hs: list
    wind: list
    fuel_t: float
    hours: float
    dist_nm: float
    cost_usd: float
    feasible: bool = True
    notes: list = field(default_factory=list)


def _env_at(env: xr.Dataset, k: int):
    d = env.isel(time=k)
    return {v: np.nan_to_num(d[v].values) for v in env.data_vars}


def _cell_kpis(ship: ShipParams, e, heading, options_ms, w_t, w_f):
    """Для всех ячеек при заданном курсе: лучшая скорость -> (сек/м, т/м, ст-ть/м, v), навигабельность."""
    wind = np.hypot(e["u10"], e["v10"])
    wind_from = np.arctan2(-e["u10"], -e["v10"])
    wind_rel = wind_from - heading
    wave_rel = np.radians(e["wave_dir_from"]) - heading
    cur_along = e["uc"] * np.sin(heading) + e["vc"] * np.cos(heading)
    env_args = (wind, wind_rel, e["hs"], wave_rel, e["sic"], e["sit"])
    vmax = max_speed(ship, *env_args, iters=22)
    nav = navigable(ship, e["sic"], e["sit"], e["hs"]) & (vmax > 1.0)
    best_cost = np.full(vmax.shape, np.inf)
    best = [np.zeros(vmax.shape) for _ in range(3)]
    for v_opt in options_ms:
        v = vmax if v_opt is None else np.full(vmax.shape, v_opt)
        ok = v <= vmax + 1e-6
        sog = np.maximum(v + cur_along, MIN_SOG)
        s_per_m = 1.0 / sog
        p = shaft_power(ship, v, *env_args)
        t_per_m = fuel_rate_t_per_h(ship, p) / 3600.0 * s_per_m
        cost = w_t * s_per_m / 3600.0 + w_f * t_per_m
        better = ok & (cost < best_cost)
        best_cost = np.where(better, cost, best_cost)
        for arr, val in zip(best, (s_per_m, t_per_m, v)):
            arr[...] = np.where(better, val, arr)
    nav &= np.isfinite(best_cost)
    return best[0], best[1], best_cost, best[2], nav


class Router:
    def __init__(self, grid: Grid, env: xr.Dataset, ship: ShipParams,
                 speed_options_kn=config.SPEED_OPTIONS_KN):
        self.grid = grid
        self.env = env
        self.ship = ship
        self.options = [None if s is None else s * KN for s in speed_options_kn]
        self.step_s = float((env.time[1] - env.time[0]) / np.timedelta64(1, "s"))
        self.nt = env.sizes["time"]
        ni, nj = grid.shape
        km_lat = haversine(70.5, 166, 70.5 + (grid.lat[1] - grid.lat[0]), 166) / 1000
        self.headings = []
        self.lengths = []  # [d] -> массив по строкам широты (м)
        for di, dj in MOVES:
            lat2 = grid.lat + di * (grid.lat[1] - grid.lat[0])
            L = haversine(grid.lat, 0.0, lat2, dj * (grid.lon[1] - grid.lon[0]))
            self.lengths.append(L)
            self.headings.append(float(bearing(70.5, 166.0, 70.5 + di * (grid.lat[1] - grid.lat[0]),
                                               166.0 + dj * (grid.lon[1] - grid.lon[0]))))
        self._km_lat = km_lat
        self._cache = {}
        # клетки, которые «задевает» отрезок каждого хода (2x2 окрестность точек вдоль отрезка)
        self.sweep = []
        for di, dj in MOVES:
            cells = set()
            for f in np.linspace(0, 1, 9):
                y, x = f * di, f * dj
                for oy in {int(np.floor(y + 1e-9)), int(np.ceil(y - 1e-9))}:
                    for ox in {int(np.floor(x + 1e-9)), int(np.ceil(x - 1e-9))}:
                        cells.add((oy, ox))
            self.sweep.append(sorted(cells))
        self._okmove = {}
        # numpy-копии полей для быстрых точечных обращений
        self.E = {v: np.nan_to_num(env[v].values) for v in env.data_vars}
        self.NAV = navigable(ship, self.E["sic"], self.E["sit"], self.E["hs"]) & ~grid.land[None]

    def _tables(self, strat: Strategy):
        """Таблицы [k][d] -> (сек/м, т/м, $/м, v, nav), считаются лениво и кэшируются."""
        key = strat.key
        if key not in self._cache:
            self._cache[key] = {}
        return self._cache[key]

    def _get(self, strat, k, d):
        tab = self._tables(strat)
        if (k, d) not in tab:
            e = _env_at(self.env, k)
            res = _cell_kpis(self.ship, e, self.headings[d], self.options, strat.hour_cost, strat.fuel_price)
            res = tuple(np.asarray(a, np.float32) if a.dtype != bool else a for a in res)
            tab[(k, d)] = res
        return tab[(k, d)]

    def ok_move(self, k, d):
        """Булев массив: ход d из клетки (i, j) на сроке k проходит только по допустимым клеткам
        (на сроке k и на следующем — с запасом на смену обстановки)."""
        key = (k, d)
        if key not in self._okmove:
            ni, nj = self.grid.shape
            k2 = min(k + 1, self.nt - 1)
            nav = self.NAV[k] & self.NAV[k2]
            ok = np.ones((ni, nj), bool)
            for oy, ox in self.sweep[d]:
                sh = np.zeros((ni, nj), bool)
                ys, ye = max(0, -oy), min(ni, ni - oy)
                xs, xe = max(0, -ox), min(nj, nj - ox)
                sh[ys:ye, xs:xe] = nav[ys + oy:ye + oy, xs + ox:xe + ox]
                ok &= sh
            self._okmove[key] = ok
        return self._okmove[key]

    def _static_nav(self):
        return ~self.grid.land

    # ------------------------------------------------------------------ поиск на сетке
    def grid_search(self, start_ij, goal_ij, strat: Strategy, depart_k=0, distance_only=False):
        ni, nj = self.grid.shape
        land = self.grid.land
        INF = np.inf
        best = np.full((ni, nj), INF)
        arr_t = np.full((ni, nj), INF)
        prev = -np.ones((ni, nj, 2), int)
        si, sj = start_ij
        gi, gj = goal_ij
        best[si, sj] = 0.0
        arr_t[si, sj] = depart_k * self.step_s
        heap = [(0.0, si, sj)]
        lat_c = np.radians(self.grid.lat)
        while heap:
            c, i, j = heapq.heappop(heap)
            if c > best[i, j]:
                continue
            if (i, j) == (gi, gj):
                break
            t = arr_t[i, j]
            k = min(int(t // self.step_s), self.nt - 1)
            for d, (di, dj) in enumerate(MOVES):
                ii, jj = i + di, j + dj
                if not (0 <= ii < ni and 0 <= jj < nj) or land[ii, jj]:
                    continue
                if not self.ok_move(k, d)[i, j]:  # суша или лёд сверх ограничений на пути хода
                    continue
                L = self.lengths[d][i]
                if distance_only:
                    s_m, _, c_m, _, nav = self._get(strat, k, d)
                    if not (nav[i, j] and nav[ii, jj]):
                        continue
                    nc = c + L
                    nt = t + L * 0.5 * (s_m[i, j] + s_m[ii, jj])
                else:
                    s_m, f_m, c_m, _, nav = self._get(strat, k, d)
                    if not (nav[i, j] and nav[ii, jj]):
                        continue
                    nc = c + L * 0.5 * (c_m[i, j] + c_m[ii, jj])
                    nt = t + L * 0.5 * (s_m[i, j] + s_m[ii, jj])
                if nc < best[ii, jj]:
                    best[ii, jj] = nc
                    arr_t[ii, jj] = nt
                    prev[ii, jj] = (i, j)
                    heapq.heappush(heap, (nc, ii, jj))
        if not np.isfinite(best[gi, gj]):
            return None
        path = [(gi, gj)]
        while path[-1] != (si, sj):
            path.append(tuple(prev[path[-1]]))
        return path[::-1]

    # ------------------------------------------------------------------ непрерывная оценка ломаной
    def evaluate(self, lats, lons, strat: Strategy, depart_k=0, step_m=2000.0, fixed_speed=None):
        """Интегрирует время/топливо вдоль ломаной с шагом ~2 км, выбирая лучшую скорость на каждом шаге."""
        g = self.grid
        t = depart_k * self.step_s
        fuel = cost = dist = 0.0
        out = dict(lat=[lats[0]], lon=[lons[0]], t=[0.0], ts=[], v=[], sog=[], sic=[], hs=[], wind=[])
        feasible = True
        for a in range(len(lats) - 1):
            seg = float(haversine(lats[a], lons[a], lats[a + 1], lons[a + 1]))
            n = max(1, int(np.ceil(seg / step_m)))
            hd = float(bearing(lats[a], lons[a], lats[a + 1], lons[a + 1]))
            for s in range(n):
                f0 = (s + 0.5) / n
                la = lats[a] + f0 * (lats[a + 1] - lats[a])
                lo = lons[a] + f0 * (lons[a + 1] - lons[a])
                i, j = g.index_of(la, lo)
                k = min(int(t // self.step_s), self.nt - 1)
                e = {v: np.atleast_1d(self.E[v][k, i, j]) for v in self.E}
                opts = self.options if fixed_speed is None else [fixed_speed]
                s_m, f_m, c_m, v, nav = _cell_kpis(self.ship, e, hd, opts, strat.hour_cost, strat.fuel_price)
                if g.land[i, j] or not nav[0] or not np.isfinite(c_m[0]):
                    feasible = False
                    s_m, f_m = np.array([1.0 / (4 * KN)]), np.array([0.0])
                    v = np.array([4 * KN])
                L = seg / n
                t += L * s_m[0]
                fuel += L * f_m[0]
                dist += L
                out["ts"].append((t - depart_k * self.step_s) / 3600.0)
                out["v"].append(float(v[0] / KN))
                out["sog"].append(float(1.0 / s_m[0] / KN))
                out["sic"].append(float(e["sic"][0]))
                out["hs"].append(float(e["hs"][0]))
                out["wind"].append(float(np.hypot(e["u10"][0], e["v10"][0])))
            out["lat"].append(lats[a + 1])
            out["lon"].append(lons[a + 1])
            out["t"].append((t - depart_k * self.step_s) / 3600.0)
        hours = (t - depart_k * self.step_s) / 3600.0
        cost = fuel * strat.fuel_price + hours * strat.hour_cost
        return dict(fuel=fuel, hours=hours, dist_nm=dist / 1852.0, cost=cost, feasible=feasible, **out)

    def _line_ok(self, la1, lo1, la2, lo2, t_s, strat):
        """Прямой отрезок проходим (вода и допустимые условия на всём протяжении)?"""
        seg = float(haversine(la1, lo1, la2, lo2))
        n = max(2, int(seg / 1500.0))
        k = min(int(t_s // self.step_s), self.nt - 1)
        k2 = min(k + int(np.ceil(seg / (6 * KN) / self.step_s)), self.nt - 1)  # пока идём по отрезку
        f = np.linspace(0, 1, n)
        la, lo = la1 + f * (la2 - la1), lo1 + f * (lo2 - lo1)
        dlat, dlon = self.grid.lat[1] - self.grid.lat[0], self.grid.lon[1] - self.grid.lon[0]
        ii = np.clip(np.round((la - self.grid.lat[0]) / dlat).astype(int), 0, self.grid.shape[0] - 1)
        jj = np.clip(np.round((lo - self.grid.lon[0]) / dlon).astype(int), 0, self.grid.shape[1] - 1)
        return bool(self.NAV[k, ii, jj].all() and self.NAV[k2, ii, jj].all())

    def smooth(self, path, strat: Strategy, depart_k=0, max_skip=40):
        """String pulling: убираем лишние вершины, если прямой отрезок проходим и не дороже."""
        lat = [float(self.grid.lat[i]) for i, _ in path]
        lon = [float(self.grid.lon[j]) for _, j in path]
        keep_lat, keep_lon = [lat[0]], [lon[0]]
        a = 0
        t_s = depart_k * self.step_s
        while a < len(lat) - 1:
            best_b = a + 1
            hi = min(len(lat) - 1, a + max_skip)
            for b in range(hi, a + 1, -1):
                if not self._line_ok(lat[a], lon[a], lat[b], lon[b], t_s, strat):
                    continue
                direct = self.evaluate([lat[a], lat[b]], [lon[a], lon[b]], strat, step_m=4000.0,
                                       depart_k=min(int(t_s // self.step_s), self.nt - 1))
                via = self.evaluate(lat[a:b + 1], lon[a:b + 1], strat, step_m=4000.0,
                                    depart_k=min(int(t_s // self.step_s), self.nt - 1))
                if direct["feasible"] and direct["cost"] <= via["cost"] * 1.002:
                    best_b = b
                    t_s += direct["hours"] * 3600
                    break
            else:
                seg = self.evaluate([lat[a], lat[a + 1]], [lon[a], lon[a + 1]], strat, step_m=4000.0,
                                    depart_k=min(int(t_s // self.step_s), self.nt - 1))
                t_s += seg["hours"] * 3600
            keep_lat.append(lat[best_b])
            keep_lon.append(lon[best_b])
            a = best_b
        return keep_lat, keep_lon

    # ------------------------------------------------------------------ полный цикл
    def solve(self, p_from, p_to, strat: Strategy, depart_k=0, distance_only=False, fixed_speed=None):
        s_ij = self.grid.nearest_water(p_from[0], p_from[1])
        g_ij = self.grid.nearest_water(p_to[0], p_to[1])
        path = self.grid_search(s_ij, g_ij, strat, depart_k, distance_only=distance_only)
        if path is None:
            return None
        lat, lon = self.smooth(path, strat, depart_k)
        r = self.evaluate(lat, lon, strat, depart_k, fixed_speed=fixed_speed)
        notes = []
        if not r["feasible"]:  # распрямление задело непроходимый участок — берём путь по сетке
            lat = [float(self.grid.lat[i]) for i, _ in path]
            lon = [float(self.grid.lon[j]) for _, j in path]
            r = self.evaluate(lat, lon, strat, depart_k, fixed_speed=fixed_speed)
            notes.append("использован несглаженный путь")
        if not r["feasible"]:
            notes.append("на части пути условия превышают ограничения класса (смена обстановки в пути)")
        return RouteResult(strategy=strat.key, label=strat.label, lat=r["lat"], lon=r["lon"],
                           t_hours=r["t"], ts=r["ts"], speed_kn=r["v"], sog_kn=r["sog"], sic=r["sic"], hs=r["hs"],
                           wind=r["wind"], fuel_t=r["fuel"], hours=r["hours"], dist_nm=r["dist_nm"],
                           cost_usd=r["cost"], feasible=r["feasible"], notes=notes)


def strategies():
    return [Strategy(k, lbl, hc) for k, (lbl, hc) in config.STRATEGIES.items()]
