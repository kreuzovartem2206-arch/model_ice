"""Пилотный расчёт: Певек ↔ прилегающие участки СМП.

Запуск на синтетическом сценарии (для проверки):
    python -m pevek_router.run_pilot --synthetic --out out/

Запуск на реальных данных (нужны ~/.cdsapirc и `copernicusmarine login`):
    python -m pevek_router.fetch_data --start 2025-10-20 --end 2025-10-25 --out data/
    python -m pevek_router.run_pilot --env data/env.nc --out out/
"""
from __future__ import annotations

import argparse
import base64
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import xarray as xr

from . import config
from .grid import make_grid
from .router import Router, strategies
from .ship import L2_SHIP


def _b64_u8(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def export_fields(env: xr.Dataset, grid, every=2):
    """Поля для карты: сплочённость, Hs, ветер — квантованные в uint8, прорежённые по пространству."""
    sic = np.nan_to_num(env.sic.values[:, ::every, ::every], nan=0)
    hs = np.nan_to_num(env.hs.values[:, ::every, ::every], nan=0)
    U = np.hypot(env.u10.values, env.v10.values)[:, ::every, ::every]
    return dict(
        lat=grid.lat[::every].round(4).tolist(), lon=grid.lon[::every].round(4).tolist(),
        times=[str(t)[:16] for t in env.time.values],
        sic=_b64_u8(np.clip(sic * 250, 0, 250)),        # шаг 0.004
        hs=_b64_u8(np.clip(hs * 25, 0, 250)),           # шаг 0.04 м, до 10 м
        wind=_b64_u8(np.clip(U * 8, 0, 250)),           # шаг 0.125 м/с, до 31 м/с
        u=np.round(env.u10.values[:, ::6, ::6], 1).tolist(),
        v=np.round(env.v10.values[:, ::6, ::6], 1).tolist(),
        uv_lat=grid.lat[::6].round(3).tolist(), uv_lon=grid.lon[::6].round(3).tolist(),
    )


def run(env: xr.Dataset, out: Path, departures_h=(0, 24, 48, 72), verbose=True):
    grid = make_grid()
    router = Router(grid, env, L2_SHIP)
    strats = strategies()
    step_h = config.TIME_STEP_H
    results = []
    for a, b in config.ROUTES:
        pa, pb = config.POINTS[a], config.POINTS[b]
        for dep in departures_h:
            k = int(dep // step_h)
            if k >= env.sizes["time"] - 1:
                continue
            item = dict(route=f"{a}-{b}", origin=pa[0], dest=pb[0], depart_h=dep,
                        depart_time=str(env.time.values[k])[:16], variants=[])
            si, sj = grid.nearest_water(*pa[1:])
            if not router.NAV[k, si, sj]:
                why = ("в пункте отхода лёд превышает ограничения судна для самостоятельного плавания "
                       "— нужна ледокольная проводка")
            else:
                why = ("по прогнозу проход к пункту назначения в пути перекрывается льдом сверх ограничений судна "
                       f"(> {L2_SHIP.ice_c_max * 10:.0f} баллов или > {L2_SHIP.ice_h_max:.1f} м) — "
                       "отход в этот срок не рекомендуется без ледокольной проводки")
            item["blocked_reason"] = None
            n_ok = 0
            for st in strats:
                r = router.solve(pa[1:], pb[1:], st, depart_k=k)
                n_ok += r is not None
                item["variants"].append(asdict(r) if r else dict(strategy=st.key, label=st.label, feasible=False,
                                                                  notes=[why]))
            if n_ok == 0:
                item["blocked_reason"] = why
            base = router.solve(pa[1:], pb[1:], strats[0], depart_k=k, distance_only=True)
            if base:
                bd = asdict(base)
                bd.update(strategy="shortest", label="Кратчайший проходимый, полный ход")
                item["variants"].append(bd)
            results.append(item)
            if verbose:
                print(item["route"], dep, [(v["label"], round(v.get("hours", 0), 1), round(v.get("fuel_t", 0), 1))
                                           for v in item["variants"]])
    for it in results:
        for v in it["variants"]:
            for key in ("lat", "lon", "t_hours", "ts", "speed_kn", "sog_kn", "sic", "hs", "wind"):
                if key in v:
                    v[key] = [round(float(x), 4 if key in ("lat", "lon") else 2) for x in v[key]]
            for key in ("fuel_t", "hours", "dist_nm", "cost_usd"):
                if key in v:
                    v[key] = round(float(v[key]), 2)
    land = np.packbits(grid.land.astype(np.uint8), axis=None)
    payload = dict(
        meta=dict(source=env.attrs.get("source", ""), ship=asdict(L2_SHIP), fuel_price=config.FUEL_PRICE_USD_T,
                  strategies={k: dict(label=l, hour_cost=round(h, 1)) for k, (l, h) in config.STRATEGIES.items()},
                  points={k: dict(name=n, lat=la, lon=lo) for k, (n, la, lo) in config.POINTS.items()}),
        grid=dict(lat=grid.lat.round(4).tolist(), lon=grid.lon.round(4).tolist(),
                  land=base64.b64encode(land.tobytes()).decode(), shape=list(grid.shape)),
        fields=export_fields(env, grid),
        results=results,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot_results.json").write_text(json.dumps(payload, ensure_ascii=False))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--env", help="NetCDF окружения (из fetch_data)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--departures", type=int, nargs="+", default=[0, 24, 48, 72],
                    help="сроки отхода, ч от начала окружения")
    a = ap.parse_args()
    if a.synthetic or not a.env:
        from .synthetic import make_scenario
        env = make_scenario(make_grid())
    else:
        env = xr.open_dataset(a.env)
    run(env, Path(a.out), departures_h=tuple(a.departures))


if __name__ == "__main__":
    main()
