"""Скачивание реальных данных и сборка окружения для района Певека.

Пример:
    python -m pevek_router.fetch_data --start 2025-10-20 --end 2025-10-25 --out data/ \\
        --cmems-phy auto --cmems-wav auto --osisaf

Что нужно заранее:
    pip install cdsapi copernicusmarine xarray netCDF4 scipy global-land-mask
    ~/.cdsapirc  — ключ Copernicus Climate Data Store (ERA5), лицензию ERA5 нужно принять на сайте CDS
    copernicusmarine login  — учётка Copernicus Marine (бесплатная регистрация)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import loaders
from .grid import make_grid


def _pick(product, must_have):
    ids = loaders.list_cmems_datasets(product)
    for i in ids:
        if all(m in i for m in must_have):
            return i
    raise SystemExit(f"Не удалось выбрать dataset в {product}. Доступны: {ids}. Укажите id явно.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--step-h", type=int, default=3)
    ap.add_argument("--no-era5", action="store_true")
    ap.add_argument("--cmems-phy", help="dataset_id TOPAZ (лёд/течения) или 'auto'")
    ap.add_argument("--cmems-wav", help="dataset_id волнового прогноза или 'auto'")
    ap.add_argument("--osisaf", action="store_true", help="сплочённость из OSI SAF (THREDDS)")
    ap.add_argument("--thickness-file", help="NetCDF CS2SMOS или Sea Ice CCI (Sentinel-3/CryoSat-2)")
    a = ap.parse_args()

    grid = make_grid()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    era5 = None if a.no_era5 else loaders.fetch_era5(grid, a.start, a.end, out, step_h=1)
    phy = a.cmems_phy
    if phy == "auto":
        phy = _pick(loaders.ARC_PHY_PRODUCT, ["arc", "phy", "anfc"])
        print("TOPAZ dataset:", phy)
    wav = a.cmems_wav
    if wav == "auto":
        wav = _pick(loaders.ARC_WAV_PRODUCT, ["arc", "wav"])
        print("Wave dataset:", wav)
    env = loaders.build_environment(grid, a.start, a.end, a.step_h, era5_files=era5, cmems_phy_id=phy,
                                    cmems_wav_id=wav, use_osisaf=a.osisaf, thickness_file=a.thickness_file)
    env.to_netcdf(out / "env.nc")
    print("Сохранено:", out / "env.nc", "|", env.attrs["source"])


if __name__ == "__main__":
    main()
