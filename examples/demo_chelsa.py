"""Real-data CHELSA V2 precipitation downscaling over Rwanda.

The demo compares bilinear interpolation, the required DEM+wind CHELSA path,
and the full paper-defined path with PBL, coarse orography, and the official
CHELSA exposure correction. It writes a PNG comparison and a NetCDF bundle.

Run from the repository root:

    uv run python examples/demo_chelsa.py

First use needs Rosetta + CDS credentials and downloads Copernicus GLO-90 DEM
tiles, monthly ERA5 fields, and a byte-range subset of CHELSA ``expocor``.
All compact derived inputs are cached under ``examples/output/demo_cache``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import numpy as np
import requests
import xarray as xr

import deepscale as ds


REGION = [-3.0, -1.0, 28.8, 31.0]  # Rwanda: south, north, west, east
ERA5_REGION = [-4.0, 0.0, 27.5, 32.5]
YEARS = (1993, 2016)
MAM = [3, 4, 5]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CACHE_DIR = OUTPUT_DIR / "demo_cache"
PLOT_PATH = OUTPUT_DIR / "demo_chelsa_rwanda.png"
DATA_PATH = OUTPUT_DIR / "demo_chelsa_rwanda.nc"

EXPOSURE_URL = (
    "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/input/static/"
    "expocor.sdat"
)
EXPOSURE_NX = 43_200
EXPOSURE_NY = 20_880
EXPOSURE_XMIN = -179.9959722222
EXPOSURE_YMIN = -89.9959722222
EXPOSURE_CELL = 0.0083333333
_LOCAL = threading.local()


def load_obs() -> xr.DataArray:
    import rosetta

    raw = rosetta.fetch(
        "obs/chirps-v2-monthly",
        "precip",
        hindcast=YEARS,
        region=REGION,
        months=MAM,
        seasonal="mean",
    )["precip"].where(lambda value: value >= 0)
    return raw.sel(year=slice(*YEARS)).sortby("lat").sortby("lon").load()


def load_gcm() -> xr.DataArray:
    import rosetta

    raw = rosetta.fetch(
        "c3s/ecmwf-monthly",
        "precip",
        init="2025-02",
        target="MAM",
        hindcast=YEARS,
        region=ERA5_REGION,
    )["precip"]
    for dim in ("lead_time", "forecastMonth"):
        if dim in raw.dims:
            raw = raw.mean(dim)
    if "number" in raw.dims:
        raw = raw.rename({"number": "member"})
    for time_dim in ("init_time", "time", "forecast_reference_time"):
        if time_dim in raw.dims:
            raw = raw.assign_coords(year=(time_dim, raw[time_dim].dt.year.values))
            raw = raw.swap_dims({time_dim: "year"}).drop_vars(time_dim)
            break
    return raw.sortby("lat").sortby("lon").load()


def load_era5(*, refresh: bool = False) -> xr.Dataset:
    """Monthly MAM wind, PBL height, and geopotential for the training years."""
    path = CACHE_DIR / "demo_chelsa_era5_mam_1993_2016.nc"
    if refresh and path.exists():
        path.unlink()
    if not path.exists():
        import cdsapi

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        south, north, west, east = ERA5_REGION
        request = {
            "product_type": "monthly_averaged_reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "boundary_layer_height",
                "geopotential",
            ],
            "year": [str(year) for year in range(YEARS[0], YEARS[1] + 1)],
            "month": [f"{month:02d}" for month in MAM],
            "time": "00:00",
            "area": [north, west, south, east],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        print(f"fetching ERA5 CHELSA predictors -> {path}")
        cdsapi.Client().retrieve(
            "reanalysis-era5-single-levels-monthly-means", request, str(path)
        )
    data = xr.open_dataset(path)
    rename = {
        old: new
        for old, new in {
            "valid_time": "time",
            "latitude": "lat",
            "longitude": "lon",
        }.items()
        if old in data.dims or old in data.coords
    }
    return data.rename(rename).sortby("lat").sortby("lon").load()


def _exposure_session() -> requests.Session:
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
    return _LOCAL.session


def _fetch_exposure_row(row: int, x0: int, x1: int) -> np.ndarray:
    start = (row * EXPOSURE_NX + x0) * 4
    end = (row * EXPOSURE_NX + x1 + 1) * 4 - 1
    response = _exposure_session().get(
        EXPOSURE_URL, headers={"Range": f"bytes={start}-{end}"}, timeout=60
    )
    response.raise_for_status()
    expected = (x1 - x0 + 1) * 4
    if response.status_code != 206 or len(response.content) != expected:
        raise RuntimeError(
            f"CHELSA exposure range request returned {response.status_code} / "
            f"{len(response.content)} bytes; expected 206 / {expected}"
        )
    return np.frombuffer(response.content, dtype="<f4").copy()


def load_exposure(*, refresh: bool = False) -> xr.DataArray:
    """Fetch only Rwanda's rows/columns from the official 3.6 GB SAGA grid."""
    path = CACHE_DIR / "demo_chelsa_expocor_rwanda.nc"
    if refresh and path.exists():
        path.unlink()
    if path.exists():
        return xr.open_dataarray(path).load()

    south, north, west, east = REGION
    x0 = max(0, int(np.floor((west - EXPOSURE_XMIN) / EXPOSURE_CELL)))
    x1 = min(EXPOSURE_NX - 1, int(np.ceil((east - EXPOSURE_XMIN) / EXPOSURE_CELL)))
    y0 = max(0, int(np.floor((south - EXPOSURE_YMIN) / EXPOSURE_CELL)))
    y1 = min(EXPOSURE_NY - 1, int(np.ceil((north - EXPOSURE_YMIN) / EXPOSURE_CELL)))
    rows = list(range(y0, y1 + 1))
    print("fetching official CHELSA exposure subset by HTTP byte range...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        values = np.stack(
            list(pool.map(lambda row: _fetch_exposure_row(row, x0, x1), rows))
        )
    values[values == -9999.0] = np.nan
    exposure = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={
            "lat": EXPOSURE_YMIN + np.asarray(rows) * EXPOSURE_CELL,
            "lon": EXPOSURE_XMIN + np.arange(x0, x1 + 1) * EXPOSURE_CELL,
        },
        name="exposure_correction",
        attrs={"source": EXPOSURE_URL, "product": "CHELSA V2.1 expocor.sdat"},
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    exposure.to_netcdf(path)
    return exposure


def parent_conservation_error(fine: xr.DataArray, coarse: xr.DataArray) -> float:
    """Maximum error between fine parent means and their coarse values."""
    iy = np.abs(fine.lat.values[:, None] - coarse.lat.values[None, :]).argmin(1)
    ix = np.abs(fine.lon.values[:, None] - coarse.lon.values[None, :]).argmin(1)
    error = 0.0
    for y in range(coarse.sizes["lat"]):
        for x in range(coarse.sizes["lon"]):
            mask = (iy[:, None] == y) & (ix[None, :] == x)
            cells = fine.values[..., mask]
            if cells.size and np.isfinite(cells).any():
                expected = coarse.values[..., y, x]
                error = max(error, float(np.nanmax(np.abs(np.nanmean(cells, axis=-1) - expected))))
    return error


def render_plot(fields: dict[str, xr.DataArray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    precip_names = {
        "CHIRPS truth",
        "Bilinear forecast",
        "CHELSA DEM + wind",
        "CHELSA full paper",
    }
    precip_values = np.concatenate(
        [np.asarray(fields[name]).ravel() for name in precip_names]
    )
    precip_values = precip_values[np.isfinite(precip_values)]
    precip_limits = tuple(np.percentile(precip_values, [2, 98]))
    for ax, (title, field) in zip(axes.ravel(), fields.items()):
        cmap = "terrain" if title == "GLO-90 elevation" else "YlGnBu"
        kwargs = {}
        if title in precip_names:
            kwargs = {"vmin": precip_limits[0], "vmax": precip_limits[1]}
        if title.startswith("Full −"):
            cmap = "RdBu"
            limit = float(np.nanmax(np.abs(field)))
            kwargs = {"vmin": -limit, "vmax": limit}
        image = ax.pcolormesh(
            field.lon, field.lat, field, shading="auto", cmap=cmap, **kwargs
        )
        ax.set_title(title)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("Paper-defined CHELSA V2 precipitation downscaling — Rwanda MAM")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-year", type=int, default=2015)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    print("loading CHIRPS, C3S/ECMWF, ERA5, terrain, and CHELSA exposure...")
    obs = load_obs()
    gcm = load_gcm()
    era5 = load_era5(refresh=args.refresh)
    terrain = ds.load_terrain(obs.mean("year"), cache_dir=CACHE_DIR)
    exposure = load_exposure(refresh=args.refresh).interp_like(obs.isel(year=0))

    train_years = obs.year.values[obs.year.values != args.target_year]
    gcm_train = gcm.sel(year=train_years)
    obs_train = obs.sel(year=train_years)
    forecast = gcm.sel(year=args.target_year)
    truth = obs.sel(year=args.target_year)
    common = {
        "forecast": forecast,
        "terrain": terrain,
        "u_wind": era5["u10"],
        "v_wind": era5["v10"],
        "max_distance_km": 75.0,
        "verbose": False,
    }
    reduced = ds.downscale(gcm_train, obs_train, method="chelsa", **common)
    full = ds.downscale(
        gcm_train,
        obs_train,
        method="chelsa",
        boundary_layer_height=era5["blh"],
        coarse_orography=era5["z"] / 9.80665,
        exposure=exposure,
        **common,
    )
    bilinear = forecast.interp(lat=obs.lat, lon=obs.lon)

    print(f"reduced conservation error: {parent_conservation_error(reduced, forecast):.3g}")
    print(f"full-paper conservation error: {parent_conservation_error(full, forecast):.3g}")
    bundle = xr.Dataset(
        {
            "bilinear": bilinear.mean("member"),
            "chelsa_reduced": reduced.mean("member"),
            "chelsa_full": full.mean("member"),
            "truth": truth,
            "elevation": terrain["elevation"],
            "exposure": exposure,
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle.to_netcdf(DATA_PATH)
    render_plot(
        {
            "GLO-90 elevation": bundle.elevation,
            "Official exposure": bundle.exposure,
            "CHIRPS truth": bundle.truth,
            "Bilinear forecast": bundle.bilinear,
            "CHELSA DEM + wind": bundle.chelsa_reduced,
            "CHELSA full paper": bundle.chelsa_full,
            "Full − reduced": bundle.chelsa_full - bundle.chelsa_reduced,
            "Full − bilinear": bundle.chelsa_full - bundle.bilinear,
        }
    )
    print(f"saved {PLOT_PATH}")
    print(f"saved {DATA_PATH}")


if __name__ == "__main__":
    main()
