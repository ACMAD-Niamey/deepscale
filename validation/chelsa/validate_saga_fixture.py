"""Validate an independent CHELSA wind-effect function against SAGA output.

This is a component check, not the end-to-end CHELSA reproduction gate.
By default it checks the prototype retained in ``~/experiments``; a different
checkout can be supplied with ``--prototype-root``.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOTYPE = Path.home() / "experiments" / "orographic-disagg"

RIDGE = np.array(
    [
        [0, 0, 0, 400, 900, 400, 0, 0, 0],
        [0, 0, 0, 500, 1100, 500, 0, 0, 0],
        [0, 0, 0, 600, 1300, 600, 0, 0, 0],
        [0, 0, 0, 700, 1500, 700, 0, 0, 0],
        [0, 0, 0, 800, 1700, 800, 0, 0, 0],
        [0, 0, 0, 700, 1500, 700, 0, 0, 0],
        [0, 0, 0, 600, 1300, 600, 0, 0, 0],
        [0, 0, 0, 500, 1100, 500, 0, 0, 0],
        [0, 0, 0, 400, 900, 400, 0, 0, 0],
    ],
    dtype=float,
)


def _load_wind_effect(root: Path):
    source = root / "methods" / "chelsa.py"
    if not source.exists():
        raise FileNotFoundError(f"CHELSA prototype not found: {source}")
    spec = importlib.util.spec_from_file_location("chelsa_candidate", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.wind_effect


def compare(prototype_root: Path) -> dict[str, float]:
    wind_effect = _load_wind_effect(prototype_root)
    spacing_degrees = 5000.0 / 111_195.08
    coords = np.arange(9) * spacing_degrees
    elevation = xr.DataArray(
        RIDGE,
        dims=("lat", "lon"),
        coords={"lat": coords, "lon": coords},
    )
    expected = np.loadtxt(HERE / "saga_wind_effect.csv", delimiter=",")
    actual = wind_effect(
        elevation,
        xr.ones_like(elevation),
        xr.zeros_like(elevation),
        max_distance_km=300.0,
    ).values
    difference = np.abs(actual - expected)
    return {
        "mean_abs_difference": float(np.nanmean(difference)),
        "max_abs_difference": float(np.nanmax(difference)),
        "central_max_abs_difference": float(
            np.nanmax(difference[1:-1, 1:-1])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype-root", type=Path, default=DEFAULT_PROTOTYPE)
    args = parser.parse_args()
    metrics = compare(args.prototype_root)
    for name, value in metrics.items():
        print(f"{name}={value:.9g}")
    if metrics["mean_abs_difference"] > 0.005:
        raise SystemExit("SAGA wind-effect component tolerance exceeded")
    print("component check passed; end-to-end CHELSA parity remains required")


if __name__ == "__main__":
    main()
