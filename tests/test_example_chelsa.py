import numpy as np
import xarray as xr

from examples.demo_chelsa import parent_conservation_error


def test_demo_parent_conservation_diagnostic():
    coarse = xr.DataArray(
        [[2.0, 4.0], [6.0, 8.0]],
        dims=("lat", "lon"),
        coords={"lat": [0.1, 0.3], "lon": [0.1, 0.3]},
    )
    fine_lat = np.arange(0.025, 0.4, 0.05)
    fine_lon = np.arange(0.025, 0.4, 0.05)
    iy = np.abs(fine_lat[:, None] - coarse.lat.values).argmin(1)
    ix = np.abs(fine_lon[:, None] - coarse.lon.values).argmin(1)
    values = coarse.values[iy[:, None], ix[None, :]]
    fine = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": fine_lat, "lon": fine_lon},
    )
    assert parent_conservation_error(fine, coarse) < 1e-12
