"""Tests for the percent-of-normal climate positioning utility."""

import numpy as np
import xarray as xr


def _obs(seed=7, n_years=20):
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    lat, lon = np.linspace(0, 4, 3), np.linspace(30, 34, 3)
    return xr.DataArray(
        rng.gamma(2.0, 2.0, (n_years, 3, 3)) + 0.5,
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": lat, "lon": lon},
    )


def test_percent_of_normal_basic_value():
    import deepscale as ds

    out = ds.percent_of_normal(xr.DataArray(50.0), xr.DataArray(100.0))
    np.testing.assert_allclose(float(out), 50.0)


def test_percent_of_normal_reduces_year_reference():
    import deepscale as ds

    obs = _obs()
    value = obs.isel(year=-1, drop=True)
    out = ds.percent_of_normal(value, obs)
    expected = 100.0 * value / obs.mean("year")
    np.testing.assert_allclose(out.values, expected.values)


def test_percent_of_normal_dry_cell_is_nan():
    import deepscale as ds

    clim = xr.DataArray(
        [[0.0, 10.0], [np.nan, -1.0]],
        dims=["lat", "lon"],
        coords={"lat": [0.0, 1.0], "lon": [30.0, 31.0]},
    )
    out = ds.percent_of_normal(xr.full_like(clim, 5.0), clim)
    assert np.isnan(out.values[0, 0])
    assert np.isnan(out.values[1, 0])
    assert np.isnan(out.values[1, 1])
    np.testing.assert_allclose(out.values[0, 1], 50.0)


def test_percent_of_normal_dry_threshold_param():
    import deepscale as ds

    clim = xr.DataArray([0.4, 2.0], dims=["lon"], coords={"lon": [30.0, 31.0]})
    value = xr.DataArray([1.0, 1.0], dims=["lon"], coords={"lon": [30.0, 31.0]})
    out = ds.percent_of_normal(value, clim, dry_threshold=0.5)
    assert np.isnan(out.values[0])
    np.testing.assert_allclose(out.values[1], 50.0)
