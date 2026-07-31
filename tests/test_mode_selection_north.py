"""Scree + North's-rule EOF mode selection, and leverage overflow safety."""
import warnings

import numpy as np
import pytest
import xarray as xr

from deepscale.methods.cca import (
    CCAMethod,
    _LEVERAGE_CLAMP,
    _north_modes,
    select_modes_north,
)


def _wrap(data, years, lat, lon, member=None):
    if member is not None:
        return xr.DataArray(
            data, dims=("year", "member", "lat", "lon"),
            coords={"year": years, "member": member, "lat": lat, "lon": lon})
    return xr.DataArray(
        data, dims=("year", "lat", "lon"),
        coords={"year": years, "lat": lat, "lon": lon})


def _planted_field(k=3, nyears=60, nlat=8, nlon=12, amps=(10.0, 6.0, 3.5),
                   noise=0.2, seed=0):
    """k orthogonal spatial modes with well-separated amplitudes + white noise."""
    rng = np.random.default_rng(seed)
    ncell = nlat * nlon
    patterns, _ = np.linalg.qr(rng.standard_normal((ncell, k)))   # orthonormal
    ts = rng.standard_normal((nyears, k)) * np.asarray(amps[:k])
    data = (ts @ patterns.T + rng.standard_normal((nyears, ncell)) * noise)
    years = np.arange(1990, 1990 + nyears)
    lat = np.linspace(-10, 10, nlat)
    lon = np.linspace(0, 33, nlon)
    return _wrap(data.reshape(nyears, nlat, nlon), years, lat, lon)


def test_north_finds_planted_mode_count():
    field = _planted_field(k=3)
    assert _north_modes(field, ceiling=10) == 3


def test_north_white_noise_returns_one():
    rng = np.random.default_rng(1)
    field = _wrap(rng.standard_normal((40, 6, 9)), np.arange(40),
                  np.linspace(-5, 5, 6), np.linspace(0, 24, 9))
    assert _north_modes(field, ceiling=10) == 1


def test_north_respects_ceiling():
    field = _planted_field(k=3)
    assert _north_modes(field, ceiling=2) == 2
    assert _north_modes(field, ceiling=1) == 1


def test_north_averages_member_dim():
    base = _planted_field(k=2, amps=(10.0, 6.0))
    rng = np.random.default_rng(2)
    members = np.stack([base.values + rng.standard_normal(base.shape) * 0.1
                        for _ in range(4)], axis=1)
    field = _wrap(members, base.year.values, base.lat.values, base.lon.values,
                  member=np.arange(4))
    assert _north_modes(field, ceiling=10) == 2


def test_north_ignores_nan_cells():
    field = _planted_field(k=2, amps=(10.0, 6.0))
    field = field.copy()
    field[:, 0, 0] = np.nan          # one all-NaN cell must not break selection
    assert _north_modes(field, ceiling=10) == 2


def test_select_modes_north_returns_min_for_cca():
    gcm = _planted_field(k=3, seed=3)
    obs = _planted_field(k=2, amps=(10.0, 6.0), nlat=10, nlon=14, seed=4)
    xe, ye, cc = select_modes_north(gcm, obs, x_ceiling=7, y_ceiling=10)
    assert (xe, ye, cc) == (3, 2, 2)


# --- leverage overflow safety ----------------------------------------------

def _fitted_cca():
    rng = np.random.default_rng(5)
    nyears, nlat, nlon, nmem = 30, 5, 7, 3
    years = np.arange(1991, 1991 + nyears)
    lat = np.linspace(-10, 10, nlat)
    lon = np.linspace(20, 44, nlon)
    hcst = _wrap(rng.standard_normal((nyears, nmem, nlat, nlon)) + 5.0,
                 years, lat, lon, member=np.arange(nmem))
    obs = _wrap(rng.standard_normal((nyears, nlat, nlon)) + 5.0, years, lat, lon)
    m = CCAMethod(x_eof_modes=3, y_eof_modes=3, cca_modes=2)
    m.fit(hcst, obs)
    return m, lat, lon, nmem


def test_leverage_normal_forecast_is_modest():
    m, lat, lon, nmem = _fitted_cca()
    rng = np.random.default_rng(6)
    fcst = _wrap(rng.standard_normal((1, nmem, len(lat), len(lon))) + 5.0,
                 [2099], lat, lon, member=np.arange(nmem))
    lev = m.leverage(fcst)
    assert np.isfinite(lev)
    assert 0.0 < lev < _LEVERAGE_CLAMP


def test_leverage_corrupt_forecast_clamps_and_warns():
    m, lat, lon, nmem = _fitted_cca()
    corrupt = _wrap(np.full((1, nmem, len(lat), len(lon)), 1e300),
                    [2099], lat, lon, member=np.arange(nmem))
    with pytest.warns(RuntimeWarning, match="leverage"):
        lev = m.leverage(corrupt)          # must not raise OverflowError
    assert lev == _LEVERAGE_CLAMP
    assert np.isfinite(lev)
