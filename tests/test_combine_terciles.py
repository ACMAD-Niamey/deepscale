"""deepscale.combine_terciles — generic tercile-forecast combination.

Added on the `acmad` branch for the component-equal objective.
"""
import numpy as np
import pytest
import xarray as xr

import deepscale


def _probs(lat, lon, fill):
    """A (tercile, lat, lon) forecast whose three categories are `fill` (a 3-tuple)."""
    a = np.empty((3, len(lat), len(lon)))
    for k in range(3):
        a[k] = fill[k]
    return xr.DataArray(a, dims=("tercile", "lat", "lon"),
                        coords={"tercile": [0, 1, 2], "lat": lat, "lon": lon})


LAT = np.array([-2.0, 0.0, 2.0])
LON = np.array([10.0, 12.0])


def test_equal_weight_average_and_simplex():
    a = _probs(LAT, LON, (0.6, 0.3, 0.1))
    b = _probs(LAT, LON, (0.2, 0.3, 0.5))
    out = deepscale.combine_terciles([a, b])
    # equal-weight mean: below .4, normal .3, above .3 — already sums to 1.
    np.testing.assert_allclose(out.sel(tercile=0).values, 0.4)
    np.testing.assert_allclose(out.sel(tercile=2).values, 0.3)
    np.testing.assert_allclose(out.sum("tercile").values, 1.0)


def test_weights_renormalized():
    a = _probs(LAT, LON, (0.9, 0.05, 0.05))
    b = _probs(LAT, LON, (0.3, 0.35, 0.35))
    out = deepscale.combine_terciles([a, b], weights=[3, 1])   # 0.75 / 0.25
    np.testing.assert_allclose(out.sel(tercile=0).values, 0.75 * 0.9 + 0.25 * 0.3)


def test_nan_component_is_skipped_per_cell():
    a = _probs(LAT, LON, (0.6, 0.3, 0.1))
    b = _probs(LAT, LON, (0.2, 0.3, 0.5))
    b.loc[dict(lat=0.0, lon=10.0)] = np.nan          # one cell missing in b
    out = deepscale.combine_terciles([a, b])
    # missing cell falls back to a alone; a present cell is the mean.
    np.testing.assert_allclose(out.sel(tercile=0, lat=0.0, lon=10.0).values, 0.6)
    np.testing.assert_allclose(out.sel(tercile=0, lat=2.0, lon=12.0).values, 0.4)


def test_hierarchy_component_equal_matches_manual():
    """Two-level component-equal (ACMAD objective) == 1/3 each of the group MMEs."""
    exp1 = deepscale.combine_terciles([_probs(LAT, LON, (0.7, 0.2, 0.1)),
                                       _probs(LAT, LON, (0.5, 0.3, 0.2))])
    exp2 = _probs(LAT, LON, (0.2, 0.3, 0.5))
    exp3 = _probs(LAT, LON, (0.3, 0.4, 0.3))
    obj = deepscale.combine_terciles([exp1, exp2, exp3])
    manual_below = np.mean([0.6, 0.2, 0.3])          # exp1 below is (0.7+0.5)/2 = 0.6
    np.testing.assert_allclose(obj.sel(tercile=0).values, manual_below)
    np.testing.assert_allclose(obj.sum("tercile").values, 1.0)


def test_regrid_to_common_grid():
    coarse = _probs(LAT, LON, (0.5, 0.3, 0.2))
    fine_lat = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    fine = _probs(fine_lat, LON, (0.3, 0.3, 0.4))
    out = deepscale.combine_terciles([coarse, fine], regrid_to=coarse)
    assert list(out.lat.values) == list(LAT)         # combined on the coarse grid
    np.testing.assert_allclose(out.sum("tercile").values, 1.0)


def test_accepts_latitude_longitude_dim_aliases():
    """combine_terciles resolves the lat/latitude/lon/longitude aliases like the
    rest of deepscale, rather than requiring lat/lon (it previously KeyError'd)."""
    def _aliased(fill):
        a = np.empty((3, len(LAT), len(LON)))
        for k in range(3):
            a[k] = fill[k]
        return xr.DataArray(a, dims=("tercile", "latitude", "longitude"),
                            coords={"tercile": [0, 1, 2], "latitude": LAT, "longitude": LON})

    out = deepscale.combine_terciles([_aliased((0.6, 0.3, 0.1)), _aliased((0.2, 0.3, 0.5))])
    assert "lat" in out.dims and "lon" in out.dims          # canonicalised on the way out
    np.testing.assert_allclose(out.sel(tercile=0).values, 0.4)
    np.testing.assert_allclose(out.sum("tercile").values, 1.0)


def test_rejects_bad_weights():
    a = _probs(LAT, LON, (0.5, 0.3, 0.2))
    with pytest.raises(ValueError):
        deepscale.combine_terciles([a, a], weights=[1])          # wrong length
    with pytest.raises(ValueError):
        deepscale.combine_terciles([a, a], weights=[0, 0])       # all zero
