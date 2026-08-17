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


# --- missing="climatology": absent components vote the flat simplex ------------------
# The two live operational conventions for an absent component. `drop` (default) is
# ACMAD's/ICPAC's per-cell nanmean; `climatology` mirrors PyCPT NextGen, whose
# `construct_mme_new` fills missing members with [33, 34, 33] before `.mean('model')`.

def test_climatology_fill_keeps_full_divisor():
    a = _probs(LAT, LON, (0.6, 0.3, 0.1))
    b = _probs(LAT, LON, (0.2, 0.3, 0.5))
    b.loc[dict(lat=0.0, lon=10.0)] = np.nan
    out = deepscale.combine_terciles([a, b], missing="climatology")
    # b abstaining would give 0.6; voting climatology gives (0.6 + 1/3) / 2.
    np.testing.assert_allclose(out.sel(tercile=0, lat=0.0, lon=10.0).values,
                               (0.6 + 1 / 3) / 2)
    # cells where both are present are unaffected.
    np.testing.assert_allclose(out.sel(tercile=0, lat=2.0, lon=12.0).values, 0.4)
    np.testing.assert_allclose(out.sum("tercile").values, 1.0)


def test_climatology_fill_pulls_toward_flat_not_away():
    """A masked component should damp the tilt, never blank or amplify it."""
    a = _probs(LAT, LON, (0.7, 0.2, 0.1))
    b = _probs(LAT, LON, (0.7, 0.2, 0.1))
    b.loc[dict(lat=0.0, lon=10.0)] = np.nan
    cell = dict(tercile=0, lat=0.0, lon=10.0)
    dropped = deepscale.combine_terciles([a, b])
    filled = deepscale.combine_terciles([a, b], missing="climatology")
    assert float(dropped.sel(**cell)) == pytest.approx(0.7)          # abstain: full tilt
    assert 1 / 3 < float(filled.sel(**cell)) < 0.7                   # vote: damped tilt


def test_all_components_absent_stays_nan_under_both():
    a = _probs(LAT, LON, (0.6, 0.3, 0.1))
    b = _probs(LAT, LON, (0.2, 0.3, 0.5))
    for da in (a, b):
        da.loc[dict(lat=0.0, lon=10.0)] = np.nan
    cell = dict(tercile=0, lat=0.0, lon=10.0)
    for mode in ("drop", "climatology"):
        out = deepscale.combine_terciles([a, b], missing=mode)
        assert np.isnan(out.sel(**cell).values), f"{mode} invented a forecast"


def test_climatology_fill_respects_weights():
    a = _probs(LAT, LON, (0.8, 0.1, 0.1))
    b = _probs(LAT, LON, (0.2, 0.4, 0.4))
    b.loc[dict(lat=0.0, lon=10.0)] = np.nan
    out = deepscale.combine_terciles([a, b], weights=[3, 1], missing="climatology")
    np.testing.assert_allclose(out.sel(tercile=0, lat=0.0, lon=10.0).values,
                               0.75 * 0.8 + 0.25 * (1 / 3))


def test_partially_nan_component_is_treated_as_absent():
    """One NaN category must not yield a mixed 2-/3-category average."""
    a = _probs(LAT, LON, (0.6, 0.3, 0.1))
    b = _probs(LAT, LON, (0.2, 0.3, 0.5))
    b.loc[dict(tercile=2, lat=0.0, lon=10.0)] = np.nan     # only the 'above' category
    cell = dict(lat=0.0, lon=10.0)
    out = deepscale.combine_terciles([a, b])
    np.testing.assert_allclose(out.sel(tercile=0, **cell).values, 0.6)   # b fully ignored
    np.testing.assert_allclose(out.sel(**cell).sum("tercile").values, 1.0)


def test_rejects_bad_missing_mode():
    a = _probs(LAT, LON, (0.5, 0.3, 0.2))
    with pytest.raises(ValueError):
        deepscale.combine_terciles([a, a], missing="climatological")
