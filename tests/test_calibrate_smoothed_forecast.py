"""Round 2 of smoothed_regression (issue #5): out-of-sample forecast application.

Fit on the hindcast, apply to a separate forecast ensemble — plus the
{model: (hindcast, forecast)} pooled super-ensemble entry and the
predictor/obs year intersection that makes the split possible.
"""
import numpy as np
import pytest
import xarray as xr

import deepscale

SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
           "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def _cube(n_year=25, n_member=6, nlat=3, nlon=4, seed=0, first_year=1991):
    rng = np.random.default_rng(seed)
    ns = len(SEASONS)
    signal = rng.standard_normal((ns, n_year, nlat, nlon))
    members = signal[:, :, None] + 0.3 * rng.standard_normal((ns, n_year, n_member, nlat, nlon))
    obs = 0.7 * signal + 0.1 * rng.standard_normal((ns, n_year, nlat, nlon))
    coords = {"season": SEASONS, "year": np.arange(first_year, first_year + n_year),
              "member": np.arange(n_member), "lat": np.linspace(50, 60, nlat),
              "lon": np.linspace(-110, -100, nlon)}
    fc = xr.DataArray(members, dims=("season", "year", "member", "lat", "lon"), coords=coords)
    ob = xr.DataArray(obs, dims=("season", "year", "lat", "lon"),
                      coords={k: coords[k] for k in ("season", "year", "lat", "lon")})
    return fc, ob


def _gamma_cube(n_year=25, n_member=6, nlat=3, nlon=4, seed=0):
    # non-negative (rainfall-like) cube for the gamma path
    rng = np.random.default_rng(seed)
    ns = len(SEASONS)
    base = rng.gamma(2.0, 50.0, (ns, n_year, nlat, nlon))
    members = base[:, :, None] * rng.gamma(8.0, 1 / 8.0, (ns, n_year, n_member, nlat, nlon))
    obs = base * rng.gamma(8.0, 1 / 8.0, (ns, n_year, nlat, nlon))
    coords = {"season": SEASONS, "year": np.arange(1991, 1991 + n_year),
              "member": np.arange(n_member), "lat": np.linspace(50, 60, nlat),
              "lon": np.linspace(-110, -100, nlon)}
    fc = xr.DataArray(members, dims=("season", "year", "member", "lat", "lon"), coords=coords)
    ob = xr.DataArray(obs, dims=("season", "year", "lat", "lon"),
                      coords={k: coords[k] for k in ("season", "year", "lat", "lon")})
    return fc, ob


def _members_of(fc, year):
    """One hindcast year's members as an out-of-sample-shaped forecast
    (season, member, lat, lon), no year coordinate."""
    return fc.sel(year=year, drop=True)


# --- in-sample equivalence: forecast= of a hindcast year's members must
# --- reproduce the round-1 forecast_year= result exactly -------------------

def test_deterministic_forecast_reproduces_hindcast_year():
    fc, ob = _cube()
    r1 = deepscale.calibrate(fc, ob, method="smoothed_regression",
                             output_type="deterministic", temporal_sigma="constant",
                             forecast_year=2010)
    r2 = deepscale.calibrate(fc, ob, method="smoothed_regression",
                             output_type="deterministic", temporal_sigma="constant",
                             forecast=_members_of(fc, 2010))
    np.testing.assert_allclose(r2.values, r1.values, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("distribution,make", [("normal", _cube), ("gamma", _gamma_cube)])
def test_tercile_forecast_reproduces_hindcast_year(distribution, make):
    fc, ob = make()
    r1 = deepscale.calibrate(fc, ob, method="smoothed_regression",
                             output_type="tercile", temporal_sigma="constant",
                             distribution=distribution, forecast_year=2010)
    r2 = deepscale.calibrate(fc, ob, method="smoothed_regression",
                             output_type="tercile", temporal_sigma="constant",
                             distribution=distribution, forecast=_members_of(fc, 2010))
    np.testing.assert_allclose(r2.values, r1.values, rtol=1e-10, atol=1e-12)


# --- genuinely out-of-sample application -----------------------------------

def test_deterministic_out_of_sample_responds_to_forecast():
    fc, ob = _cube()
    fcst = _members_of(fc, 2010)
    lo = deepscale.calibrate(fc, ob, method="smoothed_regression",
                             output_type="deterministic", temporal_sigma="constant",
                             forecast=fcst)
    hi = deepscale.calibrate(fc, ob, method="smoothed_regression",
                             output_type="deterministic", temporal_sigma="constant",
                             forecast=fcst + 1.0)
    assert lo.dims == ("season", "lat", "lon")
    assert bool(np.isfinite(lo).all())
    # planted slope is positive: a wetter/warmer forecast raises the calibrated field
    assert bool((hi > lo).all())


def test_tercile_out_of_sample_gamma_probs_valid_and_shifted():
    fc, ob = _gamma_cube()
    # per-cell 90th percentile of the hindcast members, with mild member jitter:
    # an ensemble that is unambiguously above normal everywhere
    rng = np.random.default_rng(7)
    p90 = fc.quantile(0.9, dim=["year", "member"]).drop_vars("quantile")
    jitter = xr.DataArray(rng.uniform(0.95, 1.05, (6,) + p90.shape),
                          dims=("member",) + p90.dims, coords={"member": range(6)})
    wet = (p90 * jitter).transpose("season", "member", "lat", "lon")
    out = deepscale.calibrate(fc, ob, method="smoothed_regression",
                              output_type="tercile", temporal_sigma="constant",
                              distribution="gamma", forecast=wet)
    assert out.dims == ("season", "tercile", "lat", "lon")
    sums = out.sum("tercile").values
    np.testing.assert_allclose(sums[np.isfinite(sums)], 1.0, atol=1e-9)
    above = out.sel(tercile=2).values
    below = out.sel(tercile=0).values
    ok = np.isfinite(above) & np.isfinite(below)
    assert ok.any() and bool((above[ok] > below[ok]).all())


# --- input validation -------------------------------------------------------

def test_forecast_and_forecast_year_mutually_exclusive():
    fc, ob = _cube()
    with pytest.raises(ValueError, match="mutually exclusive"):
        deepscale.calibrate(fc, ob, method="smoothed_regression",
                            output_type="deterministic",
                            forecast=_members_of(fc, 2010), forecast_year=2010)


def test_tercile_forecast_requires_member_dim():
    fc, ob = _cube()
    with pytest.raises(ValueError, match="member"):
        deepscale.calibrate(fc, ob, method="smoothed_regression",
                            output_type="tercile", temporal_sigma="constant",
                            forecast=_members_of(fc, 2010).mean("member"))


def test_forecast_singleton_year_dim_squeezed():
    fc, ob = _cube()
    flat = _members_of(fc, 2010)
    with_year = flat.expand_dims(year=[2026])
    r_flat = deepscale.calibrate(fc, ob, method="smoothed_regression",
                                 output_type="deterministic", temporal_sigma="constant",
                                 forecast=flat)
    r_year = deepscale.calibrate(fc, ob, method="smoothed_regression",
                                 output_type="deterministic", temporal_sigma="constant",
                                 forecast=with_year)
    np.testing.assert_allclose(r_year.values, r_flat.values)


def test_out_of_window_forecast_year_error_points_at_forecast_kwarg():
    fc, ob = _cube()
    with pytest.raises(ValueError, match="forecast="):
        deepscale.calibrate(fc, ob, method="smoothed_regression",
                            output_type="deterministic", forecast_year=2099)


# --- year intersection (the "(22,) (21,)" foot-gun from issue #5) -----------

def test_fit_intersects_predictor_and_obs_years():
    fc, ob = _cube(n_year=26)                  # 1991..2016
    ob = ob.sel(year=slice(1991, 2015))        # obs one year short
    trimmed = deepscale.calibrate(fc.sel(year=slice(1991, 2015)), ob,
                                  method="smoothed_regression",
                                  output_type="deterministic",
                                  temporal_sigma="constant", forecast_year=2010)
    ragged = deepscale.calibrate(fc, ob, method="smoothed_regression",
                                 output_type="deterministic",
                                 temporal_sigma="constant", forecast_year=2010)
    np.testing.assert_allclose(ragged.values, trimmed.values)


# --- multi-model pooled super-ensemble entry --------------------------------

def _two_models():
    h_a, ob = _gamma_cube(n_member=4, seed=1)
    h_b, _ = _gamma_cube(n_member=6, seed=2)
    f_a = _members_of(h_a, 2010) * 1.5
    f_b = _members_of(h_b, 2010) * 1.5
    return h_a, h_b, f_a, f_b, ob


def _pool(*ensembles):
    reindexed, offset = [], 0
    for e in ensembles:
        n = e.sizes["member"]
        reindexed.append(e.assign_coords(member=list(range(offset, offset + n))))
        offset += n
    return xr.concat(reindexed, dim="member")


def test_multimodel_tuple_dict_pools_super_ensemble():
    h_a, h_b, f_a, f_b, ob = _two_models()
    via_dict = deepscale.calibrate({"a": (h_a, f_a), "b": (h_b, f_b)}, ob,
                                   method="smoothed_regression",
                                   output_type="tercile", temporal_sigma="constant",
                                   distribution="gamma")
    manual = deepscale.calibrate(_pool(h_a, h_b), ob,
                                 method="smoothed_regression",
                                 output_type="tercile", temporal_sigma="constant",
                                 distribution="gamma", forecast=_pool(f_a, f_b))
    np.testing.assert_allclose(via_dict.values, manual.values, rtol=1e-10, atol=1e-12)


def test_multimodel_separate_forecast_dict_matches_tuple_form():
    h_a, h_b, f_a, f_b, ob = _two_models()
    tuple_form = deepscale.calibrate({"a": (h_a, f_a), "b": (h_b, f_b)}, ob,
                                     method="smoothed_regression",
                                     output_type="tercile", temporal_sigma="constant",
                                     distribution="gamma")
    dict_form = deepscale.calibrate({"a": h_a, "b": h_b}, ob,
                                    method="smoothed_regression",
                                    output_type="tercile", temporal_sigma="constant",
                                    distribution="gamma",
                                    forecast={"a": f_a, "b": f_b})
    np.testing.assert_allclose(dict_form.values, tuple_form.values)


def test_multimodel_embedded_forecasts_conflict_with_forecast_year():
    # tuple-dict entries carry forecasts; adding forecast_year must raise, not
    # silently calibrate the forecast while claiming a retro target
    h_a, h_b, f_a, f_b, ob = _two_models()
    with pytest.raises(ValueError, match="mutually exclusive"):
        deepscale.calibrate({"a": (h_a, f_a), "b": (h_b, f_b)}, ob,
                            method="smoothed_regression",
                            output_type="deterministic",
                            temporal_sigma="constant", forecast_year=2010)


def test_multimodel_hindcast_only_dict_with_forecast_year():
    h_a, h_b, _f_a, _f_b, ob = _two_models()
    via_dict = deepscale.calibrate({"a": h_a, "b": h_b}, ob,
                                   method="smoothed_regression",
                                   output_type="deterministic",
                                   temporal_sigma="constant", forecast_year=2010)
    manual = deepscale.calibrate(_pool(h_a, h_b), ob,
                                 method="smoothed_regression",
                                 output_type="deterministic",
                                 temporal_sigma="constant", forecast_year=2010)
    np.testing.assert_allclose(via_dict.values, manual.values)
