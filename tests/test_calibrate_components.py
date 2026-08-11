"""calibrate(): per-model components, eREG deterministic output, LOYO CV."""
import numpy as np
import pytest
import xarray as xr

import deepscale
from deepscale import CalibrateResult


def _grid_field(nyears=24, nlat=4, nlon=5, seed=0, offset=5.0):
    rng = np.random.default_rng(seed)
    years = np.arange(1991, 1991 + nyears)
    lat = np.linspace(-5, 5, nlat)
    lon = np.linspace(30, 42, nlon)
    signal = np.sin(np.linspace(0, 6, nyears))[:, None, None]
    data = signal + rng.standard_normal((nyears, nlat, nlon)) * 0.4 + offset
    return xr.DataArray(data, dims=("year", "lat", "lon"),
                        coords={"year": years, "lat": lat, "lon": lon})


@pytest.fixture
def obs():
    return _grid_field(seed=1)


@pytest.fixture
def ereg_models(obs):
    """Two fake GCMs on the obs grid, correlated with obs + own biases."""
    rng = np.random.default_rng(2)
    models = {}
    for i, name in enumerate(("gcm_a", "gcm_b")):
        noise = rng.standard_normal(obs.shape) * 0.5
        hcst = (obs * (0.8 + 0.2 * i) + noise + 1.5 * i).rename(name)
        models[name] = (hcst, None)
    return models


# --- return_components -----------------------------------------------------

def test_ereg_components_match_combined(obs, ereg_models):
    res = deepscale.calibrate(ereg_models, obs=obs, method="ereg",
                              forecast_year=int(obs.year[-1]),
                              return_components=True)
    assert isinstance(res, CalibrateResult)
    assert set(res.per_model) == {"gcm_a", "gcm_b"}
    for m in res.per_model.values():
        s = m.sum("tercile")
        assert np.allclose(s.values[np.isfinite(s.values)], 1.0, atol=1e-6)
    # combined is the renormalized cross-model mean of the components
    recombined = xr.concat(list(res.per_model.values()), dim="model").mean("model")
    recombined = recombined / recombined.sum("tercile")
    xr.testing.assert_allclose(res.combined, recombined.transpose(*res.combined.dims))


def test_plain_call_unchanged(obs, ereg_models):
    plain = deepscale.calibrate(ereg_models, obs=obs, method="ereg",
                                forecast_year=int(obs.year[-1]))
    res = deepscale.calibrate(ereg_models, obs=obs, method="ereg",
                              forecast_year=int(obs.year[-1]),
                              return_components=True)
    xr.testing.assert_allclose(plain, res.combined)


def test_logit_components(obs):
    rng = np.random.default_rng(3)
    idx = {}
    fc = {}
    for name in ("m1", "m2"):
        series = xr.DataArray(rng.standard_normal(obs.sizes["year"]),
                              dims="year", coords={"year": obs.year})
        idx[name] = series
        fc[name] = 0.5
    res = deepscale.calibrate(idx, obs=obs, method="logit", forecast=fc,
                              min_years=5, return_components=True)
    assert set(res.per_model) == {"m1", "m2"}
    assert res.combined.dims == res.per_model["m1"].dims


def test_components_rejected_for_unsupported_method(obs):
    with pytest.raises(ValueError, match="return_components"):
        deepscale.calibrate(obs, obs=obs, method="smoothed_regression",
                            return_components=True)


# --- eREG deterministic -----------------------------------------------------

def test_ereg_deterministic_combined_and_components(obs, ereg_models):
    res = deepscale.calibrate(ereg_models, obs=obs, method="ereg",
                              forecast_year=int(obs.year[-1]),
                              output_type="deterministic",
                              clip_negative=True, return_components=True)
    assert "tercile" not in res.combined.dims
    assert np.isfinite(res.combined.values).all()
    assert (res.combined.values >= 0).all()          # clip_negative
    stacked = xr.concat(list(res.per_model.values()), dim="model").mean("model")
    xr.testing.assert_allclose(res.combined, stacked)


def test_ereg_deterministic_tracks_obs_scale(obs, ereg_models):
    det = deepscale.calibrate(ereg_models, obs=obs, method="ereg",
                              forecast_year=int(obs.year[-1]),
                              output_type="deterministic")
    # calibrated back to obs units: same order of magnitude as obs mean
    assert abs(float(det.mean()) - float(obs.mean())) < 2.0


# --- LOYO cross-validation ---------------------------------------------------

def test_ereg_cv_shape_and_simplex(obs, ereg_models):
    cvh = deepscale.calibrate(ereg_models, obs=obs, method="ereg", cv="loyo")
    assert list(cvh.dims)[0:1] == ["year"] or "year" in cvh.dims
    assert cvh.sizes["year"] == obs.sizes["year"]
    s = cvh.sum("tercile")
    assert np.allclose(s.values[np.isfinite(s.values)], 1.0, atol=1e-6)
    # scoreable
    report = deepscale.skill(cvh, obs, metrics=["rpss"])
    assert np.isfinite(report.scores["rpss"])


def test_ereg_cv_excludes_test_year(obs, ereg_models):
    """A fold must not see its test year: with window=1 the prediction for a
    year differs from the all-years fit's prediction for that same year."""
    cvh = deepscale.calibrate(ereg_models, obs=obs, method="ereg", cv="loyo")
    full = deepscale.calibrate(ereg_models, obs=obs, method="ereg",
                               forecast_year=int(obs.year[5]))
    fold = cvh.sel(year=int(obs.year[5]))
    assert not np.allclose(fold.values, full.values, atol=1e-8)


def test_cv_rejected_for_deterministic(obs, ereg_models):
    with pytest.raises(ValueError, match="tercile"):
        deepscale.calibrate(ereg_models, obs=obs, method="ereg",
                            cv="loyo", output_type="deterministic")


def test_logit_cv_runs(obs):
    rng = np.random.default_rng(4)
    idx = {"m1": xr.DataArray(rng.standard_normal(obs.sizes["year"]),
                              dims="year", coords={"year": obs.year})}
    cvh = deepscale.calibrate(idx, obs=obs, method="logit", cv="loyo",
                              min_years=5)
    assert cvh.sizes["year"] == obs.sizes["year"]
