"""Robustness of CCA to degenerate predictors and of the MME leverage average to bad models.

Motivation: an un-guarded division by near-zero singular values in the CCA leverage/predict paths
produced CPT leverages of ~1e91 for a near-rank-deficient predictor; because seasonal_mme averages
leverages across models, that one bad model inflated the MME predictive variance without bound and
collapsed every tercile forecast to a constant [0.5, 0, 0.5] (GROC exactly 0.500). The fixes:
`_project_by_sv` drops degenerate modes at source, `fit()` raises on a rank-0 predictor, and the
seasonal_mme leverage average now skips non-finite per-model values.
"""
import numpy as np
import pytest
import xarray as xr

from deepscale.methods.cca import _project_by_sv, _SV_RTOL, CCAMethod


# ---- _project_by_sv: the numerical guard --------------------------------------------

def test_project_by_sv_is_plain_division_when_well_conditioned():
    num = np.array([3.0, -2.0, 5.0])
    sv = np.array([10.0, 4.0, 1.0])          # all comfortably above the rtol cutoff
    assert np.array_equal(_project_by_sv(num, sv), num / sv)


def test_project_by_sv_zeros_degenerate_modes():
    num = np.array([1.0, 1.0, 1.0])
    sv = np.array([1.0, 1e-20, 0.0])         # modes 2,3 negligible vs leading
    out = _project_by_sv(num, sv)
    assert out[0] == 1.0
    assert out[1] == 0.0 and out[2] == 0.0   # not inf / nan
    assert np.all(np.isfinite(out))


def test_project_by_sv_all_zero_singular_values():
    assert np.array_equal(_project_by_sv(np.array([1.0, 2.0]), np.array([0.0, 0.0])),
                          np.array([0.0, 0.0]))


def test_project_by_sv_cutoff_is_relative_to_leading():
    num = np.ones(2)
    sv = np.array([1.0, 2 * _SV_RTOL])        # second mode just above cutoff -> kept
    assert _project_by_sv(num, sv)[1] == 1.0 / (2 * _SV_RTOL)


# ---- fit() rank-0 guard --------------------------------------------------------------

def _grid(values_by_year):
    years = np.arange(len(values_by_year))
    return xr.DataArray(
        np.array(values_by_year, dtype=float)[:, None, None, :].repeat(2, axis=1),
        dims=("year", "member", "lat", "lon"),
        coords={"year": years, "member": [0, 1], "lat": [0.0], "lon": np.arange(np.shape(values_by_year)[1])},
    )


def test_fit_raises_on_zero_variance_predictor():
    n = 12
    # predictor identical every year -> rank-0
    x = xr.DataArray(
        np.ones((n, 2, 1, 4)),
        dims=("year", "member", "lat", "lon"),
        coords={"year": np.arange(n), "member": [0, 1], "lat": [0.0], "lon": np.arange(4)},
    )
    rng = np.random.default_rng(0)
    y = xr.DataArray(
        rng.normal(size=(n, 1, 4)),
        dims=("year", "lat", "lon"),
        coords={"year": np.arange(n), "lat": [0.0], "lon": np.arange(4)},
    )
    with pytest.raises(ValueError, match="no interannual variance"):
        CCAMethod().fit(x, y)


# ---- seasonal_mme leverage average skips non-finite per-model values -----------------

def test_leverage_average_skips_nonfinite():
    # Exercise the real averaging helper from pipelines/seasonal.py (a per-year mean that
    # drops non-finite entries, and is a plain mean when all are finite) — not a local copy,
    # so a regression in the shipped code path is actually caught.
    from deepscale.pipelines.seasonal import _average_leverages as avg

    healthy = {"a": [0.1, 0.2, 0.3], "b": [0.3, 0.2, 0.1]}
    assert avg(healthy) == [0.2, 0.2, 0.2]                     # identical to plain mean

    poisoned = {"a": [0.1, 0.2, 0.3], "b": [np.inf, np.nan, 0.1]}
    out = avg(poisoned)
    assert out == [0.1, 0.2, 0.2]                              # bad model dropped per-year, not poisoning

    all_bad = {"a": [np.inf], "b": [np.nan]}
    assert np.isnan(avg(all_bad)[0])
