"""Accumulation and climatological positioning."""
import numpy as np
import pytest
import xarray as xr

from deepscale.climate import (
    accumulate,
    frequency_below,
    percentile_of,
    rank_of_record,
)


@pytest.fixture
def increments():
    """(time=6, lat=2, lon=2) of ones, so totals are trivially predictable."""
    return xr.DataArray(
        np.ones((6, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(6), "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )


@pytest.fixture
def record():
    """A (year=10, lat=2) reference whose values are 0..9 at every cell."""
    values = np.tile(np.arange(10.0)[:, None], (1, 2))
    return xr.DataArray(
        values, dims=("year", "lat"), coords={"year": np.arange(2000, 2010), "lat": [0.0, 1.0]}
    )


# --- accumulate ------------------------------------------------------------


def test_accumulate_without_window_collapses_the_time_axis(increments):
    total = accumulate(increments, dim="time")
    assert "time" not in total.dims
    assert (total.values == 6).all()


def test_accumulate_with_window_keeps_the_axis_and_reports_trailing_totals(increments):
    rolled = accumulate(increments, window=3, dim="time")
    assert rolled.sizes["time"] == 6
    # First two stamps lack a full window -> NaN; the rest sum three ones.
    assert np.isnan(rolled.isel(time=[0, 1], lat=0, lon=0).values).all()
    assert (rolled.isel(time=slice(2, None)).values == 3).all()


def test_accumulate_defaults_to_refusing_a_partial_total(increments):
    holed = increments.copy()
    holed[0, 0, 0] = np.nan
    total = accumulate(holed, dim="time")
    assert np.isnan(total.isel(lat=0, lon=0).item())
    assert total.isel(lat=1, lon=1).item() == 6


def test_accumulate_min_count_1_sums_whatever_is_present(increments):
    holed = increments.copy()
    holed[0, 0, 0] = np.nan
    total = accumulate(holed, dim="time", min_count=1)
    assert total.isel(lat=0, lon=0).item() == 5


def test_accumulate_supports_reductions_other_than_sum(increments):
    scaled = increments * xr.DataArray(np.arange(1.0, 7.0), dims="time", coords={"time": increments.time})
    assert accumulate(scaled, dim="time", how="max").isel(lat=0, lon=0).item() == 6
    assert accumulate(scaled, dim="time", how="mean").isel(lat=0, lon=0).item() == 3.5


def test_accumulate_rejects_a_bad_how_or_missing_dim(increments):
    with pytest.raises(ValueError, match="how must be one of"):
        accumulate(increments, how="median")
    with pytest.raises(ValueError, match="not found on data"):
        accumulate(increments, dim="year")


def test_accumulate_rejects_a_window_longer_than_the_axis(increments):
    with pytest.raises(ValueError, match="window must be between"):
        accumulate(increments, window=99)


# --- percentile_of ---------------------------------------------------------


def test_percentile_of_places_the_median_at_one_half(record):
    # Record is 0..9. The value 4.5 has five below, none tied -> 0.5.
    frac = percentile_of(xr.DataArray(4.5), record)
    assert frac.isel(lat=0).item() == pytest.approx(0.5)


def test_percentile_of_is_bounded_by_zero_and_one(record):
    assert percentile_of(xr.DataArray(-100.0), record).isel(lat=0).item() == 0.0
    assert percentile_of(xr.DataArray(100.0), record).isel(lat=0).item() == 1.0


def test_percentile_of_a_record_member_uses_the_midrank(record):
    # The value 0 is the record's own minimum: nothing below it, one tie.
    # Mid-rank puts it at 0.5/10, not at 0 -- it is not drier than itself.
    assert percentile_of(xr.DataArray(0.0), record).isel(lat=0).item() == pytest.approx(0.05)


def test_percentile_of_weibull_never_returns_an_exact_zero_or_one(record):
    low = percentile_of(xr.DataArray(-100.0), record, method="weibull").isel(lat=0).item()
    high = percentile_of(xr.DataArray(100.0), record, method="weibull").isel(lat=0).item()
    assert 0.0 < low < high < 1.0


def test_percentile_of_gaussian_matches_the_normal_cdf_at_the_mean(record):
    frac = percentile_of(xr.DataArray(4.5), record, method="gaussian")
    assert frac.isel(lat=0).item() == pytest.approx(0.5)


def test_percentile_of_broadcasts_over_the_surviving_dims(record):
    values = xr.DataArray([0.5, 8.5], dims="lat", coords={"lat": [0.0, 1.0]})
    frac = percentile_of(values, record)
    assert frac.sel(lat=0.0).item() == pytest.approx(0.1)
    assert frac.sel(lat=1.0).item() == pytest.approx(0.9)


def test_percentile_of_gamma_is_monotone_and_bounded(record):
    # Record 0..9 has one zero year -> p0 = 0.1; positive amounts fit a Gamma.
    lo = percentile_of(xr.DataArray(1.0), record, method="gamma").isel(lat=0).item()
    mid = percentile_of(xr.DataArray(4.5), record, method="gamma").isel(lat=0).item()
    hi = percentile_of(xr.DataArray(20.0), record, method="gamma").isel(lat=0).item()
    assert 0.0 < lo < mid < hi < 1.0


def test_percentile_of_gamma_puts_zero_rain_at_half_the_dry_mass(record):
    # One of ten years is zero -> p0 = 0.1; a zero value sits at p0/2 = 0.05.
    frac = percentile_of(xr.DataArray(0.0), record, method="gamma")
    assert frac.isel(lat=0).item() == pytest.approx(0.05)


def test_percentile_of_gamma_needs_two_wet_years():
    dry = xr.DataArray(
        np.zeros((5, 1)), dims=("year", "lat"),
        coords={"year": np.arange(5), "lat": [0.0]})
    frac = percentile_of(xr.DataArray(1.0), dry, method="gamma")
    assert np.isnan(frac.isel(lat=0).item())


def test_percentile_of_propagates_nan_values_rather_than_calling_them_driest(record):
    """The naive `(clim < nan).mean()` is 0.0, which would map missing data onto
    'driest on record'. It must be NaN."""
    values = xr.DataArray([np.nan, 4.5], dims="lat", coords={"lat": [0.0, 1.0]})
    frac = percentile_of(values, record)
    assert np.isnan(frac.sel(lat=0.0).item())
    assert frac.sel(lat=1.0).item() == pytest.approx(0.5)


def test_percentile_of_excludes_nan_years_from_the_reference(record):
    holed = record.copy()
    holed[5:, 0] = np.nan  # cell 0 keeps only years 0..4
    frac = percentile_of(xr.DataArray(2.5), holed)
    assert frac.sel(lat=0.0).item() == pytest.approx(3 / 5)
    assert frac.sel(lat=1.0).item() == pytest.approx(3 / 10)


def test_percentile_of_returns_nan_where_the_record_is_empty(record):
    empty = record.copy()
    empty[:, 0] = np.nan
    frac = percentile_of(xr.DataArray(4.5), empty)
    assert np.isnan(frac.sel(lat=0.0).item())


def test_percentile_of_rejects_values_carrying_the_reference_dim(record):
    with pytest.raises(ValueError, match="must not carry the reference dim"):
        percentile_of(record, record)


def test_percentile_of_rejects_an_unknown_method(record):
    with pytest.raises(ValueError, match="method must be"):
        percentile_of(xr.DataArray(1.0), record, method="bogus")


# --- rank_of_record --------------------------------------------------------


def test_rank_one_ascending_means_driest_on_record(record):
    assert rank_of_record(xr.DataArray(-1.0), record).isel(lat=0).item() == 1


def test_rank_of_a_record_member_is_its_position_in_the_record(record):
    # Value 3 has three strictly-smaller years (0, 1, 2) -> rank 4.
    assert rank_of_record(xr.DataArray(3.0), record).isel(lat=0).item() == 4


def test_a_new_maximum_ranks_n_plus_one(record):
    assert rank_of_record(xr.DataArray(100.0), record).isel(lat=0).item() == 11


def test_descending_rank_one_means_wettest_on_record(record):
    assert rank_of_record(xr.DataArray(100.0), record, ascending=False).isel(lat=0).item() == 1


def test_ties_share_the_lower_rank(record):
    tied = record.copy()
    tied[0:2, :] = 0.0  # two equal-driest years
    assert rank_of_record(xr.DataArray(0.0), tied).isel(lat=0).item() == 1


def test_rank_propagates_nan_values(record):
    values = xr.DataArray([np.nan, 3.0], dims="lat", coords={"lat": [0.0, 1.0]})
    rank = rank_of_record(values, record)
    assert np.isnan(rank.sel(lat=0.0).item())
    assert rank.sel(lat=1.0).item() == 4


def test_rank_and_percentile_agree_on_ordering(record):
    """Whatever the estimators, a lower rank must never carry a higher
    percentile -- they are two views of the same ordering."""
    values = xr.DataArray(np.linspace(-1, 10, 12), dims="v")
    ranks = rank_of_record(values, record).isel(lat=0)
    fracs = percentile_of(values, record).isel(lat=0)
    order = np.argsort(ranks.values, kind="stable")
    assert np.all(np.diff(fracs.values[order]) >= 0)


# --- frequency_below -------------------------------------------------------


def test_frequency_below_counts_the_share_under_the_tercile(record):
    """The 1/3 quantile of 0..9 is 3.0; a sample of {0,1,2,3,9} has three of its
    five members strictly below 3.0."""
    sample = xr.DataArray(
        np.tile(np.array([0.0, 1.0, 2.0, 3.0, 9.0])[:, None], (1, 2)),
        dims=("year", "lat"),
        coords={"year": np.arange(5), "lat": [0.0, 1.0]},
    )
    freq = frequency_below(sample, record)
    assert np.allclose(freq.values, 3 / 5)


def test_frequency_below_honours_the_q_threshold(record):
    """A higher q admits more of the sample below the threshold."""
    sample = xr.DataArray(
        np.arange(10.0)[:, None] * np.ones((1, 2)),
        dims=("year", "lat"),
        coords={"year": np.arange(10), "lat": [0.0, 1.0]},
    )
    low = frequency_below(sample, record, q=0.1)
    high = frequency_below(sample, record, q=0.9)
    assert np.all(high.values >= low.values)
    assert np.all(high.values > 0.5)


def test_frequency_below_preserves_surviving_dims(record):
    """The result keeps every dim except the reduced one."""
    sample = record  # (year, lat)
    freq = frequency_below(sample, record)
    assert freq.dims == ("lat",)


def test_frequency_below_excludes_nan_sample_from_the_fraction(record):
    """A NaN sample entry is dropped, not counted as below-threshold."""
    sample = xr.DataArray(
        np.array([0.0, np.nan, 9.0])[:, None] * np.ones((1, 2)),
        dims=("year", "lat"),
        coords={"year": np.arange(3), "lat": [0.0, 1.0]},
    )
    freq = frequency_below(sample, record)  # of the two valid, one (0.0) is below 3.0
    assert np.allclose(freq.values, 0.5)


def test_frequency_below_returns_nan_where_the_climatology_is_empty():
    clim = xr.DataArray(
        np.full((10, 2), np.nan), dims=("year", "lat"),
        coords={"year": np.arange(10), "lat": [0.0, 1.0]},
    )
    sample = xr.DataArray(
        np.ones((3, 2)), dims=("year", "lat"),
        coords={"year": np.arange(3), "lat": [0.0, 1.0]},
    )
    freq = frequency_below(sample, clim)
    assert bool(freq.isnull().all())


def test_frequency_below_rejects_bad_inputs(record):
    with pytest.raises(ValueError):
        frequency_below(record.isel(year=0), record)          # sample lacks the dim
    with pytest.raises(ValueError):
        frequency_below(record, record.isel(year=0))          # climatology lacks the dim
    with pytest.raises(ValueError):
        frequency_below(record, record, q=1.5)                # q out of range
