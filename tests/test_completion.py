"""Scenario completion: splicing observations, a forecast, and analog remainders."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from deepscale.analog import analogs_from_index, analogs_from_years
from deepscale.climate import seasonal_stack
from deepscale.completion import complete

SEASON = "JJAS"
YEARS = np.arange(1981, 2027)
TARGET = 2026
N_STEPS = 12  # 4 months x 3 dekads


def _dekad_stamps(years):
    return pd.DatetimeIndex(
        [pd.Timestamp(y, m, d) for y in years for m in range(1, 13) for d in (1, 11, 21)]
    )


@pytest.fixture
def archive():
    """46 years of dekadal rainfall on a 2x2 grid. Year Y's every dekad is Y-1980
    millimetres, so a year's season total is exactly 12 * (Y - 1980)."""
    stamps = _dekad_stamps(YEARS)
    values = np.empty((len(stamps), 2, 2))
    for i, stamp in enumerate(stamps):
        values[i, :, :] = stamp.year - 1980
    return xr.DataArray(
        values, dims=("time", "lat", "lon"),
        coords={"time": stamps, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )


@pytest.fixture
def climatology(archive):
    return seasonal_stack(archive, SEASON, years=range(1981, 2026))


@pytest.fixture
def observed(archive):
    """June and July of the target year: six dekads at 100 mm each."""
    stamps = pd.DatetimeIndex(
        [pd.Timestamp(TARGET, m, d) for m in (6, 7) for d in (1, 11, 21)]
    )
    return xr.DataArray(
        np.full((6, 2, 2), 100.0), dims=("time", "lat", "lon"),
        coords={"time": stamps, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )


@pytest.fixture
def forecast():
    """The first two dekads of August: 5 mm each — a dry forecast."""
    stamps = pd.DatetimeIndex([pd.Timestamp(TARGET, 8, 1), pd.Timestamp(TARGET, 8, 11)])
    return xr.DataArray(
        np.full((2, 2, 2), 5.0), dims=("time", "lat", "lon"),
        coords={"time": stamps, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )


@pytest.fixture
def analogs():
    # Season totals of these years are 12*10=120, 12*20=240, 12*30=360.
    return analogs_from_years([1990, 2000, 2010], candidates=YEARS[:-1])


# --- seasonal_stack --------------------------------------------------------


def test_seasonal_stack_reshapes_time_into_year_by_step(archive):
    stacked = seasonal_stack(archive, SEASON, years=range(1981, 2026))
    assert stacked.dims == ("year", "step", "lat", "lon")
    assert stacked.sizes == {"year": 45, "step": N_STEPS, "lat": 2, "lon": 2}


def test_seasonal_stack_aligns_the_same_season_position_across_years(archive):
    stacked = seasonal_stack(archive, SEASON, years=[1990, 2000])
    # Step 0 is the first dekad of June in both years.
    assert float(stacked.sel(year=1990, step=0).isel(lat=0, lon=0)) == 10.0
    assert float(stacked.sel(year=2000, step=0).isel(lat=0, lon=0)) == 20.0


def test_seasonal_stack_drops_years_with_no_data_in_the_season(archive):
    stacked = seasonal_stack(archive, SEASON)
    assert stacked.year.values.min() == 1981
    assert stacked.year.values.max() == 2026


def test_seasonal_stack_pads_a_partially_covered_year_with_nan(archive):
    truncated = archive.sel(time=slice(None, f"{TARGET}-07-31"))
    stacked = seasonal_stack(truncated, SEASON, years=[TARGET])
    assert stacked.sizes["step"] == N_STEPS
    assert bool(stacked.isel(year=0, lat=0, lon=0)[:6].notnull().all())
    assert bool(stacked.isel(year=0, lat=0, lon=0)[6:].isnull().all())


def test_seasonal_stack_rejects_a_missing_time_dim(archive):
    with pytest.raises(ValueError, match="not found on data"):
        seasonal_stack(archive.rename(time="t"), SEASON)


# --- the splice ------------------------------------------------------------


def test_scenarios_carry_one_member_per_analog(observed, climatology, analogs):
    got = complete(observed, analogs, climatology=climatology, season=SEASON)
    assert got.scenarios.sizes["scenario"] == 3
    assert list(got.scenarios.scenario.values) == [1990, 2000, 2010]


def test_output_dim_order_is_scenario_step_then_the_rest(observed, climatology, analogs):
    """`xr.where` broadcasts in an arbitrary order; the result must not."""
    got = complete(observed, analogs, climatology=climatology, season=SEASON)
    assert got.scenarios.dims == ("scenario", "step", "lat", "lon")
    assert got.totals.dims == ("scenario", "lat", "lon")
    assert got.consensus.dims == ("lat", "lon")


def test_observed_steps_are_identical_across_every_scenario(observed, climatology, analogs):
    """The whole point: the observed segment is fact, not a scenario. If an
    analog's own values leaked into it, the members would differ there."""
    got = complete(observed, analogs, climatology=climatology, season=SEASON)
    observed_part = got.scenarios.isel(step=slice(0, 6))
    assert float(observed_part.std("scenario").max()) == 0.0
    np.testing.assert_allclose(observed_part.values, 100.0)


def test_analog_steps_differ_across_scenarios(observed, climatology, analogs):
    got = complete(observed, analogs, climatology=climatology, season=SEASON)
    tail = got.scenarios.isel(step=slice(6, None), lat=0, lon=0)
    np.testing.assert_allclose(tail.sel(scenario=1990).values, 10.0)
    np.testing.assert_allclose(tail.sel(scenario=2010).values, 30.0)


def test_totals_are_observed_plus_each_analog_remainder(observed, climatology, analogs):
    got = complete(observed, analogs, climatology=climatology, season=SEASON)
    totals = got.totals.isel(lat=0, lon=0)
    # 6 observed dekads at 100, plus 6 analog dekads at (year - 1980).
    assert float(totals.sel(scenario=1990)) == pytest.approx(600 + 6 * 10)
    assert float(totals.sel(scenario=2000)) == pytest.approx(600 + 6 * 20)
    assert float(totals.sel(scenario=2010)) == pytest.approx(600 + 6 * 30)


def test_forecast_segment_replaces_the_first_analog_steps(observed, forecast, climatology, analogs):
    got = complete(observed, analogs, climatology=climatology, season=SEASON,
                   forecast=forecast)
    assert list(got.segment_steps("observed")) == list(range(6))
    assert list(got.segment_steps("forecast")) == [6, 7]
    assert list(got.segment_steps("analog")) == [8, 9, 10, 11]
    np.testing.assert_allclose(
        got.scenarios.isel(step=[6, 7]).std("scenario").values, 0.0
    )


def test_a_dry_forecast_lowers_every_scenario_total(observed, forecast, climatology, analogs):
    """Running with and without the forecast isolates its contribution — the
    deck's two configurations, as one parameter."""
    without = complete(observed, analogs, climatology=climatology, season=SEASON)
    with_fcst = complete(observed, analogs, climatology=climatology, season=SEASON,
                         forecast=forecast)
    assert bool((with_fcst.totals < without.totals).all())
    # Two dekads of 5 mm replaced two of (year - 1980).
    delta = float((without.totals - with_fcst.totals).sel(scenario=2010).isel(lat=0, lon=0))
    assert delta == pytest.approx(2 * (30 - 5))


def test_completion_without_observations_is_a_pure_analog_projection(
    forecast, climatology, analogs
):
    got = complete(None, analogs, climatology=climatology, season=SEASON,
                   forecast=forecast)
    assert got.metadata["n_observed_steps"] == 0
    assert list(got.segment_steps("forecast")) == [6, 7]


def test_completion_needs_observations_or_a_forecast(climatology, analogs):
    with pytest.raises(ValueError, match="at least one of observed or forecast"):
        complete(None, analogs, climatology=climatology, season=SEASON)


# --- overlap policy --------------------------------------------------------


def _overlapping_forecast():
    stamps = pd.DatetimeIndex([pd.Timestamp(TARGET, 7, 21), pd.Timestamp(TARGET, 8, 1)])
    return xr.DataArray(
        np.full((2, 2, 2), 7.0), dims=("time", "lat", "lon"),
        coords={"time": stamps, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )


def test_observations_win_over_the_forecast_by_default(observed, climatology, analogs):
    got = complete(observed, analogs, climatology=climatology, season=SEASON,
                   forecast=_overlapping_forecast())
    assert got.segments.sel(step=5).item() == "observed"
    assert float(got.scenarios.isel(step=5, lat=0, lon=0)[0]) == 100.0


def test_overlap_forecast_lets_the_forecast_win(observed, climatology, analogs):
    got = complete(observed, analogs, climatology=climatology, season=SEASON,
                   forecast=_overlapping_forecast(), overlap="forecast")
    assert got.segments.sel(step=5).item() == "forecast"
    assert float(got.scenarios.isel(step=5, lat=0, lon=0)[0]) == 7.0


def test_overlap_error_refuses_to_choose(observed, climatology, analogs):
    with pytest.raises(ValueError, match="overlap at steps"):
        complete(observed, analogs, climatology=climatology, season=SEASON,
                 forecast=_overlapping_forecast(), overlap="error")


def test_unknown_overlap_policy_is_rejected(observed, climatology, analogs):
    with pytest.raises(ValueError, match="overlap must be one of"):
        complete(observed, analogs, climatology=climatology, season=SEASON, overlap="both")


# --- consensus and percentile ---------------------------------------------


def test_consensus_defaults_to_the_median_of_the_scenario_totals(observed, climatology, analogs):
    got = complete(observed, analogs, climatology=climatology, season=SEASON)
    assert float(got.consensus.isel(lat=0, lon=0)) == pytest.approx(600 + 6 * 20)


def test_consensus_accepts_mean_a_quantile_or_a_callable(observed, climatology, analogs):
    kwargs = dict(climatology=climatology, season=SEASON)
    mean = complete(observed, analogs, reduce="mean", **kwargs)
    assert float(mean.consensus.isel(lat=0, lon=0)) == pytest.approx(600 + 6 * 20)

    low = complete(observed, analogs, reduce=0.0, **kwargs)
    assert float(low.consensus.isel(lat=0, lon=0)) == pytest.approx(600 + 6 * 10)

    biggest = complete(observed, analogs, reduce=lambda t, dim: t.max(dim), **kwargs)
    assert float(biggest.consensus.isel(lat=0, lon=0)) == pytest.approx(600 + 6 * 30)


def test_a_bad_reduce_is_rejected(observed, climatology, analogs):
    with pytest.raises(ValueError, match="reduce must be"):
        complete(observed, analogs, climatology=climatology, season=SEASON, reduce="mode")
    with pytest.raises(ValueError, match="reduce must be"):
        complete(observed, analogs, climatology=climatology, season=SEASON, reduce=1.5)


def _index_series():
    return xr.DataArray(
        (YEARS[:-1] - 1980).astype(float), dims="year", coords={"year": YEARS[:-1]}
    )


def test_uniform_weights_reproduce_the_unweighted_consensus(observed, climatology):
    """The weighted-quantile path must agree with the plain quantile when the
    weights are equal, or the two code paths have silently diverged."""
    close = analogs_from_index(_index_series(), target=30.0, n=3)
    kwargs = dict(climatology=climatology, season=SEASON)
    plain = complete(observed, close, **kwargs)
    uniform = complete(observed, close, weights="uniform", **kwargs)
    np.testing.assert_allclose(plain.consensus.values, uniform.consensus.values)


def test_weighted_consensus_moves_toward_the_closest_analog(observed, climatology):
    """Analogs at distances 0, 1 and 5 from the target. The unweighted median is
    the middle one; weighting must pull the consensus toward the best analog.

    (Symmetric analogs would be a vacuous test: a median cannot move when the
    members are balanced about it, however they are weighted.)"""
    close = analogs_from_index(
        _index_series(), target=30.0, n=3, candidates=[2010, 2011, 2015]
    )
    kwargs = dict(climatology=climatology, season=SEASON)
    best_total = 600 + 6 * 30  # the 2010 scenario

    plain = float(complete(observed, close, **kwargs).consensus.isel(lat=0, lon=0))
    weighted = float(
        complete(observed, close, weights="inverse_distance", **kwargs)
        .consensus.isel(lat=0, lon=0)
    )
    assert abs(weighted - best_total) < abs(plain - best_total)


def test_weighted_quantile_never_leaves_the_data_range():
    """A weighted quantile at an extreme q must clamp to the outermost scenario,
    not extrapolate past it. Regression for the upper-tail extrapolation bug
    (q=0.99 on [10,20,100]/[.6,.3,.1] previously returned ~116 > 100)."""
    from deepscale.completion import _weighted_quantile

    vals = np.array([10.0, 20.0, 100.0])
    w = np.array([0.6, 0.3, 0.1])
    for q in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0):
        out = float(_weighted_quantile(vals, w, q, axis=0))
        assert vals.min() <= out <= vals.max(), (q, out)


def test_weighted_high_quantile_consensus_stays_within_scenarios(observed, climatology):
    """The end-to-end path: a high-quantile weighted consensus must lie within
    the span of the analog scenario totals it summarises."""
    close = analogs_from_index(
        _index_series(), target=30.0, n=3, candidates=[2010, 2011, 2015]
    )
    res = complete(observed, close, climatology=climatology, season=SEASON,
                   weights="inverse_distance", reduce=0.95)
    hi = res.totals.max("scenario")
    lo = res.totals.min("scenario")
    assert bool((res.consensus <= hi + 1e-9).all())
    assert bool((res.consensus >= lo - 1e-9).all())


def test_weighted_mean_matches_a_hand_computation(observed, climatology):
    close = analogs_from_index(_index_series(), target=30.0, n=3)
    weights = close.weights("inverse_distance")
    got = complete(observed, close, climatology=climatology, season=SEASON,
                   reduce="mean", weights=weights)

    totals = xr.DataArray(
        [600 + 6 * (y - 1980) for y in close.years], dims="scenario",
        coords={"scenario": close.years},
    )
    expected = float((totals * weights.rename(year="scenario")).sum() / weights.sum())
    assert float(got.consensus.isel(lat=0, lon=0)) == pytest.approx(expected)


def test_percentile_places_the_consensus_in_the_historical_record(observed, climatology, analogs):
    """The consensus total (720) exceeds every historical season total (max
    12*45=540), so it must sit at the very top of the record."""
    got = complete(observed, analogs, climatology=climatology, season=SEASON)
    assert float(got.percentile.isel(lat=0, lon=0)) == pytest.approx(1.0)


def test_percentile_of_a_dry_projection_lands_near_the_bottom(climatology, analogs):
    dry_stamps = pd.DatetimeIndex(
        [pd.Timestamp(TARGET, m, d) for m in (6, 7) for d in (1, 11, 21)]
    )
    bone_dry = xr.DataArray(
        np.zeros((6, 2, 2)), dims=("time", "lat", "lon"),
        coords={"time": dry_stamps, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    got = complete(bone_dry, analogs_from_years([1981]), climatology=climatology,
                   season=SEASON)
    # Total is 0 + 6*1 = 6 mm, drier than every year on record.
    assert float(got.percentile.isel(lat=0, lon=0)) == pytest.approx(0.0)


def test_percentile_reference_can_be_supplied_directly(observed, climatology, analogs):
    inflated = xr.DataArray(
        np.full((45, 2, 2), 10_000.0), dims=("year", "lat", "lon"),
        coords={"year": np.arange(1981, 2026), "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    got = complete(observed, analogs, climatology=climatology, season=SEASON,
                   percentile_reference=inflated)
    assert float(got.percentile.isel(lat=0, lon=0)) == pytest.approx(0.0)


# --- dimension agnosticism -------------------------------------------------


def test_completion_works_on_admin_units_not_just_a_grid(archive, analogs):
    """The engine reduces along `step` and touches nothing else, so a zonal
    (time, region) array needs no separate code path."""
    zonal = archive.isel(lat=0).rename(lon="region").assign_coords(
        region=["sebeta", "adama"]
    )
    clim = seasonal_stack(zonal, SEASON, years=range(1981, 2026))
    stamps = pd.DatetimeIndex(
        [pd.Timestamp(TARGET, m, d) for m in (6, 7) for d in (1, 11, 21)]
    )
    obs = xr.DataArray(
        np.full((6, 2), 100.0), dims=("time", "region"),
        coords={"time": stamps, "region": ["sebeta", "adama"]},
    )
    got = complete(obs, analogs, climatology=clim, season=SEASON)
    assert got.totals.dims == ("scenario", "region")
    assert got.percentile.dims == ("region",)
    assert float(got.consensus.sel(region="sebeta")) == pytest.approx(720.0)


def test_completion_works_on_a_bare_series(archive, analogs):
    series = archive.isel(lat=0, lon=0, drop=True)
    clim = seasonal_stack(series, SEASON, years=range(1981, 2026))
    stamps = pd.DatetimeIndex(
        [pd.Timestamp(TARGET, m, d) for m in (6, 7) for d in (1, 11, 21)]
    )
    obs = xr.DataArray(np.full(6, 100.0), dims="time", coords={"time": stamps})
    got = complete(obs, analogs, climatology=clim, season=SEASON)
    assert got.consensus.dims == ()
    assert float(got.consensus) == pytest.approx(720.0)


def test_completion_works_on_a_monthly_cadence(analogs):
    stamps = pd.DatetimeIndex(
        [pd.Timestamp(y, m, 1) for y in YEARS for m in range(1, 13)]
    )
    values = np.array([[[s.year - 1980]] for s in stamps], dtype=float)
    monthly = xr.DataArray(
        values, dims=("time", "lat", "lon"),
        coords={"time": stamps, "lat": [0.0], "lon": [0.0]},
    )
    clim = seasonal_stack(monthly, SEASON, years=range(1981, 2026))
    assert clim.sizes["step"] == 4  # J, J, A, S

    obs = monthly.sel(time=[f"{TARGET}-06-01", f"{TARGET}-07-01"])
    got = complete(obs, analogs, climatology=clim, season=SEASON)
    # 2 observed months at 46, plus 2 analog months at (year - 1980).
    assert float(got.totals.sel(scenario=2000).isel(lat=0, lon=0)) == pytest.approx(2 * 46 + 2 * 20)


# --- accumulation curves ---------------------------------------------------


def test_accumulation_is_the_running_total_and_agrees_with_totals(observed, climatology, analogs):
    got = complete(observed, analogs, climatology=climatology, season=SEASON)
    curves = got.accumulation()
    assert curves.dims == got.scenarios.dims
    np.testing.assert_allclose(
        curves.isel(step=-1).values, got.totals.values
    )
    # Observed segment: 100 mm per dekad, identical across scenarios.
    np.testing.assert_allclose(
        curves.isel(step=5, lat=0, lon=0).values, 600.0
    )


def test_accumulation_carries_the_segment_labels_and_target_year_stamps(
    observed, climatology, analogs
):
    got = complete(observed, analogs, climatology=climatology, season=SEASON)
    curves = got.accumulation()
    assert "segment" in curves.coords
    assert str(curves.time.values[0])[:10] == f"{TARGET}-06-01"


# --- validation ------------------------------------------------------------


def test_analog_years_absent_from_the_source_are_rejected(observed, climatology):
    with pytest.raises(ValueError, match="absent from the analog source"):
        complete(observed, analogs_from_years([1799]), climatology=climatology,
                 season=SEASON)


def test_climatology_must_be_season_stacked(observed, archive, analogs):
    with pytest.raises(ValueError, match="must have 'year' and 'step' dims"):
        complete(observed, analogs, climatology=archive, season=SEASON)


def test_observations_outside_the_season_are_rejected(climatology, analogs):
    stamps = pd.DatetimeIndex([pd.Timestamp(TARGET, 2, 1), pd.Timestamp(TARGET, 2, 11)])
    winter = xr.DataArray(
        np.ones((2, 2, 2)), dims=("time", "lat", "lon"),
        coords={"time": stamps, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    with pytest.raises(ValueError, match="fall inside season"):
        complete(winter, analogs, climatology=climatology, season=SEASON)


def test_analog_source_may_differ_from_the_percentile_climatology(
    observed, climatology, analogs
):
    """A bias-corrected archive can supply the remainders while the percentile
    is still taken against the raw record."""
    shifted = climatology + 50.0
    got = complete(observed, analogs, climatology=climatology, season=SEASON,
                   analog_source=shifted)
    assert float(got.totals.sel(scenario=1990).isel(lat=0, lon=0)) == pytest.approx(
        600 + 6 * (10 + 50)
    )


def test_metadata_records_the_segment_budget(observed, forecast, climatology, analogs):
    got = complete(observed, analogs, climatology=climatology, season=SEASON,
                   forecast=forecast)
    assert got.metadata["n_observed_steps"] == 6
    assert got.metadata["n_forecast_steps"] == 2
    assert got.metadata["n_analog_steps"] == 4
    assert got.metadata["n_scenarios"] == 3
    assert got.metadata["cadence"] == "dekad"


def test_result_keeps_the_analog_set_that_produced_it(observed, climatology, analogs):
    got = complete(observed, analogs, climatology=climatology, season=SEASON)
    assert list(got.analogs.years) == [1990, 2000, 2010]
    assert got.analogs.metadata["selector"] == "years"


def test_missing_observation_propagates_to_a_nan_total(observed, climatology, analogs):
    holed = observed.copy()
    holed[0, 0, 0] = np.nan
    got = complete(holed, analogs, climatology=climatology, season=SEASON)
    assert bool(got.totals.isel(lat=0, lon=0).isnull().all())
    assert np.isnan(float(got.consensus.isel(lat=0, lon=0)))
    assert float(got.totals.isel(lat=1, lon=1).sel(scenario=1990)) == pytest.approx(660.0)


def test_min_count_allows_a_partial_accumulation(observed, climatology, analogs):
    holed = observed.copy()
    holed[0, 0, 0] = np.nan
    got = complete(holed, analogs, climatology=climatology, season=SEASON, min_count=1)
    assert float(got.totals.isel(lat=0, lon=0).sel(scenario=1990)) == pytest.approx(
        500 + 6 * 10
    )
