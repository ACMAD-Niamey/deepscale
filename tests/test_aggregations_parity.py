"""Parity with the reference onset implementation.

Oracle transcribed from ~/seasonal-hindcasting-gha/scripts/onset_and_exceedance.py
(`compute_onset`, lines 108-129), with INIT_DOY set to 0 and the search bounds
set to the array so it returns a plain step index. Everything else, including
the loop bound that makes late triggers unverifiable, is preserved verbatim.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from deepscale.aggregations import onset

ACCUM_DAYS = 3
ACCUM_THRESHOLD = 20
DRY_SPELL_DAYS = 7
DRY_SPELL_THRESHOLD = 1.0
DRY_SPELL_WINDOW = 21


def reference_compute_onset(ts, search_end):
    """The original per-cell scan, verbatim apart from the origin."""
    n = len(ts)
    for i in range(n - ACCUM_DAYS - DRY_SPELL_WINDOW):
        if i > search_end:
            break
        if np.sum(ts[i:i + ACCUM_DAYS]) < ACCUM_THRESHOLD:
            continue
        false_start = False
        if i + ACCUM_DAYS + DRY_SPELL_WINDOW <= n:
            post = ts[i + ACCUM_DAYS:i + ACCUM_DAYS + DRY_SPELL_WINDOW]
            dry_count = 0
            for p in post:
                if p < DRY_SPELL_THRESHOLD:
                    dry_count += 1
                    if dry_count >= DRY_SPELL_DAYS:
                        false_start = True
                        break
                else:
                    dry_count = 0
        if not false_start:
            return i
    return np.nan


@pytest.fixture
def random_daily_grid():
    """Intermittent rainfall over a small grid: many cells with a real onset,
    many with false starts, some with none at all."""
    rng = np.random.default_rng(20260813)
    n_days, n_cells = 130, 60
    wet = rng.random((n_days, n_cells)) < 0.35
    amounts = rng.gamma(shape=1.4, scale=7.0, size=(n_days, n_cells))
    return np.where(wet, amounts, 0.0)


def test_vectorised_onset_matches_the_reference_scan(random_daily_grid):
    values = random_daily_grid
    time = pd.date_range("2015-03-01", periods=values.shape[0], freq="D")
    daily = xr.DataArray(values, dims=("time", "cell"),
                         coords={"time": time, "cell": np.arange(values.shape[1])})

    mam_steps = 92  # Mar 31 + Apr 30 + May 31
    ours = onset(daily, "MAM").step.sel(year=2015).values

    theirs = np.array([
        reference_compute_onset(values[:, c], search_end=mam_steps - 1)
        for c in range(values.shape[1])
    ], dtype=float)

    np.testing.assert_array_equal(
        np.isnan(ours), np.isnan(theirs),
        err_msg="disagreement on which cells have an onset at all")
    finite = ~np.isnan(theirs)
    np.testing.assert_array_equal(
        ours[finite], theirs[finite],
        err_msg="disagreement on the onset step")


def test_at_least_some_cells_found_onset_and_some_did_not(random_daily_grid):
    """Guards against the parity test passing because everything is NaN."""
    values = random_daily_grid
    time = pd.date_range("2015-03-01", periods=values.shape[0], freq="D")
    daily = xr.DataArray(values, dims=("time", "cell"),
                         coords={"time": time, "cell": np.arange(values.shape[1])})
    occurred = onset(daily, "MAM").occurred.sel(year=2015).values
    assert (occurred == 1.0).sum() >= 5
    assert (occurred == 0.0).sum() >= 1


def test_threshold_is_inclusive_unlike_the_reference_script():
    """The reference script uses `< 1.0` (line 123); the stated criterion is
    'at or below 1 mm/day'. deepscale implements the criterion. A cell whose
    dry days are all exactly 1.0 mm is a dry spell here and was not there."""
    rain = np.full(120, 5.0)
    rain[0:3] = [1.0, 12.0, 9.0]
    rain[3:24] = 1.0     # exactly at the threshold, for 21 days
    time = pd.date_range("2015-03-01", periods=len(rain), freq="D")
    daily = xr.DataArray(rain, dims="time", coords={"time": time})
    result = onset(daily, "MAM")
    assert float(result.step.sel(year=2015)) != 0.0
