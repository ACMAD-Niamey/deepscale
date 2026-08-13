"""Dry-spell statistics within a season."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from deepscale.aggregations import dry_spell


def _daily(values, start="2015-03-01"):
    time = pd.date_range(start, periods=len(values), freq="D")
    return xr.DataArray(np.asarray(values, dtype=float), dims="time",
                        coords={"time": time})


def test_longest_run_and_count_of_qualifying_runs():
    """Two qualifying runs (8 days and 7 days) and one that falls short (5)."""
    rain = np.full(92, 15.0)
    rain[10:18] = 0.0     # 8 dry days
    rain[30:35] = 0.0     # 5 dry days, below the 7-day threshold
    rain[50:57] = 0.0     # 7 dry days
    result = dry_spell(_daily(rain), "MAM")
    assert float(result.longest.sel(year=2015)) == 8.0
    assert float(result.count.sel(year=2015)) == 2.0


def test_a_long_run_counts_once_not_once_per_qualifying_day():
    rain = np.full(92, 15.0)
    rain[10:30] = 0.0     # a single 20-day run
    result = dry_spell(_daily(rain), "MAM")
    assert float(result.longest.sel(year=2015)) == 20.0
    assert float(result.count.sel(year=2015)) == 1.0


def test_no_dry_days_gives_zero_not_nan():
    result = dry_spell(_daily(np.full(92, 15.0)), "MAM")
    assert float(result.longest.sel(year=2015)) == 0.0
    assert float(result.count.sel(year=2015)) == 0.0


def test_cell_with_no_data_is_nan():
    result = dry_spell(_daily(np.full(92, np.nan)), "MAM")
    assert np.isnan(float(result.longest.sel(year=2015)))
    assert np.isnan(float(result.count.sel(year=2015)))


def test_provenance_is_attached():
    result = dry_spell(_daily(np.full(92, 15.0)), "MAM")
    assert result.longest.attrs["dry_spell_days"] == 7
    assert result.longest.attrs["season"] == "MAM"


def test_non_daily_input_raises_rather_than_reporting_no_dry_spells():
    """`obs/chirps-v2-dekadal-rhiza` is a real catalog product, and a dry-spell
    length counted in days has no meaning on a dekadal axis."""
    stamps = pd.to_datetime([f"2015-{m:02d}-{d:02d}" for m in (3, 4, 5)
                             for d in (1, 11, 21)])
    dekadal = xr.DataArray(np.full(len(stamps), 30.0), dims="time",
                           coords={"time": stamps})
    with pytest.raises(ValueError, match="needs daily rainfall"):
        dry_spell(dekadal, "MAM")


def test_short_record_warns_that_runs_are_cut():
    rain = np.full(120, 15.0)
    rain[40:] = 0.0
    with pytest.warns(RuntimeWarning, match="full season"):
        dry_spell(_daily(rain[:60]), "MAM")


def test_short_record_under_reports_the_longest_run():
    """The warning above exists because the numbers stay finite and wrong.
    Identical rainfall, two record lengths, two different answers."""
    rain = np.full(120, 15.0)
    rain[40:] = 0.0
    full = dry_spell(_daily(rain), "MAM")
    with pytest.warns(RuntimeWarning):
        short = dry_spell(_daily(rain[:60]), "MAM")
    assert float(full.longest.sel(year=2015)) == 52.0
    assert float(short.longest.sel(year=2015)) == 20.0
