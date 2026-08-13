"""The shared kernel: rolling totals and consecutive-run lengths."""
import numpy as np
import xarray as xr

from deepscale.aggregations._runs import dry_run_lengths, has_data, rolling_total


def _steps(values):
    return xr.DataArray(np.asarray(values, dtype=float), dims="step")


def test_rolling_total_sums_window_starting_at_each_position():
    out = rolling_total(_steps([1.0, 2.0, 3.0, 4.0]), 3)
    assert out.values[0] == 6.0        # 1 + 2 + 3
    assert out.values[1] == 9.0        # 2 + 3 + 4
    assert np.isnan(out.values[2])     # window runs off the end
    assert np.isnan(out.values[3])


def test_rolling_total_window_of_one_is_the_series():
    out = rolling_total(_steps([1.0, 2.0, 3.0]), 1)
    assert out.values.tolist() == [1.0, 2.0, 3.0]


def test_dry_run_lengths_counts_the_run_ending_at_each_step():
    #                        dry  dry  wet  dry  dry  dry
    out = dry_run_lengths(_steps([0.0, 0.5, 5.0, 0.0, 0.0, 0.0]), 1.0)
    assert out.values.tolist() == [1, 2, 0, 1, 2, 3]


def test_dry_run_lengths_threshold_is_inclusive():
    """The criterion is 'at or below 1 mm', not 'below 1 mm'."""
    out = dry_run_lengths(_steps([1.0, 1.0]), 1.0)
    assert out.values.tolist() == [1, 2]


def test_dry_run_lengths_treats_nan_as_not_dry():
    """A gap in the record must break a run, never manufacture one."""
    out = dry_run_lengths(_steps([0.0, np.nan, 0.0]), 1.0)
    assert out.values.tolist() == [1, 0, 1]


def test_dry_run_lengths_preserves_other_dims():
    rain = xr.DataArray(np.zeros((2, 3, 4)), dims=("year", "lat", "step"))
    out = dry_run_lengths(rain, 1.0)
    assert out.dims == ("year", "lat", "step")
    assert out.isel(year=0, lat=0).values.tolist() == [1, 2, 3, 4]


def test_has_data_distinguishes_empty_cells():
    rain = xr.DataArray([[1.0, 2.0], [np.nan, np.nan]], dims=("cell", "step"))
    assert has_data(rain).values.tolist() == [True, False]
