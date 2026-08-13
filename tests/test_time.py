"""Calendar arithmetic and season-step alignment."""
from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from deepscale.time import (
    dekad_of_year,
    dekad_start_for,
    dekad_window,
    dekads_for_issuance,
    infer_cadence,
    pentad_start_for,
    pentad_window,
    season_bounds,
    season_step,
)


def _time(stamps):
    idx = pd.DatetimeIndex(stamps)
    return xr.DataArray(idx.values, dims="time", coords={"time": idx.values})


def _dekad_starts(year, months):
    return [date(year, m, d) for m in months for d in (1, 11, 21)]


# --- dekads ----------------------------------------------------------------


@pytest.mark.parametrize(
    "day, expected", [(1, 1), (10, 1), (11, 11), (20, 11), (21, 21), (31, 21)]
)
def test_dekad_start_for_partitions_the_month_at_days_1_11_21(day, expected):
    assert dekad_start_for(date(2026, 7, day)).day == expected


def test_third_dekad_window_absorbs_the_variable_month_end():
    # February 2024 is a leap month: its third dekad is 9 days, not 10.
    _, end = dekad_window(date(2024, 2, 21))
    assert end == date(2024, 3, 1)
    _, end_non_leap = dekad_window(date(2023, 2, 21))
    assert end_non_leap == date(2023, 3, 1)


def test_dekad_of_year_is_1_to_36_and_monotonic_across_the_year():
    starts = _dekad_starts(2026, range(1, 13))
    ordinals = [dekad_of_year(d) for d in starts]
    assert ordinals == list(range(1, 37))


def test_dekads_for_issuance_covers_every_dekad_starting_inside_the_lead_window():
    # A 15-day forecast issued 5 July reaches into the dekad starting 11 July
    # and the one starting 21 July (day 16 is outside, day 21 is not).
    got = dekads_for_issuance(date(2026, 7, 5), (0, 15))
    assert got == [date(2026, 7, 1), date(2026, 7, 11)]
    got_longer = dekads_for_issuance(date(2026, 7, 5), (0, 30))
    assert got_longer == [date(2026, 7, 1), date(2026, 7, 11), date(2026, 7, 21),
                          date(2026, 8, 1)]


# --- pentads ---------------------------------------------------------------


@pytest.mark.parametrize(
    "day, expected", [(1, 1), (5, 1), (6, 6), (25, 21), (26, 26), (31, 26)]
)
def test_pentad_start_for_gives_six_pentads_per_month(day, expected):
    assert pentad_start_for(date(2026, 1, day)).day == expected


def test_sixth_pentad_absorbs_the_month_remainder():
    _, end = pentad_window(date(2026, 1, 26))  # 31-day month -> 6-day pentad
    assert end == date(2026, 2, 1)
    _, end_feb = pentad_window(date(2026, 2, 26))  # 28-day month -> 3-day pentad
    assert end_feb == date(2026, 3, 1)


# --- cadence inference -----------------------------------------------------


@pytest.mark.parametrize(
    "stamps, expected",
    [
        (pd.date_range("2026-01-01", periods=40, freq="D"), "daily"),
        (pd.date_range("2026-01-01", periods=12, freq="MS"), "monthly"),
    ],
)
def test_infer_cadence_recognizes_regular_axes(stamps, expected):
    assert infer_cadence(_time(stamps)) == expected


def test_infer_cadence_recognizes_dekads_despite_their_uneven_spacing():
    starts = pd.DatetimeIndex(_dekad_starts(2026, range(1, 13)))
    assert infer_cadence(_time(starts)) == "dekad"


def test_infer_cadence_recognizes_pentads():
    starts = pd.DatetimeIndex(
        [date(2026, m, d) for m in range(1, 13) for d in (1, 6, 11, 16, 21, 26)]
    )
    assert infer_cadence(_time(starts)) == "pentad"


def test_infer_cadence_refuses_a_single_stamp():
    with pytest.raises(ValueError, match="at least two time stamps"):
        infer_cadence(_time(pd.DatetimeIndex(["2026-07-01"])))


def test_infer_cadence_refuses_an_annual_axis():
    with pytest.raises(ValueError, match="matches no known cadence"):
        infer_cadence(_time(pd.date_range("2000-01-01", periods=5, freq="YS")))


# --- season bounds ---------------------------------------------------------


def test_season_bounds_from_month_initial_code():
    start, end = season_bounds("JJAS", 2026)
    assert (start, end) == (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-09-30"))


def test_season_bounds_wraparound_code_ends_in_the_following_year():
    start, end = season_bounds("NDJ", 2026)
    assert (start, end) == (pd.Timestamp("2026-11-01"), pd.Timestamp("2027-01-31"))


def test_season_bounds_month_pair_matches_the_code_form():
    assert season_bounds((6, 9), 2026) == season_bounds("JJAS", 2026)


def test_season_bounds_passes_through_explicit_timestamps():
    start, end = season_bounds(("2026-06-11", "2026-09-20"), 2026)
    assert (start, end) == (pd.Timestamp("2026-06-11"), pd.Timestamp("2026-09-20"))


def test_season_bounds_rejects_an_ambiguous_code():
    # A bare "M" spells both March and May. Multi-letter codes disambiguate by
    # contiguity ("MAM" can only be March-May), but a single letter cannot.
    with pytest.raises(ValueError, match="resolves to 2 month runs"):
        season_bounds("M", 2026)


def test_season_bounds_disambiguates_repeated_initials_by_contiguity():
    # "JA" could start at any J-month by first letter alone; only July works.
    assert season_bounds("JA", 2026)[0] == pd.Timestamp("2026-07-01")
    # "JJAS" likewise: month 6, not month 1 or 7.
    assert season_bounds("JJAS", 2026)[0] == pd.Timestamp("2026-06-01")


def test_season_bounds_respects_february_length_in_a_leap_year():
    _, end = season_bounds("DJF", 2023)
    assert end == pd.Timestamp("2024-02-29")


# --- season steps ----------------------------------------------------------


def test_season_step_numbers_dekads_from_zero_and_marks_outsiders():
    starts = pd.DatetimeIndex(_dekad_starts(2026, range(1, 13)))
    steps = season_step(_time(starts), "JJAS", year=2026)
    inside = steps.values[steps.values >= 0]
    assert list(inside) == list(range(12))  # 4 months x 3 dekads
    assert (steps.values[:15] == -1).all()  # Jan-May are outside


def test_season_step_aligns_the_same_season_across_years_including_leap_years():
    """The k-th dekad of JJAS gets step k in every year. This is the property
    the completion engine relies on to splice one year onto another."""
    per_year = {}
    for year in (1997, 2015, 2024, 2026):  # 2024 is a leap year
        starts = pd.DatetimeIndex(_dekad_starts(year, range(6, 10)))
        per_year[year] = season_step(_time(starts), "JJAS", year=year).values
    reference = per_year[1997]
    for year, steps in per_year.items():
        np.testing.assert_array_equal(steps, reference, err_msg=f"year {year}")
    np.testing.assert_array_equal(reference, np.arange(12))


def test_season_step_daily_alignment_is_leap_safe():
    """Daily steps are elapsed days from the season start, so 1 July is step 30
    of a JJAS season in a leap year exactly as in a common year -- which a
    day-of-year difference would get wrong."""
    for year in (2023, 2024):  # 2024 is a leap year
        stamps = pd.date_range(f"{year}-06-01", f"{year}-09-30", freq="D")
        steps = season_step(_time(stamps), "JJAS", year=year)
        july_1 = steps.sel(time=f"{year}-07-01").item()
        assert july_1 == 30, f"{year}: expected step 30, got {july_1}"


def test_season_step_stays_monotonic_across_a_wraparound_season():
    starts = pd.DatetimeIndex(_dekad_starts(2026, [10, 11, 12]) + _dekad_starts(2027, [1]))
    steps = season_step(_time(starts), "OND", year=2026)
    inside = steps.values[steps.values >= 0]
    assert list(inside) == list(range(9))
    assert (steps.values[-3:] == -1).all()  # January is outside OND


def test_season_step_handles_a_season_starting_mid_month():
    starts = pd.DatetimeIndex(_dekad_starts(2026, [6, 7]))
    steps = season_step(_time(starts), ("2026-06-11", "2026-07-31"), year=2026)
    assert steps.values[0] == -1  # 1 June precedes the window
    assert list(steps.values[1:]) == [0, 1, 2, 3, 4]


def test_season_step_infers_year_from_the_first_stamp():
    starts = pd.DatetimeIndex(_dekad_starts(2015, range(6, 10)))
    explicit = season_step(_time(starts), "JJAS", year=2015)
    inferred = season_step(_time(starts), "JJAS")
    np.testing.assert_array_equal(explicit.values, inferred.values)


def test_season_step_tail_days_admits_post_season_stamps():
    """MAM 2015: Mar 1 is step 0, May 31 is step 91, Jun 1 is step 92."""
    time = xr.DataArray(pd.date_range("2015-05-30", "2015-06-05", freq="D"), dims="time")
    plain = season_step(time, "MAM", year=2015)
    tailed = season_step(time, "MAM", year=2015, tail_days=10)
    assert plain.values.tolist() == [90, 91, -1, -1, -1, -1, -1]
    assert tailed.values.tolist() == [90, 91, 92, 93, 94, 95, 96]


def test_season_step_tail_days_default_is_unchanged():
    time = xr.DataArray(pd.date_range("2015-05-30", "2015-06-05", freq="D"), dims="time")
    assert season_step(time, "MAM", year=2015).values.tolist() == \
        season_step(time, "MAM", year=2015, tail_days=0).values.tolist()


# --- season months ---------------------------------------------------------


def test_season_months_from_a_code():
    from deepscale.time import season_months
    assert season_months("JJAS") == [6, 7, 8, 9]
    assert season_months("MAM") == [3, 4, 5]


def test_season_months_wraps_the_year_boundary():
    from deepscale.time import season_months
    assert season_months("NDJ") == [11, 12, 1]
    assert season_months("OND") == [10, 11, 12]


def test_season_months_from_a_month_pair_matches_the_code_form():
    from deepscale.time import season_months
    assert season_months((6, 9)) == season_months("JJAS")
    assert season_months((11, 1)) == season_months("NDJ")


def test_season_months_of_a_single_month():
    from deepscale.time import season_months
    assert season_months((7, 7)) == [7]


def test_season_months_rejects_a_timestamp_pair():
    from deepscale.time import season_months
    with pytest.raises(TypeError, match="month-initial code"):
        season_months(("2026-06-01", "2026-09-30"))
