"""Onset, cessation and season length."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from deepscale.aggregations import cessation, onset, season_length


def _daily(values, start="2015-03-01"):
    """A single-cell daily series beginning on `start`."""
    time = pd.date_range(start, periods=len(values), freq="D")
    return xr.DataArray(np.asarray(values, dtype=float), dims="time",
                        coords={"time": time})


def _mam_2015_with_false_start():
    """22 mm falls at step 2, then a 14-day drought. The real onset is step 20.

    Steps 2-4 total 22 mm and clear the trigger, but steps 5-19 are dry, so the
    21-day guard window catches a 7-day dry run and rejects it. Steps 20-22
    total 21 mm and are followed by sustained rain.

    The second burst is flat 7 mm rather than a spike. A spike would make the
    window *starting one step earlier* clear 20 mm too (0 + 8 + 15 = 23), and
    onset is the first qualifying window, so the answer would be step 19.
    """
    rain = np.zeros(120)
    rain[2:5] = [1.0, 12.0, 9.0]
    rain[20:23] = 7.0
    rain[23:60] = 5.0
    return rain


def _no_onset_series(n=120):
    """A series with no qualifying trigger.

    One 15 mm day keeps the units check quiet without clearing the 20 mm
    trigger, so these tests assert on onset rather than on a unit warning.
    """
    rain = np.zeros(n)
    rain[0] = 15.0
    return rain


def test_onset_rejects_a_false_start_and_finds_the_later_onset():
    result = onset(_daily(_mam_2015_with_false_start()), "MAM")
    assert float(result.step.sel(year=2015)) == 20.0
    assert float(result.occurred.sel(year=2015)) == 1.0
    assert str(result.date.sel(year=2015).values)[:10] == "2015-03-21"


def test_onset_without_the_guard_would_have_fired_early():
    """Same series, guard disabled: the false start is accepted at step 2."""
    result = onset(_daily(_mam_2015_with_false_start()), "MAM", guard_days=0)
    assert float(result.step.sel(year=2015)) == 2.0


def test_no_qualifying_trigger_reports_occurred_zero_not_nan():
    result = onset(_daily(_no_onset_series()), "MAM")
    assert np.isnan(float(result.step.sel(year=2015)))
    assert float(result.occurred.sel(year=2015)) == 0.0


def test_cell_with_no_data_reports_occurred_nan():
    """The three-state contract: absent data must not read as a failed season."""
    result = onset(_daily(np.full(120, np.nan)), "MAM")
    assert np.isnan(float(result.step.sel(year=2015)))
    assert np.isnan(float(result.occurred.sel(year=2015)))


def test_search_start_excludes_earlier_triggers():
    """Two clean onsets, at steps 2 and 40. The 2 mm background keeps both
    guards satisfied, and is below the 20 mm trigger on any 3-day window."""
    rain = np.zeros(120)
    rain[5:75] = 2.0
    rain[2:5] = [1.0, 12.0, 9.0]     # 22 mm
    rain[40:43] = 7.0                # 21 mm
    early = onset(_daily(rain), "MAM")
    late = onset(_daily(rain), "MAM", search_start=10)
    assert float(early.step.sel(year=2015)) == 2.0
    assert float(late.step.sel(year=2015)) == 40.0


def test_short_record_warns_about_the_unverifiable_guard():
    """Data stopping at the season end cannot confirm any late trigger."""
    with pytest.warns(RuntimeWarning, match="guard window"):
        onset(_daily(_no_onset_series(92)), "MAM")


def test_record_starting_mid_season_warns_rather_than_dating_its_first_day():
    """Data beginning 15 April reports step 45 with no complaint unless the
    completeness check looks at the season start as well as its end."""
    rain = np.zeros(100)
    rain[0:3] = [1.0, 12.0, 9.0]        # 22 mm on the first day available
    rain[3:] = 5.0
    with pytest.warns(RuntimeWarning, match="after the start of season"):
        result = onset(_daily(rain, start="2015-04-15"), "MAM")
    assert float(result.step.sel(year=2015)) == 45.0


def test_a_season_with_no_in_season_data_is_nan_not_a_failed_season():
    """The record begins after MAM 2015 ended. Only the guard tail overlaps it,
    and tail data is not evidence that the season was observed."""
    rain = np.zeros(400)
    rain[0] = 15.0
    daily = _daily(rain, start="2015-06-05")
    with pytest.warns(RuntimeWarning):
        result = onset(daily, "MAM")
    assert np.isnan(float(result.occurred.sel(year=2015)))
    assert float(result.occurred.sel(year=2016)) == 0.0


def test_onset_and_dry_spell_agree_on_which_years_were_observed():
    """Same record, same seasons. `dry_spell` drops a year with no in-season
    data; `onset` keeps the row but must report it as absent, not as failed."""
    from deepscale.aggregations import dry_spell

    rain = np.zeros(400)
    rain[0] = 15.0
    daily = _daily(rain, start="2015-06-05")
    with pytest.warns(RuntimeWarning):
        on = onset(daily, "MAM")
    spells = dry_spell(daily, "MAM")

    observed = {int(y) for y in on.occurred["year"].values
                if bool(on.occurred.sel(year=y).notnull())}
    assert observed == {int(y) for y in spells.longest["year"].values}


def test_interior_gap_warns_rather_than_reading_as_a_failed_season():
    """One missing day inside the guard window flips onset to NaN, which is the
    right call. Reporting it silently as `occurred=0.0` is not."""
    rain = np.zeros(120)
    rain[10:13] = [1.0, 12.0, 9.0]
    rain[13:80] = 5.0
    complete = _daily(rain)
    assert float(onset(complete, "MAM").step.sel(year=2015)) == 10.0

    gapped = complete.drop_isel(time=20)      # 2015-03-21, inside the guard
    with pytest.warns(RuntimeWarning, match="interior gaps"):
        result = onset(gapped, "MAM")
    assert np.isnan(float(result.step.sel(year=2015)))


def _dekadal(months=(3, 4, 5, 6)):
    """A dekadal series: stamps on the 1st, 11th and 21st of each month."""
    stamps = pd.to_datetime([f"2015-{m:02d}-{d:02d}" for m in months
                             for d in (1, 11, 21)])
    return xr.DataArray(np.full(len(stamps), 30.0), dims="time",
                        coords={"time": stamps})


@pytest.mark.parametrize("call", [
    lambda da: onset(da, "MAM"),
    lambda da: cessation(da, "MAM", after=0),
])
def test_non_daily_input_raises_rather_than_reporting_no_event(call):
    """Dekadal rainfall is a real catalog product and every criterion here is
    counted in days, so there is no answer to warn about."""
    with pytest.raises(ValueError, match="needs daily rainfall"):
        call(_dekadal())


def _late_trigger_series(n):
    """22 mm at steps 89-91, the last three days of MAM 2015, and nothing else.

    ``n`` days from 1 March 2015: 92 is the season alone, 116 adds the full
    24-day tail the guard needs.
    """
    rain = np.zeros(n)
    rain[89:92] = [1.0, 12.0, 9.0]
    return rain


def test_a_trigger_whose_guard_window_runs_off_the_record_is_rejected():
    """The guard fails closed. Steps 89-91 clear 20 mm, but verifying them
    needs steps 92-112, which a season-only record does not have, so the
    trigger is unverifiable and reported as no onset rather than accepted."""
    with pytest.warns(RuntimeWarning, match="guard window"):
        result = onset(_daily(_late_trigger_series(92)), "MAM")
    assert np.isnan(float(result.step.sel(year=2015)))
    assert float(result.occurred.sel(year=2015)) == 0.0


def test_the_same_late_trigger_stays_rejected_once_the_tail_is_supplied():
    """Supplying the 24 days the guard needs does not rescue this trigger: the
    tail is dry, so the guard finds a dry spell and rejects it on its merits.
    Together with the test above, unverifiable and verified-and-failed are
    separated, and neither route returns a step."""
    result = onset(_daily(_late_trigger_series(116)), "MAM")
    assert np.isnan(float(result.step.sel(year=2015)))
    assert float(result.occurred.sel(year=2015)) == 0.0


def test_metres_input_warns_rather_than_silently_reporting_no_onset():
    with pytest.warns(RuntimeWarning, match="implausibly small"):
        onset(_daily(_mam_2015_with_false_start() / 1000.0), "MAM")


def test_gridded_input_keeps_its_dims():
    """Cell 0 has a real onset, cell 1 never triggers, cell 2 has no data."""
    grid = xr.concat(
        [_daily(_mam_2015_with_false_start()),
         _daily(_no_onset_series()),
         _daily(np.full(120, np.nan))],
        dim="cell",
    )
    result = onset(grid, "MAM")
    assert set(result.step.dims) == {"year", "cell"}
    steps = result.step.sel(year=2015).values
    assert steps[0] == 20.0
    assert np.isnan(steps[1])
    assert np.isnan(steps[2])
    occurred = result.occurred.sel(year=2015).values
    assert occurred[0] == 1.0
    assert occurred[1] == 0.0
    assert np.isnan(occurred[2])


def test_leap_year_does_not_shift_the_step_axis():
    """1 Mar is step 0 in 2015 and in leap-year 2016. Day-of-year is not:
    21 Mar is DOY 80 in 2015 and DOY 81 in 2016."""
    rain = _mam_2015_with_false_start()
    both = xr.concat([_daily(rain, start="2015-03-01"),
                      _daily(rain, start="2016-03-01")], dim="time")
    result = onset(both, "MAM")
    assert float(result.step.sel(year=2015)) == 20.0
    assert float(result.step.sel(year=2016)) == 20.0
    assert str(result.date.sel(year=2015).values)[:10] == "2015-03-21"
    assert str(result.date.sel(year=2016).values)[:10] == "2016-03-21"


def test_season_crossing_the_new_year_is_indexed_from_its_own_start():
    """DJF 2015 runs 1 Dec 2015 to 29 Feb 2016. Day-of-year cannot order that
    span; season_step can, which is why step is the returned quantity."""
    rain = _no_onset_series(150)
    rain[40:43] = 7.0
    rain[43:80] = 5.0
    result = onset(_daily(rain, start="2015-12-01"), "DJF")
    assert float(result.step.sel(year=2015)) == 40.0
    assert str(result.date.sel(year=2015).values)[:10] == "2016-01-10"


def test_provenance_labels_the_default_criterion():
    result = onset(_daily(_mam_2015_with_false_start()), "MAM")
    assert result.step.attrs["onset_definition"] == "default"
    assert result.step.attrs["accum_mm"] == 20.0
    assert result.step.attrs["guard_days"] == 21
    assert result.step.attrs["season"] == "MAM"
    assert result.params["accum_days"] == 3


def test_provenance_labels_an_overridden_definition_custom():
    result = onset(_daily(_mam_2015_with_false_start()), "MAM", accum_days=2)
    assert result.step.attrs["onset_definition"] == "custom"
    assert result.step.attrs["accum_days"] == 2


def _boundary_series(dry_start):
    """Trigger at step 10, then a single 7-day dry run starting at dry_start.

    Guard window for the step-10 trigger is steps 13..33. A 7-day run ending
    inside that window is a false start; one ending after it is not.
    """
    rain = np.full(120, 5.0)
    rain[10:13] = 7.0                      # 21 mm trigger at step 10
    rain[dry_start:dry_start + 7] = 0.0
    rain[119] = 12.0    # lifts the peak above the mm/day sanity check; step 119
                        # is outside the season, so it cannot become an onset
    return rain


def test_dry_run_ending_on_the_first_disqualifying_step_is_a_false_start():
    """Run occupies steps 13-19, ending exactly at i+A+L-1 = 19, the earliest
    end position that still fits entirely inside the guard window."""
    result = onset(_daily(_boundary_series(13)), "MAM")
    assert float(result.step.sel(year=2015)) != 10.0


def test_dry_run_ending_on_the_last_guard_step_is_a_false_start():
    """Run occupies steps 27-33, ending exactly at i+A+G-1 = 33, the last step
    of the guard window."""
    result = onset(_daily(_boundary_series(27)), "MAM")
    assert float(result.step.sel(year=2015)) != 10.0


def test_dry_run_ending_one_step_past_the_guard_window_is_not_a_false_start():
    """Run occupies steps 28-34. It ends at 34, one past the window, so it is
    not fully contained and must not disqualify the trigger. The longest run
    fully inside the window is only 6 days."""
    result = onset(_daily(_boundary_series(28)), "MAM")
    assert float(result.step.sel(year=2015)) == 10.0


def test_dry_run_ending_just_before_the_guard_window_is_not_a_false_start():
    """A 7-day run occupying steps 12-18 ends at 18, one step before the guard
    window opens its disqualifying range at i+A+L-1 = 19, so it must not reject
    the trigger.

    Reaching this case needs a dry day inside the accumulation window: steps
    10-12 are 10.0 + 9.5 + 0.5 = 20.0 mm, which clears the 20 mm trigger while
    leaving step 12 at or below the 1 mm dry threshold. The split also keeps
    step 9's window at 19.5 mm, just under the threshold, so step 10 is
    genuinely the first trigger.
    """
    rain = np.zeros(120)
    rain[10] = 10.0
    rain[11] = 9.5
    rain[12] = 0.5              # dry, and the first day of the 12-18 run
    rain[13:19] = 0.0           # run is steps 12-18 inclusive, 7 days
    rain[19:80] = 5.0           # wet, so nothing later disqualifies step 10
    result = onset(_daily(rain), "MAM")
    assert float(result.step.sel(year=2015)) == 10.0


def _mam_2015_with_onset_and_cessation():
    """Onset at step 20, then rain until step 60, then a permanent dry spell."""
    rain = np.zeros(120)
    rain[20:23] = 7.0
    rain[23:60] = 5.0
    return rain


def test_cessation_finds_the_first_dry_spell_after_onset():
    daily = _daily(_mam_2015_with_onset_and_cessation())
    on = onset(daily, "MAM")
    ces = cessation(daily, "MAM", after=on)
    assert float(on.step.sel(year=2015)) == 20.0
    assert float(ces.step.sel(year=2015)) == 60.0
    assert float(ces.occurred.sel(year=2015)) == 1.0
    assert str(ces.date.sel(year=2015).values)[:10] == "2015-04-30"


def test_cessation_ignores_dry_spells_before_the_after_point():
    """The onset criterion tolerates pre-onset dry spells; cessation must not
    mistake one for the end of the season."""
    daily = _daily(_mam_2015_with_false_start())
    on = onset(daily, "MAM")
    ces = cessation(daily, "MAM", after=on)
    assert float(on.step.sel(year=2015)) == 20.0
    assert float(ces.step.sel(year=2015)) == 60.0


def test_cessation_accepts_a_plain_integer_step():
    daily = _daily(_mam_2015_with_onset_and_cessation())
    ces = cessation(daily, "MAM", after=0)
    assert float(ces.step.sel(year=2015)) == 0.0


def test_cessation_requires_after():
    with pytest.raises(TypeError):
        cessation(_daily(np.zeros(120)), "MAM")


def test_cessation_is_nan_where_onset_never_happened():
    daily = _daily(_no_onset_series())
    on = onset(daily, "MAM")
    ces = cessation(daily, "MAM", after=on)
    assert np.isnan(float(ces.step.sel(year=2015)))
    assert float(ces.occurred.sel(year=2015)) == 0.0


def test_season_length_is_the_difference():
    daily = _daily(_mam_2015_with_onset_and_cessation())
    on = onset(daily, "MAM")
    ces = cessation(daily, "MAM", after=on)
    assert float(season_length(on, ces).sel(year=2015)) == 40.0


def test_season_length_records_both_criteria_that_produced_it():
    """A shipped netCDF must not claim the default dry-spell rule when
    cessation was computed with another one. Cessation's parameters are
    prefixed so they cannot overwrite onset's same-named ones."""
    daily = _daily(_mam_2015_with_onset_and_cessation())
    on = onset(daily, "MAM")
    ces = cessation(daily, "MAM", after=on, dry_spell_days=15, dry_spell_mm=0.5)
    attrs = season_length(on, ces).attrs

    assert attrs["dry_spell_days"] == 7            # onset's guard, unchanged
    assert attrs["dry_spell_mm"] == 1.0
    assert attrs["cessation_dry_spell_days"] == 15
    assert attrs["cessation_dry_spell_mm"] == 0.5
    assert attrs["season"] == "MAM"                # shared, recorded once
    assert "cessation_season" not in attrs
    assert "cessation_deepscale_version" not in attrs


def test_season_length_is_nan_when_either_end_is_missing():
    daily = _daily(_no_onset_series())
    on = onset(daily, "MAM")
    ces = cessation(daily, "MAM", after=on)
    assert np.isnan(float(season_length(on, ces).sel(year=2015)))


def test_cessation_short_record_warns_about_the_unconfirmable_dry_spell():
    """A record stopping at the season end cannot confirm a late dry spell."""
    daily = _daily(_no_onset_series(92))
    with pytest.warns(RuntimeWarning, match="dry-spell window"):
        cessation(daily, "MAM", after=0)


def test_cessation_absent_when_the_season_never_dries():
    """Rain continues past the end of the record, so no dry spell ever starts.
    That is a real answer (0.0), not missing data (NaN)."""
    rain = np.zeros(120)
    rain[20:23] = 7.0
    rain[23:120] = 5.0          # never goes dry again
    daily = _daily(rain)
    on = onset(daily, "MAM")
    ces = cessation(daily, "MAM", after=on)
    assert float(on.step.sel(year=2015)) == 20.0
    assert np.isnan(float(ces.step.sel(year=2015)))
    assert float(ces.occurred.sel(year=2015)) == 0.0
