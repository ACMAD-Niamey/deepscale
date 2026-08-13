"""Onset over real CHIRPS daily rainfall for the Greater Horn of Africa."""
import numpy as np
import pytest

from deepscale.aggregations import cessation, dry_spell, onset, season_length

pytestmark = pytest.mark.integration

rosetta = pytest.importorskip("rosetta")

BOX = [-3.0, 3.0, 35.0, 41.0]   # [lat_s, lat_n, lon_w, lon_e], a small Kenya box


@pytest.fixture(scope="module")
def chirps_daily():
    """Daily CHIRPS for 2014-2016 over the test box.

    The season is MAM, but the record has to run at least to 24 June: onset's
    21-day guard on a 31 May trigger reads 24 days past the season end, so a
    record stopping at 31 May reports every late-season onset as no onset, with
    a warning. `months` is passed for intent, not for pruning: the sheerwater
    adapter resolves a date range and returns whole years regardless, so the
    fetch is wider than this list and the guard tail is covered either way.
    """
    return rosetta.fetch(
        "obs/chirps-v3-daily-rhiza", "precip",
        region=BOX,
        hindcast=(2014, 2016),
        months=[2, 3, 4, 5, 6],
    )["precip"]


def test_onset_falls_in_a_plausible_mam_window(chirps_daily):
    result = onset(chirps_daily, "MAM")
    steps = result.step.values
    found = steps[~np.isnan(steps)]
    assert found.size > 0, "no cell found an onset in three years of real data"
    # MAM is 92 days; onset outside it would mean the search window leaked.
    assert found.min() >= 0
    assert found.max() <= 91
    # Long-rains onset clusters in March and April, not the last week of May.
    assert np.median(found) < 70


def test_occurred_is_three_state_over_real_data(chirps_daily):
    occurred = onset(chirps_daily, "MAM").occurred.values
    present = set(np.unique(occurred[~np.isnan(occurred)]).tolist())
    assert present <= {0.0, 1.0}


def test_cessation_follows_onset_everywhere_it_exists(chirps_daily):
    on = onset(chirps_daily, "MAM")
    ces = cessation(chirps_daily, "MAM", after=on)
    length = season_length(on, ces).values
    finite = length[~np.isnan(length)]
    assert finite.size > 0
    assert (finite >= 0).all(), "cessation preceded onset somewhere"


def test_dry_spell_runs_on_real_data(chirps_daily):
    result = dry_spell(chirps_daily, "MAM")
    longest = result.longest.values
    finite = longest[~np.isnan(longest)]
    assert finite.size > 0
    assert (finite <= 92).all()
