"""Analog-year selection: every way of saying which past years resemble this one."""
import numpy as np
import pytest
import xarray as xr

from deepscale.analog import (
    AnalogSet,
    analogs_from_field,
    analogs_from_index,
    analogs_from_years,
    analogs_where,
)

YEARS = np.arange(1990, 2010)


def _index(values, years=YEARS, name="nino34"):
    return xr.DataArray(np.asarray(values, dtype=float), dims="year",
                        coords={"year": years}, name=name)


@pytest.fixture
def nino34():
    """A synthetic ENSO index: 1997 and 2005 are the strongest El Niños."""
    values = np.zeros(len(YEARS))
    values[YEARS == 1997] = 2.4
    values[YEARS == 2005] = 2.2
    values[YEARS == 1998] = -1.5
    values[YEARS == 2002] = 1.1
    values[YEARS == 1991] = 0.8
    return _index(values)


def _pattern():
    lat = np.arange(-10, 11, 5.0)
    lon = np.arange(180, 241, 5.0)
    return lat, lon, np.exp(
        -((lat[:, None] / 8.0) ** 2 + ((lon[None, :] - 210) / 20.0) ** 2)
    )


@pytest.fixture
def sst_field():
    """(year, lat, lon): each year is the same warm pattern at a growing amplitude."""
    lat, lon, pattern = _pattern()
    amplitude = np.linspace(0.5, 3.0, len(YEARS))
    values = 27.0 + amplitude[:, None, None] * pattern[None, :, :]
    return xr.DataArray(values, dims=("year", "lat", "lon"),
                        coords={"year": YEARS, "lat": lat, "lon": lon})


# --- analogs_from_years ----------------------------------------------------


def test_from_years_keeps_the_given_order():
    got = analogs_from_years([1997, 1982, 2015], candidates=np.arange(1980, 2020))
    assert list(got.years) == [1997, 1982, 2015]
    assert got.metadata["selector"] == "years"


def test_from_years_records_the_unchosen_candidates():
    got = analogs_from_years([1997], candidates=[1995, 1996, 1997])
    assert sorted(got.candidates.tolist()) == [1995, 1996, 1997]
    assert np.isnan(got.scores.sel(year=1995))
    assert got.scores.sel(year=1997) == 0.0


def test_from_years_rejects_an_empty_list():
    with pytest.raises(ValueError, match="at least one year"):
        analogs_from_years([])


# --- analogs_from_index ----------------------------------------------------


def test_from_index_ranks_by_absolute_distance(nino34):
    got = analogs_from_index(nino34, target=2.3, n=3)
    # 2.4 (1997) is 0.1 away, 2.2 (2005) is 0.1 away, 1.1 (2002) is 1.2 away.
    assert set(got.years[:2].tolist()) == {1997, 2005}
    assert got.years[2] == 2002


def test_from_index_target_year_reads_the_value_at_that_year(nino34):
    by_year = analogs_from_index(nino34, target_year=1997, n=2)
    by_value = analogs_from_index(nino34, target=2.4, n=2)
    assert list(by_year.years) == list(by_value.years)
    assert by_year.years[0] == 1997  # a year is its own best analog


def test_from_index_requires_exactly_one_of_target_and_target_year(nino34):
    with pytest.raises(ValueError, match="exactly one of target"):
        analogs_from_index(nino34)
    with pytest.raises(ValueError, match="exactly one of target"):
        analogs_from_index(nino34, target=1.0, target_year=1997)


def test_from_index_signed_metric_prefers_years_at_or_beyond_the_target(nino34):
    """With `signed`, a year stronger than the target scores better than one the
    same distance below it — the analog must be at least as extreme."""
    got = analogs_from_index(nino34, target=1.1, metric="signed", n=2)
    assert got.years[0] == 1997  # 2.4: overshoots, score -1.3
    assert got.years[1] == 2005  # 2.2: overshoots, score -1.1
    absolute = analogs_from_index(nino34, target=1.1, metric="absolute", n=1)
    assert absolute.years[0] == 2002  # exact match wins under `absolute`


def test_from_index_squared_metric_preserves_absolute_ordering(nino34):
    a = analogs_from_index(nino34, target=1.0, metric="absolute", n=5)
    s = analogs_from_index(nino34, target=1.0, metric="squared", n=5)
    assert list(a.years) == list(s.years)


def test_from_index_candidates_restrict_the_pool(nino34):
    got = analogs_from_index(nino34, target=2.3, n=1, candidates=[2002, 1991])
    assert got.years[0] == 2002
    assert np.isnan(got.scores.sel(year=1997))


def test_from_index_scores_every_candidate_not_just_the_chosen(nino34):
    got = analogs_from_index(nino34, target=2.3, n=2)
    assert got.scores.sizes["year"] == len(YEARS)
    assert bool(got.scores.notnull().all())


def test_from_index_refuses_more_analogs_than_scored_years(nino34):
    with pytest.raises(ValueError, match="only 2 candidate years scored"):
        analogs_from_index(nino34, target=1.0, n=3, candidates=[1997, 2002])


def test_from_index_rejects_a_bad_metric_or_missing_year_dim(nino34):
    with pytest.raises(ValueError, match="metric must be one of"):
        analogs_from_index(nino34, target=1.0, metric="cosine")
    with pytest.raises(ValueError, match="must have a 'year' dim"):
        analogs_from_index(nino34.rename(year="time"), target=1.0)


def test_from_index_rejects_unknown_candidate_years(nino34):
    with pytest.raises(ValueError, match="absent from the data"):
        analogs_from_index(nino34, target=1.0, candidates=[1899])


# --- analogs_from_field ----------------------------------------------------


def test_from_field_finds_the_year_whose_pattern_matches(sst_field):
    target = sst_field.sel(year=2003, drop=True)
    got = analogs_from_field(sst_field, target=target, n=1)
    assert got.years[0] == 2003


def test_from_field_target_year_is_its_own_best_analog(sst_field):
    got = analogs_from_field(sst_field, target_year=2003, n=1)
    assert got.years[0] == 2003
    assert float(got.scores.sel(year=2003)) == pytest.approx(0.0, abs=1e-12)


def test_from_field_ranks_neighbours_by_amplitude(sst_field):
    """Amplitude increases monotonically with year, so the analogs of 2003 are
    its immediate neighbours."""
    got = analogs_from_field(sst_field, target_year=2003, n=3)
    assert set(got.years.tolist()) == {2003, 2002, 2004}


def test_from_field_correlation_metric_is_amplitude_blind(sst_field):
    """Every year here is the same spatial pattern at a different amplitude, so
    `correlation` cannot separate them while `rmse` can. This is the reason to
    offer both: match the shape of an event, or match its size."""
    corr = analogs_from_field(sst_field, target_year=2009, metric="correlation")
    np.testing.assert_allclose(corr.scores.values, 0.0, atol=1e-8)

    rmse = analogs_from_field(sst_field, target_year=2009, metric="rmse", n=2)
    assert rmse.years[0] == 2009 and rmse.years[1] == 2008


def test_from_field_correlation_metric_is_sign_sensitive(sst_field):
    """Amplitude-blind is not sign-blind: an inverted pattern anticorrelates,
    scoring 1 - (-1) = 2, the worst possible distance."""
    lat, lon, pattern = _pattern()
    inverted = xr.DataArray(27.0 - 2.0 * pattern, dims=("lat", "lon"),
                            coords={"lat": lat, "lon": lon})
    got = analogs_from_field(sst_field, target=inverted, metric="correlation")
    np.testing.assert_allclose(got.scores.values, 2.0, atol=1e-8)


def test_from_field_anomaly_correlation_scores_a_year_against_itself_as_zero(sst_field):
    got = analogs_from_field(sst_field, target_year=2009, metric="anomaly_correlation")
    assert float(got.scores.sel(year=2009)) == pytest.approx(0.0, abs=1e-9)


def test_from_field_mae_and_rmse_agree_on_the_best_analog(sst_field):
    mae = analogs_from_field(sst_field, target_year=2001, metric="mae", n=1)
    rmse = analogs_from_field(sst_field, target_year=2001, metric="rmse", n=1)
    assert mae.years[0] == rmse.years[0] == 2001


def test_from_field_region_restricts_the_comparison(sst_field):
    """Scoring over a box that excludes the warm anomaly makes every year look
    alike; scoring over the box containing it does not."""
    off_pattern = analogs_from_field(
        sst_field, target_year=2009, region=[-10, -8, 180, 185], metric="rmse")
    on_pattern = analogs_from_field(
        sst_field, target_year=2009, region=[-5, 5, 200, 220], metric="rmse")
    assert float(off_pattern.scores.sel(year=1990)) < float(on_pattern.scores.sel(year=1990))


def test_from_field_averages_out_a_member_dim(sst_field):
    ensemble = sst_field.expand_dims(member=[0, 1, 2])
    got = analogs_from_field(ensemble, target_year=2003, n=1)
    assert got.years[0] == 2003


def test_from_field_cos_lat_weighting_is_the_default_and_can_be_disabled(sst_field):
    assert analogs_from_field(sst_field, target_year=2003).metadata["weights"] == "cos_lat"
    unweighted = analogs_from_field(sst_field, target_year=2003, weights=None)
    assert unweighted.metadata["weights"] is None


def test_from_field_rejects_a_target_carrying_a_year_dim(sst_field):
    with pytest.raises(ValueError, match="must not carry a 'year' dim"):
        analogs_from_field(sst_field, target=sst_field.sel(year=[2003]))


def test_from_field_rejects_a_bad_metric_or_weights(sst_field):
    with pytest.raises(ValueError, match="metric must be one of"):
        analogs_from_field(sst_field, target_year=2003, metric="hausdorff")
    with pytest.raises(ValueError, match="weights must be"):
        analogs_from_field(sst_field, target_year=2003, weights="area")


# --- analogs_where ---------------------------------------------------------


def test_where_selects_every_year_meeting_a_threshold(nino34):
    got = analogs_where(nino34 >= 0.8)
    assert sorted(got.years.tolist()) == [1991, 1997, 2002, 2005]


def test_where_marks_unselected_years_nan_rather_than_zero(nino34):
    got = analogs_where(nino34 >= 0.8)
    assert np.isnan(got.scores.sel(year=1998))
    assert float(got.scores.sel(year=1997)) == 0.0


def test_where_rejects_a_condition_matching_nothing(nino34):
    with pytest.raises(ValueError, match="selected no years"):
        analogs_where(nino34 > 99.0)


def test_where_treats_nan_in_a_float_condition_as_false():
    """A raw float condition with holes must not select the NaN years —
    `np.nan.astype(bool)` is True, so an un-guarded cast would pick them up."""
    values = np.array([1.0, np.nan, 0.0, 1.0] + [0.0] * (len(YEARS) - 4))
    condition = _index(values)
    got = analogs_where(condition)
    assert sorted(got.years.tolist()) == [YEARS[0], YEARS[3]]


# --- composition -----------------------------------------------------------


def test_and_intersects_two_criteria(nino34):
    """The compound criterion the CHC deck describes in prose: moderate-or-
    stronger El Niño *and* rapid onset. Composition, not a bespoke function."""
    strong = analogs_where(nino34 >= 0.8)
    onset = _index(np.where(np.isin(YEARS, [1997, 2002, 2015]), 1.5, 0.1))
    rapid = analogs_where(onset >= 1.0)

    both = strong & rapid
    assert sorted(both.years.tolist()) == [1997, 2002]


def test_or_unions_two_criteria(nino34):
    a = analogs_where(nino34 >= 2.0)          # 1997, 2005
    b = analogs_where(nino34 <= -1.0)         # 1998
    assert sorted((a | b).years.tolist()) == [1997, 1998, 2005]


def test_and_reranks_on_the_mean_of_both_scores(nino34):
    """Intersecting a predicate with a distance ranks the survivors by distance
    (the predicate contributes a constant 0), which is the useful behaviour."""
    strong = analogs_where(nino34 >= 0.8)
    near = analogs_from_index(nino34, target=1.1)
    combined = strong & near
    assert combined.years[0] == 2002  # nino34 == 1.1 exactly


def test_or_keeps_a_year_scored_by_only_one_side(nino34):
    """A predicate leaves non-selected years NaN. Averaging naively would poison
    the union's scores; the mean must skip the missing side."""
    predicate = analogs_where(nino34 >= 2.0)
    distance = analogs_from_index(nino34, target=-1.5, n=1)  # picks 1998
    union = predicate | distance
    assert sorted(union.years.tolist()) == [1997, 1998, 2005]
    assert bool(union.scores.sel(year=[1997, 1998, 2005]).notnull().all())


def test_composition_of_disjoint_candidate_pools_raises(nino34):
    a = analogs_from_years([1997], candidates=[1996, 1997])
    b = analogs_from_years([1801], candidates=[1800, 1801])
    with pytest.raises(ValueError, match="share no candidate years"):
        a & b


def test_filter_drops_selected_years(nino34):
    got = analogs_where(nino34 >= 0.8).filter(_index(YEARS != 1997).astype(bool))
    assert 1997 not in got.years.tolist()


# --- AnalogSet -------------------------------------------------------------


def test_top_truncates_preserving_rank(nino34):
    full = analogs_from_index(nino34, target=2.3)
    assert list(full.top(3).years) == list(full.years[:3])


def test_top_refuses_to_exceed_the_selection(nino34):
    got = analogs_from_index(nino34, target=2.3, n=2)
    with pytest.raises(ValueError, match="top 5 of only 2"):
        got.top(5)
    with pytest.raises(ValueError, match="n must be at least 1"):
        got.top(0)


def test_uniform_weights_are_equal_and_sum_to_one(nino34):
    got = analogs_from_index(nino34, target=2.3, n=4)
    w = got.weights()
    assert float(w.sum()) == pytest.approx(1.0)
    np.testing.assert_allclose(w.values, 0.25)


def test_inverse_distance_weights_favour_the_closest_analog(nino34):
    got = analogs_from_index(nino34, target=2.3, n=3)
    w = got.weights("inverse_distance")
    assert float(w.sum()) == pytest.approx(1.0)
    assert float(w.sel(year=got.years[0])) > float(w.sel(year=got.years[-1]))


def test_gaussian_weights_decay_faster_than_inverse_distance(nino34):
    got = analogs_from_index(nino34, target=2.3, n=3)
    inv = got.weights("inverse_distance", scale=1.0)
    gau = got.weights("gaussian", scale=1.0)
    assert float(gau.sel(year=got.years[-1])) < float(inv.sel(year=got.years[-1]))


def test_weights_on_an_unranked_explicit_selection_are_uniform():
    """Explicit years carry no distance, so distance weighting degenerates to
    uniform rather than dividing by zero."""
    got = analogs_from_years([1997, 1982, 2015])
    np.testing.assert_allclose(got.weights("inverse_distance").values, 1 / 3)


def test_weights_reject_a_bad_kind_or_scale(nino34):
    got = analogs_from_index(nino34, target=2.3, n=3)
    with pytest.raises(ValueError, match="kind must be"):
        got.weights("softmax")
    with pytest.raises(ValueError, match="scale must be positive"):
        got.weights("gaussian", scale=0.0)


def test_analogset_is_iterable_and_sized(nino34):
    got = analogs_from_index(nino34, target=2.3, n=3)
    assert len(got) == 3
    assert list(got) == list(got.years)


def test_analogset_rejects_years_absent_from_its_scores():
    scores = xr.DataArray([0.0], dims="year", coords={"year": [2000]})
    with pytest.raises(ValueError, match="absent from scores"):
        AnalogSet(np.array([1999]), scores)


def test_analogset_repr_names_the_selector(nino34):
    assert "index" in repr(analogs_from_index(nino34, target=2.3, n=2))
