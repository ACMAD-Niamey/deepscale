"""Rainy-season timing: when the season starts, when it ends, how long it ran.

Onset is the first day satisfying a trigger and a guard. The trigger is a short
accumulation, the guard rejects false starts by requiring that no dry spell
follows within a window. Both are rolling-window questions, so both vectorise
across grid, year and ensemble member at once.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from ..climate import seasonal_stack
from ..time import infer_cadence, season_bounds, season_times
from ._runs import dry_run_lengths, has_data, rolling_total

__all__ = ["TimingResult", "onset", "cessation", "season_length", "ONSET_DEFAULTS"]

#: The default onset criterion: 20 mm across 3 consecutive days, rejected if a
#: 7-day dry spell (1 mm/day or less) falls within the following 21 days. It is
#: a widely used definition for East African MAM seasons and matches the
#: reference implementation this module was ported from. These are the defaults
#: of :func:`onset`; the dict exists so provenance can recognise them.
#:
#: Nothing about the module assumes them. Every value is a plain keyword
#: argument, so a different regional definition is one call away.
ONSET_DEFAULTS = {
    "accum_mm": 20.0,
    "accum_days": 3,
    "dry_spell_mm": 1.0,
    "dry_spell_days": 7,
    "guard_days": 21,
}

try:
    from importlib.metadata import version as _pkg_version

    _VERSION = _pkg_version("accord-deepscale")
except Exception:  # pragma: no cover - only when running from a bare checkout
    _VERSION = "unknown"


@dataclass
class TimingResult:
    """When a timing event happened, and whether it happened at all.

    ``step`` is 0-based days since the season start, on the same axis as
    :func:`deepscale.time.season_step`. It is leap-safe and works for seasons
    crossing the new year, neither of which day-of-year is, and its dims are
    ``(year, ...)`` so it can be used as a predictand anywhere the rest of
    deepscale takes one. Handle the season-failure years before converting it
    with :func:`deepscale.to_tercile_cv`, which reads NaN as missing data:
    here a NaN ``step`` can also mean the season failed, a fourth outcome
    rather than a tail of the distribution. ``occurred`` tells the two apart.

    ``occurred`` is a float rather than a bool because there are three states
    and a bool array cannot hold NaN:

    =========================  ========  ============
    State                      ``step``  ``occurred``
    =========================  ========  ============
    Event found                index     1.0
    Season failed, no event    NaN       0.0
    No data, e.g. ocean        NaN       NaN
    =========================  ========  ============
    """

    step: xr.DataArray
    date: xr.DataArray
    occurred: xr.DataArray
    params: dict = field(default_factory=dict)


def _provenance(params: dict, season, *, definition_of: str | None = None) -> dict:
    """Resolved parameters, recorded whether or not the caller passed them.

    Defaults are invisible at the call site, so they are written into the
    output instead. ``.attrs`` rather than the dataclass alone, so provenance
    survives ``to_netcdf`` into whatever file gets shipped.
    """
    out = dict(params)
    out["season"] = str(season)
    out["deepscale_version"] = _VERSION
    if definition_of == "onset":
        matches = all(params.get(k) == v for k, v in ONSET_DEFAULTS.items())
        out["onset_definition"] = "default" if matches else "custom"
    return out


#: Provenance keys that describe the analysis rather than one criterion, so
#: they are shared by both inputs of :func:`season_length` and recorded once.
_SHARED_PROVENANCE = ("season", "deepscale_version")


def _merge_provenance(onset_params: dict, cessation_params: dict) -> dict:
    """Both criteria behind a derived quantity, with no key silently winning.

    ``season_length`` is produced by an onset criterion *and* a cessation
    criterion. Recording only one of them makes a shipped netCDF misstate what
    produced it, and merging them flat would let cessation's ``dry_spell_days``
    overwrite onset's, which is a different parameter of a different rule.
    """
    out = dict(onset_params)
    for key, value in cessation_params.items():
        if key in _SHARED_PROVENANCE:
            continue
        out[f"cessation_{key}"] = value
    return out


def _stamp(arrays: dict[str, xr.DataArray], provenance: dict) -> None:
    """Attach provenance to every returned array, in place."""
    for arr in arrays.values():
        arr.attrs.update(provenance)


def _warn_on_suspicious_units(daily: xr.DataArray) -> None:
    """Metres-instead-of-millimetres produces 'no onset anywhere', which reads
    as a science result rather than a bug. Unit conversion belongs upstream."""
    if daily.size == 0 or not bool(daily.notnull().any()):
        return
    peak = float(daily.max(skipna=True))
    if np.isfinite(peak) and peak < 10.0:
        warnings.warn(
            f"daily rainfall peaks at {peak:.4g}, which is implausibly small for "
            "mm/day. If these are metres or a rate, convert before calling: "
            "every cell will report no onset.",
            RuntimeWarning, stacklevel=3,
        )


def _require_daily(daily, time_dim: str) -> None:
    """Every criterion here is defined in days, so non-daily input has no answer.

    Dekadal rainfall is a real catalog product, and passing it silently returns
    "no onset anywhere" rather than an error: the same failure class the units
    check exists for. Raising rather than warning is the right call because
    there is no meaningful result to warn about.
    """
    if time_dim not in daily.dims:
        raise ValueError(
            f"{time_dim!r} not found on data with dims {tuple(daily.dims)}"
        )
    cadence = infer_cadence(daily[time_dim])
    if cadence != "daily":
        raise ValueError(
            f"deepscale.aggregations needs daily rainfall; the {time_dim!r} axis "
            f"has {cadence} cadence. Accumulation days, dry-spell length and the "
            f"guard window are all counted in days, so {cadence} input has no "
            "meaningful answer. Fetch a daily product instead."
        )


def _warn_on_incomplete_record(daily, season, years, tail, time_dim, *,
                               window="guard window",
                               consequence=("Late-season triggers there cannot be "
                                            "verified and are reported as no onset.")) -> None:
    """The record must cover every day each season year is asked about.

    Checking only the record's last stamp misses two other ways of not covering
    a season: starting after it has begun, and dropping days inside it. All
    three read as a science result rather than a data problem, so all three are
    named here. Without this, "the season failed in 2015" and "you did not
    supply June" are indistinguishable in the output.

    ``window`` and ``consequence`` let callers other than :func:`onset` (namely
    :func:`cessation` and :func:`dry_spell`) reuse the same detection logic with
    wording that matches their own failure mode.

    This is a property of the time axis, not of any one cell, so it warns once
    per call and never per grid cell.
    """
    stamps = pd.DatetimeIndex(np.asarray(daily[time_dim].values)).normalize()
    first, last = stamps.min(), stamps.max()
    late, early, gapped = [], [], []
    needed = None
    for year in years:
        start, end = season_bounds(season, int(year))
        want = end + pd.Timedelta(days=tail)
        expected = pd.date_range(start, want, freq="D")
        present = np.asarray(expected.isin(stamps))
        if present.all():
            continue
        here = np.flatnonzero(present)
        if here.size == 0:
            # No stamp at all inside this season; seasonal_stack normally drops
            # such a year, so this is defensive rather than reachable.
            late.append(int(year))
            early.append(int(year))
            needed = want if needed is None else max(needed, want)
            continue
        if here[0] > 0:
            late.append(int(year))
        if here[-1] < len(expected) - 1:
            early.append(int(year))
            needed = want if needed is None else max(needed, want)
        if not present[here[0]:here[-1] + 1].all():
            gapped.append(int(year))

    if not (late or early or gapped):
        return

    span = f"{tail}-day {window}" if tail else window
    parts = []
    if early:
        parts.append(f"ends {last.date()} and does not cover the {span} "
                     f"for season year(s) {early}")
    if late:
        parts.append(f"starts {first.date()}, after the start of season "
                     f"year(s) {late}")
    if gapped:
        parts.append(f"has interior gaps within season year(s) {gapped}")
    msg = "daily data " + "; ".join(parts) + "."
    if early:
        msg += f" {consequence} Supply daily data through at least {needed.date()}."
    if late or gapped:
        msg += (" A season that is only partly covered is computed from the days "
                "present, so its result reflects the missing days rather than "
                "the weather.")
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _season_widths(season, rain) -> xr.DataArray:
    """Steps in the season proper, per year, excluding the tail.

    Varies across years for any season containing February.
    """
    return xr.DataArray(
        [len(season_times(season, int(y), "daily")) for y in rain["year"].values],
        dims="year", coords={"year": rain["year"]},
    )


def _resolve_dates(step, found, rain):
    """Calendar dates for a step field, NaT where the event did not happen."""
    days = step.fillna(0).astype("int64").astype("timedelta64[D]")
    return (rain["season_start"] + days).where(found)


def onset(
    daily: xr.DataArray,
    season,
    *,
    search_start: int | None = None,
    search_end: int | None = None,
    accum_mm: float = 20.0,
    accum_days: int = 3,
    dry_spell_mm: float = 1.0,
    dry_spell_days: int = 7,
    guard_days: int = 21,
    time_dim: str = "time",
) -> TimingResult:
    """First day of the rainy season, by trigger-and-guard.

    Onset is the first step where ``accum_days`` consecutive days total at
    least ``accum_mm``, and no run of ``dry_spell_days`` or more days at or
    below ``dry_spell_mm`` falls entirely within the following ``guard_days``.
    The defaults are ONSET_DEFAULTS, described at the top of this module.

    Parameters
    ----------
    daily : xr.DataArray
        Continuous daily rainfall in mm/day with a datetime ``time_dim``. Not
        pre-stacked; ``season`` is what carves it into years. Dims other than
        time survive, so ``(time, lat, lon)`` observations and
        ``(member, time, lat, lon)`` forecasts take the same path.

        Must be daily; any other cadence raises, because every criterion here
        is counted in days. Must cover each season in full and extend
        ``accum_days + guard_days`` past its end so late triggers can be
        verified. A record that starts late, stops early or drops days inside
        that span warns.
    season : see :func:`deepscale.time.season_bounds`
    search_start, search_end : int, optional
        0-based step bounds inside the season restricting where a trigger may
        fire. They do not restrict the data available to the guard, which
        always reads into the tail. ``None`` means the season boundary.

    Returns
    -------
    TimingResult
    """
    _require_daily(daily, time_dim)
    _warn_on_suspicious_units(daily)

    tail = accum_days + guard_days
    rain = seasonal_stack(daily, season, time_dim=time_dim, tail_days=tail)
    _warn_on_incomplete_record(daily, season, rain["year"].values, tail, time_dim)

    a, g, ell = accum_days, guard_days, dry_spell_days

    trigger = rolling_total(rain, a) >= accum_mm

    # A run of `ell` dry days fitting entirely inside the guard window
    # [i+a, i+a+g-1] must end somewhere in [i+a+ell-1, i+a+g-1], which is
    # g-ell+1 positions. So the guard is a rolling max over run ends, read at
    # the far edge of the window and shifted back to the candidate.
    run_ends = dry_run_lengths(rain, dry_spell_mm) >= ell
    if g >= ell and g > 0:
        false_start = (
            run_ends.rolling(step=g - ell + 1, min_periods=1).max()
            .shift(step=-(a + g - 1))
            .fillna(0).astype(bool)
        )
    else:
        # No window wide enough to hold a dry spell: nothing can be rejected.
        false_start = xr.zeros_like(run_ends, dtype=bool)

    if g > 0:
        # A guard window that runs past the end of the record, or over
        # NaN-padded steps, cannot be evaluated. The spec treats an
        # unverifiable trigger as no onset rather than accepting it, so the
        # guard fails closed.
        guard_covered = (
            rain.notnull().rolling(step=g, min_periods=g).min()
            .shift(step=-(a + g - 1))
            .fillna(0).astype(bool)
        )
    else:
        # No guard window to cover: nothing to fail closed on.
        guard_covered = xr.ones_like(run_ends, dtype=bool)

    widths = _season_widths(season, rain)
    lo = 0 if search_start is None else int(search_start)
    hi = (widths - 1) if search_end is None else int(search_end)
    within = (rain["step"] >= lo) & (rain["step"] <= hi)

    valid = (trigger & ~false_start & guard_covered).where(within, False)
    found = valid.any("step")
    step = valid.argmax("step").where(found).astype(float)
    date = _resolve_dates(step, found, rain)
    # Data in the tail is not data in the season. Counting it would let a
    # record beginning after the season ended report that season as failed
    # rather than as unobserved, and manufacture years cessation and dry_spell
    # do not have.
    in_season = rain["step"] <= (widths - 1)
    occurred = found.astype(float).where(has_data(rain.where(in_season)))

    params = dict(accum_mm=accum_mm, accum_days=accum_days,
                  dry_spell_mm=dry_spell_mm, dry_spell_days=dry_spell_days,
                  guard_days=guard_days, search_start=search_start,
                  search_end=search_end)
    prov = _provenance(params, season, definition_of="onset")
    arrays = {"step": step, "date": date, "occurred": occurred}
    _stamp(arrays, prov)
    return TimingResult(**arrays, params=prov)


def cessation(
    daily: xr.DataArray,
    season,
    *,
    after,
    dry_spell_mm: float = 1.0,
    dry_spell_days: int = 7,
    time_dim: str = "time",
) -> TimingResult:
    """Last day of the rainy season: the first qualifying dry spell after ``after``.

    Parameters
    ----------
    daily : xr.DataArray
        Continuous daily rainfall in mm/day with a datetime ``time_dim``.

        Must be daily; any other cadence raises. Must cover each season in
        full and extend ``dry_spell_days`` past its end so a dry spell
        beginning near the end can be confirmed. An incompletely covered
        record warns, and reports those cessations as absent rather than
        guessing.
    after : TimingResult or int
        Where to start looking, required and with no default. A dry spell
        before onset is not a cessation, and the onset criterion explicitly
        tolerates pre-onset dry spells, so searching from an arbitrary fixed
        point routinely returns the wrong answer. Pass the onset result for a
        per-cell floor, or an integer step for a uniform one.

        Where ``after`` is a ``TimingResult`` with no onset, the comparison
        against NaN is False everywhere and cessation is correctly absent too.

    Returns
    -------
    TimingResult
    """
    _require_daily(daily, time_dim)
    _warn_on_suspicious_units(daily)

    # tail_days == dry_spell_days is load-bearing, not incidental: it is what
    # clamps a detectable run start to at most one day past the season's last
    # index. Widening the tail without widening the criterion would let
    # cessation drift further outside the season.
    rain = seasonal_stack(daily, season, time_dim=time_dim,
                          tail_days=dry_spell_days)
    _warn_on_incomplete_record(
        daily, season, rain["year"].values, dry_spell_days, time_dim,
        window="dry-spell window",
        consequence=("A dry spell beginning near the season end cannot be "
                     "confirmed there and is reported as no cessation."),
    )
    ell = dry_spell_days

    # A run of `ell` days *starting* at s ends at s+ell-1, so shift run ends
    # back to their starts.
    run_starts = (
        (dry_run_lengths(rain, dry_spell_mm) >= ell)
        .shift(step=-(ell - 1))
        .fillna(0).astype(bool)
    )

    floor = after.step if isinstance(after, TimingResult) else after
    within = rain["step"] >= floor

    valid = run_starts.where(within, False)
    found = valid.any("step")
    step = valid.argmax("step").where(found).astype(float)
    date = _resolve_dates(step, found, rain)
    # As in `onset`: only in-season data can say a season happened at all.
    widths = _season_widths(season, rain)
    in_season = rain["step"] <= (widths - 1)
    occurred = found.astype(float).where(has_data(rain.where(in_season)))

    params = dict(dry_spell_mm=dry_spell_mm, dry_spell_days=dry_spell_days)
    prov = _provenance(params, season)
    arrays = {"step": step, "date": date, "occurred": occurred}
    _stamp(arrays, prov)
    return TimingResult(**arrays, params=prov)


def season_length(onset: TimingResult, cessation: TimingResult) -> xr.DataArray:
    """Days from onset to cessation, NaN wherever either end is missing.

    A season that started but had not ceased within the available window has a
    length of NaN rather than a truncated value: the answer is unknown, not
    short.

    Provenance carries *both* criteria. Cessation's parameters are prefixed
    ``cessation_`` so they cannot collide with onset's same-named ones, and
    ``season`` and ``deepscale_version`` are recorded once, from ``onset``.
    """
    out = cessation.step - onset.step
    out.attrs.update(_merge_provenance(onset.params, cessation.params))
    out.name = "season_length"
    return out
