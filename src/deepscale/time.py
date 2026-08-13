"""Calendar utilities for sub-monthly climate time axes.

Two things live here.

**Dekad/pentad arithmetic.** A dekad is a 10-day calendar slice (day 1-10,
11-20, 21-end-of-month), so the third dekad of a month is 8-11 days long. A
pentad is a fixed 5-day slice. Both are identified by their start date. This
logic was previously stranded in ``scripts/s2s/dekads.py``; it is library
behaviour, not script behaviour, so it lives here and the script imports it.

**Season-step alignment.** Splicing one year's partial season onto another
year's remainder (see :mod:`deepscale.completion`) cannot be done on calendar
timestamps: 1997-08-01 is not 2026-08-01, and a leap year shifts every
subsequent day-of-year. What the two years *share* is an ordinal position
within the season -- the k-th dekad of Kiremt, the k-th day of a monsoon.
:func:`season_step` assigns that ordinal, and it is the coordinate every
cross-year alignment in deepscale joins on.

The step index is derived from the time coordinate itself rather than assumed,
so a season sampled dekadally gets dekad steps, one sampled daily gets day
steps, and neither the caller nor the completion engine needs to know which.
"""
from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "dekad_start_for",
    "dekad_window",
    "dekad_of_year",
    "dekads_for_issuance",
    "pentad_start_for",
    "pentad_window",
    "infer_cadence",
    "season_step",
    "season_bounds",
    "season_months",
    "season_times",
]

# ---------------------------------------------------------------------------
# Dekads
# ---------------------------------------------------------------------------


def dekad_start_for(d: date) -> date:
    """Return the start date of the dekad containing ``d``."""
    if d.day <= 10:
        return d.replace(day=1)
    if d.day <= 20:
        return d.replace(day=11)
    return d.replace(day=21)


def dekad_window(start: date) -> tuple[date, date]:
    """Return ``[start, end)`` for the dekad beginning at ``start``.

    ``end`` is exclusive -- the first day of the *next* dekad. The third dekad
    of a month runs to the first of the following month, so its length varies
    between 8 and 11 days.
    """
    if start.day == 1:
        return start, start.replace(day=11)
    if start.day == 11:
        return start, start.replace(day=21)
    days_in_month = monthrange(start.year, start.month)[1]
    return start, start.replace(day=days_in_month) + timedelta(days=1)


def _next_dekad_start(start: date) -> date:
    return dekad_window(start)[1]


def dekad_of_year(d: date) -> int:
    """Ordinal dekad within the calendar year, 1-36."""
    within_month = 0 if d.day <= 10 else 1 if d.day <= 20 else 2
    return (d.month - 1) * 3 + within_month + 1


def dekads_for_issuance(issuance: date, lead_days: tuple[int, int]) -> list[date]:
    """Dekad start dates covered by a forecast issuance.

    The window is ``[issuance + lead_min, issuance + lead_max]``, both
    inclusive. A dekad is included when its start date falls inside it.
    """
    lead_min, lead_max = lead_days
    window_start = issuance + timedelta(days=lead_min)
    window_end = issuance + timedelta(days=lead_max)

    cur = dekad_start_for(window_start)
    out: list[date] = []
    while cur <= window_end:
        out.append(cur)
        cur = _next_dekad_start(cur)
    return out


# ---------------------------------------------------------------------------
# Pentads
# ---------------------------------------------------------------------------


def pentad_start_for(d: date) -> date:
    """Return the start date of the pentad containing ``d``.

    Pentads are fixed 5-day slices anchored at the start of each month; the
    sixth pentad absorbs the month's remaining 3-6 days (so a 31-day month has
    six pentads, the last being 6 days long). This matches the CHIRPS pentad
    convention.
    """
    idx = min((d.day - 1) // 5, 5)
    return d.replace(day=idx * 5 + 1)


def pentad_window(start: date) -> tuple[date, date]:
    """Return ``[start, end)`` for the pentad beginning at ``start``."""
    if start.day >= 26:
        days_in_month = monthrange(start.year, start.month)[1]
        return start, start.replace(day=days_in_month) + timedelta(days=1)
    return start, start + timedelta(days=5)


# ---------------------------------------------------------------------------
# Cadence inference and season steps
# ---------------------------------------------------------------------------

# Median spacing in days -> cadence label. Bounds are deliberately loose: a
# dekadal series has spacings of 8-11 days, a pentad series 5-6, and a monthly
# series 28-31, so the classes are well separated and a couple of missing
# stamps cannot flip the answer.
_CADENCE_BOUNDS = (
    (1.5, "daily"),
    (6.5, "pentad"),
    (12.0, "dekad"),
    (45.0, "monthly"),
)


def infer_cadence(time: xr.DataArray | pd.DatetimeIndex) -> str:
    """Infer the sampling cadence of a time coordinate.

    Returns one of ``"daily"``, ``"pentad"``, ``"dekad"``, ``"monthly"``. Raises
    ``ValueError`` for a single-stamp axis (no spacing to measure) or a spacing
    that matches nothing (e.g. seasonal or annual data).
    """
    idx = pd.DatetimeIndex(np.asarray(time.values if hasattr(time, "values") else time))
    if len(idx) < 2:
        raise ValueError(
            "infer_cadence needs at least two time stamps to measure a spacing; "
            f"got {len(idx)}. Pass cadence explicitly."
        )
    spacing = float(np.median(np.diff(idx.values).astype("timedelta64[h]").astype(float)) / 24.0)
    for upper, label in _CADENCE_BOUNDS:
        if spacing < upper:
            return label
    raise ValueError(
        f"median time spacing of {spacing:.1f} days matches no known cadence "
        "(daily/pentad/dekad/monthly). Pass cadence explicitly."
    )


# Steps per calendar year, for cadences whose count is leap-invariant. Daily is
# absent on purpose: 365 vs 366 is exactly the leap trap this module exists to
# avoid, so daily steps are measured as an elapsed-day difference instead.
_STEPS_PER_YEAR = {"pentad": 72, "dekad": 36, "monthly": 12}


def _step_index(idx: pd.DatetimeIndex, cadence: str) -> np.ndarray:
    """Ordinal position of each stamp within its calendar year, 0-based."""
    if cadence == "monthly":
        return idx.month.to_numpy() - 1
    if cadence == "dekad":
        return np.array([dekad_of_year(d.date()) - 1 for d in idx])
    if cadence == "pentad":
        # Six pentads per month, the sixth possibly short. 72 per year always.
        return np.array([(d.month - 1) * 6 + min((d.day - 1) // 5, 5) for d in idx])
    raise ValueError(f"unknown cadence {cadence!r}")


def season_bounds(season, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve a season specification to inclusive ``(start, end)`` timestamps.

    ``season`` may be:

    * a ``(start, end)`` pair of anything ``pd.Timestamp`` accepts;
    * a ``(start_month, end_month)`` pair of ints, resolved against ``year``
      (a wraparound season like ``(10, 2)`` ends in ``year + 1``);
    * a month-initial season code such as ``"JJAS"`` or ``"OND"``.
    """
    if isinstance(season, str):
        months = _months_from_code(season)
        return season_bounds((months[0], months[-1]), year)

    start, end = season
    if isinstance(start, (int, np.integer)) and isinstance(end, (int, np.integer)):
        end_year = year + (1 if int(end) < int(start) else 0)
        last_day = monthrange(end_year, int(end))[1]
        return (
            pd.Timestamp(year=year, month=int(start), day=1),
            pd.Timestamp(year=end_year, month=int(end), day=last_day),
        )
    return pd.Timestamp(start), pd.Timestamp(end)


_MONTH_INITIALS = "JFMAMJJASOND"


def season_months(season) -> list[int]:
    """The calendar months a season spans, in order. ``"JJAS"`` -> ``[6, 7, 8, 9]``.

    Useful for pruning a fetch to the months an analysis actually needs, rather
    than downloading a whole year and discarding three quarters of it.
    ``season`` may be a month-initial code or an ``(start_month, end_month)``
    pair; a wraparound season wraps (``"NDJ"`` -> ``[11, 12, 1]``).
    """
    if isinstance(season, str):
        return _months_from_code(season)
    start, end = season
    if not (isinstance(start, (int, np.integer)) and isinstance(end, (int, np.integer))):
        raise TypeError(
            "season_months needs a month-initial code or an (start_month, "
            f"end_month) pair of ints, got {season!r}"
        )
    span = (int(end) - int(start)) % 12
    return [((int(start) - 1 + k) % 12) + 1 for k in range(span + 1)]


def _months_from_code(code: str) -> list[int]:
    """``"JJAS"`` -> ``[6, 7, 8, 9]``. Ambiguous initials resolve by contiguity."""
    code = code.upper()
    # Try every starting month; the season is the unique contiguous run whose
    # month initials spell `code`. ("JJAS" could start at month 1 or 6 by first
    # letter alone; only month 6 spells the whole code.)
    matches = []
    for start in range(1, 13):
        months = [((start - 1 + k) % 12) + 1 for k in range(len(code))]
        if "".join(_MONTH_INITIALS[m - 1] for m in months) == code:
            matches.append(months)
    if len(matches) != 1:
        raise ValueError(
            f"season code {code!r} resolves to {len(matches)} month runs; "
            "pass an explicit (start_month, end_month) pair instead."
        )
    return matches[0]


def season_step(
    time: xr.DataArray,
    season,
    *,
    year: int | None = None,
    cadence: str | None = None,
    tail_days: int = 0,
) -> xr.DataArray:
    """Assign each stamp in ``time`` its 0-based ordinal position in ``season``.

    This is the coordinate that makes cross-year splicing well defined. Two
    different years' dekads that occupy the same position in the season get the
    same step, regardless of leap days or the variable length of the third
    dekad.

    Parameters
    ----------
    time : xr.DataArray
        A datetime64 coordinate.
    season : see :func:`season_bounds`
        The accumulation window.
    year : int, optional
        The season's start year. Defaults to the calendar year of the first
        stamp -- correct except for a wraparound season whose axis begins after
        the new year, where it must be given explicitly.
    cadence : str, optional
        Overrides :func:`infer_cadence`.
    tail_days : int, default 0
        Extend the season's end boundary by this many calendar days. Stamps in
        the tail receive their natural continuing step index instead of -1.
        Used by rolling-window aggregations that must look past the season end
        to evaluate a window opened inside it. The default admits nothing extra.

    Returns
    -------
    xr.DataArray
        Integer steps with the same dims as ``time``. Stamps outside the season
        get ``-1``.
    """
    idx = pd.DatetimeIndex(np.asarray(time.values))
    if year is None:
        year = int(idx[0].year)
    cadence = cadence or infer_cadence(time)

    start, end = season_bounds(season, year)
    if tail_days:
        end = end + pd.Timedelta(days=int(tail_days))

    if cadence == "daily":
        # Elapsed days from the season start. Exact across leap years, which an
        # ordinal-difference on day-of-year would not be.
        steps = (idx - start).days.to_numpy()
    else:
        # Ordinal difference, unrolling the year boundary so a wraparound
        # season (e.g. OND -> J) stays monotonic. Safe because these cadences
        # have a leap-invariant number of steps per year.
        origin = _step_index(pd.DatetimeIndex([start]), cadence)[0]
        raw = _step_index(idx, cadence)
        steps = raw - origin + (idx.year.to_numpy() - start.year) * _STEPS_PER_YEAR[cadence]

    inside = (idx >= start) & (idx <= end)
    steps = np.where(inside, steps, -1)
    return xr.DataArray(steps, dims=time.dims, coords=time.coords, name="season_step")


def season_times(season, year: int, cadence: str) -> pd.DatetimeIndex:
    """The start stamp of every step in ``season``, in order.

    The inverse of :func:`season_step`: given a season and a cadence, enumerate
    the steps rather than classify stamps into them. Completion uses this to
    build the full step axis of a season before any data has been placed on it,
    so a season that is only half observed still knows how long it is.

    A season whose start falls mid-step begins at the step *containing* the
    start, matching :func:`season_step`'s ordinal convention.
    """
    start, end = season_bounds(season, year)

    if cadence == "daily":
        return pd.date_range(start, end, freq="D")
    if cadence == "monthly":
        return pd.date_range(start.replace(day=1), end, freq="MS")
    if cadence == "dekad":
        first, advance = dekad_start_for(start.date()), _next_dekad_start
    elif cadence == "pentad":
        first = pentad_start_for(start.date())
        advance = lambda d: pentad_window(d)[1]  # noqa: E731
    else:
        raise ValueError(f"unknown cadence {cadence!r}")

    stamps, cur = [], first
    while cur <= end.date():
        stamps.append(cur)
        cur = advance(cur)
    return pd.DatetimeIndex(stamps)
