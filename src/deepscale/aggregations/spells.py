"""Dry-spell statistics within a season.

The same run-length kernel that powers onset's false-start guard, read as
counting rather than dating. Emmett's observation in the design thread: because
the onset criterion already needs a dry-spell definition, the dry-spell metric
falls out of it rather than being built separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import xarray as xr

from ..climate import seasonal_stack
from ._runs import dry_run_lengths, has_data
from .timing import (
    _provenance,
    _require_daily,
    _stamp,
    _warn_on_incomplete_record,
    _warn_on_suspicious_units,
)

__all__ = ["DrySpellResult", "dry_spell"]


@dataclass
class DrySpellResult:
    """Dry-spell statistics, one value per season.

    ``longest`` is the length of the longest run at or below the threshold,
    whatever its length. ``count`` is the number of distinct runs reaching
    ``dry_spell_days``. Both are NaN for cells with no data, and 0.0 for cells
    that simply had no dry days.
    """

    longest: xr.DataArray
    count: xr.DataArray
    params: dict = field(default_factory=dict)


def dry_spell(
    daily: xr.DataArray,
    season,
    *,
    dry_spell_mm: float = 1.0,
    dry_spell_days: int = 7,
    time_dim: str = "time",
) -> DrySpellResult:
    """Longest dry run and count of qualifying dry runs, per season.

    A run is dry when each of its days receives ``dry_spell_mm`` or less. The
    defaults are the dry-spell half of the default onset criterion.

    No tail is stacked: unlike onset, this question is entirely contained
    within the season. ``daily`` must be daily; any other cadence raises,
    because a dry-spell length counted in days has no meaning on a coarser
    axis.
    """
    _require_daily(daily, time_dim)

    # Metres-instead-of-millimetres is as silently wrong here as it is for
    # onset, in the opposite direction: every day falls under the threshold and
    # the whole season reads as one dry spell.
    _warn_on_suspicious_units(daily)

    rain = seasonal_stack(daily, season, time_dim=time_dim)

    # A record that stops mid-season does not return NaN here: seasonal_stack
    # NaN-pads the gap, NaN counts as not-dry, so every run spanning the cut is
    # silently truncated and `longest` under-reports. Warn rather than let a
    # confident wrong number through.
    _warn_on_incomplete_record(
        daily, season, rain["year"].values, 0, time_dim,
        window="full season",
        consequence=("Dry runs are cut where the record ends, so longest and "
                     "count under-report rather than reporting no data."),
    )

    lengths = dry_run_lengths(rain, dry_spell_mm)
    ok = has_data(rain)

    longest = lengths.max("step").astype(float).where(ok)

    # A run reaches length L at exactly one step, the L-th day of that run, so
    # equality counts each qualifying run once regardless of how long it grows.
    count = (lengths == dry_spell_days).sum("step").astype(float).where(ok)

    params = dict(dry_spell_mm=dry_spell_mm, dry_spell_days=dry_spell_days)
    prov = _provenance(params, season)
    arrays = {"longest": longest, "count": count}
    _stamp(arrays, prov)
    return DrySpellResult(**arrays, params=prov)
