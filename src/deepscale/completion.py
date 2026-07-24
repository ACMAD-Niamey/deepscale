"""Scenario completion: finishing a partly-observed season with analog years.

Part-way through a rainy season you know what has fallen so far, you may have a
short-range forecast for the next few weeks, and beyond that you have nothing
but history. Scenario completion splices the three together — observations to
date, then the forecast, then each analog year's remainder — producing one
plausible end-of-season outcome per analog. The spread across analogs *is* the
uncertainty, and every member is a real season that really happened, which is
what makes the result explicable to someone who has to act on it.

    scenarios = complete(observed, analogs, climatology=archive, season="JJAS",
                         forecast=next_30_days)
    scenarios.percentile      # where the consensus total falls in the record
    scenarios.accumulation()  # the curves, for plotting

The engine is deliberately ignorant of three things.

**Of what the analogs mean.** It takes an
:class:`~deepscale.analog.AnalogSet`, however it was selected — nine El Niño
years by name, the ten nearest neighbours in dipole-mode-index space, the
fifteen closest SST patterns over a bounding box.

**Of the time axis.** Dekads, pentads, days and months all work, because
splicing happens on the season's ordinal step index rather than on calendar
dates, which never align across years.

**Of the spatial dimensions.** ``complete`` reduces along ``step`` and touches
nothing else, so ``(time, lat, lon)`` gives per-pixel maps, ``(time, region)``
from a zonal aggregation gives per-district curves, and ``(time,)`` gives a
single series. This is why there is no separate admin-unit code path.

The forecast segment is optional, so running with and without it isolates
exactly what the dynamic forecast contributes — the comparison a forecaster
needs before deciding to trust it.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

import numpy as np
import pandas as pd
import xarray as xr

from .analog import AnalogSet
from .climate import accumulate, percentile_of
from .time import infer_cadence, season_step, season_times

__all__ = ["complete", "CompletionResult"]

OBSERVED, FORECAST, ANALOG = "observed", "forecast", "analog"
_OVERLAP = ("observed", "forecast", "error")


@dataclass(frozen=True)
class CompletionResult:
    """The spliced scenario ensemble and its summaries.

    Attributes
    ----------
    scenarios : xr.DataArray
        ``(scenario, step, ...)`` per-step increments. One scenario per analog,
        labelled by the analog's year.
    totals : xr.DataArray
        ``(scenario, ...)`` season totals — one plausible outcome per analog.
    consensus : xr.DataArray
        ``(...)`` the totals reduced across scenarios (median by default).
    percentile : xr.DataArray
        ``(...)`` where ``consensus`` falls in the historical record, in [0, 1].
    segments : xr.DataArray
        ``(step,)`` of ``"observed"`` / ``"forecast"`` / ``"analog"``, saying
        where each step's value came from.
    analogs : AnalogSet
        The selection used, kept so a result can explain itself.
    """

    scenarios: xr.DataArray
    totals: xr.DataArray
    consensus: xr.DataArray
    percentile: xr.DataArray
    segments: xr.DataArray
    analogs: AnalogSet
    metadata: dict = _dc_field(default_factory=dict)

    def accumulation(self) -> xr.DataArray:
        """Running season-to-date totals, ``(scenario, step, ...)``.

        The accumulation curves: observed increments up to the forecast, then
        each analog fanning out. Missing steps contribute nothing rather than
        truncating the curve, so inspect :attr:`segments` and the raw
        :attr:`scenarios` before reading a flat stretch as a dry spell.
        """
        curves = self.scenarios.cumsum("step", keep_attrs=True)
        # cumsum keeps dim coords but drops the non-dim ones riding on `step`.
        extra = {
            name: coord for name, coord in self.scenarios.coords.items()
            if name not in curves.coords
        }
        return curves.assign_coords(extra).rename("scenario_accumulation")

    def segment_steps(self, segment: str) -> np.ndarray:
        """The step indices contributed by one segment."""
        if segment not in (OBSERVED, FORECAST, ANALOG):
            raise ValueError(f"unknown segment {segment!r}")
        return self.segments.step.values[self.segments.values == segment]


# ---------------------------------------------------------------------------
# Reduction across scenarios
# ---------------------------------------------------------------------------


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float, axis: int):
    """Quantile of ``values`` under ``weights`` along ``axis``.

    Uses the mid-cumulative-weight plotting position, so with equal weights it
    reproduces the ordinary linear-interpolation quantile. Any NaN along the
    reduced axis poisons that cell, matching the unweighted path.
    """
    order = np.argsort(values, axis=axis)
    sorted_values = np.take_along_axis(values, order, axis=axis)
    sorted_weights = np.take_along_axis(
        np.broadcast_to(weights, values.shape), order, axis=axis
    )

    cumulative = np.cumsum(sorted_weights, axis=axis)
    total = np.take(cumulative, [-1], axis=axis)
    position = (cumulative - 0.5 * sorted_weights) / total

    n = values.shape[axis]
    # `position` is increasing along `axis`, so counting the entries below `q`
    # locates the bracketing pair without a per-cell searchsorted.
    upper = np.clip((position < q).sum(axis=axis, keepdims=True), 0, n - 1)
    lower = np.clip(upper - 1, 0, n - 1)

    p_lo = np.take_along_axis(position, lower, axis=axis)
    p_hi = np.take_along_axis(position, upper, axis=axis)
    v_lo = np.take_along_axis(sorted_values, lower, axis=axis)
    v_hi = np.take_along_axis(sorted_values, upper, axis=axis)

    span = p_hi - p_lo
    fraction = np.where(span > 0, (q - p_lo) / np.where(span > 0, span, 1.0), 0.0)
    # Clamp to the bracketing order statistics: for q beyond the outermost
    # plotting position the interpolation would otherwise extrapolate past the
    # min/max scenario (fraction < 0 or > 1), and a completion consensus must
    # never fall outside the range of the analog scenarios it summarises.
    fraction = np.clip(fraction, 0.0, 1.0)
    out = np.squeeze(v_lo + fraction * (v_hi - v_lo), axis=axis)
    return np.where(np.isnan(values).any(axis=axis), np.nan, out)


def _reduce_scenarios(totals: xr.DataArray, reduce, weights: xr.DataArray | None):
    """Collapse the ``scenario`` dim of ``totals``."""
    if callable(reduce):
        return reduce(totals, dim="scenario")

    if reduce == "mean":
        if weights is None:
            return totals.mean("scenario", skipna=False)
        aligned = weights.rename(year="scenario").sel(scenario=totals.scenario)
        return totals.weighted(aligned).mean("scenario", skipna=False)

    q = 0.5 if reduce == "median" else reduce
    if not isinstance(q, (int, float)) or not 0.0 <= float(q) <= 1.0:
        raise ValueError(
            "reduce must be 'median', 'mean', a quantile in [0, 1], or a callable; "
            f"got {reduce!r}"
        )
    if weights is None:
        # NaN must not be skipped: dropping a scenario silently shrinks the
        # ensemble, and a quantile of a smaller ensemble is a different quantile.
        return totals.quantile(float(q), dim="scenario", skipna=False).drop_vars("quantile")

    axis = totals.get_axis_num("scenario")
    aligned = weights.rename(year="scenario").sel(scenario=totals.scenario)
    shape = [1] * totals.ndim
    shape[axis] = totals.sizes["scenario"]
    result = _weighted_quantile(
        totals.values, aligned.values.reshape(shape), float(q), axis
    )
    reduced_dims = [d for d in totals.dims if d != "scenario"]
    return xr.DataArray(
        result, dims=reduced_dims,
        coords={d: totals.coords[d] for d in reduced_dims if d in totals.coords},
    )


# ---------------------------------------------------------------------------
# Segment assembly
# ---------------------------------------------------------------------------


def _to_steps(da, season, year, cadence, time_dim, width, label):
    """Re-index ``da`` from calendar time onto the season's step axis."""
    if da is None:
        return None, np.zeros(width, dtype=bool)
    if time_dim not in da.dims:
        raise ValueError(
            f"{label} must carry the time dim {time_dim!r}; got {tuple(da.dims)}"
        )
    steps = season_step(da[time_dim], season, year=year, cadence=cadence)
    inside = steps.values >= 0
    if not inside.any():
        raise ValueError(
            f"none of {label}'s time stamps fall inside season {season!r} of {year}"
        )
    kept = steps.values[inside]
    if kept.max() >= width:
        raise ValueError(
            f"{label} reaches step {kept.max()} but the season's step axis is only "
            f"{width} long; the climatology and the season disagree"
        )
    sub = da.isel({time_dim: inside})
    sub = (
        sub.assign_coords(step=(time_dim, kept))
        .swap_dims({time_dim: "step"})
        .drop_vars(time_dim)
        .reindex(step=np.arange(width))
    )
    present = np.zeros(width, dtype=bool)
    present[kept] = True
    return sub, present


def _labels(observed_steps, forecast_steps, width, overlap):
    both = observed_steps & forecast_steps
    if both.any() and overlap == "error":
        raise ValueError(
            f"observed and forecast overlap at steps {np.flatnonzero(both).tolist()}; "
            "pass overlap='observed' or 'forecast' to say which wins"
        )
    labels = np.full(width, ANALOG, dtype=object)
    if overlap == "forecast":
        labels[observed_steps] = OBSERVED
        labels[forecast_steps] = FORECAST
    else:
        labels[forecast_steps] = FORECAST
        labels[observed_steps] = OBSERVED
    return labels


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def complete(
    observed: xr.DataArray | None,
    analogs: AnalogSet,
    *,
    climatology: xr.DataArray,
    season,
    year: int | None = None,
    forecast: xr.DataArray | None = None,
    analog_source: xr.DataArray | None = None,
    reduce="median",
    weights: str | xr.DataArray | None = None,
    percentile_reference: xr.DataArray | None = None,
    percentile_method: str = "empirical",
    overlap: str = "observed",
    time_dim: str = "time",
    cadence: str | None = None,
    min_count: int | None = None,
) -> CompletionResult:
    """Complete a partly-observed season with each analog year's remainder.

    Parameters
    ----------
    observed : xr.DataArray or None
        ``(time, ...)`` increments observed so far. ``None`` runs a pure analog
        projection, which is the useful null case: what the season would look
        like on history alone.
    analogs : AnalogSet
        Which years to complete with. See :mod:`deepscale.analog`.
    climatology : xr.DataArray
        ``(year, step, ...)`` season-aligned historical increments — the output
        of :func:`deepscale.climate.seasonal_stack`. Supplies the analog
        remainders and, by default, the reference distribution for
        ``percentile``.
    season, year : see :func:`deepscale.time.season_bounds`
        The accumulation window. ``year`` defaults to the season year of
        ``observed``'s first stamp (or ``forecast``'s, if there are no
        observations).
    forecast : xr.DataArray, optional
        ``(time, ...)`` increments for the steps after the observations. Omit it
        to isolate what the forecast contributes: run twice, once each way.
    analog_source : xr.DataArray, optional
        ``(year, step, ...)`` to draw the analog remainders from, if not
        ``climatology`` — e.g. a bias-corrected archive, or a different variable.
    reduce : {"median", "mean"}, float or callable
        How the scenario totals become one consensus number. A float is a
        quantile. A callable receives ``(totals, dim="scenario")``. Median is
        the default because a scenario ensemble is small and its tails are one
        year each.
    weights : {"uniform", "inverse_distance", "gaussian"} or xr.DataArray, optional
        Per-analog weights. A string is passed to :meth:`AnalogSet.weights`.
        ``None`` weights every analog equally.
    percentile_reference : xr.DataArray, optional
        ``(year, ...)`` historical season totals to rank ``consensus`` against.
        Defaults to summing ``climatology`` over its steps.
    overlap : {"observed", "forecast", "error"}
        Which segment wins where observations and forecast cover the same step.

    Returns
    -------
    CompletionResult
    """
    if overlap not in _OVERLAP:
        raise ValueError(f"overlap must be one of {_OVERLAP}, got {overlap!r}")
    if observed is None and forecast is None:
        raise ValueError(
            "pass at least one of observed or forecast; completing a season with "
            "neither is just the analog climatology"
        )
    for name, da in (("climatology", climatology), ("analog_source", analog_source)):
        if da is not None and not {"year", "step"} <= set(da.dims):
            raise ValueError(
                f"{name} must have 'year' and 'step' dims (see "
                f"deepscale.climate.seasonal_stack); got {tuple(da.dims)}"
            )

    source = climatology if analog_source is None else analog_source
    missing = sorted(set(np.asarray(analogs.years).tolist())
                     - set(source.year.values.tolist()))
    if missing:
        raise ValueError(f"analog years absent from the analog source: {missing}")

    anchor = observed if observed is not None else forecast
    if cadence is None:
        cadence = infer_cadence(anchor[time_dim])
    if year is None:
        year = int(pd.Timestamp(np.asarray(anchor[time_dim].values)[0]).year)

    width = int(source.sizes["step"])
    stamps = season_times(season, year, cadence)
    if len(stamps) > width:
        raise ValueError(
            f"season {season!r} of {year} has {len(stamps)} {cadence} steps but the "
            f"climatology only carries {width}; they must describe the same season"
        )

    observed_steps, observed_present = _to_steps(
        observed, season, year, cadence, time_dim, width, "observed")
    forecast_steps, forecast_present = _to_steps(
        forecast, season, year, cadence, time_dim, width, "forecast")

    labels = _labels(observed_present, forecast_present, width, overlap)
    segments = xr.DataArray(labels, dims="step",
                            coords={"step": np.arange(width)}, name="segment")

    # Start from each analog's own season, then overwrite the steps we know.
    scenarios = (
        source.sel(year=list(analogs.years))
        .rename(year="scenario")
        .reindex(step=np.arange(width))
    )
    if forecast_steps is not None:
        scenarios = xr.where(segments == FORECAST, forecast_steps, scenarios)
    if observed_steps is not None:
        scenarios = xr.where(segments == OBSERVED, observed_steps, scenarios)

    extra_coords = {"segment": ("step", labels)}
    if len(stamps) == width:
        # Calendar stamps for the target year's steps, so a plot can label its
        # x-axis. Omitted when the season is shorter than the archive's step
        # axis (only possible for a daily wraparound season across a leap year).
        extra_coords["time"] = ("step", stamps.values)
    # `xr.where` broadcasts in whatever order it likes; pin the leading dims so
    # every caller sees the same shape regardless of the spatial layout.
    scenarios = (
        scenarios.assign_coords(extra_coords)
        .transpose("scenario", "step", ...)
        .rename("scenario_increment")
    )

    totals = accumulate(scenarios, dim="step", min_count=min_count).rename("season_total")

    analog_weights = None
    if isinstance(weights, str):
        analog_weights = analogs.weights(weights)
    elif weights is not None:
        analog_weights = weights
    consensus = _reduce_scenarios(totals, reduce, analog_weights).rename("consensus_total")

    if percentile_reference is None:
        percentile_reference = accumulate(climatology, dim="step", min_count=min_count)
    percentile = percentile_of(
        consensus, percentile_reference, dim="year", method=percentile_method
    ).rename("consensus_percentile")

    return CompletionResult(
        scenarios=scenarios,
        totals=totals,
        consensus=consensus,
        percentile=percentile,
        segments=segments,
        analogs=analogs,
        metadata={
            "season": season,
            "year": year,
            "cadence": cadence,
            "reduce": reduce if isinstance(reduce, (str, float, int)) else "callable",
            "weights": weights,
            "overlap": overlap,
            "n_observed_steps": int(observed_present.sum()),
            "n_forecast_steps": int((labels == FORECAST).sum()),
            "n_analog_steps": int((labels == ANALOG).sum()),
            "n_scenarios": len(analogs),
        },
    )
