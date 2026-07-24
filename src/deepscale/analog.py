"""Analog-year selection: which past years resemble the year being forecast.

An analog ensemble is an interpretable seasonal prior — "here is what actually
happened in the N most comparable years" — and its credibility rests entirely
on how *comparable* was decided. This module makes that decision explicit,
auditable, and independent of what makes two years comparable.

Four ways to say it, all returning the same :class:`AnalogSet`:

``analogs_from_years``
    An explicit list. The expert judgement case: "these nine moderate-to-strong
    El Niños".
``analogs_from_index``
    Nearest neighbours in the value of any scalar index — Niño3.4, a dipole
    mode index, a region-averaged rainfall total, anything
    :class:`~deepscale.indices.Index` or the caller can produce.
``analogs_from_field``
    Nearest neighbours in the *pattern* of any gridded field over any region:
    SST across a tropical Pacific rectangle, 500 hPa heights over a basin,
    rainfall over a catchment.
``analogs_where``
    A boolean predicate over years, for criteria that are thresholds rather
    than distances ("every year whose Niño3.4 exceeded 0.5").

They compose. ``&`` intersects, ``|`` unions, and ``.top(n)`` truncates, so a
compound criterion — strong El Niño *and* rapid onset — is an expression rather
than a bespoke function:

    strong = analogs_where(nino34 >= 0.5)
    rapid = analogs_where((nino34 - nino34_spring) >= 1.0)
    analogs = (strong & rapid).top(9)

Every selector scores *all* candidate years, not just the chosen ones, so the
margin between the ninth and tenth analog is always inspectable.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

import numpy as np
import xarray as xr

from ._spatial import spatial_dims

__all__ = [
    "AnalogSet",
    "analogs_from_years",
    "analogs_from_index",
    "analogs_from_field",
    "analogs_where",
]

# Every metric is distance-like — lower is a better analog — including the
# correlation ones, which score `1 - r`. That uniformity is what lets `top(n)`,
# `&` and `|` mean the same thing regardless of how comparability was defined.
_FIELD_METRICS = ("rmse", "mae", "correlation", "anomaly_correlation")
_INDEX_METRICS = ("absolute", "squared", "signed")


@dataclass(frozen=True)
class AnalogSet:
    """A ranked selection of analog years, with the scores that produced it.

    Attributes
    ----------
    years : np.ndarray
        The selected years, best analog first.
    scores : xr.DataArray
        Distance-like score for *every* candidate year (lower is a better
        analog), indexed by ``year``. Years excluded by a predicate carry NaN.
        Keeping the full score vector is what lets a reviewer see how close the
        tenth candidate came to displacing the ninth.
    metadata : dict
        How the selection was made: selector name, metric, target value.
    """

    years: np.ndarray
    scores: xr.DataArray
    metadata: dict = _dc_field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "years", np.asarray(self.years))
        if "year" not in self.scores.dims:
            raise ValueError("AnalogSet.scores must be indexed by a 'year' dim")
        missing = set(self.years) - set(self.scores.year.values.tolist())
        if missing:
            raise ValueError(f"selected years absent from scores: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.years)

    def __iter__(self):
        return iter(self.years)

    def __repr__(self) -> str:
        how = self.metadata.get("selector", "?")
        return f"AnalogSet({len(self)} years via {how}: {list(self.years)})"

    @property
    def candidates(self) -> np.ndarray:
        """Every year that was scored, selected or not."""
        return self.scores.year.values

    def top(self, n: int) -> "AnalogSet":
        """The ``n`` best-scoring selected years, order preserved."""
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if n > len(self.years):
            raise ValueError(
                f"cannot take the top {n} of only {len(self.years)} selected years"
            )
        return AnalogSet(self.years[:n], self.scores, {**self.metadata, "top": n})

    def weights(self, kind: str = "uniform", *, scale: float | None = None) -> xr.DataArray:
        """Per-analog weights, summing to one, indexed by ``year``.

        ``"uniform"`` weights every analog equally — the honest default when the
        scores are not calibrated distances. ``"inverse_distance"`` and
        ``"gaussian"`` weight closer analogs more heavily; both need a positive
        score scale, so ``scale`` defaults to the median selected score.
        """
        selected = self.scores.sel(year=self.years)
        if kind == "uniform":
            raw = xr.ones_like(selected)
        else:
            positive = selected.where(selected > 0)
            default_scale = float(positive.median()) if positive.notnull().any() else 1.0
            scale = default_scale if scale is None else float(scale)
            if not scale > 0:
                raise ValueError(f"scale must be positive, got {scale}")
            if kind == "inverse_distance":
                raw = scale / (selected + scale)
            elif kind == "gaussian":
                raw = np.exp(-0.5 * (selected / scale) ** 2)
            else:
                raise ValueError(
                    "kind must be 'uniform', 'inverse_distance' or 'gaussian', "
                    f"got {kind!r}"
                )
        return (raw / raw.sum()).rename("analog_weight")

    # -- composition -------------------------------------------------------
    def _combined_scores(self, other: "AnalogSet") -> xr.DataArray:
        """Mean of the two score vectors, aligned on the shared candidate years.

        Averaging rather than picking one keeps a compound criterion's ranking
        sensitive to both of its parts. A year scored by only one side (the
        predicate selectors leave non-selected years NaN) keeps that side's
        score rather than being poisoned to NaN, so a union stays rankable.
        """
        left, right = xr.align(self.scores, other.scores, join="inner")
        if left.sizes["year"] == 0:
            raise ValueError("the two AnalogSets share no candidate years")
        stacked = xr.concat([left, right], dim="_part")
        return stacked.mean("_part", skipna=True).rename("analog_score")

    def _rank(self, years, scores) -> np.ndarray:
        if len(years) == 0:
            return np.asarray(years)
        ordered = scores.sel(year=list(years)).sortby(scores.sel(year=list(years)))
        return ordered.year.values

    def __and__(self, other: "AnalogSet") -> "AnalogSet":
        """Years selected by both, re-ranked on the mean of the two scores."""
        scores = self._combined_scores(other)
        keep = [y for y in self.years if y in set(other.years.tolist())]
        return AnalogSet(
            self._rank(keep, scores), scores,
            {"selector": "and", "parts": (self.metadata, other.metadata)},
        )

    def __or__(self, other: "AnalogSet") -> "AnalogSet":
        """Years selected by either, re-ranked on the mean of the two scores."""
        scores = self._combined_scores(other)
        union = list(self.years) + [y for y in other.years if y not in set(self.years.tolist())]
        keep = [y for y in union if y in set(scores.year.values.tolist())]
        return AnalogSet(
            self._rank(keep, scores), scores,
            {"selector": "or", "parts": (self.metadata, other.metadata)},
        )

    def filter(self, mask: xr.DataArray) -> "AnalogSet":
        """Drop selected years where the boolean ``mask`` (over ``year``) is False."""
        allowed = set(mask.year.values[mask.values.astype(bool)].tolist())
        keep = np.array([y for y in self.years if y in allowed])
        return AnalogSet(keep, self.scores, {**self.metadata, "filtered": True})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _restrict(scores: xr.DataArray, candidates) -> xr.DataArray:
    if candidates is None:
        return scores
    wanted = [y for y in np.asarray(candidates).tolist()]
    unknown = set(wanted) - set(scores.year.values.tolist())
    if unknown:
        raise ValueError(f"candidate years absent from the data: {sorted(unknown)}")
    return scores.where(scores.year.isin(wanted))


def _rank_and_take(scores: xr.DataArray, n: int | None) -> np.ndarray:
    """Years ordered by ascending score, dropping NaN (excluded) candidates."""
    finite = scores.where(scores.notnull(), drop=True)
    ordered = finite.sortby(finite).year.values
    if n is None:
        return ordered
    if n > len(ordered):
        raise ValueError(
            f"asked for {n} analogs but only {len(ordered)} candidate years scored"
        )
    return ordered[:n]


def _exactly_one(target, target_year, what: str):
    """Resolve the ``target`` / ``target_year`` pair, refusing ambiguity.

    These are kept as separate arguments rather than one overloaded argument on
    purpose. A year label is an integer and an index value is a float, but a
    rainfall total of 1997 mm is both — so guessing which the caller meant is a
    bug waiting for the right data.
    """
    if (target is None) == (target_year is None):
        raise ValueError(
            f"pass exactly one of target (a {what}) or target_year (a year to "
            "read it from)"
        )


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------


def analogs_from_years(years, *, candidates=None, scores: xr.DataArray | None = None) -> AnalogSet:
    """Take an explicit list of analog years.

    The expert-judgement path: no metric, no ranking, the order given is the
    order kept. ``candidates`` (or ``scores``) supplies the pool of years the
    selection was made from, so the result still reports what was *not* chosen.
    """
    years = np.asarray(list(years))
    if len(years) == 0:
        raise ValueError("analogs_from_years needs at least one year")
    if scores is None:
        pool = np.asarray(sorted(set(years.tolist()) | set(
            np.asarray(candidates).tolist() if candidates is not None else []
        )))
        # No metric was applied, so every candidate is equidistant.
        scores = xr.DataArray(
            np.where(np.isin(pool, years), 0.0, np.nan),
            dims="year", coords={"year": pool}, name="analog_score",
        )
    return AnalogSet(years, scores, {"selector": "years"})


def analogs_from_index(
    index: xr.DataArray,
    target=None,
    *,
    target_year: int | None = None,
    n: int | None = None,
    metric: str = "absolute",
    candidates=None,
) -> AnalogSet:
    """Rank years by how close their ``index`` value is to a target value.

    Parameters
    ----------
    index : xr.DataArray
        A scalar series over a ``year`` dim — the output of
        :meth:`deepscale.Index.reduce`, or any series the caller computed.
    target : scalar
        The index value to match, e.g. this year's forecast Niño3.4.
    target_year : int
        Alternative to ``target``: match the value ``index`` takes in this year.
        Exactly one of the two is required.
    metric : {"absolute", "squared", "signed"}
        ``"signed"`` ranks by ``target - index``, so only years *at or beyond*
        the target score well — useful when the analog must be at least as
        extreme as the forecast, not merely near it.
    n : int, optional
        How many analogs to keep. ``None`` keeps every scored year, ranked.
    candidates : sequence of int, optional
        Restrict the pool. ``target_year`` is not excluded automatically (it is
        its own perfect analog); pass ``candidates`` to leave it out.
    """
    if "year" not in index.dims:
        raise ValueError(f"index must have a 'year' dim, got {tuple(index.dims)}")
    if metric not in _INDEX_METRICS:
        raise ValueError(f"metric must be one of {_INDEX_METRICS}, got {metric!r}")
    _exactly_one(target, target_year, "value")

    value = float(index.sel(year=target_year)) if target is None else float(target)
    difference = index - value
    if metric == "absolute":
        scores = abs(difference)
    elif metric == "squared":
        scores = difference ** 2
    else:  # signed: distance is how far *short of* the target a year falls
        scores = -difference

    scores = _restrict(scores.rename("analog_score"), candidates)
    return AnalogSet(
        _rank_and_take(scores, n), scores,
        {"selector": "index", "metric": metric, "target": value,
         "target_year": target_year, "index": index.name},
    )


def _field_scores(field, target, metric, weights):
    lat, lon = spatial_dims(field, context="analogs_from_field")
    dims = [lat, lon]
    if weights == "cos_lat":
        w = np.cos(np.deg2rad(field[lat])).clip(min=0.0)
    elif weights is None:
        w = None
    elif isinstance(weights, xr.DataArray):
        w = weights
    else:
        raise ValueError(f"weights must be None, 'cos_lat' or a DataArray, got {weights!r}")

    def _mean(da):
        return da.weighted(w).mean(dims, skipna=True) if w is not None else da.mean(dims, skipna=True)

    if metric == "mae":
        return _mean(abs(field - target))
    if metric == "rmse":
        return np.sqrt(_mean((field - target) ** 2))

    # Correlation-like: turn similarity into a distance so lower is always
    # better and `top(n)` means one thing across every metric.
    if metric == "anomaly_correlation":
        # Anomalies about the climatology; no further spatial centring, which
        # is what distinguishes ACC from a plain spatial Pearson correlation.
        climatology = field.mean("year")
        field_anomaly, target_anomaly = field - climatology, target - climatology
    else:  # "correlation": centre each pattern on its own spatial mean
        field_anomaly, target_anomaly = field - _mean(field), target - _mean(target)

    covariance = _mean(field_anomaly * target_anomaly)
    denominator = np.sqrt(_mean(field_anomaly ** 2) * _mean(target_anomaly ** 2))
    correlation = covariance / xr.where(denominator < 1e-12, np.nan, denominator)
    return 1.0 - correlation


def analogs_from_field(
    field: xr.DataArray,
    target: xr.DataArray | None = None,
    *,
    target_year: int | None = None,
    n: int | None = None,
    metric: str = "rmse",
    region=None,
    weights: object = "cos_lat",
    candidates=None,
) -> AnalogSet:
    """Rank years by how closely their ``field`` pattern resembles a target pattern.

    Pattern similarity, rather than the value of a single index. Nothing here is
    SST-specific: ``field`` may be any gridded variable, and ``region`` any box
    or geometry.

    Parameters
    ----------
    field : xr.DataArray
        ``(year, lat, lon)``. A ``member`` dim is averaged out.
    target : xr.DataArray
        The ``(lat, lon)`` pattern to match — e.g. a forecast map.
    target_year : int
        Alternative to ``target``: use the pattern ``field`` takes in this year.
        Exactly one of the two is required.
    metric : {"rmse", "mae", "correlation", "anomaly_correlation"}
        The correlation metrics score ``1 - r``, so lower is a better analog in
        every case and ``top(n)`` means the same thing throughout.
    region : bbox, shapefile path or geometry, optional
        Restrict the comparison. A bbox is ``[lat_s, lat_n, lon_w, lon_e]``;
        anything else requires Rosetta, as with :class:`deepscale.Index`.
    weights : {"cos_lat", None} or xr.DataArray
        Area weighting for the spatial reduction. Defaults to ``"cos_lat"``
        because a pattern distance over a tall region is otherwise dominated by
        its high-latitude cells.
    """
    if "year" not in field.dims:
        raise ValueError(f"field must have a 'year' dim, got {tuple(field.dims)}")
    if metric not in _FIELD_METRICS:
        raise ValueError(f"metric must be one of {_FIELD_METRICS}, got {metric!r}")
    _exactly_one(target, target_year, "pattern")

    if "member" in field.dims:
        field = field.mean("member")

    if target is None:
        target = field.sel(year=int(target_year), drop=True)
    elif "year" in target.dims:
        raise ValueError("target pattern must not carry a 'year' dim")

    if region is not None:
        field = _clip(field, region)
        target = _clip(target, region)

    scores = _field_scores(field, target, metric, weights).rename("analog_score")
    scores = _restrict(scores, candidates)
    return AnalogSet(
        _rank_and_take(scores, n), scores,
        {"selector": "field", "metric": metric, "target_year": target_year,
         "region": region,
         "weights": weights if weights is None or isinstance(weights, str) else "custom"},
    )


def _clip(da: xr.DataArray, region) -> xr.DataArray:
    """Restrict ``da`` to ``region``, reusing Index's region resolution."""
    from .indices import Index

    lat, lon = spatial_dims(da, context="analogs_from_field")
    bbox, geometry = Index._resolve_region(region)
    lat_s, lat_n, lon_w, lon_e = bbox
    da = da.assign_coords({lon: da[lon] % 360}).sortby(lon)
    lon_w, lon_e = lon_w % 360, lon_e % 360
    if lon_w < lon_e:
        lon_mask = (da[lon] >= lon_w) & (da[lon] <= lon_e)
    elif lon_w > lon_e:
        lon_mask = (da[lon] >= lon_w) | (da[lon] <= lon_e)
    else:
        lon_mask = xr.ones_like(da[lon], dtype=bool)
    clipped = da.where((da[lat] >= lat_s) & (da[lat] <= lat_n) & lon_mask, drop=True)
    if geometry is not None:
        clipped = Index._mask_geometry(clipped, geometry, lat, lon)
    return clipped


def analogs_where(condition: xr.DataArray, *, scores: xr.DataArray | None = None) -> AnalogSet:
    """Select every year where the boolean ``condition`` over ``year`` holds.

    For criteria that are thresholds rather than distances, e.g.
    ``analogs_where(nino34 >= 0.5)``. Pass ``scores`` to supply a ranking; by
    default the selected years are equally good, and composing with ``&`` or
    ``|`` against a distance-based set restores an ordering.
    """
    if "year" not in condition.dims:
        raise ValueError(f"condition must have a 'year' dim, got {tuple(condition.dims)}")
    # Treat NaN as False: `np.nan.astype(bool)` is True, so a raw float condition
    # with holes would otherwise select the NaN years. Boolean conditions (the
    # common `nino34 >= 0.5` form) are unaffected.
    mask = condition.fillna(False).astype(bool)
    years = condition.year.values[mask.values]
    if len(years) == 0:
        raise ValueError("the condition selected no years")
    if scores is None:
        scores = xr.DataArray(
            np.where(mask.values, 0.0, np.nan),
            dims="year", coords={"year": condition.year.values}, name="analog_score",
        )
    return AnalogSet(years, scores, {"selector": "where"})
