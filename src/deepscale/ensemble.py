"""Ensemble combination with optional weight optimisation under overfitting
safeguards (nested CV, shrinkage, effective-N floor, acceptance gate).

The public surface is :func:`ensemble`, returning an :class:`EnsembleResult`
that carries the combined forecast, the final per-member weights, per-member
CV skill, and diagnostics for each safeguard that fired.

Spec: docs/superpowers/specs/2026-05-13-ensemble-safeguards-design.md
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from .registry import get_strategy
from .cv import get_cv
from .skill import skill
from .pev import prediction_error_variance


_DEFAULT_SAFEGUARDS = {
    "nested_cv": True,
    "shrinkage": 0.5,
    "min_effective_n": 3,
    "gate": True,
}

_VALID_SAFEGUARD_KEYS = set(_DEFAULT_SAFEGUARDS)


@dataclass
class EnsembleResult:
    """Outcome of an ensemble combination.

    See docs/superpowers/specs/2026-05-13-ensemble-safeguards-design.md
    for the field contract.

    `pev` (added in the PEV design,
    `docs/superpowers/specs/2026-05-15-prediction-error-variance-design.md`)
    is the cross-validated prediction error variance per grid cell. It is
    populated automatically by `ensemble()` from honest CV predictions
    when those are available:

    - `optimize_ensemble=False` + year-dim forecasts + obs given → from the
      uniform-combined CV hindcasts.
    - `optimize_ensemble=True`, `nested_cv=True` + year-dim forecasts + obs
      → from the pooled outer-fold CV forecasts.
    - `optimize_ensemble=True`, `nested_cv=False` → `None` (no honest CV
      predictions exist on this path; matches the existing
      `nested_cv_warning`).
    - Year-less forecasts or `obs is None` → `None`.

    `member_contributions` (added in the ensemble-diagnostics design,
    `docs/superpowers/specs/2026-05-15-ensemble-diagnostics-design.md`) is a
    dict keyed by member name. Each entry has two keys:
    `correlation_with_mme_mean` (float; per-year Pearson correlation of the
    member's spatial-mean CV hindcast with the MME's spatial-mean CV
    forecast) and `skill_delta` (xr.DataArray of MME pearson_r minus member
    pearson_r per grid cell). Populated under the same conditions as
    ``pev``; otherwise ``None``.
    """
    forecast: xr.DataArray
    weights: np.ndarray
    member_names: list
    member_cv_skill: dict = field(default_factory=dict)
    effective_n: float = 0.0
    gate_passed: bool = True
    shrinkage_lambda: float = 0.0
    safeguards_applied: dict = field(default_factory=dict)
    pev: xr.DataArray | None = None
    member_contributions: dict | None = None


def _uniform_weights(n):
    return np.full(n, 1.0 / n)


def _apply_shrinkage(w, lambda_):
    w = np.asarray(w, dtype=float)
    if lambda_ == 0.0:
        return w.copy()
    n = len(w)
    return (1.0 - lambda_) * w + lambda_ / n


def _effective_n(w):
    w = np.asarray(w, dtype=float)
    return float(1.0 / (w ** 2).sum())


def _resolve_safeguards(user_safeguards):
    if user_safeguards is None:
        return dict(_DEFAULT_SAFEGUARDS)
    unknown = set(user_safeguards) - _VALID_SAFEGUARD_KEYS
    if unknown:
        raise ValueError(
            f"unknown safeguards keys: {sorted(unknown)}; "
            f"valid keys are {sorted(_VALID_SAFEGUARD_KEYS)}"
        )
    merged = dict(_DEFAULT_SAFEGUARDS)
    merged.update(user_safeguards)
    # Floor cannot be negative — effective_N is always >= 1, so any
    # negative threshold silently disables the safeguard.
    if merged["min_effective_n"] < 0:
        raise ValueError(
            f"min_effective_n must be >= 0; got {merged['min_effective_n']}"
        )
    return merged


def _score(forecast, obs, primary_metric):
    """Score a single forecast against obs on the named metric. Returns float.

    Raises ``ValueError`` if ``primary_metric`` expands to multiple sub-metrics
    (e.g. ``roc`` → ``roc_bn``/``roc_nn``/``roc_an``). Pick a leaf metric to
    score against; silent default-selection would let the wrong AUC drive
    weight optimisation.
    """
    report = skill(forecast, obs, metrics=[primary_metric])
    if primary_metric in report.scores:
        return float(report.scores[primary_metric])
    raise ValueError(
        f"primary_metric={primary_metric!r} expands to {sorted(report.scores)}; "
        f"pick one of those leaf metrics instead."
    )


def _compute_member_contributions(forecasts, obs, combined, member_names):
    """Per-member contribution diagnostics.

    For each member i with CV hindcast f_i, compute:
      - correlation_with_mme_mean: Pearson correlation between the
        spatial-mean time series of f_i and combined, across years.
      - skill_delta: pearson_r(combined, obs, spatial) -
        pearson_r(f_i, obs, spatial), shape (lat, lon).

    Returns dict[member_name -> {"correlation_with_mme_mean": float,
    "skill_delta": xr.DataArray}].

    Assumes all inputs have a `year` dim aligned with obs.year.
    """
    from .metrics.pearson import PearsonMetric

    pearson = PearsonMetric()
    spatial_dims = [d for d in combined.dims if d != "year"]

    mme_skill_spatial = pearson.compute(combined, obs, spatial=True)
    mme_spatial_mean = combined.mean(spatial_dims)

    out = {}
    for i, name in enumerate(member_names):
        member = forecasts[i].forecast if hasattr(forecasts[i], "forecast") else forecasts[i]
        member_skill_spatial = pearson.compute(member, obs, spatial=True)
        member_spatial_mean = member.mean(spatial_dims)

        # Correlation of two per-year time series. xr.corr is the obvious
        # tool but only available in newer xarray; compute manually for
        # safety. NaN propagates when either side has zero variance.
        x = member_spatial_mean - member_spatial_mean.mean("year")
        y = mme_spatial_mean - mme_spatial_mean.mean("year")
        denom = float(np.sqrt((x ** 2).sum("year")) * np.sqrt((y ** 2).sum("year")))
        if denom == 0.0 or not np.isfinite(denom):
            corr = float("nan")
        else:
            corr = float((x * y).sum("year") / denom)

        out[name] = {
            "correlation_with_mme_mean": corr,
            "skill_delta": (mme_skill_spatial - member_skill_spatial),
        }
    return out


def _member_names(forecasts):
    names = []
    for i, f in enumerate(forecasts):
        if hasattr(f, "method"):
            names.append(f.method)
            continue
        arr = f.forecast if hasattr(f, "forecast") else f
        names.append(arr.name or f"member_{i}")
    return names


def ensemble(forecasts, obs, *, strategy="uniform", optimize_ensemble=False,
             primary_metric="rpss", safeguards=None, cv="loyo", **kwargs):
    """Combine multiple forecasts into a single forecast.

    See docs/superpowers/specs/2026-05-13-ensemble-safeguards-design.md
    for the full contract.

    ``**kwargs`` are forwarded to the strategy's ``combine()`` (and, where
    possible, ``fit()``) to preserve strategy-specific extras like
    ``hindcasts=`` (bma), ``n_drop=`` (drop_worst), and ``scores=``
    (skill_weighted, drop_worst).
    """
    if not forecasts:
        raise ValueError("forecasts list cannot be empty")
    safeguards = _resolve_safeguards(safeguards)
    strat = get_strategy(strategy)()
    member_names = _member_names(forecasts)

    if not optimize_ensemble:
        combined = strat.combine(forecasts, obs=obs, **kwargs)
        # Weights for the EnsembleResult report. Most strategies' fit() needs
        # obs (and year-dim forecasts) to evaluate per-member skill; when obs
        # is None or fit() can't consume the forwarded kwargs, fall back to
        # uniform weights for the report — the combined forecast is still
        # produced by combine(). Narrow except: (TypeError, ValueError) covers
        # the documented kwarg-mismatch and shape-mismatch cases; other
        # exceptions are real bugs and should propagate.
        if obs is None:
            w = _uniform_weights(len(forecasts))
        else:
            try:
                w = strat.fit(forecasts, obs, primary_metric=primary_metric, **kwargs)
            except (TypeError, ValueError):
                w = _uniform_weights(len(forecasts))
        pev = (
            prediction_error_variance(combined, obs)
            if obs is not None and "year" in combined.dims
            else None
        )
        member_contributions = (
            _compute_member_contributions(forecasts, obs, combined, member_names)
            if obs is not None and "year" in combined.dims
            else None
        )
        return EnsembleResult(
            forecast=combined,
            weights=w,
            member_names=member_names,
            member_cv_skill={},
            effective_n=_effective_n(w),
            gate_passed=True,
            shrinkage_lambda=0.0,
            safeguards_applied={},
            pev=pev,
            member_contributions=member_contributions,
        )

    if obs is None:
        raise ValueError("ensemble(optimize_ensemble=True) requires obs")
    years = list(obs.year.values)
    n_members = len(forecasts)
    diag = {}

    nested_cv = safeguards["nested_cv"]
    diag["nested_cv"] = nested_cv

    use_shrinkage = (
        safeguards["shrinkage"] != 0.0
        and not strat.self_shrinks()
    )
    if strat.self_shrinks() and safeguards["shrinkage"] != 0.0:
        diag["shrinkage_skipped_for_self_shrinking"] = True
    lam = safeguards["shrinkage"] if use_shrinkage else 0.0
    diag["shrinkage"] = lam

    floor_fired = []

    if not nested_cv:
        diag["nested_cv_warning"] = (
            "running unsafeguarded optimisation; OOS skill estimate will be "
            "optimistic"
        )
        warnings.warn(diag["nested_cv_warning"], RuntimeWarning, stacklevel=2)
        w_raw = strat.fit(forecasts, obs, primary_metric=primary_metric)
        w_final = w_raw if not use_shrinkage else _apply_shrinkage(w_raw, lam)
        if _effective_n(w_final) < safeguards["min_effective_n"]:
            w_final = _uniform_weights(n_members)
            floor_fired.append("in_sample_final")
        final_fcst = strat.combine(forecasts, weights=w_final)
        if floor_fired:
            diag["min_effective_n_floor_fired_on"] = floor_fired
        return EnsembleResult(
            forecast=final_fcst,
            weights=w_final,
            member_names=member_names,
            member_cv_skill={
                name: _score(forecasts[i] if not hasattr(forecasts[i], "forecast")
                             else forecasts[i].forecast,
                             obs, primary_metric)
                for i, name in enumerate(member_names)
            },
            effective_n=_effective_n(w_final),
            gate_passed=True,
            shrinkage_lambda=lam,
            safeguards_applied=diag,
            pev=None,
            member_contributions=None,
        )

    # Honest nested-CV path.
    # Pool per-fold OOS predictions then score once per path, matching the
    # pattern in optimize.py. Per-fold scoring is undefined for variance-based
    # metrics like pearson_r when a fold has a single year (e.g. loyo), so
    # accumulating into pooled (year, ...) series and scoring once is the only
    # well-defined choice for arbitrary CV schemes and metrics.
    opt_cv_forecasts = []
    unif_cv_forecasts = []
    cv_obs_list = []
    cv_fn = cv if callable(cv) else get_cv(cv)
    for train_years, test in cv_fn(years):
        test_years = test if isinstance(test, list) else [test]

        # Inner fit on the outer-train years.
        w_raw = strat.fit(forecasts, obs, years=train_years, primary_metric=primary_metric)
        w_shrunk = w_raw if not use_shrinkage else _apply_shrinkage(w_raw, lam)
        if _effective_n(w_shrunk) < safeguards["min_effective_n"]:
            w_outer = _uniform_weights(n_members)
            floor_fired.extend(test_years)
        else:
            w_outer = w_shrunk

        fcst_opt = strat.combine(forecasts, weights=w_outer).sel(year=test_years)
        fcst_unif = strat.combine(
            forecasts, weights=_uniform_weights(n_members),
        ).sel(year=test_years)

        opt_cv_forecasts.append(fcst_opt)
        unif_cv_forecasts.append(fcst_unif)
        cv_obs_list.append(obs.sel(year=test_years))

    # Schemes that overlap (e.g. lko sliding) produce multiple predictions per
    # year; average them into one prediction per year before scoring. Schemes
    # that don't overlap (loyo, blocked) pass through unchanged.
    opt_pooled = xr.concat(opt_cv_forecasts, dim="year").groupby("year").mean("year")
    unif_pooled = xr.concat(unif_cv_forecasts, dim="year").groupby("year").mean("year")
    # obs is identical across overlapping folds — `.first()` is cheaper than
    # `.mean()` and avoids NaN propagation if any obs slice is partial.
    cv_obs_pooled = xr.concat(cv_obs_list, dim="year").groupby("year").first()

    opt_cv = _score(opt_pooled, cv_obs_pooled, primary_metric)
    unif_cv = _score(unif_pooled, cv_obs_pooled, primary_metric)
    diag["gate"] = safeguards["gate"]
    diag["gate_outer_cv"] = opt_cv
    diag["uniform_outer_cv"] = unif_cv
    if floor_fired:
        diag["min_effective_n_floor_fired_on"] = sorted(set(floor_fired))

    gate_passed = (
        opt_cv >= unif_cv if safeguards["gate"]
        else True  # gate disabled → accept optimised path unconditionally
    )

    if gate_passed:
        w_final_raw = strat.fit(forecasts, obs, primary_metric=primary_metric)
        w_final = w_final_raw if not use_shrinkage else _apply_shrinkage(w_final_raw, lam)
        if _effective_n(w_final) < safeguards["min_effective_n"]:
            w_final = _uniform_weights(n_members)
            diag["min_effective_n_floor_fired_on_final"] = True
        final_fcst = strat.combine(forecasts, weights=w_final)
    else:
        w_final = _uniform_weights(n_members)
        final_fcst = strat.combine(forecasts, weights=w_final)
        warnings.warn(
            f"ensemble(optimize_ensemble=True): gate failed "
            f"(optimised CV {opt_cv:.4f} < uniform CV {unif_cv:.4f}); "
            f"returning uniform.",
            RuntimeWarning,
            stacklevel=2,
        )

    pev = prediction_error_variance(opt_pooled, obs.sel(year=opt_pooled.year))
    member_contributions = _compute_member_contributions(
        forecasts, obs.sel(year=opt_pooled.year), opt_pooled, member_names,
    )

    return EnsembleResult(
        forecast=final_fcst,
        weights=w_final,
        member_names=member_names,
        member_cv_skill={
            name: _score(forecasts[i] if not hasattr(forecasts[i], "forecast")
                         else forecasts[i].forecast,
                         obs, primary_metric)
            for i, name in enumerate(member_names)
        },
        effective_n=_effective_n(w_final),
        gate_passed=gate_passed,
        shrinkage_lambda=lam,
        safeguards_applied=diag,
        pev=pev,
        member_contributions=member_contributions,
    )


def pool_ensembles(arrays, *, member_dim="member", regrid_to="first", align_years=True):
    """Pool several per-model ensembles into one predictor cube along the member dimension.

    Each input is one model's ensemble carrying its own ``member_dim``. The members are renumbered
    to a single contiguous range so they stay unique across models, the grids are aligned by linear
    interpolation onto a reference, the shared years are intersected, and the arrays are
    concatenated along ``member_dim``. The result — a single ``(year, member, lat, lon)`` array —
    is the multi-model predictor for :func:`deepscale.optimize`. (``seasonal_mme`` pools internally
    from its ``predictor_tracks``, so it does not need this.)

    Parameters
    ----------
    arrays : sequence of xarray.DataArray
        Per-model ensembles (``None`` entries are skipped). Must contain at least one array.
    member_dim : str, default "member"
        Name of the ensemble dimension.
    regrid_to : "first" | xarray.DataArray | (lat, lon), default "first"
        Reference grid; every array is linearly interpolated onto it (a no-op where the grid
        already matches). ``"first"`` uses the first array's grid.
    align_years : bool, default True
        Intersect the ``year`` coordinate across all arrays before concatenating.
    """
    from ._spatial import spatial_dims as _resolve_spatial

    arrays = [a for a in arrays if a is not None]
    if not arrays:
        raise ValueError("pool_ensembles: no arrays to pool (all None or empty sequence).")

    lat_dim, lon_dim = _resolve_spatial(arrays[0], context="pool_ensembles")
    if regrid_to == "first":
        ref_lat, ref_lon = arrays[0][lat_dim], arrays[0][lon_dim]
    elif isinstance(regrid_to, xr.DataArray):
        rlat, rlon = _resolve_spatial(regrid_to, context="pool_ensembles regrid_to")
        ref_lat, ref_lon = regrid_to[rlat], regrid_to[rlon]
    else:
        ref_lat, ref_lon = regrid_to

    # Compare by coordinate *values*: when regrid_to is a (lat, lon) tuple of raw
    # arrays, `DataArray.equals(<ndarray>)` is always False, so without this every
    # array would be needlessly re-interpolated even when already on the grid.
    ref_lat_v = np.asarray(ref_lat.values if isinstance(ref_lat, xr.DataArray) else ref_lat)
    ref_lon_v = np.asarray(ref_lon.values if isinstance(ref_lon, xr.DataArray) else ref_lon)

    aligned = []
    for a in arrays:
        alat, alon = _resolve_spatial(a, context="pool_ensembles")
        if not (np.array_equal(a[alat].values, ref_lat_v)
                and np.array_equal(a[alon].values, ref_lon_v)):
            a = a.interp({alat: ref_lat_v, alon: ref_lon_v})
        aligned.append(a)

    if align_years:
        years = sorted(set.intersection(*[set(a["year"].values.tolist()) for a in aligned]))
        if not years:
            raise ValueError("pool_ensembles: the arrays share no common years.")
        aligned = [a.sel(year=years) for a in aligned]

    renumbered, offset = [], 0
    for a in aligned:
        n = a.sizes[member_dim]
        a = a.assign_coords({member_dim: np.arange(offset, offset + n)})
        renumbered.append(a)
        offset += n
    return xr.concat(renumbered, dim=member_dim)
