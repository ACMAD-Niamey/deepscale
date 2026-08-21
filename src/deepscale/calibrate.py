"""Calibrate family: probabilistic post-processing to tercile probabilities.

``calibrate()`` is the verb for methods that turn a predictor into below/normal/
above tercile probabilities *without changing resolution* — Model Output
Statistics / calibration, as distinct from ``downscale()`` (CCA, BCSD, CorrDiff)
which maps a coarse predictor onto a finer grid.

    p = calibrate(predictor, obs, method="ereg", forecast_year=2024)
    p = calibrate(index_per_model, obs, method="logit", forecast=fcst_index)

Both registered methods are multi-model: the predictor is a ``{model: ...}``
dict, each model is calibrated independently to its own tercile probabilities,
and the per-model maps are combined (default: cross-model average). The result
is an ``(tercile, lat, lon)`` DataArray, ``tercile=[0,1,2]`` = below/normal/above.

Registered methods:

- ``ereg`` — ensemble regression. Per model, per-grid-cell OLS of obs on the
  ensemble-mean hindcast, then parametric (Gaussian, prediction-error-variance)
  tercile probabilities; averaged across models.
- ``logit`` — logistic calibration. Per model, per-grid-cell logistic of tercile
  occurrence on a scalar predictor index (e.g. the WVG SST index); averaged
  across models. Supports independent-binomial and multinomial formulations,
  detrend (via the index), and a significance mask.
- ``smoothed_regression`` — Kharin et al. (2017) seasonally-smoothed coefficient
  postprocessing (season-aware, deterministic or tercile output). Multi-model
  dicts are pooled into ONE super-ensemble rather than averaged per model, and
  the hindcast fit can target a hindcast year (``forecast_year=``) or an
  out-of-sample forecast ensemble (``forecast=``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .indices import Index
from .registry import get_calibrator, register_calibrator
from ._warnings import quiet_all_nan_slices


@dataclass(frozen=True)
class LogitConfig:
    """Configuration for gridded-predictor logistic calibration.

    ``index`` reduces the gridded predictor hindcast/forecast to scalar index
    values. The resulting index series is passed to the same registered
    ``method="logit"`` engine used by the lower-level precomputed-index API.
    """

    index: Index
    model: str = "independent_binomial"
    predictor_level: str = "model_mean"
    detrend: bool = False
    significance: float | None = None
    regularization: float | None = None
    backend: str | None = None
    min_years: int = 10


@quiet_all_nan_slices
def calibrate(predictor=None, obs=None, *, method, forecast=None,
              forecast_year=None, predictor_hindcast=None,
              predictor_forecast=None, combine="mean", output_type="tercile",
              return_components=False, cv=None, cv_window=1,
              verbose=False, **method_kwargs):
    """Calibrate a predictor to tercile probabilities (no resolution change).

    Parameters
    ----------
    predictor
        Method-specific. ``ereg``: ``{model: (hindcast, forecast)}`` (gridded,
        on the obs grid) or a single ``(hindcast, forecast)``. ``logit``:
        ``{model: index_series}`` or a single index series.
    obs : xr.DataArray
        Predictand ``(year, lat, lon)``; sets the tercile boundaries.
    method : {"ereg", "logit"}
        Calibrate-family method name.
    forecast, forecast_year
        Method-specific forecast selector (see the registered methods).
    combine : str
        Cross-model combination of the per-model tercile maps ("mean").
    output_type : str
        ``"tercile"`` (default) preserves the existing below/normal/above
        probability output for every method. Methods that additionally
        support deterministic output (``supports_deterministic = True``:
        ``ereg`` — the calibrated rainfall field — and
        ``smoothed_regression``) accept other values; requesting a non-
        "tercile" ``output_type`` from a method that doesn't support it
        raises ``ValueError``.
    return_components : bool
        When True (``ereg``/``logit`` only), return a :class:`CalibrateResult`
        carrying both the combined map and each model's own post-calibration
        map — so members can be inspected and a subset recombined (e.g. with
        :func:`deepscale.combine_terciles`) without refitting.
    cv : {None, "loyo"}
        ``"loyo"`` (``ereg``/``logit``, tercile output only) returns the
        leave-year(s)-out CROSS-VALIDATED hindcast instead of the forecast:
        for every obs year, the model is refit without that year (window
        ``cv_window``) and the held-out year is predicted. The result gains a
        leading ``year`` dimension and is ready for :func:`deepscale.skill`.
        ``forecast``/``forecast_year`` are ignored in this mode. Honest but
        brute-force: cost is ~n_years × one full calibration — prefer a
        coarser ``obs`` grid for skill maps.
    cv_window : int
        Years left out per fold (1 = strict leave-one-out; 5 matches the
        PyCPT ``crossvalidation_window`` convention).

    Returns
    -------
    xr.DataArray or CalibrateResult
        ``(tercile, lat, lon)`` probabilities, ``tercile=[0,1,2]``, summing to 1
        per cell where defined (``output_type="tercile"``), or a method-
        specific deterministic output otherwise. With ``cv=`` a leading
        ``year`` dim of held-out predictions; with ``return_components=True``
        a :class:`CalibrateResult` instead of the bare combined array.
    """
    if predictor is None and predictor_hindcast is not None:
        predictor = predictor_hindcast
    if forecast is None and predictor_forecast is not None:
        forecast = predictor_forecast

    _component_methods = ("ereg", "logit")
    if (return_components or cv is not None) and not (
            isinstance(method, str) and method in _component_methods):
        raise ValueError(
            "calibrate: return_components/cv are supported for methods "
            f"{_component_methods}; got method={method!r}."
        )
    if cv is not None:
        if cv != "loyo":
            raise ValueError(f"calibrate: cv must be None or 'loyo'; got {cv!r}.")
        if output_type != "tercile":
            raise ValueError(
                "calibrate: cv='loyo' currently supports output_type='tercile' only."
            )
        return _calibrate_cv(
            method, predictor, obs, forecast=forecast, combine=combine,
            cv_window=cv_window, return_components=return_components,
            verbose=verbose, method_kwargs=method_kwargs,
        )
    if return_components:
        method_kwargs = dict(method_kwargs, return_components=True)

    if isinstance(method, LogitConfig):
        if output_type != "tercile":
            raise ValueError(
                f"LogitConfig calibration is tercile-only; got output_type={output_type!r}."
            )
        return _calibrate_logit_config(
            predictor, obs, forecast,
            config=method, forecast_year=forecast_year, combine=combine,
            verbose=verbose, **method_kwargs,
        )

    fn = get_calibrator(method)
    if output_type != "tercile" and not getattr(fn, "supports_deterministic", False):
        raise ValueError(
            f"method {method!r} does not support output_type={output_type!r} "
            "(only 'tercile')."
        )
    return fn(predictor, obs, forecast=forecast, forecast_year=forecast_year,
              combine=combine, output_type=output_type, verbose=verbose, **method_kwargs)


from dataclasses import dataclass, field as _dc_field


@dataclass
class CalibrateResult:
    """Outcome of ``calibrate(..., return_components=True)`` (and/or ``cv=``).

    ``combined`` is exactly what the plain call returns (the cross-model
    combination); ``per_model`` maps each input model name to ITS calibrated
    field — post-calibration tercile probabilities (or the deterministic field
    under ``output_type="deterministic"``), so a forecaster can inspect the
    members, drop models, and recombine a subset with
    :func:`deepscale.combine_terciles` without refitting anything.

    Under ``cv=`` the same fields hold the cross-validated hindcast instead
    (a leading ``year`` dimension of held-out predictions), ready for
    :func:`deepscale.skill`.
    """
    combined: "xr.DataArray"
    per_model: dict = _dc_field(default_factory=dict)


def _as_model_dict(predictor):
    """Normalize predictor into a ``{model: value}`` dict."""
    if isinstance(predictor, dict):
        return predictor
    return {"model": predictor}


def _combine_deterministic(maps: dict) -> xr.DataArray:
    """Equal-weight cross-model mean of deterministic (lat, lon[, year]) fields."""
    items = [m for m in maps.values() if m is not None]
    if not items:
        raise ValueError("calibrate: no model produced a deterministic forecast.")
    out = items[0] if len(items) == 1 else xr.concat(items, dim="model").mean("model")
    out.attrs.update(n_models=len(items))
    return out


def _combine_models(maps: dict, combine: str) -> xr.DataArray:
    """Combine per-model ``(tercile, lat, lon)`` maps. Default: cross-model mean.

    Each per-model map sums to 1 (or is all-NaN where that model can't
    calibrate); a skipna mean therefore also sums to 1 wherever at least one
    model is defined.
    """
    items = [m for m in maps.values() if m is not None]
    if not items:
        raise ValueError("calibrate: no model produced a tercile forecast.")
    if combine != "mean":
        raise ValueError(f"calibrate: unknown combine={combine!r} (expected 'mean').")
    if len(items) == 1:
        out = items[0]
    else:
        out = xr.concat(items, dim="model").mean("model")  # skipna
    out = out.transpose("tercile", ...)
    out = _probability_simplex_or_nan(out)
    out = out.transpose("tercile", ...)
    out.attrs.update(combine=combine, n_models=len(items))
    return out


def _probability_simplex_or_nan(probs: xr.DataArray) -> xr.DataArray:
    """Keep each cell as a complete finite tercile simplex, or mask it out."""
    finite = np.isfinite(probs)
    complete = finite.all("tercile")
    total = probs.sum("tercile", skipna=False)
    valid = complete & np.isfinite(total) & (total > 0)
    return xr.where(valid, probs / total, np.nan)


def _detrend_index(index: xr.DataArray, forecast_index, forecast_year=None):
    """Remove a linear trend from hindcast and forecast index values.

    The trend is fitted over the hindcast index using the ``year`` coordinate
    when present, otherwise positional years. The forecast value is adjusted by
    the same fitted line at its forecast year, resolved in priority order:
    an explicit ``forecast_year``; else the ``forecast_index``'s own ``year``
    coordinate; else one step after the hindcast period. The explicit argument
    matters for a bare-scalar forecast index (no year coordinate), where the
    one-step fallback would otherwise silently evaluate the trend at the wrong
    year whenever the real forecast year is not ``years[-1] + 1``.
    """
    x = np.asarray(index, dtype=float).reshape(-1)
    if "year" in index.coords:
        years = np.asarray(index["year"].values, dtype=float).reshape(-1)
    else:
        years = np.arange(x.size, dtype=float)
    ok = np.isfinite(years) & np.isfinite(x)
    if ok.sum() < 2 or np.unique(years[ok]).size < 2:
        return index, forecast_index
    slope, intercept = np.polyfit(years[ok], x[ok], deg=1)
    trend = xr.DataArray(
        slope * years + intercept,
        dims=index.dims,
        coords=index.coords,
    )
    detrended = index - trend

    if forecast_year is not None:
        fyear = float(forecast_year)
    elif isinstance(forecast_index, xr.DataArray) and "year" in forecast_index.coords:
        fyear = float(np.asarray(forecast_index["year"].values).reshape(-1)[0])
    else:
        fyear = float(years[-1] + 1.0)
    ftrend = slope * fyear + intercept
    return detrended, forecast_index - ftrend


def _calibrate_logit_config(
    predictor_hindcast,
    obs,
    predictor_forecast,
    *,
    config: LogitConfig,
    forecast_year=None,
    combine="mean",
    verbose=False,
    **overrides,
):
    """Reduce gridded predictors through ``config.index`` and run logit."""
    if overrides:
        unknown = set(overrides) - set(LogitConfig.__dataclass_fields__)
        if unknown:
            raise TypeError(
                "calibrate(LogitConfig) got unexpected keyword argument(s): "
                f"{sorted(unknown)}."
            )
        values = {field: getattr(config, field) for field in LogitConfig.__dataclass_fields__}
        values.update(overrides)
        config = LogitConfig(**values)

    if predictor_hindcast is None:
        raise ValueError(
            "calibrate(LogitConfig) requires predictor_hindcast or predictor."
        )
    if predictor_forecast is None:
        raise ValueError(
            "calibrate(LogitConfig) requires predictor_forecast or forecast."
        )
    if obs is None:
        raise ValueError("calibrate(LogitConfig) requires obs.")
    if config.predictor_level != "model_mean":
        raise ValueError(
            "LogitConfig.predictor_level currently supports only 'model_mean'."
        )

    hindcasts = _as_model_dict(predictor_hindcast)
    forecasts = _as_model_dict(predictor_forecast)
    if set(hindcasts) != set(forecasts):
        raise ValueError(
            "calibrate(LogitConfig): predictor_forecast keys "
            f"{sorted(forecasts)} must match predictor_hindcast keys "
            f"{sorted(hindcasts)}."
        )

    idx = {}
    fc = {}
    for name, hcst in hindcasts.items():
        forecast_field = _select_single_forecast(
            forecasts[name], forecast_year, context=f"calibrate(LogitConfig) model {name!r}"
        )
        index = config.index.reduce(hcst)
        forecast_index = config.index.reduce(forecast_field, climatology=hcst)
        forecast_index = _require_single_value(
            forecast_index, f"calibrate(LogitConfig) forecast index for model {name!r}"
        )
        if config.detrend:
            index, forecast_index = _detrend_index(index, forecast_index)
        idx[name] = index
        fc[name] = forecast_index
        if verbose:
            print(f"[calibrate:logit] {name}: reduced predictor via {config.index.name}")

    backend = config.backend
    if backend is None:
        backend = "statsmodels" if config.significance is not None else "sklearn"

    return _calibrate_logit(
        idx, obs,
        forecast=fc,
        combine=combine,
        model=config.model,
        backend=backend,
        regularization=config.regularization,
        significance_mask=config.significance,
        min_years=config.min_years,
        verbose=verbose,
    )


def _require_single_value(value, label: str):
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"{label} must contain exactly one value; got {arr.size}.")
    return value


def _select_single_forecast(forecast, forecast_year, *, context: str):
    """Select one forecast year from a DataArray, or validate an already scalar field."""
    if not isinstance(forecast, xr.DataArray) or "year" not in forecast.dims:
        return forecast
    if forecast_year is not None:
        if forecast_year not in set(forecast.year.values.tolist()):
            raise ValueError(
                f"{context}: forecast_year={forecast_year} is not available in "
                "the provided forecast."
            )
        return forecast.sel(year=[forecast_year])
    if forecast.sizes["year"] != 1:
        raise ValueError(
            f"{context}: forecast_year=None requires the provided forecast to "
            "contain exactly one year, or pass forecast_year=..."
        )
    return forecast


def _select_forecast_year_slice(source, forecast_year):
    """Return the single-year forecast slice from ``source`` for ``forecast_year``,
    whether ``year`` is a dimension to index into or must be attached to a
    year-less field. Shared by the ereg forecast-input resolution paths."""
    if "year" in source.dims:
        return source.sel(year=[forecast_year])
    return source.expand_dims(year=[forecast_year])


def _split_ereg_predictor(predictor, forecast, method="ereg"):
    """Return {model: (hindcast, forecast)} from pair or alias-style inputs."""
    predictors = _as_model_dict(predictor)
    forecasts = None if forecast is None else _as_model_dict(forecast)

    if forecasts is not None and set(predictors) != set(forecasts):
        raise ValueError(
            f"calibrate(method={method!r}): forecast keys {sorted(forecasts)} must "
            f"match predictor model keys {sorted(predictors)}."
        )

    pairs = {}
    for name, value in predictors.items():
        if isinstance(value, tuple) and len(value) == 2:
            hcst, embedded_fcst = value
        elif isinstance(value, tuple):
            raise ValueError(
                f"calibrate(method={method!r}) predictor values must be "
                "(hindcast, forecast) pairs."
            )
        else:
            hcst, embedded_fcst = value, None
        fcst = forecasts[name] if forecasts is not None else embedded_fcst
        pairs[name] = (hcst, fcst)
    return pairs


def _resolve_ereg_forecast_year(models, obs, forecast_year):
    if forecast_year is not None:
        missing = []
        for name, (hcst, fcst) in models.items():
            source = fcst if fcst is not None else hcst
            if "year" in source.dims and forecast_year not in set(source.year.values.tolist()):
                missing.append(name)
        if missing:
            raise ValueError(
                f"calibrate(method='ereg'): forecast_year={forecast_year} is "
                f"not available in models {missing}."
            )
        return int(forecast_year)

    provided = {name: fcst is not None for name, (_hcst, fcst) in models.items()}
    if all(provided.values()):
        years = {}
        for name, (_hcst, fcst) in models.items():
            if "year" not in fcst.dims:
                continue
            if fcst.sizes["year"] != 1:
                raise ValueError(
                    "calibrate(method='ereg'): forecast_year=None requires every "
                    "provided forecast to contain exactly one year. Pass "
                    "forecast_year=..."
                )
            years[name] = int(fcst.year.values[0])
        unique = set(years.values())
        if len(unique) == 1:
            return unique.pop()
        if len(unique) > 1:
            raise ValueError(
                "calibrate(method='ereg'): forecast_year=None and forecast "
                f"slices have different years: {years}. Pass forecast_year=..."
            )
        raise ValueError(
            "calibrate(method='ereg'): forecast_year=None cannot infer a year "
            "from forecasts without a year dimension. Pass forecast_year=..."
        )

    if any(provided.values()):
        raise ValueError(
            "calibrate(method='ereg'): forecast_year=None with mixed provided "
            "and missing forecasts. Pass forecast_year=..."
        )
    return int(np.max(obs.year.values))


def _common_obs_hindcast_years(hcst, obs, *, name):
    hcst_years = set(hcst.year.values.tolist())
    obs_years = set(obs.year.values.tolist())
    years = [int(y) for y in obs.year.values.tolist() if y in hcst_years]
    if not years:
        raise ValueError(
            f"calibrate(method='ereg') model {name!r}: hindcast and obs have no "
            "overlapping years."
        )
    if len(years) < len(obs_years):
        missing = sorted(obs_years - hcst_years)
        raise ValueError(
            f"calibrate(method='ereg') model {name!r}: hindcast is missing obs "
            f"years {missing}."
        )
    return years


_NATIVE_YEARS_MIN_OVERLAP = 3


def _native_obs_hindcast_years(hcst, obs, *, name):
    """native_years=True: trim to each model's own hcst ∩ obs overlap instead
    of requiring the hindcast to cover every obs year. Matches the consumer's
    ``calibrate_ereg_native_years`` (rosetta_deepscale.run_pipeline), which
    floors at 3 overlapping years."""
    obs_years = set(obs.year.values.tolist())
    years = [int(y) for y in hcst.year.values.tolist() if y in obs_years]
    if len(years) < _NATIVE_YEARS_MIN_OVERLAP:
        raise ValueError(
            f"calibrate(method='ereg') model {name!r}: fewer than "
            f"{_NATIVE_YEARS_MIN_OVERLAP} overlapping hindcast/obs years "
            f"(got {len(years)})."
        )
    return years


@register_calibrator("ereg")
def _calibrate_ereg(predictor, obs, *, forecast=None, forecast_year=None,
                    combine="mean", clip_negative=False, threshold_source="obs",
                    native_years: bool = False, output_type="tercile",
                    return_components=False, verbose=False, **_):
    """eReg calibration: per-model OLS(obs ~ ens-mean) → parametric terciles →
    cross-model average. Each model's predictor is ``(hindcast, forecast)`` with
    the GCM already on the obs grid. ``forecast_year`` selects the year; with no
    provided forecast it defaults to the maximum obs year.

    ``native_years`` (opt-in, default False): when True, each model is
    calibrated on its OWN ``hcst.year ∩ obs.year`` overlap (floor 3 years)
    instead of requiring every model's hindcast to cover every obs year.
    Default False leaves this byte-for-byte unchanged (still raises on any
    missing obs year)."""
    from .methods.ensemble_regression import EnsembleRegressionMethod

    models = _split_ereg_predictor(predictor, forecast)
    forecast_year = _resolve_ereg_forecast_year(models, obs, forecast_year)

    maps = {}
    for name, pair in models.items():
        hcst, fcst = pair
        if native_years:
            years = _native_obs_hindcast_years(hcst, obs, name=name)
        else:
            years = _common_obs_hindcast_years(hcst, obs, name=name)
        m = EnsembleRegressionMethod(clip_negative=clip_negative)
        m.fit(hcst.sel(year=years), obs.sel(year=years))
        fc = _select_forecast_year_slice(
            fcst if fcst is not None else hcst, forecast_year)
        if output_type == "deterministic":
            # the calibrated deterministic rainfall for this model — the
            # OLS-corrected ensemble-mean field behind the terciles.
            det = m.predict(fc)
            maps[name] = det.squeeze("year", drop=True) if "year" in det.dims else det
        else:
            # native_years=False: years == every obs year (enforced by
            # _common_obs_hindcast_years), so obs.sel(year=years) == obs; pass
            # obs unchanged to keep this path byte-for-byte identical to before.
            # native_years=True: years is this model's own (possibly narrower)
            # overlap, so the climatology reference must be trimmed to match —
            # this is what the consumer (calibrate_ereg_native_years) does.
            obs_climatology = obs.sel(year=years) if native_years else obs
            maps[name] = m.predict_tercile(
                fc, obs_climatology, threshold_source=threshold_source)
        if verbose:
            print(f"[calibrate:ereg] {name}: calibrated")
    if output_type == "deterministic":
        out = _combine_deterministic(maps)
    else:
        out = _combine_models(maps, combine)
    out.attrs.update(method="ereg", forecast_year=int(forecast_year),
                     output_type=output_type)
    if return_components:
        return CalibrateResult(combined=out, per_model=maps)
    return out


_calibrate_ereg.supports_deterministic = True


def _calibrate_cv(method, predictor, obs, *, forecast=None, combine="mean",
                  cv_window=1, return_components=False, verbose=False,
                  method_kwargs=None):
    """Leave-year(s)-out cross-validated hindcast for the calibrate family.

    For every obs year: refit on the remaining years (window ``cv_window``)
    and predict the held-out year from that fold's fit — the same fold logic
    as ``seasonal_mme``'s CCA path, applied to ``ereg``/``logit``. Honest
    (the held-out year never enters the fit, the tercile thresholds, or the
    detrend line) but brute-force: cost is ~n_years x one full calibration.

    Models missing a fold's test year are skipped for that fold (their maps
    are NaN there after the outer-join concat), mirroring ``native_years``.
    """
    from .cv import loyo

    method_kwargs = dict(method_kwargs or {})
    method_kwargs.pop("return_components", None)
    years = [int(y) for y in obs.year.values]

    if method == "ereg":
        models = _split_ereg_predictor(predictor, forecast)
    else:
        models = {name: series for name, series in _as_model_dict(predictor).items()}

    fold_combined = []
    fold_per = {name: [] for name in models}
    for train_years, test_year in loyo(years, window=cv_window):
        if method == "ereg":
            avail = {n: (h, None) for n, (h, _f) in models.items()
                     if test_year in set(h.year.values.tolist())}
            fc = {n: h.sel(year=[test_year]) for n, (h, _f) in avail.items()}
        else:
            avail = {n: s.sel(year=[y for y in s.year.values if int(y) != test_year
                                    and int(y) in train_years])
                     for n, s in models.items()
                     if test_year in set(s.year.values.tolist())}
            fc = {n: models[n].sel(year=test_year)
                  for n in avail}
        if not avail:
            continue
        fn = get_calibrator(method)
        res = fn(avail, obs.sel(year=[y for y in train_years]),
                 forecast=fc, forecast_year=test_year, combine=combine,
                 return_components=True, verbose=False, **method_kwargs)
        fold_combined.append(res.combined.expand_dims(year=[test_year]))
        for n, m in res.per_model.items():
            fold_per[n].append(m.expand_dims(year=[test_year]))
        if verbose:
            print(f"[calibrate:cv] {test_year}: {len(avail)} model(s)")

    if not fold_combined:
        raise ValueError("calibrate(cv='loyo'): no fold produced a forecast.")
    combined = xr.concat(fold_combined, dim="year").sortby("year")
    combined.attrs.update(cv="loyo", cv_window=int(cv_window), method=method)
    if not return_components:
        return combined
    per_model = {n: xr.concat(parts, dim="year").sortby("year")
                 for n, parts in fold_per.items() if parts}
    return CalibrateResult(combined=combined, per_model=per_model)


@register_calibrator("logit")
def _calibrate_logit(predictor, obs, *, forecast=None, forecast_year=None,
                     combine="mean", model="independent_binomial", backend="sklearn",
                     regularization=None, significance_mask=None, min_years=10,
                     tercile_edges: str = "exclusive",
                     detrend: bool = False, return_components=False,
                     verbose=False, **_):
    """Logistic calibration: per-model per-cell logistic of tercile occurrence on
    a scalar index → cross-model average. ``predictor`` is ``{model: index}`` and
    ``forecast`` the matching ``{model: index_value}`` (or single values).

    ``tercile_edges`` (opt-in, default "exclusive"): how boundary-tied obs
    values are classified into below/normal/above; see
    ``deepscale.logistic._labels_from_obs``. Default reproduces the standard
    tercile definition / legacy behavior; "inclusive" helps dry/tied cells."""
    from .logistic import logistic_forecast

    idx = _as_model_dict(predictor)
    fc = _as_model_dict(forecast) if forecast is not None else None
    if fc is None:
        raise ValueError(
            "calibrate(method='logit') requires forecast=<index value(s)>: the "
            "predictor index for the forecast year, per model."
        )

    if set(idx) != set(fc):
        raise ValueError(
            f"calibrate(method='logit'): forecast keys {sorted(fc)} must match "
            f"predictor model keys {sorted(idx)}."
        )
    maps = {}
    for name, index in idx.items():
        _require_single_value(fc[name], f"calibrate(method='logit') forecast for model {name!r}")
        fval = float(np.asarray(fc[name]).reshape(-1)[0])
        if detrend:
            index, fval = _detrend_index(index, fc[name], forecast_year=forecast_year)
            fval = float(np.asarray(fval).reshape(-1)[0])
        maps[name] = logistic_forecast(
            index, obs, fval, model=model, backend=backend,
            regularization=regularization, significance_mask=significance_mask,
            min_years=min_years, tercile_edges=tercile_edges,
        )
        if verbose:
            print(f"[calibrate:logit] {name}: fit on index")
    out = _combine_models(maps, combine)
    out.attrs.update(method="logit", model=model, backend=backend)
    if return_components:
        return CalibrateResult(combined=out, per_model=maps)
    return out


def _concat_super_ensemble(ensembles):
    """Concat DataArrays along ``member`` with globally unique integer member ids
    (a member-less entry contributes one member). Order follows the input order."""
    reindexed, offset = [], 0
    for e in ensembles:
        if "member" in e.dims:
            n = e.sizes["member"]
            e = e.assign_coords(member=list(range(offset, offset + n)))
        else:
            n = 1
            e = e.expand_dims(member=[offset])
        offset += n
        reindexed.append(e)
    return xr.concat(reindexed, dim="member")


def _normalize_smoothed_forecast(forecast):
    """Strip the (at most singleton) ``year`` dim/coord from an out-of-sample
    forecast field so it broadcasts as a plain (season, [member,] lat, lon) target."""
    if "year" in forecast.dims:
        if forecast.sizes["year"] != 1:
            raise ValueError(
                "smoothed_regression: `forecast` must be a single target field "
                "(season, [member,] lat, lon), optionally with a length-1 year dim; "
                f"got a year dim of size {forecast.sizes['year']}."
            )
        forecast = forecast.isel(year=0, drop=True)
    if "year" in forecast.coords:
        forecast = forecast.drop_vars("year")
    return forecast


def _pool_smoothed_super_ensemble(predictor, forecast):
    """Resolve smoothed_regression's multi-model entry shapes into a single pooled
    ``(hindcast, forecast)`` pair.

    Accepts the ereg-style shapes — ``{model: (hindcast, forecast)}``,
    ``{model: hindcast}`` with ``forecast=None`` or ``{model: forecast}``, or a bare
    ``(hindcast, forecast)`` tuple — and pools members across models into one
    super-ensemble with reindexed member ids (hindcast years intersected across
    models first). A bare DataArray predictor passes through unchanged."""
    if isinstance(predictor, xr.DataArray) and not isinstance(forecast, dict):
        return predictor, forecast
    models = _split_ereg_predictor(predictor, forecast, method="smoothed_regression")

    with_fcst = [name for name, (_h, f) in models.items() if f is not None]
    if with_fcst and len(with_fcst) != len(models):
        missing = sorted(set(models) - set(with_fcst))
        raise ValueError(
            f"calibrate(method='smoothed_regression'): models {missing} have no "
            "forecast — the pooled super-ensemble needs a forecast from every model "
            "(or from none, for hindcast-mode `forecast_year` targeting)."
        )

    hindcasts = [h for h, _f in models.values()]
    common = hindcasts[0]["year"].values
    for h in hindcasts[1:]:
        common = np.intersect1d(common, h["year"].values)
    if common.size == 0:
        raise ValueError(
            "calibrate(method='smoothed_regression'): the model hindcasts share "
            "no common years — the pooled super-ensemble needs an overlapping "
            "hindcast window."
        )
    pooled_h = _concat_super_ensemble([h.sel(year=common) for h in hindcasts])
    pooled_f = None
    if with_fcst:
        pooled_f = _concat_super_ensemble(
            [_normalize_smoothed_forecast(f) for _h, f in models.values()])
    return pooled_h, pooled_f


@register_calibrator("smoothed_regression")
def _calibrate_smoothed_regression(predictor, obs, *, forecast=None, forecast_year=None,
                                   combine="mean", output_type="deterministic",
                                   temporal_sigma=None, distribution="normal",
                                   constrained=True, verbose=False, **_):
    """Kharin et al. (2017) postprocessing. predictor: ensemble hindcast
    (season, year, member, lat, lon); obs: (season, year, lat, lon), same grid.
    temporal_sigma: None (per-season) | float (Gaussian smoothing) | 'constant' (one
    coefficient for the year; for output_type="tercile" this is the mean of the
    per-season a/b coefficients, matching the Kharin et al. (2017) probabilistic
    experiment — deliberately different from the deterministic path's pooled-regression
    'constant', see `methods.smoothed_regression.smooth_ab`).

    output_type="deterministic" (default) returns the calibrated forecast (season, lat,
    lon) for the resolved forecast year (mean-scaling only, ``a * anomaly``).

    output_type="tercile" returns below/near/above-normal category probabilities
    (season, tercile, lat, lon), tercile=[0,1,2], summing to 1, for the resolved forecast
    year. Mean AND spread are scaled (a, b) and turned into a Gaussian predictive
    distribution against per-(season,lat,lon) tercile boundaries:
      - distribution="normal" (default): boundaries are the 33.3rd/66.7th percentiles of
        the observed anomaly, per (season, lat, lon).
      - distribution="gamma": intended for non-negative variables (e.g. precipitation). A
        method-of-moments gamma is fit to the obs per (season, lat, lon); every ensemble
        member and the obs are mapped through the gamma CDF into standard-normal space
        (`methods.smoothed_regression.gamma_to_normal`) before the ensemble mean/spread,
        anomalies, and (a, b) fit are computed — matching the Kharin et al. (2017)
        probabilistic experiment (`prob_realdata/crpss_reliability.py`,
        `fig2_fig3_illustrative.py`). Tercile boundaries are the empirical 33.3rd/66.7th
        percentiles of the (transformed) observed anomaly, the same as the normal path:
        the gamma-transformed obs are only approximately standard normal after the
        method-of-moments fit, so the empirical percentiles are the faithful choice.
    ``constrained`` (tercile only) is passed to
    `methods.smoothed_regression.fit_ab_field`: True (default) sizes the spread
    coefficient analytically; False numerically minimizes mean Gaussian CRPS.

    Two targeting modes (mutually exclusive):

    - ``forecast_year=`` (hindcast-mode, like ereg's retro-forecasts): the target year
      must be one of the years present in `predictor`, and the calibrated field is the
      re-scaled hindcast for that year.
    - ``forecast=`` (out-of-sample, round 2 / issue #5): a separate forecast ensemble
      ``(season, member, lat, lon)`` — member optional for deterministic output,
      required for tercile — not in the obs years (a singleton ``year`` dim is
      tolerated and squeezed). The (a, b) fit, climatologies, tercile boundaries, and
      gamma parameters all come from the hindcast; only the application target changes.

    The predictor and obs are aligned to their common years before fitting, so a
    hindcast carrying extra years (e.g. beyond the obs record) is trimmed rather than
    raising a broadcast error.

    Multi-model entry (pooled super-ensemble): `predictor` may also be an ereg-style
    ``{model: (hindcast, forecast)}`` dict (or ``{model: hindcast}`` with
    ``forecast={model: forecast}``). Members are pooled across models with reindexed
    member ids into ONE super-ensemble — the Kharin experiment design — not calibrated
    per model and averaged like ereg/logit.

    Cross-validated scoring is a caller concern."""
    if output_type not in ("deterministic", "tercile"):
        raise NotImplementedError(
            f"smoothed_regression: unknown output_type={output_type!r}. Supported "
            "output_type values are 'deterministic' and 'tercile'."
        )
    predictor, forecast = _pool_smoothed_super_ensemble(predictor, forecast)
    if forecast is not None and forecast_year is not None:
        # after pooling, so forecasts embedded in {model: (hindcast, forecast)}
        # tuples conflict too instead of silently shadowing forecast_year
        raise ValueError(
            "smoothed_regression: `forecast` and `forecast_year` are mutually "
            "exclusive — pass `forecast_year` to target a hindcast year (with "
            "hindcast-only predictors), or `forecast` to apply the hindcast fit "
            "to an out-of-sample ensemble."
        )
    if forecast is not None:
        forecast = _normalize_smoothed_forecast(forecast)

    common_years = np.intersect1d(predictor["year"].values, obs["year"].values)
    if common_years.size == 0:
        raise ValueError(
            "smoothed_regression: predictor hindcast and obs share no common years."
        )
    if predictor.sizes["year"] != common_years.size:
        predictor = predictor.sel(year=common_years)
    if obs.sizes["year"] != common_years.size:
        obs = obs.sel(year=common_years)

    if output_type == "deterministic":
        from .methods.smoothed_regression import seasonal_coefficients
        a = seasonal_coefficients(predictor, obs, temporal_sigma=temporal_sigma)  # (season,lat,lon)
        fbar = predictor.mean("member") if "member" in predictor.dims else predictor
        f_clim = fbar.mean("year")
        o_clim = obs.mean("year")
        if forecast is not None:
            f_target = forecast.mean("member") if "member" in forecast.dims else forecast
        else:
            year = int(forecast_year) if forecast_year is not None else int(fbar["year"].max())
            hindcast_years = set(int(y) for y in fbar["year"].values)
            if year not in hindcast_years:
                raise ValueError(
                    f"forecast_year={year} is not in the hindcast years "
                    f"[{min(hindcast_years)}..{max(hindcast_years)}]. Pass the target's "
                    "ensemble via `forecast=` to apply the hindcast fit out-of-sample."
                )
            f_target = fbar.sel(year=year)
        calibrated = o_clim + a * (f_target - f_clim)          # (season, lat, lon)
        calibrated.attrs["method"] = "smoothed_regression"
        calibrated.attrs["temporal_sigma"] = "None" if temporal_sigma is None else str(temporal_sigma)
        calibrated.attrs["out_of_sample"] = str(forecast is not None)
        return calibrated

    return _calibrate_smoothed_regression_tercile(
        predictor, obs, forecast=forecast, forecast_year=forecast_year,
        temporal_sigma=temporal_sigma, distribution=distribution,
        constrained=constrained, verbose=verbose,
    )


_calibrate_smoothed_regression.supports_deterministic = True


def _fit_gamma_field(obs_values):
    """Method-of-moments gamma fit per (season, lat, lon) on ``obs_values`` (season, year,
    lat, lon). Returns ``(shape, scale)``, each (season, lat, lon); NaN at any gridpoint/
    season where the fit is degenerate (near-zero variance, too few finite values) —
    mirrors the experiment's ``prob_realdata/crpss_reliability.py::_fit_gamma_field``."""
    from .methods import smoothed_regression as pb

    ns, _ny, nla, nlo = obs_values.shape
    shape = np.full((ns, nla, nlo), np.nan)
    scale = np.full((ns, nla, nlo), np.nan)
    for s in range(ns):
        for i in range(nla):
            for j in range(nlo):
                shape[s, i, j], scale[s, i, j] = pb.fit_gamma(obs_values[s, :, i, j])
    return shape, scale


def _gamma_transform_field(x, shape, scale):
    """Map ``x`` (season, ..., lat, lon) through ``gamma_to_normal`` season-by-season;
    ``shape``/``scale`` are (season, lat, lon). ``gamma_to_normal`` broadcasts shape/scale
    over trailing (lat, lon) axes but not over season, so season is looped explicitly —
    mirrors the experiment's ``prob_realdata/crpss_reliability.py::_gamma_transform``. A
    NaN shape/scale (degenerate gridpoint/season) propagates to NaN output there."""
    from .methods import smoothed_regression as pb

    out = np.full(x.shape, np.nan)
    with np.errstate(invalid="ignore"):
        for s in range(x.shape[0]):
            out[s] = pb.gamma_to_normal(x[s], shape[s], scale[s])
    return out


def _calibrate_smoothed_regression_tercile(predictor, obs, *, forecast, forecast_year,
                                           temporal_sigma, distribution, constrained,
                                           verbose=False):
    """``output_type="tercile"`` path for smoothed_regression: mean/spread scaling (a, b)
    fit per (season, lat, lon) (`methods.smoothed_regression.fit_ab_field`), smoothed
    across seasons (`smooth_ab`), applied to the resolved forecast year, and turned into
    below/near/above category probabilities via a Gaussian predictive distribution
    (`normal_category_probs`). Mirrors the Kharin et al. (2017) probabilistic experiment
    (`prob_realdata/crpss_reliability.py`, `fig2_fig3_illustrative.py`); see
    `_calibrate_smoothed_regression`'s docstring for the normal/gamma distinction.

    Fit-and-apply on the full hindcast (no leave-one-out), consistent with the
    deterministic path: the target is either a year present in `predictor`
    (``forecast_year``) or a separate out-of-sample ensemble ``forecast``
    (season, member, lat, lon) pushed through the hindcast-fitted (a, b), gamma
    parameters, and tercile boundaries.
    """
    from .methods import smoothed_regression as pb

    if "member" not in predictor.dims:
        raise ValueError(
            "smoothed_regression output_type='tercile' requires an ensemble predictor "
            "with a 'member' dimension (to estimate the ensemble spread)."
        )
    predictor = predictor.transpose("season", "year", "member", "lat", "lon")
    obs = obs.transpose("season", "year", "lat", "lon")
    seasons = predictor["season"].values
    lat = predictor["lat"].values
    lon = predictor["lon"].values

    yi = None
    if forecast is not None:
        if "member" not in forecast.dims:
            raise ValueError(
                "smoothed_regression output_type='tercile' requires an ensemble "
                "`forecast` with a 'member' dimension (to estimate the ensemble spread)."
            )
        forecast = forecast.transpose("season", "member", "lat", "lon")
    else:
        year = int(forecast_year) if forecast_year is not None else int(predictor["year"].max())
        hindcast_years = set(int(y) for y in predictor["year"].values)
        if year not in hindcast_years:
            raise ValueError(
                f"forecast_year={year} is not in the hindcast years "
                f"[{min(hindcast_years)}..{max(hindcast_years)}]. Pass the target's "
                "ensemble via `forecast=` to apply the hindcast fit out-of-sample."
            )
        yi = int(np.where(predictor["year"].values == year)[0][0])

    mu_fc = sigma_fc = None
    if distribution == "gamma":
        # Fit the gamma to the obs, then map every ensemble member and the obs through the
        # gamma CDF into standard-normal space *before* reducing over "member" — not just
        # the ensemble mean — matching the experiment's _gamma_transform/precip_example.
        shape, scale = _fit_gamma_field(obs.values)
        pred_hat = _gamma_transform_field(predictor.values, shape, scale)  # (s,y,m,lat,lon)
        obs_hat = _gamma_transform_field(obs.values, shape, scale)         # (s,y,lat,lon)
        mu_f = pred_hat.mean(axis=2)             # ensemble mean, normal space (s,y,lat,lon)
        sigma_f = pred_hat.std(axis=2)           # ensemble spread, normal space
        o = obs_hat
        if forecast is not None:
            # the forecast members go through the HINDCAST-fitted gamma — same
            # climate, out-of-sample application.
            fc_hat = _gamma_transform_field(forecast.values, shape, scale)  # (s,m,lat,lon)
            mu_fc = fc_hat.mean(axis=1)
            sigma_fc = fc_hat.std(axis=1)
    elif distribution == "normal":
        mu_f = predictor.values.mean(axis=2)
        sigma_f = predictor.values.std(axis=2)
        o = obs.values
        if forecast is not None:
            mu_fc = forecast.values.mean(axis=1)
            sigma_fc = forecast.values.std(axis=1)
    else:
        raise ValueError(
            f"smoothed_regression output_type='tercile': unknown distribution="
            f"{distribution!r} (expected 'normal' or 'gamma')."
        )

    f_clim = mu_f.mean(axis=1)                    # (season, lat, lon)
    o_clim = o.mean(axis=1)
    mu_anom = mu_f - f_clim[:, None]              # (season, year, lat, lon)
    o_anom = o - o_clim[:, None]

    # Tercile boundaries are the empirical 33.3/66.7 percentiles of the observed anomaly,
    # for both distributions -- matching the experiment (crpss_reliability._tercile_bounds,
    # fig2_fig3_illustrative). For gamma these are percentiles of the gamma_to_normal-space
    # obs anomaly, which is only approximately (not exactly) standard normal after the
    # method-of-moments fit, so the empirical percentiles differ from norm.ppf(1/3),(2/3)
    # and are the faithful choice.
    t_lo = np.percentile(o_anom, 100.0 / 3.0, axis=1)       # (season, lat, lon)
    t_hi = np.percentile(o_anom, 200.0 / 3.0, axis=1)
    degenerate = ~(t_lo < t_hi)
    if np.any(degenerate):
        t_hi = np.where(degenerate, t_lo + 1e-6, t_hi)

    a, b = pb.fit_ab_field(mu_anom, sigma_f, o_anom, constrained=constrained)
    a, b = pb.smooth_ab(a, b, temporal_sigma)

    if forecast is not None:
        mu_ho = a * (mu_fc - f_clim)              # (season, lat, lon)
        sig_ho = np.abs(b) * sigma_fc
    else:
        mu_ho = a * mu_anom[:, yi]                # (season, lat, lon)
        sig_ho = np.abs(b) * sigma_f[:, yi]

    probs = pb.normal_category_probs(mu_ho, sig_ho, t_lo, t_hi)   # (3, season, lat, lon)

    out = xr.DataArray(
        np.moveaxis(probs, 0, 1),
        dims=("season", "tercile", "lat", "lon"),
        coords={"season": seasons, "tercile": [0, 1, 2], "lat": lat, "lon": lon},
    )
    out.attrs["method"] = "smoothed_regression"
    out.attrs["distribution"] = distribution
    out.attrs["constrained"] = bool(constrained)
    out.attrs["temporal_sigma"] = "None" if temporal_sigma is None else str(temporal_sigma)
    out.attrs["out_of_sample"] = str(forecast is not None)
    if verbose:
        target = "out-of-sample forecast" if forecast is not None else f"forecast_year={year}"
        print(f"[calibrate:smoothed_regression] tercile ({distribution}) {target}")
    return out
