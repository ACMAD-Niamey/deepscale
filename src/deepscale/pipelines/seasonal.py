"""Seasonal multi-model ensemble (MME) orchestrator.

See docs/superpowers/specs/2026-05-15-seasonal-mme-orchestrator-design.md
for the full contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import xarray as xr

from ..cv import get_cv
from ..ensemble import EnsembleResult, ensemble
from ..registry import get_method
from ..skill import SkillReport, skill
from ..tercile import to_tercile_cv, to_tercile, cpt_tercile_forecast
from .._warnings import quiet_all_nan_slices

_PROBABILISTIC_METHODS = frozenset({"corrdiff"})


def _average_leverages(per_model_leverages):
    """Per-year mean of the per-(track, model) leverages, skipping non-finite values.

    A single degenerate model must not poison the MME leverage estimate: a NaN/Inf
    would flow into ``pesd = sqrt(s2*(1+lev))`` and collapse the whole tercile
    forecast. ``cca._project_by_sv`` prevents the finite-but-absurd case at source;
    this handles the non-finite case defensively. For all-finite inputs — the
    healthy path — it is identical to a plain mean. Returns one leverage per year,
    ``nan`` for a year where every model was non-finite.
    """
    levs = []
    for vals in zip(*per_model_leverages.values()):
        finite = [v for v in vals if np.isfinite(v)]
        levs.append(sum(finite) / len(finite) if finite else np.nan)
    return levs


@dataclass
class SeasonalMMEResult:
    """Outcome of `seasonal_mme()`. Field semantics: see the design doc."""
    forecast: xr.DataArray
    tercile_forecast: xr.DataArray
    tercile_cv: xr.DataArray
    skill_report: SkillReport
    ensemble_result: EnsembleResult
    pev: xr.DataArray | None
    per_model_methods: dict = field(default_factory=dict)
    per_model_cv_hindcasts: dict = field(default_factory=dict)
    per_model_forecasts: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@quiet_all_nan_slices
def seasonal_mme(
    predictor_tracks,
    obs,
    *,
    method: str = "cca",
    cv: str = "loyo",
    cpt_args: dict | None = None,
    skill_metrics=None,
    tercile_method: str | None = None,
    probability_aggregation: str = "pooled",
    forecast_year: int | None = None,
    optimize_ensemble: bool = False,
    primary_metric: str = "rpss",
    verbose: bool = True,
    native_years: bool = False,
) -> SeasonalMMEResult:
    """Run the full PyCPT-style multi-track seasonal pipeline end-to-end.

    See docs/superpowers/specs/2026-05-15-seasonal-mme-orchestrator-design.md
    for parameter semantics, the four-case `forecast_year` resolution rules,
    and the error-handling table.

    `native_years` (opt-in, default False): when True, each model is
    calibrated on its OWN `hcst.year ∩ obs.year` overlap instead of the
    single global intersection across all models. Only supported for
    `method="cca"` + `probability_aggregation="cpt_per_model"` — pooling
    ensemble members across models with different year sets is undefined, so
    `native_years=True` with `probability_aggregation="pooled"` raises. This
    reproduces looping `seasonal_mme()` one model at a time (each call's own
    global intersection = that model's overlap) and combining, in a single
    call. Default False leaves all other behavior byte-for-byte unchanged.
    """
    if native_years and probability_aggregation != "cpt_per_model":
        raise ValueError(
            "native_years=True requires probability_aggregation='cpt_per_model'; "
            "pooling ensemble members across models with different year sets "
            f"is undefined. Got probability_aggregation={probability_aggregation!r}."
        )
    if native_years and (cpt_args or {}).get("skillmask_threshold") is not None:
        raise ValueError(
            "native_years=True is not supported together with "
            "cpt_args['skillmask_threshold'] (the skill mask needs a coherent "
            "shared-obs baseline, which native per-model years do not provide)."
        )
    if not predictor_tracks:
        raise ValueError(
            "seasonal_mme: predictor_tracks is empty; at least one track with "
            "one model is required."
        )
    empty_tracks = [name for name, models in predictor_tracks.items() if not models]
    if empty_tracks:
        raise ValueError(
            f"seasonal_mme: tracks {empty_tracks} contain zero models. "
            f"Each track must have at least one (model_name, (hcst, fcst)) entry."
        )
    if method in _PROBABILISTIC_METHODS:
        raise NotImplementedError(
            f"seasonal_mme V1 supports deterministic methods only; method "
            f"{method!r} produces samples. Track this in the V2 roadmap."
        )
    if probability_aggregation not in ("pooled", "cpt_per_model"):
        raise ValueError(
            "probability_aggregation must be 'pooled' or 'cpt_per_model'; "
            f"got {probability_aggregation!r}."
        )
    if probability_aggregation == "cpt_per_model" and method != "cca":
        raise ValueError(
            "probability_aggregation='cpt_per_model' requires method='cca' "
            f"(the per-model Student-t/leverage path); got method={method!r}."
        )

    # native_years=True: skip the global _intersect_years floor (it would
    # reject exactly the case this knob exists for — models whose individual
    # overlaps with obs are fine but don't all agree). Each model gets its
    # own obs slice below (per_model_obs); `years`/`obs_sliced` here become a
    # "reference" set (union of the per-model native years) used only by the
    # best-effort pooled/ensemble/skill machinery that cpt_per_model's
    # tercile_forecast does not depend on (see per_model_obs threading below).
    per_model_obs: dict = {}
    if native_years:
        years = _union_native_years(predictor_tracks, obs)
    else:
        years = _intersect_years(predictor_tracks, obs)
    obs_sliced = obs.sel(year=years)
    resolved_forecast_year = _resolve_forecast_year(predictor_tracks, years, forecast_year)
    resolved_tercile_method = _resolve_tercile_method(method, tercile_method)

    per_model_cv_hindcasts: dict = {}
    per_model_forecasts: dict = {}
    per_model_methods: dict = {}
    per_model_leverages: dict = {}

    for track_name, models in predictor_tracks.items():
        for model_name, (hcst, fcst) in models.items():
            key = (track_name, model_name)
            if native_years:
                model_years = _native_years_for_model(
                    track_name, model_name, hcst, obs)
                model_obs = obs.sel(year=model_years)
                per_model_obs[key] = model_obs
            else:
                model_obs = obs_sliced
            cv_hindcast, forecast_pred, m_full, levs = _per_model_cv(
                hcst, fcst, model_obs,
                method=method, cv_scheme=cv, cpt_args=cpt_args,
                forecast_year=resolved_forecast_year, verbose=verbose,
            )
            per_model_cv_hindcasts[key] = cv_hindcast
            per_model_forecasts[key] = forecast_pred
            per_model_methods[key] = m_full
            if levs is not None:
                per_model_leverages[key] = levs

    pooled_cv = _pool_members(per_model_cv_hindcasts)
    pooled_fcst = _pool_members(per_model_forecasts)

    if native_years:
        # The pooled ensemble/skill machinery below (ensemble(), skill(),
        # the tercile_cv fallback) is not part of cpt_per_model's
        # tercile_forecast contract, but it still runs and must not crash.
        # Under native_years the per-model CV hindcasts are ragged (each
        # spans that model's own native years), so `uniform.combine()`'s
        # plain arithmetic (`sum(arrays) / len(arrays)`) inner-joins them
        # down to their actual intersection. Rebind obs_sliced to that same
        # intersection so ensemble()'s internal PEV/skill calls see matching
        # year sets instead of raising on a mismatch against the (wider)
        # union computed above. If the per-model CV hindcasts are fully
        # disjoint, `combined` itself degenerates to an empty year dim (same
        # inner-join mechanism) — rebind to that same empty intersection so
        # the two sides still agree; ensemble()/skill() handle an empty year
        # dim without crashing (unlike a *mismatched* one).
        cv_year_sets = [set(arr.year.values.tolist()) for arr in pooled_cv]
        cv_intersection = sorted(set.intersection(*cv_year_sets)) if cv_year_sets else []
        obs_sliced = obs.sel(year=cv_intersection)

    ensemble_result = ensemble(
        pooled_cv, obs_sliced,
        strategy="uniform",
        optimize_ensemble=optimize_ensemble,
        primary_metric=primary_metric,
        cv=cv,
    )

    # tercile_cv is what the probabilistic skill report is scored against, so it
    # must be built the same way as the published tercile_forecast. Under
    # cpt_per_model that is the per-model-averaged CPT construction, not the
    # pooled one — otherwise the reported RPSS/ROC characterize a different
    # forecast than the one returned.
    if probability_aggregation == "cpt_per_model":
        tercile_cv = _cpt_per_model_tercile_cv(
            per_model_cv_hindcasts, per_model_methods, per_model_leverages,
            obs_sliced, fallback_forecast=ensemble_result.forecast,
            fallback_method=resolved_tercile_method,
            per_model_obs=per_model_obs if native_years else None,
        )
    else:
        tercile_kwargs = {}
        if resolved_tercile_method == "cpt":
            if not per_model_leverages:
                raise ValueError(
                    "seasonal_mme: tercile_method='cpt' but no per-model leverages "
                    "were collected; was a non-CCA method used?"
                )
            # Average leverages across (track, model) per year to get an MME-level estimate.
            tercile_kwargs["leverages"] = _average_leverages(per_model_leverages)
            tercile_kwargs["n_modes"] = (cpt_args or {}).get("n_modes", 3)

        tercile_cv = to_tercile_cv(
            ensemble_result.forecast, obs_sliced,
            method=resolved_tercile_method,
            **tercile_kwargs,
        )

    # Skill is scored against two forecast shapes:
    #   - deterministic ensemble mean → deterministic metrics
    #     (pearson_r, rmse, spearman, 2afc): they detect the lack of a
    #     `tercile` dim and skip the mean-collapse step.
    #   - tercile-probability CV forecast → probabilistic metrics
    #     (rpss, roc, reliability, heidke, generalized_roc): they require a
    #     `tercile` dim of size 3.
    # We call skill() twice and merge the reports. Each `metrics="all"` call
    # uses the existing skip_incompatible path to filter metrics that don't
    # match the supplied forecast shape.
    # The dual skill() call is by design: each metric is computed against the
    # input shape it expects, and the irrelevant half is skipped. The skip
    # warnings are useful when a caller invokes skill() directly with a single
    # input but are pure noise here, so suppress them locally rather than
    # forwarding them to seasonal_mme()'s caller.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=r"^skill\(metrics='all'\): skipping ",
        )
        det_report = skill(
            ensemble_result.forecast, obs_sliced,
            metrics=skill_metrics or "all",
        )
        prob_report = skill(
            tercile_cv, obs_sliced,
            metrics=skill_metrics or "all",
        )
    # Each metric raises ValueError when handed an input shape it can't
    # interpret, so the two reports have disjoint key sets in practice:
    # deterministic metrics live in det_report, probabilistic in prob_report.
    # Merge order therefore doesn't matter for correctness.
    skill_report = SkillReport(
        scores={**det_report.scores, **prob_report.scores},
        spatial={**det_report.spatial, **prob_report.spatial},
        metadata={**det_report.metadata, **prob_report.metadata},
        diagrams={**det_report.diagrams, **prob_report.diagrams},
    )
    # Surface ensemble-level diagnostics into the SkillReport's diagrams
    # payload so renderers (svslrf) can pick them up via the same convention
    # used for roc/reliability diagrams.
    if ensemble_result.member_contributions is not None:
        skill_report.diagrams["member_contributions"] = ensemble_result.member_contributions

    # Build the single-year MME tercile forecast. Two aggregation models:
    #   - "pooled" (default): pool per-model forecasts as ensemble members and
    #     convert the spread to terciles (today's behavior, unchanged).
    #   - "cpt_per_model": compute CPT-style per-model Student-t tercile
    #     probabilities and average the per-model probability maps.
    if probability_aggregation == "cpt_per_model":
        # CPT-style per-model tercile probabilities, averaged across models.
        # Student-t with leverage-inflated prediction-error variance (the shared
        # `cpt_tercile_forecast` kernel — same math the CV path uses). Boundaries
        # use the CPT convention (`_cpt_boundaries`), matching the per-model
        # tercile_cv built above so the forecast and its skill report agree.
        #
        # native_years=False (default): boundaries/dof/s2 are computed once off
        # the shared global `obs_sliced`/`years`, exactly as before.
        # native_years=True: each model's boundaries/dof/s2 use that model's
        # OWN obs slice (per_model_obs), matching the "loop one model at a
        # time" reference semantics — each standalone seasonal_mme() call's
        # `obs_sliced` in that loop IS that one model's native-year slice.
        if not native_years:
            t33, t67 = _cpt_spatial_boundaries(obs_sliced)
        per_model_maps = []
        for key, cv_pred in per_model_cv_hindcasts.items():
            track_name, model_name = key
            m = per_model_methods[key]
            fc_pred = per_model_forecasts[key]
            n_modes = int(getattr(m, "x_eof_modes_", getattr(m, "n_modes", 3)))
            if native_years:
                model_obs = per_model_obs[key]
                t33, t67 = _cpt_spatial_boundaries(model_obs)
                dof = len(model_obs.year) - n_modes - 1
            else:
                model_obs = obs_sliced
                dof = len(years) - n_modes - 1
            if dof <= 1:
                continue
            # Forecast predictor for this model = the input track's forecast
            # field. When no forecast slice was supplied (fcst is None), the
            # forecast year lives inside the hindcast, mirroring _per_model_cv's
            # forecast-input resolution. The real ICPAC pipeline always supplies
            # a forecast slice, so that branch reproduces the lifted math exactly.
            hcst_predictor, fcst_predictor = predictor_tracks[track_name][model_name]
            if fcst_predictor is None:
                fcst_predictor = hcst_predictor.sel(year=[resolved_forecast_year])
            elif "year" in fcst_predictor.dims:
                fcst_predictor = fcst_predictor.sel(year=[resolved_forecast_year])
            leverage = m.leverage(fcst_predictor)
            if not np.isfinite(leverage):
                leverage = 0.0
            s2 = ((cv_pred - model_obs) ** 2).sum("year") / dof
            per_model_maps.append(
                cpt_tercile_forecast(fc_pred, t33, t67, s2, dofr=dof, leverage=leverage))
        if not per_model_maps:
            raise RuntimeError(
                "probability_aggregation='cpt_per_model': no per-model CPT "
                "probability maps could be computed (dof<=1 for all models)."
            )
        out = xr.concat(per_model_maps, dim="model", join="outer").mean("model", skipna=True)
        total = out.sum("tercile", skipna=False)
        tercile_forecast = xr.where(
            np.isfinite(total) & (total > 0), out / total, np.nan)
        # Match the pooled path's canonical (tercile, lat, lon) dim ordering.
        tercile_forecast = tercile_forecast.transpose("tercile", ...)
        forecast_mean = xr.concat(
            list(per_model_forecasts.values()), dim="model").mean("model")
    else:
        # Build a single-year MME forecast (member dim = each per-model forecast).
        forecast_members = xr.concat(
            list(per_model_forecasts.values()),
            dim=xr.DataArray(
                [f"{t}__{m}" for (t, m) in per_model_forecasts.keys()],
                dims="member",
                name="member",
            ),
        )

        # `to_tercile` produces (tercile, lat, lon) probabilities using member-
        # counting or Gaussian fit. We use Gaussian when the ensemble has fewer
        # than 10 members (counting becomes noisy); otherwise counting.
        tercile_forecast_method = "counting" if forecast_members.sizes.get("member", 0) >= 10 else "gaussian"
        tercile_forecast = to_tercile(
            forecast_members, obs_sliced, method=tercile_forecast_method,
        )

        # MME mean for the forecast year (consumer-facing 'forecast' scalar map).
        forecast_mean = forecast_members.mean("member")

    # Skillmask (§7): where cross-validated Pearson skill is below the
    # threshold, replace the forecast with climatology. This changes forecast
    # *values* (deterministic -> climatological mean; tercile -> uniform 1/3),
    # not just a display mask. Cells with undefined skill (NaN) are left as-is.
    skillmask_threshold = (cpt_args or {}).get("skillmask_threshold")
    if skillmask_threshold is not None:
        pearson = xr.corr(ensemble_result.forecast, obs_sliced, dim="year")
        low_skill = pearson < skillmask_threshold
        climo_mean = obs_sliced.mean("year")
        forecast_mean = xr.where(low_skill, climo_mean, forecast_mean)
        tercile_forecast = xr.where(low_skill, 1.0 / 3.0, tercile_forecast)

    metadata = {
        "years_used": list(years),
        "cv": cv,
        "method": method,
        "tercile_method": resolved_tercile_method,
        "probability_aggregation": probability_aggregation,
        "tracks": list(predictor_tracks.keys()),
        "n_members": len(per_model_forecasts),
        "forecast_year": int(resolved_forecast_year),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "native_years": native_years,
    }
    if native_years:
        # Under native_years, "years_used" (above) is the union across models
        # for backward-compatible shape; this is the per-model breakdown each
        # model actually calibrated on (mirrors calibrate_cca_per_model_years'
        # years_used dict in the one-model-at-a-time reference loop).
        metadata["per_model_years_used"] = {
            f"{track}__{model}": list(per_model_obs[(track, model)].year.values)
            for (track, model) in per_model_forecasts
        }

    return SeasonalMMEResult(
        forecast=forecast_mean,
        tercile_forecast=tercile_forecast,
        tercile_cv=tercile_cv,
        skill_report=skill_report,
        ensemble_result=ensemble_result,
        pev=ensemble_result.pev,
        per_model_methods=per_model_methods,
        per_model_cv_hindcasts=per_model_cv_hindcasts,
        per_model_forecasts=per_model_forecasts,
        metadata=metadata,
    )


def _resolve_forecast_year(predictor_tracks, intersection_years, forecast_year):
    """Resolve forecast_year per the four-case rules in the design doc.

    Returns an int. Raises ValueError on mismatch/missing.
    """
    # Inventory forecast-slice years per (track, model).
    fcst_years_per_model = {}
    fcst_provided = {}
    for track_name, models in predictor_tracks.items():
        for model_name, (_hcst, fcst) in models.items():
            key = (track_name, model_name)
            fcst_provided[key] = fcst is not None
            if fcst is not None:
                fcst_years_per_model[key] = sorted(fcst.year.values.tolist())

    if forecast_year is not None:
        # Rule 1: explicit year must be available everywhere.
        missing = []
        for track_name, models in predictor_tracks.items():
            for model_name, (hcst, fcst) in models.items():
                hcst_years = set(hcst.year.values.tolist())
                fcst_years = set() if fcst is None else set(fcst.year.values.tolist())
                if forecast_year not in (hcst_years | fcst_years):
                    missing.append(f"{track_name}/{model_name}")
        if missing:
            raise ValueError(
                f"seasonal_mme: forecast_year={forecast_year} is not available "
                f"in models {missing} (neither in their hindcast nor in a "
                f"provided forecast slice)."
            )
        return forecast_year

    all_provided = all(fcst_provided.values())
    none_provided = not any(fcst_provided.values())

    if all_provided:
        # Rule 2: every model has a forecast slice — infer year.
        single_years = {k: ys for k, ys in fcst_years_per_model.items() if len(ys) == 1}
        if len(single_years) != len(fcst_years_per_model):
            multi = [k for k, ys in fcst_years_per_model.items() if len(ys) != 1]
            raise ValueError(
                f"seasonal_mme: forecast_year=None requires every forecast "
                f"slice to be a single year. Multi-year slices: {multi}. "
                f"Pass an explicit forecast_year=..."
            )
        years = {ys[0] for ys in single_years.values()}
        if len(years) != 1:
            raise ValueError(
                f"seasonal_mme: forecast_year=None and forecast slices have "
                f"different years: {fcst_years_per_model}. Pass an explicit "
                f"forecast_year=..."
            )
        return next(iter(years))

    if none_provided:
        # Rule 3: fall back to last year of the intersected hindcast range.
        return max(intersection_years)

    # Rule 4: mixed → require explicit year.
    raise ValueError(
        "seasonal_mme: forecast_year=None with mixed fcst=None / "
        "fcst=DataArray across models. Pass an explicit forecast_year=..."
    )


_METHOD_PARAMS = (
    "n_modes", "x_eof_modes", "y_eof_modes", "cca_modes",
    "device", "n_samples", "target_variable", "standardize",
    # CPT_ARGS knobs applied inside the method (§7). skillmask_threshold and
    # crossvalidation_window and mode_selection are handled by the orchestrator,
    # not the method.
    "transform_predictand", "tailoring", "drymask_threshold",
)


def _per_model_cv(hcst, fcst, obs_sliced, *, method, cv_scheme, cpt_args,
                  forecast_year, verbose):
    """Run a CV-folded fit-and-predict for one (hcst, fcst) pair.

    Returns (cv_hindcast, forecast_pred, fitted_method_full, leverages).

    - `cv_hindcast`: (year, lat, lon) — held-out predictions, one per year in
      obs_sliced.
    - `forecast_pred`: (lat, lon) — single-year prediction for `forecast_year`.
    - `fitted_method_full`: method object trained on all years of (hcst, obs).
      Used by the orchestrator if leverages or other introspection are needed.
    - `leverages`: list[float] aligned to obs_sliced.year, or None if the
      method does not expose a `.leverage` attribute.
    """
    method_cls = get_method(method)
    method_kwargs = {k: v for k, v in (cpt_args or {}).items() if k in _METHOD_PARAMS}

    # Slice hcst to the intersection years used for fitting/scoring.
    # Use .values to avoid xarray carrying along any non-dimension scalar
    # coordinates (e.g. a stray 'member' scalar from CDS-fetched data) that
    # would conflict with dimension coordinates on hcst.
    hcst_sliced = hcst.sel(year=obs_sliced.year.values)
    all_years = list(obs_sliced.year.values)

    # ---- CV loop: held-out predictions for each obs year. ----
    import inspect
    cv_fn = get_cv(cv_scheme) if isinstance(cv_scheme, str) else cv_scheme
    # crossvalidation_window (§7): widen the leave-out window for schemes that
    # accept it (e.g. loyo). deepscale's own default stays the scheme default
    # (loyo=1); a PyCPT CPT_ARGS dict passes its own (typically 5).
    cv_window = (cpt_args or {}).get("crossvalidation_window")
    if cv_window is not None and "window" in inspect.signature(cv_fn).parameters:
        fold_iter = cv_fn(all_years, window=cv_window)
    else:
        fold_iter = cv_fn(all_years)

    mode_sel = (cpt_args or {}).get("mode_selection")
    if method == "cca" and mode_sel == "north":
        # Eigenvalue-spectrum selection (Scree + North's rule), the CPT.x
        # dry-run approach: one SVD per field, no CV requirement (so no LOYO
        # restriction), selected per model since this runs inside the per-model
        # loop. Ceilings reuse the x/y_eof_range upper bounds (ACMAD reference:
        # 10 for obs-SST predictors, 7 for NMME predictors).
        from ..methods.cca import select_modes_north
        xe, ye, cc = select_modes_north(
            hcst_sliced,
            obs_sliced,
            x_ceiling=(cpt_args or {}).get("x_eof_range", (1, 10))[1],
            y_ceiling=(cpt_args or {}).get("y_eof_range", (1, 10))[1],
        )
        method_kwargs.update({
            "x_eof_modes": xe,
            "y_eof_modes": ye,
            "cca_modes": cc,
        })
    elif method == "cca" and mode_sel in ("auto", "cpt"):
        if cv_scheme != "loyo":
            raise ValueError(
                "seasonal_mme: CCA mode_selection='auto' currently requires "
                "cv='loyo', matching CPT's cross-validated mode-selection path."
            )
        from ..methods.cca import select_modes
        mode_window = cv_window if cv_window is not None else 1
        x_range = (cpt_args or {}).get("x_eof_range", (1, 10))
        y_range = (cpt_args or {}).get("y_eof_range", (1, 10))
        cca_range = (cpt_args or {}).get("cca_range", (1, 10))
        # Default the fallback to the smallest searched combo so a model whose
        # CV goodness is never finite falls back to minimal modes instead of
        # aborting the whole MME run; callers can override via cpt_args.
        fallback_modes = (cpt_args or {}).get(
            "mode_selection_fallback",
            (x_range[0], y_range[0], cca_range[0]),
        )
        xe, ye, cc, goodness, _cv, _lev = select_modes(
            hcst_sliced,
            obs_sliced,
            all_years,
            window=mode_window,
            x_eof_range=x_range,
            y_eof_range=y_range,
            cca_range=cca_range,
            fallback_modes=fallback_modes,
        )
        method_kwargs.update({
            "x_eof_modes": xe,
            "y_eof_modes": ye,
            "cca_modes": cc,
        })
        method_kwargs["_mode_selection_goodness"] = goodness

    mode_selection_goodness = method_kwargs.pop("_mode_selection_goodness", None)
    fold_predictions = []
    leverages = [] if hasattr(method_cls, "leverage") else None
    can_leverage = leverages is not None

    for train_years, test in fold_iter:
        test_years = test if isinstance(test, list) else [test]
        m = method_cls(**method_kwargs)
        m.fit(hcst_sliced.sel(year=train_years), obs_sliced.sel(year=train_years),
              **method_kwargs)
        test_forecast = hcst_sliced.sel(year=test_years)
        pred = m.predict(test_forecast, **method_kwargs)
        # `predict` may return a result with the year dim or with year squeezed
        # out depending on the method. Normalize to (year, lat, lon).
        if "year" not in pred.dims:
            pred = pred.expand_dims(year=test_years)
        fold_predictions.append(pred)
        if can_leverage:
            # Leverage is computed on the same test_forecast for each test year.
            for yr in test_years:
                leverages.append(m.leverage(test_forecast.sel(year=[yr])))

    cv_hindcast = xr.concat(fold_predictions, dim="year").sortby("year")
    # Schemes that overlap (lko sliding) produce multiple predictions per year.
    cv_hindcast = cv_hindcast.groupby("year").mean("year")
    # Reduce over any member dim so cv_hindcast is (year, lat, lon), as
    # `ensemble()` expects a list of (year, lat, lon) DataArrays.
    if "member" in cv_hindcast.dims:
        cv_hindcast = cv_hindcast.mean("member")

    # ---- Full fit + forecast-year prediction. ----
    m_full = method_cls(**method_kwargs)
    m_full.fit(hcst_sliced, obs_sliced, **method_kwargs)
    if mode_selection_goodness is not None:
        m_full.mode_selection_goodness_ = mode_selection_goodness

    if fcst is not None:
        # Caller-provided forecast slice.
        if "year" in fcst.dims:
            fc_input = fcst.sel(year=[forecast_year])
        else:
            fc_input = fcst.expand_dims(year=[forecast_year])
    elif forecast_year in [int(y) for y in hcst.year.values]:
        # Forecast year lives inside the original hindcast range.
        fc_input = hcst.sel(year=[forecast_year])
    else:
        # Should not happen: _resolve_forecast_year should have caught this.
        raise ValueError(
            f"_per_model_cv: forecast_year={forecast_year} is unavailable in "
            f"both the provided forecast and the original hindcast."
        )

    forecast_pred = m_full.predict(fc_input, **method_kwargs)

    if "year" in forecast_pred.dims and forecast_pred.sizes["year"] == 1:
        forecast_pred = forecast_pred.isel(year=0, drop=True)
    # Reduce over any member dim so forecast_pred is (lat, lon), as the spec
    # requires per_model_forecasts to be per-model (lat, lon) scalar maps.
    if "member" in forecast_pred.dims:
        forecast_pred = forecast_pred.mean("member")

    if verbose:
        print(f"[deepscale.seasonal_mme] per-model CV complete: "
              f"{len(all_years)} CV years, forecast_year={forecast_year}")

    return cv_hindcast, forecast_pred, m_full, leverages


def _cpt_spatial_boundaries(obs):
    """CPT-convention tercile boundaries (t33, t67) as spatial DataArrays.

    Same boundary convention `to_tercile_cv(method='cpt')` uses, so the
    cpt_per_model forecast and its CV-scored skill report share boundaries.
    """
    from ..metrics.rpss import _cpt_boundaries

    t33_arr, t67_arr = _cpt_boundaries(obs.values)
    spatial_dims = [d for d in obs.dims if d != "year"]
    spatial_coords = {k: v for k, v in obs.coords.items()
                      if k != "year" and set(obs[k].dims).issubset(set(spatial_dims))}
    t33 = xr.DataArray(t33_arr, dims=spatial_dims, coords=spatial_coords)
    t67 = xr.DataArray(t67_arr, dims=spatial_dims, coords=spatial_coords)
    return t33, t67


def _cpt_per_model_tercile_cv(per_model_cv_hindcasts, per_model_methods,
                              per_model_leverages, obs, *,
                              fallback_forecast, fallback_method,
                              per_model_obs=None):
    """Per-model CPT CV terciles averaged across models.

    The CV-scoring twin of the cpt_per_model forecast: each model's CV
    predictions go through the same `to_tercile_cv(method='cpt')` construction
    (per-model EOF count and per-year leverages), and the maps are averaged and
    renormalized exactly like the forecast. Falls back to the pooled
    construction only if no model supports CPT terciles (dof <= 1 everywhere).

    `per_model_obs`: optional {key: obs_slice} override (native_years=True) —
    each model's own obs slice, used in place of the shared `obs` for that
    model's dof (`n`) and `to_tercile_cv` call. When None (native_years=False,
    the default), every model uses the shared `obs`, unchanged from before.
    """
    maps = []
    for key, cv_pred in per_model_cv_hindcasts.items():
        model_obs = per_model_obs[key] if per_model_obs is not None else obs
        n = len(model_obs.year)
        m = per_model_methods[key]
        n_modes = int(getattr(m, "x_eof_modes_", getattr(m, "n_modes", 3)))
        if n - n_modes - 1 <= 1:
            continue
        maps.append(to_tercile_cv(
            cv_pred, model_obs, method="cpt",
            leverages=per_model_leverages.get(key), n_modes=n_modes))
    if not maps:
        return to_tercile_cv(fallback_forecast, obs, method=fallback_method)
    # join="outer" (xarray's current concat default, pinned explicitly): under
    # native_years the per-model CV maps carry ragged `year` coords, so we take
    # their union and skipna-average — a model contributes to the years it has.
    stacked = xr.concat(maps, dim="model", join="outer").mean("model", skipna=True)
    total = stacked.sum("tercile", skipna=False)
    out = xr.where(np.isfinite(total) & (total > 0), stacked / total, np.nan)
    # Match the (year, tercile, lat, lon) ordering the pooled to_tercile_cv path
    # returns, so downstream scoring/consumers see a consistent shape.
    return out.transpose("year", "tercile", ...)


def _pool_members(per_model_arrays):
    """Convert {(track, model): DataArray} into a list of DataArrays, each
    carrying a `method` attribute of `"{track}__{model}"` for downstream
    identification by `ensemble`'s `_member_names`.
    """
    pooled = []
    for (track, model), arr in per_model_arrays.items():
        tagged = arr.copy()
        tagged.attrs["method"] = f"{track}__{model}"
        pooled.append(tagged)
    return pooled


def _resolve_tercile_method(method, tercile_method):
    """Resolve the tercile-conversion method.

    Returns the resolved string. Raises ValueError if the caller asked for
    `cpt` with a non-CCA method (no leverages available).
    """
    if tercile_method is None:
        return "cpt" if method == "cca" else "bootstrap"
    if tercile_method == "cpt" and method != "cca":
        raise ValueError(
            f"seasonal_mme: tercile_method='cpt' requires method='cca' (which "
            f"exposes hat-matrix leverages); got method={method!r}. Use "
            f"tercile_method='bootstrap' or 'gaussian_loo' for non-CCA methods."
        )
    return tercile_method


def _intersect_years(predictor_tracks, obs):
    """Return the sorted list of years common to obs and every hindcast.

    Raises ValueError with diagnostic year ranges if the intersection has
    fewer than 5 years.
    """
    year_sets = {"obs": set(obs.year.values.tolist())}
    for track_name, models in predictor_tracks.items():
        for model_name, (hcst, _fcst) in models.items():
            year_sets[f"{track_name}/{model_name}"] = set(hcst.year.values.tolist())
    intersection = sorted(set.intersection(*year_sets.values()))
    if len(intersection) < 5:
        ranges = {
            name: (min(ys), max(ys)) if ys else (None, None)
            for name, ys in year_sets.items()
        }
        raise ValueError(
            f"seasonal_mme: year intersection across obs and all hindcasts "
            f"has only {len(intersection)} year(s); need at least 5. "
            f"Per-source year ranges: {ranges}. "
            f"Intersection: {intersection}."
        )
    return intersection


def _native_years_for_model(track_name, model_name, hcst, obs):
    """Return the sorted years common to `obs` and this one model's hindcast.

    The native_years=True per-model counterpart to `_intersect_years`: same
    >=5-year floor, but scoped to a single (track, model) instead of the
    intersection across every model. Raising per-model (rather than
    pre-checking a union) keeps the diagnostic message pinned to the model
    that actually failed the floor.
    """
    obs_years = set(obs.year.values.tolist())
    hcst_years = set(hcst.year.values.tolist())
    intersection = sorted(obs_years & hcst_years)
    if len(intersection) < 5:
        raise ValueError(
            f"seasonal_mme: native_years=True year intersection for "
            f"{track_name}/{model_name} has only {len(intersection)} "
            f"year(s); need at least 5. obs range: "
            f"{(min(obs_years), max(obs_years)) if obs_years else (None, None)}. "
            f"{track_name}/{model_name} hindcast range: "
            f"{(min(hcst_years), max(hcst_years)) if hcst_years else (None, None)}. "
            f"Intersection: {intersection}."
        )
    return intersection


def _union_native_years(predictor_tracks, obs):
    """Union of every model's own obs-intersected native years.

    Used only as the "reference" year set for native_years=True's best-effort
    pooled/ensemble/skill scaffolding (forecast_year fallback resolution,
    `metadata['years_used']`, `tercile_cv`'s fallback path) — none of which
    the cpt_per_model `tercile_forecast` this knob targets actually depends
    on. Each model's own overlap is validated separately by
    `_native_years_for_model`, so this never raises the <5-year floor itself
    (a union of per-model-valid sets already has >=5 years whenever at least
    one model does, and the per-model call raises first for one that can't).
    """
    obs_years = set(obs.year.values.tolist())
    union: set = set()
    for track_name, models in predictor_tracks.items():
        for model_name, (hcst, _fcst) in models.items():
            union |= obs_years & set(hcst.year.values.tolist())
    return sorted(union)
