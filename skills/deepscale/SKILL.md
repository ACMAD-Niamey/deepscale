---
name: deepscale
description: Downscale, calibrate, ensemble, and verify seasonal climate forecasts with the accord-deepscale Python package. Use when bias-correcting or downscaling GCM hindcasts (BCSD, CCA, quantile mapping), producing tercile (below/normal/above) probability forecasts, calibrating with ensemble regression or logistic/WVG indices, building multi-model ensembles, computing skill metrics (RPSS, ROC, reliability), running the seasonal_mme PyCPT-style pipeline, generating SVSLRF verification reports and forecast maps, selecting analog years and completing a partly-observed season into scenarios (SMPG), positioning seasonal totals in a historical record (percentile / rank-of-record), computing teleconnection indices (Niño, ONI, RONI, DMI/IOD, WVG), or testing predictor significance (permutation test, FDR).
license: MIT
metadata:
  author: accord-research
  package: accord-deepscale
compatibility: Requires Python 3.10+. Real-data examples need accord-rosetta and CDS credentials; CorrDiff needs an NVIDIA GPU with torch/earth2studio installed manually.
---

# DeepScale — seasonal forecast downscaling, calibration, and verification

DeepScale turns coarse GCM seasonal hindcasts/forecasts plus fine-resolution observations into calibrated high-resolution forecasts (continuous fields or below/normal/above tercile probabilities) and scores them with cross-validated skill metrics. It operates purely on xarray and is source-agnostic — it pairs naturally with **rosetta** for data acquisition but does not require it.

- Install: `pip install accord-deepscale` — **import name is `deepscale`** (conventionally `import deepscale as ds`).
- Methods, metrics, ensemble strategies, and CV schemes are **registries looked up by name strings** — you select behavior with plain strings like `method="cca"`, `metrics="rpss"`, `strategy="bma"`, `cv="loyo"`.
- CCA is validated against CPT Fortran 17.8.3 / PyCPT to r≈0.9996 on predictions.

## Data conventions (get these right first)

| Object | Dims | Notes |
|---|---|---|
| GCM hindcast | `(year, member, lat, lon)` | `year` = consecutive integers |
| GCM forecast | `(member, lat, lon)` | year squeezed out |
| Observations (predictand) | `(year, lat, lon)` | fine grid |
| Tercile forecast | `(tercile, lat, lon)` | `tercile=[0,1,2]` = below/normal/above, sums to 1 per valid cell |
| CV tercile hindcasts | `(year, tercile, lat, lon)` | |

`lat`/`latitude`/`Y` and `lon`/`longitude`/`X` aliases are accepted, but prefer `lat`/`lon`. **CV years must be consecutive integers** (gap of 1) or CV schemes raise. NaN cells (ocean/dry) propagate cleanly. Rosetta's `fetch(..., year_index=True)` / `assemble()` produce exactly these shapes.

## Quick start — pick a method, forecast, verify

```python
import deepscale as ds
from deepscale.tercile import to_tercile, to_tercile_cv
from deepscale.cv import loyo

# gcm: (year, member, lat, lon) hindcast; obs: (year, lat, lon)
best = ds.optimize(gcm, obs, methods=["bcsd", "cca"], cv="loyo", primary_metric="rpss")
print(best.method, best.score)          # winning method + CV RPSS
tercile_fc = to_tercile(best.forecast, obs)   # (tercile, lat, lon)

# Honest cross-validated verification (see leakage warning below)
cv_preds = []
for train_years, test_year in loyo(gcm.year.values):
    model = ds.train(best.method, gcm.sel(year=train_years), obs.sel(year=train_years))
    cv_preds.append(model.predict(gcm.sel(year=test_year)).expand_dims(year=[test_year]))
import xarray as xr
cv_fcst = xr.concat(cv_preds, dim="year")
cv_terc = to_tercile_cv(cv_fcst, obs, method="bootstrap")

report = ds.skill(cv_terc, obs, metrics=["rpss", "roc", "reliability", "hss"], spatial=True)
report.to_pdf("skill.pdf")               # WMO-SVSLRF verification report
ds.write_terciles(tercile_fc, "forecast.nc", title="MAM precip")
ds.plot_terciles(tercile_fc, title="MAM 2024")
```

Or run the whole PyCPT-style pipeline in one call:

```python
result = ds.seasonal_mme({"prcp": {"ECMWF": (gcm, None)}}, obs, method="cca", cv="loyo")
result.tercile_forecast; result.skill_report.scores; result.metadata
```

## The core verbs

| Verb | Purpose | Key output |
|---|---|---|
| `ds.downscale(predictor_hindcast, obs, method="bcsd", output_type="continuous")` | One method, one GCM → downscaled field or terciles | `(member,lat,lon)` or `(tercile,lat,lon)` |
| `ds.optimize(gcm, obs, methods=[...], cv="loyo")` | Try methods under CV, keep the best | `OptimizeResult(method, score, forecast)` |
| `ds.train(name, hindcast, obs, save_to=path)` | Fit once, checkpoint; required for `requires_training` (DL) methods | fitted `MethodBase` |
| `ds.calibrate(predictor, obs, method="ereg"\|"logit"\|"smoothed_regression"\|LogitConfig)` | MOS calibration, **no regridding** | `(tercile,lat,lon)`; `smoothed_regression` is season-aware → `(season,lat,lon)` deterministic or `(season,tercile,lat,lon)` |
| `ds.ensemble(forecasts, obs, strategy="uniform", optimize_ensemble=...)` | Combine forecasts | `EnsembleResult(forecast, weights, ...)` |
| `ds.skill(forecast, obs, metrics=..., spatial=...)` | Score a forecast | `SkillReport` (`.scores`, `.spatial`, `.to_pdf`) |
| `ds.skill_compare({name: fc}, obs)` | Score several forecasts side by side | `ComparisonReport` (`.to_table`, `.to_heatmap`) |
| `ds.seasonal_mme(tracks, obs, method="cca")` | Full multi-track MME pipeline | `SeasonalMMEResult` |
| `ds.prediction_error_variance(cv_preds, obs)` | Per-cell PEV from CV residuals | `(lat,lon)` |
| `ds.flex_forecast(det_fcst, pev, obs, threshold)` | Exceedance probability P(Y > threshold) | `FlexForecastResult` |

Downscale methods: `bcsd`, `cca`, `qm`, `dqm`, `delta`, `climatology`, `rank-analog`, `corrdiff` (GPU). Ensemble strategies: `uniform`, `skill_weighted`, `bma`, `drop_worst`. CV schemes: `loyo`, `lko`, `blocked`, `expanding`. Metrics: `rpss`, `roc`, `groc`, `reliability`, `hss`, `pearson_r`, `spearman`, `2afc`, `rmse`, `msss`, `crpss`, `spread_error_ratio`, `spread_error_correlation` (+ presets `"svslrf"`, `"all"`). Also exported: `ds.seasonal_coefficients` — the fitted Kharin-2017 seasonally-smoothed regression coefficient field behind the `smoothed_regression` calibrator (see [references/methods.md](references/methods.md)). Full parameter tables: [references/methods.md](references/methods.md), [references/metrics-and-terciles.md](references/metrics-and-terciles.md).

## Analog completion & climate positioning (SMPG)

A second workflow, alongside the downscale/calibrate pipeline: pick the past years that resemble the one being forecast, splice a partly-observed season forward with each, and say where the result falls in the record. All pure xarray, all agnostic to whether the dims are `(lat, lon)`, `(region,)`, or a single series — full detail in [references/analog-completion.md](references/analog-completion.md).

```python
import deepscale as ds
clim = ds.seasonal_stack(archive, "JJAS")               # (time,…) -> (year, step, …) season-aligned
nino = ds.Index.named("nino34").reduce(sst_hcst)        # a teleconnection index series over year
analogs = ds.analogs_from_index(nino, target=1.4, n=9)  # the 9 years most like a forecast Niño3.4 of 1.4
result = ds.complete(observed_to_date, analogs, climatology=clim,
                     season="JJAS", forecast=next_30_days)  # -> CompletionResult
result.consensus; result.percentile; result.accumulation()  # median outcome + where it sits + curves
ds.plot_accumulation_scenarios(result, climatology=clim)
```

- **Analog selection** (`ds.analogs_from_years / _from_index / _from_field / where`) → an `AnalogSet` (scores every candidate, composes with `&` `|` `.top(n)`).
- **Climate positioning** (`ds.percentile_of`, `ds.rank_of_record`, `ds.frequency_below`, `ds.accumulate`, `ds.seasonal_reduce`, `ds.seasonal_stack`) → where a value sits in a reference record.
- **Scenario completion** (`ds.complete` → `CompletionResult`) → one plausible end-of-season per analog; omit `forecast=` and run twice to isolate what a dynamic forecast adds.
- **Scalar-series calibration** (`ds.quantile_map`, `ds.error_bounds`) → bias-correct / bracket a forecast index, not a field.
- **Teleconnection indices** (`ds.Index`) now cover `wvg`, `wvg2`, `nino12/3/34/4`, `oni`, `roni`, `dmi` (`iod`), `wtio`, `setio`, `wio`, `wpac`, with configurable `transform=` (`"standardize"`/`"anomaly"`/`"raw"`), `weights=` (`"cos_lat"`), and `baseline=` — see [references/api.md](references/api.md).
- **Combine & mask** (`ds.combine_terciles`, `ds.mask_by_skill`, `ds.dry_mask`) and **pool** (`ds.pool_ensembles`) — see [references/methods.md](references/methods.md).
- **Predictor significance** (`ds.loo_corr`, `ds.permutation_test`, `ds.fdr`) — see [references/metrics-and-terciles.md](references/metrics-and-terciles.md).

Calendar helpers (`deepscale.time.season_step`, `season_bounds`, dekad/pentad arithmetic) are module-qualified — reference them as `deepscale.time.<fn>`, not `ds.<fn>`.

## Critical discipline rules

1. **Tercile leakage:** never score CV hindcasts with `to_tercile(pred, obs)` — full obs leaks the held-out year. Use `to_tercile_cv(cv_predictions, obs, method="bootstrap"|"cpt"|...)`. `to_tercile` is for the production forecast only.
2. **Grid rule:** `calibrate()` and `skill_compare()` do **not** regrid — put the GCM on the obs grid first (`gcm.interp(lat=obs.lat, lon=obs.lon)`). `downscale()` methods map coarse→fine themselves.
3. **Metric/forecast pairing:** probabilistic metrics (`rpss`, `roc`, `groc`, `reliability`, `hss`) need a `tercile` dim of size 3; continuous metrics (`pearson_r`, `spearman`, `2afc`, `rmse`, `spread_error_*`) need the deterministic ensemble and reject tercile input. Score each family in its own `skill()` call.
4. **Don't nest `optimize()` inside a CV loop** — double CV plus non-consecutive inner years breaks the CV schemes. Use `train()` + `.predict()` in manual loops (see quick start).
5. **`primary_metric` must be a leaf metric** — `roc_an`, not `roc` (which expands to a dict).
6. **DL methods** (`requires_training=True`) refuse inline fitting: `ds.train(name, ..., save_to=path)` then `ds.downscale(..., weights_path=path)`.
7. **Mask discipline when comparing forecasts:** RPSS masks its climatology reference only where *obs* is NaN, so forecasts with different NaN footprints are silently scored over different cell sets (a uniform-1/3 forecast has scored +0.26 this way). Apply one common valid mask to every forecast and the obs before scoring, and self-check that a uniform `[1/3,1/3,1/3]` forecast scores ≈ 0 — see [references/metrics-and-terciles.md](references/metrics-and-terciles.md).

## Getting data in (rosetta)

Any xarray source works; rosetta produces the exact shapes deepscale expects:

```python
import rosetta
gcm = rosetta.fetch("c3s/ecmwf-monthly", "precip", init="2024-02", target="MAM",
                    region=[-5, 15, 33, 48], hindcast=(1993, 2016),
                    year_index=True)["precip"]                    # (year, member, lat, lon)
obs = rosetta.fetch("obs/era5", "precip", region=[-5, 15, 33, 48],
                    hindcast=(1993, 2016), target="MAM",
                    seasonal="mean")["precip"]                    # (year, lat, lon)
```

`rosetta.assemble(roster, ...)` returns `{label: (hindcast, forecast)}` already shaped `(year, member, lat, lon)` with a guaranteed `member` dim. Rosetta is optional at runtime — only shapefile/geometry `Index` regions import it.

## Common pitfalls (errors + environment: [references/troubleshooting.md](references/troubleshooting.md))

- `downscale(gcm=...)` is deprecated → `predictor_hindcast=`.
- `ensemble(optimize_ensemble=True)` requires `obs` and can silently fall back to uniform weights (gate; `RuntimeWarning`).
- Degenerate/dry cells get NaN tercile boundaries and are excluded from skill — expected, not a bug.
- Plotting/reporting/GeoTIFF need the `plotting` extra (`pip install accord-deepscale[plotting]`); errors say so.
- CorrDiff GPU deps (`torch`, `earth2studio`, `nvidia-physicsnemo`) are not on PyPI — install manually; its `save`/`load` raise `NotImplementedError`.

## Runnable examples (also see the repo's `examples/` directory)

- [examples/end_to_end_forecast.py](examples/end_to_end_forecast.py) — optimize → forecast → honest CV verification → PDF report
- [examples/seasonal_mme_pipeline.py](examples/seasonal_mme_pipeline.py) — one-call multi-model MME with `seasonal_mme`
- [examples/calibration.py](examples/calibration.py) — eReg multi-model calibration, logistic/WVG index calibration, and season-aware smoothed_regression (deterministic + tercile)
- [examples/ensemble_and_reporting.py](examples/ensemble_and_reporting.py) — strategies, safeguards, skill comparison, plots
- [examples/smoothed_calibration.py](examples/smoothed_calibration.py) — Kharin-2017 function layer in a manual honest-CV loop: gamma→normal, fit/smooth a·b, CRPSS
- [examples/analog_completion.py](examples/analog_completion.py) — SMPG workflow: seasonal_stack → analogs_from_index/where → complete → CompletionResult (consensus, percentile, rank-of-record)

## Reference files

- [references/api.md](references/api.md) — full signatures for the core forecasting verbs/dataclasses and the generalized `Index`
- [references/analog-completion.md](references/analog-completion.md) — SMPG subsystem: analog selection, scenario completion, climate positioning, scalar-series calibration, calendar/season-step utilities
- [references/methods.md](references/methods.md) — downscale methods, calibrators, ensemble strategies (+ `pool_ensembles`), tercile combination/masking (`combine`), CV schemes, registries
- [references/metrics-and-terciles.md](references/metrics-and-terciles.md) — every metric's semantics + tercile conversion discipline + predictor-significance tools
- [references/plotting-reporting.md](references/plotting-reporting.md) — which plot for which artifact, forecast/skill maps, field maps & choropleths, scenario/index plots, SVSLRF PDFs, GeoTIFF/NetCDF export, headless figure handling
- [references/troubleshooting.md](references/troubleshooting.md) — error → cause table, environment/install setup, test markers, operational scripts, convention caveats
